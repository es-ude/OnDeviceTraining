#define SOURCE_FILE "ODT_DIFFDRIVE"

/*
 * odt_diffdrive -- the C side of the differential numerics harness
 * (test/differential/README.md).
 *
 *   odt_diffdrive <spec.json> <in_dir> <out_dir>
 *
 * Reads one op spec (JSON) plus its operands (.npy integer CODES for
 * quantized dtypes, float32 for FLOAT32), runs exactly one op through the
 * real libraries, and writes every produced tensor back as CODES + scale
 * (never dequantized floats: the comparison needs the scale separately).
 *
 * Ops and tiers (opTable[] at the bottom):
 *   conv1d  kernel/forward   conv1dKernelSymInt32: accumulator-range int32 at
 *                            scale s_in * s_w, which the funnel never exposes
 *           layer/forward    layerFunctions[CONV1D].forward through executeOp
 *   linear  kernel/forward   linearForwardSymInt32 (the funnel's kernel adapter
 *                            target: transpose dance + matmulSymInt32TensorsWithBias)
 *           kernel/dx        linearCalcPropLossSymInt32 (matmulSymInt32Tensors)
 *           kernel/weightGrad linearCalcWeightGradsSymInt32 (+ the raw bias-grad
 *                            value sum linearCalcBiasGradsSymInt32 when a bias
 *                            operand is given)
 *           layer/forward    layerFunctions[LINEAR].forward: FLOAT32 or SYM_INT32
 *                            input, packed SYM or SYM_INT32 weight/bias
 *           layer/dx, layer/weightGrad  one layerFunctions[LINEAR].backward
 *                            emitting propLoss + weightGrad (+ biasGrad); the
 *                            grads are stored FLOAT32 (the production default)
 *   convert convert/encode           executeConvert FLOAT32 -> SYM_INT32
 *           convert/requant_dynamic  executeConvert SYM_INT32 -> SYM_INT32 (the
 *                                    conversionMatrix diagonal, fresh scale)
 *           convert/requant_fixed    requantSymInt32TensorToScale into the
 *                                    pre-set spec.output.scale
 *
 * The structure (operand loader by dtype, op table keyed on op/tier/direction,
 * output registry) is meant to grow: later steps add the acc, optim, sr and
 * BFP modes as new table rows, loader arms and quantization builders -- not
 * by rewriting this file.
 *
 * Spec errors fail fast on stderr with the key path and exit 2. Library-side
 * fail-fasts (PRINT_ERROR + exit(1)) are visible only when the build defines
 * DEBUG_MODE_ERROR (the `differential` preset does).
 *
 * Allocation rule: no malloc/calloc here -- fixed static buffers for the spec
 * text and the JSON node pool, reserveMemory/freeReservedMemory for tensors.
 */

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dKernel.h"
#include "ExecuteOp.h"
#include "Kernel.h"
#include "Layer.h"
#include "Linear.h"
#include "NPYLoaderApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "SlidingWindow1d.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "npy_writer.h"

#define DIFF_PATH_MAX 4096

static void failf(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void failf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fputs("odt_diffdrive: ", stderr);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
    exit(2);
}

/* ------------------------------------------------------------------------- */
/* Minimal JSON: recursive descent into a static node pool. Strings are      */
/* unescaped in place inside the text buffer and NUL-terminated there.       */
/* ------------------------------------------------------------------------- */

typedef enum { J_NULL, J_BOOL, J_NUMBER, J_STRING, J_ARRAY, J_OBJECT } jsonType_t;

typedef struct jsonNode {
    jsonType_t type;
    const char *key; /* member name when the parent is an object, else NULL */
    const char *str; /* J_STRING payload */
    double num;      /* J_NUMBER payload */
    bool boolean;    /* J_BOOL payload */
    int firstChild;  /* J_ARRAY / J_OBJECT: index of the first child, -1 if none */
    int next;        /* next sibling, -1 at the end */
    int count;       /* J_ARRAY / J_OBJECT: number of children */
} jsonNode_t;

#define JSON_MAX_NODES 8192
#define JSON_MAX_TEXT (512 * 1024)

static jsonNode_t jsonPool[JSON_MAX_NODES];
static int jsonPoolUsed;
static char jsonText[JSON_MAX_TEXT];

typedef struct jsonParser {
    char *s;
    size_t pos;
    size_t len;
    const char *file;
} jsonParser_t;

static void jsonFail(const jsonParser_t *p, const char *what) __attribute__((noreturn));
static void jsonFail(const jsonParser_t *p, const char *what) {
    size_t line = 1;
    size_t col = 1;
    for (size_t i = 0; i < p->pos && i < p->len; i++) {
        if (p->s[i] == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
    }
    failf("%s:%zu:%zu: JSON: %s", p->file, line, col, what);
}

static int jsonNewNode(const jsonParser_t *p, jsonType_t type, const char *key) {
    if (jsonPoolUsed >= JSON_MAX_NODES) {
        jsonFail(p, "spec has more nodes than the static pool holds");
    }
    jsonNode_t *n = &jsonPool[jsonPoolUsed];
    memset(n, 0, sizeof(*n));
    n->type = type;
    n->key = key;
    n->firstChild = -1;
    n->next = -1;
    return jsonPoolUsed++;
}

static void jsonSkipWs(jsonParser_t *p) {
    while (p->pos < p->len && (p->s[p->pos] == ' ' || p->s[p->pos] == '\t' ||
                               p->s[p->pos] == '\n' || p->s[p->pos] == '\r')) {
        p->pos++;
    }
}

static char jsonPeek(const jsonParser_t *p) {
    return p->pos < p->len ? p->s[p->pos] : '\0';
}

static void jsonExpect(jsonParser_t *p, char c, const char *what) {
    if (jsonPeek(p) != c) {
        jsonFail(p, what);
    }
    p->pos++;
}

/* Parses a string literal starting at the opening quote; returns the
 * NUL-terminated, unescaped payload (written in place: out <= in). */
static const char *jsonParseString(jsonParser_t *p) {
    jsonExpect(p, '"', "expected '\"'");
    char *out = p->s + p->pos;
    char *start = out;
    while (true) {
        if (p->pos >= p->len) {
            jsonFail(p, "unterminated string");
        }
        char c = p->s[p->pos++];
        if (c == '"') {
            break;
        }
        if (c == '\\') {
            if (p->pos >= p->len) {
                jsonFail(p, "unterminated escape");
            }
            char e = p->s[p->pos++];
            switch (e) {
            case '"':
            case '\\':
            case '/':
                *out++ = e;
                break;
            case 'n':
                *out++ = '\n';
                break;
            case 't':
                *out++ = '\t';
                break;
            case 'r':
                *out++ = '\r';
                break;
            default:
                jsonFail(p, "unsupported escape (only \\\" \\\\ \\/ \\n \\t \\r)");
            }
        } else {
            *out++ = c;
        }
    }
    *out = '\0';
    return start;
}

static int jsonParseValue(jsonParser_t *p, const char *key);

static int jsonParseContainer(jsonParser_t *p, const char *key, bool isObject) {
    int node = jsonNewNode(p, isObject ? J_OBJECT : J_ARRAY, key);
    p->pos++; /* '{' or '[' */
    char close = isObject ? '}' : ']';
    int last = -1;
    jsonSkipWs(p);
    if (jsonPeek(p) == close) {
        p->pos++;
        return node;
    }
    while (true) {
        jsonSkipWs(p);
        const char *memberKey = NULL;
        if (isObject) {
            memberKey = jsonParseString(p);
            jsonSkipWs(p);
            jsonExpect(p, ':', "expected ':' after member name");
        }
        int child = jsonParseValue(p, memberKey);
        if (last < 0) {
            jsonPool[node].firstChild = child;
        } else {
            jsonPool[last].next = child;
        }
        last = child;
        jsonPool[node].count++;
        jsonSkipWs(p);
        char c = jsonPeek(p);
        if (c == ',') {
            p->pos++;
            continue;
        }
        if (c == close) {
            p->pos++;
            return node;
        }
        jsonFail(p, isObject ? "expected ',' or '}'" : "expected ',' or ']'");
    }
}

static int jsonParseValue(jsonParser_t *p, const char *key) {
    jsonSkipWs(p);
    char c = jsonPeek(p);
    if (c == '{') {
        return jsonParseContainer(p, key, true);
    }
    if (c == '[') {
        return jsonParseContainer(p, key, false);
    }
    if (c == '"') {
        const char *s = jsonParseString(p);
        int node = jsonNewNode(p, J_STRING, key);
        jsonPool[node].str = s;
        return node;
    }
    if (c == 't' && p->pos + 4 <= p->len && strncmp(p->s + p->pos, "true", 4) == 0) {
        p->pos += 4;
        int node = jsonNewNode(p, J_BOOL, key);
        jsonPool[node].boolean = true;
        return node;
    }
    if (c == 'f' && p->pos + 5 <= p->len && strncmp(p->s + p->pos, "false", 5) == 0) {
        p->pos += 5;
        int node = jsonNewNode(p, J_BOOL, key);
        jsonPool[node].boolean = false;
        return node;
    }
    if (c == 'n' && p->pos + 4 <= p->len && strncmp(p->s + p->pos, "null", 4) == 0) {
        p->pos += 4;
        return jsonNewNode(p, J_NULL, key);
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *end = NULL;
        errno = 0;
        double v = strtod(p->s + p->pos, &end);
        if (end == p->s + p->pos || errno != 0) {
            jsonFail(p, "bad number");
        }
        p->pos = (size_t)(end - p->s);
        int node = jsonNewNode(p, J_NUMBER, key);
        jsonPool[node].num = v;
        return node;
    }
    jsonFail(p, "unexpected character");
}

static const jsonNode_t *jsonParseFile(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        failf("cannot open spec %s: %s", path, strerror(errno));
    }
    size_t n = fread(jsonText, 1, sizeof(jsonText) - 1, f);
    if (ferror(f)) {
        failf("cannot read spec %s", path);
    }
    if (!feof(f)) {
        failf("spec %s exceeds the %d-byte static buffer", path, (int)sizeof(jsonText));
    }
    fclose(f);
    jsonText[n] = '\0';
    jsonParser_t p = {.s = jsonText, .pos = 0, .len = n, .file = path};
    jsonPoolUsed = 0;
    int root = jsonParseValue(&p, NULL);
    jsonSkipWs(&p);
    if (p.pos != p.len) {
        jsonFail(&p, "trailing characters after the top-level value");
    }
    if (jsonPool[root].type != J_OBJECT) {
        jsonFail(&p, "top-level value must be an object");
    }
    return &jsonPool[root];
}

/* --- typed getters; `path` is the dotted location used in error messages --- */

static const jsonNode_t *jsonMember(const jsonNode_t *obj, const char *key) {
    if (obj == NULL || obj->type != J_OBJECT) {
        return NULL;
    }
    for (int i = obj->firstChild; i >= 0; i = jsonPool[i].next) {
        if (jsonPool[i].key != NULL && strcmp(jsonPool[i].key, key) == 0) {
            return &jsonPool[i];
        }
    }
    return NULL;
}

static const char *jsonTypeName(jsonType_t t) {
    switch (t) {
    case J_NULL:
        return "null";
    case J_BOOL:
        return "bool";
    case J_NUMBER:
        return "number";
    case J_STRING:
        return "string";
    case J_ARRAY:
        return "array";
    case J_OBJECT:
        return "object";
    }
    return "?";
}

static const jsonNode_t *jsonRequire(const jsonNode_t *obj, const char *key, jsonType_t type,
                                     const char *path) {
    const jsonNode_t *m = jsonMember(obj, key);
    if (m == NULL) {
        failf("spec: missing required member %s.%s", path, key);
    }
    if (m->type != type) {
        failf("spec: %s.%s must be a %s (got %s)", path, key, jsonTypeName(type),
              jsonTypeName(m->type));
    }
    return m;
}

static const jsonNode_t *jsonRequireObject(const jsonNode_t *obj, const char *key,
                                           const char *path) {
    return jsonRequire(obj, key, J_OBJECT, path);
}

static const char *jsonRequireString(const jsonNode_t *obj, const char *key, const char *path) {
    return jsonRequire(obj, key, J_STRING, path)->str;
}

static const char *jsonStringOr(const jsonNode_t *obj, const char *key, const char *dflt,
                                const char *path) {
    const jsonNode_t *m = jsonMember(obj, key);
    if (m == NULL) {
        return dflt;
    }
    if (m->type != J_STRING) {
        failf("spec: %s.%s must be a string", path, key);
    }
    return m->str;
}

static size_t jsonSizeFromNode(const jsonNode_t *m, const char *key, const char *path) {
    if (m->type != J_NUMBER || m->num < 0 || m->num != floor(m->num) || m->num > 1e12) {
        failf("spec: %s.%s must be a non-negative integer", path, key);
    }
    return (size_t)m->num;
}

static size_t jsonRequireSize(const jsonNode_t *obj, const char *key, const char *path) {
    const jsonNode_t *m = jsonMember(obj, key);
    if (m == NULL) {
        failf("spec: missing required member %s.%s", path, key);
    }
    return jsonSizeFromNode(m, key, path);
}

static size_t jsonSizeOr(const jsonNode_t *obj, const char *key, size_t dflt, const char *path) {
    const jsonNode_t *m = jsonMember(obj, key);
    return m == NULL ? dflt : jsonSizeFromNode(m, key, path);
}

/* A float32 given as a C99 hexadecimal float literal in a JSON string
 * ("0x1.2p-10"): exact, no decimal round trip. Rejected unless the value is
 * exactly representable as float32 and finite. */
static float jsonRequireFloat32Hex(const jsonNode_t *obj, const char *key, const char *path) {
    const char *s = jsonRequireString(obj, key, path);
    const char *digits = (s[0] == '-' || s[0] == '+') ? s + 1 : s;
    if (strncmp(digits, "0x", 2) != 0 && strncmp(digits, "0X", 2) != 0) {
        failf("spec: %s.%s must be a hex float string such as \"0x1.2p-10\" (got \"%s\")", path,
              key, s);
    }
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || *end != '\0' || !isfinite(v)) {
        failf("spec: %s.%s: \"%s\" is not a finite hex float", path, key, s);
    }
    float f = (float)v;
    if ((double)f != v) {
        failf("spec: %s.%s: \"%s\" is not exactly representable as float32", path, key, s);
    }
    return f;
}

/* ------------------------------------------------------------------------- */
/* Spec vocabulary                                                           */
/* ------------------------------------------------------------------------- */

static roundingMode_t parseRounding(const char *s, const char *path) {
    if (strcmp(s, "HALF_AWAY") == 0) {
        return HALF_AWAY;
    }
    if (strcmp(s, "SR_HALF_AWAY") == 0) {
        /* A keyed stream (spec "sr") is the later `sr` mode's job; nothing in
         * this step opens one, and HALF_AWAY needs none. */
        failf("spec: %s: SR_HALF_AWAY is not supported yet (no stream plumbing in this step)",
              path);
    }
    failf("spec: %s: unknown rounding mode \"%s\" (HALF_AWAY | SR_HALF_AWAY)", path, s);
}

static arithmeticType_t parseArithmeticType(const char *s, const char *path) {
    if (strcmp(s, "FLOAT32") == 0) {
        return ARITH_FLOAT32;
    }
    if (strcmp(s, "SYM_INT32") == 0) {
        return ARITH_SYM_INT32;
    }
    if (strcmp(s, "BFP") == 0) {
        return ARITH_BFP;
    }
    failf("spec: %s: unknown arithmetic type \"%s\" (FLOAT32 | SYM_INT32 | BFP)", path, s);
}

/* spec.arith = {"type": "SYM_INT32", "rounding": "HALF_AWAY"} */
static arithmetic_t parseArithmetic(const jsonNode_t *spec) {
    const jsonNode_t *a = jsonRequireObject(spec, "arith", "spec");
    arithmetic_t arith;
    arith.type = parseArithmeticType(jsonRequireString(a, "type", "spec.arith"), "spec.arith.type");
    arith.roundingMode = parseRounding(jsonStringOr(a, "rounding", "HALF_AWAY", "spec.arith"),
                                       "spec.arith.rounding");
    return arith;
}

/* A storage config from {"dtype": ..., "qMaxBits": ..., "rounding": ...}.
 * Heap (reserveMemory) through the QuantizationApi factories; the caller
 * frees with freeQuantization unless a tensor took ownership. */
static quantization_t *makeQuantization(const jsonNode_t *q, const char *path) {
    const char *dtype = jsonRequireString(q, "dtype", path);
    if (strcmp(dtype, "FLOAT32") == 0) {
        return quantizationInitFloat();
    }
    if (strcmp(dtype, "SYM_INT32") == 0) {
        size_t bits = jsonSizeOr(q, "qMaxBits", ODT_SYM_OPERAND_QMAXBITS, path);
        if (bits < 2 || bits > 31) {
            failf("spec: %s.qMaxBits must be in [2, 31]", path);
        }
        roundingMode_t rm = parseRounding(jsonStringOr(q, "rounding", "HALF_AWAY", path), path);
        return quantizationInitSymInt32WithBits(rm, (uint8_t)bits);
    }
    failf("spec: %s.dtype \"%s\" is not supported in this step (FLOAT32 | SYM_INT32)", path, dtype);
}

/* ------------------------------------------------------------------------- */
/* Tensors                                                                   */
/* ------------------------------------------------------------------------- */

static tensor_t *makeTensor(size_t ndim, const size_t *dims, quantization_t *q) {
    size_t *ownedDims = reserveMemory(ndim * sizeof(size_t));
    memcpy(ownedDims, dims, ndim * sizeof(size_t));
    size_t *order = reserveMemory(ndim * sizeof(size_t));
    setOrderOfDimsForNewTensor(ndim, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, ndim, order);
    return initTensor(shape, q, NULL);
}

static void joinPath(char *out, size_t outSize, const char *dir, const char *file) {
    int n = snprintf(out, outSize, "%s/%s", dir, file);
    if (n < 0 || (size_t)n >= outSize) {
        failf("path too long: %s/%s", dir, file);
    }
}

/* Codes must sit on the declared grid: |code| <= qMax = 2^(bits-1)-1. */
static void checkCodeRange(const int32_t *codes, size_t n, size_t bits, const char *path) {
    const int32_t qMax = (int32_t)((1u << (bits - 1)) - 1u);
    for (size_t i = 0; i < n; i++) {
        if (codes[i] < -qMax || codes[i] > qMax) {
            failf("spec: %s: code %ld at flat index %zu is outside [-%ld, %ld] (%zu bits)", path,
                  (long)codes[i], i, (long)qMax, (long)qMax, bits);
        }
    }
}

/* Loads operand `name` of spec.operands:
 *   {"file": "x.npy", "dtype": "SYM_INT32", "scale": "0x1p-9",
 *    "qMaxBits": 12, "rounding": "HALF_AWAY"}
 * SYM_INT32: the .npy holds int32 CODES; the tensor is rebuilt with a
 * SYM_INT32 quantization carrying the given scale/width/rounding (same
 * int32 payload layout, copied). SYM ("qBits" instead of "qMaxBits"): the
 * .npy holds int32 CODES that are bit-packed into a per-tensor sub-byte SYM
 * tensor through repackSymInt32ToSymNoRescale, which carries the given scale
 * verbatim (the FLOAT32 -> SYM pack would re-derive it). FLOAT32: the .npy
 * holds float32 values. Returns NULL when the operand is absent and
 * `required` is false. */
static tensor_t *loadOperand(const jsonNode_t *operands, const char *name, const char *inDir,
                             bool required) {
    const jsonNode_t *o = jsonMember(operands, name);
    if (o == NULL) {
        if (required) {
            failf("spec: missing required operand spec.operands.%s", name);
        }
        return NULL;
    }
    char path[64];
    snprintf(path, sizeof(path), "spec.operands.%s", name);
    if (o->type != J_OBJECT) {
        failf("spec: %s must be an object", path);
    }
    const char *file = jsonRequireString(o, "file", path);
    const char *dtype = jsonRequireString(o, "dtype", path);

    char full[DIFF_PATH_MAX];
    joinPath(full, sizeof(full), inDir, file);
    tensor_t *loaded = npyLoadFlat(full);

    if (strcmp(dtype, "FLOAT32") == 0) {
        if (loaded->quantization->type != FLOAT32) {
            failf("%s: %s declares FLOAT32 but the .npy is not float32", path, file);
        }
        return loaded;
    }
    if (strcmp(dtype, "SYM_INT32") == 0) {
        if (loaded->quantization->type != INT32) {
            failf("%s: %s declares SYM_INT32 but the .npy is not int32 (codes expected)", path,
                  file);
        }
        size_t bits = jsonSizeOr(o, "qMaxBits", ODT_SYM_OPERAND_QMAXBITS, path);
        if (bits < 2 || bits > 31) {
            failf("spec: %s.qMaxBits must be in [2, 31]", path);
        }
        roundingMode_t rm = parseRounding(jsonStringOr(o, "rounding", "HALF_AWAY", path), path);
        float scale = jsonRequireFloat32Hex(o, "scale", path);
        if (!(scale > 0.0f)) {
            failf("spec: %s.scale must be > 0", path);
        }
        tensor_t *t = makeTensor(loaded->shape->numberOfDimensions, loaded->shape->dimensions,
                                 quantizationInitSymInt32WithBits(rm, (uint8_t)bits));
        ((symInt32QConfig_t *)t->quantization->qConfig)->scale = scale;
        size_t n = calcNumberOfElementsByTensor(loaded);
        memcpy(t->data, loaded->data, calcNumberOfBytesForData(t->quantization, n));
        freeTensor(loaded);
        checkCodeRange((const int32_t *)t->data, n, bits, path);
        return t;
    }
    if (strcmp(dtype, "SYM") == 0) {
        if (loaded->quantization->type != INT32) {
            failf("%s: %s declares SYM but the .npy is not int32 (codes expected)", path, file);
        }
        size_t bits = jsonSizeOr(o, "qBits", 8, path);
        if (bits < 2 || bits > 16) {
            failf("spec: %s.qBits must be in [2, 16]", path);
        }
        roundingMode_t rm = parseRounding(jsonStringOr(o, "rounding", "HALF_AWAY", path), path);
        float scale = jsonRequireFloat32Hex(o, "scale", path);
        if (!(scale > 0.0f)) {
            failf("spec: %s.scale must be > 0", path);
        }
        size_t n = calcNumberOfElementsByTensor(loaded);
        checkCodeRange((const int32_t *)loaded->data, n, bits, path);
        tensor_t *staged = makeTensor(loaded->shape->numberOfDimensions, loaded->shape->dimensions,
                                      quantizationInitSymInt32WithBits(rm, (uint8_t)bits));
        ((symInt32QConfig_t *)staged->quantization->qConfig)->scale = scale;
        memcpy(staged->data, loaded->data, n * sizeof(int32_t));
        tensor_t *packed = makeTensor(loaded->shape->numberOfDimensions, loaded->shape->dimensions,
                                      quantizationInitSym((uint8_t)bits, rm));
        repackSymInt32ToSymNoRescale(staged, packed);
        freeTensor(staged);
        freeTensor(loaded);
        return packed;
    }
    failf("spec: %s.dtype \"%s\" is not supported in this step (FLOAT32 | SYM_INT32 | SYM)", path,
          dtype);
}

/* The spec vocabulary for dtypes (quantTypeToString spells SYM_INT32 as
 * "SYMINT32"; the Python side keys on these names). */
static const char *dtypeName(qtype_t t) {
    switch (t) {
    case INT32:
        return "INT32";
    case FLOAT32:
        return "FLOAT32";
    case SYM_INT32:
        return "SYM_INT32";
    case SYM:
        return "SYM";
    case ASYM:
        return "ASYM";
    case BOOL:
        return "BOOL";
    case BFP:
        return "BFP";
    }
    return "?";
}

static void requireDtype(const tensor_t *t, qtype_t want, const char *what) {
    if (t->quantization->type != want) {
        failf("%s must be stored %s for this op/tier (got %s)", what, dtypeName(want),
              dtypeName(t->quantization->type));
    }
}

/* ------------------------------------------------------------------------- */
/* Output registry: <name>.npy (codes or float32) + result.json             */
/* ------------------------------------------------------------------------- */

#define MAX_OUTPUTS 8

typedef struct outputEntry {
    const char *name;
    tensor_t *tensor; /* owned by the registry after addOutput */
} outputEntry_t;

static outputEntry_t outputs[MAX_OUTPUTS];
static size_t nOutputs;

static void addOutput(const char *name, tensor_t *t) {
    if (nOutputs >= MAX_OUTPUTS) {
        failf("more than %d outputs", MAX_OUTPUTS);
    }
    outputs[nOutputs].name = name;
    outputs[nOutputs].tensor = t;
    nOutputs++;
}

static const char *roundingName(roundingMode_t rm) {
    return rm == SR_HALF_AWAY ? "SR_HALF_AWAY" : "HALF_AWAY";
}

static void writeOneOutput(FILE *json, const char *outDir, const outputEntry_t *e, bool last) {
    tensor_t *t = e->tensor;
    char fileName[128];
    snprintf(fileName, sizeof(fileName), "%s.npy", e->name);
    char full[DIFF_PATH_MAX];
    joinPath(full, sizeof(full), outDir, fileName);

    const size_t *dims = t->shape->dimensions;
    size_t ndim = t->shape->numberOfDimensions;
    int rc;
    switch (t->quantization->type) {
    case FLOAT32:
        rc = npyWriteFloat32(full, (const float *)t->data, dims, ndim);
        break;
    case INT32:
    case SYM_INT32:
        rc = npyWriteInt32(full, (const int32_t *)t->data, dims, ndim);
        break;
    default:
        failf("output %s: dtype %s cannot be written yet (codes-only sinks for SYM/ASYM/BFP are "
              "later steps)",
              e->name, dtypeName(t->quantization->type));
    }
    if (rc != 0) {
        failf("writing %s failed (npy writer rc=%d)", full, rc);
    }

    fprintf(json, "    \"%s\": {\n", e->name);
    fprintf(json, "      \"file\": \"%s\",\n", fileName);
    fprintf(json, "      \"dtype\": \"%s\",\n", dtypeName(t->quantization->type));
    fprintf(json, "      \"shape\": [");
    for (size_t i = 0; i < ndim; i++) {
        fprintf(json, "%s%zu", i ? ", " : "", dims[i]);
    }
    fprintf(json, "]");
    if (t->quantization->type == SYM_INT32) {
        const symInt32QConfig_t *qc = t->quantization->qConfig;
        fprintf(json, ",\n      \"scale_hex\": \"%a\",\n", (double)qc->scale);
        fprintf(json, "      \"scale\": %.9g,\n", (double)qc->scale);
        fprintf(json, "      \"qMaxBits\": %u,\n", (unsigned)qc->qMaxBits);
        fprintf(json, "      \"rounding\": \"%s\"", roundingName(qc->roundingMode));
    }
    fprintf(json, "\n    }%s\n", last ? "" : ",");
}

static void writeOutputs(const char *outDir) {
    char full[DIFF_PATH_MAX];
    joinPath(full, sizeof(full), outDir, "result.json");
    FILE *json = fopen(full, "w");
    if (json == NULL) {
        failf("cannot open %s for writing: %s", full, strerror(errno));
    }
    fprintf(json, "{\n  \"driver\": \"odt_diffdrive\",\n  \"outputs\": {\n");
    for (size_t i = 0; i < nOutputs; i++) {
        writeOneOutput(json, outDir, &outputs[i], i + 1 == nOutputs);
    }
    fprintf(json, "  }\n}\n");
    if (fclose(json) != 0) {
        failf("closing %s failed", full);
    }
    for (size_t i = 0; i < nOutputs; i++) {
        freeTensor(outputs[i].tensor);
        outputs[i].tensor = NULL;
    }
    nOutputs = 0;
}

/* ------------------------------------------------------------------------- */
/* Conv1d geometry                                                           */
/* ------------------------------------------------------------------------- */

/* spec.geometry = {"kernelSize": K, "stride": 1, "dilation": 1,
 *                  "padding": "VALID" | "SAME" | "EXPLICIT", "paddingAmount": p,
 *                  "groups": 1} */
static void parseConvGeometry(const jsonNode_t *spec, kernel_t *kernel, size_t *groups) {
    const jsonNode_t *g = jsonRequireObject(spec, "geometry", "spec");
    const char *path = "spec.geometry";
    size_t kernelSize = jsonRequireSize(g, "kernelSize", path);
    size_t stride = jsonSizeOr(g, "stride", 1, path);
    size_t dilation = jsonSizeOr(g, "dilation", 1, path);
    *groups = jsonSizeOr(g, "groups", 1, path);
    if (kernelSize == 0 || stride == 0 || dilation == 0 || *groups == 0) {
        failf("spec: %s: kernelSize/stride/dilation/groups must be >= 1", path);
    }
    const char *padding = jsonStringOr(g, "padding", "VALID", path);
    if (strcmp(padding, "VALID") == 0) {
        initKernel(kernel, kernelSize, VALID, dilation, stride);
    } else if (strcmp(padding, "SAME") == 0) {
        initKernel(kernel, kernelSize, SAME, dilation, stride);
    } else if (strcmp(padding, "EXPLICIT") == 0) {
        size_t amount = jsonRequireSize(g, "paddingAmount", path);
        initKernelExplicit(kernel, kernelSize, amount, dilation, stride);
    } else {
        failf("spec: %s.padding \"%s\" unknown (VALID | SAME | EXPLICIT)", path, padding);
    }
}

static void requireRank(const tensor_t *t, size_t rank, const char *what) {
    if (t->shape->numberOfDimensions != rank) {
        failf("%s must be rank %zu, got rank %zu", what, rank, t->shape->numberOfDimensions);
    }
}

/* The checks both Conv1d tiers run before touching a library entry (the
 * kernel does not validate geometry; a wrong spec would read out of bounds
 * or produce a shape the Python side cannot interpret): dtypes, ranks,
 * kernelSize vs weight dim 2, Cin == groups * weight dim 1, Cout % groups,
 * bias [Cout]. Code range is checked at load (loadOperand). */
static void validateConv1dOperands(const tensor_t *input, const tensor_t *weight,
                                   const tensor_t *bias, const kernel_t *kernel, size_t groups) {
    requireRank(input, 3, "spec.operands.input");
    requireRank(weight, 3, "spec.operands.weight");
    requireDtype(input, SYM_INT32, "spec.operands.input");
    requireDtype(weight, SYM_INT32, "spec.operands.weight");
    const size_t cIn = input->shape->dimensions[1];
    const size_t cOut = weight->shape->dimensions[0];
    const size_t cInPerGroup = weight->shape->dimensions[1];
    if (kernel->size != weight->shape->dimensions[2]) {
        failf("spec.geometry.kernelSize (%zu) != spec.operands.weight dim 2 (%zu)", kernel->size,
              weight->shape->dimensions[2]);
    }
    if (cIn != groups * cInPerGroup) {
        failf("spec.operands.input dim 1 (Cin=%zu) != spec.geometry.groups (%zu) * "
              "spec.operands.weight dim 1 (%zu)",
              cIn, groups, cInPerGroup);
    }
    if (cOut % groups != 0) {
        failf("spec.operands.weight dim 0 (Cout=%zu) is not a multiple of spec.geometry.groups "
              "(%zu)",
              cOut, groups);
    }
    if (bias != NULL) {
        requireRank(bias, 1, "spec.operands.bias");
        requireDtype(bias, SYM_INT32, "spec.operands.bias");
        if (bias->shape->dimensions[0] != cOut) {
            failf("spec.operands.bias length (%zu) != spec.operands.weight dim 0 (Cout=%zu)",
                  bias->shape->dimensions[0], cOut);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Op runners                                                                */
/* ------------------------------------------------------------------------- */

typedef struct runCtx {
    const jsonNode_t *spec;
    const char *inDir;
} runCtx_t;

typedef void (*opRunFn_t)(const runCtx_t *ctx);

/* conv1d / kernel / forward: conv1dKernelSymInt32 on SYM_INT32 codes. The
 * output is the kernel's own raw accumulator-range int32 at the scale the
 * kernel writes into the output qConfig (s_in * s_w). */
static void runConv1dKernelForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    if (arith.type != ARITH_SYM_INT32) {
        failf("conv1d/kernel/forward: only arith.type SYM_INT32 is implemented in this step");
    }
    kernel_t kernel;
    size_t groups;
    parseConvGeometry(ctx->spec, &kernel, &groups);

    const jsonNode_t *operands = jsonRequireObject(ctx->spec, "operands", "spec");
    tensor_t *input = loadOperand(operands, "input", ctx->inDir, true);
    tensor_t *weight = loadOperand(operands, "weight", ctx->inDir, true);
    tensor_t *bias = loadOperand(operands, "bias", ctx->inDir, false);
    validateConv1dOperands(input, weight, bias, &kernel, groups);

    windowGeometry1d_t geom = windowGeometry1dCalc(input->shape->dimensions[2], &kernel);
    size_t outDims[3] = {input->shape->dimensions[0], weight->shape->dimensions[0],
                         geom.outputLength};
    /* Raw wire: the kernel overwrites the scale; width/rounding are carried
     * for the record only (nothing restores here). */
    tensor_t *out = makeTensor(
        3, outDims, quantizationInitSymInt32WithBits(arith.roundingMode, ODT_SYM_OPERAND_QMAXBITS));

    conv1dKernelSymInt32(input, weight, bias, &kernel, groups, out, NULL);
    addOutput("output", out);

    freeTensor(input);
    freeTensor(weight);
    if (bias != NULL) {
        freeTensor(bias);
    }
}

/* conv1d / layer / forward: layerFunctions[CONV1D].forward through the funnel.
 * Config hand-wired like test/unit/support/include/BorrowedLayer.h: borrowed
 * parameter_t's, forwardMath = spec.arith, outputQ = spec.output. The output
 * tensor carries the spec.output storage config: the OUT_WRITE width-restore
 * takes the wire WIDTH (qMaxBits) from that target qConfig, while its
 * ROUNDING is operation-owned (docs/conventions/arithmetic-sym.md, #282):
 * cfg.forwardMath.roundingMode, i.e. spec.arith.rounding, not the target's. */
static void runConv1dLayerForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    kernel_t kernel;
    size_t groups;
    parseConvGeometry(ctx->spec, &kernel, &groups);
    quantization_t *outputQ =
        makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"), "spec.output");

    const jsonNode_t *operands = jsonRequireObject(ctx->spec, "operands", "spec");
    tensor_t *input = loadOperand(operands, "input", ctx->inDir, true);
    tensor_t *weightTensor = loadOperand(operands, "weight", ctx->inDir, true);
    tensor_t *biasTensor = loadOperand(operands, "bias", ctx->inDir, false);
    validateConv1dOperands(input, weightTensor, biasTensor, &kernel, groups);

    parameter_t *weights = parameterInit(weightTensor, gradInitFloat(weightTensor, NULL));
    parameter_t *bias =
        biasTensor != NULL ? parameterInit(biasTensor, gradInitFloat(biasTensor, NULL)) : NULL;

    conv1dConfig_t cfg;
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, groups, outputQ, outputQ,
                                       outputQ, outputQ);
    cfg.forwardMath = arith; /* the declared op arithmetic, independent of the wire's default */
    cfg.ownsQuantizations = false;
    layerConfig_t layerCfg = {.conv1d = &cfg};
    layer_t layer;
    initLayer(&layer, CONV1D, &layerCfg);

    size_t outDims[3];
    size_t outOrder[3];
    shape_t outShape = {
        .numberOfDimensions = 3, .dimensions = outDims, .orderOfDimensions = outOrder};
    layerFunctions[CONV1D].calcOutputShape(&layer, input->shape, &outShape);
    tensor_t *out = makeTensor(3, outDims, getQLike(outputQ));

    layerFunctions[CONV1D].forward(&layer, input, out, NULL);
    addOutput("output", out);

    freeTensor(input);
    freeParameter(weights);
    if (bias != NULL) {
        freeParameter(bias);
    }
    freeQuantization(outputQ);
}

/* ------------------------------------------------------------------------- */
/* Linear                                                                    */
/* ------------------------------------------------------------------------- */

static void requireDtype2(const tensor_t *t, qtype_t a, qtype_t b, const char *what) {
    if (t->quantization->type != a && t->quantization->type != b) {
        failf("%s must be stored %s or %s for this op/tier (got %s)", what, dtypeName(a),
              dtypeName(b), dtypeName(t->quantization->type));
    }
}

/* The checks both Linear tiers run before touching a library entry: ranks
 * (input [B, inF], weight [outF, inF], bias [outF], loss [B, outF]), the
 * shared dims, and the storage dtypes each tier accepts. The kernel entries
 * take SYM_INT32 only; the layer tier stages a FLOAT32 input and unpacks
 * packed SYM weights/bias in the funnel prologue. The loss is the int12 dx
 * wire of the layer above and arrives SYM_INT32 on both tiers. Code range
 * is checked at load (loadOperand). */
static void validateLinearOperands(const tensor_t *input, const tensor_t *weight,
                                   const tensor_t *bias, const tensor_t *loss, bool kernelTier) {
    requireRank(input, 2, "spec.operands.input");
    requireRank(weight, 2, "spec.operands.weight");
    if (kernelTier) {
        requireDtype(input, SYM_INT32, "spec.operands.input");
        requireDtype(weight, SYM_INT32, "spec.operands.weight");
    } else {
        requireDtype2(input, FLOAT32, SYM_INT32, "spec.operands.input");
        requireDtype2(weight, SYM, SYM_INT32, "spec.operands.weight");
    }
    const size_t batch = input->shape->dimensions[0];
    const size_t inF = input->shape->dimensions[1];
    const size_t outF = weight->shape->dimensions[0];
    if (weight->shape->dimensions[1] != inF) {
        failf("spec.operands.weight dim 1 (%zu) != spec.operands.input dim 1 (inF=%zu)",
              weight->shape->dimensions[1], inF);
    }
    if (bias != NULL) {
        requireRank(bias, 1, "spec.operands.bias");
        if (kernelTier) {
            requireDtype(bias, SYM_INT32, "spec.operands.bias");
        } else {
            requireDtype2(bias, SYM, SYM_INT32, "spec.operands.bias");
        }
        if (bias->shape->dimensions[0] != outF) {
            failf("spec.operands.bias length (%zu) != spec.operands.weight dim 0 (outF=%zu)",
                  bias->shape->dimensions[0], outF);
        }
    }
    if (loss != NULL) {
        requireRank(loss, 2, "spec.operands.loss");
        requireDtype(loss, SYM_INT32, "spec.operands.loss");
        if (loss->shape->dimensions[0] != batch || loss->shape->dimensions[1] != outF) {
            failf("spec.operands.loss must be [B=%zu, outF=%zu], got [%zu, %zu]", batch, outF,
                  loss->shape->dimensions[0], loss->shape->dimensions[1]);
        }
    }
}

typedef struct linearOperands {
    tensor_t *input;
    tensor_t *weight;
    tensor_t *bias; /* NULL without a bias operand */
    tensor_t *loss; /* NULL unless requested */
} linearOperands_t;

static linearOperands_t loadLinearOperands(const runCtx_t *ctx, bool withLoss, bool kernelTier) {
    const jsonNode_t *operands = jsonRequireObject(ctx->spec, "operands", "spec");
    linearOperands_t ops;
    ops.input = loadOperand(operands, "input", ctx->inDir, true);
    ops.weight = loadOperand(operands, "weight", ctx->inDir, true);
    ops.bias = loadOperand(operands, "bias", ctx->inDir, false);
    ops.loss = withLoss ? loadOperand(operands, "loss", ctx->inDir, true) : NULL;
    validateLinearOperands(ops.input, ops.weight, ops.bias, ops.loss, kernelTier);
    return ops;
}

static void freeLinearOperands(linearOperands_t *ops) {
    freeTensor(ops->input);
    freeTensor(ops->weight);
    if (ops->bias != NULL) {
        freeTensor(ops->bias);
    }
    if (ops->loss != NULL) {
        freeTensor(ops->loss);
    }
}

/* A raw SYM_INT32 wire for a kernel-tier output: the kernel overwrites the
 * scale; width/rounding are carried for the record only. */
static tensor_t *makeRawWire(size_t ndim, const size_t *dims, roundingMode_t rm) {
    return makeTensor(ndim, dims, quantizationInitSymInt32WithBits(rm, ODT_SYM_OPERAND_QMAXBITS));
}

static arithmetic_t requireSymArith(const runCtx_t *ctx, const char *what) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    if (arith.type != ARITH_SYM_INT32) {
        failf("%s: only arith.type SYM_INT32 is implemented in this step", what);
    }
    return arith;
}

/* linear / kernel / forward: linearForwardSymInt32 -- the entry the funnel's
 * kernel adapter calls (transpose dance around matmulSymInt32TensorsWithBias).
 * Calling it rather than matmul directly keeps the weight orientation under
 * test: Leg A's y = x W^T is the documented contract, not the driver's. Raw
 * accumulator-range int32 at s_in * s_w, the bias refolded per column. */
static void runLinearKernelForward(const runCtx_t *ctx) {
    arithmetic_t arith = requireSymArith(ctx, "linear/kernel/forward");
    linearOperands_t ops = loadLinearOperands(ctx, false, true);
    size_t outDims[2] = {ops.input->shape->dimensions[0], ops.weight->shape->dimensions[0]};
    tensor_t *out = makeRawWire(2, outDims, arith.roundingMode);
    linearForwardSymInt32(ops.weight, ops.bias, ops.input, out, NULL);
    addOutput("output", out);
    freeLinearOperands(&ops);
}

/* linear / kernel / dx: linearCalcPropLossSymInt32 = matmulSymInt32Tensors
 * (loss [B, outF], W [outF, inF]) -> raw [B, inF] at s_loss * s_w. */
static void runLinearKernelDx(const runCtx_t *ctx) {
    arithmetic_t arith = requireSymArith(ctx, "linear/kernel/dx");
    linearOperands_t ops = loadLinearOperands(ctx, true, true);
    size_t outDims[2] = {ops.input->shape->dimensions[0], ops.input->shape->dimensions[1]};
    tensor_t *out = makeRawWire(2, outDims, arith.roundingMode);
    linearCalcPropLossSymInt32(ops.weight, ops.loss, out);
    addOutput("propLoss", out);
    freeLinearOperands(&ops);
}

/* linear / kernel / weightGrad: linearCalcWeightGradsSymInt32 = loss^T @ x
 * -> raw [outF, inF] at s_loss * s_in; with a bias operand also the raw
 * bias-grad value sum over the batch (linearCalcBiasGradsSymInt32, at
 * s_loss). Both are what the funnel's backward kernel adapters emit before
 * the ACC epilogue. */
static void runLinearKernelWeightGrad(const runCtx_t *ctx) {
    arithmetic_t arith = requireSymArith(ctx, "linear/kernel/weightGrad");
    linearOperands_t ops = loadLinearOperands(ctx, true, true);
    size_t wgDims[2] = {ops.weight->shape->dimensions[0], ops.weight->shape->dimensions[1]};
    tensor_t *wg = makeRawWire(2, wgDims, arith.roundingMode);
    linearCalcWeightGradsSymInt32(ops.loss, ops.input, wg);
    addOutput("weightGrad", wg);
    if (ops.bias != NULL) {
        size_t bgDims[1] = {ops.weight->shape->dimensions[0]};
        tensor_t *bg = makeRawWire(1, bgDims, arith.roundingMode);
        linearCalcBiasGradsSymInt32(bg, ops.loss);
        addOutput("biasGrad", bg);
    }
    freeLinearOperands(&ops);
}

/* Hand-wired Linear layer (test/unit/support/include/BorrowedLayer.h pattern):
 * borrowed parameter_t's, every math slot = spec.arith, the produced wires =
 * the given storage configs, the packed-grad D1 accumulate modes at
 * linearInitConfig's defaults (dynamic-rescale weights, fixed-scale bias --
 * on FLOAT32 grads both are the exact float add). ownsQuantizations=false:
 * the caller frees the configs. */
static void wireLinearLayer(layer_t *layer, layerConfig_t *layerCfg, linearConfig_t *cfg,
                            parameter_t *weights, parameter_t *bias, arithmetic_t arith,
                            quantization_t *outputQ, quantization_t *propLossQ) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->weights = weights;
    cfg->bias = bias;
    cfg->forwardMath = arith;
    cfg->weightGradMath = arith;
    cfg->biasGradMath = arith;
    cfg->propLossMath = arith;
    cfg->outputQ = outputQ;
    cfg->propLossQ = propLossQ;
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    cfg->ownsQuantizations = false;
    cfg->frozen = false;
    layerCfg->linear = cfg;
    initLayer(layer, LINEAR, layerCfg);
}

/* linear / layer / forward: layerFunctions[LINEAR].forward through the
 * funnel. A FLOAT32 input is staged by the prologue (the harness's Leg A
 * assumes the documented encode at the int12 operand default); packed SYM
 * weights/bias are unpacked there. The output carries spec.output (width
 * from the target, rounding op-owned per #282, as for Conv1d). */
static void runLinearLayerForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    quantization_t *outputQ =
        makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"), "spec.output");
    linearOperands_t ops = loadLinearOperands(ctx, false, false);

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    linearConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireLinearLayer(&layer, &layerCfg, &cfg, weights, bias, arith, outputQ, outputQ);

    size_t outDims[2];
    size_t outOrder[2];
    shape_t outShape = {
        .numberOfDimensions = 2, .dimensions = outDims, .orderOfDimensions = outOrder};
    layerFunctions[LINEAR].calcOutputShape(&layer, ops.input->shape, &outShape);
    tensor_t *out = makeTensor(2, outDims, getQLike(outputQ));

    layerFunctions[LINEAR].forward(&layer, ops.input, out, NULL);
    addOutput("output", out);

    freeTensor(ops.input);
    freeParameter(weights); /* frees ops.weight and its grad */
    if (bias != NULL) {
        freeParameter(bias);
    }
    freeQuantization(outputQ);
}

/* linear / layer / {dx, weightGrad}: ONE layerFunctions[LINEAR].backward with
 * every backward math slot = spec.arith, emitting propLoss (OUT_WRITE onto
 * spec.output, the dx wire), weightGrad and biasGrad (the ACC epilogues into
 * FLOAT32 grads -- spec.gradStorage, FLOAT32 only in this step, the
 * production default). The Python side evaluates the output its cell's
 * direction names; the others ride along in result.json. */
static void runLinearLayerBackward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    quantization_t *propLossQ =
        makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"), "spec.output");
    const char *gradStorage = jsonStringOr(ctx->spec, "gradStorage", "FLOAT32", "spec");
    if (strcmp(gradStorage, "FLOAT32") != 0) {
        failf("spec.gradStorage \"%s\": only FLOAT32 grad storage is implemented in this step",
              gradStorage);
    }
    linearOperands_t ops = loadLinearOperands(ctx, true, false);

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    linearConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireLinearLayer(&layer, &layerCfg, &cfg, weights, bias, arith, propLossQ, propLossQ);

    size_t dxDims[2] = {ops.input->shape->dimensions[0], ops.input->shape->dimensions[1]};
    tensor_t *propLoss = makeTensor(2, dxDims, getQLike(propLossQ));

    layerFunctions[LINEAR].backward(&layer, ops.input, ops.loss, propLoss, NULL);
    addOutput("propLoss", propLoss);
    /* The grads move to the output registry (which frees them); detach them
     * from the parameters so freeParameter releases only the params. */
    addOutput("weightGrad", weights->grad);
    weights->grad = NULL;
    if (bias != NULL) {
        addOutput("biasGrad", bias->grad);
        bias->grad = NULL;
    }

    freeTensor(ops.input);
    freeTensor(ops.loss);
    freeParameter(weights);
    if (bias != NULL) {
        freeParameter(bias);
    }
    freeQuantization(propLossQ);
}

/* ------------------------------------------------------------------------- */
/* Convert tier: one conversionMatrix cell / requant primitive              */
/* ------------------------------------------------------------------------- */

static tensor_t *loadConvertInput(const runCtx_t *ctx, qtype_t want) {
    const jsonNode_t *operands = jsonRequireObject(ctx->spec, "operands", "spec");
    tensor_t *input = loadOperand(operands, "input", ctx->inDir, true);
    requireDtype(input, want, "spec.operands.input");
    return input;
}

static tensor_t *makeConvertTarget(const runCtx_t *ctx, const tensor_t *input) {
    quantization_t *targetQ =
        makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"), "spec.output");
    if (targetQ->type != SYM_INT32) {
        failf("spec.output.dtype: the convert tier targets SYM_INT32 in this step");
    }
    return makeTensor(input->shape->numberOfDimensions, input->shape->dimensions, targetQ);
}

/* convert / convert / encode: executeConvert FLOAT32 -> SYM_INT32, i.e. the
 * conversionMatrix[FLOAT32][SYM_INT32] encode; a bare conversion rounds by
 * the TARGET's storage roundingMode (#282), spec.output.rounding. */
static void runConvertEncode(const runCtx_t *ctx) {
    tensor_t *input = loadConvertInput(ctx, FLOAT32);
    tensor_t *out = makeConvertTarget(ctx, input);
    executeConvert(input, out, NULL);
    addOutput("output", out);
    freeTensor(input);
}

/* convert / convert / requant_dynamic: executeConvert SYM_INT32 -> SYM_INT32
 * routes through the conversionMatrix diagonal (#192 D1 dynamic requant: fresh
 * absmax scale at the target's qMaxBits) -- the OUT_WRITE width restore
 * every SYM producer runs, here in isolation and at a chosen width pair. */
static void runConvertRequantDynamic(const runCtx_t *ctx) {
    tensor_t *input = loadConvertInput(ctx, SYM_INT32);
    tensor_t *out = makeConvertTarget(ctx, input);
    executeConvert(input, out, NULL);
    addOutput("output", out);
    freeTensor(input);
}

/* convert / convert / requant_fixed: requantSymInt32TensorToScale into the
 * pre-set spec.output.scale (#192 D1 fixed-scale requant; not a matrix cell). */
static void runConvertRequantFixed(const runCtx_t *ctx) {
    tensor_t *input = loadConvertInput(ctx, SYM_INT32);
    tensor_t *out = makeConvertTarget(ctx, input);
    float target = jsonRequireFloat32Hex(jsonRequireObject(ctx->spec, "output", "spec"), "scale",
                                         "spec.output");
    if (!(target > 0.0f)) {
        failf("spec.output.scale must be > 0 for requant_fixed");
    }
    ((symInt32QConfig_t *)out->quantization->qConfig)->scale = target;
    requantSymInt32TensorToScale(input, out, NULL);
    addOutput("output", out);
    freeTensor(input);
}

typedef struct opEntry {
    const char *op;
    const char *tier;
    const char *direction;
    opRunFn_t run;
} opEntry_t;

/* Later steps append rows here: the acc, optim and sr modes, SYM grad
 * storage for the Linear backward, and the BFP twins. */
static const opEntry_t opTable[] = {
    {"conv1d", "kernel", "forward", runConv1dKernelForward},
    {"conv1d", "layer", "forward", runConv1dLayerForward},
    {"linear", "kernel", "forward", runLinearKernelForward},
    {"linear", "kernel", "dx", runLinearKernelDx},
    {"linear", "kernel", "weightGrad", runLinearKernelWeightGrad},
    {"linear", "layer", "forward", runLinearLayerForward},
    {"linear", "layer", "dx", runLinearLayerBackward},
    {"linear", "layer", "weightGrad", runLinearLayerBackward},
    {"convert", "convert", "encode", runConvertEncode},
    {"convert", "convert", "requant_dynamic", runConvertRequantDynamic},
    {"convert", "convert", "requant_fixed", runConvertRequantFixed},
};

/* ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: odt_diffdrive <spec.json> <in_dir> <out_dir>\n");
        return 2;
    }
    const jsonNode_t *spec = jsonParseFile(argv[1]);
    const char *op = jsonRequireString(spec, "op", "spec");
    const char *tier = jsonRequireString(spec, "tier", "spec");
    const char *direction = jsonRequireString(spec, "direction", "spec");

    const opEntry_t *entry = NULL;
    for (size_t i = 0; i < sizeof(opTable) / sizeof(opTable[0]); i++) {
        if (strcmp(opTable[i].op, op) == 0 && strcmp(opTable[i].tier, tier) == 0 &&
            strcmp(opTable[i].direction, direction) == 0) {
            entry = &opTable[i];
            break;
        }
    }
    if (entry == NULL) {
        fprintf(stderr, "odt_diffdrive: no runner for op=%s tier=%s direction=%s; supported:\n", op,
                tier, direction);
        for (size_t i = 0; i < sizeof(opTable) / sizeof(opTable[0]); i++) {
            fprintf(stderr, "  %s/%s/%s\n", opTable[i].op, opTable[i].tier, opTable[i].direction);
        }
        return 2;
    }

    runCtx_t ctx = {.spec = spec, .inDir = argv[2]};
    entry->run(&ctx);
    writeOutputs(argv[3]);
    return 0;
}
