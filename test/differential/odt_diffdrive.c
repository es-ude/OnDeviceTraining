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
 *           convert/pack_bfp         executeConvert FLOAT32 -> BFP (spec.output
 *                                    = the BFP target: widths, groupSize)
 *
 * BFP (step 3, spec.arith.type "BFP" or "FLOAT32" over BFP storage):
 *   linear  kernel/forward,dx,weightGrad  linearForwardBfp / linearCalcPropLossBfp
 *                            / linearCalcWeightGradsBfp (+ linearCalcBiasGradsBfp)
 *                            on the funnel's UNPACKED-BFP scratch form (int32
 *                            codes under a live bfpQConfig_t); raw FLOAT32 out
 *   conv1d  kernel/forward   conv1dKernelBfp, same form
 *   convt1d kernel/forward   convTranspose1dKernelBfpGather (D9), same form
 *   linear/conv1d/convt1d layer/forward  layerFunctions[].forward with BFP-stored
 *                            (packed) or FLOAT32 operands; output = spec.output
 *                            (FLOAT32 raw, or a BFP wire = the OUT_WRITE pack)
 *   linear/conv1d/convt1d layer/dx,weightGrad  ONE layerFunctions[].backward:
 *                            propLoss onto spec.output (FLOAT32 or BFP wire),
 *                            weightGrad + biasGrad into FLOAT32 grads (the
 *                            ACC epilogue is the exact float add there, so the
 *                            grads ARE the kernels' raw output)
 *   A BFP operand spec carries the int32 CODES (.npy) plus mantissaBits,
 *   exponentBits, groupSize and the stored exponents; the kernel tier gets
 *   them as scratch, the layer tier as a packed tensor (code-domain pack via
 *   the INT32 -> BFP cell, exponents then set verbatim). A BFP output is
 *   written as codes (BFP -> INT32 int_repr) + its exponents in result.json.
 *
 * The structure (operand loader by dtype, op table keyed on op/tier/direction,
 * output registry) is meant to grow: later steps add the acc, optim and sr
 * modes as new table rows, loader arms and quantization builders -- not by
 * rewriting this file.
 *
 * Spec errors fail fast on stderr with the key path and exit 2. Library-side
 * fail-fasts are exit 1; their PRINT_ERROR text appears only when the build
 * defines DEBUG_MODE_ERROR, which the `differential` preset does NOT (it sets
 * only ODT_BUILD_DIFFERENTIAL / ODT_BUILD_TESTS), so the Python side
 * classifies on the exit status (1 = ABORT) and must not rely on stderr.
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
#include "Conv1dTransposed.h"
#include "ConvTranspose1dKernel.h"
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

/* BFP widths/geometry of a spec object: {"mantissaBits": m, "exponentBits": e,
 * "groupSize": 0 | g}. groupSize follows the wire rule (arithmetic-bfp 5.5):
 * 0 or >= n is per-tensor {1,0}; otherwise numGroups = n / groupSize, a
 * non-divisor fails fast. */
typedef struct bfpSpec {
    uint8_t mantissaBits;
    uint8_t exponentBits;
    size_t numGroups;
    size_t groupSize;
    roundingMode_t rounding;
} bfpSpec_t;

static bfpSpec_t parseBfpSpec(const jsonNode_t *q, const char *path, size_t n) {
    bfpSpec_t s;
    size_t m = jsonRequireSize(q, "mantissaBits", path);
    size_t e = jsonRequireSize(q, "exponentBits", path);
    if (m < 2 || m > 16) {
        failf("spec: %s.mantissaBits must be in [2, 16]", path);
    }
    if (e < 2 || e > 8) {
        failf("spec: %s.exponentBits must be in [2, 8]", path);
    }
    s.mantissaBits = (uint8_t)m;
    s.exponentBits = (uint8_t)e;
    s.groupSize = jsonSizeOr(q, "groupSize", 0, path);
    if (s.groupSize == 0 || s.groupSize >= n) {
        s.groupSize = 0;
        s.numGroups = 1;
    } else {
        if (n % s.groupSize != 0) {
            failf("spec: %s.groupSize %zu does not divide the element count %zu", path, s.groupSize,
                  n);
        }
        s.numGroups = n / s.groupSize;
    }
    s.rounding = parseRounding(jsonStringOr(q, "rounding", "HALF_AWAY", path), path);
    return s;
}

static quantization_t *makeBfpQuantization(const bfpSpec_t *s) {
    if (s->numGroups == 1) {
        return quantizationInitBfp(s->mantissaBits, s->exponentBits, s->rounding);
    }
    return quantizationInitBfpGrouped(s->mantissaBits, s->exponentBits, s->rounding, s->numGroups,
                                      s->groupSize);
}

/* A storage config from {"dtype": ..., "qMaxBits": ..., "rounding": ...} (or
 * the BFP fields above; `n` is the element count the config will attach to,
 * needed for a grouped BFP wire). Heap (reserveMemory) through the
 * QuantizationApi factories; the caller frees with freeQuantization unless a
 * tensor took ownership. */
static quantization_t *makeQuantization(const jsonNode_t *q, const char *path, size_t n) {
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
    if (strcmp(dtype, "BFP") == 0) {
        bfpSpec_t s = parseBfpSpec(q, path, n);
        return makeBfpQuantization(&s);
    }
    failf("spec: %s.dtype \"%s\" is not supported (FLOAT32 | SYM_INT32 | BFP)", path, dtype);
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

/* How a BFP operand is materialized: the funnel's UNPACKED scratch form for a
 * kernel-tier call (int32 codes under a live bfpQConfig_t -- the form the
 * kernels take, see Matmul.h), or the PACKED storage form for the layer tier
 * (what a real parameter/wire is). */
typedef enum { BFP_SCRATCH, BFP_PACKED } bfpForm_t;

/* Reads spec.operands.<name>.exponents (an array of numGroups stored
 * exponents, each in [0, 2^exponentBits - 1]) into qc->exponents. */
static void readBfpExponents(const jsonNode_t *o, const char *path, bfpQConfig_t *qc) {
    const jsonNode_t *arr = jsonRequire(o, "exponents", J_ARRAY, path);
    if ((size_t)arr->count != qc->numGroups) {
        failf("spec: %s.exponents has %d entries, numGroups is %zu", path, arr->count,
              qc->numGroups);
    }
    size_t cap = (1u << qc->exponentBits) - 1u;
    size_t g = 0;
    for (int i = arr->firstChild; i >= 0; i = jsonPool[i].next, g++) {
        const jsonNode_t *m = &jsonPool[i];
        if (m->type != J_NUMBER) {
            failf("spec: %s.exponents[%zu] must be a number", path, g);
        }
        double v = m->num;
        if (v < 0 || v > (double)cap || v != (double)(size_t)v) {
            failf("spec: %s.exponents[%zu] = %g is outside [0, %zu]", path, g, v, cap);
        }
        qc->exponents[g] = (uint8_t)v;
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
 * holds float32 values. BFP ("mantissaBits", "exponentBits", "groupSize",
 * "exponents"): the .npy holds int32 CODES; `bfpForm` picks the scratch form
 * (int32 payload under a BFP quantization) or the packed form (INT32 -> BFP
 * code-domain pack, which aborts on a code outside the mantissa grid, then
 * the given exponents set verbatim). Returns NULL when the operand is absent
 * and `required` is false. */
static tensor_t *loadOperand(const jsonNode_t *operands, const char *name, const char *inDir,
                             bool required, bfpForm_t bfpForm) {
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
    if (strcmp(dtype, "BFP") == 0) {
        if (loaded->quantization->type != INT32) {
            failf("%s: %s declares BFP but the .npy is not int32 (codes expected)", path, file);
        }
        size_t n = calcNumberOfElementsByTensor(loaded);
        bfpSpec_t s = parseBfpSpec(o, path, n);
        checkCodeRange((const int32_t *)loaded->data, n, s.mantissaBits, path);
        quantization_t *bfpQ = makeBfpQuantization(&s);
        if (bfpForm == BFP_SCRATCH) {
            /* Scratch form: the int32 payload stays, the quantization becomes
             * BFP (UnitTestMatmul's stack-fixture idiom, on the heap). */
            readBfpExponents(o, path, bfpQ->qConfig);
            quantization_t *int32Q = loaded->quantization;
            loaded->quantization = bfpQ;
            freeQuantization(int32Q);
            return loaded;
        }
        tensor_t *packed =
            makeTensor(loaded->shape->numberOfDimensions, loaded->shape->dimensions, bfpQ);
        executeConvert(loaded, packed, NULL); /* INT32 -> BFP: codes verbatim, exponents = bias */
        readBfpExponents(o, path, packed->quantization->qConfig);
        freeTensor(loaded);
        return packed;
    }
    failf("spec: %s.dtype \"%s\" is not supported (FLOAT32 | SYM_INT32 | SYM | BFP)", path, dtype);
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
    case BFP: {
        /* Codes via the BFP -> INT32 cell (int_repr: sign-extended mantissas,
         * exponents dropped); the exponents go into result.json below. */
        tensor_t *codes = makeTensor(ndim, dims, quantizationInitInt32());
        executeConvert(t, codes, NULL);
        rc = npyWriteInt32(full, (const int32_t *)codes->data, dims, ndim);
        freeTensor(codes);
        break;
    }
    default:
        failf("output %s: dtype %s cannot be written yet (codes-only sinks for SYM/ASYM are "
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
    } else if (t->quantization->type == BFP) {
        const bfpQConfig_t *qc = t->quantization->qConfig;
        fprintf(json, ",\n      \"mantissaBits\": %u,\n", (unsigned)qc->mantissaBits);
        fprintf(json, "      \"exponentBits\": %u,\n", (unsigned)qc->exponentBits);
        fprintf(json, "      \"numGroups\": %zu,\n", qc->numGroups);
        fprintf(json, "      \"groupSize\": %zu,\n", qc->groupSize);
        fprintf(json, "      \"exponents\": [");
        for (size_t g = 0; g < qc->numGroups; g++) {
            fprintf(json, "%s%u", g ? ", " : "", (unsigned)qc->exponents[g]);
        }
        fprintf(json, "],\n");
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
/* Conv geometry                                                             */
/* ------------------------------------------------------------------------- */

/* spec.geometry = {"kernelSize": K, "stride": 1, "dilation": 1,
 *                  "padding": "VALID" | "SAME" | "EXPLICIT", "paddingAmount": p,
 *                  "groups": 1, "outputPadding": 0 (convt1d only)} */
static void parseConvGeometry(const jsonNode_t *spec, kernel_t *kernel, size_t *groups,
                              size_t *outputPadding) {
    const jsonNode_t *g = jsonRequireObject(spec, "geometry", "spec");
    const char *path = "spec.geometry";
    size_t kernelSize = jsonRequireSize(g, "kernelSize", path);
    size_t stride = jsonSizeOr(g, "stride", 1, path);
    size_t dilation = jsonSizeOr(g, "dilation", 1, path);
    *groups = jsonSizeOr(g, "groups", 1, path);
    if (kernelSize == 0 || stride == 0 || dilation == 0 || *groups == 0) {
        failf("spec: %s: kernelSize/stride/dilation/groups must be >= 1", path);
    }
    if (outputPadding != NULL) {
        *outputPadding = jsonSizeOr(g, "outputPadding", 0, path);
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

/* The storage dtypes an operand may arrive in. Kernel tier: exactly the
 * arithmetic's own form (SYM_INT32 codes for ARITH_SYM_INT32, unpacked-BFP
 * scratch for ARITH_BFP). Layer tier: anything the funnel prologue converts
 * or borrows -- FLOAT32 (staged under SYM/BFP math, native under FLOAT32),
 * SYM_INT32, packed SYM (weights/bias) and BFP (borrowed under ARITH_BFP,
 * dequantized under ARITH_FLOAT32 = the fake-quant twin). */
static void requireDtypeIn(const tensor_t *t, const qtype_t *allowed, size_t n, const char *what) {
    for (size_t i = 0; i < n; i++) {
        if (t->quantization->type == allowed[i]) {
            return;
        }
    }
    failf("%s: storage dtype %s is not accepted for this op/tier", what,
          dtypeName(t->quantization->type));
}

static const qtype_t kLayerActDtypes[] = {FLOAT32, SYM_INT32, BFP};
static const qtype_t kLayerParamDtypes[] = {FLOAT32, SYM, SYM_INT32, BFP};

static void requireOperandDtype(const tensor_t *t, bool kernelTier, arithmeticType_t arith,
                                bool isParam, const char *what) {
    if (kernelTier) {
        requireDtype(t, arith == ARITH_BFP ? BFP : SYM_INT32, what);
    } else if (isParam) {
        requireDtypeIn(t, kLayerParamDtypes, 4, what);
    } else {
        requireDtypeIn(t, kLayerActDtypes, 3, what);
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

typedef struct convOperands {
    tensor_t *input;
    tensor_t *weight;
    tensor_t *bias; /* NULL without a bias operand */
    tensor_t *loss; /* NULL unless requested */
} convOperands_t;

/* Loads input/weight/bias (+loss) for a conv-family op and checks what the
 * library entries do not: dtypes per tier, ranks, kernelSize vs the weight's
 * tap dim, the channel/group identities, bias length, loss shape. Conv1d:
 * weight [Cout, Cin/groups, K], Cin == groups * dim1, Cout % groups == 0.
 * ConvT1d: weight [Cin, Cout/groups, K], Cin == dim0, Cout = groups * dim1.
 * `outLen` is the forward output length the loss must match. */
static convOperands_t loadConvOperands(const runCtx_t *ctx, bool transposed, bool withLoss,
                                       bool kernelTier, arithmeticType_t arith,
                                       const kernel_t *kernel, size_t groups, size_t outLen) {
    const jsonNode_t *operands = jsonRequireObject(ctx->spec, "operands", "spec");
    bfpForm_t form = kernelTier ? BFP_SCRATCH : BFP_PACKED;
    convOperands_t ops;
    ops.input = loadOperand(operands, "input", ctx->inDir, true, form);
    ops.weight = loadOperand(operands, "weight", ctx->inDir, true, form);
    ops.bias = loadOperand(operands, "bias", ctx->inDir, false, form);
    ops.loss = withLoss ? loadOperand(operands, "loss", ctx->inDir, true, form) : NULL;

    requireRank(ops.input, 3, "spec.operands.input");
    requireRank(ops.weight, 3, "spec.operands.weight");
    requireOperandDtype(ops.input, kernelTier, arith, false, "spec.operands.input");
    requireOperandDtype(ops.weight, kernelTier, arith, true, "spec.operands.weight");
    const size_t batch = ops.input->shape->dimensions[0];
    const size_t cIn = ops.input->shape->dimensions[1];
    size_t cOut;
    if (kernel->size != ops.weight->shape->dimensions[2]) {
        failf("spec.geometry.kernelSize (%zu) != spec.operands.weight dim 2 (%zu)", kernel->size,
              ops.weight->shape->dimensions[2]);
    }
    if (!transposed) {
        cOut = ops.weight->shape->dimensions[0];
        if (cIn != groups * ops.weight->shape->dimensions[1]) {
            failf("spec.operands.input dim 1 (Cin=%zu) != spec.geometry.groups (%zu) * "
                  "spec.operands.weight dim 1 (%zu)",
                  cIn, groups, ops.weight->shape->dimensions[1]);
        }
        if (cOut % groups != 0) {
            failf("spec.operands.weight dim 0 (Cout=%zu) is not a multiple of spec.geometry.groups "
                  "(%zu)",
                  cOut, groups);
        }
    } else {
        if (cIn != ops.weight->shape->dimensions[0]) {
            failf("spec.operands.input dim 1 (Cin=%zu) != spec.operands.weight dim 0 (%zu)", cIn,
                  ops.weight->shape->dimensions[0]);
        }
        if (cIn % groups != 0) {
            failf("spec.operands.input dim 1 (Cin=%zu) is not a multiple of spec.geometry.groups "
                  "(%zu)",
                  cIn, groups);
        }
        cOut = ops.weight->shape->dimensions[1] * groups;
    }
    if (ops.bias != NULL) {
        requireRank(ops.bias, 1, "spec.operands.bias");
        requireOperandDtype(ops.bias, kernelTier, arith, true, "spec.operands.bias");
        if (ops.bias->shape->dimensions[0] != cOut) {
            failf("spec.operands.bias length (%zu) != Cout (%zu)", ops.bias->shape->dimensions[0],
                  cOut);
        }
    }
    if (ops.loss != NULL) {
        requireRank(ops.loss, 3, "spec.operands.loss");
        requireOperandDtype(ops.loss, kernelTier, arith, false, "spec.operands.loss");
        const size_t *ld = ops.loss->shape->dimensions;
        if (ld[0] != batch || ld[1] != cOut || ld[2] != outLen) {
            failf("spec.operands.loss must be [B=%zu, Cout=%zu, Lout=%zu], got [%zu, %zu, %zu]",
                  batch, cOut, outLen, ld[0], ld[1], ld[2]);
        }
    }
    return ops;
}

static void freeConvOperands(convOperands_t *ops) {
    freeTensor(ops->input);
    freeTensor(ops->weight);
    if (ops->bias != NULL) {
        freeTensor(ops->bias);
    }
    if (ops->loss != NULL) {
        freeTensor(ops->loss);
    }
}

/* The forward output length from the operands' shapes (the layer's own
 * calcOutputShape is cross-checked against it where a layer is built). */
static size_t convForwardOutLen(const runCtx_t *ctx, bool transposed, const kernel_t *kernel,
                                size_t outputPadding) {
    const jsonNode_t *operands = jsonRequireObject(ctx->spec, "operands", "spec");
    const jsonNode_t *o = jsonRequireObject(operands, "input", "spec.operands");
    char full[DIFF_PATH_MAX];
    joinPath(full, sizeof(full), ctx->inDir, jsonRequireString(o, "file", "spec.operands.input"));
    tensor_t *peek = npyLoadFlat(full);
    requireRank(peek, 3, "spec.operands.input");
    size_t length = peek->shape->dimensions[2];
    freeTensor(peek);
    if (transposed) {
        return convTranspose1dOutputLength(length, kernel, outputPadding);
    }
    return windowGeometry1dCalc(length, kernel).outputLength;
}

static size_t elemCount3(size_t a, size_t b, size_t c) {
    return a * b * c;
}

/* A raw SYM_INT32 wire for a kernel-tier output: the kernel overwrites the
 * scale; width/rounding are carried for the record only. */
static tensor_t *makeRawWire(size_t ndim, const size_t *dims, roundingMode_t rm) {
    return makeTensor(ndim, dims, quantizationInitSymInt32WithBits(rm, ODT_SYM_OPERAND_QMAXBITS));
}

/* A raw FLOAT32 wire: what every ARITH_BFP kernel emits (D7). */
static tensor_t *makeFloatWire(size_t ndim, const size_t *dims) {
    return makeTensor(ndim, dims, quantizationInitFloat());
}

static void requireKernelArith(arithmetic_t arith, const char *what) {
    if (arith.type != ARITH_SYM_INT32 && arith.type != ARITH_BFP) {
        failf("%s: the kernel tier implements arith.type SYM_INT32 and BFP", what);
    }
}

static void checkOutputShape(const shape_t *got, size_t ndim, const size_t *want,
                             const char *what) {
    if (got->numberOfDimensions != ndim) {
        failf("%s: calcOutputShape rank %zu != expected %zu", what, got->numberOfDimensions, ndim);
    }
    for (size_t i = 0; i < ndim; i++) {
        if (got->dimensions[i] != want[i]) {
            failf("%s: calcOutputShape dim %zu = %zu != expected %zu", what, i, got->dimensions[i],
                  want[i]);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Conv1d                                                                    */
/* ------------------------------------------------------------------------- */

/* conv1d / kernel / forward: conv1dKernelSymInt32 on SYM_INT32 codes (raw
 * accumulator-range int32 at s_in * s_w) or conv1dKernelBfp on unpacked-BFP
 * scratch (raw FLOAT32, D7). */
static void runConv1dKernelForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    requireKernelArith(arith, "conv1d/kernel/forward");
    kernel_t kernel;
    size_t groups;
    parseConvGeometry(ctx->spec, &kernel, &groups, NULL);
    convOperands_t ops = loadConvOperands(ctx, false, false, true, arith.type, &kernel, groups, 0);

    windowGeometry1d_t geom = windowGeometry1dCalc(ops.input->shape->dimensions[2], &kernel);
    size_t outDims[3] = {ops.input->shape->dimensions[0], ops.weight->shape->dimensions[0],
                         geom.outputLength};
    tensor_t *out;
    if (arith.type == ARITH_BFP) {
        out = makeFloatWire(3, outDims);
        conv1dKernelBfp(ops.input, ops.weight, ops.bias, &kernel, groups, out);
    } else {
        out = makeRawWire(3, outDims, arith.roundingMode);
        conv1dKernelSymInt32(ops.input, ops.weight, ops.bias, &kernel, groups, out, NULL);
    }
    addOutput("output", out);
    freeConvOperands(&ops);
}

/* Hand-wired Conv1d layer (test/unit/support/include/BorrowedLayer.h pattern):
 * borrowed parameter_t's, every math slot = spec.arith, produced wires =
 * the given storage configs, the packed-grad D1 accumulate modes at the
 * factory defaults (dynamic-rescale weights, fixed-scale bias -- on FLOAT32
 * grads both are the exact float add). The OUT_WRITE width-restore takes the
 * wire WIDTH from the target qConfig, its ROUNDING is operation-owned (#282):
 * spec.arith.rounding. */
static void wireConv1dLayer(layer_t *layer, layerConfig_t *layerCfg, conv1dConfig_t *cfg,
                            kernel_t *kernel, parameter_t *weights, parameter_t *bias,
                            size_t groups, arithmetic_t arith, quantization_t *outputQ,
                            quantization_t *propLossQ) {
    initConv1dConfigWithWeightsAndBias(cfg, kernel, weights, bias, groups, outputQ, outputQ,
                                       outputQ, propLossQ);
    cfg->forwardMath = arith;
    cfg->weightGradMath = arith;
    cfg->biasGradMath = arith;
    cfg->propLossMath = arith;
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    cfg->ownsQuantizations = false;
    layerCfg->conv1d = cfg;
    initLayer(layer, CONV1D, layerCfg);
}

/* conv1d / layer / forward: layerFunctions[CONV1D].forward through the funnel. */
static void runConv1dLayerForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    kernel_t kernel;
    size_t groups;
    parseConvGeometry(ctx->spec, &kernel, &groups, NULL);
    convOperands_t ops = loadConvOperands(ctx, false, false, false, arith.type, &kernel, groups, 0);
    windowGeometry1d_t geom = windowGeometry1dCalc(ops.input->shape->dimensions[2], &kernel);
    size_t outDims[3] = {ops.input->shape->dimensions[0], ops.weight->shape->dimensions[0],
                         geom.outputLength};
    quantization_t *outputQ =
        makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"), "spec.output",
                         elemCount3(outDims[0], outDims[1], outDims[2]));

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    conv1dConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireConv1dLayer(&layer, &layerCfg, &cfg, &kernel, weights, bias, groups, arith, outputQ,
                    outputQ);

    size_t shapeDims[3];
    size_t shapeOrder[3];
    shape_t outShape = {
        .numberOfDimensions = 3, .dimensions = shapeDims, .orderOfDimensions = shapeOrder};
    layerFunctions[CONV1D].calcOutputShape(&layer, ops.input->shape, &outShape);
    checkOutputShape(&outShape, 3, outDims, "conv1d");
    tensor_t *out = makeTensor(3, outDims, getQLike(outputQ));

    layerFunctions[CONV1D].forward(&layer, ops.input, out, NULL);
    addOutput("output", out);

    freeTensor(ops.input);
    freeParameter(weights);
    if (bias != NULL) {
        freeParameter(bias);
    }
    freeQuantization(outputQ);
}

/* conv1d / layer / {dx, weightGrad}: ONE layerFunctions[CONV1D].backward
 * emitting propLoss (OUT_WRITE onto spec.output, the dx wire: FLOAT32 raw or
 * a BFP wire), weightGrad and biasGrad (ACC epilogues into zeroed FLOAT32
 * grads = the kernels' raw output). */
static void runConv1dLayerBackward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    kernel_t kernel;
    size_t groups;
    parseConvGeometry(ctx->spec, &kernel, &groups, NULL);
    size_t outLen = convForwardOutLen(ctx, false, &kernel, 0);
    convOperands_t ops =
        loadConvOperands(ctx, false, true, false, arith.type, &kernel, groups, outLen);
    const size_t *in = ops.input->shape->dimensions;
    quantization_t *propLossQ = makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"),
                                                 "spec.output", elemCount3(in[0], in[1], in[2]));
    const char *gradStorage = jsonStringOr(ctx->spec, "gradStorage", "FLOAT32", "spec");
    if (strcmp(gradStorage, "FLOAT32") != 0) {
        failf("spec.gradStorage \"%s\": only FLOAT32 grad storage is implemented", gradStorage);
    }

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    conv1dConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireConv1dLayer(&layer, &layerCfg, &cfg, &kernel, weights, bias, groups, arith, propLossQ,
                    propLossQ);

    size_t dxDims[3] = {in[0], in[1], in[2]};
    tensor_t *propLoss = makeTensor(3, dxDims, getQLike(propLossQ));
    layerFunctions[CONV1D].backward(&layer, ops.input, ops.loss, propLoss, NULL);
    addOutput("propLoss", propLoss);
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
/* Conv1dTransposed (VALID + outputPadding)                                  */
/* ------------------------------------------------------------------------- */

/* convt1d / kernel / forward: convTranspose1dKernelBfpGather (D9 gather) on
 * unpacked-BFP scratch; raw FLOAT32 out. */
static void runConvT1dKernelForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    if (arith.type != ARITH_BFP) {
        failf("convt1d/kernel/forward: only arith.type BFP is implemented");
    }
    kernel_t kernel;
    size_t groups;
    size_t outputPadding;
    parseConvGeometry(ctx->spec, &kernel, &groups, &outputPadding);
    convOperands_t ops = loadConvOperands(ctx, true, false, true, arith.type, &kernel, groups, 0);
    size_t outLen =
        convTranspose1dOutputLength(ops.input->shape->dimensions[2], &kernel, outputPadding);
    size_t outDims[3] = {ops.input->shape->dimensions[0], ops.weight->shape->dimensions[1] * groups,
                         outLen};
    tensor_t *out = makeFloatWire(3, outDims);
    convTranspose1dKernelBfpGather(ops.input, ops.weight, ops.bias, &kernel, groups, outputPadding,
                                   out);
    addOutput("output", out);
    freeConvOperands(&ops);
}

static void wireConvT1dLayer(layer_t *layer, layerConfig_t *layerCfg, conv1dTransposedConfig_t *cfg,
                             kernel_t *kernel, parameter_t *weights, parameter_t *bias,
                             size_t groups, size_t outputPadding, arithmetic_t arith,
                             quantization_t *outputQ, quantization_t *propLossQ) {
    initConv1dTransposedConfigWithWeightsAndBias(cfg, kernel, weights, bias, groups, outputPadding,
                                                 outputQ, outputQ, outputQ, propLossQ);
    cfg->forwardMath = arith;
    cfg->weightGradMath = arith;
    cfg->biasGradMath = arith;
    cfg->propLossMath = arith;
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    cfg->ownsQuantizations = false;
    layerCfg->conv1dTransposed = cfg;
    initLayer(layer, CONV1D_TRANSPOSED, layerCfg);
}

/* convt1d / layer / forward: layerFunctions[CONV1D_TRANSPOSED].forward. */
static void runConvT1dLayerForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    kernel_t kernel;
    size_t groups;
    size_t outputPadding;
    parseConvGeometry(ctx->spec, &kernel, &groups, &outputPadding);
    convOperands_t ops = loadConvOperands(ctx, true, false, false, arith.type, &kernel, groups, 0);
    size_t outLen =
        convTranspose1dOutputLength(ops.input->shape->dimensions[2], &kernel, outputPadding);
    size_t outDims[3] = {ops.input->shape->dimensions[0], ops.weight->shape->dimensions[1] * groups,
                         outLen};
    quantization_t *outputQ =
        makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"), "spec.output",
                         elemCount3(outDims[0], outDims[1], outDims[2]));

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    conv1dTransposedConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireConvT1dLayer(&layer, &layerCfg, &cfg, &kernel, weights, bias, groups, outputPadding, arith,
                     outputQ, outputQ);

    size_t shapeDims[3];
    size_t shapeOrder[3];
    shape_t outShape = {
        .numberOfDimensions = 3, .dimensions = shapeDims, .orderOfDimensions = shapeOrder};
    layerFunctions[CONV1D_TRANSPOSED].calcOutputShape(&layer, ops.input->shape, &outShape);
    checkOutputShape(&outShape, 3, outDims, "convt1d");
    tensor_t *out = makeTensor(3, outDims, getQLike(outputQ));

    layerFunctions[CONV1D_TRANSPOSED].forward(&layer, ops.input, out, NULL);
    addOutput("output", out);

    freeTensor(ops.input);
    freeParameter(weights);
    if (bias != NULL) {
        freeParameter(bias);
    }
    freeQuantization(outputQ);
}

/* convt1d / layer / {dx, weightGrad}: one layerFunctions[CONV1D_TRANSPOSED]
 * .backward, outputs as for conv1d. */
static void runConvT1dLayerBackward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    kernel_t kernel;
    size_t groups;
    size_t outputPadding;
    parseConvGeometry(ctx->spec, &kernel, &groups, &outputPadding);
    size_t outLen = convForwardOutLen(ctx, true, &kernel, outputPadding);
    convOperands_t ops =
        loadConvOperands(ctx, true, true, false, arith.type, &kernel, groups, outLen);
    const size_t *in = ops.input->shape->dimensions;
    quantization_t *propLossQ = makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"),
                                                 "spec.output", elemCount3(in[0], in[1], in[2]));
    const char *gradStorage = jsonStringOr(ctx->spec, "gradStorage", "FLOAT32", "spec");
    if (strcmp(gradStorage, "FLOAT32") != 0) {
        failf("spec.gradStorage \"%s\": only FLOAT32 grad storage is implemented", gradStorage);
    }

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    conv1dTransposedConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireConvT1dLayer(&layer, &layerCfg, &cfg, &kernel, weights, bias, groups, outputPadding, arith,
                     propLossQ, propLossQ);

    size_t dxDims[3] = {in[0], in[1], in[2]};
    tensor_t *propLoss = makeTensor(3, dxDims, getQLike(propLossQ));
    layerFunctions[CONV1D_TRANSPOSED].backward(&layer, ops.input, ops.loss, propLoss, NULL);
    addOutput("propLoss", propLoss);
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
/* Linear                                                                    */
/* ------------------------------------------------------------------------- */

typedef struct linearOperands {
    tensor_t *input;
    tensor_t *weight;
    tensor_t *bias; /* NULL without a bias operand */
    tensor_t *loss; /* NULL unless requested */
} linearOperands_t;

/* The checks both Linear tiers run before touching a library entry: ranks
 * (input [B, inF], weight [outF, inF], bias [outF], loss [B, outF]), the
 * shared dims, and the storage dtypes each tier accepts (requireOperandDtype).
 * Code range is checked at load (loadOperand). */
static linearOperands_t loadLinearOperands(const runCtx_t *ctx, bool withLoss, bool kernelTier,
                                           arithmeticType_t arith) {
    const jsonNode_t *operands = jsonRequireObject(ctx->spec, "operands", "spec");
    bfpForm_t form = kernelTier ? BFP_SCRATCH : BFP_PACKED;
    linearOperands_t ops;
    ops.input = loadOperand(operands, "input", ctx->inDir, true, form);
    ops.weight = loadOperand(operands, "weight", ctx->inDir, true, form);
    ops.bias = loadOperand(operands, "bias", ctx->inDir, false, form);
    ops.loss = withLoss ? loadOperand(operands, "loss", ctx->inDir, true, form) : NULL;

    requireRank(ops.input, 2, "spec.operands.input");
    requireRank(ops.weight, 2, "spec.operands.weight");
    requireOperandDtype(ops.input, kernelTier, arith, false, "spec.operands.input");
    requireOperandDtype(ops.weight, kernelTier, arith, true, "spec.operands.weight");
    const size_t batch = ops.input->shape->dimensions[0];
    const size_t inF = ops.input->shape->dimensions[1];
    const size_t outF = ops.weight->shape->dimensions[0];
    if (ops.weight->shape->dimensions[1] != inF) {
        failf("spec.operands.weight dim 1 (%zu) != spec.operands.input dim 1 (inF=%zu)",
              ops.weight->shape->dimensions[1], inF);
    }
    if (ops.bias != NULL) {
        requireRank(ops.bias, 1, "spec.operands.bias");
        requireOperandDtype(ops.bias, kernelTier, arith, true, "spec.operands.bias");
        if (ops.bias->shape->dimensions[0] != outF) {
            failf("spec.operands.bias length (%zu) != spec.operands.weight dim 0 (outF=%zu)",
                  ops.bias->shape->dimensions[0], outF);
        }
    }
    if (ops.loss != NULL) {
        requireRank(ops.loss, 2, "spec.operands.loss");
        requireOperandDtype(ops.loss, kernelTier, arith, false, "spec.operands.loss");
        if (ops.loss->shape->dimensions[0] != batch || ops.loss->shape->dimensions[1] != outF) {
            failf("spec.operands.loss must be [B=%zu, outF=%zu], got [%zu, %zu]", batch, outF,
                  ops.loss->shape->dimensions[0], ops.loss->shape->dimensions[1]);
        }
    }
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

/* linear / kernel / forward: linearForwardSymInt32 / linearForwardBfp -- the
 * entries the funnel's kernel adapters call (transpose dance around the
 * matmul). Calling them rather than matmul directly keeps the weight
 * orientation under test: Leg A's y = x W^T is the documented contract, not
 * the driver's. SYM: raw accumulator-range int32 at s_in * s_w, the bias
 * refolded per column. BFP: raw FLOAT32, the bias dequant-seeded (D7). */
static void runLinearKernelForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    requireKernelArith(arith, "linear/kernel/forward");
    linearOperands_t ops = loadLinearOperands(ctx, false, true, arith.type);
    size_t outDims[2] = {ops.input->shape->dimensions[0], ops.weight->shape->dimensions[0]};
    tensor_t *out;
    if (arith.type == ARITH_BFP) {
        out = makeFloatWire(2, outDims);
        linearForwardBfp(ops.weight, ops.bias, ops.input, out);
    } else {
        out = makeRawWire(2, outDims, arith.roundingMode);
        linearForwardSymInt32(ops.weight, ops.bias, ops.input, out, NULL);
    }
    addOutput("output", out);
    freeLinearOperands(&ops);
}

/* linear / kernel / dx: linearCalcPropLossSymInt32 (matmulSymInt32Tensors) /
 * linearCalcPropLossBfp: (loss [B, outF], W [outF, inF]) -> raw [B, inF]. */
static void runLinearKernelDx(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    requireKernelArith(arith, "linear/kernel/dx");
    linearOperands_t ops = loadLinearOperands(ctx, true, true, arith.type);
    size_t outDims[2] = {ops.input->shape->dimensions[0], ops.input->shape->dimensions[1]};
    tensor_t *out;
    if (arith.type == ARITH_BFP) {
        out = makeFloatWire(2, outDims);
        linearCalcPropLossBfp(ops.loss, ops.weight, out);
    } else {
        out = makeRawWire(2, outDims, arith.roundingMode);
        linearCalcPropLossSymInt32(ops.weight, ops.loss, out);
    }
    addOutput("propLoss", out);
    freeLinearOperands(&ops);
}

/* linear / kernel / weightGrad: loss^T @ x -> raw [outF, inF] (SYM: int32 at
 * s_loss * s_in; BFP: FLOAT32); with a bias operand also the raw bias-grad
 * sum over the batch (SYM: int32 at s_loss; BFP: FLOAT32 sum-fold). Both are
 * what the funnel's backward kernel adapters emit before the ACC epilogue. */
static void runLinearKernelWeightGrad(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    requireKernelArith(arith, "linear/kernel/weightGrad");
    linearOperands_t ops = loadLinearOperands(ctx, true, true, arith.type);
    size_t wgDims[2] = {ops.weight->shape->dimensions[0], ops.weight->shape->dimensions[1]};
    size_t bgDims[1] = {ops.weight->shape->dimensions[0]};
    if (arith.type == ARITH_BFP) {
        tensor_t *wg = makeFloatWire(2, wgDims);
        linearCalcWeightGradsBfp(ops.loss, ops.input, wg);
        addOutput("weightGrad", wg);
        if (ops.bias != NULL) {
            tensor_t *bg = makeFloatWire(1, bgDims);
            linearCalcBiasGradsBfp(ops.loss, bg);
            addOutput("biasGrad", bg);
        }
    } else {
        tensor_t *wg = makeRawWire(2, wgDims, arith.roundingMode);
        linearCalcWeightGradsSymInt32(ops.loss, ops.input, wg);
        addOutput("weightGrad", wg);
        if (ops.bias != NULL) {
            tensor_t *bg = makeRawWire(1, bgDims, arith.roundingMode);
            linearCalcBiasGradsSymInt32(bg, ops.loss);
            addOutput("biasGrad", bg);
        }
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
 * funnel. A FLOAT32 input is staged by the prologue (int12 under SYM math,
 * per-tensor BFP at the weight's widths under BFP math); packed SYM / BFP
 * weights and bias are unpacked / borrowed there. The output carries
 * spec.output (width from the target, rounding op-owned per #282). */
static void runLinearLayerForward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    linearOperands_t ops = loadLinearOperands(ctx, false, false, arith.type);
    size_t outDims[2] = {ops.input->shape->dimensions[0], ops.weight->shape->dimensions[0]};
    quantization_t *outputQ = makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"),
                                               "spec.output", outDims[0] * outDims[1]);

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    linearConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireLinearLayer(&layer, &layerCfg, &cfg, weights, bias, arith, outputQ, outputQ);

    size_t shapeDims[2];
    size_t shapeOrder[2];
    shape_t outShape = {
        .numberOfDimensions = 2, .dimensions = shapeDims, .orderOfDimensions = shapeOrder};
    layerFunctions[LINEAR].calcOutputShape(&layer, ops.input->shape, &outShape);
    checkOutputShape(&outShape, 2, outDims, "linear");
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
 * FLOAT32 grads -- spec.gradStorage, FLOAT32 only, the production default).
 * The Python side evaluates the output its cell's direction names; the
 * others ride along in result.json. */
static void runLinearLayerBackward(const runCtx_t *ctx) {
    arithmetic_t arith = parseArithmetic(ctx->spec);
    const char *gradStorage = jsonStringOr(ctx->spec, "gradStorage", "FLOAT32", "spec");
    if (strcmp(gradStorage, "FLOAT32") != 0) {
        failf("spec.gradStorage \"%s\": only FLOAT32 grad storage is implemented", gradStorage);
    }
    linearOperands_t ops = loadLinearOperands(ctx, true, false, arith.type);
    size_t dxDims[2] = {ops.input->shape->dimensions[0], ops.input->shape->dimensions[1]};
    quantization_t *propLossQ = makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"),
                                                 "spec.output", dxDims[0] * dxDims[1]);

    parameter_t *weights = parameterInit(ops.weight, gradInitFloat(ops.weight, NULL));
    parameter_t *bias =
        ops.bias != NULL ? parameterInit(ops.bias, gradInitFloat(ops.bias, NULL)) : NULL;
    linearConfig_t cfg;
    layerConfig_t layerCfg;
    layer_t layer;
    wireLinearLayer(&layer, &layerCfg, &cfg, weights, bias, arith, propLossQ, propLossQ);

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
    tensor_t *input = loadOperand(operands, "input", ctx->inDir, true, BFP_PACKED);
    requireDtype(input, want, "spec.operands.input");
    return input;
}

static tensor_t *makeConvertTarget(const runCtx_t *ctx, const tensor_t *input, qtype_t want) {
    quantization_t *targetQ =
        makeQuantization(jsonRequireObject(ctx->spec, "output", "spec"), "spec.output",
                         calcNumberOfElementsByTensor((tensor_t *)input));
    if (targetQ->type != want) {
        failf("spec.output.dtype must be %s for this convert direction (got %s)", dtypeName(want),
              dtypeName(targetQ->type));
    }
    return makeTensor(input->shape->numberOfDimensions, input->shape->dimensions, targetQ);
}

/* convert / convert / encode: executeConvert FLOAT32 -> SYM_INT32, i.e. the
 * conversionMatrix[FLOAT32][SYM_INT32] encode; a bare conversion rounds by
 * the TARGET's storage roundingMode (#282), spec.output.rounding. */
static void runConvertEncode(const runCtx_t *ctx) {
    tensor_t *input = loadConvertInput(ctx, FLOAT32);
    tensor_t *out = makeConvertTarget(ctx, input, SYM_INT32);
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
    tensor_t *out = makeConvertTarget(ctx, input, SYM_INT32);
    executeConvert(input, out, NULL);
    addOutput("output", out);
    freeTensor(input);
}

/* convert / convert / requant_fixed: requantSymInt32TensorToScale into the
 * pre-set spec.output.scale (#192 D1 fixed-scale requant; not a matrix cell). */
static void runConvertRequantFixed(const runCtx_t *ctx) {
    tensor_t *input = loadConvertInput(ctx, SYM_INT32);
    tensor_t *out = makeConvertTarget(ctx, input, SYM_INT32);
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

/* convert / convert / pack_bfp: executeConvert FLOAT32 -> BFP, the
 * conversionMatrix[FLOAT32][BFP] value-domain pack (arithmetic-bfp sections
 * 1-3: per-group absmax -> exponent rule -> D6 clamp -> round/clamp); the
 * target's widths/geometry come from spec.output, its rounding is the
 * target's storage mode (#282, bare conversion). Output: codes + exponents. */
static void runConvertPackBfp(const runCtx_t *ctx) {
    tensor_t *input = loadConvertInput(ctx, FLOAT32);
    tensor_t *out = makeConvertTarget(ctx, input, BFP);
    executeConvert(input, out, NULL);
    addOutput("output", out);
    freeTensor(input);
}

typedef struct opEntry {
    const char *op;
    const char *tier;
    const char *direction;
    opRunFn_t run;
} opEntry_t;

/* Later steps append rows here: the acc, optim and sr modes, and SYM/BFP grad
 * storage for the backward runners. */
static const opEntry_t opTable[] = {
    {"conv1d", "kernel", "forward", runConv1dKernelForward},
    {"conv1d", "layer", "forward", runConv1dLayerForward},
    {"conv1d", "layer", "dx", runConv1dLayerBackward},
    {"conv1d", "layer", "weightGrad", runConv1dLayerBackward},
    {"convt1d", "kernel", "forward", runConvT1dKernelForward},
    {"convt1d", "layer", "forward", runConvT1dLayerForward},
    {"convt1d", "layer", "dx", runConvT1dLayerBackward},
    {"convt1d", "layer", "weightGrad", runConvT1dLayerBackward},
    {"linear", "kernel", "forward", runLinearKernelForward},
    {"linear", "kernel", "dx", runLinearKernelDx},
    {"linear", "kernel", "weightGrad", runLinearKernelWeightGrad},
    {"linear", "layer", "forward", runLinearLayerForward},
    {"linear", "layer", "dx", runLinearLayerBackward},
    {"linear", "layer", "weightGrad", runLinearLayerBackward},
    {"convert", "convert", "encode", runConvertEncode},
    {"convert", "convert", "requant_dynamic", runConvertRequantDynamic},
    {"convert", "convert", "requant_fixed", runConvertRequantFixed},
    {"convert", "convert", "pack_bfp", runConvertPackBfp},
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
