"""Test examples/_shared/npy_writer.{h,c} by compiling a tiny C harness
that calls npyWriteFloat32 / npyWriteInt32 with known buffers, then
np.load()-ing the output and asserting it matches.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compile_writer_test_harness(harness_src: Path, *, sanitize: bool = False) -> str:
    """Compile harness_src against examples/_shared/npy_writer.c.

    When sanitize=True, instrument with AddressSanitizer + UndefinedBehaviorSanitizer
    so OOB writes / unsigned overflows are caught at runtime. Requires clang.
    """
    writer_c = REPO_ROOT / "examples" / "_shared" / "npy_writer.c"
    writer_h = REPO_ROOT / "examples" / "_shared"
    binary = tempfile.NamedTemporaryFile(suffix="", delete=False).name
    cmd = [
        "cc" if not sanitize else "clang",
        "-std=c11", "-O0", f"-I{writer_h}",
        str(harness_src), str(writer_c), "-o", binary,
    ]
    if sanitize:
        cmd[1:1] = [
            "-fsanitize=address,undefined",
            "-fno-sanitize=function",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
            "-g",
        ]
    subprocess.run(cmd, check=True)
    return binary


def test_npy_writer_round_trips_float32(tmp_path):
    harness = tmp_path / "harness.c"
    out = tmp_path / "out.npy"
    harness.write_text(f"""
        #include "npy_writer.h"
        #include <stddef.h>
        int main(void) {{
            float data[] = {{1.5f, -2.25f, 0.0f, 3.125f, 4.0f, -5.5f}};
            size_t shape[] = {{2, 3}};
            return npyWriteFloat32("{out}", data, shape, 2);
        }}
    """)
    binary = _compile_writer_test_harness(harness)
    subprocess.run([binary], check=True)
    arr = np.load(out)
    assert arr.dtype == np.float32
    assert arr.shape == (2, 3)
    assert np.array_equal(arr, np.array([[1.5, -2.25, 0.0], [3.125, 4.0, -5.5]], dtype=np.float32))


def test_npy_writer_round_trips_int32(tmp_path):
    harness = tmp_path / "harness.c"
    out = tmp_path / "out.npy"
    harness.write_text(f"""
        #include "npy_writer.h"
        #include <stddef.h>
        #include <stdint.h>
        int main(void) {{
            int32_t data[] = {{0, 1, 2, 3, 4, 5, 6, 7}};
            size_t shape[] = {{8}};
            return npyWriteInt32("{out}", data, shape, 1);
        }}
    """)
    binary = _compile_writer_test_harness(harness)
    subprocess.run([binary], check=True)
    arr = np.load(out)
    assert arr.dtype == np.int32
    assert arr.shape == (8,)
    assert np.array_equal(arr, np.arange(8, dtype=np.int32))


def test_npy_writer_rejects_high_rank_shape_that_overflows_buf(tmp_path):
    """High-rank shapes serialize to >256 chars and used to silently overflow
    the internal shape_buf[256] (issue #165). Compiled with ASan+UBSan so any
    OOB write or unsigned underflow trips the sanitizers; the test also asserts
    that npyWriteFloat32 returns a non-zero rc instead of writing a corrupt
    file."""
    if shutil.which("clang") is None:
        pytest.skip("clang required for sanitizer build")
    harness = tmp_path / "harness.c"
    out = tmp_path / "out.npy"
    # ndim=200 with all-1 dims: leading "(" + 200 * "1," = 1 + 400 = 401 chars,
    # well past the 256-byte shape_buf. Without the fix, the unsigned subtraction
    # `sizeof(shape_buf) - shape_len` underflows and snprintf writes OOB.
    harness.write_text(f"""
        #include "npy_writer.h"
        #include <stddef.h>
        int main(void) {{
            size_t shape[200];
            for (size_t i = 0; i < 200; ++i) shape[i] = 1;
            float data[1] = {{1.0f}};
            return npyWriteFloat32("{out}", data, shape, 200);
        }}
    """)
    binary = _compile_writer_test_harness(harness, sanitize=True)
    result = subprocess.run(
        [binary],
        env={
            "ASAN_OPTIONS": "abort_on_error=0:halt_on_error=1:strict_string_checks=1",
            "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1",
        },
        capture_output=True,
        text=True,
    )
    # rc must be non-zero (the writer must detect shape-buffer truncation),
    # and the sanitizers must not have aborted.
    assert "AddressSanitizer" not in result.stderr, (
        f"ASan tripped on stack-buffer overflow:\n{result.stderr}"
    )
    assert "runtime error" not in result.stderr, (
        f"UBSan tripped on unsigned overflow / OOB index:\n{result.stderr}"
    )
    assert result.returncode != 0, (
        f"writeNpy returned 0 on a shape that overflows shape_buf; expected "
        f"a non-zero error code. stderr: {result.stderr!r}"
    )
