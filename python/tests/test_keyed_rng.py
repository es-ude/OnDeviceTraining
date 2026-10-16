"""Test the Python mirror of the keyed RNG core in examples/_shared/keyed_rng.py.

The golden values are those of test/unit/rng/UnitTestRNG.c. The mirror is also
compared with the C code through a harness compiled against src/rng/RNG.c."""
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

# Make examples/ importable from python/tests/.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.keyed_rng import (  # noqa: E402
    TAGS,
    UINT32_MAX,
    fold,
    key_from_seed,
    keyed_shuffle,
    next_bounded,
    next_u32,
    next_unit_float,
    open_stream,
    xorshift32_next,
)

# Threefry-2x32 known answers at 20 rounds, read as fold((KEY0, KEY1), CTR0, CTR1).
# Source: tests/kat_vectors of https://github.com/DEShawResearch/random123 at
# commit 9545ff64 (Random123, Copyright 2010-2011 D. E. Shaw Research, BSD-3-Clause).
KNOWN_ANSWERS = [
    ((0x00000000, 0x00000000), 0x00000000, 0x00000000, (0x6B200159, 0x99BA4EFE)),
    ((0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF, 0xFFFFFFFF, (0x1CB996FC, 0xBB002BE7)),
    ((0x13198A2E, 0x03707344), 0x243F6A88, 0x85A308D3, (0xC4923A9C, 0x483DF7A0)),
]

# fold(fold(key_from_seed(42), 7, 0), 3, 5), the derived key of the C tests.
DERIVED_KEY = (0x929C8A01, 0xEE4A7510)


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def test_xorshift32_next_known_vectors():
    """The first four steps from state 1 with shifts (13, 17, 5)."""
    state, drawn = 1, []
    for _ in range(4):
        state = xorshift32_next(state)
        drawn.append(state)
    assert drawn == [270369, 67634689, 2647435461, 307599695]


def test_fold_matches_threefry_known_answers():
    for parent, tag, index, child in KNOWN_ANSWERS:
        assert fold(parent, tag, index) == child


def test_fold_two_edge_path_matches_reference():
    first = fold((0x13198A2E, 0x03707344), 7, 0)
    assert first == (0x627945FD, 0xA1F54519)
    assert fold(first, 3, 5) == (0x2B6FC7A8, 0x5D69D71D)


def test_key_from_seed_matches_reference():
    assert key_from_seed(0) == (0x6B200159, 0x99BA4EFE)
    assert key_from_seed(42) == (0xA608025A, 0x0EF6F004)
    assert key_from_seed(1 << 32) == (0x375F238F, 0xCDDB151D)
    assert fold(fold(key_from_seed(42), 7, 0), 3, 5) == DERIVED_KEY


def test_open_stream_reduces_the_key_to_a_non_zero_state():
    """w[0], else w[1], else 1: state 0 would stick, so it never occurs."""
    assert open_stream(DERIVED_KEY) == 0x929C8A01
    assert open_stream((0, 0xEE4A7510)) == 0xEE4A7510
    assert open_stream((0, 0)) == 1


def test_draws_match_reference_on_the_derived_key():
    state, u32 = open_stream(DERIVED_KEY), []
    for _ in range(4):
        state, value = next_u32(state)
        u32.append(value)
    assert u32 == [0x7849D60F, 0x138AFE2B, 0xC5B8B98C, 0x83A349C8]
    state, bits = open_stream(DERIVED_KEY), []
    for _ in range(4):
        state, value = next_unit_float(state)
        bits.append(_float_bits(value))
    assert bits == [0x3EF093AC, 0x3D9C57F0, 0x3F45B8B9, 0x3F03A349]
    state, drawn = open_stream(DERIVED_KEY), []
    for bound in (10, 10, 10, 10, 1000, 1000, 1, UINT32_MAX):
        state, value = next_bounded(state, bound)
        drawn.append(value)
    assert drawn == [7, 9, 6, 2, 485, 721, 0, 1286403703]


def test_out_of_range_inputs_raise():
    """C would truncate these silently; the mirror refuses them."""
    for call in (
        lambda: key_from_seed(-1),
        lambda: key_from_seed(1 << 64),
        lambda: fold((0, 1 << 32), 1, 0),
        lambda: fold((0, 0), 1 << 32, 0),
        lambda: fold((0, 0), 1, -1),
        lambda: open_stream((-1, 0)),
        lambda: next_u32(0),
        lambda: next_unit_float(1 << 32),
        lambda: next_bounded(1, -1),
    ):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError, match="rngNextBounded: bound is 0"):
        next_bounded(1, 0)
    with pytest.raises(ValueError, match="rngNextBounded: bound exceeds UINT32_MAX"):
        next_bounded(1, UINT32_MAX + 1)


def test_keyed_shuffle_is_fisher_yates_from_the_identity_order():
    """From the last slot down, slot i swaps with a draw below i + 1. The derived
    key's draws for the bounds 5, 4, 3, 2 are 2, 3, 1, 0, so [0, 1, 2, 3, 4]
    becomes [0, 1, 4, 3, 2], stays, then [0, 4, 1, 3, 2], then [4, 0, 1, 3, 2]."""
    state, drawn = open_stream(DERIVED_KEY), []
    for bound in (5, 4, 3, 2):
        state, value = next_bounded(state, bound)
        drawn.append(value)
    assert drawn == [2, 3, 1, 0]
    assert keyed_shuffle(5, DERIVED_KEY) == [4, 0, 1, 3, 2]


def test_keyed_shuffle_draws_nothing_below_two_samples():
    assert keyed_shuffle(0, DERIVED_KEY) == []
    assert keyed_shuffle(1, DERIVED_KEY) == [0]
    with pytest.raises(ValueError):
        keyed_shuffle(-1, DERIVED_KEY)


def _c_tags(tmp_path: Path) -> dict[str, int]:
    """RNG.h's tag table as the C preprocessor of the default build defines it."""
    probe = tmp_path / "tags.c"
    probe.write_text('#include "RNG.h"\n')
    result = subprocess.run(
        ["cc", "-std=c11", "-dM", "-E", f"-I{REPO_ROOT / 'src' / 'rng' / 'include'}", str(probe)],
        check=True, capture_output=True, text=True)
    tags = {}
    for line in result.stdout.splitlines():
        if line.startswith("#define RNG_TAG_"):
            match = re.fullmatch(r"#define RNG_TAG_([A-Z0-9_]+) ([0-9]+)u", line)
            assert match, f"a tag define this test cannot read: {line!r}"
            tags[match.group(1)] = int(match.group(2))
    return tags


def test_tags_mirror_the_c_tag_table(tmp_path):
    """Read through the preprocessor, so a tag added to RNG.h but not to TAGS fails
    here. A tag behind a build flag is checked only in a run with that flag."""
    c_tags = _c_tags(tmp_path)
    assert c_tags, "no RNG_TAG_ define reached the preprocessor"
    assert TAGS == c_tags


def test_tags_are_nonzero_and_unique():
    """0 is never a tag and no two tags share a value, so every key path decodes
    one way. Gaps are allowed: a tag behind a build flag keeps its value."""
    values = list(TAGS.values())
    assert all(1 <= value <= UINT32_MAX for value in values)
    assert len(set(values)) == len(values)


@pytest.fixture(scope="module")
def harness(tmp_path_factory) -> Path:
    """Compile the C harness against src/rng/RNG.c once per module."""
    binary = tmp_path_factory.mktemp("keyed_rng") / "harness"
    subprocess.run(
        [
            "cc", "-std=c11", "-O0",
            # As CMake's Common target: a bad bound prints its named error.
            "-DDEBUG_MODE_ERROR=true",
            f"-I{REPO_ROOT / 'src' / 'rng' / 'include'}",
            f"-I{REPO_ROOT / 'src' / 'common' / 'include'}",
            str(REPO_ROOT / "examples" / "_shared" / "verify_keyed_rng_harness.c"),
            str(REPO_ROOT / "src" / "rng" / "RNG.c"),
            "-o", str(binary),
        ],
        check=True,
    )
    return binary


def _run(harness: Path, *args: int | str) -> list[int]:
    result = subprocess.run([str(harness), *map(str, args)], check=True, capture_output=True,
                            text=True)
    return [int(tok, 16) for tok in result.stdout.split()]


def test_mirror_matches_c_key_from_seed(harness):
    for seed in (0, 1, 42, 0xFFFFFFFF, 1 << 32, 0x0123456789ABCDEF, (1 << 64) - 1):
        assert tuple(_run(harness, "key", seed)) == key_from_seed(seed), seed


def test_mirror_matches_c_fold(harness):
    cases = [(parent, tag, index) for parent, tag, index, _ in KNOWN_ANSWERS]
    cases += [(DERIVED_KEY, 7, 1), ((0, 0xEE4A7510), 0, UINT32_MAX), ((5, 0), 3, 9)]
    for parent, tag, index in cases:
        assert tuple(_run(harness, "fold", *parent, tag, index)) == fold(parent, tag, index)


KEYS = [DERIVED_KEY, (1, 0xDEADBEEF), (0, 0xEE4A7510), (0, 0), (UINT32_MAX, 0), (0xFC7D0B53, 0)]


def test_mirror_matches_c_u32_and_float(harness):
    for key in KEYS:
        state, u32, bits = open_stream(key), [], []
        for _ in range(200):
            state, value = next_u32(state)
            u32.append(value)
        state = open_stream(key)
        for _ in range(200):
            state, value = next_unit_float(state)
            bits.append(_float_bits(value))
        assert _run(harness, "u32", *key, 200) == u32, key
        assert _run(harness, "float", *key, 200) == bits, key


def test_mirror_matches_c_bounded(harness):
    """(0xFC7D0B53, 0)'s first draw is exactly the rejection limit for 1000."""
    bounds = [1000, 1, UINT32_MAX, 10, (1 << 31) + 1, 7, 1000, 3]
    for key in KEYS:
        state, drawn = open_stream(key), []
        for bound in bounds:
            state, value = next_bounded(state, bound)
            drawn.append(value)
        result = subprocess.run([str(harness), "bounded", *map(str, key), *map(str, bounds)],
                                check=True, capture_output=True, text=True)
        assert [int(tok) for tok in result.stdout.split()] == drawn, key


def test_mirror_matches_c_keyed_shuffle(harness):
    """n = 0 and 1 draw nothing; 1000 crosses many rejection limits."""
    for key in KEYS:
        for n in (0, 1, 2, 3, 8, 100, 1000):
            result = subprocess.run([str(harness), "shuffle", *map(str, key), str(n)],
                                    check=True, capture_output=True, text=True)
            assert [int(tok) for tok in result.stdout.split()] == keyed_shuffle(n, key), (key, n)


# The harness's largest shuffle on a 64-bit host: SIZE_MAX / sizeof(size_t).
SHUFFLE_LIMIT = ((1 << 64) - 1) // 8


def _shuffle(harness: Path, n: int) -> subprocess.CompletedProcess:
    return subprocess.run([str(harness), "shuffle", "1", "0", str(n)], capture_output=True,
                          text=True)


def test_harness_names_a_shuffle_whose_buffer_size_would_wrap(harness):
    result = _shuffle(harness, SHUFFLE_LIMIT + 1)
    assert result.returncode == 2
    assert "would overflow the buffer size" in result.stderr


def test_harness_names_a_failed_allocation(harness):
    """SHUFFLE_LIMIT passes the size check and asks for about 2**64 bytes, which no
    64-bit host grants."""
    result = _shuffle(harness, SHUFFLE_LIMIT)
    assert result.returncode == 2
    assert "cannot allocate" in result.stderr
