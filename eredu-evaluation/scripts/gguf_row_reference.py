#!/usr/bin/env python3
"""Freeze small row-decoder fixtures using hash-pinned upstream C equations.

Inputs are downloaded outside the source tree. No Eredu implementation is imported.
Requires a little-endian host and clang with _Float16 support. Output comparisons
use exact F32 bits with contraction disabled, including signed zero and infinity.
"""
import argparse
import hashlib
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

REVISION = "2145525a4081d66ff1a87cf43ef809f95a85ac0c"
QUANTS_HASH = "7878680cc60493f98469a116e9b14af8b84789292ccf891c249230b2fa3d157f"
IMPL_HASH = "43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3"
FORMATS = [(2, "q4_0", 18, 32), (3, "q4_1", 20, 32), (6, "q5_0", 22, 32),
           (7, "q5_1", 24, 32), (8, "q8_0", 34, 32),
           (12, "q4_K", 144, 256), (13, "q5_K", 176, 256), (14, "q6_K", 210, 256),
           (10, "q2_K", 84, 256), (11, "q3_K", 110, 256), (39, "mxfp4", 17, 32)]


def read_pinned(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"reference hash mismatch: {path}")
    return data.decode()


def function(source, signature):
    start = source.index(signature)
    return source[start:source.index("\n}", start) + 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quants", required=True)
    parser.add_argument("--impl", dest="implementation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("upstream reference harness requires little-endian storage")
    quants = read_pinned(args.quants, QUANTS_HASH)
    implementation = read_pinned(args.implementation, IMPL_HASH)
    # Storage declarations have exactly the pinned GGML layouts. The function
    # bodies below come verbatim from the verified references, not translations.
    source = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GGML_RESTRICT restrict
#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define K_SCALE_SIZE 12
#define QK_K 256
#define QK_MXFP4 32
typedef uint16_t ggml_half;
static float from_half(ggml_half bits) { _Float16 x; memcpy(&x, &bits, 2); return (float)x; }
#define GGML_FP16_TO_FP32(x) from_half(x)
typedef struct { ggml_half d; uint8_t qs[16]; } block_q4_0;
typedef struct { ggml_half d, m; uint8_t qs[16]; } block_q4_1;
typedef struct { ggml_half d; uint8_t qh[4], qs[16]; } block_q5_0;
typedef struct { ggml_half d, m; uint8_t qh[4], qs[16]; } block_q5_1;
typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;
typedef struct { ggml_half d, dmin; uint8_t scales[12], qs[128]; } block_q4_K;
typedef struct { ggml_half d, dmin; uint8_t scales[12], qh[32], qs[128]; } block_q5_K;
typedef struct { uint8_t ql[128], qh[64]; int8_t scales[16]; ggml_half d; } block_q6_K;
typedef struct { uint8_t scales[16], qs[64]; ggml_half d, dmin; } block_q2_K;
typedef struct { uint8_t hmask[32], qs[64], scales[12]; ggml_half d; } block_q3_K;
typedef struct { uint8_t e, qs[16]; } block_mxfp4;
static const int8_t kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
"""
    source += function(implementation, "static inline float ggml_e8m0_to_fp32_half(")
    source += "\n#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)\n"
    source += function(quants, "static inline void get_scale_min_k4(")
    for _, name, size, _ in FORMATS:
        source += f'\n_Static_assert(sizeof(block_{name}) == {size}, "block size");\n'
        source += function(quants, f"void dequantize_row_{name}(") + "\n"
    source += """
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    union { uint64_t alignment; uint8_t data[4096]; } input;
    float output[4096];
    size_t n = fread(input.data, 1, sizeof(input.data), stdin), count = 0;
    switch (atoi(argv[1])) {
"""
    for code, name, size, values in FORMATS:
        source += (f"case {code}: if (n % {size}) return 2; count = n / {size} * {values}; "
                   f"if (count > 4096) return 3; dequantize_row_{name}"
                   f"((const block_{name} *)input.data, output, count); break;\n")
    source += "default: return 4; } return fwrite(output, sizeof(float), count, stdout) == count ? 0 : 5; }\n"
    rng = random.Random(0x4E58)
    records = [f"# GGML {REVISION}; F32-bit tolerance 0; seed 0x4e58; three blocks per format"]
    with tempfile.TemporaryDirectory(prefix="gguf-row-reference-") as temporary:
        temporary = Path(temporary)
        c = temporary / "reference.c"
        binary = temporary / "reference"
        c.write_text(source)
        subprocess.run(["clang", "-std=c11", "-O0", "-ffp-contract=off", str(c), "-o", str(binary)], check=True)
        for code, name, size, values in FORMATS:
            payload = bytearray()
            for i, scale in enumerate([0.375, -0.8125, 16384.]):
                block = bytearray(rng.randrange(256) for _ in range(size))
                if name == "mxfp4":
                    block[0] = [0, 127, 255][i]
                else:
                    at = {"q2_K": 80, "q3_K": 108, "q6_K": 208}.get(name, 0)
                    block[at:at + 2] = struct.pack("<e", scale)
                    if name in ("q4_1", "q5_1", "q2_K", "q4_K", "q5_K"):
                        at = 82 if name == "q2_K" else 2
                        block[at:at + 2] = struct.pack("<e", [0.125, -0.3125, 3.][i])
                payload.extend(block)
            result = subprocess.run([str(binary), str(code)], input=payload, capture_output=True, check=True).stdout
            assert len(result) == 3 * values * 4
            records.append(f"{code}|{payload.hex()}|{result.hex()}")
    args.output.write_text("\n".join(records) + "\n")
    print(f"{args.output}: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
