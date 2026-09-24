// Adapted from MLX GEMVKernel (gemv.h), preserving its FP32 operation order.
// Copyright © 2023-2024 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

constexpr int BLOCK_M = BM * SM * TM;
constexpr int BLOCK_N = BN * SN * 4;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
int thr_m = SN != 32 ? lane / SN : 0;
int thr_n = SN != 32 ? lane % SN : int(lane);
int sg_n = BN != 1 ? group % BN : 0;
int simd_m = BN != 1 ? SM * (group / BN) : SM * group;
int simd_n = BN != 1 ? SN * (group % BN) : 0;
int bm = (simd_m + thr_m) * TM;
int bn = (simd_n + thr_n) * 4;
int row = threadgroup_position_in_grid.x * BLOCK_M + bm;
if (row >= OUTPUTS) return;
int first_output = row;
row = row + TM <= OUTPUTS ? row : OUTPUTS - TM;
float result[TM] = {0};

// Each lane consumes four consecutive coefficients in the same order as MLX's
// F32 GEMV. The only change is converting the narrow matrix element on load.
for (int block = 0; block < WIDTH / BLOCK_N; ++block) {
    float values[4];
    #pragma clang loop unroll(full)
    for (int n = 0; n < 4; ++n) values[n] = input[bn + n];
    #pragma clang loop unroll(full)
    for (int m = 0; m < TM; ++m) {
        #pragma clang loop unroll(full)
        for (int n = 0; n < 4; ++n) {
            result[m] += float(weight[size_t(row + m) * WIDTH + bn + n]) * values[n];
        }
    }
    bn += BLOCK_N;
}
if (WIDTH % BLOCK_N != 0) {
    float values[4];
    #pragma clang loop unroll(full)
    for (int n = 0; n < 4; ++n) values[n] = bn + n < WIDTH ? input[bn + n] : 0.0f;
    #pragma clang loop unroll(full)
    for (int m = 0; m < TM; ++m) {
        #pragma clang loop unroll(full)
        for (int n = 0; n < 4; ++n) {
            float w = bn + n < WIDTH ? float(weight[size_t(row + m) * WIDTH + bn + n]) : 0.0f;
            result[m] += w * values[n];
        }
    }
}
#pragma clang loop unroll(full)
for (int m = 0; m < TM; ++m) {
    #pragma clang loop unroll(full)
    for (ushort offset = SN / 2; offset >= 1; offset >>= 1) {
        result[m] += simd_shuffle_down(result[m], offset);
    }
}
if (BN > 1) {
    threadgroup float partials[BN * (BLOCK_M + TM)];
    if (thr_n == 0) {
        #pragma clang loop unroll(full)
        for (int m = 0; m < TM; ++m) partials[sg_n * (BLOCK_M + TM) + bm + m] = result[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_n == 0) {
        #pragma clang loop unroll(full)
        for (int g = 1; g < BN; ++g) {
            #pragma clang loop unroll(full)
            for (int m = 0; m < TM; ++m) result[m] += partials[g * (BLOCK_M + TM) + bm + m];
        }
    }
}
if (simd_n == 0 && thr_n == 0) {
    #pragma clang loop unroll(full)
    for (int m = 0; m < TM; ++m) {
        // The tail shifts its loads inward like MLX, but only writes its own
        // outputs so neighboring SIMD groups never race on overlapping rows.
        if (row + m >= first_output) output[row + m] = result[m];
    }
}
