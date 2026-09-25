/* Copyright © 2026 Eredu contributors. */
#ifndef MLX_MIXED_GEMM_H
#define MLX_MIXED_GEMM_H
#include <stdbool.h>
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Inference-only operation: F32 [...,K] times narrow [N,K] transposed.
 * Unsupported cases return success with supported=false and leave res unchanged.
 * Does not evaluate inputs. Requires settled contiguous weights and native SIMD
 * Metal GEMM dispatch. Bias remains a separate ordinary add operation. */
int mlx_try_mixed_storage_gemm(
    mlx_array* res, bool* supported, mlx_array input, mlx_array weight,
    mlx_stream stream);
#ifdef __cplusplus
}
#endif
#endif
