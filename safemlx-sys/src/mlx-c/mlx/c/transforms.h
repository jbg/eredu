/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_TRANSFORMS_H
#define MLX_TRANSFORMS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/event.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup transforms Transform operations
 */
/**@{*/

int mlx_async_eval(const mlx_vector_array outputs);
/**
 * Submit evaluation of `outputs` and set `event` to its completion token.
 *
 * This is the operation which records lazy graphs for execution. The existing
 * `mlx_async_eval` API is preserved for callers which do not need a token.
 */
int mlx_async_eval_with_event(
    mlx_event* event,
    const mlx_vector_array outputs);
/**
 * Submit evaluation between timestamp markers on `stream`.
 *
 * The markers are recorded on the execution timeline without synchronizing
 * the calling thread. The output graph must be rooted on exactly `stream`.
 */
int mlx_async_eval_timed(
    mlx_event* event,
    const mlx_vector_array outputs,
    const mlx_stream stream);
int mlx_checkpoint(mlx_closure* res, const mlx_closure fun);
int mlx_custom_function(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp /* may be null */,
    const mlx_closure_custom_jvp fun_jvp /* may be null */,
    const mlx_closure_custom_vmap fun_vmap /* may be null */);
int mlx_custom_vjp(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp);
int mlx_eval(const mlx_vector_array outputs);
int mlx_jvp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array tangents);
int mlx_value_and_grad(
    mlx_closure_value_and_grad* res,
    const mlx_closure fun,
    const int* argnums,
    size_t argnums_num);
int mlx_vjp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array cotangents);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
