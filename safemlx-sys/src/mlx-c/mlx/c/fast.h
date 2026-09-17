/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FAST_H
#define MLX_FAST_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold retained Metal SDPA scratch override; zero selects device defaults.
int mlx_fast_sdpa_blocks_override(int* res);

/**
 * \defgroup fast Fast custom operations
 */
/**@{*/

typedef struct mlx_fast_cuda_kernel_config_ {
  void* ctx;
} mlx_fast_cuda_kernel_config;
mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void);
void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls);

int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value);
int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_cuda_kernel_ {
  void* ctx;
} mlx_fast_cuda_kernel;

mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);

void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls);

int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);

int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);

typedef struct mlx_fast_metal_kernel_config_ {
  void* ctx;
} mlx_fast_metal_kernel_config;
/* Concrete one-output C configuration/bridge host controls; zero is unqualified. */
size_t mlx_fast_metal_single_output_control_bytes(size_t templates, size_t rank, size_t maximum_name);
mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);

int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value);
int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_metal_kernel_ {
  void* ctx;
} mlx_fast_metal_kernel;

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);

void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);

/* Synchronous borrowed fixed configuration. Names/descriptors never escape. */
typedef struct mlx_fast_template_view_ {
  const char* name;
  int kind; /* 0=int, 1=bool, 2=dtype */
  int value;
} mlx_fast_template_view;
typedef struct mlx_fast_output_view_ {
  const int* shape;
  size_t ndim;
  mlx_dtype dtype;
} mlx_fast_output_view;
/* One immutable definition block; no array, Device, source-cache or library
 * construction. Every input text loan is copied before successful return. */
typedef struct mlx_fast_text_view_ { const char* data; size_t size; } mlx_fast_text_view;
typedef struct mlx_fast_definition_view_ {
  mlx_fast_text_view name, source, header;
  const mlx_fast_text_view* inputs;
  size_t input_count;
  const mlx_fast_text_view* outputs;
  size_t output_count;
  bool ensure_row_contiguous, atomic_outputs;
} mlx_fast_definition_view;
typedef struct mlx_fast_definition_layout_ {
  size_t requested_bytes, alignment, construction_controls;
} mlx_fast_definition_layout;
typedef struct mlx_fast_prepared_definition_ { void* ctx; } mlx_fast_prepared_definition;
int mlx_fast_metal_definition_layout(mlx_fast_definition_layout*, const mlx_fast_definition_view*);
size_t mlx_fast_metal_definition_static_bytes(void);
int mlx_fast_metal_definition_new(mlx_fast_prepared_definition*, const mlx_fast_definition_view*);
void mlx_fast_metal_definition_free(mlx_fast_prepared_definition);
int mlx_fast_metal_definition_apply_fixed(
    mlx_array* outputs, size_t output_count,
    mlx_fast_prepared_definition kernel, const mlx_array* inputs, size_t input_count,
    const mlx_fast_output_view* output_shapes,
    const mlx_fast_template_view* templates, size_t template_count,
    const int* grid, const int* thread_group, mlx_stream stream);
/* Closed finite source/cache declarations, never arbitrary future growth. */
typedef struct mlx_fast_input_signature_ { mlx_dtype dtype; bool scalar, constant; } mlx_fast_input_signature;
typedef struct mlx_fast_specialization_ {
  mlx_fast_input_signature inputs[8];
  mlx_dtype outputs[4];
  mlx_fast_template_view templates[4];
  size_t template_count;
} mlx_fast_specialization;
typedef struct mlx_fast_kernel_family_layout_ {
  size_t definition_bytes, family_bytes, alignment, control_bytes;
} mlx_fast_kernel_family_layout;
typedef struct mlx_fast_kernel_family_ { void* ctx; } mlx_fast_kernel_family;
int mlx_fast_kernel_family_layout_for(mlx_fast_kernel_family_layout*, const mlx_fast_definition_view*, const mlx_fast_specialization*, size_t);
size_t mlx_fast_kernel_family_static_bytes(void);
size_t mlx_fast_kernel_family_control_bytes(void);
int mlx_fast_kernel_family_new(mlx_fast_kernel_family*, const mlx_fast_definition_view*, const mlx_fast_specialization*, size_t, void*, void (*)(void*));
void mlx_fast_kernel_family_free(mlx_fast_kernel_family);
int mlx_fast_kernel_family_apply_fixed(
    mlx_array*, size_t, mlx_fast_kernel_family, const mlx_array*, size_t,
    const mlx_fast_output_view*, const mlx_fast_template_view*, size_t,
    const int*, const int*, mlx_stream);
size_t mlx_fast_metal_fixed_control_bytes(size_t inputs, size_t outputs,
    size_t templates, size_t maximum_rank);
int mlx_fast_metal_kernel_apply_fixed(
    mlx_array* outputs, size_t output_count,
    mlx_fast_metal_kernel kernel, const mlx_array* inputs, size_t input_count,
    const mlx_fast_output_view* output_shapes,
    const mlx_fast_template_view* templates, size_t template_count,
    const int* grid, const int* thread_group, mlx_stream stream);

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
