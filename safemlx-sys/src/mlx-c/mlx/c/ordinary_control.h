// Copyright © 2026 Eredu contributors.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mlx_ordinary_control_population_ {
  size_t observed_host_controls;
  size_t control_allocations;
  size_t platform_events;
} mlx_ordinary_control_population;
bool mlx_ordinary_frontend_control_layout(mlx_ordinary_control_population*,size_t,size_t,size_t,size_t);
bool mlx_ordinary_cpu_eval_control_layout(mlx_ordinary_control_population*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
bool mlx_ordinary_metal_eval_control_layout(mlx_ordinary_control_population*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
bool mlx_ordinary_metal_router_eval_control_layout(mlx_ordinary_control_population*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
bool mlx_ordinary_cpu_dispatch_envelope(mlx_ordinary_control_population*,size_t);
bool mlx_ordinary_metal_wait_record_control_layout(mlx_ordinary_control_population*);
bool mlx_ordinary_cpu_wait_control_layout(mlx_ordinary_control_population*);
bool mlx_ordinary_reserved_array_vector_control_layout(mlx_ordinary_control_population*,size_t);
bool mlx_ordinary_array_vector_control_layout(mlx_ordinary_control_population*,size_t);
bool mlx_ordinary_array_wrapper_controls(size_t*,unsigned,size_t);
bool mlx_ordinary_fast_wrapper_controls(size_t*,unsigned);
#ifdef __cplusplus
}
#endif
