#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mlx_scheduler_initialization_layout {
  size_t object_bytes, object_alignment, controls;
} mlx_scheduler_initialization_layout;
typedef struct mlx_scheduler_initialization_static_layout {
  size_t bytes;
  unsigned qualified;
} mlx_scheduler_initialization_static_layout;
// Pure owning static/type/code facts. No runtime, thread identity or Scheduler
// is initialized. Dynamic qualification checks the actually loaded constructor.
void mlx_scheduler_initialization_static_layout_for(mlx_scheduler_initialization_static_layout*);
unsigned mlx_scheduler_initialization_layout_for(mlx_scheduler_initialization_layout*);
// Fixed statuses and unchanged output on refusal. Success transfers the exact
// raw custody permanently, matching this process singleton's real lifetime.
unsigned mlx_scheduler_initialize(uint64_t*, mlx_scheduler_initialization_layout,
    void*, void (*)(void*));
unsigned mlx_scheduler_initialized_borrow(uint64_t);
#ifdef __cplusplus
}
#endif
