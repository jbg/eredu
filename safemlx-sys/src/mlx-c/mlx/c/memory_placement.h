// Copyright © 2026 Eredu contributors.
#ifndef MLX_C_MEMORY_PLACEMENT_H
#define MLX_C_MEMORY_PLACEMENT_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
// Fixed scalar facts. A managed allocation names host plus GPUs [0, device_count).
typedef struct mlx_memory_placement {
  uint32_t kind; // 0 unknown, 1 host, 2 fixed device, 3 CUDA managed, 4 GPU envelope
  int32_t device;
  uint32_t device_count;
} mlx_memory_placement;
// Selected ordinary allocator mechanism; does not manufacture execution authority.
int mlx_default_memory_placement(mlx_memory_placement* placement);
int mlx_gpu_allocation_placement(mlx_memory_placement* placement);
// An immutable observer admits backing and its host controls before allocation.
// A returned payload is released only after physical retirement (including cache).
typedef bool (*mlx_physical_backing_publish)(void*, void**, void (**)(void*));
typedef bool (*mlx_physical_backing_observer)(void*, uint64_t, size_t, size_t,
    mlx_memory_placement, void**, void (**)(void*), mlx_physical_backing_publish*);
size_t mlx_physical_backing_control_bytes(void);
size_t mlx_physical_backing_static_bytes(void);
int mlx_observe_physical_backings(void*, mlx_physical_backing_observer);
// Hardware fact for a registered GPU; never inferred from accessibility.
// Discovery does not construct MLX devices or consume prepared-owner provenance.
int mlx_memory_device_count(int* count);
int mlx_memory_device_shares_host(bool* shared, int device);
#ifdef __cplusplus
}
#endif
#endif
