#pragma once
#include "mlx/c/memory_placement.h"
#include "mlx/allocator.h"
inline mlx_memory_placement mlx_placement_to_c(mlx::core::allocator::MemoryPlacement p) noexcept {
  return {p.kind, p.device, p.device_count};
}
inline bool mlx_same_placement(mlx_memory_placement a, mlx_memory_placement b) noexcept {
  return a.kind == b.kind && a.device == b.device && a.device_count == b.device_count;
}
