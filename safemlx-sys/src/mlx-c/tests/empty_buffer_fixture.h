#pragma once
#include "mlx/allocator.h"
#include "mlx/c/original_buffer.h"
namespace empty_buffer_tests {
inline bool has_backing() {
  bool backing=false;
  REQUIRE(mlx::core::allocator::original_buffer_zero_has_backing(backing));
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  mlx_original_buffer_population_layout layout{};
  REQUIRE(mlx_original_buffer_request_layout_for(&layout,runtime,0)==0);
  CHECK((layout.capacity!=0)==backing);
  return backing;
}
}
