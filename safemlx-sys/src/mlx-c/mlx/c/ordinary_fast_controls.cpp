// Copyright © 2026 Eredu contributors.
#include "mlx/c/ordinary_control.h"
#include "mlx/c/fast.h"
#include "mlx/c/private/mlx.h"
#include <limits>
#include <optional>
#include <string>

// The actual ordinary fast.cpp signatures. Primitive/descriptor/Eval storage
// is supplied independently by the selected numerical source.
extern "C" bool mlx_ordinary_fast_wrapper_controls(size_t* out, unsigned kind) {
  using namespace mlx::core;
  if (!out) return false;
  size_t bytes = 0;
  const auto add = [&](size_t value) {
    if (bytes > std::numeric_limits<size_t>::max() - value) return false;
    bytes += value;
    return true;
  };
  // The C output owner, core return value, output argument and status.
  if (!add(sizeof(array) * 2) || !add(sizeof(mlx_array*)) || !add(sizeof(int))) return false;
  if (kind == 0) { // mlx_fast_scaled_dot_product_attention with the empty mask mode
    if (!add(sizeof(mlx_array) * 5) || !add(sizeof(float)) ||
        !add(sizeof(const char*)) || !add(sizeof(std::string)) ||
        !add(sizeof(std::optional<array>) * 2) || !add(sizeof(mlx_stream))) return false;
  } else if (kind == 1) { // one mlx_fast_rope call, with a scalar offset
    if (!add(sizeof(mlx_array) * 2) || !add(sizeof(int) * 2) ||
        !add(sizeof(bool)) || !add(sizeof(mlx_optional_float)) ||
        !add(sizeof(float)) || !add(sizeof(std::optional<float>)) ||
        !add(sizeof(std::optional<array>)) || !add(sizeof(mlx_stream))) return false;
  } else {
    return false;
  }
  *out = bytes;
  return true;
}
