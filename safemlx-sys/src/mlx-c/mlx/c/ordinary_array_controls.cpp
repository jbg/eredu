// Copyright © 2026 Eredu contributors.
#include "mlx/c/ordinary_control.h"
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include "mlx/c/private/mlx.h"
#include <limits>
#include <vector>
#include <optional>

// These are the actual ordinary C call signatures used by array.cpp/ops.cpp.
// ArrayDesc/Data/primitive owners and backend work are separately observed.
extern "C" bool mlx_ordinary_array_wrapper_controls(
    size_t* out, unsigned kind, size_t rank) {
  using namespace mlx::core;
  if (!out) return false;
  size_t bytes = 0;
  const auto add = [&](size_t value) {
    if (bytes > std::numeric_limits<size_t>::max() - value) return false;
    bytes += value;
    return true;
  };
  // A returned C array owner and the core operation's by-value result.
  if (kind != 10 && kind != 16 && kind != 20 && kind != 21 && (!add(sizeof(array)) || !add(sizeof(array)) ||
      !add(sizeof(mlx_array*)) || !add(sizeof(int)))) return false;
  switch (kind) {
    case 0: // mlx_array_set_data, used by scalar/from-slice
      if (rank > Shape::inline_capacity || !add(sizeof(Shape)) ||
          !add(sizeof(Dtype)) || !add(sizeof(mlx_dtype)) ||
          !add(sizeof(const void*)) || !add(sizeof(const int*)) ||
          !add(sizeof(int))) return false;
      break;
    case 1: // log
      if (!add(sizeof(mlx_array)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 2: // less / multiply / subtract share this C signature
      if (!add(sizeof(mlx_array)) || !add(sizeof(mlx_array)) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 3: // reshape / broadcast_to create one inline Shape from the borrowed rank vector
      if (rank > Shape::inline_capacity || !add(sizeof(Shape)) ||
          !add(sizeof(mlx_array)) || !add(sizeof(const int*)) ||
          !add(sizeof(size_t)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 4: // transpose_axes / squeeze_axes range-construct std::vector<int>
      if (rank > Shape::inline_capacity || rank > SIZE_MAX / sizeof(int) ||
          !add(sizeof(std::vector<int>)) || !add(rank * sizeof(int)) ||
          !add(sizeof(mlx_array)) || !add(sizeof(const int*)) ||
          !add(sizeof(size_t)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 5: // astype / view
      if (!add(sizeof(mlx_array)) || !add(sizeof(mlx_dtype)) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 6: // take_axis
      if (!add(sizeof(mlx_array)) || !add(sizeof(mlx_array)) ||
          !add(sizeof(int)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 7: // concatenate_axis / stack_axis; C vector shell queried separately
      if (!add(sizeof(mlx_vector_array)) || !add(sizeof(int)) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 8: // contiguous
    case 9: // all (all axes), with explicit keepdims
      if (!add(sizeof(mlx_array)) || !add(sizeof(bool)) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 10: // mlx_array_eval followed by mlx_array_item_bool; borrowed array
      if (!add(sizeof(mlx_array)) || !add(sizeof(int)) ||
          !add(sizeof(mlx_array)) || !add(sizeof(bool*)) ||
          !add(sizeof(bool)) || !add(sizeof(int))) return false;
      break;
    case 11: // expand_dims: one explicit axis
      if (!add(sizeof(mlx_array)) || !add(sizeof(int)) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 12: // mean_axis / sum_axis / argmin_axis
      if (!add(sizeof(mlx_array)) || !add(sizeof(int)) ||
          !add(sizeof(bool)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 13: // where: condition, selected and alternate arrays
      if (!add(sizeof(mlx_array) * 3) || !add(sizeof(mlx_stream))) return false;
      break;
    case 14: // fast_rms_norm: value, optional learned gain and epsilon
      if (!add(sizeof(mlx_array) * 2) || !add(sizeof(float)) ||
          !add(sizeof(std::optional<array>) * 2) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 15: // addmm: input, matrices and alpha/beta
      if (!add(sizeof(mlx_array) * 3) || !add(sizeof(float) * 2) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 16: // mlx_array_eval followed by mlx_array_item_uint32
      if (!add(sizeof(mlx_array)) || !add(sizeof(int)) ||
          !add(sizeof(mlx_array)) || !add(sizeof(uint32_t*)) ||
          !add(sizeof(uint32_t)) || !add(sizeof(int))) return false;
      break;
    case 17: // zeros/ones range-construct an inline Shape
      if (rank > Shape::inline_capacity || !add(sizeof(Shape)) ||
          !add(sizeof(const int*)) || !add(sizeof(size_t)) ||
          !add(sizeof(mlx_dtype)) || !add(sizeof(Dtype)) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 18: // scatter_add_axis
      if (!add(sizeof(mlx_array) * 3) || !add(sizeof(int)) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 19: // slice: three inline coordinate Shapes
      if (rank > Shape::inline_capacity || !add(sizeof(Shape) * 3) ||
          !add(sizeof(const int*) * 3) || !add(sizeof(size_t) * 3) ||
          !add(sizeof(mlx_array)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 20: // VectorArray new/append/free and synchronous mlx_eval
      if (!add(sizeof(ArrayVector) * 2) || !add(sizeof(mlx_vector_array) * 4) ||
          !add(sizeof(mlx_array) * 2) || !add(sizeof(int) * 3)) return false;
      break;
    case 21: // mlx_array_eval on one borrowed array
      if (!add(sizeof(mlx_array)) || !add(sizeof(int))) return false;
      break;
    case 22: // topk_axis / argpartition_axis / swapaxes: two integers
      if (!add(sizeof(mlx_array)) || !add(sizeof(int) * 2) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 23: // cumsum: explicit axis and two Boolean options
      if (!add(sizeof(mlx_array)) || !add(sizeof(int)) ||
          !add(sizeof(bool) * 2) || !add(sizeof(mlx_stream))) return false;
      break;
    case 24: // full: borrowed scalar, typed result and inline Shape
      if (rank > Shape::inline_capacity || !add(sizeof(Shape)) ||
          !add(sizeof(const int*)) || !add(sizeof(size_t)) ||
          !add(sizeof(mlx_array)) || !add(sizeof(mlx_dtype)) ||
          !add(sizeof(Dtype)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 25: // gather_mm: four array handles, optional core aliases and flag
      if (!add(sizeof(mlx_array) * 6) || !add(sizeof(std::optional<array>) * 2) ||
          !add(sizeof(bool)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 26: // clip: actual optional min/max array aliases and their C handles
      if (!add(sizeof(mlx_array) * 5) || !add(sizeof(std::optional<array>) * 2) ||
          !add(sizeof(mlx_stream))) return false;
      break;
    case 27: // arange: converted endpoints/step, explicit dtype and stream
      if (!add(sizeof(double) * 3) || !add(sizeof(mlx_dtype)) ||
          !add(sizeof(Dtype)) || !add(sizeof(mlx_stream))) return false;
      break;
    case 28: // slice_update: source/update handles and three inline Shapes
      if (rank > Shape::inline_capacity || !add(sizeof(Shape) * 3) ||
          !add(sizeof(const int*) * 3) || !add(sizeof(size_t) * 3) ||
          !add(sizeof(mlx_array) * 2) || !add(sizeof(mlx_stream))) return false;
      break;
    default: return false;
  }
  *out = bytes;
  return true;
}
