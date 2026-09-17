#include "mlx/c/graph_rows.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/graph_construction.h"
#include "mlx/backend/common/utils.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/submission.h"
#include "mlx/transforms.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <memory>

namespace {
using namespace mlx::core;
using namespace mlx::core::submission;
struct Rows {
  Rows(GraphQuota* graph, uint64_t scope, size_t count) noexcept
      : values(GraphAllocator<array>(graph)), identity(scope), capacity(count) {}
  ArrayVector values;
  uint64_t identity;
  size_t capacity;
  bool consumed{false};
};
Scope* current(mlx_submission_observer observer) noexcept {
  auto* scope = static_cast<Scope*>(observer.ctx);
  if (!scope || !scope->valid_original_observer() || !scope->active_on_owner_thread() ||
      current_scope() != scope) return nullptr;
  const auto controls = current_native_controls();
  if (!controls.original || controls.failure != NativeControlFailure::none ||
      controls.scope_identity != scope->identity() || !controls.graph ||
      controls.graph != scope->graph_quota()) return nullptr;
  return scope;
}
bool matches(const Rows& rows, const Scope& scope) noexcept {
  return rows.identity == scope.identity() &&
      rows.values.get_allocator().resource() == scope.graph_quota();
}
void destroy(Rows* rows) noexcept {
  if (!rows) return;
  // The local allocator keeps the account alive through element destruction,
  // vector backing deallocation and the final header deallocation, in order.
  GraphAllocator<Rows> allocator(rows->values.get_allocator().resource());
  rows->~Rows();
  allocator.deallocate(rows, 1);
}
unsigned failed(Scope& scope) noexcept {
  scope.failure_owner().get()->capture_current_exception();
  return static_cast<unsigned>(ScopedEvaluation::failed);
}
}

extern "C" bool mlx_graph_array_rows_inspect(
    mlx_graph_array_rows_layout* out, size_t capacity, size_t rank) {
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 210106 || __cplusplus < 202002L
  return false;
#else
  if (!out || !capacity || !rank) return false;
  GraphAllocator<array> allocator(nullptr); // cold query never reads TLS
  const auto limit = std::min<size_t>(
      std::allocator_traits<decltype(allocator)>::max_size(allocator),
      std::numeric_limits<ArrayVector::difference_type>::max());
  if (capacity > limit) return false;
  const size_t operands = std::max<size_t>(4, capacity);
  ResidentGraphLayout bank;
  if (!resident_graph_layout_with_operands(1, 0, rank, operands, bank) ||
      bank.request_counts[2] < 1 || bank.request_counts[3] < 1 ||
      sizeof(Rows) > bank.request_bytes[2] ||
      alignof(Rows) > bank.reserved_alignment ||
      capacity > bank.request_bytes[3] / sizeof(array)) return false;
  if (rank > Shape::inline_capacity &&
      (bank.request_counts[4] < 2 ||
       rank > bank.request_bytes[4] / sizeof(Shape::value_type))) return false;
  // One extra resident control envelope supplies this actual Rows header,
  // one exactly reserved ArrayVector backing and the two borrowed-prefix Shape
  // copies. Its unused descriptor/shell slots remain conservative capacity.
  // All ordinary numerical primitives, concat copies and outputs are separate.
  constexpr size_t controls = sizeof(bank) + sizeof(Rows*) +
      sizeof(GraphAllocator<Rows>) + sizeof(GraphAllocator<array>) +
      sizeof(mlx_graph_array_rows) + sizeof(mlx_graph_array_rows_layout) +
      sizeof(NativeControlBinding) + sizeof(Scope*) + sizeof(GraphQuota*) +
      sizeof(uint64_t) + 4 * sizeof(size_t) + sizeof(bool) +
      // concat's moved argument and borrowed-prefix reshape's Shape/stream.
      sizeof(ArrayVector) + 2 * sizeof(Shape) + 2 * sizeof(StreamOrDevice) +
      3 * sizeof(array) + 4 * sizeof(mlx_array) + sizeof(mlx_stream) + 2 * sizeof(int);
  if (bank.named_control_bytes > std::numeric_limits<size_t>::max() - controls)
    return false;
  *out = {capacity, operands, 1, bank.named_control_bytes + controls};
  return true;
#endif
}
extern "C" unsigned mlx_graph_array_rows_new(
    mlx_graph_array_rows* out, mlx_submission_observer observer, size_t capacity) {
  auto* scope = current(observer);
  if (!scope || !out || out->ctx) return 4;
  mlx_graph_array_rows_layout layout{};
  if (!mlx_graph_array_rows_inspect(&layout, capacity, 2)) return 1;
  if (scope->failure_owner().get()->borrow()) return 7;
  if (scope->failure_owner().get()->pending_publication()) return 10;
  mark_native_control_construction();
  Rows* rows = nullptr;
  try {
    GraphAllocator<Rows> allocator(scope->graph_quota());
    auto* storage = allocator.allocate(1);
    // All members construct without allocation; the published partial owner is
    // destroyed below if its one backing reservation fails.
    rows = ::new (storage) Rows(scope->graph_quota(), scope->identity(), capacity);
    rows->values.reserve(capacity);
    if (rows->values.capacity() != capacity) { destroy(rows); return 1; }
    out->ctx = rows;
    return 0;
  } catch (...) {
    const auto status = failed(*scope);
    destroy(rows);
    return status;
  }
}
extern "C" unsigned mlx_graph_array_rows_push(
    mlx_graph_array_rows value, mlx_submission_observer observer, mlx_array input) {
  auto* scope = current(observer);
  auto* rows = static_cast<Rows*>(value.ctx);
  if (!scope || !rows || !matches(*rows, *scope)) return 4;
  if (rows->consumed || rows->values.size() == rows->capacity || !input.ctx) return 1;
  try { rows->values.emplace_back(mlx_array_get_(input)); return 0; }
  catch (...) { return failed(*scope); }
}
extern "C" unsigned mlx_graph_array_rows_concatenate(
    mlx_array* out, mlx_graph_array_rows value, mlx_submission_observer observer,
    int axis, mlx_stream stream) {
  auto* scope = current(observer);
  auto* rows = static_cast<Rows*>(value.ctx);
  if (!scope || !rows || !matches(*rows, *scope) || !stream.ctx || !out || out->ctx) return 4;
  if (rows->consumed || rows->values.size() != rows->capacity) return 1;
  const auto ready = preflight_original_submission_stream(*scope, mlx_stream_get_(stream));
  if (ready != ScopedEvaluation::complete) return static_cast<unsigned>(ready);
  rows->consumed = true; // a failed constructor never permits a second attempt
  try {
    mlx_array_set_(*out, concatenate(std::move(rows->values), axis, mlx_stream_get_(stream)));
    return 0;
  } catch (...) { return failed(*scope); }
}
extern "C" void mlx_graph_array_rows_free(mlx_graph_array_rows value) {
  destroy(static_cast<Rows*>(value.ctx));
}
extern "C" int mlx_reshape_like_prefix(mlx_array* out, mlx_array input,
    mlx_array source, int final_dimension, mlx_stream stream) {
  try {
    auto shape = mlx_array_get_(source).shape();
    if (shape.empty()) {
      mlx_error("reshape prefix requires a nonempty source shape");
      return 1;
    }
    shape.back() = final_dimension;
    mlx_array_set_(*out, reshape(mlx_array_get_(input), std::move(shape), mlx_stream_get_(stream)));
    return 0;
  } catch (const std::exception& cause) { mlx_error(cause.what()); return 1; }
}


extern "C" bool mlx_reshape_like_prefix_control_bytes(size_t* out) {
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 210106 || __cplusplus != 202002L
  return false;
#else
  if (!out) return false;
  // Actual borrowed-prefix constructor and C exception adapter frames. The
  // copied Shape backing belongs to the separately inspected host Graph bank.
  *out = sizeof(Shape) + sizeof(array) + 3 * sizeof(mlx_array) +
      sizeof(mlx_array*) + sizeof(mlx_stream) + sizeof(int) +
      sizeof(mlx::core::StreamOrDevice) + sizeof(std::exception_ptr);
  return true;
#endif
}

namespace {
bool graph_copy_worker_inspect(
    mlx_graph_copy_worker_layout* out, size_t rank, size_t reshapes,
    size_t concatenations, size_t concatenate_inputs, size_t byte_views) {
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 210106 || __cplusplus != 202002L
  return false;
#else
  if (!out || rank > size_t(INT_MAX) ||
      rank > Shape::max_size() / 2 || rank > Strides::max_size() ||
      (concatenations == 0) != (concatenate_inputs == 0) ||
      concatenate_inputs < concatenations) return false;
  size_t result = 0;
  auto add = [](size_t a, size_t b, size_t& sum) {
    if (b > SIZE_MAX - a) return false;
    sum = a + b; return true;
  };
  auto mul = [](size_t a, size_t b, size_t& product) {
    if (a && b > SIZE_MAX / a) return false;
    product = a * b; return true;
  };
  auto extent = [&](size_t bytes, size_t alignment, size_t count) {
    if (!bytes || !count) return true;
    size_t each, total;
    return GraphQuota::allocation_extent(bytes, alignment, each) &&
        mul(each, count, total) && add(result, total, result);
  };
  // Mirror SmallVector::grow. Fresh size/range constructors allocate once
  // at max(requested, twice-inline), which also bounds exact-size copies;
  // push loops can visit every doubling. Count
  // every request, including the final one if its allocator refuses it.
  auto vectors = [&]<class Vector>(size_t fresh, size_t growing) {
    size_t capacity = Vector::inline_capacity;
    if (rank <= capacity) return true;
    const size_t maximum = Vector::max_size();
    const auto next = [&](size_t old) {
      return old > maximum / 2 ? maximum : 2 * old;
    };
    size_t bytes;
    if (!mul(std::max(rank, next(capacity)), sizeof(typename Vector::value_type), bytes) ||
        !extent(bytes, alignof(typename Vector::value_type), fresh)) return false;
    while (capacity < rank) {
      const size_t advanced = next(capacity);
      if (advanced <= capacity) return false;
      capacity = advanced;
      if (!mul(capacity, sizeof(typename Vector::value_type), bytes) ||
          !extent(bytes, alignof(typename Vector::value_type), growing)) return false;
    }
    return true;
  };
  size_t copies, twice_inputs, fresh_shapes, fresh_strides, growing_strides;
  // prepare_reshape: one growing Shape and two growing Strides (single-array
  // collapse plus output strides); returning the two pairs copies one Shape
  // and two Strides. make_contiguous_strides adds one fresh Strides.
  // copy_gpu_inplace: its collapse result tuple copies one Shape/two Strides;
  // two additional range Strides feed set_vector_bytes.
  // Concatenate adds one copied output Strides per invocation, and per input a
  // temporary descriptor's Shape/Strides plus the copy worker. Its frontend
  // also permits one AsType copy worker per input; no dtype sameness assumed.
  if (!mul(concatenate_inputs, 2, twice_inputs) ||
      !add(reshapes, twice_inputs, copies) ||
      !add(copies, byte_views, copies) ||
      !mul(reshapes, 2, fresh_shapes)) return false;
  size_t extra;
  if (!mul(concatenate_inputs, 3, extra) ||
      !add(fresh_shapes, extra, fresh_shapes) ||
      !mul(reshapes, 7, fresh_strides) ||
      !mul(concatenate_inputs, 9, extra) ||
      !add(fresh_strides, extra, fresh_strides) ||
      !add(fresh_strides, concatenations, fresh_strides) ||
      // View: copied frontend shape, fallback temporary shape/descriptor,
      // copy_gpu collapse tuple and range views; the alias branch additionally
      // copies its stride vector. Keep both branches in this finite union.
      !mul(byte_views, 4, extra) ||
      !add(fresh_shapes, extra, fresh_shapes) ||
      !mul(byte_views, 8, extra) ||
      !add(fresh_strides, extra, fresh_strides) ||
      !mul(reshapes, 2, growing_strides) ||
      !vectors.template operator()<Shape>(fresh_shapes, reshapes) ||
      !vectors.template operator()<Strides>(fresh_strides, growing_strides)) return false;
  CollapseStorageLayout collapse;
  size_t collapse_extents;
  if (!collapse_contiguous_dims_layout(2, rank, true, collapse) ||
      !mul(collapse.graph_extent_sum, copies, collapse_extents) ||
      !add(result, collapse_extents, result)) return false;
  // The existing worker query already pays descriptor/Data shells, temporary
  // arrays, concatenate prefix sizes, kernels, resource/receipt owners and
  // ordinary inline-rank collapse. This is additive rank-dependent metadata.
  constexpr size_t frames = 3 * sizeof(Shape) + 9 * sizeof(Strides) +
      sizeof(CollapseStorageLayout) + sizeof(CollapsedDimensions) +
      sizeof(std::pair<bool, Strides>) + sizeof(std::pair<Shape, Strides>) +
      sizeof(std::tuple<Shape, Strides, Strides>) + sizeof(array) +
      sizeof(GraphAllocator<int>) + sizeof(mlx_graph_copy_worker_layout) +
      12 * sizeof(size_t);
  size_t controls = frames;
  if (byte_views) {
    // ops::view and View::eval_gpu plus their shared general-copy fallback.
    // Descriptor/Data/primitive owners are independently in the resident bank.
    constexpr size_t view_frames = 3 * sizeof(array) + 2 * sizeof(Shape) +
        2 * sizeof(Strides) + sizeof(array::Flags) + 2 * sizeof(Dtype) +
        sizeof(StreamOrDevice) + sizeof(mlx_array) * 2 + sizeof(mlx_array*) +
        sizeof(mlx_stream) + sizeof(mlx_dtype) + sizeof(std::exception_ptr) +
        sizeof(Shape::const_iterator) + sizeof(int) + 6 * sizeof(size_t);
    size_t all;
    if (!mul(byte_views, view_frames, all) || !add(controls, all, controls)) return false;
  }
  *out = {result, controls};
  return true;
#endif
}
} // namespace
extern "C" bool mlx_graph_copy_worker_inspect(
    mlx_graph_copy_worker_layout* out, size_t rank, size_t reshapes,
    size_t concatenations, size_t concatenate_inputs) {
  return graph_copy_worker_inspect(out, rank, reshapes, concatenations, concatenate_inputs, 0);
}
extern "C" bool mlx_graph_byte_view_worker_inspect(
    mlx_graph_copy_worker_layout* out, size_t rank) {
  return graph_copy_worker_inspect(out, rank, 0, 0, 0, 1);
}

extern "C" bool mlx_original_convolution_inspect(
    mlx_original_convolution_layout* out, const mlx_original_convolution_source* source) {
  if (!out || !source) return false;
  ResidentConvolutionSource input;
  input.spatial_dimensions = source->spatial_dimensions;
  input.groups = source->groups;
  for (size_t i = 0; i != 4; ++i) {
    input.input[i] = source->input[i]; input.weight[i] = source->weight[i];
  }
  for (size_t i = 0; i != 2; ++i) {
    input.stride[i] = source->stride[i]; input.padding[i] = source->padding[i];
    input.dilation[i] = source->dilation[i];
  }
  ResidentConvolutionLayout layout;
  if (!resident_convolution_layout(input, layout)) return false;
  constexpr size_t controls = sizeof(input) + sizeof(layout) + sizeof(out) +
      sizeof(source) + sizeof(size_t) + sizeof(bool);
  if (layout.named_control_bytes > SIZE_MAX - controls) return false;
  *out = {layout.primitives, layout.edges, layout.backing_births,
      layout.named_control_bytes + controls,
      {layout.output[0], layout.output[1], layout.output[2], layout.output[3]}};
  return true;
}
