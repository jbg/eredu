// Append-only private backing. The safe wrapper grants each suffix once and
// never exposes the capacity array; callers can observe only immutable prefixes.
#include "mlx/c/private/mlx.h"
#include "mlx/c/error.h"
#include "mlx/c/private/prepared_array_output.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/primitives.h"
#include "mlx/transforms_impl.h"
#include <atomic>
#ifdef MLX_C_PREFIX_APPEND_GPU
#include "mlx/backend/gpu/copy.h"
#endif

namespace {
using namespace mlx::core;
class PrefixAppend final : public UnaryPrimitive {
 public:
  PrefixAppend(Stream stream, int axis, int start)
      : UnaryPrimitive(stream), axis_(axis), start_(start) {}
  const char* name() const override { return "PrefixAppend"; }
  void eval_cpu(const std::vector<array>& inputs, array& out) override {
    claim();
    out.copy_shared_buffer(inputs[0]);
    copy_cpu_inplace(inputs[1], out, inputs[1].shape(), inputs[1].strides(),
        out.strides(), 0, start_ * int64_t(out.strides()[axis_]),
        CopyType::GeneralGeneral, stream());
  }
  void eval_gpu(const std::vector<array>& inputs, array& out) override {
#ifdef MLX_C_PREFIX_APPEND_GPU
    claim();
    out.copy_shared_buffer(inputs[0]);
    copy_gpu_inplace(inputs[1], out, inputs[1].shape(), inputs[1].strides(),
        out.strides(), 0, start_ * int64_t(out.strides()[axis_]),
        CopyType::GeneralGeneral, stream());
#else
    throw std::runtime_error("Prefix append GPU execution unavailable");
#endif
  }
 private:
  void claim() {
    // A stateful node must never be replayed by graph transforms/compilation.
    if (detail::InTracing::in_tracing() || used_.exchange(true)) {
      throw std::invalid_argument("Prefix append is a one-use state operation");
    }
  }
  int axis_, start_;
  std::atomic<bool> used_{false};
};
}

extern "C" int _mlx_prefix_append(mlx_array* result, mlx_array backing,
    mlx_array update, int axis, int start, mlx_stream stream) {
  try {
    if (mlx::core::detail::InTracing::in_tracing()) {
      throw std::invalid_argument("Prefix append is unavailable in graph transforms");
    }
    auto& old = mlx_array_get_(backing);
    auto& next = mlx_array_get_(update);
    if (axis < 0 || axis >= old.ndim() || start < 0 || old.dtype() != next.dtype()
        || old.ndim() != next.ndim() || next.shape(axis) > old.shape(axis) - start) {
      throw std::invalid_argument("Invalid prefix append geometry");
    }
    for (int i = 0; i < old.ndim(); ++i) {
      if (i != axis && old.shape(i) != next.shape(i)) {
        throw std::invalid_argument("Prefix append non-axis dimensions differ");
      }
    }
    mlx_array_output_preparation_ publication(*result);
    array output(old.shape(), old.dtype(), std::make_shared<PrefixAppend>(
        mlx_stream_get_(stream), axis, start), {old, next});
    publication.publish(std::move(output));
    return 0;
  } catch (std::exception& error) {
    mlx_error(error.what());
    return 1;
  }
}
