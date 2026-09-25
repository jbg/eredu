// Copyright © 2026 Eredu contributors.
#include "mlx/c/mixed_gemm.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

#ifdef MLX_C_MIXED_STORAGE_GEMM
#include "mlx/backend/gpu/device_info.h"
#include "mlx/primitives.h"

namespace {
using namespace mlx::core;
// A distinct inference-only primitive prevents native Matmul transforms or
// graph optimizations from assuming equal operand dtypes. Unsupported transforms
// retain Primitive's typed exception; no implicit cast or alternate reduction.
class MixedStorageGemm : public UnaryPrimitive {
 public:
  explicit MixedStorageGemm(Stream s) : UnaryPrimitive(s) {}
  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("MixedStorageGemm requires validated Metal execution");
  }
  void eval_gpu(const std::vector<array>& inputs, array& output) override {
    Matmul native(stream());
    native.eval_gpu(inputs, output);
  }
  DEFINE_NAME(MixedStorageGemm)
};
} // namespace
#endif

extern "C" int mlx_try_mixed_storage_gemm(
    mlx_array* res, bool* supported, mlx_array input, mlx_array weight,
    mlx_stream stream) {
  try {
    if (!res || !supported) {
      throw std::invalid_argument("MixedStorageGemm requires non-null outputs");
    }
    *supported = false;
#ifdef MLX_C_MIXED_STORAGE_GEMM
    const auto& a = mlx_array_get_(input);
    const auto& w = mlx_array_get_(weight);
    const auto& s = mlx_stream_get_(stream);
    if (s.device.type != Device::gpu || a.ndim() != 2 || w.ndim() != 2 ||
        a.dtype() != float32 || (w.dtype() != float16 && w.dtype() != bfloat16) ||
        a.shape(0) <= 1 || w.shape(0) <= 1 || a.shape(1) <= 0 ||
        a.shape(0) > 2000 || w.shape(0) > 65536 || a.shape(1) > 8192 ||
        a.shape(1) != w.shape(1) || !a.is_available() || !w.is_available() ||
        !a.flags().row_contiguous || !w.flags().row_contiguous) {
      return 0;
    }
    // Stage-two validation covers this device only. In particular, do not route
    // NAX/TF32 hardware or unvalidated SIMD geometries through this prototype.
    const auto& info = gpu::device_info(s.device.index);
    auto name = info.find("device_name");
    if (name == info.end() ||
        std::get<std::string>(name->second) != "Apple M3 Ultra") {
      return 0;
    }
    auto b = transpose(w, s);
    array output({a.shape(0), w.shape(0)}, float32,
        std::make_shared<MixedStorageGemm>(s), {a, b});
    mlx_array_set_(*res, std::move(output));
    *supported = true;
#endif
    return 0;
  } catch (const std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
