#include "mlx/c/stream_copy.h"
#include "mlx/stream.h"
#include "mlx/backend/cpu/matmul_storage.h"
#include <new>
#include <type_traits>
namespace {
using mlx::core::Stream;
using mlx::core::Device;
static_assert(std::is_trivially_copy_constructible_v<Stream>);
static_assert(std::is_trivially_destructible_v<Stream>);
static_assert(std::is_trivially_destructible_v<Device>);
}
extern "C" unsigned mlx_stream_copy_snapshot(mlx_stream_copy_value* out, mlx_stream source) {
  if (!out || !source.ctx) return 2;
  const auto& value = *static_cast<const Stream*>(source.ctx);
  const auto kind = value.device.type;
  if (kind != Device::cpu && kind != Device::gpu) return 2;
  const auto kernel=static_cast<unsigned>(value.cpu_matmul());
  if(kernel>2||(kind!=Device::cpu&&kernel!=0))return 2;
  *out = {value.index, value.device.index, kind == Device::cpu ? 0u : 1u, kernel};
  return 0;
}
extern "C" unsigned mlx_stream_copy_layout_for(mlx_stream_copy_layout* out) {
  if (!out) return 2;
  *out = {sizeof(Stream), alignof(Stream),
      sizeof(mlx_stream_copy_layout) + sizeof(mlx_stream_copy_value) * 2 +
      sizeof(mlx_stream) * 3 + sizeof(Stream) + sizeof(Device) +
      sizeof(Device::DeviceType) + sizeof(void*) * 4 + sizeof(unsigned) * 6 +
      sizeof(mlx::core::cpu::TiledMatmulFacts)};
  return 0;
}
extern "C" unsigned mlx_stream_copy_new(mlx_stream* out, mlx_stream_copy_value value) {
  if (!out || out->ctx || value.device_kind > 1 || value.cpu_matmul > 2 ||
      (value.device_kind != 0 && value.cpu_matmul != 0)) return 2;
  // Same exact native value as ordinary cloning, without new_stream, default
  // lookup, worker initialization or mutable access to the source wrapper.
  auto* copy = new (std::nothrow) Stream(value.index,
      Device(value.device_kind == 0 ? Device::cpu : Device::gpu, value.device_index),
      static_cast<mlx::core::CpuMatmulKernel>(value.cpu_matmul));
  if (!copy) return 4;
  out->ctx = copy;
  return 0;
}
extern "C" void mlx_stream_copy_free(mlx_stream value) {
  delete static_cast<Stream*>(value.ctx);
}

extern "C" unsigned mlx_cpu_matmul_facts_for(mlx_cpu_matmul_facts* out) {
  if(!out)return 2;
  mlx::core::cpu::TiledMatmulFacts source;
  if(!mlx::core::cpu::tiled_matmul_facts(source))return 2;
  const size_t controls=sizeof(source)+sizeof(*out)+sizeof(out)+sizeof(size_t)*2;
  const bool half=mlx::core::cpu::float16_tiled_matmul_facts(source);
  const bool platform=half&&mlx::core::cpu::platform_float16_uses_tiles();
  *out={source.tile_edge,source.reduction_lanes,source.max_rank,source.max_elements,
      controls+sizeof(bool)*2,half,platform};return 0;
}
extern "C" unsigned mlx_stream_copy_select_cpu_matmul(mlx_stream_copy_value* value,unsigned kernel) {
  if(!value||value->device_kind!=0||value->cpu_matmul>2||kernel>2)return 2;
  mlx::core::cpu::TiledMatmulFacts source;
  if(kernel==1&&!mlx::core::cpu::tiled_matmul_facts(source))return 2;
  if(kernel==2&&!mlx::core::cpu::float16_tiled_matmul_facts(source))return 2;
  value->cpu_matmul=kernel;return 0;
}
