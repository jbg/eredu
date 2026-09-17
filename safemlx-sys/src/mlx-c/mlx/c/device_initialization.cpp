#include "mlx/c/device_initialization.h"
#include "mlx/device_initialization.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/submission.h"
using namespace mlx::core;
namespace {
metal::EmbeddedMetallib source(mlx_device_initialization_source value) noexcept {
  return {value.compressed_data, value.compressed_size, value.uncompressed_size};
}
size_t transports() noexcept {
  return sizeof(mlx_device_initialization_source) + sizeof(metal::EmbeddedMetallib) +
      sizeof(mlx_device_initialization_layout) * 2 + sizeof(metal::DeviceInitializationLayout) * 2 +
      sizeof(metal::DeviceInitializationOwner) + sizeof(metal::DeviceInitializationCause) +
      sizeof(mlx_device_initialization_static_layout) + sizeof(metal::DeviceInitializationStaticLayout) +
      sizeof(uint64_t) * 2 + sizeof(void*) * 2 + sizeof(void (*)(void*)) + sizeof(size_t) * 2;
}
}
extern "C" void mlx_device_initialization_static_layout_for(mlx_device_initialization_static_layout* out) {
  if (!out) return;
  const auto value = metal::device_initialization_static_layout();
  *out = {value.bytes, unsigned(value.qualified), unsigned(value.required)};
}
extern "C" unsigned mlx_device_initialization_layout_for(mlx_device_initialization_layout* out,
    mlx_device_initialization_source input) {
  if (!out) return unsigned(metal::DeviceInitializationCause::invalid);
  metal::DeviceInitializationLayout value;
  const auto status = metal::device_initialization_layout(source(input), value);
  if (status != metal::DeviceInitializationCause::success) return unsigned(status);
  if (value.controls > SIZE_MAX - transports()) return unsigned(metal::DeviceInitializationCause::invalid);
  *out = {value.object_bytes, value.decoded_bytes, value.scratch_bytes, value.dispatch_copy_bytes,
      value.override_bytes, value.controls + transports()};
  return 0;
}
extern "C" unsigned mlx_device_initialize(uint64_t* output, mlx_device_initialization_source input,
    mlx_device_initialization_layout layout, void* owner, void (*retire)(void*)) {
  if (!output || *output || layout.controls < transports() || !submission::submission_context_empty())
    return unsigned(metal::DeviceInitializationCause::invalid);
  const metal::DeviceInitializationLayout native{layout.object_bytes, layout.decoded_bytes,
      layout.scratch_bytes, layout.dispatch_copy_bytes, layout.override_bytes, layout.controls - transports()};
  return unsigned(metal::initialize_device(source(input), native, {owner, retire}, *output));
}
extern "C" unsigned mlx_device_initialized_borrow(uint64_t identity) {
  if (!submission::submission_context_empty()) return unsigned(metal::DeviceInitializationCause::invalid);
  return unsigned(metal::borrow_initialized_device(identity));
}
