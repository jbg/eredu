#include "mlx/c/stream_registration.h"
#include "mlx/stream_registration.h"
#include "mlx/submission.h"
#include <new>
using namespace mlx::core;
namespace {
size_t transports() noexcept {
  return sizeof(mlx_stream_registration_layout) * 2 +
      sizeof(StreamRegistrationLayout) * 2 + sizeof(StreamRegistrationOwner) +
      sizeof(StreamRegistrationCause) * 2 + sizeof(mlx_stream) * 2 +
      sizeof(Stream) + sizeof(Stream*) * 2 + sizeof(void*) +
      sizeof(void (*)(void*)) + sizeof(size_t) * 2;
}
}
extern "C" void mlx_stream_registration_static_layout_for(
    mlx_stream_registration_static_layout* out) {
  if (!out) return;
  const auto value = stream_registration_static_layout();
  *out = {value.bytes, unsigned(value.qualified)};
}
extern "C" unsigned mlx_stream_registration_layout_for(
    mlx_stream_registration_layout* out) {
  if (!out) return unsigned(StreamRegistrationCause::invalid);
  StreamRegistrationLayout value;
  const auto status = stream_registration_layout(value);
  if (status != StreamRegistrationCause::success) return unsigned(status);
  if (value.controls > SIZE_MAX - transports())
    return unsigned(StreamRegistrationCause::invalid);
  *out = {value.object_bytes, value.object_alignment,
      sizeof(Stream), alignof(Stream), value.controls + transports()};
  return 0;
}
extern "C" unsigned mlx_stream_register_cpu(mlx_stream* out,
    mlx_stream_registration_layout layout, void* owner, void (*retire)(void*)) {
  if (!out || out->ctx || !owner || !retire ||
      !submission::submission_context_empty() || layout.controls < transports() ||
      layout.wrapper_bytes != sizeof(Stream) || layout.wrapper_alignment != alignof(Stream))
    return unsigned(StreamRegistrationCause::invalid);
  const StreamRegistrationLayout native{
      layout.object_bytes, layout.object_alignment, layout.controls - transports()};
  // Revalidate the cold query before even the wrapper allocation. Native repeats
  // this check before its own independently owned registration allocation.
  StreamRegistrationLayout actual;
  const auto qualified = stream_registration_layout(actual);
  if (qualified != StreamRegistrationCause::success) return unsigned(qualified);
  if (actual.object_bytes != native.object_bytes ||
      actual.object_alignment != native.object_alignment || actual.controls != native.controls)
    return unsigned(StreamRegistrationCause::layout_changed);
  auto* wrapper = new (std::nothrow) Stream(-1, Device::cpu);
  if (!wrapper) return unsigned(StreamRegistrationCause::allocation_failed);
  const auto status = register_prepared_cpu_stream(native, {owner, retire}, *wrapper);
  if (status != StreamRegistrationCause::success) {
    delete wrapper;
    return unsigned(status);
  }
  out->ctx = wrapper;
  return 0;
}
extern "C" unsigned mlx_stream_registration_borrow(mlx_stream stream, const void* owner) {
  if (!stream.ctx) return unsigned(StreamRegistrationCause::invalid);
  return unsigned(borrow_stream_registration(*static_cast<const Stream*>(stream.ctx), owner));
}
