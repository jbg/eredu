#include "mlx/c/gpu_stream_registration.h"
#include "mlx/gpu_stream_registration.h"
#include "mlx/device_initialization.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"
#include <algorithm>
#include <new>
using namespace mlx::core;
namespace {
using Cause = GpuStreamRegistrationCause;
size_t transports() noexcept {
  return 2 * sizeof(mlx_gpu_stream_registration_layout) + 2 * sizeof(GpuStreamRegistrationLayout) +
      sizeof(StreamRegistrationOwner) + 2 * sizeof(mlx_gpu_stream_target) +
      2 * sizeof(mlx_stream) + 2 * sizeof(Stream*) + sizeof(Stream) +
      sizeof(Cause) * 2 + sizeof(void*) + sizeof(void (*)(void*)) +
      sizeof(std::shared_ptr<void>);
}
Cause target(mlx_gpu_stream_target value) noexcept {
  const auto device = metal::borrow_initialized_device(value.device_identity);
  if (device == metal::DeviceInitializationCause::busy) return Cause::busy;
  if (device != metal::DeviceInitializationCause::success) return Cause::identity_mismatch;
  const auto scheduler = scheduler::borrow_initialized_scheduler(value.scheduler_identity);
  if (scheduler == scheduler::SchedulerInitializationCause::busy) return Cause::busy;
  return scheduler == scheduler::SchedulerInitializationCause::success
      ? Cause::success : Cause::identity_mismatch;
}
}
extern "C" unsigned mlx_gpu_stream_registration_layout_for(mlx_gpu_stream_registration_layout* out) {
  if (!out) return unsigned(Cause::invalid);
  GpuStreamRegistrationLayout native;
  const auto status = gpu_stream_registration_layout(native);
  if (status != Cause::success) return unsigned(status);
  if (native.controls > SIZE_MAX - transports()) return unsigned(Cause::invalid);
  mlx_gpu_stream_registration_layout value{};
  std::copy_n(native.bytes, 6, value.bytes);
  std::copy_n(native.alignments, 6, value.alignments);
  value.wrapper_bytes = sizeof(Stream); value.wrapper_alignment = alignof(Stream);
  value.controls = native.controls + transports();
  value.platform_queues = native.platform_queues;
  value.platform_buffers = native.platform_buffers;
  value.autorelease_pools = native.autorelease_pools;
  *out = value;
  return 0;
}
extern "C" unsigned mlx_stream_register_gpu(mlx_stream* out,
    mlx_gpu_stream_registration_layout layout, mlx_gpu_stream_target selected,
    void* owner, void (*retire)(void*)) {
  if (!out || out->ctx || !owner || !retire || !submission::submission_context_empty() ||
      layout.wrapper_bytes != sizeof(Stream) || layout.wrapper_alignment != alignof(Stream) ||
      layout.controls < transports()) return unsigned(Cause::invalid);
  const auto authenticated = target(selected);
  if (authenticated != Cause::success) return unsigned(authenticated);
  GpuStreamRegistrationLayout native;
  std::copy_n(layout.bytes, 6, native.bytes);
  std::copy_n(layout.alignments, 6, native.alignments);
  native.controls = layout.controls - transports();
  native.platform_queues = layout.platform_queues;
  native.platform_buffers = layout.platform_buffers;
  native.autorelease_pools = layout.autorelease_pools;
  GpuStreamRegistrationLayout actual;
  auto status = gpu_stream_registration_layout(actual);
  if (status != Cause::success) return unsigned(status);
  if (!(actual == native)) return unsigned(Cause::layout_changed);
  auto* wrapper = new (std::nothrow) Stream(-1, Device::gpu);
  if (!wrapper) return unsigned(Cause::allocation_failed);
  status = register_prepared_gpu_stream(native, selected.device_identity,
      selected.scheduler_identity, {owner, retire}, *wrapper);
  if (status != Cause::success) { delete wrapper; return unsigned(status); }
  out->ctx = wrapper;
  return 0;
}
extern "C" unsigned mlx_stream_gpu_registration_borrow(mlx_stream stream,
    const void* owner, mlx_gpu_stream_target selected) {
  if (!stream.ctx || !owner) return unsigned(Cause::invalid);
  const auto authenticated = target(selected);
  if (authenticated != Cause::success) return unsigned(authenticated);
  const auto& value = *static_cast<const Stream*>(stream.ctx);
  if (value.device != Device::gpu) return unsigned(Cause::identity_mismatch);
  std::shared_ptr<void> encoder;
  return unsigned(gpu::borrow_registered_encoder(value, owner, encoder));
}

extern "C" unsigned mlx_stream_gpu_registration_observe_idle(mlx_stream stream,
    const void* owner, mlx_gpu_stream_target selected) {
  if (!stream.ctx || !owner) return unsigned(Cause::invalid);
  const auto authenticated = target(selected);
  if (authenticated != Cause::success) return unsigned(authenticated);
  const auto& value = *static_cast<const Stream*>(stream.ctx);
  if (value.device != Device::gpu) return unsigned(Cause::identity_mismatch);
  return unsigned(gpu::registered_encoder_idle(value, owner));
}
