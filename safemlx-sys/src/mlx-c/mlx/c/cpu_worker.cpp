#include "mlx/c/cpu_worker.h"
#include "mlx/worker_startup.h"
#include "mlx/scheduler.h"
#include "mlx/stream_registration.h"
using namespace mlx::core;
using namespace mlx::core::scheduler;
namespace {
size_t transports() noexcept {
  return sizeof(mlx_cpu_worker_layout) * 2 + sizeof(CpuWorkerStartupLayout) * 2 +
      sizeof(mlx_cpu_worker_result) * 4 + sizeof(CpuWorkerStartupResult) * 3 +
      sizeof(mlx_cpu_worker_target) * 3 + sizeof(CpuWorkerStartupOwner) +
      sizeof(SchedulerInitializationCause) + sizeof(StreamRegistrationCause) +
      sizeof(mlx_stream) + sizeof(Stream) * 2 + sizeof(Stream*) +
      sizeof(void*) * 5 + sizeof(void (*)(void*)) + sizeof(uint64_t) + sizeof(size_t);
}
mlx_cpu_worker_result result(CpuWorkerStartupResult r) noexcept {
  return {unsigned(r.cause), r.system_value, r.system_category};
}
Stream target_stream(mlx_cpu_worker_target target) noexcept {
  return Stream(target.stream_index, Device(Device::cpu, target.device_index));
}
}
extern "C" void mlx_cpu_worker_static_layout_for(mlx_cpu_worker_static_layout* out) {
  if (!out) return;
  const auto value = cpu_worker_startup_static_layout();
  *out = {value.bytes, unsigned(value.qualified)};
}
extern "C" mlx_cpu_worker_result mlx_cpu_worker_layout_for(mlx_cpu_worker_layout* out) {
  if (!out) return result({CpuWorkerStartupCause::invalid});
  CpuWorkerStartupLayout value;
  const auto query = cpu_worker_startup_layout(value);
  if (query.cause != CpuWorkerStartupCause::success) return result(query);
  if (value.controls > SIZE_MAX - transports()) return result({CpuWorkerStartupCause::invalid});
  *out = {value.entry_bytes, value.entry_alignment,
      value.fallback_control_bytes, value.fallback_char_bytes,
      value.thread_handle_bytes, value.thread_implementation_bytes, value.thread_packet_bytes,
      value.failure_requests, value.controls + transports()};
  return {};
}
extern "C" mlx_cpu_worker_result mlx_cpu_worker_target_for(mlx_cpu_worker_target* out,
    uint64_t scheduler_identity, mlx_stream input, const void* stream_birth) {
  if (!out || !input.ctx || !stream_birth) return result({CpuWorkerStartupCause::invalid});
  const auto& stream = *static_cast<const Stream*>(input.ctx);
  if (stream.device != Device::cpu) return result({CpuWorkerStartupCause::identity_mismatch});
  const auto scheduler = borrow_initialized_scheduler(scheduler_identity);
  if (scheduler == SchedulerInitializationCause::busy) return result({CpuWorkerStartupCause::busy});
  if (scheduler == SchedulerInitializationCause::unsupported) return result({CpuWorkerStartupCause::unsupported});
  if (scheduler != SchedulerInitializationCause::success) return result({CpuWorkerStartupCause::identity_mismatch});
  const auto registration = borrow_stream_registration(stream, stream_birth);
  if (registration == StreamRegistrationCause::busy) return result({CpuWorkerStartupCause::busy});
  if (registration == StreamRegistrationCause::unsupported) return result({CpuWorkerStartupCause::unsupported});
  if (registration != StreamRegistrationCause::success) return result({CpuWorkerStartupCause::identity_mismatch});
  *out = {scheduler_identity, stream.index, stream.device.index, stream_birth};
  return {};
}
extern "C" mlx_cpu_worker_result mlx_cpu_worker_initialize(const void** out,
    mlx_cpu_worker_layout layout, mlx_cpu_worker_target target,
    void* owner, void (*retire)(void*)) {
  if (!out || *out || layout.controls < transports()) return result({CpuWorkerStartupCause::invalid});
  const CpuWorkerStartupLayout native{layout.entry_bytes, layout.entry_alignment,
      layout.fallback_control_bytes, layout.fallback_char_bytes,
      layout.thread_handle_bytes, layout.thread_implementation_bytes, layout.thread_packet_bytes,
      layout.failure_requests, layout.controls - transports()};
  return result(initialize_cpu_worker(target.scheduler_identity, target_stream(target),
      target.stream_birth, native, {owner, retire}, *out));
}
extern "C" mlx_cpu_worker_result mlx_cpu_worker_borrow(
    mlx_cpu_worker_target target, const void* worker_birth) {
  return result(borrow_initialized_cpu_worker(target.scheduler_identity,
      target_stream(target), target.stream_birth, worker_birth));
}

extern "C" mlx_cpu_worker_result mlx_cpu_worker_observe_idle(
    mlx_cpu_worker_target target, const void* worker_birth) {
  return result(observe_initialized_cpu_worker_idle(target.scheduler_identity,
      target_stream(target), target.stream_birth, worker_birth));
}
