#include "mlx/c/scheduler_initialization.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"
using namespace mlx::core::scheduler;
namespace {
size_t transports() noexcept {
  return sizeof(mlx_scheduler_initialization_layout) * 2 +
      sizeof(SchedulerInitializationLayout) * 2 + sizeof(SchedulerInitializationOwner) +
      sizeof(SchedulerInitializationCause) + sizeof(mlx_scheduler_initialization_static_layout) +
      sizeof(SchedulerInitializationStaticLayout) + sizeof(uint64_t) * 2 +
      sizeof(void*) * 3 + sizeof(void (*)(void*)) + sizeof(size_t) * 2;
}
}
extern "C" void mlx_scheduler_initialization_static_layout_for(
    mlx_scheduler_initialization_static_layout* out) {
  if (!out) return;
  const auto value = scheduler_initialization_static_layout();
  *out = {value.bytes, unsigned(value.qualified)};
}
extern "C" unsigned mlx_scheduler_initialization_layout_for(
    mlx_scheduler_initialization_layout* out) {
  if (!out) return unsigned(SchedulerInitializationCause::invalid);
  SchedulerInitializationLayout value;
  const auto status = scheduler_initialization_layout(value);
  if (status != SchedulerInitializationCause::success) return unsigned(status);
  if (value.controls > SIZE_MAX - transports())
    return unsigned(SchedulerInitializationCause::invalid);
  *out = {value.object_bytes, value.object_alignment, value.controls + transports()};
  return 0;
}
extern "C" unsigned mlx_scheduler_initialize(uint64_t* output,
    mlx_scheduler_initialization_layout layout, void* owner, void (*retire)(void*)) {
  if (!output || *output || layout.controls < transports() ||
      !mlx::core::submission::submission_context_empty())
    return unsigned(SchedulerInitializationCause::invalid);
  const SchedulerInitializationLayout native{
      layout.object_bytes, layout.object_alignment, layout.controls - transports()};
  return unsigned(initialize_scheduler(native, {owner, retire}, *output));
}
extern "C" unsigned mlx_scheduler_initialized_borrow(uint64_t identity) {
  return unsigned(borrow_initialized_scheduler(identity));
}
