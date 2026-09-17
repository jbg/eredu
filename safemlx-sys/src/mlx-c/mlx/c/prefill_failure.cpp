#include "mlx/c/prefill_failure.h"
#include "mlx/failure.h"
#include "mlx/submission.h"
#include <cstring>
using namespace mlx::core;
namespace {
FailureCarrier* get(mlx_prefill_failure value) noexcept {
  return static_cast<FailureCarrier*>(value.ctx);
}
struct Construction {
  FailureCarrierRef value;
  mlx_prefill_failure* out;
  void* owner;
  void (*retire)(void*);
};
}
extern "C" unsigned mlx_prefill_failure_layout_for(mlx_prefill_failure_layout* out) {
  if (!out) return 1;
  // Named constructor, one successful capture, borrowed inspection and
  // return/alias controls. Original throw/ABI and event/task controls remain
  // explicitly separate, not disguised as part of this fixed shell.
  *out = {sizeof(FailureCarrier), alignof(FailureCarrier),
      sizeof(Construction) +
      // create/into_raw handoff, alias copy, copy-assignment handoff and the
      // displaced Event carrier retained through old event/control replacement.
      sizeof(FailureCarrierRef) * 4 +
      // from_exception/capture_native local and finish's moved parameter.
      sizeof(FailureValue) * 2 + sizeof(std::exception_ptr) * 2 +
      // Native source return and explicit copy-assignment temporary.
      sizeof(NativeFailure) * 2 +
      // Borrowed get/view source pointers and native NSString source parameters.
      sizeof(void*) * 4 + sizeof(size_t) * 5 + sizeof(bool) +
      sizeof(mlx_prefill_failure_view) + sizeof(mlx_prefill_failure_layout),
      FailureCarrier::retirement_control_bytes()};
  return 0;
}
extern "C" unsigned mlx_prefill_failure_new_retaining(
    mlx_prefill_failure* out, void* owner, void (*retire)(void*)) {
  if (!out || out->ctx || !owner || !retire) return 1;
  Construction local{{}, out, owner, retire};
  local.value = FailureCarrier::create(local.owner, local.retire);
  if (!local.value) return 3;
  local.out->ctx = FailureCarrier::into_raw(std::move(local.value));
  return 0;
}
extern "C" void mlx_prefill_failure_retain(mlx_prefill_failure value) {
  if (auto* owner = get(value)) (void)FailureCarrier::into_raw(owner->clone());
}
extern "C" void mlx_prefill_failure_free(mlx_prefill_failure value) {
  FailureCarrier::release_raw(get(value));
}
extern "C" unsigned mlx_prefill_failure_view_get(
    mlx_prefill_failure_view* out, mlx_prefill_failure value) {
  const auto* owner = get(value);
  if (!out || !owner) return 1;
  const auto* source = owner->borrow();
  if (!source) return owner->pending_publication() ? 2 : 1;
  *out = {static_cast<unsigned>(source->kind), source->message,
      source->message_size, nullptr, 0, 0};
  if (source->kind == FailureKind::exception && source->exception_type) {
    out->source_type = source->exception_type->name();
    out->source_type_size = std::strlen(out->source_type);
  } else if (source->kind == FailureKind::native) {
    out->native_code = source->native.code();
  }
  return 0;
}

extern "C" unsigned mlx_prefill_failure_native_text(
    size_t* used, size_t* remaining, mlx_prefill_failure value, unsigned field,
    size_t offset, uint16_t* output, size_t capacity) {
  const auto* owner = get(value);
  if (!owner || !used || !remaining || field > 1 || (!output && capacity)) return 1;
  const auto* source = owner->borrow();
  if (!source || source->kind != FailureKind::native) return 1;
  *used = source->native.copy_utf16(field == 1, offset, output, capacity, *remaining);
  return 0;
}

extern "C" unsigned mlx_prefill_failure_bind_original_scope(
    mlx_prefill_failure failure, mlx_submission_scope scope) {
  using namespace mlx::core;
  auto* owner = static_cast<FailureCarrier*>(failure.ctx);
  auto* target = static_cast<submission::Scope*>(scope.ctx);
  if (!owner || !target || !target->original_controls_required() ||
      !target->active_on_owner_thread())
    return static_cast<unsigned>(submission::NativeControlFailure::invalid_scope);
  if (!target->bind_failure(owner->clone()))
    return static_cast<unsigned>(submission::NativeControlFailure::foreign_domain);
  return 0;
}
