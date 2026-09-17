#pragma once
#include "mlx/c/event.h"
#include "mlx/completion.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/submission.h"
#include "mlx/transforms.h"
#include "mlx/graph_construction.h"
#include <optional>

namespace mlx_c_detail {
using namespace mlx::core;
struct OperationEvent final : submission::ScopedRetirement {
  OperationEvent(submission::Scope* owner, submission::GraphQuota* arena) noexcept
      : scope(owner), quota(arena), roots(submission::GraphAllocator<array>(arena)) {}
  submission::Scope* scope;
  submission::GraphQuota* quota;
  // Unpublished scheduled-root builders restore only after their roots retire.
  std::optional<submission::GraphNestedEvaluation> nested;
  Completion completion;
  ArrayVector roots;
  enum class RootState : uint8_t { dynamic, failed, ready };
  RootState root_state{RootState::dynamic};
  size_t expected_roots{0};
  bool submitted{false};
};
inline OperationEvent* operation_get(mlx_operation_event value) noexcept {
  return static_cast<OperationEvent*>(value.ctx);
}
inline bool operation_current(const OperationEvent& event) noexcept {
  const auto controls = submission::current_native_controls();
  return event.scope->valid_original_observer() && event.scope->active_on_owner_thread() &&
      submission::current_scope() == event.scope && controls.original &&
      controls.failure == submission::NativeControlFailure::none &&
      controls.scope_identity == event.scope->identity() && controls.graph == event.quota;
}
inline bool operation_accepts_root(const OperationEvent& event) noexcept {
  return event.root_state == OperationEvent::RootState::dynamic ||
      (event.root_state == OperationEvent::RootState::ready &&
       event.roots.size() < event.expected_roots);
}
inline bool operation_roots_complete(const OperationEvent& event) noexcept {
  return event.root_state == OperationEvent::RootState::dynamic ||
      (event.root_state == OperationEvent::RootState::ready &&
       event.roots.size() == event.expected_roots);
}
inline unsigned operation_failure(OperationEvent& event) noexcept {
  event.scope->failure_owner().get()->capture_current_exception();
  return static_cast<unsigned>(ScopedEvaluation::failed);
}
} // namespace mlx_c_detail
