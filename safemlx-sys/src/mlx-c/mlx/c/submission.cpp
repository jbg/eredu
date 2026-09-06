#include "mlx/c/submission.h"
#include "mlx/submission.h"

namespace {
using mlx::core::submission::Activity;
using mlx::core::submission::Scope;

Scope* get_scope(mlx_submission_scope scope) noexcept {
  auto* value = static_cast<Scope*>(scope.ctx);
  return value && value->on_owner_thread() ? value : nullptr;
}

void write_status(mlx_submission_status* out, Scope* scope) noexcept {
  auto status = scope->query();
  switch (status.activity) {
    case Activity::none:
      out->activity = MLX_SUBMISSION_ACTIVITY_NONE;
      break;
    case Activity::pending:
      out->activity = MLX_SUBMISSION_ACTIVITY_PENDING;
      break;
    case Activity::terminal:
      out->activity = MLX_SUBMISSION_ACTIVITY_TERMINAL;
      break;
  }
  out->failed = status.failed;
  out->blocked = status.blocked;
}
} // namespace

extern "C" int mlx_submission_scope_new(mlx_submission_scope* out) {
  if (!out || out->ctx) {
    return 1;
  }
  try {
    out->ctx = new Scope;
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int mlx_submission_scope_seal(mlx_submission_scope scope) {
  if (auto* value = get_scope(scope)) {
    value->seal();
    return 0;
  }
  return 1;
}

extern "C" int mlx_submission_scope_query(
    mlx_submission_status* out, mlx_submission_scope scope) {
  if (auto* value = get_scope(scope); value && out) {
    write_status(out, value);
    return 0;
  }
  return 1;
}

extern "C" int mlx_submission_scope_progress(
    mlx_submission_status* out, mlx_submission_scope scope) {
  if (auto* value = get_scope(scope); value && out) {
    value->progress();
    write_status(out, value);
    return 0;
  }
  return 1;
}

extern "C" int mlx_submission_scope_free(mlx_submission_scope scope) {
  if (auto* value = get_scope(scope)) {
    value->seal();
    value->release();
    return 0;
  }
  return 1;
}
