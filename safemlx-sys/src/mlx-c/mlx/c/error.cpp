/* Copyright © 2023-2024 Apple Inc. */

#include "mlx/c/error.h"
#include "mlx/record_quota.h"
#include "mlx/allocator.h"
#include "mlx/graph_quota.h"
#include "mlx/submission.h"
#include <exception>

#include <memory>
#include <vector>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

static void mlx_error_handler_default_(const char* msg, void* data) {
  printf("MLX error: %s\n", msg);
  exit(-1);
}

static std::shared_ptr<void> mlx_error_handler_data_ = nullptr;
static mlx_error_handler_func mlx_error_handler_ = mlx_error_handler_default_;

extern "C" size_t mlx_error_static_storage_bytes(void) {
  return sizeof(mlx_error_handler_data_) + sizeof(mlx_error_handler_);
}

extern "C" void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*)) {
  if (dtor) {
    mlx_error_handler_data_ = std::shared_ptr<void>(data, dtor);
  } else {
    mlx_error_handler_data_ = nullptr;
  }
  if (handler) {
    mlx_error_handler_ = handler;
  } else {
    mlx_error_handler_ = mlx_error_handler_default_;
  }
}

namespace {
thread_local unsigned submission_tracking_failure = 0;
thread_local unsigned graph_metadata_failure = 0;
thread_local void* physical_backing_failure = nullptr;
}
extern "C" void* mlx_error_physical_backing_failure(void) { return physical_backing_failure; }
extern "C" unsigned mlx_error_submission_tracking_failure(void) {
  return submission_tracking_failure;
}
extern "C" unsigned mlx_error_graph_metadata_failure(void) {
  return graph_metadata_failure;
}
extern "C" void
_mlx_error(const char* file, const int line, const char* fmt, ...) {
  // Original safe operations translate the returned status through their exact
  // retained observer. Do not format an additional diagnostic or invoke the
  // process-wide handler. The carrier explicitly records source_unavailable
  // when a C error is reported outside a catch; it invents no native exception.
  auto* scope = mlx::core::submission::current_scope();
  if (scope && scope->original_controls_required()) {
    if (auto* carrier = scope->failure_owner().get()) {
      carrier->capture_current_exception();
    }
    return;
  }
  submission_tracking_failure = 0;
  graph_metadata_failure = 0;
  physical_backing_failure = nullptr;
  if (auto current = std::current_exception()) {
    try { std::rethrow_exception(current); }
    catch (const mlx::core::submission::RecordQuotaError& error) {
      submission_tracking_failure = static_cast<unsigned>(error.cause());
    } catch (const mlx::core::submission::GraphQuotaError& error) {
      graph_metadata_failure = static_cast<unsigned>(error.cause());
    } catch (const mlx::core::allocator::PhysicalBackingAdmissionError& error) {
      physical_backing_failure = error.cause.get();
    } catch (...) {}
  }
  va_list args, args_copy;
  va_start(args, fmt);

  // compute total size
  va_copy(args_copy, args);
  int size = vsnprintf(nullptr, 0, fmt, args_copy);
  va_end(args_copy);
  int size_loc = snprintf(nullptr, 0, " at %s:%d", file, line);

  std::vector<char> msg(size + size_loc + 1); // \0 at the end
  size = vsnprintf(msg.data(), size + 1, fmt, args);
  snprintf(msg.data() + size, size_loc + 1, " at %s:%d", file, line);
  va_end(args);

  mlx_error_handler_(msg.data(), mlx_error_handler_data_.get());
}
