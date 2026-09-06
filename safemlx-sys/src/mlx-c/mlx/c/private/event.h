/* Copyright © 2026 Apple Inc. */

#ifndef MLX_EVENT_PRIVATE_H
#define MLX_EVENT_PRIVATE_H

#include "mlx/c/event.h"
#include "mlx/c/private/prepared_output.h"
#include "mlx/mlx.h"

inline mlx_event mlx_event_new_() {
  return mlx_event({new mlx::core::Completion()});
}

// Prepare publication storage before a producer can enqueue native work.
// Publishing its successful result must not introduce a later allocation or
// exception that would discard the only completion handle.
using mlx_event_preparation_ =
    mlx_output_preparation_<mlx::core::Completion, mlx_event>;

inline mlx::core::Completion& mlx_event_get_(mlx_event event) {
  if (!event.ctx) {
    throw std::runtime_error("expected a non-empty mlx_event");
  }
  return *static_cast<mlx::core::Completion*>(event.ctx);
}

inline void mlx_event_free_(mlx_event event) {
  delete static_cast<mlx::core::Completion*>(event.ctx);
}

#endif
