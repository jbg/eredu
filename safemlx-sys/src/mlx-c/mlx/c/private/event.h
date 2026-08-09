/* Copyright © 2026 Apple Inc. */

#ifndef MLX_EVENT_PRIVATE_H
#define MLX_EVENT_PRIVATE_H

#include "mlx/c/event.h"
#include "mlx/mlx.h"

inline mlx_event mlx_event_new_() {
  return mlx_event({new mlx::core::Completion()});
}

inline mlx_event mlx_event_new_(mlx::core::Completion&& event) {
  return mlx_event({new mlx::core::Completion(std::move(event))});
}

inline mlx_event& mlx_event_set_(
    mlx_event& destination,
    mlx::core::Completion&& event) {
  if (destination.ctx) {
    *static_cast<mlx::core::Completion*>(destination.ctx) = std::move(event);
  } else {
    destination.ctx = new mlx::core::Completion(std::move(event));
  }
  return destination;
}

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
