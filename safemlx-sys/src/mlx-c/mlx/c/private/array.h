/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_ARRAY_PRIVATE_H
#define MLX_ARRAY_PRIVATE_H

#include "mlx/c/array.h"
#include "mlx/mlx.h"
#include "mlx/prepared_input.h"
#include "mlx/submission.h"

inline mlx::core::submission::GraphQuota* mlx_array_original_graph_() {
  using namespace mlx::core::submission;
  const auto controls = current_native_controls();
  if (!controls.original) return nullptr;
  if (controls.failure != NativeControlFailure::none)
    throw NativeControlError(controls.failure);
  mark_native_control_construction();
  return controls.graph;
}

inline mlx_array mlx_array_new_original_(const mlx::core::array& value,
    mlx::core::submission::GraphQuota* graph) {
  using mlx::core::PreparedInputArray;
  // The produced value stays owned by the caller until the final handle is
  // published. If allocation refuses, its partial graph retires normally.
  void* storage = graph->allocate(sizeof(PreparedInputArray), alignof(PreparedInputArray));
  auto* owner = PreparedInputArray::publish_storage(storage, value, graph);
  return mlx_array({&owner->value(), owner});
}

inline mlx_array mlx_array_new_() {
  return mlx_array({nullptr, nullptr});
}

inline mlx_array mlx_array_new_(const mlx::core::array& s) {
  if (auto* graph = mlx_array_original_graph_()) return mlx_array_new_original_(s, graph);
  return mlx_array({new mlx::core::array(s), nullptr});
}

inline mlx_array mlx_array_new_(mlx::core::array&& s) {
  if (auto* graph = mlx_array_original_graph_()) return mlx_array_new_original_(s, graph);
  return mlx_array({new mlx::core::array(std::move(s)), nullptr});
}


inline mlx_array& mlx_array_set_(mlx_array& d, const mlx::core::array& s) {
  if (d.ctx) {
    *static_cast<mlx::core::array*>(d.ctx) = s;
  } else {
    d = mlx_array_new_(s);
  }
  return d;
}

inline mlx_array& mlx_array_set_(mlx_array& d, mlx::core::array&& s) {
  if (d.ctx) {
    *static_cast<mlx::core::array*>(d.ctx) = std::move(s);
  } else {
    d = mlx_array_new_(std::move(s));
  }
  return d;
}


inline mlx::core::array& mlx_array_get_(mlx_array d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_array");
  }
  return *static_cast<mlx::core::array*>(d.ctx);
}

inline void mlx_array_free_(mlx_array d) {
  if (d.prepared_owner) {
    static_cast<mlx::core::PreparedInputArray*>(d.prepared_owner)->destroy();
  } else if (d.ctx) {
    delete static_cast<mlx::core::array*>(d.ctx);
  }
}

#endif
