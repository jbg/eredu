// Copyright © 2026 Apple Inc.

#ifndef MLX_HOST_TRANSFER_PRIVATE_H
#define MLX_HOST_TRANSFER_PRIVATE_H

#include "mlx/c/host_transfer.h"
#include "mlx/host_transfer.h"
#include "mlx/submission.h"

namespace mlx_c_detail {
struct OriginalHostBuffer final : mlx::core::submission::ScopedRetirement {
  explicit OriginalHostBuffer(mlx::core::submission::Scope* owner) noexcept
      : scope(owner), quota(owner->graph_quota()) {}
  mlx::core::HostTransferBuffer value;
  mlx::core::submission::Scope* scope;
  mlx::core::submission::GraphQuota* quota;
  void release() noexcept {
    auto* retained = scope;
    auto* arena = quota;
    this->~OriginalHostBuffer();
    arena->deallocate(this, sizeof(OriginalHostBuffer), alignof(OriginalHostBuffer));
    retained->release();
  }
};
}

inline mlx_host_transfer_buffer mlx_host_transfer_buffer_new_() {
  return mlx_host_transfer_buffer({nullptr});
}

inline mlx_host_transfer_buffer mlx_host_transfer_buffer_new_(
    mlx::core::HostTransferBuffer&& buffer) {
  return mlx_host_transfer_buffer(
      {new mlx::core::HostTransferBuffer(std::move(buffer))});
}

inline mlx::core::HostTransferBuffer& mlx_host_transfer_buffer_get_(
    mlx_host_transfer_buffer buffer) {
  if (!buffer.ctx) {
    throw std::runtime_error("expected a non-empty mlx_host_transfer_buffer");
  }
  return *static_cast<mlx::core::HostTransferBuffer*>(buffer.ctx);
}

inline void mlx_host_transfer_buffer_set_(
    mlx_host_transfer_buffer& destination,
    mlx::core::HostTransferBuffer&& source) {
  if (destination.ctx) {
    *static_cast<mlx::core::HostTransferBuffer*>(destination.ctx) =
        std::move(source);
  } else {
    destination.ctx =
        new mlx::core::HostTransferBuffer(std::move(source));
  }
}

inline void mlx_host_transfer_buffer_free_(mlx_host_transfer_buffer buffer) {
  if (buffer.prepared_owner) {
    static_cast<mlx_c_detail::OriginalHostBuffer*>(buffer.prepared_owner)->release();
  } else {
    delete static_cast<mlx::core::HostTransferBuffer*>(buffer.ctx);
  }
}

#endif
