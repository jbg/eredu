// Copyright © 2026 Apple Inc.

#include "mlx/c/host_transfer.h"

#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

namespace {

mlx::core::HostTransferPolicy policy_to_cpp(mlx_host_transfer_policy policy) {
  switch (policy) {
    case MLX_HOST_TRANSFER_POLICY_TRANSFER:
      return mlx::core::HostTransferPolicy::transfer;
    case MLX_HOST_TRANSFER_POLICY_MANAGED:
      return mlx::core::HostTransferPolicy::managed;
  }
  throw std::invalid_argument("Unknown mlx_host_transfer_policy value.");
}

mlx_host_transfer_policy policy_from_cpp(
    mlx::core::HostTransferPolicy policy) {
  switch (policy) {
    case mlx::core::HostTransferPolicy::transfer:
      return MLX_HOST_TRANSFER_POLICY_TRANSFER;
    case mlx::core::HostTransferPolicy::managed:
      return MLX_HOST_TRANSFER_POLICY_MANAGED;
  }
  throw std::invalid_argument("Unknown HostTransferPolicy value.");
}

mlx_host_transfer_storage_kind storage_kind_from_cpp(
    mlx::core::HostTransferStorageKind kind) {
  switch (kind) {
    case mlx::core::HostTransferStorageKind::cpu:
      return MLX_HOST_TRANSFER_STORAGE_CPU;
    case mlx::core::HostTransferStorageKind::metal_shared:
      return MLX_HOST_TRANSFER_STORAGE_METAL_SHARED;
    case mlx::core::HostTransferStorageKind::cuda_pinned:
      return MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED;
    case mlx::core::HostTransferStorageKind::cuda_managed:
      return MLX_HOST_TRANSFER_STORAGE_CUDA_MANAGED;
  }
  throw std::invalid_argument("Unknown HostTransferStorageKind value.");
}

} // namespace

extern "C" int mlx_host_transfer_buffer_new(
    mlx_host_transfer_buffer* buffer,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    mlx_host_transfer_policy policy) {
  try {
    if (!buffer || dim < 0 || (dim > 0 && !shape)) {
      throw std::invalid_argument("Invalid host transfer buffer arguments.");
    }
    auto cpp_shape = dim > 0 ? mlx::core::Shape(shape, shape + dim)
                             : mlx::core::Shape{};
    mlx_host_transfer_buffer_set_(
        *buffer,
        mlx::core::HostTransferBuffer(
            std::move(cpp_shape),
            mlx_dtype_to_cpp(dtype),
            policy_to_cpp(policy)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_host_transfer_buffer_free(
    mlx_host_transfer_buffer buffer) {
  try {
    mlx_host_transfer_buffer_free_(buffer);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_copy_to_host(
    mlx_host_transfer_buffer* buffer,
    mlx_event* event,
    mlx_array source,
    mlx_host_transfer_policy policy,
    mlx_stream stream) {
  try {
    if (!buffer || !event) {
      throw std::invalid_argument("Host copy output pointers must be non-null.");
    }
    auto transfer = mlx::core::copy_to_host(
        mlx_array_get_(source), policy_to_cpp(policy), mlx_stream_get_(stream));
    mlx_host_transfer_buffer_set_(*buffer, std::move(transfer.buffer));
    mlx_event_set_(*event, std::move(transfer.completion));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_copy_from_host(
    mlx_array* array,
    mlx_event* event,
    mlx_host_transfer_buffer source,
    mlx_stream stream) {
  try {
    if (!array || !event) {
      throw std::invalid_argument("Device copy output pointers must be non-null.");
    }
    auto transfer = mlx::core::copy_from_host(
        mlx_host_transfer_buffer_get_(source), mlx_stream_get_(stream));
    mlx_array_set_(*array, std::move(transfer.value));
    mlx_event_set_(*event, std::move(transfer.completion));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

#define MLX_HOST_BUFFER_GETTER(name, type, expression)                       \
  extern "C" int name(type* output, mlx_host_transfer_buffer buffer) {       \
    try {                                                                     \
      if (!output) {                                                          \
        throw std::invalid_argument("Host transfer output must be non-null."); \
      }                                                                       \
      *output = (expression);                                                 \
      return 0;                                                               \
    } catch (std::exception & e) {                                            \
      mlx_error(e.what());                                                    \
      return 1;                                                               \
    }                                                                         \
  }

MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_data,
    const void*,
    mlx_host_transfer_buffer_get_(buffer).data())
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_data_mut,
    void*,
    mlx_host_transfer_buffer_get_(buffer).data())
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_shape,
    const int*,
    mlx_host_transfer_buffer_get_(buffer).shape().data())
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_ndim,
    size_t,
    mlx_host_transfer_buffer_get_(buffer).shape().size())
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_size,
    size_t,
    mlx_host_transfer_buffer_get_(buffer).size())
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_nbytes,
    size_t,
    mlx_host_transfer_buffer_get_(buffer).nbytes())
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_capacity,
    size_t,
    mlx_host_transfer_buffer_get_(buffer).capacity())
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_dtype,
    mlx_dtype,
    mlx_dtype_to_c(mlx_host_transfer_buffer_get_(buffer).dtype()))
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_policy,
    mlx_host_transfer_policy,
    policy_from_cpp(mlx_host_transfer_buffer_get_(buffer).policy()))
MLX_HOST_BUFFER_GETTER(
    mlx_host_transfer_buffer_storage_kind,
    mlx_host_transfer_storage_kind,
    storage_kind_from_cpp(
        mlx_host_transfer_buffer_get_(buffer).storage_kind()))

#undef MLX_HOST_BUFFER_GETTER
