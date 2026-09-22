// Copyright © 2026 Eredu contributors.
#include "mlx/c/host_transfer.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/private/prepared_array_output.h"
#include "mlx/host_transfer.h"
#include <limits>

extern "C" bool mlx_ordinary_host_buffer_observed_controls(
    size_t* out, mlx_prepared_input_runtime runtime, size_t rank) {
  if (!out || !runtime.allocator) return false;
  return mlx::core::ordinary_host_buffer_observed_controls(
      *static_cast<mlx::core::allocator::Allocator*>(runtime.allocator), rank, *out);
}

extern "C" size_t mlx_ordinary_host_buffer_wrapper_controls(size_t rank) {
  using namespace mlx::core;
  if (rank > Shape::inline_capacity) return 0;
  // Final C-owned HostTransferBuffer, its moved construction value and C call
  // arguments. Its native shared owner is observed separately, not counted here.
  return sizeof(HostTransferBuffer) * 2 + sizeof(Shape) + sizeof(mlx_host_transfer_buffer) +
      sizeof(mlx_host_transfer_buffer*) + sizeof(const int*) + sizeof(int) +
      sizeof(mlx_dtype) + sizeof(mlx_host_transfer_policy) + sizeof(size_t) * 2;
}

extern "C" size_t mlx_ordinary_host_copy_wrapper_controls(void) {
  using namespace mlx::core;
  // Actual ordinary output shells and preparers. Core graph construction,
  // eval_impl, dispatch and physical output storage require separate facts.
  return sizeof(array) + sizeof(Completion) + sizeof(mlx_array_output_preparation_) +
      sizeof(mlx_event_preparation_) + sizeof(DeviceTransfer) + sizeof(mlx_array*) +
      sizeof(mlx_event*) + sizeof(mlx_host_transfer_buffer) + sizeof(mlx_stream) + sizeof(int);
}

extern "C" size_t mlx_ordinary_array_to_host_wrapper_controls(void) {
  using namespace mlx::core;
  return sizeof(HostTransferBuffer) + sizeof(Completion) +
      sizeof(mlx_output_preparation_<HostTransferBuffer, mlx_host_transfer_buffer>) +
      sizeof(mlx_event_preparation_) + sizeof(HostTransfer) +
      sizeof(mlx_host_transfer_buffer*) + sizeof(mlx_event*) + sizeof(mlx_array) +
      sizeof(mlx_host_transfer_policy) + sizeof(mlx_stream) + sizeof(int);
}

extern "C" unsigned mlx_ordinary_host_buffer_capacity(
    size_t* capacity, mlx_memory_placement* placement,
    mlx_prepared_input_runtime runtime, size_t bytes) {
  if (!capacity || !placement || !runtime.allocator || bytes > static_cast<size_t>(PTRDIFF_MAX)) return 2;
  try {
    auto& selected = *static_cast<mlx::core::allocator::Allocator*>(runtime.allocator);
    const auto policy = mlx::core::allocator::HostTransferPolicy::transfer;
    const auto kind = selected.host_transfer_storage_kind(policy);
    // The retained initializer captured this same allocator/policy placement.
    // A different current mechanism cannot inherit its physical-domain fact.
    if (static_cast<unsigned>(kind) != runtime.storage_kind) return 1;
    const auto value = selected.host_transfer_capacity_upper_bound(bytes, policy);
    if (value > static_cast<size_t>(PTRDIFF_MAX) || value < bytes) return 2;
    *capacity = value;
    *placement = runtime.placement;
    return 0;
  } catch (const std::overflow_error&) { return 2; }
    catch (const std::length_error&) { return 2; }
    catch (...) { return 1; }
}

extern "C" size_t mlx_ordinary_event_wrapper_controls(void) {
  using namespace mlx::core;
  // ArrayVector element buffers use GraphAllocator and are observed separately.
  // The C vector and Completion shells are ordinary wrapper allocations.
  return sizeof(ArrayVector) + sizeof(Completion) + sizeof(mlx_event_preparation_) +
      sizeof(mlx_vector_array) + sizeof(mlx_event*) + sizeof(int) * 2;
}

extern "C" size_t mlx_ordinary_stream_clone_wrapper_controls(void) {
  return sizeof(mlx::core::Stream) + sizeof(mlx_stream) * 2 +
      sizeof(mlx_stream*) + sizeof(int);
}
