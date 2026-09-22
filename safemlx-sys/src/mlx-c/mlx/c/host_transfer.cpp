// Copyright © 2026 Apple Inc.

#include "mlx/c/host_transfer.h"

#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/private/allocation_owner.h"
#include "mlx/c/private/prepared_array_output.h"
#include "mlx/c/private/operation_event.h"
#include "mlx/prepared_input.h"

extern "C" unsigned mlx_copy_from_host_operation(
    mlx_array* output, mlx_operation_event token,
    mlx_host_transfer_buffer source, mlx_stream stream) {
  using namespace mlx::core;
  auto* event = mlx_c_detail::operation_get(token);
  if (!output || output->ctx || output->prepared_owner || !source.ctx || !stream.ctx ||
      !event || event->submitted || !mlx_c_detail::operation_current(*event)) return 4;
  auto* quota = event->quota;
  void* storage = quota->try_allocate(sizeof(PreparedInputArray), alignof(PreparedInputArray));
  if (!storage) return 2;
  event->submitted = true;
  try {
    auto transfer = copy_from_host(mlx_host_transfer_buffer_get_(source), mlx_stream_get_(stream));
    auto* owner = PreparedInputArray::publish_storage(storage, transfer.value, quota);
    event->completion = std::move(transfer.completion);
    *output = {&owner->value(), owner};
    return 0;
  } catch (const ScopedEvaluationError& refusal) {
    quota->deallocate(storage, sizeof(PreparedInputArray), alignof(PreparedInputArray));
    return static_cast<unsigned>(refusal.outcome());
  } catch (...) {
    quota->deallocate(storage, sizeof(PreparedInputArray), alignof(PreparedInputArray));
    return mlx_c_detail::operation_failure(*event);
  }
}

extern "C" unsigned mlx_copy_to_prepared_host_operation(
    mlx_array* output, mlx_operation_event token,
    mlx_array source, void* prepared_destination, mlx_stream stream) {
  using namespace mlx::core;
  auto* event = mlx_c_detail::operation_get(token);
  if (!output || output->ctx || output->prepared_owner || !source.ctx ||
      !prepared_destination || !stream.ctx || !event || event->submitted ||
      !mlx_c_detail::operation_current(*event)) return 4;
  auto* quota = event->quota;
  void* storage = quota->try_allocate(sizeof(PreparedInputArray), alignof(PreparedInputArray));
  if (!storage) return 2;
  event->submitted = true;
  try {
    auto transfer = copy_to_host_into(mlx_array_get_(source),
        static_cast<PreparedHostTransferSource*>(prepared_destination)->value(),
        mlx_stream_get_(stream));
    auto* owner = PreparedInputArray::publish_storage(storage, transfer.value, quota);
    event->completion = std::move(transfer.completion);
    *output = {&owner->value(), owner};
    return 0;
  } catch (const ScopedEvaluationError& refusal) {
    quota->deallocate(storage, sizeof(PreparedInputArray), alignof(PreparedInputArray));
    return static_cast<unsigned>(refusal.outcome());
  } catch (...) {
    quota->deallocate(storage, sizeof(PreparedInputArray), alignof(PreparedInputArray));
    return mlx_c_detail::operation_failure(*event);
  }
}

extern "C" unsigned mlx_copy_to_host_operation(
    mlx_host_transfer_buffer* output, mlx_operation_event token,
    mlx_array source, mlx_host_transfer_policy policy, mlx_stream stream) {
  using namespace mlx::core;
  using mlx_c_detail::OriginalHostBuffer;
  auto* event = mlx_c_detail::operation_get(token);
  if (!output || output->ctx || output->prepared_owner || !source.ctx || !stream.ctx ||
      !event || event->submitted || !mlx_c_detail::operation_current(*event)) return 4;
  if (policy != MLX_HOST_TRANSFER_POLICY_TRANSFER && policy != MLX_HOST_TRANSFER_POLICY_MANAGED) return 1;
  void* storage = event->quota->try_allocate(sizeof(OriginalHostBuffer), alignof(OriginalHostBuffer));
  if (!storage) return 2;
  event->scope->retain_observer();
  auto* owner = ::new (storage) OriginalHostBuffer(event->scope);
  event->submitted = true;
  try {
    auto transfer = copy_to_host(mlx_array_get_(source),
        policy == MLX_HOST_TRANSFER_POLICY_TRANSFER ? HostTransferPolicy::transfer : HostTransferPolicy::managed,
        mlx_stream_get_(stream));
    owner->value = std::move(transfer.buffer);
    event->completion = std::move(transfer.completion);
    *output = {&owner->value, owner};
    return 0;
  } catch (const ScopedEvaluationError& refusal) {
    owner->release();
    return static_cast<unsigned>(refusal.outcome());
  } catch (...) {
    owner->release();
    return mlx_c_detail::operation_failure(*event);
  }
}

extern "C" void mlx_host_transfer_buffer_defer_original(mlx_host_transfer_buffer value) {
  if (auto* owner = static_cast<mlx_c_detail::OriginalHostBuffer*>(value.prepared_owner)) {
    owner->destroy = [](mlx::core::submission::ScopedRetirement* node) noexcept {
      static_cast<mlx_c_detail::OriginalHostBuffer*>(node)->release();
    };
    owner->scope->defer_retirement(owner);
  }
}
extern "C" bool mlx_original_host_copy_layout(size_t rank, mlx_dtype dtype,
    size_t* controls, size_t* direct_graph_extent) {
  using namespace mlx::core;
  size_t native = 0, extent = 0;
  if (!controls || !direct_graph_extent || dtype < MLX_BOOL || dtype > MLX_COMPLEX64 ||
      !original_host_copy_control_bytes(rank, mlx_dtype_to_cpp(dtype), native) ||
      !submission::GraphQuota::allocation_extent(sizeof(PreparedInputArray),
          alignof(PreparedInputArray), extent)) return false;
  const size_t fixed = sizeof(mlx_array) + sizeof(mlx_operation_event) +
      sizeof(mlx_host_transfer_buffer) + sizeof(mlx_stream) + 6 * sizeof(void*) +
      4 * sizeof(size_t) + sizeof(mlx_dtype) + sizeof(unsigned);
  if (native > SIZE_MAX - fixed) return false;
  *controls = native + fixed;
  *direct_graph_extent = extent;
  return true;
}

extern "C" bool mlx_original_host_store_layout(size_t rank, mlx_dtype dtype,
    size_t* controls, size_t* direct_graph_extent) {
  using namespace mlx::core;
  size_t native = 0, extent = 0;
  if (!controls || !direct_graph_extent || dtype < MLX_BOOL || dtype > MLX_COMPLEX64 ||
      !original_host_store_control_bytes(rank, mlx_dtype_to_cpp(dtype), native) ||
      !submission::GraphQuota::allocation_extent(sizeof(PreparedInputArray),
          alignof(PreparedInputArray), extent)) return false;
  const size_t fixed = sizeof(mlx_array) + sizeof(mlx_operation_event) +
      sizeof(mlx_host_transfer_buffer) + sizeof(mlx_stream) + 6 * sizeof(void*) +
      4 * sizeof(size_t) + sizeof(mlx_dtype) + sizeof(unsigned);
  if (native > SIZE_MAX - fixed) return false;
  *controls = native + fixed;
  *direct_graph_extent = extent;
  return true;
}

extern "C" size_t mlx_operation_host_transfer_control_bytes(void) {
  return sizeof(mlx_c_detail::OriginalHostBuffer) + sizeof(mlx::core::PreparedInputArray) +
      sizeof(mlx::core::HostTransfer) + sizeof(mlx::core::DeviceTransfer) +
      sizeof(mlx_array) + sizeof(mlx_host_transfer_buffer) + sizeof(void*) * 4 + sizeof(unsigned);
}

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

mlx::core::HostTransferStorageKind storage_kind_to_cpp(
    mlx_host_transfer_storage_kind kind) {
  switch (kind) {
    case MLX_HOST_TRANSFER_STORAGE_CPU:
      return mlx::core::HostTransferStorageKind::cpu;
    case MLX_HOST_TRANSFER_STORAGE_METAL_SHARED:
      return mlx::core::HostTransferStorageKind::metal_shared;
    case MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED:
      return mlx::core::HostTransferStorageKind::cuda_pinned;
    case MLX_HOST_TRANSFER_STORAGE_CUDA_MANAGED:
      return mlx::core::HostTransferStorageKind::cuda_managed;
  }
  throw std::invalid_argument("Unknown mlx_host_transfer_storage_kind value.");
}

} // namespace

extern "C" int mlx_host_transfer_memory_stats_get(
    mlx_host_transfer_memory_stats* stats,
    mlx_host_transfer_storage_kind kind) {
  try {
    if (!stats) {
      throw std::invalid_argument("Host transfer stats output must be non-null.");
    }
    const auto cpp_stats =
        mlx::core::host_transfer_memory_stats(storage_kind_to_cpp(kind));
    *stats = {
        cpp_stats.active_bytes,
        cpp_stats.peak_bytes,
        cpp_stats.active_allocations,
        cpp_stats.peak_allocations};
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_host_transfer_memory_stats_reset_peak(
    mlx_host_transfer_storage_kind kind) {
  try {
    mlx::core::reset_host_transfer_peak_memory(storage_kind_to_cpp(kind));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_host_transfer_capacity_upper_bound(
    size_t* capacity,
    size_t nbytes,
    mlx_host_transfer_policy policy) {
  try {
    if (!capacity) {
      throw std::invalid_argument(
          "Host transfer capacity output must be non-null.");
    }
    *capacity = mlx::core::host_transfer_capacity_upper_bound(
        nbytes, policy_to_cpp(policy));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

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
    mlx_output_preparation_<
        mlx::core::HostTransferBuffer,
        mlx_host_transfer_buffer>
        buffer_output(*buffer);
    mlx_event_preparation_ event_output(*event);
    auto transfer = mlx::core::copy_to_host(
        mlx_array_get_(source), policy_to_cpp(policy), mlx_stream_get_(stream));
    buffer_output.publish(std::move(transfer.buffer));
    event_output.publish(std::move(transfer.completion));
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
    mlx_array_output_preparation_ array_output(*array);
    mlx_event_preparation_ event_output(*event);
    auto transfer = mlx::core::copy_from_host(
        mlx_host_transfer_buffer_get_(source), mlx_stream_get_(stream));
    array_output.publish(std::move(transfer.value));
    event_output.publish(std::move(transfer.completion));
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
    mlx_host_transfer_buffer_allocation_identity,
    uint64_t,
    mlx_host_transfer_buffer_get_(buffer).allocation_identity())

extern "C" int mlx_host_transfer_buffer_attach_prepared_allocation_owner(
    int* outcome, mlx_host_transfer_buffer buffer, void* node,
    void* payload, void (*release)(void*)) {
  try {
    if (!outcome || !node || !payload || !release) {
      throw std::invalid_argument("Invalid prepared host allocation owner.");
    }
    *outcome = 0;
    const auto& value = mlx_host_transfer_buffer_get_(buffer);
    if (!value.allocation_identity()) {
      *outcome = 1;
      return 0;
    }
    if (!value.retain_prepared_allocation_owner(
            static_cast<mlx::core::AllocationOwners::Node*>(node), payload, release)) {
      return 0;
    }
    *outcome = 2;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" size_t mlx_host_transfer_buffer_prepared_owner_control_bytes(void) {
  // This C adapter, HostTransferBuffer's source check, storage delegation and
  // AllocationOwners::append_prepared all borrow the same supplied nodes.
  return sizeof(int*) + sizeof(mlx_host_transfer_buffer) + sizeof(int) +
      sizeof(const mlx::core::HostTransferBuffer*) + sizeof(std::exception*) +
      4 * (sizeof(mlx::core::AllocationOwners::Node*) + sizeof(void*) +
           sizeof(void (*)(void*))) + sizeof(mlx::core::AllocationOwners*) +
      sizeof(uint64_t) + sizeof(bool);
}

extern "C" int mlx_host_transfer_buffer_retain_allocation_owner(
    bool* attached, mlx_host_transfer_buffer buffer, void* payload, void (*release)(void*)) {
  try {
    if (!attached || !payload || !release) {
      throw std::invalid_argument("Invalid host allocation-owner attachment.");
    }
    *attached = false;
    const auto& value = mlx_host_transfer_buffer_get_(buffer);
    if (value.allocation_identity() == 0) {
      return 0;
    }
    auto owner = std::make_shared<mlx_c_detail::AllocationOwnerPayload>();
    if (!value.retain_allocation_owner(owner)) {
      return 0;
    }
    // No throwing operation follows the ownership-transfer point.
    owner->release = release;
    owner->payload = payload;
    *attached = true;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

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

namespace {
mlx::core::allocator::PreparedInputFacts prepared_host_facts(mlx_prepared_input_runtime runtime) noexcept {
  return {runtime.page_size,runtime.maximum,
      static_cast<mlx::core::HostTransferStorageKind>(runtime.storage_kind),runtime.controls, {runtime.placement.kind, runtime.placement.device, runtime.placement.device_count}};
}
bool prepared_host_dtype(mlx_dtype dtype) noexcept {
  return static_cast<unsigned>(dtype) <= MLX_COMPLEX64;
}
}
extern "C" unsigned mlx_prepared_host_transfer_layout_for(mlx_prepared_host_transfer_layout* out,
    mlx_prepared_input_runtime runtime, const int* shape, size_t rank, mlx_dtype dtype, size_t array_handles) {
  using namespace mlx::core;
  if (!out || !runtime.allocator || !prepared_host_dtype(dtype)) return 2;
  PreparedHostTransferLayout layout;
  if (!PreparedHostTransferSource::layout(prepared_host_facts(runtime),shape,rank,mlx_dtype_to_cpp(dtype),array_handles,layout)) return 1;
  *out = {layout.metadata_bytes,layout.backing_bytes,layout.logical_bytes,
      layout.controls + sizeof(mlx_prepared_host_transfer_layout) + sizeof(mlx_prepared_input_runtime) +
      sizeof(mlx_submission_graph_quota) + sizeof(mlx_host_transfer_buffer) +
      sizeof(PreparedHostTransferSource*) + sizeof(unsigned) + sizeof(int*) + sizeof(size_t) +
      sizeof(mlx_array) + sizeof(mlx_array*) + sizeof(PreparedInputArray*) +
      sizeof(allocator::PreparedInputCause) + sizeof(void*)};
  return 0;
}
extern "C" unsigned mlx_prepared_host_transfer_new(mlx_host_transfer_buffer* out, void** owner,
    mlx_prepared_input_runtime runtime, mlx_submission_graph_quota arena,
    const int* shape, size_t rank, mlx_dtype dtype, size_t array_handles) {
  using namespace mlx::core;
  if (!out || out->ctx || out->prepared_owner || !owner || *owner || !runtime.allocator || !arena.ctx || !prepared_host_dtype(dtype)) return 2;
  PreparedHostTransferSource* source = nullptr;
  auto cause = PreparedHostTransferSource::create(*static_cast<allocator::Allocator*>(runtime.allocator),
      prepared_host_facts(runtime),static_cast<submission::GraphQuota*>(arena.ctx),shape,rank,mlx_dtype_to_cpp(dtype),array_handles,source);
  if (cause == allocator::PreparedInputCause::success) {
    *owner = source;
    *out = {&source->value(),nullptr};
  }
  return static_cast<unsigned>(cause);
}
extern "C" void mlx_prepared_host_transfer_free(void* source) {
  if (source) static_cast<mlx::core::PreparedHostTransferSource*>(source)->destroy();
}
extern "C" void* mlx_prepared_host_transfer_data(void* source) {
  return source ? static_cast<mlx::core::PreparedHostTransferSource*>(source)->value().data() : nullptr;
}

extern "C" unsigned mlx_prepared_host_transfer_array(mlx_array* out,void* value) {
  using namespace mlx::core;
  if (!out || out->ctx || out->prepared_owner || !value) return unsigned(allocator::PreparedInputCause::invalid);
  PreparedInputArray* owned=nullptr;
  const auto status=static_cast<PreparedHostTransferSource*>(value)->make_array(owned);
  if (status==allocator::PreparedInputCause::success) *out=mlx_array{&owned->value(),owned};
  return unsigned(status);
}

extern "C" uint64_t mlx_prepared_host_transfer_identity(void* source) {
  return source ? static_cast<mlx::core::PreparedHostTransferSource*>(source)->value().allocation_identity() : 0;
}
