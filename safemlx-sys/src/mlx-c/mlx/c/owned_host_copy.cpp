#include "mlx/c/owned_host_copy.h"
#include "mlx/c/private/enums.h"
#include "mlx/c/private/memory_placement.h"
#include "mlx/prepared_input.h"
#include "mlx/submission.h"

using namespace mlx::core;
namespace {
using Cause = PreparedHostCopyCause;
allocator::PreparedInputFacts facts(mlx_prepared_input_runtime runtime) noexcept {
  return {runtime.page_size, runtime.maximum,
      static_cast<allocator::HostTransferStorageKind>(runtime.storage_kind), runtime.controls, {runtime.placement.kind, runtime.placement.device, runtime.placement.device_count}};
}
bool dtype_valid(mlx_dtype dtype) noexcept {
  return static_cast<unsigned>(dtype) > MLX_BOOL &&
      static_cast<unsigned>(dtype) <= MLX_COMPLEX64;
}
}
extern "C" unsigned mlx_owned_host_copy_layout_for(mlx_owned_host_copy_layout* out,
    mlx_prepared_input_runtime runtime, const int* shape, size_t rank, mlx_dtype dtype) {
  if (!out || !runtime.allocator || !dtype_valid(dtype)) return unsigned(Cause::invalid);
  PreparedHostCopyLayout layout;
  if (!PreparedHostCopySlot::layout(facts(runtime), shape, rank, mlx_dtype_to_cpp(dtype), layout))
    return unsigned(Cause::invalid);
  *out = {layout.metadata_bytes, layout.backing_bytes, layout.copy_bytes,
      layout.handle_bytes, layout.handle_alignment, layout.controls +
      sizeof(mlx_owned_host_copy_layout) + sizeof(mlx_owned_host_copy_slot) +
      sizeof(mlx_prepared_input_runtime) + sizeof(mlx_submission_graph_quota) +
      sizeof(mlx_submission_observer) + sizeof(mlx_array) + sizeof(array*) +
      sizeof(Cause) + sizeof(const void*) + sizeof(size_t) +
      sizeof(const array::Data*) + sizeof(mlx_original_buffer_info) +
      sizeof(mlx_original_buffer_info*), MLX_OWNED_HOST_COPY_STRATEGY};
  return unsigned(Cause::success);
}
extern "C" unsigned mlx_owned_host_copy_new(mlx_owned_host_copy_slot* out,
    mlx_prepared_input_runtime runtime, mlx_submission_graph_quota arena,
    const int* shape, size_t rank, mlx_dtype dtype) {
  if (!out || out->ctx || !runtime.allocator || !arena.ctx || !dtype_valid(dtype))
    return unsigned(Cause::invalid);
  PreparedHostCopySlot* slot = nullptr;
  const auto status = PreparedHostCopySlot::create(
      *static_cast<allocator::Allocator*>(runtime.allocator), facts(runtime),
      static_cast<submission::GraphQuota*>(arena.ctx), shape, rank,
      mlx_dtype_to_cpp(dtype), slot);
  if (status == Cause::success) out->ctx = slot;
  return unsigned(status);
}
extern "C" void mlx_owned_host_copy_free(mlx_owned_host_copy_slot value) {
  if (value.ctx) static_cast<PreparedHostCopySlot*>(value.ctx)->destroy();
}
namespace {
unsigned fill(mlx_array* out, mlx_original_buffer_info* birth,
    mlx_owned_host_copy_slot value, mlx_submission_observer observer,
    const void* data, size_t bytes) {
  if (!out || out->ctx || out->prepared_owner || !value.ctx) return unsigned(Cause::invalid);
  if (!observer.ctx) return unsigned(Cause::domain);
  PreparedInputArray* published = nullptr;
  const auto status = static_cast<PreparedHostCopySlot*>(value.ctx)->fill(
      *static_cast<submission::Scope*>(observer.ctx), data, bytes, published);
  if (status == Cause::success) {
    // The one successful producer still owns this exact Data. Reading its
    // immutable fields cannot fail or call a provider after handle publication.
    if (birth) {
      const auto* actual = published->value().data_shared_ptr().get();
      *birth = {actual->allocation_generation != 0, actual->allocation_generation,
          actual->original_input->capacity,
          mlx_placement_to_c(allocator::memory_placement(actual->original_input->buffer)), 0};
    }
    *out = mlx_array{&published->value(), published};
  }
  return unsigned(status);
}
} // namespace
extern "C" unsigned mlx_owned_host_copy_fill(mlx_array* out,
    mlx_owned_host_copy_slot value, mlx_submission_observer observer,
    const void* data, size_t bytes) {
  return fill(out, nullptr, value, observer, data, bytes);
}
extern "C" unsigned mlx_owned_host_copy_fill_completed(mlx_array* out,
    mlx_original_buffer_info* birth, mlx_owned_host_copy_slot value,
    mlx_submission_observer observer, const void* data, size_t bytes) {
  if (!birth) return unsigned(Cause::invalid);
  return fill(out, birth, value, observer, data, bytes);
}
extern "C" unsigned mlx_owned_host_copy_state(mlx_owned_host_copy_slot value) {
  return value.ctx ? unsigned(static_cast<PreparedHostCopySlot*>(value.ctx)->attempted()) : 2;
}
