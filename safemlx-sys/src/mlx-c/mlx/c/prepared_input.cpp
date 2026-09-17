#include "mlx/c/prepared_input.h"
#include "mlx/c/original_buffer.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/prepared_input.h"
#include "mlx/input_allocator.h"
#include "mlx/stream.h"
#include "mlx/submission.h"

using namespace mlx::core;
namespace {
allocator::PreparedInputFacts facts(mlx_prepared_input_runtime runtime) noexcept {
  return {runtime.page_size, runtime.maximum,
      static_cast<allocator::HostTransferStorageKind>(runtime.storage_kind), runtime.controls};
}
PreparedInputSource source(mlx_prepared_input_source value) noexcept {
  return {value.data, value.shape, value.rank, value.elements, value.kind};
}
}
extern "C" int mlx_prepared_input_runtime_prepare(mlx_prepared_input_runtime* out) {
  if (!out || out->allocator) return 1;
  try {
    auto& runtime = allocator::allocator();
    allocator::PreparedInputFacts prepared;
    if (!runtime.prepare_input_runtime(prepared)) return 1;
    *out = {&runtime, prepared.page_size, prepared.maximum, static_cast<unsigned>(prepared.kind), prepared.controls};
    return 0;
  } catch (const std::exception& error) { mlx_error(error.what()); return 2; }
}
extern "C" int mlx_prepared_input_layout_for(mlx_prepared_input_layout* out,
    mlx_prepared_input_runtime runtime, mlx_prepared_input_source input) {
  if (!out || !runtime.allocator) return 1;
  PreparedInputSlotLayout layout;
  if (!PreparedInputLeaf::layout(facts(runtime), source(input), layout)) return 1;
  *out = {layout.metadata_bytes, layout.backing_bytes, layout.controls +
      sizeof(mlx_prepared_input_runtime) + sizeof(mlx_prepared_input_source) +
      sizeof(mlx_prepared_input_layout) + sizeof(PreparedInputLeaf*) + sizeof(unsigned)};
  return 0;
}
extern "C" unsigned mlx_prepared_input_leaf_new(mlx_prepared_input_leaf* out,
    mlx_prepared_input_runtime runtime, mlx_submission_graph_quota arena,
    mlx_prepared_input_source input) {
  if (!out || out->ctx || !runtime.allocator || !arena.ctx) return 2;
  PreparedInputLeaf* leaf = nullptr;
  auto result = PreparedInputLeaf::create(*static_cast<allocator::Allocator*>(runtime.allocator),
      facts(runtime), static_cast<submission::GraphQuota*>(arena.ctx), source(input), leaf);
  if (result == allocator::PreparedInputCause::success) out->ctx = leaf;
  return static_cast<unsigned>(result);
}
extern "C" void mlx_prepared_input_leaf_free(mlx_prepared_input_leaf value) {
  if (value.ctx) static_cast<PreparedInputLeaf*>(value.ctx)->destroy();
}
extern "C" int mlx_prepared_input_leaf_clone_array(mlx_array* out, mlx_prepared_input_leaf value) {
  if (!out || !value.ctx) return 1;
  try {
    mlx_array_set_(*out, static_cast<PreparedInputLeaf*>(value.ctx)->value());
    return 0;
  } catch (const std::exception& error) { mlx_error(error.what()); return 1; }
}
extern "C" int mlx_prepared_input_array_layout(size_t* bytes, size_t* controls) {
  if (!bytes || !controls || !PreparedInputArray::layout(*bytes, *controls)) return 1;
  *controls += sizeof(mlx_array) + sizeof(PreparedInputArray*) + sizeof(unsigned);
  return 0;
}
extern "C" unsigned mlx_prepared_input_leaf_array(mlx_array* out, mlx_prepared_input_leaf leaf) {
  if (!out || out->ctx || out->prepared_owner || !leaf.ctx)
    return static_cast<unsigned>(allocator::PreparedInputCause::invalid);
  PreparedInputArray* owned = nullptr;
  auto status = static_cast<PreparedInputLeaf*>(leaf.ctx)->make_array(owned);
  if (status == allocator::PreparedInputCause::success)
    *out = mlx_array{&owned->value(), owned};
  return static_cast<unsigned>(status);
}
extern "C" int mlx_prepared_input_leaf_info(uint64_t* identity, size_t* capacity,
    mlx_prepared_input_leaf value) {
  if (!identity || !capacity || !value.ctx) return 1;
  const auto& data = static_cast<PreparedInputLeaf*>(value.ctx)->value().data_shared_ptr();
  if (!data || !data->original_input || !data->allocation_generation) return 1;
  *identity = data->allocation_generation;
  *capacity = data->original_input->capacity;
  return 0;
}

extern "C" int mlx_prepared_input_target_matches(mlx_stream stream, mlx_device device) {
  if (!stream.ctx || !device.ctx) return 0;
  return static_cast<const Stream*>(stream.ctx)->device ==
      *static_cast<const Device*>(device.ctx);
}

extern "C" size_t mlx_prepared_input_target_controls(void) {
  return sizeof(mlx_stream) + sizeof(mlx_device) + sizeof(const Stream*) +
      sizeof(const Device*) + sizeof(int);
}

extern "C" int mlx_original_mutable_pair_context_empty(void) {
  return submission::submission_context_empty() ? 1 : 0;
}
extern "C" unsigned mlx_original_mutable_pair_layout_for(
    mlx_original_mutable_pair_layout* out, mlx_prepared_input_runtime runtime) {
  using Status = PreparedHostCopyCause;
  if (!out || !runtime.allocator) return unsigned(Status::invalid);
  OriginalMutablePairLayout value;
  if (!original_mutable_pair_layout(facts(runtime), value)) return unsigned(Status::unsupported);
  size_t controls = value.controls;
  const size_t extra = sizeof(OriginalMutablePairLayout) + sizeof(mlx_original_mutable_pair_layout) +
      sizeof(mlx_prepared_input_runtime) + sizeof(mlx_submission_graph_quota) +
      sizeof(mlx_submission_scope) + sizeof(mlx_original_buffer_budget) +
      sizeof(mlx_array) + sizeof(PreparedInputArray*) + sizeof(Status) + sizeof(const uint32_t*);
  if (controls > SIZE_MAX - extra) return unsigned(Status::invalid);
  controls += extra;
  *out = {value.metadata_bytes, value.backing_bytes, value.copy_bytes,
      value.record_minimum_capacity, controls, value.module_bytes, value.thread_bytes,
      value.graph_requests, unsigned(value.constant_registry)};
  return unsigned(Status::success);
}
extern "C" unsigned mlx_original_mutable_pair_new(mlx_array* out,
    mlx_prepared_input_runtime runtime, mlx_submission_graph_quota graph,
    mlx_submission_scope scope, mlx_original_buffer_budget budget, const uint32_t* input) {
  using Status = PreparedHostCopyCause;
  if (!out || out->ctx || out->prepared_owner || !runtime.allocator ||
      !graph.ctx || !scope.ctx || !budget.ctx || !input) return unsigned(Status::invalid);
  PreparedInputArray* value = nullptr;
  const auto status = construct_original_mutable_pair(
      *static_cast<allocator::Allocator*>(runtime.allocator), facts(runtime),
      static_cast<submission::GraphQuota*>(graph.ctx),
      *static_cast<submission::Scope*>(scope.ctx),
      *static_cast<allocator::OriginalBufferBudget*>(budget.ctx), input, value);
  if (status == Status::success) *out = mlx_array{&value->value(), value};
  return unsigned(status);
}

extern "C" void mlx_input_allocator_layout_for(mlx_input_allocator_layout* out) {
  if (!out) return;
  const auto layout = allocator::input_allocator_layout();
  const size_t transport = sizeof(allocator::InputAllocatorLayout) +
      sizeof(allocator::InitializedInputAllocator) + sizeof(allocator::InputAllocatorOwner) +
      sizeof(allocator::InputAllocatorCause) + sizeof(mlx_input_allocator_layout) +
      sizeof(mlx_prepared_input_runtime) + sizeof(uint64_t) + sizeof(void*) * 4 +
      sizeof(size_t) * 2; // query's transport and checked-controls locals
  if (layout.controls > SIZE_MAX - transport) { *out = {}; return; }
  const size_t controls = layout.controls + transport;
  *out = {layout.static_bytes, layout.object_bytes, controls,
      unsigned(layout.qualified), unsigned(layout.requires_device)};
}
extern "C" unsigned mlx_input_allocator_initialize(mlx_prepared_input_runtime* out,
    uint64_t* identity, void* owner, void (*retire)(void*)) {
  using Cause = allocator::InputAllocatorCause;
  if (!out || out->allocator || !identity || *identity || !owner || !retire ||
      !submission::submission_context_empty()) return unsigned(Cause::invalid);
  allocator::InitializedInputAllocator initialized{};
  const auto status = allocator::initialize_input_allocator({owner, retire}, initialized);
  if (status == Cause::success) {
    const auto& facts = initialized.facts;
    *out = {initialized.allocator, facts.page_size, facts.maximum,
        unsigned(facts.kind), facts.controls};
    *identity = initialized.identity;
  }
  return unsigned(status);
}
extern "C" size_t mlx_input_allocator_borrow_controls(void) {
  if (!allocator::input_allocator_layout_qualified) return 0;
  // C wrapper: output/identity, initialized facts, status and the facts loan.
  // Forwarding function and InputAllocatorSlot::borrow: their arguments/result,
  // the successful try-lock guard and aggregate output assignment. The existing
  // singleton/mutex/allocator storage remains in the original source account.
  return sizeof(mlx_prepared_input_runtime*) + sizeof(uint64_t) +
      sizeof(allocator::InitializedInputAllocator) + sizeof(allocator::InputAllocatorCause) +
      sizeof(const allocator::PreparedInputFacts*) + sizeof(unsigned) +
      sizeof(mlx_prepared_input_runtime) + // aggregate C output assignment
      sizeof(uint64_t) + sizeof(allocator::InitializedInputAllocator*) +
      sizeof(allocator::InputAllocatorCause) +
      sizeof(void*) + sizeof(uint64_t) + sizeof(allocator::InitializedInputAllocator*) +
      sizeof(std::lock_guard<std::mutex>) + sizeof(allocator::InitializedInputAllocator) +
      sizeof(allocator::InputAllocatorCause) + sizeof(std::mutex*) + sizeof(bool);
}
extern "C" unsigned mlx_input_allocator_borrow(mlx_prepared_input_runtime* out,
    uint64_t identity) {
  using Cause = allocator::InputAllocatorCause;
  // This only borrows an already admitted immutable singleton. The native
  // try-lock/identity check does not initialize, wait, or allocate in a role.
  if (!out || out->allocator) return unsigned(Cause::invalid);
  allocator::InitializedInputAllocator initialized{};
  const auto status = allocator::borrow_input_allocator(identity, initialized);
  if (status == Cause::success) {
    const auto& facts = initialized.facts;
    *out = {initialized.allocator, facts.page_size, facts.maximum,
        unsigned(facts.kind), facts.controls};
  }
  return unsigned(status);
}

extern "C" unsigned mlx_original_prompt_input_layout_for(
    mlx_original_prompt_input_layout* out, mlx_prepared_input_runtime runtime, size_t elements) {
  using Status = PreparedHostCopyCause;
  if (!out || !runtime.allocator) return unsigned(Status::invalid);
  OriginalPromptInputLayout value;
  if (!original_prompt_input_layout(facts(runtime), elements, value)) return unsigned(Status::unsupported);
  const size_t controls = sizeof(OriginalPromptInputLayout) + sizeof(mlx_original_prompt_input_layout) +
      sizeof(mlx_prepared_input_runtime) + sizeof(mlx_array) + sizeof(PreparedInputArray*) +
      sizeof(const uint32_t*) + sizeof(size_t) * 2 + sizeof(Status);
  if (value.controls > SIZE_MAX - controls) return unsigned(Status::invalid);
  *out = {value.metadata_bytes, value.backing_bytes, value.controls + controls, value.ordinary_handle_bytes};
  return unsigned(Status::success);
}
extern "C" unsigned mlx_original_prompt_input_new(
    mlx_array* out, const uint32_t* input, size_t elements) {
  using Status = PreparedHostCopyCause;
  if (!out || out->ctx || out->prepared_owner) return unsigned(Status::invalid);
  PreparedInputArray* value = nullptr;
  const auto status = construct_original_prompt_input(input, elements, value);
  if (status == Status::success) *out = mlx_array{&value->value(), value};
  return unsigned(status);
}

extern "C" unsigned mlx_original_prediction_input_new(
    mlx_array* out, const int32_t* input, size_t elements) {
  using Status = PreparedHostCopyCause;
  if (!out || out->ctx || out->prepared_owner) return unsigned(Status::invalid);
  PreparedInputArray* value = nullptr;
  const auto status = construct_original_prediction_input(input, elements, value);
  if (status == Status::success) *out = mlx_array{&value->value(), value};
  return unsigned(status);
}
