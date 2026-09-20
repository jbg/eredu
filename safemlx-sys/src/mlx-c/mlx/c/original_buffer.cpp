#include "mlx/c/original_buffer.h"
#include "mlx/original_buffer.h"
#include "mlx/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/host_transfer.h"
#include "mlx/host_transfer.h"
#include "mlx/submission.h"
#include <new>
#include <limits>
#include <optional>

namespace {
using namespace mlx::core;
using Budget = allocator::OriginalBufferBudget;
unsigned status(allocator::OriginalBufferCause cause) noexcept {
  using C = allocator::OriginalBufferCause;
  switch (cause) {
    case C::missing_binding: return MLX_ORIGINAL_BUFFER_MISSING;
    case C::foreign_domain: return MLX_ORIGINAL_BUFFER_FOREIGN;
    case C::invalid_scope: return MLX_ORIGINAL_BUFFER_SCOPE;
    case C::already_bound: return MLX_ORIGINAL_BUFFER_BOUND;
    case C::invalid_layout: return MLX_ORIGINAL_BUFFER_LAYOUT;
    case C::capacity: return MLX_ORIGINAL_BUFFER_CAPACITY;
    case C::busy: return MLX_ORIGINAL_BUFFER_BUSY;
    case C::allocation_failed: return MLX_ORIGINAL_BUFFER_ALLOCATION;
    case C::unsupported: return MLX_ORIGINAL_BUFFER_UNSUPPORTED;
    case C::identity_exhausted: return MLX_ORIGINAL_BUFFER_IDENTITY;
  }
  return MLX_ORIGINAL_BUFFER_UNEXPECTED;
}
Budget* budget(mlx_original_buffer_budget value) noexcept {
  return static_cast<Budget*>(value.ctx);
}
enum class StorageKind { unknown, empty, original, ordinary, immutable };
struct MutableBirth {
  array::Data* data{nullptr};
  mlx_original_buffer_info info{};
  StorageKind kind{StorageKind::unknown};
};
unsigned inspect_storage(MutableBirth& out, mlx_array source) noexcept {
  mlx_array_descriptor descriptor{};
  if (mlx_array_descriptor_read(&descriptor, source)) return MLX_ORIGINAL_BUFFER_LAYOUT;
  // The shared descriptor kernel proves readiness, deleter/source provenance and
  // real capacity. Unknown or merely logical storage never becomes ordinary.
  if (!descriptor.known || descriptor.host_transfer) return MLX_ORIGINAL_BUFFER_OK;
  if (!descriptor.data) {
    if (!descriptor.elements && !descriptor.buffer && !descriptor.identity &&
        !descriptor.allocation_bytes) out.kind = StorageKind::empty;
    return MLX_ORIGINAL_BUFFER_OK;
  }
  auto* data = static_cast<const array::Data*>(descriptor.data);
  if (data->original_input) return MLX_ORIGINAL_BUFFER_OK;
  using NativeDeleter = void (*)(allocator::Buffer);
  const auto* deleter = data->d.target<NativeDeleter>();
  if (!deleter || *deleter != &allocator::free || !data->allocation_generation)
    return MLX_ORIGINAL_BUFFER_OK;
  if (data->buffer.original_buffer_budget()) {
    if (descriptor.identity != data->buffer.original_allocation_generation() ||
        descriptor.allocation_bytes != data->buffer.original_allocation_capacity())
      return MLX_ORIGINAL_BUFFER_LAYOUT;
    out.kind = StorageKind::original;
  } else if (!descriptor.buffer) {
    if (!descriptor.elements && !descriptor.identity && !descriptor.allocation_bytes)
      out.kind = StorageKind::empty;
    return MLX_ORIGINAL_BUFFER_OK;
  } else {
    if (descriptor.identity != data->allocation_generation)
      return MLX_ORIGINAL_BUFFER_LAYOUT;
    out.kind = StorageKind::ordinary;
  }
  // No second Array lookup. Only this serialized, same Data's sidecar may change.
  out.data = const_cast<array::Data*>(data);
  out.info = {true, descriptor.identity, descriptor.allocation_bytes};
  return MLX_ORIGINAL_BUFFER_OK;
}
unsigned inspect_immutable(MutableBirth& out, mlx_array source) noexcept {
  mlx_array_descriptor descriptor{};
  if (mlx_array_descriptor_read(&descriptor, source)) return MLX_ORIGINAL_BUFFER_LAYOUT;
  // descriptor.data is supplied only for actually settled storage. A prepared
  // zero-copy slot has Data but no generation; do not infer this from bytes.
  if (!descriptor.data || descriptor.host_transfer) return MLX_ORIGINAL_BUFFER_OK;
  auto* data = static_cast<const array::Data*>(descriptor.data);
  const auto* input = data->original_input;
  if (!input || data->buffer.ptr() != input->buffer.ptr()) return MLX_ORIGINAL_BUFFER_OK;
  if (!data->allocation_generation) {
    if (!descriptor.elements && !descriptor.buffer && !input->buffer.ptr() &&
        !input->capacity && !descriptor.identity && !descriptor.allocation_bytes)
      out.kind = StorageKind::empty;
    return MLX_ORIGINAL_BUFFER_OK;
  }
  if (!descriptor.known || descriptor.identity != data->allocation_generation ||
      descriptor.allocation_bytes != input->capacity) return MLX_ORIGINAL_BUFFER_LAYOUT;
  out.data = const_cast<array::Data*>(data);
  out.info = {true, descriptor.identity, descriptor.allocation_bytes};
  out.kind = StorageKind::immutable;
  return MLX_ORIGINAL_BUFFER_OK;
}
unsigned inspect_birth(MutableBirth& out, mlx_array source, Budget* expected) noexcept {
  MutableBirth actual;
  const auto result = inspect_storage(actual, source);
  if (result) return result;
  if (actual.kind != StorageKind::original) return MLX_ORIGINAL_BUFFER_OK;
  if (expected && actual.data->buffer.original_buffer_budget() != expected)
    return MLX_ORIGINAL_BUFFER_FOREIGN;
  out = actual;
  return MLX_ORIGINAL_BUFFER_OK;
}
unsigned attach_storage(mlx_array source, Budget* budget, StorageKind kind,
    const mlx_original_buffer_info* expected, void* node, void* payload,
    void (*release)(void*)) noexcept {
  if (!expected || !expected->known || !expected->identity || !node || !payload || !release)
    return MLX_ORIGINAL_BUFFER_LAYOUT;
  MutableBirth actual;
  const auto status = kind == StorageKind::immutable
      ? inspect_immutable(actual, source) : inspect_storage(actual, source);
  if (status) return status;
  if (!actual.info.known) return MLX_ORIGINAL_BUFFER_UNCERTIFIED;
  if (actual.kind != kind) return kind == StorageKind::original
      ? MLX_ORIGINAL_BUFFER_UNCERTIFIED : MLX_ORIGINAL_BUFFER_CHANGED;
  if (budget && actual.data->buffer.original_buffer_budget() != budget)
    return MLX_ORIGINAL_BUFFER_FOREIGN;
  if (actual.info.identity != expected->identity ||
      actual.info.charged_bytes != expected->charged_bytes) return MLX_ORIGINAL_BUFFER_CHANGED;
  // All checks precede arming. This is the exact Data captured above, not a
  // re-query of an Array which might now describe different storage.
  actual.data->allocation_owners.append_prepared(
      static_cast<AllocationOwners::Node*>(node), payload, release);
  return MLX_ORIGINAL_BUFFER_OK;
}
}

extern "C" unsigned mlx_original_buffer_layout_for(mlx_original_buffer_layout* out) {
  if (!out) return MLX_ORIGINAL_BUFFER_LAYOUT;
  // Actual fixed C/native locals and transport. Allocator preparation's fixed
  // controls are returned by the separately prepared input runtime.
  *out = {Budget::control_bytes(),
      sizeof(mlx_original_buffer_layout) + sizeof(mlx_original_buffer_budget) * 2 +
      sizeof(mlx_prepared_input_runtime) + sizeof(allocator::PreparedInputFacts) +
      sizeof(Budget*) * 3 + sizeof(void*) + sizeof(void (*)(void*)) +
      sizeof(size_t) * 2 + sizeof(unsigned) +
      sizeof(mlx_submission_scope) + sizeof(submission::Scope*) +
      sizeof(mlx_original_buffer_info) + sizeof(mlx_array_descriptor) + mlx_array_descriptor_control_bytes() +
      sizeof(mlx_array) + sizeof(const array::Data*) + sizeof(MutableBirth) * 2 + sizeof(StorageKind)};
  return MLX_ORIGINAL_BUFFER_OK;
}

extern "C" unsigned mlx_original_buffer_population_layout_for(
    mlx_original_buffer_population_layout* out, mlx_prepared_input_runtime runtime,
    size_t requested_bytes, size_t maximum_births) {
  using Kind = allocator::HostTransferStorageKind;
  if (!out || !runtime.allocator) return MLX_ORIGINAL_BUFFER_LAYOUT;
  if (runtime.storage_kind != static_cast<unsigned>(Kind::metal_shared) &&
      runtime.storage_kind != static_cast<unsigned>(Kind::cpu))
    return MLX_ORIGINAL_BUFFER_UNSUPPORTED;
  const auto kind = static_cast<Kind>(runtime.storage_kind);
  const allocator::PreparedInputFacts facts{runtime.page_size, runtime.maximum,
      kind, runtime.controls};
  size_t quantum = 0;
  // Use the exact physical rounding worker used by malloc_original; this does
  // not initialize/query the allocator. A one-byte request occupies one page.
  if (!allocator::original_buffer_physical_capacity(facts, 1, quantum) ||
      !quantum || quantum != runtime.page_size)
    return MLX_ORIGINAL_BUFFER_UNSUPPORTED;
  if (requested_bytes && !maximum_births) return MLX_ORIGINAL_BUFFER_LAYOUT;
  const size_t header = kind == Kind::cpu ? sizeof(size_t) : 0;
  if (quantum - 1 > std::numeric_limits<size_t>::max() - header)
    return MLX_ORIGINAL_BUFFER_LAYOUT;
  const size_t padding = quantum - 1 + header;
  size_t capacity = 0;
  if (requested_bytes || header) {
    if (maximum_births && padding >
        (std::numeric_limits<size_t>::max() - requested_bytes) / maximum_births)
      return MLX_ORIGINAL_BUFFER_LAYOUT;
    // Each physical birth costs at most r + header + quantum - 1. CPU includes
    // zero-payload births because their headers still occupy a page; Metal
    // empty requests have no backing. Retain every generation without credit
    // for donation, eager retirement or cache reuse.
    capacity = requested_bytes + maximum_births * padding;
  }
  *out = {capacity, sizeof(mlx_original_buffer_population_layout) +
      sizeof(mlx_prepared_input_runtime) + sizeof(allocator::PreparedInputFacts) +
      sizeof(size_t) * 7 + sizeof(unsigned) + sizeof(Kind)};
  return MLX_ORIGINAL_BUFFER_OK;
}


extern "C" size_t mlx_original_buffer_request_control_bytes(void) {
  return sizeof(mlx_original_buffer_population_layout)+sizeof(mlx_prepared_input_runtime)+
      sizeof(allocator::PreparedInputFacts)+sizeof(allocator::HostTransferStorageKind)+
      sizeof(void*)+sizeof(size_t)*5+sizeof(unsigned)+sizeof(bool)*3;
}
extern "C" unsigned mlx_original_buffer_request_layout_for(
    mlx_original_buffer_population_layout* out,mlx_prepared_input_runtime runtime,size_t requested_bytes) {
  using Kind=allocator::HostTransferStorageKind;
  if(!out || !runtime.allocator)return MLX_ORIGINAL_BUFFER_LAYOUT;
  if(runtime.storage_kind!=static_cast<unsigned>(Kind::cpu) &&
      runtime.storage_kind!=static_cast<unsigned>(Kind::metal_shared))return MLX_ORIGINAL_BUFFER_UNSUPPORTED;
  const allocator::PreparedInputFacts facts{runtime.page_size,runtime.maximum,
      static_cast<Kind>(runtime.storage_kind),runtime.controls};
  size_t capacity=0;
  if(!allocator::original_buffer_physical_capacity(facts,requested_bytes,capacity))return MLX_ORIGINAL_BUFFER_LAYOUT;
  *out={capacity,mlx_original_buffer_request_control_bytes()};
  return MLX_ORIGINAL_BUFFER_OK;
}

extern "C" unsigned mlx_original_buffer_budget_new_retaining(
    mlx_original_buffer_budget* out, mlx_prepared_input_runtime runtime,
    size_t capacity, void* owner, void (*retire)(void*)) {
  if (!out || out->ctx || !runtime.allocator || !owner || !retire)
    return MLX_ORIGINAL_BUFFER_LAYOUT;
  if (!mlx_submission_runtime_preparation_allowed()) return MLX_ORIGINAL_BUFFER_SCOPE;
  try {
    out->ctx = Budget::create(*static_cast<allocator::Allocator*>(runtime.allocator),
        capacity, owner, retire);
    return MLX_ORIGINAL_BUFFER_OK;
  } catch (const allocator::OriginalBufferError& error) { return status(error.cause()); }
  catch (const std::bad_alloc&) { return MLX_ORIGINAL_BUFFER_ALLOCATION; }
  catch (...) { return MLX_ORIGINAL_BUFFER_UNEXPECTED; }
}
extern "C" void mlx_original_buffer_budget_retain(mlx_original_buffer_budget value) {
  if (auto* p = budget(value)) p->retain();
}
extern "C" void mlx_original_buffer_budget_release(mlx_original_buffer_budget value) {
  if (auto* p = budget(value)) p->release();
}
extern "C" size_t mlx_original_buffer_budget_capacity(mlx_original_buffer_budget value) {
  return budget(value) ? budget(value)->capacity() : 0;
}
extern "C" size_t mlx_original_buffer_budget_occupied(mlx_original_buffer_budget value) {
  return budget(value) ? budget(value)->occupied_bytes() : 0;
}
extern "C" unsigned mlx_original_buffer_budget_bind(
    mlx_submission_scope scope, mlx_original_buffer_budget value) {
  auto* target = static_cast<submission::Scope*>(scope.ctx);
  if (!target || !budget(value)) return MLX_ORIGINAL_BUFFER_MISSING;
  try {
    target->bind_original_buffer_budget(*budget(value));
    return MLX_ORIGINAL_BUFFER_OK;
  } catch (const allocator::OriginalBufferError& error) { return status(error.cause()); }
  catch (const submission::NativeControlError& error) {
    return MLX_ORIGINAL_BUFFER_NATIVE_CONTROL_BASE + static_cast<unsigned>(error.failure());
  } catch (...) { return MLX_ORIGINAL_BUFFER_UNEXPECTED; }
}
extern "C" unsigned mlx_original_buffer_array_info(mlx_original_buffer_info* out,
    mlx_array source, mlx_original_buffer_budget expected) {
  if (!out || !budget(expected)) return MLX_ORIGINAL_BUFFER_LAYOUT;
  *out = {};
  MutableBirth actual;
  const auto status = inspect_birth(actual, source, budget(expected));
  if (!status) *out = actual.info;
  return status;
}
extern "C" unsigned mlx_original_buffer_array_alias_info(
    mlx_original_buffer_info* out, mlx_array source) {
  if (!out) return MLX_ORIGINAL_BUFFER_LAYOUT;
  *out = {};
  MutableBirth actual;
  const auto status = inspect_birth(actual, source, nullptr);
  if (!status) *out = actual.info;
  return status;
}
extern "C" size_t mlx_original_buffer_inspection_control_bytes(void) {
  return sizeof(MutableBirth) * 2 + sizeof(StorageKind) + sizeof(mlx_array_descriptor) +
      mlx_array_descriptor_control_bytes() + sizeof(mlx_original_buffer_info) +
      sizeof(mlx_array) + sizeof(Budget*) * 2 + sizeof(const array::Data*) +
      sizeof(const allocator::PreparedInputAllocation*) + sizeof(unsigned);
}
extern "C" size_t mlx_original_buffer_attachment_control_bytes(void) {
  return sizeof(MutableBirth) * 2 + sizeof(StorageKind) + sizeof(mlx_array_descriptor) +
      mlx_array_descriptor_control_bytes() + sizeof(mlx_original_buffer_info) +
      sizeof(mlx_array) + sizeof(mlx_original_buffer_budget) +
      sizeof(Budget*) * 2 + sizeof(const array::Data*) +
      sizeof(const allocator::PreparedInputAllocation*) + sizeof(const mlx_original_buffer_info*) + sizeof(void*) * 2 +
      sizeof(AllocationOwners::Node*) + sizeof(void (*)(void*)) + sizeof(unsigned) * 2 + sizeof(StorageKind);
}
extern "C" unsigned mlx_original_buffer_array_attach(mlx_array source,
    mlx_original_buffer_budget expected_budget, const mlx_original_buffer_info* expected,
    void* node, void* payload, void (*release)(void*)) {
  if (!budget(expected_budget)) return MLX_ORIGINAL_BUFFER_LAYOUT;
  return attach_storage(source, budget(expected_budget), StorageKind::original, expected, node, payload, release);
}
extern "C" unsigned mlx_original_buffer_array_alias_attach(mlx_array source,
    const mlx_original_buffer_info* expected, void* node, void* payload, void (*release)(void*)) {
  return attach_storage(source, nullptr, StorageKind::original, expected, node, payload, release);
}

extern "C" unsigned mlx_ordinary_buffer_array_info(uint32_t* kind,
    mlx_original_buffer_info* out, mlx_array source) {
  if (!kind || !out) return MLX_ORIGINAL_BUFFER_LAYOUT;
  *kind = MLX_ORDINARY_BUFFER_UNKNOWN;
  *out = {};
  MutableBirth actual;
  const auto status = inspect_storage(actual, source);
  if (status) return status;
  if (actual.kind == StorageKind::empty) {
    *kind = MLX_ORDINARY_BUFFER_EMPTY;
  } else if (actual.kind == StorageKind::ordinary) {
    *kind = MLX_ORDINARY_BUFFER_ALLOCATION;
    *out = actual.info;
  }
  return MLX_ORIGINAL_BUFFER_OK;
}
extern "C" unsigned mlx_ordinary_buffer_array_attach(mlx_array source,
    const mlx_original_buffer_info* expected, void* node, void* payload, void (*release)(void*)) {
  return attach_storage(source, nullptr, StorageKind::ordinary, expected, node, payload, release);
}

extern "C" unsigned mlx_immutable_source_array_info(uint32_t* kind,
    mlx_original_buffer_info* out, mlx_array source) {
  if (!kind || !out) return MLX_ORIGINAL_BUFFER_LAYOUT;
  MutableBirth actual;
  const auto status = inspect_immutable(actual, source);
  if (status) return status;
  *kind = actual.kind == StorageKind::immutable ? MLX_ORDINARY_BUFFER_ALLOCATION :
      actual.kind == StorageKind::empty ? MLX_ORDINARY_BUFFER_EMPTY : MLX_ORDINARY_BUFFER_UNKNOWN;
  *out = actual.info;
  return MLX_ORIGINAL_BUFFER_OK;
}
extern "C" unsigned mlx_immutable_source_array_attach(mlx_array source,
    const mlx_original_buffer_info* expected, void* node, void* payload, void (*release)(void*)) {
  return attach_storage(source, nullptr, StorageKind::immutable, expected, node, payload, release);
}

namespace {
void host_source_facts(mlx_immutable_host_transfer_info& out,
    const std::optional<HostTransferAllocationInfo>& actual) noexcept {
  if (actual && actual->identity)
    out = {{true, actual->identity, actual->capacity}, actual->prepared_source};
}
}
extern "C" unsigned mlx_immutable_host_transfer_inspect(
    mlx_immutable_host_transfer_info* out, mlx_host_transfer_buffer source) {
  if (!out || !source.ctx) return MLX_ORIGINAL_BUFFER_LAYOUT;
  *out = {};
  // This real immutable Host owner needs no Array, completion or source slot.
  host_source_facts(*out, mlx_host_transfer_buffer_get_(source).allocation_info());
  return out->backing.known ? MLX_ORIGINAL_BUFFER_OK : MLX_ORIGINAL_BUFFER_UNCERTIFIED;
}
extern "C" unsigned mlx_host_transfer_array_alias_info(
    mlx_immutable_host_transfer_info* out, mlx_array source) {
  if (!out || !source.ctx) return MLX_ORIGINAL_BUFFER_LAYOUT;
  *out = {};
  host_source_facts(*out, host_transfer_allocation_info(mlx_array_get_(source)));
  return MLX_ORIGINAL_BUFFER_OK;
}
extern "C" unsigned mlx_immutable_host_transfer_attach(mlx_host_transfer_buffer source,
    const mlx_immutable_host_transfer_info* expected, void* node, void* payload,
    void (*release)(void*)) {
  if (!expected || !expected->backing.known || !expected->backing.identity ||
      !node || !payload || !release) return MLX_ORIGINAL_BUFFER_LAYOUT;
  mlx_immutable_host_transfer_info actual{};
  const auto result = mlx_immutable_host_transfer_inspect(&actual, source);
  if (result) return result;
  if (actual.backing.identity != expected->backing.identity ||
      actual.backing.charged_bytes != expected->backing.charged_bytes ||
      actual.prepared_source != expected->prepared_source) return MLX_ORIGINAL_BUFFER_CHANGED;
  return mlx_host_transfer_buffer_get_(source).retain_prepared_allocation_owner(
      static_cast<AllocationOwners::Node*>(node), payload, release)
      ? MLX_ORIGINAL_BUFFER_OK : MLX_ORIGINAL_BUFFER_CHANGED;
}
extern "C" size_t mlx_immutable_host_transfer_control_bytes(void) {
  return sizeof(mlx_immutable_host_transfer_info) * 2 +
      sizeof(mlx_immutable_host_transfer_info*) + sizeof(const mlx_immutable_host_transfer_info*) +
      sizeof(std::optional<HostTransferAllocationInfo>) + sizeof(HostTransferAllocationInfo) +
      sizeof(const std::optional<HostTransferAllocationInfo>*) +
      sizeof(mlx_host_transfer_buffer) * 2 + sizeof(const HostTransferBuffer*) * 2 +
      sizeof(AllocationOwners::Node*) + sizeof(void*) * 2 + sizeof(void (*)(void*)) +
      sizeof(unsigned) + sizeof(bool);
}

// A settled Host-transfer Array is an alias of its immutable Host allocation.
// The neutral registry must authenticate its existing prepaid source; this ABI
// creates no source, budget, allocation or completion authority.
extern "C" unsigned mlx_host_transfer_array_alias_attach(mlx_array source,
    const mlx_immutable_host_transfer_info* expected, void* node, void* payload,
    void (*release)(void*)) {
  if (!expected || !expected->backing.known || !expected->backing.identity || !node || !payload || !release)
    return MLX_ORIGINAL_BUFFER_LAYOUT;
  mlx_immutable_host_transfer_info actual{};
  const auto status = mlx_host_transfer_array_alias_info(&actual, source);
  if (status) return status;
  if (!actual.backing.known) return MLX_ORIGINAL_BUFFER_UNCERTIFIED;
  if (actual.backing.identity != expected->backing.identity || actual.backing.charged_bytes != expected->backing.charged_bytes ||
      actual.prepared_source != expected->prepared_source)
    return MLX_ORIGINAL_BUFFER_CHANGED;
  // The caller holds the same exclusive runtime loan across inspection and this
  // no-allocation sidecar append. No callback or descriptor mutation intervenes.
  if (!retain_prepared_host_transfer_allocation_owner(mlx_array_get_(source),
      static_cast<AllocationOwners::Node*>(node), payload, release))
    return MLX_ORIGINAL_BUFFER_CHANGED;
  return MLX_ORIGINAL_BUFFER_OK;
}
extern "C" size_t mlx_host_transfer_array_alias_control_bytes(void) {
  return mlx_immutable_host_transfer_control_bytes() + sizeof(mlx_original_buffer_info) * 2 + sizeof(mlx_array) +
      sizeof(const mlx_original_buffer_info*) + sizeof(AllocationOwners::Node*) +
      sizeof(std::optional<HostTransferAllocationInfo>) + sizeof(void*) * 2 +
      sizeof(void (*)(void*)) + sizeof(unsigned) * 2 + sizeof(const array*) +
      sizeof(array::Status) + sizeof(const std::shared_ptr<array::Data>*) +
      sizeof(const void*); // actual settled-Data and named Host deleter borrows
}

namespace {
unsigned inspect_host_view(mlx_host_transfer_view_info& out,
    array::Data*& data, mlx_array source) noexcept {
  mlx_array_descriptor descriptor{};
  if (mlx_array_descriptor_read(&descriptor, source)) return MLX_ORIGINAL_BUFFER_LAYOUT;
  if (!descriptor.known || !descriptor.host_transfer || !descriptor.identity ||
      !descriptor.data) return MLX_ORIGINAL_BUFFER_OK;
  const auto* actual = static_cast<const array::Data*>(descriptor.data);
  if (!actual->allocation_generation) return MLX_ORIGINAL_BUFFER_LAYOUT;
  out = {{true, descriptor.identity, descriptor.allocation_bytes},
      actual->allocation_generation};
  data = const_cast<array::Data*>(actual);
  return MLX_ORIGINAL_BUFFER_OK;
}
}
extern "C" unsigned mlx_host_transfer_array_view_info(
    mlx_host_transfer_view_info* out, mlx_array source) {
  if (!out) return MLX_ORIGINAL_BUFFER_LAYOUT;
  *out = {};
  array::Data* data = nullptr;
  return inspect_host_view(*out, data, source);
}
extern "C" unsigned mlx_host_transfer_array_view_attach(mlx_array source,
    const mlx_host_transfer_view_info* expected, void* node, void* payload,
    void (*release)(void*)) {
  if (!expected || !expected->backing.known || !expected->backing.identity ||
      !expected->view_identity || !node || !payload || !release)
    return MLX_ORIGINAL_BUFFER_LAYOUT;
  mlx_host_transfer_view_info actual{};
  array::Data* data = nullptr;
  const auto status = inspect_host_view(actual, data, source);
  if (status) return status;
  if (!actual.backing.known) return MLX_ORIGINAL_BUFFER_UNCERTIFIED;
  if (actual.backing.identity != expected->backing.identity ||
      actual.backing.charged_bytes != expected->backing.charged_bytes ||
      actual.view_identity != expected->view_identity)
    return MLX_ORIGINAL_BUFFER_CHANGED;
  // Append to the exact positively inspected Data, whose ordinary destructor
  // retires shared Device views before this payload. The immutable Host source
  // owns a separate storage object and cannot pin this view's residency charge.
  data->allocation_owners.append_prepared(
      static_cast<AllocationOwners::Node*>(node), payload, release);
  return MLX_ORIGINAL_BUFFER_OK;
}
extern "C" size_t mlx_host_transfer_array_view_control_bytes(void) {
  return mlx_array_descriptor_control_bytes() + sizeof(mlx_array_descriptor) +
      sizeof(mlx_host_transfer_view_info) * 2 + sizeof(mlx_host_transfer_view_info*) +
      sizeof(const mlx_host_transfer_view_info*) + sizeof(array::Data*) * 2 +
      sizeof(const array::Data*) + sizeof(mlx_array) * 2 +
      sizeof(AllocationOwners::Node*) + sizeof(void*) * 2 +
      sizeof(void (*)(void*)) + sizeof(unsigned);
}
