/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>
#include <limits>
#include <iterator>
#include <new>
#include <type_traits>

#include "mlx/c/array.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/private/allocation_owner.h"
#include "mlx/c/private/prepared_array_output.h"
#include "mlx/c/string.h"
#include "mlx/host_transfer.h"
#include "mlx/c/prefill_roots.h"
#include "mlx/scoped_evaluation.h"

namespace {
// Array::init copies exactly size() elements from [begin, begin + size()).
// The factory validates that range before constructing either the descriptor
// or buffer. An empty range never dereferences its possibly zero divisor.
struct RepeatedI32Iterator {
  using value_type = int32_t;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::input_iterator_tag;
  using pointer = const int32_t*;
  using reference = int32_t;

  size_t position;
  size_t repeats;
  int32_t operator*() const noexcept {
    return static_cast<int32_t>(position / repeats);
  }
  RepeatedI32Iterator& operator++() noexcept { ++position; return *this; }
  RepeatedI32Iterator operator++(int) noexcept {
    auto old = *this;
    ++*this;
    return old;
  }
  RepeatedI32Iterator operator+(size_t count) const noexcept {
    return {position + count, repeats};
  }
  bool operator==(const RepeatedI32Iterator& other) const noexcept {
    return position == other.position && repeats == other.repeats;
  }
  bool operator!=(const RepeatedI32Iterator& other) const noexcept {
    return !(*this == other);
  }
};
} // namespace

extern "C" size_t mlx_array_repeated_i32_control_bytes(void) {
  // Named factory/array-init/copy iterator controls only. Descriptor, Data,
  // wrapper and the one final buffer retain their existing Graph/P owners.
  return sizeof(mlx_array*) + sizeof(size_t) * 4 + sizeof(int32_t) +
      sizeof(RepeatedI32Iterator) * 3 + sizeof(mlx::core::Shape) +
      sizeof(mlx::core::array) + sizeof(const std::exception*) +
      sizeof(const char*) + sizeof(int);
}

extern "C" int mlx_array_set_repeated_i32(
    mlx_array* result, size_t groups, size_t repeats) {
  try {
    const auto maximum = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    if (!result || groups > maximum || repeats > maximum ||
        (repeats && groups > maximum / repeats)) {
      throw std::invalid_argument("Invalid repeated I32 input extent.");
    }
    const size_t count = groups * repeats;
    if (count > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
      throw std::invalid_argument("Repeated I32 input extent overflow.");
    }
    // Same eager descriptor/Data/buffer producer as mlx_array_set_data, with
    // no numerical staging allocation. Publication adopts its actual owners;
    // the active original Scope retains a failed constructor's native prefix.
    mlx_array_set_(*result, mlx::core::array(
        RepeatedI32Iterator{0, repeats},
        mlx::core::Shape{static_cast<int32_t>(count)}, mlx::core::int32));
    return 0;
  } catch (const std::exception& error) {
    mlx_error(error.what());
    return 1;
  }
}

extern "C" int mlx_array_new_host(
    mlx_array* result,
    void** data,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
  try {
    if (!result || !data || dim < 0 || (dim > 0 && !shape)) {
      throw std::invalid_argument("Invalid host array arguments.");
    }
    using namespace mlx::core;
    auto kind = allocator::allocator().host_transfer_storage_kind(
        allocator::HostTransferPolicy::transfer);
    if (kind != allocator::HostTransferStorageKind::cpu &&
        kind != allocator::HostTransferStorageKind::metal_shared) {
      throw std::invalid_argument("Direct host arrays require CPU or Metal storage.");
    }
    auto cpp_dtype = mlx_dtype_to_cpp(dtype);
    size_t bytes = size_of(cpp_dtype);
    // Check suffix extents as well as the total: a leading zero must not
    // conceal overflowing strides in the array descriptor.
    for (int i = dim; i-- > 0;) {
      if (shape[i] < 0 ||
          (shape[i] && bytes > std::numeric_limits<size_t>::max() / shape[i])) {
        throw std::invalid_argument("Invalid or overflowing host array shape.");
      }
      bytes *= shape[i];
    }
    mlx_array_output_preparation_ publication(*result);
    auto cpp_shape = dim ? Shape(shape, shape + dim) : Shape{};
    array value(allocator::Buffer{nullptr}, std::move(cpp_shape), cpp_dtype);
    value.set_data(allocator::malloc(bytes));
    auto pointer = value.data<void>();
    if (bytes) {
      if (!pointer) {
        throw std::runtime_error("Unable to allocate host array storage.");
      }
      std::memset(pointer, 0, bytes);
    }
    publication.publish(std::move(value));
    *data = pointer;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" size_t mlx_dtype_size(mlx_dtype dtype) {
  return mlx_dtype_to_cpp(dtype).size();
}

extern "C" int mlx_array_tostring(mlx_string* str_, const mlx_array arr) {
  try {
    std::ostringstream os;
    os << mlx_array_get_(arr);
    std::string str = os.str();
    mlx_string_set_(*str_, str);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_array_free(mlx_array arr) {
  try {
    mlx_array_free_(arr);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_array mlx_array_new(void) {
  try {
    return mlx_array_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}

extern "C" size_t mlx_array_clone_handle_bytes(void) {
  return sizeof(mlx::core::array);
}

extern "C" int mlx_array_clone_storage_new(void** storage) {
  try {
    if (!storage || *storage) throw std::invalid_argument("Invalid clone storage destination.");
    static_assert(alignof(mlx::core::array) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
    // No array object exists until fill. Keep the ordinary operator-new/delete
    // pairing used by mlx_array_output_preparation_ and mlx_array_free_.
    *storage = ::operator new(sizeof(mlx::core::array));
    return 0;
  } catch (const std::exception& error) {
    mlx_error(error.what());
    return 1;
  }
}

extern "C" uint32_t mlx_array_clone_storage_new_fixed(void** storage) {
  if (!storage || *storage) return 4;
  static_assert(alignof(mlx::core::array) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
  *storage = ::operator new(sizeof(mlx::core::array), std::nothrow);
  return *storage ? 0 : 6;
}

extern "C" void mlx_array_clone_storage_free(void* storage) {
  ::operator delete(storage);
}

extern "C" uint32_t mlx_array_clone_storage_fill(
    mlx_array* output, void** storage, const mlx_array source) {
  static_assert(std::is_nothrow_copy_constructible_v<mlx::core::array>);
  if (!output || output->ctx || output->prepared_owner || !storage || !*storage ||
      !source.ctx || *storage == source.ctx) return 4;
  // Copy construction only increments the existing shared descriptor owner.
  // No destructor or user callback runs between construction and publication.
  auto* value = ::new (*storage) mlx::core::array(
      *static_cast<const mlx::core::array*>(source.ctx));
  *output = mlx_array{value, nullptr};
  *storage = nullptr;
  return 0;
}

extern "C" size_t mlx_array_clone_storage_control_bytes(void) {
  return sizeof(void**) * 2 + sizeof(void*) + sizeof(mlx_array*) +
      sizeof(mlx_array) * 2 + sizeof(mlx::core::array*) + sizeof(uint32_t) +
      sizeof(const std::exception*) + sizeof(const char*);
}

extern "C" int mlx_array_set(mlx_array* arr, const mlx_array src) {
  try {
    mlx_array_set_(*arr, mlx_array_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_set_bool(mlx_array* arr, bool val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_array mlx_array_new_bool(bool val) {
  try {
    return mlx_array_new_(mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" int mlx_array_set_int(mlx_array* arr, int val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_array mlx_array_new_int(int val) {
  try {
    return mlx_array_new_(mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" int mlx_array_set_float32(mlx_array* arr, float val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_set_float(mlx_array* arr, float val) {
  return mlx_array_set_float32(arr, val);
}
extern "C" int mlx_array_set_float64(mlx_array* arr, double val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val, mlx::core::float64));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_set_double(mlx_array* arr, double val) {
  return mlx_array_set_float64(arr, val);
}
extern "C" mlx_array mlx_array_new_float32(float val) {
  try {
    return mlx_array_new_(mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_float(float val) {
  return mlx_array_new_float32(val);
}
extern "C" mlx_array mlx_array_new_float64(double val) {
  try {
    return mlx_array_new_(mlx::core::array(val, mlx::core::float64));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_double(double val) {
  return mlx_array_new_float64(val);
}
extern "C" int
mlx_array_set_complex(mlx_array* arr, float real_val, float imag_val) {
  try {
    std::complex<float> cpp_val(real_val, imag_val);
    mlx_array_set_(*arr, mlx::core::array(cpp_val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_array mlx_array_new_complex(float real_val, float imag_val) {
  try {
    std::complex<float> cpp_val(real_val, imag_val);
    return mlx_array_new_(mlx::core::array(cpp_val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
namespace {
// Both public host-data construction and the original isolated-copy path use
// this exact typed initializer. No byte conversion or alternative copy engine.
bool array_set_data(mlx_array& arr, const void* data,
    const mlx::core::Shape& cpp_shape, mlx::core::Dtype cpp_dtype) {
  switch (cpp_dtype) {
      case mlx::core::bool_:
        mlx_array_set_(
            arr, mlx::core::array((bool*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint8:
        mlx_array_set_(
            arr, mlx::core::array((uint8_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint16:
        mlx_array_set_(
            arr, mlx::core::array((uint16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint32:
        mlx_array_set_(
            arr, mlx::core::array((uint32_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint64:
        mlx_array_set_(
            arr, mlx::core::array((uint64_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int8:
        mlx_array_set_(
            arr, mlx::core::array((int8_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int16:
        mlx_array_set_(
            arr, mlx::core::array((int16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int32:
        mlx_array_set_(
            arr, mlx::core::array((int32_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int64:
        mlx_array_set_(
            arr, mlx::core::array((int64_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::float16:
        mlx_array_set_(
            arr,
            mlx::core::array(
                (mlx::core::float16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::float32:
        mlx_array_set_(
            arr, mlx::core::array((float*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::float64:
        mlx_array_set_(
            arr, mlx::core::array((double*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::bfloat16:
        mlx_array_set_(
            arr,
            mlx::core::array(
                (mlx::core::bfloat16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::complex64:
        mlx_array_set_(
            arr,
            mlx::core::array(
                (mlx::core::complex64_t*)data, cpp_shape, cpp_dtype));
        break;
      default:
        return false;
    }
  return true;
}
}
extern "C" int mlx_array_set_data(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
  try {
    mlx::core::Shape cpp_shape(shape, shape + dim);
    mlx::core::Dtype cpp_dtype = mlx_dtype_to_cpp(dtype);
    if (!array_set_data(*arr, data, cpp_shape, cpp_dtype)) {
      mlx_error("unknown data type");
      return 1;
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" size_t mlx_array_deep_copy_scoped_control_bytes(size_t rank) {
  using namespace mlx::core;
  // The actual source and by-value native constructor shapes/strides stay
  // inline. Larger ranks need their own counted metadata producer.
  if (rank > Shape::inline_capacity || rank > Strides::inline_capacity) return 0;
  return sizeof(Shape) * 3 + sizeof(Strides) + sizeof(array) * 2 +
      sizeof(array::Data) + sizeof(std::shared_ptr<array::Data>) +
      sizeof(allocator::Buffer) + sizeof(Dtype) + sizeof(const void*) +
      sizeof(mlx_array) + sizeof(mlx_array*) + sizeof(mlx_prefill_failure*) +
      sizeof(submission::NativeControlBinding) + sizeof(submission::Scope*) +
      sizeof(ScopedEvaluation) + sizeof(unsigned) + sizeof(bool) + sizeof(size_t);
}
extern "C" unsigned mlx_array_deep_copy_scoped(
    mlx_array* out, mlx_array source, mlx_prefill_failure* error) {
  using namespace mlx::core;
  const auto status = mlx_array_eval_scoped_failure(error);
  if (status) return status;
  if (!out || out->ctx || out->prepared_owner || !source.ctx) return 1;
  auto* scope = original_evaluation_scope();
  const auto controls = submission::current_native_controls();
  if (!scope || submission::current_scope() != scope || !controls.original ||
      controls.failure != submission::NativeControlFailure::none ||
      controls.scope_identity != scope->identity() || !scope->on_owner_thread()) return 4;
  if (scope->failure_owner().get()->borrow()) return 7;
  if (scope->failure_owner().get()->pending_publication()) return 10;
  if (scope->query().failed) return 9;
  try {
    auto& value = mlx_array_get_(source);
    const auto ready = validate_scoped_array(value, *scope);
    if (ready != ScopedEvaluation::complete) return static_cast<unsigned>(ready);
    if (!value.flags().row_contiguous ||
        !mlx_array_deep_copy_scoped_control_bytes(value.ndim())) return 1;
    // Ordinary deep_clone uses this same synchronous typed constructor. The
    // selected Metal allocation is CPU-addressable; no implicit compaction,
    // stream selection, error formatting, handler or housekeeping occurs here.
    if (!array_set_data(*out, value.data<void>(), value.shape(), value.dtype())) return 1;
    return 0;
  } catch (...) {
    scope->failure_owner().get()->capture_current_exception();
    return 7;
  }
}
extern "C" mlx_array mlx_array_new_data(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
  try {
    mlx_array arr = mlx_array_new_();
    if (mlx_array_set_data(&arr, data, shape, dim, dtype)) {
      return mlx_array_();
    }
    return arr;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_data_managed_payload(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void* payload,
    void (*dtor)(void*)) {
  try {
    mlx::core::Shape cpp_shape(shape, shape + dim);
    mlx::core::Dtype cpp_dtype = mlx_dtype_to_cpp(dtype);
    std::function<void(void*)> cpp_deleter = [dtor, payload](void*) {
      dtor(payload);
    };
    return mlx_array_new_(
        mlx::core::array(data, cpp_shape, cpp_dtype, cpp_deleter));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_data_managed(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void (*dtor)(void*)) {
  return mlx_array_new_data_managed_payload(
      data, shape, dim, dtype, data, dtor);
}

extern "C" size_t mlx_array_itemsize(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).itemsize();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" size_t mlx_array_size(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" size_t mlx_array_nbytes(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).nbytes();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

namespace {
struct AllocationFacts {
  bool known{false};
  bool host_transfer{false};
  uint64_t identity{0};
  size_t bytes{0};
};
using NativeDeleter = void (*)(mlx::core::allocator::Buffer);
struct AllocationInspection {
  mlx::core::array::Status status;
  const mlx::core::array::Data* data{nullptr};
  std::optional<mlx::core::HostTransferAllocationInfo> host;
  const NativeDeleter* deleter{nullptr};
};
struct DescriptorInspection {
  const mlx::core::array* value{nullptr};
  int type{0};
  mlx_dtype dtype{};
  size_t itemsize{0};
  AllocationFacts backing;
  mlx::core::array::Status status;
  bool event_present{false};
  bool complete{false};
  const mlx::core::array::Data* data{nullptr};
};
// Legal retained descriptors reach no allocation/format/callback. Recognized
// allocator-owned backing proves the selected singleton was initialized by its
// successful allocation. The concrete CPU/CUDA size workers read a header; Metal
// reads the retained buffer length. Foreign/custom storage never reaches size().
AllocationFacts allocation_facts(const mlx::core::array& value) noexcept {
  AllocationFacts result;
  AllocationInspection inspection{};
  // is_available() can poll and detach an event. Never use it in this kernel.
  inspection.status = value.status();
  if (inspection.status != mlx::core::array::Status::available &&
      !(inspection.status == mlx::core::array::Status::evaluated &&
        !value.event().valid())) return result;
  inspection.data = value.data_shared_ptr().get();
  if (!inspection.data) {
    result.known = value.size() == 0;
    return result;
  }
  inspection.host = mlx::core::host_transfer_allocation_info(value);
  if (inspection.host) {
    result.known = true;
    result.host_transfer = true;
    result.identity = inspection.host->identity;
    result.bytes = inspection.host->capacity;
    return result;
  }
  if (inspection.data->original_input &&
      inspection.data->buffer.ptr() == inspection.data->original_input->buffer.ptr() &&
      inspection.data->allocation_generation != 0) {
    result.known = true;
    result.identity = inspection.data->allocation_generation;
    result.bytes = inspection.data->original_input->capacity;
    return result;
  }
  inspection.deleter = inspection.data->d.target<NativeDeleter>();
  // Arbitrary caller-owned CPU pointers cannot be passed to allocator size().
  // Unknown custom/mapped storage stays unknown; logical bytes are not capacity.
  if (!inspection.deleter || *inspection.deleter != &mlx::core::allocator::free ||
      inspection.data->allocation_generation == 0) return result;
  if (inspection.data->buffer.original_buffer_budget()) {
    // Original mutable backing carries a complete physical charge, including
    // the CPU header. Ordinary buffer_size() describes only usable bytes.
    // All alias/descriptor/publication observations must use this same birth.
    if (inspection.data->allocation_generation !=
        inspection.data->buffer.original_allocation_generation()) return result;
    result.identity = inspection.data->allocation_generation;
    result.bytes = inspection.data->buffer.original_allocation_capacity();
    result.known = true;
    return result;
  }
  if (!inspection.data->buffer.ptr()) {
    result.known = value.size() == 0;
    return result;
  }
  result.identity = inspection.data->allocation_generation;
  result.bytes = value.buffer_size();
  result.known = true;
  return result;
}

uint32_t descriptor_read(
    mlx_array_descriptor* out, const mlx_array arr) noexcept {
  if (!out) return 1;
  if (!arr.ctx) return 2;
  DescriptorInspection inspection{};
  inspection.value = static_cast<const mlx::core::array*>(arr.ctx);
  if (!inspection.value->id()) return 2;
  inspection.type = static_cast<int>(inspection.value->dtype().val());
  if (inspection.type < static_cast<int>(mlx::core::Dtype::Val::bool_) ||
      inspection.type > static_cast<int>(mlx::core::Dtype::Val::complex64)) return 4;
  // Same constant enum maps as ordinary C queries, with a checked input index.
  inspection.dtype = mlx_dtype_to_c(inspection.value->dtype());
  inspection.itemsize = inspection.value->itemsize();
  if (inspection.itemsize != mlx_dtype_to_cpp(inspection.dtype).size()) return 4;
  if (!inspection.itemsize || inspection.value->size() >
      std::numeric_limits<size_t>::max() / inspection.itemsize) return 3;
  inspection.backing = allocation_facts(*inspection.value);
  inspection.status = inspection.value->status();
  inspection.event_present = inspection.value->event().valid();
  inspection.complete = inspection.status == mlx::core::array::Status::available ||
      (inspection.status == mlx::core::array::Status::evaluated && !inspection.event_present);
  inspection.data = inspection.complete ? inspection.value->data_shared_ptr().get() : nullptr;
  *out = mlx_array_descriptor{
      inspection.value->id(), inspection.value->shape().data(), inspection.value->ndim(),
      inspection.dtype, inspection.value->size(), inspection.value->size() * inspection.itemsize,
      static_cast<uint32_t>(inspection.status), inspection.event_present,
      inspection.data, inspection.data ? inspection.data->buffer.ptr() : nullptr,
      inspection.backing.known, inspection.backing.host_transfer,
      inspection.backing.identity, inspection.backing.bytes};
  return 0;
}

bool same_descriptor(
    const mlx_array_descriptor& a, const mlx_array_descriptor& b) noexcept {
  return a.descriptor == b.descriptor && a.shape == b.shape && a.rank == b.rank &&
      a.dtype == b.dtype && a.elements == b.elements &&
      a.logical_bytes == b.logical_bytes && a.status == b.status &&
      a.event_present == b.event_present && a.data == b.data &&
      a.buffer == b.buffer && a.known == b.known &&
      a.host_transfer == b.host_transfer && a.identity == b.identity &&
      a.allocation_bytes == b.allocation_bytes;
}
} // namespace

extern "C" int mlx_array_allocation_info(
    bool* known, bool* host_transfer, uint64_t* identity, size_t* bytes, const mlx_array arr) {
  try {
    if (!known || !host_transfer || !identity || !bytes) {
      throw std::invalid_argument("Invalid allocation-info outputs.");
    }
    *known = false;
    *host_transfer = false;
    *identity = 0;
    *bytes = 0;
    const auto result = allocation_facts(mlx_array_get_(arr));
    *known = result.known;
    *host_transfer = result.host_transfer;
    *identity = result.identity;
    *bytes = result.bytes;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" uint32_t mlx_array_descriptor_read(
    mlx_array_descriptor* out, const mlx_array arr) {
  return descriptor_read(out, arr);
}

extern "C" uint32_t mlx_array_descriptor_fill_shape(
    const mlx_array arr, const mlx_array_descriptor* expected,
    int* destination, size_t count) {
  if (!expected) return 1;
  mlx_array_descriptor actual;
  const auto status = descriptor_read(&actual, arr);
  if (status) return status;
  if (!same_descriptor(actual, *expected)) return 5;
  if (count != actual.rank) return 6;
  if (count && !destination) return 7;
  // Caller owns a disjoint exact destination and retains this descriptor loan.
  // Every refusal precedes any destination write, including zero-length fills.
  if (count) std::memcpy(destination, actual.shape, count * sizeof(int));
  return 0;
}

extern "C" size_t mlx_array_descriptor_control_bytes(void) {
  // Named query locals/results and the actual borrowed C call representations.
  // No dynamic backing is hidden here: source shape and fill destination belong
  // to their existing owners. This is not a compiler stack-usage guarantee.
  return sizeof(AllocationInspection) + sizeof(DescriptorInspection) +
      sizeof(AllocationFacts) + 2 * sizeof(mlx_array_descriptor) +
      sizeof(std::optional<mlx::core::HostTransferAllocationInfo>) +
      sizeof(mlx::core::array::Status) + 2 * sizeof(mlx::core::Dtype) +
      3 * sizeof(mlx_array) + sizeof(mlx_array_descriptor*) +
      2 * sizeof(const mlx_array_descriptor*) + sizeof(int*) +
      sizeof(size_t) + sizeof(uint32_t) + 2 * sizeof(const void*);
}

extern "C" size_t mlx_allocation_owner_node_bytes(void) {
  return sizeof(mlx::core::AllocationOwners::Node);
}
extern "C" size_t mlx_allocation_owner_list_bytes(void) {
  return sizeof(mlx::core::AllocationOwners);
}
extern "C" void* mlx_allocation_owner_node_new(void) {
  return new (std::nothrow) mlx::core::AllocationOwners::Node();
}
extern "C" void mlx_allocation_owner_node_free(void* node) {
  delete static_cast<mlx::core::AllocationOwners::Node*>(node);
}

extern "C" int mlx_array_attach_prepared_allocation_owner(
    int* outcome, const mlx_array arr, void* node,
    void* payload, void (*release)(void*)) {
  try {
    if (!outcome || !node || !payload || !release) {
      throw std::invalid_argument("Invalid prepared allocation owner.");
    }
    *outcome = 0;
    bool known = false;
    bool host = false;
    uint64_t identity = 0;
    size_t bytes = 0;
    if (mlx_array_allocation_info(&known, &host, &identity, &bytes, arr)) {
      return 1;
    }
    if (!known) {
      return 0;
    }
    if (!identity) {
      *outcome = 1;
      return 0;
    }
    auto* prepared = static_cast<mlx::core::AllocationOwners::Node*>(node);
    const auto& value = mlx_array_get_(arr);
    if (host) {
      if (!mlx::core::retain_prepared_host_transfer_allocation_owner(
              value, prepared, payload, release)) {
        return 0;
      }
    } else {
      value.data_shared_ptr()->allocation_owners.append_prepared(
          prepared, payload, release);
    }
    *outcome = 2;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_array_retain_allocation_owner(
    bool* attached, const mlx_array arr, void* payload, void (*release)(void*)) {
  try {
    if (!attached || !payload || !release) {
      throw std::invalid_argument("Invalid allocation-owner attachment.");
    }
    *attached = false;
    bool known = false;
    bool host_transfer = false;
    uint64_t identity = 0;
    size_t bytes = 0;
    if (mlx_array_allocation_info(&known, &host_transfer, &identity, &bytes, arr)) {
      return 1;
    }
    if (!known || identity == 0) {
      return 0;
    }
    // Keep the callback disarmed through every fallible allocation. In
    // particular shared_ptr construction may otherwise invoke its deleter when
    // control-block allocation fails, consuming payload on an error return.
    auto owner = std::make_shared<mlx_c_detail::AllocationOwnerPayload>();
    const auto& value = mlx_array_get_(arr);
    if (host_transfer) {
      if (!mlx::core::retain_host_transfer_allocation_owner(value, owner)) {
        return 0;
      }
    } else {
      value.data_shared_ptr()->allocation_owners.push_back(owner);
    }
    // No throwing operation follows this ownership-transfer point.
    owner->release = release;
    owner->payload = payload;
    *attached = true;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_array_retain_deferred_allocation_owner(
    bool* attached, const mlx_array arr, void* payload, void (*release)(void*)) {
  try {
    if (!attached || !payload || !release) {
      throw std::invalid_argument("Invalid deferred allocation-owner attachment.");
    }
    *attached = false;
    auto owner = std::make_shared<mlx_c_detail::AllocationOwnerPayload>();
    mlx_array_get_(arr).retain_deferred_allocation_owner(owner);
    // Descriptor/backing insertion has completed. No throwing operation follows
    // this transfer point, including for an as-yet unmaterialized descriptor.
    owner->release = release;
    owner->payload = payload;
    *attached = true;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" size_t mlx_array_ndim(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).ndim();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" const int* mlx_array_shape(const mlx_array arr) {
  try {
    return (int*)mlx_array_get_(arr).shape().data();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const size_t* mlx_array_strides(const mlx_array arr) {
  try {
    return (size_t*)mlx_array_get_(arr).strides().data();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int64_t* mlx_array_signed_strides(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).strides().data();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" int mlx_array_dim(const mlx_array arr, int dim) {
  try {
    return mlx_array_get_(arr).shape(dim);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" mlx_dtype mlx_array_dtype(const mlx_array arr) {
  try {
    return mlx_dtype_to_c(mlx_array_get_(arr).dtype());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return MLX_BOOL;
  }
}

extern "C" int mlx_array_eval(mlx_array arr) {
  try {
    mlx_array_get_(arr).eval();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_array_item_bool(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<bool>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint8(uint8_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint16(uint16_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint32(uint32_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint64(uint64_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int8(int8_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int16(int16_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int32(int32_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int64(int64_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_float32(float* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<float>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_float64(double* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<double>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_complex64(
    mlx_complex64_t* res,
    const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<mlx_complex64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

#ifdef HAS_FLOAT16
extern "C" int mlx_array_item_float16(float16_t* res, const mlx_array arr) {
  try {
    auto value = mlx_array_get_(arr).item<mlx::core::float16_t>();
    static_assert(sizeof(value) == sizeof(*res));
    std::memcpy(res, &value, sizeof(value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
#endif

#ifdef HAS_BFLOAT16
extern "C" int mlx_array_item_bfloat16(bfloat16_t* res, const mlx_array arr) {
  try {
    auto value = mlx_array_get_(arr).item<mlx::core::bfloat16_t>();
    static_assert(sizeof(value) == sizeof(*res));
    std::memcpy(res, &value, sizeof(value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
#endif

extern "C" const bool* mlx_array_data_bool(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<bool>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint8_t* mlx_array_data_uint8(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint16_t* mlx_array_data_uint16(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint32_t* mlx_array_data_uint32(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint64_t* mlx_array_data_uint64(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int8_t* mlx_array_data_int8(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int16_t* mlx_array_data_int16(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int32_t* mlx_array_data_int32(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int64_t* mlx_array_data_int64(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const float* mlx_array_data_float32(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<float>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const double* mlx_array_data_float64(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<double>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const mlx_complex64_t* mlx_array_data_complex64(
    const mlx_array arr) {
  try {
    // std::complex<float> and mlx_complex64_t have the same memory layout
    return reinterpret_cast<const mlx_complex64_t*>(
        mlx_array_get_(arr).data<std::complex<float>>());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}

#ifdef HAS_FLOAT16
extern "C" const float16_t* mlx_array_data_float16(const mlx_array arr) {
  try {
    static_assert(sizeof(float16_t) == sizeof(mlx::core::float16_t));
    return reinterpret_cast<const float16_t*>(
        mlx_array_get_(arr).data<mlx::core::float16_t>());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
#endif

#ifdef HAS_BFLOAT16
extern "C" const bfloat16_t* mlx_array_data_bfloat16(const mlx_array arr) {
  try {
    static_assert(sizeof(bfloat16_t) == sizeof(mlx::core::bfloat16_t));
    return reinterpret_cast<const bfloat16_t*>(
        mlx_array_get_(arr).data<mlx::core::bfloat16_t>());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
#endif

extern "C" int _mlx_array_is_available(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int _mlx_array_wait(const mlx_array arr) {
  try {
    mlx_array_get_(arr).wait();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int _mlx_array_is_contiguous(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).flags().contiguous;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int _mlx_array_is_row_contiguous(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).flags().row_contiguous;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int _mlx_array_is_col_contiguous(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).flags().col_contiguous;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
