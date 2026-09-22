/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_ARRAY_H
#define MLX_ARRAY_H

#include "mlx/c/string.h"
#include "mlx/c/memory_placement.h"

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// Complex number support
#ifdef _MSC_VER
#define _CRT_USE_C_COMPLEX_H
#include <complex.h>
typedef _Fcomplex mlx_complex64_t;
#else
#include <complex.h>
typedef float _Complex mlx_complex64_t;
#endif

#include "half.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_array Array
 * MLX N-dimensional array object.
 */
/**@{*/

/**
 * A N-dimensional array object.
 */
typedef struct mlx_array_ {
  void* ctx;
  // Null for ordinary new/delete. Prepared inputs and prepaid handle clones use
  // this closed arena owner. Shallow C copies still obey unique-handle custody.
  void* prepared_owner;
} mlx_array;

static mlx_array mlx_array_empty;

/**
 * Array element type.
 */
typedef enum mlx_dtype_ {
  MLX_BOOL,
  MLX_UINT8,
  MLX_UINT16,
  MLX_UINT32,
  MLX_UINT64,
  MLX_INT8,
  MLX_INT16,
  MLX_INT32,
  MLX_INT64,
  MLX_FLOAT16,
  MLX_FLOAT32,
  MLX_FLOAT64,
  MLX_BFLOAT16,
  MLX_COMPLEX64,
} mlx_dtype;

/**
 * Size of given mlx_dtype datatype in bytes.
 */
size_t mlx_dtype_size(mlx_dtype dtype);

/**
 * Get array description.
 */
int mlx_array_tostring(mlx_string* str, const mlx_array arr);

/**
 * New empty array.
 */
mlx_array mlx_array_new(void);

/**
 * Free an array.
 */
int mlx_array_free(mlx_array arr);

/**
 * Cold storage for one future ordinary array handle, without constructing an
 * ArrayDesc, Data, graph or numerical buffer. `storage` must initially be null.
 * On failure it is unchanged. The unfilled storage has unique ownership.
 */
int mlx_array_clone_storage_new(void** storage);
/** Same single handle extent; fixed status and no formatted error storage. */
uint32_t mlx_array_clone_storage_new_fixed(void** storage);
/** Free only unfilled storage returned by mlx_array_clone_storage_new. */
void mlx_array_clone_storage_free(void* storage);
/**
 * Share the actual source descriptor into that final handle storage. The source
 * and storage must be live, uniquely owned as applicable, and externally
 * serialized. Output must be empty. Success consumes storage (sets it null)
 * and publishes an ordinary mlx_array, paired with mlx_array_free. No allocation,
 * evaluation, hooks or error formatting. 0 success; 4 invalid/consumed storage
 * or arguments. Refusal changes neither output nor storage nor source.
 */
uint32_t mlx_array_clone_storage_fill(
    mlx_array* output, void** storage, const mlx_array source);
/** Named native call controls, excluding the separately reported handle bytes. */
size_t mlx_array_clone_storage_control_bytes(void);

/**
 * Eager I32[groups * repeats] input: each integer in [0, groups) appears
 * `repeats` times. Writes directly to the final owned native allocation.
 * Both arguments and their product must fit INT32_MAX; all extents are checked
 * before construction. Zero groups/repeats produces an empty rank-one array.
 * Same destination, original-scope and error conventions as set_data.
 */
int mlx_array_set_repeated_i32(mlx_array* result, size_t groups, size_t repeats);
/** Named fixed controls; excludes the ordinary Graph/P array and buffer owners. */
size_t mlx_array_repeated_i32_control_bytes(void);


/**
 * New array from a bool scalar.
 */
mlx_array mlx_array_new_bool(bool val);
/**
 * New array from a int scalar.
 */
mlx_array mlx_array_new_int(int val);
/**
 * New array from a float32 scalar.
 */
mlx_array mlx_array_new_float32(float val);
/**
 * New array from a float scalar.
 * Same as float32.
 */
mlx_array mlx_array_new_float(float val);
/**
 * New array from a float64 scalar.
 */
mlx_array mlx_array_new_float64(double val);
/**
 * New array from a double scalar.
 * Same as float64.
 */
mlx_array mlx_array_new_double(double val);
/**
 * New array from a complex scalar.
 */
mlx_array mlx_array_new_complex(float real_val, float imag_val);
/**
 * New array from existing buffer.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 */
mlx_array mlx_array_new_data(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);
/**
 * Allocate a zero-initialized, host-addressable array in native storage.
 * The returned writable pointer is exclusively owned by the new array and must
 * not be used after the array is shared or submitted. CPU and Metal only.
 */
int mlx_array_new_host(
    mlx_array* result,
    void** data,
    const int* shape,
    int dim,
    mlx_dtype dtype);
/**
 * New array from existing buffer.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 * @param dtor Callback for when the buffer is no longer needed.
 */
mlx_array mlx_array_new_data_managed(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void (*dtor)(void*));
/**
 * New array from existing buffer.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 * @param payload Payload pointer passed to the `dtor` callback instead of
 * `data`.
 * @param dtor Callback for when the buffer is no longer needed.
 */
mlx_array mlx_array_new_data_managed_payload(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void* payload,
    void (*dtor)(void*));
/**
 * Set array to provided src array.
 */
int mlx_array_set(mlx_array* arr, const mlx_array src);
/**
 * Bytes of the single C++ handle allocated when copying into an empty mlx_array.
 * The copy shares the existing ArrayDesc and does not copy its shape or backing.
 * Excludes allocator bookkeeping, source storage, and caller-owned wrappers.
 * Reads no array or runtime state and does not allocate or acquire a lock.
 */
size_t mlx_array_clone_handle_bytes(void);
/**
 * Set array to a bool scalar.
 */
int mlx_array_set_bool(mlx_array* arr, bool val);
/**
 * Set array to a int scalar.
 */
int mlx_array_set_int(mlx_array* arr, int val);
/**
 * Set array to a float32 scalar.
 */
int mlx_array_set_float32(mlx_array* arr, float val);
/**
 * Set array to a float scalar.
 */
int mlx_array_set_float(mlx_array* arr, float val);
/**
 * Set array to a float64 scalar.
 */
int mlx_array_set_float64(mlx_array* arr, double val);
/**
 * Set array to a double scalar.
 */
int mlx_array_set_double(mlx_array* arr, double val);
/**
 * Set array to a complex scalar.
 */
int mlx_array_set_complex(mlx_array* arr, float real_val, float imag_val);
/**
 * Set array to specified data and shape.
 * @param arr Destination array.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 */
int mlx_array_set_data(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);

/**
 * The size of the array's datatype in bytes.
 */
size_t mlx_array_itemsize(const mlx_array arr);
/**
 * Number of elements in the array.
 */
size_t mlx_array_size(const mlx_array arr);
/**
 * The number of bytes in the array.
 */
size_t mlx_array_nbytes(const mlx_array arr);
/**
 * Read completed allocator-owned or certified host-transfer backing without
 * evaluating, polling, or waiting. Unknown for unfinished or foreign storage.
 * Nonzero identity generations are never reused by this linked native runtime,
 * even after backing retires. They do not retain storage or persist across runs.
 */
int mlx_array_allocation_info(
    bool* known, bool* host_transfer, uint64_t* identity, size_t* bytes, const mlx_array arr);
/**
 * Borrowed descriptor witness. No field owns or pins an array, shape, backing,
 * completion or allowance. Caller must retain the actual array and serialize
 * its descriptor access for the entire read/fill loan. Completed backing is
 * certified by the same kernel as mlx_array_allocation_info; unknown stays
 * unknown. Shape is immutable descriptor storage, not a completion certificate.
 */
typedef struct mlx_array_descriptor_ {
  uintptr_t descriptor;
  const int* shape;
  size_t rank;
  mlx_dtype dtype;
  size_t elements;
  size_t logical_bytes;
  uint32_t status;
  bool event_present;
  const void* data;
  const void* buffer;
  bool known;
  bool host_transfer;
  uint64_t identity;
  size_t allocation_bytes;
  mlx_memory_placement placement;
  size_t host_control_bytes;
} mlx_array_descriptor;

/** Fixed status, no allocation/evaluation/poll/reap/error callback. Output is
 * written only on success. 0 success; 1 invalid argument; 2 empty descriptor;
 * 3 logical-byte overflow; 4 invalid datatype representation; 5 changed source;
 * 6 wrong destination length; 7 null nonempty destination. Legal borrowed
 * descriptors use audited nonthrowing native queries, never catch/erase a cause.
 */
/** Synchronous copy from completed authenticated backing into exact host storage.
 * Never evaluates, allocates, compacts or relocates the source. Returns -1 for
 * invalid source/destination; positive values preserve a native CUDA error code.
 * A native refusal may leave a copied prefix, but retains no destination access.
 */
int mlx_array_copy_completed_data(const mlx_array arr, void* destination, size_t bytes);
size_t mlx_array_completed_readback_control_bytes(void);
/** Synchronous logical scalar copy for caller-owned inline conversion. */
int mlx_array_copy_completed_element(const mlx_array arr, size_t index, void* destination, size_t bytes);
uint32_t mlx_array_descriptor_read(mlx_array_descriptor* out, const mlx_array arr);
/** Rechecks the same witness before writing an exact, disjoint caller-owned
 * shape destination. No rank cap or destination allocation. Zero rank accepts
 * a null destination and still validates source identity/current-state facts.
 */
uint32_t mlx_array_descriptor_fill_shape(
    const mlx_array arr, const mlx_array_descriptor* expected,
    int* destination, size_t count);
/** Concrete native scalar/control bytes only. No array, runtime or callback. */
size_t mlx_array_descriptor_control_bytes(void);

/* Prepared opaque owner node. new returns NULL on allocation failure; free is
 * valid only before successful handoff. Both are runtime-independent, and the
 * empty node owns no payload. Facts describe exact linked native objects. */
size_t mlx_allocation_owner_node_bytes(void);
size_t mlx_allocation_owner_list_bytes(void);
void* mlx_allocation_owner_node_new(void);
void mlx_allocation_owner_node_free(void* node);

/* outcome: 0 = unfinished/uncertified, 1 = no allocation, 2 = consumed.
 * The caller exclusively owns a fresh node and payload until outcome 2.
 * Successful handoff never allocates and cannot subsequently fail. Caller
 * serializes attachment and keeps the actual backing alive. Release may run on
 * any native thread and must not throw, block, allocate or enter the runtime.
 * Payload must not itself retain the destination backing or graph. */
int mlx_array_attach_prepared_allocation_owner(
    int* outcome, const mlx_array arr, void* node,
    void* payload, void (*release)(void*));

/**
 * Attach opaque ownership to completed certified physical backing, preserving
 * allocation identity and all native/host aliases. Caller serializes attachment.
 * Ownership transfers only when this succeeds with *attached == true. Otherwise
 * payload remains caller-owned and release is never called. Release can run on
 * any native thread and must not throw, block, allocate or reenter the runtime.
 */
int mlx_array_retain_allocation_owner(
    bool* attached, const mlx_array arr, void* payload, void (*release)(void*));
/**
 * Retain payload-free ownership on a shared lazy descriptor and publish it to
 * certified backing when materialized. Does not evaluate, poll or wait. Existing
 * descriptor clones and subsequent physical aliases share this retention;
 * independently allocated results require their own attachment. Unrecognized
 * future backing fails materialization before publication. Payload must not
 * retain this descriptor, its graph, or any covered backing. Handoff and callback
 * requirements are identical to mlx_array_retain_allocation_owner.
 */
int mlx_array_retain_deferred_allocation_owner(
    bool* attached, const mlx_array arr, void* payload, void (*release)(void*));
/**
 * The array's dimension.
 */
size_t mlx_array_ndim(const mlx_array arr);
/**
 * The shape of the array.
 * Returns: a pointer to the sizes of each dimension.
 */
const int* mlx_array_shape(const mlx_array arr);
/**
 * The strides of the array.
 * Returns: a pointer to the sizes of each dimension.
 */
const size_t* mlx_array_strides(const mlx_array arr);
/** Signed native strides, in elements, including reversed views. */
const int64_t* mlx_array_signed_strides(const mlx_array arr);
/**
 * The shape of the array in a particular dimension.
 */
int mlx_array_dim(const mlx_array arr, int dim);
/**
 * The array element type.
 */
mlx_dtype mlx_array_dtype(const mlx_array arr);

/**
 * Evaluate the array.
 */
int mlx_array_eval(mlx_array arr);

/**
 * Access the value of a scalar array.
 */
int mlx_array_item_bool(bool* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint8(uint8_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint16(uint16_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint32(uint32_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint64(uint64_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int8(int8_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int16(int16_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int32(int32_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int64(int64_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_float32(float* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_float64(double* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_complex64(mlx_complex64_t* res, const mlx_array arr);

#ifdef HAS_FLOAT16
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_float16(float16_t* res, const mlx_array arr);
#endif

#ifdef HAS_BFLOAT16
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_bfloat16(bfloat16_t* res, const mlx_array arr);
#endif

/**
 * Returns a pointer to the array data, cast to `bool*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const bool* mlx_array_data_bool(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint8_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint8_t* mlx_array_data_uint8(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint16_t* mlx_array_data_uint16(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint32_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint32_t* mlx_array_data_uint32(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint64_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint64_t* mlx_array_data_uint64(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int8_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int8_t* mlx_array_data_int8(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int16_t* mlx_array_data_int16(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int32_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int32_t* mlx_array_data_int32(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int64_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int64_t* mlx_array_data_int64(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `float32*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const float* mlx_array_data_float32(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `float64*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const double* mlx_array_data_float64(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `_Complex*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const mlx_complex64_t* mlx_array_data_complex64(const mlx_array arr);

#ifdef HAS_FLOAT16
/**
 * Returns a pointer to the array data, cast to `float16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const float16_t* mlx_array_data_float16(const mlx_array arr);
#endif

#ifdef HAS_BFLOAT16
/**
 * Returns a pointer to the array data, cast to `bfloat16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const bfloat16_t* mlx_array_data_bfloat16(const mlx_array arr);
#endif

/**
 * Check if the array is available.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_available(bool* res, const mlx_array arr);

/** Fixed availability-query caller transports; no evaluation or wait. */
size_t mlx_array_availability_control_bytes(void);

/**
 * Wait on the array to be available. After this `_mlx_array_is_available`
 * returns `true`. Internal function: use at your own risk.
 */
int _mlx_array_wait(const mlx_array arr);

/**
 * Whether the array is contiguous in memory.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_contiguous(bool* res, const mlx_array arr);

/**
 * Whether the array's rows are contiguous in memory.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_row_contiguous(bool* res, const mlx_array arr);

/**
 * Whether the array's columns are contiguous in memory.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_col_contiguous(bool* res, const mlx_array arr);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
