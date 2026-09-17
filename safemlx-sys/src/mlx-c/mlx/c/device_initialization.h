#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mlx_device_initialization_source {
  const uint8_t* compressed_data;
  size_t compressed_size;
  size_t uncompressed_size;
} mlx_device_initialization_source;
typedef struct mlx_device_initialization_layout {
  size_t object_bytes, decoded_bytes, scratch_bytes, dispatch_copy_bytes;
  size_t override_bytes, controls;
} mlx_device_initialization_layout;
typedef struct mlx_device_initialization_static_layout {
  size_t bytes;
  unsigned qualified;
  unsigned required;
} mlx_device_initialization_static_layout;
// Fixed native statuses, no error handler and no source registration. Output is
// unchanged on refusal. Success alone transfers the supplied retained owner.
void mlx_device_initialization_static_layout_for(mlx_device_initialization_static_layout*);
unsigned mlx_device_initialization_layout_for(mlx_device_initialization_layout*, mlx_device_initialization_source);
unsigned mlx_device_initialize(uint64_t*, mlx_device_initialization_source,
    mlx_device_initialization_layout, void*, void (*)(void*));
unsigned mlx_device_initialized_borrow(uint64_t);
#ifdef __cplusplus
}
#endif
