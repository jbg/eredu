#pragma once
#include <stddef.h>
#include "mlx/c/stream.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mlx_stream_registration_layout_ {
  size_t object_bytes;
  size_t object_alignment;
  size_t wrapper_bytes;
  size_t wrapper_alignment;
  size_t controls;
} mlx_stream_registration_layout;
typedef struct mlx_stream_registration_static_layout_ {
  size_t bytes;
  unsigned qualified;
} mlx_stream_registration_static_layout;
void mlx_stream_registration_static_layout_for(mlx_stream_registration_static_layout*);
unsigned mlx_stream_registration_layout_for(mlx_stream_registration_layout*);
/* CPU registration and inline encoder only. No std::thread, event or GPU queue.
   All failures preserve output and owner. Success keeps raw owner until actual
   process registration destruction. The returned wrapper is an independent
   exact Stream object; mlx_stream_copy_free retires only that wrapper. */
unsigned mlx_stream_register_cpu(mlx_stream*, mlx_stream_registration_layout,
    void* owner, void (*retire)(void*));
unsigned mlx_stream_registration_borrow(mlx_stream, const void* expected_owner);
#ifdef __cplusplus
}
#endif
