/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_ERROR_H
#define MLX_ERROR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_error Error management
 */
/**@{*/

typedef void (*mlx_error_handler_func)(const char* msg, void* data);

/* Exact static handler slots only; no TLS or dynamically supplied handler owner.
 * Pure query: does not install a handler or invoke native runtime entry. */
size_t mlx_error_static_storage_bytes(void);

/**
 * Set the error handler.
 */
void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*));

/**
 * Throw an error.
 */
/** Fixed code for the error currently delivered to the synchronous handler.
 *  Zero is unrelated; 1 exhausted, 2 layout, 3 parent, 4 unbound factory.
 *  Does not query/progress native submission or establish terminal status. */
unsigned mlx_error_submission_tracking_failure(void);
unsigned mlx_error_graph_metadata_failure(void);
void* mlx_error_physical_backing_failure(void);
void _mlx_error(const char* file, const int line, const char* fmt, ...);

/**
 * Throw an error. Macro which passes file name and line number to _mlx_error().
 */
#define mlx_error(...) _mlx_error(__FILE__, __LINE__, __VA_ARGS__)

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
