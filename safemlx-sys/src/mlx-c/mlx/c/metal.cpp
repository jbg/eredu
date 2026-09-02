/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/metal.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_metal_is_available(bool* res) {
  try {
    *res = mlx::core::metal::is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_set_metallib_path(const char* path) {
  try {
    if (path == nullptr) {
      throw std::invalid_argument("[mlx_metal_set_metallib_path] path is null");
    }
    mlx::core::metal::set_metallib_path(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_set_embedded_metallib(
    const uint8_t* compressed_data,
    size_t compressed_size,
    size_t uncompressed_size) {
  try {
    if (compressed_data == nullptr || compressed_size == 0 ||
        uncompressed_size == 0) {
      throw std::invalid_argument(
          "[mlx_metal_set_embedded_metallib] metallib data and sizes must be non-empty");
    }
    mlx::core::metal::set_embedded_metallib(
        compressed_data, compressed_size, uncompressed_size);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_start_capture(const char* path) {
  try {
    mlx::core::metal::start_capture(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_stop_capture(void) {
  try {
    mlx::core::metal::stop_capture();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
