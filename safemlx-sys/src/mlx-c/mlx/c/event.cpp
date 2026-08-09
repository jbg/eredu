/* Copyright © 2026 Apple Inc. */

#include "mlx/c/error.h"
#include "mlx/c/event.h"
#include "mlx/c/private/mlx.h"

extern "C" mlx_event mlx_event_new(void) {
  try {
    return mlx_event_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_event({nullptr});
  }
}

extern "C" int mlx_event_free(mlx_event event) {
  try {
    mlx_event_free_(event);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_synchronize(mlx_event event) {
  try {
    mlx_event_get_(event).wait();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_query(bool* complete, mlx_event event) {
  try {
    *complete = mlx_event_get_(event).is_complete();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_stream_wait_event(mlx_stream stream, mlx_event event) {
  try {
    mlx_event_get_(event).wait(mlx_stream_get_(stream));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_has_device(bool* has_device, mlx_event event) {
  try {
    *has_device = mlx_event_get_(event).device().has_value();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_get_device(mlx_device* device, mlx_event event) {
  try {
    auto& producer = mlx_event_get_(event).device();
    if (!producer) {
      throw std::runtime_error(
          "[mlx_event_get_device] Event has no producer device identity.");
    }
    mlx_device_set_(*device, *producer);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_get_backend(
    mlx_event_backend* backend,
    mlx_event event) {
  try {
    switch (mlx_event_get_(event).backend()) {
      case mlx::core::CompletionBackend::none:
        *backend = MLX_EVENT_BACKEND_NONE;
        break;
      case mlx::core::CompletionBackend::cpu:
        *backend = MLX_EVENT_BACKEND_CPU;
        break;
      case mlx::core::CompletionBackend::metal:
        *backend = MLX_EVENT_BACKEND_METAL;
        break;
      case mlx::core::CompletionBackend::cuda:
        *backend = MLX_EVENT_BACKEND_CUDA;
        break;
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
