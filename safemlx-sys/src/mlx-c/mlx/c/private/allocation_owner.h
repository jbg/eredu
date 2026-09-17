#pragma once

namespace mlx_c_detail {
// Remains disarmed through every fallible allocation and insertion. In
// particular shared_ptr construction must not consume caller ownership when
// allocation of its control block fails.
struct AllocationOwnerPayload {
  void* payload{nullptr};
  void (*release)(void*){nullptr};
  ~AllocationOwnerPayload() {
    if (payload) {
      release(payload);
    }
  }
};
} // namespace mlx_c_detail
