#ifndef MLX_PREPARED_ARRAY_OUTPUT_PRIVATE_H
#define MLX_PREPARED_ARRAY_OUTPUT_PRIVATE_H

#include "mlx/array.h"
#include "mlx/c/array.h"

#include <new>
#include <type_traits>
#include <utility>

// Reserve only wrapper storage: constructing an empty array here would also
// allocate temporary ArrayDesc/Data objects and a zero-sized native buffer.
class mlx_array_output_preparation_ {
 public:
  explicit mlx_array_output_preparation_(mlx_array& destination)
      : destination_(destination) {
    static_assert(
        alignof(mlx::core::array) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
    if (!destination_.ctx) {
      storage_ = ::operator new(sizeof(mlx::core::array));
    }
  }

  mlx_array_output_preparation_(const mlx_array_output_preparation_&) = delete;
  mlx_array_output_preparation_& operator=(const mlx_array_output_preparation_&) = delete;
  mlx_array_output_preparation_(mlx_array_output_preparation_&&) = delete;
  mlx_array_output_preparation_& operator=(mlx_array_output_preparation_&&) = delete;

  ~mlx_array_output_preparation_() {
    // No array exists in this storage until publication. Ordinary allocation
    // and placement construction preserve mlx_array_free's delete contract.
    ::operator delete(storage_);
  }

  void publish(mlx::core::array&& value) noexcept {
    static_assert(std::is_nothrow_move_constructible_v<mlx::core::array>);
    static_assert(std::is_nothrow_move_assignable_v<mlx::core::array>);
    if (storage_) {
      destination_.ctx = ::new (storage_) mlx::core::array(std::move(value));
      storage_ = nullptr;
    } else {
      *static_cast<mlx::core::array*>(destination_.ctx) = std::move(value);
    }
  }

 private:
  mlx_array& destination_;
  void* storage_{nullptr};
};

#endif
