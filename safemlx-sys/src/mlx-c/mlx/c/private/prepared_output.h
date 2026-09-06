#ifndef MLX_PREPARED_OUTPUT_PRIVATE_H
#define MLX_PREPARED_OUTPUT_PRIVATE_H

#include <memory>
#include <type_traits>
#include <utility>

// Reserve a C output's owning wrapper before native submission. A failed
// producer restores an initially empty output; successful publication needs no
// new wrapper and uses noexcept move assignment. Destruction of a reused native
// value retains its existing behavior. Values are untouched until publication.
template <class Value, class Handle>
class mlx_output_preparation_ {
 public:
  template <class... Args>
  explicit mlx_output_preparation_(Handle& destination, Args&&... args)
      : destination_(destination) {
    if (!destination_.ctx) {
      storage_ = std::make_unique<Value>(std::forward<Args>(args)...);
      destination_.ctx = storage_.get();
    }
  }

  ~mlx_output_preparation_() {
    if (storage_) {
      destination_.ctx = nullptr;
    }
  }

  void publish(Value&& value) noexcept {
    static_assert(std::is_nothrow_move_assignable_v<Value>);
    *static_cast<Value*>(destination_.ctx) = std::move(value);
    storage_.release();
  }

 private:
  Handle& destination_;
  std::unique_ptr<Value> storage_;
};

#endif
