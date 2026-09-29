#include "doctest/doctest.h"

#include "mlx/c/array.h"
#include "mlx/c/private/array.h"
#include "mlx/mlx.h"

#include <vector>

using namespace mlx::core;

TEST_CASE("array storage custom backing never queries allocator headers") {
  // The pointer is deliberately not an MLX allocation. Calling allocator.size
  // would inspect unrelated bytes on CPU or an invalid device buffer elsewhere.
  std::vector<float> external{1.0f, -2.0f, 3.0f, 4.0f};
  size_t retired = 0;
  {
    auto deleter = [&](allocator::Buffer) { ++retired; };
    array first(allocator::Buffer{external.data()}, {4}, float32, deleter);
    array second(allocator::Buffer{external.data()}, {4}, float32, deleter);
    const auto owners = first.data_shared_ptr().use_count();
    auto first_handle = mlx_array_new_(first);
    auto second_handle = mlx_array_new_(second);
    uintptr_t first_identity = 0, second_identity = 0;
    bool owned = true;
    size_t capacity = 123;
    REQUIRE(_mlx_array_storage_metadata(
                &first_identity, &owned, &capacity, first_handle) == 0);
    CHECK(first_identity != 0);
    CHECK_FALSE(owned);
    CHECK(capacity == 0);
    CHECK(first.data_shared_ptr().use_count() == owners);
    REQUIRE(_mlx_array_storage_metadata(
                &second_identity, &owned, &capacity, second_handle) == 0);
    // Distinct Data wrappers can refer to the same external pointer. The C
    // identity is wrapper equality only; it does not prove disjoint storage.
    CHECK(second_identity != first_identity);
    CHECK_FALSE(owned);
    CHECK(capacity == 0);
    CHECK(retired == 0);
    mlx_array_free_(first_handle);
    mlx_array_free_(second_handle);
  }
  CHECK(retired == 2);
  const std::vector<float> expected{1.0f, -2.0f, 3.0f, 4.0f};
  CHECK(external == expected);
}
