#include "mlx/backend/common/utils.h"

TEST_CASE(
    "stride collapse local layouts distinguish inline owners and "
    "borrowed inputs") {
  CollapseStorageLayout layout;
  REQUIRE(collapse_contiguous_dims_layout(3, 0, true, layout));
  CHECK(layout.output_headers.bytes_each == 3 * sizeof(Strides));
  CHECK(layout.markers.count == 0);
  CHECK(layout.output_shape.count == 0);
  CHECK(layout.output_strides.count == 0);
  CHECK(layout.fixed_reference_bytes == 3 * sizeof(BorrowedStrides::Reference));
  REQUIRE(collapse_contiguous_dims_layout(3, 5, false, layout));
  CHECK(layout.fixed_reference_bytes == 0);
  CHECK(layout.markers.count == 0);
  REQUIRE(collapse_contiguous_dims_layout(3, 6, false, layout));
  CHECK(layout.markers.bytes_each == 20 * sizeof(ShapeElem));
  CHECK(layout.output_strides.count == 0);
  REQUIRE(collapse_contiguous_dims_layout(3, 12, true, layout));
  CHECK(layout.output_shape.bytes_each == 20 * sizeof(ShapeElem));
  CHECK(layout.output_strides.count == 3);
  CHECK(layout.output_strides.bytes_each == 20 * sizeof(int64_t));
  CHECK(layout.graph_extent_sum >= layout.requested_bytes);
  const auto saved = layout.requested_bytes;
  CHECK_FALSE(collapse_contiguous_dims_layout(SIZE_MAX, 12, true, layout));
  CHECK_FALSE(collapse_contiguous_dims_layout(3, SIZE_MAX, true, layout));
  CHECK(layout.requested_bytes == saved);
}

TEST_CASE(
    "stride collapse borrows old geometry and retains only its own "
    "result arena") {
  Counts a_count, b_count;
  Arena a(a_count, 64 * 1024), b(b_count, 64 * 1024);
  std::optional<Shape> shape;
  std::optional<Strides> first, second;
  std::optional<CollapsedDimensions> result;
  {
    Scope scope(a.value);
    shape.emplace(12, 1);
    (*shape)[0] = 2;
    (*shape)[11] = 3;
    first.emplace(12, 0);
    second.emplace(12, 0);
    (*first)[0] = 3;
    (*first)[11] = 1;
    (*second)[11] = 1;
  }
  const auto* source_address = first->data();
  {
    Scope scope(b.value);
    result.emplace(collapse_contiguous_dims(*shape, {*first, *second}));
    CHECK(first->data() == source_address);
    CHECK(std::get<0>(*result) == Shape({2, 3}));
    CHECK(std::get<1>(*result)[0] == Strides({3, 1}));
    CHECK(std::get<1>(*result)[1] == Strides({0, 1}));
    CHECK(std::get<1>(*result).get_allocator().resource() == b.value);
    CHECK(std::get<1>(*result)[0].get_allocator().resource() == b.value);
  }
  a.reset();
  shape.reset();
  first.reset();
  second.reset();
  CHECK(a_count.retired == 1); // result retains no source loan or source arena
  b.reset();
  CHECK(b_count.retired == 0);
  result.reset();
  CHECK(b_count.retired == 1);

  Strides empty;
  auto zero = collapse_contiguous_dims(Shape{}, {empty, empty});
  CHECK(std::get<0>(zero).empty());
  CHECK(std::get<1>(zero).size() == 2);
  CHECK(std::get<1>(zero)[0].empty());
  Strides broadcast{0, 0, 0}, packed{4, 2, 1};
  auto singleton =
      collapse_contiguous_dims(Shape{1, 1, 1}, {broadcast, packed});
  CHECK(std::get<0>(singleton) == Shape({1}));
  CHECK(std::get<1>(singleton)[0] == Strides({0}));
  CHECK(std::get<1>(singleton)[1] == Strides({0}));
}

TEST_CASE(
    "stride collapse refusal preserves borrowed input and submits no work") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  const auto before = scheduler::cpu_stream_progress(stream);
  REQUIRE(before.has_value());
  Shape shape{2, 3, 4};
  Strides first{12, 4, 1}, second{0, 4, 1};
  const auto* first_address = first.data();
  Counts counts;
  Arena tiny(counts, 128);
  {
    Scope scope(tiny.value);
    bool refused = false;
    try {
      (void)collapse_contiguous_dims(shape, {first, second});
    } catch (const submission::GraphQuotaError& error) {
      refused = true;
      CHECK(error.cause() == submission::GraphFailure::exhausted);
    }
    CHECK(refused);
    CHECK(tiny.value->occupied_bytes() == 0);
  }
  CHECK(first.data() == first_address);
  CHECK(first == Strides({12, 4, 1}));
  CHECK(shape == Shape({2, 3, 4}));
  const auto after = scheduler::cpu_stream_progress(stream);
  REQUIRE(after.has_value());
  CHECK(after->accepted == before->accepted);
}

TEST_CASE(
    "stride collapse owning result survives source drop and sealed "
    "worker capture") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto gate = std::make_shared<DefaultGate>();
  OpenDefaultGate release{gate};
  struct State {
    std::atomic<int64_t> value{0};
  };
  auto state = std::make_shared<State>();
  BufferFixture fixture(1 << 20, 2 << 20);
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    Shape source{2, 3, 4};
    Strides a{12, 4, 1}, b{0, 4, 1};
    auto result = collapse_contiguous_dims(source, {a, b});
    enqueue_default_record(
        role, stream, [gate, state, result = std::move(result)] {
          while (!gate->open.load(std::memory_order_acquire))
            std::this_thread::yield();
          state->value = std::get<0>(result)[1] + std::get<1>(result)[0][0];
        });
  }
  role.scope->seal();
  CHECK(state->value == 0);
  CHECK(fixture.graph->occupied_bytes() > 0);
  gate->open.store(true, std::memory_order_release);
  REQUIRE(settle_default_role(*role.scope));
  CHECK(state->value == 24);
  CHECK(fixture.graph->occupied_bytes() == 0);
}

TEST_CASE(
    "stride collapse worker refusal preserves its accepted native prefix") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  BufferFixture fixture(1 << 20, 2 << 20);
  // Genuine caller-owned dynamic input, prepared cold. The helper borrows it;
  // this test does not claim its ordinary storage as part of the helper layout.
  std::vector<Strides> inputs(fixture.graph->capacity() / sizeof(Strides) + 1);
  struct State {
    std::atomic<bool> prefix{false};
    std::atomic<int> cause{0};
  };
  auto state = std::make_shared<State>();
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  auto* expected = fixture.budget.get();
  enqueue_default_record(
      role, stream, [state, expected, inputs = std::move(inputs)] {
        auto prefix = allocator::malloc_owned(24);
        state->prefix = prefix.get().original_buffer_budget() == expected;
        try {
          (void)collapse_contiguous_dims(Shape{}, inputs);
        } catch (const submission::GraphQuotaError& error) {
          state->cause = static_cast<int>(error.cause());
          throw;
        }
      });
  role.scope->seal();
  REQUIRE(settle_default_role(*role.scope));
  CHECK(state->prefix);
  CHECK(state->cause == static_cast<int>(submission::GraphFailure::exhausted));
  CHECK(role.scope->query().failed);
  REQUIRE(role.failure.get()->borrow());
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.counts->allocated == 1);
  CHECK(fixture.counts->freed == 1);
  CHECK(fixture.graph->occupied_bytes() == 0);
}

namespace {
void stride_collapse_numerical(Device::DeviceType type) {
  auto stream = new_stream(type);
  prepare_default_streams(stream, stream);
  Shape shape(12, 1);
  shape[10] = 2;
  shape[11] = 3;
  auto raw = reshape(array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}), shape);
  auto axes = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10};
  auto left = transpose(raw, axes, stream);
  Shape row_shape(12, 1);
  row_shape[10] = 3;
  auto right = reshape(array({10.f, 20.f, 30.f}), row_shape);
  auto condition = greater(left, array(3.f), stream);
  eval(left, right, condition);
  auto produce = [&] {
    auto sum = add(left, right, stream);
    return where(condition, sum, left, stream);
  };
  eval(
      produce(),
      contiguous(left, false, stream)); // actual cold kernel preparation
  BufferFixture fixture(1 << 20, 2 << 20);
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    auto result = produce();
    auto copied = contiguous(left, false, stream);
    Completion completion;
    REQUIRE(
        async_eval_with_completion_scoped_on_stream(
            ArrayVector{result, copied}, *role.scope, completion, stream) ==
        submission::ScopedProgress::observed);
    role.scope->seal();
    REQUIRE(
        wait_scoped_completion(*role.scope, completion) ==
        ScopedEvaluation::complete);
    REQUIRE(settle_default_role(*role.scope));
    REQUIRE(
        validate_scoped_array(result, *role.scope) ==
        ScopedEvaluation::complete);
    REQUIRE(
        validate_scoped_array(copied, *role.scope) ==
        ScopedEvaluation::complete);
    const float copy_expected[] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    for (size_t i = 0; i < 6; ++i)
      CHECK(copied.data<float>()[i] == copy_expected[i]);
    const float expected[] = {1.f, 14.f, 2.f, 25.f, 3.f, 36.f};
    for (size_t i = 0; i < 6; ++i)
      CHECK(result.data<float>()[i] == expected[i]);
    CHECK(result.buffer().original_buffer_budget() == fixture.budget.get());
  }
  REQUIRE(settle_default_role(*role.scope));
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.graph->occupied_bytes() == 0);
}
} // namespace
TEST_CASE(
    "stride collapse nonzero high rank CPU binary ternary and copy "
    "preserve values") {
  stride_collapse_numerical(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE(
    "stride collapse nonzero high rank Metal binary ternary and copy "
    "preserve values") {
  stride_collapse_numerical(Device::gpu);
}
#endif
