// Included once by operation_observer_tests.cpp; uses its actual Owner/runtime
// fixture and terminal wait. No second native producer is implemented here.
#include "mlx/c/private/operation_event.h"
#include <cstring>

TEST_CASE("exact original root layout is pure checked and leaves failed outputs intact") {
  CHECK(current_scope() == nullptr);
  mlx_operation_root_storage_layout zero{}, one{};
  REQUIRE(mlx_operation_event_root_storage_layout(&zero, 0));
  REQUIRE(mlx_operation_event_root_storage_layout(&one, 1));
  CHECK(zero.root_count == 0);
  CHECK(zero.roots_bytes == 0);
  CHECK(zero.roots_graph_extent == 0);
  CHECK(zero.graph_blocks == 1);
  CHECK(zero.object_bytes == sizeof(mlx_c_detail::OperationEvent));
  CHECK(zero.object_alignment == alignof(mlx_c_detail::OperationEvent));
  CHECK(one.roots_bytes == sizeof(array));
  CHECK(one.roots_alignment == alignof(array));
  CHECK(one.graph_blocks == 2);
  CHECK(one.graph_request_extent == one.object_graph_extent + one.roots_graph_extent);
  CHECK(!mlx_operation_event_root_storage_layout(nullptr, 1));
  auto unchanged = one;
  CHECK(!mlx_operation_event_root_storage_layout(&unchanged, std::numeric_limits<size_t>::max()));
  CHECK(std::memcmp(&unchanged, &one, sizeof(one)) == 0);
  CHECK(current_scope() == nullptr);
  Owner owner;
  const auto graph = owner.graph->occupied_bytes();
  const auto records = owner.records->occupied_bytes();
  REQUIRE(mlx_operation_event_root_storage_layout(&one, 17));
  CHECK(owner.graph->occupied_bytes() == graph);
  CHECK(owner.records->occupied_bytes() == records);
  CHECK(current_scope() == owner.scope);
}

TEST_CASE("exact original roots retain cardinality refusals and dynamic positive control") {
  auto stream = new_stream(Device::cpu);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  array source(7.0f);
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto baseline = owner.graph->occupied_bytes();
  const auto records = owner.records->occupied_bytes();
  mlx_operation_event event{};
  REQUIRE(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, 2) == 0);
  auto* native = mlx_c_detail::operation_get(event);
  const auto* buffer = native->roots.data();
  const auto occupied = owner.graph->occupied_bytes();
  CHECK(native->roots.capacity() == 2);
  CHECK(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, 2) == 4);
  REQUIRE(mlx_operation_event_append(event, mlx_array{&source}) == 0);
  CHECK(mlx_operation_event_submit(event) == 4);
  CHECK(mlx_operation_event_submit_on_stream(event, mlx_stream{&stream}) == 4);
  CHECK(!native->submitted);
  CHECK(native->roots.size() == 1);
  REQUIRE(mlx_operation_event_append(event, mlx_array{&source}) == 0);
  CHECK(mlx_operation_event_append(event, mlx_array{&source}) == 4);
  CHECK(native->roots.size() == 2);
  CHECK(native->roots.data() == buffer);
  CHECK(native->roots[0].id() == source.id());
  CHECK(native->roots[1].id() == source.id());
  CHECK(owner.graph->occupied_bytes() == occupied);
  CHECK(owner.records->occupied_bytes() == records);
  mlx_operation_event_free(event);
  CHECK(owner.graph->occupied_bytes() == baseline);
  mlx_operation_event dynamic{};
  REQUIRE(mlx_operation_event_new(&dynamic, observer) == 0);
  for (unsigned i = 0; i != 3; ++i)
    REQUIRE(mlx_operation_event_append(dynamic, mlx_array{&source}) == 0);
  REQUIRE(mlx_operation_event_submit(dynamic) == 0);
  CHECK(mlx_operation_event_query(dynamic) == 0);
  mlx_operation_event_free(dynamic);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == baseline);
  mlx_submission_observer_release(observer);
}

TEST_CASE("exact original reserve failure returns an owning failed wrapper and actual cause") {
  auto stream = new_stream(Device::cpu);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  array source(11.0f);
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto baseline = owner.graph->occupied_bytes();
  const auto records = owner.records->occupied_bytes();
  const auto roots = owner.graph->capacity() / sizeof(array) + 1;
  mlx_operation_root_storage_layout layout{};
  REQUIRE(mlx_operation_event_root_storage_layout(&layout, roots));
  REQUIRE(layout.roots_bytes > owner.graph->capacity());
  mlx_operation_event failed{};
  CHECK(mlx_operation_event_new_exact(&failed, observer, mlx_stream{&stream}, roots) == 7);
  REQUIRE(failed.ctx != nullptr);
  const auto* native = mlx_c_detail::operation_get(failed);
  CHECK(native->root_state == mlx_c_detail::OperationEvent::RootState::failed);
  CHECK(native->scope == owner.scope);
  CHECK(native->quota == owner.graph);
  CHECK(native->roots.empty());
  CHECK(native->roots.capacity() == 0);
  CHECK(mlx_operation_event_append(failed, mlx_array{&source}) == 4);
  CHECK(mlx_operation_event_submit(failed) == 4);
  CHECK(mlx_operation_event_submit_on_stream(failed, mlx_stream{&stream}) == 4);
  CHECK(owner.records->occupied_bytes() == records);
  CHECK(!owner.scope->query_records().pending);
  REQUIRE(owner.failure.get()->borrow() != nullptr);
  CHECK(std::strcmp(owner.failure.get()->borrow()->message,
      "graph metadata capacity exhausted") == 0);
  CHECK(owner.graph->occupied_bytes() > baseline);
  mlx_operation_event_defer(failed);
  CHECK(owner.graph->occupied_bytes() > baseline);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == baseline);
  CHECK(std::strcmp(owner.failure.get()->borrow()->message,
      "graph metadata capacity exhausted") == 0);
  mlx_submission_observer_release(observer);
}

TEST_CASE("exact original constructor validates stream owner and count before mutation") {
  auto stream = new_stream(Device::cpu);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph = owner.graph->occupied_bytes();
  const auto records = owner.records->occupied_bytes();
  mlx_operation_event event{};
  auto missing = Stream{-1, stream.device};
  CHECK(mlx_operation_event_new_exact(&event, observer, mlx_stream{&missing}, 1) == 4);
  CHECK(event.ctx == nullptr);
  CHECK(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream},
      std::numeric_limits<size_t>::max()) == 1);
  CHECK(event.ctx == nullptr);
  auto* child = new Scope;
  CHECK(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, 1) == 4);
  CHECK(event.ctx == nullptr);
  child->seal(); child->release();
  owner.scope->seal();
  CHECK(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, 1) == 4);
  CHECK(event.ctx == nullptr);
  CHECK(owner.graph->occupied_bytes() == graph);
  CHECK(owner.records->occupied_bytes() == records);
  CHECK(owner.failure.get()->borrow() == nullptr);
  mlx_submission_observer_release(observer);
}

void original_exact_root_frontiers(Device::DeviceType device) {
  auto stream = new_stream(device);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  const float values[] = {2.0f, -3.0f, 7.0f};
  array input(values, Shape{3}, float32);
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph = owner.graph->occupied_bytes();
  const auto records = owner.records->occupied_bytes();
  for (unsigned mode = 0; mode != 3; ++mode) {
    {
      auto lazy = negative(input, stream);
      const size_t count = mode == 0 ? 0 : 3;
      mlx_operation_event event{};
      REQUIRE(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, count) == 0);
      auto* native = mlx_c_detail::operation_get(event);
      const auto* buffer = native->roots.data();
      const auto prepared = owner.graph->occupied_bytes();
      if (count) {
        REQUIRE(mlx_operation_event_append(event, mlx_array{&input}) == 0);
        REQUIRE(mlx_operation_event_append(event, mlx_array{&input}) == 0);
        REQUIRE(mlx_operation_event_append(event, mode == 1 ? mlx_array{&input} : mlx_array{&lazy}) == 0);
      }
      CHECK(native->roots.data() == buffer);
      CHECK(owner.graph->occupied_bytes() == prepared);
      REQUIRE(mlx_operation_event_submit_on_stream(event, mlx_stream{&stream}) == 0);
      REQUIRE(mlx_operation_event_wait(event) == 0);
      await_original_terminal(owner);
      REQUIRE(validate_scoped_array(input, *owner.scope) == ScopedEvaluation::complete);
      for (unsigned i = 0; i != 3; ++i) CHECK(input.data<float>()[i] == values[i]);
      if (mode == 2) {
        REQUIRE(validate_scoped_array(lazy, *owner.scope) == ScopedEvaluation::complete);
        for (unsigned i = 0; i != 3; ++i) CHECK(lazy.data<float>()[i] == -values[i]);
      }
      mlx_operation_event_free(event);
    }
    await_original_terminal(owner);
    REQUIRE(mlx_submission_observer_retire(observer) == 0);
    CHECK(owner.graph->occupied_bytes() == graph);
    CHECK(owner.records->occupied_bytes() == records);
  }
  mlx_submission_observer_release(observer);
}
TEST_CASE("exact original CPU root buffer supports empty identity and nonzero duplicate frontiers") {
  original_exact_root_frontiers(Device::cpu);
}
