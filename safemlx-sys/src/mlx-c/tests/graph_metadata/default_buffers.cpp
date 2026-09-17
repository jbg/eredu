// Shared real fixture owners from data_controls.cpp and original_buffers.cpp.
#include <cmath>
#include <cstdlib>
#include <numeric>
#include "mlx/linalg.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/reduce.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/ternary.h"
#include "mlx/scoped_evaluation.h"

namespace {
void prepare_default_streams(Stream a, Stream b) {
  auto input = array({1.0f, 2.0f, 3.0f});
  eval(add(negative(input, a), input, b));
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(
      mlx_submission_prepare_runtime(
          &baseline, mlx_stream{&a}, mlx_stream{&b}) == 0);
}
bool settle_default_role(submission::Scope& scope) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (scope.query().activity == submission::Activity::pending) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    (void)scope.progress_scoped();
    std::this_thread::yield();
  }
  for (;;) {
    auto retired = scope.retire_scoped_records();
    if (retired == submission::ScopedProgress::observed)
      return true;
    if (retired != submission::ScopedProgress::busy ||
        std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::yield();
  }
}
struct DefaultGate {
  std::atomic<bool> open{false};
};
struct OpenDefaultGate {
  std::shared_ptr<DefaultGate> gate;
  ~OpenDefaultGate() {
    gate->open.store(true, std::memory_order_release);
  }
};
template <class F>
void enqueue_default_record(DataRole& role, Stream stream, F&& body) {
  auto pending = submission::Record::create<DataCaptureRecord>();
  pending->enter();
  auto* record = pending.release();
  BufferRecordFinish guard{record};
  record->reserve_streams(1);
  record->prepare_stream(stream);
  {
    submission::RecordDispatchGuard dispatch(*record);
    scheduler::enqueue(stream, std::forward<F>(body));
  }
  record->finish(false);
  guard.value = nullptr;
}
struct DefaultChild {
  FailureCarrierRef failure{data_failure_owner()};
  std::unique_ptr<submission::Scope, DataScopeRelease> scope{
      new submission::Scope};
  DefaultChild() {
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(
        scope->require_original_controls() ==
        submission::NativeControlFailure::none);
    REQUIRE(scope->bind_failure(failure));
    REQUIRE(
        scope->enable_original_controls() ==
        submission::NativeControlFailure::none);
  }
};
} // namespace

TEST_CASE(
    "default original buffers preserve callable allocator and refuse shared cache fallback") {
  auto ordinary = allocator::malloc_owned(24);
  CHECK_FALSE(ordinary.get().has_original_allocation());
  ordinary.reset(); // leave a genuine ordinary cache opportunity
  const auto cached = get_cache_memory();
  {
    BufferFixture no_room(0);
    DataRole role(no_room.graph.get());
    (void)bind_buffer(role, no_room);
    auto* callable = &allocator::malloc;
    buffer_refusal([&] { (void)callable(24); }, BufferCause::capacity);
    CHECK(no_room.counts->allocated == 0);
    CHECK(no_room.budget->occupied_bytes() == 0);
    CHECK(get_cache_memory() == cached);
    buffer_refusal(
        [&] { (void)allocator::allocator().malloc(24); },
        BufferCause::unsupported);
    buffer_refusal(
        [&] { (void)allocator::make_buffer(nullptr, 0); },
        BufferCause::unsupported);
    buffer_refusal(
        [&] {
          (void)allocator::allocator().malloc_host_transfer(
              24, allocator::HostTransferPolicy::transfer);
        },
        BufferCause::unsupported);
    CHECK(no_room.counts->allocated == 0);
  }
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    // Both factories use the same available, primitive-free descriptor path.
    auto prepared =
        array::with_allocated_buffer(6 * sizeof(float), Shape{2, 3}, float32);
    array prior(allocator::malloc(6 * sizeof(float)), Shape{2, 3}, float32);
    CHECK(prepared.status() == array::Status::available);
    CHECK(prepared.status() == prior.status());
    CHECK(prepared.shape() == prior.shape());
    CHECK(prepared.strides() == prior.strides());
    CHECK(prepared.data_size() == prior.data_size());
    CHECK(prepared.flags().contiguous == prior.flags().contiguous);
    CHECK(prepared.flags().row_contiguous == prior.flags().row_contiguous);
    CHECK(prepared.flags().col_contiguous == prior.flags().col_contiguous);
  }
  for (size_t bytes : {size_t{0}, sizeof(float), size_t{24}}) {
    size_t charged = 0;
    REQUIRE(fixture.budget->physical_capacity(bytes, charged));
    auto buffer = allocator::malloc_owned(bytes);
    CHECK(fixture.budget->occupied_bytes() == charged);
    CHECK(buffer.get().has_original_allocation() == (charged != 0));
    if (bytes) {
      *static_cast<float*>(buffer.get().raw_ptr()) = 7.25f;
      CHECK(*static_cast<float*>(buffer.get().raw_ptr()) == 7.25f);
    }
  }
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(get_cache_memory() == cached);
}

TEST_CASE(
    "default original buffers require the same budget through unbound child ancestry") {
  BufferFixture a, b;
  DataRole parent(a.graph.get());
  (void)bind_buffer(parent, a);
  {
    DefaultChild child;
    buffer_refusal(
        [&] { (void)allocator::malloc(24); }, BufferCause::missing_binding);
    {
      DefaultChild grandchild;
      buffer_refusal(
          [&] { grandchild.scope->bind_original_buffer_budget(*b.budget); },
          BufferCause::foreign_domain);
      buffer_refusal(
          [&] { (void)allocator::malloc(24); }, BufferCause::missing_binding);
      grandchild.scope->bind_original_buffer_budget(*a.budget);
      auto buffer = allocator::malloc_owned(24);
      CHECK(buffer.get().original_buffer_budget() == a.budget.get());
    }
    CHECK(a.budget->occupied_bytes() == 0);
    // The grandchild marked native construction through its ancestry. Binding
    // remains a cold, once-only operation; settlement does not restore it.
    buffer_refusal(
        [&] { child.scope->bind_original_buffer_budget(*a.budget); },
        BufferCause::invalid_scope);
  }
  {
    DefaultChild fresh_child;
    REQUIRE(fresh_child.scope->active_on_owner_thread());
    REQUIRE(fresh_child.scope->query().activity == submission::Activity::none);
    REQUIRE(fresh_child.scope->original_buffer_budget() == nullptr);
    fresh_child.scope->bind_original_buffer_budget(*a.budget);
    auto buffer = allocator::malloc_owned(24);
    CHECK(buffer.get().original_buffer_budget() == a.budget.get());
  }
  CHECK(a.budget->occupied_bytes() == 0);
  CHECK(b.counts->allocated == 0);
}

TEST_CASE(
    "default original buffers cover real matmul and cross-stream evaluator output") {
  auto producer =
      new_stream(is_available(Device::gpu) ? Device::gpu : Device::cpu);
  auto consumer = new_stream(Device::cpu);
  prepare_default_streams(producer, consumer);
  auto input = reshape(array({1.0f, 2.0f, 3.0f, 4.0f}), {2, 2});
  eval(add(matmul(input, input, producer), input, consumer));
  BufferFixture fixture(1 << 20, 2 << 20);
  const auto cached = get_cache_memory();
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    auto product = matmul(input, input, producer);
    auto result = add(product, input, consumer);
    Completion completion;
    REQUIRE(
        async_eval_with_completion_scoped_on_stream(
            ArrayVector{product, result}, *role.scope, completion, consumer) ==
        submission::ScopedProgress::observed);
    role.scope->seal();
    REQUIRE(
        wait_scoped_completion(*role.scope, completion) ==
        ScopedEvaluation::complete);
    REQUIRE(settle_default_role(*role.scope));
    REQUIRE(
        validate_scoped_array(product, *role.scope) ==
        ScopedEvaluation::complete);
    REQUIRE(
        validate_scoped_array(result, *role.scope) ==
        ScopedEvaluation::complete);
    CHECK(product.buffer().original_buffer_budget() == fixture.budget.get());
    CHECK(result.buffer().original_buffer_budget() == fixture.budget.get());
    const float expected[] = {8, 12, 18, 26};
    for (size_t i = 0; i < 4; ++i)
      CHECK(result.data<float>()[i] == expected[i]);
    CHECK(fixture.budget->occupied_bytes() > 0);
    CHECK(get_cache_memory() == cached);
  }
  REQUIRE(settle_default_role(*role.scope));
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.counts->allocated == fixture.counts->freed);
  CHECK(fixture.graph->occupied_bytes() == 0);
}

TEST_CASE(
    "default original buffers follow real inverse and QR worker allocations after scope seal") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto input = reshape(array({4.0f, 1.0f, 2.0f, 3.0f}), {2, 2});
  eval(linalg::inv(input, stream));
  auto warm_qr = linalg::qr(input, stream);
  eval(warm_qr.first, warm_qr.second);
  auto gate = std::make_shared<DefaultGate>();
  OpenDefaultGate unlock{gate};
  BufferFixture fixture(1 << 20, 2 << 20);
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  enqueue_default_record(role, stream, [gate] {
    while (!gate->open.load(std::memory_order_acquire))
      std::this_thread::yield();
  });
  {
    auto inverse = linalg::inv(input, stream);
    auto [q, r] = linalg::qr(input, stream);
    Completion completion;
    REQUIRE(
        async_eval_with_completion_scoped_on_stream(
            ArrayVector{inverse, q, r}, *role.scope, completion, stream) ==
        submission::ScopedProgress::observed);
    const auto host_allocations = fixture.counts->allocated.load();
    role.scope->seal();
    gate->open.store(true, std::memory_order_release);
    REQUIRE(
        wait_scoped_completion(*role.scope, completion) ==
        ScopedEvaluation::complete);
    REQUIRE(settle_default_role(*role.scope));
    REQUIRE(
        validate_scoped_array(inverse, *role.scope) ==
        ScopedEvaluation::complete);
    REQUIRE(
        validate_scoped_array(q, *role.scope) == ScopedEvaluation::complete);
    REQUIRE(
        validate_scoped_array(r, *role.scope) == ScopedEvaluation::complete);
    CHECK(
        fixture.counts->allocated >
        host_allocations); // actual LAPACK scratch on worker
    const float expected[] = {0.3f, -0.1f, -0.2f, 0.4f};
    for (size_t i = 0; i < 4; ++i)
      CHECK(
          inverse.data<float>()[i] ==
          doctest::Approx(expected[i]).epsilon(1e-5));
    for (size_t row = 0; row < 2; ++row)
      for (size_t col = 0; col < 2; ++col) {
        float value = 0;
        for (size_t k = 0; k < 2; ++k)
          value += q.data<float>()[row * 2 + k] * r.data<float>()[k * 2 + col];
        CHECK(
            value ==
            doctest::Approx(input.data<float>()[row * 2 + col]).epsilon(1e-5));
      }
    CHECK(inverse.buffer().original_buffer_budget() == fixture.budget.get());
    CHECK(q.buffer().original_buffer_budget() == fixture.budget.get());
    CHECK(r.buffer().original_buffer_budget() == fixture.budget.get());
  }
  REQUIRE(settle_default_role(*role.scope));
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.counts->allocated == fixture.counts->freed);
}

TEST_CASE(
    "default original worker capacity failure frees its real scratch prefix and keeps cause") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  BufferFixture fixture;
  struct State {
    std::atomic<bool> saw_prefix{false};
  };
  auto state = std::make_shared<State>();
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  auto* expected = fixture.budget.get();
  enqueue_default_record(role, stream, [state, expected] {
    auto prefix = allocator::malloc_owned(24);
    state->saw_prefix = prefix.get().original_buffer_budget() == expected;
    // The actual remaining budget is smaller than one complete capacity-sized
    // allocation while prefix is live. No substituted physical refusal flag.
    auto refused = allocator::malloc_owned(expected->capacity());
  });
  role.scope->seal();
  REQUIRE(settle_default_role(*role.scope));
  CHECK(state->saw_prefix);
  CHECK(role.scope->query().failed);
  REQUIRE(role.failure.get()->borrow());
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.counts->allocated == 1);
  CHECK(fixture.counts->freed == 1);
}

TEST_CASE(
    "default original evaluator donation keeps the previous request birth") {
  auto stream =
      new_stream(is_available(Device::gpu) ? Device::gpu : Device::cpu);
  prepare_default_streams(stream, stream);
  auto warm = array({1.0f, 2.0f, 3.0f});
  eval(negative(warm, stream));
  BufferFixture a, b;
  std::optional<array> source(array({1.0f, 2.0f, 3.0f}));
  uint64_t generation = 0;
  void* pointer = nullptr;
  {
    DataRole first(a.graph.get());
    (void)bind_buffer(first, a);
    source->allocate_data(24);
    fill_buffer(source->buffer());
    generation = source->buffer().original_allocation_generation();
    pointer = source->buffer().ptr();
  }
  DataRole second(b.graph.get());
  (void)bind_buffer(second, b);
  {
    auto output = negative(*source, stream);
    source.reset();
    Completion completion;
    REQUIRE(
        async_eval_with_completion_scoped_on_stream(
            ArrayVector{output}, *second.scope, completion, stream) ==
        submission::ScopedProgress::observed);
    second.scope->seal();
    REQUIRE(
        wait_scoped_completion(*second.scope, completion) ==
        ScopedEvaluation::complete);
    REQUIRE(settle_default_role(*second.scope));
    REQUIRE(
        validate_scoped_array(output, *second.scope) ==
        ScopedEvaluation::complete);
    CHECK(output.buffer().ptr() == pointer);
    CHECK(output.buffer().original_allocation_generation() == generation);
    CHECK(output.buffer().original_buffer_budget() == a.budget.get());
    CHECK(output.data<float>()[0] == -5.0f);
    CHECK(output.data<float>()[1] == -7.0f);
    CHECK(output.data<float>()[2] == -11.0f);
  }
  REQUIRE(settle_default_role(*second.scope));
  CHECK(a.budget->occupied_bytes() == 0);
  CHECK(b.budget->occupied_bytes() == 0);
}

TEST_CASE(
    "default original A and B tasks retain distinct births through callable tail and foreign free") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto gate = std::make_shared<DefaultGate>();
  OpenDefaultGate unlock{gate};
  scheduler::enqueue(stream, [gate] {
    while (!gate->open.load(std::memory_order_acquire))
      std::this_thread::yield();
  });
  struct State {
    std::optional<array> a, b;
    std::atomic<unsigned> tails{0}, mismatches{0};
  };
  struct Tail {
    std::shared_ptr<State> state;
    allocator::OriginalBufferBudget* expected;
    Tail(
        std::shared_ptr<State> state,
        allocator::OriginalBufferBudget* expected)
        : state(std::move(state)), expected(expected) {}
    Tail(Tail&& other) noexcept
        : state(std::move(other.state)), expected(other.expected) {}
    Tail(const Tail&) = delete;
    ~Tail() {
      if (!state)
        return;
      try {
        auto temporary = allocator::malloc_owned(sizeof(float));
        if (temporary.get().original_buffer_budget() != expected)
          ++state->mismatches;
      } catch (...) {
        ++state->mismatches;
      }
      ++state->tails;
    }
  };
  auto state = std::make_shared<State>();
  BufferFixture a, b;
  auto a_counts = a.counts, b_counts = b.counts;
  DataRole a_role(a.graph.get());
  (void)bind_buffer(a_role, a);
  enqueue_default_record(
      a_role, stream, [state, tail = Tail(state, a.budget.get())] {
        state->a.emplace(array::with_allocated_buffer(24, Shape{6}, float32));
        fill_buffer(state->a->buffer());
      });
  a_role.scope->seal();
  DataRole b_role(b.graph.get());
  (void)bind_buffer(b_role, b);
  enqueue_default_record(
      b_role, stream, [state, tail = Tail(state, b.budget.get())] {
        state->b.emplace(array::with_allocated_buffer(24, Shape{6}, float32));
        fill_buffer(state->b->buffer());
      });
  b_role.scope->seal();
  gate->open.store(true, std::memory_order_release);
  REQUIRE(settle_default_role(*a_role.scope));
  REQUIRE(settle_default_role(*b_role.scope));
  REQUIRE(state->a.has_value());
  REQUIRE(state->b.has_value());
  CHECK(state->tails == 2);
  CHECK(state->mismatches == 0);
  CHECK(state->a->buffer().original_buffer_budget() == a.budget.get());
  CHECK(state->b->buffer().original_buffer_budget() == b.budget.get());
  CHECK(state->a->data<float>()[5] == 19.0f);
  CHECK(state->b->data<float>()[1] == 7.0f);
  auto alias = *state->a;
  const auto generation = alias.buffer().original_allocation_generation();
  state->a.reset();
  CHECK(alias.buffer().original_allocation_generation() == generation);
  a_role.scope.reset();
  b_role.scope.reset();
  a.budget.reset();
  a.graph.reset();
  b.budget.reset();
  b.graph.reset();
  CHECK(a_counts->budget_retired == 0);
  CHECK(b_counts->budget_retired == 0);
  std::thread retire([state, alias = std::move(alias)]() mutable {
    state->b.reset();
    // The remaining A alias and both Data controls retire on this different
    // thread, without a Scope/Record/worker allocation context.
  });
  retire.join();
  CHECK(a_counts->budget_retired == 1);
  CHECK(b_counts->budget_retired == 1);
  CHECK(a_counts->allocated == a_counts->freed);
  CHECK(b_counts->allocated == b_counts->freed);
}

TEST_CASE(
    "default original worker allocation authority does not grant nested submission") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  BufferFixture fixture;
  struct State {
    std::atomic<unsigned> refused{0}, nested{0};
    std::atomic<bool> graph_data{false};
  };
  auto state = std::make_shared<State>();
  auto cold_context = scheduler::prepare_task(stream);
  auto* scheduler_owner = cold_context.scheduler;
  auto* worker = cold_context.thread;
  auto cold = [&] {
    return scheduler::make_task(
        nullptr, {}, scheduler_owner, false, [state] { ++state->nested; });
  };
  auto host_node = cold();
  auto dispatch_node = cold();
  auto worker_node = cold();
  auto cold_record = std::make_unique<submission::Record>();
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  auto accepted = worker->progress().accepted;
  CHECK_THROWS_AS(
      scheduler_owner->submit_task(worker, std::move(host_node), false),
      submission::NativeControlError);
  CHECK(worker->progress().accepted == accepted);
  auto pending = submission::Record::create<DataCaptureRecord>();
  pending->enter();
  auto* record = pending.release();
  BufferRecordFinish finish{record};
  record->reserve_streams(1);
  record->prepare_stream(stream);
  {
    submission::RecordDispatchGuard dispatch(*record);
    CHECK_THROWS_AS(
        scheduler_owner->submit_task(worker, std::move(dispatch_node), false),
        submission::NativeControlError);
    CHECK(worker->progress().accepted == accepted);
    auto* expected = fixture.budget.get();
    scheduler::enqueue(
        stream,
        [state,
         expected,
         stream,
         scheduler_owner,
         worker,
         cold_node = std::move(worker_node),
         cold_record = std::move(cold_record)]() mutable {
          auto value = array::with_allocated_buffer(24, Shape{6}, float32);
          state->graph_data =
              value.buffer().original_buffer_budget() == expected &&
              submission::current_graph_quota() != nullptr;
          auto refused = [&](auto&& operation) {
            try {
              operation();
            } catch (const submission::NativeControlError& error) {
              if (error.failure() ==
                      submission::NativeControlFailure::invalid_scope ||
                  error.failure() ==
                      submission::NativeControlFailure::erased_callable)
                ++state->refused;
            }
          };
          refused([] {
            std::unique_ptr<submission::Scope, DataScopeRelease> scope(
                new submission::Scope);
          });
          refused([] {
            auto record = submission::Record::create<DataCaptureRecord>();
          });
          refused([] { auto record = std::make_unique<submission::Record>(); });
          refused([&] { cold_record->enter(); });
          refused([&] { Event event(stream); });
          refused([&] { scheduler::enqueue(stream, [] {}); });
          const auto before = worker->progress().accepted;
          refused([&] {
            scheduler_owner->submit_task(worker, std::move(cold_node), false);
          });
          if (worker->progress().accepted != before)
            ++state->nested;
        });
  }
  record->finish(false);
  finish.value = nullptr;
  role.scope->seal();
  REQUIRE(settle_default_role(*role.scope));
  CHECK(state->graph_data);
  CHECK(state->refused == 7);
  CHECK(state->nested == 0);
  CHECK_FALSE(role.scope->query().failed);
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.graph->occupied_bytes() == 0);
}

namespace {
struct DefaultPrefixState {
  bool refused{false};
  std::weak_ptr<array::Data> output;
};
struct DefaultRefuseAfterAdd : Add {
  std::shared_ptr<DefaultPrefixState> state;
  size_t impossible_while_output_live;
  DefaultRefuseAfterAdd(
      Stream stream,
      std::shared_ptr<DefaultPrefixState> state,
      size_t capacity)
      : Add(stream),
        state(std::move(state)),
        impossible_while_output_live(capacity) {}
  void refuse(array& output, bool gpu_stream) {
    state->output = output.data_shared_ptr();
    try {
      auto rejected = allocator::malloc_owned(impossible_while_output_live);
    } catch (const allocator::OriginalBufferError&) {
      state->refused = true;
      // Fund this exact already-entered dispatch before outer failure closes
      // stream admission. No new post-failure Record or global flush.
      if (gpu_stream)
        gpu::finalize(stream());
      throw;
    }
    throw std::runtime_error(
        "capacity fixture unexpectedly allocated a second full budget");
  }
  void eval_cpu(const ArrayVector& inputs, array& output) override {
    Add::eval_cpu(inputs, output);
    refuse(output, false);
  }
  void eval_gpu(const ArrayVector& inputs, array& output) override {
    Add::eval_gpu(inputs, output);
    refuse(output, true);
  }
};
} // namespace

TEST_CASE(
    "default original allocation failure preserves real encoded prefix and exact retirement") {
  auto stream =
      new_stream(is_available(Device::gpu) ? Device::gpu : Device::cpu);
  prepare_default_streams(stream, stream);
  auto input = array({2.0f, -3.0f, 7.0f});
  eval(add(input, input, stream));
  auto state = std::make_shared<DefaultPrefixState>();
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    auto output = add(input, input, stream);
    output.primitive_ptr() =
        submission::make_graph_primitive<DefaultRefuseAfterAdd>(
            stream, state, fixture.budget->capacity());
    Completion completion;
    bool refused = false;
    try {
      (void)async_eval_with_completion_scoped_on_stream(
          ArrayVector{output}, *role.scope, completion, stream);
    } catch (const allocator::OriginalBufferError& error) {
      CHECK(error.cause() == BufferCause::capacity);
      role.failure.get()->capture_current_exception();
      refused = true;
    }
    REQUIRE(refused);
    REQUIRE(state->refused);
    REQUIRE_FALSE(state->output.expired());
    role.scope->seal();
    REQUIRE(settle_default_role(*role.scope));
    CHECK(role.scope->query().failed);
    REQUIRE(role.failure.get()->borrow());
    auto retained = state->output.lock();
    REQUIRE(retained != nullptr);
    CHECK(retained->buffer.original_buffer_budget() == fixture.budget.get());
    auto* values = static_cast<const float*>(retained->buffer.raw_ptr());
    CHECK(values[0] == 4.0f);
    CHECK(values[1] == -6.0f);
    CHECK(values[2] == 14.0f);
    CHECK(fixture.budget->occupied_bytes() > 0);
  }
  REQUIRE(settle_default_role(*role.scope));
  CHECK(state->output.expired());
  state->output.reset(); // final weak Data control still owns its Graph block
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.counts->allocated == fixture.counts->freed);
  CHECK(fixture.graph->occupied_bytes() == 0);
}

TEST_CASE(
    "default original adoption owns backing across metadata control and container refusal") {
  for (bool strided : {false, true}) {
    BufferFixture fixture;
    auto value = array({2.0f, 3.0f, 5.0f});
    auto old = value.data_shared_ptr();
    DataRole role(fixture.graph.get());
    (void)bind_buffer(role, fixture);
    Strides strides({2}, Strides::allocator_type(fixture.graph.get()));
    Strides dynamic_strides(
        12, 2, Strides::allocator_type(fixture.graph.get()));
    {
      FilledDataArena full(fixture.graph.get());
      const auto before = fixture.counts->allocated.load();
      CHECK_THROWS_AS(
          value.allocate_data(24, 6, dynamic_strides, value.flags()),
          submission::GraphQuotaError);
      CHECK_THROWS_AS(
          array::with_allocated_buffer(24, Shape{6}, float32),
          submission::GraphQuotaError);
      CHECK(fixture.counts->allocated == before);
    }
    std::optional<FilledDataArena> full;
    auto allocator_after_birth = [&](size_t bytes) {
      auto owned = allocator::malloc_owned(bytes);
      full.emplace(fixture.graph.get());
      return owned.release();
    };
    if (strided) {
      CHECK_THROWS_AS(
          value.allocate_data_with(
              24, allocator_after_birth, 6, std::move(strides), value.flags()),
          submission::GraphQuotaError);
    } else {
      CHECK_THROWS_AS(
          value.allocate_data_with(24, allocator_after_birth),
          submission::GraphQuotaError);
    }
    full.reset();
    CHECK(value.data_shared_ptr() == old);
    CHECK(value.data<float>()[2] == 5.0f);
    CHECK(fixture.budget->occupied_bytes() == 0);
    CHECK(fixture.counts->allocated == 1);
    CHECK(fixture.counts->freed == 1);
    submission::GraphVector<array::Data> rows{
        submission::GraphAllocator<array::Data>(fixture.graph.get())};
    auto incoming = allocator::malloc_owned(24);
    {
      FilledDataArena exhausted(fixture.graph.get());
      CHECK_THROWS_AS(
          rows.push_back(array::Data{incoming.release()}),
          submission::GraphQuotaError);
    }
    CHECK(rows.empty());
    CHECK(fixture.budget->occupied_bytes() == 0);
    CHECK(fixture.counts->freed == 2);
  }
}

TEST_CASE(
    "default original strided adoption prepares the old descriptor domain before publication") {
  BufferFixture a, b;
  DataRole first(a.graph.get());
  (void)bind_buffer(first, a);
  {
    Shape shape(12, 1);
    shape.back() = 3;
    auto value = array::with_allocated_buffer(
        3 * sizeof(float), std::move(shape), float32);
    value.data<float>()[0] = 2.0f;
    value.data<float>()[1] = 3.0f;
    value.data<float>()[2] = 5.0f;
    auto old = value.data_shared_ptr();
    const auto old_size = value.data_size();
    const auto old_offset = value.offset();
    const auto old_flags = value.flags();
    const auto old_strides = value.strides();
    auto destination_allocator = value.strides().get_allocator();
    first.scope->seal();
    DataRole second(b.graph.get());
    (void)bind_buffer(second, b);
    auto flags = old_flags;
    flags.contiguous = false;
    flags.row_contiguous = false;
    flags.col_contiguous = false;
    {
      Strides incoming(12, 2, Strides::allocator_type(b.graph.get()));
      FilledDataArena full(a.graph.get());
      CHECK_THROWS_AS(
          value.allocate_data(
              6 * sizeof(float), 6, std::move(incoming), flags, sizeof(float)),
          submission::GraphQuotaError);
      CHECK(value.data_shared_ptr() == old);
      CHECK(value.data_size() == old_size);
      CHECK(value.offset() == old_offset);
      CHECK(value.strides() == old_strides);
      CHECK(value.flags().contiguous == old_flags.contiguous);
      CHECK(value.flags().row_contiguous == old_flags.row_contiguous);
      CHECK(value.flags().col_contiguous == old_flags.col_contiguous);
      CHECK(value.data<float>()[2] == 5.0f);
      CHECK(b.counts->allocated == 1);
      CHECK(b.counts->freed == 1);
      CHECK(b.budget->occupied_bytes() == 0);
    }
    Strides incoming(12, 2, Strides::allocator_type(b.graph.get()));
    value.allocate_data(
        6 * sizeof(float), 6, std::move(incoming), flags, sizeof(float));
    CHECK(value.data_shared_ptr() != old);
    CHECK(value.strides().get_allocator() == destination_allocator);
    CHECK(value.strides().size() == 12);
    CHECK(value.strides().back() == 2);
    CHECK(value.data_size() == 6);
    CHECK(value.offset() == sizeof(float));
    CHECK_FALSE(value.flags().contiguous);
    CHECK_FALSE(value.flags().row_contiguous);
    CHECK_FALSE(value.flags().col_contiguous);
    CHECK(value.buffer().original_buffer_budget() == b.budget.get());
    auto* raw = static_cast<float*>(value.buffer().raw_ptr());
    for (int i = 0; i != 6; ++i)
      raw[i] = 10.0f + i;
    CHECK(value.data<float>()[0] == 11.0f);
    CHECK(value.data<float>()[4] == 15.0f);
  }
  CHECK(a.counts->allocated == a.counts->freed);
  CHECK(b.counts->allocated == b.counts->freed);
  CHECK(a.budget->occupied_bytes() == 0);
  CHECK(b.budget->occupied_bytes() == 0);
}

TEST_CASE(
    "default original routing layouts price actual binding task and adoption representations") {
  submission::OriginalBufferRoutingLayout layout{};
  REQUIRE(submission::original_buffer_routing_layout(layout));
  CHECK(layout.binding_bytes == sizeof(allocator::OriginalBufferBinding));
  CHECK(layout.binding_alignment == alignof(allocator::OriginalBufferBinding));
  CHECK(layout.worker_guard_bytes == sizeof(submission::BufferWorkerGuard));
  size_t extent = 0;
  REQUIRE(
      submission::GraphQuota::allocation_extent(
          layout.birth_bytes, layout.birth_alignment, extent));
  CHECK(layout.birth_graph_extent == extent);
  const auto tasks = scheduler::task_routing_layout();
  CHECK(tasks.context_bytes == sizeof(scheduler::TaskContext));
  CHECK(tasks.base_bytes == sizeof(scheduler::Task));
  auto body = [] {};
  REQUIRE(scheduler::task_graph_extent<decltype(body)>(extent));
  CHECK(extent >= sizeof(scheduler::TaskNode<decltype(body)>));
  const auto adoption = array::buffer_adoption_layout();
  CHECK(adoption.scratch_owner_bytes == sizeof(allocator::BufferOwner));
  CHECK(
      adoption.incoming_bytes >= sizeof(allocator::Buffer) + sizeof(Deleter*));
}

TEST_CASE(
    "default original Fence refuses fast backing before dispatch and retains complete Buffer") {
  auto producer =
      new_stream(is_available(Device::gpu) ? Device::gpu : Device::cpu);
  auto consumer = new_stream(Device::cpu);
  auto input = array({5.0f, 7.0f, 11.0f});
  auto output = array({0.0f, 0.0f, 0.0f});
  // Warm exactly the ordinary streams and existing device policy, including
  // actual cross-device Fence kernels where that policy selects them.
  eval(add(add(input, input, producer), input, consumer));
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(
      mlx_submission_prepare_runtime(
          &baseline, mlx_stream{&producer}, mlx_stream{&consumer}) == 0);
  // Read the same actual prepared device and retained environment/configuration
  // used by FenceImpl; this is independent of observing our allocation count.
  unsigned expected_allocations = 0;
#ifdef MLX_C_PATCH_TEST_METAL
  if (producer.device == Device::gpu) {
    auto* prepared = metal::prepared_device(producer.device);
    REQUIRE(prepared != nullptr);
    if (prepared->mtl_device()->supportsFamily(MTL::GPUFamilyMetal3)) {
      if (__builtin_available(macOS 15, iOS 18, *)) {
        expected_allocations = env::metal_fast_synch() ? 1 : 0;
      }
    }
  }
#endif
  // A dedicated forced-fast process sets this test requirement as well as
  // MLX_METAL_FAST_SYNCH=1; unsupported hardware cannot earn fast-path credit.
  if (std::getenv("EREDU_REQUIRE_FAST_DEFAULT_FENCE"))
    REQUIRE(expected_allocations == 1);
  if (expected_allocations == 1) {
    BufferFixture capacity_source;
    size_t needed = 0;
    REQUIRE(
        capacity_source.budget->physical_capacity(sizeof(uint32_t), needed));
    REQUIRE(needed > 0);
    for (size_t capacity : {size_t{0}, needed - 1}) {
      BufferFixture refused(capacity);
      DataRole refusal_role(refused.graph.get());
      (void)bind_buffer(refusal_role, refused);
      const auto graph_before = refused.graph->occupied_bytes();
      buffer_refusal([&] { Fence ordinary(producer); }, BufferCause::capacity);
      CHECK(refused.counts->allocated == 0);
      CHECK(refused.counts->freed == 0);
      CHECK(refused.budget->occupied_bytes() == 0);
      CHECK(refused.graph->occupied_bytes() == graph_before);
      CHECK(refusal_role.scope->query().activity == submission::Activity::none);
      CHECK_FALSE(refusal_role.scope->query_records().pending);
    }
  }
  BufferFixture fixture;
  const auto cached = get_cache_memory();
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  auto pending = submission::Record::create<DataCaptureRecord>();
  pending->enter();
  auto* record = pending.release();
  BufferRecordFinish record_guard{record};
  record->reserve_streams(2);
  record->prepare_stream(producer);
  record->prepare_stream(consumer);
  struct Gate {
    std::atomic<bool> open{false};
  };
  auto gate = std::make_shared<Gate>();
  struct ReleaseGate {
    std::shared_ptr<Gate> value;
    ~ReleaseGate() {
      value->open.store(true);
    }
  } gate_guard{gate};
  {
    submission::RecordDispatchGuard dispatch(*record);
    scheduler::enqueue(consumer, [gate] {
      while (!gate->open.load())
        std::this_thread::yield();
    });
    {
      Fence fence(producer);
      fence.update(producer, input, producer.device != consumer.device);
      fence.wait(consumer, input);
      auto* source = input.data<float>();
      auto* destination = output.data<float>();
      // A failed assertion may unwind before the queued task runs. Keep the
      // actual source/output owners with that task, independently of locals.
      scheduler::enqueue(
          consumer,
          [source,
           destination,
           source_owner = input.data_shared_ptr(),
           output_owner = output.data_shared_ptr()] {
            for (size_t i = 0; i < 3; ++i)
              destination[i] = 2 * source[i];
          });
      if (producer.device == Device::gpu)
        gpu::finalize(producer);
    }
    // Fast Metal mode made exactly one direct allocation. CPU/slow Event mode
    // makes none. The controlled consumer still retains the actual Fence owner.
    CHECK(fixture.counts->allocated == expected_allocations);
    CHECK(fixture.counts->freed == 0);
    CHECK(
        (fixture.budget->occupied_bytes() != 0) == (expected_allocations == 1));
    CHECK(get_cache_memory() == cached);
  }
  record->finish(false);
  record_guard.value = nullptr;
  role.scope->seal();
  gate->open.store(true);
  REQUIRE(settle_default_role(*role.scope));
  CHECK_FALSE(role.scope->query().failed);
  CHECK(output.data<float>()[0] == 10.0f);
  CHECK(output.data<float>()[1] == 14.0f);
  CHECK(output.data<float>()[2] == 22.0f);
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.counts->allocated == expected_allocations);
  CHECK(fixture.counts->freed == expected_allocations);
  CHECK(get_cache_memory() == cached);
}

TEST_CASE(
    "default original shared-buffer publication preflights low-level stride growth") {
  // Ordinary matching-capacity aliasing keeps its existing stride storage.
  {
    Shape shape(12, 1);
    shape.back() = 3;
    auto source =
        array::with_allocated_buffer(3 * sizeof(float), shape, float32);
    auto output =
        array::with_allocated_buffer(3 * sizeof(float), shape, float32);
    auto* old_storage = output.strides().data();
    output.copy_shared_buffer(source);
    CHECK(output.strides().data() == old_storage);
    CHECK(output.data_shared_ptr() == source.data_shared_ptr());
    CHECK(output.strides() == source.strides());
  }
  BufferFixture a, b;
  DataRole first(a.graph.get());
  (void)bind_buffer(first, a);
  {
    auto output =
        array::with_allocated_buffer(3 * sizeof(float), Shape{3}, float32);
    output.data<float>()[0] = 17.0f;
    auto old = output.data_shared_ptr();
    const auto old_size = output.data_size();
    const auto old_offset = output.offset();
    const auto old_flags = output.flags();
    const auto old_strides = output.strides();
    first.scope->seal();
    DataRole second(b.graph.get());
    (void)bind_buffer(second, b);
    Shape shape(12, 1);
    shape.back() = 3;
    auto source = array::with_allocated_buffer(
        3 * sizeof(float), std::move(shape), float32);
    source.data<float>()[0] = 2.0f;
    source.data<float>()[1] = 3.0f;
    source.data<float>()[2] = 5.0f;
    const auto physical_before = b.counts->allocated.load();
    {
      FilledDataArena full(a.graph.get());
      // This exercises the public low-level descriptor transport's supplied
      // stride population, not a claim that same-rank model aliases must grow.
      CHECK_THROWS_AS(
          output.copy_shared_buffer(source), submission::GraphQuotaError);
      CHECK(output.data_shared_ptr() == old);
      CHECK(output.data_size() == old_size);
      CHECK(output.offset() == old_offset);
      CHECK(output.strides() == old_strides);
      CHECK(output.flags().contiguous == old_flags.contiguous);
      CHECK(output.flags().row_contiguous == old_flags.row_contiguous);
      CHECK(output.flags().col_contiguous == old_flags.col_contiguous);
      CHECK(output.data<float>()[0] == 17.0f);
      CHECK(b.counts->allocated == physical_before);
    }
    output.copy_shared_buffer(source);
    CHECK(output.data_shared_ptr() == source.data_shared_ptr());
    CHECK(output.strides() == source.strides());
    CHECK(output.data<float>()[2] == 5.0f);
    CHECK(output.buffer().original_buffer_budget() == b.budget.get());
    CHECK(b.counts->allocated == physical_before);
    // Equal populated capacity now succeeds even while the old Graph is full.
    {
      FilledDataArena full(a.graph.get());
      auto* storage = output.strides().data();
      output.copy_shared_buffer(source);
      CHECK(output.strides().data() == storage);
      CHECK(output.data_shared_ptr() == source.data_shared_ptr());
    }
  }
  CHECK(a.counts->allocated == a.counts->freed);
  CHECK(b.counts->allocated == b.counts->freed);
}

// Exercise the actual shared materializer used by CPU/Metal/CUDA binary workers.
// Existing fixture capacities are explicit component storage, never fit facts.
namespace {
array binary_adoption_values(size_t rank, size_t elements) {
  const float values[] = {2.f, 3.f, 5.f, 7.f, 11.f, 13.f};
  Shape shape(rank, 1);
  shape.back() = static_cast<ShapeElem>(elements);
  return array(values, std::move(shape), float32);
}
array binary_adoption_strided(size_t rank) {
  auto value = binary_adoption_values(rank, 3);
  auto storage = binary_adoption_values(rank, 6);
  auto strides = value.strides();
  strides.back() = 2;
  value.copy_shared_buffer(storage, strides, array::Flags{false, false, false}, 6);
  return value;
}
std::pair<array, array> binary_adoption_inputs(BinaryOpType type, size_t rank) {
  switch (type) {
    case BinaryOpType::ScalarScalar:
      return {binary_adoption_values(rank, 1), binary_adoption_values(rank, 1)};
    case BinaryOpType::ScalarVector:
      return {binary_adoption_values(rank, 1), binary_adoption_values(rank, 3)};
    case BinaryOpType::VectorScalar:
      return {binary_adoption_values(rank, 3), binary_adoption_values(rank, 1)};
    case BinaryOpType::VectorVector:
      return {binary_adoption_values(rank, 3), binary_adoption_values(rank, 3)};
    case BinaryOpType::General:
      return {binary_adoption_strided(rank), binary_adoption_strided(rank)};
  }
  throw std::logic_error("invalid binary fixture type");
}
}

TEST_CASE("binary adoption refuses copied strides before calling its allocator") {
  for (auto type : {BinaryOpType::ScalarScalar, BinaryOpType::ScalarVector,
                   BinaryOpType::VectorScalar, BinaryOpType::VectorVector}) {
    auto [a, b] = binary_adoption_inputs(type, 12);
    auto keep_a = a, keep_b = b; // the actual donation predicate must be false
    auto output = binary_adoption_values(12, std::max(a.size(), b.size()));
    const auto old = output.data_shared_ptr();
    const auto old_strides = output.strides();
    REQUIRE(get_binary_op_type(a, b) == type);
    BufferFixture fixture(1 << 20, 4096);
    DataRole role(fixture.graph.get());
    (void)bind_buffer(role, fixture);
    unsigned calls = 0;
    auto allocate = [&](size_t bytes) { ++calls; return allocator::malloc(bytes); };
    {
      FilledDataArena full(fixture.graph.get());
      const auto occupied = fixture.graph->occupied_bytes();
      CHECK_THROWS_AS(set_binary_op_output_data(a, b, output, type, allocate),
                      submission::GraphQuotaError);
      CHECK(calls == 0);
      CHECK(fixture.counts->allocated == 0);
      CHECK(fixture.budget->occupied_bytes() == 0);
      CHECK(fixture.graph->occupied_bytes() == occupied);
      CHECK(output.data_shared_ptr() == old);
      CHECK(output.strides() == old_strides);
      CHECK(output.data<float>()[0] == 2.f);
    }
  }
}

TEST_CASE("binary adoption owns every allocating branch through Data control refusal") {
  for (size_t rank : {size_t{1}, size_t{12}}) {
    for (auto type : {BinaryOpType::ScalarScalar, BinaryOpType::ScalarVector,
                     BinaryOpType::VectorScalar, BinaryOpType::VectorVector,
                     BinaryOpType::General}) {
      auto [a, b] = binary_adoption_inputs(type, rank);
      auto keep_a = a, keep_b = b;
      auto output = binary_adoption_values(rank, std::max(a.size(), b.size()));
      const auto old = output.data_shared_ptr();
      const auto old_strides = output.strides();
      const auto old_size = output.data_size();
      REQUIRE(get_binary_op_type(a, b) == type);
      BufferFixture fixture(1 << 20, 4096);
      DataRole role(fixture.graph.get());
      (void)bind_buffer(role, fixture);
      std::optional<FilledDataArena> full;
      unsigned calls = 0;
      auto allocate = [&](size_t bytes) {
        ++calls;
        auto incoming = allocator::malloc_owned(bytes);
        // Native backing and its real P debit already exist. Refuse the next
        // Data control; the existing owning helper must free this buffer once.
        full.emplace(fixture.graph.get());
        return incoming.release();
      };
      CHECK_THROWS_AS(set_binary_op_output_data(a, b, output, type, allocate),
                      submission::GraphQuotaError);
      full.reset();
      CHECK(calls == 1);
      CHECK(fixture.counts->allocated == 1);
      CHECK(fixture.counts->freed == 1);
      CHECK(fixture.budget->occupied_bytes() == 0);
      CHECK(fixture.graph->occupied_bytes() == 0);
      CHECK(output.data_shared_ptr() == old);
      CHECK(output.strides() == old_strides);
      CHECK(output.data_size() == old_size);
      CHECK(output.offset() == 0);
      CHECK(output.data<float>()[0] == 2.f);
    }
  }
}

TEST_CASE("binary adoption preserves donation and actual P through final Data weak retirement") {
  auto a = binary_adoption_values(1, 3), b = binary_adoption_values(1, 1);
  auto keep_a = a, keep_b = b;
  std::optional<array> first(binary_adoption_values(1, 3));
  std::optional<array> last(binary_adoption_values(1, 3));
  std::weak_ptr<array::Data> weak;
  BufferFixture fixture;
  auto counts = fixture.counts;
  size_t charge = 0;
  REQUIRE(fixture.budget->physical_capacity(3 * sizeof(float), charge));
  unsigned calls = 0;
  {
    DataRole role(fixture.graph.get());
    (void)bind_buffer(role, fixture);
    auto pending = submission::Record::create<DataCaptureRecord>();
    pending->enter();
    auto* record = pending.release();
    BufferRecordFinish finish{record};
    record->begin_primitive();
    auto allocate = [&](size_t bytes) { ++calls; return allocator::malloc(bytes); };
    REQUIRE(get_binary_op_type(a, b) == BinaryOpType::VectorScalar);
    set_binary_op_output_data(a, b, *first, BinaryOpType::VectorScalar, allocate);
    REQUIRE(first->buffer().original_buffer_budget() == fixture.budget.get());
    CHECK(first->buffer().original_allocation_capacity() == charge);
    CHECK(first->buffer().original_allocation_generation() != 0);
    CHECK(first->data_shared_ptr()->original_input == nullptr);
    weak = first->data_shared_ptr();
    const auto* identity = first->data_shared_ptr().get();
    first->data<float>()[0] = 17.f;
    first->data<float>()[1] = 19.f;
    first->data<float>()[2] = 23.f;
    // Successful primitive output retention uses PhysicalPin. The temporary
    // capture remains a real alias until the next primitive boundary.
    record->retain_output(first->data_shared_ptr());
    record->end_primitive();
    record->begin_primitive();
    REQUIRE(first->data_shared_ptr()->recovery_pins.load() == 1);
    REQUIRE(first->is_donatable());
    set_binary_op_output_data(*first, b, *last, BinaryOpType::VectorScalar, allocate);
    CHECK(calls == 1);
    CHECK(last->data_shared_ptr().get() == identity);
    CHECK(fixture.budget->occupied_bytes() == charge);
    record->end_primitive();
    record->finish(false); finish.value = nullptr;
    first.reset();
    role.scope->seal();
    REQUIRE(settle_default_role(*role.scope));
    CHECK(last->data<float>()[0] == 17.f);
    CHECK(last->data<float>()[2] == 23.f);
  }
  fixture.budget.reset(); fixture.graph.reset();
  CHECK(counts->budget_retired == 0);
  CHECK(counts->graph_retired == 0);
  last.reset();
  CHECK(weak.expired());
  CHECK(counts->allocated == 1); CHECK(counts->freed == 1);
  CHECK(counts->budget_retired == 1);
  CHECK(counts->graph_retired == 0); // actual shared control survives its Data
  weak.reset();
  CHECK(counts->graph_retired == 1);
}

namespace {
enum class SharedAdoption { Unary, UnaryGeneral, ScalarSelect, VectorSelect, GeneralSelect };
void shared_adoption(SharedAdoption kind, const array& a, const array& b,
                     const array& c, array& output,
                     const std::function<allocator::Buffer(size_t)>& allocate) {
  if (kind == SharedAdoption::Unary || kind == SharedAdoption::UnaryGeneral)
    set_unary_output_data(b, output, allocate);
  else
    set_ternary_op_output_data(a, b, c, output, get_ternary_op_type(a, b, c), allocate);
}
}

TEST_CASE("shared unary and Select adoption preserve every allocated failure prefix") {
  for (auto kind : {SharedAdoption::Unary, SharedAdoption::UnaryGeneral,
                    SharedAdoption::ScalarSelect, SharedAdoption::VectorSelect,
                    SharedAdoption::GeneralSelect}) {
    for (bool before_allocate : {true, false}) {
      const bool scalar = kind == SharedAdoption::ScalarSelect;
      const bool general = kind == SharedAdoption::UnaryGeneral ||
                           kind == SharedAdoption::GeneralSelect;
      if (before_allocate && general) continue; // these branches copy no strides
      const size_t count = scalar ? 1 : 3;
      const bool conditions[] = {true, false, true};
      Shape shape(12, 1); shape.back() = static_cast<ShapeElem>(count);
      auto a = array(conditions, shape, bool_);
      auto b = general ? binary_adoption_strided(12) : binary_adoption_values(12, count);
      auto c = general ? binary_adoption_strided(12) : binary_adoption_values(12, count);
      auto keep_b = b, keep_c = c;
      auto output = binary_adoption_values(12, count);
      const auto old = output.data_shared_ptr();
      const auto strides = output.strides();
      if (kind == SharedAdoption::ScalarSelect)
        REQUIRE(get_ternary_op_type(a, b, c) == TernaryOpType::ScalarScalarScalar);
      if (kind == SharedAdoption::VectorSelect)
        REQUIRE(get_ternary_op_type(a, b, c) == TernaryOpType::VectorVectorVector);
      if (kind == SharedAdoption::GeneralSelect)
        REQUIRE(get_ternary_op_type(a, b, c) == TernaryOpType::General);
      BufferFixture fixture(1 << 20, 4096);
      DataRole role(fixture.graph.get()); (void)bind_buffer(role, fixture);
      std::optional<FilledDataArena> full;
      if (before_allocate) full.emplace(fixture.graph.get());
      unsigned calls = 0;
      auto allocate = [&](size_t bytes) {
        ++calls;
        auto incoming = allocator::malloc_owned(bytes);
        full.emplace(fixture.graph.get());
        return incoming.release();
      };
      CHECK_THROWS_AS(shared_adoption(kind, a, b, c, output, allocate), submission::GraphQuotaError);
      CHECK(calls == (before_allocate ? 0 : 1));
      CHECK(fixture.counts->allocated == calls);
      CHECK(fixture.counts->freed == calls);
      CHECK(fixture.budget->occupied_bytes() == 0);
      CHECK(output.data_shared_ptr() == old);
      CHECK(output.strides() == strides);
      CHECK(output.data<float>()[0] == 2.f);
      full.reset();
      CHECK(fixture.graph->occupied_bytes() == 0);
    }
  }
}

TEST_CASE("shared unary and Select donation retain the original donor ordering") {
  unsigned calls = 0;
  auto allocate = [&](size_t) -> allocator::Buffer {
    ++calls; throw std::logic_error("donation must not allocate");
  };
  {
    auto input = binary_adoption_values(1, 3);
    auto output = binary_adoption_values(1, 3);
    const auto* identity = input.data_shared_ptr().get();
    REQUIRE(input.is_donatable());
    set_unary_output_data(input, output, allocate);
    CHECK(output.data_shared_ptr().get() == identity);
    CHECK(output.data<float>()[2] == 5.f);
  }
  {
    auto condition = array({true, false, true}, bool_);
    auto b = binary_adoption_values(1, 3), c = binary_adoption_values(1, 3);
    auto output = binary_adoption_values(1, 3);
    const auto* first = b.data_shared_ptr().get();
    const auto* second = c.data_shared_ptr().get();
    REQUIRE(b.is_donatable()); REQUIRE(c.is_donatable());
    REQUIRE(get_ternary_op_type(condition, b, c) == TernaryOpType::VectorVectorVector);
    set_ternary_op_output_data(condition, b, c, output,
                              TernaryOpType::VectorVectorVector, allocate);
    CHECK(output.data_shared_ptr().get() == first);
    CHECK(c.data_shared_ptr().get() == second);
    CHECK(output.data<float>()[2] == 5.f);
  }
  CHECK(calls == 0);
}


TEST_CASE("resident view axes preserve ordinary values and duplicate ordering") {
  auto stream = default_stream(Device::cpu);
  auto input = array({3.0f, -7.0f, 11.0f});
  auto expanded = expand_dims(input, std::vector<int>{-1, 0}, stream);
  CHECK(expanded.shape() == Shape({1, 3, 1}));
  auto restored = squeeze(expanded, std::vector<int>{-1, 0}, stream);
  eval(restored);
  REQUIRE(restored.shape() == Shape({3}));
  CHECK(restored.data<float>()[0] == 3.0f);
  CHECK(restored.data<float>()[1] == -7.0f);
  CHECK(restored.data<float>()[2] == 11.0f);
  CHECK_THROWS_WITH_AS(squeeze(expanded, std::vector<int>{0, -3}, stream),
      "[squeeze] Received duplicate axes.", std::invalid_argument);
  CHECK_THROWS_WITH_AS(expand_dims(input, std::vector<int>{0, -3}, stream),
      "[expand_dims] Received duplicate axes.", std::invalid_argument);
  CHECK_THROWS_AS(squeeze(input, 0, stream), std::invalid_argument);
}

TEST_CASE("resident view axes refuse before result replacement and retire retained backing") {
  auto stream = default_stream(Device::cpu);
  std::vector<int> axes(11);
  std::iota(axes.begin(), axes.end(), 0);
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  auto input = array::with_allocated_buffer(2 * sizeof(float), Shape{2}, float32);
  input.data<float>()[0] = 5.0f;
  input.data<float>()[1] = -9.0f;
  auto output = input;
  const auto id = output.id();
  const auto allocated = fixture.counts->allocated.load();
  const auto baseline = fixture.graph->occupied_bytes();
  {
    FilledDataArena full(fixture.graph.get());
    CHECK_THROWS_AS(output = expand_dims(input, axes, stream), submission::GraphQuotaError);
    CHECK(output.id() == id);
    CHECK(output.shape() == Shape({2}));
    CHECK(output.data<float>()[0] == 5.0f);
    CHECK(fixture.counts->allocated == allocated);
  }
  {
    auto expanded = expand_dims(input, axes, stream);
    CHECK(expanded.ndim() == 12);
    CHECK(expanded.shape(-1) == 2);
    CHECK(fixture.graph->occupied_bytes() > baseline);
    // The actual primitive retains the axis backing until its last descriptor
    // dies. No worker or synthetic registry reset is needed for this owner.
    auto retained = expanded;
    expanded = input;
    CHECK(fixture.graph->occupied_bytes() > baseline);
  }
  CHECK(fixture.graph->occupied_bytes() == baseline);
  CHECK(fixture.counts->allocated == allocated);
}


TEST_CASE("resident reduction axes preserve numerical and ordinary axis semantics") {
  auto check = [](Stream stream) {
    auto input = array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, Shape{2, 3});
    auto transposed = transpose(input, std::vector<int>{1, 0}, stream);
    auto sums = sum(transposed, 1, false, stream);
    auto means = mean(transposed, 0, false, stream);
    auto maxima = max(transposed, 0, false, stream);
    auto products = prod(transposed, 1, false, stream);
    auto total = sum(transposed, std::vector<int>{-1, 0}, false, stream);
    auto scalar_softmax = softmax(input, 1, true, stream);
    auto vector_softmax = softmax(input, std::vector<int>{1}, true, stream);
    eval(sums, means, maxima, products, total, scalar_softmax, vector_softmax);
    CHECK(sums.shape() == Shape({3}));
    CHECK(sums.data<float>()[0] == 5.f);
    CHECK(sums.data<float>()[2] == 9.f);
    CHECK(means.data<float>()[0] == 2.f);
    CHECK(means.data<float>()[1] == 5.f);
    CHECK(maxima.data<float>()[0] == 3.f);
    CHECK(maxima.data<float>()[1] == 6.f);
    CHECK(products.data<float>()[0] == 4.f);
    CHECK(products.data<float>()[2] == 18.f);
    CHECK(total.item<float>() == 21.f);
    for (size_t i = 0; i < scalar_softmax.size(); ++i)
      CHECK(scalar_softmax.data<float>()[i] == doctest::Approx(vector_softmax.data<float>()[i]));
    auto flags = array({true, false}, bool_);
    const std::vector<int> empty;
    CHECK(all(flags, empty, false, stream).id() == flags.id());
    CHECK(any(flags, empty, false, stream).id() == flags.id());
    CHECK(sum(input, empty, false, stream).id() == input.id());
    CHECK_THROWS_AS(sum(input, std::vector<int>{0, 0, 5}, false, stream), std::out_of_range);
    CHECK_THROWS_WITH_AS(sum(input, std::vector<int>{0, -2}, false, stream),
        "Duplicate axes detected in reduction.", std::invalid_argument);
    CHECK_THROWS_AS(transpose(input, std::vector<int>{0, 0}, stream), std::invalid_argument);
  };
  check(default_stream(Device::cpu));
#if defined(MLX_C_PATCH_TEST_METAL)
  check(default_stream(Device::gpu));
#endif
}

TEST_CASE("resident reduction and transpose axes refuse before replacing retained output") {
  auto stream = default_stream(Device::cpu);
  Shape shape(12, 1); shape.back() = 2;
  const float values[] = {5.f, -9.f};
  auto input = array(values, shape, float32);
  std::vector<int> axes(12);
  std::iota(axes.begin(), axes.end(), 0);
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  auto output = input;
  const auto baseline = fixture.graph->occupied_bytes();
  const auto allocated = fixture.counts->allocated.load();
  {
    FilledDataArena full(fixture.graph.get());
    CHECK_THROWS_AS(output = sum(input, axes, true, stream), submission::GraphQuotaError);
    CHECK(output.id() == input.id());
    CHECK_THROWS_AS(output = transpose(input, axes, stream), submission::GraphQuotaError);
    CHECK(output.id() == input.id());
    CHECK(output.data<float>()[1] == -9.f);
    CHECK(fixture.counts->allocated == allocated);
  }
  {
    auto reduced = sum(input, axes, true, stream);
    auto transposed = transpose(input, axes, stream);
    CHECK(reduced.ndim() == 12);
    CHECK(reduced.size() == 1);
    CHECK(transposed.shape() == input.shape());
    CHECK(fixture.graph->occupied_bytes() > baseline);
    auto retained_reduce = reduced;
    auto retained_transpose = transposed;
    reduced = input; transposed = input;
    CHECK(fixture.graph->occupied_bytes() > baseline);
  }
  CHECK(fixture.graph->occupied_bytes() == baseline);
  CHECK(fixture.counts->allocated == allocated);
}

TEST_CASE("resident reduction planner owns high rank scratch and returned geometry") {
  // A valid broadcast view forces the non-row-contiguous planner. No kernel or
  // huge backing allocation is needed to exercise its real pair storage.
  auto input = array(Shape(12, 2), float32, nullptr, {});
  input.allocate_data(sizeof(float), 1, Strides(12, 0),
                      array::Flags{false, false, false});
  input.data<float>()[0] = 7.f;
  std::vector<int> axes(12);
  std::iota(axes.begin(), axes.end(), 0);
  const std::vector<int> remove{0};
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  const auto baseline = fixture.graph->occupied_bytes();
  {
    FilledDataArena full(fixture.graph.get());
    CHECK_THROWS_AS(get_reduction_plan(input, axes), submission::GraphQuotaError);
    CHECK_THROWS_AS(shapes_without_reduction_axes(input, remove), submission::GraphQuotaError);
  }
  {
    auto plan = get_reduction_plan(input, axes);
    REQUIRE(plan.shape.size() == 1);
    REQUIRE(plan.strides.size() == 1);
    CHECK(plan.type == GeneralReduce);
    CHECK(plan.shape[0] == 4096);
    CHECK(plan.strides[0] == 0);
    // Pair scratch has retired; the collapsed plan is inline.
    CHECK(fixture.graph->occupied_bytes() == baseline);
    auto geometry = shapes_without_reduction_axes(input, remove);
    CHECK(geometry.first.size() == 11);
    CHECK(geometry.second.size() == 11);
    CHECK(fixture.graph->occupied_bytes() > baseline);
    auto retained = std::move(geometry);
    CHECK(retained.first.back() == 2);
    CHECK(retained.second.back() == 0);
  }
  CHECK(fixture.graph->occupied_bytes() == baseline);
  CHECK(fixture.counts->allocated == 0);
}


#if defined(MLX_C_PATCH_TEST_METAL)
#include "mlx/backend/metal/device.h"

TEST_CASE("Metal library cache borrows builders through refusal reuse and dispatch") {
  auto stream = default_stream(Device::gpu);
  auto& device = metal::device(stream.device);
  const std::string name = "eredu_borrowed_builder_lifetime";
  device.clear_library(name);
  struct Refusal final : std::exception {};
  struct Builder {
    unsigned& calls;
    bool& refuse;
    std::unique_ptr<unsigned> capture{std::make_unique<unsigned>(1)};
    Builder(unsigned& calls, bool& refuse) : calls(calls), refuse(refuse) {}
    Builder(const Builder&) = delete;
    Builder(Builder&&) = delete;
    std::string operator()() const {
      ++calls;
      if (refuse) throw Refusal{};
      REQUIRE(*capture == 1);
      return "#include <metal_stdlib>\nusing namespace metal;\n"
          "kernel void eredu_borrowed_increment(device float* out [[buffer(0)]], "
          "uint i [[thread_position_in_grid]]) { out[i] += 1.0f; }";
    }
  };
  unsigned calls = 0;
  bool refuse = true;
  MTL::Library* library = nullptr;
  {
    Builder builder(calls, refuse);
    CHECK_THROWS_AS(device.get_library_borrowed(name, builder), Refusal);
    CHECK(calls == 1);
    refuse = false;
    library = device.get_library_borrowed(name, builder);
    REQUIRE(library != nullptr);
    CHECK(calls == 2);
    // A cache hit must neither invoke nor copy the now-refusing builder.
    refuse = true;
    CHECK(device.get_library_borrowed(name, builder) == library);
    CHECK(calls == 2);
  }
  // The actual library owns its compiled code after the noncopyable builder
  // and all its captured state have died. Retain ordinary std::function API.
  const std::function<std::string(void)> old_api = [&]() -> std::string {
    throw Refusal{};
  };
  CHECK(device.get_library(name, old_api) == library);
  auto* kernel = device.get_kernel("eredu_borrowed_increment", library);
  REQUIRE(kernel != nullptr);
  auto values = array({3.f, -2.f, 7.f});
  auto& encoder = metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_output_array(values, 0);
  encoder.dispatch_threads(MTL::Size(3, 1, 1), MTL::Size(3, 1, 1));
  synchronize(stream);
  CHECK(values.data<float>()[0] == 4.f);
  CHECK(values.data<float>()[1] == -1.f);
  CHECK(values.data<float>()[2] == 8.f);
  device.clear_library(name);
}
#endif
