#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/cpu/retention.h"
#include "mlx/transforms_impl.h"

namespace {
template <class F>
void with_cpu_retention_record(Stream stream, F&& body) {
  auto pending = submission::Record::create<DataCaptureRecord>();
  pending->enter();
  auto* record = pending.release();
  BufferRecordFinish finish{record};
  record->reserve_streams(1);
  record->prepare_stream(stream);
  {
    submission::RecordDispatchGuard dispatch(*record);
    record->begin_primitive();
    body();
    record->end_primitive();
  }
  record->finish(false);
  finish.value = nullptr;
}

struct CpuRetentionState {
  std::atomic<unsigned> entered{0}, completed{0};
  std::weak_ptr<array::Data> temporary;
  std::shared_ptr<DefaultGate> gate{std::make_shared<DefaultGate>()};
};
struct CpuRetentionProbe : Primitive {
  enum class Mode { ordinary, temporary_refusal, enqueue_refusal };
  std::shared_ptr<CpuRetentionState> state;
  Mode mode;
  CpuRetentionProbe(
      Stream stream,
      std::shared_ptr<CpuRetentionState> state,
      Mode mode = Mode::ordinary)
      : Primitive(stream), state(std::move(state)), mode(mode) {}
  const char* name() const override {
    return "CPU retention native prefix";
  }
  void eval_cpu(const ArrayVector&, ArrayVector& outputs) override {
    ++state->entered;
    outputs[0].allocate_data(sizeof(float));
    auto temporary =
        array::with_allocated_buffer(sizeof(float), Shape{1}, float32);
    state->temporary = temporary.data_shared_ptr();
    auto* temporary_data = temporary.data<float>();
    auto* output = outputs[0].data<float>();
    auto& encoder = cpu::get_command_encoder(stream());
    encoder.dispatch([state = state, temporary_data, output] {
      while (!state->gate->open.load(std::memory_order_acquire))
        std::this_thread::yield();
      *temporary_data = 37.f;
      *output = *temporary_data + 5.f;
      ++state->completed;
    });
    if (mode == Mode::temporary_refusal) {
      FilledDataArena fill(submission::current_graph_quota());
      encoder.add_temporary(std::move(temporary));
      FAIL("actual exhausted temporary header must refuse");
    } else {
      encoder.add_temporary(std::move(temporary));
    }
    if (mode == Mode::enqueue_refusal) {
      scheduler::CpuStreamToken token;
      REQUIRE(
          scheduler::prepared_cpu_stream(stream(), token) ==
          submission::NativeControlFailure::none);
      scheduler::block_cpu_stream(token);
    }
  }
  void eval_gpu(const ArrayVector&, ArrayVector&) override {
    std::abort();
  }
};
struct CpuDonationProbe : Primitive {
  std::shared_ptr<std::atomic<bool>> donated;
  CpuDonationProbe(Stream stream, std::shared_ptr<std::atomic<bool>> donated)
      : Primitive(stream), donated(std::move(donated)) {}
  const char* name() const override {
    return "CPU retention donation order";
  }
  void eval_cpu(const ArrayVector& inputs, ArrayVector& outputs) override {
    const bool donate = inputs[0].is_donatable();
    *donated = donate;
    if (donate)
      outputs[0].copy_shared_buffer(inputs[0]);
    else
      outputs[0].allocate_data(sizeof(float));
    auto* out = outputs[0].data<float>();
    auto* in = inputs[0].data<float>();
    cpu::get_command_encoder(stream()).dispatch(
        [out, in] { *out = *in + 3.f; });
  }
  void eval_gpu(const ArrayVector&, ArrayVector&) override {
    std::abort();
  }
};
} // namespace

TEST_CASE(
    "CPU retention layouts describe exact slots headers and fixed transports") {
  cpu::RetentionLayout layout;
  REQUIRE(cpu::CommandEncoder::retention_layout(0, 0, layout));
  CHECK(layout.data_slots.bytes == 0);
  CHECK(layout.data_slots.graph_extent == 0);
  CHECK(
      layout.cleanup_task.bytes ==
      sizeof(scheduler::TaskNode<cpu::detail::Cleanup>));
  CHECK(layout.temporary_batch.bytes == sizeof(cpu::detail::TemporaryBatch));
  CHECK(layout.encoder_bytes == sizeof(cpu::CommandEncoder));
  REQUIRE(cpu::CommandEncoder::retention_layout(3, 2, layout));
  CHECK(layout.maximum_data_slots == 5);
  CHECK(layout.data_slots.bytes == 5 * sizeof(std::shared_ptr<array::Data>));
  CHECK(layout.prepared_cleanup_bytes == sizeof(cpu::detail::PreparedCleanup));
  const auto saved = layout.data_slots.bytes;
  CHECK_FALSE(cpu::CommandEncoder::retention_layout(SIZE_MAX, 1, layout));
  CHECK_FALSE(cpu::CommandEncoder::retention_layout(SIZE_MAX, 0, layout));
  CHECK(layout.data_slots.bytes == saved);
}

TEST_CASE(
    "CPU retention reserves without references then deduplicates actual Data identity") {
  array first({1.f, 3.f}), second({5.f, 7.f});
  ArrayVector inputs{first, first, second}, siblings{second, first};
  const auto first_owners = first.data_shared_ptr().use_count();
  const auto second_owners = second.data_shared_ptr().use_count();
  BufferFixture fixture;
  {
    cpu::detail::DataRetention retained(fixture.graph.get(), 5);
    CHECK(retained.size() == 0);
    CHECK(first.data_shared_ptr().use_count() == first_owners);
    CHECK(second.data_shared_ptr().use_count() == second_owners);
    retained.fill(inputs, siblings, first.data_shared_ptr().get());
    REQUIRE(retained.size() == 1);
    CHECK(retained.data()[0].get() == second.data_shared_ptr().get());
    CHECK(first.data_shared_ptr().use_count() == first_owners);
    CHECK(second.data_shared_ptr().use_count() == second_owners + 1);
    CHECK_THROWS_AS(
        retained.fill(inputs, siblings, nullptr), submission::GraphQuotaError);
    CHECK(retained.size() == 1);
  }
  CHECK(second.data_shared_ptr().use_count() == second_owners);
  CHECK(fixture.graph->occupied_bytes() == 0);
  {
    cpu::detail::DataRetention empty(fixture.graph.get(), 0);
    empty.fill({}, {}, nullptr);
    CHECK(empty.data() == nullptr);
    CHECK(fixture.graph->occupied_bytes() == 0);
  }
}

TEST_CASE(
    "CPU temporary batches preserve old vector storage duplicates and header births") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  array first({2.f}), second({7.f});
  BufferFixture a, b;
  auto a_count = a.counts, b_count = b.counts;
  cpu::CommandEncoder encoder(stream);
  std::optional<ArrayVector> source;
  {
    Scope scope(a.graph.get());
    source.emplace(ArrayVector{first, second, first});
  }
  const auto* original = source->data();
  {
    Scope scope(b.graph.get());
    encoder.add_temporaries(std::move(*source));
    encoder.add_temporary(second);
  }
  source.reset();
  auto owned = encoder.take_temporaries();
  REQUIRE(owned.size() == 4);
  REQUIRE(owned.first());
  REQUIRE(owned.first()->multiple.has_value());
  CHECK(owned.first()->multiple->data() == original);
  CHECK(owned.first()->multiple->get_allocator().resource() == a.graph.get());
  CHECK(owned.first()->birth.get() == b.graph.get());
  CHECK((*owned.first()->multiple)[0].id() == first.id());
  CHECK((*owned.first()->multiple)[2].id() == first.id());
  REQUIRE(owned.first()->next);
  CHECK(owned.first()->next->single->id() == second.id());
  CHECK(encoder.take_temporaries().size() == 0);
  a.graph.reset();
  b.graph.reset();
  CHECK(a_count->graph_retired == 0);
  CHECK(b_count->graph_retired == 0);
  owned.clear();
  CHECK(a_count->graph_retired == 1);
  CHECK(b_count->graph_retired == 1);
  {
    ArrayVector ordinary{first, first};
    const auto* address = ordinary.data();
    encoder.add_temporaries(ordinary);
    auto copy = encoder.take_temporaries();
    CHECK(ordinary.data() == address);
    CHECK(copy.first()->multiple->data() != address);
    CHECK(copy.first()->multiple->size() == 2);
    CHECK(copy.first()->birth.get() == nullptr);
  }
}

TEST_CASE(
    "CPU cleanup refusal precedes primitive and physical output allocation") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto state = std::make_shared<CpuRetentionState>();
  OpenDefaultGate release{state->gate};
  array output(
      Shape{1},
      float32,
      std::make_shared<CpuRetentionProbe>(stream, state),
      {});
  BufferFixture fixture(1 << 20, 128);
  cpu::RetentionLayout layout;
  REQUIRE(cpu::CommandEncoder::retention_layout(0, 0, layout));
  REQUIRE(layout.cleanup_task.graph_extent > fixture.graph->capacity());
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  const auto before = scheduler::cpu_stream_progress(stream);
  REQUIRE(before);
  bool refused = false;
  try {
    with_cpu_retention_record(stream, [&] { cpu::eval(output); });
  } catch (const submission::GraphQuotaError& error) {
    refused = true;
    CHECK(error.cause() == submission::GraphFailure::exhausted);
  }
  CHECK(refused);
  CHECK(state->entered == 0);
  CHECK(fixture.counts->allocated == 0);
  CHECK(scheduler::cpu_stream_progress(stream)->accepted == before->accepted);
  role.scope->seal();
  REQUIRE(settle_default_role(*role.scope));
  CHECK(fixture.graph->occupied_bytes() == 0);
}

TEST_CASE(
    "CPU temporary and cleanup enqueue refusals retain real accepted prefix Data") {
  for (auto mode :
       {CpuRetentionProbe::Mode::temporary_refusal,
        CpuRetentionProbe::Mode::enqueue_refusal}) {
    auto stream = new_stream(Device::cpu);
    prepare_default_streams(stream, stream);
    auto state = std::make_shared<CpuRetentionState>();
    OpenDefaultGate release{state->gate};
    BufferFixture prior_fixture;
    auto prior_counts = prior_fixture.counts;
    std::weak_ptr<array::Data> prior_data;
    {
      Scope prior_scope(prior_fixture.graph.get());
      auto prior =
          array::with_allocated_buffer(sizeof(float), Shape{1}, float32);
      prior_data = prior.data_shared_ptr();
      cpu::get_command_encoder(stream).add_temporary(std::move(prior));
    }
    prior_fixture.graph.reset();
    std::optional<array> output;
    output.emplace(
        Shape{1},
        float32,
        std::make_shared<CpuRetentionProbe>(stream, state, mode),
        ArrayVector{});
    BufferFixture fixture;
    DataRole role(fixture.graph.get());
    (void)bind_buffer(role, fixture);
    const auto before = scheduler::cpu_stream_progress(stream);
    REQUIRE(before);
    bool refused = false;
    try {
      with_cpu_retention_record(stream, [&] { cpu::eval(*output); });
    } catch (const submission::GraphQuotaError& error) {
      refused = true;
      CHECK(mode == CpuRetentionProbe::Mode::temporary_refusal);
      CHECK(error.cause() == submission::GraphFailure::exhausted);
    } catch (const submission::NativeControlError& error) {
      refused = true;
      CHECK(mode == CpuRetentionProbe::Mode::enqueue_refusal);
      CHECK(
          error.failure() == submission::NativeControlFailure::stopped_worker);
    }
    CHECK(refused);
    CHECK(state->entered == 1);
    CHECK(state->completed == 0);
    CHECK(
        scheduler::cpu_stream_progress(stream)->accepted ==
        before->accepted + 1);
    CHECK_FALSE(state->temporary.expired());
    CHECK_FALSE(prior_data.expired());
    output.reset();
    role.scope->seal();
    CHECK_FALSE(state->temporary.expired());
    CHECK(fixture.counts->freed == 0);
    state->gate->open.store(true, std::memory_order_release);
    REQUIRE(settle_default_role(*role.scope));
    CHECK(state->completed == 1);
    CHECK_FALSE(prior_data.expired());
    CHECK(prior_counts->graph_retired == 0);
    // The permanently blocked stream retains restored batches until its actual
    // encoder is destroyed. Terminal Record retirement alone does not refund
    // it.
    CHECK(
        state->temporary.expired() ==
        (mode == CpuRetentionProbe::Mode::temporary_refusal));
    REQUIRE(cpu::get_command_encoders().erase(stream.index) == 1);
    CHECK(state->temporary.expired());
    CHECK(prior_data.expired());
    prior_data.reset();
    CHECK(prior_counts->graph_retired == 1);
    state->temporary.reset(); // final weak Data control has its own Graph birth
    CHECK(fixture.counts->allocated == 2);
    CHECK(fixture.counts->freed == 2);
    CHECK(fixture.budget->occupied_bytes() == 0);
    CHECK(fixture.graph->occupied_bytes() == 0);
  }
}

TEST_CASE(
    "CPU prepared cleanup counts the actual tenth dispatch and retires after seal") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto gate = std::make_shared<DefaultGate>();
  OpenDefaultGate release{gate};
  auto calls = std::make_shared<std::atomic<unsigned>>(0);
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  cpu::CommandEncoder encoder(
      stream); // genuine fresh per-encoder dispatch count
  const auto before_counted = scheduler::n_active_tasks();
  const auto before = scheduler::cpu_stream_progress(stream);
  REQUIRE(before);
  with_cpu_retention_record(stream, [&] {
    auto cleanup = encoder.prepare_cleanup(0);
    encoder.dispatch([gate] {
      while (!gate->open.load(std::memory_order_acquire))
        std::this_thread::yield();
    });
    for (unsigned i = 0; i < 8; ++i)
      encoder.dispatch([calls] { ++*calls; });
    CHECK(scheduler::n_active_tasks() == before_counted);
    encoder.dispatch_cleanup(std::move(cleanup));
    CHECK(scheduler::n_active_tasks() == before_counted + 1);
  });
  role.scope->seal();
  CHECK(
      scheduler::cpu_stream_progress(stream)->accepted ==
      before->accepted + 10);
  CHECK(*calls == 0);
  gate->open.store(true, std::memory_order_release);
  REQUIRE(settle_default_role(*role.scope));
  CHECK(*calls == 8);
  CHECK(scheduler::n_active_tasks() == before_counted);
  CHECK(fixture.graph->occupied_bytes() == 0);
}

TEST_CASE(
    "CPU refused cleanup restores mixed prior owners for a later successful cleanup") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto gate = std::make_shared<DefaultGate>();
  OpenDefaultGate release{gate};
  auto value = std::make_shared<std::atomic<float>>(0.f);
  auto& encoder = cpu::get_command_encoder(stream);
  std::weak_ptr<array::Data> ordinary_data, original_data;
  {
    auto ordinary = array({5.f});
    ordinary_data = ordinary.data_shared_ptr();
    encoder.add_temporary(std::move(ordinary));
  }
  BufferFixture a, b, c;
  std::optional<cpu::detail::PreparedCleanup> first;
  DataRole a_role(a.graph.get());
  (void)bind_buffer(a_role, a);
  with_cpu_retention_record(stream, [&] {
    first.emplace(encoder.prepare_cleanup(0));
    auto temporary =
        array::with_allocated_buffer(sizeof(float), Shape{1}, float32);
    *temporary.data<float>() = 19.f;
    original_data = temporary.data_shared_ptr();
    auto* data = temporary.data<float>();
    encoder.add_temporary(std::move(temporary));
    encoder.dispatch([gate, value, data] {
      while (!gate->open.load(std::memory_order_acquire))
        std::this_thread::yield();
      *value = *data + 4.f;
    });
  });
  a_role.scope->seal();
  DataRole b_role(b.graph.get());
  (void)bind_buffer(b_role, b);
  const auto before = scheduler::cpu_stream_progress(stream);
  REQUIRE(before);
  bool refused = false;
  try {
    with_cpu_retention_record(
        stream, [&] { encoder.dispatch_cleanup(std::move(*first)); });
  } catch (const submission::NativeControlError& error) {
    refused = true;
    CHECK(error.failure() == submission::NativeControlFailure::foreign_domain);
  }
  first.reset();
  CHECK(refused);
  CHECK(scheduler::cpu_stream_progress(stream)->accepted == before->accepted);
  CHECK_FALSE(ordinary_data.expired());
  CHECK_FALSE(original_data.expired());
  b_role.scope->seal();
  DataRole c_role(c.graph.get());
  (void)bind_buffer(c_role, c);
  with_cpu_retention_record(
      stream, [&] { encoder.dispatch_cleanup(encoder.prepare_cleanup(0)); });
  c_role.scope->seal();
  CHECK_FALSE(original_data.expired());
  gate->open.store(true, std::memory_order_release);
  REQUIRE(settle_default_role(*a_role.scope));
  REQUIRE(settle_default_role(*b_role.scope));
  REQUIRE(settle_default_role(*c_role.scope));
  CHECK(*value == 23.f);
  CHECK(ordinary_data.expired());
  CHECK(original_data.expired());
  ordinary_data.reset();
  original_data.reset();
  CHECK(a.graph->occupied_bytes() == 0);
  CHECK(b.graph->occupied_bytes() == 0);
  CHECK(c.graph->occupied_bytes() == 0);
  CHECK(a.budget->occupied_bytes() == 0);
}

TEST_CASE(
    "CPU cleanup keeps donation and tracer input ownership at primitive time") {
  for (bool tracer : {false, true}) {
    auto stream = new_stream(Device::cpu);
    prepare_default_streams(stream, stream);
    auto donated = std::make_shared<std::atomic<bool>>(false);
    std::optional<detail::InTracing> tracing;
    if (tracer)
      tracing.emplace(); // actual native tracing state, prepared cold
    std::optional<array> output;
    output.emplace(
        Shape{1},
        float32,
        std::make_shared<CpuDonationProbe>(stream, donated),
        ArrayVector{array({11.f})});
    output->set_tracer(tracer);
    REQUIRE(output->inputs()[0].is_donatable());
    BufferFixture fixture;
    DataRole role(fixture.graph.get());
    (void)bind_buffer(role, fixture);
    with_cpu_retention_record(stream, [&] { cpu::eval(*output); });
    role.scope->seal();
    REQUIRE(settle_default_role(*role.scope));
    CHECK(*donated == !tracer);
    CHECK(output->data<float>()[0] == 14.f);
    output.reset();
    CHECK(fixture.graph->occupied_bytes() == 0);
    CHECK(fixture.budget->occupied_bytes() == 0);
  }
}

TEST_CASE(
    "CPU cleanup preserves noncontiguous matmul temporaries and sibling outputs") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto raw = reshape(
      array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f}),
      {2, 6});
  auto left = slice(raw, {0, 0}, {2, 6}, {1, 2}, stream);
  auto right = reshape(array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}), {3, 2});
  auto divisor = array(4.f);
  eval(left, right);
  auto expected = matmul(left, right, stream);
  auto expected_pair = divmod(left, divisor, stream);
  eval(expected, expected_pair[0], expected_pair[1]);
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    auto result = matmul(left, right, stream);
    auto pair = divmod(left, divisor, stream);
    Completion completion;
    REQUIRE(
        async_eval_with_completion_scoped_on_stream(
            ArrayVector{result, pair[0], pair[1]},
            *role.scope,
            completion,
            stream) == submission::ScopedProgress::observed);
    role.scope->seal();
    REQUIRE(
        wait_scoped_completion(*role.scope, completion) ==
        ScopedEvaluation::complete);
    REQUIRE(settle_default_role(*role.scope));
    for (auto* value : {&result, &pair[0], &pair[1]})
      REQUIRE(
          validate_scoped_array(*value, *role.scope) ==
          ScopedEvaluation::complete);
    const float reference[] = {35.f, 44.f, 89.f, 116.f};
    for (size_t i = 0; i < 4; ++i) {
      CHECK(result.data<float>()[i] == reference[i]);
      CHECK(result.data<float>()[i] == expected.data<float>()[i]);
    }
    for (size_t i = 0; i < 6; ++i) {
      CHECK(pair[0].data<float>()[i] == expected_pair[0].data<float>()[i]);
      CHECK(pair[1].data<float>()[i] == expected_pair[1].data<float>()[i]);
    }
  }
  REQUIRE(settle_default_role(*role.scope));
  CHECK(fixture.graph->occupied_bytes() == 0);
  CHECK(fixture.budget->occupied_bytes() == 0);
}

TEST_CASE(
    "CPU cleanup retains actual noncontiguous quantized matmul temporary storage") {
  auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  auto raw =
      reshape(add(arange(128, float32, stream), array(1.f), stream), {2, 64});
  auto x = slice(raw, {0, 0}, {2, 64}, {1, 2}, stream);
  auto weights =
      reshape(add(arange(64, float32, stream), array(1.f), stream), {2, 32});
  auto packed = quantize(weights, 32, 4, "affine", std::nullopt, stream);
  auto produce = [&] {
    return quantized_matmul(
        x, packed[0], packed[1], packed[2], true, 32, 4, "affine", stream);
  };
  auto expected = produce();
  eval(expected);
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    auto result = produce();
    Completion completion;
    REQUIRE(
        async_eval_with_completion_scoped_on_stream(
            ArrayVector{result}, *role.scope, completion, stream) ==
        submission::ScopedProgress::observed);
    role.scope->seal();
    REQUIRE(
        wait_scoped_completion(*role.scope, completion) ==
        ScopedEvaluation::complete);
    REQUIRE(settle_default_role(*role.scope));
    REQUIRE(
        validate_scoped_array(result, *role.scope) ==
        ScopedEvaluation::complete);
    REQUIRE(result.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(result.data<float>()[i] > 0.f);
      CHECK(result.data<float>()[i] == expected.data<float>()[i]);
    }
  }
  REQUIRE(settle_default_role(*role.scope));
  CHECK(fixture.graph->occupied_bytes() == 0);
  CHECK(fixture.budget->occupied_bytes() == 0);
}

#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE(
    "CPU cleanup compacts a real noncontiguous Metal source for host copying") {
  auto gpu = new_stream(Device::gpu), cpu = new_stream(Device::cpu);
  prepare_default_streams(gpu, cpu);
  auto matrix = reshape(array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}), {2, 3});
  auto source = transpose(add(matrix, array(10.f), gpu), gpu);
  eval(source);
  eval(contiguous(source, false, cpu)); // exact cold CPU compaction path
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  (void)bind_buffer(role, fixture);
  {
    auto compact = contiguous(source, false, cpu);
    Completion completion;
    REQUIRE(
        async_eval_with_completion_scoped_on_stream(
            ArrayVector{compact}, *role.scope, completion, cpu) ==
        submission::ScopedProgress::observed);
    role.scope->seal();
    REQUIRE(
        wait_scoped_completion(*role.scope, completion) ==
        ScopedEvaluation::complete);
    REQUIRE(settle_default_role(*role.scope));
    REQUIRE(
        validate_scoped_array(compact, *role.scope) ==
        ScopedEvaluation::complete);
    const float expected[] = {11.f, 14.f, 12.f, 15.f, 13.f, 16.f};
    for (size_t i = 0; i < 6; ++i)
      CHECK(compact.data<float>()[i] == expected[i]);
    CHECK(compact.buffer().original_buffer_budget() == fixture.budget.get());
  }
  REQUIRE(settle_default_role(*role.scope));
  CHECK(fixture.graph->occupied_bytes() == 0);
  CHECK(fixture.budget->occupied_bytes() == 0);
}
#endif
