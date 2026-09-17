#include "doctest/doctest.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/event.h"
#include "mlx/c/private/array.h"
#include "mlx/c/submission.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/submission.h"
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <thread>

using namespace mlx::core;
using namespace mlx::core::submission;
namespace {
bool resource_lock_free(metal::CommandEncoder &) noexcept;
struct Retirement {
  std::weak_ptr<metal::CommandEncoder> encoder;
  std::atomic<unsigned> calls{0};
  std::atomic<bool> unlocked{false};
};
FailureCarrierRef failure_owner() {
  auto token = std::make_unique<unsigned>(0);
  auto result = FailureCarrier::create(
      token.get(), [](void *raw) { delete static_cast<unsigned *>(raw); });
  if (!result)
    throw std::bad_alloc();
  token.release();
  return result;
}
GraphQuota *graph_owner(std::shared_ptr<Retirement> state) {
  if (!state)
    return GraphQuota::create(4 << 20, nullptr, nullptr);
  auto token = std::make_unique<std::shared_ptr<Retirement>>(std::move(state));
  auto *graph = GraphQuota::create(4 << 20, token.get(), [](void *raw) {
    std::unique_ptr<std::shared_ptr<Retirement>> token(
        static_cast<std::shared_ptr<Retirement> *>(raw));
    if (auto encoder = (*token)->encoder.lock())
      (*token)->unlocked = resource_lock_free(*encoder);
    ++(*token)->calls;
  });
  if (!graph)
    throw std::bad_alloc();
  token.release();
  return graph;
}
struct GraphRelease {
  void operator()(GraphQuota *p) const noexcept { p->release(); }
};
struct RecordRelease {
  void operator()(RecordQuota *p) const noexcept { p->release(); }
};
struct ScopeRelease {
  void operator()(Scope *p) const noexcept {
    p->seal();
    p->release();
  }
};
struct Owner {
  FailureCarrierRef failure{failure_owner()};
  std::unique_ptr<GraphQuota, GraphRelease> graph;
  std::unique_ptr<RecordQuota, RecordRelease> records{
      RecordQuota::create(4 << 20, nullptr, nullptr)};
  std::unique_ptr<Scope, ScopeRelease> scope;
  explicit Owner(std::shared_ptr<Retirement> retirement = {})
      : graph(graph_owner(std::move(retirement))),
        scope(new Scope(nullptr, nullptr, records.get(), graph.get())) {
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == NativeControlFailure::none);
    REQUIRE(scope->bind_failure(failure));
    REQUIRE(scope->enable_original_controls() == NativeControlFailure::none);
  }
};
struct TestRecord : Record {
  explicit TestRecord(Allocation allocation) : Record(allocation) {}
  bool scoped_observation_supported() const noexcept override { return true; }
};
TestRecord *entered() {
  auto record = Record::create<TestRecord>();
  record->enter();
  return record.release();
}
Stream prepared_stream() {
  auto stream = new_stream(Device::gpu);
  mlx_submission_runtime_baseline facts{};
  REQUIRE(mlx_submission_prepare_runtime(&facts, mlx_stream{&stream},
                                         mlx_stream{&stream}) == 0);
  return stream;
}
template <class F> void await(F &&condition) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!condition()) {
    if (std::chrono::steady_clock::now() >= end)
      FAIL("bounded native callback or record wait expired");
    std::this_thread::yield();
  }
}
void settle(Owner &owner, bool failed = false) {
  await([&] {
    const auto progress = owner.scope->progress_scoped();
    REQUIRE((progress == ScopedProgress::observed ||
             (failed && progress == ScopedProgress::failed)));
    return owner.scope->query().activity != Activity::pending;
  });
  CHECK(owner.scope->query().failed == failed);
  REQUIRE(owner.scope->retire_scoped_records() == ScopedProgress::observed);
}
struct Operation {
  mlx_submission_observer observer{};
  mlx_operation_event event{};
  explicit Operation(Stream stream) {
    REQUIRE(mlx_submission_observer_current(&observer) == 0);
    const auto result =
        mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, 1);
    if (result) {
      mlx_submission_observer_release(observer);
      observer = {};
    }
    REQUIRE(result == 0);
  }
  ~Operation() {
    mlx_operation_event_free(event);
    mlx_submission_observer_release(observer);
  }
};
struct Gate {
  std::atomic<bool> started{false}, release{false};
};
struct ReleaseGates {
  std::shared_ptr<Gate> a, b;
  ~ReleaseGates() {
    a->release = true;
    b->release = true;
  }
};
auto blocking(std::shared_ptr<Gate> gate) {
  return [gate = std::move(gate)](MTL::CommandBuffer *) {
    gate->started = true;
    while (!gate->release.load())
      std::this_thread::yield();
  };
}
// Controlled exhaustion of the actual arena, with every successful allocation
// owned before the next attempt. No platform-OOM claim or production hook.
struct ArenaDrain {
  GraphQuota *graph;
  struct Block {
    void *value;
    size_t bytes;
  };
  std::array<Block, 512> blocks{};
  size_t count{0};
  explicit ArenaDrain(GraphQuota *graph) : graph(graph) {}
  void fill() {
    size_t bytes = graph->capacity() - graph->occupied_bytes();
    for (bytes /= 2; bytes; bytes /= 2) {
      while (true) {
        REQUIRE(count < blocks.size());
        auto *value = graph->try_allocate(bytes, 1);
        if (!value)
          break;
        blocks[count++] = {value, bytes};
      }
    }
    REQUIRE(count < blocks.size());
    auto *last = graph->try_allocate(1, 1);
    if (last)
      blocks[count++] = {last, 1};
    REQUIRE(last == nullptr);
  }
  ~ArenaDrain() {
    while (count) {
      auto block = blocks[--count];
      graph->deallocate(block.value, block.bytes, 1);
    }
  }
};
} // namespace

namespace mlx::core::metal {
struct PersistentStorageTestAccess {
  static size_t resources(CommandEncoder &encoder) {
    return encoder.resources_.size();
  }
  static size_t bytes(CommandEncoder &encoder) { return encoder.buffer_sizes_; }
  static uint8_t bits(CommandEncoder &encoder, const void *key) {
    auto *entry = encoder.resources_.find(key);
    return entry ? entry->bits : 0;
  }
  static bool barrier(CommandEncoder &encoder) {
    return encoder.needs_barrier_;
  }
  static GraphQuota *resource_birth(CommandEncoder &encoder, const void *key) {
    auto *entry = encoder.resources_.find(key);
    return entry ? entry->birth.get() : nullptr;
  }
  static std::pair<GraphQuota *, const detail::EncoderFence *>
  prior(CommandEncoder &encoder, const void *key) {
    std::lock_guard lock(encoder.outputs_mtx_);
    auto *entry = encoder.prev_ce_outputs_.find(key);
    return entry ? std::pair{entry->birth.get(), entry->fence.get()}
                 : std::pair<GraphQuota *, const detail::EncoderFence *>{};
  }
  static const detail::EncoderFence *current_fence(CommandEncoder &encoder) {
    return encoder.fence_.get();
  }
  static bool unlocked(CommandEncoder &encoder) noexcept {
    if (!encoder.outputs_mtx_.try_lock())
      return false;
    encoder.outputs_mtx_.unlock();
    return true;
  }
  static const array *temporary_address(CommandEncoder &encoder, size_t row) {
    const array *result = nullptr;
    size_t seen = 0;
    encoder.temporaries_.each([&](const array &value) {
      if (seen++ == row)
        result = &value;
    });
    return result;
  }
  static void temporary_rows(CommandEncoder &encoder, size_t rows) {
    CHECK(encoder.temporaries_.size() == rows);
  }
};
} // namespace mlx::core::metal
namespace {
bool resource_lock_free(metal::CommandEncoder &encoder) noexcept {
  return metal::PersistentStorageTestAccess::unlocked(encoder);
}
using Access = metal::PersistentStorageTestAccess;
} // namespace

TEST_CASE("Metal persistent resource aliases preserve byte accounting barriers "
          "and nonzero execution") {
  auto stream = prepared_stream();
  array input({2.0f, -3.0f, 7.0f});
  eval(add(input, input, stream));
  array alias = input;
  auto output = add(input, input, stream);
  auto encoder = metal::get_command_encoder_owner(stream);
  Owner owner;
  auto *record = entered();
  {
    RecordDispatchGuard dispatch(*record);
    record->prepare_stream(stream);
    encoder->record_submission();
    auto *buffer = static_cast<const MTL::Buffer *>(input.buffer().ptr());
    encoder->set_buffer(buffer, 0);
    CHECK(Access::bytes(*encoder) == 0);
    encoder->set_input_array(input, 0);
    encoder->set_input_array(alias, 1);
    CHECK(Access::resources(*encoder) == 1);
    CHECK(Access::bytes(*encoder) ==
          0); // prior set_buffer already marks all-input
    CHECK(Access::resource_birth(*encoder, buffer) == owner.graph.get());
    encoder->maybeInsertBarrier();
    encoder->register_output_array(alias);
    CHECK(Access::barrier(*encoder));
    encoder->maybeInsertBarrier();
    CHECK_FALSE(Access::barrier(*encoder));
    {
      auto concurrent = encoder->start_concurrent();
      encoder->register_output_array(input);
      CHECK((Access::bits(*encoder, buffer) &
             metal::detail::ResourceEntry::concurrent_output) != 0);
    }
    CHECK((Access::bits(*encoder, buffer) &
           metal::detail::ResourceEntry::concurrent_output) == 0);
    encoder->set_input_array(input, 0);
    CHECK(Access::barrier(*encoder));
    gpu::eval(output);
    gpu::finalize(stream);
  }
  record->finish(false);
  settle(owner);
  CHECK(output.data<float>()[0] == 4.0f);
  CHECK(output.data<float>()[1] == -6.0f);
  CHECK(output.data<float>()[2] == 14.0f);
  CHECK(Access::resources(*encoder) == 0);
  CHECK(Access::prior(*encoder, input.buffer().ptr()).first == nullptr);
}

TEST_CASE("Metal persistent missing resource refuses full arena before binding "
          "while encoded prefix survives") {
  auto stream = prepared_stream();
  array input({2.0f, -3.0f, 7.0f}), other({11.0f, 13.0f, 17.0f});
  eval(add(input, input, stream));
  auto output = add(input, input, stream);
  auto encoder = metal::get_command_encoder_owner(stream);
  Owner owner;
  auto *record = entered();
  {
    RecordDispatchGuard dispatch(*record);
    record->prepare_stream(stream);
    encoder->record_submission();
    const auto accepted = encoder->submission_progress().accepted;
    {
      ArenaDrain drain(owner.graph.get());
      drain.fill();
      CHECK_THROWS_AS(encoder->set_input_array(other, 0), GraphQuotaError);
      CHECK(Access::resources(*encoder) == 0);
      CHECK(Access::bytes(*encoder) == 0);
      CHECK(encoder->submission_progress().accepted == accepted);
    }
    gpu::eval(output); // genuine nonzero encoded prefix before second refusal
    const auto count = Access::resources(*encoder),
               bytes = Access::bytes(*encoder);
    {
      ArenaDrain drain(owner.graph.get());
      drain.fill();
      CHECK_THROWS_AS(encoder->set_output_array(other, 0), GraphQuotaError);
      CHECK(Access::resources(*encoder) == count);
      CHECK(Access::bytes(*encoder) == bytes);
      CHECK(Access::bits(*encoder, other.buffer().ptr()) == 0);
      CHECK_FALSE(encoder->submission_idle());
    }
    gpu::finalize(stream);
  }
  record->finish(false);
  settle(owner);
  CHECK(output.data<float>()[0] == 4.0f);
  CHECK(output.data<float>()[1] == -6.0f);
  CHECK(output.data<float>()[2] == 14.0f);
}

TEST_CASE("Metal persistent later writer survives prior callback and retains "
          "original node birth") {
  auto stream = prepared_stream();
  auto encoder = metal::get_command_encoder_owner(stream);
  array input({2.0f, -3.0f, 7.0f});
  eval(add(input, input, stream));
  std::optional<array> output_a(add(input, input, stream));
  std::optional<array> output_b(add(input, input, stream));
  auto retired = std::make_shared<Retirement>();
  retired->encoder = encoder;
  auto gate_a = std::make_shared<Gate>(), gate_b = std::make_shared<Gate>();
  ReleaseGates release{gate_a, gate_b};
  auto a_cleanup_finished = std::make_shared<std::atomic<bool>>(false);
  std::optional<Owner> a;
  a.emplace(retired);
  auto *first_graph = a->graph.get();
  auto *first = entered();
  {
    RecordDispatchGuard dispatch(*first);
    first->prepare_stream(stream);
    encoder->record_submission();
    encoder->add_completed_handler(blocking(gate_a));
    encoder->set_buffer(static_cast<const MTL::Buffer *>(input.buffer().ptr()),
                        0);
    gpu::eval(*output_a);
    encoder->end_encoding();
    encoder->add_completed_handler([a_cleanup_finished](MTL::CommandBuffer *) {
      *a_cleanup_finished = true;
    });
    gpu::finalize(stream);
  }
  first->finish(false);
  a->scope->seal();
  await([&] { return gate_a->started.load(); });
  CHECK(Access::prior(*encoder, input.buffer().ptr()).first == first_graph);
  Owner b;
  auto *second = entered();
  const metal::detail::EncoderFence *second_fence = nullptr;
  {
    RecordDispatchGuard dispatch(*second);
    second->prepare_stream(stream);
    encoder->record_submission();
    encoder->add_completed_handler(blocking(gate_b));
    encoder->set_buffer(static_cast<const MTL::Buffer *>(input.buffer().ptr()),
                        0);
    gpu::eval(*output_b);
    second_fence = Access::current_fence(*encoder);
    gpu::finalize(stream);
  }
  second->finish(false);
  CHECK(Access::prior(*encoder, input.buffer().ptr()).first == first_graph);
  CHECK(Access::prior(*encoder, input.buffer().ptr()).second == second_fence);
  gate_a->release = true;
  await([&] { return a_cleanup_finished->load(); });
  // A's cleanup ran, but same-stream completion does not advance while B's
  // receipt is pending. Keep A's actual Record until both receipts complete.
  CHECK(a->scope->query().activity == Activity::pending);
  CHECK(retired->calls == 0);
  CHECK(Access::prior(*encoder, input.buffer().ptr()).first == first_graph);
  CHECK(Access::prior(*encoder, input.buffer().ptr()).second == second_fence);
  gate_b->release = true;
  settle(b);
  settle(*a);
  CHECK(output_a->data<float>()[2] == 14.0f);
  output_a.reset();
  a.reset();
  CHECK(output_b->data<float>()[0] == 4.0f);
  CHECK(output_b->data<float>()[1] == -6.0f);
  output_b.reset();
  await([&] { return retired->calls.load() != 0; });
  CHECK(retired->calls == 1);
  // This checks final owner retirement after records settle. The Record itself
  // pins A during the callbacks, so sole-node release outside outputs_mtx_ is
  // a source-reviewed ordering property, not an isolated oracle in this case.
  CHECK(retired->unlocked);
  CHECK(Access::prior(*encoder, input.buffer().ptr()).first == nullptr);
  CHECK(b.graph->occupied_bytes() == 0);
}

TEST_CASE("Metal persistent temporary batches adopt actual vector allocation "
          "and retain duplicate rows") {
  auto stream = prepared_stream();
  auto encoder = metal::get_command_encoder_owner(stream);
  array input({2.0f, -3.0f, 7.0f});
  eval(add(input, input, stream));
  std::optional<array> output(add(input, input, stream));
  Owner a;
  std::optional<ArrayVector> moved(std::in_place,
                                   ArrayVector::allocator_type(a.graph.get()));
  moved->reserve(9);
  moved->push_back(input);
  moved->push_back(input);
  std::optional<ArrayVector> copied(std::in_place,
                                    ArrayVector::allocator_type(a.graph.get()));
  copied->reserve(5);
  copied->push_back(input);
  ArrayVector empty(ArrayVector::allocator_type(a.graph.get()));
  empty.reserve(8);
  const auto *exact = moved->data();
  const auto *source = copied->data();
  a.scope->seal();
  Owner b;
  auto *record = entered();
  {
    RecordDispatchGuard dispatch(*record);
    record->prepare_stream(stream);
    gpu::eval(*output);
    encoder->add_temporaries(std::move(*moved));
    CHECK(Access::temporary_address(*encoder, 0) == exact);
    CHECK(Access::temporary_address(*encoder, 1) == exact + 1);
    CHECK(moved->empty());
    encoder->add_temporaries(*copied);
    CHECK(copied->data() == source);
    CHECK(Access::temporary_address(*encoder, 2) != source);
    encoder->add_temporary(input);
    encoder->add_temporaries(std::move(empty));
    CHECK(empty.capacity() == 0);
    Access::temporary_rows(*encoder, 4);
    CHECK(a.graph->occupied_bytes() > 0);
    gpu::finalize(stream);
    Access::temporary_rows(*encoder, 0);
  }
  record->finish(false);
  settle(b);
  CHECK(output->data<float>()[0] == 4.0f);
  CHECK(output->data<float>()[1] == -6.0f);
  CHECK(output->data<float>()[2] == 14.0f);
  output.reset();
  copied.reset();
  moved.reset();
  CHECK(a.graph->occupied_bytes() == 0);
  CHECK(b.graph->occupied_bytes() == 0);
}

TEST_CASE("Metal persistent concurrent concatenate with repeated source "
          "matches ordinary values") {
  auto stream = prepared_stream();
  array a({2.0f, -3.0f, 7.0f}), b({11.0f, 13.0f, 17.0f});
  auto ordinary = concatenate({a, b, a}, 0, stream);
  eval(ordinary);
  Owner owner;
  {
    auto original = concatenate({a, b, a}, 0, stream);
    Operation operation(stream);
    REQUIRE(mlx_operation_event_append(operation.event, mlx_array{&original}) ==
            0);
    REQUIRE(mlx_operation_event_submit_on_stream(operation.event,
                                                 mlx_stream{&stream}) == 0);
    REQUIRE(mlx_operation_event_wait(operation.event) == 0);
    settle(owner);
    REQUIRE(mlx_submission_observer_validate_array(operation.observer,
                                                   mlx_array{&original}) == 0);
    const std::array<float, 9> expected{2, -3, 7, 11, 13, 17, 2, -3, 7};
    REQUIRE(original.size() == expected.size());
    for (size_t i = 0; i != expected.size(); ++i) {
      CHECK(original.data<float>()[i] == expected[i]);
      CHECK(original.data<float>()[i] == ordinary.data<float>()[i]);
    }
  }
  settle(owner);
  CHECK(owner.graph->occupied_bytes() == 0);
}

namespace {
struct TemporaryRefusalState {
  bool refused{false};
  std::weak_ptr<array::Data> temporary;
};
struct RefuseTemporaryAfterCopy : Add {
  std::shared_ptr<TemporaryRefusalState> state;
  RefuseTemporaryAfterCopy(Stream stream,
                           std::shared_ptr<TemporaryRefusalState> state)
      : Add(stream), state(std::move(state)) {}
  void eval_gpu(const ArrayVector &inputs, array &output) override {
    Add::eval_gpu(inputs, output);
    array temporary(inputs.front().shape(), inputs.front().dtype(), nullptr,
                    {});
    copy_gpu(inputs.front(), temporary, CopyType::General, stream());
    state->temporary = temporary.data_shared_ptr();
    auto encoder = metal::get_command_encoder_owner(stream());
    try {
      ArenaDrain drain(current_graph_quota());
      drain.fill();
      encoder->add_temporary(std::move(temporary));
    } catch (const GraphQuotaError &) {
      state->refused = true;
      // The controlled drain has released its blocks. Fund this SAME entered
      // dispatch before outer finish(true) blocks new submissions. The real
      // failed primitive's Data captures remain owned until record retirement.
      gpu::finalize(stream());
      throw;
    }
    FAIL("actual full Graph arena unexpectedly accepted a temporary batch");
  }
};
} // namespace

TEST_CASE("Metal persistent temporary refusal after real copy retains exact "
          "Data through funded completion") {
  auto stream = prepared_stream();
  array input({2.0f, -3.0f, 7.0f});
  eval(add(input, input, stream));
  // Warm the same general-copy kernel before entering original controls.
  {
    array warm(input.shape(), input.dtype(), nullptr, {});
    copy_gpu(input, warm, CopyType::General, stream);
    synchronize(stream);
  }
  auto state = std::make_shared<TemporaryRefusalState>();
  Owner owner;
  {
    auto output = add(input, input, stream);
    output.primitive_ptr() =
        make_graph_primitive<RefuseTemporaryAfterCopy>(stream, state);
    Operation operation(stream);
    REQUIRE(mlx_operation_event_append(operation.event, mlx_array{&output}) ==
            0);
    CHECK(mlx_operation_event_submit_on_stream(operation.event,
                                               mlx_stream{&stream}) != 0);
    REQUIRE(state->refused);
    REQUIRE_FALSE(state->temporary.expired());
    auto *cause = owner.failure.get()->borrow();
    REQUIRE(cause != nullptr);
    CHECK(cause->kind == FailureKind::exception);
    CHECK(cause->exception_type == &typeid(GraphQuotaError));
    // The original entered dispatch funded its exact prefix before throwing.
    // Failure now blocks new submissions; observation does not bypass that
    // gate.
    await([&] {
      const auto progress = owner.scope->progress_scoped();
      REQUIRE((progress == ScopedProgress::observed ||
               progress == ScopedProgress::failed));
      return owner.scope->query().activity != Activity::pending;
    });
    {
      auto data = state->temporary.lock();
      REQUIRE(data != nullptr);
      const auto *values = static_cast<const float *>(data->buffer.raw_ptr());
      REQUIRE(values != nullptr);
      CHECK(values[0] == 2.0f);
      CHECK(values[1] == -3.0f);
      CHECK(values[2] == 7.0f);
    }
    REQUIRE(owner.scope->retire_scoped_records() == ScopedProgress::observed);
    CHECK(state->temporary.expired());
    // The Data-control successor retains its shared control through the final
    // weak release, independently of the expired Data payload.
    state->temporary.reset();
    CHECK(owner.scope->query().failed);
  }
  settle(owner, true);
  CHECK(owner.graph->occupied_bytes() == 0);
}

TEST_CASE("Metal persistent controls report actual fixed storage without "
          "accepting a workspace fit") {
  metal::PersistentControlLayout facts{};
  REQUIRE(metal::CommandEncoder::persistent_control_layout(facts));
  CHECK(facts.encoder_bytes == sizeof(metal::CommandEncoder));
  CHECK(facts.resource_entry.bytes == sizeof(metal::detail::ResourceEntry));
  CHECK(facts.fence_entry.bytes == sizeof(metal::detail::FenceEntry));
  CHECK(facts.temporary_batch.bytes == sizeof(metal::detail::TemporaryBatch));
  CHECK(facts.resource_entry.graph_extent >= facts.resource_entry.bytes);
  CHECK(facts.fence_entry.graph_extent >= facts.fence_entry.bytes);
  CHECK(facts.encoder_fence.graph_extent >= facts.encoder_fence.bytes);
}
