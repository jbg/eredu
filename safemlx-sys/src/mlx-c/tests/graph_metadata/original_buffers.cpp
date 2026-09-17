// Included after data_controls.cpp; exercises the explicit new mechanism only.
#include "mlx/original_buffer.h"
#include "mlx/fence.h"
#ifdef MLX_C_PATCH_TEST_METAL
#include "mlx/backend/metal/device.h"
#include "mlx/utils.h"
#endif
#include <barrier>
#include <chrono>
#include <limits>

namespace {
using BufferCause = allocator::OriginalBufferCause;
struct BufferCounts {
  std::atomic<unsigned> allocated{0}, freed{0}, budget_retired{0}, graph_retired{0};
  std::atomic<bool> graph_before_budget{false};
};
struct BufferAllocator final : allocator::Allocator {
  enum class Refusal { none, before, after };
  allocator::Allocator& actual{allocator::allocator()};
  std::shared_ptr<BufferCounts> counts;
  Refusal refusal{Refusal::none};
  explicit BufferAllocator(std::shared_ptr<BufferCounts> counts) : counts(std::move(counts)) {}
  allocator::Buffer malloc(size_t n) override { return actual.malloc(n); }
  void free(allocator::Buffer p) override { allocator::free(p); }
  size_t size(allocator::Buffer p) const override { return actual.size(p); }
  bool prepare_input_runtime(allocator::PreparedInputFacts& facts) override {
    return actual.prepare_input_runtime(facts);
  }
  allocator::PreparedInputCause allocate_prepared_buffer(size_t n,
      allocator::PreparedInputAllocation& out) noexcept override {
    ++counts->allocated;
    if (refusal == Refusal::before) return allocator::PreparedInputCause::busy;
    auto cause = actual.allocate_prepared_buffer(n, out);
    if (out.buffer.ptr()) out.allocator = this;
    if (cause == allocator::PreparedInputCause::success && refusal == Refusal::after)
      return allocator::PreparedInputCause::capacity;
    return cause;
  }
  void free_prepared_input(allocator::PreparedInputAllocation& out) noexcept override {
    ++counts->freed;
    actual.free_prepared_input(out);
  }
};
struct BufferBudgetRelease {
  void operator()(allocator::OriginalBufferBudget* p) const noexcept { p->release(); }
};
struct BufferGraphRelease {
  void operator()(submission::GraphQuota* p) const noexcept { p->release(); }
};
struct BufferRecordFinish {
  submission::Record* value;
  ~BufferRecordFinish() { if (value) value->finish(true); }
};
struct BufferFixture {
  std::shared_ptr<BufferCounts> counts{std::make_shared<BufferCounts>()};
  std::shared_ptr<BufferAllocator> physical{std::make_shared<BufferAllocator>(counts)};
  std::unique_ptr<allocator::OriginalBufferBudget, BufferBudgetRelease> budget;
  std::unique_ptr<submission::GraphQuota, BufferGraphRelease> graph;
  explicit BufferFixture(size_t capacity = 1 << 20, size_t graph_bytes = 1 << 20) {
    auto graph_owner = std::make_unique<std::shared_ptr<BufferCounts>>(counts);
    graph.reset(submission::GraphQuota::create(graph_bytes, graph_owner.get(), [](void* p) {
      auto* owner = static_cast<std::shared_ptr<BufferCounts>*>(p);
      (*owner)->graph_before_budget = (*owner)->budget_retired.load() == 0;
      ++(*owner)->graph_retired; delete owner;
    }));
    graph_owner.release();
    // A native allocation may outlive this fixture. Its concrete allocator and
    // counters therefore belong to the allowance's independent heap owner.
    auto owner = std::make_unique<std::shared_ptr<BufferAllocator>>(physical);
    budget.reset(allocator::OriginalBufferBudget::create(*physical, capacity, owner.get(), [](void* p) {
      auto* owner = static_cast<std::shared_ptr<BufferAllocator>*>(p);
      ++(*owner)->counts->budget_retired; delete owner;
    }));
    owner.release();
  }
};
allocator::OriginalBufferBinding bind_buffer(DataRole& role, BufferFixture& fixture) {
  role.scope->bind_original_buffer_budget(*fixture.budget);
  return allocator::OriginalBufferBinding::capture(*fixture.budget);
}
template<class F> void buffer_refusal(F&& work, BufferCause cause) {
  try { work(); FAIL_CHECK("explicit original buffer operation must refuse"); }
  catch (const allocator::OriginalBufferError& error) { CHECK(error.cause() == cause); }
}
void fill_buffer(allocator::Buffer buffer) {
  const float values[] = {5, 7, 11, 13, 17, 19};
  std::copy(std::begin(values), std::end(values), static_cast<float*>(buffer.raw_ptr()));
}
}

TEST_CASE("original buffer direct capacity zero and scalar preserve physical strategy") {
  BufferFixture fixture;
  const auto cached = get_cache_memory();
  DataRole role(fixture.graph.get());
  auto binding = bind_buffer(role, fixture);
  for (size_t bytes : {size_t{0}, sizeof(float), size_t{24}}) {
    size_t capacity = 0;
    REQUIRE(fixture.budget->physical_capacity(bytes, capacity));
    const auto before = fixture.counts->allocated.load();
    auto buffer = allocator::allocate_original(bytes, binding);
    CHECK(fixture.budget->occupied_bytes() == capacity);
    if (capacity) {
      CHECK(buffer.ptr() != nullptr);
      CHECK(buffer.original_buffer_budget() == fixture.budget.get());
      CHECK(buffer.original_allocation_generation() != 0);
      CHECK(buffer.original_allocation_capacity() == capacity);
      CHECK(fixture.counts->allocated == before + 1);
      if (bytes) { *static_cast<float*>(buffer.raw_ptr()) = 3.5f;
        CHECK(*static_cast<float*>(buffer.raw_ptr()) == 3.5f); }
    } else {
      CHECK(buffer.ptr() == nullptr);
      CHECK(fixture.counts->allocated == before);
    }
    allocator::free(buffer);
    CHECK(fixture.budget->occupied_bytes() == 0);
  }
  const auto before = fixture.counts->allocated.load();
  buffer_refusal([&] { (void)allocator::allocate_original(std::numeric_limits<size_t>::max(), binding); },
                 BufferCause::invalid_layout);
  CHECK(fixture.counts->allocated == before);
  CHECK(get_cache_memory() == cached);
  allocator::PreparedInputAllocation source;
  CHECK(fixture.physical->actual.allocate_prepared_input(0, source) == allocator::PreparedInputCause::invalid);
}

TEST_CASE("original buffer missing foreign and exhausted allowances never enter physical worker") {
  const auto stream = new_stream(Device::cpu);
  BufferFixture a(0), b;
  std::optional<allocator::OriginalBufferBinding> from_a;
  {
    DataRole role(a.graph.get());
    buffer_refusal([&] { (void)allocator::OriginalBufferBinding::capture(*a.budget); }, BufferCause::missing_binding);
    auto binding = bind_buffer(role, a);
    from_a.emplace(binding);
    buffer_refusal([&] { role.scope->bind_original_buffer_budget(*a.budget); }, BufferCause::already_bound);
    buffer_refusal([&] { (void)allocator::allocate_original(4, binding); }, BufferCause::capacity);
    buffer_refusal([&] { role.scope->bind_original_buffer_budget(*a.budget); }, BufferCause::invalid_scope);
    CHECK(a.counts->allocated == 0);
    CHECK(a.budget->occupied_bytes() == 0);
  }
  {
    DataRole role(b.graph.get());
    auto unbound_record = submission::Record::create<DataCaptureRecord>();
    auto binding = bind_buffer(role, b);
    buffer_refusal([&] { unbound_record->enter(); }, BufferCause::foreign_domain);
    CHECK(role.scope->query().activity == submission::Activity::none);
    buffer_refusal([&] { (void)allocator::allocate_original(4, *from_a); }, BufferCause::foreign_domain);
    CHECK(b.counts->allocated == 0);
    allocator::OriginalBufferBinding missing;
    buffer_refusal([&] { (void)allocator::allocate_original(4, missing); }, BufferCause::missing_binding);
    buffer_refusal([&] { Fence rejected(stream, missing); }, BufferCause::missing_binding);
  }
}

TEST_CASE("original buffer allocation refusals roll back only their own real debit") {
  BufferFixture fixture;
  DataRole role(fixture.graph.get());
  auto binding = bind_buffer(role, fixture);
  auto first = allocator::allocate_original(24, binding);
  const auto held = fixture.budget->occupied_bytes();
  const auto baseline = fixture.graph->occupied_bytes();
  for (auto mode : {BufferAllocator::Refusal::before, BufferAllocator::Refusal::after}) {
    fixture.physical->refusal = mode;
    const auto freed = fixture.counts->freed.load();
    buffer_refusal([&] { (void)allocator::allocate_original(24, binding); },
        mode == BufferAllocator::Refusal::before ? BufferCause::busy : BufferCause::capacity);
    CHECK(fixture.budget->occupied_bytes() == held);
    CHECK(fixture.graph->occupied_bytes() == baseline);
    CHECK(fixture.counts->freed == freed + (mode == BufferAllocator::Refusal::after ? 1 : 0));
  }
  fixture.physical->refusal = BufferAllocator::Refusal::none;
  { FilledDataArena full(fixture.graph.get());
    const auto calls = fixture.counts->allocated.load();
    CHECK_THROWS_AS(allocator::allocate_original(24, binding), submission::GraphQuotaError);
    CHECK(fixture.counts->allocated == calls);
    CHECK(fixture.budget->occupied_bytes() == held);
  }
  allocator::free(first);
  CHECK(fixture.budget->occupied_bytes() == 0);
}

TEST_CASE("original buffer both Data refusal paths preserve old output and one physical free") {
  for (bool strided : {false, true}) {
    BufferFixture fixture;
    auto value = array({2.0f, 3.0f, 5.0f});
    auto old = value.data_shared_ptr();
    DataRole role(fixture.graph.get());
    auto binding = bind_buffer(role, fixture);
    auto incoming = allocator::allocate_original(24, binding);
    fill_buffer(incoming);
    Strides strides{2};
    { FilledDataArena full(fixture.graph.get());
      CHECK_THROWS_AS(install_data(value, incoming, strided, strides, allocator::free), submission::GraphQuotaError);
      CHECK(value.data_shared_ptr() == old);
      CHECK(value.data<float>()[2] == 5.0f);
      CHECK(fixture.budget->occupied_bytes() == 0);
      CHECK(fixture.counts->freed == 1);
    }
  }
}

TEST_CASE("original buffer mutable Data keeps one generation through donation aliases and weak controls") {
  BufferFixture a, b;
  auto counts = a.counts;
  std::optional<array> value(array({2.0f, 3.0f, 5.0f})), alias;
  std::weak_ptr<array::Data> weak;
  uint64_t generation = 0;
  size_t capacity = 0;
  {
    DataRole role(a.graph.get());
    auto binding = bind_buffer(role, a);
    auto buffer = allocator::allocate_original(24, binding);
    fill_buffer(buffer);
    generation = buffer.original_allocation_generation(); capacity = buffer.original_allocation_capacity();
    value->set_data(buffer);
    CHECK(value->data_shared_ptr()->allocation_generation == generation);
    CHECK(value->data_shared_ptr()->original_input == nullptr);
    CHECK(value->is_donatable());
    weak = value->data_shared_ptr();
  }
  {
    DataRole role(b.graph.get());
    auto binding = bind_buffer(role, b);
    alias.emplace(array({1.0f, 2.0f, 3.0f}));
    alias->copy_shared_buffer(*value);
    CHECK(alias->buffer().original_buffer_budget() == a.budget.get());
    CHECK(alias->data_shared_ptr()->allocation_generation == generation);
    CHECK(b.budget->occupied_bytes() == 0);
    value.reset();
    CHECK(alias->is_donatable());
    CHECK(alias->data<float>()[2] == 11.0f);
  }
  CHECK(a.budget->occupied_bytes() == capacity);
  a.budget.reset(); a.graph.reset();
  CHECK(counts->budget_retired == 0);
  std::thread release([held = std::move(alias)]() mutable { held.reset(); });
  release.join();
  CHECK(weak.expired());
  CHECK(counts->freed == 1);
  CHECK(counts->budget_retired == 1);
  CHECK(counts->graph_retired == 0);
  weak.reset();
  CHECK(counts->graph_retired == 1);
}

TEST_CASE("original buffer active Record keeps real failed Data after Scope end") {
  const auto stream = new_stream(Device::cpu);
  BufferFixture fixture;
  std::weak_ptr<array::Data> weak;
  auto value = array({2.0f, 3.0f, 5.0f});
  DataRole role(fixture.graph.get());
  auto binding = bind_buffer(role, fixture);
  auto pending = submission::Record::create<DataCaptureRecord>();
  pending->enter(); auto* record = pending.release();
  BufferRecordFinish record_guard{record};
  record->begin_primitive();
  role.scope->seal();
  buffer_refusal([&] { Fence rejected(stream, binding); }, BufferCause::invalid_scope);
  auto buffer = allocator::allocate_original(24, binding);
  fill_buffer(buffer); value.set_data(buffer);
  weak = value.data_shared_ptr();
  record->finish(true); record_guard.value = nullptr;
  value = array({1.0f});
  CHECK_FALSE(weak.expired());
  CHECK(fixture.budget->occupied_bytes() == buffer.original_allocation_capacity());
  REQUIRE(role.scope->progress_scoped() == submission::ScopedProgress::observed);
  REQUIRE(role.scope->retire_scoped_records() == submission::ScopedProgress::observed);
  CHECK(weak.expired());
  CHECK(fixture.counts->freed == 1);
  CHECK(fixture.budget->occupied_bytes() == 0);
  weak.reset();
}

TEST_CASE("original buffer concurrent roles share one finite physical allowance") {
  BufferFixture owner;
  size_t page;
  REQUIRE(owner.budget->physical_capacity(4, page));
  BufferFixture finite(page);
  std::barrier ready(2), finished(2);
  std::atomic<unsigned> accepted{0};
  auto attempt = [&] {
    BufferFixture local;
    DataRole role(local.graph.get());
    role.scope->bind_original_buffer_budget(*finite.budget);
    auto binding = allocator::OriginalBufferBinding::capture(*finite.budget);
    ready.arrive_and_wait();
    allocator::Buffer buffer{nullptr};
    try { buffer = allocator::allocate_original(4, binding); ++accepted; }
    catch (const allocator::OriginalBufferError& error) {
      CHECK(error.cause() == BufferCause::capacity);
    }
    CHECK(finite.budget->occupied_bytes() <= page);
    finished.arrive_and_wait();
    allocator::free(buffer);
  };
  std::thread first(attempt), second(attempt);
  first.join(); second.join();
  CHECK(accepted == 1);
  CHECK(finite.counts->allocated == 1);
  CHECK(finite.counts->freed == 1);
  CHECK(finite.budget->occupied_bytes() == 0);
}

TEST_CASE("original buffer concurrent allocation and retirement do not refuse available capacity") {
  constexpr size_t workers = 4;
  constexpr size_t rounds = 128;
  BufferFixture size_probe;
  size_t page;
  REQUIRE(size_probe.budget->physical_capacity(32, page));
  BufferFixture shared(page * workers);
  std::array<BufferFixture, workers> local;
  std::barrier ready(workers);
  std::array<std::thread, workers> threads;
  std::atomic<unsigned> failed{0}, accepted{0};
  for (size_t worker = 0; worker < workers; ++worker) {
    threads[worker] = std::thread([&, worker] {
      DataRole role(local[worker].graph.get());
      role.scope->bind_original_buffer_budget(*shared.budget);
      auto binding = allocator::OriginalBufferBinding::capture(*shared.budget);
      for (size_t round = 0; round < rounds; ++round) {
        ready.arrive_and_wait();
        allocator::Buffer buffer{nullptr};
        try {
          buffer = allocator::allocate_original(32, binding);
          ++accepted;
          const auto tag = static_cast<unsigned char>(worker + round + 1);
          std::memset(buffer.raw_ptr(), tag, 32);
          const auto* bytes = static_cast<const unsigned char*>(buffer.raw_ptr());
          for (size_t byte = 0; byte < 32; ++byte) CHECK(bytes[byte] == tag);
          CHECK(shared.budget->occupied_bytes() <= page * workers);
        } catch (...) {
          ++failed;
        }
        // This return may overlap another role's debit or native allocation.
        // All workers fit simultaneously; no capacity refusal is legitimate.
        allocator::free(buffer);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  CHECK(failed.load() == 0);
  CHECK(accepted.load() == workers * rounds);
  CHECK(shared.counts->allocated.load() == workers * rounds);
  CHECK(shared.counts->freed.load() == workers * rounds);
  CHECK(shared.budget->occupied_bytes() == 0);
}

TEST_CASE("original buffer final raw free retains allowance across Graph record retirement") {
  allocator::Buffer raw{nullptr};
  std::shared_ptr<BufferCounts> counts;
  {
    BufferFixture fixture;
    counts = fixture.counts;
    DataRole role(fixture.graph.get());
    auto binding = bind_buffer(role, fixture);
    raw = allocator::allocate_original(24, binding);
    fill_buffer(raw);
  }
  REQUIRE(counts->budget_retired == 0);
  REQUIRE(counts->graph_retired == 0);
  CHECK(static_cast<float*>(raw.raw_ptr())[5] == 19.0f);
  // No scope/record, allocator fixture, binding or quota handle survives here.
  allocator::free(raw);
  CHECK(counts->freed == 1);
  CHECK(counts->graph_retired == 1);
  CHECK(counts->graph_before_budget);
  CHECK(counts->budget_retired == 1);
}

TEST_CASE("original buffer Data copy and capture failures preserve incoming birth cleanup") {
  for (bool strided : {false, true}) {
    for (unsigned fail_at : {1u, 2u}) {
      BufferFixture fixture;
      auto value = array({2.0f, 3.0f, 5.0f});
      auto old = value.data_shared_ptr();
      DataCopyState state;
      Deleter deleter = DataCopyDeleter(state);
      state.copies = 0; state.fail_at = fail_at;
      DataRole role(fixture.graph.get());
      auto binding = bind_buffer(role, fixture);
      auto raw = allocator::allocate_original(24, binding);
      Strides strides{2};
      CHECK_THROWS_AS(install_data(value, raw, strided, strides, std::move(deleter)), DataCopyFailure);
      CHECK(value.data_shared_ptr() == old);
      CHECK(state.freed == 1);
      CHECK(fixture.counts->freed == 1);
      CHECK(fixture.budget->occupied_bytes() == 0);
    }
    BufferFixture fixture;
    auto value = array({2.0f, 3.0f, 5.0f});
    auto old = value.data_shared_ptr();
    DataRole role(fixture.graph.get());
    auto binding = bind_buffer(role, fixture);
    auto raw = allocator::allocate_original(24, binding);
    auto pending = submission::Record::create<DataCaptureRecord>();
    pending->enter(); auto* record = pending.release();
    BufferRecordFinish record_guard{record};
    record->begin_primitive();
    try { for (;;) { auto capture = submission::prepare_capture(); } }
    catch (const submission::RecordQuotaError& error) {
      CHECK(error.cause() == submission::QuotaFailure::exhausted);
    }
    Strides strides{2};
    CHECK_THROWS_AS(install_data(value, raw, strided, strides, allocator::free), submission::RecordQuotaError);
    CHECK(value.data_shared_ptr() == old);
    CHECK(fixture.counts->freed == 1);
    CHECK(fixture.budget->occupied_bytes() == 0);
    record->finish(true); record_guard.value = nullptr;
    REQUIRE(role.scope->progress_scoped() == submission::ScopedProgress::observed);
    REQUIRE(role.scope->retire_scoped_records() == submission::ScopedProgress::observed);
  }
}

TEST_CASE("original buffer real evaluator donation preserves immutable birth and nonzero values") {
  BufferFixture fixture;
  std::optional<array> source(array({1.0f, 2.0f, 3.0f}));
  uint64_t generation;
  void* pointer;
  {
    DataRole role(fixture.graph.get());
    auto binding = bind_buffer(role, fixture);
    auto buffer = allocator::allocate_original(24, binding);
    fill_buffer(buffer); source->set_data(buffer);
    generation = buffer.original_allocation_generation(); pointer = source->data<void>();
  }
  // Existing evaluator remains unchanged; only the source's concrete backing
  // was explicitly allocated. This does not claim interception of new outputs.
  auto output = negative(*source, Device::cpu);
  source.reset();
  auto completion = async_eval_with_completion({output}); completion.wait();
  CHECK(output.data<void>() == pointer);
  CHECK(output.buffer().original_allocation_generation() == generation);
  CHECK(output.data<float>()[0] == -5.0f);
  CHECK(output.data<float>()[1] == -7.0f);
  CHECK(output.data<float>()[2] == -11.0f);
  CHECK(output.data_shared_ptr()->original_input == nullptr);
  submission::progress_records(); submission::retire_records();
}

TEST_CASE("original buffer explicit Fence retains actual backing through cross-stream completion") {
  auto producer = new_stream(is_available(Device::gpu) ? Device::gpu : Device::cpu);
  auto consumer = new_stream(Device::cpu);
  auto input = array({5.0f, 7.0f, 11.0f});
  auto output = array({0.0f, 0.0f, 0.0f});
  // Warm exactly the ordinary streams and existing device policy, including
  // actual cross-device Fence kernels where that policy selects them.
  eval(add(add(input, input, producer), input, consumer));
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&producer}, mlx_stream{&consumer}) == 0);
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
  BufferFixture fixture;
  const auto cached = get_cache_memory();
  DataRole role(fixture.graph.get());
  auto binding = bind_buffer(role, fixture);
  auto pending = submission::Record::create<DataCaptureRecord>();
  pending->enter(); auto* record = pending.release();
  BufferRecordFinish record_guard{record};
  record->reserve_streams(2); record->prepare_stream(producer); record->prepare_stream(consumer);
  struct Gate { std::atomic<bool> open{false}; };
  auto gate = std::make_shared<Gate>();
  struct ReleaseGate { std::shared_ptr<Gate> value; ~ReleaseGate() { value->open.store(true); } } gate_guard{gate};
  {
    submission::RecordDispatchGuard dispatch(*record);
    scheduler::enqueue(consumer, [gate] { while (!gate->open.load()) std::this_thread::yield(); });
    {
      Fence fence(producer, binding);
      fence.update(producer, input, producer.device != consumer.device);
      fence.wait(consumer, input);
      auto* source = input.data<float>(); auto* destination = output.data<float>();
      // A failed assertion may unwind before the queued task runs. Keep the
      // actual source/output owners with that task, independently of locals.
      scheduler::enqueue(consumer, [source, destination,
          source_owner = input.data_shared_ptr(), output_owner = output.data_shared_ptr()] {
        for (size_t i = 0; i < 3; ++i) destination[i] = 2 * source[i];
      });
      if (producer.device == Device::gpu) gpu::finalize(producer);
    }
    // Fast Metal mode made exactly one direct allocation. CPU/slow Event mode
    // makes none. The controlled consumer still retains the actual Fence owner.
    CHECK(fixture.counts->allocated == expected_allocations);
    CHECK(fixture.counts->freed == 0);
    CHECK((fixture.budget->occupied_bytes() != 0) == (expected_allocations == 1));
    CHECK(get_cache_memory() == cached);
  }
  record->finish(false); record_guard.value = nullptr;
  role.scope->seal();
  gate->open.store(true);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (role.scope->query().activity == submission::Activity::pending) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    REQUIRE(role.scope->progress_scoped() == submission::ScopedProgress::observed);
    std::this_thread::yield();
  }
  REQUIRE(role.scope->retire_scoped_records() == submission::ScopedProgress::observed);
  CHECK_FALSE(role.scope->query().failed);
  CHECK(output.data<float>()[0] == 10.0f);
  CHECK(output.data<float>()[1] == 14.0f);
  CHECK(output.data<float>()[2] == 22.0f);
  CHECK(fixture.budget->occupied_bytes() == 0);
  CHECK(fixture.counts->allocated == expected_allocations);
  CHECK(fixture.counts->freed == expected_allocations);
  CHECK(get_cache_memory() == cached);
}
