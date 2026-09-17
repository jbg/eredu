// Included by graph_metadata_quota_tests.cpp; shares its actual Arena counters.
#include "mlx/failure.h"
#include "mlx/record_quota.h"
#include <vector>

namespace {
FailureCarrierRef data_failure_owner() {
  // Records may outlive a failed test assertion and this fixture's stack.
  auto token = std::make_unique<unsigned>(0);
  auto failure = FailureCarrier::create(token.get(), [](void* value) {
    delete static_cast<unsigned*>(value);
  });
  if (!failure) throw std::bad_alloc(); // token still owns allocation failure
  token.release();
  return failure;
}
struct DataRecordRelease {
  void operator()(submission::RecordQuota* value) const noexcept { value->release(); }
};
struct DataScopeRelease {
  void operator()(submission::Scope* value) const noexcept {
    value->seal(); value->release();
  }
};
struct DataRole {
  FailureCarrierRef failure{data_failure_owner()};
  std::unique_ptr<submission::RecordQuota, DataRecordRelease> records{
      submission::RecordQuota::create(64 * 1024, nullptr, nullptr)};
  std::unique_ptr<submission::Scope, DataScopeRelease> scope;
  explicit DataRole(submission::GraphQuota* graph)
      : scope(new submission::Scope(nullptr, nullptr, records.get(), graph)) {
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == submission::NativeControlFailure::none);
    REQUIRE(scope->bind_failure(failure));
    REQUIRE(scope->enable_original_controls() == submission::NativeControlFailure::none);
  }
};
struct DataCaptureRecord final : submission::Record {
  explicit DataCaptureRecord(Allocation allocation) : Record(allocation) {}
  bool scoped_observation_supported() const noexcept override { return true; }
};
allocator::Buffer data_buffer() {
  auto buffer = allocator::malloc(6 * sizeof(float));
  // The ordinary CPU allocator may return null; Metal's allocator throws for
  // nonzero failure and supplies shared storage. Check before initializing.
  if (!buffer.ptr()) throw std::bad_alloc();
  try {
    auto* values = static_cast<float*>(buffer.raw_ptr());
    if (!values) throw std::runtime_error("Data fixture backing is not host accessible");
    const float nonzero[] = {5, 7, 11, 13, 17, 19};
    std::copy(std::begin(nonzero), std::end(nonzero), values);
  } catch (...) {
    allocator::free(buffer);
    throw;
  }
  return buffer;
}
void install_data(array& value, allocator::Buffer incoming, bool strided,
                  Strides& strides, Deleter deleter) {
  if (strided) {
    value.set_data(incoming, 6, std::move(strides),
                   array::Flags{false, false, false}, sizeof(float),
                   std::move(deleter));
  } else {
    value.set_data(incoming, std::move(deleter));
  }
}
void check_data(const array& value, bool strided) {
  const auto* data = value.data<float>();
  CHECK(data[0] == (strided ? 7.0f : 5.0f));
  CHECK(data[strided ? 2 : 1] == (strided ? 13.0f : 7.0f));
  CHECK(data[strided ? 4 : 2] == (strided ? 19.0f : 11.0f));
  CHECK(value.data_size() == (strided ? 6 : 3));
  CHECK(value.strides()[0] == (strided ? 2 : 1));
  CHECK(value.offset() == (strided ? sizeof(float) : 0));
}
struct FilledDataArena {
  submission::GraphQuota* graph;
  std::vector<void*> blocks;
  explicit FilledDataArena(submission::GraphQuota* graph) : graph(graph) {
    // Test-only storage tracks real successful arena allocations, not a
    // substituted failure flag or a fit/capacity selection procedure.
    while (auto* block = graph->try_allocate(1, alignof(std::max_align_t)))
      blocks.push_back(block);
  }
  ~FilledDataArena() {
    for (auto* block : blocks) graph->deallocate(block, 1, alignof(std::max_align_t));
  }
};
struct DataCopyFailure final : std::exception {
  const char* what() const noexcept override { return "Data deleter copy probe"; }
};
struct DataCopyState { unsigned copies{0}, fail_at{0}, freed{0}; };
struct DataCopyDeleter {
  DataCopyState* state;
  explicit DataCopyDeleter(DataCopyState& state) : state(&state) {}
  DataCopyDeleter(const DataCopyDeleter& other) : state(other.state) {
    if (++state->copies == state->fail_at) throw DataCopyFailure{};
  }
  void operator()(allocator::Buffer buffer) const {
    ++state->freed;
    allocator::free(buffer);
  }
};
} // namespace

TEST_CASE("graph quota Data controls preserve ordinary and original set_data geometry") {
  for (bool strided : {false, true}) {
    for (bool original : {false, true}) {
      Counts counts;
      Arena arena(counts, 64 * 1024);
      unsigned freed = 0;
      std::optional<array> value(array({2.0f, 3.0f, 5.0f}));
      auto run = [&] {
        Strides strides{2};
        auto incoming = data_buffer();
        install_data(*value, incoming, strided, strides,
                     [&](allocator::Buffer buffer) { ++freed; allocator::free(buffer); });
        CHECK(value->buffer().ptr() == incoming.ptr());
        check_data(*value, strided);
        CHECK(value->is_donatable());
      };
      if (original) { DataRole role(arena.value); run(); }
      else { run(); }
      CHECK((arena.value->occupied_bytes() > 0) == original);
      CHECK(freed == 0);
      value.reset();
      CHECK(freed == 1);
      CHECK(arena.value->occupied_bytes() == 0);
    }
  }
}

TEST_CASE("graph quota Data exhaustion preserves old data and frees incoming once") {
  for (bool strided : {false, true}) {
    Counts counts;
    Arena arena(counts, 4096);
    auto value = array({2.0f, 3.0f, 5.0f});
    const auto old = value.data_shared_ptr();
    unsigned freed = 0;
    // Prepare the argument's actual allocator before exhausting its domain;
    // this exercises Data allocation, not a by-value Strides entry failure.
    Strides strides({2}, Strides::allocator_type(arena.value));
    DataRole role(arena.value);
    const auto baseline = arena.value->occupied_bytes();
    {
      FilledDataArena full(arena.value);
      REQUIRE(!full.blocks.empty());
      const auto occupied = arena.value->occupied_bytes();
      auto incoming = data_buffer();
      try {
        install_data(value, incoming, strided, strides,
                     [&](allocator::Buffer buffer) { ++freed; allocator::free(buffer); });
        FAIL_CHECK("Data control must refuse the physically exhausted Graph arena");
      } catch (const submission::GraphQuotaError& error) {
        CHECK(error.cause() == submission::GraphFailure::exhausted);
      }
      CHECK(freed == 1);
      CHECK(value.data_shared_ptr() == old);
      CHECK(value.data<float>()[0] == 2.0f);
      CHECK(value.data<float>()[2] == 5.0f);
      CHECK(value.offset() == 0);
      CHECK(value.strides()[0] == 1);
      // One stride is inline; no Data block or incoming-buffer owner
      // survives the failed attempt.
      CHECK(arena.value->occupied_bytes() == occupied);
    }
    CHECK(arena.value->occupied_bytes() == baseline);
  }
}

TEST_CASE("graph quota Data control retains its birth arena through aliases and final weak release") {
  Counts a_count, b_count;
  Arena a(a_count, 64 * 1024), b(b_count, 64 * 1024);
  unsigned a_freed = 0, b_freed = 0;
  bool a_guard_during_free = false, b_guard_during_free = false;
  std::optional<array> first(array({1.0f, 2.0f, 3.0f}));
  std::optional<array> second(array({1.0f, 2.0f, 3.0f}));
  std::optional<array> alias(array({1.0f, 2.0f, 3.0f}));
  std::weak_ptr<array::Data> weak_a, weak_b;
  {
    DataRole role(a.value);
    first->set_data(data_buffer(), [&](allocator::Buffer buffer) {
      a_guard_during_free = a_count.retired == 0;
      ++a_freed; allocator::free(buffer);
    });
    weak_a = first->data_shared_ptr();
    alias->copy_shared_buffer(*first);
  }
  const auto a_bytes = a.value->occupied_bytes();
  {
    DataRole role(b.value);
    Strides strides{2};
    install_data(*second, data_buffer(), true, strides, [&](allocator::Buffer buffer) {
      b_guard_during_free = b_count.retired == 0;
      ++b_freed; allocator::free(buffer);
    });
    weak_b = second->data_shared_ptr();
    CHECK(a.value->occupied_bytes() == a_bytes);
    REQUIRE(b.value->occupied_bytes() > 0);
    first.reset();
    CHECK(a_freed == 0); // independent descriptor still owns the same Data
  }
  a.reset(); b.reset();
  alias.reset(); second.reset();
  CHECK(a_freed == 1); CHECK(b_freed == 1);
  CHECK(a_guard_during_free); CHECK(b_guard_during_free);
  CHECK(weak_a.expired()); CHECK(weak_b.expired());
  CHECK(a_count.retired == 0); CHECK(b_count.retired == 0);
  // Drop A's final header while an unrelated B-owned header remains live.
  std::thread final_a([weak = std::move(weak_a)]() mutable { weak.reset(); });
  final_a.join();
  CHECK(a_count.retired == 1); CHECK(b_count.retired == 0);
  weak_b.reset();
  CHECK(b_count.retired == 1);
}

TEST_CASE("graph quota Data refuses an original-required missing domain without ordinary fallback") {
  for (bool strided : {false, true}) {
    auto value = array({2.0f, 3.0f, 5.0f});
    const auto old = value.data_shared_ptr();
    Strides strides{2};
    unsigned freed = 0;
    // Keep the Data-control oracle independent of default allocation admission.
    // This owner also covers assertion unwind before adoption takes custody.
    allocator::BufferOwner incoming(data_buffer());
    Scope scope;
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(scope.value->enable_original_controls() == submission::NativeControlFailure::missing_graph);
    try {
      auto refused = allocator::malloc_owned(6 * sizeof(float));
      FAIL_CHECK("required original default allocation cannot enter an ordinary allocator");
    } catch (const submission::NativeControlError& error) {
      CHECK(error.failure() == submission::NativeControlFailure::invalid_scope);
    }
    CHECK(freed == 0);
    CHECK(value.data_shared_ptr() == old);
    try {
      install_data(value, incoming.release(), strided, strides,
                   [&](allocator::Buffer buffer) { ++freed; allocator::free(buffer); });
      FAIL_CHECK("required original Data cannot choose an ordinary control block");
    } catch (const submission::NativeControlError& error) {
      CHECK(error.failure() == submission::NativeControlFailure::invalid_scope);
    }
    CHECK(freed == 1);
    CHECK(value.data_shared_ptr() == old);
  }
}

TEST_CASE("graph quota Data constructor copy failures return control storage and preserve cause") {
  for (bool strided : {false, true}) {
    for (unsigned fail_at : {1u, 2u}) {
      Counts counts;
      Arena arena(counts, 64 * 1024);
      auto value = array({2.0f, 3.0f, 5.0f});
      const auto old = value.data_shared_ptr();
      DataCopyState state;
      Deleter deleter = DataCopyDeleter(state);
      state.copies = 0; state.fail_at = fail_at;
      DataRole role(arena.value);
      Strides strides{2};
      const auto before = arena.value->occupied_bytes();
      auto incoming = data_buffer();
      CHECK_THROWS_AS(install_data(value, incoming, strided, strides,
                                  std::move(deleter)), DataCopyFailure);
      CHECK(state.copies == fail_at);
      CHECK(state.freed == 1);
      CHECK(value.data_shared_ptr() == old);
      CHECK(arena.value->occupied_bytes() == before);
    }
  }
}

TEST_CASE("graph quota Data uses an active Record birth domain after its Scope is sealed") {
  for (bool strided : {false, true}) {
    Counts counts;
    Arena arena(counts, 64 * 1024);
    std::optional<array> value(array({2.0f, 3.0f, 5.0f}));
    unsigned freed = 0;
    std::weak_ptr<array::Data> weak;
    {
      DataRole role(arena.value);
      auto pending = submission::Record::create<DataCaptureRecord>();
      pending->enter();
      auto* record = pending.release();
      record->begin_primitive();
      role.scope->seal();
      CHECK(submission::current_scope() == nullptr);
      CHECK(submission::current_graph_quota() == arena.value);
      Strides strides{2};
      install_data(*value, data_buffer(), strided, strides,
                   [&](allocator::Buffer buffer) { ++freed; allocator::free(buffer); });
      weak = value->data_shared_ptr();
      record->finish(true); // retained failed captures, no native dispatch in this case
      value.reset();
      CHECK_FALSE(weak.expired()); CHECK(freed == 0);
      REQUIRE(role.scope->progress_scoped() == submission::ScopedProgress::observed);
      REQUIRE(role.scope->retire_scoped_records() == submission::ScopedProgress::observed);
      CHECK(weak.expired()); CHECK(freed == 1);
    }
    arena.reset();
    CHECK(counts.retired == 0);
    weak.reset();
    CHECK(counts.retired == 1);
  }
}

TEST_CASE("graph quota Data shares the real nonzero evaluator without changing donation") {
  Counts counts;
  Arena arena(counts, 1 << 20);
  {
    Scope scope(arena.value); // existing graph-only ordinary mechanism remains valid
    std::optional<array> source(array({3.0f, -5.0f, 7.0f}));
    auto* backing = source->data<float>();
    auto output = negative(*source, Device::cpu);
    source.reset();
    auto completion = async_eval_with_completion({output});
    completion.wait();
    CHECK(output.data<float>() == backing);
    CHECK(output.data<float>()[0] == -3.0f);
    CHECK(output.data<float>()[1] == 5.0f);
    CHECK(output.data<float>()[2] == -7.0f);
    submission::progress_records();
    submission::retire_records();
  }
  submission::progress_records();
  submission::retire_records();
  CHECK(arena.value->occupied_bytes() == 0);
  arena.reset();
  CHECK(counts.retired == 1);
}

TEST_CASE("graph quota Data preserves capture refusal before its own allocation") {
  for (bool strided : {false, true}) {
    Counts counts;
    Arena arena(counts, 4096);
    auto value = array({2.0f, 3.0f, 5.0f});
    const auto old = value.data_shared_ptr();
    unsigned freed = 0;
    DataRole role(arena.value);
    auto pending = submission::Record::create<DataCaptureRecord>();
    pending->enter();
    auto* record = pending.release();
    record->begin_primitive();
    // Fill the actual Record capture vector, independently of Graph storage.
    try {
      for (;;) { auto slot = submission::prepare_capture(); }
    } catch (const submission::RecordQuotaError& error) {
      CHECK(error.cause() == submission::QuotaFailure::exhausted);
    }
    Strides strides{2};
    {
      FilledDataArena full(arena.value);
      auto incoming = data_buffer();
      try {
        install_data(value, incoming, strided, strides,
                     [&](allocator::Buffer buffer) { ++freed; allocator::free(buffer); });
        FAIL_CHECK("Record capture refusal must precede Data Graph exhaustion");
      } catch (const submission::RecordQuotaError& error) {
        CHECK(error.cause() == submission::QuotaFailure::exhausted);
      }
      CHECK(freed == 1);
      CHECK(value.data_shared_ptr() == old);
    }
    record->finish(true);
    REQUIRE(role.scope->progress_scoped() == submission::ScopedProgress::observed);
    REQUIRE(role.scope->retire_scoped_records() == submission::ScopedProgress::observed);
    CHECK(arena.value->occupied_bytes() == 0);
  }
}
