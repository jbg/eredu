#include "doctest/doctest.h"
#include "mlx/c/event.h"
#include "mlx/c/ops.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/mlx.h"
#include "mlx/submission.h"
#include <atomic>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

using namespace mlx::core;
namespace {
struct Evidence {
  std::atomic<int> arena_retired{0};
  std::atomic<int> records_destroyed{0};
  std::atomic<bool> terminal{false};
};
void retire_evidence(void* pointer) {
  std::unique_ptr<std::shared_ptr<Evidence>> owner(
      static_cast<std::shared_ptr<Evidence>*>(pointer));
  ++(*owner)->arena_retired;
}
struct Arena {
  std::shared_ptr<Evidence> evidence = std::make_shared<Evidence>();
  submission::RecordQuota* value;
  explicit Arena(size_t capacity) {
    auto owner = std::make_unique<std::shared_ptr<Evidence>>(evidence);
    value = submission::RecordQuota::create(capacity, owner.get(), retire_evidence);
    owner.release();
  }
  ~Arena() {
    evidence->terminal = true;
    submission::progress_records(false);
    submission::retire_records();
    if (value) value->release();
  }
};
struct Scope {
  submission::Scope* value;
  explicit Scope(submission::RecordQuota* quota = nullptr)
      : value(new submission::Scope(nullptr, nullptr, quota)) {}
  ~Scope() { if (value) { value->seal(); value->release(); } }
};
struct HeldRecord : submission::Record {
  HeldRecord(Allocation allocation, std::shared_ptr<Evidence> evidence)
      : Record(allocation), evidence(std::move(evidence)) {}
  ~HeldRecord() override { ++evidence->records_destroyed; }
  bool side_effects_terminal() const noexcept override { return evidence->terminal; }
  std::shared_ptr<Evidence> evidence;
};
HeldRecord* entered(const std::shared_ptr<Evidence>& evidence) {
  auto record = submission::Record::create<HeldRecord>(evidence);
  record->enter();
  return record.release();
}
} // namespace

TEST_CASE("submission recovery quota nested scopes share actual nonzero eval storage") {
  Arena arena(256 * 1024); // explicit fixture capacity, not a provider/default fit bound
  {
    Scope parent(arena.value);
    Scope child;
    CHECK(child.value->record_quota() == arena.value);
    auto x = add(array({1.0f, 3.0f, 7.0f}), array({2.0f, 5.0f, 11.0f}), Device::cpu);
    eval(x);
    CHECK(x.data<float>()[0] == 3.0f);
    CHECK(x.data<float>()[1] == 8.0f);
    CHECK(x.data<float>()[2] == 18.0f);
    submission::progress_records();
    submission::retire_records();
    CHECK(arena.value->occupied_bytes() == 0);
    Arena foreign(256 * 1024);
    CHECK_THROWS_AS(Scope(foreign.value), submission::RecordQuotaError);
    CHECK(submission::current_record_quota() == arena.value);
  }
  auto evidence = arena.evidence;
  arena.value->release();
  arena.value = nullptr;
  CHECK(evidence->arena_retired.load() == 1);
}

TEST_CASE("submission recovery quota retains occupied blocks until actual record destruction") {
  Arena arena(16 * 1024);
  {
    Scope scope(arena.value);
    size_t accepted = 0;
    while (true) {
      try {
        auto* record = entered(arena.evidence);
        record->finish(false);
        ++accepted;
      } catch (const submission::RecordQuotaError& error) {
        CHECK(error.cause() == submission::QuotaFailure::exhausted);
        break;
      }
    }
    REQUIRE(accepted > 1);
    const auto full = arena.value->occupied_bytes();
    submission::progress_records(false);
    submission::retire_records();
    CHECK(arena.value->occupied_bytes() == full);
    CHECK(arena.evidence->records_destroyed.load() == 0);
    CHECK_THROWS_AS(entered(arena.evidence), submission::RecordQuotaError);
    CHECK(arena.value->occupied_bytes() == full); // failed constructor returned its block
    arena.evidence->terminal = true;
    for (size_t i = 0; i < accepted; ++i) submission::progress_records(false);
    // Terminal observation alone has returned no physical block.
    CHECK(arena.value->occupied_bytes() == full);
    submission::retire_records();
    CHECK(arena.value->occupied_bytes() == 0);
    CHECK(arena.evidence->records_destroyed.load() == static_cast<int>(accepted));
    auto* reused = entered(arena.evidence);
    reused->finish(false);
    submission::progress_records(false);
    submission::retire_records();
    CHECK(arena.value->occupied_bytes() == 0);
  }
}

TEST_CASE("submission recovery quota capture refusal protects old data and incoming buffer") {
  auto values = array({3.0f, 5.0f, 7.0f});
  auto original = values.data_shared_ptr();
  Arena arena(16 * 1024);
  Scope scope(arena.value);
  auto* record = entered(arena.evidence);
  record->begin_primitive();
  // First capture contains genuine initialized Data, then indexed placeholders
  // fill this same record's bounded vector without losing earlier ownership.
  auto first = submission::prepare_capture();
  auto nested = submission::prepare_capture();
  nested.commit(original);
  first.commit(original);
  try {
    while (true) { auto slot = submission::prepare_capture(); }
  } catch (const submission::RecordQuotaError& error) {
    CHECK(error.cause() == submission::QuotaFailure::exhausted);
  }
  auto incoming_freed = std::make_shared<std::atomic<int>>(0);
  auto incoming = allocator::malloc(values.nbytes());
  try {
    values.set_data(incoming, [incoming_freed](allocator::Buffer buffer) {
      ++*incoming_freed;
      allocator::free(buffer);
    });
    FAIL_CHECK("capture reservation must refuse before replacing Data");
  } catch (const submission::RecordQuotaError& error) {
    CHECK(error.cause() == submission::QuotaFailure::exhausted);
  }
  CHECK(incoming_freed->load() == 1);
  CHECK(values.data_shared_ptr() == original);
  CHECK(values.data<float>()[0] == 3.0f);
  CHECK(values.data<float>()[2] == 7.0f);
  std::weak_ptr<array::Data> captured = original;
  original.reset();
  record->finish(true); // ends only the capture channel, retaining failed owners
  values = array({13.0f});
  CHECK_FALSE(captured.expired());
  arena.evidence->terminal = true;
  submission::progress_records(false);
  submission::retire_records();
  CHECK(captured.expired());
  CHECK(arena.value->occupied_bytes() == 0);
}

TEST_CASE("submission recovery quota refuses nonzero graph before tracked allocation growth") {
  Arena arena(4096);
  auto source = array({1.0f, 2.0f, 3.0f});
  auto value = source;
  for (int i = 0; i < 128; ++i) value = add(value, source, Device::cpu);
  {
    Scope scope(arena.value);
    CHECK_THROWS_AS(eval(value), submission::RecordQuotaError);
    CHECK(source.data<float>()[0] == 1.0f);
    CHECK(source.data<float>()[2] == 3.0f);
    submission::progress_records(false);
    submission::retire_records();
    CHECK(arena.value->occupied_bytes() == 0);
  }
}

TEST_CASE("submission recovery quota tracks actual eval and cross-stream wait records") {
  Arena arena(256 * 1024);
  Scope scope(arena.value);
  const auto producer = new_stream(Device::cpu);
  const auto consumer = new_stream(Device::cpu);
  auto value = multiply(array({2.0f, 3.0f, 5.0f}), array({7.0f, 11.0f, 13.0f}), producer);
  auto completion = async_eval_with_completion({value});
  completion.wait(consumer);
  CHECK(arena.value->occupied_bytes() > 0);
  completion.wait();
  synchronize(consumer);
  submission::progress_records();
  submission::retire_records();
  CHECK(value.data<float>()[0] == 14.0f);
  CHECK(value.data<float>()[1] == 33.0f);
  CHECK(value.data<float>()[2] == 65.0f);
  CHECK(arena.value->occupied_bytes() == 0);
}

#include "wait_record_layout_tests.cpp"

#include "eval_record_layout_tests.cpp"
#include "eval_traversal_tests.cpp"

#include "graph_construction_tests.cpp"

#include "eval_synchronizer_graph_tests.cpp"

#include "gpu_eval_prologue_tests.cpp"

#include "constant_pad_tests.cpp"

TEST_CASE("submission record physical request extent matches actual occupied blocks") {
  size_t first = 0, second = 0, request = 0;
  REQUIRE(submission::RecordQuota::minimum_capacity(request));
  const size_t align = alignof(std::max_align_t);
  REQUIRE(submission::RecordQuota::allocation_extent(request, align, first));
  REQUIRE(submission::RecordQuota::allocation_extent(512, align, second));
  const size_t capacity = first + second;
  auto* exact = submission::RecordQuota::create(capacity, nullptr, nullptr);
  auto* a = exact->allocate(request, align);
  CHECK(exact->occupied_bytes() == first);
  auto* b = exact->allocate(512, align);
  CHECK(exact->occupied_bytes() == capacity);
  exact->deallocate(b, 512, align);
  exact->deallocate(a, request, align);
  CHECK(exact->occupied_bytes() == 0);
  exact->release();
  auto* short_arena = submission::RecordQuota::create(capacity - 1, nullptr, nullptr);
  a = short_arena->allocate(request, align);
  CHECK_THROWS_AS(short_arena->allocate(512, align), submission::RecordQuotaError);
  CHECK(short_arena->occupied_bytes() == first);
  short_arena->deallocate(a, request, align);
  short_arena->release();
  size_t unchanged = 19;
  CHECK_FALSE(submission::RecordQuota::allocation_extent(SIZE_MAX, align, unchanged));
  CHECK_FALSE(submission::RecordQuota::allocation_extent(1, 0, unchanged));
  CHECK_FALSE(submission::RecordQuota::allocation_extent(1, align * 2, unchanged));
  CHECK(unchanged == 19);
}


TEST_CASE("fresh Record lifetime capacity survives fragmented retirement and failed prefixes") {
  constexpr std::array<size_t, 9> sizes{257, 4097, 113, 769, 250, 4098, 89, 1537, 8193};
  size_t extents = 0;
  for (size_t size : sizes) {
    size_t extent = 0;
    REQUIRE(submission::RecordQuota::allocation_extent(size, alignof(std::max_align_t), extent));
    extents += extent;
  }
  size_t capacity = 0;
  REQUIRE(submission::RecordQuota::fresh_capacity_for_extents(extents, capacity));
  REQUIRE(capacity > extents);
  for (unsigned retired = 0; retired < 16; ++retired) {
    Arena arena(capacity);
    std::array<void*, sizes.size()> pointers{};
    for (size_t i = 0; i < sizes.size(); ++i) {
      // Retire every possible subset of the initial constructor prefix before
      // later requests. Larger requests must survive holes that cannot fit them.
      if (i == 4) {
        for (size_t j = 0; j < i; ++j) if (retired & (1u << j)) {
          arena.value->deallocate(pointers[j], sizes[j], alignof(std::max_align_t));
          pointers[j] = nullptr;
        }
      }
      pointers[i] = arena.value->allocate(sizes[i], alignof(std::max_align_t));
      std::memset(pointers[i], static_cast<int>(i + 1), sizes[i]);
    }
    for (size_t i = 0; i < sizes.size(); ++i) if (pointers[i]) {
      const auto* bytes = static_cast<const unsigned char*>(pointers[i]);
      CHECK(bytes[0] == i + 1);
      CHECK(bytes[sizes[i] - 1] == i + 1);
      arena.value->deallocate(pointers[i], sizes[i], alignof(std::max_align_t));
    }
    CHECK(arena.value->occupied_bytes() == 0);
  }
  size_t unchanged = 73;
  const size_t overflow = std::numeric_limits<size_t>::max() & ~(alignof(std::max_align_t) - 1);
  CHECK_FALSE(submission::RecordQuota::fresh_capacity_for_extents(overflow, unchanged));
  CHECK(unchanged == 73);
  CHECK_FALSE(submission::RecordQuota::fresh_capacity_for_extents(1, unchanged));
  CHECK(unchanged == 73);
}
