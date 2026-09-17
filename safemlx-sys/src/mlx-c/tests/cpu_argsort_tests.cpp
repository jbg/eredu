#include "doctest/doctest.h"
#include "cpu_argsort_oracle.h"
#include "mlx/backend/cpu/argsort_f32.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

using namespace mlx::core;
namespace oracle = cpu_argsort_test;
extern size_t native_recovery_without_allocations(void (*)(void*), void*);

namespace {
void check_worker(const std::vector<float>& values) {
  const auto expected = oracle::ordinary(values.data(), 1, values.size());
  std::vector<uint32_t> output(values.size() * 3 + 1, 0xf00dcafe);
  cpu::detail::argsort_f32_row(values.data(), 1, output.data(), 3, values.size());
  std::vector<uint32_t> actual(values.size());
  bool guards = output.back() == 0xf00dcafe;
  for (size_t i = 0; i < values.size(); ++i) {
    actual[i] = output[3 * i];
    guards &= output[3 * i + 1] == 0xf00dcafe && output[3 * i + 2] == 0xf00dcafe;
  }
  CHECK(actual == expected);
  CHECK(guards);
}

void check_native(const array& row, const std::vector<float>& expected_values, Stream stream) {
  const auto expected = oracle::ordinary(expected_values.data(), 1, expected_values.size());
  for (const bool flattened : {false, true}) {
    auto sorted = flattened ? argsort(row, stream) : argsort(row, -1, stream);
    eval(sorted);
    CHECK(sorted.dtype() == uint32);
    REQUIRE(sorted.shape() == Shape{static_cast<int>(expected.size())});
    const std::vector<uint32_t> actual(sorted.data<uint32_t>(), sorted.data<uint32_t>() + sorted.size());
    CHECK(actual == expected);
  }
}

struct Gate {
  std::promise<void> release;
  std::shared_future<void> wait{release.get_future().share()};
  bool open = false;
  void finish() { if (!open) { open = true; release.set_value(); } }
  ~Gate() { finish(); }
};

struct Scope {
  submission::Scope* value;
  explicit Scope(submission::RecordQuota* records = nullptr, submission::GraphQuota* graph = nullptr)
      : value(new submission::Scope(nullptr, nullptr, records, graph)) {}
  ~Scope() { value->seal(); value->release(); }
};

bool settle(submission::Scope& scope) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    submission::progress_records(false);
    submission::retire_records();
    if (!scope.query_records().pending) return true;
    std::this_thread::yield();
  }
  return false;
}

bool retired(submission::Scope& scope, const std::weak_ptr<array::Data>& first,
             const std::weak_ptr<array::Data>& second) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    submission::progress_records(false);
    submission::retire_records();
    if (!scope.query_records().pending && first.expired() && second.expired()) return true;
    std::this_thread::yield();
  }
  return false;
}

// Inject a later synchronous consumer failure after the real Argsort task has
// been accepted. This is not a naturally occurring sort/allocator failure.
class AfterSortFailure : public UnaryPrimitive {
 public:
  explicit AfterSortFailure(Stream stream) : UnaryPrimitive(stream) {}
  void eval_cpu(const ArrayVector& inputs, array& output) override {
    output.copy_shared_buffer(inputs[0]);
    throw std::runtime_error("injected consumer failure after actual CPU Argsort");
  }
  void eval_gpu(const ArrayVector&, array&) override { std::abort(); }
  DEFINE_NAME(AfterSortFailure);
};
} // namespace

TEST_CASE("CPU argsort F32 worker matches ordinary stable special-value permutations") {
  const std::array<float, 8> alphabet{
      -2.0f, 2.0f, -0.0f, 0.0f,
      oracle::from_bits(0xff800000), oracle::from_bits(0x7f800000),
      oracle::from_bits(0x7fc00001), oracle::from_bits(0xffc00123)};
  size_t combinations = 1;
  for (size_t length = 0; length <= 5; ++length) {
    std::vector<float> values(length);
    for (size_t code = 0; code < combinations; ++code) {
      auto remaining = code;
      for (auto& value : values) {
        value = alphabet[remaining % alphabet.size()];
        remaining /= alphabet.size();
      }
      check_worker(values);
    }
    combinations *= alphabet.size();
  }
  for (const size_t length : {1, 2, 3, 17, 2048, 2049, 32768}) {
    for (unsigned profile = 0; profile < 7; ++profile) {
      check_worker(oracle::values(length, profile));
    }
  }
}

TEST_CASE("CPU argsort actual F32 views preserve signed strides and flattened results") {
  const auto stream = new_stream(Device::cpu);
  const std::vector<float> values{
      9.0f, oracle::from_bits(0x7fc00001), -0.0f, 5.0f,
      oracle::from_bits(0xff800000), 5.0f, 0.0f, oracle::from_bits(0xffc00123),
      -3.0f, oracle::from_bits(0x7f800000), 2.0f, -3.0f};
  const array source(values.data(), {static_cast<int>(values.size())}, float32);
  struct View { int width; int64_t stride; size_t offset; };
  for (const auto view : {View{12, 1, 0}, View{5, 2, 1}, View{6, -2, 11}, View{7, 0, 3}}) {
    auto row = as_strided(source, {view.width}, {view.stride}, view.offset, stream);
    eval(row);
    REQUIRE(row.strides()[0] == view.stride);
    std::vector<float> logical(view.width);
    for (int i = 0; i < view.width; ++i) {
      logical[i] = values[static_cast<int64_t>(view.offset) + i * view.stride];
    }
    check_native(row, logical, stream);
  }
  const auto matrix = reshape(source, {3, 4}, stream);
  auto flat = argsort(matrix, stream);
  eval(flat);
  const auto expected = oracle::ordinary(values.data(), 1, values.size());
  CHECK(std::vector<uint32_t>(flat.data<uint32_t>(), flat.data<uint32_t>() + flat.size()) == expected);
  // Nonselected multidimensional-axis and F64 paths retain their old worker.
  auto axes = argsort(matrix, 1, stream);
  eval(axes);
  for (size_t row = 0; row < 3; ++row) {
    const auto reference = oracle::ordinary(values.data() + 4 * row, 1, 4);
    CHECK(std::vector<uint32_t>(axes.data<uint32_t>() + 4 * row, axes.data<uint32_t>() + 4 * row + 4) == reference);
  }
  auto doubles = astype(source, float64, stream);
  check_native(doubles, values, stream);
  // These go through the compiled MLX translation unit, not just the inline
  // header test. In particular, compiler flags must preserve NaN semantics.
  for (const size_t width : {2, 17, 2048, 2049, 32768}) {
    const auto raw = oracle::values(width, 6);
    check_native(array(raw.data(), {static_cast<int>(width)}, float32), raw, stream);
  }
  CHECK_THROWS_AS(argsort(source, 1, stream), std::invalid_argument);
  CHECK_THROWS_AS(argsort(source, -2, stream), std::invalid_argument);
  for (size_t i = 0; i < values.size(); ++i) {
    CHECK(oracle::bits(source.data<float>()[i]) == oracle::bits(values[i]));
  }
}

TEST_CASE("CPU argsort empty row is safe and singleton keeps its original ID") {
  const auto stream = new_stream(Device::cpu);
  const float unused = 0;
  const array empty(&unused, Shape{0}, float32);
  for (const int axis : {0, -1}) {
    auto sorted = argsort(empty, axis, stream);
    eval(sorted);
    CHECK(sorted.shape() == Shape{0});
    CHECK(sorted.dtype() == uint32);
  }
  for (const float value : {3.0f, oracle::from_bits(0x7fc01234)}) {
    check_native(array({value}), {value}, stream);
  }
  // Null pointers are valid for the private worker's zero-element case.
  cpu::detail::argsort_f32_row(nullptr, -1, nullptr, 1, 0);
}

TEST_CASE("CPU argsort worker uses no auxiliary new allocations with active positive control") {
  auto values = oracle::values(32768, 6);
  std::vector<uint32_t> output(values.size());
  struct Call { const float* values; uint32_t* output; size_t length; } call{values.data(), output.data(), values.size()};
  const auto attempts = native_recovery_without_allocations([](void* p) {
    const auto& call = *static_cast<Call*>(p);
    cpu::detail::argsort_f32_row(call.values, 1, call.output, 1, call.length);
  }, &call);
  CHECK(attempts == 0);
  CHECK(output == oracle::ordinary(values.data(), 1, values.size()));
  bool refused = false;
  const auto positive = native_recovery_without_allocations([](void* p) {
    try {
      auto* storage = ::operator new(32);
      ::operator delete(storage);
    } catch (const std::bad_alloc&) { *static_cast<bool*>(p) = true; }
  }, &refused);
  CHECK(positive == 1);
  CHECK(refused);
}

TEST_CASE("CPU argsort pending completion and K view retain full original backing") {
  Scope scope;
  Gate gate;
  const auto stream = new_stream(Device::cpu);
  scheduler::enqueue(stream, [wait = gate.wait] { wait.wait(); });
  std::weak_ptr<array::Data> source_data, full_data;
  std::optional<array> selected;
  Completion completion;
  {
    auto values = oracle::values(257, 4);
    auto source = array(values.data(), {257}, float32);
    source_data = source.data_shared_ptr();
    auto sorted = argsort(source, stream);
    selected.emplace(slice(sorted, {250}, {257}, {1}, stream));
    completion = async_eval_with_completion({*selected});
    full_data = sorted.data_shared_ptr();
    CHECK(selected->data_shared_ptr() == sorted.data_shared_ptr());
    CHECK(selected->buffer_size() == sorted.buffer_size());
    CHECK(selected->buffer_size() >= 257 * sizeof(uint32_t));
    CHECK_FALSE(completion.is_complete());
  }
  CHECK_FALSE(source_data.expired());
  CHECK_FALSE(full_data.expired());
  // Failure-safe release occurs before any assertion that could abort scope.
  gate.finish();
  completion.wait();
  scope.value->seal();
  REQUIRE(settle(*scope.value));
  const auto values = oracle::values(257, 4);
  const auto expected = oracle::ordinary(values.data(), 1, values.size());
  CHECK(std::vector<uint32_t>(selected->data<uint32_t>(), selected->data<uint32_t>() + 7) ==
        std::vector<uint32_t>(expected.end() - 7, expected.end()));
  CHECK_FALSE(full_data.expired());
  selected.reset();
  completion = Completion{};
  REQUIRE(retired(*scope.value, source_data, full_data));
  CHECK(source_data.expired());
  CHECK(full_data.expired());
}

TEST_CASE("CPU argsort later consumer error preserves queued sort until real settlement") {
  Scope scope;
  Gate gate;
  const auto stream = new_stream(Device::cpu);
  scheduler::enqueue(stream, [wait = gate.wait] { wait.wait(); });
  std::weak_ptr<array::Data> source_data, sorted_data;
  std::optional<array> sorted;
  {
    auto source = array({3.0f, 7.0f, 7.0f, -2.0f});
    source_data = source.data_shared_ptr();
    sorted.emplace(argsort(source, stream));
    auto failing = array(Shape{4}, uint32, std::make_shared<AfterSortFailure>(stream), {*sorted});
    CHECK_THROWS_WITH_AS(async_eval_with_completion({failing}),
                        "injected consumer failure after actual CPU Argsort", std::runtime_error);
    sorted_data = sorted->data_shared_ptr();
  }
  scope.value->seal();
  const auto pending = scope.value->query_records();
  CHECK(pending.pending);
  CHECK(pending.failed);
  CHECK_FALSE(source_data.expired());
  CHECK_FALSE(sorted_data.expired());
  gate.finish();
  REQUIRE(settle(*scope.value));
  CHECK(std::vector<uint32_t>(sorted->data<uint32_t>(), sorted->data<uint32_t>() + 4) ==
        std::vector<uint32_t>{3, 0, 1, 2});
  sorted.reset();
  REQUIRE(retired(*scope.value, source_data, sorted_data));
  CHECK(source_data.expired());
  CHECK(sorted_data.expired());
}

TEST_CASE("CPU argsort actual graph and record quota refusals preserve source") {
  const auto stream = new_stream(Device::cpu);
  auto source = array({3.0f, -1.0f, 3.0f, 2.0f});
  const auto original = source.data_shared_ptr();
  // Small explicit test capacities force real allocator refusals. They are not
  // an admission formula and this does not claim a managed CPU operation.
  auto* graph = submission::GraphQuota::create(128, nullptr, nullptr);
  {
    Scope scope(nullptr, graph);
    bool refused = false;
    try { (void)argsort(source, stream); }
    catch (const submission::GraphQuotaError& error) {
      refused = true;
      CHECK(error.cause() == submission::GraphFailure::exhausted);
    }
    CHECK(refused);
    CHECK(graph->occupied_bytes() == 0);
    CHECK(source.data_shared_ptr() == original);
  }
  graph->release();
  auto sorted = argsort(source, stream);
  // Query the exact smallest valid arena before constructing it. This covers
  // the concrete Eval object and allocator headers, not its later containers.
  size_t capacity = submission::minimum_record_object_bytes();
  submission::RecordQuotaLayout layout{};
  while (!submission::RecordQuota::layout(capacity, layout)) {
    REQUIRE(capacity < static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()));
    ++capacity;
  }
  REQUIRE(capacity > 0);
  submission::RecordQuotaLayout below_minimum{};
  REQUIRE_FALSE(submission::RecordQuota::layout(capacity - 1, below_minimum));
  auto* records = submission::RecordQuota::create(capacity, nullptr, nullptr);
  {
    Scope scope(records);
    bool refused = false;
    try { eval(sorted); }
    catch (const submission::RecordQuotaError& error) {
      refused = true;
      CHECK(error.cause() == submission::QuotaFailure::exhausted);
    }
    CHECK(refused);
    scope.value->seal();
    CHECK(settle(*scope.value));
    CHECK(records->occupied_bytes() == 0);
    CHECK(source.data_shared_ptr() == original);
    CHECK(source.data<float>()[0] == 3.0f);
    CHECK(source.data<float>()[1] == -1.0f);
  }
  records->release();
}
