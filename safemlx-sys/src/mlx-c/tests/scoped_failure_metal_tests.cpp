#include <thread>
#include <exception>
#include "mlx/c/submission.h"
#include "doctest/doctest.h"
#include "mlx/failure.h"
#include "mlx/c/prefill_failure.h"
#include "mlx/c/prefill_roots.h"
#include "mlx/c/private/array.h"
#include "mlx/submission.h"
#include "mlx/ops.h"
#include "mlx/backend/metal/device.h"
#include <array>
#include <atomic>
#include <chrono>
#include <vector>
using namespace mlx::core;
namespace {
struct Pool {
  NS::AutoreleasePool* value{NS::AutoreleasePool::alloc()->init()};
  ~Pool() { value->drain(); }
};
void retired(void* raw) { ++*static_cast<std::atomic<unsigned>*>(raw); }
struct FailureHandle {
  mlx_prefill_failure value{};
  ~FailureHandle() { mlx_prefill_failure_free(value); }
};
void capture(FailureHandle& handle, std::vector<uint16_t>& units, bool empty_domain) {
  Pool pool;
  auto* text = units.empty() ? NS::String::alloc()->init()
      : NS::String::alloc()->init(units.data(), units.size() * sizeof(uint16_t),
          NS::UTF16LittleEndianStringEncoding, false);
  REQUIRE(text);
  auto* dictionary = NS::Dictionary::dictionary(text, NS::LocalizedDescriptionKey);
  auto* domain = NS::String::string(empty_domain ? "" : "original.native.domain", NS::UTF8StringEncoding);
  auto* error = NS::Error::error(domain, 4097, dictionary);
  REQUIRE(error);
  static_cast<FailureCarrier*>(handle.value.ctx)->capture_native_error(error);
  text->release();
  // The error, dictionary, domain and conversion context above all leave the
  // autorelease pool. Only the carrier's actual error/string retains survive.
}
std::vector<uint16_t> copied(mlx_prefill_failure value, unsigned field) {
  std::vector<uint16_t> result;
  size_t offset = 0;
  for (;;) {
    std::array<uint16_t, 64> part{};
    size_t used = 0, remaining = 0;
    REQUIRE(mlx_prefill_failure_native_text(&used, &remaining, value, field,
        offset, part.data(), part.size()) == 0);
    REQUIRE(used <= part.size());
    result.insert(result.end(), part.begin(), part.begin() + used);
    offset += used;
    if (!remaining) return result;
    REQUIRE(used != 0);
  }
}
}
TEST_CASE("scoped native error text survives autorelease drainage and split surrogate copies") {
  std::atomic<unsigned> count{0};
  {
    FailureHandle owner;
    REQUIRE(mlx_prefill_failure_new_retaining(&owner.value, &count, retired) == 0);
    std::vector<uint16_t> units(63, 'x');
    units.insert(units.end(), {0xd83d, 0xde42, 0, 'z'});
    capture(owner, units, false);
    for (unsigned iteration = 0; iteration < 3; ++iteration) {
      { Pool drained; }
      CHECK(copied(owner.value, 0) == units);
      const std::vector<uint16_t> domain{'o','r','i','g','i','n','a','l','.','n','a','t','i','v','e','.','d','o','m','a','i','n'};
      CHECK(copied(owner.value, 1) == domain);
      mlx_prefill_failure_view view{};
      REQUIRE(mlx_prefill_failure_view_get(&view, owner.value) == 0);
      CHECK(view.kind == unsigned(FailureKind::native));
      CHECK(view.native_code == 4097);
      CHECK(view.message == nullptr);
      CHECK(view.source_type == nullptr);
    }
    CHECK(count == 0);
  }
  CHECK(count == 1);
}
TEST_CASE("scoped native error preserves empty description and domain") {
  std::atomic<unsigned> count{0};
  {
    FailureHandle owner;
    REQUIRE(mlx_prefill_failure_new_retaining(&owner.value, &count, retired) == 0);
    std::vector<uint16_t> empty;
    capture(owner, empty, true);
    CHECK(copied(owner.value, 0).empty());
    CHECK(copied(owner.value, 1).empty());
    CHECK(count == 0);
  }
  CHECK(count == 1);
}

TEST_CASE("scoped Metal roots observe actual committed work after scope sealing") {
  const auto stream = new_stream(Device::gpu);
  std::atomic<unsigned> count{0};
  struct Owners {
    submission::GraphQuota* graph{submission::GraphQuota::try_create(1 << 20, nullptr, nullptr)};
    mlx_prefill_roots roots{};
    submission::Scope* scope{nullptr};
    ~Owners() {
      if (scope) { scope->seal(); scope->release(); }
      mlx_prefill_roots_free(roots);
      if (graph) graph->release();
    }
  };
  {
    FailureHandle failure;
    REQUIRE(mlx_prefill_failure_new_retaining(&failure.value, &count, retired) == 0);
    Owners owner;
    REQUIRE(owner.graph);
    REQUIRE(mlx_prefill_roots_new_owned(&owner.roots, 1,
        mlx_submission_graph_quota{owner.graph}, failure.value) == 0);
    owner.scope = new submission::Scope(nullptr, nullptr, nullptr, owner.graph);
    REQUIRE(owner.scope->enable_scoped_observation());
    const mlx_submission_scope scope{owner.scope};
    REQUIRE(mlx_prefill_roots_bind(owner.roots, scope) == 0);
    auto source = array({0.5f, -2.0f, 3.75f});
    auto output = add(source, array(2.0f), stream);
    REQUIRE(mlx_prefill_roots_append(owner.roots, mlx_array{&output, nullptr}) == 0);
    REQUIRE(mlx_prefill_roots_submit_scoped(owner.roots, scope) == 0);
    owner.scope->seal();
    // This path cannot commit an open command buffer: success requires the
    // submitted actual receipt to be observable, including its callback frontier.
    REQUIRE(mlx_prefill_roots_wait_scoped(owner.roots, scope) == 0);
    REQUIRE(mlx_prefill_roots_validate_scoped(owner.roots, scope) == 0);
    CHECK(output.data<float>()[0] == 2.5f);
    CHECK(output.data<float>()[1] == 0.0f);
    CHECK(output.data<float>()[2] == 5.75f);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (owner.scope->query_records().pending) {
      REQUIRE(owner.scope->progress_scoped() == submission::ScopedProgress::observed);
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    CHECK(count == 0);
  }
  submission::progress_records();
  submission::retire_records();
  CHECK(count == 1);
}

TEST_CASE("scoped Metal progress refuses an open receipt and distinguishes its earlier frontier") {
  const auto stream = new_stream(Device::gpu);
  auto encoder = metal::get_command_encoder_owner(stream);
  struct ReceiptRecord final : submission::Record {
    explicit ReceiptRecord(Allocation value) : Record(value) {}
    bool scoped_observation_supported() const noexcept override { return true; }
  };
  auto* scope = new submission::Scope;
  struct Release {
    submission::Scope* value;
    ~Release() { value->seal(); value->release(); }
  } release{scope};
  REQUIRE(scope->enable_scoped_observation());
  auto pending = submission::Record::create<ReceiptRecord>();
  pending->enter();
  auto* record = pending.release();
  record->prepare_stream(stream);
  encoder->record_submission();
  const auto frontier = encoder->submission_progress();
  auto* const buffer = encoder->get_command_buffer();
  REQUIRE(frontier.accepted > frontier.completed);
  CHECK(record->wait_for_scoped_progress() == submission::ScopedProgress::funded_progress);
  CHECK(encoder->get_command_buffer() == buffer);
  CHECK(encoder->submission_progress().accepted == frontier.accepted);
  CHECK(encoder->observation_requirement(frontier.accepted + 1) == gpu::ObservationRequirement::unobservable);
  record->finish(false);
  CHECK(scope->progress_scoped() == submission::ScopedProgress::funded_progress);
  CHECK(scope->query_records().pending);
  CHECK(encoder->get_command_buffer() == buffer);
  scope->seal();
  // Explicit ordinary fixture preparation resumes the exact receipt. The
  // refused scoped observer above never commits or manufactures a receipt.
  encoder->end_encoding();
  encoder->commit();
  encoder->record_submission();
  const auto later = encoder->submission_progress();
  REQUIRE(later.accepted > frontier.accepted);
  CHECK(encoder->observation_requirement(frontier.accepted) == gpu::ObservationRequirement::observe);
  CHECK(encoder->observation_requirement(later.accepted) == gpu::ObservationRequirement::funded_progress);
  encoder->end_encoding();
  encoder->commit();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scope->query_records().pending) {
    REQUIRE(scope->progress_scoped() == submission::ScopedProgress::observed);
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  submission::retire_records();
}

TEST_CASE("original Metal Event controls complete nonzero work after autorelease pools drain") {
  auto stream = new_stream(Device::gpu);
  // Run this exact case in a fresh process as well as in the complete suite.
  // No Event, task or numerical graph precedes actual selected initialization.
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&stream},
      mlx_stream{&stream}) == 0);
  CHECK(baseline.selected_streams == 1);
  CHECK(baseline.cpu_workers == 0);
  CHECK(baseline.native_threads == 0);
  CHECK(baseline.worker_object_bytes == 0);
  CHECK(baseline.scheduler_object_bytes > 0);
  CHECK((baseline.unpriced_populations & 1) == 1);
  std::atomic<unsigned> count{0};
  struct Owners {
    submission::GraphQuota* graph{submission::GraphQuota::try_create(1 << 20, nullptr, nullptr)};
    submission::RecordQuota* records{
        submission::RecordQuota::create(1 << 20, nullptr, nullptr)};
    mlx_prefill_roots roots{};
    submission::Scope* scope{nullptr};
    ~Owners() {
      if (scope) { scope->seal(); scope->release(); }
      mlx_prefill_roots_free(roots);
      if (graph) graph->release();
      if (records) records->release();
    }
  };
  {
    FailureHandle failure;
    REQUIRE(mlx_prefill_failure_new_retaining(&failure.value, &count, retired) == 0);
    Owners owner;
    REQUIRE(owner.graph);
    REQUIRE(mlx_prefill_roots_new_owned(&owner.roots, 1,
        mlx_submission_graph_quota{owner.graph}, failure.value) == 0);
    owner.scope = new submission::Scope(nullptr, nullptr, owner.records, owner.graph);
    REQUIRE(owner.scope->enable_scoped_observation());
    REQUIRE(owner.scope->require_original_controls() == submission::NativeControlFailure::none);
    const mlx_submission_scope scope{owner.scope};
    REQUIRE(mlx_prefill_roots_bind(owner.roots, scope) == 0);
    REQUIRE(owner.scope->enable_original_controls() == submission::NativeControlFailure::none);
    auto source = array({0.5f, -2.0f, 3.75f});
    auto output = add(source, array(2.0f), stream);
    REQUIRE(mlx_prefill_roots_append(owner.roots, mlx_array{&output, nullptr}) == 0);
    REQUIRE(mlx_prefill_roots_submit_scoped(owner.roots, scope) == 0);
    owner.scope->seal();
    // This path cannot commit an open command buffer: success requires the
    // submitted actual receipt to be observable, including its callback frontier.
    REQUIRE(mlx_prefill_roots_wait_scoped(owner.roots, scope) == 0);
    REQUIRE(mlx_prefill_roots_validate_scoped(owner.roots, scope) == 0);
    for (unsigned i = 0; i < 3; ++i) {
      { Pool drained; }
      CHECK(mlx_prefill_roots_query_scoped(owner.roots, scope) == 0);
    }
    CHECK(output.data<float>()[0] == 2.5f);
    CHECK(output.data<float>()[1] == 0.0f);
    CHECK(output.data<float>()[2] == 5.75f);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (owner.scope->query_records().pending) {
      REQUIRE(owner.scope->progress_scoped() == submission::ScopedProgress::observed);
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    CHECK(count == 0);
  }
  submission::progress_records();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (submission::try_retire_records() == submission::RetirementPass::busy) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  CHECK(count == 1);
}

TEST_CASE("original synchronous Metal evaluation first use uses only selected initialized runtime") {
  // Must also run this exact case in a fresh process: no preceding inference,
  // Event, numerical allocation or test warmup is an initialization prerequisite.
  Pool pool;
  const auto stream = new_stream(Device::gpu);
  auto prepared_stream = stream;
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&prepared_stream}, mlx_stream{&prepared_stream}) == 0);
  REQUIRE(baseline.selected_streams == 1);
  REQUIRE(baseline.cpu_workers == 0);
  REQUIRE(baseline.scheduler_object_bytes > 0);
  auto* graph = submission::GraphQuota::try_create(1 << 20, nullptr, nullptr);
  auto* records = submission::RecordQuota::create(1 << 20, nullptr, nullptr);
  REQUIRE(graph); REQUIRE(records);
  std::atomic<unsigned> count{0};
  {
    FailureHandle failure;
    REQUIRE(mlx_prefill_failure_new_retaining(&failure.value, &count, retired) == 0);
    auto* scope = new submission::Scope(nullptr, nullptr, records, graph);
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == submission::NativeControlFailure::none);
    REQUIRE(mlx_prefill_failure_bind_original_scope(failure.value, mlx_submission_scope{scope}) == 0);
    REQUIRE(scope->enable_original_controls() == submission::NativeControlFailure::none);
    {
      auto output = add(array({2.0f, -3.0f, 7.0f}), array(4.0f), stream);
      FailureHandle returned;
      REQUIRE(mlx_array_eval_scoped(mlx_array{&output, nullptr}, &returned.value) == 0);
      CHECK(returned.value.ctx == failure.value.ctx);
      CHECK(output.status() == array::Status::available);
      CHECK(output.data<float>()[0] == 6.0f);
      CHECK(output.data<float>()[1] == 1.0f);
      CHECK(output.data<float>()[2] == 11.0f);
    }
    scope->seal();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scope->query_records().pending) {
      CHECK(scope->progress_scoped() == submission::ScopedProgress::observed);
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    scope->release();
  }
  submission::progress_records(); submission::retire_records();
  graph->release(); records->release();
  CHECK(count == 1);
}

// The same two-window nonzero transfer contract runs on an actual Metal stream.
void original_host_copy_roundtrip(mlx::core::Device::DeviceType);
TEST_CASE("original native Metal host copies preserve values and reclaim an active role window") {
  Pool pool;
  original_host_copy_roundtrip(Device::gpu);
}

void original_selected_stream_frontier(mlx::core::Device::DeviceType);
void original_completed_array_validation(mlx::core::Device::DeviceType);
void original_exact_root_frontiers(mlx::core::Device::DeviceType);
TEST_CASE("exact original Metal root buffer supports empty identity and nonzero duplicate frontiers") {
  Pool pool;
  original_exact_root_frontiers(Device::gpu);
}
TEST_CASE("original Metal completed array validation preserves refusal and detaches after seal") {
  Pool pool;
  original_completed_array_validation(Device::gpu);
}
void original_prepared_global_stream_frontier(mlx::core::Stream);
TEST_CASE("original Metal selected stream closes empty identity and foreign-stream result frontiers") {
  Pool pool;
  original_selected_stream_frontier(Device::gpu);
  auto stream = new_thread_unsafe_stream(Device::gpu);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  std::exception_ptr failure;
  std::thread fresh([stream, &failure] {
    try {
      Pool pool;
      original_prepared_global_stream_frontier(stream);
    } catch (...) { failure = std::current_exception(); }
  });
  fresh.join();
  if (failure) std::rethrow_exception(failure);
}
