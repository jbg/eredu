// Included after original_buffers.cpp: the counting allocator delegates every
// successful physical allocation/free to the real selected CPU/Metal allocator.
#include "mlx/prepared_input.h"
#include <cstdlib>
#include <cstring>
#include "mlx/c/original_buffer.h"

namespace {
struct PairHandleDelete {
  void operator()(PreparedInputArray* value) const noexcept { if (value) value->destroy(); }
};
using PairHandle = std::unique_ptr<PreparedInputArray, PairHandleDelete>;
struct PairRole {
  FailureCarrierRef failure{data_failure_owner()};
  std::unique_ptr<submission::RecordQuota, DataRecordRelease> records;
  std::unique_ptr<submission::Scope, DataScopeRelease> scope;
  PairRole(submission::GraphQuota* graph, allocator::OriginalBufferBudget& budget,
      size_t minimum) : records(submission::RecordQuota::create(minimum, nullptr, nullptr)),
      scope(new submission::Scope(nullptr, nullptr, records.get(), graph)) {
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == submission::NativeControlFailure::none);
    REQUIRE(scope->bind_failure(failure));
    REQUIRE(scope->enable_original_controls() == submission::NativeControlFailure::none);
    scope->bind_original_buffer_budget(budget);
  }
};
struct PairPlan {
  allocator::PreparedInputFacts facts{};
  OriginalMutablePairLayout layout{};
  bool qualified{false};
  PairPlan() {
    REQUIRE(allocator::allocator().prepare_input_runtime(facts));
    qualified = original_mutable_pair_layout(facts, layout);
    if (!qualified) {
      CHECK(layout.graph_requests == 0);
      CHECK(layout.metadata_bytes == 0);
      CHECK(submission::submission_context_empty());
      const char* required = std::getenv("EREDU_REQUIRE_MUTABLE_PAIR_QUALIFICATION");
      REQUIRE((!required || std::strcmp(required, "1") != 0));
      MESSAGE("mutable-pair qualification=native-layout-unknown; no owner preparation");
    }
  }
};
PreparedHostCopyCause pair_call(BufferFixture& fixture, PairRole& role,
    const PairPlan& plan, const uint32_t* input, PairHandle& output) {
  PreparedInputArray* raw = nullptr;
  const auto status = construct_original_mutable_pair(*fixture.physical, plan.facts,
      fixture.graph.get(), *role.scope, *fixture.budget, input, raw);
  output.reset(raw);
  return status;
}
}

TEST_CASE("mutable pair exact controls produce nonzero mutable birth surviving private scope") {
  PairPlan plan;
  if (!plan.qualified) return;
  REQUIRE(plan.layout.constant_registry);
  CHECK(plan.layout.graph_requests == 5);
  CHECK(plan.layout.copy_bytes == sizeof(uint32_t) * 2);
  CHECK(plan.layout.backing_bytes >= plan.layout.copy_bytes);
  CHECK(plan.layout.module_bytes == submission::submission_runtime_layout().module_bytes);
  CHECK(plan.layout.thread_bytes == submission::submission_runtime_layout().thread_bytes);
  const uint32_t values[] = {17, 0xf1234567u};
  const array ordinary(values, Shape{2}, uint32);
  std::optional<array> alias;
  std::shared_ptr<BufferCounts> counts;
  uint64_t generation = 0;
  {
    BufferFixture fixture(plan.layout.backing_bytes, plan.layout.metadata_bytes);
    counts = fixture.counts;
    PairHandle output;
    {
      PairRole role(fixture.graph.get(), *fixture.budget, plan.layout.record_minimum_capacity);
      CHECK_FALSE(submission::submission_context_empty());
      REQUIRE(pair_call(fixture, role, plan, values, output) == PreparedHostCopyCause::success);
      REQUIRE(output);
      CHECK(role.scope->query().activity == submission::Activity::none);
      CHECK_FALSE(role.failure.get()->borrow());
      CHECK(fixture.counts->allocated == 1);
      CHECK(fixture.budget->occupied_bytes() == plan.layout.backing_bytes);
      const auto& value = output->value();
      CHECK(value.dtype() == uint32);
      CHECK(value.shape() == Shape{2});
      CHECK(value.data<uint32_t>()[0] == ordinary.data<uint32_t>()[0]);
      CHECK(value.data<uint32_t>()[1] == ordinary.data<uint32_t>()[1]);
      CHECK(value.data_shared_ptr()->original_input == nullptr);
      CHECK(value.buffer().original_buffer_budget() == fixture.budget.get());
      CHECK(value.buffer().original_allocation_capacity() == plan.layout.backing_bytes);
      generation = value.buffer().original_allocation_generation();
      REQUIRE(generation != 0);
      alias.emplace(value);
    }
    CHECK(submission::submission_context_empty());
    output.reset();
    CHECK(counts->freed == 0);
  }
  CHECK(counts->budget_retired == 0);
  CHECK(counts->graph_retired == 0);
  REQUIRE(alias);
  CHECK(alias->buffer().original_allocation_generation() == generation);
  CHECK(alias->data<uint32_t>()[1] == 0xf1234567u);
  alias.reset();
  CHECK(counts->freed == 1);
  CHECK(counts->budget_retired == 1);
  CHECK(counts->graph_retired == 1);
}

TEST_CASE("mutable pair one short metadata and backing refuse before real physical allocation") {
  PairPlan plan;
  if (!plan.qualified) return;
  const uint32_t values[] = {31, 47};
  for (bool short_metadata : {true, false}) {
    BufferFixture fixture(plan.layout.backing_bytes - (short_metadata ? 0 : 1),
        plan.layout.metadata_bytes - (short_metadata ? 1 : 0));
    PairRole role(fixture.graph.get(), *fixture.budget, plan.layout.record_minimum_capacity);
    PairHandle output;
    const auto status = pair_call(fixture, role, plan, values, output);
    CHECK_FALSE(output);
    CHECK(fixture.counts->allocated == 0);
    CHECK(fixture.budget->occupied_bytes() == 0);
    CHECK(fixture.graph->occupied_bytes() == 0);
    CHECK(role.scope->query().activity == submission::Activity::none);
    if (short_metadata) {
      CHECK(status == PreparedHostCopyCause::capacity);
      CHECK_FALSE(role.failure.get()->borrow());
    } else {
      CHECK(status == PreparedHostCopyCause::failed);
      const auto* failure = role.failure.get()->borrow();
      REQUIRE(failure);
      REQUIRE(failure->exception);
      try { std::rethrow_exception(failure->exception); FAIL_CHECK("retained original cause"); }
      catch (const allocator::OriginalBufferError& cause) {
        CHECK(cause.cause() == allocator::OriginalBufferCause::capacity);
      }
    }
  }
}

TEST_CASE("mutable pair physical refusal retains real cause and rolls back unpublished prefix") {
  PairPlan plan;
  if (!plan.qualified) return;
  const uint32_t values[] = {59, 61};
  for (auto refusal : {BufferAllocator::Refusal::before, BufferAllocator::Refusal::after}) {
    BufferFixture fixture(plan.layout.backing_bytes, plan.layout.metadata_bytes);
    fixture.physical->refusal = refusal;
    PairRole role(fixture.graph.get(), *fixture.budget, plan.layout.record_minimum_capacity);
    PairHandle output;
    CHECK(pair_call(fixture, role, plan, values, output) == PreparedHostCopyCause::failed);
    CHECK_FALSE(output);
    CHECK(fixture.counts->allocated == 1);
    CHECK(fixture.counts->freed == (refusal == BufferAllocator::Refusal::after ? 1 : 0));
    CHECK(fixture.budget->occupied_bytes() == 0);
    CHECK(fixture.graph->occupied_bytes() == 0);
    const auto* failure = role.failure.get()->borrow();
    REQUIRE(failure);
    REQUIRE(failure->exception);
    try { std::rethrow_exception(failure->exception); FAIL_CHECK("retained actual refusal"); }
    catch (const allocator::OriginalBufferError& cause) {
      CHECK(cause.cause() == (refusal == BufferAllocator::Refusal::before ?
          allocator::OriginalBufferCause::busy : allocator::OriginalBufferCause::capacity));
    }
    // The retained failure blocks a second call before allocator access, even
    // though rollback returned its unused physical and metadata debit.
    CHECK(pair_call(fixture, role, plan, values, output) == PreparedHostCopyCause::failed);
    CHECK(fixture.counts->allocated == 1);
  }
}

TEST_CASE("mutable pair context checks reject nested and entered record without new payload") {
  PairPlan plan;
  if (!plan.qualified) return;
  const auto stream = new_stream(Device::cpu);
  prepare_default_streams(stream, stream);
  struct WorkerResult {
    std::atomic<bool> entered{false}, nonempty{false}, isolated{true}, output{true};
    std::atomic<PreparedHostCopyCause> cause{PreparedHostCopyCause::success};
  };
  // Callback state is prepared cold. The queued closure owns every native
  // argument, so an assertion unwind cannot leave dangling host-stack borrows.
  auto worker_result = std::make_shared<WorkerResult>();
  const uint32_t values[] = {71, 73};
  REQUIRE(submission::submission_context_empty());
  // This negative context fixture separately permits a real Record capture.
  // Its larger test arena is not used as the exact producer certificate.
  BufferFixture fixture(plan.layout.backing_bytes, 64 * 1024);
  PairRole role(fixture.graph.get(), *fixture.budget, 64 * 1024);
  CHECK(role.scope->isolated_current());
  PairHandle output;
  {
    Scope nested(fixture.graph.get());
    CHECK_FALSE(role.scope->isolated_current());
    CHECK_FALSE(submission::submission_context_empty());
    CHECK(pair_call(fixture, role, plan, values, output) == PreparedHostCopyCause::domain);
  }
  CHECK(role.scope->isolated_current());
  auto pending = submission::Record::create<DataCaptureRecord>();
  pending->enter();
  auto* record = pending.release(); // registry now owns final destruction
  {
    BufferRecordFinish finish{record};
    record->begin_primitive();
    CHECK_FALSE(role.scope->isolated_current());
    CHECK_FALSE(submission::submission_context_empty());
    CHECK(pair_call(fixture, role, plan, values, output) == PreparedHostCopyCause::domain);
    record->end_primitive();
    record->reserve_streams(1);
    record->prepare_stream(stream);
    {
      submission::RecordDispatchGuard dispatch(*record);
      CHECK_FALSE(role.scope->isolated_current());
      CHECK(pair_call(fixture, role, plan, values, output) == PreparedHostCopyCause::domain);
      // An observer alias releases only its reference on a foreign worker.
      // PairRole retains the sole host-thread seal/lexical unlink authority.
      auto release_observer = [](submission::Scope* scope) noexcept {
        if (scope) scope->release();
      };
      auto retained_scope = std::unique_ptr<submission::Scope, decltype(release_observer)>(
          submission::Scope::retain_current_original(), release_observer);
      REQUIRE(retained_scope);
      scheduler::enqueue(stream,
          [worker_result, physical = fixture.physical,
           graph = submission::GraphQuotaRef(fixture.graph.get()),
           budget = allocator::OriginalBufferBudgetRef(fixture.budget.get()),
           scope = std::move(retained_scope), facts = plan.facts] {
            worker_result->nonempty = !submission::submission_context_empty();
            worker_result->isolated = scope->isolated_current();
            const uint32_t input[] = {71, 73};
            PreparedInputArray* raw = nullptr;
            worker_result->cause = construct_original_mutable_pair(*physical,
                facts, graph.get(), *scope, *budget.get(), input, raw);
            PairHandle output(raw);
            worker_result->output = bool(output);
            worker_result->entered = true;
          });
    }
    record->finish(false);
    finish.value = nullptr;
  }
  role.scope->seal();
  REQUIRE(submission::submission_context_empty());
  REQUIRE(settle_default_role(*role.scope));
  CHECK(worker_result->entered);
  CHECK(worker_result->nonempty);
  CHECK_FALSE(worker_result->isolated);
  CHECK(worker_result->cause == PreparedHostCopyCause::domain);
  CHECK_FALSE(worker_result->output);
  CHECK_FALSE(role.scope->query().failed);
  CHECK(submission::submission_context_empty());
  CHECK_FALSE(output);
  CHECK(fixture.counts->allocated == 0);
  CHECK(fixture.budget->occupied_bytes() == 0);
}
