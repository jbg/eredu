#include "prepared_metal_fixture.h"
#include "mlx/backend/cpu/sampling_storage.h"
#include "mlx/backend/cpu/alias_storage.h"
#include "mlx/c/original_buffer.h"
#include <numeric>
// Included after the prepared Eval fixtures. Only changed host construction and
// its transition into the existing real Eval/worker path are exercised here.
#include "mlx/graph_construction.h"
namespace pointwise_graph_tests {
using namespace eval_record_facts;
using submission::GraphConstruction;
struct Bank {
  void* value{nullptr};
  ~Bank() { reset(); }
  void reset() { mlx_operation_event_finish_pointwise_graph(value); value = nullptr; }
};
struct Observer {
  mlx_submission_observer value{nullptr};
  Observer() { REQUIRE(mlx_submission_observer_current(&value) == 0); }
  ~Observer() { mlx_submission_observer_release(value); }
};
struct GraphHoles {
  submission::GraphQuota* quota;
  std::vector<Block> blocks;
  explicit GraphHoles(submission::GraphQuota* q) : quota(q) {
    blocks.reserve(q->capacity() / alignof(std::max_align_t) + 8);
  }
  void add(size_t bytes, size_t alignment) {
    auto* pointer = quota->try_allocate(bytes, alignment);
    REQUIRE(pointer);
    blocks.push_back({pointer, bytes, alignment});
  }
  void fill() {
    while (auto* pointer = quota->try_allocate(1, 1)) blocks.push_back({pointer, 1, 1});
  }
  void release(size_t index) {
    auto& b = blocks[index];
    quota->deallocate(b.pointer, b.bytes, b.alignment);
    b.pointer = nullptr;
  }
  ~GraphHoles() {
    for (const auto& b : blocks) if (b.pointer) quota->deallocate(b.pointer, b.bytes, b.alignment);
  }
};
void numerical(Device device, bool expect_cpu_source_refusal = false) {
  const auto stream = new_stream(device);
  prepare(stream, stream);
  // Rank 11/12 exercises real exact copies and grow-to-20 buffers; the cold
  // rank-21 envelope must satisfy them through smaller-request matching.
  Shape a_shape(11, 1), b_shape(12, 1), c_shape(12, 1);
  a_shape.back() = 2; b_shape[b_shape.size()-2] = 3;
  c_shape[c_shape.size()-2] = 3; c_shape.back() = 2;
  std::vector<mlx::core::float16_t> a_values{mlx::core::float16_t(2.f), mlx::core::float16_t(4.f)};
  std::vector<float> b_values{1.f, 3.f, 5.f}, c_values(6, 2.f);
  array left(a_values.begin(), a_shape, float16);
  array right(b_values.begin(), b_shape, float32);
  array scale(c_values.begin(), c_shape, float32);
  unsigned physical_retired = 0;
  struct Budget {
    mlx_original_buffer_budget value{};
    ~Budget() { mlx_original_buffer_budget_release(value); }
  } budget;
  if (expect_cpu_source_refusal) {
    mlx_prepared_input_runtime runtime{};
    REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
    mlx_original_buffer_population_layout physical{};
    // The actual cast and two binary outputs need at most three positive
    // births. A missing physical source must not mask the CPU Eval refusal.
    const size_t bytes = (a_values.size() + 2 * c_values.size()) * sizeof(float);
    REQUIRE(mlx_original_buffer_metal_population_layout_for(&physical, runtime, bytes, 3) == 0);
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime,
        physical.capacity, &physical_retired,
        [](void* owner) { ++*static_cast<unsigned*>(owner); }) == 0);
  }
  Role role;
  if (expect_cpu_source_refusal)
    REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
  Observer observer;
  Bank bank;
  REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value, observer.value, 2, 21) == 0);
  const auto reserved = role.graph->occupied_bytes();
  auto* native = static_cast<GraphConstruction*>(bank.value);
  const auto available = native->remaining();
  REQUIRE(available == 74);
  auto value = multiply(add(left, right, stream), scale, stream);
  CHECK(native->remaining() < available);
  // Physical occupancy cannot rise while all host requests use reserved blocks.
  CHECK(role.graph->occupied_bytes() <= reserved);
  CHECK(value.status() == array::Status::unscheduled);
  auto alias = value;
  bank.reset(); // worker allocations must not consume the exhausted host recipe
  CHECK(role.graph->occupied_bytes() > 0);
  Operation operation;
  operation.append(value);
  const mlx_operation_eval_traversal_limits limits{1, 14, 11, 13, 11, 1, 8};
  if (expect_cpu_source_refusal) {
    // Inspect the actual first cast and following Broadcast. The rank-21 cold
    // constructor envelope does not qualify every CPU Eval in this graph.
    const auto& addition = value.inputs()[0];
    const auto& broadcast = addition.inputs()[0];
    const auto& cast = broadcast.inputs()[0];
    REQUIRE(typeid(cast.primitive()) == typeid(AsType));
    REQUIRE(typeid(broadcast.primitive()) == typeid(Broadcast));
    cpu::CopyEvalStorage source;
    REQUIRE(cpu::copy_eval_storage(cast, source));
    CHECK_FALSE(cpu::alias_eval_layout(cpu::AliasOperation::Broadcast,
        cast.ndim(), broadcast.ndim(), false, source));
    CHECK_FALSE(cpu::alias_eval_storage(broadcast, source));
    REQUIRE(eval_traversal_tests::submit(operation, stream, limits) ==
        static_cast<unsigned>(ScopedEvaluation::failed));
    // Submission failure alone does not prove that the earlier cast's queued
    // task has released its borrowed source. Observe and retire its Record.
    settle(role);
    CHECK(role.scope->query_records().pending == 0);
    CHECK(role.records->occupied_bytes() == 0);
    REQUIRE(role.error.get()->borrow());
    const auto* failure = role.error.get()->borrow();
    REQUIRE(failure->exception_type == &typeid(submission::GraphQuotaError));
    try {
      std::rethrow_exception(failure->exception);
      FAIL("CPU source refusal must preserve its original typed cause");
    } catch (const submission::GraphQuotaError& error) {
      CHECK(error.cause() == submission::GraphFailure::invalid_layout);
    }
    return;
  }
  REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
  eval_traversal_tests::complete(role, operation, value);
  const float expected[] = {6.f, 10.f, 10.f, 14.f, 14.f, 18.f};
  for (size_t i = 0; i != 6; ++i) CHECK(value.data<float>()[i] == expected[i]);
  CHECK(alias.data_shared_ptr() == value.data_shared_ptr());
}
} // namespace pointwise_graph_tests

TEST_CASE("pointwise Graph layout distinguishes exact copy growth and zero operations"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  static_assert(sizeof(mlx_pointwise_graph_layout) == 41 * sizeof(size_t));
  static_assert(std::is_same_v<decltype(&mlx_operation_event_prepare_pointwise_graph),
      unsigned (*)(void**, mlx_submission_observer, size_t, size_t)>);
  mlx_pointwise_graph_layout layout{};
  REQUIRE(mlx_operation_event_pointwise_graph_layout(&layout, 2, 11));
  CHECK(layout.blocks == 74);
  CHECK(layout.request_bytes[7] == 11 * sizeof(Shape::value_type));
  CHECK(layout.request_bytes[8] == 20 * sizeof(Shape::value_type));
  CHECK(layout.request_bytes[9] == 20 * sizeof(Strides::value_type));
  CHECK(layout.request_counts[7] == 18);
  CHECK(layout.request_counts[8] == 2);
  CHECK(layout.request_counts[9] == 10);
  CHECK(layout.header_bytes > 0);
  CHECK(layout.slots_bytes > 0);
  size_t bytes = layout.header_bytes + layout.slots_bytes;
  for (size_t i = 0; i != 10; ++i) bytes += layout.request_bytes[i] * layout.request_counts[i];
  CHECK(layout.requested_bytes == bytes);
  const auto saved = layout;
  CHECK_FALSE(mlx_operation_event_pointwise_graph_layout(&layout, SIZE_MAX, 11));
  CHECK(std::memcmp(&layout, &saved, sizeof(layout)) == 0);
  CHECK_FALSE(mlx_operation_event_pointwise_graph_layout(&layout, 1, SIZE_MAX));
  CHECK(std::memcmp(&layout, &saved, sizeof(layout)) == 0);
  for (size_t rank : {size_t{0}, size_t{1}, size_t{10}}) {
    REQUIRE(mlx_operation_event_pointwise_graph_layout(&layout, 1, rank));
    CHECK(layout.blocks == 22);
    CHECK(layout.request_counts[7] == 0);
    CHECK(layout.request_counts[8] == 0);
    CHECK(layout.request_counts[9] == 0);
  }
  REQUIRE(mlx_operation_event_pointwise_graph_layout(&layout, 0, 0));
  CHECK(layout.blocks == 0);
  CHECK(layout.slots_bytes == 0);
  CHECK(layout.requested_bytes == layout.header_bytes);
}

TEST_CASE("pointwise Graph preflight rolls back real fragmented prefixes before construction"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  array input({2.f, 3.f});
  const auto identity = input.id();
  mlx_pointwise_graph_layout layout{};
  REQUIRE(mlx_operation_event_pointwise_graph_layout(&layout, 1, 0));
  for (size_t prefix : {size_t{0}, size_t{1}, size_t{2}, size_t{3}}) {
    CAPTURE(prefix);
    Role role;
    Observer observer;
    GraphHoles holes(role.graph.get());
    holes.add(layout.header_bytes, layout.header_alignment); holes.add(1, 1);
    holes.add(layout.slots_bytes, layout.slots_alignment); holes.add(1, 1);
    holes.add(layout.request_bytes[0], layout.reserved_alignment); holes.add(1, 1);
    holes.fill();
    for (size_t i = 0; i != prefix; ++i) holes.release(2*i);
    const auto before = role.graph->occupied_bytes();
    scheduler::CpuStreamToken worker;
    REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
    const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
    Bank bank;
    CHECK(mlx_operation_event_prepare_pointwise_graph(&bank.value, observer.value, 1, 0) == 2);
    CHECK(bank.value == nullptr);
    CHECK(role.graph->occupied_bytes() == before);
    CHECK(scheduler::cpu_stream_progress(worker).accepted == accepted);
    CHECK(input.id() == identity);
    CHECK(input.data<float>()[1] == 3.f);
    CHECK_FALSE(role.scope->query().failed);
  }
}

TEST_CASE("pointwise Graph bank refuses nested foreign and spent requests without fallback"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  Role role;
  Observer observer;
  const auto baseline = role.graph->occupied_bytes();
  Bank bank;
  REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value, observer.value, 0, 0) == 0);
  Bank nested;
  CHECK(mlx_operation_event_prepare_pointwise_graph(&nested.value, observer.value, 1, 0) == 10);
  CHECK(nested.value == nullptr);
  try {
    (void)role.graph->allocate(1, 1);
    FAIL("empty recipe must not allocate from remaining Graph capacity");
  } catch (const submission::GraphQuotaError& error) {
    CHECK(error.cause() == submission::GraphFailure::construction_mismatch);
  }
  {
    std::unique_ptr<submission::Scope, wait_record_facts::ReleaseScope> child(new submission::Scope());
    try {
      (void)role.graph->allocate(1, 1);
      FAIL("foreign child must not consume the parent host bank");
    } catch (const submission::GraphQuotaError& error) {
      CHECK(error.cause() == submission::GraphFailure::foreign_parent);
    }
  }
  bank.reset();
  CHECK(role.graph->occupied_bytes() == baseline);
  // A fresh once-owned bank remains valid after an intact refusal.
  REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value, observer.value, 0, 0) == 0);
  bank.reset();
  CHECK(role.graph->occupied_bytes() == baseline);
}

TEST_CASE("pointwise Graph high rank constructor does not authorize unsupported CPU Broadcast"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  pointwise_graph_tests::numerical(Device::cpu, true);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("pointwise Graph high rank cast broadcast owns prefix through Metal completion"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  pointwise_graph_tests::numerical(Device::gpu);
}
#endif
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 210106 || __cplusplus != 202002L
TEST_CASE("unqualified pointwise Graph producer preserves explicit unknown sentinel") {
  using namespace wait_record_facts;
  const auto* required = std::getenv("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT");
  CHECK_FALSE(required && std::strcmp(required, "1") == 0);
  check_unknown_unchanged<mlx_pointwise_graph_layout>([](auto& value) {
    return mlx_operation_event_pointwise_graph_layout(&value, 1, 0);
  });
}
#endif

TEST_CASE("pointwise Graph scalar singleton and empty geometry use the same host constructors"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  for (Shape shape : {Shape{}, Shape(12, 1), Shape{0}}) {
    const std::vector<float> values{2.f};
    array input(values.begin(), shape, float32);
    Role role;
    Observer observer;
    Bank bank;
    REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value, observer.value, 1, shape.size()) == 0);
    const auto held = role.graph->occupied_bytes();
    {
      auto value = add(input, input, stream);
      CHECK(value.shape() == input.shape());
      CHECK(value.status() == array::Status::unscheduled);
      CHECK(role.graph->occupied_bytes() <= held);
      bank.reset();
      CHECK(role.graph->occupied_bytes() > 0);
    }
    CHECK(role.graph->occupied_bytes() == 0);
  }
}

TEST_CASE("pointwise Graph host bank leaves already accepted CPU worker allocation independent"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  struct State {
    std::atomic<bool> release{false}, completed{false}, success{false};
  };
  struct Release {
    std::shared_ptr<State> state;
    ~Release() { state->release.store(true, std::memory_order_release); }
  };
  struct WorkerRecord : submission::Record {
    explicit WorkerRecord(Allocation allocation) : Record(allocation) {}
    bool scoped_observation_supported() const noexcept override { return true; }
  };
  struct Finish {
    submission::Record* record;
    ~Finish() { if (record) record->finish(true); }
  };
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  auto state = std::make_shared<State>();
  Role role;
  Release release{state}; // releases before Role even on assertion unwind
  Observer observer;
  auto pending = submission::Record::create<WorkerRecord>();
  pending->enter();
  auto* record = pending.release();
  Finish finish{record};
  record->reserve_streams(1);
  record->prepare_stream(stream);
  {
    submission::RecordDispatchGuard dispatch(*record);
    scheduler::enqueue(stream, [state, graph = submission::GraphQuotaRef(role.graph.get())] {
      while (!state->release.load(std::memory_order_acquire)) std::this_thread::yield();
      try {
        submission::GraphAllocator<unsigned char> allocator(graph.get());
        auto* block = allocator.allocate(64);
        block[63] = 37;
        state->success.store(block[63] == 37, std::memory_order_relaxed);
        allocator.deallocate(block, 64);
      } catch (...) {
        state->success.store(false, std::memory_order_relaxed);
      }
      state->completed.store(true, std::memory_order_release);
    });
  }
  record->finish(false);
  finish.record = nullptr;
  Bank bank;
  REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value, observer.value, 0, 0) == 0);
  state->release.store(true, std::memory_order_release);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!state->completed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  REQUIRE(state->completed.load(std::memory_order_acquire));
  CHECK(state->success.load(std::memory_order_relaxed));
  CHECK(static_cast<GraphConstruction*>(bank.value)->remaining() == 0);
  bank.reset();
  settle(role);
}

TEST_CASE("pointwise Graph size classes preserve best fit and never recycle spent slots"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  for (size_t rank : {size_t{0}, size_t{11}, size_t{21}}) {
    for (bool small_first : {false, true}) {
      CAPTURE(rank);
      CAPTURE(small_first);
      Role role;
      Observer observer;
      mlx_pointwise_graph_layout layout{};
      REQUIRE(mlx_operation_event_pointwise_graph_layout(&layout, 4, rank));
      std::vector<size_t> requests;
      for (size_t c = 0; c < 10; ++c)
        for (size_t n = 0; n < layout.request_counts[c]; ++n)
          requests.push_back(layout.request_bytes[c]);
      std::sort(requests.begin(), requests.end());
      Bank bank;
      REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value, observer.value, 4, rank) == 0);
      auto* native = static_cast<GraphConstruction*>(bank.value);
      REQUIRE(native->remaining() == requests.size());
      const auto held = role.graph->occupied_bytes();
      for (const auto& bad : {submission::GraphQuota::Request{1, 3},
                             submission::GraphQuota::Request{SIZE_MAX, 1}}) {
        try {
          (void)role.graph->allocate(bad.bytes, bad.alignment);
          FAIL("invalid requests must not consume a size-class cursor");
        } catch (const submission::GraphQuotaError& error) {
          CHECK(error.cause() == submission::GraphFailure::construction_mismatch);
        }
        CHECK(native->remaining() == requests.size());
        CHECK(role.graph->occupied_bytes() == held);
      }
      size_t lower = 0, upper = requests.size();
      bool small = small_first;
      while (lower < upper) {
        // Tiny requests must consume the smallest class, leaving every exact
        // large request in this alternating sequence satisfiable.
        const auto bytes = small ? (lower++, size_t{1}) : requests[--upper];
        small = !small;
        auto* block = role.graph->allocate(bytes, 1);
        REQUIRE(block);
        static_cast<unsigned char*>(block)[bytes - 1] = 37;
        CHECK(role.graph->occupied_bytes() <= held);
        role.graph->deallocate(block, bytes, 1);
        CHECK(native->remaining() == upper - lower);
      }
      try {
        (void)role.graph->allocate(1, 1);
        FAIL("physical retirement must not restore construction credit");
      } catch (const submission::GraphQuotaError& error) {
        CHECK(error.cause() == submission::GraphFailure::construction_mismatch);
      }
      CHECK(native->remaining() == 0);
      bank.reset();
      CHECK(role.graph->occupied_bytes() == 0);
    }
  }
}

#include "mlx/c/distributed_group.h"
#include "mlx/distributed/constructor.h"
#include "mlx/distributed/ops.h"
namespace distributed_constructor_tests {
using namespace pointwise_graph_tests;
struct Result {
  mlx_array value{nullptr,nullptr};
  ~Result(){mlx_array_free(value);}
};
}
TEST_CASE("distributed original constructor preserves actual singleton source and Graph retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace distributed_constructor_tests;
  auto stream=new_stream(Device::cpu);
  prepare(stream,stream);
  auto group=distributed::init(false,"ring");
  Shape shape(11,1);shape.back()=3;
  const int values[]={7,-11,23};
  array input(values,shape,int32);
  mlx_distributed_constructor_storage facts{};
  REQUIRE(mlx_distributed_group_constructor_storage(&facts,{&group},{&input},0,0));
  CHECK(facts.output_rank==11);
  CHECK(facts.output_elements==3);
  CHECK(facts.primitives==0);
  CHECK(facts.input_edges==0);
  CHECK(facts.blocks==1);
  auto sentinel=facts;
  CHECK_FALSE(mlx_distributed_group_constructor_storage(&facts,{&group},{&input},4,0));
  CHECK(std::memcmp(&facts,&sentinel,sizeof(facts))==0);
  CHECK_FALSE(mlx_distributed_group_constructor_storage(&facts,{nullptr},{&input},0,0));
  CHECK(std::memcmp(&facts,&sentinel,sizeof(facts))==0);
  Role role;
  Observer observer;
  const auto before=role.graph->occupied_bytes();
  {
    Result sum,gather;
    REQUIRE(mlx_distributed_construct_original(&sum.value,observer.value,{&group},{&input},0,0,{&stream})==0);
    REQUIRE(mlx_distributed_construct_original(&gather.value,observer.value,{&group},{&input},3,0,{&stream})==0);
    CHECK(sum.value.prepared_owner!=nullptr);
    CHECK(gather.value.prepared_owner!=nullptr);
    CHECK(mlx_array_get_(sum.value).data_shared_ptr()==input.data_shared_ptr());
    CHECK(mlx_array_get_(gather.value).data_shared_ptr()==input.data_shared_ptr());
    CHECK(mlx_array_get_(sum.value).data<int>()[1]==-11);
    CHECK(role.graph->occupied_bytes()>before);
    // Unused/header blocks retired: another bank is available immediately.
    Bank bank;
    REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value,observer.value,0,0)==0);
    Result refused;
    CHECK(mlx_distributed_construct_original(&refused.value,observer.value,{&group},{&input},0,0,{&stream})==10);
    CHECK(refused.value.ctx==nullptr);
  }
  CHECK(role.graph->occupied_bytes()==before);
  {
    GraphHoles holes(role.graph.get());holes.fill();
    const auto held=role.graph->occupied_bytes();
    Result refused;
    CHECK(mlx_distributed_construct_original(&refused.value,observer.value,{&group},{&input},0,0,{&stream})==2);
    CHECK(refused.value.ctx==nullptr);
    CHECK(role.graph->occupied_bytes()==held);
  }
  CHECK(role.graph->occupied_bytes()==before);
}

#include "mlx/backend/cpu/copy.h"
#include "mlx/prepared_input.h"
TEST_CASE("General CPU copy uses source-derived weak wrappers and high-rank worker destinations"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  constexpr size_t rank=14, count=size_t{1}<<rank;
  Shape shape(rank,2);
  std::vector<int> values(count);
  std::vector<int> axes(rank);
  for(size_t i=0;i<count;++i)values[i]=static_cast<int>((i*17)%997)-491;
  for(size_t i=0;i<rank;++i)axes[i]=static_cast<int>(rank-1-i);
  array base(values.begin(),shape,int32);
  auto input=transpose(base,axes,stream);eval(input);
  REQUIRE_FALSE(input.flags().row_contiguous);
  CpuCopyDispatchStorage layout;
  REQUIRE(contiguous_copy_dispatch_storage(input,layout));
  CHECK(layout.rank==rank);
  CHECK(layout.weak_wrappers==2);
  CHECK(layout.data_captures==2);
  CHECK(layout.weak_graph_extent_sum>0);
  CHECK(layout.worker_graph_extent_bound>0);
  CHECK(layout.task_graph_extent>layout.task_bytes);
  CHECK(layout.controls>0);
  // Run the existing ordinary copy producer; the oracle computes independent
  // logical-to-physical bit reversal and includes both signs and nonzero values.
  auto copied=contiguous_copy_cpu(input,stream);
  synchronize(stream);
  for(size_t i=0;i<count;++i) {
    size_t physical=0,bits=i;
    for(size_t d=0;d<rank;++d){physical=(physical<<1)|(bits&1);bits>>=1;}
    CHECK(copied.data<int>()[i]==values[physical]);
  }
  CHECK(copied.flags().row_contiguous);
  WeakCopyStorage weak;
  REQUIRE(PreparedInputLeaf::weak_copy_storage(rank,weak));
  CHECK(weak.count==6);
  CHECK(layout.weak_graph_extent_sum==2*weak.graph_extent_sum);
  {
    Role role;
    const auto before=role.graph->occupied_bytes();
    {
      auto alias=array::unsafe_weak_copy(input);
      CHECK(alias.data<int>()==input.data<int>());
      CHECK(alias.strides()==input.strides());
      CHECK(alias.data_shared_ptr()!=input.data_shared_ptr());
      CHECK(role.graph->occupied_bytes()>before);
      CHECK(role.graph->occupied_bytes()-before<=weak.graph_extent_sum);
    }
    CHECK(role.graph->occupied_bytes()==before);
    GraphHoles holes(role.graph.get());holes.fill();
    const auto full=role.graph->occupied_bytes();
    CHECK_THROWS_AS(array::unsafe_weak_copy(input),submission::GraphQuotaError);
    CHECK(role.graph->occupied_bytes()==full);
  }
  // The single-stride collapse is the same equation, with fixed fresh
  // destinations. Preserve meaningful unit dimensions and noncontiguous cuts.
  Shape cut_shape{2,1,3,4};Strides cut_strides{41,41,7,1};
  auto [collapsed_shape,collapsed_strides]=collapse_contiguous_dims(cut_shape,cut_strides);
  CHECK((collapsed_shape==Shape{2,3,4}));
  CHECK((collapsed_strides==Strides{41,7,1}));
  CollapseStorageLayout empty,large;
  REQUIRE(collapse_single_dims_layout(0,empty));CHECK(empty.graph_extent_sum==0);
  REQUIRE(collapse_single_dims_layout(rank,large));CHECK(large.graph_extent_sum>0);
  auto saved=large;
  CHECK_FALSE(collapse_single_dims_layout(SIZE_MAX,large));
  CHECK(std::memcmp(&saved,&large,sizeof(saved))==0);
}


#include "mlx/threadpool.h"
#include "mlx/distributed/ring/socket_tasks.h"
namespace accepted_graph_worker_tests {
struct State {
  std::atomic<bool> release{false}, retired{false}, retirement_source{false};
};
struct Release {
  State& state;
  ~Release() { state.release.store(true, std::memory_order_release); }
};
struct Job {
  State* state;
  submission::GraphQuota* graph;
  submission::GraphQuota* foreign;
  bool armed{true};
  Job(State& state, submission::GraphQuota* graph, submission::GraphQuota* foreign)
      : state(&state), graph(graph), foreign(foreign) {}
  Job(Job&& other) noexcept
      : state(other.state), graph(other.graph), foreign(other.foreign),
        armed(std::exchange(other.armed,false)) {}
  Job(const Job&) = delete;
  ~Job() {
    if (!armed) return;
    // Callable destruction is still part of the accepted task lifetime. Its
    // explicit metadata allocation must not borrow a concurrent host bank.
    try {
      const auto context=submission::current_graph_construction_context();
      submission::GraphAllocator<int> allocator(graph);
      auto* value=allocator.allocate(1);*value=73;
      state->retirement_source.store(context.allocation_worker==graph &&
          context.independent_worker && *value==73,std::memory_order_relaxed);
      allocator.deallocate(value,1);
    } catch (...) { state->retirement_source.store(false,std::memory_order_relaxed); }
    state->retired.store(true,std::memory_order_release);
  }
  int operator()() const {
    while(!state->release.load(std::memory_order_acquire))std::this_thread::yield();
    const auto context=submission::current_graph_construction_context();
    if (context.allocation_worker!=graph || !context.independent_worker ||
        submission::current_graph_quota() || submission::current_record_quota() ||
        submission::current_scope() || submission::GraphAllocator<int>{}.resource())
      return -1;
    const auto controls=submission::current_native_controls();
    if (!controls.original || controls.failure!=submission::NativeControlFailure::invalid_scope)
      return -2;
    bool rejected=false;
    try { auto* forbidden=new submission::Scope;forbidden->seal();forbidden->release(); }
    catch(const submission::NativeControlError& e) {
      rejected=e.failure()==submission::NativeControlFailure::invalid_scope;
    }
    if(!rejected)return -3;
    if(foreign) {
      rejected=false;
      try { auto* value=foreign->allocate(16,alignof(int));foreign->deallocate(value,16,alignof(int)); }
      catch(const submission::GraphQuotaError& e) {
        rejected=e.cause()==submission::GraphFailure::foreign_parent;
      }
      if(!rejected)return -4;
    }
    submission::GraphVector<int> values{submission::GraphAllocator<int>(graph)};
    values.reserve(37);
    int sum=0;
    for(int i=0;i<37;++i){values.push_back(i*3-17);sum+=values.back();}
    // Empty socket completion is synchronous in the accepted parent and must
    // keep its existing loan instead of installing a second worker context.
    using mlx::core::distributed::ring::SocketTaskQueue;
    auto socket=SocketTaskQueue::prepare(nullptr,0,graph);
    auto receipt=SocketTaskQueue::complete_empty(socket);receipt.get();
    return sum;
  }
};
} // namespace accepted_graph_worker_tests

TEST_CASE("accepted Ring worker Graph loan preserves concurrent host bank and source identity"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  using namespace accepted_graph_worker_tests;
  State state;
  std::unique_ptr<submission::GraphQuota,wait_record_facts::ReleaseGraph> foreign{
      submission::GraphQuota::create(4096,nullptr,nullptr)};
  ThreadPool pool(1);
  Release release{state};
  Role role;
  Observer observer;
  auto failed=pool.enqueue_with_graph(role.graph.get(),[&state] {
    while(!state.release.load(std::memory_order_acquire))std::this_thread::yield();
    throw submission::GraphQuotaError(submission::GraphFailure::invalid_layout);
  });
  auto value=pool.enqueue_with_graph(role.graph.get(),Job(state,role.graph.get(),foreign.get()));
  Bank bank;
  REQUIRE(mlx_operation_event_prepare_pointwise_graph(&bank.value,observer.value,1,0)==0);
  auto* native=static_cast<GraphConstruction*>(bank.value);
  const auto remaining=native->remaining();
  REQUIRE(remaining>0);
  state.release.store(true,std::memory_order_release);
  CHECK_THROWS_AS(failed.get(),submission::GraphQuotaError);
  CHECK(value.get()==1369);
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
  while(!state.retired.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
    std::this_thread::yield();
  REQUIRE(state.retired.load(std::memory_order_acquire));
  CHECK(state.retirement_source.load(std::memory_order_relaxed));
  CHECK(native->remaining()==remaining);
  CHECK(foreign->occupied_bytes()==0);
  // The worker did not consume/refund a reserved host destination.
  auto* own=role.graph->allocate(1,1);
  CHECK(native->remaining()==remaining-1);
  role.graph->deallocate(own,1,1);
  bank.reset();
}

TEST_CASE("accepted Ring worker Graph loan survives creator retirement without default allocation authority"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace accepted_graph_worker_tests;
  State state;
  std::atomic<unsigned> retired{0};
  {
    ThreadPool pool(1);
    Release release{state};
    auto* graph=submission::GraphQuota::create(128<<10,&retired,[](void* value) {
      static_cast<std::atomic<unsigned>*>(value)->fetch_add(1,std::memory_order_relaxed);
    });
    auto value=pool.enqueue_with_graph(graph,Job(state,graph,nullptr));
    graph->release();
    CHECK(retired.load()==0);
    state.release.store(true,std::memory_order_release);
    CHECK(value.get()==1369);
  }
  CHECK(state.retired.load(std::memory_order_acquire));
  CHECK(state.retirement_source.load(std::memory_order_relaxed));
  CHECK(retired.load()==1);
}


#include "mlx/backend/cpu/distributed_storage.h"
#include "mlx/distributed/primitives.h"
TEST_CASE("CPU distributed Eval storage refuses identities and undeclared native producers"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace mlx::core::distributed;
  const auto stream=new_stream(Device::cpu);
  auto group=init(false,"ring");
  GroupStorageInventory inventory;
  REQUIRE(group.storage_inventory(inventory));
  REQUIRE(inventory.kind==GroupStorageKind::empty);
  array input({7,-11,23});
  // Ordinary singleton identity is a real leaf, not a fabricated Ring Eval.
  auto identity=all_sum(input,group,stream);
  REQUIRE(identity.id()==input.id());
  CHECK_FALSE(is_cpu_distributed_primitive(identity));
  CpuDistributedStorage untouched;
  untouched.backing_births=91;untouched.logical_backing_bytes=73;
  CHECK_FALSE(cpu_distributed_storage(identity,untouched));
  CHECK(untouched.backing_births==91);CHECK(untouched.logical_backing_bytes==73);
  CHECK(cpu_distributed_storage_inspection_controls(identity)>0);
  // An actual typed primitive still cannot turn EmptyGroup into a Ring source.
  auto primitive=std::make_shared<AllReduce>(stream,group,AllReduce::Sum);
  array explicit_identity(input.shape(),input.dtype(),primitive,{input});
  CHECK(is_cpu_distributed_primitive(explicit_identity));
  CHECK_FALSE(cpu_distributed_storage(explicit_identity,untouched));
  CHECK(untouched.backing_births==91);CHECK(untouched.logical_backing_bytes==73);
  struct Derived final:AllReduce {using AllReduce::AllReduce;};
  array subclass(input.shape(),input.dtype(),std::make_shared<Derived>(stream,group,AllReduce::Sum),{input});
  CHECK_FALSE(is_cpu_distributed_primitive(subclass));
  CHECK_FALSE(cpu_distributed_storage(subclass,untouched));
  array unsupported(input.shape(),input.dtype(),std::make_shared<AllReduce>(stream,group,AllReduce::Prod),{input});
  CHECK_FALSE(is_cpu_distributed_primitive(unsupported));
  mlx_distributed_cpu_eval_storage native{};native.backing_births=39;
  CHECK_FALSE(mlx_distributed_query_cpu_eval_storage(&native,mlx_array{&explicit_identity}));
  CHECK(native.backing_births==39);
  CHECK_FALSE(mlx_distributed_query_cpu_eval_storage(&native,mlx_array{nullptr}));
  CHECK(native.backing_births==39);
  CHECK(mlx_distributed_cpu_eval_storage_controls(mlx_array{&explicit_identity})>0);
  CHECK(mlx_distributed_cpu_eval_storage_controls(mlx_array{nullptr})>0);
}


TEST_CASE("distributed source constructor borrows resident bank and restores exhausted prefix"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace distributed_constructor_tests;
  auto stream=new_stream(Device::cpu);
  prepare(stream,stream);
  auto group=distributed::init(false,"ring");
  const int values[]={7,-11,23};
  array input(values,Shape{3},int32);
  Role role;
  Observer observer;
  const auto baseline=role.graph->occupied_bytes();
  {
    Bank resident;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&resident.value,observer.value,2,0,1)==0);
    auto* bank=static_cast<GraphConstruction*>(resident.value);
    auto prefix=add(input,input,stream);
    const auto remaining=bank->remaining();
    REQUIRE(remaining>0);
    GraphConstruction* strict=nullptr;
    CHECK(GraphConstruction::create_distributed(*role.graph,role.scope->identity(),group,input,
        distributed::GroupWorkerOperation::sum,0,strict)==submission::GraphFailure::construction_busy);
    CHECK(strict==nullptr);
    const auto held=role.graph->occupied_bytes();
    {
      Result value;
      REQUIRE(mlx_distributed_construct_original(&value.value,observer.value,{&group},{&input},0,0,{&stream})==0);
      CHECK(bank->remaining()==remaining);
      CHECK(mlx_array_get_(value.value).data_shared_ptr()==input.data_shared_ptr());
      CHECK(mlx_array_get_(value.value).data<int>()[1]==-11);
      CHECK(role.graph->occupied_bytes()>held);
    }
    CHECK(role.graph->occupied_bytes()==held);
    {
      GraphHoles holes(role.graph.get());holes.fill();
      const auto exhausted=role.graph->occupied_bytes();
      Result refused;
      CHECK(mlx_distributed_construct_original(&refused.value,observer.value,{&group},{&input},0,0,{&stream})==2);
      CHECK(refused.value.ctx==nullptr);
      CHECK(refused.value.prepared_owner==nullptr);
      CHECK(bank->remaining()==remaining);
      CHECK(role.graph->occupied_bytes()==exhausted);
      // Even while the free arena is exhausted the restored resident bank owns
      // the next ordinary equation's already reserved destination.
      auto suffix=add(prefix,input,stream);
      CHECK(suffix.status()==array::Status::unscheduled);
      CHECK(bank->remaining()<remaining);
      CHECK(role.graph->occupied_bytes()<=exhausted);
    }
    const auto after=bank->remaining();
    Result again;
    REQUIRE(mlx_distributed_construct_original(&again.value,observer.value,{&group},{&input},3,0,{&stream})==0);
    CHECK(bank->remaining()==after);
    CHECK(mlx_array_get_(again.value).data<int>()[2]==23);
    resident.reset();
    CHECK(role.graph->occupied_bytes()>baseline);
  }
  CHECK(role.graph->occupied_bytes()==baseline);
}

#include "mlx/backend/cpu/copy_storage.h"
#include "mlx/c/original_buffer.h"
TEST_CASE("CPU copy Eval source matches actual primitive and refuses malformed geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  const int values[] = {7, -11, 23, 13, 17, -19};
  array base(values, Shape{2, 3}, int32);
  auto source = transpose(base, stream);
  eval(source);
  auto value = contiguous(source, false, stream);
  cpu::CopyEvalStorage cold, actual;
  REQUIRE(cpu::copy_eval_layout(2, 1, true, false, cold));
  REQUIRE(cpu::copy_eval_storage(value, actual));
  CHECK(actual.allocation_extents == cold.allocation_extents);
  CHECK(actual.worker_graph_extents == cold.worker_graph_extents);
  CHECK(actual.request_counts[3] == 3);
  CHECK(actual.backing_births == 1);
  CHECK(cold.named_control_bytes > 0);
  cpu::CopyEvalStorage high;
  REQUIRE(cpu::copy_eval_layout(14, 1, true, false, high));
  CHECK(high.worker_graph_extents > cold.worker_graph_extents);
  CHECK(high.request_counts[8] == 7);
  std::array<unsigned char, sizeof(high)> saved;
  std::memcpy(saved.data(), &high, sizeof(high));
  CHECK_FALSE(cpu::copy_eval_layout(SIZE_MAX, 1, true, false, high));
  CHECK(std::memcmp(saved.data(), &high, sizeof(high)) == 0);
  CHECK_FALSE(cpu::copy_eval_layout(2, 2, true, false, high));
  CHECK(std::memcmp(saved.data(), &high, sizeof(high)) == 0);
  CHECK_FALSE(cpu::copy_eval_storage(base, high));
  auto unrelated = add(base, base, stream);
  CHECK_FALSE(cpu::copy_eval_storage(unrelated, high));
  auto malformed = array(Shape{6}, int32, std::make_shared<Contiguous>(stream, false), {source});
  CHECK_FALSE(cpu::copy_eval_storage(malformed, high));
  CHECK(std::memcmp(saved.data(), &high, sizeof(high)) == 0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_copy_eval_layout(&raw, 2, 1, true, false));
  CHECK(raw.graph_extents == cold.allocation_extents);
  CHECK(raw.worker_graph_extents == cold.worker_graph_extents);
  REQUIRE(mlx_operation_event_cpu_copy_eval_layout(&raw, 0, 1, false, false));
  CHECK(raw.backing_births == 0);
  CHECK(raw.worker_graph_extents == 0);
  const auto previous = raw;
  CHECK_FALSE(mlx_operation_event_cpu_copy_eval_layout(&raw, 0, SIZE_MAX, false, false));
  CHECK(std::memcmp(&previous, &raw, sizeof(raw)) == 0);
}

TEST_CASE("CPU copy Eval executes the existing nonzero strided worker under original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  unsigned retired = 0;
  struct Budget {
    mlx_original_buffer_budget value{};
    ~Budget() { mlx_original_buffer_budget_release(value); }
  } budget;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20,
      &retired, [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
  const int values[] = {7, -11, 23, 13, 17, -19};
  array base(values, Shape{2, 3}, int32);
  auto source = transpose(base, stream);
  eval(source);
  REQUIRE_FALSE(source.flags().row_contiguous);
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer;
    Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, 2) == 0);
    auto value = contiguous(source, false, stream);
    bank.reset();
    Operation operation;
    operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1, 3, 2, 2, 2, 1, 8};
    REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
    eval_traversal_tests::complete(role, operation, value);
    const int expected[] = {7, 13, -11, 17, 23, -19};
    CHECK(value.flags().row_contiguous);
    for (size_t i = 0; i != 6; ++i) CHECK(value.data<int>()[i] == expected[i]);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
    CHECK(info.known);
    CHECK(info.identity != 0);
    CHECK(info.charged_bytes >= sizeof(expected));
    CHECK(retired == 0);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value) == 0);
  CHECK(retired == 0);
}

#include "mlx/backend/cpu/unary_storage.h"
TEST_CASE("CPU unary Eval source matches the exact worker and rejects unsupported geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  const float values[] = {-2.0f, 0.5f, 3.0f};
  array source(values, Shape{3}, float32);
  auto value = sigmoid(source, stream);
  cpu::UnaryEvalStorage cold, actual;
  REQUIRE(cpu::unary_eval_layout(cpu::UnaryEvalKind::sigmoid, float32, 1, false, cold));
  REQUIRE(cpu::unary_eval_storage(value, actual));
  CHECK(actual.kind == cpu::UnaryEvalKind::sigmoid);
  CHECK(actual.dtype == float32);
  CHECK(actual.allocation_extents == cold.allocation_extents);
  CHECK(actual.worker_graph_extents == cold.worker_graph_extents);
  CHECK(actual.backing_births == 1);
  CHECK(actual.request_counts[3] == 3);
  CHECK(actual.named_control_bytes > 0);
  cpu::UnaryEvalStorage high;
  REQUIRE(cpu::unary_eval_layout(cpu::UnaryEvalKind::square, int32, 14, true, high));
  CHECK(high.worker_graph_extents > cold.worker_graph_extents);
  CHECK(high.request_counts[8] == 7);
  CHECK(high.request_counts[2] == 2);
  std::array<unsigned char, sizeof(high)> saved;
  std::memcpy(saved.data(), &high, sizeof(high));
  CHECK_FALSE(cpu::unary_eval_layout(cpu::UnaryEvalKind::sigmoid, int32, 1, false, high));
  CHECK_FALSE(cpu::unary_eval_layout(cpu::UnaryEvalKind::erf, complex64, 1, false, high));
  CHECK_FALSE(cpu::unary_eval_layout(cpu::UnaryEvalKind::square, int32, SIZE_MAX, false, high));
  CHECK_FALSE(cpu::unary_eval_layout(static_cast<cpu::UnaryEvalKind>(999), int32, 1, false, high));
  CHECK(std::memcmp(saved.data(), &high, sizeof(high)) == 0);
  auto wrong_shape = array(Shape{1,3}, float32, std::make_shared<Sigmoid>(stream), {source});
  auto wrong_dtype = array(Shape{3}, int32, std::make_shared<Sigmoid>(stream), {source});
  auto wrong_inputs = array(Shape{3}, float32, std::make_shared<Sigmoid>(stream), {source,source});
  CHECK_FALSE(cpu::unary_eval_storage(wrong_shape, high));
  CHECK_FALSE(cpu::unary_eval_storage(wrong_dtype, high));
  CHECK_FALSE(cpu::unary_eval_storage(wrong_inputs, high));
  CHECK_FALSE(cpu::unary_eval_storage(source, high));
  auto unrelated = add(source, source, stream);
  CHECK_FALSE(cpu::unary_eval_storage(unrelated, high));
  struct Derived final : Sigmoid { using Sigmoid::Sigmoid; };
  auto subclass = array(Shape{3}, float32, std::make_shared<Derived>(stream), {source});
  CHECK_FALSE(cpu::unary_eval_storage(subclass, high));
  CHECK(std::memcmp(saved.data(), &high, sizeof(high)) == 0);
  mlx_cpu_unary_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_unary_eval_layout(&raw,
      uint32_t(cpu::UnaryEvalKind::sigmoid), MLX_FLOAT32, 1, false));
  CHECK(raw.graph_extents == cold.allocation_extents);
  CHECK(raw.worker_graph_extents == cold.worker_graph_extents);
  const auto previous = raw;
  CHECK_FALSE(mlx_operation_event_cpu_unary_eval_layout(&raw, UINT32_MAX, MLX_FLOAT32, 1, false));
  CHECK_FALSE(mlx_operation_event_cpu_unary_eval_layout(&raw, 0, static_cast<mlx_dtype>(999), 1, false));
  CHECK(std::memcmp(&previous, &raw, sizeof(raw)) == 0);
}

TEST_CASE("CPU unary Eval preserves nonzero strided output and escaped original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  unsigned retired = 0;
  struct Budget {
    mlx_original_buffer_budget value{};
    ~Budget() { mlx_original_buffer_budget_release(value); }
  } budget;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20,
      &retired, [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
  const int values[] = {7,99,-11,99,23,99,13,99,17,99,-19,99};
  Shape shape(14,1); shape[12]=2; shape[13]=6;
  Shape starts(14,0), steps(14,1); steps[13]=2;
  array base(values,shape,int32);
  auto source = slice(base,starts,shape,steps,stream);
  eval(source);
  REQUIRE_FALSE(source.flags().contiguous);
  REQUIRE(source.ndim()==14);
  std::optional<array> escaped;
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer;
    Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, 14) == 0);
    auto value = square(source,stream);
    bank.reset();
    Operation operation;
    operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    CHECK(value.flags().row_contiguous);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);
    CHECK(info.identity!=0);
    CHECK(info.charged_bytes>=6*sizeof(int));
    escaped.emplace(value);
  }
  const int expected[]={49,121,529,169,289,361};
  REQUIRE(escaped.has_value());
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  for(size_t i=0;i!=6;++i) CHECK(escaped->data<int>()[i]==expected[i]);
  CHECK(retired==0);
  escaped.reset();
  CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/backend/cpu/binary_storage.h"
#include "mlx/backend/common/utils.h"
TEST_CASE("CPU binary Eval source preserves arity dtype and checked loop geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu); prepare(stream,stream);
  const float values[]={-2.0f,0.5f,3.0f};
  array source(values,Shape{3},float32);
  auto value=greater(source,source,stream);
  cpu::BinaryEvalStorage cold,actual;
  REQUIRE(cpu::binary_eval_layout(cpu::BinaryEvalKind::greater,float32,1,3,false,cold));
  REQUIRE(cpu::binary_eval_storage(value,actual));
  CHECK(actual.kind==cpu::BinaryEvalKind::greater);
  CHECK(actual.dtype==float32);
  CHECK(actual.elements==3);
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.request_counts[3]==4);
  CHECK(actual.backing_births==1);
  cpu::BinaryEvalStorage high;
  REQUIRE(cpu::binary_eval_layout(cpu::BinaryEvalKind::add,int32,14,6,true,high));
  CHECK(high.worker_graph_extents>cold.worker_graph_extents);
  CHECK(high.request_counts[8]==10);
  CHECK(high.request_counts[2]==2);
  ContiguousIteratorLayout iterator;
  REQUIRE(contiguous_iterator_layout(13,iterator));
  CHECK(iterator.graph_extent_sum>0);
  CHECK(iterator.controls>0);
  std::array<unsigned char,sizeof(high)> saved;
  std::memcpy(saved.data(),&high,sizeof(high));
  CHECK_FALSE(cpu::binary_eval_layout(cpu::BinaryEvalKind::add,int32,SIZE_MAX,6,false,high));
  CHECK_FALSE(cpu::binary_eval_layout(cpu::BinaryEvalKind::add,int32,1,size_t(INT_MAX)+1,false,high));
  CHECK_FALSE(cpu::binary_eval_layout(static_cast<cpu::BinaryEvalKind>(999),int32,1,3,false,high));
  CHECK(std::memcmp(saved.data(),&high,sizeof(high))==0);
  auto wrong_shape=array(Shape{1,3},float32,std::make_shared<Add>(stream),{source,source});
  auto wrong_dtype=array(Shape{3},float32,std::make_shared<Greater>(stream),{source,source});
  auto wrong_inputs=array(Shape{3},float32,std::make_shared<Add>(stream),{source});
  auto nan_equal=array(Shape{3},bool_,std::make_shared<Equal>(stream,true),{source,source});
  CHECK_FALSE(cpu::binary_eval_storage(wrong_shape,high));
  CHECK_FALSE(cpu::binary_eval_storage(wrong_dtype,high));
  CHECK_FALSE(cpu::binary_eval_storage(wrong_inputs,high));
  CHECK_FALSE(cpu::binary_eval_storage(nan_equal,high));
  CHECK_FALSE(cpu::binary_eval_storage(source,high));
  struct Derived final:Add {using Add::Add;};
  auto subclass=array(Shape{3},float32,std::make_shared<Derived>(stream),{source,source});
  CHECK_FALSE(cpu::binary_eval_storage(subclass,high));
  CHECK(std::memcmp(saved.data(),&high,sizeof(high))==0);
  // Arity expansion of the shared constructor grants no broader Copy source.
  cpu::CopyEvalStorage copy;
  CHECK_FALSE(cpu::copy_eval_layout(1,2,true,false,copy));
  mlx_cpu_binary_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_binary_eval_layout(&raw,
      uint32_t(cpu::BinaryEvalKind::greater),MLX_FLOAT32,1,3,false));
  CHECK(raw.graph_extents==cold.allocation_extents);
  const auto previous=raw;
  CHECK_FALSE(mlx_operation_event_cpu_binary_eval_layout(&raw,UINT32_MAX,MLX_FLOAT32,1,3,false));
  CHECK_FALSE(mlx_operation_event_cpu_binary_eval_layout(&raw,0,static_cast<mlx_dtype>(999),1,3,false));
  CHECK(std::memcmp(&previous,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU binary Eval preserves repeated strided inputs and escaped original output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu); prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  unsigned retired=0;
  struct Budget {mlx_original_buffer_budget value{};
    ~Budget(){mlx_original_buffer_budget_release(value);}} budget;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,
      &retired,[](void* p){++*static_cast<unsigned*>(p);})==0);
  const int values[]={7,99,-11,99,23,99,13,99,17,99,-19,99};
  Shape shape(14,1);shape[12]=2;shape[13]=6;
  Shape starts(14,0),steps(14,1);steps[13]=2;
  array base(values,shape,int32);
  auto source=slice(base,starts,shape,steps,stream);eval(source);
  REQUIRE_FALSE(source.flags().contiguous);
  std::optional<array> escaped;
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,14)==0);
    auto value=add(source,source,stream);
    bank.reset();
    Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,2,3,2,1,12};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    CHECK(value.flags().row_contiguous);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity!=0);CHECK(info.charged_bytes>=6*sizeof(int));
    escaped.emplace(value);
  }
  const int expected[]={14,-22,46,26,34,-38};
  REQUIRE(escaped.has_value());
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  for(size_t i=0;i!=6;++i)CHECK(escaped->data<int>()[i]==expected[i]);
  CHECK(retired==0);escaped.reset();
  CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

TEST_CASE("CPU cast Eval source shares copy storage and validates conversion geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream,stream);
  const int values[]={7,-11,23,13,17,-19};
  array source(values,Shape{2,3},int32);
  auto value=astype(source,float32,stream);
  cpu::CopyEvalStorage cold,actual,copy;
  REQUIRE(cpu::cast_eval_layout(2,int32,float32,6,false,cold));
  REQUIRE(cpu::copy_eval_layout(2,1,true,false,copy));
  REQUIRE(cpu::copy_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.allocation_extents==copy.allocation_extents);
  CHECK(actual.request_counts[3]==3);CHECK(actual.backing_births==1);
  CHECK(actual.named_control_bytes>copy.named_control_bytes);
  cpu::CopyEvalStorage high;
  REQUIRE(cpu::cast_eval_layout(14,int32,float32,6,true,high));
  CHECK(high.worker_graph_extents>cold.worker_graph_extents);
  CHECK(high.request_counts[8]==7);CHECK(high.request_counts[2]==2);
  std::array<unsigned char,sizeof(high)> saved;
  std::memcpy(saved.data(),&high,sizeof(high));
  CHECK_FALSE(cpu::cast_eval_layout(SIZE_MAX,int32,float32,6,false,high));
  CHECK_FALSE(cpu::cast_eval_layout(2,int32,float32,size_t(INT_MAX)+1,false,high));
  CHECK_FALSE(cpu::cast_eval_layout(2,Dtype{static_cast<Dtype::Val>(999),1},float32,6,false,high));
  CHECK_FALSE(cpu::cast_eval_layout(2,Dtype{Dtype::Val::int32,1},float32,6,false,high));
  auto shape=array(Shape{6},float32,std::make_shared<AsType>(stream,float32),{source});
  auto dtype=array(Shape{2,3},float32,std::make_shared<AsType>(stream,int32),{source});
  auto inputs=array(Shape{2,3},float32,std::make_shared<AsType>(stream,float32),{source,source});
  struct Derived final : AsType {using AsType::AsType;};
  auto subclass=array(Shape{2,3},float32,std::make_shared<Derived>(stream,float32),{source});
  CHECK_FALSE(cpu::copy_eval_storage(shape,high));
  CHECK_FALSE(cpu::copy_eval_storage(dtype,high));
  CHECK_FALSE(cpu::copy_eval_storage(inputs,high));
  CHECK_FALSE(cpu::copy_eval_storage(subclass,high));
  CHECK(std::memcmp(saved.data(),&high,sizeof(high))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_cast_eval_layout(&raw,MLX_INT32,MLX_FLOAT32,2,6,false));
  CHECK(raw.graph_extents==cold.allocation_extents);
  CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
  CHECK(raw.signal_graph_extents==0);
  auto previous=raw;
  CHECK_FALSE(mlx_operation_event_cpu_cast_eval_layout(&raw,MLX_INT32,MLX_FLOAT32,2,SIZE_MAX,false));
  CHECK(std::memcmp(&raw,&previous,sizeof(raw))==0);
}

TEST_CASE("CPU cast Eval preserves nonzero strided conversion and escaped original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  unsigned retired=0;
  struct Budget { mlx_original_buffer_budget value{};
    ~Budget(){mlx_original_buffer_budget_release(value);} } budget;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,
      &retired,[](void* p){++*static_cast<unsigned*>(p);})==0);
  const int values[]={7,99,-11,99,23,99,13,99,17,99,-19,99};
  Shape shape(14,1);shape[12]=2;shape[13]=6;
  Shape starts(14,0),steps(14,1);steps[13]=2;
  array base(values,shape,int32);
  auto source=slice(base,starts,shape,steps,stream);eval(source);
  REQUIRE_FALSE(source.flags().contiguous);
  std::optional<array> escaped;
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,14)==0);
    auto value=astype(source,float32,stream);
    bank.reset();
    Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    CHECK(value.flags().row_contiguous);CHECK(value.dtype()==float32);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity!=0);CHECK(info.charged_bytes>=6*sizeof(float));
    escaped.emplace(value);
  }
  const float expected[]={7,-11,23,13,17,-19};
  REQUIRE(escaped.has_value());
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  for(size_t i=0;i!=6;++i)CHECK(escaped->data<float>()[i]==expected[i]);
  CHECK(retired==0);escaped.reset();
  CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/backend/cpu/slice_storage.h"
TEST_CASE("CPU Slice source preserves fixed range geometry and refuses unqualified views"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const int values[]={7,-11,23,13,17,-19};array source(values,Shape{2,3},int32);
  auto value=slice(source,Shape{0,1},Shape{2,3},Shape{1,1},stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::slice_eval_layout(2,false,false,cold));
  REQUIRE(cpu::slice_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.backing_births==0);CHECK(actual.worker_graph_extents==0);
  CHECK(actual.request_counts[3]==0);CHECK(actual.request_counts[6]==0);
  std::array<unsigned char,sizeof(actual)> saved;
  std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::slice_eval_layout(0,false,false,actual));
  CHECK_FALSE(cpu::slice_eval_layout(5,false,false,actual));
  auto strided=slice(source,Shape{0,0},Shape{2,3},Shape{1,2},stream);
  CHECK(cpu::slice_eval_storage(strided,actual));
  auto wrong_step=array(Shape{2,3},int32,std::make_shared<Slice>(stream,Shape{0,0},Shape{2,3},Shape{1,2}),{source});
  CHECK_FALSE(cpu::slice_eval_storage(wrong_step,actual));
  auto zero_step=array(Shape{2,2},int32,std::make_shared<Slice>(stream,Shape{0,0},Shape{2,3},Shape{1,0}),{source});
  CHECK_FALSE(cpu::slice_eval_storage(zero_step,actual));
  auto wrong=array(Shape{2,1},int32,std::make_shared<Slice>(stream,Shape{0,1},Shape{2,3},Shape{1,1}),{source});
  CHECK_FALSE(cpu::slice_eval_storage(wrong,actual));
  auto wrong_empty=array(Shape{2,0},int32,std::make_shared<Slice>(stream,Shape{0,1},Shape{2,2},Shape{1,1}),{source});
  CHECK_FALSE(cpu::slice_eval_storage(wrong_empty,actual));
  struct Derived final : Slice { using Slice::Slice; };
  auto subclass=array(Shape{2,2},int32,std::make_shared<Derived>(stream,Shape{0,1},Shape{2,3},Shape{1,1}),{source});
  CHECK_FALSE(cpu::slice_eval_storage(subclass,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_slice_eval_layout(&raw,2,false,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==0);
  CHECK(raw.signal_graph_extents==0);auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_slice_eval_layout(&raw,SIZE_MAX,false,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
}
TEST_CASE("CPU Slice aliases exact original backing through escaped view retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
  unsigned retired=0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void* p){++*static_cast<unsigned*>(p);})==0);
  const int values[]={7,99,-11,99,23,99,13,99,17,99,-19,99};
  array base(values,Shape{2,6},int32);
  auto strided=slice(base,Shape{0,0},Shape{2,6},Shape{1,2},stream);eval(strided);
  std::optional<array> source,escaped;
  uint64_t identity=0;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=contiguous(strided,false,stream);bank.reset();
    Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    REQUIRE(info.known);identity=info.identity;source.emplace(value);
  }
  auto occupied=mlx_original_buffer_budget_occupied(budget.value);REQUIRE(occupied>0);
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=slice(*source,Shape{0,1},Shape{2,3},Shape{1,1},stream);bank.reset();
    Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity==identity);
    CHECK(value.shape()==Shape{2,2});CHECK_FALSE(value.flags().row_contiguous);
    CHECK(value.data<int>()[0]==-11);CHECK(value.data<int>()[1]==23);
    CHECK(value.data<int>()[value.strides()[0]]==17);
    CHECK(value.data<int>()[value.strides()[0]+1]==-19);
    escaped.emplace(value);
  }
  source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==occupied);
  CHECK(retired==0);escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/backend/cpu/softmax_storage.h"
TEST_CASE("CPU Softmax source counts actual optional copy and rejects malformed rows"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-2.5f,0.25f,3.75f,-0.125f};array source(data,Shape{1,4},float32);
  auto value=softmax(source,-1,true,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::softmax_eval_layout(2,4,1,false,cold));REQUIRE(cpu::softmax_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[4]==3);CHECK(actual.request_counts[5]==3);
  CHECK(actual.request_counts[6]==1);CHECK(actual.request_counts[9]==1);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::softmax_eval_layout(2,0,1,false,actual));
  CHECK_FALSE(cpu::softmax_eval_layout(2,4,SIZE_MAX,false,actual));
  CHECK_FALSE(cpu::softmax_eval_layout(4,4,1,false,actual));
  auto wrong=array(Shape{4,1},float32,std::make_shared<Softmax>(stream,true),{source});
  auto dtype=array(Shape{1,4},int32,std::make_shared<Softmax>(stream,true),{source});
  struct Derived final:Softmax{using Softmax::Softmax;};
  auto subclass=array(Shape{1,4},float32,std::make_shared<Derived>(stream,true),{source});
  CHECK_FALSE(cpu::softmax_eval_storage(wrong,actual));CHECK_FALSE(cpu::softmax_eval_storage(dtype,actual));
  CHECK_FALSE(cpu::softmax_eval_storage(subclass,actual));CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_softmax_eval_layout(&raw,2,4,1,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);CHECK(raw.signal_graph_extents==0);
}
TEST_CASE("CPU Softmax preserves ordinary nonzero strided numerics and escaped original output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  const float data[]={-2.5f,99,0.25f,99,3.75f,99,-0.125f,99,1.5f,99,-3.0f,99,0.75f,99,2.0f,99};
  array base(data,Shape{2,8},float32);auto source=slice(base,Shape{0,0},Shape{2,8},Shape{1,2},stream);eval(source);
  REQUIRE_FALSE(source.flags().contiguous);auto ordinary=softmax(source,-1,true,stream);eval(ordinary);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,2)==0);
    auto value=softmax(source,-1,true,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,12};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=8;++i){CHECK(value.data<float>()[i]>0);CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);}
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.charged_bytes>=8*sizeof(float));escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  for(size_t i=0;i!=8;++i)CHECK(escaped->data<float>()[i]==ordinary.data<float>()[i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/backend/cpu/greedy_storage.h"
TEST_CASE("CPU greedy source preserves reduction and squeeze geometry and rejects malformed descriptors"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-2.5f,3.75f,3.75f,-0.125f,1.5f};array source(data,Shape{1,1,5},float32);
  auto reduction=argmax(source,-1,true,stream);
  auto value=squeeze(reduction,-1,stream);
  cpu::CopyEvalStorage cold,actual,alias;
  REQUIRE(cpu::arg_reduce_eval_layout(3,5,1,false,cold));REQUIRE(cpu::greedy_eval_storage(reduction,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[4]==2);CHECK(actual.request_counts[6]==1);
  eval(reduction); // The alias source authenticates an actual completed backing.
  REQUIRE(cpu::squeeze_eval_layout(3,false,alias));REQUIRE(cpu::greedy_eval_storage(value,actual));
  CHECK(actual.allocation_extents==alias.allocation_extents);CHECK(actual.backing_births==0);
  CHECK(actual.worker_graph_extents==0);CHECK(actual.request_counts[3]==0);CHECK(actual.request_counts[6]==0);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::arg_reduce_eval_layout(3,0,1,false,actual));
  CHECK_FALSE(cpu::arg_reduce_eval_layout(3,size_t(INT_MAX)+1,1,false,actual));
  CHECK_FALSE(cpu::arg_reduce_eval_layout(3,5,size_t(UINT32_MAX)+1,false,actual));
  CHECK_FALSE(cpu::arg_reduce_eval_layout(5,5,1,false,actual));
  CHECK_FALSE(cpu::arg_reduce_eval_layout(3,size_t(INT_MAX),size_t(UINT32_MAX),false,actual));
  CHECK_FALSE(cpu::squeeze_eval_layout(0,false,actual));CHECK_FALSE(cpu::squeeze_eval_layout(6,false,actual));
  auto wrong=array(Shape{1,1,2},uint32,std::make_shared<ArgReduce>(stream,ArgReduce::ArgMax,2),{source});
  auto axis=array(Shape{1,1,1},uint32,std::make_shared<ArgReduce>(stream,ArgReduce::ArgMax,3),{source});
  auto dtype=array(Shape{1,1,1},int32,std::make_shared<ArgReduce>(stream,ArgReduce::ArgMax,2),{source});
  auto kind=array(Shape{1,1,1},uint32,std::make_shared<ArgReduce>(stream,static_cast<ArgReduce::ReduceType>(99),2),{source});
  struct Derived final:ArgReduce{using ArgReduce::ArgReduce;};
  auto subclass=array(Shape{1,1,1},uint32,std::make_shared<Derived>(stream,ArgReduce::ArgMax,2),{source});
  CHECK_FALSE(cpu::greedy_eval_storage(wrong,actual));CHECK_FALSE(cpu::greedy_eval_storage(axis,actual));
  CHECK_FALSE(cpu::greedy_eval_storage(dtype,actual));CHECK_FALSE(cpu::greedy_eval_storage(kind,actual));
  CHECK_FALSE(cpu::greedy_eval_storage(subclass,actual));
  auto duplicate=array(Shape{1},uint32,std::make_shared<Squeeze>(stream,std::vector<int>{1,1}),{reduction});
  auto negative=array(Shape{1,1},uint32,std::make_shared<Squeeze>(stream,std::vector<int>{-1}),{reduction});
  auto wide=array(Shape{1,1},float32,std::make_shared<Squeeze>(stream,std::vector<int>{2}),{source});
  CHECK_FALSE(cpu::greedy_eval_storage(duplicate,actual));CHECK_FALSE(cpu::greedy_eval_storage(negative,actual));
  CHECK_FALSE(cpu::greedy_eval_storage(wide,actual));CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_greedy_eval_layout(&raw,3,5,1,true,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);CHECK(raw.signal_graph_extents==0);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_greedy_eval_layout(&raw,SIZE_MAX,5,1,true,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
}
TEST_CASE("CPU greedy preserves first ties and escaped original U32 output for strided input"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  const float data[]={-2.5f,99,3.75f,99,3.75f,99,-0.125f,99,1.5f,99,-3.0f,99,2.0f,99,2.0f,99};
  array base(data,Shape{2,8},float32);auto source=slice(base,Shape{0,0},Shape{2,8},Shape{1,2},stream);eval(source);
  REQUIRE_FALSE(source.flags().contiguous);auto ordinary=argmax(source,-1,false,stream);eval(ordinary);
  REQUIRE(ordinary.data<uint32_t>()[0]==1);REQUIRE(ordinary.data<uint32_t>()[1]==2);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,2)==0);
    auto value=argmax(source,-1,false,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,12};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    CHECK(value.shape()==Shape{2});CHECK(value.dtype()==uint32);
    CHECK(value.data<uint32_t>()[0]==1);CHECK(value.data<uint32_t>()[1]==2);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.charged_bytes>=2*sizeof(uint32_t));escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  for(size_t i=0;i!=2;++i)CHECK(escaped->data<uint32_t>()[i]==ordinary.data<uint32_t>()[i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/backend/cpu/reshape_storage.h"
TEST_CASE("CPU reshape source matches scalar aliases and distinguishes actual copy branches"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-2.5f,0.25f,3.75f,-0.125f,1.5f,2.0f};array source(data,Shape{1,1,6},float32);
  auto scalar=slice(source,Shape{0,0,4},Shape{1,1,5},Shape{1,1,1},stream);eval(scalar);
  REQUIRE(scalar.flags().row_contiguous);
  auto value=reshape(scalar,Shape{},stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::reshape_alias_eval_layout(3,0,false,cold));REQUIRE(cpu::reshape_alias_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==0);
  CHECK(actual.worker_graph_extents==0);CHECK(actual.request_counts[3]==0);CHECK(actual.request_counts[6]==0);
  auto matrix=reshape(source,Shape{2,3},stream);REQUIRE(cpu::reshape_alias_eval_storage(matrix,actual));
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::reshape_alias_eval_layout(6,0,false,actual));CHECK_FALSE(cpu::reshape_alias_eval_layout(3,6,false,actual));
  auto bad_declared=array(Shape{2,3},float32,std::make_shared<Reshape>(stream,Shape{3,2}),{source});
  auto bad_size=array(Shape{2,2},float32,std::make_shared<Reshape>(stream,Shape{2,2}),{source});
  auto bad_type=array(Shape{2,3},uint32,std::make_shared<Reshape>(stream,Shape{2,3}),{source});
  struct Derived final:Reshape{using Reshape::Reshape;};
  auto subclass=array(Shape{2,3},float32,std::make_shared<Derived>(stream,Shape{2,3}),{source});
  CHECK_FALSE(cpu::reshape_alias_eval_storage(bad_declared,actual));CHECK_FALSE(cpu::reshape_alias_eval_storage(bad_size,actual));
  CHECK_FALSE(cpu::reshape_alias_eval_storage(bad_type,actual));CHECK_FALSE(cpu::reshape_alias_eval_storage(subclass,actual));
  auto strided=transpose(matrix,stream);eval(strided);
  REQUIRE_FALSE(strided.flags().row_contiguous);auto copy=reshape(strided,Shape{6},stream);
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  REQUIRE(cpu::reshape_alias_eval_storage(copy,actual));CHECK(actual.backing_births==1);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_reshape_alias_eval_layout(&raw,3,0,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==0);CHECK(raw.signal_graph_extents==0);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_reshape_alias_eval_layout(&raw,SIZE_MAX,0,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
}
TEST_CASE("CPU probability alias preserves offset and actual original backing through retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  const float data[]={7,99,-11,99,23,99,13,99,17,99,-19,99};array base(data,Shape{1,1,12},float32);
  auto strided=slice(base,Shape{0,0,0},Shape{1,1,12},Shape{1,1,2},stream);eval(strided);
  std::optional<array> source,escaped;uint64_t identity=0;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,3)==0);
    auto value=contiguous(strided,false,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    REQUIRE(info.known);identity=info.identity;source.emplace(value);
  }
  const auto occupied=mlx_original_buffer_budget_occupied(budget.value);REQUIRE(occupied>0);
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,3)==0);
    auto selected=slice(*source,Shape{0,0,4},Shape{1,1,5},Shape{1,1,1},stream);
    auto value=reshape(selected,Shape{},stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,12};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    CHECK(value.ndim()==0);CHECK(value.data<float>()[0]==17.0f);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity==identity);escaped.emplace(value);
  }
  source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==occupied);CHECK(retired==0);
  CHECK(escaped->data<float>()[0]==17.0f);escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/backend/cpu/random_storage.h"
#include "mlx/random.h"
TEST_CASE("CPU RandomBits source counts actual raw task without weak aliases and rejects malformed keys"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);auto key=random::key(0x123456789abcdef0ULL);
  auto value=random::split(key,3,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::random_bits_eval_layout(2,6,false,cold));REQUIRE(cpu::random_bits_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[3]==1);CHECK(actual.request_counts[4]==0);CHECK(actual.request_counts[5]==0);
  CHECK(actual.request_counts[6]==1);CHECK(actual.worker_graph_extents==0);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::random_bits_eval_layout(2,0,false,actual));
  CHECK_FALSE(cpu::random_bits_eval_layout(2,size_t(UINT32_MAX)+1,false,actual));
  CHECK_FALSE(cpu::random_bits_eval_layout(5,6,false,actual));
  auto shape=array(Shape{2,3},uint32,std::make_shared<RandomBits>(stream,Shape{3,2},4),{key});
  auto width=array(Shape{3,2},uint32,std::make_shared<RandomBits>(stream,Shape{3,2},2),{key});
  auto dtype=array(Shape{3,2},int32,std::make_shared<RandomBits>(stream,Shape{3,2},4),{key});
  const uint32_t words[]={1,2,3,4};array batched(words,Shape{2,2},uint32);
  auto batch=array(Shape{3,2},uint32,std::make_shared<RandomBits>(stream,Shape{3,2},4),{batched});
  struct Derived final:RandomBits{using RandomBits::RandomBits;};
  auto subclass=array(Shape{3,2},uint32,std::make_shared<Derived>(stream,Shape{3,2},4),{key});
  CHECK_FALSE(cpu::random_bits_eval_storage(shape,actual));CHECK_FALSE(cpu::random_bits_eval_storage(width,actual));
  CHECK_FALSE(cpu::random_bits_eval_storage(dtype,actual));CHECK_FALSE(cpu::random_bits_eval_storage(batch,actual));
  CHECK_FALSE(cpu::random_bits_eval_storage(subclass,actual));CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_random_bits_eval_layout(&raw,2,6,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);CHECK(raw.signal_graph_extents==0);
}
TEST_CASE("CPU RandomBits preserves ordinary strided-key words and escaped original output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  const uint32_t words[]={0x12345678,99,0x9abcdef0,99};array base(words,Shape{4},uint32);
  auto key=slice(base,Shape{0},Shape{4},Shape{2},stream);eval(key);REQUIRE_FALSE(key.flags().contiguous);
  auto ordinary=random::bits(Shape{5},4,key,stream);eval(ordinary);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
    auto value=random::bits(Shape{5},4,key,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=5;++i)CHECK(value.data<uint32_t>()[i]==ordinary.data<uint32_t>()[i]);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.charged_bytes>=5*sizeof(uint32_t));escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  for(size_t i=0;i!=5;++i)CHECK(escaped->data<uint32_t>()[i]==ordinary.data<uint32_t>()[i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/backend/cpu/matmul_storage.h"
TEST_CASE("CPU BF16 Matmul source matches ordinary row geometry and rejects foreign equations"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const mlx::core::bfloat16_t ad[]={1,2,-1,3,0.5f,-2},bd[]={2,1,-3,4,0.5f,2};
  array a(ad,Shape{2,3},bfloat16),b(bd,Shape{3,2},bfloat16);
  auto value=matmul(a,b,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::bf16_matmul_eval_layout(2,2,2,3,1,false,cold));
  REQUIRE(cpu::bf16_matmul_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[3]==1);CHECK(actual.request_counts[4]==0);
  CHECK(actual.request_counts[6]==1);CHECK(actual.request_counts[9]==1);
  CHECK(actual.worker_graph_extents==0);
  auto bt=transpose(a,stream);eval(bt);auto transposed=matmul(a,bt,stream);
  REQUIRE(cpu::bf16_matmul_eval_storage(transposed,actual));
  auto batches=broadcast_to(reshape(a,Shape{1,2,3},stream),Shape{2,2,3},stream);
  auto batch_b=broadcast_to(reshape(b,Shape{1,3,2},stream),Shape{2,3,2},stream);eval(batches,batch_b);
  auto batched=matmul(batches,batch_b,stream);
  REQUIRE(cpu::bf16_matmul_eval_storage(batched,actual));
  REQUIRE(cpu::bf16_matmul_eval_layout(3,2,2,3,2,false,cold));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::bf16_matmul_eval_layout(1,2,2,3,1,false,actual));
  CHECK_FALSE(cpu::bf16_matmul_eval_layout(6,2,2,3,1,false,actual));
  CHECK_FALSE(cpu::bf16_matmul_eval_layout(2,2,2,0,1,false,actual));
  CHECK_FALSE(cpu::bf16_matmul_eval_layout(2,size_t(INT_MAX),2,2,1,false,actual));
  CHECK_FALSE(cpu::bf16_matmul_eval_layout(2,2,2,2,SIZE_MAX,false,actual));
  auto wrong=array(Shape{2,3},bfloat16,std::make_shared<Matmul>(stream),{a,b});
  auto dtype=array(Shape{2,2},float32,std::make_shared<Matmul>(stream),{a,b});
  struct Derived final:Matmul{using Matmul::Matmul;};
  auto subclass=array(Shape{2,2},bfloat16,std::make_shared<Derived>(stream),{a,b});
  auto addmm_value=array(Shape{2,2},bfloat16,std::make_shared<AddMM>(stream,1.0f,0.0f),{a,b,a});
  CHECK_FALSE(cpu::bf16_matmul_eval_storage(wrong,actual));CHECK_FALSE(cpu::bf16_matmul_eval_storage(dtype,actual));
  CHECK_FALSE(cpu::bf16_matmul_eval_storage(subclass,actual));CHECK_FALSE(cpu::bf16_matmul_eval_storage(addmm_value,actual));
  const mlx::core::bfloat16_t wide[]={1,99,2,99,-1,99,3,99,0.5f,99,-2,99};
  array raw(wide,Shape{2,6},bfloat16);auto strided=slice(raw,Shape{0,0},Shape{2,6},Shape{1,2},stream);eval(strided);
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  auto copies=matmul(strided,b,stream);REQUIRE(cpu::bf16_matmul_eval_storage(copies,actual));
  cpu::CopyEvalStorage copy_cold;REQUIRE(cpu::bf16_matmul_copy_eval_layout(2,2,2,3,1,1,false,copy_cold));
  CHECK(actual.allocation_extents==copy_cold.allocation_extents);CHECK(actual.backing_births==2);
  mlx_cpu_copy_eval_layout query{};REQUIRE(mlx_operation_event_cpu_bf16_matmul_eval_layout(&query,3,2,2,3,2,false));
  CHECK(query.graph_extents==cold.allocation_extents);CHECK(query.backing_births==1);CHECK(query.signal_graph_extents==0);
  auto prior=query;CHECK_FALSE(mlx_operation_event_cpu_bf16_matmul_eval_layout(&query,3,2,2,3,SIZE_MAX,false));
  CHECK(std::memcmp(&prior,&query,sizeof(query))==0);
}
TEST_CASE("CPU BF16 Matmul preserves nonzero row results and escaped original backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  const mlx::core::bfloat16_t ad[]={1,2,-1,3,0.5f,-2},bd[]={2,1,-3,4,0.5f,2};
  array a(ad,Shape{2,3},bfloat16),b(bd,Shape{3,2},bfloat16);
  auto ordinary=matmul(a,b,stream);eval(ordinary);
  const float expected[]={-4.5f,7.0f,3.5f,1.0f};
  for(size_t i=0;i!=4;++i)REQUIRE(float(ordinary.data<mlx::core::bfloat16_t>()[i])==expected[i]);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=matmul(a,b,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,12};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=4;++i)CHECK(float(value.data<mlx::core::bfloat16_t>()[i])==expected[i]);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.charged_bytes>=4*sizeof(mlx::core::bfloat16_t));escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  for(size_t i=0;i!=4;++i)CHECK(float(escaped->data<mlx::core::bfloat16_t>()[i])==expected[i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

#include "mlx/c/stream_copy.h"
TEST_CASE("CPU tiled Matmul choice is explicit immutable and source geometry checked"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_cpu_matmul_facts facts{};REQUIRE(mlx_cpu_matmul_facts_for(&facts)==0);
  CHECK(facts.tile_edge==16);CHECK(facts.reduction_lanes>0);CHECK(facts.max_rank==5);
  mlx_stream_copy_value captured{};REQUIRE(mlx_stream_copy_snapshot(&captured,{&stream})==0);
  REQUIRE(captured.cpu_matmul==0);auto chosen=captured;
  REQUIRE(mlx_stream_copy_select_cpu_matmul(&chosen,1)==0);CHECK(captured.cpu_matmul==0);
  struct Copy{mlx_stream value{};~Copy(){mlx_stream_copy_free(value);}}copy;
  REQUIRE(mlx_stream_copy_new(&copy.value,chosen)==0);
  const auto selected=*static_cast<Stream*>(copy.value.ctx);CHECK(selected==stream);
  CHECK(selected.cpu_matmul()==CpuMatmulKernel::Float32Tiles);
  mlx_stream_copy_value recaptured{};REQUIRE(mlx_stream_copy_snapshot(&recaptured,copy.value)==0);CHECK(recaptured.cpu_matmul==1);
  auto bad=chosen;bad.device_kind=1;auto prior=bad;
  CHECK(mlx_stream_copy_select_cpu_matmul(&bad,1)==2);CHECK(std::memcmp(&bad,&prior,sizeof(bad))==0);
  CHECK(mlx_stream_copy_select_cpu_matmul(&chosen,3)==2);CHECK(chosen.cpu_matmul==1);
  const float ad[]={1,2,-1,3,0.5f,-2},bd[]={2,1,-3,4,0.5f,2};
  array a(ad,Shape{2,3},float32),b(bd,Shape{3,2},float32);
  auto platform=matmul(a,b,stream),tiled=matmul(a,b,selected);
  CHECK_FALSE(platform.primitive().is_equivalent(tiled.primitive()));
  CHECK_FALSE(tiled.primitive().is_equivalent(platform.primitive()));
  cpu::CopyEvalStorage cold,actual;REQUIRE(cpu::tiled_matmul_eval_layout(2,2,2,3,1,false,cold));
  REQUIRE(cpu::tiled_matmul_eval_storage(tiled,actual));CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.backing_births==1);CHECK(actual.request_counts[6]==1);CHECK(actual.request_counts[9]==1);
  CHECK(actual.worker_graph_extents==0);CHECK(actual.request_counts[4]==0);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::tiled_matmul_eval_storage(platform,actual));
  CHECK_FALSE(cpu::tiled_matmul_eval_layout(2,size_t(INT_MAX),1,1,1,false,actual));
  CHECK_FALSE(cpu::tiled_matmul_eval_layout(3,65536,65536,1,1,false,actual));
  CHECK_FALSE(cpu::tiled_matmul_eval_layout(2,2,2,0,1,false,actual));
  CHECK_FALSE(cpu::tiled_matmul_eval_layout(2,2,2,3,2,false,actual));
  auto malformed=array(Shape{2,3},float32,std::make_shared<Matmul>(selected),{a,b});
  CHECK_FALSE(cpu::tiled_matmul_eval_storage(malformed,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  AddMM ordinary_add(stream,0.75f,-0.5f),selected_add(selected,0.75f,-0.5f);
  CHECK_FALSE(ordinary_add.is_equivalent(selected_add));
  mlx_cpu_copy_eval_layout query{};REQUIRE(mlx_operation_event_cpu_tiled_matmul_eval_layout(&query,2,2,2,3,1,false));
  CHECK(query.graph_extents==cold.allocation_extents);CHECK(query.backing_births==1);
}
TEST_CASE("CPU selected tiled Matmul matches shared ordinary tails transpose batches and escaped custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  constexpr int m=17,n=19,k=23;
  std::vector<float> left(m*k),right(k*n);
  for(size_t i=0;i!=left.size();++i)left[i]=(int(i%29)-14)*0.03125f;
  for(size_t i=0;i!=right.size();++i)right[i]=(int(i%17)-8)*0.0625f;
  array a(left.data(),Shape{m,k},float32),b(right.data(),Shape{k,n},float32);
  auto platform=matmul(a,b,stream),ordinary=matmul(a,b,selected);eval(platform,ordinary);
  for(size_t i=0;i!=size_t(m*n);++i)CHECK(std::abs(ordinary.data<float>()[i]-platform.data<float>()[i])<1e-5f);
  auto at=transpose(a,selected),bt=transpose(b,selected);eval(at,bt);
  auto transposed=matmul(bt,at,selected);eval(transposed);
  for(int i=0;i!=m;++i)for(int j=0;j!=n;++j)CHECK(transposed.data<float>()[j*m+i]==ordinary.data<float>()[i*n+j]);
  auto batched_a=broadcast_to(reshape(a,Shape{1,m,k},selected),Shape{2,m,k},selected);
  auto batched_b=broadcast_to(reshape(b,Shape{1,k,n},selected),Shape{2,k,n},selected);
  auto batched=matmul(batched_a,batched_b,selected);eval(batched);
  for(size_t i=0;i!=size_t(2*m*n);++i)CHECK(batched.data<float>()[i]==ordinary.data<float>()[i%(m*n)]);
  // AddMM uses the same retained choice and shared alpha/beta worker; its
  // original Eval source remains separately unqualified in this increment.
  auto c=array(0.25f,float32);auto sum=addmm(c,a,b,0.75f,-0.5f,selected);eval(sum);
  for(size_t i=0;i!=size_t(m*n);++i)CHECK(sum.data<float>()[i]==0.75f*ordinary.data<float>()[i]-0.125f);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=matmul(a,b,selected);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,12};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=size_t(m*n);++i)CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  for(size_t i=0;i!=size_t(m*n);++i)CHECK(escaped->data<float>()[i]==ordinary.data<float>()[i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

TEST_CASE("CPU F32 selection preserves existing BF16 row execution and source qualification"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  const mlx::core::bfloat16_t ad[]={1,2,-1,3,0.5f,-2},bd[]={2,1,-3,4,0.5f,2};
  array a(ad,Shape{2,3},bfloat16),b(bd,Shape{3,2},bfloat16);
  auto platform=matmul(a,b,stream),chosen=matmul(a,b,selected);
  cpu::CopyEvalStorage actual;
  REQUIRE(cpu::bf16_matmul_eval_storage(chosen,actual));
  CHECK_FALSE(cpu::tiled_matmul_eval_storage(chosen,actual));
  eval(platform,chosen);
  const float expected[]={-4.5f,7.0f,3.5f,1.0f};
  for(size_t i=0;i!=4;++i){CHECK(float(chosen.data<mlx::core::bfloat16_t>()[i])==expected[i]);CHECK(chosen.data<mlx::core::bfloat16_t>()[i]==platform.data<mlx::core::bfloat16_t>()[i]);}
}

#include "mlx/backend/cpu/alias_storage.h"
TEST_CASE("CPU shape alias source validates actual permutation broadcast and floating reshape geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-3,0.5f,7,2,-1.25f,11};array source(data,Shape{2,3},float32);
  auto transposed=transpose(source,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::alias_eval_layout(cpu::AliasOperation::Transpose,2,2,false,cold));
  REQUIRE(cpu::alias_eval_storage(transposed,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==0);
  CHECK(actual.worker_graph_extents==0);CHECK(actual.request_counts[3]==0);CHECK(actual.request_counts[6]==0);
  auto broadcast=broadcast_to(source,Shape{4,2,3},stream);
  REQUIRE(cpu::alias_eval_layout(cpu::AliasOperation::Broadcast,2,3,false,cold));
  REQUIRE(cpu::alias_eval_storage(broadcast,actual));CHECK(actual.allocation_extents==cold.allocation_extents);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::alias_eval_layout(cpu::AliasOperation::Transpose,2,3,false,actual));
  CHECK_FALSE(cpu::alias_eval_layout(cpu::AliasOperation::Broadcast,3,2,false,actual));
  CHECK_FALSE(cpu::alias_eval_layout(cpu::AliasOperation::Broadcast,2,6,false,actual));
  CHECK_FALSE(cpu::alias_eval_layout(static_cast<cpu::AliasOperation>(99),2,2,false,actual));
  auto repeated=array(Shape{2,2},float32,std::make_shared<Transpose>(stream,std::vector<int>{0,0}),{source});
  auto missing=array(Shape{3,2},float32,std::make_shared<Transpose>(stream,std::vector<int>{1}),{source});
  auto negative=array(Shape{3,2},float32,std::make_shared<Transpose>(stream,std::vector<int>{1,-1}),{source});
  auto wrong_shape=array(Shape{2,3},float32,std::make_shared<Transpose>(stream,std::vector<int>{1,0}),{source});
  auto incompatible=array(Shape{4,3},float32,std::make_shared<Broadcast>(stream,Shape{4,3}),{source});
  auto wrong_declared=array(Shape{4,2,3},float32,std::make_shared<Broadcast>(stream,Shape{5,2,3}),{source});
  struct Derived final:Transpose{using Transpose::Transpose;};
  auto subclass=array(Shape{3,2},float32,std::make_shared<Derived>(stream,std::vector<int>{1,0}),{source});
  for(const auto* invalid:{&repeated,&missing,&negative,&wrong_shape,&incompatible,&wrong_declared,&subclass})
    CHECK_FALSE(cpu::alias_eval_storage(*invalid,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_alias_eval_layout(&raw,1,2,3,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==0);auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_alias_eval_layout(&raw,99,2,3,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
  for(Dtype dtype:{float16,bfloat16}) {
    auto floating=astype(source,dtype,stream);eval(floating);
    auto flattened=reshape(floating,Shape{6},stream);REQUIRE(cpu::reshape_alias_eval_storage(flattened,actual));
    CHECK(actual.backing_births==0);eval(flattened);
    CHECK(flattened.data_size()==floating.data_size());
  }
}
TEST_CASE("CPU aliases preserve nonzero strided views and exact original backing after source retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  const float data[]={-3,99,0.5f,99,7,99,2,99,-1.25f,99,11,99};array base(data,Shape{2,6},float32);
  auto strided=slice(base,Shape{0,0},Shape{2,6},Shape{1,2},stream);eval(strided);
  std::optional<array> source,escaped;uint64_t identity=0;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=contiguous(strided,false,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    REQUIRE(info.known);identity=info.identity;source.emplace(value);
  }
  const auto occupied=mlx_original_buffer_budget_occupied(budget.value);REQUIRE(occupied>0);
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,3)==0);
    auto row=reshape(*source,Shape{1,2,3},stream);
    auto repeated=broadcast_to(row,Shape{4,2,3},stream);
    auto value=transpose(repeated,std::vector<int>{0,2,1},stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,5,4,4,4,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    REQUIRE(value.shape()==Shape{4,3,2});CHECK(value.strides()[0]==0);
    for(int batch=0;batch!=4;++batch)for(int row=0;row!=3;++row)for(int col=0;col!=2;++col)
      CHECK(value.data<float>()[batch*value.strides()[0]+row*value.strides()[1]+col*value.strides()[2]]==data[col*6+row*2]);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity==identity);escaped.emplace(value);
  }
  source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==occupied);CHECK(retired==0);
  CHECK(escaped->data<float>()[escaped->strides()[1]*2+escaped->strides()[2]]==11.0f);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

TEST_CASE("CPU selected dense frontend preserves nonzero flattened projection bias and escaped original output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  const float xd[]={-3,0.5f,7,2,-1.25f,11,1,4,-2,-0.25f,3,2};
  const float wd[]={1,-2,0.5f,3,0.25f,-1,-2,1,4,0.5f,-3,2,2,1,-0.5f};
  const float bd[]={0.5f,-1,2,0.25f,-3};
  array input(xd,Shape{2,2,3},float32),weight(wd,Shape{5,3},float32),bias(bd,Shape{5},float32);
  auto ordinary=add(matmul(input,transpose(weight,selected),selected),bias,selected);eval(ordinary);
  REQUIRE(ordinary.shape()==Shape{2,2,5});
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,6,0,3)==0);
    auto value=add(matmul(input,transpose(weight,selected),selected),bias,selected);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,11,7,10,7,1,24};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=20;++i)CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);
    CHECK(value.data<float>()[0]==0.0f);CHECK(value.data<float>()[1]==-16.875f);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  for(size_t i=0;i!=20;++i)CHECK(escaped->data<float>()[i]==ordinary.data<float>()[i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

TEST_CASE("CPU selected BF16 dense frontend preserves row rounding aliases bias and original retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  const mlx::core::bfloat16_t xd[]={-3,0.5f,7,2,-1.25f,11,1,4,-2,-0.25f,3,2};
  const mlx::core::bfloat16_t wd[]={1,-2,0.5f,3,0.25f,-1,-2,1,4,0.5f,-3,2,2,1,-0.5f};
  const mlx::core::bfloat16_t bd[]={0.5f,-1,2,0.25f,-3};
  array input(xd,Shape{2,2,3},bfloat16),weight(wd,Shape{5,3},bfloat16),bias(bd,Shape{5},bfloat16);
  auto ordinary=add(matmul(input,transpose(weight,selected),selected),bias,selected);eval(ordinary);
  REQUIRE(ordinary.dtype()==bfloat16);REQUIRE(ordinary.shape()==Shape{2,2,5});
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,6,0,3)==0);
    auto value=add(matmul(input,transpose(weight,selected),selected),bias,selected);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,11,7,10,7,1,24};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=20;++i)CHECK(value.data<mlx::core::bfloat16_t>()[i]==ordinary.data<mlx::core::bfloat16_t>()[i]);
    CHECK(float(value.data<mlx::core::bfloat16_t>()[0])==0.0f);CHECK(float(value.data<mlx::core::bfloat16_t>()[1])==-16.875f);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  for(size_t i=0;i!=20;++i)CHECK(escaped->data<mlx::core::bfloat16_t>()[i]==ordinary.data<mlx::core::bfloat16_t>()[i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

TEST_CASE("CPU F32 tile choice preserves ordinary empty branches without qualifying their original source"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  const float zero=0.0f;
  array a(&zero,Shape{2,0},float32),b(&zero,Shape{0,3},float32);
  auto ordinary=matmul(a,b,stream),chosen=matmul(a,b,selected);
  cpu::CopyEvalStorage unused;
  CHECK_FALSE(cpu::tiled_matmul_eval_storage(chosen,unused));
  eval(ordinary,chosen);
  for(size_t i=0;i!=6;++i){CHECK(chosen.data<float>()[i]==0.0f);CHECK(chosen.data<float>()[i]==ordinary.data<float>()[i]);}
  const float bias_data[]={1,-2,3};array bias(bias_data,Shape{3},float32);
  auto add_ordinary=addmm(bias,a,b,0.75f,-0.5f,stream),add_chosen=addmm(bias,a,b,0.75f,-0.5f,selected);
  eval(add_ordinary,add_chosen);
  for(size_t i=0;i!=6;++i)CHECK(add_chosen.data<float>()[i]==add_ordinary.data<float>()[i]);
  array empty(&zero,Shape{0,3},float32),weight(&zero,Shape{3,0},float32);
  auto empty_ordinary=matmul(empty,weight,stream),empty_chosen=matmul(empty,weight,selected);
  eval(empty_ordinary,empty_chosen);CHECK(empty_chosen.size()==0);CHECK(empty_chosen.shape()==empty_ordinary.shape());
}

TEST_CASE("CPU pointwise physical source refuses larger inherited contiguous backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float values[]={-3,0.5f,7,2,-1.25f,11};array source(values,Shape{6},float32);
  array shorter(Shape{3},float32,nullptr,{});shorter.copy_shared_buffer(source);
  REQUIRE(shorter.flags().contiguous);REQUIRE(shorter.data_size()>shorter.size());
  auto unary=square(shorter,stream);cpu::UnaryEvalStorage unary_source;
  CHECK_FALSE(cpu::unary_eval_storage(unary,unary_source));
  array right(values,Shape{3},float32);auto binary=add(shorter,right,stream);cpu::BinaryEvalStorage binary_source;
  CHECK_FALSE(cpu::binary_eval_storage(binary,binary_source));
  auto valid=square(source,stream);CHECK(cpu::unary_eval_storage(valid,unary_source));
}
TEST_CASE("CPU residual pointwise chain preserves nonzero ordinary values and escaped original backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float xd[]={-3,0.5f,7,2,-1.25f,11},gd[]={0.25f,-0.5f,0.125f};
  array input(xd,Shape{2,3},float32),gain(gd,Shape{3},float32);
  auto ordinary=add(tanh(square(multiply(input,gain,stream),stream),stream),input,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,5,0,2)==0);
    auto value=add(tanh(square(multiply(input,gain,stream),stream),stream),input,stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,9,6,8,6,1,20};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=6;++i)CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);
    CHECK(value.data<float>()[0]<-2.0f);CHECK(value.data<float>()[5]>11.0f);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}


#include "mlx/backend/cpu/gather_storage.h"
TEST_CASE("CPU Gather source validates exact one-index geometry counts and failure atomicity"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-3,0.5f,7,2,-1.25f,11,1,4,-2,0.25f,-4,9};
  const int32_t picks[]={2,-1,0,1};array source(data,Shape{4,3},float32),index(picks,Shape{2,2},int32);
  auto value=gather(source,index,0,Shape{1,3},stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::gather_eval_layout(float32,int32,2,2,12,4,3,false,cold));
  REQUIRE(cpu::gather_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.worker_graph_extents>0);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[3]==4);CHECK(actual.request_counts[4]==3);
  CHECK(actual.request_counts[5]==3);CHECK(actual.request_counts[9]==1);
  cpu::CopyEvalStorage empty;
  REQUIRE(cpu::gather_eval_layout(float32,int32,2,2,12,0,3,false,empty));
  CHECK(empty.backing_births==0);CHECK(empty.request_counts[3]==4);
  CHECK(empty.request_counts[6]==1);CHECK(empty.request_counts[7]==0);CHECK(empty.request_counts[9]==1);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::gather_eval_layout(float64,int32,2,2,12,4,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(float32,uint64,2,2,12,4,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(float32,int32,2,4,12,4,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(float32,int32,0,2,12,4,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(float32,int32,2,0,12,4,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(float32,int32,2,2,12,4,0,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(float32,int32,2,2,SIZE_MAX,4,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(float32,int32,2,2,12,size_t(INT_MAX),3,false,actual));
  auto shape_mismatch=array(Shape{2,2,1,2},float32,std::make_shared<Gather>(stream,std::vector<int>{0},Shape{1,3}),{source,index});
  auto wrong_axis=array(Shape{2,2,1,3},float32,std::make_shared<Gather>(stream,std::vector<int>{2},Shape{1,3}),{source,index});
  auto negative_axis=array(Shape{2,2,1,3},float32,std::make_shared<Gather>(stream,std::vector<int>{-1},Shape{1,3}),{source,index});
  auto partial=array(Shape{2,2,1,2},float32,std::make_shared<Gather>(stream,std::vector<int>{0},Shape{1,2}),{source,index});
  auto multiple=array(Shape{2,2,1,1},float32,std::make_shared<Gather>(stream,std::vector<int>{0,1},Shape{1,1}),{source,index,index});
  struct Derived final:Gather{using Gather::Gather;};
  auto subclass=array(Shape{2,2,1,3},float32,std::make_shared<Derived>(stream,std::vector<int>{0},Shape{1,3}),{source,index});
  for(const auto* invalid:{&shape_mismatch,&wrong_axis,&negative_axis,&partial,&multiple,&subclass})
    CHECK_FALSE(cpu::gather_eval_storage(*invalid,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_gather_eval_layout(&raw,
      MLX_FLOAT32,MLX_INT32,2,2,12,4,3,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_gather_eval_layout(&raw,
      static_cast<mlx_dtype>(99),MLX_INT32,2,2,12,4,3,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
}
TEST_CASE("CPU Gather and floating Squeeze preserve strided negative-index rows and escaped source custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float full_data[]={-3,99,0.5f,99,7,99,2,99,-1.25f,99,11,99,1,99,4,99,-2,99,0.25f,99,-4,99,9,99};
  const int32_t signed_data[]={2,99,-1,99,0,99,1,99};
  const uint32_t unsigned_data[]={2,99,3,99,0,99,1,99};
  array full(full_data,Shape{4,6},float32);
  for(Dtype dtype:{float32,bfloat16})for(Dtype index_dtype:{int32,uint32}) {
    auto typed=astype(full,dtype,stream);eval(typed);
    auto source=slice(typed,Shape{0,0},Shape{4,6},Shape{1,2},stream);eval(source);
    auto indices=index_dtype==int32?array(signed_data,Shape{2,4},int32):array(unsigned_data,Shape{2,4},uint32);
    auto index=slice(indices,Shape{0,0},Shape{2,4},Shape{1,2},stream);eval(index);
    REQUIRE_FALSE(source.flags().row_contiguous);REQUIRE_FALSE(index.flags().row_contiguous);
    auto ordinary=take(source,index,0,stream);eval(ordinary);REQUIRE(ordinary.shape()==Shape{2,2,3});
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,4)==0);
      auto value=take(source,index,0,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,6,4,5,4,1,16};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      for(size_t i=0;i!=12;++i) {
        const float actual=dtype==float32?value.data<float>()[i]:float(value.data<mlx::core::bfloat16_t>()[i]);
        const float expected=dtype==float32?ordinary.data<float>()[i]:float(ordinary.data<mlx::core::bfloat16_t>()[i]);
        CHECK(actual==expected);
      }
      const float first=dtype==float32?value.data<float>()[0]:float(value.data<mlx::core::bfloat16_t>()[0]);
      const float last=dtype==float32?value.data<float>()[11]:float(value.data<mlx::core::bfloat16_t>()[11]);
      CHECK(first==1.0f);CHECK(last==11.0f);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}


#include "mlx/backend/cpu/reduction_storage.h"
#include "mlx/backend/common/reduce.h"
TEST_CASE("CPU reduction source authenticates boolean whole-axis and F32 row geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const bool mask_data[]={true,false,true,true,false,true};array mask(mask_data,Shape{2,3},bool_);
  auto all_value=all(mask,true,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::BooleanAll,2,6,1,false,cold));
  REQUIRE(cpu::reduction_eval_storage(all_value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.worker_graph_extents==0);CHECK(actual.request_counts[3]==3);
  auto any_value=any(mask,true,stream);REQUIRE(cpu::reduction_eval_storage(any_value,actual));
  const float values[]={-3,0.5f,7,2,-1.25f,11};array input(values,Shape{2,3},float32);
  auto sum_value=sum(input,-1,true,stream);
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32Rows,2,3,2,false,cold));
  REQUIRE(cpu::reduction_eval_storage(sum_value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.named_control_bytes==cold.named_control_bytes);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::BooleanAny,2,6,2,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::BooleanAll,0,6,1,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32Rows,5,3,2,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32Rows,2,0,2,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32Rows,2,SIZE_MAX,2,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32Rows,2,size_t(INT_MAX),2,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(static_cast<cpu::ReductionEvalKind>(99),2,6,1,false,actual));
  auto partial=any(mask,0,true,stream);
  auto wrong_shape=array(Shape{2,2},float32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{1}),{input});
  auto wrong_axes=array(Shape{1,1},bool_,std::make_shared<Reduce>(stream,Reduce::Or,std::vector<int>{0,0}),{mask});
  struct Derived final:Reduce{using Reduce::Reduce;};
  auto subclass=array(Shape{2,1},float32,std::make_shared<Derived>(stream,Reduce::Sum,std::vector<int>{1}),{input});
  for(const auto* invalid:{&partial,&wrong_shape,&wrong_axes,&subclass})
    CHECK_FALSE(cpu::reduction_eval_storage(*invalid,actual));
  auto strided=slice(mask,Shape{0,0},Shape{2,3},Shape{1,2},stream);eval(strided);
  auto sparse=any(strided,true,stream);CHECK_FALSE(cpu::reduction_eval_storage(sparse,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_reduction_eval_layout(&raw,2,2,3,2,false));
  CHECK(raw.graph_extents==cold.allocation_extents);auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_reduction_eval_layout(&raw,99,2,3,2,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
  cpu::UnaryEvalStorage unary;REQUIRE(cpu::unary_eval_layout(cpu::UnaryEvalKind::logical_not,bool_,2,false,unary));
  CHECK_FALSE(cpu::unary_eval_layout(cpu::UnaryEvalKind::logical_not,float32,2,false,unary));
}
TEST_CASE("CPU boolean validation reductions preserve SIMD tails aliases and escaped retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(unsigned pattern=0;pattern!=3;++pattern)for(bool every:{false,true}) {
    std::array<bool,1537> data;
    for(size_t i=0;i!=data.size();++i)data[i]=pattern==0?false:pattern==1?true:i!=1536;
    array input(data.data(),Shape{29,53},bool_);
    auto ordinary=every?all(logical_not(input,stream),false,stream):any(logical_not(input,stream),false,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,2)==0);
      auto value=every?all(logical_not(input,stream),false,stream):any(logical_not(input,stream),false,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,6,5,5,5,1,16};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      CHECK(value.size()==1);CHECK(value.data<bool>()[0]==ordinary.data<bool>()[0]);
      CHECK(value.data<bool>()[0]==(every?pattern==0:pattern!=1));
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}
TEST_CASE("CPU F32 row sums keep the existing cascade and exact original result ownership"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,3*1031> data;const float pattern[]={-3,0.5f,7,2,-1.25f,11};
  for(size_t i=0;i!=data.size();++i)data[i]=pattern[i%6];
  array input(data.data(),Shape{3,1031},float32);auto ordinary=sum(input,-1,false,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,2)==0);
    auto value=sum(input,-1,false,stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,5,4,4,4,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t row=0;row!=3;++row) {
      double reference=0;for(size_t col=0;col!=1031;++col)reference+=data[row*1031+col];
      CHECK(value.data<float>()[row]==ordinary.data<float>()[row]);
      CHECK(value.data<float>()[row]==float(reference));CHECK(value.data<float>()[row]>0.0f);
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  escaped.reset();CHECK(retired==1);
}


#include "mlx/backend/cpu/selection_storage.h"
TEST_CASE("CPU selection and scalar Full source validate physical backing and immutable queries"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const bool md[]={true,false,true,false};const int32_t vd[]={2,-1,3,1};
  array mask(md,Shape{2,2},bool_),values(vd,Shape{2,2},int32);
  auto zeros=zeros_like(values,stream);auto selected=where(mask,values,zeros,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::scalar_full_eval_layout(int32,2,4,false,cold));
  // The actual Eval prologue runs after dependencies: inspect the completed
  // scalar broadcast backing, while keeping the Full output lazy.
  eval(zeros.inputs());
  REQUIRE(cpu::selection_eval_storage(zeros,actual));CHECK(actual.allocation_extents==cold.allocation_extents);
  eval(zeros);
  REQUIRE(cpu::select_eval_layout(int32,2,4,false,cold));
  REQUIRE(cpu::selection_eval_storage(selected,actual));CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.request_counts[3]==5);CHECK(actual.request_counts[4]==4);CHECK(actual.worker_graph_extents==0);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::select_eval_layout(float16,2,4,false,actual));
  CHECK_FALSE(cpu::select_eval_layout(int32,5,4,false,actual));
  CHECK_FALSE(cpu::select_eval_layout(int32,2,SIZE_MAX,false,actual));
  auto invalid=array(Shape{2,2},int32,std::make_shared<Select>(stream),{values,values,zeros});
  auto vector_full=array(Shape{2,2},int32,std::make_shared<Full>(stream),{values});
  CHECK_FALSE(cpu::selection_eval_storage(invalid,actual));CHECK_FALSE(cpu::selection_eval_storage(vector_full,actual));
  array too_large(Shape{2,1},int32,nullptr,{});too_large.copy_shared_buffer(values);
  auto short_mask=array(md,Shape{2,1},bool_);auto short_zero=array(vd,Shape{2,1},int32);
  auto inherited=array(Shape{2,1},int32,std::make_shared<Select>(stream),{short_mask,too_large,short_zero});
  CHECK_FALSE(cpu::selection_eval_storage(inherited,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_selection_eval_layout(&raw,MLX_INT32,2,4,false,false));
  CHECK(raw.graph_extents==cold.allocation_extents);auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_selection_eval_layout(&raw,static_cast<mlx_dtype>(99),2,4,false,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU strict lookup retains its real validation root safe indices and completed output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float wd[]={-3,0.5f,7,2,-1.25f,11,1,4,-2,0.25f,-4,9};
  array original_weight(wd,Shape{4,3},float32);
  for(Dtype dtype:{float32,bfloat16})for(bool invalid:{false,true}) {
    auto weight=astype(original_weight,dtype,stream);eval(weight);
    const int32_t td[]={2,invalid?-1:3,0,1};array tokens(td,Shape{2,2},int32);
    auto lookup=[&]() {
      auto normalized=astype(tokens,int32,stream);
      auto valid=logical_and(greater_equal(normalized,array(0),stream),less(normalized,array(4),stream),stream);
      auto rejected=any(logical_not(valid,stream),false,stream);
      normalized=astype(normalized,int32,stream);
      auto selected=logical_and(greater_equal(normalized,array(0),stream),less(normalized,array(4),stream),stream);
      auto safe=where(selected,normalized,zeros_like(normalized,stream),stream);
      return std::make_pair(take(weight,safe,0,stream),rejected);
    };
    auto ordinary=lookup();eval(ordinary.first,ordinary.second);CHECK(ordinary.second.data<bool>()[0]==invalid);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,50,5,4)==0);
      auto value=lookup();bank.reset();Operation operation;operation.append(value.first);operation.append(value.second);
      const mlx_operation_eval_traversal_limits limits{2,60,51,61,51,1,200};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value.first);
      CHECK(value.second.data<bool>()[0]==invalid);
      for(size_t row=0;row!=4;++row)for(size_t col=0;col!=3;++col) {
        const size_t offset=row*3+col;const float actual=dtype==float32?value.first.data<float>()[offset]:float(value.first.data<mlx::core::bfloat16_t>()[offset]);
        const float expected=dtype==float32?ordinary.first.data<float>()[offset]:float(ordinary.first.data<mlx::core::bfloat16_t>()[offset]);
        CHECK(actual==expected);CHECK(actual==wd[(td[row]<0?0:td[row])*3+col]);
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value.first},budget.value)==0);
      CHECK(info.known);escaped.emplace(value.first);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}

#include "mlx/fast.h"
TEST_CASE("CPU RMS fallback controls validate exact geometry without constructor side effects"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  size_t controls=0;
  REQUIRE(fast::cpu_rms_fallback_control_bytes(float32,3,17,6,controls));
  CHECK(controls>0);const auto saved=controls;
  CHECK_FALSE(fast::cpu_rms_fallback_control_bytes(float64,3,17,6,controls));
  CHECK_FALSE(fast::cpu_rms_fallback_control_bytes(float32,0,17,6,controls));
  CHECK_FALSE(fast::cpu_rms_fallback_control_bytes(float32,5,17,6,controls));
  CHECK_FALSE(fast::cpu_rms_fallback_control_bytes(float32,3,0,6,controls));
  CHECK_FALSE(fast::cpu_rms_fallback_control_bytes(float32,3,SIZE_MAX,6,controls));
  CHECK_FALSE(fast::cpu_rms_fallback_control_bytes(float32,3,17,SIZE_MAX,controls));
  CHECK(controls==saved);
  REQUIRE(mlx_operation_event_cpu_rms_fallback_control_bytes(&controls,MLX_BFLOAT16,3,17,6));
  CHECK(controls>saved);const auto raw=controls;
  CHECK_FALSE(mlx_operation_event_cpu_rms_fallback_control_bytes(&controls,static_cast<mlx_dtype>(99),3,17,6));
  CHECK(controls==raw);
}
namespace cpu_rms_rounding_tests {
void run(Dtype dtype) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(size_t width:{size_t(1),size_t(17),size_t(1031)}) {
    std::vector<float> data(3*width),gain(width);
    for(size_t i=0;i<data.size();++i)data[i]=(int(i%23)-11)*0.3125f;
    for(size_t i=0;i<width;++i)gain[i]=0.25f+float(i%7)*0.1875f;
    array wide_input(data.data(),Shape{3,int(width)},float32),wide_gain(gain.data(),Shape{int(width)},float32);
    auto input=astype(wide_input,dtype,stream),weight=astype(wide_gain,dtype,stream);eval(input,weight);
    auto normalize=[&]() {
      // Match input_precision_rms's actual CPU-declined custom-kernel probe.
      if(dtype!=float32) {auto probe=astype(input,float32,stream);(void)probe;}
      return fast::rms_norm(input,weight,1e-6f,stream);
    };
    auto ordinary=normalize();eval(ordinary);
    auto wide=astype(input,float32,stream);
    auto normalized=multiply(wide,rsqrt(add(divide(sum(square(wide,stream),-1,true,stream),
        array(float(width),float32),stream),array(1e-6f,float32),stream),stream),stream);
    auto expected=multiply(astype(normalized,dtype,stream),weight,stream);eval(expected);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,27,2,2)==0);
      auto value=normalize();bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,34,28,52,28,1,120};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      CHECK(value.shape()==input.shape());CHECK(value.dtype()==dtype);
      bool nonzero=false;
      for(size_t i=0;i<data.size();++i) {
        const float actual=dtype==float32?value.data<float>()[i]:dtype==bfloat16?float(value.data<mlx::core::bfloat16_t>()[i]):float(value.data<mlx::core::float16_t>()[i]);
        const float reference=dtype==float32?expected.data<float>()[i]:dtype==bfloat16?float(expected.data<mlx::core::bfloat16_t>()[i]):float(expected.data<mlx::core::float16_t>()[i]);
        const float same=dtype==float32?ordinary.data<float>()[i]:dtype==bfloat16?float(ordinary.data<mlx::core::bfloat16_t>()[i]):float(ordinary.data<mlx::core::float16_t>()[i]);
        CHECK(actual==same);CHECK(actual==reference);nonzero|=actual!=0;
      }
      CHECK(nonzero);mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}
}
TEST_CASE("CPU RMS shared fallback preserves cascade rounding gain and escaped original backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  for(Dtype dtype:{float32,bfloat16})cpu_rms_rounding_tests::run(dtype);
}
TEST_CASE("CPU F16 RMS shares exact fallback controls and half gain rounding"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  size_t controls=0;
  REQUIRE(fast::cpu_rms_fallback_control_bytes(float16,3,17,6,controls));
  const size_t native=controls;CHECK(native>0);
  REQUIRE(mlx_operation_event_cpu_rms_fallback_control_bytes(&controls,MLX_FLOAT16,3,17,6));
  CHECK(controls>native);
  cpu_rms_rounding_tests::run(float16);
}

TEST_CASE("CPU activation sources retain F32 intermediates and original escaped results"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-11.5f,-5.25f,-2.0f,-0.25f,0.0f,0.125f,1.25f,3.5f,9.25f};
  array original(data,Shape{3,3},float32);
  for(Dtype dtype:{float32,bfloat16})for(bool silu:{false,true}) {
    auto input=astype(original,dtype,stream);eval(input);
    auto activation=[&]() {
      auto work=dtype==bfloat16?astype(input,float32,stream):input;
      auto output=silu?divide(work,add(exp(negative(work,stream),stream),array(1.0f,float32),stream),stream)
          :sigmoid(work,stream);
      return astype(output,dtype,stream);
    };
    auto ordinary=activation();eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,14,silu?1:0,2)==0);
      auto value=activation();bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,20,15,30,15,1,60};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      CHECK(value.dtype()==dtype);CHECK(value.shape()==input.shape());
      for(size_t i=0;i<9;++i) {
        const float actual=dtype==float32?value.data<float>()[i]:float(value.data<mlx::core::bfloat16_t>()[i]);
        const float reference=dtype==float32?ordinary.data<float>()[i]:float(ordinary.data<mlx::core::bfloat16_t>()[i]);
        const float expected=(silu?data[i]:1.0f)/(1.0f+std::exp(-data[i]));
        CHECK(actual==reference);CHECK(std::abs(actual-expected)<(dtype==float32?1e-5f:0.04f));
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU ExpandDims source validates exact inserted axes without allocating backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float values[]={-3,0.5f,7,2,-1.25f,11};array input(values,Shape{2,3},float32);
  auto expanded=expand_dims(input,1,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::alias_eval_layout(cpu::AliasOperation::ExpandDims,2,3,false,cold));
  REQUIRE(cpu::alias_eval_storage(expanded,actual));CHECK(actual.backing_births==0);
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.named_control_bytes==cold.named_control_bytes);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::alias_eval_layout(cpu::AliasOperation::ExpandDims,2,2,false,actual));
  CHECK_FALSE(cpu::alias_eval_layout(cpu::AliasOperation::ExpandDims,2,6,false,actual));
  auto wrong=array(Shape{1,2,3},float32,std::make_shared<ExpandDims>(stream,std::vector<int>{1}),{input});
  auto duplicate=array(Shape{2,1,1,3},float32,std::make_shared<ExpandDims>(stream,std::vector<int>{1,1}),{input});
  CHECK_FALSE(cpu::alias_eval_storage(wrong,actual));CHECK_FALSE(cpu::alias_eval_storage(duplicate,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_alias_eval_layout(&raw,2,2,3,false));
  CHECK(raw.backing_births==0);
}
TEST_CASE("CPU row views retain the exact original backing through every escaped alias"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float values[]={-3,0.5f,7,2,-1.25f,11};array original(values,Shape{2,3},float32);
  for(Dtype dtype:{float32,bfloat16}) {
    auto input=astype(original,dtype,stream);eval(input);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> first,second;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,4,0,3)==0);
      auto squared=square(input,stream);auto expanded=expand_dims(squared,1,stream);
      auto reshaped=reshape(expanded,Shape{1,6,1},stream);auto value=squeeze(reshaped,std::vector<int>{0,2},stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,7,5,6,5,1,24};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      CHECK(value.shape()==Shape{6});CHECK(value.dtype()==dtype);
      for(size_t i=0;i<6;++i) {
        const float actual=dtype==float32?value.data<float>()[i]:float(value.data<mlx::core::bfloat16_t>()[i]);
        CHECK(actual==values[i]*values[i]);
      }
      if(dtype==float32)CHECK(value.data<float>()==squared.data<float>());
      else CHECK(value.data<mlx::core::bfloat16_t>()==squared.data<mlx::core::bfloat16_t>());
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);
      first.emplace(squared);second.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    first.reset();CHECK(retired==0);second.reset();CHECK(retired==1);
  }
}

#include "mlx/backend/cpu/concatenate_storage.h"
TEST_CASE("CPU concatenation source counts actual slices weak copies and queued jobs"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float av[]={-3,0.5f,7,2,-1.25f,11},bv[]={9,-4,1,3};
  array a(av,Shape{2,3},float32),b(bv,Shape{2,2},float32);
  auto value=concatenate({a,b},1,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::concatenate_eval_layout(float32,2,6,4,false,cold));
  REQUIRE(cpu::concatenate_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.backing_births==1);CHECK(actual.request_counts[3]==5);
  CHECK(actual.request_counts[4]==6);CHECK(actual.request_counts[5]==6);CHECK(actual.request_counts[6]==2);
  cpu::CopyEvalStorage supported;
  REQUIRE(cpu::concatenate_eval_layout(float16,2,6,4,false,supported));
  CHECK(supported.backing_births==1);CHECK(supported.request_counts[6]==2);
  REQUIRE(cpu::concatenate_eval_layout(float32,2,0,4,false,supported));
  CHECK(supported.backing_births==1);CHECK(supported.request_counts[3]==5);
  CHECK(supported.request_counts[6]==2);
  auto arity=concatenate({a,a,a},1,stream);
  REQUIRE(cpu::concatenate_many_eval_layout(float32,2,3,18,false,cold));
  REQUIRE(cpu::concatenate_eval_storage(arity,supported));
  CHECK(supported.allocation_extents==cold.allocation_extents);
  CHECK(supported.named_control_bytes==cold.named_control_bytes);
  CHECK(supported.backing_births==1);CHECK(supported.request_counts[3]==7);
  CHECK(supported.request_counts[6]==3);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::concatenate_eval_layout(float64,2,6,4,false,actual));
  CHECK_FALSE(cpu::concatenate_eval_layout(float32,0,6,4,false,actual));
  CHECK_FALSE(cpu::concatenate_eval_layout(float32,5,6,4,false,actual));
  CHECK_FALSE(cpu::concatenate_eval_layout(float32,2,size_t(INT_MAX),1,false,actual));
  auto wrong=array(Shape{2,4},float32,std::make_shared<Concatenate>(stream,1),{a,b});
  auto axis=array(Shape{2,5},float32,std::make_shared<Concatenate>(stream,2),{a,b});
  CHECK_FALSE(cpu::concatenate_eval_storage(wrong,actual));CHECK_FALSE(cpu::concatenate_eval_storage(axis,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
}
TEST_CASE("CPU Gemma entrance concatenation preserves row order and escaped backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,64> av,bv;for(size_t i=0;i<64;++i){av[i]=float(int(i)-32)*0.125f;bv[i]=float(i+1)*0.25f;}
  array left(av.data(),Shape{2,1,32},float32),right(bv.data(),Shape{2,1,32},float32);
  for(Dtype dtype:{float32,bfloat16}) {
    auto a=astype(left,dtype,stream),b=astype(right,dtype,stream);eval(a,b);
    auto ordinary=concatenate({a,b},-1,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,3)==0);
      auto value=concatenate({a,b},-1,stream);bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,7,4,8,4,1,32};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      CHECK(value.shape()==Shape{2,1,64});
      for(size_t row=0;row<2;++row)for(size_t col=0;col<64;++col) {
        const size_t i=row*64+col;
        const float actual=dtype==float32?value.data<float>()[i]:float(value.data<mlx::core::bfloat16_t>()[i]);
        const float expected=col<32?av[row*32+col]:bv[row*32+col-32];
        const float prior=dtype==float32?ordinary.data<float>()[i]:float(ordinary.data<mlx::core::bfloat16_t>()[i]);
        CHECK(actual==prior);CHECK(actual==expected);
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);
      escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);escaped.reset();CHECK(retired==1);
  }
}

#include "mlx/backend/cpu/arange_storage.h"
TEST_CASE("CPU rotary range source matches finite primitive geometry and immutable query"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(int length:{1,4,1031})for(int step:{-1,1}) {
    auto value=arange(0.0,double(step)*length,double(step),float32,stream);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::arange_float_eval_layout(size_t(length),false,cold));
    REQUIRE(cpu::arange_float_eval_storage(value,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.backing_births==1);CHECK(actual.inputs==0);CHECK(actual.worker_graph_extents==0);
    std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
    auto malformed=array(Shape{length+1},float32,std::make_shared<Arange>(stream,0.0,double(step)*length,double(step)),{});
    CHECK_FALSE(cpu::arange_float_eval_storage(malformed,actual));
    CHECK_FALSE(cpu::arange_float_eval_layout(0,false,actual));
    CHECK_FALSE(cpu::arange_float_eval_layout(size_t(INT_MAX)+1,false,actual));
    CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  }
  size_t controls=0;REQUIRE(fast::cpu_rope_fallback_control_bytes(4,8,32,controls));
  const auto saved=controls;
  REQUIRE(fast::cpu_rope_fallback_control_bytes(3,8,32,controls));CHECK(controls==saved);
  CHECK_FALSE(fast::cpu_rope_fallback_control_bytes(4,7,32,controls));CHECK(controls==saved);
  CHECK_FALSE(fast::cpu_rope_fallback_control_bytes(2,8,32,controls));CHECK(controls==saved);
  CHECK_FALSE(fast::cpu_rope_fallback_control_bytes(5,8,32,controls));CHECK(controls==saved);
  CHECK_FALSE(fast::cpu_rope_fallback_control_bytes(3,8,0,controls));CHECK(controls==saved);
}
TEST_CASE("CPU Gemma rotary fallback preserves nonzero offsets ranges and escaped backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::vector<float> data(96);for(size_t i=0;i<data.size();++i)data[i]=(int(i%19)-9)*0.1875f;
  array input(data.data(),Shape{1,4,3,8},float32);
  auto ordinary=fast::rope(input,8,false,10000.0f,1.0f,2,{},stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<23,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,80,3,4)==0);
    auto value=fast::rope(input,8,false,10000.0f,1.0f,2,{},stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,100,81,160,81,1,400};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    CHECK(value.shape()==input.shape());CHECK(value.dtype()==float32);bool changed=false;
    for(size_t i=0;i<data.size();++i) {CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);changed|=value.data<float>()[i]!=data[i];}
    CHECK(changed);mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU supplied rotary frequencies preserve rank three values and independent source custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::vector<float> data(24);for(size_t i=0;i<data.size();++i)data[i]=(int(i%17)-8)*0.125f;
  const float denominators[]={1.0f,3.0f,11.0f,31.0f};
  array input(data.data(),Shape{1,3,8},float32),freqs(denominators,Shape{4},float32);
  const auto frequency_backing=freqs.data_shared_ptr();
  auto ordinary=fast::rope(input,8,false,{},1.0f,5,freqs,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<23,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,80,3,4)==0);
    auto value=fast::rope(input,8,false,{},1.0f,5,freqs,stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,100,81,160,81,1,400};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    CHECK(value.shape()==input.shape());CHECK(value.dtype()==float32);
    for(size_t row=0;row<3;++row)for(size_t col=0;col<4;++col) {
      const float angle=float(row+5)/denominators[col];
      const float left=data[row*8+col],right=data[row*8+col+4];
      CHECK(value.data<float>()[row*8+col]==doctest::Approx(left*std::cos(angle)-right*std::sin(angle)).epsilon(2e-6));
      CHECK(value.data<float>()[row*8+col+4]==doctest::Approx(right*std::cos(angle)+left*std::sin(angle)).epsilon(2e-6));
    }
    for(size_t i=0;i<data.size();++i)CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);
    CHECK(freqs.data_shared_ptr()==frequency_backing);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);
    escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  escaped.reset();CHECK(retired==1);
  CHECK(freqs.data_shared_ptr()==frequency_backing);
}

TEST_CASE("CPU Gemma grouped attention uses rank five source and unchanged selected equations"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  std::vector<float> qv(32),kv(48),vv(48);
  for(size_t i=0;i<qv.size();++i)qv[i]=(int(i%11)-5)*0.125f;
  for(size_t i=0;i<kv.size();++i){kv[i]=(int(i%17)-8)*0.0625f;vv[i]=(int(i%13)-6)*0.1875f;}
  array q(qv.data(),Shape{1,4,1,8},float32),k(kv.data(),Shape{1,2,3,8},float32),v(vv.data(),Shape{1,2,3,8},float32);
  const float scale=1.0f/std::sqrt(8.0f);
  auto ordinary=fast::scaled_dot_product_attention(q,k,v,scale,"",{},{},selected);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<23,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,24,1,5)==0);
    auto value=fast::scaled_dot_product_attention(q,k,v,scale,"",{},{},selected);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,48,25,60,25,1,120};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);eval_traversal_tests::complete(role,operation,value);
    CHECK(value.shape()==q.shape());CHECK(value.dtype()==float32);bool nonzero=false;
    for(size_t i=0;i<qv.size();++i){CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);nonzero|=value.data<float>()[i]!=0;}
    CHECK(nonzero);
    for(int head=0;head<4;++head){
      double scores[3],denominator=0;const int shared=head/2;
      for(int key=0;key<3;++key){double dot=0;
        for(int d=0;d<8;++d)dot+=double(qv[head*8+d])*kv[(shared*3+key)*8+d];
        scores[key]=std::exp(dot*scale);denominator+=scores[key];
      }
      for(int d=0;d<8;++d){double expected=0;
        for(int key=0;key<3;++key)expected+=scores[key]/denominator*vv[(shared*3+key)*8+d];
        CHECK(std::abs(double(value.data<float>()[head*8+d])-expected)<1e-5);
      }
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU ordered partition source preserves rank three geometry and immutable refusal"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={1,7,-3,4,2,2,0,3,1,8,-2,7,7,0};
  array input(data,Shape{1,2,7},float32);
  auto partition=argpartition(input,-2,-1,stream);
  submission::CpuArgPartitionLayout cold,actual;
  REQUIRE(submission::cpu_argpartition_source_layout(3,14,false,cold));
  REQUIRE(submission::cpu_argpartition_source_storage(partition,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.request_counts[3]==3);CHECK(actual.request_counts[4]==1);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(submission::cpu_argpartition_source_layout(4,14,false,actual));
  CHECK_FALSE(submission::cpu_argpartition_source_layout(0,14,false,actual));
  CHECK_FALSE(submission::cpu_argpartition_source_layout(3,0,false,actual));
  CHECK_FALSE(submission::cpu_argpartition_source_layout(3,SIZE_MAX,false,actual));
  auto bad_axis=array(input.shape(),uint32,std::make_shared<ArgPartition>(stream,1,1),{input});
  auto bad_kth=array(input.shape(),uint32,std::make_shared<ArgPartition>(stream,7,2),{input});
  auto bad_shape=array(Shape{1,2,6},uint32,std::make_shared<ArgPartition>(stream,5,2),{input});
  auto bad_dtype=array(input.shape(),int32,std::make_shared<ArgPartition>(stream,5,2),{input});
  for(const auto* value:{&bad_axis,&bad_kth,&bad_shape,&bad_dtype})
    CHECK_FALSE(submission::cpu_argpartition_source_storage(*value,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_argpartition_layout raw{};
  REQUIRE(mlx_operation_event_cpu_argpartition_source_layout(&raw,3,14,false));
  CHECK(raw.allocation_extents==cold.allocation_extents);auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_argpartition_source_layout(&raw,3,SIZE_MAX,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU ordered rank three partition keeps shared equal key and NaN ordering"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,42> data;
  for(size_t i=0;i<data.size();++i)data[i]=float(int(i%7)-3);
  data[2]=data[1];data[8]=std::numeric_limits<float>::quiet_NaN();data[17]=data[16];
  array input(data.data(),Shape{2,3,7},float32);
  auto ordinary=argpartition(input,-2,-1,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,3)==0);
    auto value=argpartition(input,-2,-1,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,2,3,2,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i<data.size();++i)CHECK(value.data<uint32_t>()[i]==ordinary.data<uint32_t>()[i]);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU ordered Gather source binds integer tables and rank five selection"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const int32_t tokens[]={8,1,5,3,11,0,9,2,7,10,4,6};
  const uint32_t picks[]={2,0,3,1,1,2,0,3};
  array source(tokens,Shape{4,3},int32),index(picks,Shape{2,2,2},uint32);
  auto value=gather(source,index,0,Shape{1,3},stream);
  REQUIRE(value.shape()==Shape{2,2,2,1,3});
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::gather_eval_layout(int32,uint32,2,3,12,8,3,false,cold));
  REQUIRE(cpu::gather_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.request_counts[9]==1);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::gather_eval_layout(int64,uint32,2,3,12,8,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(int32,uint32,2,4,12,8,3,false,actual));
  CHECK_FALSE(cpu::gather_eval_layout(int32,uint32,2,3,12,SIZE_MAX,3,false,actual));
  auto invalid=array(Shape{2,2,2,1,2},int32,std::make_shared<Gather>(stream,
      std::vector<int>{0},Shape{1,3}),{source,index});
  CHECK_FALSE(cpu::gather_eval_storage(invalid,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_gather_eval_layout(&raw,MLX_INT32,MLX_UINT32,2,3,12,8,3,false));
  CHECK(raw.graph_extents==cold.allocation_extents);
  CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
}
TEST_CASE("CPU ordered Gather preserves integer permutations repeated picks and escaped backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const int32_t tokens[]={8,1,5,3,11,0,9,2,7,10,4,6};
  const uint32_t picks[]={2,0,3,1,1,2,0,3};
  array source(tokens,Shape{4,3},int32),index(picks,Shape{2,2,2},uint32);
  auto ordinary=take(source,index,0,stream);eval(ordinary);
  REQUIRE(ordinary.shape()==Shape{2,2,2,3});
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,5)==0);
    auto value=take(source,index,0,stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,6,4,5,4,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t i=0;i!=24;++i) {
      CHECK(value.data<int32_t>()[i]==ordinary.data<int32_t>()[i]);
      CHECK(value.data<int32_t>()[i]==tokens[picks[i/3]*3+i%3]);
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(escaped->data<int32_t>()[0]==9);CHECK(escaped->data<int32_t>()[23]==6);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU ordered mask source matches row minimum and stored broadcast geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,42> data;for(size_t i=0;i!=data.size();++i)data[i]=float(int(i%19)-9)*0.25f;
  array input(data.data(),Shape{2,3,7},float32);
  auto minimum=min(input,-1,true,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32MinimumRows,3,7,6,false,cold));
  REQUIRE(cpu::reduction_eval_storage(minimum,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==0);CHECK(actual.named_control_bytes==cold.named_control_bytes);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32MinimumRows,3,1,6,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32MinimumRows,3,SIZE_MAX,6,false,actual));
  auto wrong_axis=min(input,0,true,stream);CHECK_FALSE(cpu::reduction_eval_storage(wrong_axis,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  eval(minimum);auto masked=full(Shape{2,3,11},minimum,float32,stream);eval(masked.inputs());
  REQUIRE(cpu::row_full_eval_layout(float32,3,11,6,false,cold));
  REQUIRE(cpu::selection_eval_storage(masked,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::row_full_eval_layout(bfloat16,3,11,6,false,actual));
  CHECK_FALSE(cpu::row_full_eval_layout(float32,3,11,0,false,actual));
  CHECK_FALSE(cpu::row_full_eval_layout(float32,3,11,SIZE_MAX,false,actual));
  array full_input(data.data(),Shape{2,3,7},float32);
  auto arbitrary=full(Shape{2,3,7},full_input,float32,stream);eval(arbitrary.inputs());
  CHECK_FALSE(cpu::selection_eval_storage(arbitrary,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_row_full_eval_layout(&raw,3,11,6,false));
  CHECK(raw.graph_extents==cold.allocation_extents);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_row_full_eval_layout(&raw,3,11,SIZE_MAX,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
}
#include "mlx/backend/common/utils.h"
TEST_CASE("CPU ordered mask preserves row minima NaNs fill values and escaped output custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,42> data;for(size_t i=0;i!=data.size();++i)data[i]=float(int((i*7)%19)-9)*0.25f;
  data[17]=std::numeric_limits<float>::quiet_NaN();
  array input(data.data(),Shape{2,3,7},float32);
  auto ordinary=full(Shape{2,3,11},min(input,-1,true,stream),float32,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  // Full's existing Vector copy preserves a compact per-row broadcast view.
  // Logical elements must use the actual strides, including after escape.
  const auto logical=[](const array& value,int position) {
    return value.data<float>()[elem_to_loc(position,value.shape(),value.strides())];
  };
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,4,0,3)==0);
    auto value=full(Shape{2,3,11},min(input,-1,true,stream),float32,stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,7,5,6,5,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t row=0;row!=6;++row) {
      float expected=std::numeric_limits<float>::infinity();bool has_nan=false;
      for(size_t column=0;column!=7;++column) {
        const float x=data[row*7+column];has_nan|=std::isnan(x);expected=std::min(expected,x);
      }
      for(size_t column=0;column!=11;++column) {
        const float actual=logical(value,int(row*11+column)),reference=logical(ordinary,int(row*11+column));
        CHECK(((std::isnan(actual)&&std::isnan(reference))||actual==reference));
        // The independent scalar minimum covers finite rows; NaN rows
        // retain the exact ordinary SIMD reduction behavior checked above.
        if(!has_nan)CHECK(actual==expected);
      }
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(logical(*escaped,0)<0.0f);
  const float actual=logical(*escaped,22),reference=logical(ordinary,22);
  CHECK(((std::isnan(actual)&&std::isnan(reference))||actual==reference));
  escaped.reset();CHECK(retired==1);
}

#include "mlx/backend/cpu/scatter_storage.h"
TEST_CASE("CPU ordered scatter source counts copy and overwrite jobs with exact geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float source_data[]={-3,2,1,7,-2,4,0,3,8,-1,2,4,5,6};
  const int32_t picks[]={-1,2,2,0,1,1,4,-2};
  const float updates_data[]={0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f};
  array source(source_data,Shape{1,2,7},float32),index(picks,Shape{1,2,4},int32),updates(updates_data,Shape{1,2,4},float32);
  auto value=put_along_axis(source,index,updates,-1,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::scatter_axis_eval_layout(int32,3,14,8,false,cold));
  REQUIRE(cpu::scatter_axis_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.request_counts[3]==6);CHECK(actual.request_counts[4]==5);
  CHECK(actual.request_counts[5]==5);CHECK(actual.request_counts[6]==2);CHECK(actual.backing_births==1);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::scatter_axis_eval_layout(int64,3,14,8,false,actual));
  CHECK_FALSE(cpu::scatter_axis_eval_layout(int32,0,14,8,false,actual));
  CHECK_FALSE(cpu::scatter_axis_eval_layout(int32,3,SIZE_MAX,8,false,actual));
  CHECK_FALSE(cpu::scatter_axis_eval_layout(int32,3,14,SIZE_MAX,false,actual));
  auto sum=array(Shape{1,2,7},float32,std::make_shared<ScatterAxis>(stream,ScatterAxis::Sum,2),{source,index,updates});
  auto axis=array(Shape{1,2,7},float32,std::make_shared<ScatterAxis>(stream,ScatterAxis::None,1),{source,index,updates});
  auto wrong_shape=array(Shape{1,2,6},float32,std::make_shared<ScatterAxis>(stream,ScatterAxis::None,2),{source,index,updates});
  for(const auto* invalid:{&sum,&axis,&wrong_shape})CHECK_FALSE(cpu::scatter_axis_eval_storage(*invalid,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_scatter_axis_eval_layout(&raw,MLX_INT32,3,14,8,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
}
TEST_CASE("CPU ordered scatter preserves duplicate negative indices and escaped overwrite custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,42> source_data;for(size_t i=0;i!=source_data.size();++i)source_data[i]=float(int(i)-21)*0.25f;
  std::array<float,24> updates_data;for(size_t i=0;i!=updates_data.size();++i)updates_data[i]=float(i+1)*0.125f;
  std::array<int32_t,24> signed_picks;std::array<uint32_t,24> unsigned_picks;
  const int32_t row_picks[]={-1,2,2,0};
  for(size_t i=0;i!=24;++i){signed_picks[i]=row_picks[i%4];unsigned_picks[i]=signed_picks[i]<0?6:uint32_t(signed_picks[i]);}
  array source(source_data.data(),Shape{2,3,7},float32),updates(updates_data.data(),Shape{2,3,4},float32);
  for(bool signed_index:{false,true}) {
    auto index=signed_index?array(signed_picks.data(),Shape{2,3,4},int32):array(unsigned_picks.data(),Shape{2,3,4},uint32);
    auto ordinary=put_along_axis(source,index,updates,-1,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      // Same constructor candidates as the shared composed recipe: one cast,
      // two broadcast inputs, three ignored-axis broadcasts, then Scatter.
      // Even elided nodes build seven geometric ArrayVector destinations;
      // reserving only the one executed primitive supplies just three slots.
      constexpr size_t candidates=1+2+3+1;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,candidates,0,3)==0);
      auto value=put_along_axis(source,index,updates,-1,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,6,3,5,3,1,16};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      for(size_t row=0;row!=6;++row)for(size_t column=0;column!=7;++column) {
        float expected=source_data[row*7+column];
        for(size_t pick=0;pick!=4;++pick)if(unsigned_picks[row*4+pick]==column)expected=updates_data[row*4+pick];
        CHECK(value.data<float>()[row*7+column]==expected);
        CHECK(value.data<float>()[row*7+column]==ordinary.data<float>()[row*7+column]);
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(escaped->data<float>()[2]==updates_data[2]);CHECK(escaped->data<float>()[41]==updates_data[20]);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU Reshape source resolves one inferred dimension and rejects malformed retained requests"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,32> data;for(size_t i=0;i<data.size();++i)data[i]=float(int(i)-13)*0.125f;
  array source(data.data(),Shape{1,1,32},float32);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::reshape_alias_eval_layout(3,4,false,cold));
  const Shape resolved{1,1,4,8};
  for(size_t axis=0;axis!=resolved.size();++axis) {
    auto requested=resolved;requested[axis]=-1;
    auto value=reshape(source,requested,stream);
    REQUIRE(value.shape()==resolved);
    REQUIRE(cpu::reshape_alias_eval_storage(value,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.backing_births==0);CHECK(actual.worker_graph_extents==0);
    eval(value);for(size_t i=0;i!=data.size();++i)CHECK(value.data<float>()[i]==data[i]);
  }
  const auto saved=actual;
  for(const auto& requested:std::vector<Shape>{{-1,-1,4,8},{1,0,4,8},{1,-2,4,8},{1,2,-1,8},{1,1,-1,7}}) {
    auto value=array(resolved,float32,std::make_shared<Reshape>(stream,requested),{source});
    CHECK_FALSE(cpu::reshape_alias_eval_storage(value,actual));
    CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
  }
  auto mismatched=array(Shape{1,1,4,7},float32,std::make_shared<Reshape>(stream,Shape{1,1,-1,7}),{source});
  CHECK_FALSE(cpu::reshape_alias_eval_storage(mismatched,actual));
  CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
}
TEST_CASE("CPU inferred Reshape preserves original nonzero backing and escaped alias retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::array<float,64> data;for(size_t i=0;i!=data.size();++i)data[i]=float(int(i)-19)*0.0625f;
  array base(data.data(),Shape{1,1,64},float32);
  auto strided=slice(base,Shape{0,0,0},Shape{1,1,64},Shape{1,1,2},stream);eval(strided);
  std::optional<array> source,escaped;uint64_t identity=0;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,3)==0);
    auto value=contiguous(strided,false,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    REQUIRE(info.known);identity=info.identity;source.emplace(value);
  }
  const auto occupied=mlx_original_buffer_budget_occupied(budget.value);REQUIRE(occupied>0);
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,4)==0);
    auto value=reshape(*source,Shape{1,1,-1,8},stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    CHECK(value.shape()==Shape{1,1,4,8});
    for(size_t i=0;i!=32;++i)CHECK(value.data<float>()[i]==data[2*i]);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity==identity);escaped.emplace(value);
  }
  source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==occupied);CHECK(retired==0);
  for(size_t i=0;i!=32;++i)CHECK(escaped->data<float>()[i]==data[2*i]);
  escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}


TEST_CASE("CPU all-axis F32 sum source authenticates complete axes and exact contiguous backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-3.f,0.5f,7.f,2.f,-1.25f,11.f};array input(data,Shape{2,3},float32);
  auto value=sum(input,true,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32All,2,6,1,false,cold));
  REQUIRE(cpu::reduction_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);CHECK(actual.backing_births==1);
  auto saved=actual;
  auto repeated=array(Shape{1,1},float32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{0,0}),{input});
  auto missing=array(Shape{1,1},float32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{0}),{input});
  auto wrong_shape=array(Shape{1,3},float32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{0,1}),{input});
  struct Derived final:Reduce{using Reduce::Reduce;};
  auto subclass=array(Shape{1,1},float32,std::make_shared<Derived>(stream,Reduce::Sum,std::vector<int>{0,1}),{input});
  auto sparse=slice(input,Shape{0,0},Shape{2,3},Shape{1,2},stream);eval(sparse);
  auto sparse_sum=sum(sparse,true,stream);
  for(const auto* invalid:{&repeated,&missing,&wrong_shape,&subclass,&sparse_sum})
    CHECK_FALSE(cpu::reduction_eval_storage(*invalid,actual));
  CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32All,1,6,1,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32All,2,1,1,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32All,2,6,2,false,actual));
  CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32All,2,SIZE_MAX,1,false,actual));
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_reduction_eval_layout(&raw,4,2,6,1,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
}
TEST_CASE("CPU all-axis F32 sum preserves ordinary SIMD tails and escaped source custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,3*1031> data;const float pattern[]={-3,0.5f,7,2,-1.25f,11};
  double expected=0;for(size_t i=0;i!=data.size();++i){data[i]=pattern[i%6];expected+=data[i];}
  array input(data.data(),Shape{3,1031},float32);auto ordinary=sum(input,false,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,2)==0);
    auto value=sum(input,false,stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,5,4,4,4,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    CHECK(value.ndim()==0);CHECK(value.data<float>()[0]==ordinary.data<float>()[0]);
    CHECK(value.data<float>()[0]==float(expected));CHECK(value.data<float>()[0]>0.f);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(escaped->data<float>()[0]==float(expected));escaped.reset();CHECK(retired==1);
}


TEST_CASE("CPU I32 coordinate source checks exact endpoints and final increment"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(int offset:{0,16777217,INT_MAX-7}) {
    auto value=arange(double(offset),double(offset)+7.0,1.0,int32,stream);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::arange_int_eval_layout(int32,7,false,cold));REQUIRE(cpu::arange_int_eval_storage(value,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.inputs==0);CHECK(actual.backing_births==1);CHECK(actual.worker_graph_extents==0);
    std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
    for(const auto& endpoints:std::array<std::array<double,3>,5>{{
        {{-1.0,6.0,1.0}},{{0.5,7.5,1.0}},{{0.0,15.0,2.0}},
        {{double(INT_MAX)-6.0,double(INT_MAX)+1.0,1.0}},{{7.0,0.0,-1.0}}}}) {
      auto malformed=array(Shape{7},int32,std::make_shared<Arange>(stream,endpoints[0],endpoints[1],endpoints[2]),{});
      CHECK_FALSE(cpu::arange_int_eval_storage(malformed,actual));
      CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
    }
    auto wrong=array(Shape{8},int32,std::make_shared<Arange>(stream,double(offset),double(offset)+7.0,1.0),{});
    CHECK_FALSE(cpu::arange_int_eval_storage(wrong,actual));
    CHECK_FALSE(cpu::arange_int_eval_layout(int32,0,false,actual));
    CHECK_FALSE(cpu::arange_int_eval_layout(int32,size_t(INT_MAX)+1,false,actual));
    CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  }
}
TEST_CASE("CPU I32 coordinates preserve values above float precision and escaped source custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  for(int offset:{0,16777217,INT_MAX-7}) {
    auto ordinary=arange(double(offset),double(offset)+7.0,1.0,int32,stream);eval(ordinary);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
      auto value=arange(double(offset),double(offset)+7.0,1.0,int32,stream);bank.reset();
      Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,16};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      CHECK(value.dtype()==int32);CHECK(value.shape()==Shape{7});
      for(int i=0;i<7;++i){CHECK(value.data<int32_t>()[i]==offset+i);CHECK(value.data<int32_t>()[i]==ordinary.data<int32_t>()[i]);}
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);
      escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU broadcast Select source retains exact readable spans and scratch populations"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Shape shape{2,3,2,3,5};
  std::array<bool,20> masks;std::array<float,180> numbers;
  for(size_t i=0;i<masks.size();++i)masks[i]=i%3!=1;
  for(size_t i=0;i<numbers.size();++i)numbers[i]=float(i)-89.f;
  auto mask=broadcast_to(array(masks.data(),Shape{2,1,2,1,5},bool_),shape,stream);
  array values(numbers.data(),shape,float32);
  auto floor=broadcast_to(array(-7.f),shape,stream);eval(mask,floor);
  auto selected=where(mask,values,floor,stream);eval(selected.inputs());
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::select_broadcast_eval_layout(5,180,false,cold));
  REQUIRE(cpu::selection_eval_storage(selected,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CollapseStorageLayout collapse;ContiguousIteratorLayout iterator;
  REQUIRE(collapse_contiguous_dims_layout(4,5,true,collapse));
  REQUIRE(contiguous_iterator_layout(3,iterator));
  CHECK(cold.worker_graph_extents==collapse.graph_extent_sum+3*iterator.graph_extent_sum);
  CHECK(cold.worker_graph_extents>0);CHECK(cold.backing_births==1);
  const auto saved=actual;
  CHECK_FALSE(cpu::select_broadcast_eval_layout(0,180,false,actual));
  CHECK_FALSE(cpu::select_broadcast_eval_layout(6,180,false,actual));
  CHECK_FALSE(cpu::select_broadcast_eval_layout(5,0,false,actual));
  CHECK_FALSE(cpu::select_broadcast_eval_layout(5,SIZE_MAX,false,actual));
  auto wrong=array(shape,float32,std::make_shared<Select>(stream),{values,values,floor});
  CHECK_FALSE(cpu::selection_eval_storage(wrong,actual));
  auto short_values=array(Shape{2,3,2,3,5},float32,nullptr,{});
  short_values.copy_shared_buffer(array(1.f));
  auto outside=array(shape,float32,std::make_shared<Select>(stream),{mask,short_values,floor});
  CHECK_FALSE(cpu::selection_eval_storage(outside,actual));
  CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_select_broadcast_eval_layout(&raw,5,180,false));
  CHECK(raw.graph_extents==cold.allocation_extents);
  CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
  const auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_select_broadcast_eval_layout(&raw,6,180,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
}
TEST_CASE("CPU broadcast Select preserves ordinary mask values and escaped native backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(bool grouped:{false,true}) {
    const Shape shape=grouped?Shape{2,3,2,3,5}:Shape{2,3,5};
    const Shape mask_shape=grouped?Shape{2,1,2,1,5}:Shape{2,1,5};
    const size_t count=grouped?180:30,mask_count=grouped?20:10;
    std::array<bool,20> masks;std::array<float,180> numbers;
    for(size_t i=0;i<mask_count;++i)masks[i]=i%3!=1;
    for(size_t i=0;i<count;++i)numbers[i]=i%17==0?std::numeric_limits<float>::quiet_NaN():float(i)-11.25f;
    auto mask=broadcast_to(array(masks.data(),mask_shape,bool_),shape,stream);
    array values(numbers.data(),shape,float32);
    auto floor=broadcast_to(array(std::numeric_limits<float>::lowest()),shape,stream);
    eval(mask,floor);
    auto ordinary=where(mask,values,floor,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,16,0,5)==0);
      auto output=where(mask,values,floor,stream);bank.reset();
      Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,32,17,48,17,1,64};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      for(size_t i=0;i<count;++i) {
        const size_t mask_offset=grouped?((i/90)*2+(i/15)%2)*5+i%5:(i/15)*5+i%5;
        const float expected=masks[mask_offset]?numbers[i]:std::numeric_limits<float>::lowest();
        const float actual=output.data<float>()[i],reference=ordinary.data<float>()[i];
        CHECK(((std::isnan(actual)&&std::isnan(reference))||actual==reference));
        CHECK(((std::isnan(actual)&&std::isnan(expected))||actual==expected));
      }
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);CHECK(info.known);
      escaped.emplace(output);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU explicit SDPA mask controls cover the same borrowed worker and refuse overflow"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  size_t base=0,masked=0;
  REQUIRE(fast::cpu_sdpa_fallback_control_bytes(5,96,160,160,60,base));
  REQUIRE(fast::cpu_sdpa_array_mask_control_bytes(5,96,160,160,60,masked));
  CHECK(masked>base);const auto saved=masked;
  CHECK_FALSE(fast::cpu_sdpa_array_mask_control_bytes(3,96,160,160,60,masked));
  CHECK_FALSE(fast::cpu_sdpa_array_mask_control_bytes(5,96,0,160,60,masked));
  CHECK_FALSE(fast::cpu_sdpa_array_mask_control_bytes(5,96,160,160,SIZE_MAX,masked));
  CHECK(masked==saved);
  size_t wrapper=0;
  REQUIRE(mlx_operation_event_cpu_sdpa_array_mask_control_bytes(&wrapper,5,96,160,160,60));
  CHECK(wrapper>masked);const auto prior=wrapper;
  CHECK_FALSE(mlx_operation_event_cpu_sdpa_array_mask_control_bytes(&wrapper,6,96,160,160,60));
  CHECK(wrapper==prior);
}
TEST_CASE("CPU explicit masked SDPA preserves nonzero MHA GQA equations and final custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  for(int heads:{2,4}) {
    constexpr int queries=3,keys=5,width=8,kv_heads=2;
    std::vector<float> qv(size_t(heads*queries*width)),kv(kv_heads*keys*width),vv(kv.size());
    std::array<bool,queries*keys> mask_values;
    for(size_t i=0;i<qv.size();++i)qv[i]=(int(i%11)-5)*0.125f;
    for(size_t i=0;i<kv.size();++i){kv[i]=(int(i%17)-8)*0.0625f;vv[i]=(int(i%13)-6)*0.1875f;}
    for(int query=0;query<queries;++query)for(int key=0;key<keys;++key)
      mask_values[query*keys+key]=key<=query+1&&key>=query;
    array q(qv.data(),Shape{1,heads,queries,width},float32),
        k(kv.data(),Shape{1,kv_heads,keys,width},float32),v(vv.data(),Shape{1,kv_heads,keys,width},float32),
        mask(mask_values.data(),Shape{queries,keys},bool_);
    const float scale=1.0f/std::sqrt(float(width));
    auto ordinary=fast::scaled_dot_product_attention(q,k,v,scale,"array",mask,{},selected);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<23,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,40,2,5)==0);
      auto output=fast::scaled_dot_product_attention(q,k,v,scale,"array",mask,{},selected);
      bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,80,41,100,41,1,200};
      REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==q.shape());CHECK(output.dtype()==float32);bool nonzero=false;
      for(size_t i=0;i<qv.size();++i){CHECK(output.data<float>()[i]==ordinary.data<float>()[i]);nonzero|=output.data<float>()[i]!=0;}
      CHECK(nonzero);
      for(int head=0;head<heads;++head)for(int query=0;query<queries;++query) {
        double scores[keys],denominator=0;const int shared=head/(heads/kv_heads);
        for(int key=0;key<keys;++key) {
          double dot=0;for(int d=0;d<width;++d)
            dot+=double(qv[(head*queries+query)*width+d])*kv[(shared*keys+key)*width+d];
          scores[key]=mask_values[query*keys+key]?std::exp(dot*scale):0;denominator+=scores[key];
        }
        for(int d=0;d<width;++d) {
          double expected=0;for(int key=0;key<keys;++key)
            expected+=scores[key]/denominator*vv[(shared*keys+key)*width+d];
          CHECK(std::abs(double(output.data<float>()[(head*queries+query)*width+d])-expected)<1e-5);
        }
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU final-axis row Sum source preserves general plan and refuses unknown strides"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,2*3*4*17> data;for(size_t i=0;i<data.size();++i)data[i]=(int(i%29)-14)*0.125f;
  array base(data.data(),Shape{2,3,4,17},float32);
  auto input=transpose(base,{0,2,1,3},stream);eval(input);
  REQUIRE_FALSE(input.flags().row_contiguous);REQUIRE(input.strides().back()==1);
  auto output=sum(input,-1,true,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32Rows,4,17,24,false,cold));
  REQUIRE(cpu::reduction_eval_storage(output,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.backing_births==1);CHECK(actual.worker_graph_extents==0);
  auto plan=get_reduction_plan(input,std::vector<int>{3});
  CHECK(plan.type==GeneralContiguousReduce);CHECK(plan.shape==Shape{17});CHECK(plan.strides==Strides{1});
  const auto saved=actual;
  auto wrong=transpose(base,{0,1,3,2},stream);eval(wrong);auto bad=sum(wrong,-1,true,stream);
  CHECK_FALSE(cpu::reduction_eval_storage(bad,actual));
  auto reversed=slice(input,Shape{0,0,0,16},Shape{2,4,3,-18},Shape{1,1,1,-1},stream);eval(reversed);
  auto reversed_sum=sum(reversed,-1,true,stream);CHECK_FALSE(cpu::reduction_eval_storage(reversed_sum,actual));
  CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
}
TEST_CASE("CPU strided head RMS keeps ordinary general-row rounding and escaped custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(size_t width:{size_t(7),size_t(17),size_t(1031)}) {
    const size_t rows=2*3*4;std::vector<float> data(rows*width),gain(width);
    for(size_t i=0;i<data.size();++i)data[i]=(int(i%29)-14)*0.15625f;
    for(size_t i=0;i<width;++i)gain[i]=0.25f+float(i%7)*0.1875f;
    array base(data.data(),Shape{2,3,4,int(width)},float32),weight(gain.data(),Shape{int(width)},float32);
    auto input=transpose(base,{0,2,1,3},stream);eval(input,weight);
    REQUIRE_FALSE(input.flags().row_contiguous);
    auto ordinary=fast::rms_norm(input,weight,1e-6f,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<24,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,27,2,2)==0);
      auto value=fast::rms_norm(input,weight,1e-6f,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,34,28,52,28,1,120};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      REQUIRE(value.flags().row_contiguous);CHECK(value.shape()==input.shape());
      bool nonzero=false;
      for(size_t b=0;b<2;++b)for(size_t h=0;h<4;++h)for(size_t q=0;q<3;++q) {
        const size_t source=((b*3+q)*4+h)*width,output=((b*4+h)*3+q)*width;
        double sum=0;for(size_t d=0;d<width;++d)sum+=double(data[source+d])*data[source+d];
        for(size_t d=0;d<width;++d) {
          const auto actual=value.data<float>()[output+d];
          const double reference=double(data[source+d])/std::sqrt(sum/width+1e-6)*gain[d];
          CHECK(actual==ordinary.data<float>()[output+d]);
          CHECK(std::abs(double(actual)-reference)<2e-5);nonzero|=actual!=0;
        }
      }
      CHECK(nonzero);mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU tiled Matmul fallback source counts actual temporary copies and preserves refusal"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  std::array<float,2*3*5*7> ad;std::array<float,2*7*5*11> bd;
  for(size_t i=0;i<ad.size();++i)ad[i]=(int(i%17)-8)*0.125f;
  for(size_t i=0;i<bd.size();++i)bd[i]=(int(i%19)-9)*0.0625f;
  array a_source(ad.data(),Shape{2,3,5,7},float32),b_source(bd.data(),Shape{2,7,5,11},float32);
  auto a=transpose(a_source,{0,2,1,3},selected),b=transpose(b_source,{0,2,1,3},selected);eval(a,b);
  auto output=matmul(a,b,selected);cpu::CopyEvalStorage cold,actual,no_copy;
  REQUIRE(cpu::tiled_matmul_eval_layout(4,3,11,7,10,false,no_copy));
  REQUIRE(cpu::tiled_matmul_copy_eval_layout(4,3,11,7,10,2,false,cold));
  REQUIRE(cpu::tiled_matmul_eval_storage(output,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.backing_births==3);CHECK(actual.request_counts[6]==3);
  CHECK(actual.request_counts[4]==6);CHECK(actual.request_counts[5]==6);
  CHECK(actual.request_counts[9]==1);CHECK(actual.request_counts[2]==no_copy.request_counts[2]+1);
  const auto prior=actual;
  CHECK_FALSE(cpu::tiled_matmul_copy_eval_layout(4,3,11,7,10,3,false,actual));
  CHECK_FALSE(cpu::tiled_matmul_copy_eval_layout(4,3,11,7,SIZE_MAX,2,false,actual));
  auto platform=matmul(a,b,stream);CHECK_FALSE(cpu::tiled_matmul_eval_storage(platform,actual));
  CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_tiled_matmul_copy_eval_layout(&raw,4,3,11,7,10,2,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==3);
  const auto raw_prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_tiled_matmul_copy_eval_layout(&raw,4,3,11,7,10,3,false));
  CHECK(std::memcmp(&raw_prior,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU selected Matmul two-copy fallback keeps numeric order and final custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  constexpr int m=3,n=5,k=7,batches=2,heads=3;
  std::vector<float> av(batches*m*heads*k),bv(batches*k*heads*n);
  for(size_t i=0;i<av.size();++i)av[i]=(int(i%17)-8)*0.125f;
  for(size_t i=0;i<bv.size();++i)bv[i]=(int(i%19)-9)*0.0625f;
  array a_source(av.data(),Shape{batches,m,heads,k},float32),b_source(bv.data(),Shape{batches,k,heads,n},float32);
  auto a=transpose(a_source,{0,2,1,3},selected),b=transpose(b_source,{0,2,1,3},selected);eval(a,b);
  auto ordinary=matmul(a,b,selected);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,2)==0);
    auto value=matmul(a,b,selected);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,8,4,8,4,1,32};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);eval_traversal_tests::complete(role,operation,value);
    bool nonzero=false;
    for(int batch=0;batch<batches;++batch)for(int head=0;head<heads;++head)
      for(int row=0;row<m;++row)for(int col=0;col<n;++col) {
        double expected=0;for(int inner=0;inner<k;++inner)
          expected+=double(av[((batch*m+row)*heads+head)*k+inner])*bv[((batch*k+inner)*heads+head)*n+col];
        const size_t index=((batch*heads+head)*m+row)*n+col;
        CHECK(value.data<float>()[index]==ordinary.data<float>()[index]);
        CHECK(value.data<float>()[index]==float(expected));nonzero|=value.data<float>()[index]!=0;
      }
    CHECK(nonzero);mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  escaped.reset();CHECK(retired==1);
}
TEST_CASE("CPU masked attention copies strided values with exact ordinary output and retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  for(int heads:{2,4}) {
    constexpr int queries=3,keys=5,width=8,kv_heads=2;
    std::vector<float> qv(size_t(heads*queries*width)),kv(kv_heads*keys*width),vv(kv.size());
    std::array<bool,queries*keys> mask_values;
    for(size_t i=0;i<qv.size();++i)qv[i]=(int(i%11)-5)*0.125f;
    for(size_t i=0;i<kv.size();++i){kv[i]=(int(i%17)-8)*0.0625f;vv[i]=(int(i%13)-6)*0.1875f;}
    for(int query=0;query<queries;++query)for(int key=0;key<keys;++key)
      mask_values[query*keys+key]=key<=query+1&&key>=query;
    array q(qv.data(),Shape{1,heads,queries,width},float32),
        k(kv.data(),Shape{1,kv_heads,keys,width},float32),v_source(vv.data(),Shape{1,keys,kv_heads,width},float32),
        mask(mask_values.data(),Shape{queries,keys},bool_);
    auto v=transpose(v_source,{0,2,1,3},selected);eval(v);
    REQUIRE_FALSE(v.flags().row_contiguous);
    const float scale=1.0f/std::sqrt(float(width));
    auto ordinary=fast::scaled_dot_product_attention(q,k,v,scale,"array",mask,{},selected);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<23,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,40,2,5)==0);
      auto output=fast::scaled_dot_product_attention(q,k,v,scale,"array",mask,{},selected);
      bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,80,41,100,41,1,200};
      REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==q.shape());CHECK(output.dtype()==float32);bool nonzero=false;
      for(size_t i=0;i<qv.size();++i){CHECK(output.data<float>()[i]==ordinary.data<float>()[i]);nonzero|=output.data<float>()[i]!=0;}
      CHECK(nonzero);
      for(int head=0;head<heads;++head)for(int query=0;query<queries;++query) {
        double scores[keys],denominator=0;const int shared=head/(heads/kv_heads);
        for(int key=0;key<keys;++key) {
          double dot=0;for(int d=0;d<width;++d)
            dot+=double(qv[(head*queries+query)*width+d])*kv[(shared*keys+key)*width+d];
          scores[key]=mask_values[query*keys+key]?std::exp(dot*scale):0;denominator+=scores[key];
        }
        for(int d=0;d<width;++d) {
          double expected=0;for(int key=0;key<keys;++key)
            expected+=scores[key]/denominator*vv[(key*kv_heads+shared)*width+d];
          CHECK(std::abs(double(output.data<float>()[(head*queries+query)*width+d])-expected)<1e-5);
        }
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU fixed reshape source preserves collapse decisions and rejects malformed extents"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,2*3*4*8> data;for(size_t i=0;i<data.size();++i)data[i]=(int(i%31)-15)*0.125f;
  array base(data.data(),Shape{2,4,3,8},float32);
  auto input=transpose(base,{0,2,1,3},stream);eval(input);
  REQUIRE_FALSE(input.flags().row_contiguous);
  for(const auto& shape:std::vector<Shape>{{2,3,32},{2,3,4,8,1},{2,3,4,2,4},{6,32},{192}}) {
    auto output=reshape(input,shape,stream);FixedReshapePlan plan;
    REQUIRE(fixed_reshape_plan(input.shape().data(),input.strides().data(),input.ndim(),false,
        shape.data(),shape.size(),plan));
    // The existing unbounded collapse worker supplies the independent prior
    // alias decision and strides; numeric execution shares the fixed worker.
    auto collapsed=collapse_contiguous_dims(input.shape(),input.strides(),int64_t(INT_MAX));Strides expected;bool copied=false;size_t j=0;
    for(int dimension:shape) {
      if(j<collapsed.first.size()&&collapsed.first[j]%dimension==0) {
        collapsed.first[j]/=dimension;expected.push_back(collapsed.first[j]*collapsed.second[j]);
        j+=size_t(collapsed.first[j]==1);
      } else if(dimension==1)expected.push_back(expected.back());
      else{copied=true;break;}
    }
    CHECK(plan.copy==copied);if(!copied)for(size_t i=0;i<shape.size();++i)CHECK(plan.strides[i]==expected[i]);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::reshape_eval_layout(input.shape().data(),input.strides().data(),input.ndim(),
        shape.data(),shape.size(),false,cold));
    REQUIRE(cpu::reshape_alias_eval_storage(output,actual));
    CHECK(actual.backing_births==size_t(copied));CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  }
  cpu::CopyEvalStorage actual;
  const int copied_shape[]={2,3,32};
  REQUIRE(cpu::reshape_eval_layout(input.shape().data(),input.strides().data(),4,copied_shape,3,false,actual));
  const auto saved=actual;const int bad_shape[]={2,3,31};const int huge[]={INT_MAX,2};const int small[]={1};
  CHECK_FALSE(cpu::reshape_eval_layout(input.shape().data(),input.strides().data(),4,bad_shape,3,false,actual));
  CHECK_FALSE(cpu::reshape_eval_layout(huge,input.strides().data(),2,small,1,false,actual));
  auto negative=input.strides();negative[1]=-negative[1];
  CHECK_FALSE(cpu::reshape_eval_layout(input.shape().data(),negative.data(),4,copied_shape,3,false,actual));
  CHECK_FALSE(cpu::reshape_eval_layout(input.shape().data(),input.strides().data(),6,copied_shape,3,false,actual));
  CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_reshape_eval_layout(&raw,input.shape().data(),input.strides().data(),4,copied_shape,3,false));
  CHECK(raw.backing_births==1);CHECK(raw.graph_extents==saved.allocation_extents);
  const auto saved_raw=raw;
  CHECK_FALSE(mlx_operation_event_cpu_reshape_eval_layout(&raw,input.shape().data(),input.strides().data(),4,bad_shape,3,false));
  CHECK(std::memcmp(&saved_raw,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU head joining reshape preserves ordinary values and copy or alias custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(bool copied:{false,true}) {
    constexpr int batches=2,heads=4,queries=3,width=8;
    std::array<float,batches*heads*queries*width*2> data;
    for(size_t i=0;i<data.size();++i)data[i]=(int(i%37)-18)*0.0625f;
    array base(data.data(),Shape{batches,heads,queries,width*2},float32);
    auto every_other=slice(base,Shape{0,0,0,0},Shape{batches,heads,queries,width*2},Shape{1,1,1,2},stream);eval(every_other);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> source,escaped;uint64_t source_identity=0;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,4)==0);
      auto value=contiguous(every_other,false,stream);bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      REQUIRE(info.known);source_identity=info.identity;source.emplace(value);
    }
    {
      auto input=transpose(*source,{0,2,1,3},stream);eval(input);
      const Shape shape=copied?Shape{batches,queries,heads*width}:Shape{batches,queries,heads,width,1};
      auto ordinary=reshape(input,shape,stream);eval(ordinary);
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,5)==0);
      auto value=reshape(input,shape,stream);bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
      CHECK(value.flags().row_contiguous==copied);CHECK(value.shape()==shape);bool nonzero=false;
      for(int b=0;b<batches;++b)for(int q=0;q<queries;++q)for(int h=0;h<heads;++h)for(int d=0;d<width;++d) {
        const size_t source_index=(((b*heads+h)*queries+q)*width+d)*2;
        const size_t location=copied?((b*queries+q)*heads+h)*width+d:
            b*value.strides(0)+q*value.strides(1)+h*value.strides(2)+d*value.strides(3);
        CHECK(value.data<float>()[location]==data[source_index]);
        CHECK(value.data<float>()[location]==ordinary.data<float>()[location]);nonzero|=value.data<float>()[location]!=0;
      }
      CHECK(nonzero);mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);REQUIRE(info.known);
      CHECK((info.identity!=source_identity)==copied);escaped.emplace(value);
    }
    source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("ordinary Eval uses the same ready input availability transition"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const auto stream = new_stream(Device::gpu);
  prepare(stream, stream);
  array input({-1.5f, 0.25f, 2.75f});
  auto first = add(input, input, stream);
  async_eval({first});
  REQUIRE(first.event().valid());
  first.event().wait();
  REQUIRE(first.status() == array::Status::evaluated);
  auto second = add(first, input, stream);
  async_eval({second});
  CHECK(first.status() == array::Status::available);
  CHECK_FALSE(first.event().valid());
  second.wait();
  CHECK(second.data<float>()[0] == -4.5f);
  CHECK(second.data<float>()[1] == 0.75f);
  CHECK(second.data<float>()[2] == 8.25f);
}

TEST_CASE("CPU preview reuses exact rank-one Slice and retains original flattened backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={7.f,99.f,-11.f,99.f,23.f,99.f,13.f,99.f,17.f,99.f,-19.f,99.f,31.f,99.f,43.f,99.f};
  array base(data,Shape{1,1,16},float32);
  auto strided=slice(base,Shape{0,0,0},Shape{1,1,16},Shape{1,1,2},stream);eval(strided);
  array flat_source(data,Shape{16},float32);
  auto prefix=slice(flat_source,Shape{2},Shape{5},Shape{1},stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::slice_eval_layout(1,false,false,cold));
  REQUIRE(cpu::slice_eval_storage(prefix,actual));
  CHECK(actual.rank==1);CHECK(actual.backing_births==0);
  CHECK(actual.worker_graph_extents==0);CHECK(actual.allocation_extents==cold.allocation_extents);
  auto stepped=slice(flat_source,Shape{0},Shape{8},Shape{2},stream);
  CHECK(cpu::slice_eval_storage(stepped,actual));
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
  unsigned retired=0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void* p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> source,escaped;uint64_t identity=0;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,3)==0);
    auto value=contiguous(strided,false,stream);bank.reset();
    Operation operation;operation.append(value);
    REQUIRE(eval_traversal_tests::submit(operation,stream,{1,3,2,2,2,1,8})==0);
    eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    REQUIRE(info.known);identity=info.identity;source.emplace(value);
  }
  const auto occupied=mlx_original_buffer_budget_occupied(budget.value);REQUIRE(occupied>0);
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,3)==0);
    auto rectangle=slice(*source,Shape{0,0,1},Shape{1,1,7},Shape{1,1,1},stream);
    auto flat=reshape(rectangle,Shape{6},stream);
    auto value=slice(flat,Shape{0},Shape{3},Shape{1},stream);bank.reset();
    Operation operation;operation.append(value);
    REQUIRE(eval_traversal_tests::submit(operation,stream,{1,5,4,4,4,1,8})==0);
    eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity==identity);CHECK(value.shape()==Shape{3});
    CHECK(value.flags().row_contiguous);CHECK(value.data<float>()[0]==-11.f);
    CHECK(value.data<float>()[1]==23.f);CHECK(value.data<float>()[2]==13.f);
    escaped.emplace(value);
  }
  source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==occupied);
  CHECK(escaped->data<float>()[0]==-11.f);escaped.reset();
  CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

TEST_CASE("CPU flat count and maximum sources validate complete scalar reductions"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const uint32_t counts[]={1,0,1,1,0,1};
  const float values[]={-7.f,0.5f,3.f,-1.f,2.f,9.f};
  for(bool count:{true,false}) {
    array input=count?array(counts,Shape{6},uint32):array(values,Shape{6},float32);
    const auto dtype=count?uint32:float32;
    const auto kind=count?Reduce::Sum:Reduce::Max;
    const auto source=count?cpu::ReductionEvalKind::Uint32FlatSum:cpu::ReductionEvalKind::Float32FlatMaximum;
    auto value=count?sum(input,true,stream):max(input,true,stream);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::reduction_eval_layout(source,1,6,1,false,cold));
    REQUIRE(cpu::reduction_eval_storage(value,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.backing_births==1);CHECK(actual.inputs==1);
    std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
    auto bad_axis=array(Shape{1},dtype,std::make_shared<Reduce>(stream,kind,std::vector<int>{1}),{input});
    auto repeated=array(Shape{1},dtype,std::make_shared<Reduce>(stream,kind,std::vector<int>{0,0}),{input});
    auto bad_shape=array(Shape{2},dtype,std::make_shared<Reduce>(stream,kind,std::vector<int>{0}),{input});
    auto bad_dtype=array(Shape{1},count?float32:uint32,
        std::make_shared<Reduce>(stream,kind,std::vector<int>{0}),{input});
    struct Derived final:Reduce {using Reduce::Reduce;};
    auto subclass=array(Shape{1},dtype,std::make_shared<Derived>(stream,kind,std::vector<int>{0}),{input});
    auto sparse=slice(input,Shape{0},Shape{6},Shape{2},stream);eval(sparse);
    auto sparse_value=count?sum(sparse,true,stream):max(sparse,true,stream);
    for(const auto* malformed:{&bad_axis,&repeated,&bad_shape,&bad_dtype,&subclass,&sparse_value}) {
      CHECK_FALSE(cpu::reduction_eval_storage(*malformed,actual));
      CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
    }
    for(size_t width:{size_t(0),size_t(1),SIZE_MAX})
      CHECK_FALSE(cpu::reduction_eval_layout(source,1,width,1,false,actual));
    CHECK_FALSE(cpu::reduction_eval_layout(source,2,6,1,false,actual));
    CHECK_FALSE(cpu::reduction_eval_layout(source,1,6,2,false,actual));
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_reduction_eval_layout(&raw,count?5:6,1,6,1,false));
    CHECK(raw.graph_extents==cold.allocation_extents);
    CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
  }
}
TEST_CASE("CPU flat count and maximum preserve ordinary SIMD tails and escaped ownership"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<uint32_t,1031> counts;std::array<float,1031> values;
  uint32_t expected_count=0;float expected_max=-1000.f;
  for(size_t i=0;i!=counts.size();++i) {
    counts[i]=uint32_t(i%3!=0);expected_count+=counts[i];
    values[i]=float(int(i%29)-14)*0.25f;expected_max=std::max(expected_max,values[i]);
  }
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  for(bool count:{true,false}) {
    array input=count?array(counts.data(),Shape{1031},uint32):array(values.data(),Shape{1031},float32);
    auto ordinary=count?sum(input,false,stream):max(input,false,stream);eval(ordinary);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,2)==0);
      auto value=count?sum(input,false,stream):max(input,false,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,5,4,4,4,1,16};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,value);
      CHECK(value.ndim()==0);CHECK(value.dtype()==input.dtype());
      if(count) {
        CHECK(value.data<uint32_t>()[0]==ordinary.data<uint32_t>()[0]);
        CHECK(value.data<uint32_t>()[0]==expected_count);CHECK(expected_count>0);
      } else {
        CHECK(value.data<float>()[0]==ordinary.data<float>()[0]);
        CHECK(value.data<float>()[0]==expected_max);CHECK(expected_max>0.f);
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    if(count) {CHECK(escaped->data<uint32_t>()[0]==expected_count);}
    else {CHECK(escaped->data<float>()[0]==expected_max);}
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU scalar overwrite source prices both copy tasks and rejects changed geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float values[]={1,2,3,4,5};const float replacement[]={-INFINITY};
  array source(values,Shape{5},float32),update(replacement,Shape{1},float32);
  auto output=slice_update(source,update,Shape{2},Shape{3},Shape{1},stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::scalar_update_eval_layout(5,false,cold));
  REQUIRE(cpu::copy_eval_storage(output,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[3]==5);CHECK(actual.request_counts[4]==4);
  CHECK(actual.request_counts[5]==4);CHECK(actual.request_counts[6]==2);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::scalar_update_eval_layout(1,false,actual));
  CHECK_FALSE(cpu::scalar_update_eval_layout(size_t(INT_MAX)+1,false,actual));
  auto invalid=[&](SliceUpdate::ReduceType kind,Shape starts,Shape ends,Shape strides) {
    return array(Shape{5},float32,std::make_shared<SliceUpdate>(stream,kind,starts,ends,strides),{source,update});
  };
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::Sum,{2},{3},{1}),actual));
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::None,{5},{6},{1}),actual));
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::None,{2},{4},{1}),actual));
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::None,{2},{3},{2}),actual));
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::None,{},{},{1}),actual));
  struct Derived final:SliceUpdate {using SliceUpdate::SliceUpdate;};
  auto derived=array(Shape{5},float32,std::make_shared<Derived>(stream,SliceUpdate::None,Shape{2},Shape{3},Shape{1}),{source,update});
  CHECK_FALSE(cpu::copy_eval_storage(derived,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_scalar_update_eval_layout(&raw,5,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_scalar_update_eval_layout(&raw,0,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU scalar overwrite preserves ordinary values input aliases and escaped original backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(int width:{2,19,1025})for(int index:{0,width-1}) {
    std::vector<float> values(width);for(int i=0;i<width;++i)values[i]=float((i%19)-9)*.25f;
    const float replacement=-INFINITY;array source(values.data(),Shape{width},float32);
    array update(&replacement,Shape{1},float32);array retained=source;
    auto ordinary=slice_update(source,update,Shape{index},Shape{index+1},Shape{1},stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
      auto output=slice_update(source,update,Shape{index},Shape{index+1},Shape{1},stream);
      bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,3,2,3,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==Shape{width});CHECK(output.dtype()==float32);
      for(int i=0;i<width;++i) {
        CHECK(output.data<float>()[i]==ordinary.data<float>()[i]);
        CHECK(output.data<float>()[i]==(i==index?replacement:values[i]));
        CHECK(retained.data<float>()[i]==values[i]);
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);CHECK(info.charged_bytes>=size_t(width)*sizeof(float));escaped.emplace(output);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}

#include "mlx/backend/cpu/argsort_f32.h"
TEST_CASE("CPU candidate ArgSort source authenticates F32 rows and one U32 output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float values[]={-2.0f,99,3.0f,99,3.0f,99,1.0f,99};
  array base(values,Shape{8},float32);auto input=slice(base,Shape{0},Shape{8},Shape{2},stream);eval(input);
  auto output=argsort(input,stream);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::argsort_eval_layout(float32,1,4,1,false,cold));REQUIRE(cpu::argsort_eval_storage(output,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[4]==2);CHECK(actual.request_counts[6]==1);CHECK(actual.worker_graph_extents==0);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::argsort_eval_layout(float32,1,0,1,false,actual));CHECK_FALSE(cpu::argsort_eval_layout(float32,1,size_t(INT_MAX)+1,1,false,actual));
  auto wrong_axis=array(Shape{4},uint32,std::make_shared<ArgSort>(stream,1),{input});
  auto wrong_dtype=array(Shape{4},int32,std::make_shared<ArgSort>(stream,0),{input});
  auto wrong_shape=array(Shape{3},uint32,std::make_shared<ArgSort>(stream,0),{input});
  struct Derived final:ArgSort {using ArgSort::ArgSort;};
  auto derived=array(Shape{4},uint32,std::make_shared<Derived>(stream,0),{input});
  CHECK_FALSE(cpu::argsort_eval_storage(wrong_axis,actual));CHECK_FALSE(cpu::argsort_eval_storage(wrong_dtype,actual));
  CHECK_FALSE(cpu::argsort_eval_storage(wrong_shape,actual));CHECK_FALSE(cpu::argsort_eval_storage(derived,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_argsort_eval_layout(&raw,MLX_FLOAT32,1,4,1,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_argsort_eval_layout(&raw,MLX_FLOAT32,1,0,1,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU candidate ArgSort preserves stable ties NaNs and escaped original ordering"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(int width:{1,19,4097}) {
    std::vector<float> values(width);for(int i=0;i<width;++i)values[i]=float((i%13)-6)*.25f;
    if(width>2){values[0]=-0.0f;values[1]=0.0f;values[2]=NAN;values[width-1]=NAN;}
    array input(values.data(),Shape{width},float32);auto ordinary=argsort(input,stream);eval(ordinary);
    std::vector<uint32_t> expected(width);std::iota(expected.begin(),expected.end(),0);
    std::stable_sort(expected.begin(),expected.end(),[&](uint32_t a,uint32_t b){
      if(std::isnan(values[a]))return false;if(std::isnan(values[b]))return true;return values[a]<values[b];
    });
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
      auto output=argsort(input,stream);bank.reset();Operation operation;operation.append(output);
      // Source, ArgSort result and the selected completion Synchronizer.
      // Record::bounded_storage_layout requires its initial eight capture slots.
      const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==Shape{width});CHECK(output.dtype()==uint32);
      for(int i=0;i<width;++i){CHECK(output.data<uint32_t>()[i]==ordinary.data<uint32_t>()[i]);CHECK(output.data<uint32_t>()[i]==expected[i]);}
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);CHECK(info.charged_bytes>=size_t(width)*sizeof(uint32_t));escaped.emplace(output);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU static rectangle update source authenticates rank coordinates and one output birth"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::vector<float> values(42,2.5f),replacement(12,-3.0f);
  array source(values.data(),Shape{2,3,7},float32),update(replacement.data(),Shape{2,2,3},float32);
  auto output=slice_update(source,update,Shape{0,1,2},Shape{2,3,5},Shape{1,1,1},stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::static_update_eval_layout(3,42,12,false,cold));
  REQUIRE(cpu::copy_eval_storage(output,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[3]==5);CHECK(actual.request_counts[4]==4);
  CHECK(actual.request_counts[5]==4);CHECK(actual.request_counts[6]==2);
  CpuCopyDispatchStorage copy;REQUIRE(static_update_copy_dispatch_storage(3,copy));
  CHECK(cold.worker_graph_extents==copy.worker_graph_extent_bound);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  for(size_t rank:{size_t{0},size_t{5}})CHECK_FALSE(cpu::static_update_eval_layout(rank,42,12,false,actual));
  CHECK_FALSE(cpu::static_update_eval_layout(3,42,0,false,actual));
  CHECK_FALSE(cpu::static_update_eval_layout(3,42,42,false,actual));
  CHECK_FALSE(cpu::static_update_eval_layout(3,size_t(INT_MAX)+1,12,false,actual));
  auto invalid=[&](SliceUpdate::ReduceType kind,Shape starts,Shape ends,Shape strides) {
    return array(Shape{2,3,7},float32,std::make_shared<SliceUpdate>(stream,kind,starts,ends,strides),{source,update});
  };
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::Sum,{0,1,2},{2,3,5},{1,1,1}),actual));
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::None,{0,1,2},{2,3,6},{1,1,1}),actual));
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::None,{0,1,2},{2,3,5},{1,1,2}),actual));
  CHECK_FALSE(cpu::copy_eval_storage(invalid(SliceUpdate::None,{0,1},{2,3,5},{1,1,1}),actual));
  struct Derived final:SliceUpdate {using SliceUpdate::SliceUpdate;};
  auto derived=array(Shape{2,3,7},float32,std::make_shared<Derived>(stream,SliceUpdate::None,Shape{0,1,2},Shape{2,3,5},Shape{1,1,1}),{source,update});
  CHECK_FALSE(cpu::copy_eval_storage(derived,actual));
  auto half_source=astype(source,float16,stream),half_update=astype(update,float16,stream);
  eval(half_source,half_update);
  auto half=slice_update(half_source,half_update,Shape{0,1,2},Shape{2,3,5},Shape{1,1,1},stream);
  CHECK_FALSE(cpu::copy_eval_storage(half,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_static_update_eval_layout(&raw,3,42,12,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
  CHECK(raw.backing_births==1);auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_static_update_eval_layout(&raw,3,42,42,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
}
TEST_CASE("CPU static rectangle update preserves batched ordinary values exact bits and escaped ownership"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(Shape shape:{Shape{19},Shape{3,7},Shape{2,3,7},Shape{2,2,3,7}}) {
    Shape starts(shape.size(),0),ends=shape,strides(shape.size(),1),update_shape=shape;
    starts.back()=1;ends.back()-=1;update_shape.back()-=2;
    if(shape.size()>1){starts[shape.size()-2]=1;update_shape[shape.size()-2]-=1;}
    const size_t count=std::accumulate(shape.begin(),shape.end(),size_t{1},std::multiplies<size_t>());
    const size_t update_count=std::accumulate(update_shape.begin(),update_shape.end(),size_t{1},std::multiplies<size_t>());
    std::vector<float> values(count),replacements(update_count);
    for(size_t i=0;i<count;++i)values[i]=float(int(i%19)-9)*.25f;
    for(size_t i=0;i<update_count;++i)replacements[i]=float(i)*-.125f;
    replacements.front()=-0.0f;if(update_count>1)replacements[1]=-INFINITY;
    if(update_count>2)replacements[2]=NAN;
    auto expected=values;
    for(size_t i=0;i<count;++i) {
      size_t offset=i,update_index=0,multiplier=1;bool selected=true;
      for(size_t axis=shape.size();axis-->0;) {
        const int coordinate=offset%shape[axis];offset/=shape[axis];
        if(coordinate<starts[axis]||coordinate>=ends[axis])selected=false;
        if(coordinate>=starts[axis])update_index+=size_t(coordinate-starts[axis])*multiplier;
        multiplier*=update_shape[axis];
      }
      if(selected)expected[i]=replacements[update_index];
    }
    array source(values.data(),shape,float32),update(replacements.data(),update_shape,float32),retained=source;
    auto ordinary=slice_update(source,update,starts,ends,strides,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
      auto output=slice_update(source,update,starts,ends,strides,stream);
      bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,3,2,3,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==shape);CHECK(output.dtype()==float32);
      CHECK(std::memcmp(output.data<float>(),ordinary.data<float>(),count*sizeof(float))==0);
      CHECK(std::memcmp(output.data<float>(),expected.data(),count*sizeof(float))==0);
      CHECK(std::memcmp(retained.data<float>(),values.data(),count*sizeof(float))==0);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);CHECK(info.charged_bytes>=count*sizeof(float));escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<float>(),expected.data(),count*sizeof(float))==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU F16 Gather preserves strided exact bits repeated picks and escaped source"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  // Signed zero and a NaN payload must survive direct half copies.
  const std::array<uint16_t,24> bits{0x8000,0,0x3c00,0,0x7e35,0,
      0x4000,0,0xbd00,0,0x4980,0,0x3c00,0,0x4400,0,0xc000,0,
      0x3400,0,0xc400,0,0x4880,0};
  std::array<mlx::core::float16_t,24> data;
  static_assert(sizeof(data)==sizeof(bits));std::memcpy(data.data(),bits.data(),sizeof(bits));
  const int32_t picks[]={-1,99,0,99,0,99,2,99};
  array full(data.data(),Shape{4,6},float16),all_indices(picks,Shape{2,4},int32);
  auto source=slice(full,Shape{0,0},Shape{4,6},Shape{1,2},stream);eval(source);
  auto index=slice(all_indices,Shape{0,0},Shape{2,4},Shape{1,2},stream);eval(index);
  REQUIRE_FALSE(source.flags().row_contiguous);REQUIRE_FALSE(index.flags().row_contiguous);
  auto lazy=gather(source,index,0,Shape{1,3},stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::gather_eval_layout(float16,int32,2,2,12,4,3,false,cold));
  REQUIRE(cpu::gather_eval_storage(lazy,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);CHECK(actual.backing_births==1);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_gather_eval_layout(&raw,
      MLX_FLOAT16,MLX_INT32,2,2,12,4,3,false));CHECK(raw.graph_extents==cold.allocation_extents);
  auto ordinary=take(source,index,0,stream);
  // take is the shared Gather followed by Squeeze; qualify both actual
  // descriptors before entering the original role, including the half alias.
  // The evaluator resolves Gather before inspecting Squeeze's backing span.
  // Preserve the lazy Squeeze descriptor while completing that exact input.
  auto gathered=ordinary.inputs()[0];eval(gathered);
  cpu::CopyEvalStorage squeeze,expected_squeeze;
  REQUIRE(cpu::greedy_eval_storage(ordinary,squeeze));
  REQUIRE(cpu::squeeze_eval_layout(4,false,expected_squeeze));
  CHECK(squeeze.allocation_extents==expected_squeeze.allocation_extents);
  CHECK(squeeze.backing_births==0);
  eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,4)==0);
    auto value=take(source,index,0,stream);
    bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,6,4,5,4,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    REQUIRE(value.dtype()==float16);REQUIRE(value.shape()==Shape{2,2,3});
    CHECK(std::memcmp(value.data<mlx::core::float16_t>(),ordinary.data<mlx::core::float16_t>(),24)==0);
    const size_t selected[]={3,0,0,2};
    for(size_t row=0;row!=4;++row)for(size_t col=0;col!=3;++col) {
      uint16_t observed=0;std::memcpy(&observed,value.data<mlx::core::float16_t>()+row*3+col,2);
      CHECK(observed==bits[selected[row]*6+col*2]);
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(std::memcmp(escaped->data<mlx::core::float16_t>(),ordinary.data<mlx::core::float16_t>(),24)==0);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU precise half Softmax source binds dtype precision and exact row geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-2.5f,0.25f,3.75f,-0.125f,1.5f,-3.0f,0.75f,2.0f};
  array base(data,Shape{2,4},float32);
  for(Dtype dtype:{float16,bfloat16}) {
    auto source=astype(base,dtype,stream);eval(source);
    auto value=softmax(source,-1,true,stream);cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::softmax_typed_eval_layout(dtype,true,2,4,2,false,cold));
    REQUIRE(cpu::softmax_eval_storage(value,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
    CHECK(actual.backing_births==1);CHECK(actual.request_counts[9]==1);
    std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
    CHECK_FALSE(cpu::softmax_typed_eval_layout(dtype,false,2,4,2,false,actual));
    auto imprecise=softmax(source,-1,false,stream);
    CHECK_FALSE(cpu::softmax_eval_storage(imprecise,actual));
    auto wrong=array(Shape{2,4},float32,std::make_shared<Softmax>(stream,true),{source});
    CHECK_FALSE(cpu::softmax_eval_storage(wrong,actual));
    CHECK_FALSE(cpu::softmax_typed_eval_layout(dtype,true,2,0,2,false,actual));
    CHECK_FALSE(cpu::softmax_typed_eval_layout(dtype,true,2,4,SIZE_MAX,false,actual));
    CHECK(std::memcmp(&actual,saved.data(),sizeof(actual))==0);
    mlx_cpu_copy_eval_layout raw{};
    const auto scalar=dtype==float16?MLX_FLOAT16:MLX_BFLOAT16;
    REQUIRE(mlx_operation_event_cpu_typed_softmax_eval_layout(&raw,scalar,true,2,4,2,false));
    CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);
    const auto raw_saved=raw;
    CHECK_FALSE(mlx_operation_event_cpu_typed_softmax_eval_layout(&raw,scalar,false,2,4,2,false));
    CHECK_FALSE(mlx_operation_event_cpu_typed_softmax_eval_layout(&raw,MLX_FLOAT64,true,2,4,2,false));
    CHECK(std::memcmp(&raw,&raw_saved,sizeof(raw))==0);
  }
}

TEST_CASE("CPU precise half Softmax preserves scalar strided tails and escaped original output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(Dtype dtype:{float16,bfloat16})for(int columns:{1,19}) {
    const bool strided=columns>1;const int step=strided?2:1;
    std::vector<float> data(size_t(2*columns*step),99.0f);
    for(int row=0;row!=2;++row)for(int col=0;col!=columns;++col)
      data[size_t((row*columns+col)*step)]=float((col*7+row*3)%23-11)*0.25f;
    array full(data.data(),Shape{2,columns*step},float32);
    auto typed=astype(full,dtype,stream);eval(typed);
    auto source=strided?slice(typed,Shape{0,0},Shape{2,columns*step},Shape{1,step},stream):typed;
    eval(source);CHECK(source.flags().row_contiguous==!strided);
    auto ordinary=softmax(source,-1,true,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,2)==0);
      auto value=softmax(source,-1,true,stream);bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,12};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,value);
      REQUIRE(value.dtype()==dtype);REQUIRE(value.shape()==Shape{2,columns});
      const auto bytes=size_t(2*columns)*2;
      const void* actual_data=dtype==float16?static_cast<const void*>(value.data<mlx::core::float16_t>()):value.data<mlx::core::bfloat16_t>();
      const void* expected_data=dtype==float16?static_cast<const void*>(ordinary.data<mlx::core::float16_t>()):ordinary.data<mlx::core::bfloat16_t>();
      CHECK(std::memcmp(actual_data,expected_data,bytes)==0);
      for(int row=0;row!=2;++row) {
        double denominator=0.0;
        for(int col=0;col!=columns;++col)denominator+=std::exp(double(data[size_t((row*columns+col)*step)]));
        double observed_sum=0.0;
        for(int col=0;col!=columns;++col) {
          const size_t offset=size_t(row*columns+col);
          const float observed=dtype==float16?float(value.data<mlx::core::float16_t>()[offset]):float(value.data<mlx::core::bfloat16_t>()[offset]);
          const double expected=std::exp(double(data[offset*step]))/denominator;
          CHECK(std::isfinite(observed));CHECK(observed>0.0f);
          CHECK(std::abs(double(observed)-expected)<(dtype==float16?0.0006:0.003));observed_sum+=observed;
        }
        CHECK(std::abs(observed_sum-1.0)<(dtype==float16?0.002:0.01));
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);CHECK(info.charged_bytes>=bytes);escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    const void* actual_data=dtype==float16?static_cast<const void*>(escaped->data<mlx::core::float16_t>()):escaped->data<mlx::core::bfloat16_t>();
    const void* expected_data=dtype==float16?static_cast<const void*>(ordinary.data<mlx::core::float16_t>()):ordinary.data<mlx::core::bfloat16_t>();
    CHECK(std::memcmp(actual_data,expected_data,size_t(2*columns)*2)==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU BF16 rank five Matmul binds exact copies and preserves row results and custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  constexpr int batches=2,heads=2,groups=3,m=3,n=5,k=7;
  std::vector<mlx::core::bfloat16_t> av(batches*m*heads*groups*k),bv(batches*k*heads*groups*n);
  for(size_t i=0;i<av.size();++i)av[i]=(int(i%17)-8)*0.125f;
  for(size_t i=0;i<bv.size();++i)bv[i]=(int(i%19)-9)*0.0625f;
  array a_source(av.data(),Shape{batches,m,heads,groups,k},bfloat16),
      b_source(bv.data(),Shape{batches,k,heads,groups,n},bfloat16);
  auto a=transpose(a_source,{0,2,3,1,4},selected),b=transpose(b_source,{0,2,3,1,4},selected);eval(a,b);
  auto ac=contiguous(a,false,selected),bc=contiguous(b,false,selected);eval(ac,bc);
  cpu::CopyEvalStorage base;
  REQUIRE(cpu::bf16_matmul_eval_layout(5,m,n,k,batches*heads*groups,false,base));
  for(size_t copies=0;copies<=2;++copies) {
    auto output=matmul(copies?a:ac,copies==2?b:bc,selected);cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::bf16_matmul_copy_eval_layout(5,m,n,k,batches*heads*groups,copies,false,cold));
    REQUIRE(cpu::bf16_matmul_eval_storage(output,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.backing_births==1+copies);CHECK(actual.request_counts[6]==1+copies);
    CHECK(actual.request_counts[9]==1);
    CHECK(actual.request_counts[2]==base.request_counts[2]+size_t(copies!=0));
  }
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::bf16_matmul_copy_eval_layout(5,m,n,k,batches*heads*groups,2,false,cold));
  actual=cold;std::array<unsigned char,sizeof(actual)> prior;std::memcpy(prior.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::bf16_matmul_copy_eval_layout(5,m,n,k,batches*heads*groups,3,false,actual));
  CHECK_FALSE(cpu::bf16_matmul_copy_eval_layout(6,m,n,k,batches*heads*groups,2,false,actual));
  CHECK_FALSE(cpu::bf16_matmul_copy_eval_layout(5,m,n,k,SIZE_MAX,2,false,actual));
  CHECK(std::memcmp(prior.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_bf16_matmul_copy_eval_layout(&raw,5,m,n,k,batches*heads*groups,2,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==3);
  auto raw_saved=raw;
  CHECK_FALSE(mlx_operation_event_cpu_bf16_matmul_copy_eval_layout(&raw,5,m,n,k,batches*heads*groups,3,false));
  CHECK(std::memcmp(&raw,&raw_saved,sizeof(raw))==0);
  auto ordinary=matmul(a,b,selected);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,2)==0);
    auto value=matmul(a,b,selected);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,8,4,8,4,1,32};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    REQUIRE(value.shape()==Shape{batches,heads,groups,m,n});REQUIRE(value.dtype()==bfloat16);
    CHECK(std::memcmp(value.data<mlx::core::bfloat16_t>(),ordinary.data<mlx::core::bfloat16_t>(),value.size()*2)==0);
    bool nonzero=false;
    for(int batch=0;batch<batches;++batch)for(int head=0;head<heads;++head)for(int group=0;group<groups;++group)
      for(int row=0;row<m;++row)for(int col=0;col<n;++col) {
        double expected=0;
        for(int inner=0;inner<k;++inner)
          expected+=double(float(av[(((batch*m+row)*heads+head)*groups+group)*k+inner]))*
              float(bv[(((batch*k+inner)*heads+head)*groups+group)*n+col]);
        const size_t index=(((batch*heads+head)*groups+group)*m+row)*n+col;
        const float observed=float(value.data<mlx::core::bfloat16_t>()[index]);
        CHECK(observed==float(mlx::core::bfloat16_t(float(expected))));nonzero|=observed!=0;
      }
    CHECK(nonzero);mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(std::memcmp(escaped->data<mlx::core::bfloat16_t>(),ordinary.data<mlx::core::bfloat16_t>(),ordinary.size()*2)==0);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU selected F16 tiles bind rank five copies default facts and escaped exact results"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32AndFloat16Tiles);
  mlx_cpu_matmul_facts facts{};REQUIRE(mlx_cpu_matmul_facts_for(&facts)==0);
  CHECK(facts.float16_tiles);CHECK(facts.platform_float16_tiles==cpu::platform_float16_uses_tiles());
  mlx_stream_copy_value captured{};REQUIRE(mlx_stream_copy_snapshot(&captured,{&stream})==0);
  auto chosen=captured;REQUIRE(mlx_stream_copy_select_cpu_matmul(&chosen,2)==0);
  CHECK(captured.cpu_matmul==0);CHECK(chosen.cpu_matmul==2);
  struct StreamOwner{mlx_stream value{};~StreamOwner(){mlx_stream_copy_free(value);}} copy;
  REQUIRE(mlx_stream_copy_new(&copy.value,chosen)==0);
  mlx_stream_copy_value recaptured{};REQUIRE(mlx_stream_copy_snapshot(&recaptured,copy.value)==0);
  CHECK(recaptured.cpu_matmul==2);
  constexpr int batches=2,heads=2,groups=3,m=19,n=17,k=23;
  std::vector<mlx::core::float16_t> av(batches*m*heads*groups*k),bv(batches*k*heads*groups*n);
  for(size_t i=0;i<av.size();++i)av[i]=(int(i%17)-8)*0.125f;
  for(size_t i=0;i<bv.size();++i)bv[i]=(int(i%19)-9)*0.0625f;
  array a_source(av.data(),Shape{batches,m,heads,groups,k},float16),
      b_source(bv.data(),Shape{batches,k,heads,groups,n},float16);
  auto a=transpose(a_source,{0,2,3,1,4},selected),b=transpose(b_source,{0,2,3,1,4},selected);eval(a,b);
  auto ac=contiguous(a,false,selected),bc=contiguous(b,false,selected);eval(ac,bc);
  cpu::CopyEvalStorage base;
  REQUIRE(cpu::float16_matmul_copy_eval_layout(5,m,n,k,batches*heads*groups,0,false,base));
  for(size_t copies=0;copies<=2;++copies) {
    auto output=matmul(copies?a:ac,copies==2?b:bc,selected);cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::float16_matmul_copy_eval_layout(5,m,n,k,batches*heads*groups,copies,false,cold));
    REQUIRE(cpu::tiled_matmul_eval_storage(output,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.backing_births==1+copies);CHECK(actual.request_counts[6]==1+copies);
    CHECK(actual.request_counts[9]==1);
    CHECK(actual.request_counts[2]==base.request_counts[2]+size_t(copies!=0));
  }
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::float16_matmul_copy_eval_layout(5,m,n,k,batches*heads*groups,2,false,cold));
  actual=cold;std::array<unsigned char,sizeof(actual)> prior;std::memcpy(prior.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::float16_matmul_copy_eval_layout(5,m,n,k,batches*heads*groups,3,false,actual));
  CHECK_FALSE(cpu::float16_matmul_copy_eval_layout(6,m,n,k,batches*heads*groups,2,false,actual));
  CHECK_FALSE(cpu::float16_matmul_copy_eval_layout(5,m,n,k,SIZE_MAX,2,false,actual));
  CHECK(std::memcmp(prior.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_f16_matmul_copy_eval_layout(&raw,5,m,n,k,batches*heads*groups,2,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==3);
  auto raw_saved=raw;
  CHECK_FALSE(mlx_operation_event_cpu_f16_matmul_copy_eval_layout(&raw,5,m,n,k,batches*heads*groups,3,false));
  CHECK(std::memcmp(&raw,&raw_saved,sizeof(raw))==0);
  auto platform=matmul(a,b,stream);cpu::CopyEvalStorage platform_source;
  CHECK(cpu::tiled_matmul_eval_storage(platform,platform_source)==facts.platform_float16_tiles);
  const Stream f32_only(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  auto inherited=matmul(a,b,f32_only);
  CHECK(cpu::tiled_matmul_eval_storage(inherited,platform_source)==facts.platform_float16_tiles);
  auto ordinary=matmul(a,b,selected);eval(ordinary);
  if(facts.platform_float16_tiles) {
    eval(platform);
    CHECK(std::memcmp(platform.data<mlx::core::float16_t>(),ordinary.data<mlx::core::float16_t>(),ordinary.size()*2)==0);
  }
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,2)==0);
    auto value=matmul(a,b,selected);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,8,4,8,4,1,32};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    REQUIRE(value.shape()==Shape{batches,heads,groups,m,n});REQUIRE(value.dtype()==float16);
    CHECK(std::memcmp(value.data<mlx::core::float16_t>(),ordinary.data<mlx::core::float16_t>(),value.size()*2)==0);
    bool nonzero=false;
    for(int batch=0;batch<batches;++batch)for(int head=0;head<heads;++head)for(int group=0;group<groups;++group)
      for(int row=0;row<m;++row)for(int col=0;col<n;++col) {
        double expected=0;
        for(int inner=0;inner<k;++inner)
          expected+=double(float(av[(((batch*m+row)*heads+head)*groups+group)*k+inner]))*
              float(bv[(((batch*k+inner)*heads+head)*groups+group)*n+col]);
        const size_t index=(((batch*heads+head)*groups+group)*m+row)*n+col;
        const float observed=float(value.data<mlx::core::float16_t>()[index]);
        CHECK(observed==float(mlx::core::float16_t(float(expected))));nonzero|=observed!=0;
      }
    CHECK(nonzero);mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(std::memcmp(escaped->data<mlx::core::float16_t>(),ordinary.data<mlx::core::float16_t>(),ordinary.size()*2)==0);
  escaped.reset();CHECK(retired==1);
}


TEST_CASE("CPU half SDPA shares precise masked and unmasked GQA sources and escaped custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32AndFloat16Tiles);
  for(Dtype dtype:{float16,bfloat16}) for(bool masked:{false,true}) {
    constexpr int heads=4;
    constexpr int queries=3,keys=5,width=8,kv_heads=2;
    std::vector<float> qv(size_t(heads*queries*width)),kv(kv_heads*keys*width),vv(kv.size());
    std::array<bool,queries*keys> mask_values;
    for(size_t i=0;i<qv.size();++i)qv[i]=(int(i%11)-5)*0.125f;
    for(size_t i=0;i<kv.size();++i){kv[i]=(int(i%17)-8)*0.0625f;vv[i]=(int(i%13)-6)*0.1875f;}
    for(int query=0;query<queries;++query)for(int key=0;key<keys;++key)
      mask_values[query*keys+key]=key<=query+1&&key>=query;
    array qsource(qv.data(),Shape{1,heads,queries,width},float32),
        ksource(kv.data(),Shape{1,kv_heads,keys,width},float32),vsource(vv.data(),Shape{1,keys,kv_heads,width},float32),
        mask(mask_values.data(),Shape{queries,keys},bool_);
    auto q=astype(qsource,dtype,selected),k=astype(ksource,dtype,selected);
    auto v=transpose(astype(vsource,dtype,selected),{0,2,1,3},selected);eval(q,k,v);
    REQUIRE_FALSE(v.flags().row_contiguous);REQUIRE(v.strides().back()==1);
    const auto mask_arg=masked?std::optional<array>(mask):std::nullopt;
    const char* mode=masked?"array":"";
    cpu::CopyEvalStorage selection;
    REQUIRE(cpu::typed_select_broadcast_eval_layout(dtype,5,heads*queries*keys,false,selection));
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_typed_select_broadcast_eval_layout(&raw,
        dtype==float16?MLX_FLOAT16:MLX_BFLOAT16,5,heads*queries*keys,false));
    CHECK(raw.graph_extents==selection.allocation_extents);CHECK(raw.backing_births==selection.backing_births);
    auto prior=raw;
    CHECK_FALSE(mlx_operation_event_cpu_typed_select_broadcast_eval_layout(&raw,MLX_INT32,5,heads*queries*keys,false));
    CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
    const float scale=1.0f/std::sqrt(float(width));
    auto ordinary=fast::scaled_dot_product_attention(q,k,v,scale,mode,mask_arg,{},selected);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<23,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,40,2,5)==0);
      auto output=fast::scaled_dot_product_attention(q,k,v,scale,mode,mask_arg,{},selected);
      bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,80,41,100,41,1,200};
      REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==q.shape());CHECK(output.dtype()==dtype);bool nonzero=false;
      CHECK(std::memcmp(output.data<void>(),ordinary.data<void>(),output.size()*2)==0);
      const auto observed=[&](size_t i){return dtype==float16?float(output.data<mlx::core::float16_t>()[i]):
          float(output.data<mlx::core::bfloat16_t>()[i]);};
      for(size_t i=0;i<qv.size();++i)nonzero|=observed(i)!=0;
      CHECK(nonzero);
      for(int head=0;head<heads;++head)for(int query=0;query<queries;++query) {
        double scores[keys],denominator=0;const int shared=head/(heads/kv_heads);
        for(int key=0;key<keys;++key) {
          double dot=0;for(int d=0;d<width;++d)
            dot+=double(qv[(head*queries+query)*width+d])*kv[(shared*keys+key)*width+d];
          scores[key]=(!masked||mask_values[query*keys+key])?std::exp(dot*scale):0;denominator+=scores[key];
        }
        for(int d=0;d<width;++d) {
          double expected=0;for(int key=0;key<keys;++key)
            expected+=scores[key]/denominator*vv[(key*kv_heads+shared)*width+d];
          CHECK(std::abs(double(observed((head*queries+query)*width+d))-expected)<(dtype==float16?0.005:0.05));
        }
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<void>(),ordinary.data<void>(),ordinary.size()*2)==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU byte View binds exact alias and one-copy reinterpretation with escaped source"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float values[]={1.0f,-2.0f,0.5f,3.25f,-4.0f,6.0f};
  array floats(values,Shape{2,3},float32);
  auto bytes=view(floats,uint8,stream);cpu::CopyEvalStorage alias,actual;
  REQUIRE(cpu::byte_view_eval_layout(float32,uint8,2,sizeof(values),false,false,alias));
  REQUIRE(cpu::copy_eval_storage(bytes,actual));CHECK(actual.backing_births==0);
  CHECK(actual.allocation_extents==alias.allocation_extents);
  std::array<uint8_t,sizeof(values)> interleaved{};
  const auto* raw=reinterpret_cast<const uint8_t*>(values);
  for(size_t row=0;row<2;++row)for(size_t col=0;col<12;++col)interleaved[col*2+row]=raw[row*12+col];
  array stored(interleaved.data(),Shape{12,2},uint8);
  auto input=transpose(stored,{1,0},stream);eval(input);REQUIRE_FALSE(input.flags().row_contiguous);
  auto ordinary=view(input,float32,stream);cpu::CopyEvalStorage copy;
  REQUIRE(cpu::byte_view_eval_layout(uint8,float32,2,sizeof(values),true,false,copy));
  REQUIRE(cpu::copy_eval_storage(ordinary,actual));CHECK(actual.backing_births==1);
  CHECK(actual.allocation_extents==copy.allocation_extents);
  auto old=copy;CHECK_FALSE(cpu::byte_view_eval_layout(uint8,float32,2,sizeof(values)-1,true,false,copy));
  CHECK_FALSE(cpu::byte_view_eval_layout(uint32,float32,2,sizeof(values),true,false,copy));
  CHECK_FALSE(cpu::byte_view_eval_layout(uint8,float32,6,sizeof(values),true,false,copy));
  CHECK(std::memcmp(&old,&copy,sizeof(copy))==0);
  eval(ordinary);CHECK(std::memcmp(ordinary.data<float>(),values,sizeof(values))==0);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto output=view(input,float32,stream);bank.reset();Operation operation;operation.append(output);
    const mlx_operation_eval_traversal_limits limits{1,8,4,8,4,1,32};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,output);
    CHECK(output.shape()==Shape{2,3});CHECK(std::memcmp(output.data<float>(),values,sizeof(values))==0);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);CHECK(info.known);
    escaped.emplace(output);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(std::memcmp(escaped->data<float>(),ordinary.data<float>(),sizeof(values))==0);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU byte frame uses exact reshape concatenate slice and alignment sources with escaped payload"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float values[]={1.f,-2.f,.5f,3.25f,-4.f,6.f};
  const uint8_t header_bytes[]={17,29,255,3};
  array source(values,Shape{1,2,3},float32);
  // Actual strided U8 reshape exercises the quoted possible-copy branch.
  const uint8_t interleaved[]={1,4,2,5,3,6};
  array stored(interleaved,Shape{3,2},uint8);
  auto strided=transpose(stored,{1,0},stream);eval(strided);
  auto flat=reshape(strided,{-1},stream);
  cpu::CopyEvalStorage copy,actual;
  REQUIRE(cpu::reshape_copy_eval_layout(2,1,false,copy));
  REQUIRE(cpu::reshape_alias_eval_storage(flat,actual));
  CHECK(actual.backing_births==1);CHECK(actual.allocation_extents==copy.allocation_extents);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_reshape_copy_eval_layout(&raw,2,1,false));
  CHECK(raw.graph_extents==copy.allocation_extents);CHECK(raw.backing_births==1);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_reshape_copy_eval_layout(&raw,6,1,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
  eval(flat);const uint8_t flat_expected[]={1,2,3,4,5,6};
  CHECK(std::memcmp(flat.data<uint8_t>(),flat_expected,sizeof(flat_expected))==0);
  for(int header_len:{3,4}) {
    array header(header_bytes,Shape{header_len},uint8);
    auto run=[&]() {
      auto bytes=reshape(view(source,uint8,stream),{-1},stream);
      auto frame=concatenate({header,bytes},0,stream);
      auto returned_header=slice(frame,{0},{header_len},{1},stream);
      auto payload=slice(frame,{header_len},{int(frame.size())},{1},stream);
      if(header_len%int(source.itemsize())!=0)payload=add(payload,array(uint8_t{0}),stream);
      auto result=reshape(view(payload,float32,stream),source.shape(),stream);
      return std::pair<array,array>{returned_header,result};
    };
    auto ordinary=run();eval(ordinary.first,ordinary.second);
    CHECK(std::memcmp(ordinary.first.data<uint8_t>(),header_bytes,header_len)==0);
    CHECK(std::memcmp(ordinary.second.data<float>(),values,sizeof(values))==0);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      // View, Reshape, Concatenate, two Slice, optional Broadcast/Add,
      // typed View and final Reshape; exactly one eager zero seed if needed.
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,
          header_len==3?9:7,header_len==3?1:0,3)==0);
      auto output=run();bank.reset();Operation operation;operation.append(output.first);operation.append(output.second);
      // Nine byte operations plus Synchronizer (or seven when aligned),
      // three/two leaves, and one output per actual primitive.
      const mlx_operation_eval_traversal_limits limits=header_len==3
          ? mlx_operation_eval_traversal_limits{2,13,10,13,10,1,64}
          : mlx_operation_eval_traversal_limits{2,10,8,10,8,1,64};
      mlx_operation_eval_traversal_layout traversal{};
      REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal,&limits));
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output.second);
      CHECK(output.second.shape()==source.shape());CHECK(output.second.dtype()==float32);
      CHECK(std::memcmp(output.first.data<uint8_t>(),header_bytes,header_len)==0);
      CHECK(std::memcmp(output.second.data<float>(),ordinary.second.data<float>(),sizeof(values))==0);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output.second},budget.value)==0);
      CHECK(info.known);escaped.emplace(output.second);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<float>(),values,sizeof(values))==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU F16 scalar Full preserves exact source bits and escaped destination"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  // Full receives an actual stored half, including values not preserved by an
  // invented F32 fill or arithmetic conversion (signed zero and NaN payload).
  for(uint16_t bits:{uint16_t{0x8000},uint16_t{0x3555},uint16_t{0x7e35}}) {
    mlx::core::float16_t scalar;std::memcpy(&scalar,&bits,sizeof(bits));
    array source(&scalar,Shape{},float16);
    auto ordinary=full(Shape{2,3,19},source,float16,stream);eval(ordinary.inputs());
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::scalar_full_eval_layout(float16,3,114,false,cold));
    REQUIRE(cpu::selection_eval_storage(ordinary,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_selection_eval_layout(&raw,MLX_FLOAT16,3,114,true,false));
    CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);
    auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_selection_eval_layout(&raw,MLX_FLOAT64,3,114,true,false));
    CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
    eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,3)==0);
      auto output=full(Shape{2,3,19},source,float16,stream);bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,4,3,4,3,1,16};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.dtype()==float16);CHECK(output.shape()==Shape{2,3,19});
      CHECK(std::memcmp(output.data<void>(),ordinary.data<void>(),114*sizeof(bits))==0);
      for(size_t i=0;i<114;++i){uint16_t actual_bits;std::memcpy(&actual_bits,output.data<mlx::core::float16_t>()+i,2);CHECK(actual_bits==bits);}
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);CHECK(info.known);
      escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    uint16_t last;std::memcpy(&last,escaped->data<mlx::core::float16_t>()+113,2);CHECK(last==bits);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU I32 broadcast Select preserves local indexes and exact escaped source"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<int32_t,114> indexes;for(size_t i=0;i<indexes.size();++i)indexes[i]=int32_t(i)-61;
  indexes[0]=INT32_MIN;indexes.back()=INT32_MAX;
  const bool mask_values[]={true,false,true};
  array local(indexes.data(),Shape{2,3,19},int32),valid(mask_values,Shape{1,3,1},bool_),zero(int32_t{0});
  auto ordinary=where(valid,local,zero,stream);eval(ordinary.inputs());
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::typed_select_broadcast_eval_layout(int32,3,114,false,cold));
  REQUIRE(cpu::selection_eval_storage(ordinary,actual));CHECK(actual.backing_births==1);
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);CHECK(actual.named_control_bytes==cold.named_control_bytes);
  mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_typed_select_broadcast_eval_layout(&raw,MLX_INT32,3,114,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_typed_select_broadcast_eval_layout(&raw,MLX_INT64,3,114,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
  eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,3)==0);
    auto output=where(valid,local,zero,stream);bank.reset();Operation operation;operation.append(output);
    // Three leaves, two broadcasts, Select and Synchronizer; six edges.
    const mlx_operation_eval_traversal_limits limits{1,7,4,6,4,1,24};
    mlx_operation_eval_traversal_layout traversal{};
    REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal,&limits));
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,output);
    CHECK(output.dtype()==int32);CHECK(output.shape()==local.shape());
    CHECK(std::memcmp(output.data<int32_t>(),ordinary.data<int32_t>(),sizeof(indexes))==0);
    for(size_t i=0;i<indexes.size();++i)CHECK(output.data<int32_t>()[i]==(mask_values[(i/19)%3]?indexes[i]:0));
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);CHECK(info.known);
    escaped.emplace(output);
  }
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(escaped->data<int32_t>()[0]==INT32_MIN);CHECK(escaped->data<int32_t>()[113]==INT32_MAX);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU ordered member Stack shares exact N copy jobs and escaped half or float output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(size_t count:{size_t{2},size_t{8}})for(auto dtype:{float32,float16,bfloat16}) {
    ArrayVector members;members.reserve(count);
    for(size_t member=0;member<count;++member) {
      std::array<float,6> values;
      for(size_t i=0;i<values.size();++i)values[i]=float((count-member)*8+i)*.25f;
      members.emplace_back(astype(array(values.data(),Shape{2,3},float32),dtype,stream));
    }
    eval(members);
    auto ordinary=stack(members,0,stream);eval(ordinary.inputs());
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::concatenate_many_eval_layout(dtype,3,count,count*6,false,cold));
    REQUIRE(cpu::concatenate_eval_storage(ordinary,actual));
    CHECK(actual.backing_births==1);CHECK(actual.request_counts[3]==2*count+1);
    CHECK(actual.request_counts[4]==3*count);CHECK(actual.request_counts[5]==3*count);
    CHECK(actual.request_counts[6]==count);CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    mlx_cpu_copy_eval_layout raw{};
    const auto raw_dtype=dtype==float32?MLX_FLOAT32:dtype==float16?MLX_FLOAT16:MLX_BFLOAT16;
    REQUIRE(mlx_operation_event_cpu_concatenate_many_eval_layout(&raw,raw_dtype,3,count,count*6,false));
    CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
    const auto prior=raw;
    CHECK_FALSE(mlx_operation_event_cpu_concatenate_many_eval_layout(&raw,raw_dtype,3,1,6,false));
    CHECK_FALSE(mlx_operation_event_cpu_concatenate_many_eval_layout(&raw,raw_dtype,3,count,count-1,false));
    CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
    eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph_with_operands(&bank.value,observer.value,count+1,0,3,std::max(count,size_t(4)))==0);
      auto output=stack(members,0,stream);bank.reset();Operation operation;operation.append(output);
      // N leaves, N ExpandDims, one Concatenate, one Synchronizer.
      const mlx_operation_eval_traversal_limits limits{1,2*count+2,count+2,2*count+1,count+2,1,4*count+8};
      mlx_operation_eval_traversal_layout traversal{};REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal,&limits));
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==Shape{int(count),2,3});CHECK(output.dtype()==dtype);
      CHECK(std::memcmp(output.data<void>(),ordinary.data<void>(),count*6*dtype.size())==0);
      for(size_t i=0;i<count*6;++i) {
        const float value=dtype==float32?output.data<float>()[i]:dtype==float16?
            float(output.data<mlx::core::float16_t>()[i]):float(output.data<mlx::core::bfloat16_t>()[i]);
        CHECK(value==float((count-i/6)*8+i%6)*.25f);
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);CHECK(info.known);
      escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<void>(),ordinary.data<void>(),count*6*dtype.size())==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU F16 residual pointwise uses exact half workers and escaped original backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(size_t width:{size_t(1),size_t(19),size_t(513)}) {
    std::vector<mlx::core::float16_t> values(2*width),gains(width);
    for(size_t i=0;i<values.size();++i)values[i]=mlx::core::float16_t((int(i%17)-8)*.137f);
    for(size_t i=0;i<gains.size();++i)gains[i]=mlx::core::float16_t(.187f+float(i%7)*.031f);
    array input(values.begin(),Shape{2,int(width)},float16),gain(gains.begin(),Shape{int(width)},float16);
    auto calculate=[&](){return add(tanh(square(multiply(input,gain,stream),stream),stream),input,stream);};
    auto ordinary=calculate();eval(ordinary);
    REQUIRE(ordinary.dtype()==float16);
    cpu::BinaryEvalStorage binary;cpu::UnaryEvalStorage unary;
    REQUIRE(cpu::binary_eval_layout(cpu::BinaryEvalKind::multiply,float16,2,2*width,false,binary));
    REQUIRE(cpu::unary_eval_layout(cpu::UnaryEvalKind::square,float16,2,false,unary));
    mlx_cpu_binary_eval_layout raw_binary{};mlx_cpu_unary_eval_layout raw_unary{};
    REQUIRE(mlx_operation_event_cpu_binary_eval_layout(&raw_binary,uint32_t(cpu::BinaryEvalKind::multiply),MLX_FLOAT16,2,2*width,false));
    REQUIRE(mlx_operation_event_cpu_unary_eval_layout(&raw_unary,uint32_t(cpu::UnaryEvalKind::tanh),MLX_FLOAT16,2,false));
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,5,0,2)==0);
      auto value=calculate();bank.reset();Operation operation;operation.append(value);
      // Two completed inputs, Broadcast/Multiply/Square/Tanh/Add, Synchronizer.
      const mlx_operation_eval_traversal_limits limits{1,8,6,8,6,1,20};
      mlx_operation_eval_traversal_layout traversal{};REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal,&limits));
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,value);
      REQUIRE(value.dtype()==float16);REQUIRE(value.shape()==ordinary.shape());
      CHECK(std::memcmp(value.data<void>(),ordinary.data<void>(),values.size()*sizeof(mlx::core::float16_t))==0);
      for(size_t i=0;i<values.size();++i)CHECK(std::isfinite(float(value.data<mlx::core::float16_t>()[i])));
      CHECK(float(value.data<mlx::core::float16_t>()[0])<-.5f);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);escaped.emplace(value);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<void>(),ordinary.data<void>(),values.size()*sizeof(mlx::core::float16_t))==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU half row Sum retains ordinary SIMD rounding source and escaped backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(Dtype dtype:{float16,bfloat16})for(int rank:{1,3})for(int width:{3,17,1031}) {
    const int rows=rank==1?1:6;
    std::vector<float> values(size_t(rows*width));
    // Exactly representable positive terms make the mathematical row sum an
    // independent oracle while still covering the native SIMD tail.
    for(int row=0;row<rows;++row)for(int col=0;col<width;++col)
      values[size_t(row*width+col)]=col<3?float(row+1)*.125f:0.f;
    Shape shape=rank==1?Shape{width}:Shape{2,3,width};
    array base(values.data(),shape,float32);auto input=astype(base,dtype,stream);eval(input);
    auto ordinary=sum(input,-1,true,stream);
    const auto kind=dtype==float16?cpu::ReductionEvalKind::Float16Rows:cpu::ReductionEvalKind::Bfloat16Rows;
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::reduction_eval_layout(kind,rank,width,rows,false,cold));
    REQUIRE(cpu::reduction_eval_storage(ordinary,actual));
    CHECK(actual.named_control_bytes==cold.named_control_bytes);CHECK(actual.backing_births==1);
    CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.worker_graph_extents==0);
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_reduction_eval_layout(&raw,unsigned(kind),rank,width,rows,false));
    CHECK(raw.named_control_bytes>cold.named_control_bytes);
    const auto saved=cold;
    CHECK_FALSE(cpu::reduction_eval_layout(kind,rank,1,rows,false,cold));
    CHECK_FALSE(cpu::reduction_eval_layout(kind,rank,SIZE_MAX,rows,false,cold));
    CHECK(std::memcmp(&saved,&cold,sizeof(cold))==0);
    if(rank==3) {
      auto wrong=transpose(input,{0,2,1},stream);eval(wrong);
      auto wrong_sum=sum(wrong,-1,true,stream);
      CHECK_FALSE(cpu::reduction_eval_storage(wrong_sum,actual));
    }
    eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,size_t(rank))==0);
      auto output=sum(input,-1,true,stream);bank.reset();Operation operation;operation.append(output);
      // One completed source, Reduce and Synchronizer, with two read edges.
      const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
      mlx_operation_eval_traversal_layout traversal{};
      REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal,&limits));
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.dtype()==dtype);CHECK(output.size()==size_t(rows));
      CHECK(std::memcmp(output.data<void>(),ordinary.data<void>(),size_t(rows)*dtype.size())==0);
      for(int row=0;row<rows;++row) {
        const float value=dtype==float16?float(output.data<mlx::core::float16_t>()[row]):
            float(output.data<mlx::core::bfloat16_t>()[row]);
        CHECK(value==float(row+1)*.375f);
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<void>(),ordinary.data<void>(),size_t(rows)*dtype.size())==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU weightless half RMS preserves half mean then F32 epsilon boundary"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(Dtype dtype:{float16,bfloat16})for(int width:{1,17,1031}) {
    constexpr int rows=6;
    std::vector<float> values(size_t(rows*width));
    for(size_t i=0;i<values.size();++i)values[i]=.0625f+float(i%23)*.046875f;
    array base(values.data(),Shape{2,3,width},float32);auto input=astype(base,dtype,stream);eval(input);
    auto unit=[&]() {
      auto variance=mean(square(input,stream),-1,true,stream);
      auto denominator=rsqrt(add(variance,array(1e-6f,float32),stream),stream);
      return astype(multiply(input,denominator,stream),dtype,stream);
    };
    // This is the existing ordinary worker: half square, half Sum/Divide,
    // then the eager F32 epsilon and final cast back to the input precision.
    auto ordinary=unit();eval(ordinary);
    auto squared=square(input,stream);auto mean_value=mean(squared,-1,true,stream);eval(squared,mean_value);
    REQUIRE(squared.dtype()==dtype);REQUIRE(mean_value.dtype()==dtype);
    auto promoted=add(mean_value,array(1e-6f,float32),stream);eval(promoted);REQUIRE(promoted.dtype()==float32);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,19,2,3)==0);
      auto output=unit();bank.reset();Operation operation;operation.append(output);
      // Nineteen frontend populations plus two scalar seeds and the input;
      // identity casts/broadcasts can elide nodes. Keep their constructor census.
      const mlx_operation_eval_traversal_limits limits{1,23,20,24,20,1,64};
      mlx_operation_eval_traversal_layout traversal{};
      REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal,&limits));
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==input.shape());CHECK(output.dtype()==dtype);
      CHECK(std::memcmp(output.data<void>(),ordinary.data<void>(),values.size()*dtype.size())==0);
      for(size_t i=0;i<values.size();++i) {
        const float value=dtype==float16?float(output.data<mlx::core::float16_t>()[i]):float(output.data<mlx::core::bfloat16_t>()[i]);
        const float x=dtype==float16?float(input.data<mlx::core::float16_t>()[i]):float(input.data<mlx::core::bfloat16_t>()[i]);
        const float denominator=std::sqrt(promoted.data<float>()[i/size_t(width)]);
        CHECK(std::isfinite(value));CHECK(value>0);
        CHECK(std::abs(value-x/denominator)<(dtype==float16?.004f:.03f));
      }
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<void>(),ordinary.data<void>(),values.size()*dtype.size())==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU transposed head rotary shares exact ordinary slices and escaped output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(Shape dims:{Shape{1,2,2,4},Shape{2,3,2,8}})for(int offset:{0,3}) {
    size_t count=1;for(int extent:dims)count*=size_t(extent);
    std::vector<float> data(count);for(size_t i=0;i<count;++i)data[i]=(int(i%19)-9)*.1875f;
    array base(data.data(),dims,float32);
    auto input=transpose(base,{0,2,1,3},stream);eval(input);
    REQUIRE_FALSE(input.flags().row_contiguous);REQUIRE(input.strides().back()==1);
    const int width=dims.back();
    // Dense and strided invocation preserve the same scalar operation order.
    auto dense=contiguous(input,false,stream);eval(dense);
    auto ordinary=fast::rope(dense,width,false,1000000.0f,1.0f,offset,{},stream);eval(ordinary);
    auto strided=fast::rope(input,width,false,1000000.0f,1.0f,offset,{},stream);eval(strided);
    CHECK(std::memcmp(strided.data<void>(),ordinary.data<void>(),count*sizeof(float))==0);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<23,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      // Same complete RoPE frontend census as the contiguous invocation.
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,80,3,4)==0);
      auto output=fast::rope(input,width,false,1000000.0f,1.0f,offset,{},stream);
      bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,100,81,160,81,1,400};
      mlx_operation_eval_traversal_layout traversal{};
      REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal,&limits));
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==input.shape());CHECK(output.dtype()==float32);CHECK(output.flags().row_contiguous);
      CHECK(std::memcmp(output.data<void>(),ordinary.data<void>(),count*sizeof(float))==0);
      bool changed=false;for(size_t i=0;i<count;++i)changed|=output.data<float>()[i]!=dense.data<float>()[i];
      CHECK(changed);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    CHECK(std::memcmp(escaped->data<void>(),ordinary.data<void>(),count*sizeof(float))==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU sampler leaf sources authenticate complete single rows and exact modes"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  float values[7]={2,-1,3,0,3,4,-2};array input(values,Shape{1,1,7},float32);
  auto sorted=argsort(input,-1,stream),partitioned=partition(input,3,-1,stream);
  auto scanned=cumsum(input,-1,false,true,stream),maximum=max(input,-1,true,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::argsort_eval_layout(float32,3,7,1,false,cold));
  REQUIRE(cpu::argsort_eval_storage(sorted,actual));CHECK(actual.allocation_extents==cold.allocation_extents);
  REQUIRE(cpu::argsort_eval_storage(argsort(input,2,stream),actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK_FALSE(cpu::argsort_eval_storage(argsort(input,0,stream),actual));
  REQUIRE(cpu::partition_row_eval_layout(3,7,false,cold));
  REQUIRE(cpu::partition_row_eval_storage(partitioned,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[3]==4);CHECK(actual.request_counts[4]==3);
  CHECK(actual.request_counts[6]==1);CHECK(actual.request_counts[9]==1);
  REQUIRE(cpu::scan_sum_row_eval_layout(3,7,false,cold));
  REQUIRE(cpu::scan_sum_row_eval_storage(scanned,actual));CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.backing_births==1);CHECK(actual.worker_graph_extents==0);
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32FlatMaximum,3,7,1,false,cold));
  REQUIRE(cpu::reduction_eval_storage(maximum,actual));CHECK(actual.allocation_extents==cold.allocation_extents);
  auto before=actual;
  CHECK_FALSE(cpu::scan_sum_row_eval_storage(cumsum(input,-1,true,true,stream),actual));
  CHECK_FALSE(cpu::scan_sum_row_eval_storage(cumsum(input,-1,false,false,stream),actual));
  CHECK_FALSE(cpu::scan_sum_row_eval_storage(cumprod(input,-1,false,true,stream),actual));
  CHECK_FALSE(cpu::partition_row_eval_storage(partition(input,0,0,stream),actual));
  CHECK(std::memcmp(&before,&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_argsort_eval_layout(&raw,MLX_FLOAT32,3,7,1,false));
  REQUIRE(mlx_operation_event_cpu_partition_row_eval_layout(&raw,3,7,false));CHECK(raw.backing_births==1);
  REQUIRE(mlx_operation_event_cpu_scan_sum_row_eval_layout(&raw,3,7,false));CHECK(raw.backing_births==1);
  REQUIRE(mlx_operation_event_cpu_maximum_row_eval_layout(&raw,3,7,false));CHECK(raw.backing_births==1);
  const auto prior=raw;
  CHECK_FALSE(mlx_operation_event_cpu_partition_row_eval_layout(&raw,4,7,false));
  CHECK_FALSE(mlx_operation_event_cpu_scan_sum_row_eval_layout(&raw,3,0,false));
  CHECK(std::memcmp(&prior,&raw,sizeof(raw))==0);
}

TEST_CASE("CPU sampler leaves preserve ordinary ordering scan rounding and escaped backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(int rank:{2,3})for(int width:{19,4097})for(int kind=0;kind<4;++kind) {
    std::vector<float> values(width);for(int i=0;i<width;++i)values[i]=float(i%13-6)*0.125f;
    if(kind<2){values[0]=-0.0f;values[1]=0.0f;values[2]=NAN;values.back()=NAN;}
    Shape shape(rank,1);shape.back()=width;array input(values.data(),shape,float32);
    auto make=[&](){switch(kind) {
      case 0:return argsort(input,-1,stream);
      case 1:return partition(input,width/2,-1,stream);
      case 2:return cumsum(input,-1,false,true,stream);
      default:return max(input,-1,true,stream);
    }};
    auto ordinary=make();eval(ordinary);
    std::vector<uint32_t> ids(width);std::iota(ids.begin(),ids.end(),0);
    std::stable_sort(ids.begin(),ids.end(),[&](uint32_t a,uint32_t b){
      if(std::isnan(values[a]))return false;if(std::isnan(values[b]))return true;return values[a]<values[b];
    });
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,rank)==0);
      auto output=make();bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==ordinary.shape());CHECK(output.dtype()==ordinary.dtype());
      CHECK(std::memcmp(output.data<void>(),ordinary.data<void>(),output.nbytes())==0);
      if(kind==0)for(int i=0;i<width;++i)CHECK(output.data<uint32_t>()[i]==ids[i]);
      if(kind==1)CHECK(output.data<float>()[width/2]==values[ids[width/2]]);
      if(kind==2){float sum=0;for(int i=0;i<width;++i){sum+=values[i];CHECK(output.data<float>()[i]==sum);}}
      if(kind==3)CHECK(output.data<float>()[0]==*std::max_element(values.begin(),values.end()));
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}


TEST_CASE("CPU sampling GatherAxis retains exact ordering indices and escaped scores"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(int width:{19,4097}) {
    std::vector<float> values(width);std::vector<uint32_t> order(width);
    for(int i=0;i<width;++i){values[i]=float(i%17-8)*0.25f;order[i]=width-1-i;}
    array input(values.data(),Shape{1,1,width},float32),indices(order.data(),Shape{1,1,width},uint32);
    auto ordinary=take_along_axis(input,indices,-1,stream);cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::gather_axis_row_eval_layout(3,width,false,cold));
    REQUIRE(cpu::gather_axis_row_eval_storage(ordinary,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);CHECK(actual.backing_births==1);
    auto signed_indices=astype(indices,int32,stream);eval(signed_indices);
    REQUIRE(cpu::gather_axis_row_eval_storage(take_along_axis(input,signed_indices,-1,stream),actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    auto wide_indices=astype(indices,int64,stream);eval(wide_indices);
    auto before=actual;CHECK_FALSE(cpu::gather_axis_row_eval_storage(take_along_axis(input,wide_indices,-1,stream),actual));
    CHECK(std::memcmp(&before,&actual,sizeof(actual))==0);
    mlx_cpu_copy_eval_layout raw{};REQUIRE(mlx_operation_event_cpu_gather_axis_row_eval_layout(&raw,3,width,false));
    CHECK(raw.graph_extents==cold.allocation_extents);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<22,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,3)==0);
      auto output=take_along_axis(input,indices,-1,stream);bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,4,2,3,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      for(int i=0;i<width;++i){CHECK(output.data<float>()[i]==values[order[i]]);CHECK(output.data<float>()[i]==ordinary.data<float>()[i]);}
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);CHECK(info.known);
      escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU positive Slice keeps exact stepped coordinates and original escaped backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
  unsigned retired=0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void* p){++*static_cast<unsigned*>(p);})==0);
  const float values[]={7,99,-11,99,23,99,13,99,17,99,-19,99,31,99,43,99};
  array base(values,Shape{2,1,8},float32);
  auto strided=slice(base,Shape{0,0,0},Shape{2,1,8},Shape{1,1,2},stream);eval(strided);
  auto ordinary_source=contiguous(strided,false,stream);eval(ordinary_source);
  auto ordinary=slice(ordinary_source,Shape{0,0,0},Shape{2,1,3},Shape{1,1,2},stream);eval(ordinary);
  std::optional<array> source,escaped;
  uint64_t identity=0;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=contiguous(strided,false,stream);bank.reset();
    Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    REQUIRE(info.known);identity=info.identity;source.emplace(value);
  }
  const auto occupied=mlx_original_buffer_budget_occupied(budget.value);REQUIRE(occupied>0);
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=slice(*source,Shape{0,0,0},Shape{2,1,3},Shape{1,1,2},stream);bank.reset();
    Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);CHECK(info.identity==identity);
    CHECK(value.shape()==Shape{2,1,2});CHECK_FALSE(value.flags().row_contiguous);
    CHECK(value.strides()[2]==2);
    const float expected[]={7,23,17,31};
    for(size_t row=0;row!=2;++row)for(size_t col=0;col!=2;++col) {
      auto offset=row*value.strides()[0]+col*value.strides()[2];
      auto reference=row*ordinary.strides()[0]+col*ordinary.strides()[2];
      CHECK(value.data<float>()[offset]==expected[row*2+col]);
      CHECK(value.data<float>()[offset]==ordinary.data<float>()[reference]);
    }
    escaped.emplace(value);
  }
  source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==occupied);
  CHECK(retired==0);escaped.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
}

TEST_CASE("CPU General Select preserves stepped input span and escaped output custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const std::array<float,9> data{-4.f,91.f,2.f,92.f,-7.f,93.f,8.f,94.f,11.f};
  const std::array<bool,5> choices{true,false,true,true,false};
  auto values=slice(array(data.data(),Shape{9},float32),{0},{9},{2},stream);
  array mask(choices.data(),Shape{5},bool_);
  auto floor=broadcast_to(array(-3.f),Shape{5},stream);eval(values,floor);
  CHECK_FALSE(values.flags().contiguous);CHECK(values.strides()[0]==2);
  auto descriptor=where(mask,values,floor,stream);eval(descriptor.inputs());
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::select_broadcast_eval_layout(1,5,false,cold));
  REQUIRE(cpu::selection_eval_storage(descriptor,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  const auto prior=actual;
  auto short_values=array(Shape{5},float32,nullptr,{});
  short_values.copy_shared_buffer(array(1.f));
  auto outside=array(Shape{5},float32,std::make_shared<Select>(stream),{mask,short_values,floor});
  CHECK_FALSE(cpu::selection_eval_storage(outside,actual));
  CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
  eval(descriptor);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
    auto output=where(mask,values,floor,stream);bank.reset();
    Operation operation;operation.append(output);
    // Three completed source leaves, one Select and its Synchronizer.
    const mlx_operation_eval_traversal_limits limits{1,5,2,4,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,output);
    for(size_t i=0;i<choices.size();++i) {
      const float expected=choices[i]?data[2*i]:-3.f;
      CHECK(output.data<float>()[i]==expected);
      CHECK(output.data<float>()[i]==descriptor.data<float>()[i]);
    }
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);CHECK(info.known);
    escaped.emplace(output);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  escaped.reset();CHECK(retired==1);
}


#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("MXFP4 explicit-index grouped projection owns five operands through Metal completion"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  prepared_metal_fixture::StreamOwner registered;
  const auto stream=registered.stream();
  prepare(stream,stream);
  size_t controls=0;
  REQUIRE(mlx_mxfp4_gather_control_bytes(&controls));
  CHECK(controls>0);
  CHECK_FALSE(mlx_mxfp4_gather_control_bytes(nullptr));
  for (int width : {32,512}) for (bool strided : {false,true}) {
    CAPTURE(width); CAPTURE(strided);
    constexpr int routes=3,groups=2,rows=8;
    const int stride=strided?2:1;
    std::vector<float> xs(routes*width*stride,0.f);
    std::vector<uint32_t> ws(groups*rows*(width/8)*stride,0);
    std::vector<uint8_t> ss(groups*rows*(width/32)*stride,0);
    std::vector<float> expected(routes*rows);
    for(int r=0;r<routes;++r) {
      float sum=0;
      for(int k=0;k<width;++k) {const float v=float((r+k)%9-4)/4;xs[(r*width+k)*stride]=v;sum+=v;}
      for(int n=0;n<rows;++n) expected[r*rows+n]=sum*(r==1?-0.5f:1.f);
    }
    for(int g=0;g<groups;++g) {
      for(int i=0;i<rows*width/8;++i) ws[(g*rows*width/8+i)*stride]=g?0xaaaaaaaau:0x22222222u;
      for(int i=0;i<rows*width/32;++i) ss[(g*rows*width/32+i)*stride]=g?126:127;
    }
    auto x=as_strided(array(xs.begin(),Shape{int(xs.size())},float32),Shape{routes,width},Strides{Strides::value_type(width*stride),Strides::value_type(stride)},0,stream);
    auto w=as_strided(array(ws.begin(),Shape{int(ws.size())},uint32),Shape{groups,rows,width/8},Strides{Strides::value_type(rows*width/8*stride),Strides::value_type(width/8*stride),Strides::value_type(stride)},0,stream);
    auto scales=as_strided(array(ss.begin(),Shape{int(ss.size())},uint8),Shape{groups,rows,width/32},Strides{Strides::value_type(rows*width/32*stride),Strides::value_type(width/32*stride),Strides::value_type(stride)},0,stream);
    array ids({0,1,0},int32);
    eval({x,w,scales,ids});
    auto ordinary=reshape(gather_qmm(reshape(x,{routes,1,width},stream),w,scales,std::nullopt,
        arange(0,routes,1,uint32,stream),ids,true,32,4,"mxfp4",true,stream),{routes,rows},stream);
    eval(ordinary);
    mlx_prepared_input_runtime runtime{};
    REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget { mlx_original_buffer_budget value{};
      ~Budget(){mlx_original_buffer_budget_release(value);} } budget;
    unsigned retired=0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void* p){++*static_cast<unsigned*>(p);})==0);
    struct Pipeline { mlx_pipeline_cache value{};
      ~Pipeline(){mlx_pipeline_cache_free(value);} } pipeline;
    REQUIRE(mlx_pipeline_cache_new_retaining(&pipeline.value,8,new int(0),wait_record_facts::release_token)==0);
    std::optional<array> escaped;
    {
      Role role(pipeline.value);
      REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      // Arange, input reshape, signed route cast, GatherQMM and final reshape.
      REQUIRE(mlx_operation_event_prepare_resident_graph_with_operands(&bank.value,observer.value,5,0,3,5)==0);
      auto value=reshape(gather_qmm(reshape(x,{routes,1,width},stream),w,scales,std::nullopt,
          arange(0,routes,1,uint32,stream),ids,true,32,4,"mxfp4",true,stream),{routes,rows},stream);
      escaped=value;
      bank.reset();
      Operation operation;operation.append(value);
      // Four input leaves + five nodes + Synchronizer; the five-input QMM
      // and its three possible row compactions retain at most nine buffers.
      const mlx_operation_eval_traversal_limits limits{1,10,6,9,6,1,9};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,value);
      for(size_t i=0;i<expected.size();++i) {
        CHECK(value.data<float>()[i]==doctest::Approx(expected[i]));
        CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);
      }
      CHECK(escaped->data_shared_ptr()==value.data_shared_ptr());
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      CHECK(info.known);
      CHECK(role.records->occupied_bytes()==0);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    for(size_t i=0;i<expected.size();++i) CHECK(escaped->data<float>()[i]==doctest::Approx(expected[i]));
    escaped.reset();CHECK(retired==1);
  }
}
#endif


#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("Metal original indexed sum accumulates duplicate rows and retains output custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  prepared_metal_fixture::StreamOwner registered;
  auto stream=registered.stream();prepare(stream,stream);
  const float source_data[]={10,20,30,40,50,60,70,80};
  const float contiguous_updates[]={1,2,3,4,5,6};
  const float transposed_updates[]={1,3,5,2,4,6};
  const float expected[]={15,26,30,40,54,66,70,80};
  for(auto dtype:{float32,float16,bfloat16})for(bool signed_index:{false,true})
  for(bool update_contiguous:{false,true})for(bool index_contiguous:{false,true}) {
    CAPTURE(dtype);CAPTURE(signed_index);CAPTURE(update_contiguous);CAPTURE(index_contiguous);
    auto source=astype(array(source_data,Shape{4,2},float32),dtype,stream);
    auto updates=astype(array(update_contiguous?contiguous_updates:transposed_updates,
        update_contiguous?Shape{3,2}:Shape{2,3},float32),dtype,stream);
    if(!update_contiguous)updates=transpose(updates,stream);
    const int32_t signed_contiguous[]={2,2,2,2,-4,-4};
    const int32_t signed_transposed[]={2,2,-4,2,2,-4};
    const uint32_t unsigned_contiguous[]={2,2,2,2,0,0};
    const uint32_t unsigned_transposed[]={2,2,0,2,2,0};
    const Shape index_shape=index_contiguous?Shape{3,2}:Shape{2,3};
    auto index=signed_index
        ?array(index_contiguous?signed_contiguous:signed_transposed,index_shape,int32)
        :array(index_contiguous?unsigned_contiguous:unsigned_transposed,index_shape,uint32);
    if(!index_contiguous)index=transpose(index,stream);
    eval({source,index,updates});
    REQUIRE(updates.flags().row_contiguous==update_contiguous);
    REQUIRE(index.flags().row_contiguous==index_contiguous);
    auto ordinary=scatter_add_axis(source,index,updates,0,stream);eval(ordinary);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    struct Pipeline{mlx_pipeline_cache value{};~Pipeline(){mlx_pipeline_cache_free(value);}}pipeline;
    REQUIRE(mlx_pipeline_cache_new_retaining(&pipeline.value,8,new int(0),wait_record_facts::release_token)==0);
    std::optional<array> escaped;
    {
      Role role(pipeline.value);
      REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      // The ordinary adapter has one cast, five possible broadcasts and the
      // same ScatterAxis sum. The admitted constructor bank covers all seven.
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,7,0,2)==0);
      auto value=scatter_add_axis(source,index,updates,0,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,6,3,5,3,1,16};
      const auto status=eval_traversal_tests::submit(operation,stream,limits);
      const auto* failure=role.error.get()->borrow();
      INFO("indexed sum retained source: ",doctest::String(failure&&failure->message?failure->message:"none"));
      INFO("indexed sum retained type: ",doctest::String(failure&&failure->exception_type?failure->exception_type->name():"none"));
      REQUIRE(status==0);
      eval_traversal_tests::complete(role,operation,value);
      CHECK(role.records->occupied_bytes()==0);
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);CHECK(info.known);
      escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    for(size_t i=0;i!=8;++i) {
      const float actual=dtype==float32?escaped->data<float>()[i]:dtype==float16
          ?float(escaped->data<mlx::core::float16_t>()[i]):float(escaped->data<mlx::core::bfloat16_t>()[i]);
      const float reference=dtype==float32?ordinary.data<float>()[i]:dtype==float16
          ?float(ordinary.data<mlx::core::float16_t>()[i]):float(ordinary.data<mlx::core::bfloat16_t>()[i]);
      CHECK(actual==expected[i]);CHECK(actual==reference);
    }
    escaped.reset();CHECK(retired==1);
  }
}
#endif


TEST_CASE("CPU flat Scatter source binds general overwrite and preserves invalid-query output"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-3,2,1,7,-2,4,0,3,8,-1,2,4,5};
  const int32_t picks[]={-1,2,2,0,4};
  const float changes[]={0.5f,1.5f,2.5f,3.5f,4.5f};
  array source(data,Shape{13},float32),index(picks,Shape{5},int32),updates(changes,Shape{5,1},float32);
  auto value=scatter(source,index,updates,0,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::scatter_eval_layout(float32,int32,1,13,5,false,cold));
  REQUIRE(cpu::flat_scatter_eval_storage(value,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.backing_births==1);CHECK(actual.request_counts[6]==2);
  CHECK(actual.request_counts[8]==1);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  CHECK_FALSE(cpu::scatter_eval_layout(float32,int64,1,13,5,false,actual));
  CHECK_FALSE(cpu::scatter_eval_layout(float32,int32,1,13,0,false,actual));
  CHECK_FALSE(cpu::scatter_eval_layout(float32,int32,1,SIZE_MAX,5,false,actual));
  auto sum=array(Shape{13},float32,std::make_shared<Scatter>(stream,Scatter::Sum,std::vector<int>{0}),{source,index,updates});
  auto wrong=array(Shape{12},float32,std::make_shared<Scatter>(stream,Scatter::None,std::vector<int>{0}),{source,index,updates});
  for(const auto* invalid:{&sum,&wrong})CHECK_FALSE(cpu::flat_scatter_eval_storage(*invalid,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_scatter_eval_layout(&raw,MLX_FLOAT32,MLX_INT32,1,13,5,false));
  CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.worker_graph_extents==cold.worker_graph_extents);
}
TEST_CASE("CPU flat Scatter preserves ordinary row order and escaped original copy custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float data[]={-3,2,1,7,-2,4,0,3,8,-1,2,4,5};
  const int32_t picks[]={-1,2,2,0,4};
  const float changes[]={0.5f,1.5f,2.5f,3.5f,4.5f};
  array source(data,Shape{13},float32),index(picks,Shape{5},int32),updates(changes,Shape{5,1},float32);
  auto ordinary=scatter(source,index,updates,0,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    // The unchanged native constructor has two cast candidates and Scatter.
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,2)==0);
    auto value=scatter(source,index,updates,0,stream);
    bank.reset();Operation operation;operation.append(value);
    // The traversal envelope reserves one output pin per possible tape row,
    // including its Synchronizer; the query rejects fewer pins before Eval.
    const mlx_operation_eval_traversal_limits limits{1,6,3,5,3,1,16};
    mlx_operation_eval_traversal_layout frontier{};
    REQUIRE(mlx_operation_event_eval_traversal_layout(&frontier,&limits));
    const auto status=eval_traversal_tests::submit(operation,stream,limits);
    const auto* failure=role.error.get()->borrow();
    INFO("flat Scatter retained source: ",doctest::String(failure&&failure->message?failure->message:"none"));
    INFO("flat Scatter retained type: ",doctest::String(failure&&failure->exception_type?failure->exception_type->name():"none"));
    REQUIRE(status==0);
    eval_traversal_tests::complete(role,operation,value);
    for(size_t column=0;column!=13;++column) {
      float expected=data[column];
      for(size_t pick=0;pick!=5;++pick) {
        int32_t actual=picks[pick]<0?picks[pick]+13:picks[pick];
        if(size_t(actual)==column)expected=changes[pick];
      }
      CHECK(value.data<float>()[column]==expected);
      CHECK(value.data<float>()[column]==ordinary.data<float>()[column]);
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(escaped->data<float>()[2]==changes[2]);CHECK(escaped->data<float>()[12]==changes[0]);
  escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU paged row maximum source preserves exact geometry and query refusal"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,168> data;for(size_t i=0;i!=data.size();++i)data[i]=float(int(i%23)-11)*0.125f;
  array input(data.data(),Shape{2,3,4,7},float32);
  auto output=max(input,-1,true,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32MaximumRows,4,7,24,false,cold));
  REQUIRE(cpu::reduction_eval_storage(output,actual));
  CHECK(cold.allocation_extents==actual.allocation_extents);
  CHECK(cold.named_control_bytes==actual.named_control_bytes);
  CHECK(actual.backing_births==1);CHECK(actual.worker_graph_extents==0);
  std::array<unsigned char,sizeof(actual)> saved;std::memcpy(saved.data(),&actual,sizeof(actual));
  for(const auto& geometry:std::array<std::array<size_t,3>,5>{{{0,7,24},{5,7,24},{4,1,24},{4,7,0},{4,SIZE_MAX,24}}}) {
    CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32MaximumRows,
        geometry[0],geometry[1],geometry[2],false,actual));
    CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  }
  auto wrong_axis=max(input,1,true,stream);CHECK_FALSE(cpu::reduction_eval_storage(wrong_axis,actual));
  CHECK(std::memcmp(saved.data(),&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_reduction_eval_layout(&raw,9,4,7,24,false));
  auto prior=raw;CHECK_FALSE(mlx_operation_event_cpu_reduction_eval_layout(&raw,9,4,SIZE_MAX,24,false));
  CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
}
TEST_CASE("CPU paged row maxima retain ordinary SIMD values and escaped source custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,168> data;for(size_t i=0;i!=data.size();++i)data[i]=float(int((i*7)%23)-11)*0.125f;
  data[17]=std::numeric_limits<float>::quiet_NaN();
  for(const auto& shape:{Shape{2,3,4,7},Shape{1,1,1,7}}) {
  array input(data.data(),shape,float32);
  const auto rows=input.size()/7;
  auto ordinary=max(input,-1,true,stream);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,4)==0);
    auto value=max(input,-1,true,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(size_t row=0;row!=rows;++row) {
      float expected=-std::numeric_limits<float>::infinity();bool has_nan=false;
      for(size_t col=0;col!=7;++col){const auto x=data[row*7+col];has_nan|=std::isnan(x);expected=std::max(expected,x);}
      const auto actual=value.data<float>()[row],reference=ordinary.data<float>()[row];
      CHECK(((std::isnan(actual)&&std::isnan(reference))||actual==reference));
      if(!has_nan)CHECK(actual==expected);
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value)>0);CHECK(retired==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(escaped->data<float>()[0]==ordinary.data<float>()[0]);
  escaped.reset();CHECK(retired==1);CHECK(input.data<float>()[0]==data[0]);
  }
}
TEST_CASE("CPU paged score rows preserve singleton transposed query copies and original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const Stream selected(stream.index,stream.device,CpuMatmulKernel::Float32Tiles);
  constexpr int heads=3,queries=1,keys=5,width=4;
  std::array<float,heads*queries*width> qv;
  std::array<float,heads*keys*width> kv;
  for(size_t i=0;i<qv.size();++i)qv[i]=(int(i%11)-5)*0.125f;
  for(size_t i=0;i<kv.size();++i)kv[i]=(int((i*7)%17)-8)*0.0625f;
  array q_source(qv.data(),Shape{1,queries,heads,width},float32),k_source(kv.data(),Shape{1,heads,keys,width},float32);
  auto q=transpose(q_source,{0,2,1,3},selected),k=swapaxes(k_source,-1,-2,selected);eval(q,k);
  REQUIRE(q.flags().row_contiguous);CHECK(q.strides()[2]!=width);
  auto score=matmul(q,k,selected);cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::tiled_matmul_copy_eval_layout(4,queries,keys,width,heads,1,false,cold));
  REQUIRE(cpu::tiled_matmul_eval_storage(score,actual));
  CHECK(actual.backing_births==cold.backing_births);CHECK(actual.backing_births==2);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  auto ordinary=max(score,-1,true,selected);eval(ordinary);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,4,0,4)==0);
    auto value=max(matmul(q,k,selected),-1,true,selected);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,9,5,10,5,1,32};
    REQUIRE(eval_traversal_tests::submit(operation,selected,limits)==0);eval_traversal_tests::complete(role,operation,value);
    for(int h=0;h<heads;++h) {
      float expected=-std::numeric_limits<float>::infinity();
      for(int pos=0;pos<keys;++pos){float product=0;for(int d=0;d<width;++d)product+=qv[h*width+d]*kv[(h*keys+pos)*width+d];expected=std::max(expected,product);}
      CHECK(value.data<float>()[h]==ordinary.data<float>()[h]);CHECK(value.data<float>()[h]==expected);
    }
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
    CHECK(info.known);escaped.emplace(value);
  }
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(escaped->data<float>()[0]==ordinary.data<float>()[0]);escaped.reset();CHECK(retired==1);
  CHECK(q_source.data<float>()[0]==qv[0]);CHECK(k_source.data<float>()[0]==kv[0]);
}

#include "mlx/host_transfer.h"
TEST_CASE("CPU Host transfer source preserves scalar copy geometry and rejects foreign backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(const auto dtype:{float16,bfloat16,float32}) {
    for(size_t rank=1;rank<=4;++rank) {
      for(bool store:{false,true}) {
        cpu::CopyEvalStorage cold;
        REQUIRE(cpu::host_transfer_eval_layout(dtype,rank,store,false,cold));
        CHECK(cold.backing_births==size_t(!store));
        CHECK(cold.allocation_extents>0);CHECK(cold.named_control_bytes>0);
        const auto before=cold;
        CHECK_FALSE(cpu::host_transfer_eval_layout(dtype,0,store,false,cold));
        CHECK(std::memcmp(&before,&cold,sizeof(cold))==0);
        CHECK_FALSE(cpu::host_transfer_eval_layout(dtype,5,store,false,cold));
        CHECK(std::memcmp(&before,&cold,sizeof(cold))==0);
        CHECK_FALSE(cpu::host_transfer_eval_layout(int32,rank,store,false,cold));
        CHECK(std::memcmp(&before,&cold,sizeof(cold))==0);
      }
    }
  }
  const float data[]={2,-3,7,11,-5,13};
  array input(data,Shape{2,3},float32);
  HostTransferBuffer host(Shape{2,3},float32,HostTransferPolicy::transfer);
  auto stored=array(input.shape(),input.dtype(),std::make_shared<CopyToHostTransfer>(stream,host),{input});
  cpu::CopyEvalStorage actual;
  REQUIRE(cpu::copy_eval_storage(stored,actual));CHECK(actual.backing_births==0);
  eval(stored);
  for(size_t i=0;i<6;++i)CHECK(static_cast<const float*>(host.data())[i]==data[i]);
  auto leaf=stored; // actual completed Host-backed output, without fabricating a source
  auto loaded=array(Shape{2,3},float32,std::make_shared<CopyFromHostTransfer>(stream),{leaf});
  REQUIRE(cpu::copy_eval_storage(loaded,actual));CHECK(actual.backing_births==1);
  eval(loaded);
  for(size_t i=0;i<6;++i)CHECK(loaded.data<float>()[i]==data[i]);
  const auto before=actual;
  auto foreign=array(Shape{2,3},float32,std::make_shared<CopyFromHostTransfer>(stream),{input});
  CHECK_FALSE(cpu::copy_eval_storage(foreign,actual));CHECK(std::memcmp(&before,&actual,sizeof(actual))==0);
  auto malformed=array(Shape{6},float32,std::make_shared<CopyFromHostTransfer>(stream),{leaf});
  CHECK_FALSE(cpu::copy_eval_storage(malformed,actual));CHECK(std::memcmp(&before,&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout query{};
  REQUIRE(mlx_operation_event_cpu_host_transfer_eval_layout(&query,MLX_FLOAT32,2,false,false));
  CHECK(query.backing_births==1);const auto unchanged=query;
  CHECK_FALSE(mlx_operation_event_cpu_host_transfer_eval_layout(&query,MLX_FLOAT32,SIZE_MAX,false,false));
  CHECK(std::memcmp(&unchanged,&query,sizeof(query))==0);
}

#include "cpu_broadcast_reshape_tests.cpp"
#include "cpu_strided_sum_tests.cpp"

#include "cpu_grouped_index_tests.cpp"

#include "cpu_tiled_gather_mm_tests.cpp"
#include "cpu_grouped_gather_axis_tests.cpp"
#include "cpu_empty_typed_join_tests.cpp"

#include "cpu_row_movement_tests.cpp"

#include "cpu_empty_structural_alias_tests.cpp"

#include "cpu_routing_selection_tests.cpp"
#include "cpu_empty_slice_tests.cpp"

#include "cpu_quantization_unary_tests.cpp"

#include "cpu_quantization_integer_tests.cpp"
#include "cpu_quantization_half_tests.cpp"
