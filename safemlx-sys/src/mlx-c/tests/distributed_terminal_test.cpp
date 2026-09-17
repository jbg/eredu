// Isolated canonical native group fencing; no real transport is synthesized.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <optional>
#include <iostream>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <thread>
#include <utility>

#include "mlx/c/distributed_group.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/ring/socket_tasks.h"
#include "mlx/distributed/ring/reduction_plan.h"
#include "mlx/distributed/ring/worker_geometry.h"
#include "mlx/distributed/ring/worker_storage.h"
#include "mlx/threadpool.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/c/submission.h"
#include "mlx/submission.h"

namespace {
using namespace mlx::core;
namespace dist = mlx::core::distributed;
void check(bool ok, const char* what) {
  if (!ok) {
    throw std::runtime_error(what);
  }
}
void ring_worker_partitions_and_storage() {
  using namespace dist::ring;
  WorkerPartitions gather{};
  check(gather_partitions(1048581, 4, gather) && gather.jobs == 4 && gather.step == 262146 &&
      gather.range(3) == std::pair<size_t,size_t>{786438,1048581}, "gather partition differs");
  ReductionWorkers reduce{};
  check(reduction_workers(786433, 4, 3, 4, reduce) && !reduce.padded &&
      reduce.parts.jobs == 4 && reduce.parts.range(3) == std::pair<size_t,size_t>{589827,786433},
      "reduction partition differs");
  check(reduction_workers(2, 4, 3, 4, reduce) && reduce.padded && reduce.parts.items == 3 &&
      reduce.parts.jobs == 1, "small reduction did not preserve padded worker");
  DirectWorkers direct{};
  check(direct_workers(4097,3,direct) && direct.parts.jobs == 3 && direct.reserved_futures == 3 &&
      direct.parts.range(2) == std::pair<size_t,size_t>{2732,4097}, "direct partition differs");
  check(direct_workers(0,3,direct) && direct.parts.jobs == 0 && direct.reserved_futures == 3,
      "empty direct operation changed reserved destination");
  check(!gather_partitions(17,0,gather) && !direct_workers(17,0,direct) &&
      !reduction_workers(std::numeric_limits<size_t>::max(),8,3,4,reduce), "invalid worker geometry accepted");
  WorkerStorageCounter counter;
  dist::GroupWorkerStorage storage{};
  ReductionPlanLayout plan{};
  SocketTaskLayout socket{};
  const bool native_layout = PreparedReductionPlan::layout(7,3,4,8*1024*1024,plan) && socket_task_layout(socket);
  if (native_layout) {
    check(counter.plan(7,3,4,8*1024*1024) && counter.finish(0,storage), "actual worker storage refused");
    check(storage.pool_jobs == 0 && storage.socket_attempts == 8 && storage.destination_arrays == 2 &&
        storage.task_graph_extent == 8*(socket.node_extent+socket.promise_extent) &&
        storage.destination_graph_extent == 2*plan.plan_extent && storage.controls > 0,
        "worker storage did not consume exact socket/plan layouts");
    WorkerStorageCounter overflow;
    check(!overflow.sockets(std::numeric_limits<size_t>::max()), "overflowed worker population accepted");
  } else {
    check(!counter.plan(7,3,4,8*1024*1024), "unknown native ABI granted storage");
  }
  // Actual EmptyGroup must not become a Ring operation source from requested
  // backend or scalar geometry. Query leaves the caller's sentinel untouched.
  auto group = dist::init(false,"ring");
  array input({7,-11,23});
  dist::GroupWorkerStorage unchanged{};
  unchanged.pool_jobs = 73;
  check(!group.worker_storage(dist::GroupWorkerOperation::sum,input,0,unchanged) &&
      unchanged.pool_jobs == 73, "singleton fallback fabricated Ring worker source");
  mlx_distributed_worker_storage null_result{};
  null_result.pool_jobs = 91;
  check(!mlx_distributed_group_worker_storage(&null_result,{nullptr},{nullptr},0,0) &&
      null_result.pool_jobs == 91, "invalid native source changed destination");
}

struct SourceDispatchProbe {
  submission::GraphQuotaRef allocation;
  const int* input;
  int* output;
  void operator()() const {
    check(submission::current_graph_quota() == nullptr, "outer worker unexpectedly has Graph TLS");
    check(allocation.get() != nullptr, "outer callback lost creator source");
    *output = input[0] * 3 + input[1] * 2 - input[2];
  }
};
void outer_dispatch_type_and_source_retirement() {
  cpu::DispatchLayout layout{};
  check(cpu::CommandEncoder::dispatch_layout<SourceDispatchProbe>(layout), "outer task layout refused");
  using Bound = decltype(std::bind(std::declval<SourceDispatchProbe>()));
  check(layout.task_bytes == sizeof(scheduler::TaskNode<Bound>) &&
      layout.task_alignment == alignof(scheduler::TaskNode<Bound>) && layout.controls > 0,
      "encoder layout differs from its actual bound task");
  size_t capacity;
  check(submission::GraphQuota::fresh_capacity_for_extents(layout.graph_extent,capacity), "outer capacity overflow");
  int retired=0, result=0;
  int input[]={7,-11,23};
  auto* graph=submission::GraphQuota::create(capacity,&retired,[](void* p){++*static_cast<int*>(p);});
  {
    auto body=std::bind(SourceDispatchProbe{submission::GraphQuotaRef(graph),input,&result});
    auto task=scheduler::make_task(graph,{},nullptr,false,std::move(body));
    graph->release();
    check(retired==0,"outer source retired before accepted task");
    std::thread worker([task=std::move(task)]() mutable {
      auto* value=task.get(); value->invoke(value); task.reset();
    });
    worker.join();
    check(result==-24,"outer callable nonzero result changed");
  }
  check(retired==1,"outer task/body did not retire exact creator source");
  size_t tiny_capacity;
  check(submission::GraphQuota::fresh_capacity_for_extents(0,tiny_capacity),"tiny capacity unavailable");
  auto* tiny=submission::GraphQuota::create(tiny_capacity,nullptr,nullptr);
  bool refused=false;
  try {
    auto body=std::bind(SourceDispatchProbe{submission::GraphQuotaRef(tiny),input,&result});
    auto task=scheduler::make_task(tiny,{},nullptr,false,std::move(body));
  } catch(const submission::GraphQuotaError&) {refused=true;}
  check(refused && tiny->occupied_bytes()==0,"outer refusal retained partial task");
  tiny->release();
  dist::GroupDispatchStorage untouched{};
  untouched.task_bytes=83;
  auto group=dist::init(false,"ring");
  array value({7,-11,23});
  check(!group.dispatch_storage(dist::GroupWorkerOperation::sum,value,0,untouched) &&
      untouched.task_bytes==83,"singleton fallback fabricated a Ring dispatch source");
}

struct SplitGate {
  std::atomic<bool> entered{false}, release{false};
};
struct Probe final : dist::detail::GroupImpl {
  explicit Probe(std::shared_ptr<SplitGate> gate = {}) : gate(std::move(gate)) {}
  std::shared_ptr<SplitGate> gate;
  Stream communication_stream(StreamOrDevice s) override { return to_stream(s); }
  int rank() override { return 0; }
  int size() override { return 2; }
  std::shared_ptr<dist::detail::GroupImpl> split(int, int) override {
    if (gate) {
      gate->entered.store(true);
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!gate->release.load()) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("split gate timed out");
        }
        std::this_thread::yield();
      }
    }
    return std::make_shared<Probe>();
  }
  void all_sum(const array&, array&, Stream) override {
    throw std::logic_error("no test transport");
  }
  void all_gather(const array&, array&, Stream) override {
    throw std::logic_error("no test transport");
  }
  void send(const array&, int, Stream) override {
    throw std::logic_error("no test transport");
  }
  void recv(array&, int, Stream) override {
    throw std::logic_error("no test transport");
  }
  void all_max(const array&, array&, Stream) override {
    throw std::logic_error("no test transport");
  }
  void all_min(const array&, array&, Stream) override {
    throw std::logic_error("no test transport");
  }
  void sum_scatter(const array&, array&, Stream) override {
    throw std::logic_error("no test transport");
  }
  void all_to_all_v(
      const array&,
      array&,
      const std::vector<size_t>&,
      const std::vector<size_t>&,
      Stream) override {
    throw std::logic_error("no test transport");
  }
};
template <class F>
void rejected(F&& f) {
  bool failed = false;
  try {
    f();
  } catch (const std::runtime_error&) {
    failed = true;
  }
  check(failed, "terminal operation was not rejected");
}
struct Scope {
  mlx_submission_scope value{nullptr};
  Scope() {
    check(mlx_submission_scope_new(&value) == 0, "scope creation failed");
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() { mlx_submission_scope_free(value); }
};
struct HeldRecord final : submission::Record {
  HeldRecord(dist::Group group, std::shared_ptr<std::atomic<int>> retired)
      : group(std::move(group)), retired(std::move(retired)) {}
  ~HeldRecord() override { ++*retired; }
  dist::Group group;
  std::shared_ptr<std::atomic<int>> retired;
};
void reduction_plan_geometry_source_and_intervals() {
  using namespace dist::ring;
  constexpr size_t packet_bytes = 8 * 1024 * 1024;
  const ReductionSpan expected_send[] = {{3,6},{0,3},{6,7},{3,6}};
  const ReductionSpan expected_recv[] = {{0,3},{6,7},{3,6},{0,3}};
  PreparedReductionPlan ordinary(7, 3, 1, -1, sizeof(float), packet_bytes, nullptr);
  check(ordinary.sends().size() == 4 && ordinary.recvs().size() == 4,
        "nondivisible reduction plan population changed");
  for (size_t i = 0; i != 4; ++i)
    check(ordinary.sends()[i] == expected_send[i] && ordinary.recvs()[i] == expected_recv[i],
          "scatter/gather segment routing changed");
  ReductionPlanLayout layout{};
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106 && __cplusplus == 202002L
  check(PreparedReductionPlan::layout(262147, 3, sizeof(float), packet_bytes, layout),
        "multi-packet actual plan geometry refused");
  check(layout.geometry.segment_size == 87383 && layout.geometry.buffer_size == 43691 &&
        layout.geometry.packets == 3 && layout.geometry.steps == 12,
        "actual uneven packet geometry changed");
  size_t capacity;
  check(submission::GraphQuota::fresh_capacity_for_extents(2 * layout.plan_extent, capacity),
        "reduction source capacity overflow");
  int retired = 0;
  auto* graph = submission::GraphQuota::create(capacity, &retired,
      [](void* p) { ++*static_cast<int*>(p); });
  {
    PreparedReductionPlan paid(262147, 3, 1, 1, sizeof(float), packet_bytes, graph);
    graph->release();
    check(retired == 0 && paid.sends().size() == 12, "prepared plan lost exact source");
    check(paid.sends()[0] == ReductionSpan(87383, 131074) &&
          paid.sends()[2] == ReductionSpan(174765, 174766) &&
          paid.recvs()[2] == ReductionSpan(262147, 262147),
          "partial final packet did not preserve clipped intervals");
    for (size_t i = 0; i != paid.sends().size(); ++i) {
      const auto send = paid.sends()[i], recv = paid.recvs()[i];
      check(send.first <= send.second && send.second <= 262147 &&
            recv.first <= recv.second && recv.second <= 262147,
            "reduction interval escaped actual input");
    }
  }
  check(retired == 1, "plan vectors did not retire original source");
  size_t one_plan_capacity;
  check(submission::GraphQuota::fresh_capacity_for_extents(layout.plan_extent, one_plan_capacity),
        "single destination capacity overflow");
  auto* small = submission::GraphQuota::create(one_plan_capacity, nullptr, nullptr);
  bool refused = false;
  try { PreparedReductionPlan too_large(262147, 3, 1, 1, sizeof(float), packet_bytes, small); }
  catch (const submission::GraphQuotaError&) { refused = true; }
  check(refused && small->occupied_bytes() == 0, "plan reserve refusal retained a partial destination");
  small->release();
  check(!PreparedReductionPlan::layout(1, 0, sizeof(float), packet_bytes, layout) &&
        !PreparedReductionPlan::layout(1, 2, 0, packet_bytes, layout), "invalid geometry was admitted");
  check(!PreparedReductionPlan::layout(std::numeric_limits<size_t>::max(),
            std::numeric_limits<size_t>::max(), 1, packet_bytes, layout),
        "overflowing plan population was admitted");
#else
  check(!PreparedReductionPlan::layout(262147, 3, sizeof(float), packet_bytes, layout),
        "unknown vector reserve ABI was qualified");
#endif
}

void pool_task_source_results_and_failure() {
  ThreadPool pool(2);
  auto plain = pool.enqueue([](int x) { return x * 3 - 2; }, 11);
  check(plain.get() == 31, "ordinary pool result changed");
  int alias = 37;
  auto reference = pool.enqueue([&]() -> int& { return alias; });
  reference.get() = -19;
  check(alias == -19, "ordinary pool reference result lost identity");
  auto body = [](int x) { return x * -4 + 7; };
  threadpool_detail::TaskLayout layout{};
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106 && __cplusplus == 202002L
  check(ThreadPool::task_layout<decltype(body), int>(layout), "pool callable layout missing");
  size_t capacity;
  check(submission::GraphQuota::fresh_capacity_for_extents(
            8 * (layout.node_extent + layout.promise_extent), capacity), "pool capacity overflow");
  int retired = 0;
  auto* graph = submission::GraphQuota::create(capacity, &retired,
      [](void* p) { ++*static_cast<int*>(p); });
  auto result = pool.enqueue_with_graph(graph, body, 13);
  auto failure = pool.enqueue_with_graph(graph, []() -> int { throw std::runtime_error("pool cause"); });
  graph->release();
  check(retired == 0, "accepted result lost exact Graph source");
  check(result.get() == -45, "source pool result differs from ordinary expression");
  bool cause_preserved = false;
  try { failure.get(); } catch (const std::runtime_error& e) {
    cause_preserved = std::string(e.what()) == "pool cause";
  }
  check(cause_preserved, "pool task exception source was discarded");
  pool.resize(0); // existing join: every fulfilled task node has now retired
  check(retired == 1, "final pool task/promise source was not retired");
  auto* small = submission::GraphQuota::create(128, nullptr, nullptr);
  bool refused = false;
  try { auto unused = pool.enqueue_with_graph(small, body, 9); }
  catch (const submission::GraphQuotaError&) { refused = true; }
  check(refused && small->occupied_bytes() == 0, "pool refusal lost an unaccepted Graph prefix");
  small->release();
#else
  check(!ThreadPool::task_layout<decltype(body), int>(layout), "unknown promise ABI was qualified");
#endif
}

void socket_task_source_and_failure_retirement() {
  using namespace dist::ring;
  SocketTaskLayout layout{};
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106 && __cplusplus == 202002L
  check(socket_task_layout(layout), "actual promise/node layout missing");
  check(layout.node_bytes > 0 && layout.promise_bytes > 0 && layout.controls > 0,
        "empty socket task layout");
  size_t capacity;
  check(submission::GraphQuota::fresh_capacity_for_extents(
            4 * (layout.node_extent + layout.promise_extent), capacity),
        "socket source arena extent overflow");
  int retired = 0;
  auto* graph = submission::GraphQuota::create(capacity, &retired,
      [](void* p) { ++*static_cast<int*>(p); });
  {
    auto empty = SocketTaskQueue::prepare(nullptr, 0, graph);
    auto completed_empty = SocketTaskQueue::complete_empty(empty);
    completed_empty.get();
    check(graph->occupied_bytes() == 0, "empty socket result retained native work");
    int payload[] = {7, -11, 23};
    SocketTaskQueue queue;
    auto ready = SocketTaskQueue::prepare(payload, sizeof(payload), graph);
    check(graph->occupied_bytes() > 0, "socket task escaped its Graph source");
    auto receipt = queue.push(ready);
    check(!queue.empty(), "accepted task missing from same queue");
    graph->release();
    check(retired == 0, "accepted task lost its original allocation source");
    std::thread worker([&] {
      auto* actual = queue.front();
      check(actual && actual->size == sizeof(payload), "queue payload changed");
      auto* values = static_cast<int*>(actual->buffer);
      for (size_t i = 0; i != 3; ++i) values[i] *= -2;
      queue.complete_front();
    });
    receipt.wait();
    worker.join();
    check(payload[0] == -14 && payload[1] == 22 && payload[2] == -46,
          "shared socket task changed payload/notification ordering");
    check(queue.empty() && retired == 0, "escaped completed receipt lost Graph custody");
  }
  check(retired == 1, "final socket receipt did not retire Graph custody");
  auto* small = submission::GraphQuota::create(128, nullptr, nullptr);
  bool refused = false;
  try { auto unused = SocketTaskQueue::prepare(nullptr, 1, small); }
  catch (const submission::GraphQuotaError&) { refused = true; }
  check(refused && small->occupied_bytes() == 0,
        "refused prefix allocated outside Graph or retained an unaccepted promise");
  small->release();
  // A native callback can have no Graph TLS while its accepted caller owns
  // the exact allocation source. Capture that source, never rediscover it.
  int worker_retired = 0;
  auto* worker_graph = submission::GraphQuota::create(capacity, &worker_retired,
      [](void* p) { ++*static_cast<int*>(p); });
  std::optional<SocketFuture> escaped;
  {
    submission::GraphQuotaRef source(worker_graph);
    worker_graph->release();
    std::thread worker([source, &escaped] {
      check(submission::current_graph_quota() == nullptr, "test worker unexpectedly has Graph TLS");
      auto ready = SocketTaskQueue::prepare(nullptr, 0, source.get());
      escaped.emplace(SocketTaskQueue::complete_empty(ready));
    });
    worker.join();
  }
  check(worker_retired == 0, "worker source retired before escaped result");
  escaped.reset();
  check(worker_retired == 1, "explicit worker source was not retired");
  // An accepted receipt joins on destruction before its buffer can retire.
  SocketTaskQueue queue;
  int ordinary = 5;
  auto ready = SocketTaskQueue::prepare(&ordinary, sizeof(ordinary), nullptr);
  std::thread worker;
  {
    auto receipt = queue.push(ready);
    worker = std::thread([&] { ordinary += 17; queue.complete_front(); });
  }
  worker.join();
  check(ordinary == 22, "ordinary receipt did not keep accepted buffer alive");
  // Join all issued siblings before propagating the first exact task error.
  std::promise<void> one, two;
  PoolFutures joined(2, nullptr);
  joined.append(one.get_future()); joined.append(two.get_future());
  one.set_exception(std::make_exception_ptr(std::runtime_error("first task")));
  bool sibling_finished = false;
  std::thread sibling([&] { sibling_finished = true; two.set_value(); });
  bool first_preserved = false;
  try { joined.join(); } catch (const std::runtime_error& e) {
    first_preserved = std::string(e.what()) == "first task";
  }
  sibling.join();
  check(first_preserved && sibling_finished, "pool join lost error or retired before sibling");
#else
  check(!socket_task_layout(layout), "unknown promise ABI was qualified");
#endif
}

void retained_storage_and_exact_implementation() {
  auto implementation = std::make_shared<Probe>();
  dist::Group first(implementation), alias(implementation);
  auto child = first.split(0);
  dist::GroupStorageInventory source, split;
  check(first.storage_inventory(source) && child.storage_inventory(split),
        "actual retained implementation inventory missing");
  check(first.same_implementation(alias) && !first.same_implementation(child),
        "canonical ancestor mistaken for exact implementation");
  check(source.kind == dist::GroupStorageKind::unknown &&
            (source.unresolved & dist::group_unknown_implementation),
        "unknown native implementation was qualified from rank/size");
  check(!(source.unresolved & dist::group_inherited_owner) &&
            (split.unresolved & dist::group_inherited_owner),
        "inherited terminal owner missing from unresolved inventory");
  mlx_distributed_storage_inventory fixed{};
  check(!mlx_distributed_group_storage_inventory(&fixed, {nullptr}) &&
            !mlx_distributed_group_same_implementation({nullptr}, {nullptr}),
        "null wrappers manufactured native source evidence");
}

void family_and_split_race() {
  auto implementation = std::make_shared<Probe>();
  dist::Group root(implementation);
  auto child = root.split(0), sibling = root.split(1), grandchild = child.split(0);
  child.mark_terminal_submission();
  check(
      root.terminal_submission() && sibling.terminal_submission() &&
          grandchild.terminal_submission(),
      "split family bypassed canonical fence");
  auto rewrapped = dist::Group(grandchild.raw_group());
  check(rewrapped.terminal_submission(),
        "raw native implementation rewrap bypassed fence");
  rejected([&] { root.split(0); });
  rejected([&] { sibling.split(0); });
  auto gate = std::make_shared<SplitGate>();
  dist::Group raced(std::make_shared<Probe>(gate));
  auto result = std::async(std::launch::async, [raced] { return raced.split(0); });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!gate->entered.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  bool entered = gate->entered.load();
  raced.mark_terminal_submission();
  gate->release.store(true);
  auto raced_child = result.get();
  check(entered && raced_child.terminal_submission(),
        "concurrent split published an unfenced child");
  rejected([&] { raced_child.split(0); });
}
void cached_and_zero_operations() {
#ifdef _WIN32
  _putenv_s("MLX_HOSTFILE", "");
  _putenv_s("MLX_RANK", "");
#else
  unsetenv("MLX_HOSTFILE");
  unsetenv("MLX_RANK");
#endif
  auto group = dist::init(false, "ring");
  check(group.size() == 1, "isolated fallback must be singleton");
  auto again = dist::init(false, "ring");
  check(group.raw_group() == again.raw_group(),
        "init did not use actual cached implementation");
  dist::GroupStorageInventory inventory;
  check(group.storage_inventory(inventory) &&
            inventory.kind == dist::GroupStorageKind::empty &&
            inventory.implementation_bytes > 0 &&
            inventory.unresolved == dist::group_shared_control,
        "requested Ring was confused with actual EmptyGroup fallback");
  auto stream = new_stream(Device::cpu);
  auto input = array({1.25f, -2.5f, 3.0f});
  auto empty = array(std::initializer_list<float>{}, {0});
  auto prior = dist::all_sum(input, group, stream);
  group.mark_terminal_submission();
  auto after = dist::init(false, "ring");
  check(again.terminal_submission() && after.terminal_submission(),
        "cached aliases bypassed terminal state");
  rejected([&] { dist::all_sum(input, after, stream); });
  rejected([&] { dist::all_gather(empty, after, stream); });
  rejected([&] { dist::all_to_all_v(empty, {0}, {0}, after, stream); });
  rejected([&] {
    dist::AllReduce primitive(stream, after, dist::AllReduce::Sum);
  });
  prior.eval();
  check(prior.data<float>()[0] == 1.25f && prior.data<float>()[1] == -2.5f,
        "mark altered already accepted values");
}
void outstanding_record_retains_original_owner() {
  auto implementation = std::make_shared<Probe>();
  std::weak_ptr<Probe> weak = implementation;
  auto retired = std::make_shared<std::atomic<int>>(0);
  Scope scope;
  {
    dist::Group group(implementation);
    auto record = std::make_unique<HeldRecord>(group, retired);
    record->enter();
    record->finish(false);
    record.release();
    mlx_submission_status before{}, after{};
    check(mlx_submission_scope_query(&before, scope.value) == 0 &&
              before.activity == MLX_SUBMISSION_ACTIVITY_PENDING,
          "scope must retain unfinished record");
    group.mark_terminal_submission();
    check(mlx_submission_scope_query(&after, scope.value) == 0 &&
              before.activity == after.activity &&
              before.failed == after.failed && before.blocked == after.blocked,
          "mark changed unfinished authority");
  }
  implementation.reset();
  check(!weak.expired() && retired->load() == 0,
        "mark retired original record owner");
  check(mlx_submission_scope_seal(scope.value) == 0, "scope seal failed");
  submission::progress_records(false);
  check(submission::try_retire_records() ==
            submission::RetirementPass::complete_snapshot,
        "ordinary retirement unavailable");
  check(weak.expired() && retired->load() == 1,
        "positive retirement did not release exact owner");
}
} // namespace

int main() {
  try {
    mlx_distributed_group_mark_terminal_submission({nullptr});
    check(mlx_distributed_group_terminal_submission({nullptr}),
          "empty C handle must conservatively reject");
    outer_dispatch_type_and_source_retirement();
    ring_worker_partitions_and_storage();
    reduction_plan_geometry_source_and_intervals();
    pool_task_source_results_and_failure();
    socket_task_source_and_failure_retirement();
    retained_storage_and_exact_implementation();
    family_and_split_race();
    cached_and_zero_operations();
    outstanding_record_retains_original_owner();
    std::cout << "canonical terminal group tests passed; GroupImpl="
              << sizeof(dist::detail::GroupImpl)
              << " atomic=" << sizeof(std::atomic<bool>)
              << " root_link=" << sizeof(std::shared_ptr<dist::detail::GroupImpl>) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
