// Isolated native process: exact allocations, no-allocation destruction and
// thread-affine reentrancy. No graph-size or original-admission claim.
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

namespace probe {
struct Entry {
  uintptr_t address{0};
  size_t bytes{0};
  bool alive{false};
};
// Single-thread allocation tracing is deliberately separate from concurrent
// tests. No TLS, dynamically allocated table or callback-local container.
constexpr size_t slots = 1 << 18;
Entry entries[slots];
std::atomic<bool> tracing{false}, deny_cpp{false}, deny_c{false};
std::atomic<size_t> denied_cpp{0}, denied_c{0};
size_t live = 0, allocations = 0, attempt = 0, fail_at = 0;
size_t fail_size = 0, fail_match = 0, matches = 0;
bool failure_fired = false;
size_t after_failure_attempts = 0;
std::array<uintptr_t, 8> recent{};
size_t index(uintptr_t address) { return (address >> 4) & (slots - 1); }
Entry* find(uintptr_t address, bool insert) {
  for (size_t n = 0, i = index(address); n < slots; ++n, i = (i + 1) & (slots - 1)) {
    if (entries[i].address == address) return &entries[i];
    if (entries[i].address == 0) return insert ? &entries[i] : nullptr;
  }
  std::abort();
}
void allocated(void* p, size_t bytes) {
  if (!tracing.load(std::memory_order_relaxed)) return;
  auto& e = *find(reinterpret_cast<uintptr_t>(p), true);
  if (e.alive) std::abort();
  e = {reinterpret_cast<uintptr_t>(p), bytes, true};
  if (allocations < recent.size()) recent[allocations] = e.address;
  ++live; ++allocations;
}
void deallocated(uintptr_t p) noexcept {
  if (!tracing.load(std::memory_order_relaxed) || !p) return;
  if (auto* e = find(p, false); e && e->alive) { e->alive = false; --live; }
}
bool alive(uintptr_t p) {
  auto* e = find(p, false);
  return e && e->alive;
}
void begin(size_t nth = 0, size_t bytes = 0, size_t occurrence = 0) {
  for (auto& e : entries) e = {};
  recent.fill(0);
  live = allocations = attempt = matches = 0;
  fail_at = nth; fail_size = bytes; fail_match = occurrence;
  failure_fired = false; after_failure_attempts = 0;
  tracing = true;
}
size_t end() { tracing = false; fail_at = fail_size = fail_match = 0; return live; }
struct NoNew {
  NoNew() { denied_cpp = 0; denied_c = 0; deny_c = true; deny_cpp = true; }
  ~NoNew() { deny_cpp = false; deny_c = false; }
};
}
void* operator new(size_t n) {
  if (probe::deny_cpp.load(std::memory_order_relaxed)) {
    ++probe::denied_cpp; throw std::bad_alloc();
  }
  if (probe::tracing.load(std::memory_order_relaxed)) {
    if (probe::failure_fired) { ++probe::after_failure_attempts; throw std::bad_alloc(); }
    ++probe::attempt;
    if ((probe::fail_at && probe::attempt == probe::fail_at) ||
        (probe::fail_size && probe::fail_size == n && ++probe::matches == probe::fail_match)) {
      probe::failure_fired = true;
      throw std::bad_alloc();
    }
  }
  if (auto* p = std::malloc(n ? n : 1)) { probe::allocated(p, n); return p; }
  throw std::bad_alloc();
}
void* operator new[](size_t n) { return ::operator new(n); }
static void free_observed(void* p) noexcept {
  const auto address = reinterpret_cast<uintptr_t>(p);
  std::free(p);
  probe::deallocated(address); // Observe completion of the actual deallocation.
}
void operator delete(void* p) noexcept { free_observed(p); }
void operator delete[](void* p) noexcept { free_observed(p); }
void operator delete(void* p, size_t) noexcept { free_observed(p); }
void operator delete[](void* p, size_t) noexcept { free_observed(p); }

#if defined(__APPLE__)
// dyld interposition also sees first-use TLV malloc/calloc, which an operator
// new counter alone would miss. The disabled wrapper calls the original API.
static void* checked_malloc(size_t n) {
  if (probe::deny_c.load(std::memory_order_relaxed)) { ++probe::denied_c; return nullptr; }
  return std::malloc(n);
}
static void* checked_calloc(size_t n, size_t bytes) {
  if (probe::deny_c.load(std::memory_order_relaxed)) { ++probe::denied_c; return nullptr; }
  return std::calloc(n, bytes);
}
static void* checked_realloc(void* p, size_t n) {
  if (probe::deny_c.load(std::memory_order_relaxed)) { ++probe::denied_c; return nullptr; }
  return std::realloc(p, n);
}
__attribute__((used, section("__DATA,__interpose"))) static const struct {
  const void* replacement;
  const void* original;
} interpositions[] = {
  {reinterpret_cast<const void*>(&checked_malloc), reinterpret_cast<const void*>(&malloc)},
  {reinterpret_cast<const void*>(&checked_calloc), reinterpret_cast<const void*>(&calloc)},
  {reinterpret_cast<const void*>(&checked_realloc), reinterpret_cast<const void*>(&realloc)},
};
#endif

using namespace mlx::core;
static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct Counts {
  std::atomic<size_t> primitives{0}, data{0};
  std::atomic<uintptr_t> stack_low{UINTPTR_MAX}, stack_high{0};
};
struct PrimitiveProbe : Primitive {
  Counts* counts;
  void (*callback)(void*) noexcept{nullptr};
  void* context{nullptr};
  // Actual primitive-held graph ownership, distinct from declared input edges.
  std::optional<array> retained;
  PrimitiveProbe(Stream stream, Counts& counts) : Primitive(stream), counts(&counts) {}
  ~PrimitiveProbe() override {
    char marker;
    const auto address = reinterpret_cast<uintptr_t>(&marker);
    auto low = counts->stack_low.load();
    while (address < low && !counts->stack_low.compare_exchange_weak(low, address)) {}
    auto high = counts->stack_high.load();
    while (address > high && !counts->stack_high.compare_exchange_weak(high, address)) {}
    ++counts->primitives;
    if (callback) callback(context);
  }
  void eval_cpu(const ArrayVector&, ArrayVector&) override { std::abort(); }
  void eval_gpu(const ArrayVector&, ArrayVector&) override { std::abort(); }
  const char* name() const override { return "descriptor retirement witness"; }
};
struct OwnerProbe {
  void (*callback)(void*) noexcept;
  void* context;
  OwnerProbe(void (*callback)(void*) noexcept, void* context) : callback(callback), context(context) {}
  ~OwnerProbe() { callback(context); }
};
static array leaf(Counts& counts) {
  array a(Shape{1}, float32, nullptr, {});
  // Buffer::ptr includes backend metadata; only allocator-created buffers are
  // valid. CPU host-transfer storage is uncached and releases its actual pages
  // without allocating before publishing the retirement observation.
  auto buffer = allocator::allocator().malloc_host_transfer(
      sizeof(float), allocator::HostTransferPolicy::transfer);
  *static_cast<float*>(buffer.raw_ptr()) = 3.25f;
  a.set_data(buffer, [&counts](allocator::Buffer owned) {
    allocator::allocator().free_host_transfer(
        owned, allocator::HostTransferPolicy::transfer);
    ++counts.data;
  });
  require(a.data<float>()[0] == 3.25f, "leaf buffer did not preserve its value");
  return a;
}
static array node(Stream stream, Counts& counts, ArrayVector inputs) {
  return array(Shape{1}, float32, std::make_shared<PrimitiveProbe>(stream, counts), std::move(inputs));
}
static void no_attempts() {
  require(probe::denied_cpp == 0 && probe::denied_c == 0, "retirement attempted allocation");
}

static void deep_known_edges_keep_parent_authority(Stream stream) {
  constexpr size_t depth = 12000;
  Counts counts;
  probe::begin();
  std::optional<array> root(leaf(counts));
  std::array<uintptr_t, 4> watched{root->id(), 0, 0, 0};
  for (size_t i = 0; i < depth; ++i) {
    auto next = node(stream, counts, {*root});
    if (i == 0) watched[1] = next.id();
    if (i == depth / 2) watched[2] = next.id();
    if (i == depth - 2) watched[3] = next.id();
    *root = std::move(next);
  }
  struct Check { Counts* counts; decltype(watched)* addresses; bool called=false, correct=false; } check{&counts, &watched};
  root->retain_deferred_allocation_owner(std::make_shared<OwnerProbe>([](void* p) noexcept {
    auto& c = *static_cast<Check*>(p);
    c.called = true;
    c.correct = c.counts->data == 1 && c.counts->primitives == depth - 1;
    for (auto address : *c.addresses) c.correct &= !probe::alive(address);
  }, &check));
  { probe::NoNew denied; root.reset(); }
  const auto remaining = probe::end();
  no_attempts();
  require(check.called && check.correct, "parent authority preceded known descendants");
  require(counts.primitives == depth && counts.data == 1 && remaining == 0, "deep graph leaked storage");
  require(counts.stack_high.load() - counts.stack_low.load() < 4096,
          "known-edge destruction retained recursive stack frames");
}

static void repeated_dag_edges_preserve_external_aliases(Stream stream) {
  constexpr size_t width = 512;
  Counts counts;
  auto input = leaf(counts);
  ArrayVector branches;
  for (size_t i = 0; i < width; ++i) branches.push_back(node(stream, counts, {input, input}));
  auto survivor = branches[17];
  ArrayVector repeated = branches;
  repeated.insert(repeated.end(), branches.begin(), branches.end());
  std::optional<array> root(node(stream, counts, std::move(repeated)));
  const auto incoming = survivor.id();
  input = array(Shape{}, float32, nullptr, {});
  branches.clear();
  { probe::NoNew denied; root.reset(); }
  no_attempts();
  require(counts.primitives == width && counts.data == 0 && survivor.id() == incoming,
          "DAG release invalidated a surviving branch");
  require(survivor.inputs()[0].data<float>()[0] == 3.25f, "surviving input changed");
  survivor = array(Shape{}, float32, nullptr, {});
  require(counts.primitives == width + 1 && counts.data == 1, "last DAG alias did not retire");
}

static void sibling_replacement_and_self_aliases(Stream stream) {
  for (int mode = 0; mode < 4; ++mode) {
    Counts counts;
    auto input = leaf(counts);
    auto primitive = std::make_shared<PrimitiveProbe>(stream, counts);
    auto outputs = array::make_arrays({{1}, {1}, {1}}, {float32, float32, float32}, primitive, {input});
    primitive.reset();
    input = array(Shape{}, float32, nullptr, {});
    auto replacement = array(7.5f);
    const auto identity = replacement.id();
    { probe::NoNew denied;
      for (auto& output : outputs) {
        if (mode == 0) output = replacement;
        else if (mode == 1) { auto copy = replacement; output = std::move(copy); }
        else if (mode == 2) output.overwrite_descriptor(replacement);
      }
      if (mode == 3) { outputs.erase(outputs.begin() + 1); outputs.clear(); }
    }
    no_attempts();
    require(counts.primitives == 1 && counts.data == 1, "sibling replacement retained cycle");
    for (auto& output : outputs) require(output.id() == identity && output.data<float>()[0] == 7.5f,
                                        "replacement changed incoming value");
  }
  Counts counts;
  auto input = leaf(counts);
  auto graph = node(stream, counts, {input});
  const auto graph_id = graph.id(), input_id = input.id();
  { probe::NoNew denied;
    auto& same = graph;
    graph = same;
    graph = std::move(same);
  }
  no_attempts();
  require(graph.id() == graph_id && counts.primitives == 0, "self assignment lost graph");
  { probe::NoNew denied; graph = graph.inputs()[0]; }
  no_attempts();
  require(graph.id() == input_id && counts.primitives == 1, "internal input was not retained before replacement");
  auto alias = graph;
  { probe::NoNew denied; graph = std::move(alias); }
  no_attempts();
  require(alias.id() == 0 && graph.id() == input_id, "same-descriptor move did not consume source");
  auto moving_parent = node(stream, counts, {graph});
  { probe::NoNew denied; moving_parent = std::move(moving_parent.inputs()[0]); }
  no_attempts();
  require(moving_parent.id() == input_id && counts.primitives == 2,
          "move from internal input lost the incoming owner");
}

static void partial_sibling_publication_rolls_back(Stream stream) {
  Counts counts;
  auto input = leaf(counts);
  auto primitive = std::make_shared<PrimitiveProbe>(stream, counts);
  std::vector<Shape> shapes{{1}, {1}, {1}};
  const std::vector<Dtype> types{float32, float32, float32};
  const ArrayVector inputs{input};
  bool caught = false;
  // Each sibling publication copies the three actual array owners. Fail its
  // third allocation after two successful publications, not a synthetic throw.
  probe::begin(0, 3 * sizeof(array), 3);
  try { (void)array::make_arrays(std::move(shapes), types, primitive, inputs); }
  catch (const std::bad_alloc&) { caught = true; }
  const auto matched = probe::matches;
  const auto remaining = probe::end();
  require(caught && matched == 3 && remaining == 0 && probe::after_failure_attempts == 0, "partial sibling publication leaked or changed exception");
  require(primitive.use_count() == 1 && input.data<float>()[0] == 3.25f,
          "failure lost caller-owned inputs or primitive");
}

static void descriptor_constructor_and_control_failures(Stream stream) {
  for (size_t fail = 1; fail <= 3; ++fail) {
    Counts counts;
    auto input = leaf(counts);
    auto primitive = std::make_shared<PrimitiveProbe>(stream, counts);
    // Twenty dimensions require a real dynamic Strides allocation in init.
    Shape shape(20, 1);
    ArrayVector inputs{input};
    bool caught = false;
    probe::begin(fail);
    try { (void)array(std::move(shape), float32, primitive, std::move(inputs)); }
    catch (const std::bad_alloc&) { caught = true; }
    const auto remaining = probe::end();
    require(caught && remaining == 0 && probe::after_failure_attempts == 0, "constructor/control allocation failure leaked");
    require(primitive.use_count() == 1 && input.data<float>()[0] == 3.25f,
            "construction failure consumed caller's roots");
  }
}

static size_t actual_descriptor_request = 0, actual_control_request = 0;
static void reentrant_control_dies_before_queued_object(Stream stream) {
  Counts counts;
  auto outer_primitive = std::make_shared<PrimitiveProbe>(stream, counts);
  std::optional<array> outer(array(Shape{1}, float32, outer_primitive, {}));
  struct Check { std::optional<array> inner; uintptr_t descriptor=0, control=0; bool correct=false; } check;
  probe::begin();
  check.inner.emplace(Shape{}, float32, nullptr, ArrayVector{});
  check.descriptor = check.inner->id();
  // This constructor has no payload/vector allocation; use observed addresses,
  // not a guessed private control-block type or allocator size.
  require(probe::allocations == 2, "unexpected empty descriptor construction allocations");
  check.control = probe::recent[0] == check.descriptor ? probe::recent[1] : probe::recent[0];
  actual_descriptor_request = probe::find(check.descriptor, false)->bytes;
  actual_control_request = probe::find(check.control, false)->bytes;
  outer_primitive->callback = [](void* p) noexcept {
    auto& c = *static_cast<Check*>(p);
    c.inner.reset();
    c.correct = probe::alive(c.descriptor) && !probe::alive(c.control);
  };
  outer_primitive->context = &check;
  outer_primitive.reset();
  { probe::NoNew denied; outer.reset(); }
  const auto remaining = probe::end();
  no_attempts();
  require(check.correct && remaining == 0, "raw queued object relied on expired control block");
}

static void failed_control_construction_during_drain(Stream stream) {
  Counts counts;
  auto primitive = std::make_shared<PrimitiveProbe>(stream, counts);
  std::optional<array> outer(array(Shape{1}, float32, primitive, {}));
  struct Check { bool caught=false; } check;
  primitive->callback = [](void* p) noexcept {
    auto& c = *static_cast<Check*>(p);
    // This arbitrary user callback is allowed to allocate. Fail only the
    // control allocation after its descriptor was successfully allocated.
    probe::fail_at = probe::attempt + 2;
    try { (void)array(Shape{}, float32, nullptr, {}); }
    catch (const std::bad_alloc&) { c.caught = true; }
    probe::fail_at = 0;
  };
  primitive->context = &check;
  primitive.reset();
  probe::begin();
  outer.reset();
  const auto remaining = probe::end();
  require(check.caught && remaining == 0 && probe::after_failure_attempts == 0, "reentrant failed control lost queued descriptor");
}

static void primitive_owned_and_external_graphs_remain_distinct(Stream stream) {
  Counts counts;
  auto input = leaf(counts);
  auto primitive = std::make_shared<PrimitiveProbe>(stream, counts);
  primitive->retained = input;
  std::optional<array> outer(array(Shape{1}, float32, primitive, {input}));
  input = array(Shape{}, float32, nullptr, {});
  bool authority_released = false;
  outer->retain_deferred_allocation_owner(std::make_shared<OwnerProbe>([](void* p) noexcept {
    *static_cast<bool*>(p) = true;
  }, &authority_released));
  { probe::NoNew denied; outer.reset(); }
  no_attempts();
  require(authority_released && counts.data == 0, "test lost independent primitive ownership");
  require(primitive->retained->data<float>()[0] == 3.25f, "surviving primitive root invalidated");
  { probe::NoNew denied; primitive.reset(); }
  no_attempts();
  require(counts.data == 1 && counts.primitives == 1, "primitive-held root did not retire");
}

static void fresh_foreign_thread_and_reentrant_affinity(Stream stream) {
  Counts counts;
  auto primitive = std::make_shared<PrimitiveProbe>(stream, counts);
  auto graph = node(stream, counts, {leaf(counts)});
  primitive->retained = std::move(graph);
  struct Check { std::thread::id expected, actual; } check;
  primitive->callback = [](void* p) noexcept {
    auto& c = *static_cast<Check*>(p); c.actual = std::this_thread::get_id();
  };
  primitive->context = &check;
  std::optional<array> root(array(Shape{1}, float32, primitive, {}));
  primitive.reset();
  std::thread worker([root = std::move(root), &check]() mutable {
    check.expected = std::this_thread::get_id();
    // This thread has not touched an MLX array, TLS, or descriptor drain yet.
    { probe::NoNew denied; root.reset(); }
  });
  worker.join();
  no_attempts();
  require(check.actual == check.expected && counts.primitives == 2 && counts.data == 1,
          "fresh-thread reentrant retirement lost affinity or descendants");
}

static void simultaneous_frames_do_not_hold_lock_across_callbacks(Stream stream) {
  Counts first_counts, other_counts;
  struct Gate { std::atomic<bool> entered{false}, release{false}; } gate;
  auto primitive = std::make_shared<PrimitiveProbe>(stream, first_counts);
  primitive->callback = [](void* p) noexcept {
    auto& g = *static_cast<Gate*>(p); g.entered = true;
    while (!g.release.load()) std::this_thread::yield();
  };
  primitive->context = &gate;
  std::optional<array> first(array(Shape{1}, float32, primitive, {}));
  primitive.reset();
  std::optional<array> other(node(stream, other_counts, {leaf(other_counts)}));
  std::atomic<bool> other_done{false};
  std::thread blocked([root=std::move(first)]() mutable { root.reset(); });
  const auto first_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!gate.entered.load() && std::chrono::steady_clock::now() < first_deadline) std::this_thread::yield();
  const bool entered = gate.entered.load();
  std::thread independent([root=std::move(other), &other_done]() mutable { root.reset(); other_done=true; });
  // Bounded wait and unconditional release/join precede all assertions.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!other_done.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  const bool progressed = other_done.load();
  gate.release = true;
  blocked.join(); independent.join();
  require(entered && progressed && first_counts.primitives == 1 && other_counts.primitives == 1 && other_counts.data == 1,
          "another native frame could not drain during callback");
}

static Counts exit_counts;
struct CheckAtExit {
  ~CheckAtExit() { if (exit_counts.primitives != 1 || exit_counts.data != 1) std::abort(); }
} exit_check;
static void thread_and_static_teardown(Stream stream) {
  Counts counts;
  auto graph = node(stream, counts, {leaf(counts)});
  std::thread worker([graph=std::move(graph)]() mutable {
    thread_local std::optional<array> at_exit;
    at_exit = std::move(graph);
  });
  worker.join();
  require(counts.primitives == 1 && counts.data == 1, "thread-exit descriptor did not drain");
  // Initialize after the allocator's first use so this root retires before the
  // allocator singleton during static teardown, with exit_check still alive.
  static std::optional<array> exit_root;
  exit_root = node(stream, exit_counts, {leaf(exit_counts)});
}

int main() {
  static_assert(std::is_nothrow_move_assignable_v<array>);
  static_assert(std::is_nothrow_move_constructible_v<array>);
  const auto stream = default_stream(Device::cpu);
  // Ordinary allocator preparation precedes graph allocation tracing. Metal's
  // process-owned allocator/device-info/heap storage outlives every graph even
  // when its primitive stream is CPU. Do not warm any descriptor or graph.
  {
    auto buffer = allocator::allocator().malloc_host_transfer(
        sizeof(float), allocator::HostTransferPolicy::transfer);
    allocator::allocator().free_host_transfer(
        buffer, allocator::HostTransferPolicy::transfer);
  }
  deep_known_edges_keep_parent_authority(stream);
  repeated_dag_edges_preserve_external_aliases(stream);
  sibling_replacement_and_self_aliases(stream);
  partial_sibling_publication_rolls_back(stream);
  descriptor_constructor_and_control_failures(stream);
  reentrant_control_dies_before_queued_object(stream);
  failed_control_construction_during_drain(stream);
  primitive_owned_and_external_graphs_remain_distinct(stream);
  fresh_foreign_thread_and_reentrant_affinity(stream);
  simultaneous_frames_do_not_hold_lock_across_callbacks(stream);
  thread_and_static_teardown(stream);
  std::cout << "11 descriptor retirement cases passed (static teardown follows); "
            << "actual descriptor request=" << actual_descriptor_request
            << ", actual shared-control request=" << actual_control_request << '\n';
}
