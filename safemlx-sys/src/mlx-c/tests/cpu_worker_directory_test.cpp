#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include "mlx/mlx.h"
#include "mlx/scheduler.h"

namespace mlx::core::scheduler {
// Existing private test boundary; production gains no allocator or thread hook.
struct SchedulerControlTestAccess {
  static size_t entry_bytes() { return sizeof(Scheduler::WorkerEntry); }
  static size_t entries(const Scheduler& owner) {
    std::shared_lock lock(owner.threads_mtx_);
    return owner.threads_.size();
  }
  static const void* entry(const Scheduler& owner, Stream stream) {
    std::shared_lock lock(owner.threads_mtx_);
    return owner.threads_.find(stream.index);
  }
  static bool stopping(const Scheduler& owner) {
    return owner.stopping_.load(std::memory_order_acquire);
  }
  static bool registry_unlocked(Scheduler& owner) {
    std::unique_lock lock(owner.threads_mtx_, std::try_to_lock);
    return lock.owns_lock();
  }
};
}

namespace {
using namespace mlx::core;
using Access = scheduler::SchedulerControlTestAccess;
using Failure = submission::NativeControlFailure;
thread_local bool observe_allocations = false;
thread_local bool refuse_constructor_next = false;
std::atomic<size_t> entry_size{0};
std::atomic<unsigned> entry_attempts{0}, freed_entries{0}, constructor_refusals{0};
std::atomic<bool> refuse_entry{false}, refuse_constructor{false};
std::array<std::atomic<void*>, 16> entry_addresses{};
std::atomic<scheduler::Scheduler*> observed_owner{nullptr};
std::atomic<bool> free_unlocked{true};

void require(bool condition, const char* message) {
  if (!condition) { std::fprintf(stderr, "%s\n", message); std::abort(); }
}
template<class F> void await(F&& ready, const char* message) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!ready()) {
    require(std::chrono::steady_clock::now() < end, message);
    std::this_thread::yield();
  }
}
struct Observe {
  Observe() { observe_allocations = true; }
  ~Observe() { observe_allocations = false; refuse_constructor_next = false; }
};
void observe_free(void* pointer) noexcept {
  if (!pointer) return;
  for (auto& address : entry_addresses) {
    void* expected = pointer;
    if (address.compare_exchange_strong(expected, nullptr)) {
      if (auto* owner = observed_owner.load())
        if (!Access::registry_unlocked(*owner)) free_unlocked = false;
      ++freed_entries;
      break;
    }
  }
}
bool tracked(const void* pointer) {
  for (const auto& address : entry_addresses) if (address.load() == pointer) return true;
  return false;
}
void watch(scheduler::Scheduler& owner) {
  observed_owner = &owner;
  entry_size = Access::entry_bytes();
}

void refusal_case() {
  const Stream stream = new_thread_unsafe_stream(Device::cpu);
  scheduler::CpuStreamToken token;
  {
    scheduler::Scheduler owner;
    watch(owner);
    refuse_entry = true;
    bool rejected = false;
    try { Observe observe; owner.prepare_cpu_stream(stream); }
    catch (const std::bad_alloc&) { rejected = true; }
    require(rejected && entry_attempts == 1, "real entry refusal absent");
    require(Access::entries(owner) == 0, "refusal published a worker");
    require(owner.prepared_cpu_stream(stream, token) == Failure::missing_worker && !token.state,
            "refusal manufactured a prepared token");
    refuse_entry = false;
    refuse_constructor = true;
    rejected = false;
    try { Observe observe; owner.prepare_cpu_stream(stream); }
    catch (const std::bad_alloc&) { rejected = true; }
    refuse_constructor = false;
    require(rejected && constructor_refusals == 1, "constructor allocation refusal absent");
    require(freed_entries == 1 && free_unlocked, "failed entry did not retire outside directory loan");
    require(Access::entries(owner) == 0, "constructor failure published a worker");
    { Observe observe; token = owner.prepare_cpu_stream(stream); }
    require(token.state && tracked(Access::entry(owner, stream)), "retry did not publish actual tracked entry");
    const auto before = entry_attempts.load();
    refuse_entry = true;
    {
      Observe observe;
      require(owner.prepare_cpu_stream(stream).state == token.state, "existing lookup replaced worker");
      scheduler::CpuStreamToken borrowed;
      require(owner.prepared_cpu_stream(stream, borrowed) == Failure::none && borrowed.state == token.state,
              "prepared lookup changed identity");
    }
    refuse_entry = false;
    require(entry_attempts == before, "existing lookup allocated an entry");
    std::atomic<int> result{0};
    owner.enqueue(stream, [&] { result = 3 * 7 + 5; }, true);
    await([&] { return scheduler::cpu_stream_progress(token).completed == 1; }, "retry work did not complete");
    require(result == 26 && owner.n_active_tasks() == 0, "retry FIFO/count result differs");
  }
  observed_owner = nullptr;
  require(freed_entries == 2 && free_unlocked, "successful worker entry did not retire after join");
  std::puts("CPU worker directory allocation and constructor refusal passed");
}

void race_and_values_case() {
  const Stream stream = new_thread_unsafe_stream(Device::cpu);
  constexpr size_t count = 4;
  std::array<scheduler::CpuStreamToken, count> tokens{};
  std::array<std::thread, count> callers;
  std::atomic<unsigned> ready{0};
  std::atomic<bool> go{false};
  {
    scheduler::Scheduler owner;
    watch(owner);
    // Another preparer may hold the mutex after this candidate unlocks. Counts
    // still observe every race loser; probe lock state only after callers join.
    observed_owner = nullptr;
    for (size_t i = 0; i < count; ++i) callers[i] = std::thread([&, i] {
      ++ready;
      while (!go.load()) std::this_thread::yield();
      Observe observe;
      tokens[i] = owner.prepare_cpu_stream(stream);
    });
    await([&] { return ready == count; }, "preparation callers did not start");
    go = true;
    for (auto& caller : callers) caller.join();
    observed_owner = &owner;
    require(Access::entries(owner) == 1, "same-stream race published multiple workers");
    require(tokens[0].state && tracked(Access::entry(owner, stream)), "actual winning entry untracked");
    for (const auto& token : tokens) require(token.state == tokens[0].state, "race changed worker identity");
    std::atomic<int> value{0};
    std::atomic<bool> actual_worker{false};
    owner.enqueue(stream, [&] {
      actual_worker = scheduler::current_worker() == tokens[0].state;
      value = 5;
    }, true);
    owner.enqueue(stream, [&] { value = value * 7; }, true);
    owner.enqueue(stream, [&] { value = value.load() + 11; }, true);
    await([&] { return scheduler::cpu_stream_progress(tokens[0]).completed == 3; }, "race worker did not drain");
    require(actual_worker && value == 46 && owner.n_active_tasks() == 0, "FIFO/worker/count behavior changed");
  }
  observed_owner = nullptr;
  require(freed_entries == entry_attempts && free_unlocked, "winner or empty race candidate leaked/retired locked");
  // Ordinary real MLX CPU lowering uses the same changed process directory.
  auto output = multiply(add(array({3.f, 5.f}), array({2.f, 7.f}), stream),
                         array({4.f, 2.f}), stream);
  eval(output);
  require(output.data<float>()[0] == 20.f && output.data<float>()[1] == 24.f,
          "real CPU lowering values differ");
  auto first = scheduler::prepare_cpu_stream(stream);
  require(first.state == scheduler::prepare_cpu_stream(stream).state, "process worker lookup changed identity");
  std::puts("CPU worker directory racing reuse and numerical values passed");
}

void teardown_case() {
  const Stream stream = new_thread_unsafe_stream(Device::cpu);
  struct State {
    std::atomic<bool> entered{false}, release{false}, callback_unlocked{false};
    std::atomic<unsigned> retired{0}, completed{0};
    std::atomic<bool> reentry_refused{false};
  };
  auto state = std::make_shared<State>();
  std::thread release;
  {
    scheduler::Scheduler owner;
    watch(owner);
    // This guard is after owner: a constructor/assertion unwind cannot leave
    // its worker parked while the earlier owner destructor tries to join.
    struct Release {
      std::shared_ptr<State> state;
      ~Release() { if (state) state->release = true; }
    } unblock{state};
    scheduler::CpuStreamToken token;
    { Observe observe; token = owner.prepare_cpu_stream(stream); }
    auto custody = std::shared_ptr<int>(new int(23), [state, &owner](int* pointer) {
      delete pointer;
      state->callback_unlocked = Access::registry_unlocked(owner);
      ++state->retired;
    });
    owner.enqueue(stream, [state, custody, &owner, stream] {
      state->entered = true;
      while (!state->release.load()) std::this_thread::yield();
      try { owner.prepare_cpu_stream(stream); }
      catch (const std::runtime_error& error) {
        state->reentry_refused = std::strcmp(error.what(),
            "Cannot enqueue work after CPU stream is stopped or blocked.") == 0;
      }
      ++state->completed;
    }, true);
    owner.enqueue(stream, [state, custody] { ++state->completed; }, true);
    custody.reset();
    await([&] { return state->entered.load(); }, "teardown worker did not start");
    release = std::thread([state, &owner] {
      while (!Access::stopping(owner)) std::this_thread::yield();
      state->release = true;
    });
    // The normal guard must let Scheduler set stopping_ before releasing work.
    unblock.state.reset();
  }
  release.join();
  observed_owner = nullptr;
  require(state->completed == 2 && state->retired == 1 && state->reentry_refused,
          "teardown lost FIFO captures or accepted reentrant preparation");
  require(state->callback_unlocked && free_unlocked && freed_entries == 1,
          "teardown callback/entry return occurred under directory loan");
  std::puts("CPU worker directory terminal FIFO retirement passed");
}
}

// Forwarding test allocator; only the invoking preparation thread observes its
// exact private WorkerEntry size. No production hook or synthetic directory reset.
void* operator new(size_t size) {
  if (observe_allocations && refuse_constructor_next) {
    refuse_constructor_next = false;
    ++constructor_refusals;
    throw std::bad_alloc();
  }
  const bool entry = observe_allocations && size == entry_size.load();
  if (entry) {
    ++entry_attempts;
    if (refuse_entry.load()) throw std::bad_alloc();
  }
  void* result = std::malloc(size ? size : 1);
  if (!result) throw std::bad_alloc();
  if (entry) {
    bool stored = false;
    for (auto& address : entry_addresses) {
      void* empty = nullptr;
      if (address.compare_exchange_strong(empty, result)) { stored = true; break; }
    }
    require(stored, "test entry observation slots exhausted");
    refuse_constructor_next = refuse_constructor.load();
  }
  return result;
}
void operator delete(void* pointer) noexcept { observe_free(pointer); std::free(pointer); }
void operator delete(void* pointer, size_t) noexcept { observe_free(pointer); std::free(pointer); }

int main(int argc, char** argv) {
  require(argc == 2, "expected refusal, race, or teardown");
  if (std::strcmp(argv[1], "refusal") == 0) refusal_case();
  else if (std::strcmp(argv[1], "race") == 0) race_and_values_case();
  else if (std::strcmp(argv[1], "teardown") == 0) teardown_case();
  else require(false, "unknown worker case");
}
