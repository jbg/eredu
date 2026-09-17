// Fresh process per case; real std::thread and actual process Scheduler.
#include "mlx/c/cpu_worker.h"
#include "mlx/c/scheduler_initialization.h"
#include "mlx/c/stream_registration.h"
#include "mlx/c/stream_copy.h"
#include "mlx/scheduler.h"
#include "mlx/mlx.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <pthread.h>

namespace {
using namespace mlx::core;
std::atomic<bool> refuse_thread{false};
std::atomic<unsigned> thread_attempts{0};
using Start = void* (*)(void*);
using Create = int (*)(pthread_t*, const pthread_attr_t*, Start, void*);
Create actual_create = nullptr; // test-only cold resolution, before observation
void check(bool value, const char* why) {
  if (!value) { std::fprintf(stderr, "%s\n", why); std::abort(); }
}
struct Owner { std::shared_ptr<std::atomic<unsigned>> drops; };
void retire(void* raw) {
  std::unique_ptr<Owner> owner(static_cast<Owner*>(raw)); ++*owner->drops;
}
struct Fixture {
  uint64_t scheduler{0};
  mlx_stream wrapper{};
  mlx_cpu_worker_target target{};
  std::shared_ptr<std::atomic<unsigned>> prefix_drops = std::make_shared<std::atomic<unsigned>>(0);
  Fixture() {
    mlx_scheduler_initialization_layout scheduler_layout{};
    check(mlx_scheduler_initialization_layout_for(&scheduler_layout) == 0, "qualified process Scheduler required");
    auto scheduler_owner = std::make_unique<Owner>(Owner{prefix_drops});
    check(mlx_scheduler_initialize(&scheduler, scheduler_layout, scheduler_owner.get(), retire) == 0,
        "actual admitted process Scheduler required"); scheduler_owner.release();
    mlx_stream_registration_layout stream_layout{};
    check(mlx_stream_registration_layout_for(&stream_layout) == 0, "qualified registration required");
    auto stream_owner = std::make_unique<Owner>(Owner{prefix_drops});
    const void* birth = stream_owner.get();
    check(mlx_stream_register_cpu(&wrapper, stream_layout, stream_owner.get(), retire) == 0,
        "actual source registration required"); stream_owner.release();
    check(mlx_cpu_worker_target_for(&target, scheduler, wrapper, birth).cause == 0,
        "actual scalar source target required");
  }
  ~Fixture() { mlx_stream_copy_free(wrapper); }
  Stream stream() const { return *static_cast<const Stream*>(wrapper.ctx); }
};
void check_missing(Stream stream) {
  scheduler::CpuStreamToken token;
  check(scheduler::prepared_cpu_stream(stream, token) == submission::NativeControlFailure::missing_worker && !token.state,
      "failed constructor published a worker");
}
void values(Fixture& f, const void* birth) {
  auto* owner = scheduler::prepared_scheduler();
  auto token = owner->prepare_cpu_stream(f.stream());
  check(token.state, "successful startup has no real worker");
  auto value = std::make_shared<std::atomic<int>>(0);
  auto actual = std::make_shared<std::atomic<bool>>(false);
  owner->enqueue(f.stream(), [value, actual, token] {
    actual->store(scheduler::current_worker() == token.state); value->store(7 * 11);
  }, true);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (scheduler::cpu_stream_progress(token).completed != 1) {
    check(std::chrono::steady_clock::now() < deadline, "actual worker timed out");
    std::this_thread::yield();
  }
  check(value->load() == 77 && actual->load() && owner->n_active_tasks() == 0,
      "genuine worker FIFO/count/value differs");
  array left({2.0f, 7.0f}), right({3.0f, 11.0f});
  auto output = add(left, right, f.stream()); eval(output);
  check(output.data<float>()[0] == 5.0f && output.data<float>()[1] == 18.0f,
      "ordinary nonzero lowering changed");
  const auto attempts = thread_attempts.load();
  auto reused = owner->prepare_cpu_stream(f.stream());
  check(reused.state == token.state && attempts == thread_attempts,
      "already prepared worker replaced its thread");
  check(mlx_cpu_worker_borrow(f.target, birth).cause == 0, "exact worker birth failed");
  auto wrong = f.target; ++wrong.stream_index;
  check(mlx_cpu_worker_borrow(wrong, birth).cause == 8, "wrong worker source accepted");
}
void run(const char* mode, const mlx_cpu_worker_layout& layout) {
  Fixture fixture;
  auto drops = std::make_shared<std::atomic<unsigned>>(0);
  auto owner = std::make_unique<Owner>(Owner{drops});
  const void* birth = nullptr;
  if (std::strcmp(mode, "ordinary") == 0) {
    scheduler::prepare_cpu_stream(fixture.stream());
    check(mlx_cpu_worker_initialize(&birth, layout, fixture.target, owner.get(), retire).cause == 6,
        "ordinary worker was retrospectively admitted");
    check(!birth && *drops == 0, "ordinary refusal lost source");
    retire(owner.release()); check(*drops == 1, "ordinary refused source did not retire once");
    return;
  }
  auto wrong = layout; ++wrong.thread_packet_bytes;
  check(mlx_cpu_worker_initialize(&birth, wrong, fixture.target, owner.get(), retire).cause == 11,
      "changed cold layout must refuse"); check_missing(fixture.stream());
  auto foreign = fixture.target; ++foreign.scheduler_identity;
  check(mlx_cpu_worker_initialize(&birth, layout, foreign, owner.get(), retire).cause == 8,
      "wrong process Scheduler must refuse");
  check(!birth && *drops == 0 && thread_attempts == 0, "pre-construction refusal mutated owner");
  if (std::strcmp(mode, "thread-failure") == 0) {
    refuse_thread = true;
    const auto result = mlx_cpu_worker_initialize(&birth, layout, fixture.target, owner.get(), retire);
    refuse_thread = false;
    check(result.cause == 12 && result.system_value == EAGAIN && result.system_category == 1,
        "actual std::thread system_error code/category not retained");
    check(thread_attempts == 1 && !birth && *drops == 0, "thread failure source/prefix differs");
    check_missing(fixture.stream());
  } else check(std::strcmp(mode, "admitted") == 0, "unknown mode");
  check(mlx_cpu_worker_initialize(&birth, layout, fixture.target, owner.get(), retire).cause == 0,
      "qualified actual worker startup failed");
  check(birth == owner.get(), "worker adopted different source"); owner.release();
  values(fixture, birth);
  check(*drops == 0 && *fixture.prefix_drops == 0, "birth account refunded after completion");
  auto rejected = std::make_unique<Owner>(Owner{drops}); const void* duplicate = nullptr;
  check(mlx_cpu_worker_initialize(&duplicate, layout, fixture.target, rejected.get(), retire).cause == 7,
      "duplicate worker not refused");
  check(!duplicate && *drops == 0, "duplicate source lost");
  retire(rejected.release()); check(*drops == 1, "only unused duplicate should retire");
}
}
// Test interposition retains std::thread's actual constructor/throw/catch path.
// Only pthread_create's result is injected. Success forwards to the real API.
extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attributes, Start start, void* argument) {
  ++thread_attempts;
  if (refuse_thread.load()) return EAGAIN;
  return actual_create(thread, attributes, start, argument);
}
int main(int argc, char** argv) {
  check(argc == 2, "fresh explicit worker startup mode required");
  actual_create = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
  check(actual_create && actual_create != &pthread_create, "actual platform create resolution failed");
  mlx_cpu_worker_layout layout;
  std::memset(&layout, 0x5a, sizeof(layout)); const auto saved = layout;
  const auto result = mlx_cpu_worker_layout_for(&layout);
  if (std::getenv("EREDU_REQUIRE_CPU_WORKER_STARTUP_QUALIFICATION"))
    check(result.cause == 0, "positive loaded worker startup qualification required");
  if (result.cause == 1) {
    check(std::memcmp(&layout, &saved, sizeof(layout)) == 0, "unknown query changed output");
    check(!scheduler::prepared_scheduler(), "unknown query initialized Scheduler");
    std::puts("CPU worker startup qualification=unknown; fixed refusal passed"); return 0;
  }
  check(result.cause == 0 && layout.failure_requests > 0 && layout.thread_implementation_bytes == 48,
      "qualified real startup recipe required");
  check(!scheduler::prepared_scheduler() && thread_attempts == 0, "query initialized worker/Scheduler");
  run(argv[1], layout);
  std::printf("CPU worker startup qualification=actual; case passed: %s\n", argv[1]);
}
