// Fresh process per mode: the actual singleton has no reset/test replacement.
// These are native owner mechanisms. Real Pool comparisons are backend tests;
// ordinary stream/worker/kernel fixture setup is outside the Scheduler recipe.
#include "mlx/c/scheduler_initialization.h"
#include "mlx/scheduler.h"
#include "mlx/scheduler_initialization.h"
#include "mlx/utils.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#ifdef MLX_C_PATCH_TEST_METAL
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#endif
using namespace mlx::core;
namespace {
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template <class Predicate>
void await(Predicate ready) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (!ready()) {
    check(std::chrono::steady_clock::now() < deadline, "bounded callback/worker wait timed out");
    std::this_thread::yield();
  }
}
struct Owner { std::shared_ptr<std::atomic<unsigned>> retired; };
void retire(void* raw) {
  std::unique_ptr<Owner> owner(static_cast<Owner*>(raw));
  owner->retired->fetch_add(1);
}
struct Gate {
  std::atomic<bool> started{false}, release{false}, tail{false};
  std::atomic<int> value{0};
};
struct Release {
  std::shared_ptr<Gate> gate;
  ~Release() { gate->release.store(true); }
};
struct Initialized {
  std::shared_ptr<std::atomic<unsigned>> retired;
  uint64_t identity;
  scheduler::Scheduler* actual;
};
Initialized initialize(const mlx_scheduler_initialization_layout& layout) {
  auto retired = std::make_shared<std::atomic<unsigned>>(0);
  auto owner = std::make_unique<Owner>(Owner{retired});
  auto wrong = layout; ++wrong.controls;
  uint64_t identity = 0;
  check(mlx_scheduler_initialize(&identity, wrong, owner.get(), retire) == 11,
      "exact stale layout must refuse before construction");
  check(!identity && !scheduler::prepared_scheduler() && *retired == 0,
      "layout refusal preserves unpublished owner");
  check(mlx_scheduler_initialize(&identity, layout, owner.get(), retire) == 0,
      "actual Scheduler initialization");
  owner.release();
  auto* actual = scheduler::prepared_scheduler();
  check(actual && identity && mlx_scheduler_initialized_borrow(identity) == 0,
      "exact published Scheduler borrow");
  check(mlx_scheduler_initialized_borrow(identity + 1) == 8, "foreign identity");
  check(&scheduler::scheduler() == actual, "ordinary getter must share admitted winner");
  auto duplicate = std::make_unique<Owner>(Owner{retired});
  uint64_t second = 0;
  check(mlx_scheduler_initialize(&second, layout, duplicate.get(), retire) == 7,
      "duplicate admitted constructor refused");
  check(!second && *retired == 0, "duplicate does not retire either source");
  retire(duplicate.release());
  check(*retired == 1, "only rejected owner retires");
  return {std::move(retired), identity, actual};
}
void query(const mlx_scheduler_initialization_layout& layout) {
  check(layout.object_bytes == sizeof(scheduler::Scheduler) &&
      layout.object_alignment == alignof(scheduler::Scheduler) && layout.controls > 0,
      "actual Scheduler layout");
#if MLX_SCHEDULER_CONSTANT_STORAGE
  std::array<unsigned char, 68> changed;
  std::memcpy(changed.data(), scheduler::detail::shared_mutex_constructor, changed.size());
  check(scheduler::detail::matches_shared_mutex_constructor(changed.data(), changed.size()),
      "exact matcher baseline");
  changed[0] ^= 1;
  check(!scheduler::detail::matches_shared_mutex_constructor(changed.data(), changed.size()),
      "different implementation cannot qualify");
  check(!scheduler::detail::matches_shared_mutex_constructor(changed.data(), 67),
      "short readable prefix must refuse");
#endif
  // The quote above ran on this thread. A distinct first actual identity call
  // must still claim the identity: no quote may eagerly initialize it for us.
  bool child_first = false;
  std::thread child([&] { child_first = is_main_thread(); }); child.join();
  check(child_first && !is_main_thread(), "query initialized native first-caller identity");
  check(!scheduler::prepared_scheduler(), "query initialized Scheduler");
}
void admitted_cpu(const mlx_scheduler_initialization_layout& layout) {
  const auto owner = initialize(layout);
  check(is_main_thread(), "constructor preserves actual first-caller thread");
  {
    scheduler::Scheduler local;
    check(local.n_active_tasks() == 0 && scheduler::prepared_scheduler() == owner.actual,
        "public stack Scheduler does not replace process owner");
  }
  auto stream = new_stream(Device::cpu);
  auto gate = std::make_shared<Gate>();
  Release release{gate}; // release before any asynchronous owner can be destroyed
  scheduler::enqueue_counted(stream, [gate] {
    gate->started.store(true);
    while (!gate->release.load()) std::this_thread::yield();
    gate->value.store(7 * 11);
    gate->tail.store(true);
  });
  await([&] { return gate->started.load(); });
  check(owner.actual->n_active_tasks() == 1 && *owner.retired == 1,
      "accepted worker must retain admitted Scheduler");
  gate->release.store(true);
  synchronize(stream);
  check(gate->tail.load() && gate->value.load() == 77, "nonzero counted worker value");
  check(owner.actual->n_active_tasks() == 0 && *owner.retired == 1,
      "worker completion cannot refund permanent Scheduler owner");
  check(mlx_scheduler_initialized_borrow(owner.identity) == 0, "post-completion same owner");
}
void ordinary(const mlx_scheduler_initialization_layout& layout) {
  auto* prior = &scheduler::scheduler();
  auto retired = std::make_shared<std::atomic<unsigned>>(0);
  auto owner = std::make_unique<Owner>(Owner{retired});
  uint64_t identity = 0;
  check(mlx_scheduler_initialize(&identity, layout, owner.get(), retire) == 6,
      "ordinary predecessor must refuse without promotion");
  check(!identity && *retired == 0 && scheduler::prepared_scheduler() == prior,
      "ordinary refusal retains exact owner and original singleton");
  retire(owner.release()); check(*retired == 1, "refused owner retires once");
}
void race(const mlx_scheduler_initialization_layout& layout) {
  constexpr size_t count = 4;
  std::array<std::shared_ptr<std::atomic<unsigned>>, count> retired;
  std::array<std::unique_ptr<Owner>, count> owners;
  std::array<unsigned, count> status{};
  std::array<uint64_t, count> identity{};
  std::array<bool, count> first{};
  for (size_t i = 0; i < count; ++i) {
    retired[i] = std::make_shared<std::atomic<unsigned>>(0);
    owners[i] = std::make_unique<Owner>(Owner{retired[i]});
  }
  std::atomic<unsigned> ready{0};
  auto gate = std::make_shared<Gate>();
  std::array<std::jthread, count> workers;
  Release release{gate}; // also releases a partially constructed thread prefix
  for (size_t i = 0; i < count; ++i) {
    workers[i] = std::jthread([&, i, gate] {
      ++ready;
      while (!gate->release.load()) std::this_thread::yield();
      status[i] = mlx_scheduler_initialize(&identity[i], layout, owners[i].get(), retire);
      if (status[i] == 0) { owners[i].release(); first[i] = is_main_thread(); }
    });
  }
  await([&] { return ready.load() == count; }); gate->release.store(true);
  for (auto& worker : workers) worker.join();
  unsigned winners = 0;
  for (size_t i = 0; i < count; ++i) {
    if (status[i] == 0) {
      ++winners;
      check(identity[i] && first[i] && !owners[i] && *retired[i] == 0,
          "exact winning custody and first-caller identity");
      check(mlx_scheduler_initialized_borrow(identity[i]) == 0, "winning identity borrow");
    } else {
      check(status[i] == 3 || status[i] == 7, "loser is Busy or already initialized");
      check(!identity[i] && owners[i] && *retired[i] == 0, "losing owner remains intact");
      retire(owners[i].release()); check(*retired[i] == 1, "losing source retires once");
    }
  }
  check(winners == 1 && !is_main_thread(), "one winning constructor from real worker race");
}
#ifdef MLX_C_PATCH_TEST_METAL
void metal_counted(const mlx_scheduler_initialization_layout& layout) {
  const auto owner = initialize(layout);
  auto stream = new_stream(Device::gpu); // ordinary fixture Device/encoder/JIT setup
  auto gate = std::make_shared<Gate>();
  array left({2.f, 7.f}), right({3.f, 11.f});
  auto output = add(left, right, stream);
  // Execute the existing GPU primitive lowerer. The component under test is
  // its real command-buffer callback's Scheduler owner, not a model-fit claim.
  gpu::eval(output);
  auto encoder = metal::get_command_encoder_owner(stream);
  auto counted = std::make_shared<metal::CountedCompletion>(owner.actual);
  Release release{gate}; // releases before encoder/array owners on every unwind
  encoder->end_encoding();
  encoder->add_completed_handler([gate](MTL::CommandBuffer*) {
    gate->started.store(true);
    while (!gate->release.load()) std::this_thread::yield();
  });
  encoder->add_completed_handler([gate, counted](MTL::CommandBuffer*) {
    counted->complete(); gate->tail.store(true);
  });
  owner.actual->notify_new_task(stream); counted->counted.store(true);
  try { encoder->commit(); } catch (...) { counted->complete(); throw; }
  await([&] { return gate->started.load(); });
  check(owner.actual->n_active_tasks() == 1 && *owner.retired == 1,
      "actual Metal callback retains admitted Scheduler while delayed");
  gate->release.store(true);
  await([&] { return gate->tail.load(); });
  gpu::synchronize(stream);
  check(output.data<float>()[0] == 5.f && output.data<float>()[1] == 18.f,
      "nonzero real GPU result");
  check(owner.actual->n_active_tasks() == 0, "actual counted callback completes once");
  counted->complete();
  check(owner.actual->n_active_tasks() == 0 && *owner.retired == 1,
      "duplicate completion preserves count and permanent birth");
  check(scheduler::prepared_scheduler() == owner.actual &&
      mlx_scheduler_initialized_borrow(owner.identity) == 0, "same Scheduler after GPU completion");
}
#endif
}
int main(int argc, char** argv) {
  try {
    check(argc == 2, "explicit isolated Scheduler mode required");
    const auto* mode = argv[1];
    mlx_scheduler_initialization_static_layout fixed{};
    mlx_scheduler_initialization_static_layout_for(&fixed);
    check(!scheduler::prepared_scheduler(), "fresh process required");
    mlx_scheduler_initialization_layout layout{11, 13, 17};
    const auto status = mlx_scheduler_initialization_layout_for(&layout);
    if (std::getenv("EREDU_REQUIRE_SCHEDULER_INITIALIZATION_QUALIFICATION"))
      check(fixed.qualified && status == 0, "positive actual loaded Scheduler qualification required");
    if (status == 1) {
      check(layout.object_bytes == 11 && layout.object_alignment == 13 && layout.controls == 17,
          "unknown output remains intact");
      auto retired = std::make_shared<std::atomic<unsigned>>(0);
      auto owner = std::make_unique<Owner>(Owner{retired}); uint64_t identity = 0;
      // Supply valid C transport extent while retaining a deliberately inert
      // recipe: unqualified native code must refuse before layout/constructor.
      layout.controls = SIZE_MAX;
      check(mlx_scheduler_initialize(&identity, layout, owner.get(), retire) == 1,
          "unknown constructor refusal");
      check(!identity && !scheduler::prepared_scheduler() && *retired == 0,
          "unknown source custody");
      retire(owner.release()); check(*retired == 1, "unknown owner release");
      std::cout << "Scheduler qualification=unknown; fixed refusal passed\n"; return 0;
    }
    check(status == 0 && fixed.qualified, "unexpected Scheduler layout status");
    if (std::strcmp(mode, "query") == 0) query(layout);
    else if (std::strcmp(mode, "admitted") == 0) admitted_cpu(layout);
    else if (std::strcmp(mode, "ordinary") == 0) ordinary(layout);
    else if (std::strcmp(mode, "race") == 0) race(layout);
#ifdef MLX_C_PATCH_TEST_METAL
    else if (std::strcmp(mode, "metal-counted") == 0) metal_counted(layout);
#endif
    else throw std::runtime_error("unknown Scheduler mode");
    std::cout << "Scheduler qualification=actual; case passed: " << mode << '\n'; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
