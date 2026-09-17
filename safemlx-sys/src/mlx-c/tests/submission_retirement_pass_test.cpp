// Isolated registry test: no tensors, streams, workers or completion polls are
// needed to verify the exact retirement operation. The gate below holds the
// actual registry through an existing test Record observation; production gains
// no test hook or lock-manipulation API.
#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include "mlx/c/submission.h"
#include "mlx/submission.h"

using namespace mlx::core;

namespace {
void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

submission::RetirementPass native_pass() {
  mlx_submission_retirement result = MLX_SUBMISSION_RETIREMENT_BUSY;
  check(mlx_submission_retire_completed(&result) == 0, "C retirement failed");
  if (result == MLX_SUBMISSION_RETIREMENT_COMPLETE_SNAPSHOT) {
    return submission::RetirementPass::complete_snapshot;
  }
  check(result == MLX_SUBMISSION_RETIREMENT_BUSY, "unknown C retirement result");
  return submission::RetirementPass::busy;
}

struct Counts {
  std::atomic<int> destroyed{0};
  std::atomic<int> observations{0};
  std::atomic<bool> terminal{false};
  std::atomic<bool> wrong_thread{false};
  std::atomic<bool> destructor_locked{false};
};

struct CountedRecord : submission::Record {
  explicit CountedRecord(std::shared_ptr<Counts> value)
      : counts(std::move(value)), owner(std::this_thread::get_id()) {}
  ~CountedRecord() override {
    if (owner != std::this_thread::get_id()) {
      counts->wrong_thread = true;
    }
    try {
      // There are no remaining eligible siblings in the registry: a complete
      // result proves destruction is outside its lock, including nested entry.
      if (submission::try_retire_records() !=
          submission::RetirementPass::complete_snapshot) {
        counts->destructor_locked = true;
      }
    } catch (...) {
      counts->destructor_locked = true;
    }
    ++counts->destroyed;
  }
  bool side_effects_terminal() const noexcept override {
    ++counts->observations;
    return counts->terminal.load();
  }
  std::shared_ptr<Counts> counts;
  std::thread::id owner;
};

void append(const std::shared_ptr<Counts>& counts, int n) {
  for (int i = 0; i < n; ++i) {
    auto record = std::make_unique<CountedRecord>(counts);
    record->enter();
    record->finish(false);
    record.release(); // Registry owns it until positive progress then retirement.
  }
}

void no_progress_and_unlocked_destruction() {
  auto counts = std::make_shared<Counts>();
  append(counts, 1);
  check(native_pass() == submission::RetirementPass::complete_snapshot,
        "uncontended unresolved snapshot did not complete");
  check(counts->destroyed == 0 && counts->observations == 0,
        "retirement polled or destroyed an unresolved record");
  counts->terminal = true;
  check(native_pass() == submission::RetirementPass::complete_snapshot,
        "second unresolved snapshot failed");
  check(counts->destroyed == 0 && counts->observations == 0,
        "retirement promoted terminal side effects without recorded proof");
  submission::progress_records(false);
  check(counts->observations == 1 && counts->destroyed == 0,
        "existing progress must establish proof without destruction");
  check(native_pass() == submission::RetirementPass::complete_snapshot,
        "settled snapshot failed");
  check(counts->destroyed == 1 && !counts->wrong_thread && !counts->destructor_locked,
        "settled owner was not destroyed on its thread outside the registry lock");
}

void full_entry_snapshot_exceeds_progress_batch() {
  constexpr int count = 129;
  auto counts = std::make_shared<Counts>();
  counts->terminal = true;
  append(counts, count);
  // Existing bounded progress visits at most 64 each call. This establishes
  // exact proof before the one full retirement pass under test.
  for (int i = 0; i < 3; ++i) {
    submission::progress_records(false);
  }
  check(counts->observations == count && counts->destroyed == 0,
        "bounded progress did not leave all positively settled owners retained");
  check(native_pass() == submission::RetirementPass::complete_snapshot,
        "full retirement snapshot failed");
  check(counts->destroyed == count && !counts->wrong_thread && !counts->destructor_locked,
        "retirement only inspected a bounded batch or violated destruction rules");
}

struct Gate {
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> released{release.get_future().share()};
  std::atomic<bool> opened{false};
  std::atomic<int> observations{0};
  std::atomic<int> destroyed{0};
  std::atomic<bool> wrong_thread{false};
  void open() {
    if (!opened.exchange(true)) {
      release.set_value();
    }
  }
};

struct GatedRecord : submission::Record {
  explicit GatedRecord(std::shared_ptr<Gate> value)
      : gate(std::move(value)), owner(std::this_thread::get_id()) {}
  ~GatedRecord() override {
    gate->wrong_thread = owner != std::this_thread::get_id();
    ++gate->destroyed;
  }
  bool side_effects_terminal() const noexcept override {
    // Test-only deterministic contention: progress already owns the registry
    // before invoking this existing virtual observation. No native work runs.
    if (gate->observations.fetch_add(1) == 0) {
      gate->entered.set_value();
    }
    gate->released.wait();
    return true;
  }
  std::shared_ptr<Gate> gate;
  std::thread::id owner;
};

void registry_busy_retains_current_and_foreign_owners() {
  auto current = std::make_shared<Counts>();
  current->terminal = true;
  append(current, 1);
  submission::progress_records(false);
  auto gate = std::make_shared<Gate>();
  auto entered = gate->entered.get_future();
  std::promise<void> returned;
  auto return_signal = returned.get_future();
  std::atomic<bool> watchdog_opened{false};
  std::thread worker([gate] {
    auto record = std::make_unique<GatedRecord>(gate);
    record->enter();
    record->finish(false);
    record.release();
    submission::progress_records(false);
    // Only this original owner may destroy the foreign record.
    submission::retire_records();
  });
  std::thread watchdog([&] {
    if (return_signal.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
      watchdog_opened = true;
      gate->open();
    }
  });
  // Always release/join on an assertion, exception, or failed implementation.
  struct Cleanup {
    std::shared_ptr<Gate> gate;
    std::promise<void>& returned;
    std::thread& worker;
    std::thread& watchdog;
    ~Cleanup() {
      gate->open();
      try { returned.set_value(); } catch (const std::future_error&) {}
      worker.join();
      watchdog.join();
    }
  };
  submission::RetirementPass result;
  int current_during_busy = -1;
  int foreign_during_busy = -1;
  int observations_during_busy = -1;
  {
    Cleanup cleanup{gate, returned, worker, watchdog};
    check(entered.wait_for(std::chrono::seconds(3)) == std::future_status::ready,
          "fixture did not acquire the actual registry");
    result = native_pass();
    returned.set_value();
    current_during_busy = current->destroyed.load();
    foreign_during_busy = gate->destroyed.load();
    observations_during_busy = gate->observations.load();
  }
  check(!watchdog_opened && result == submission::RetirementPass::busy,
        "retirement waited for the registry or hid contention");
  check(current_during_busy == 0 && foreign_during_busy == 0 && observations_during_busy == 1,
        "busy retirement changed ownership or progressed a record");
  check(gate->destroyed == 1 && !gate->wrong_thread,
        "foreign owner did not retire its own record");
  check(native_pass() == submission::RetirementPass::complete_snapshot,
        "positive retry after contention did not complete");
  check(current->destroyed == 1 && !current->wrong_thread && !current->destructor_locked,
        "retry lost current owner or destroyed while locked");
}
} // namespace

int main() {
  try {
    no_progress_and_unlocked_destruction();
    full_entry_snapshot_exceeds_progress_batch();
    registry_busy_retains_current_and_foreign_owners();
    std::cout << "retirement pass: no progress; full snapshot; Busy retains owners; unlocked destruction\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "retirement pass fixture failed: " << error.what() << '\n';
    return 1;
  }
}
