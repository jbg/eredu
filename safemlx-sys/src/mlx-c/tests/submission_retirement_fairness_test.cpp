// Isolated native registry regression, deliberately separate from numerical
// tests: the 128-record list shape is part of this deterministic fixture.
// This uses the same derived Record/destructor observations as the existing
// submission_recovery_tests.cpp SecondaryRecord fixture. It submits no work.
#include <atomic>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

#include "mlx/submission.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {
constexpr int kOwnerRecords = 64;

struct Counts {
  std::atomic<int> destroyed{0};
  std::atomic<bool> wrong_thread{false};
};

struct RetainedPayload {
  explicit RetainedPayload(std::shared_ptr<Counts> value)
      : counts(std::move(value)), owner(std::this_thread::get_id()) {}
  ~RetainedPayload() {
    if (std::this_thread::get_id() != owner) {
      counts->wrong_thread.store(true);
    }
    ++counts->destroyed;
  }
  std::shared_ptr<Counts> counts;
  std::thread::id owner;
};

struct CountedRecord : submission::Record {
  explicit CountedRecord(const std::shared_ptr<Counts>& counts)
      : payload(std::make_unique<RetainedPayload>(counts)) {}
  std::unique_ptr<RetainedPayload> payload;
};

void append_records(const std::shared_ptr<Counts>& counts) {
  for (int i = 0; i < kOwnerRecords; ++i) {
    auto record = std::make_unique<CountedRecord>(counts);
    record->enter();
    // No streams or side effects were submitted. Normal progress can establish
    // their terminal state; neither this test nor a counter fabricates it.
    record->finish(false);
    record.release(); // The native registry now owns this exact payload.
  }
}

// All created records have already finished and have no asynchronous work.
// Existing progress establishes terminal proof; retirement remains owner-local.
// Direct retirement calls are cleanup, outside the Completion::wait assertion.
struct JoinOwner {
  std::promise<void>& release;
  std::thread& worker;
  ~JoinOwner() {
    submission::progress_records(false);
    submission::progress_records(false);
    for (int i = 0; i < 4; ++i) {
      submission::retire_records();
    }
    release.set_value();
    worker.join();
  }
};
} // namespace

int main() {
  try {
    auto current = std::make_shared<Counts>();
    auto foreign = std::make_shared<Counts>();
    // No previous tests or numerical constructors may precede this isolated
    // setup: registry order is exactly current[64], foreign[64].
    append_records(current);
    int retired_by_wait = -1;
    int foreign_retired_by_wait = -1;
    {
      std::promise<void> ready;
      auto ready_signal = ready.get_future();
      std::promise<void> release;
      auto release_signal = release.get_future();
      std::thread worker([&] {
        try {
          append_records(foreign);
          ready.set_value();
        } catch (...) {
          ready.set_exception(std::current_exception());
        }
        release_signal.wait();
        // Only this original owner can retire the foreign half. The current
        // thread has neither destroyed it nor changed the ownership filter.
        for (int i = 0; i < 4; ++i) {
          submission::retire_records();
        }
      });
      JoinOwner join{release, worker};
      ready_signal.get();
      // Visit both halves without retirement, restoring exact initial order.
      submission::progress_records(false);
      submission::progress_records(false);
      if (current->destroyed != 0 || foreign->destroyed != 0) {
        throw std::runtime_error("progress unexpectedly destroyed a payload");
      }
      // A completed empty event is the existing ordinary-host retirement seam.
      // It must make progress on settled current-owner resources despite the
      // foreign-owner half. It must never reclaim that foreign half itself.
      const Completion completed;
      for (int i = 0; i < 16; ++i) {
        completed.wait();
      }
      retired_by_wait = current->destroyed.load();
      foreign_retired_by_wait = foreign->destroyed.load();
    } // Cleanup each half on its actual owner, even when the regression fails.

    const bool progress = retired_by_wait == kOwnerRecords;
    const bool ownership = foreign_retired_by_wait == 0 &&
        !current->wrong_thread && !foreign->wrong_thread;
    const bool cleanup = current->destroyed == kOwnerRecords &&
        foreign->destroyed == kOwnerRecords;
    std::cout << "completion current=" << retired_by_wait
              << " foreign=" << foreign_retired_by_wait
              << "; after owner cleanup current=" << current->destroyed.load()
              << " foreign=" << foreign->destroyed.load() << '\n';
    if (!progress || !ownership || !cleanup) {
      std::cerr << "terminal current-owner retirement did not progress safely\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fixture failed: " << error.what() << '\n';
    return 2;
  }
}
