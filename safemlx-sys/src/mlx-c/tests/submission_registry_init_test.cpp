// Isolated process: every worker's first native call is scope construction.
// No ordinary Scope, Record or progress call pre-initializes the registry.
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include "mlx/c/submission.h"
#include "mlx/submission.h"

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Counts {
  std::atomic<int> retired{0};
  std::atomic<bool> wrong_thread{false};
};
struct Record final : mlx::core::submission::Record {
  explicit Record(std::shared_ptr<Counts> counts)
      : counts(counts), owner(std::this_thread::get_id()) {}
  ~Record() override {
    if (owner != std::this_thread::get_id()) counts->wrong_thread.store(true);
    ++counts->retired;
  }
  std::shared_ptr<Counts> counts;
  std::thread::id owner;
};
struct Scope {
  mlx_submission_scope value{nullptr};
  Scope() {
    check(mlx_submission_scope_new(&value) == 0 && value.ctx,
          "cold scope construction failed");
  }
  ~Scope() { mlx_submission_scope_free(value); }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};
} // namespace

int main() {
  constexpr int workers = 16;
  // Every worker and registry-owned record retains its counters, even when a
  // failure leaves the record in the immortal registry after the worker exits.
  // Release the start gate before joining if thread creation throws.
  std::vector<std::shared_ptr<Counts>> counts;
  std::atomic<int> waiting{0};
  std::vector<std::future<void>> results;
  std::promise<void> start;
  struct ReleaseStart {
    std::promise<void>& start;
    bool armed{true};
    ~ReleaseStart() { if (armed) start.set_value(); }
  } release{start};
  auto ready = start.get_future().share();
  counts.reserve(workers);
  results.reserve(workers);
  for (int i = 0; i < workers; ++i) {
    counts.push_back(std::make_shared<Counts>());
    auto count = counts.back();
    results.push_back(std::async(std::launch::async, [ready, &waiting, count] {
      ++waiting;
      ready.wait();
      Scope scope;
      mlx_submission_status status{};
      check(mlx_submission_scope_query(&status, scope.value) == 0,
            "cold scope query failed");
      check(status.activity == MLX_SUBMISSION_ACTIVITY_NONE && !status.failed &&
                !status.blocked, "construction accepted work");
      // The real shared registry owns this record and its scope until positive
      // existing progress and exact owner-thread retirement destroy the record.
      auto record = std::make_unique<Record>(count);
      record->enter();
      record->finish(false);
      record.release();
      check(mlx_submission_scope_seal(scope.value) == 0, "scope seal failed");
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (count->retired.load() == 0) {
        mlx::core::submission::progress_records(false);
        mlx::core::submission::try_retire_records();
        check(std::chrono::steady_clock::now() < deadline,
              "published registry lost a record or did not retire it");
        std::this_thread::yield();
      }
      check(mlx_submission_scope_query(&status, scope.value) == 0,
            "final scope query failed");
      check(status.activity == MLX_SUBMISSION_ACTIVITY_NONE && !status.failed &&
                !status.blocked, "empty record did not settle its actual scope");
    }));
  }
  while (waiting.load() != workers) std::this_thread::yield();
  start.set_value();
  release.armed = false;
  try {
    for (auto& result : results) result.get();
    for (auto& count : counts) {
      check(count->retired == 1 && !count->wrong_thread,
            "record did not retire exactly once on its constructing thread");
    }
    std::cout << "cold concurrent registry construction and retirement passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
