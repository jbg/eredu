#include "mlx/submission_identity.h"
#include <algorithm>
#include <limits>

TEST_CASE("submission owner sequence is unique across concurrent fresh caches") {
  std::atomic<uint64_t> sequence{1};
  std::array<std::array<uint64_t, 64>, 8> values{};
  std::array<std::thread, 8> workers;
  for (size_t worker = 0; worker < workers.size(); ++worker) {
    workers[worker] = std::thread([&, worker] {
      for (auto& value : values[worker]) {
        value = submission::detail::claim_owner_identity(sequence, value);
        const auto same = submission::detail::claim_owner_identity(sequence, value);
        CHECK(same == value);
      }
    });
  }
  for (auto& worker : workers) worker.join();
  std::array<uint64_t, 512> ordered{};
  size_t index = 0;
  for (const auto& worker : values)
    for (auto value : worker) ordered[index++] = value;
  std::sort(ordered.begin(), ordered.end());
  for (size_t i = 0; i < ordered.size(); ++i) CHECK(ordered[i] == i + 1);
  CHECK(sequence.load() == 513);
}

TEST_CASE("submission owner exhaustion is sticky and preserves existing identities") {
  constexpr auto maximum = std::numeric_limits<uint64_t>::max();
  std::atomic<uint64_t> sequence{maximum - 1};
  uint64_t last = 0;
  CHECK(submission::detail::claim_owner_identity(sequence, last) == maximum - 1);
  CHECK(sequence.load() == maximum);
  for (int attempt = 0; attempt < 8; ++attempt) {
    uint64_t fresh = 0;
    CHECK_THROWS_AS(submission::detail::claim_owner_identity(sequence, fresh),
        submission::detail::OwnerIdentityExhausted);
    CHECK(fresh == 0);
    CHECK(sequence.load() == maximum);
  }
  CHECK(submission::detail::claim_owner_identity(sequence, last) == maximum - 1);
  std::atomic<uint64_t> unavailable{0};
  uint64_t fresh = 0;
  CHECK_THROWS_AS(submission::detail::claim_owner_identity(unavailable, fresh),
      submission::detail::OwnerIdentityExhausted);
  CHECK(unavailable.load() == 0);
}

TEST_CASE("submission thread identity does not reuse a departed owner") {
  submission::Scope* old = nullptr;
  std::thread first([&] {
    old = new submission::Scope;
    CHECK(old->on_owner_thread());
    old->seal();
  });
  first.join();
  REQUIRE(old != nullptr);
  CHECK_FALSE(old->on_owner_thread());
  std::thread second([&] {
    Scope current;
    CHECK(current.value->on_owner_thread());
    CHECK_FALSE(old->on_owner_thread());
  });
  second.join();
  CHECK_FALSE(old->on_owner_thread());
  old->release();
}

namespace {
struct RuntimeOwnerRecord final : submission::Record {
  std::shared_ptr<std::atomic<size_t>> retired;
  RuntimeOwnerRecord(Allocation allocation,
      const std::shared_ptr<std::atomic<size_t>>& count)
      : Record(allocation), retired(count) {}
  ~RuntimeOwnerRecord() override { ++*retired; }
  bool scoped_observation_supported() const noexcept override { return true; }
};
}
TEST_CASE("submission registry preserves concurrent record retirement and scope affinity") {
  auto retired = std::make_shared<std::atomic<size_t>>(0);
  std::array<std::thread, 8> workers;
  for (auto& worker : workers) {
    worker = std::thread([&] {
      Scope current;
      CHECK(current.value->on_owner_thread());
      auto record = submission::Record::create<RuntimeOwnerRecord>(retired);
      record->enter();
      record->finish(false);
      record.release();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      auto status = submission::RetirementPass::busy;
      do {
        submission::progress_records(false);
        status = submission::try_retire_records();
        if (status == submission::RetirementPass::complete_snapshot &&
            current.value->query().activity != submission::Activity::pending)
          break;
        std::this_thread::yield();
      } while (std::chrono::steady_clock::now() < deadline);
      CHECK(status == submission::RetirementPass::complete_snapshot);
      CHECK(current.value->query().activity == submission::Activity::none);
    });
  }
  for (auto& worker : workers) worker.join();
  CHECK(retired->load() == workers.size());
}
