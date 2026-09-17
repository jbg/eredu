#include "doctest/doctest.h"
#include "mlx/input_allocator.h"
#include <atomic>
#include <future>
#include <memory>
#include <thread>

namespace {
using namespace mlx::core::allocator;
struct Counts {
  std::atomic<unsigned> constructed{0}, destroyed{0}, prefixes{0}, retired{0};
  std::atomic<bool> early_retirement{false};
};
struct Owner { std::shared_ptr<Counts> counts; };
void retire_owner(void* ptr) {
  std::unique_ptr<Owner> owner(static_cast<Owner*>(ptr));
  if (owner->counts->destroyed == 0) owner->counts->early_retirement = true;
  ++owner->counts->retired;
}
struct PrefixDelete {
  std::shared_ptr<Counts> counts;
  void operator()(unsigned* data) { delete[] data; ++counts->prefixes; }
};
struct LocalAllocator : Allocator {
  std::shared_ptr<Counts> counts;
  std::unique_ptr<unsigned[], PrefixDelete> backing;
  explicit LocalAllocator(std::shared_ptr<Counts> c, bool fail = false)
      : counts(std::move(c)), backing(new unsigned[3]{3, 17, 41}, PrefixDelete{counts}) {
    ++counts->constructed;
    if (fail) throw std::bad_alloc(); // real already-allocated prefix unwinds
  }
  ~LocalAllocator() override { ++counts->destroyed; }
  Buffer malloc(size_t) override { std::abort(); }
  void free(Buffer) override { std::abort(); }
  size_t size(Buffer) const override { std::abort(); }
  bool prepare_input_runtime(PreparedInputFacts& facts) override {
    facts = {4096, SIZE_MAX, HostTransferStorageKind::cpu, 0}; return true;
  }
};
}
TEST_CASE("input allocator slot chooses one winner and retires storage before custody") {
  auto counts = std::make_shared<Counts>();
  auto loser = std::make_unique<Owner>(Owner{counts});
  {
    InputAllocatorSlot<LocalAllocator, false> slot;
    std::promise<void> constructing, release;
    auto started = constructing.get_future(); auto proceed = release.get_future();
    auto winner = std::make_unique<Owner>(Owner{counts});
    auto* owner = winner.get();
    InitializedInputAllocator first{};
    auto pending = std::async(std::launch::async, [&] {
      return slot.initialize([&](void* storage) {
        constructing.set_value(); proceed.wait();
        return new (storage) LocalAllocator(counts);
      }, {owner, retire_owner}, first);
    });
    started.wait();
    InitializedInputAllocator refused{};
    auto result = slot.initialize([&](void* storage) {
      return new (storage) LocalAllocator(counts);
    }, {loser.get(), retire_owner}, refused);
    release.set_value();
    const auto winning = pending.get();
    if (winning == InputAllocatorCause::success) winner.release();
    REQUIRE(winning == InputAllocatorCause::success);
    CHECK(result == InputAllocatorCause::busy);
    CHECK_FALSE(refused.allocator);
    CHECK(counts->constructed == 1);
    InitializedInputAllocator alias{};
    REQUIRE(slot.borrow(first.identity, alias) == InputAllocatorCause::success);
    CHECK(alias.allocator == first.allocator);
    CHECK(alias.facts.page_size == first.facts.page_size);
    InitializedInputAllocator foreign{};
    CHECK(slot.borrow(first.identity + 1, foreign) == InputAllocatorCause::identity_mismatch);
    CHECK_FALSE(foreign.allocator);
    CHECK(counts->retired == 0);
  }
  CHECK(counts->destroyed == 1);
  CHECK(counts->prefixes == 1);
  CHECK(counts->retired == 1);
  CHECK_FALSE(counts->early_retirement);
}
TEST_CASE("input allocator slot retains caller owner after a real failed constructor prefix") {
  auto counts = std::make_shared<Counts>();
  auto owner = std::make_unique<Owner>(Owner{counts});
  InputAllocatorSlot<LocalAllocator, false> slot;
  InitializedInputAllocator output{};
  auto status = slot.initialize([&](void* storage) {
    return new (storage) LocalAllocator(counts, true);
  }, {owner.get(), retire_owner}, output);
  CHECK(status == InputAllocatorCause::allocation_failed);
  CHECK_FALSE(output.allocator);
  CHECK(counts->constructed == 1);
  CHECK(counts->prefixes == 1);
  CHECK(counts->retired == 0);
  InitializedInputAllocator missing{};
  CHECK(slot.borrow(1, missing) == InputAllocatorCause::identity_mismatch);
  CHECK_FALSE(missing.allocator);
}
TEST_CASE("input allocator slot refuses an ordinary predecessor without promotion") {
  auto counts = std::make_shared<Counts>();
  auto owner = std::make_unique<Owner>(Owner{counts});
  {
    InputAllocatorSlot<LocalAllocator, false> slot;
    auto& ordinary = slot.ordinary([&](void* storage) { return new (storage) LocalAllocator(counts); });
    InitializedInputAllocator output{};
    CHECK(slot.initialize([&](void* storage) { return new (storage) LocalAllocator(counts); },
        {owner.get(), retire_owner}, output) == InputAllocatorCause::ordinary_predecessor);
    CHECK_FALSE(output.allocator);
    CHECK(counts->constructed == 1);
    CHECK(&slot.ordinary([](void*) -> LocalAllocator* { std::abort(); }) == &ordinary);
    CHECK(counts->retired == 0);
  }
  CHECK(counts->destroyed == 1);
  CHECK(counts->retired == 0);
}
