// Deterministic generation tests; no production-counter reset or allocator
// reuse heuristic is needed to exercise the address-reuse/exhaustion boundary.
#include <array>
#include <atomic>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mlx/array.h"
#include "mlx/c/array.h"
#include "mlx/c/private/array.h"

using namespace mlx::core;

static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

static void exhaustion_never_reissues_a_generation() {
  detail::AllocationGenerationCounter sequence(UINT64_MAX - 2);
  std::array<uint64_t, 16> results{};
  std::vector<std::thread> workers;
  for (size_t i = 0; i < results.size(); ++i) {
    workers.emplace_back([&, i] { results[i] = sequence.next(); });
  }
  for (auto& worker : workers) worker.join();
  std::set<uint64_t> issued;
  size_t exhausted = 0;
  for (auto generation : results) {
    if (generation == 0) ++exhausted;
    else require(issued.insert(generation).second, "generation reused under contention");
  }
  require(issued == std::set<uint64_t>{UINT64_MAX - 2, UINT64_MAX - 1, UINT64_MAX},
          "last generations were not issued exactly once");
  require(exhausted == 13, "exhausted callers did not fail closed");
  for (int i = 0; i < 64; ++i) require(sequence.next() == 0, "exhaustion was not permanent");
}

static void reused_addresses_do_not_alias_retained_old_charges() {
  alignas(64) std::array<unsigned char, 64> memory{};
  std::set<uint64_t> retained_charges;
  size_t physical_releases = 0;
  for (int i = 0; i < 64; ++i) {
    auto owner = std::make_shared<array::Data>(
        allocator::Buffer(memory.data()),
        [&](allocator::Buffer buffer) {
          require(buffer.ptr() == memory.data(), "backing address changed");
          ++physical_releases;
        });
    const auto generation = owner->allocation_generation;
    require(generation != 0, "unexpected production generation exhaustion");
    // Old accounting identities deliberately remain alive while the exact
    // same backing address is assigned to a new physical ownership lifetime.
    require(retained_charges.insert(generation).second, "reused address collided with old charge");
    auto alias = owner;
    require(alias->allocation_generation == generation, "alias changed generation");
    array::Data moved(std::move(*owner));
    require(moved.allocation_generation == generation, "move/donation changed generation");
    require(owner->allocation_generation == 0, "moved-from owner retained an identity");
  }
  require(physical_releases == 64, "identity retention leaked physical storage");
}

static void unavailable_generation_keeps_storage_usable_but_uncertified() {
  auto value = array(3.5f);
  auto handle = mlx_array_new_(value);
  bool known = false;
  bool host = false;
  uint64_t identity = 0;
  size_t bytes = 0;
  require(mlx_array_allocation_info(&known, &host, &identity, &bytes, handle) == 0,
          "initial allocation query failed");
  require(known && identity != 0, "initial native backing not certified");
  // Inject only this allocation's exhausted-factory result. The actual global
  // counter is never reset, advanced artificially or shared with this test.
  value.data_shared_ptr()->allocation_generation = 0;
  require(mlx_array_allocation_info(&known, &host, &identity, &bytes, handle) == 0,
          "unknown generation query failed");
  require(!known && identity == 0 && bytes == 0, "exhausted generation remained certified");
  bool attached = false;
  int releases = 0;
  require(mlx_array_retain_allocation_owner(
              &attached, handle, &releases, [](void* payload) { ++*static_cast<int*>(payload); }) == 0,
          "unavailable generation attachment failed unexpectedly");
  require(!attached && releases == 0, "unavailable identity consumed attachment ownership");
  require(value.item<float>() == 3.5f, "exhausted identity damaged storage");
  require(mlx_array_free(handle) == 0, "native storage release failed");
}

int main() {
  exhaustion_never_reissues_a_generation();
  reused_addresses_do_not_alias_retained_old_charges();
  unavailable_generation_keeps_storage_usable_but_uncertified();
}
