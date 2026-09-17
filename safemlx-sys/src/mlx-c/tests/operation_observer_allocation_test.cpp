#include "mlx/c/submission.h"
#include "mlx/failure.h"
#include "mlx/graph_quota.h"
#include "mlx/record_quota.h"
#include "mlx/submission.h"
#include "mlx/fence.h"
#include "mlx/stream.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
namespace {
std::atomic<size_t> calls{0};
std::atomic<bool> measured{false};
void count() { if (measured.load(std::memory_order_relaxed)) ++calls; }
void* allocate(size_t bytes) {
  count();
  if (auto* value = std::malloc(bytes ? bytes : 1)) return value;
  throw std::bad_alloc();
}
void* allocate(size_t bytes, size_t alignment) {
  count();
  void* value = nullptr;
  if (posix_memalign(&value, alignment, bytes ? bytes : 1) == 0) return value;
  throw std::bad_alloc();
}
size_t positive() {
  calls = 0; measured = true;
  auto* value = ::operator new(37);
  *static_cast<volatile unsigned char*>(value) = 19;
  ::operator delete(value);
  measured = false;
  return calls.load();
}
}
void* operator new(size_t size) { return allocate(size); }
void* operator new[](size_t size) { return allocate(size); }
void* operator new(size_t size, std::align_val_t alignment) { return allocate(size, size_t(alignment)); }
void* operator new[](size_t size, std::align_val_t alignment) { return allocate(size, size_t(alignment)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { std::free(p); }
int main() {
  using namespace mlx::core;
  using namespace mlx::core::submission;
  // Cold default slow-Fence setup; opaque platform allocations are not this
  // C++ new counter. A separate fast-Fence numerical run covers that branch.
  if (setenv("MLX_METAL_FAST_SYNCH", "0", 1) != 0) return 2;
  const auto fence_stream = default_stream(Device::cpu);
  { Fence warm(fence_stream); }
  unsigned retired = 0;
  auto failure = FailureCarrier::create(&retired, [](void* value) { ++*static_cast<unsigned*>(value); });
  auto* graph = GraphQuota::create(1 << 20, nullptr, nullptr);
  auto* records = RecordQuota::create(1 << 20, nullptr, nullptr);
  mlx_array source = mlx_array_new_float32(3.25f);
  void* clone_storage = nullptr;
  if (!source.ctx || mlx_array_clone_storage_new(&clone_storage) != 0) return 2;
  auto* scope = new Scope(nullptr, nullptr, records, graph);
  if (!scope->enable_scoped_observation() || scope->require_original_controls() != NativeControlFailure::none ||
      !scope->bind_failure(failure) || scope->enable_original_controls() != NativeControlFailure::none) return 2;
  const auto before = positive();
  calls = 0; measured = true;
  unsigned status = 0;
  { Fence original(fence_stream); }
  mlx_array clone{};
  status |= mlx_array_clone_storage_fill(&clone, &clone_storage, source);
  status |= mlx_array_free(clone);
  for (unsigned i = 0; i != 100; ++i) {
    mlx_submission_observer observer{};
    mlx_submission_status observation{};
    status |= mlx_submission_observer_current(&observer);
    status |= mlx_submission_observer_retain(observer);
    status |= mlx_submission_observer_query(&observation, observer);
    status |= mlx_submission_observer_progress(&observation, observer);
    status |= mlx_submission_observer_retire(observer);
    mlx_submission_observer_release(observer);
    mlx_submission_observer_release(observer);
  }
  measured = false;
  const auto actual = calls.load();
  const auto after = positive();
  const bool current = current_scope() == scope && scope->active_on_owner_thread();
  mlx_array_clone_storage_free(clone_storage);
  mlx_array_free(source);
  scope->seal(); scope->release(); graph->release(); records->release(); failure = {};
  std::printf("{\"before\":%zu,\"actual\":%zu,\"after\":%zu,\"status\":%u,\"current\":%s,\"retired\":%u}\n",
      before, actual, after, status, current ? "true" : "false", retired);
  return before > 0 && actual == 0 && after > 0 && status == 0 && current && retired == 1 ? 0 : 1;
}
