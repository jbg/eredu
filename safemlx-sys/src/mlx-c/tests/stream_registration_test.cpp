#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include "mlx/mlx.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/scheduler.h"
#include "mlx/stream_registration.h"
#include "mlx/c/stream_registration.h"
#include "mlx/c/stream_copy.h"

namespace {
std::atomic<size_t> watched_size{0};
std::atomic<void*> watched_entry{nullptr};
std::atomic<unsigned> entry_attempts{0};
std::atomic<bool> refuse_entry{false};
std::atomic<bool> entry_freed{false};
std::atomic<unsigned> retired{0};
bool expect_retirement = false;
void require(bool value, const char* message) {
  if (!value) { std::fprintf(stderr, "%s\n", message); std::abort(); }
}
void observe_free(void* pointer) noexcept {
  if (pointer && pointer == watched_entry.load()) entry_freed.store(true);
}
struct Token { unsigned marker{23}; };
void retire(void* raw) {
  auto* token = static_cast<Token*>(raw);
  require(token->marker == 23, "wrong constructor owner");
  require(entry_freed.load(), "source retired before actual entry deallocation");
  require(++retired == 1, "source retired more than once");
  delete token;
}
void after_registry() {
  require(!expect_retirement || retired == 1, "process registration did not retire");
  if (expect_retirement) std::puts("stream registration physical retirement passed");
}
}

// Test-only forwarding allocator. A matching request is the actual production
// entry allocation; refusal is injected allocation-entry failure, not claimed
// platform exhaustion. Every other allocation keeps the normal malloc route.
void* operator new(size_t size) {
  const bool watched = watched_size.load() && size == watched_size.load();
  if (watched) { ++entry_attempts; if (refuse_entry.load()) throw std::bad_alloc(); }
  auto* result = std::malloc(size ? size : 1);
  if (!result) throw std::bad_alloc();
  if (watched) watched_entry.store(result);
  return result;
}
void operator delete(void* pointer) noexcept { observe_free(pointer); std::free(pointer); }
void operator delete(void* pointer, size_t) noexcept { observe_free(pointer); std::free(pointer); }

int main(int argc, char** argv) {
  require(argc == 2, "expected ownership or refusal");
  const bool ownership = std::strcmp(argv[1], "ownership") == 0;
  require(ownership || std::strcmp(argv[1], "refusal") == 0, "unknown mode");
  mlx_stream_registration_layout layout{};
  const auto status = mlx_stream_registration_layout_for(&layout);
  if (std::getenv("EREDU_REQUIRE_STREAM_REGISTRATION_QUALIFICATION"))
    require(status == 0, "required native qualification is absent");
  if (status == 1) {
    mlx_stream_registration_layout sentinel{11, 22, 33, 44, 55};
    const auto before = sentinel;
    require(mlx_stream_registration_layout_for(&sentinel) == 1, "wrong unknown cause");
    require(std::memcmp(&sentinel, &before, sizeof(before)) == 0, "unknown changed output");
    std::puts("stream registration unknown-layout contract passed");
    return 0;
  }
  require(status == 0 && layout.object_bytes > layout.wrapper_bytes, "invalid layout");
  require(mlx::core::scheduler::prepared_scheduler() == nullptr, "query initialized Scheduler");
  // Register before any stream constructor: this callback runs after the actual
  // function-static registry destructor. No reset/destructor test hook exists.
  require(std::atexit(after_registry) == 0, "atexit registration failed");
  if (!ownership) {
    auto* owner = new Token;
    mlx_stream output{nullptr};
    auto changed = layout;
    ++changed.object_bytes;
    watched_size = layout.object_bytes;
    require(mlx_stream_register_cpu(&output, changed, owner, retire) == 11, "wrong layout refusal");
    require(!output.ctx && entry_attempts == 0 && retired == 0, "layout refusal mutated ownership");
    refuse_entry = true;
    require(mlx_stream_register_cpu(&output, layout, owner, retire) == 4, "wrong allocation refusal");
    watched_size = 0;
    require(!output.ctx && entry_attempts == 1 && retired == 0, "allocation refusal adopted owner");
    require(mlx::core::get_streams().empty(), "failed entry consumed stream index");
    delete owner;
    std::puts("stream registration allocation refusal passed");
    return 0;
  }
  using namespace mlx::core;
  const auto ordinary = new_thread_unsafe_stream(Device::cpu);
  const auto local = new_stream(Device::cpu);
  const auto before = get_streams();
  auto* owner = new Token;
  mlx_stream output{nullptr};
  watched_size = layout.object_bytes;
  require(mlx_stream_register_cpu(&output, layout, owner, retire) == 0, "registration failed");
  watched_size = 0;
  expect_retirement = true;
  require(entry_attempts == 1 && watched_entry != nullptr, "not the one physical entry request");
  require(scheduler::prepared_scheduler() == nullptr, "registration created worker Scheduler");
  const auto registered = *static_cast<const Stream*>(output.ctx);
  const auto after = get_streams();
  require(after.size() == before.size() + 1 && after.back() == registered, "registration order changed");
  require(after[0] == ordinary && after[1] == local, "ordinary identities changed");
  require(cpu::registered_global_encoder(ordinary) && cpu::registered_global_encoder(registered), "global encoder absent");
  require(!cpu::registered_global_encoder(local), "local encoder became global");
  require(&cpu::get_command_encoder(local) != cpu::registered_global_encoder(registered), "local lookup changed");
  require(mlx_stream_registration_borrow(output, owner) == 0, "actual constructor token refused");
  Token foreign;
  require(mlx_stream_registration_borrow(output, &foreign) == 8, "foreign token accepted");
  require(borrow_stream_registration(ordinary, owner) == StreamRegistrationCause::identity_mismatch, "ordinary entry promoted");
  // Worker/event/input-array initialization is ordinary and explicitly outside
  // this constructor's proof. Real CPU lowering still uses the registered encoder.
  auto result = add(array({3.f, 5.f}), array({2.f, 7.f}), registered);
  eval(result);
  require(result.data<float>()[0] == 5.f && result.data<float>()[1] == 12.f, "CPU values differ");
  auto local_result = multiply(array({2.f, 3.f}), array({4.f, 5.f}), local);
  eval(local_result);
  require(local_result.data<float>()[0] == 8.f && local_result.data<float>()[1] == 15.f, "local CPU values differ");
  mlx_stream_copy_free(output);
  require(retired == 0 && !entry_freed, "wrapper drop retired process registration");
  std::puts("stream registration ownership and CPU values passed");
}
