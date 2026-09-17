// Isolated process: the actual global allocator has no earlier test history.
// This is native mechanism evidence; Device/Graph/Record fixture setup is not a
// neutral admission claim. The neutral tests exercise the real Pool comparison.
#include "mlx/c/prepared_input.h"
#include "mlx/input_allocator.h"
#include "mlx/prepared_input.h"
#include "mlx/original_buffer.h"
#include "mlx/submission.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#ifdef MLX_C_PATCH_TEST_METAL
#include "mlx/backend/metal/allocator.h"
namespace mlx::core::metal {
struct InputAllocatorTestAccess {
  static bool heap_present(MetalAllocator& value) { return value.heap_.get() != nullptr; }
  static bool heap_initialized(MetalAllocator& value) { return value.ordinary_heap_initialized_; }
  static bool virtual_machine(MetalAllocator& value) { return value.virtual_machine_; }
};
}
#endif
using namespace mlx::core;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Owner { std::shared_ptr<std::atomic<unsigned>> retired; };
void retire(void* ptr) {
  std::unique_ptr<Owner> owner(static_cast<Owner*>(ptr)); ++*owner->retired;
}
struct GraphDrop { void operator()(submission::GraphQuota* p) const { if (p) p->release(); } };
struct RecordDrop { void operator()(submission::RecordQuota* p) const { if (p) p->release(); } };
struct ScopeDrop { void operator()(submission::Scope* p) const { if (p) { p->seal(); p->release(); } } };
struct BudgetDrop { void operator()(allocator::OriginalBufferBudget* p) const { if (p) p->release(); } };
struct LeafDrop { void operator()(PreparedInputLeaf* p) const { if (p) p->destroy(); } };
struct ArrayDrop { void operator()(PreparedInputArray* p) const { if (p) p->destroy(); } };
void source_buffers(allocator::Allocator& actual, const allocator::PreparedInputFacts& facts) {
  const uint32_t values[]{17, 0xf1234567u}; const size_t shape[]{2};
  PreparedInputSlotLayout immutable{}; OriginalMutablePairLayout mutable_layout{};
  check(PreparedInputLeaf::layout(facts, {values, shape, 1, 2, 0}, immutable), "immutable layout");
  check(original_mutable_pair_layout(facts, mutable_layout), "mutable layout");
  const auto capacity = std::max(immutable.metadata_bytes, mutable_layout.metadata_bytes);
  std::unique_ptr<submission::GraphQuota, GraphDrop> graph(submission::GraphQuota::try_create(capacity, nullptr, nullptr));
  check(bool(graph), "graph allocation");
  PreparedInputLeaf* raw_leaf = nullptr;
  check(PreparedInputLeaf::create(actual, facts, graph.get(), {values, shape, 1, 2, 0}, raw_leaf) ==
      allocator::PreparedInputCause::success, "immutable birth");
  std::unique_ptr<PreparedInputLeaf, LeafDrop> leaf(raw_leaf);
  check(leaf->value().data<uint32_t>()[1] == values[1], "immutable value");
  check(bool(leaf->value().data_shared_ptr()->original_input), "immutable marker");
  auto immutable_alias = std::make_optional<array>(leaf->value());
  leaf.reset();
  check(immutable_alias->data<uint32_t>()[0] == values[0], "immutable alias");
  immutable_alias.reset();
  check(graph->occupied_bytes() == 0, "immutable final graph retirement");

  std::unique_ptr<allocator::OriginalBufferBudget, BudgetDrop> budget(
      allocator::OriginalBufferBudget::create(actual, mutable_layout.backing_bytes));
  check(bool(budget) && budget->uses_allocator(actual), "actual allocator budget");
  auto token = std::make_unique<unsigned>(0);
  auto failure = FailureCarrier::create(token.get(), [](void* p) { delete static_cast<unsigned*>(p); });
  check(bool(failure), "failure owner"); token.release();
  std::unique_ptr<submission::RecordQuota, RecordDrop> records(
      submission::RecordQuota::create(mutable_layout.record_minimum_capacity, nullptr, nullptr));
  check(bool(records), "record allocation");
  std::optional<array> alias;
  {
    std::unique_ptr<submission::Scope, ScopeDrop> scope(new submission::Scope(nullptr, nullptr, records.get(), graph.get()));
    check(scope->enable_scoped_observation(), "observation");
    check(scope->require_original_controls() == submission::NativeControlFailure::none, "require controls");
    check(scope->bind_failure(failure), "failure binding");
    check(scope->enable_original_controls() == submission::NativeControlFailure::none, "enable controls");
    scope->bind_original_buffer_budget(*budget);
    bool refused = false;
    try { (void)actual.malloc(17); }
    catch (const allocator::OriginalBufferError& e) { refused = e.cause() == allocator::OriginalBufferCause::unsupported; }
    check(refused, "ordinary bypass must refuse before lazy heap");
#ifdef MLX_C_PATCH_TEST_METAL
    check(!metal::InputAllocatorTestAccess::heap_initialized(metal::allocator()), "bypass initialized ordinary heap");
#endif
    PreparedInputArray* raw = nullptr;
    check(construct_original_mutable_pair(actual, facts, graph.get(), *scope, *budget, values, raw) ==
        PreparedHostCopyCause::success, "mutable birth");
    std::unique_ptr<PreparedInputArray, ArrayDrop> output(raw);
    check(output->value().data<uint32_t>()[1] == values[1], "mutable value");
    check(!output->value().data_shared_ptr()->original_input, "mutable marker");
    check(output->value().buffer().original_buffer_budget() == budget.get(), "mutable budget identity");
    alias.emplace(output->value());
  }
  check(budget->occupied_bytes() == mutable_layout.backing_bytes, "alias physical custody");
  check(alias->data<uint32_t>()[0] == values[0], "mutable alias"); alias.reset();
  check(budget->occupied_bytes() == 0, "mutable final physical retirement");
}
}
int main(int argc, char** argv) {
  try {
    const bool ordinary = argc > 1 && std::strcmp(argv[1], "ordinary") == 0;
    const bool missing = argc > 1 && std::strcmp(argv[1], "missing-device") == 0;
    mlx_input_allocator_layout layout{}; mlx_input_allocator_layout_for(&layout);
    check(bool(layout.qualified) == allocator::input_allocator_layout_qualified, "compile-time qualification");
    auto retired = std::make_shared<std::atomic<unsigned>>(0);
    auto owner = std::make_unique<Owner>(Owner{retired});
    mlx_prepared_input_runtime output{}; uint64_t identity = 0;
    if (std::getenv("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION"))
      check(layout.qualified == 1, "selected run requires positive initializer qualification");
    if (!layout.qualified) {
      check(mlx_input_allocator_initialize(&output, &identity, owner.get(), retire) == 1, "unqualified refusal");
      check(!output.allocator && identity == 0 && *retired == 0, "unqualified ownership");
      std::cout << "qualification=unknown; fixed refusal preserved owner\n"; return 0;
    }
#ifdef MLX_C_PATCH_TEST_METAL
    if (missing) {
      check(mlx_input_allocator_initialize(&output, &identity, owner.get(), retire) == 5, "missing Device refusal");
      check(!output.allocator && identity == 0 && *retired == 0, "missing Device ownership");
      check(!metal::prepared_device(Device::gpu), "missing Device initialized implicitly");
      std::cout << "missing Device refused without initialization\n"; return 0;
    }
    // Ordinary fixture preparation, explicitly outside this allocator component.
    (void)metal::device(Device::gpu);
#else
    check(!missing, "missing-device mode is Metal-only");
#endif
    if (ordinary) {
      auto& prior = allocator::allocator(); (void)prior;
      check(mlx_input_allocator_initialize(&output, &identity, owner.get(), retire) == 6, "ordinary predecessor refusal");
      check(!output.allocator && identity == 0 && *retired == 0, "ordinary predecessor ownership");
      std::cout << "ordinary predecessor refused without promotion\n"; return 0;
    }
    check(mlx_input_allocator_initialize(&output, &identity, owner.get(), retire) == 0, "actual initializer");
    owner.release(); check(identity != 0 && output.allocator, "completed identity");
    mlx_prepared_input_runtime alias{};
    check(mlx_input_allocator_borrow(&alias, identity) == 0 && alias.allocator == output.allocator, "same singleton borrow");
    mlx_prepared_input_runtime foreign{};
    check(mlx_input_allocator_borrow(&foreign, identity + 1) == 8 && !foreign.allocator, "foreign identity refusal");
    auto& actual = *static_cast<allocator::Allocator*>(output.allocator);
    check(&allocator::allocator() == &actual, "ordinary uses same allocator");
#ifdef MLX_C_PATCH_TEST_METAL
    check(!metal::InputAllocatorTestAccess::heap_present(metal::allocator()), "input-only heap backing");
    check(!metal::InputAllocatorTestAccess::heap_initialized(metal::allocator()), "input-only heap transition");
#endif
    allocator::PreparedInputFacts facts{}; check(actual.prepare_input_runtime(facts), "actual facts");
    source_buffers(actual, facts);
    auto buffer = actual.malloc(17);
    check(buffer.ptr() != nullptr, "ordinary allocation");
    std::memset(buffer.raw_ptr(), 0x57, 17);
    check(static_cast<unsigned char*>(buffer.raw_ptr())[16] == 0x57, "ordinary values");
    actual.free(buffer);
#ifdef MLX_C_PATCH_TEST_METAL
    check(metal::InputAllocatorTestAccess::heap_initialized(metal::allocator()), "ordinary heap transition missing");
    check(metal::InputAllocatorTestAccess::heap_present(metal::allocator()) ==
        !metal::InputAllocatorTestAccess::virtual_machine(metal::allocator()), "ordinary VM heap policy");
#endif
    check(*retired == 0, "live singleton lost accounting owner");
    std::cout << "qualified actual initializer, immutable/mutable aliases, bypass and ordinary transition passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
