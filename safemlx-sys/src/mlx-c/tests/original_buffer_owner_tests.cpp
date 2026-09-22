#include "mlx/c/original_buffer.h"
#include "mlx/c/owned_host_copy.h"
#include "mlx/c/private/array.h"
#include "doctest/doctest.h"
#include "mlx/mlx.h"
#include "mlx/original_buffer.h"
#include "mlx/prepared_input.h"
#include "mlx/host_transfer.h"
#include "mlx/primitives.h"
#include <array>
#include "mlx/failure.h"
#include "mlx/record_quota.h"
#include "mlx/submission.h"
#include <memory>
#include <optional>
#include <cstring>

using namespace mlx::core;
namespace {
struct GraphRelease { void operator()(submission::GraphQuota* p) const noexcept { p->release(); } };
struct RecordRelease { void operator()(submission::RecordQuota* p) const noexcept { p->release(); } };
struct ScopeRelease { void operator()(submission::Scope* p) const noexcept { p->seal(); p->release(); } };
FailureCarrierRef make_failure() {
  auto owner = std::make_unique<unsigned>(0);
  auto carrier = FailureCarrier::create(owner.get(), [](void* p) { delete static_cast<unsigned*>(p); });
  if (!carrier) throw std::bad_alloc();
  owner.release();
  return carrier;
}
struct BudgetHandle {
  mlx_original_buffer_budget raw{};
  BudgetHandle(mlx_prepared_input_runtime runtime, unsigned& retired) {
    auto status = mlx_original_buffer_budget_new_retaining(&raw, runtime, 1 << 20,
        &retired, [](void* p) { ++*static_cast<unsigned*>(p); }, nullptr);
    if (status) throw std::runtime_error("budget fixture creation failed");
  }
  ~BudgetHandle() { mlx_original_buffer_budget_release(raw); }
  allocator::OriginalBufferBudget& native() const { return *static_cast<allocator::OriginalBufferBudget*>(raw.ctx); }
};
struct Role {
  std::unique_ptr<submission::GraphQuota, GraphRelease> graph{
      submission::GraphQuota::create(64 << 10, nullptr, nullptr)};
  std::unique_ptr<submission::RecordQuota, RecordRelease> records{
      submission::RecordQuota::create(64 << 10, nullptr, nullptr)};
  FailureCarrierRef failure{make_failure()};
  std::unique_ptr<submission::Scope, ScopeRelease> scope{
      new submission::Scope(nullptr, nullptr, records.get(), graph.get())};
  Role() {
    REQUIRE(failure.get());
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == submission::NativeControlFailure::none);
    REQUIRE(scope->bind_failure(failure));
    REQUIRE(scope->enable_original_controls() == submission::NativeControlFailure::none);
  }
  mlx_submission_scope raw() const { return {scope.get()}; }
};
mlx_prepared_input_runtime runtime() {
  mlx_prepared_input_runtime value{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&value) == 0);
  return value;
}
mlx_array borrow(array& value) { return {&value, nullptr}; }
void check_info(array& value, const BudgetHandle& expected, uint64_t generation, size_t capacity) {
  mlx_original_buffer_info original{};
  REQUIRE(mlx_original_buffer_array_info(&original, borrow(value), expected.raw) == 0);
  CHECK(original.known);
  CHECK(original.identity == generation);
  CHECK(original.charged_bytes == capacity);
  bool known = false, host = true;
  uint64_t identity = 0;
  size_t bytes = 0;
  REQUIRE(mlx_array_allocation_info(&known, &host, &identity, &bytes, borrow(value)) == 0);
  CHECK(known);
  CHECK_FALSE(host);
  CHECK(identity == generation);
  CHECK(bytes == capacity);
}
}

TEST_CASE("original buffer safe ABI authenticates charged capacity across mutable aliases") {
  auto prepared = runtime();
  unsigned a_retired = 0, b_retired = 0;
  BudgetHandle a(prepared, a_retired), b(prepared, b_retired);
  std::optional<array> value(array({2.0f, 3.0f, 5.0f}));
  array alias({7.0f, 11.0f, 13.0f});
  uint64_t generation = 0;
  size_t capacity = 0;
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), a.raw) == 0);
    auto binding = allocator::OriginalBufferBinding::capture(a.native());
    auto incoming = allocator::allocate_original(3 * sizeof(float), binding);
    generation = incoming.original_allocation_generation();
    capacity = incoming.original_allocation_capacity();
    auto* data = static_cast<float*>(incoming.raw_ptr());
    data[0] = 17; data[1] = 19; data[2] = 23;
    value->set_data(incoming);
    check_info(*value, a, generation, capacity);
    mlx_original_buffer_info foreign{};
    CHECK(mlx_original_buffer_array_info(&foreign, borrow(*value), b.raw) == MLX_ORIGINAL_BUFFER_FOREIGN);
    CHECK_FALSE(foreign.known);
    alias.copy_shared_buffer(*value);
  }
  value.reset();
  CHECK(alias.is_donatable());
  check_info(alias, a, generation, capacity);
  CHECK(alias.data<float>()[0] == 17);
  CHECK(alias.data<float>()[2] == 23);
  CHECK(a.native().occupied_bytes() == capacity);
  CHECK(b.native().occupied_bytes() == 0);
}

TEST_CASE("original buffer safe ABI rejects custom deleters and generation mismatch") {
  auto prepared = runtime();
  unsigned retired = 0;
  BudgetHandle budget(prepared, retired);
  array custom({2.0f, 3.0f, 5.0f}), ordinary({7.0f, 11.0f, 13.0f});
  mlx_original_buffer_info info{};
  CHECK(mlx_original_buffer_array_info(&info, borrow(ordinary), budget.raw) == 0);
  CHECK_FALSE(info.known);
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), budget.raw) == 0);
    auto binding = allocator::OriginalBufferBinding::capture(budget.native());
    auto incoming = allocator::allocate_original(3 * sizeof(float), binding);
    const auto generation = incoming.original_allocation_generation();
    custom.set_data(incoming, [](allocator::Buffer p) { allocator::free(p); });
    REQUIRE(custom.data_shared_ptr()->allocation_generation == generation);
    CHECK(mlx_original_buffer_array_info(&info, borrow(custom), budget.raw) == 0);
    CHECK_FALSE(info.known);
    // A real mutable birth with a mismatched Data generation must stay unknown
    // even through ordinary inspection. Restore it before normal destruction.
    auto actual = allocator::allocate_original(3 * sizeof(float), binding);
    ordinary.set_data(actual);
    const auto correct = ordinary.data_shared_ptr()->allocation_generation;
    ordinary.data_shared_ptr()->allocation_generation = generation;
    REQUIRE(generation != correct);
    CHECK(mlx_original_buffer_array_info(&info, borrow(ordinary), budget.raw) == 0);
    CHECK_FALSE(info.known);
    ordinary.data_shared_ptr()->allocation_generation = correct;
  }
}

TEST_CASE("original buffer safe ABI binds once and preserves owner on refusal") {
  auto prepared = runtime();
  unsigned retired = 0;
  mlx_original_buffer_budget duplicate{};
  {
    BudgetHandle budget(prepared, retired);
    CHECK(mlx_original_buffer_budget_bind({}, budget.raw) == MLX_ORIGINAL_BUFFER_MISSING);
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), budget.raw) == 0);
    CHECK(mlx_original_buffer_budget_bind(role.raw(), budget.raw) == MLX_ORIGINAL_BUFFER_BOUND);
    CHECK(mlx_original_buffer_budget_new_retaining(&duplicate, prepared, 1,
        &retired, [](void* p) { ++*static_cast<unsigned*>(p); }, nullptr) == MLX_ORIGINAL_BUFFER_SCOPE);
    CHECK(duplicate.ctx == nullptr);
    CHECK(retired == 0);
    submission::mark_native_control_construction();
    CHECK(mlx_original_buffer_budget_bind(role.raw(), budget.raw) == MLX_ORIGINAL_BUFFER_SCOPE);
  }
  CHECK(retired == 1);
}

namespace {
struct AttachmentState { unsigned retired{0}; };
struct AttachmentPayload { std::shared_ptr<AttachmentState> state; };
struct EmptyNodeRelease {
  void operator()(void* node) const noexcept { mlx_allocation_owner_node_free(node); }
};
void retire_attachment(void* pointer) noexcept {
  auto* payload = static_cast<AttachmentPayload*>(pointer);
  auto state = std::move(payload->state);
  delete payload;
  ++state->retired;
}
struct PreparedAttachment {
  // Reverse C++ destruction frees the native shell before payload custody.
  std::unique_ptr<AttachmentPayload> payload;
  std::unique_ptr<void, EmptyNodeRelease> node;
  explicit PreparedAttachment(const std::shared_ptr<AttachmentState>& state)
      : payload(std::make_unique<AttachmentPayload>(AttachmentPayload{state})),
        node(mlx_allocation_owner_node_new()) {
    if (!node) throw std::bad_alloc();
  }
  unsigned attach(array& value, const mlx_original_buffer_info& expected,
      const BudgetHandle* budget = nullptr) {
    const auto status = budget
        ? mlx_original_buffer_array_attach(borrow(value), budget->raw, &expected,
            node.get(), payload.get(), retire_attachment)
        : mlx_original_buffer_array_alias_attach(borrow(value), &expected,
            node.get(), payload.get(), retire_attachment);
    if (status == MLX_ORIGINAL_BUFFER_OK) {
      node.release();
      payload.release();
    }
    return status;
  }
};
void fill_original(array& value, allocator::OriginalBufferBinding& binding,
    float first, float second, float third) {
  auto incoming = allocator::allocate_original(3 * sizeof(float), binding);
  auto* values = static_cast<float*>(incoming.raw_ptr());
  values[0] = first; values[1] = second; values[2] = third;
  value.set_data(incoming);
}
mlx_original_buffer_info alias_info(array& value) {
  mlx_original_buffer_info info{};
  REQUIRE(mlx_original_buffer_array_alias_info(&info, borrow(value)) == MLX_ORIGINAL_BUFFER_OK);
  return info;
}
}

TEST_CASE("original birth alias attachment survives originating safe handle and shared Data") {
  auto prepared = runtime();
  unsigned budget_retired = 0;
  auto state = std::make_shared<AttachmentState>();
  std::optional<BudgetHandle> budget;
  budget.emplace(prepared, budget_retired);
  std::optional<array> value(array({2.0f, 3.0f, 5.0f}));
  std::optional<array> alias(array({7.0f, 11.0f, 13.0f}));
  mlx_original_buffer_info expected{};
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), budget->raw) == 0);
    auto binding = allocator::OriginalBufferBinding::capture(budget->native());
    fill_original(*value, binding, 17, 19, 23);
    expected = alias_info(*value);
    REQUIRE(expected.known);
    alias->copy_shared_buffer(*value);
  }
  budget.reset();
  CHECK(budget_retired == 0);
  const auto surviving = alias_info(*alias);
  REQUIRE(surviving.known);
  CHECK(surviving.identity == expected.identity);
  CHECK(surviving.charged_bytes == expected.charged_bytes);
  PreparedAttachment owner(state);
  REQUIRE(owner.attach(*alias, expected) == MLX_ORIGINAL_BUFFER_OK);
  CHECK_FALSE(owner.node);
  CHECK_FALSE(owner.payload);
  value.reset();
  CHECK(state->retired == 0);
  CHECK(alias->data<float>()[0] == 17);
  CHECK(alias->data<float>()[2] == 23);
  CHECK(alias->is_donatable());
  alias.reset();
  CHECK(state->retired == 1);
  CHECK(budget_retired == 1);
}

TEST_CASE("original birth attachment refuses every changed input before consuming exact nodes") {
  auto prepared = runtime();
  unsigned a_retired = 0, b_retired = 0;
  BudgetHandle a(prepared, a_retired), b(prepared, b_retired);
  auto state = std::make_shared<AttachmentState>();
  std::optional<array> value(array({2.0f, 3.0f, 5.0f}));
  array ordinary({7.0f, 11.0f, 13.0f});
  PreparedAttachment owner(state);
  void* node = owner.node.get();
  void* payload = owner.payload.get();
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), a.raw) == 0);
    auto binding = allocator::OriginalBufferBinding::capture(a.native());
    fill_original(*value, binding, 29, 31, 37);
    const auto expected = alias_info(*value);
    REQUIRE(expected.known);
    auto wrong = expected;
    ++wrong.identity;
    CHECK(owner.attach(*value, wrong, &a) == MLX_ORIGINAL_BUFFER_CHANGED);
    wrong = expected;
    ++wrong.charged_bytes;
    CHECK(owner.attach(*value, wrong, &a) == MLX_ORIGINAL_BUFFER_CHANGED);
    CHECK(owner.attach(*value, expected, &b) == MLX_ORIGINAL_BUFFER_FOREIGN);
    wrong = expected;
    wrong.known = false;
    CHECK(owner.attach(*value, wrong, &a) == MLX_ORIGINAL_BUFFER_LAYOUT);
    CHECK(owner.attach(ordinary, expected) == MLX_ORIGINAL_BUFFER_UNCERTIFIED);
    CHECK(mlx_original_buffer_array_alias_attach({}, &expected, node, payload,
        retire_attachment) == MLX_ORIGINAL_BUFFER_LAYOUT);
    CHECK(mlx_original_buffer_array_attach(borrow(*value), {}, &expected, node,
        payload, retire_attachment) == MLX_ORIGINAL_BUFFER_LAYOUT);
    CHECK(owner.node.get() == node);
    CHECK(owner.payload.get() == payload);
    CHECK(state->retired == 0);
    CHECK(value->data<float>()[1] == 31);
    REQUIRE(owner.attach(*value, expected, &a) == MLX_ORIGINAL_BUFFER_OK);
  }
  CHECK(state->retired == 0);
  value.reset();
  CHECK(state->retired == 1);
}

TEST_CASE("original birth stale descriptor refusal leaves the old alias attachable") {
  auto prepared = runtime();
  unsigned retired = 0;
  BudgetHandle budget(prepared, retired);
  auto state = std::make_shared<AttachmentState>();
  std::optional<array> value(array({2.0f, 3.0f, 5.0f}));
  std::optional<array> old_alias(array({7.0f, 11.0f, 13.0f}));
  PreparedAttachment owner(state);
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), budget.raw) == 0);
    auto binding = allocator::OriginalBufferBinding::capture(budget.native());
    fill_original(*value, binding, 41, 43, 47);
    const auto old = alias_info(*value);
    REQUIRE(old.known);
    old_alias->copy_shared_buffer(*value);
    fill_original(*value, binding, 53, 59, 61);
    const auto current = alias_info(*value);
    REQUIRE(current.known);
    CHECK(current.identity != old.identity);
    CHECK(current.charged_bytes == old.charged_bytes);
    CHECK(owner.attach(*value, old) == MLX_ORIGINAL_BUFFER_CHANGED);
    CHECK(state->retired == 0);
    REQUIRE(owner.attach(*old_alias, old) == MLX_ORIGINAL_BUFFER_OK);
  }
  CHECK(old_alias->data<float>()[0] == 41);
  CHECK(value->data<float>()[0] == 53);
  value.reset();
  CHECK(state->retired == 0);
  old_alias.reset();
  CHECK(state->retired == 1);
}

namespace {
mlx_original_buffer_info ordinary_info(const array& value, uint32_t expected_kind) {
  uint32_t kind = 99;
  mlx_original_buffer_info info{};
  // Read-only C inspection accepts the existing borrowed const descriptor.
  mlx_array borrowed{const_cast<array*>(&value), nullptr};
  REQUIRE(mlx_ordinary_buffer_array_info(&kind, &info, borrowed) == MLX_ORIGINAL_BUFFER_OK);
  REQUIRE(kind == expected_kind);
  return info;
}
unsigned attach_ordinary(PreparedAttachment& owner, array& value,
    const mlx_original_buffer_info& expected) {
  const auto result = mlx_ordinary_buffer_array_attach(borrow(value), &expected,
      owner.node.get(), owner.payload.get(), retire_attachment);
  if (result == MLX_ORIGINAL_BUFFER_OK) {
    owner.node.release();
    owner.payload.release();
  }
  return result;
}
}

TEST_CASE("ordinary buffer witness positively identifies only settled default backing") {
  auto prepared = runtime();
  unsigned retired = 0;
  BudgetHandle budget(prepared, retired);
  array ordinary({67.0f, 71.0f, 73.0f});
  array custom({79.0f, 83.0f, 89.0f});
  auto lazy = add(ordinary, ordinary, default_stream(Device::cpu));
  const auto known = ordinary_info(ordinary, MLX_ORDINARY_BUFFER_ALLOCATION);
  REQUIRE(known.known);
  CHECK(known.identity != 0);
  CHECK(known.charged_bytes >= 3 * sizeof(float));
  ordinary_info(lazy, MLX_ORDINARY_BUFFER_UNKNOWN);
  // A complete zero-element descriptor can genuinely have no physical Data.
  array empty(Shape{0}, float32, nullptr, {});
  empty.set_status(array::Status::available);
  const auto absent = ordinary_info(empty, MLX_ORDINARY_BUFFER_EMPTY);
  CHECK_FALSE(absent.known);
  CHECK(absent.identity == 0);
  CHECK(absent.charged_bytes == 0);
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), budget.raw) == 0);
    auto binding = allocator::OriginalBufferBinding::capture(budget.native());
    fill_original(ordinary, binding, 97, 101, 103);
    ordinary_info(ordinary, MLX_ORDINARY_BUFFER_UNKNOWN);
    auto incoming = allocator::allocate_original(3 * sizeof(float), binding);
    custom.set_data(incoming, [](allocator::Buffer value) { allocator::free(value); });
    ordinary_info(custom, MLX_ORDINARY_BUFFER_UNKNOWN);
    // CPU zero backing owns a charged page; it must never appear allocation-free.
    if (budget.native().facts().kind == allocator::HostTransferStorageKind::cpu) {
      array zero(std::initializer_list<float>{});
      zero.set_data(allocator::allocate_original(0, binding));
      const auto native = alias_info(zero);
      CHECK(native.known);
      CHECK(native.charged_bytes != 0);
      ordinary_info(zero, MLX_ORDINARY_BUFFER_UNKNOWN);
    }
  }
}

TEST_CASE("ordinary buffer checked attachment refuses a changed native kind and retains exact owner") {
  auto prepared = runtime();
  unsigned retired = 0;
  BudgetHandle budget(prepared, retired);
  auto state = std::make_shared<AttachmentState>();
  std::optional<array> value(array({107.0f, 109.0f, 113.0f}));
  std::optional<array> old_alias(array({139.0f, 149.0f, 151.0f}));
  old_alias->copy_shared_buffer(*value);
  const auto expected = ordinary_info(*value, MLX_ORDINARY_BUFFER_ALLOCATION);
  PreparedAttachment owner(state);
  auto* node = owner.node.get();
  auto* payload = owner.payload.get();
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind(role.raw(), budget.raw) == 0);
    auto binding = allocator::OriginalBufferBinding::capture(budget.native());
    fill_original(*value, binding, 127, 131, 137);
    CHECK(attach_ordinary(owner, *value, expected) == MLX_ORIGINAL_BUFFER_CHANGED);
    CHECK(owner.node.get() == node);
    CHECK(owner.payload.get() == payload);
    CHECK(state->retired == 0);
  }
  auto wrong = expected;
  ++wrong.charged_bytes;
  CHECK(attach_ordinary(owner, *old_alias, wrong) == MLX_ORIGINAL_BUFFER_CHANGED);
  REQUIRE(attach_ordinary(owner, *old_alias, expected) == MLX_ORIGINAL_BUFFER_OK);
  CHECK(old_alias->data<float>()[0] == 107);
  CHECK(value->data<float>()[0] == 127);
  value.reset();
  CHECK(state->retired == 0);
  old_alias.reset();
  CHECK(state->retired == 1);
}

#ifdef _LIBCPP_VERSION
TEST_CASE("ordinary buffer witness never replaces an immutable original input source proof") {
  auto& selected = allocator::allocator();
  allocator::PreparedInputFacts facts{};
  REQUIRE(selected.prepare_input_runtime(facts));
  const std::array<size_t, 1> shape{3};
  const std::array<int32_t, 3> values{157, -163, 167};
  const PreparedInputSource source{values.data(), shape.data(), 1, 3, 1};
  PreparedInputSlotLayout layout{};
  REQUIRE(PreparedInputLeaf::layout(facts, source, layout));
  std::unique_ptr<submission::GraphQuota, GraphRelease> graph(
      submission::GraphQuota::create(layout.metadata_bytes, nullptr, nullptr));
  PreparedInputLeaf* incoming = nullptr;
  REQUIRE(PreparedInputLeaf::create(selected, facts, graph.get(), source, incoming)
      == allocator::PreparedInputCause::success);
  struct LeafRelease { void operator()(PreparedInputLeaf* p) const noexcept { p->destroy(); } };
  std::unique_ptr<PreparedInputLeaf, LeafRelease> leaf(incoming);
  REQUIRE(leaf->value().data_shared_ptr()->original_input != nullptr);
  ordinary_info(leaf->value(), MLX_ORDINARY_BUFFER_UNKNOWN);
  CHECK(leaf->value().data<int32_t>()[0] == 157);
  CHECK(leaf->value().data<int32_t>()[1] == -163);
  CHECK(leaf->value().data<int32_t>()[2] == 167);
}
#endif

namespace {
// This source owner is independent of the fixture's stack lifetime, including
// a REQUIRE unwind while an alias retains Data or its Graph control block.
struct ImmutableCopyFixture {
  std::shared_ptr<AttachmentState> state{std::make_shared<AttachmentState>()};
  std::unique_ptr<submission::GraphQuota, GraphRelease> graph;
  mlx_owned_host_copy_slot slot{};
  mlx_array output{};
  mlx_owned_host_copy_layout layout{};
  mlx_original_buffer_info birth{};
  ImmutableCopyFixture(mlx_prepared_input_runtime prepared, int extent) {
    REQUIRE(mlx_owned_host_copy_layout_for(&layout, prepared, &extent, 1, MLX_FLOAT32) == 0);
    auto payload = std::make_unique<AttachmentPayload>(AttachmentPayload{state});
    graph.reset(submission::GraphQuota::create(layout.metadata_bytes, payload.get(), retire_attachment));
    REQUIRE(graph);
    payload.release();
    REQUIRE(mlx_owned_host_copy_new(&slot, prepared, mlx_submission_graph_quota{graph.get()}, &extent, 1, MLX_FLOAT32) == 0);
  }
  void fill(mlx_submission_observer observer, const float* values) {
    REQUIRE(mlx_owned_host_copy_fill_completed(&output, &birth, slot, observer, values, layout.copy_bytes) == 0);
  }
  void close() {
    mlx_array_free(output); output = {};
    mlx_owned_host_copy_free(slot); slot = {};
    graph.reset();
  }
  ~ImmutableCopyFixture() { close(); }
};
unsigned attach_immutable(PreparedAttachment& owner, array& value, const mlx_original_buffer_info& expected) {
  const auto status = mlx_immutable_source_array_attach(borrow(value), &expected,
      owner.node.get(), owner.payload.get(), retire_attachment);
  if (status == MLX_ORIGINAL_BUFFER_OK) { owner.node.release(); owner.payload.release(); }
  return status;
}
mlx_original_buffer_info immutable_info(array& value, uint32_t expected_kind) {
  uint32_t kind = 99;
  mlx_original_buffer_info info{};
  REQUIRE(mlx_immutable_source_array_info(&kind, &info, borrow(value)) == MLX_ORIGINAL_BUFFER_OK);
  CHECK(kind == expected_kind);
  return info;
}
}

TEST_CASE("immutable completed copy attaches the exact Data and retires after final alias") {
  const auto prepared = runtime();
  ImmutableCopyFixture copy(prepared, 3);
  Role role;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const float values[] = {2.5f, -0.0f, -7.25f};
  copy.fill(observer, values);
  REQUIRE(copy.birth.known);
  CHECK(copy.birth.charged_bytes == copy.layout.backing_bytes);
  auto& actual = mlx_array_get_(copy.output);
  const auto facts = immutable_info(actual, MLX_ORDINARY_BUFFER_ALLOCATION);
  CHECK(facts.identity == copy.birth.identity);
  CHECK(facts.charged_bytes == copy.birth.charged_bytes);
  CHECK_FALSE(actual.is_donatable());
  std::optional<array> alias(actual);
  auto owner_state = std::make_shared<AttachmentState>();
  PreparedAttachment owner(owner_state);
  REQUIRE(attach_immutable(owner, *alias, copy.birth) == MLX_ORIGINAL_BUFFER_OK);
  CHECK_FALSE(owner.node);
  std::weak_ptr<array::Data> weak = actual.data_shared_ptr();
  copy.close();
  CHECK(copy.state->retired == 0);
  CHECK(owner_state->retired == 0);
  CHECK(std::memcmp(alias->data<float>(), values, sizeof(values)) == 0);
  CHECK_FALSE(alias->is_donatable());
  alias.reset();
  CHECK(weak.expired());
  CHECK(owner_state->retired == 1);
  // Data's Graph control block can outlive its physical payload.
  CHECK(copy.state->retired == 0);
  weak.reset();
  CHECK(copy.state->retired == 1);
  mlx_submission_observer_release(observer);
}

TEST_CASE("immutable attachment refuses swapped completed births and preserves both exact nodes") {
  const auto prepared = runtime();
  ImmutableCopyFixture a(prepared, 3), b(prepared, 3);
  Role role;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const float values[] = {3.0f, 5.0f, 11.0f};
  a.fill(observer, values); b.fill(observer, values);
  REQUIRE(a.birth.identity != b.birth.identity);
  auto state = std::make_shared<AttachmentState>();
  PreparedAttachment owner(state);
  const auto node = owner.node.get(); const auto payload = owner.payload.get();
  CHECK(attach_immutable(owner, mlx_array_get_(b.output), a.birth) == MLX_ORIGINAL_BUFFER_CHANGED);
  auto wrong = a.birth; ++wrong.charged_bytes;
  CHECK(attach_immutable(owner, mlx_array_get_(a.output), wrong) == MLX_ORIGINAL_BUFFER_CHANGED);
  wrong = a.birth; wrong.known = false;
  CHECK(attach_immutable(owner, mlx_array_get_(a.output), wrong) == MLX_ORIGINAL_BUFFER_LAYOUT);
  // The separate mutable alias path never recognizes original_input backing.
  CHECK(owner.attach(mlx_array_get_(a.output), a.birth) == MLX_ORIGINAL_BUFFER_UNCERTIFIED);
  CHECK(owner.node.get() == node); CHECK(owner.payload.get() == payload);
  CHECK(state->retired == 0);
  mlx_array refused{};
  mlx_original_buffer_info sentinel{true, 123, 456};
  CHECK(mlx_owned_host_copy_fill_completed(&refused, &sentinel, a.slot, observer, values, sizeof(values)) == 7);
  CHECK(refused.ctx == nullptr);
  CHECK(sentinel.known);
  CHECK(sentinel.identity == 123);
  CHECK(sentinel.charged_bytes == 456);
  REQUIRE(attach_immutable(owner, mlx_array_get_(a.output), a.birth) == MLX_ORIGINAL_BUFFER_OK);
  a.close();
  CHECK(state->retired == 1);
  mlx_submission_observer_release(observer);
}

TEST_CASE("immutable completion distinguishes actual allocation-free Data from nonzero source birth") {
  const auto prepared = runtime();
  ImmutableCopyFixture empty(prepared, 0), nonzero(prepared, 1);
  Role role;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const float value = -0.0f;
  empty.fill(observer, nullptr); nonzero.fill(observer, &value);
  auto& actual_empty = mlx_array_get_(empty.output);
  REQUIRE(actual_empty.data_shared_ptr());
  CHECK_FALSE(empty.birth.known);
  CHECK(empty.birth.identity == 0); CHECK(empty.birth.charged_bytes == 0);
  immutable_info(actual_empty, MLX_ORDINARY_BUFFER_EMPTY);
  REQUIRE(nonzero.birth.known);
  immutable_info(mlx_array_get_(nonzero.output), MLX_ORDINARY_BUFFER_ALLOCATION);
  auto state = std::make_shared<AttachmentState>();
  PreparedAttachment owner(state);
  CHECK(attach_immutable(owner, actual_empty, nonzero.birth) == MLX_ORIGINAL_BUFFER_UNCERTIFIED);
  CHECK(owner.node); CHECK(owner.payload); CHECK(state->retired == 0);
  mlx_submission_observer_release(observer);
}

TEST_CASE("Host array alias preserves exact backing and failed attachment custody until final owner") {
  const auto stream = default_stream(Device::cpu);
  HostTransferBuffer host(Shape{3}, float32, HostTransferPolicy::transfer);
  array input({2.5f, -0.0f, -7.25f});
  std::optional<array> output(array(input.shape(), input.dtype(),
      submission::make_graph_primitive<CopyToHostTransfer>(stream, host),
      {contiguous(input, false, stream)}));
  mlx_immutable_host_transfer_info facts{};
  REQUIRE(mlx_host_transfer_array_alias_info(&facts, borrow(*output)) == MLX_ORIGINAL_BUFFER_OK);
  CHECK_FALSE(facts.backing.known); // no eager evaluation or readiness inference
  output->eval();
  REQUIRE(mlx_host_transfer_array_alias_info(&facts, borrow(*output)) == MLX_ORIGINAL_BUFFER_OK);
  REQUIRE(facts.backing.known);
  CHECK_FALSE(facts.prepared_source);
  CHECK(facts.backing.identity == host.allocation_identity());
  CHECK(facts.backing.charged_bytes == host.capacity());
  CHECK(static_cast<float*>(host.data())[0] == 2.5f);
  CHECK(static_cast<float*>(host.data())[2] == -7.25f);
  auto state = std::make_shared<AttachmentState>();
  PreparedAttachment owner(state);
  auto attach = [&](array& value, const mlx_immutable_host_transfer_info& expected) {
    const auto status = mlx_host_transfer_array_alias_attach(borrow(value), &expected,
        owner.node.get(), owner.payload.get(), retire_attachment);
    if (status == MLX_ORIGINAL_BUFFER_OK) { owner.node.release(); owner.payload.release(); }
    return status;
  };
  auto wrong_kind = facts;
  wrong_kind.prepared_source = true;
  CHECK(attach(*output, wrong_kind) == MLX_ORIGINAL_BUFFER_CHANGED);
  CHECK(owner.node); CHECK(owner.payload);
  auto wrong_capacity = facts;
  ++wrong_capacity.backing.charged_bytes;
  CHECK(attach(*output, wrong_capacity) == MLX_ORIGINAL_BUFFER_CHANGED);
  CHECK(owner.node);
  CHECK(owner.payload);
  HostTransferBuffer unrelated(Shape{3}, float32, HostTransferPolicy::transfer);
  auto wrong_generation = facts;
  wrong_generation.backing.identity = unrelated.allocation_identity();
  REQUIRE(wrong_generation.backing.identity != facts.backing.identity);
  CHECK(attach(*output, wrong_generation) == MLX_ORIGINAL_BUFFER_CHANGED);
  CHECK(owner.node);
  CHECK(attach(input, facts) == MLX_ORIGINAL_BUFFER_UNCERTIFIED);
  CHECK(owner.node);
  CHECK(state->retired == 0);
  std::optional<array> alias(*output);
  REQUIRE(attach(*output, facts) == MLX_ORIGINAL_BUFFER_OK);
  CHECK_FALSE(owner.node);
  output.reset();
  CHECK(state->retired == 0);
  alias.reset();
  CHECK(state->retired == 0); // independent actual Host owner still exists
  host = HostTransferBuffer();
  CHECK(state->retired == 1);
}
