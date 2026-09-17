#include "mlx/c/submission.h"
#include "doctest/doctest.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/submission.h"
#include "mlx/backend/cpu/encoder.h"
#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <thread>
#include <cstring>
#include <limits>

using namespace mlx::core;
namespace {
struct Counts {
  std::atomic<size_t> retired{0};
  std::atomic<size_t> primitives{0};
};
void retired(void* p) { ++static_cast<Counts*>(p)->retired; }
struct Arena {
  Counts& counts;
  submission::GraphQuota* value;
  Arena(Counts& counts, size_t capacity)
      : counts(counts), value(submission::GraphQuota::create(capacity, &counts, retired)) {}
  ~Arena() { reset(); }
  void reset() { if (value) { auto* old = value; value = nullptr; old->release(); } }
};
struct Scope {
  submission::Scope* value;
  explicit Scope(submission::GraphQuota* graph = nullptr)
      : value(new submission::Scope(nullptr, nullptr, nullptr, graph)) {}
  ~Scope() { value->seal(); value->release(); }
};
struct Probe : Primitive {
  Counts* counts;
  std::optional<array> held;
  void (*callback)(void*) noexcept = nullptr;
  void* context = nullptr;
  Probe(Stream stream, Counts& counts) : Primitive(stream), counts(&counts) {}
  ~Probe() override { ++counts->primitives; if (callback) callback(context); }
  const char* name() const override { return "graph metadata lifetime probe"; }
  void eval_cpu(const ArrayVector&, ArrayVector&) override { std::abort(); }
  void eval_gpu(const ArrayVector&, ArrayVector&) override { std::abort(); }
};
} // namespace

TEST_CASE("graph quota constructors assignment and swap preserve original allocator domains") {
  Counts a_count, b_count;
  Arena a(a_count, 64 * 1024), b(b_count, 64 * 1024);
  std::optional<Shape> from_a, empty_a;
  std::optional<ArrayVector> roots_a;
  auto value = array({3.0f, 5.0f}); // explicit pre-existing legacy source
  {
    Scope scope(a.value);
    from_a.emplace(32, 7);
    empty_a.emplace();
    roots_a.emplace(3, value);
  }
  const auto* original = from_a->data();
  const auto a_bytes = a.value->occupied_bytes();
  {
    Scope scope(b.value);
    Shape copied(*from_a);
    CHECK(copied.get_allocator().resource() == b.value);
    CHECK(copied == *from_a);
    Shape moved(std::move(*from_a));
    CHECK(moved.get_allocator().resource() == b.value);
    CHECK(moved.data() != original);
    CHECK(from_a->empty());
    CHECK(from_a->data() == original); // empty source still owns its old block
    CHECK(a.value->occupied_bytes() == a_bytes);
    Shape legacy(24, 9, Shape::allocator_type(nullptr));
    Shape imported(std::move(legacy));
    CHECK(imported.get_allocator().resource() == b.value);
    CHECK(imported.size() == 24);
    CHECK(legacy.empty());
    *from_a = imported; // fixed existing A destination, even under B
    CHECK(from_a->get_allocator().resource() == a.value);
    CHECK((*from_a)[23] == 9);
    from_a->swap(moved);
    CHECK(from_a->get_allocator().resource() == a.value);
    CHECK(moved.get_allocator().resource() == b.value);
    CHECK((*from_a)[31] == 7);
    CHECK(moved[23] == 9);
    ArrayVector legacy_roots(2, value, ArrayVector::allocator_type(nullptr));
    const auto* legacy_buffer = legacy_roots.data();
    ArrayVector imported_roots(std::move(legacy_roots));
    CHECK(imported_roots.get_allocator().resource() == b.value);
    CHECK(imported_roots.size() == 2);
    CHECK(legacy_roots.empty());
    CHECK(legacy_roots.data() == legacy_buffer);
    ArrayVector roots_b(std::move(*roots_a));
    CHECK(roots_b.get_allocator().resource() == b.value);
    CHECK(roots_a->empty());
    CHECK(roots_b[1].id() == value.id());
    roots_a->push_back(value);
    roots_a->swap(roots_b);
    CHECK(roots_a->get_allocator().resource() == a.value);
    CHECK(roots_b.get_allocator().resource() == b.value);
    CHECK(roots_a->size() == 3);
    CHECK(roots_b.size() == 1);
  }
  from_a.reset(); roots_a.reset();
  CHECK(a.value->occupied_bytes() == 0);
  a.reset();
  CHECK(a_count.retired.load() == 0); // even an empty container protects its allocator
  empty_a.reset();
  CHECK(a_count.retired.load() == 1);
  b.reset();
  CHECK(b_count.retired.load() == 1);
}

TEST_CASE("graph quota cross-domain refusal preserves source and destination contents") {
  Counts a_count, b_count;
  Arena a(a_count, 64 * 1024), b(b_count, 512);
  Shape source(1024, 11, Shape::allocator_type(a.value));
  Shape destination({2, 3}, Shape::allocator_type(b.value));
  const auto* source_buffer = source.data();
  const auto before_a = a.value->occupied_bytes();
  const auto before_b = b.value->occupied_bytes();
  CHECK_THROWS_AS(destination = source, submission::GraphQuotaError);
  CHECK_THROWS_AS(destination = std::move(source), submission::GraphQuotaError);
  CHECK_THROWS_AS(destination.swap(source), submission::GraphQuotaError);
  CHECK(source.size() == 1024);
  CHECK(source[1023] == 11);
  CHECK(source.data() == source_buffer);
  CHECK(destination == Shape({2, 3}));
  CHECK(a.value->occupied_bytes() == before_a);
  CHECK(b.value->occupied_bytes() == before_b);
  CHECK_THROWS_AS(destination.reserve(std::numeric_limits<size_t>::max()), std::length_error);
  CHECK(destination.size() == 2);
  auto leaf = array({2.0f, 5.0f});
  ArrayVector source_roots(128, leaf, ArrayVector::allocator_type(a.value));
  ArrayVector destination_roots(1, leaf, ArrayVector::allocator_type(b.value));
  const auto* roots_buffer = source_roots.data();
  const auto before_roots_a = a.value->occupied_bytes();
  const auto before_roots_b = b.value->occupied_bytes();
  CHECK_THROWS_AS(destination_roots = source_roots, submission::GraphQuotaError);
  CHECK_THROWS_AS(destination_roots = std::move(source_roots), submission::GraphQuotaError);
  CHECK_THROWS_AS(destination_roots.swap(source_roots), submission::GraphQuotaError);
  CHECK(source_roots.data() == roots_buffer);
  CHECK(source_roots.size() == 128);
  CHECK(destination_roots.size() == 1);
  CHECK(destination_roots[0].id() == leaf.id());
  CHECK(a.value->occupied_bytes() == before_roots_a);
  CHECK(b.value->occupied_bytes() == before_roots_b);
}

TEST_CASE("graph quota descriptor construction refusal returns every partial allocation") {
  // Exhaust every byte boundary over this fixed test interval. This is failure
  // coverage, never a fit report or a method for selecting a production ceiling.
  const auto stream = default_stream(Device::cpu);
  size_t refused = 0, constructed = 0;
  for (size_t capacity = 128; capacity <= 4096; ++capacity) {
    Counts counts;
    Arena arena(counts, capacity);
    {
      Scope scope(arena.value);
      try {
        auto primitive = submission::make_graph_primitive<Probe>(stream, counts);
        auto leaf = array(Shape(32, 1), float32, primitive, {});
        auto parent = array(Shape{1}, float32, primitive, ArrayVector(13, leaf));
        CHECK(parent.inputs().size() == 13);
        CHECK(leaf.ndim() == 32);
        ++constructed;
      } catch (const submission::GraphQuotaError& error) {
        CHECK(error.cause() == submission::GraphFailure::exhausted);
        ++refused;
      }
      CHECK(arena.value->occupied_bytes() == 0);
    }
    arena.reset();
    CHECK(counts.retired.load() == 1);
  }
  CHECK(refused > 0);
  CHECK(constructed > 0);
}

TEST_CASE("graph quota multi-output partial publication leaves no sibling cycle") {
  const auto stream = default_stream(Device::cpu);
  size_t refused = 0, constructed = 0;
  for (size_t capacity = 256; capacity <= 8192; capacity += 16) {
    Counts counts;
    Arena arena(counts, capacity);
    {
      Scope scope(arena.value);
      try {
        auto primitive = submission::make_graph_primitive<Probe>(stream, counts);
        auto outputs = array::make_arrays(
            {Shape{1}, Shape{1}, Shape{1}, Shape{1}},
            {float32, float32, float32, float32}, primitive, {});
        CHECK(outputs.size() == 4);
        CHECK(outputs[0].siblings().size() == 3);
        ++constructed;
      } catch (const submission::GraphQuotaError& error) {
        CHECK(error.cause() == submission::GraphFailure::exhausted);
        ++refused;
      }
      CHECK(arena.value->occupied_bytes() == 0);
    }
  }
  CHECK(refused > 0);
  CHECK(constructed > 0);
}

TEST_CASE("graph quota survives external primitive roots and final weak control release") {
  Counts counts;
  Arena arena(counts, 64 * 1024);
  std::optional<array> root;
  std::shared_ptr<Probe> primitive;
  std::weak_ptr<Probe> weak;
  {
    Scope scope(arena.value);
    primitive = submission::make_graph_primitive<Probe>(default_stream(Device::cpu), counts);
    primitive->held.emplace(Shape{1}, float32, nullptr, ArrayVector{});
    root.emplace(Shape{1}, float32, primitive, ArrayVector{});
    weak = primitive;
  }
  root.reset(); // primitive's independent root remains
  CHECK(arena.value->occupied_bytes() > 0);
  arena.reset();
  CHECK(counts.retired.load() == 0);
  primitive.reset();
  CHECK(counts.primitives.load() == 1);
  CHECK(weak.expired());
  CHECK(counts.retired.load() == 0); // actual shared control block remains allocated
  std::thread final_weak([weak = std::move(weak)]() mutable { weak.reset(); });
  final_weak.join();
  CHECK(counts.retired.load() == 1);
}

TEST_CASE("graph quota queued descriptor survives early shared control release") {
  Counts counts;
  Arena arena(counts, 64 * 1024);
  std::optional<array> trigger, released;
  struct Context { std::optional<array>* released; Counts* counts; bool protected_after_release=false; };
  Context context{&released, &counts};
  {
    Scope scope(arena.value);
    released.emplace(Shape(24, 1), float32, nullptr, ArrayVector{});
    auto primitive = submission::make_graph_primitive<Probe>(default_stream(Device::cpu), counts);
    primitive->context = &context;
    primitive->callback = [](void* pointer) noexcept {
      auto& c = *static_cast<Context*>(pointer);
      c.released->reset(); // reentrant deleter queues raw descriptor, control dies
      c.protected_after_release = c.counts->retired == 0;
    };
    trigger.emplace(Shape{1}, float32, std::move(primitive), ArrayVector{});
  }
  arena.reset();
  trigger.reset();
  CHECK(context.protected_after_release);
  CHECK(counts.primitives.load() == 1);
  CHECK(counts.retired.load() == 1);
}

TEST_CASE("graph quota actual nonzero evaluation and nested record association share one arena") {
  Counts counts, foreign_counts;
  Arena arena(counts, 1 << 20), foreign(foreign_counts, 1 << 20);
  {
    Scope parent(arena.value);
    Scope child;
    CHECK(child.value->graph_quota() == arena.value);
    CHECK_THROWS_AS(Scope(foreign.value), submission::GraphQuotaError);
    auto result = multiply(add(array({1.0f, 3.0f, 7.0f}), array({2.0f, 5.0f, 11.0f}), Device::cpu), array(2.0f), Device::cpu);
    auto completion = async_eval_with_completion({result});
    completion.wait();
    CHECK(result.data<float>()[0] == 6.0f);
    CHECK(result.data<float>()[1] == 16.0f);
    CHECK(result.data<float>()[2] == 36.0f);
    submission::progress_records();
    submission::retire_records();
  }
  submission::progress_records();
  submission::retire_records();
  CHECK(arena.value->occupied_bytes() == 0);
  arena.reset();
  CHECK(counts.retired.load() == 1);
}

TEST_CASE("graph quota deep duplicate-edge DAG retires without recursive allocation") {
  Counts counts;
  Arena arena(counts, 8 << 20);
  std::optional<array> root;
  {
    Scope scope(arena.value);
    const auto stream = default_stream(Device::cpu);
    root.emplace(Shape{1}, float32, nullptr, ArrayVector{});
    for (size_t i = 0; i < 2048; ++i) {
      auto primitive = submission::make_graph_primitive<Probe>(stream, counts);
      auto next = array(Shape{1}, float32, std::move(primitive), {*root, *root});
      *root = std::move(next);
    }
  }
  arena.reset();
  CHECK(counts.retired.load() == 0);
  root.reset();
  CHECK(counts.primitives.load() == 2048);
  CHECK(counts.retired.load() == 1);
}

TEST_CASE("graph quota refusal after native submission retains the accepted prefix") {
  Counts counts;
  Arena arena(counts, 64 * 1024);
  {
    Scope scope(arena.value);
    auto result = add(array({2.0f, 7.0f}), array({11.0f, 13.0f}), Device::cpu);
    auto completion = async_eval_with_completion({result});
    ArrayVector roots;
    bool refused = false;
    try {
      for (;;) roots.push_back(add(result, result, Device::cpu));
    } catch (const submission::GraphQuotaError& error) {
      CHECK(error.cause() == submission::GraphFailure::exhausted);
      refused = true;
    }
    CHECK(refused);
    CHECK(arena.value->occupied_bytes() > 0);
    // Graph refusal is not completion. The existing accepted completion is used.
    completion.wait();
    CHECK(result.data<float>()[0] == 13.0f);
    CHECK(result.data<float>()[1] == 20.0f);
    roots.clear();
    submission::progress_records();
    submission::retire_records();
  }
  submission::progress_records();
  submission::retire_records();
  CHECK(arena.value->occupied_bytes() == 0);
  arena.reset();
  CHECK(counts.retired.load() == 1);
}

TEST_CASE("persistent encoder batches retain allocation births across original submissions") {
  Counts first_count, second_count;
  Arena first(first_count, 64 * 1024), second(second_count, 64 * 1024);
  std::unique_ptr<cpu::CommandEncoder> encoder;
  {
    Scope scope(first.value);
    encoder = std::make_unique<cpu::CommandEncoder>(default_stream(Device::cpu));
    CHECK(encoder->take_temporaries().size() == 0);
    encoder->add_temporary(array(Shape{1}, float32, nullptr, ArrayVector{}));
    auto held = encoder->take_temporaries();
    REQUIRE(held.first());
    CHECK(held.first()->birth.get() == first.value);
    // No work was submitted: return the actual array to a newly owned batch.
    encoder->add_temporary(std::move(*held.first()->single));
  }
  first.reset();
  CHECK(first_count.retired.load() == 0); // actual array AND batch birth are held
  encoder->take_temporaries().clear(); // no work submitted in this lifecycle case
  CHECK(first_count.retired.load() == 1);
  std::optional<ArrayVector> callback_roots;
  {
    Scope scope(second.value);
    CHECK(encoder->take_temporaries().size() == 0);
    encoder->add_temporary(array(Shape{1}, float32, nullptr, ArrayVector{}));
    auto detached = encoder->take_temporaries();
    REQUIRE(detached.first());
    CHECK(detached.first()->birth.get() == second.value);
    // The deliberately ordinary caller-owned callback vector remains a
    // separate root owner; it does not replace or rebind batch storage.
    callback_roots.emplace(ArrayVector::allocator_type(nullptr));
    callback_roots->push_back(*detached.first()->single);
    CHECK(callback_roots->get_allocator().resource() == nullptr);
    CHECK(encoder->take_temporaries().size() == 0);
    second.reset();
    CHECK(second_count.retired.load() == 0);
    detached.clear();
  }
  CHECK(second_count.retired.load() == 0); // actual root, not callback list storage
  callback_roots->clear();
  CHECK(second_count.retired.load() == 1);
  CHECK(first_count.retired.load() == 1);
  encoder.reset(); // an empty encoder cannot pin either earlier arena
}

#include "graph_metadata/data_controls.cpp"
#include "graph_metadata/original_buffers.cpp"

#include "graph_metadata/default_buffers.cpp"

#include "graph_metadata/stride_collapse.cpp"
#include "graph_metadata/cpu_retention.cpp"

#include "graph_metadata/submission_runtime.cpp"

#include "graph_metadata/mutable_pair.cpp"

TEST_CASE("fresh Graph population fit preserves holes and failed constructor prefixes") {
  constexpr std::array<size_t, 7> sizes{113, 2049, 17, 511, 8193, 245, 4097};
  size_t bytes = 0, exact = 0;
  for (size_t size : sizes) {
    size_t extent = 0;
    REQUIRE(submission::GraphQuota::allocation_extent(size, alignof(std::max_align_t), extent));
    bytes += size;
    exact += extent;
  }
  size_t population = 0, capacity = 0;
  REQUIRE(submission::GraphQuota::allocation_population_extent(bytes, sizes.size(), population));
  CHECK(population >= exact);
  REQUIRE(submission::GraphQuota::fresh_capacity_for_extents(population, capacity));
  for (unsigned retired_mask = 0; retired_mask < 8; ++retired_mask) {
    Counts counts;
    Arena arena(counts, capacity);
    std::array<void*, sizes.size()> pointers{};
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (i == 3) {
        for (size_t j = 0; j < i; ++j) if (retired_mask & (1u << j)) {
          arena.value->deallocate(pointers[j], sizes[j], alignof(std::max_align_t));
          pointers[j] = nullptr;
        }
      }
      pointers[i] = arena.value->allocate(sizes[i], alignof(std::max_align_t));
      std::memset(pointers[i], int(i + 1), sizes[i]);
    }
    for (size_t i = 0; i < sizes.size(); ++i) if (pointers[i]) {
      const auto* p = static_cast<const unsigned char*>(pointers[i]);
      CHECK(p[0] == i + 1);
      CHECK(p[sizes[i] - 1] == i + 1);
      arena.value->deallocate(pointers[i], sizes[i], alignof(std::max_align_t));
    }
    CHECK(arena.value->occupied_bytes() == 0);
    arena.reset();
    CHECK(counts.retired == 1);
  }
  size_t unchanged = 73;
  CHECK_FALSE(submission::GraphQuota::allocation_population_extent(1, 2, unchanged));
  CHECK(unchanged == 73);
  CHECK_FALSE(submission::GraphQuota::allocation_population_extent(1, 0, unchanged));
  CHECK(unchanged == 73);
  CHECK_FALSE(submission::GraphQuota::allocation_population_extent(
      std::numeric_limits<size_t>::max(), 1, unchanged));
  CHECK(unchanged == 73);
  CHECK_FALSE(submission::GraphQuota::fresh_capacity_for_extents(1, unchanged));
  CHECK(unchanged == 73);
  REQUIRE(submission::GraphQuota::allocation_population_extent(0, 0, unchanged));
  CHECK(unchanged == 0);
}

#include "graph_metadata/next_fit.cpp"
