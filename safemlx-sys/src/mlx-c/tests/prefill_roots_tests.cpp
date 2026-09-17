#include "doctest/doctest.h"
#include "mlx/c/prefill_roots.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/graph_quota.h"
#include "mlx/submission.h"
#include "mlx/ops.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <limits>
#include <optional>
using namespace mlx::core;
namespace {
using Graph = submission::GraphQuota;
struct Count { std::atomic<size_t> retired{0}; };
struct CountErrors {
  size_t count{0};
  CountErrors() {
    mlx_set_error_handler([](const char*, void* raw) {
      ++*static_cast<size_t*>(raw);
    }, &count, [](void*) {});
  }
  ~CountErrors() { mlx_set_error_handler(nullptr, nullptr, nullptr); }
};
void retired(void* value) { ++static_cast<Count*>(value)->retired; }
struct Arena {
  Graph* value;
  Arena(size_t capacity, Count& count) : value(Graph::try_create(capacity, &count, retired)) {}
  ~Arena() { reset(); }
  void reset() { if (auto* old=value) { value=nullptr; old->release(); } }
};
struct Enter {
  submission::Scope* value;
  explicit Enter(Graph* graph=nullptr) : value(new submission::Scope(nullptr,nullptr,nullptr,graph)) {}
  ~Enter() { value->seal(); value->release(); }
};
struct Roots {
  mlx_prefill_roots value{nullptr};
  ~Roots() { reset(); }
  void reset() { if(value.ctx) { auto old=value; value.ctx=nullptr; mlx_prefill_roots_free(old); } }
  unsigned create(size_t count, Graph* graph=nullptr) {
    return mlx_prefill_roots_new(&value,count,mlx_submission_graph_quota{graph});
  }
};
// Lexically borrowed only; never passed to a C owning-array destructor.
mlx_array view(array& value) { return mlx_array{&value,nullptr}; }
}
TEST_CASE("prefill roots minimum extent covers alignment fragmentation and absorbed tails") {
  bool absorbed=false;
  for(size_t alignment=1;alignment<=alignof(std::max_align_t);alignment*=2) {
    for(size_t bytes:{size_t{0},size_t{1},size_t{7},size_t{16},size_t{53}}) {
      size_t minimum=0,maximum=0;
      REQUIRE(Graph::minimum_allocation_extent(bytes,alignment,minimum));
      REQUIRE(Graph::allocation_extent(bytes,alignment,maximum)); CHECK(minimum<=maximum);
      Count count; Arena arena(5*maximum,count); REQUIRE(arena.value);
      auto* a=arena.value->try_allocate(bytes,alignment); REQUIRE(a);
      CHECK(arena.value->occupied_bytes()>=minimum);
      CHECK(reinterpret_cast<uintptr_t>(a)%alignment==0);
      auto* b=arena.value->try_allocate(bytes,alignment); REQUIRE(b);
      auto* c=arena.value->try_allocate(bytes,alignment); REQUIRE(c);
      arena.value->deallocate(b,bytes,alignment);
      const auto fragmented=arena.value->occupied_bytes();
      b=arena.value->try_allocate(bytes,alignment); REQUIRE(b);
      CHECK(arena.value->occupied_bytes()-fragmented>=minimum);
      arena.value->deallocate(a,bytes,alignment); arena.value->deallocate(b,bytes,alignment);
      arena.value->deallocate(c,bytes,alignment); CHECK(arena.value->occupied_bytes()==0);
      Count tail_count; Arena tail(minimum+alignof(std::max_align_t),tail_count); REQUIRE(tail.value);
      auto* p=tail.value->try_allocate(bytes,alignment); REQUIRE(p);
      CHECK(tail.value->occupied_bytes()>=minimum);
      absorbed |= tail.value->occupied_bytes()>minimum;
      tail.value->deallocate(p,bytes,alignment); CHECK(tail.value->occupied_bytes()==0);
    }
  }
  CHECK(absorbed); size_t out=0;
  CHECK_FALSE(Graph::minimum_allocation_extent(1,3,out));
  CHECK_FALSE(Graph::minimum_allocation_extent(SIZE_MAX,alignof(std::max_align_t),out));
  REQUIRE(array::graph_descriptor_minimum_extent(out)); CHECK(out>sizeof(array));
}
TEST_CASE("prefill roots fixed buffers refuse atomically and retain their actual arena") {
  mlx_prefill_roots_layout layout{}; REQUIRE(mlx_prefill_roots_layout_for(&layout,3)==0);
  size_t block=0; REQUIRE(Graph::minimum_allocation_extent(3*sizeof(array),alignof(array),block));
  Count short_count;
  { Arena arena(2*block-1,short_count); REQUIRE(arena.value); Roots roots;
    CHECK(roots.create(3,arena.value)==2); CHECK_FALSE(roots.value.ctx);
    CHECK(arena.value->occupied_bytes()==0); }
  CHECK(short_count.retired==1);
  Count count; Arena arena(layout.graph_bytes,count); REQUIRE(arena.value);
  Roots roots; REQUIRE(roots.create(3,arena.value)==0);
  CHECK(arena.value->occupied_bytes()>=2*block); CHECK(arena.value->occupied_bytes()<=layout.graph_bytes);
  auto source=array({2.0f,5.0f});
  for(size_t n=0;n<3;++n) REQUIRE(mlx_prefill_roots_append(roots.value,view(source))==0);
  CHECK(mlx_prefill_roots_append(roots.value,view(source))==2); CHECK(mlx_prefill_roots_size(roots.value)==3);
  arena.reset(); CHECK(count.retired==0); roots.reset(); CHECK(count.retired==1);
  CHECK(mlx_prefill_roots_layout_for(&layout,SIZE_MAX)==1);
}
TEST_CASE("prefill roots ordinary completion preserves nonzero values and empty sites") {
  const auto stream=default_stream(Device::cpu); Enter entered;
  auto source=array({2.0f,-3.0f,7.0f}); auto output=add(source,array(4.0f),stream);
  Roots roots; REQUIRE(roots.create(2)==0);
  REQUIRE(mlx_prefill_roots_append(roots.value,view(output))==0);
  REQUIRE(mlx_prefill_roots_append(roots.value,view(source))==0);
  REQUIRE(mlx_prefill_roots_submit(roots.value)==0); REQUIRE(mlx_prefill_roots_wait(roots.value)==0);
  REQUIRE(mlx_prefill_roots_query(roots.value)==0); REQUIRE(mlx_prefill_roots_validate(roots.value)==0);
  CHECK(output.data<float>()[0]==6.0f); CHECK(output.data<float>()[1]==1.0f); CHECK(output.data<float>()[2]==11.0f);
  CHECK(mlx_prefill_roots_append(roots.value,view(source))==5); CHECK(mlx_prefill_roots_submit(roots.value)==5);
  Roots empty; REQUIRE(empty.create(0)==0); REQUIRE(mlx_prefill_roots_submit(empty.value)==0);
  REQUIRE(mlx_prefill_roots_wait(empty.value)==0); REQUIRE(mlx_prefill_roots_validate(empty.value)==0);
}
TEST_CASE("prefill validation roots authenticate descriptor identity without arbitrary uniqueness") {
  const auto stream=default_stream(Device::cpu);
  Count count; Arena arena(1<<20,count); REQUIRE(arena.value);
  Roots roots; REQUIRE(roots.create(2,arena.value)==0); auto foreign=array(false);
  { Enter entered(arena.value);
    auto fresh=logical_not(foreign,stream); auto no_op=any(fresh,stream);
    CHECK(no_op.id()==fresh.id()); // bool scalar reduction keeps the private fresh input
    auto equal=logical_not(foreign,stream); CHECK(equal.id()!=fresh.id());
    REQUIRE(fresh.has_graph_descriptor(arena.value)); REQUIRE(equal.has_graph_descriptor(arena.value));
    REQUIRE(mlx_prefill_roots_append_validation(roots.value,view(fresh))==0);
    REQUIRE(mlx_prefill_roots_append_validation(roots.value,view(no_op))==0);
    CHECK(mlx_prefill_roots_size(roots.value)==1);
    REQUIRE(mlx_prefill_roots_append_validation(roots.value,view(equal))==0);
    CHECK(mlx_prefill_roots_size(roots.value)==2);
    CHECK(mlx_prefill_roots_append_validation(roots.value,view(foreign))==4);
    CHECK(mlx_prefill_roots_size(roots.value)==2); }
  CHECK(mlx_prefill_roots_submit(roots.value)==4); // wrong active domain, still unspent
  roots.reset(); CHECK(arena.value->occupied_bytes()==0);
  Roots ordinary; REQUIRE(ordinary.create(2)==0);
  REQUIRE(mlx_prefill_roots_append_validation(ordinary.value,view(foreign))==0);
  REQUIRE(mlx_prefill_roots_append_validation(ordinary.value,view(foreign))==0);
  CHECK(mlx_prefill_roots_size(ordinary.value)==2);
  CHECK(mlx_prefill_roots_append_validation(ordinary.value,view(foreign))==2);
}
TEST_CASE("prefill root submit refusal preserves populated roots and spent is not completion") {
  const auto stream=default_stream(Device::cpu);
  Count count; Arena arena(1<<20,count); REQUIRE(arena.value);
  size_t root_block=0;
  REQUIRE(Graph::minimum_allocation_extent(sizeof(array),alignof(array),root_block));
  Roots roots; REQUIRE(roots.create(1,arena.value)==0);
  // These are the first two blocks in the empty arena; neither absorbs a tail.
  REQUIRE(arena.value->occupied_bytes()==2*root_block);
  {
    Enter entered(arena.value);
    auto source=array({3.0f,9.0f}); auto output=add(source,array(2.0f),stream);
    const auto source_identity=source.id(), output_identity=output.id();
    REQUIRE(mlx_prefill_roots_append(roots.value,view(output))==0);
    // Test-only exhaustion, never a production fit estimate or a replacement grant.
    std::vector<void*> filler;
    while(auto* value=arena.value->try_allocate(1,1)) filler.push_back(value);
    const auto occupied=arena.value->occupied_bytes();
    CountErrors errors;
    CHECK(mlx_prefill_roots_submit(roots.value)==7); CHECK(mlx_prefill_roots_size(roots.value)==1);
    CHECK(mlx_prefill_roots_query(roots.value)==5); CHECK(mlx_prefill_roots_wait(roots.value)==5);
    CHECK(mlx_prefill_roots_validate(roots.value)==5); CHECK(mlx_prefill_roots_submit(roots.value)==5);
    CHECK(errors.count==1);
    // The moved submitted vector unwinds; the retained vector and root stay live.
    CHECK(arena.value->occupied_bytes()==occupied-root_block);
    CHECK(source.id()==source_identity); CHECK(output.id()==output_identity);
    CHECK(source.data<float>()[0]==3.0f); CHECK(source.data<float>()[1]==9.0f);
    CHECK(output.status()==array::Status::unscheduled);
    CHECK(count.retired==0);
    // Eval enters its recovery record before the graph constructor refuses.
    // All spent probes above leave that actual record pending, never complete.
    CHECK(entered.value->query().activity==submission::Activity::pending);
    CHECK(entered.value->query_records().pending);
    CHECK(entered.value->query_records().failed);
    for(auto* value:filler) arena.value->deallocate(value,1,1);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(entered.value->query_records().pending) {
      submission::progress_records(false);
      REQUIRE(std::chrono::steady_clock::now()<deadline);
      std::this_thread::yield();
    }
    CHECK_FALSE(entered.value->query_records().pending);
    CHECK(entered.value->query_records().failed);
    CHECK_FALSE(entered.value->query_records().blocked);
    CHECK(entered.value->query().activity==submission::Activity::none);
    entered.value->seal();
    while(submission::try_retire_records()==submission::RetirementPass::busy) {
      REQUIRE(std::chrono::steady_clock::now()<deadline);
      std::this_thread::yield();
    }
    CHECK(mlx_prefill_roots_size(roots.value)==1);
    CHECK(mlx_prefill_roots_query(roots.value)==5);
    CHECK(errors.count==1); CHECK(count.retired==0);
  }
  // All external array/scope aliases are gone after real recovery retirement.
  // The failed roots alone keep their native backing and arena custody alive.
  arena.reset(); CHECK(count.retired==0);
  roots.reset(); CHECK(count.retired==1);
}
