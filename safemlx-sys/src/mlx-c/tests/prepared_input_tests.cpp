#include "doctest/doctest.h"
#include "mlx/prepared_input.h"
#include "mlx/c/prepared_input.h"
#include "mlx/c/private/array.h"
#include "mlx/stream.h"
#include <array>
#include <atomic>
#include <optional>
#include <thread>

using namespace mlx::core;
namespace {
using Cause = allocator::PreparedInputCause;
struct Counts { std::atomic<size_t> retired{0}; };
void retired(void* value) { ++static_cast<Counts*>(value)->retired; }
struct Arena {
  submission::GraphQuota* value;
  Arena(size_t capacity, Counts& counts)
      : value(submission::GraphQuota::try_create(capacity, &counts, retired)) {}
  ~Arena() { reset(); }
  void reset() { if (value) { auto* old=value; value=nullptr; old->release(); } }
};
struct Leaf {
  PreparedInputLeaf* value=nullptr;
  ~Leaf() { reset(); }
  void reset() { if (value) { auto* old=value; value=nullptr; old->destroy(); } }
};
struct AllocatorProbe : allocator::Allocator {
  allocator::Allocator& actual;
  size_t calls=0, frees=0, refuse=SIZE_MAX;
  explicit AllocatorProbe(allocator::Allocator& value) : actual(value) {}
  allocator::Buffer malloc(size_t) override { std::abort(); }
  void free(allocator::Buffer) override { std::abort(); }
  size_t size(allocator::Buffer value) const override { return actual.size(value); }
  Cause allocate_prepared_input(size_t bytes, allocator::PreparedInputAllocation& out) noexcept override {
    if (calls++ == refuse) return Cause::allocation_failed;
    auto result=actual.allocate_prepared_input(bytes,out);
    if (result==Cause::success) out.allocator=this;
    return result;
  }
  void free_prepared_input(allocator::PreparedInputAllocation& value) noexcept override {
    ++frees; actual.free_prepared_input(value);
  }
};
}

TEST_CASE("prepared target compares exact retained scalar devices and rejects nulls") {
  Device cpu(Device::cpu,0), other(Device::cpu,1), gpu(Device::gpu,0);
  Stream stream(3,cpu);
  CHECK(mlx_prepared_input_target_matches(mlx_stream{&stream},mlx_device{&cpu})==1);
  CHECK(mlx_prepared_input_target_matches(mlx_stream{&stream},mlx_device{&other})==0);
  CHECK(mlx_prepared_input_target_matches(mlx_stream{&stream},mlx_device{&gpu})==0);
  CHECK(mlx_prepared_input_target_matches(mlx_stream{nullptr},mlx_device{&cpu})==0);
  CHECK(mlx_prepared_input_target_matches(mlx_stream{&stream},mlx_device{nullptr})==0);
  CHECK(sizeof(mlx_array)==2*sizeof(void*));
  CHECK(alignof(mlx_array)==alignof(void*));
}

#ifndef _LIBCPP_VERSION
TEST_CASE("prepared input unknown shared control ABI rejects its cold layout") {
  const size_t shape[]{1};const uint32_t value[]{3};
  allocator::PreparedInputFacts facts{4096,SIZE_MAX,allocator::HostTransferStorageKind::cpu};
  PreparedInputSlotLayout layout;
  CHECK_FALSE(PreparedInputLeaf::layout(facts,{value,shape,1,1,0},layout));
}
#else
TEST_CASE("prepared input actual ABI blocks hold typed values without eval or donation") {
  auto& actual=allocator::allocator(); allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts)); // explicit ordinary initialization
  AllocatorProbe allocator(actual);
  const std::array<size_t,2> shape{2,3};
  const std::array<uint32_t,6> u{1,3,5,7,11,13};
  const std::array<int32_t,6> i{-3,2,8,-7,4,19};
  const std::array<float,6> f{0.25f,-1.5f,3.0f,2.75f,8.0f,-0.125f};
  const PreparedInputSource sources[]={{u.data(),shape.data(),2,6,0},{i.data(),shape.data(),2,6,1},{f.data(),shape.data(),2,6,2}};
  for (const auto& source:sources) {
    PreparedInputSlotLayout layout; REQUIRE(PreparedInputLeaf::layout(facts,source,layout));
    Counts counts; Arena arena(layout.metadata_bytes,counts); REQUIRE(arena.value);
    Leaf leaf; REQUIRE(PreparedInputLeaf::create(allocator,facts,arena.value,source,leaf.value)==Cause::success);
    const auto& value=leaf.value->value();
    CHECK(value.is_available()); CHECK_FALSE(value.has_primitive()); CHECK(value.inputs().empty());
    CHECK_FALSE(value.event().valid()); CHECK_FALSE(value.is_donatable());
    CHECK(value.shape()==Shape{2,3}); CHECK(value.strides()==Strides{3,1});
    CHECK(value.data_shared_ptr()->original_input->capacity==layout.backing_bytes);
    CHECK(value.data_shared_ptr()->allocation_generation!=0);
    for(size_t n=0;n<6;++n) {
      if(source.kind==0) CHECK(value.data<uint32_t>()[n]==u[n]);
      else if(source.kind==1) CHECK(value.data<int32_t>()[n]==i[n]);
      else CHECK(value.data<float>()[n]==f[n]);
    }
    arena.reset(); CHECK(counts.retired==0); leaf.reset(); CHECK(counts.retired==1);
  }
  CHECK(allocator.calls==3); CHECK(allocator.frees==3);
}

TEST_CASE("prepared input rank spills use source blocks and failed block prefixes return") {
  auto& actual=allocator::allocator(); allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts)); AllocatorProbe allocator(actual);
  std::array<size_t,13> shape; shape.fill(1); shape[12]=3;
  const std::array<float,3> values{1.0f,-2.5f,4.25f};
  PreparedInputSource source{values.data(),shape.data(),shape.size(),values.size(),2};
  PreparedInputSlotLayout layout; REQUIRE(PreparedInputLeaf::layout(facts,source,layout));
  size_t failed=0;
  for(size_t capacity=128;capacity<=layout.metadata_bytes;capacity+=16) {
    Counts counts; Arena arena(capacity,counts); if(!arena.value)continue;
    Leaf leaf; const auto before=allocator.calls;
    auto result=PreparedInputLeaf::create(allocator,facts,arena.value,source,leaf.value);
    if(result==Cause::capacity) {
      ++failed; CHECK_FALSE(leaf.value); CHECK(allocator.calls==before); CHECK(arena.value->occupied_bytes()==0);
    } else {
      REQUIRE(result==Cause::success); CHECK(leaf.value->value().ndim()==13);
      CHECK(leaf.value->value().shape().get_allocator().resource()==arena.value);
      CHECK(leaf.value->value().strides().get_allocator().resource()==arena.value);
      CHECK(leaf.value->value().data<float>()[2]==4.25f); leaf.reset(); CHECK(arena.value->occupied_bytes()==0);
    }
    arena.reset(); CHECK(counts.retired==1);
  }
  CHECK(failed>0);
  Counts counts; Arena exact(layout.metadata_bytes,counts); Leaf leaf;
  REQUIRE(PreparedInputLeaf::create(allocator,facts,exact.value,source,leaf.value)==Cause::success);
}

TEST_CASE("prepared input each backing refusal preserves preceding real leaves and arena") {
  auto& actual=allocator::allocator(); allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts));
  const size_t shape[]{1,2}; const uint32_t values[]{17,29};
  const PreparedInputSource source{values,shape,2,2,0};
  PreparedInputSlotLayout layout; REQUIRE(PreparedInputLeaf::layout(facts,source,layout));
  for(size_t fail=0;fail<5;++fail) {
    AllocatorProbe allocator(actual); allocator.refuse=fail;
    Counts counts; Arena arena(layout.metadata_bytes*5,counts); std::array<Leaf,5> leaves;
    for(size_t n=0;n<fail;++n)
      REQUIRE(PreparedInputLeaf::create(allocator,facts,arena.value,source,leaves[n].value)==Cause::success);
    const auto occupied=arena.value->occupied_bytes();
    CHECK(PreparedInputLeaf::create(allocator,facts,arena.value,source,leaves[fail].value)==Cause::allocation_failed);
    CHECK_FALSE(leaves[fail].value); CHECK(arena.value->occupied_bytes()==occupied);
    CHECK(allocator.frees==0); CHECK(allocator.calls==fail+1);
    for(size_t n=0;n<fail;++n) CHECK(leaves[n].value->value().data<uint32_t>()[1]==29);
    arena.reset(); CHECK(counts.retired==(fail==0?1:0));
    for(auto& leaf:leaves) leaf.reset();
    CHECK(allocator.frees==fail); CHECK(counts.retired==1);
  }
}

TEST_CASE("prepared input backing object and weak control precede account callback") {
  auto& actual=allocator::allocator(); allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts)); AllocatorProbe allocator(actual);
  const size_t shape[]{1,2}; const float values[]{-3.0f,9.5f};
  const PreparedInputSource source{values,shape,2,2,2};
  PreparedInputSlotLayout layout; REQUIRE(PreparedInputLeaf::layout(facts,source,layout));
  Counts counts; Arena arena(layout.metadata_bytes,counts); Leaf leaf;
  REQUIRE(PreparedInputLeaf::create(allocator,facts,arena.value,source,leaf.value)==Cause::success);
  // Internal C++ weak control is not exposed by the Rust source.
  std::weak_ptr<array::Data> weak=leaf.value->value().data_shared_ptr();
  std::optional<array> alias(leaf.value->value());
  arena.reset(); leaf.reset(); CHECK(counts.retired==0); CHECK(allocator.frees==0);
  CHECK(alias->data<float>()[1]==9.5f); alias.reset();
  CHECK(weak.expired()); CHECK(allocator.frees==1); CHECK(counts.retired==0);
  std::thread worker([weak=std::move(weak)]()mutable{weak.reset();}); worker.join();
  CHECK(counts.retired==1);
}

TEST_CASE("prepared input checked invalid geometry performs no native allocation") {
  auto& actual=allocator::allocator(); allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts)); AllocatorProbe allocator(actual);
  const size_t shape[]{1,SIZE_MAX}; const float values[]{1};
  PreparedInputSlotLayout layout;
  CHECK_FALSE(PreparedInputLeaf::layout(facts,{values,shape,2,1,2},layout));
  CHECK_FALSE(PreparedInputLeaf::layout(facts,{values,shape,0,1,2},layout));
  CHECK_FALSE(PreparedInputLeaf::layout(facts,{values,shape,1,1,3},layout));
  CHECK(allocator.calls==0);
}

TEST_CASE("prepared C handles consume exact arena blocks and preserve disposition through overwrite") {
  auto& actual=allocator::allocator();allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts));AllocatorProbe allocator(actual);
  const size_t shape[]{1,3};const float values[]{0.25f,-1.5f,7.0f};
  PreparedInputSource source{values,shape,2,3,2};PreparedInputSlotLayout layout;
  REQUIRE(PreparedInputLeaf::layout(facts,source,layout));
  size_t holder=0,controls=0;REQUIRE(mlx_prepared_input_array_layout(&holder,&controls)==0);
  REQUIRE(holder>sizeof(PreparedInputArray));REQUIRE(controls>0);
  Counts counts;Arena arena(layout.metadata_bytes+holder+holder,counts);REQUIRE(arena.value);
  Leaf leaf;REQUIRE(PreparedInputLeaf::create(allocator,facts,arena.value,source,leaf.value)==Cause::success);
  mlx_array first{nullptr,nullptr},second{nullptr,nullptr},failed{nullptr,nullptr};
  REQUIRE(mlx_prepared_input_leaf_array(&first,mlx_prepared_input_leaf{leaf.value})==0);
  REQUIRE(mlx_prepared_input_leaf_array(&second,mlx_prepared_input_leaf{leaf.value})==0);
  size_t minimum=0;REQUIRE(submission::GraphQuota::allocation_extent(1,1,minimum));
  const auto remaining=arena.value->capacity()-arena.value->occupied_bytes();
  const auto filler_bytes=remaining>=minimum?remaining-minimum+1:0;
  void* filler=filler_bytes?arena.value->try_allocate(filler_bytes,1):nullptr;
  const auto occupied=arena.value->occupied_bytes();
  CHECK(mlx_prepared_input_leaf_array(&failed,mlx_prepared_input_leaf{leaf.value})==static_cast<unsigned>(Cause::capacity));
  CHECK_FALSE(failed.ctx);CHECK_FALSE(failed.prepared_owner);CHECK(arena.value->occupied_bytes()==occupied);
  CHECK(allocator.calls==1);CHECK(first.prepared_owner!=second.prepared_owner);
  CHECK(mlx_array_get_(first).data<float>()[2]==7.0f);
  auto* disposition=first.prepared_owner;
  REQUIRE(mlx_array_set(&first,second)==0);CHECK(first.prepared_owner==disposition);
  mlx_array ordinary=mlx_array_new();REQUIRE(mlx_array_set(&ordinary,first)==0);
  CHECK_FALSE(ordinary.prepared_owner);CHECK(ordinary.ctx!=first.ctx);
  // An ordinary wrapper may overwrite a prepared value; its own allocation
  // disposition remains ordinary and destruction never claims a prepared block.
  REQUIRE(mlx_array_set(&ordinary,second)==0);CHECK_FALSE(ordinary.prepared_owner);
  if(filler)arena.value->deallocate(filler,filler_bytes,1);
  arena.reset();leaf.reset();CHECK(counts.retired==0);
  CHECK(mlx_array_free(first)==0);CHECK(counts.retired==0);
  CHECK(mlx_array_free(ordinary)==0);CHECK(counts.retired==0);
  CHECK(mlx_array_get_(second).data<float>()[1]==-1.5f);
  std::thread final([second]{CHECK(mlx_array_free(second)==0);});final.join();
  CHECK(allocator.calls==1);CHECK(allocator.frees==1);CHECK(counts.retired==1);
}
TEST_CASE("prepared handle rejects a genuinely exhausted source arena without publication") {
  auto& actual=allocator::allocator();allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts));AllocatorProbe allocator(actual);
  const size_t shape[]{1,2};const uint32_t values[]{17,29};
  PreparedInputSource source{values,shape,2,2,0};PreparedInputSlotLayout layout;
  REQUIRE(PreparedInputLeaf::layout(facts,source,layout));size_t holder=0,controls=0;
  REQUIRE(PreparedInputArray::layout(holder,controls));
  Counts counts;Arena arena(layout.metadata_bytes+holder,counts);REQUIRE(arena.value);
  Leaf leaf;REQUIRE(PreparedInputLeaf::create(allocator,facts,arena.value,source,leaf.value)==Cause::success);
  size_t minimum=0;REQUIRE(submission::GraphQuota::allocation_extent(1,1,minimum));
  const auto remaining=arena.value->capacity()-arena.value->occupied_bytes();REQUIRE(remaining>=minimum);
  const auto fill_bytes=remaining-minimum+1;void* fill=arena.value->try_allocate(fill_bytes,1);REQUIRE(fill);
  mlx_array out{nullptr,nullptr};
  CHECK(mlx_prepared_input_leaf_array(&out,mlx_prepared_input_leaf{leaf.value})==static_cast<unsigned>(Cause::capacity));
  CHECK_FALSE(out.ctx);CHECK_FALSE(out.prepared_owner);CHECK(allocator.calls==1);
  arena.value->deallocate(fill,fill_bytes,1);
  REQUIRE(mlx_prepared_input_leaf_array(&out,mlx_prepared_input_leaf{leaf.value})==0);
  CHECK(mlx_array_get_(out).data<uint32_t>()[1]==29);CHECK(mlx_array_free(out)==0);
  leaf.reset();CHECK(arena.value->occupied_bytes()==0);arena.reset();CHECK(counts.retired==1);
}

#endif
