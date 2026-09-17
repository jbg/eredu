#include "doctest/doctest.h"
#include "mlx/prepared_input.h"
#include "mlx/memory.h"
#include "mlx/backend/metal/device.h"

using namespace mlx::core;
namespace {
struct WiredLimit {
  size_t old;
  WiredLimit():old(set_wired_limit(0)){}
  ~WiredLimit(){set_wired_limit(old);}
};
struct Source {
  submission::GraphQuota* arena;
  PreparedInputLeaf* leaf=nullptr;
  ~Source(){reset();if(arena)arena->release();}
  void reset(){if(leaf){auto* old=leaf;leaf=nullptr;old->destroy();}}
};
}
TEST_CASE("prepared input Metal records preserve unwired grow shrink and direct release") {
  auto& actual=allocator::allocator(); allocator::PreparedInputFacts facts;
  REQUIRE(actual.prepare_input_runtime(facts));
  auto& residency=metal::device(Device::gpu).residency_set();
  if(!residency.mtl_residency_set()) {
    WARN("actual Metal device has no residency-set API; wired transition remains unexecuted");return;
  }
  WiredLimit restore;
  const size_t shape[]{1,3};const float values[]{0.5f,-2.f,3.75f};
  const PreparedInputSource source{values,shape,2,3,2};
  PreparedInputSlotLayout layout; REQUIRE(PreparedInputLeaf::layout(facts,source,layout));
  Source first{submission::GraphQuota::try_create(layout.metadata_bytes,nullptr,nullptr)};
  Source second{submission::GraphQuota::try_create(layout.metadata_bytes,nullptr,nullptr)};
  REQUIRE(first.arena);REQUIRE(second.arena);
  REQUIRE(PreparedInputLeaf::create(actual,facts,first.arena,source,first.leaf)==allocator::PreparedInputCause::success);
  REQUIRE(PreparedInputLeaf::create(actual,facts,second.arena,source,second.leaf)==allocator::PreparedInputCause::success);
  auto* a=first.leaf->value().data_shared_ptr()->original_input;
  auto* b=second.leaf->value().data_shared_ptr()->original_input;
  CHECK(a->residency.installed);CHECK(b->residency.installed);
  CHECK_FALSE(a->residency.wired);CHECK_FALSE(b->residency.wired);
  set_wired_limit(layout.backing_bytes*2);
  CHECK(a->residency.wired);CHECK(b->residency.wired);
  set_wired_limit(0);
  CHECK_FALSE(a->residency.wired);CHECK_FALSE(b->residency.wired);
  first.reset(); // removal of an unwired intrusive member, no ordinary hash insertion
  CHECK(second.leaf->value().data<float>()[2]==3.75f);
  set_wired_limit(layout.backing_bytes);
  CHECK(b->residency.wired);second.reset(); // wired direct release and commit
  CHECK(residency.mtl_residency_set()->allocationCount()==0);
}
