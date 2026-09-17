#include "doctest/doctest.h"
#include "mlx/c/array.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/host_transfer.h"
#include "mlx/ops.h"
#include "mlx/prepared_input.h"
#include "mlx/primitives.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <optional>
#include <vector>
using namespace mlx::core;
extern size_t native_recovery_without_allocations(void (*)(void*), void*);
namespace {
mlx_array borrowed(const array& value) {
  return mlx_array{const_cast<array*>(&value), nullptr};
}
struct Handler {
  unsigned calls{0};
  Handler() { mlx_set_error_handler([](const char*, void* p) { ++*static_cast<unsigned*>(p); }, &calls, [](void*) {}); }
  ~Handler() { mlx_set_error_handler(nullptr, nullptr, nullptr); }
};
mlx_array_descriptor observe(const array& value) {
  mlx_array_descriptor out{};
  REQUIRE(mlx_array_descriptor_read(&out, borrowed(value)) == 0);
  return out;
}
void ordinary_agrees(const array& value, const mlx_array_descriptor& out) {
  bool known=false, host=false; uint64_t identity=0; size_t bytes=0;
  REQUIRE(mlx_array_allocation_info(&known, &host, &identity, &bytes, borrowed(value)) == 0);
  CHECK(known==out.known); CHECK(host==out.host_transfer);
  CHECK(identity==out.identity); CHECK(bytes==out.allocation_bytes);
}
}

TEST_CASE("fixed descriptor native and lazy view facts reuse ordinary full backing classification") {
  const auto stream=default_stream(Device::cpu);
  array source({2,-3,5,7,11,-13}, Shape{2,3}, int32);
  auto lazy=slice(source,{0,2},{2,3},{1,1},stream);
  Handler handler;
  auto root=observe(source);ordinary_agrees(source,root);
  CHECK(root.rank==2);CHECK(root.logical_bytes==24);CHECK(root.dtype==MLX_INT32);
  auto pending=observe(lazy);ordinary_agrees(lazy,pending);
  CHECK_FALSE(pending.known);CHECK(pending.logical_bytes==8);
  lazy.eval();
  auto view=observe(lazy);ordinary_agrees(lazy,view);
  CHECK(view.known);CHECK(view.identity==root.identity);CHECK(view.allocation_bytes==root.allocation_bytes);
  CHECK(view.allocation_bytes>view.logical_bytes);
  std::array<int,2> shape{-1,-1};
  CHECK(mlx_array_descriptor_fill_shape(borrowed(lazy),&view,shape.data(),shape.size())==0);
  CHECK(shape==std::array<int,2>{2,1});CHECK(handler.calls==0);
}

TEST_CASE("fixed descriptor refuses equal content substitution and changed state before destination writes") {
  const auto stream=default_stream(Device::cpu);
  array a({17,19},Shape{2},int32), b({17,19},Shape{2},int32);
  auto lazy=square(a,stream); Handler handler;
  auto expected=observe(a);int destination=97;
  CHECK(mlx_array_descriptor_fill_shape(borrowed(b),&expected,&destination,1)==5);
  CHECK(destination==97);
  CHECK(mlx_array_descriptor_fill_shape(borrowed(a),&expected,&destination,0)==6);
  CHECK(destination==97);
  auto pending=observe(lazy);lazy.eval();
  CHECK(mlx_array_descriptor_fill_shape(borrowed(lazy),&pending,&destination,1)==5);
  CHECK(destination==97);CHECK(handler.calls==0);
}

TEST_CASE("fixed descriptor scalar zero and unbounded rank fill retain exact caller geometry") {
  array scalar(23,int32);
  array empty(std::initializer_list<int32_t>{},Shape{2,0,3},int32);
  const Shape dimensions(37,1);
  array wide({29},dimensions,int32);
  Handler handler;
  auto zero=observe(scalar);
  CHECK(zero.rank==0);CHECK(mlx_array_descriptor_fill_shape(borrowed(scalar),&zero,nullptr,0)==0);
  auto vacant=observe(empty);ordinary_agrees(empty,vacant);
  CHECK(vacant.rank==3);CHECK(vacant.logical_bytes==0);
  auto many=observe(wide);std::vector<int> shape(many.rank,-1);
  CHECK(mlx_array_descriptor_fill_shape(borrowed(wide),&many,shape.data(),shape.size())==0);
  CHECK(std::equal(shape.begin(),shape.end(),dimensions.begin(),dimensions.end()));CHECK(handler.calls==0);
  array equal(23,int32);
  CHECK(mlx_array_descriptor_fill_shape(borrowed(equal),&zero,nullptr,0)==5);
}

TEST_CASE("fixed descriptor checked errors never enter ordinary native error transport") {
  array source({31,37},Shape{2},int32);Handler handler;
  mlx_array_descriptor sentinel{};sentinel.logical_bytes=101;
  CHECK(mlx_array_descriptor_read(nullptr,borrowed(source))==1);
  CHECK(mlx_array_descriptor_read(&sentinel,mlx_array{nullptr,nullptr})==2);
  CHECK(sentinel.logical_bytes==101);
  auto expected=observe(source);
  CHECK(mlx_array_descriptor_fill_shape(borrowed(source),nullptr,nullptr,0)==1);
  CHECK(mlx_array_descriptor_fill_shape(borrowed(source),&expected,nullptr,1)==7);
  array moved(std::move(source));
  CHECK(mlx_array_descriptor_read(&sentinel,borrowed(source))==2);
  CHECK(handler.calls==0);
  bool known=false,host=false;uint64_t identity=0;size_t bytes=0;
  CHECK(mlx_array_allocation_info(&known,&host,&identity,&bytes,mlx_array{nullptr,nullptr})==1);
  CHECK(handler.calls==1); // preserved ordinary adapter and its actual error
  array invalid(std::initializer_list<int32_t>{},Shape{0},Dtype(Dtype::Val::int32,8));
  sentinel.logical_bytes=103;
  CHECK(mlx_array_descriptor_read(&sentinel,borrowed(invalid))==4);
  CHECK(sentinel.logical_bytes==103);CHECK(handler.calls==1);
}

TEST_CASE("fixed descriptor custom backing remains unknown and host transfer capacity stays certified") {
  auto custom=allocator::malloc(2*sizeof(int32_t));
  array foreign(custom,Shape{2},int32,[](allocator::Buffer value){allocator::free(value);});
  foreign.data<int32_t>()[0]=41;foreign.data<int32_t>()[1]=-43;
  HostTransferBuffer host(Shape{2},int32,allocator::HostTransferPolicy::transfer);
  const auto stream=default_stream(Device::cpu);
  array source({47,-53},Shape{2},int32);
  // Retain copy_to_host's actual output array so its certified backing can be
  // inspected after completion; copy_from_host creates a separate device array.
  auto certified=array(source.shape(),source.dtype(),
      submission::make_graph_primitive<CopyToHostTransfer>(stream,host),
      {contiguous(source,false,stream)});
  certified.eval();
  CHECK(static_cast<int32_t*>(host.data())[0]==47);
  CHECK(static_cast<int32_t*>(host.data())[1]==-53);
  Handler handler;
  auto unknown=observe(foreign);ordinary_agrees(foreign,unknown);
  CHECK_FALSE(unknown.known);CHECK(unknown.logical_bytes==8);
  auto known=observe(certified);ordinary_agrees(certified,known);
  CHECK(known.known);CHECK(known.host_transfer);CHECK(known.identity!=0);
  CHECK(known.allocation_bytes>=known.logical_bytes);CHECK(handler.calls==0);
}

#ifdef _LIBCPP_VERSION
TEST_CASE("fixed descriptor borrows actual original native input arena capacity without adding an owner") {
  auto& native_allocator=allocator::allocator();allocator::PreparedInputFacts facts;
  REQUIRE(native_allocator.prepare_input_runtime(facts));
  const std::array<size_t,2> shape{2,2};const std::array<int32_t,4> values{59,-61,67,71};
  const PreparedInputSource source{values.data(),shape.data(),2,4,1};
  PreparedInputSlotLayout layout;REQUIRE(PreparedInputLeaf::layout(facts,source,layout));
  std::atomic<size_t> retired{0};
  auto* arena=submission::GraphQuota::try_create(layout.metadata_bytes,&retired,[](void* p){++*static_cast<std::atomic<size_t>*>(p);});
  REQUIRE(arena);
  PreparedInputLeaf* leaf=nullptr;
  REQUIRE(PreparedInputLeaf::create(native_allocator,facts,arena,source,leaf)==allocator::PreparedInputCause::success);
  auto observed=observe(leaf->value());ordinary_agrees(leaf->value(),observed);
  CHECK(observed.known);CHECK_FALSE(observed.host_transfer);
  CHECK(observed.allocation_bytes==layout.backing_bytes);CHECK(observed.logical_bytes==sizeof(values));
  std::array<int,2> destination{-1,-1};
  CHECK(mlx_array_descriptor_fill_shape(borrowed(leaf->value()),&observed,destination.data(),2)==0);
  arena->release();CHECK(retired==0);
  leaf->destroy();CHECK(retired==1); // witness has no hidden native/account owner
}
#endif

TEST_CASE("fixed descriptor controls expose concrete scalar witnesses without source or runtime work") {
  Handler handler;
  const auto first=mlx_array_descriptor_control_bytes();
  CHECK(first>=2*sizeof(mlx_array_descriptor));
  CHECK(mlx_array_descriptor_control_bytes()==first);
  CHECK(sizeof(mlx_array)==2*sizeof(void*));
  CHECK(handler.calls==0);
}

TEST_CASE("fixed descriptor query and fill use no native allocation or legacy error transport") {
  array source({73,-79,83,89},Shape{2,2},int32);
  struct Probe {
    mlx_array source;
    mlx_array_descriptor descriptor{};
    std::array<int,2> shape{-1,-1};
    uint32_t read{99}, fill{99}, refused{99};
  } probe{borrowed(source)};
  Handler handler;
  const auto allocations=native_recovery_without_allocations([](void* raw) {
    auto& p=*static_cast<Probe*>(raw);
    p.read=mlx_array_descriptor_read(&p.descriptor,p.source);
    p.fill=mlx_array_descriptor_fill_shape(p.source,&p.descriptor,p.shape.data(),p.shape.size());
    p.refused=mlx_array_descriptor_fill_shape(p.source,&p.descriptor,p.shape.data(),0);
  }, &probe);
  CHECK(allocations==0);CHECK(probe.read==0);CHECK(probe.fill==0);CHECK(probe.refused==6);
  CHECK(probe.shape==std::array<int,2>{2,2});CHECK(handler.calls==0);
}
