// Copyright © 2026 Eredu contributors.
#include <doctest/doctest.h>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"
#include "mlx/backend/cpu/binary_storage.h"
#include "mlx/backend/cpu/copy_storage.h"
#include "mlx/ordinary_control.h"
#include "mlx/graph_quota.h"
#include "mlx/record_quota.h"
#include "mlx/submission.h"
using namespace mlx::core;
namespace {
void check(bool value,const char* reason) { if(!value) throw std::runtime_error(reason); }
struct Counts;
struct Slot { Counts* owner{}; size_t charge{}; bool live{}; };
struct Counts {
  size_t charged{}, births{}, retired{}, total_controls{}, payload{};
  bool reject{}, owner_retired{};
  std::array<Slot,1024> slots{};
};
bool observe(void* context, uint64_t, size_t bytes, size_t controls,
    allocator::MemoryPlacement placement, void** owner,
    void (**release)(void*), allocator::PhysicalBackingPublish* publish) {
  auto& count=*static_cast<Counts*>(context);
  check(placement.kind==1,"CPU source was not host placement");
  if(count.reject) { *owner=&count; *release=[](void*){}; *publish=nullptr; return false; }
  for(auto& slot:count.slots) if(!slot.live) {
    slot={&count,controls,true}; count.charged+=controls;count.total_controls+=controls;count.payload+=bytes;++count.births;
    *owner=&slot;*publish=nullptr;
    *release=[](void* value){auto& slot=*static_cast<Slot*>(value);check(slot.live,"duplicate retirement");slot.live=false;slot.owner->charged-=slot.charge;++slot.owner->retired;};
    return true;
  }
  throw std::runtime_error("test observation slots exhausted");
}
submission::Scope* begin(Counts& count) {
  auto* scope=new submission::Scope();
  check(scope->bind_physical_observer(&count,observe,[](void* c){static_cast<Counts*>(c)->owner_retired=true;}),"scope bind");
  check(!scope->original_controls_enabled() && !scope->original_controls_required(),"ordinary observer granted original authority");
  return scope;
}
}
namespace {
void ordinary_control_source_regression() {
  Counts a,b;
  auto* first=begin(a);
  std::optional<submission::GraphAllocator<int>> source(std::in_place);
  submission::OrdinaryControlLayout one;
  check(submission::ordinary_control_layout(sizeof(int)*4,alignof(int),one),"layout");
  int* initial=source->allocate(4);
  check(a.charged==one.host_controls && a.births==1,"actual one-allocation quote differs");
  auto* second=begin(b);
  int* grown=source->allocate(9);
  check(a.births==2 && b.births==0,"container changed funding source in a foreign current scope");
  a.reject=true;
  bool refused=false;
  try { (void)source->allocate(1); } catch(const allocator::PhysicalBackingAdmissionError&) { refused=true; }
  check(refused && a.births==2 && b.births==0,"refused source fell back to the foreign scope");
  a.reject=false;
  source->deallocate(initial,4);source->deallocate(grown,9);
  check(a.charged==0 && a.retired==2,"explicit blocks did not retire");
  // New independent construction captures the genuinely current second scope.
  auto owned=std::allocate_shared<int>(submission::GraphAllocator<int>{},41);
  std::shared_ptr<const int> alias(owned,owned.get());owned.reset();
  check(b.births==1 && b.charged>sizeof(int),"shared control was not observed");
  second->seal();second->release();
  check(!b.owner_retired && *alias==41,"source retired before last shared alias");
  alias.reset();check(b.owner_retired && b.charged==0 && b.retired==1,"last alias did not release exact source");
  first->seal();first->release();
  check(!a.owner_retired,"allocator lost its actual source association");
  source.reset();check(a.owner_retired,"empty allocator source failed to retire");
  Counts c,d;
  auto* record_source=begin(c);
  std::optional<submission::RecordAllocator<int>> records(std::in_place);
  auto* record_foreign=begin(d);
  int* record_memory=records->allocate(7);
  check(c.births==1 && d.births==0,"Record allocator changed source");
  records->deallocate(record_memory,7);
  record_foreign->seal();record_foreign->release();
  record_source->seal();record_source->release();
  check(d.owner_retired && !c.owner_retired && c.charged==0,"Record source lifetime differs");
  records.reset();check(c.owner_retired,"Record allocator source leaked");
  Counts callback_source;
  auto* callback_scope=begin(callback_source);
  auto callable=[state=std::array<int,8>{1,2,3,4,5,6,7,8}](int index){return state[index];};
  const auto callback_bytes=submission::graph_function_control_bytes<int(int),decltype(callable)>();
  submission::OrdinaryControlLayout callback_layout;
  check(callback_bytes && submission::ordinary_control_layout(callback_bytes,alignof(std::max_align_t),callback_layout),"callback source layout");
  auto callback=submission::make_graph_function<int(int)>(std::move(callable));
  check(callback_source.births==1 && callback_source.charged==callback_layout.host_controls,"callable capture bypassed originating payer");
  auto retained_callback=callback;callback={};
  callback_scope->seal();callback_scope->release();
  check(!callback_source.owner_retired && retained_callback(6)==7,"callable source retired before alias");
  retained_callback={};
  check(callback_source.owner_retired && callback_source.charged==0,"callable source failed to retire");
  submission::OrdinaryControlPopulation bound{};
  check(!submission::ordinary_cpu_eval_control_layout({1,2,3,3,3,1,4},bound),"malformed traversal accepted");
  check(!submission::ordinary_frontend_control_layout(SIZE_MAX,1,2,4,bound),"overflow accepted");
  check(submission::ordinary_cpu_eval_control_layout({2,25,19,27,19,1,6},bound) && bound.control_allocations>0,"finite CPU source missing");
  set_default_device(Device(Device::cpu,0));
  const auto stream=default_stream(Device(Device::cpu,0));
  (void)scheduler::prepare_cpu_stream(stream);
  array input({1.f,2.f,3.f},{3},float32);
  cpu::BinaryEvalStorage multiply_source,add_source;
  cpu::CopyEvalStorage completion_source;
  check(cpu::binary_eval_layout(cpu::BinaryEvalKind::multiply,float32,1,3,false,multiply_source),"multiply source");
  check(cpu::binary_eval_layout(cpu::BinaryEvalKind::add,float32,1,3,false,add_source),"add source");
  check(cpu::copy_eval_layout(0,1,false,false,completion_source),"completion source");
  submission::OrdinaryControlPopulation frontend,evaluation,worker,vector,wait;
  check(submission::ordinary_frontend_control_layout(10,0,1,4,frontend),"frontend source");
  check(submission::ordinary_cpu_eval_control_layout({1,4,3,5,3,1,6},evaluation),"eval source");
  check(submission::ordinary_cpu_dispatch_envelope(multiply_source.allocation_extents+
      multiply_source.worker_graph_extents+add_source.allocation_extents+
      add_source.worker_graph_extents+completion_source.allocation_extents,worker),"dispatch source");
  check(submission::ordinary_array_vector_control_layout(1,vector),"vector source");
  check(submission::ordinary_cpu_wait_control_layout(wait),"wait source");
  const auto quote=wait.observed_host_controls+frontend.observed_host_controls+evaluation.observed_host_controls+
      worker.observed_host_controls+vector.observed_host_controls+
      3*allocator::ordinary_allocation_control_bytes();
  Counts numerical;
  auto* scope=begin(numerical);
  {
    auto result=add(multiply(input,input,stream),input,stream);
    auto done=async_eval_with_completion(ArrayVector{result});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(!done.is_complete()) {check(std::chrono::steady_clock::now()<deadline,"completion timeout");std::this_thread::yield();}
    done.wait(stream);
    check(result.data<float>()[0]==2.f && result.data<float>()[1]==6.f && result.data<float>()[2]==12.f,"nonzero numerical result");
    check(numerical.total_controls<=quote,"actual ordinary controls exceed source allowance");
  }
  scope->seal();scope->release();
  for(int pass=0;pass<4;++pass) {submission::progress_records(false);submission::try_retire_records();}
  clear_cache();
  check(numerical.owner_retired && numerical.charged==0 && numerical.births==numerical.retired,"completed ordinary graph did not retire");
  std::cout<<"ordinary source ownership, refusal, alias and nonzero Eval PASS; observed controls="<<numerical.total_controls<<" allowance="<<quote<<"\n";
}

}
TEST_CASE("Ordinary native controls preserve exact source refusal and final alias retirement") {
  REQUIRE_NOTHROW(ordinary_control_source_regression());
}
