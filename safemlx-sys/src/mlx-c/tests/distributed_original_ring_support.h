#pragma once
// Run as two actual localhost Ring processes; see run_local_ring.py.
// Graph/Record/backing capacities come from the actual retained operation source.
// Runtime, source-host and role authorities are explicit fixture owners.
#include "mlx/c/distributed_group.h"
#include "mlx/c/event.h"
#include "mlx/c/original_buffer.h"
#include "mlx/c/private/array.h"
#include "mlx/mlx.h"
#include "mlx/distributed/ops.h"
#include "mlx/backend/cpu/distributed_storage.h"
#include "mlx/record_quota.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/submission.h"
#include "mlx/failure.h"
#include "mlx/transforms_impl.h"
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>


namespace original_ring_test_support {
using namespace mlx::core;
namespace dist=mlx::core::distributed;
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
struct GraphRelease {void operator()(submission::GraphQuota* p)const noexcept{p->release();}};
struct RecordRelease {void operator()(submission::RecordQuota* p)const noexcept{p->release();}};
struct ScopeRelease {void operator()(submission::Scope* p)const noexcept{p->seal();p->release();}};
struct Role {
  std::unique_ptr<submission::GraphQuota,GraphRelease> graph;
  std::unique_ptr<submission::RecordQuota,RecordRelease> records;
  static FailureCarrierRef make_failure(){
    static unsigned source=0;return FailureCarrier::create(&source,[](void*){});
  }
  FailureCarrierRef failure{make_failure()};
  std::unique_ptr<submission::Scope,ScopeRelease> scope{
      new submission::Scope(nullptr,nullptr,records.get(),graph.get())};
  explicit Role(const mlx_distributed_cpu_completion_storage& source)
      :graph(submission::GraphQuota::create(source.graph_capacity,nullptr,nullptr)),
       records(submission::RecordQuota::create(source.record_capacity,nullptr,nullptr)) {
    require(bool(failure),"failure carrier");require(scope->enable_scoped_observation(),"scoped observation");
    require(scope->require_original_controls()==submission::NativeControlFailure::none,"require original");
    require(scope->bind_failure(failure),"bind failure");
    require(scope->enable_original_controls()==submission::NativeControlFailure::none,"enable original");
  }
  void settle(){
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(scope->query_records().pending){
      require(std::chrono::steady_clock::now()<deadline,"record completion timeout");
      require(scope->progress_scoped()==submission::ScopedProgress::observed,"record completion");
      std::this_thread::yield();
    }
    for(;;){
      const auto result=scope->retire_scoped_records();
      if(result==submission::ScopedProgress::observed)break;
      require(result==submission::ScopedProgress::busy && std::chrono::steady_clock::now()<deadline,"record retirement");
      std::this_thread::yield();
    }
  }
};
struct BufferBudget {
  mlx_original_buffer_budget value{};
  BufferBudget(mlx_prepared_input_runtime runtime,size_t capacity,const std::shared_ptr<std::atomic<unsigned>>& retired){
    auto owner=std::make_unique<std::shared_ptr<std::atomic<unsigned>>>(retired);
    require(mlx_original_buffer_budget_new_retaining(&value,runtime,capacity,owner.get(),
        [](void* p){std::unique_ptr<std::shared_ptr<std::atomic<unsigned>>> held(
            static_cast<std::shared_ptr<std::atomic<unsigned>>*>(p));(*held)->fetch_add(1);}, nullptr)==0,"buffer budget");
    owner.release();
  }
  ~BufferBudget(){mlx_original_buffer_budget_release(value);}
};
struct Output {mlx_array value{};~Output(){if(value.ctx)mlx_array_free(value);}};
struct Event {
  mlx_submission_observer observer{};
  mlx_operation_event value{};
  explicit Event(Stream& stream,size_t roots=1){
    require(mlx_submission_observer_current(&observer)==0,"observer");
    require(mlx_operation_event_new_exact(&value,observer,{&stream},roots)==0,"event roots");
  }
  ~Event(){if(value.ctx)mlx_operation_event_free(value);if(observer.ctx)mlx_submission_observer_release(observer);}
};
}
