/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>
#include <algorithm>
#include "mlx/c/event.h"
#include "mlx/distributed/constructor.h"
#include "mlx/distributed/ops.h"
#include "mlx/transforms_impl.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/scheduler.h"

#include "mlx/c/distributed_group.h"
#include "mlx/c/private/distributed_storage.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/distributed/distributed_impl.h"

extern "C" mlx_distributed_group mlx_distributed_group_new(void) {
  try {
    return mlx_distributed_group_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return {nullptr};
  }
}

extern "C" int mlx_distributed_group_free(mlx_distributed_group group) {
  try {
    mlx_distributed_group_free_(group);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int
mlx_distributed_init(mlx_distributed_group* res, bool strict, const char* bk) {
  try {
    if (bk) {
      mlx_distributed_group_set_(
          *res, mlx::core::distributed::init(strict, bk));
    } else {
      mlx_distributed_group_set_(*res, mlx::core::distributed::init(strict));
    }
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" void mlx_distributed_group_mark_terminal_submission(
    mlx_distributed_group group) {
  if (group.ctx) {
    static_cast<mlx::core::distributed::Group*>(group.ctx)
        ->mark_terminal_submission();
  }
}

extern "C" bool mlx_distributed_group_terminal_submission(
    mlx_distributed_group group) {
  return !group.ctx ||
      static_cast<mlx::core::distributed::Group*>(group.ctx)->terminal_submission();
}

extern "C" bool mlx_distributed_group_storage_inventory(
    mlx_distributed_storage_inventory* out, mlx_distributed_group group) {
  if (!out) return false;
  *out = {};
  if (!group.ctx) return false;
  mlx::core::distributed::GroupStorageInventory value;
  if (!static_cast<const mlx::core::distributed::Group*>(group.ctx)
          ->storage_inventory(value)) return false;
  out->kind = static_cast<unsigned>(value.kind);
  out->unresolved = value.unresolved;
  out->wrapper_bytes = sizeof(mlx::core::distributed::Group);
  out->implementation_bytes = value.implementation_bytes;
  out->socket_handles = value.socket_handles;
  out->socket_capacity_bytes = value.socket_capacity_bytes;
  out->buffer_bytes = value.buffer_bytes;
  out->pool_workers = value.pool_workers;
  out->pool_worker_capacity_bytes = value.pool_worker_capacity_bytes;
  out->socket_workers = value.socket_workers;
  out->socket_map_buckets = value.socket_map_buckets;
  return true;
}

extern "C" bool mlx_distributed_group_same_implementation(
    mlx_distributed_group left, mlx_distributed_group right) {
  return left.ctx && right.ctx &&
      static_cast<const mlx::core::distributed::Group*>(left.ctx)
          ->same_implementation(
              *static_cast<const mlx::core::distributed::Group*>(right.ctx));
}

extern "C" size_t mlx_distributed_group_storage_inventory_controls(void) {
  using namespace mlx::core::distributed;
  return sizeof(GroupStorageInventory) + sizeof(mlx_distributed_storage_inventory) +
      sizeof(mlx_distributed_group) * 3 + sizeof(void*) * 8 + sizeof(bool) * 3 +
      sizeof(size_t) * 12;
}

extern "C" bool mlx_distributed_group_worker_storage(mlx_distributed_worker_storage* out,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer) {
  if (!out || !group.ctx || !input.ctx || operation > 5) return false;
  mlx::core::distributed::GroupWorkerStorage value;
  if (!static_cast<const mlx::core::distributed::Group*>(group.ctx)->worker_storage(
        static_cast<mlx::core::distributed::GroupWorkerOperation>(operation),
        *static_cast<const mlx::core::array*>(input.ctx), peer, value)) return false;
  *out = {value.pool_jobs, value.socket_attempts, value.destination_arrays,
      value.task_graph_extent, value.destination_graph_extent, value.controls};
  return true;
}
extern "C" size_t mlx_distributed_group_worker_storage_controls(mlx_distributed_group group) {
  return sizeof(mlx_distributed_worker_storage) + sizeof(mlx::core::distributed::GroupWorkerStorage) +
      sizeof(mlx_distributed_group) + sizeof(mlx_array) + sizeof(unsigned) + sizeof(int) +
      sizeof(void*) * 4 + sizeof(bool) * 2 +
      (group.ctx ? static_cast<const mlx::core::distributed::Group*>(group.ctx)
          ->worker_storage_inspection_controls() : 0);
}

extern "C" bool mlx_distributed_group_dispatch_storage(mlx_distributed_dispatch_storage* out,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer) {
  if (!out || !group.ctx || !input.ctx || operation > 5) return false;
  mlx::core::distributed::GroupDispatchStorage value;
  if (!static_cast<const mlx::core::distributed::Group*>(group.ctx)->dispatch_storage(
        static_cast<mlx::core::distributed::GroupWorkerOperation>(operation),
        *static_cast<const mlx::core::array*>(input.ctx), peer, value)) return false;
  *out = {value.task_bytes,value.task_alignment,value.graph_extent,value.controls};
  return true;
}
extern "C" size_t mlx_distributed_group_dispatch_storage_controls(mlx_distributed_group group) {
  return sizeof(mlx_distributed_dispatch_storage) + sizeof(mlx::core::distributed::GroupDispatchStorage) +
      sizeof(mlx_distributed_group) + sizeof(mlx_array) + sizeof(unsigned) + sizeof(int) +
      sizeof(void*) * 4 + sizeof(bool) * 2 +
      (group.ctx ? static_cast<const mlx::core::distributed::Group*>(group.ctx)
          ->dispatch_storage_inspection_controls() : 0);
}

extern "C" int mlx_distributed_group_rank(mlx_distributed_group group) {
  try {
    return mlx_distributed_group_get_(group).rank();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" int mlx_distributed_group_size(mlx_distributed_group group) {
  try {
    return mlx_distributed_group_get_(group).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" int mlx_distributed_group_communication_stream(
    mlx_stream* res,
    mlx_distributed_group group) {
  try {
    mlx_distributed_group_get_(group).ensure_submission_available();
    mlx_stream_set_(
        *res,
        mlx_distributed_group_get_(group).raw_group()->communication_stream());
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_distributed_group_split(
    mlx_distributed_group* res,
    mlx_distributed_group group,
    int color,
    int key) {
  try {
    mlx_distributed_group_set_(
        *res, mlx_distributed_group_get_(group).split(color, key));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" bool mlx_distributed_is_available(const char* bk) {
  try {
    if (bk) {
      return mlx::core::distributed::is_available(bk);
    } else {
      return mlx::core::distributed::is_available();
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return false;
  }
}

extern "C" size_t mlx_distributed_group_constructor_storage_controls(void) {
  return mlx::core::distributed::constructor_storage_inspection_controls() +
      sizeof(mlx_distributed_constructor_storage) + sizeof(mlx_distributed_group) +
      sizeof(mlx_array) * 3 + sizeof(mlx_submission_observer) + sizeof(mlx_stream) +
      sizeof(void*) * 8 + sizeof(unsigned) * 3 + sizeof(int) + sizeof(bool) * 2 +
      sizeof(mlx::core::submission::NativeControlBinding) + sizeof(mlx::core::scheduler::CpuStreamToken) +
      sizeof(mlx::core::submission::GraphConstructionContext) +
      sizeof(mlx::core::ScopedEvaluation) + sizeof(mlx::core::submission::GraphFailure) +
      sizeof(mlx::core::submission::Status) + sizeof(mlx::core::submission::NativeControlFailure);
}
extern "C" bool mlx_distributed_group_constructor_storage(mlx_distributed_constructor_storage* out,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer) {
  using namespace mlx::core::distributed;
  if (!out || !group.ctx || !input.ctx || operation > 5) return false;
  GroupConstructorStorage native;
  if (!constructor_storage(*static_cast<const Group*>(group.ctx),
        *static_cast<const mlx::core::array*>(input.ctx),static_cast<GroupWorkerOperation>(operation),peer,native)) return false;
  return mlx_distributed_publish_constructor_storage(out,native);
}
extern "C" unsigned mlx_distributed_construct_original(mlx_array* out, mlx_submission_observer observer,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer, mlx_stream stream) {
  using namespace mlx::core;
  using namespace mlx::core::submission;
  if (!out || out->ctx || out->prepared_owner || !group.ctx || !input.ctx || !stream.ctx || operation > 5) return 4;
  if (auto status=mlx_operation_event_validate_traversal_context(observer)) return status;
  auto* scope=static_cast<Scope*>(observer.ctx);
  const auto& selected=*static_cast<const Stream*>(stream.ctx);
  // Ring's existing communication_stream returns this exact supplied CPU
  // stream. The producer never creates an implicit default stream or encoder.
  if (selected.device != Device::cpu) return 4;
  if (auto status=preflight_original_submission_stream(*scope,selected); status != ScopedEvaluation::complete)
    return static_cast<unsigned>(status);
  const auto& actual_group=*static_cast<const distributed::Group*>(group.ctx);
  const auto& actual_input=*static_cast<const array*>(input.ctx);
  // The exact resident model bank remains alive and reserved. Only this
  // source-bound constructor may borrow its association; child destruction
  // precedes restoration on success, refusal and native exception paths.
  GraphNestedConstruction parent;
  struct Bank { GraphConstruction* value{nullptr}; ~Bank(){if(value)value->destroy();} } bank;
  const auto failure=GraphConstruction::create_distributed_with_resident_loan(*scope->graph_quota(),scope->identity(),
      actual_group,actual_input,static_cast<distributed::GroupWorkerOperation>(operation),peer,bank.value,parent);
  if (failure == GraphFailure::exhausted) return 2;
  if (failure == GraphFailure::construction_busy) return 10;
  if (static_cast<unsigned>(failure)) return 4;
  try {
    // Exactly the existing six constructors. Optional Group aliases and lazy
    // primitive/input owners preserve their ordinary construction/drop order.
    switch(operation) {
      case 0: mlx_array_set_(*out,distributed::all_sum(actual_input,actual_group,selected)); break;
      case 1: mlx_array_set_(*out,distributed::all_max(actual_input,actual_group,selected)); break;
      case 2: mlx_array_set_(*out,distributed::all_min(actual_input,actual_group,selected)); break;
      case 3: mlx_array_set_(*out,distributed::all_gather(actual_input,actual_group,selected)); break;
      case 4: mlx_array_set_(*out,distributed::send(actual_input,peer,actual_group,selected)); break;
      case 5: mlx_array_set_(*out,distributed::recv_like(actual_input,peer,actual_group,selected)); break;
    }
    return 0;
  } catch (std::exception& cause) {
    // Existing original error handler retains the exact native exception in
    // the active Scope; this does not allocate an ordinary formatted adapter.
    mlx_error(cause.what()); return 7;
  }
}

extern "C" unsigned mlx_distributed_construct_variable_original(mlx_array* out,
    mlx_submission_observer observer, mlx_distributed_group group, mlx_array input,
    const size_t* matrix, size_t matrix_entries, bool transposed, mlx_stream stream) {
  using namespace mlx::core;
  using namespace mlx::core::submission;
  if (!out || out->ctx || out->prepared_owner || !group.ctx || !input.ctx || !stream.ctx ||
      !matrix || !matrix_entries) return 4;
  if (auto status=mlx_operation_event_validate_traversal_context(observer)) return status;
  auto* scope=static_cast<Scope*>(observer.ctx);
  const auto& selected=*static_cast<const Stream*>(stream.ctx);
  if (selected.device != Device::cpu) return 4;
  if (auto status=preflight_original_submission_stream(*scope,selected); status != ScopedEvaluation::complete)
    return static_cast<unsigned>(status);
  const auto& actual_group=*static_cast<const distributed::Group*>(group.ctx);
  const auto& actual_input=*static_cast<const array*>(input.ctx);
  const auto counts=std::span<const size_t>(matrix,matrix_entries);
  GraphNestedConstruction parent;
  struct Bank { GraphConstruction* value{nullptr}; ~Bank(){if(value)value->destroy();} } bank;
  const auto failure=GraphConstruction::create_distributed_variable_with_resident_loan(
      *scope->graph_quota(),scope->identity(),actual_group,actual_input,counts,transposed,bank.value,parent);
  if (failure == GraphFailure::exhausted) return 2;
  if (failure == GraphFailure::construction_busy) return 10;
  if (static_cast<unsigned>(failure)) return 4;
  try {
    mlx_array_set_(*out,distributed::all_to_all_v_prepared(actual_input,counts,transposed,actual_group,selected));
    return 0;
  } catch (std::exception& cause) {
    mlx_error(cause.what()); return 7;
  }
}
