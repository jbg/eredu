// Linked representation, no-new handoff and actual physical-lifetime tests.
#include <atomic>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>
#include "mlx/allocation_owners.h"
#include "mlx/array.h"
#include "mlx/c/array.h"
#include "mlx/c/host_transfer.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/host_transfer.h"
#include "mlx/host_transfer.h"

static thread_local bool forbid_new = false;
static thread_local void* watched_delete = nullptr;
static thread_local bool watched_deleted = false;
static void free_observed(void* p) noexcept {
  const bool watched = p == watched_delete && p != nullptr;
  std::free(p);
  if (watched) watched_deleted = true;
}
void* operator new(std::size_t n) {
  if (forbid_new) throw std::bad_alloc();
  if (auto* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { free_observed(p); }
void operator delete[](void* p) noexcept { free_observed(p); }
void operator delete(void* p, std::size_t) noexcept { free_observed(p); }
void operator delete[](void* p, std::size_t) noexcept { free_observed(p); }
struct NoNew {
  NoNew() { forbid_new = true; }
  ~NoNew() { forbid_new = false; }
};
static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
using namespace mlx::core;
struct Payload { bool* backing_freed; size_t* retired; };
static void release(void* raw) noexcept {
  auto& p = *static_cast<Payload*>(raw);
  if (p.backing_freed && !*p.backing_freed) std::abort();
  ++*p.retired;
}

static void release_after_exact_node_delete(void* raw) noexcept {
  if (!watched_deleted) std::abort();
  release(raw);
}
static void node_is_freed_before_accounting_callback() {
  bool freed=false; size_t retired=0; Payload payload{&freed,&retired};
  auto data=std::make_shared<array::Data>(allocator::Buffer(reinterpret_cast<void*>(1)),
      [&](allocator::Buffer) { freed=true; });
  auto* node=new AllocationOwners::Node();
  watched_delete=node; watched_deleted=false;
  data->allocation_owners.append_prepared(node,&payload,release_after_exact_node_delete);
  // The exact node's operator delete must finish before the callback can enqueue
  // an accounting owner for immediate reclamation on another host thread.
  { NoNew denied; data.reset(); }
  require(watched_deleted && freed && retired==1,"native node/backing outlived accounting callback");
  watched_delete=nullptr;
}
static void exact_layout_and_no_growth() {
  require(mlx_allocation_owner_node_bytes() == sizeof(AllocationOwners::Node), "wrong linked node size");
  require(mlx_allocation_owner_list_bytes() == sizeof(AllocationOwners), "wrong linked list size");
  require(sizeof(AllocationOwners) <= sizeof(std::vector<std::shared_ptr<void>>), "common owner representation grew");
  size_t retired = 0; Payload payload{nullptr, &retired};
  std::vector<AllocationOwners::Node*> nodes;
  for (int i=0; i<4096; ++i) nodes.push_back(new AllocationOwners::Node());
  {
    AllocationOwners list;
    { NoNew denied;
      for (auto* node : nodes) list.append_prepared(node, &payload, release);
      AllocationOwners moved(std::move(list));
    }
    require(retired == nodes.size(), "iterative moved-list retirement lost owners");
  }
}
static void legacy_identity_and_null_entries_are_unchanged() {
  size_t retired = 0; Payload payload{nullptr, &retired};
  AllocationOwners list;
  list.append_prepared(new AllocationOwners::Node(), &payload, release);
  require(!list.contains({}), "prepared callback confused with legacy null shared owner");
  list.push_back({}); require(list.contains({}), "legacy null owner lost");
  auto owner = std::make_shared<int>(7);
  auto equal = std::make_shared<int>(7);
  list.push_back(owner); list.push_back(owner); // Data retains duplicates.
  require(list.contains(owner) && !list.contains(equal), "shared pointer identity changed");
  require(owner.use_count() == 3, "legacy duplicate append was silently deduplicated");
}
static void data_move_frees_backing_before_callbacks() {
  bool freed=false; size_t retired=0; Payload payload{&freed,&retired};
  auto data=std::make_shared<array::Data>(allocator::Buffer(reinterpret_cast<void*>(1)),
      [&](allocator::Buffer) { freed=true; });
  data->allocation_owners.append_prepared(new AllocationOwners::Node(),&payload,release);
  auto alias=data;
  { array::Data moved(std::move(*data));
    data.reset(); alias.reset(); require(!freed && retired==0,"moved-from Data retired payload");
  }
  require(freed && retired==1,"Data did not retire after backing");
}
static void actual_array_handoff_has_no_new_or_identity_change() {
  size_t retired=0; Payload payload{nullptr,&retired};
  auto value=array(3.5f); auto alias=value;
  auto handle=mlx_array_new_(value);
  bool known=false,host=false; uint64_t identity=0; size_t bytes=0;
  require(!mlx_array_allocation_info(&known,&host,&identity,&bytes,handle) && known && identity,"missing actual backing");
  auto* node=mlx_allocation_owner_node_new(); require(node,"node allocation failed");
  int outcome=0,status=-1;
  { NoNew denied; status=mlx_array_attach_prepared_allocation_owner(&outcome,handle,node,&payload,release); }
  require(status==0 && outcome==2,"prepared actual array handoff allocated or rejected");
  bool known2=false,host2=false; uint64_t id2=0;size_t bytes2=0;
  require(!mlx_array_allocation_info(&known2,&host2,&id2,&bytes2,handle) && known2 && id2==identity && bytes2==bytes,"backing changed");
  mlx_array_free(handle); require(retired==0,"C handle drop lost earlier alias");
  value=array(1.0f); require(retired==0,"earlier native alias lost custody");
  alias=array(2.0f); require(retired==1,"last native alias failed to retire");
}
static void actual_host_handoff_uses_same_physical_storage() {
  size_t retired=0; Payload payload{nullptr,&retired};
  HostTransferBuffer buffer({2},int32); auto alias=buffer;
  auto handle=mlx_host_transfer_buffer_new_(HostTransferBuffer(buffer));
  auto* node=mlx_allocation_owner_node_new();require(node,"node allocation failed");
  int outcome=0,status=-1;
  { NoNew denied; status=mlx_host_transfer_buffer_attach_prepared_allocation_owner(&outcome,handle,node,&payload,release); }
  require(status==0 && outcome==2,"prepared actual host handoff allocated or rejected");
  mlx_host_transfer_buffer_free(handle);buffer=HostTransferBuffer();
  require(retired==0,"shared host alias lost owner"); alias=HostTransferBuffer();
  require(retired==1,"host owner did not retire exactly once");
}
int main() {
  node_is_freed_before_accounting_callback();
  exact_layout_and_no_growth(); legacy_identity_and_null_entries_are_unchanged();
  data_move_frees_backing_before_callbacks(); actual_array_handoff_has_no_new_or_identity_change();
  actual_host_handoff_uses_same_physical_storage();
}
