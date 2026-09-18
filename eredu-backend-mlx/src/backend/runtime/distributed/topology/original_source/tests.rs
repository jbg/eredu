use super::*;
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};

#[test]
fn native_retained_communication_source_binds_exact_world_without_submission_and_retains_failures() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let world = NativeGroup::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let foreign = NativeGroup::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let id = CollectiveGroupId::new(7);
    let descriptor = CommunicationGroupDescriptor::new(id, 0, vec![0], Some(0),
        eredu_runtime::CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![descriptor], vec![]).unwrap()
        .with_completion_policy(eredu_runtime::CommunicationCompletionPolicy::new(std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete).unwrap());
    let actual = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
    let authority = PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 20).unwrap();
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    let source = actual.bind_original_source(&manifest, &world, &authority, &funding).unwrap();
    source.validate().unwrap();
    assert_eq!(source.group(0).unwrap().1.id(), id);
    assert!(source.group(1).is_none() && source.route(0).is_none());
    assert!(source.world().native_group().shares_native_handle(&world));
    assert!(source.source().host_storage_bytes().unwrap() > 0);
    let escaped = source.group(0).unwrap().0.clone();
    let identity = source.source().clone();
    drop(source);
    let failure = actual.bind_original_source(&manifest, &foreign, &authority, &funding)
        .err().expect("equal native rank/size cannot replace the exact world");
    assert_eq!(crate::backend::runtime::distributed::group::native_collective_submissions(), 0);
    drop((actual, authority, funding));
    assert!(escaped.retained_source().unwrap().same_source(&identity));
    assert!(pool.used_bytes().unwrap() > 0, "the original failure retains its account");
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}


#[test]
fn paid_native_inventory_keeps_actual_source_and_h_without_submission_or_total_fit() {
    use safemlx::distributed::{GroupStorageKind, GroupStorageDomain};
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let world = NativeGroup::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let id = CollectiveGroupId::new(7);
    let descriptor = CommunicationGroupDescriptor::new(id, 0, vec![0], Some(0),
        eredu_runtime::CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![descriptor], vec![]).unwrap()
        .with_completion_policy(eredu_runtime::CommunicationCompletionPolicy::new(std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete).unwrap());
    let actual = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
    let authority = PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 20).unwrap();
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    let source = actual.bind_original_source(&manifest, &world, &authority, &funding).unwrap();
    let inventory = source.world_inventory().unwrap();
    let group = source.group_inventory(0).unwrap();
    assert!(inventory.source().same_source(source.source()));
    assert!(inventory.native().same_implementation(group.native()));
    assert_eq!(inventory.native().kind(), GroupStorageKind::Empty);
    assert!(inventory.native().has_unqualified_storage());
    assert!(inventory.native().is_unqualified(GroupStorageDomain::SharedControl));
    let failure = source.group_inventory(1).err().expect("no fabricated source row");
    assert_eq!(crate::backend::runtime::distributed::group::native_collective_submissions(), 0);
    drop(group);
    drop(source);
    drop(funding);
    drop(failure);
    assert!(pool.used_bytes().unwrap() > 0, "paid inventory loan retains H");
    assert!(inventory.native().implementation_bytes() > 0);
    drop(inventory);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unqualified_actual_worker_source_refuses_without_submission_and_retains_failure_h() {
    use safemlx::{Array, distributed::GroupWorkerOperation};
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let world = NativeGroup::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let id = CollectiveGroupId::new(7);
    let descriptor = CommunicationGroupDescriptor::new(id, 0, vec![0], Some(0),
        eredu_runtime::CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![descriptor], vec![]).unwrap()
        .with_completion_policy(eredu_runtime::CommunicationCompletionPolicy::new(std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete).unwrap());
    let actual = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
    let authority = PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 20).unwrap();
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    let source = actual.bind_original_source(&manifest, &world, &authority, &funding).unwrap();
    let input = Array::from_slice(&[7_i32,-11,23], &[3]);
    let failure = source.world_worker_storage(&input,GroupWorkerOperation::Sum)
        .err().expect("actual singleton fallback cannot supply Ring workers");
    assert!(source.group_worker_storage(1,&input,GroupWorkerOperation::Gather).is_err());
    assert!(source.route_worker_storage(0,&input,GroupWorkerOperation::Send {peer:0}).is_err());
    assert_eq!(crate::backend::runtime::distributed::group::native_collective_submissions(),0);
    drop(source);
    drop((actual,authority,funding,input));
    assert!(pool.used_bytes().unwrap()>0,"refused worker query keeps exact source/H");
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(),0);
}

#[test]
fn repeated_source_query_errors_each_pay_retained_failure_and_keep_h() {
    use safemlx::{Array, distributed::GroupWorkerOperation};
    use std::sync::{Arc, atomic::{AtomicUsize,Ordering}};
    #[derive(Debug)]
    struct Counted { inner: HostMetadataFunding, reserved: Arc<AtomicUsize> }
    impl eredu_nn::workspace::HostMetadataAccount for Counted {
        fn reserve_metadata(&self, bytes:usize)->Result<(),HostMetadataFundingError> {
            self.inner.reserve_metadata(bytes)?;
            self.reserved.fetch_add(bytes,Ordering::SeqCst);
            Ok(())
        }
    }
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let world = NativeGroup::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let id = CollectiveGroupId::new(7);
    let descriptor = CommunicationGroupDescriptor::new(id, 0, vec![0], Some(0),
        eredu_runtime::CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![descriptor], vec![]).unwrap()
        .with_completion_policy(eredu_runtime::CommunicationCompletionPolicy::new(std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete).unwrap());
    let actual = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
    let authority = PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 20).unwrap();
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    let reserved=Arc::new(AtomicUsize::new(0));
    let funding=HostMetadataFunding::new(Counted {inner:funding,reserved:reserved.clone()}).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let input=Array::from_slice(&[7_i32,-11,23],&[3]);
    let allowance=failure_control_bytes().unwrap();
    let before=reserved.load(Ordering::SeqCst);
    let first=source.group_worker_storage(99,&input,GroupWorkerOperation::Gather).err().unwrap();
    let after_first=reserved.load(Ordering::SeqCst);
    assert!(after_first-before>=allowance);
    let second=source.group_worker_storage(99,&input,GroupWorkerOperation::Gather).err().unwrap();
    let after_second=reserved.load(Ordering::SeqCst);
    assert!(after_second-after_first>=allowance,"escaped errors cannot reuse a source allowance");
    let third=source.group_inventory(99).err().unwrap();
    assert!(reserved.load(Ordering::SeqCst)-after_second>=allowance);
    let before_validation=reserved.load(Ordering::SeqCst);
    source.validate().unwrap();
    assert!(reserved.load(Ordering::SeqCst)-before_validation>=allowance);
    assert_eq!(crate::backend::runtime::distributed::group::native_collective_submissions(),0);
    drop(source);
    drop((actual,authority,funding,input));
    drop(first);
    drop(second);
    assert!(pool.used_bytes().unwrap()>0,"each retained error owns its source/account");
    drop(third);
    assert_eq!(pool.used_bytes().unwrap(),0);
}

#[test]
fn retained_native_persistent_owners_are_paid_before_lending_and_keep_source_h(){
    use std::sync::{Arc,atomic::{AtomicUsize,Ordering}};
    #[derive(Debug)]
    struct Counted{inner:HostMetadataFunding,reserved:Arc<AtomicUsize>}
    impl eredu_nn::workspace::HostMetadataAccount for Counted{
        fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError>{
            self.inner.reserve_metadata(bytes)?;self.reserved.fetch_add(bytes,Ordering::SeqCst);Ok(())
        }
    }
    let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
    let world=NativeGroup::init(false,safemlx::distributed::Backend::Ring).unwrap();
    let descriptor=CommunicationGroupDescriptor::new(CollectiveGroupId::new(7),0,vec![0],Some(0),
        eredu_runtime::CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let manifest=CommunicationManifest::new(1,0,vec![descriptor],vec![]).unwrap()
        .with_completion_policy(eredu_runtime::CommunicationCompletionPolicy::new(std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete).unwrap());
    let actual=ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap();
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool=WorkingMemoryPool::new(1<<20,0).unwrap();
    let raw=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),1<<20).unwrap();
    let reserved=Arc::new(AtomicUsize::new(0));
    let funding=HostMetadataFunding::new(Counted{inner:raw,reserved:reserved.clone()}).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let before=reserved.load(Ordering::SeqCst);
    let persistent=source.world_persistent().unwrap();
    assert!(reserved.load(Ordering::SeqCst)-before>=persistent.native().retained_owner_bytes().unwrap());
    assert!(persistent.source().same_source(source.source()));
    let group=source.group_persistent(0).unwrap();
    assert!(persistent.native().same_implementation(group.native()));
    assert!(!persistent.native().has_unqualified_storage());
    let failure=source.group_persistent(99).err().expect("exact declaration required");
    drop(group);drop(source);drop(funding);drop(failure);
    assert!(pool.used_bytes().unwrap()>0,"paid persistent native loan retains H");
    assert!(persistent.native().retained_owner_bytes().unwrap()>0);
    drop(persistent);
    assert_eq!(pool.used_bytes().unwrap(),0);
}
