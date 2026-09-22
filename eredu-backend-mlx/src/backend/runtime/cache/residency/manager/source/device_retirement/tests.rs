use super::*;
use eredu_nn::workspace::{
    HostMetadataAccount, HostMetadataFundingError, WorkspaceEffectDestination,
    WorkspaceFactMechanisms, WorkspaceHostDestination, WorkspaceHostFacts, WorkspaceMechanisms,
    WorkspaceOperation, WorkspaceOperationBound, WorkspaceOperationFacts, WorkspaceOperationView,
};
use eredu_runtime::{CachePoolLimits, CacheResidencyPool};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("backing retirement issues no tensor operation")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("backing retirement inspects no tensor equation")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("backing retirement emits no tensor equation")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("backing retirement inspects no host equation")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("backing retirement emits no host equation")
    }
}
#[derive(Debug)]
struct State {
    remaining: AtomicUsize,
    calls: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0.calls.fetch_add(1, Ordering::SeqCst);
        self.0
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(bytes))
            .map(|_| ())
            .map_err(|remaining| HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: remaining as u64,
            })
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn context() -> (WorkspaceContext, Arc<State>) {
    let state = Arc::new(State {
        remaining: AtomicUsize::new(1 << 24),
        calls: AtomicUsize::new(0),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding).unwrap(),
        state,
    )
}

#[test]
fn device_retirement_refuses_before_attachment_construction_and_retains_payer() {
    for kind in [AttachmentKind::Device, AttachmentKind::Host] {
        let (context, state) = context();
        let bytes = DeviceRetirement::controls(kind).unwrap();
        let calls = state.calls.load(Ordering::SeqCst);
        state.remaining.store(bytes - 1, Ordering::SeqCst);
        let error = DeviceRetirement::prepare_kind(&context, kind)
            .err()
            .expect("first debit must refuse");
        assert_eq!(state.calls.load(Ordering::SeqCst), calls + 1);
        assert_eq!(state.remaining.load(Ordering::SeqCst), bytes - 1);
        assert!(matches!(
            error.cause(),
            CacheSourceFailureCause::Metadata(_)
        ));
        drop(context);
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(error);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}

#[test]
#[ignore = "requires actual native backing retirement"]
fn device_retirement_keeps_escaped_alias_charge_and_reuses_exact_two_page_cap() {
    let (context, state) = context();
    let pool = CacheResidencyPool::new(CachePoolLimits::new(256, 0, 256, 0).unwrap());
    let usage = CachePoolUsage {
        device_bytes: 128,
        ..CachePoolUsage::default()
    };
    let other_page = pool
        .prepare_reservation(&context)
        .unwrap()
        .reserve(usage)
        .unwrap();
    let mut replaced = Some(
        pool.prepare_reservation(&context)
            .unwrap()
            .reserve(usage)
            .unwrap(),
    );
    let first = Array::from_slice(&[1.25f32; 16], &[1, 1, 2, 8]);
    let second = Array::from_slice(&[-2.5f32; 16], &[1, 1, 2, 8]);
    first.evaluated().unwrap();
    second.evaluated().unwrap();
    let escaped = first.clone();
    let mut retirement = DeviceRetirement::prepare(&context).unwrap();
    retirement.attach([&first, &second]).unwrap();
    retirement.publish(&mut replaced);
    assert!(replaced.is_none());
    drop((first, second));
    retirement.reclaim();
    assert_eq!(pool.report().unwrap().current_device_bytes, 256);
    let refused = pool
        .prepare_reservation(&context)
        .unwrap()
        .reserve(usage)
        .unwrap_err();
    assert!(matches!(
        refused.cause(),
        eredu_runtime::CachePoolError::BudgetExceeded {
            resource: eredu_runtime::CachePoolResource::Device,
            required: 384,
            budget: 256
        }
    ));
    drop(refused);
    // An unrelated ready payload must remain queued: this is a specific drain.
    struct Probe(Arc<AtomicBool>);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let unrelated_dropped = Arc::new(AtomicBool::new(false));
    let unrelated = Array::from_slice(&[3f32], &[1]);
    unrelated.evaluated().unwrap();
    PreparedAllocationOwner::try_new(Probe(unrelated_dropped.clone()))
        .unwrap()
        .try_attach(&unrelated)
        .unwrap();
    drop(unrelated);
    drop(escaped);
    retirement.reclaim();
    assert!(!unrelated_dropped.load(Ordering::SeqCst));
    assert_eq!(pool.report().unwrap().current_device_bytes, 128);
    let next_page = pool
        .prepare_reservation(&context)
        .unwrap()
        .reserve(usage)
        .unwrap();
    assert_eq!(pool.report().unwrap().current_device_bytes, 256);
    drop((next_page, other_page, retirement, context));
    assert_eq!(pool.report().unwrap().current_device_bytes, 0);
    assert!(state.retired.load(Ordering::SeqCst));
    safemlx::reclaim_allocation_owners();
    assert!(unrelated_dropped.load(Ordering::SeqCst));
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires actual Metal Host-backed Data view retirement"]
fn device_retirement_host_views_reuse_two_page_cap_while_immutable_sources_survive() {
    use safemlx::{
        Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy, Stream,
        ops::indexing::TryIndexOp,
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let (context, state) = context();
    let pool = CacheResidencyPool::new(CachePoolLimits::new(256, 0, 256, 0).unwrap());
    let usage = CachePoolUsage {
        device_bytes: 128,
        ..CachePoolUsage::default()
    };
    let other_page = pool
        .prepare_reservation(&context)
        .unwrap()
        .reserve(usage)
        .unwrap();
    let hosts = [1.25f32, -2.5f32].map(|value| {
        let mut host =
            HostTransferBuffer::new(&[1, 1, 2, 8], Dtype::Float32, HostTransferPolicy::Transfer)
                .unwrap();
        for bytes in host.as_bytes_mut().unwrap().chunks_exact_mut(4) {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        host.freeze()
    });
    for _ in 0..3 {
        let mut replaced = Some(
            pool.prepare_reservation(&context)
                .unwrap()
                .reserve(usage)
                .unwrap(),
        );
        let first = hosts[0]
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let second = hosts[1]
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        first.evaluated().unwrap();
        second.evaluated().unwrap();
        assert_eq!(
            first.allocation_info().unwrap(),
            Some(hosts[0].allocation_info().unwrap())
        );
        assert_eq!(
            second.allocation_info().unwrap(),
            Some(hosts[1].allocation_info().unwrap())
        );
        assert!(first.inspect_host_transfer_view().unwrap().is_some());
        assert!(second.inspect_host_transfer_view().unwrap().is_some());
        let escaped = first.clone();
        let view = first.try_index_device((.., .., 1.., ..), &stream).unwrap();
        view.evaluated().unwrap();
        let mut retirement = DeviceRetirement::prepare(&context).unwrap();
        retirement.attach([&first, &second]).unwrap();
        retirement.publish(&mut replaced);
        drop((first, second));
        stream.synchronize().unwrap();
        retirement.reclaim();
        assert_eq!(pool.report().unwrap().current_device_bytes, 256);
        let refused = pool
            .prepare_reservation(&context)
            .unwrap()
            .reserve(usage)
            .unwrap_err();
        assert!(matches!(
            refused.cause(),
            eredu_runtime::CachePoolError::BudgetExceeded {
                resource: eredu_runtime::CachePoolResource::Device,
                required: 384,
                budget: 256
            }
        ));
        drop(refused);
        drop(escaped);
        retirement.reclaim();
        assert_eq!(pool.report().unwrap().current_device_bytes, 256);
        assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &[1.25f32; 8]);
        drop(view);
        stream.synchronize().unwrap();
        retirement.reclaim();
        assert_eq!(
            pool.report().unwrap().current_device_bytes,
            128,
            "completed Device view retires while reusable immutable Host sources remain"
        );
    }
    assert!(
        hosts
            .iter()
            .all(|host| host.allocation_info().unwrap().bytes() >= 64)
    );
    drop((other_page, context));
    assert_eq!(pool.report().unwrap().current_device_bytes, 0);
    assert!(state.retired.load(Ordering::SeqCst));
    drop(hosts);
}

#[test]
#[ignore = "requires actual native Host copy and backing retirement"]
fn host_staging_retirement_releases_copied_source_before_device_return() {
    host_staging_retirement(safemlx::DeviceType::Cpu, false);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires actual Metal Host-backed view retirement"]
fn host_staging_retirement_keeps_shared_view_charge_until_last_alias() {
    host_staging_retirement(safemlx::DeviceType::Gpu, true);
}

fn host_staging_retirement(device: safemlx::DeviceType, shared: bool) {
    use safemlx::{
        Device, Dtype, HostTransferBuffer, HostTransferPolicy, Stream, ops::indexing::TryIndexOp,
    };
    let stream = Stream::new_with_device(&Device::new(device, 0));
    // Select the same fresh backing policy as ordinary request Work. This
    // fixture's logical pool owner is tested separately from the native
    // observer admission/custody cases in scoped_physical_funding.
    struct FixturePhysicalSource;
    impl safemlx::PhysicalBackingObserver for FixturePhysicalSource {
        fn admit(
            &self,
            _: safemlx::AllocationInfo,
        ) -> safemlx::error::Result<safemlx::PhysicalBackingCustody> {
            Ok(safemlx::PhysicalBackingCustody::new(()))
        }
    }
    let observer =
        safemlx::ScopedPhysicalBackingObserver::new_fresh(Arc::new(FixturePhysicalSource)).unwrap();
    let mut scope = safemlx::SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    let (context, payer) = context();
    let hosts = [1.25f32, -2.5f32].map(|value| {
        let mut buffer =
            HostTransferBuffer::new(&[1, 1, 2, 8], Dtype::Float32, HostTransferPolicy::Transfer)
                .unwrap();
        for bytes in buffer.as_bytes_mut().unwrap().chunks_exact_mut(4) {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        buffer.freeze()
    });
    let host_bytes = hosts
        .iter()
        .map(|host| u64::try_from(host.allocation_info().unwrap().bytes()).unwrap())
        .sum::<u64>();
    let host_usage = CachePoolUsage {
        host_bytes,
        ..CachePoolUsage::default()
    };
    let device_usage = CachePoolUsage {
        device_bytes: 128,
        ..CachePoolUsage::default()
    };
    let pool =
        CacheResidencyPool::new(CachePoolLimits::new(128, host_bytes, host_bytes, 0).unwrap());
    let member = pool.register_manager(7).unwrap();
    pool.update_manager(7, host_usage).unwrap();
    // Both independently prepared table destinations exist before either token
    // is installed. The empty return token grants no additional occupancy.
    let promotion_slot = pool.prepare_reservation_population(2, &context).unwrap();
    let return_slot = pool.prepare_reservation_population(2, &context).unwrap();
    let mut replaced_host = Some(
        promotion_slot
            .reserve(CachePoolUsage {
                transfer_in_flight_bytes: host_bytes,
                ..device_usage
            })
            .unwrap(),
    );
    let mut future_return = Some(return_slot.reserve(CachePoolUsage::default()).unwrap());
    let admitted = pool.report().unwrap();
    let mut host_retirement = DeviceRetirement::prepare_host(&context).unwrap();
    host_retirement.attach_host([&hosts[0], &hosts[1]]).unwrap();
    let outputs = hosts.each_ref().map(|host| {
        let value = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
        value.evaluated().unwrap();
        value
    });
    if shared {
        assert_eq!(
            outputs[0].allocation_info().unwrap(),
            Some(hosts[0].allocation_info().unwrap())
        );
    } else {
        assert_ne!(
            outputs[0].allocation_info().unwrap(),
            Some(hosts[0].allocation_info().unwrap())
        );
    }
    let escaped = if shared {
        let view = outputs[0]
            .try_index_device((.., .., 1.., ..), &stream)
            .unwrap();
        view.evaluated().unwrap();
        view
    } else {
        outputs[0].clone()
    };
    scope.seal();
    drop((scope, observer));
    replaced_host
        .as_mut()
        .unwrap()
        .publish_retaining_replaced_storage(&member, device_usage)
        .unwrap();
    assert_eq!(
        pool.report().unwrap(),
        admitted,
        "publication changes ownership only"
    );
    assert_eq!(
        future_return.as_ref().unwrap().remaining_usage().unwrap(),
        CachePoolUsage::default()
    );
    host_retirement.publish(&mut replaced_host);
    assert!(replaced_host.is_none());
    drop(hosts);
    stream.synchronize().unwrap();
    host_retirement.reclaim();
    let after_host_drop = pool.report().unwrap();
    assert_eq!(after_host_drop.current_device_bytes, 128);
    assert_eq!(
        after_host_drop.current_host_bytes,
        if shared { host_bytes } else { 0 }
    );
    assert_eq!(
        after_host_drop.current_transfer_in_flight_bytes,
        if shared { host_bytes } else { 0 }
    );
    assert_eq!(after_host_drop.peak_host_bytes, admitted.peak_host_bytes);
    if !shared {
        // The copied Device destination remains live while another exact Host
        // staging page fits the same one-page cap.
        let next = pool
            .prepare_reservation(&context)
            .unwrap()
            .reserve(host_usage)
            .unwrap();
        assert_eq!(pool.report().unwrap().current_host_bytes, host_bytes);
        drop(next);
    }
    let mut device_retirement = DeviceRetirement::prepare(&context).unwrap();
    device_retirement
        .attach([&outputs[0], &outputs[1]])
        .unwrap();
    future_return
        .as_mut()
        .unwrap()
        .publish_retaining_replaced_storage(&member, CachePoolUsage::default())
        .unwrap();
    assert_eq!(
        future_return.as_ref().unwrap().remaining_usage().unwrap(),
        device_usage
    );
    device_retirement.publish(&mut future_return);
    drop((outputs, context));
    stream.synchronize().unwrap();
    host_retirement.reclaim();
    device_retirement.reclaim();
    assert_eq!(pool.report().unwrap().current_device_bytes, 128);
    assert!(!payer.retired.load(Ordering::SeqCst));
    let values = escaped.evaluated().unwrap();
    assert!(
        values
            .try_as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 1.25)
    );
    drop(values);
    drop(escaped);
    stream.synchronize().unwrap();
    host_retirement.reclaim();
    device_retirement.reclaim();
    let retired = pool.report().unwrap();
    assert_eq!(retired.current_device_bytes, 0);
    assert_eq!(retired.current_host_bytes, 0);
    assert_eq!(retired.current_transfer_in_flight_bytes, 0);
    assert_eq!(retired.peak_device_bytes, admitted.peak_device_bytes);
    assert_eq!(retired.peak_host_bytes, admitted.peak_host_bytes);
    drop((host_retirement, device_retirement, member));
    assert!(payer.retired.load(Ordering::SeqCst));
}
