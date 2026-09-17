use super::*;
use crate::backend::nn::workspace::ExistingArrayProjection;
use eredu_core::cache::{
    LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
    StateTensorPolicy, StateTensorRole,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
};
use safemlx::{ops::indexing::TryIndexOp, Device, DeviceType};
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

const CONV: StateTensorRole = StateTensorRole::Convolution { slot: 2 };
const RECURRENT: StateTensorRole = StateTensorRole::Recurrent;
const PREFIX: StateTensorRole = StateTensorRole::PrefixEmbedding;

#[derive(Debug)]
struct Unpriced;
impl WorkspaceMechanisms for Unpriced {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}

fn slots() -> FixedStateSlots {
    let tensors = [PREFIX, RECURRENT, CONV]
        .into_iter()
        .map(|role| {
            StateTensorPolicy::new(
                role,
                vec![
                    StateTensorDimension::fixed(2).unwrap(),
                    StateTensorDimension::fixed(2).unwrap(),
                ],
                StateTensorDtype::Float32,
                if role == RECURRENT {
                    MutableStateResidency::LayerScopedOffloadable
                } else {
                    MutableStateResidency::AlwaysDeviceMutable
                },
            )
            .unwrap()
        })
        .collect();
    FixedStateSlots::from_policy(&LayerCachePolicy::fixed_only(tensors).unwrap()).unwrap()
}

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn view(stream: &Stream) -> Array {
    Array::from_slice(&[1_f32, 2., 3., 4., 5., 6., 7., 8.], &[2, 4])
        .try_index_device((.., 1..3), stream)
        .unwrap()
}

fn fill(source: &mut FixedStateSlots, array: Array) {
    *source.get_mut(&CONV).unwrap() = Some(MlxTensor::from_array(array.clone()));
    *source.get_mut(&PREFIX).unwrap() = Some(MlxTensor::from_array(array));
}

fn source_array(source: &FixedStateSlots, role: StateTensorRole) -> &Array {
    source
        .iter()
        .find(|(key, _)| **key == role)
        .unwrap()
        .1
        .as_ref()
        .unwrap()
        .as_array()
}

fn numbers(array: &Array) -> Vec<f32> {
    array.evaluated().unwrap().try_to_vec::<f32>().unwrap()
}

thread_local! {
    static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) };
    static FAIL_AFTER_SLOT: Cell<bool> = const { Cell::new(false) };
}
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct HousekeepingGuard;
impl Drop for HousekeepingGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[derive(Debug)]
struct InjectedCopyFailure;
impl std::fmt::Display for InjectedCopyFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("fixed-state copy failed after first native destination")
    }
}
impl std::error::Error for InjectedCopyFailure {}

struct FailureGuard(bool);
impl FailureGuard {
    fn after_first_slot() -> Self {
        Self(FAIL_AFTER_SLOT.replace(true))
    }
}
impl Drop for FailureGuard {
    fn drop(&mut self) {
        FAIL_AFTER_SLOT.set(self.0);
    }
}
pub(super) fn after_slot_copy() -> Result<(), Exception> {
    if FAIL_AFTER_SLOT.replace(false) {
        Err(Exception::from_source(InjectedCopyFailure))
    } else {
        Ok(())
    }
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn prepared_operands_borrow_original_descriptors_without_settling_lazy_views() {
    let stream = stream();
    let mut source = slots();
    fill(&mut source, view(&stream));
    assert!(source_array(&source, CONV)
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none());
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = HousekeepingGuard;
    HOUSEKEEPING.set(0);
    let plan = source.prepare_copy();
    assert_eq!(plan.payload_bytes(), source.payload_bytes());
    let mut seen = [None, None];
    let mut count = 0;
    plan.visit_operands(&mut |array| {
        seen[count] = Some(array);
        count += 1;
    });
    assert_eq!(count, 2);
    assert!(std::ptr::eq(seen[0].unwrap(), source_array(&source, CONV)));
    assert!(std::ptr::eq(
        seen[1].unwrap(),
        source_array(&source, PREFIX)
    ));
    let context = WorkspaceContext::new(Unpriced);
    let mut projection = ExistingArrayProjection::new(&context);
    for array in seen.into_iter().flatten() {
        projection.project(array).unwrap();
        assert!(array
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .is_none());
    }
    assert!(!projection.is_complete());
    assert!(context.report(&[]).unwrap().operations.is_empty());
    assert_eq!(HOUSEKEEPING.get(), 0);
    drop(guard);
    let mut copied = plan.copy(&stream).unwrap();
    assert!(!copied.metadata().same_storage(source.metadata()));
    assert!(copied.get_mut(&RECURRENT).unwrap().is_none());
    assert_eq!(numbers(source_array(&copied, CONV)), [2., 3., 6., 7.]);
    assert_eq!(numbers(source_array(&copied, PREFIX)), [2., 3., 6., 7.]);
}

#[test]
fn aliased_source_roles_keep_one_opening_root_and_two_independent_destinations() {
    let stream = stream();
    let mut source = slots();
    fill(&mut source, view(&stream));
    let expected = numbers(source_array(&source, CONV));
    source_array(&source, PREFIX).evaluated().unwrap();
    let source_identity = source_array(&source, CONV)
        .allocation_info()
        .unwrap()
        .unwrap()
        .identity();
    let plan = source.prepare_copy();
    let mut operands = Vec::new();
    plan.visit_operands(&mut |array| operands.push(array));
    let context = WorkspaceContext::new(Unpriced);
    let mut projection = ExistingArrayProjection::new(&context);
    for array in &operands {
        projection.project(array).unwrap();
    }
    assert!(projection.is_complete());
    let inventory = projection.into_storage();
    assert_eq!(inventory.iter().len(), 1);
    assert_eq!(inventory.iter().next().unwrap().0, source_identity);
    let mut copied = plan.copy(&stream).unwrap();
    assert_eq!(copied.payload_bytes(), source.payload_bytes());
    assert_eq!(
        copied.iter().map(|(role, _)| *role).collect::<Vec<_>>(),
        [CONV, RECURRENT, PREFIX]
    );
    assert!(copied.get_mut(&RECURRENT).unwrap().is_none());
    let mut identities = Vec::new();
    for role in [CONV, PREFIX] {
        let copied_array = source_array(&copied, role);
        assert_eq!(numbers(copied_array), expected);
        let identity = copied_array.allocation_info().unwrap().unwrap().identity();
        assert_ne!(identity, source_identity);
        identities.push(identity);
    }
    assert_ne!(identities[0], identities[1]);
    *copied.get_mut(&CONV).unwrap() = None;
    drop(operands);
    assert!(source.get_mut(&CONV).unwrap().is_some());
    drop(inventory);
    drop(source);
    assert_eq!(numbers(source_array(&copied, PREFIX)), expected);
}

#[test]
fn retained_worker_appends_all_temporary_and_destination_roots_and_preserves_empty_slots() {
    let stream = stream();
    let mut source = slots();
    fill(&mut source, view(&stream));
    let roots = RefCell::new(vec![Array::from_slice(&[19_f32], &[1])]);
    let copied = source
        .prepare_copy()
        .copy_retained(&stream, &roots)
        .unwrap();
    assert_eq!(roots.borrow().len(), 5);
    assert!(!copied.metadata().same_storage(source.metadata()));
    assert_eq!(numbers(&roots.borrow()[0]), [19.]);
    for (index, role) in [CONV, PREFIX].into_iter().enumerate() {
        assert_eq!(numbers(source_array(&copied, role)), [2., 3., 6., 7.]);
        let destination = source_array(&copied, role)
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity();
        let retained = &roots.borrow()[2 + 2 * index];
        retained.evaluated().unwrap();
        assert_eq!(
            retained.allocation_info().unwrap().unwrap().identity(),
            destination
        );
    }
    drop(copied);
    drop(source);
    for root in roots.borrow().iter().skip(1) {
        assert_eq!(numbers(root), [2., 3., 6., 7.]);
    }
    roots.borrow_mut().clear();
    let empty = FixedStateSlots::from_policy(&LayerCachePolicy::NoState).unwrap();
    let absent = slots();
    for source in [&empty, &absent] {
        let mut visits = 0;
        source.prepare_copy().visit_operands(&mut |_| visits += 1);
        assert_eq!(visits, 0);
        let copied = source
            .prepare_copy()
            .copy_retained(&stream, &roots)
            .unwrap();
        assert!(copied.values().all(Option::is_none));
        assert_eq!(copied.payload_bytes(), source.payload_bytes());
        assert!(roots.borrow().is_empty());
    }
}

#[test]
fn partial_failure_keeps_actual_native_roots_until_collector_and_raw_alias_retire() {
    let stream = stream();
    let mut source = slots();
    fill(&mut source, view(&stream));
    let before = source.payload_bytes();
    let roots = RefCell::new(Vec::new());
    let failure = FailureGuard::after_first_slot();
    let error = source
        .prepare_copy()
        .copy_retained(&stream, &roots)
        .unwrap_err();
    drop(failure);
    assert!(std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<InjectedCopyFailure>()
        .is_some());
    assert_eq!(
        roots.borrow().len(),
        2,
        "later source roles must not run after the original failure"
    );
    assert_eq!(source.payload_bytes(), before);
    assert!(source.get_mut(&RECURRENT).unwrap().is_none());
    assert_eq!(numbers(source_array(&source, CONV)), [2., 3., 6., 7.]);
    assert_eq!(numbers(source_array(&source, PREFIX)), [2., 3., 6., 7.]);
    let retired = Arc::new(AtomicUsize::new(0));
    let escaped = {
        let retained = roots.borrow();
        let destination = &retained[1];
        destination.evaluated().unwrap();
        destination
            .retain_allocation_owner(Retired(retired.clone()))
            .unwrap();
        destination.clone()
    };
    drop(source);
    for root in roots.borrow().iter() {
        assert_eq!(numbers(root), [2., 3., 6., 7.]);
    }
    roots.borrow_mut().clear();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(numbers(&escaped), [2., 3., 6., 7.]);
    drop(escaped);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        retired.load(Ordering::SeqCst) == 1
    });
}
