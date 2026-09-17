use super::*;
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
use safemlx::{Device, DeviceType, Stream, ops::indexing::TryIndexOp};
use std::cell::Cell;

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("projection must not execute equations")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("projection cannot quote equations")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("projection cannot emit equations")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("projection cannot quote host equations")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("projection cannot emit host equations")
    }
}
thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
fn hook() {
    HOOKS.with(|n| n.set(n.get() + 1));
}
struct Hooks;
impl Drop for Hooks {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(hook);
    }
}
fn cold<T>(f: impl FnOnce() -> T) -> T {
    safemlx::register_thread_runtime_housekeeping(hook);
    let _hooks = Hooks;
    HOOKS.with(|n| n.set(0));
    let result = f();
    assert_eq!(HOOKS.with(Cell::get), 0);
    result
}
#[test]
#[ignore = "requires native CPU array sources"]
fn transient_projection_deduplicates_closed_loans_and_keeps_failed_prefix_custody() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5], &[6]);
    let other = Array::from_slice(&[2.0f32, 3.0], &[2]);
    let excess = Array::from_slice(&[4.0f32], &[1]);
    root.evaluated().unwrap();
    other.evaluated().unwrap();
    excess.evaluated().unwrap();
    let id = root.try_allocation_info().unwrap().unwrap().identity();
    let capacity = 1 << 22;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), capacity)
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let mut destination = None;
    let (projected, whole) = {
        let mut projection =
            cold(|| OwnedArrayProjection::prepare(&mut destination, &context, 2)).unwrap();
        let projected = {
            let transient = root.try_index_device(2..5, &stream).unwrap();
            transient.evaluated().unwrap();
            let tensor = cold(|| projection.project_prepared(&transient)).unwrap();
            assert_eq!(tensor.layout().shape(), [3]);
            tensor
        };
        let whole = cold(|| projection.project_prepared(&root)).unwrap();
        assert_eq!(projection.storage().iter().len(), 1);
        cold(|| projection.project_prepared(&other)).unwrap();
        assert_eq!(projection.storage().iter().len(), 2);
        assert!(matches!(
            cold(|| projection.project_prepared(&excess)),
            Err(ProjectionSourceError::Inventory(
                ProjectionInventoryError::Exhausted
            ))
        ));
        assert_eq!(projection.storage().iter().len(), 2);
        assert!(projection.storage().is_complete());
        (projected, whole)
    };
    let duplicate = cold(|| OwnedArrayProjection::prepare(&mut destination, &context, 2));
    assert!(matches!(
        &duplicate,
        Err(ProjectionSourceError::Inventory(
            ProjectionInventoryError::Destination
        ))
    ));
    drop(duplicate);
    drop((root, other, excess, context, funding));
    assert!(pool.used_bytes().unwrap() > 0);
    let storage = destination.unwrap();
    assert_eq!(
        storage
            .native_array(id)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        [0.75, 1.0, 1.25]
    );
    // Raw workspace tensors retain trace identity, not an independent paying
    // account. Their enclosing state/context normally owns H; retire them while
    // this actual native storage owner still retains the same funding alias.
    drop((projected, whole));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(storage);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
#[ignore = "requires Metal shared host backing"]
fn actual_host_array_aliases_share_one_projection_root_and_retain_failed_prefix() {
    use safemlx::{Dtype, HostTransferBuffer, HostTransferPolicy};
    use std::sync::Arc;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for host_first in [true, false] {
        let mut raw =
            HostTransferBuffer::new(&[2, 3], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
        let values = [0.25f32, -0.5, 0.75, 1.0, 1.25, -1.5];
        let bytes = values
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        raw.as_bytes_mut().unwrap().copy_from_slice(&bytes);
        let host = Arc::new(raw.freeze());
        let weak = Arc::downgrade(&host);
        // The ordinary Metal transfer source exposes its real shared backing;
        // no fabricated allocation token or native alias is used by the test.
        let array = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
        array.evaluated().unwrap();
        let allocation = host.try_allocation_info().unwrap();
        assert_eq!(array.try_allocation_info().unwrap(), Some(allocation));
        let excess = Array::from_slice(&[3.0f32, 4.0], &[2]);
        excess.evaluated().unwrap();
        let capacity = 1 << 22;
        let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
        let funding = pool
            .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), capacity)
            .unwrap();
        let context =
            WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
        let mut destination = None;
        let symbols = {
            let mut projection =
                cold(|| OwnedArrayProjection::prepare(&mut destination, &context, 1)).unwrap();
            let first = if host_first {
                cold(|| projection.project_host::<4>(&host)).unwrap()
            } else {
                cold(|| projection.project_prepared(&array)).unwrap()
            };
            let second = if host_first {
                cold(|| projection.project_prepared(&array)).unwrap()
            } else {
                cold(|| projection.project_host::<4>(&host)).unwrap()
            };
            assert_eq!(projection.storage().iter().len(), 1);
            assert!(
                projection
                    .storage()
                    .native_array(allocation.identity())
                    .is_some()
            );
            assert!(Arc::ptr_eq(
                projection
                    .storage()
                    .native_host(allocation.identity())
                    .unwrap(),
                &host
            ));
            assert!(matches!(
                cold(|| projection.project_prepared(&excess)),
                Err(ProjectionSourceError::Inventory(
                    ProjectionInventoryError::Exhausted
                ))
            ));
            assert_eq!(projection.storage().iter().len(), 1);
            (first, second)
        };
        drop((array, host, excess, context, funding));
        assert!(weak.upgrade().is_some());
        assert!(pool.used_bytes().unwrap() > 0);
        let storage = destination.unwrap();
        assert_eq!(
            storage
                .native_array(allocation.identity())
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            values
        );
        drop(symbols);
        drop(storage);
        assert!(weak.upgrade().is_none());
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
