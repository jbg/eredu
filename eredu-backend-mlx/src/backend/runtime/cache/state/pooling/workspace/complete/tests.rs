use super::super::tests::{layout, step};
use super::*;
use crate::backend::runtime::cache::residency::{CacheResidencyError, CacheSourceFailureCause};
use eredu_nn::{workspace::*, Error, Tensor};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger},
    CacheLifecycleError, PagedCacheOptions,
};
use safemlx::{Device, DeviceType};
use std::{cell::Cell, collections::BTreeSet};

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

#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        (operation.outputs.get(output)?.dtype() == WorkspaceDtype::Float32).then_some(
            WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true),
        )
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|layout| {
                    if matches!(
                        op.kind,
                        WorkspaceOperationKind::Index { .. }
                            | WorkspaceOperationKind::StaticSlice { .. }
                            | WorkspaceOperationKind::View(_)
                            | WorkspaceOperationKind::Transpose(_)
                    ) {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        layout.bytes().map(WorkspaceOutputStorage::Allocate)
                    }
                })
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "fixture exact allocation and native view aliases".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture has no host payload workspace".into(),
        }))
    }
}

#[test]
#[ignore = "requires native CPU paged pooling source storage"]
fn pooling_paged_projection_preserves_visible_history_partial_streams_and_failed_source_custody() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap();
    let manager = CacheResidencyManager::new(options).unwrap();
    let rank = Some(CacheRankIdentity::new(Some(1), Some(2), Some(3)));
    let mut native =
        MlxPoolingAttentionStateFactory::paged(layout(1, 5), manager.clone(), 7, 1, rank).unwrap();
    step(native.layer(0).unwrap(), 1, 5, &stream);
    for tensor in RuntimeLayerState::<MlxNeuralBackend>::retained_values(native.layer(0).unwrap()) {
        tensor.as_array().evaluated().unwrap();
    }
    // Deliberately retain one actual allocation across paged tail and a
    // declared pending pooling component. No synthetic identity witness.
    let tail = match native.layer(0).unwrap().local() {
        LiveKeyValueCache::Paged(cache) => cache.retained_arrays()[0]
            .reshape(&[2, 1, 8], &stream)
            .unwrap(),
        _ => unreachable!(),
    };
    let pool = native.layer(0).unwrap().pool_mut(0).unwrap();
    let slots = pool.state_arrays();
    let gates = slots[1].unwrap().clone();
    let pooled = slots[2].unwrap().clone();
    pool.restore_state(
        PoolingCacheState {
            pending_values: Some(tail.clone()),
            pending_gates: Some(gates.clone()),
            pooled: Some(pooled.clone()),
            ..Default::default()
        },
        5,
    )
    .unwrap();
    tail.evaluated().unwrap();
    let capacity = 1 << 25;
    let owner = crate::memory_fixture::ledger(capacity, 0).unwrap();
    let funding = owner
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(capacity),
        )
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let mut expected = BTreeSet::new();
    let mut operands = 0;
    let cache = native.layer(0).unwrap();
    if let LiveKeyValueCache::Paged(local) = cache.local() {
        local
            .with_workspace_source(&context, |source| {
                for block in source.manager_source().blocks() {
                    for array in block.device().unwrap() {
                        expected.insert(array.try_allocation_info().unwrap().unwrap().identity());
                        operands += 1;
                    }
                }
                for arrays in source.tail_arrays() {
                    for array in arrays {
                        expected.insert(array.try_allocation_info().unwrap().unwrap().identity());
                        operands += 1;
                    }
                }
                Ok(())
            })
            .unwrap();
    }
    for array in cache.pool(0).unwrap().arrays() {
        expected.insert(array.try_allocation_info().unwrap().unwrap().identity());
        operands += 1;
    }
    let report = manager.report().unwrap();
    let projected = cold(|| {
        MlxPoolingAttentionStateFactory::project_complete_workspace_with_storage(
            &native,
            NonZeroU32::new(2).unwrap(),
            &context,
        )
    })
    .unwrap();
    assert!(projected.storage.is_complete());
    assert_eq!(
        projected
            .storage
            .iter()
            .map(|(id, _, _)| id)
            .collect::<BTreeSet<_>>(),
        expected
    );
    assert!(
        expected.len() < operands,
        "tail/pool alias must share one native root"
    );
    assert_eq!(projected.storage.paged_sources().len(), 1);
    let source = &projected.storage.paged_sources()[0];
    assert_eq!(source.geometry().global_layer, 7);
    assert_eq!(source.geometry().rank, rank);
    assert_eq!(
        (source.geometry().offset, source.geometry().tail_start),
        (5, 4)
    );
    if let LiveKeyValueCache::Paged(local) = native.layer(0).unwrap().local() {
        cold(|| source.validate_source(local, &projected.storage, &context)).unwrap();
    }
    assert_eq!(manager.report().unwrap().demand_hits, report.demand_hits);
    assert_eq!(
        manager.report().unwrap().demand_misses,
        report.demand_misses
    );
    let id = source.geometry().blocks[0].id.clone();

    let equation_context = WorkspaceContext::new(Facts);
    let mut equation = cold(|| {
        MlxPoolingAttentionStateFactory::project_resident_workspace_with_storage(
            &native,
            NonZeroU32::new(2).unwrap(),
            &equation_context,
        )
    })
    .unwrap();
    let WorkspaceResidentLayerState::Pooling(state) = &mut equation.state.as_mut()[0] else {
        panic!("pooling semantics retained")
    };
    assert_eq!(state.pooling_ratio(0), Some(4));
    assert_eq!(state.offset(), 5);
    let saved = state.checkpoint().unwrap();
    equation_context
        .begin_state_span(state.retained_values())
        .unwrap();
    let input = WorkspaceTensor::full_f32(100., &[2, 7, 8], &equation_context).unwrap();
    let visible = state.append_local(input, &equation_context).unwrap();
    assert_eq!(visible.shape(), [2, 11, 8]);
    assert_eq!(state.offset(), 12);
    let next = (0..2 * 7 * 8)
        .map(|n| 100. + n as f32 / 8.)
        .collect::<Vec<_>>();
    let native_input = MlxTensor::from_f32_slice(&next, &[2, 7, 8], &stream).unwrap();
    let actual = native
        .layer(0)
        .unwrap()
        .append_local(native_input, &stream)
        .unwrap();
    let initial = (0..2 * 5 * 8)
        .map(|n| 0.25 + n as f32 / 32.)
        .collect::<Vec<_>>();
    let expected_values = (0..2)
        .flat_map(|b| {
            initial[b * 40 + 8..(b + 1) * 40]
                .iter()
                .copied()
                .chain(next[b * 56..(b + 1) * 56].iter().copied())
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.shape(), visible.shape());
    assert_eq!(
        actual.as_array().evaluated().unwrap().as_slice::<f32>(),
        expected_values.as_slice()
    );
    // Checkpoint metadata keeps the exact old frontier and independent COW.
    state.restore(&saved, &equation_context).unwrap();
    assert_eq!(state.offset(), 5);
    state.clear().unwrap();
    assert_eq!(state.offset(), 0);
    assert_eq!(state.retained_values().count(), 0);
    drop((saved, visible, actual, equation, equation_context));

    // This is an invalid pooling frontier after the successful local-only
    // update: refusal follows accepted source pins and must own that prefix.
    let failure = match cold(|| {
        MlxPoolingAttentionStateFactory::project_complete_workspace_with_storage(
            &native,
            NonZeroU32::new(2).unwrap(),
            &context,
        )
    }) {
        Err(failure) => failure,
        Ok(_) => panic!("pooling stream frontier must match local source"),
    };
    assert_eq!(failure.retained_storage().unwrap().paged_sources().len(), 1);
    assert!(matches!(
        failure.cause().cause(),
        CacheSourceFailureCause::Metadata(_)
    ));
    drop((native, context, funding, tail, gates, pooled));
    assert!(owner.fixture_host_charge().unwrap() > 0);
    drop(projected);
    assert!(matches!(
        manager.remove_block(&id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    drop(failure);
    manager.remove_block(&id).unwrap();
    drop(manager);
    assert_eq!(owner.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
