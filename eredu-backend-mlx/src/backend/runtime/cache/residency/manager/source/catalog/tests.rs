use super::*;
use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("catalog installation runs no equations")
    }
}

#[test]
#[ignore = "requires native cache sources"]
fn paid_manager_catalog_preserves_actual_backing_pins_and_failed_installation_custody() {
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let first = Array::from_slice(&[1.25f32, -2.5], &[1, 1, 2, 1]);
    let second = Array::from_slice(&[3.5f32, 4.75], &[1, 1, 2, 1]);
    let id = manager
        .seal_block(
            7,
            0,
            2,
            None,
            CacheBlockArrays::KeyValue {
                keys: first,
                values: second,
            },
            true,
        )
        .unwrap();
    let selection = CacheBlockSelection::new(7, CacheRepresentation::KeyValue, 0, 2, 0);
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 24)
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let (pins, busy, prepared, identities) = manager
        .with_source_loan(selection, &context, |mut loan| {
            let pair = loan.blocks().next().unwrap().device().unwrap();
            let identities =
                pair.map(|array| array.try_allocation_info().unwrap().unwrap().identity());
            let busy = loan.prepare_catalog(3, 1, &context)?;
            let prepared = loan.prepare_catalog(3, 1, &context)?;
            let pins = loan.pin_selected(&context)?;
            Ok((pins, busy, prepared, identities))
        })
        .unwrap();
    let guard = manager.inner.state.lock().unwrap();
    let failure = busy
        .install()
        .err()
        .expect("busy manager refuses installation");
    assert!(matches!(
        failure.cause.cause(),
        CacheSourceFailureCause::Source(CacheSourceError::Busy)
    ));
    drop(guard);
    let installed = prepared.install().unwrap();
    assert!(installed.manager().same_catalog(&manager));
    assert_eq!(installed.initial_generation(), 0);
    manager
        .with_source_loan(selection, &context, |loan| {
            pins.validate_loan(&loan).unwrap();
            let pair = loan.blocks().next().unwrap().device().unwrap();
            assert_eq!(
                pair.map(|array| array.try_allocation_info().unwrap().unwrap().identity()),
                identities
            );
            assert_eq!(loan.lifecycle.lease_count(&id).unwrap(), 1);
            assert!(loan.lifecycle.is_protected_prefix(&id).unwrap());
            Ok(())
        })
        .unwrap();
    {
        let state = manager.inner.state.lock().unwrap();
        assert!(state.blocks.validate_prepared_population(3).is_ok());
        assert!(state.blocks.validate_prepared_population(4).is_err());
    }
    manager.record_attention_scan(7, true, 1, 16, 8).unwrap();
    let reference_report = manager.report().unwrap();
    {
        let mut state = manager.inner.state.lock().unwrap();
        super::super::super::reporting::update_report_totals_prepared(&mut state).unwrap();
        assert_eq!(state.telemetry.report, reference_report);
        assert!(state.report_rows.as_ref().unwrap().is_empty());
        assert!(state.report_rows.as_ref().unwrap().validate_prepared_population(1).is_ok());
        assert!(state.report_rows.as_ref().unwrap().validate_prepared_population(2).is_err());
    }
    drop(pins);
    let stale = manager
        .with_source_loan(selection, &context, |loan| {
            loan.prepare_catalog(3, 1, &context)
        })
        .unwrap();
    manager.clear().unwrap();
    let stale = stale
        .install()
        .err()
        .expect("changed generation refuses installation");
    assert!(matches!(
        stale.cause.cause(),
        CacheSourceFailureCause::Source(CacheSourceError::Identity)
    ));
    drop((context, funding, installed));
    assert!(pool.used_bytes().unwrap() > 0);
    // Both rejected destinations retain their original account after external
    // manager handles retire. They contain no cloned native backing.
    drop(manager);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(failure);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(stale);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

impl eredu_nn::workspace::WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("catalog preparation cannot quote equations")
    }
    fn write_operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceEffectDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("catalog preparation cannot emit equations")
    }
    fn host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("catalog preparation cannot quote host equations")
    }
    fn write_host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceHostDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("catalog preparation cannot emit host equations")
    }
}

#[test]
#[ignore = "requires native cache arrays"]
fn prepared_publication_metadata_validates_actual_arrays_and_outlives_catalog_failure() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 20).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let arrays = CacheBlockArrays::KeyValue {
        keys: Array::from_slice(&[1.25f32, -2.5], &[1, 1, 2, 1]),
        values: Array::from_slice(&[3.5f32, 4.75], &[1, 1, 2, 1]),
    };
    let metadata = CacheBlockMetadata::prepare_f32(CacheRepresentation::KeyValue,
        [&[1, 1, 2, 1], &[1, 1, 2, 1]], &context).unwrap();
    metadata.validate_arrays(&arrays).unwrap();
    let wrong = CacheBlockArrays::KeyValue {
        keys: Array::from_slice(&[1_i32, 2], &[1, 1, 2, 1]),
        values: Array::from_slice(&[3_i32, 4], &[1, 1, 2, 1]),
    };
    assert!(matches!(metadata.validate_arrays(&wrong), Err(CacheSourceError::Geometry)));
    let manager = CacheResidencyManager::new(PagedCacheOptions::new(2, 1 << 20, 0, 1).unwrap()).unwrap();
    let id = CacheBlockId { session_id: manager.session_id(), global_layer: 3, representation: CacheRepresentation::KeyValue,
        start: 0, end: 2, rank: None };
    // This record is deliberately never published. Its ID is descriptive and
    // carries no source/manager permission; escaped metadata still needs H.
    let record = metadata.into_record(id, arrays, false);
    assert_eq!(record.shapes, [vec![1, 1, 2, 1], vec![1, 1, 2, 1]]);
    assert_eq!(record.dtypes, ["Float32", "Float32"]);
    drop((context, funding, wrong, manager));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(record);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}


#[test]
#[ignore = "requires native cache sources"]
fn prepared_source_block_pin_preserves_demand_telemetry_foreign_refusal_and_escaped_lease_custody() {
    let make = || {
        let manager = CacheResidencyManager::new(PagedCacheOptions::new(2, 1 << 20, 0, 1).unwrap().with_full_attention(true)).unwrap();
        let id = manager.seal_block(7, 0, 2, None, CacheBlockArrays::KeyValue {
            keys: Array::from_slice(&[1.25f32, -2.5], &[1, 1, 2, 1]),
            values: Array::from_slice(&[3.5f32, 4.75], &[1, 1, 2, 1]),
        }, false).unwrap();
        (manager, id)
    };
    let (manager, id) = make();
    let (foreign, foreign_id) = make();
    let selection = CacheBlockSelection::new(7, CacheRepresentation::KeyValue, 0, 2, 0);
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 24).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let (pin, prepared) = manager.with_source_loan(selection, &context, |mut loan| {
        let prepared = loan.prepare_catalog(1, 0, &context)?;
        context.charge_metadata(PinnedCacheBlock::fixed_controls().unwrap())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), &context))?;
        assert!(matches!(loan.pin_prepared_block(&foreign_id, context.metadata_funding()), Err(CacheSourceError::Identity)));
        let pin = loan.pin_prepared_block(&id, context.metadata_funding())
            .map_err(|cause| CacheSourceFailure::source(cause, &context))?;
        Ok((pin, prepared))
    }).unwrap();
    let installed = prepared.install().unwrap();
    assert_eq!(manager.report().unwrap().demand_hits, 0);
    let lease = pin.acquire().unwrap();
    assert_eq!(manager.report().unwrap().demand_hits, 1);
    {
        let state = manager.inner.state.lock().unwrap();
        assert_eq!(state.lifecycle.lease_count(&id).unwrap(), 2);
    }
    assert!(matches!(manager.remove_block(&id), Err(CacheResidencyError::Lifecycle(CacheLifecycleError::BlockLeased(_)))));
    drop(pin);
    {
        let state = manager.inner.state.lock().unwrap();
        assert_eq!(state.lifecycle.lease_count(&id).unwrap(), 1);
    }
    drop((foreign, context, funding, installed, manager));
    assert!(pool.used_bytes().unwrap() > 0);
    // No array or native permission was fabricated by the count-only lease.
    // Its real manager/backing and host account survive every other owner.
    drop(lease);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}


#[test]
#[ignore = "requires native cache source workers"]
fn paid_independent_manager_preserves_source_namespace_pool_and_last_owner_custody() {
    let source = CacheResidencyManager::new(PagedCacheOptions::new(2, 1 << 20, 0, 1).unwrap()).unwrap();
    let id = source.seal_block(7, 0, 2, None, CacheBlockArrays::KeyValue {
        keys: Array::from_slice(&[1.25f32, -2.5], &[1,1,2,1]),
        values: Array::from_slice(&[3.5f32, 4.75], &[1,1,2,1]),
    }, false).unwrap();
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 24).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let prepared = source.with_source_loan(CacheBlockSelection::new(7, CacheRepresentation::KeyValue, 0, 2, 0), &context,
        |loan| loan.prepare_independent_manager(&context)).unwrap();
    assert!(prepared.matches_source(&source, 0));
    assert!(!prepared.matches_source(&source, 1));
    let destination = prepared.destination().clone();
    assert_ne!(source.session_id(), destination.session_id());
    assert!(!prepared.matches_source(&destination, 0));
    assert_eq!(source.pool(), destination.pool());
    assert_eq!(source.pool().report().unwrap().managers, 2);
    assert_eq!(source.report().unwrap().key_value_blocks, 1);
    assert_eq!(destination.report().unwrap().key_value_blocks, 0);
    assert!(Arc::ptr_eq(&source.inner.host_demotion_worker, &destination.inner.host_demotion_worker));
    assert!(Arc::ptr_eq(source.inner.disk_worker.as_ref().unwrap(), destination.inner.disk_worker.as_ref().unwrap()));
    {
        let state = destination.inner.state.lock().unwrap();
        state.blocks.validate_prepared_population(1).unwrap();
        assert!(state.blocks.validate_prepared_population(2).is_err());
    }
    let duplicate = source.pool().prepare_manager_registration(&context).unwrap()
        .register(destination.session_id()).unwrap_err();
    assert!(matches!(duplicate.cause(), CachePoolError::DuplicateManager { manager } if *manager == destination.session_id()));
    // No source record or array is reidentified, copied or published by this
    // destination constructor. Numerical source/copy authority remains required.
    assert!(source.inner.state.lock().unwrap().blocks.contains_key(&id));
    drop((prepared, context, funding, source));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(destination.pool().report().unwrap().managers, 1);
    drop(destination);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(duplicate);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
