use super::*;
use crate::working_memory::{quote_inference_workspace_with_context, WorkspaceReportMetadata};
use std::convert::Infallible;

// This fixture retains a real 64-byte source and adds a 128-byte complete
// enclosing demand. Its equation spans leave that source unchanged.
#[derive(Debug)]
struct UnchangedSource;
impl WorkspaceMechanisms for UnchangedSource {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("unchanged source fixture must not invoke ordinary tensor facts")
    }
}
impl WorkspaceFactMechanisms for UnchangedSource {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("unchanged source fixture must not invoke tensor operations")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        unreachable!()
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        unreachable!()
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        unreachable!()
    }
}

#[test]
fn planned_reservation_refuses_before_q_publication_and_retains_final_metadata_or_node() {
    for keep_allocation in [false, true] {
        let capacity = 1 << 20;
        let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let original = pool.register_storage([(1u32, 64)]).unwrap();
        let funding = pool
            .prepare_workspace_metadata(&execution, capacity)
            .unwrap();
        let g = geometry();
        let quote = {
            let context =
                WorkspaceContext::new_with_metadata_funding(UnchangedSource, funding.clone())
                    .unwrap();
            let root = WorkspaceExistingStorage::try_new(Some(64), &context).unwrap();
            let layout = RegisteredWorkspaceStorageLayout::<u32>::new(1).unwrap();
            context.charge_metadata(layout.requested_bytes()).unwrap();
            let registered = layout
                .construct(&pool, &context, [(1u32, root.clone())])
                .unwrap();
            let current = WorkspaceTensor::existing_with_storage(
                context.layout(&[2], WorkspaceDtype::Float32).unwrap(),
                &root,
                &context,
            )
            .unwrap();
            let equations = quote_inference_workspace_with_context(g, &context, |_| {
                context.begin_state_span([&current])?;
                context.finish_report(std::slice::from_ref(&current))
            })
            .unwrap();
            ResidualInferenceQuote::compose_metadata(
                &equations,
                state(g),
                outside(g, 128),
                &registered,
                WorkspaceReportMetadata::new(&context),
            )
            .unwrap()
            .into_incremental()
        };
        let admission = Admission {
            requested_positions: g.cached_positions + g.input_positions + g.max_output_tokens,
            state: quote.state().clone(),
            incremental_required_bytes: quote.incremental_bytes(),
            available_memory_bytes: None,
        };
        assert!(admission.incremental_required_bytes >= 32);

        // Another real planning account leaves one byte. The new reservation's
        // first constructor must refuse before a Q node or numeric commit exists.
        let blocker = pool
            .prepare_workspace_metadata(&execution, capacity)
            .unwrap();
        blocker
            .reserve_metadata((capacity - pool.used_bytes().unwrap() - 1) as usize)
            .unwrap();
        let before = {
            let usage = pool.0.usage.lock().unwrap();
            (usage.reserved, usage.reservations, usage.next_funding)
        };
        assert!(
            matches!(quote.reserve(&pool, &execution, &admission, capacity, &[]),
            Err(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes: 1 }) if required_bytes > 1)
        );
        {
            let usage = pool.0.usage.lock().unwrap();
            assert_eq!(
                (usage.reserved, usage.reservations, usage.next_funding),
                before
            );
        }
        drop(blocker);

        let reservation = quote
            .reserve(&pool, &execution, &admission, capacity, &[])
            .unwrap();
        assert_eq!(reservation.admission(), &admission);
        let q = reservation.bytes();
        let planning = pool.used_bytes().unwrap() - 64 - q;
        assert!(planning > 0);
        let (metadata, run) = reservation.into_funding().unwrap();
        let allocation = if keep_allocation {
            let scope = run.scope().unwrap();
            let allocation = scope
                .adopt_storage_individually([(2u32, 32)])
                .unwrap()
                .remove(&2)
                .unwrap();
            scope.certify().unwrap();
            Some(allocation)
        } else {
            None
        };
        drop(quote);
        drop(admission);
        drop(funding);
        run.close().unwrap();

        if keep_allocation {
            drop(metadata);
            // Only the real Q allocation remains: it keeps the Q node, which
            // must independently retain the planning account that built it.
            assert_eq!(pool.used_bytes().unwrap(), 64 + 32 + planning);
            drop(allocation);
        } else {
            // Run closure refunds Q, while the historical Admission copy stays
            // charged until the last reservation metadata owner retires.
            assert_eq!(pool.used_bytes().unwrap(), 64 + planning);
            assert_eq!(metadata.admission().incremental_required_bytes, q);
            drop(metadata);
        }
        assert_eq!(pool.used_bytes().unwrap(), 64);
        drop(original);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
