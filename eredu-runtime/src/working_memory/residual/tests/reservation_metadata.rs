use super::*;
use crate::working_memory::{WorkspaceReportMetadata, quote_inference_workspace_with_context};
use std::convert::Infallible;

// This fixture retains a real 64-byte source and adds a 128-byte complete
// enclosing demand. Its equation spans leave that source unchanged.
#[derive(Debug)]
struct UnchangedSource;
impl WorkspaceMechanisms for UnchangedSource {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("unchanged source fixture must not invoke ordinary tensor facts")
    }
}
impl WorkspaceFactMechanisms for UnchangedSource {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

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

fn funded_quote(
    pool: &MemoryLedger,
    funding: &HostMetadataFunding,
    g: InferenceGeometry,
) -> IncrementalInferenceQuote {
    let context =
        WorkspaceContext::new_with_metadata_funding(UnchangedSource, funding.clone()).unwrap();
    let root = WorkspaceExistingStorage::try_new_placed(
        Some(64),
        crate::working_memory::memory_fixture::host_placement(),
        &context,
    )
    .unwrap();
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
        outside(g, 128 + publication_controls()),
        &registered,
        WorkspaceReportMetadata::new(&context),
    )
    .unwrap()
    .into_incremental()
}

#[test]
fn quote_aliases_share_paid_diagnostics_and_sealing_copies_only_shared_reports() {
    let capacity = 1 << 22;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let quote = funded_quote(&pool, &funding, geometry());
    let before = (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool));
    let alias = quote.clone();
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        before
    );
    assert!(std::ptr::eq(quote.state(), alias.state()));
    let original_retained = quote
        .state()
        .execution_workspace
        .as_ref()
        .unwrap()
        .retained
        .bytes()
        .unwrap();

    let sealed = alias.with_span_workspace().unwrap();
    assert!(!std::ptr::eq(quote.state(), sealed.state()));
    assert_eq!(
        quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .retained
            .bytes(),
        Some(original_retained)
    );
    assert!(
        sealed
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .retained
            .bytes()
            .unwrap()
            > original_retained
    );
    let before = (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool));
    let sealed_alias = sealed.clone();
    assert!(std::ptr::eq(sealed.state(), sealed_alias.state()));
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        before
    );
    drop((quote, sealed, funding));
    assert!((pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)) > 64);
    assert!(sealed_alias.state().execution_workspace.is_some());
    drop(sealed_alias);
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        64
    );

    // A unique candidate can add the same diagnostics without replacing its
    // report shell or copying any existing string/window backing.
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let quote = funded_quote(&pool, &funding, geometry());
    let address = quote.state() as *const RuntimeStateEstimate;
    let sealed = quote.with_span_workspace().unwrap();
    assert_eq!(address, sealed.state() as *const RuntimeStateEstimate);
    drop((sealed, funding));
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        64
    );
    drop(original);
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        0
    );
}

#[test]
fn shared_quote_report_refusal_preserves_original_diagnostics_and_custody() {
    let capacity = 1 << 22;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let quote = funded_quote(&pool, &funding, geometry());
    let mut alias = quote.clone();
    let blocker = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    blocker
        .reserve_metadata((capacity - physical_used(&pool)) as usize)
        .unwrap();
    let before = (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool));
    let failure = alias.state.make_mut().unwrap_err();
    assert!(matches!(
        failure,
        crate::working_memory::WorkspaceReportError::Metadata(_)
    ));
    assert!(std::ptr::eq(quote.state(), alias.state()));
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        before
    );
    drop((blocker, failure, quote, funding));
    assert!((pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)) > 64);
    // The surviving alias can still fund its unique mutation after the blocker
    // and earlier callers retire. This never refunds the retained report early.
    alias.state.make_mut().unwrap();
    drop(alias);
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        64
    );
    drop(original);
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        0
    );
}

#[test]
fn planned_reservation_refuses_before_q_publication_and_retains_final_metadata_or_node() {
    for keep_allocation in [false, true] {
        let capacity = 1 << 20;
        let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let original = pool.register_host_storage([(1u32, 64)]).unwrap();
        let funding = pool
            .prepare_workspace_metadata(
                &execution,
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
            )
            .unwrap();
        let g = geometry();
        let quote = funded_quote(&pool, &funding, g);
        let admission = Admission {
            requested_positions: g.cached_positions + g.input_positions + g.max_output_tokens,
            state: quote.state().clone(),
            incremental_required_bytes: quote.incremental_bytes(),
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
        };
        assert!(admission.incremental_required_bytes >= Some(32));

        // Another real planning account leaves one byte. The new reservation's
        // first constructor must refuse before a Q node or numeric commit exists.
        let blocker = pool
            .prepare_workspace_metadata(
                &execution,
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
            )
            .unwrap();
        blocker
            .reserve_metadata((capacity - physical_used(&pool) - 1) as usize)
            .unwrap();
        let before = {
            let usage = pool.0.usage.lock().unwrap();
            (usage.reserved, usage.reservations, usage.next_funding)
        };
        assert!(
            matches!(quote.reserve(&pool, &execution, &admission, crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity), &[]),
            Err(capacity_error) if matches!(capacity_numbers(&capacity_error), Some((required_bytes, 1)) if required_bytes > 1))
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
            .reserve(
                &pool,
                &execution,
                &admission,
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
                &[],
            )
            .unwrap();
        assert_eq!(reservation.admission(), &admission);
        let q = reservation_payload_bytes(&reservation);
        let planning =
            (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)) - 64 - q;
        assert!(planning > 0);
        let (metadata, run) = reservation.into_funding().unwrap();
        let allocation = if keep_allocation {
            let scope = run.scope().unwrap();
            let allocation = scope
                .adopt_host_storage_individually([(2u32, 32)])
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
            assert_eq!(
                (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
                64 + 32 + planning
            );
            drop(allocation);
        } else {
            // Run closure refunds Q, while the historical Admission copy stays
            // charged until the last reservation metadata owner retires.
            assert_eq!(
                (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
                64 + planning
            );
            assert_eq!(metadata.admission().incremental_required_bytes, Some(q));
            drop(metadata);
        }
        assert_eq!(
            (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
            64
        );
        drop(original);
        assert_eq!(
            (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
            0
        );
    }
}

#[test]
fn incremental_candidate_pays_diagnostics_before_reservation_and_retains_them() {
    let capacity = 1 << 20;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let g = geometry();
    let quote = funded_quote(&pool, &funding, g);
    let blocker = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    blocker
        .reserve_metadata((capacity - physical_used(&pool) - 1) as usize)
        .unwrap();
    let before = {
        let usage = pool.0.usage.lock().unwrap();
        (usage.reserved, usage.reservations, usage.next_funding)
    };
    let mut quote = Some(quote);
    let error = plan_prefill_incremental_with_capacity(
        &execution,
        &pool,
        &capabilities(),
        request(g),
        g,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        |_| {
            Ok(quote
                .take()
                .expect("one-position candidate is attempted once"))
        },
    )
    .unwrap_err();
    assert!(matches!(error, PrefillPlanningError::Reservation(
        capacity_error
    ) if matches!(capacity_numbers(&capacity_error), Some((_, 1)))));
    {
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(
            (usage.reserved, usage.reservations, usage.next_funding),
            before
        );
    }
    drop(blocker);
    let mut quote = Some(funded_quote(&pool, &funding, g));
    let (reservation, quote) = plan_prefill_incremental_with_capacity(
        &execution,
        &pool,
        &capabilities(),
        request(g),
        g,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        |_| {
            Ok(quote
                .take()
                .expect("one-position candidate is attempted once"))
        },
    )
    .unwrap();
    assert_eq!(&reservation.admission().state, quote.state());
    drop(quote);
    drop(funding);
    assert!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool))
            > 64 + reservation_payload_bytes(&reservation)
    );
    drop(reservation);
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        64
    );
    drop(original);
    assert_eq!(
        (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
        0
    );
}

#[test]
fn retained_policy_diagnostic_preserves_retry_classification_and_account() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    for retryable in [false, true] {
        let funding = pool
            .prepare_workspace_metadata(
                &execution,
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 20),
            )
            .unwrap();
        let metadata = WorkspaceReportMetadata::with_funding(&funding);
        let reason = metadata
            .text(format_args!("exact selected mechanism has no bound"))
            .unwrap();
        let rejection = if retryable {
            AdmissionRejection::EstimationUnsupported { reason }
        } else {
            {
                drop(reason);
                AdmissionRejection::PromptExceedsContext {
                    prompt_positions: 2,
                    maximum_positions: 1,
                }
            }
        };
        let error: PrefillPlanningError =
            crate::working_memory::reservation_metadata::neural_error(
                metadata.source(PrefillPlanningError::Admission(rejection)),
                &funding,
            )
            .into();
        assert_eq!(
            crate::working_memory::candidate_can_shrink(&error),
            retryable
        );
        drop(funding);
        assert!((pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)) > 0);
        drop(error);
        assert_eq!(
            (pool.payload_used_bytes().unwrap() + reservation_controls_used(&pool)),
            0
        );
    }
}
