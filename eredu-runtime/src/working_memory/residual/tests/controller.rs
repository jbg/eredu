use super::*;
use crate::working_memory::{ControllerStorageContract, ControllerWorkspaceContribution};
use eredu_core::{
    SharedTokenFilter, TextControllerStorage, TextControllerWorkspace, TokenFilter,
    TokenFilterController,
};

struct HostController {
    sources: [SharedTokenFilter; 1],
}

impl TokenFilterController for HostController {
    type Error = std::convert::Infallible;

    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: self.sources[0].as_ref().into(),
            // The immutable source and twelve bytes of independent run state
            // coexist. The final emitted mask belongs to the outside fixture.
            additional_host_bytes: self.sources[0].capacity_bytes()?.checked_add(12)?,
        })
    }

    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithSharedFilters(&self.sources)
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("cold composition must not query a controller decision")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("cold composition must not commit a token")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("cold composition must not query controller state")
    }
}

fn host(pool: &MemoryLedger) -> HostController {
    HostController {
        sources: [pool
            .prepare_shared_token_filter(|| {
                let mut bits = Vec::with_capacity(32);
                bits.extend([true, false, true]);
                TokenFilter::allowed(bits).unwrap()
            })
            .unwrap()],
    }
}

fn controller_controls() -> u64 {
    crate::working_memory::controller::controller_publication_layout()
        .unwrap()
        .requested_bytes()
        .checked_add(MemoryLedger::storage_metadata_control_bytes().unwrap())
        .unwrap()
}

fn contribution(
    pool: &MemoryLedger,
    g: InferenceGeometry,
    host: &HostController,
    credited: bool,
    outside: ExecutionWorkspaceEstimate,
) -> ControllerWorkspaceEstimate {
    let contract = ControllerStorageContract::inspect(host).unwrap();
    let registered = credited.then(|| contract.pin_registered(host, pool).unwrap());
    ControllerWorkspaceContribution::new(
        g,
        host.inference_workspace(g.max_output_tokens).unwrap(),
        3,
        &contract,
        pool,
        registered.as_ref(),
    )
    .unwrap()
    .compose(outside)
    .unwrap()
}

fn combined(pool: &MemoryLedger, host: &HostController) -> IncrementalInferenceQuote {
    let g = geometry();
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let registered =
        RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())]).unwrap();
    let equations = replacement_report(&context, &root, g);
    let enclosing = contribution(pool, g, host, true, outside(g, 40));
    ResidualInferenceQuote::compose_with_controller(&equations, state(g), enclosing, &registered)
        .unwrap()
        .into_incremental()
}

fn plan_incremental_quote(
    pool: &MemoryLedger,
    quote: &IncrementalInferenceQuote,
    request: AdmissionRequest,
    capacity: u64,
) -> Result<(WorkingMemoryReservation, IncrementalInferenceQuote), PrefillPlanningError> {
    let capacity = payload_capacity_with_quote(pool, quote, &request, capacity);
    plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &capabilities(),
        request,
        quote.geometry(),
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        |_| Ok(quote.clone()),
    )
}

#[test]
fn fixed_controller_credit_preserves_full_equations_and_combines_with_decoder_identity_credit() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let g = geometry();
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let registered =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let equations = replacement_report(&context, &root, g);
    let full = IncrementalInferenceQuote::compose_controller(
        &equations,
        state(g),
        contribution(&pool, g, &host, false, outside(g, 40)),
    )
    .unwrap();
    let host_only = IncrementalInferenceQuote::compose_controller(
        &equations,
        state(g),
        contribution(&pool, g, &host, true, outside(g, 40)),
    )
    .unwrap();
    let both = ResidualInferenceQuote::compose_with_controller(
        &equations,
        state(g),
        contribution(&pool, g, &host, true, outside(g, 40)),
        &registered,
    )
    .unwrap();
    assert_eq!(
        full.incremental_bytes().unwrap(),
        (228 + controller_controls())
    );
    assert_eq!(
        host_only.incremental_bytes().unwrap(),
        (196 + controller_controls())
    );
    assert_eq!(
        both.incremental_bytes().unwrap(),
        (148 + controller_controls())
    );
    assert_eq!(full.state(), host_only.state());
    assert_eq!(full.state(), both.state());
    assert!(both.registered_storage().pool().same_ledger(&pool));
    let both = both.into_incremental();
    assert!(both.pool().same_ledger(&pool));
    assert_eq!(
        both.controller_contract().unwrap().additional_host_bytes(),
        44
    );
    assert_ne!(
        both.incremental_bytes().unwrap(),
        full.incremental_bytes().unwrap() - 64 - 32,
        "decoder replacement overlap cannot be discounted from a scalar peak"
    );
    assert!(matches!(
        plan_incremental_quote(&pool, &both, request(g), (243 + controller_controls())),
        Err(PrefillPlanningError::Reservation(
            capacity_error
        )) if matches!(capacity_numbers(&capacity_error), Some((required, available)) if required == available + 1)));
    assert_eq!(pool.payload_used_bytes().unwrap(), 96);
    let (reservation, accepted) =
        plan_incremental_quote(&pool, &both, request(g), (244 + controller_controls())).unwrap();
    assert_eq!(reservation.admission().state, *full.state());
    assert_eq!(
        reservation_payload_bytes(&reservation),
        (148 + controller_controls())
    );
    assert!(reservation.requires_funding_scope());
    assert!(matches!(
        pool.reserve(
            &InferenceExecutionIdentity::default(),
            reservation.admission()
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut complete = reservation.admission().clone();
    complete.incremental_required_bytes = complete.state.requested_state_bytes.checked_add(
        complete
            .state
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .unwrap(),
    );
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &complete),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        (244 + controller_controls())
    );
    drop((
        reservation,
        accepted,
        both,
        host_only,
        full,
        registered,
        host,
        original,
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn combined_pins_transfer_to_run_and_scopes_without_becoming_historical_metadata() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let quote = combined(&pool, &host);
    let (reservation, accepted) = plan_incremental_quote(
        &pool,
        &quote,
        request(geometry()),
        (244 + controller_controls()),
    )
    .unwrap();
    drop((quote, accepted, host, original));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        (244 + controller_controls())
    );
    let (metadata, run) = reservation.into_funding().unwrap();
    let first = run.scope().unwrap();
    let last = run.scope().unwrap();
    drop(run);
    first.certify().unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        (244 + controller_controls())
    );
    last.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(metadata);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn abandoned_combined_work_quarantines_both_sources_after_request_metadata_drops() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let quote = combined(&pool, &host);
    let (reservation, accepted) = plan_incremental_quote(
        &pool,
        &quote,
        request(geometry()),
        (244 + controller_controls()),
    )
    .unwrap();
    let (metadata, run) = reservation.into_funding().unwrap();
    let work = run.scope().unwrap();
    drop((metadata, run, quote, accepted, host, original));
    drop(work);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        (244 + controller_controls())
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn incremental_policy_keeps_full_completeness_geometry_and_safety_requirements() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let quote = combined(&pool, &host);
    let g = geometry();
    let mut req = request(g);
    req.additional_headroom = eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 7)]);
    req.memory_limits = crate::working_memory::memory_fixture::host_limits(0);
    let exact = payload_capacity_with_quote(&pool, &quote, &req, (251 + controller_controls()));
    req.memory_limits = crate::working_memory::memory_fixture::host_limits(exact - 1);
    assert!(matches!(
        plan_incremental_quote(&pool, &quote, req.clone(), 1000),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    ));
    req.memory_limits = crate::working_memory::memory_fixture::host_limits(exact);
    assert!(matches!(
        plan_incremental_quote(&pool, &quote, req.clone(), 250),
        Err(PrefillPlanningError::Reservation(
            capacity_error
        )) if matches!(capacity_numbers(&capacity_error), Some((_, _)))));
    let (reservation, accepted) =
        plan_incremental_quote(&pool, &quote, req, (251 + controller_controls())).unwrap();
    assert_eq!(
        reservation_payload_bytes(&reservation),
        155 + controller_controls()
    );
    drop((reservation, accepted));
    let foreign = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    assert!(matches!(
        plan_incremental_quote(&foreign, &quote, request(g), 1000),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));

    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let equations = replacement_report(&context, &root, g);
    let changed = InferenceGeometry {
        max_output_tokens: g.max_output_tokens + 1,
        ..g
    };
    assert!(matches!(
        IncrementalInferenceQuote::compose_controller(
            &equations,
            state(g),
            contribution(&pool, changed, &host, true, outside(changed, 40))
        ),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let mut partial = state(g);
    partial.persistent_state_completeness = EstimationCompleteness::PersistentStateOnly;
    let partial = IncrementalInferenceQuote::compose_controller(
        &equations,
        partial,
        contribution(&pool, g, &host, true, outside(g, 40)),
    )
    .unwrap();
    let mut permissive = request(g);
    assert!(matches!(
        plan_incremental_quote(&pool, &partial, permissive, 1000),
        Err(PrefillPlanningError::Admission(
            AdmissionRejection::EstimationUnsupported { .. }
        ))
    ));
    let mut unknown = outside(g, 40);
    unknown.attention = WorkspaceBound::Unknown {
        reason: "no primitive fact".into(),
    };
    assert!(matches!(
        IncrementalInferenceQuote::compose_controller(
            &equations,
            state(g),
            contribution(&pool, g, &host, true, unknown)
        ),
        Err(ResidualQuoteError::IncompleteWorkspace(_))
    ));
    drop((partial, quote, host, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn combined_credit_uses_the_shared_smaller_chunk_planner() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let g = InferenceGeometry {
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 3,
        ..geometry()
    };
    let mut candidates = Vec::new();
    let mut provider =
        |candidate: InferenceGeometry| -> Result<IncrementalInferenceQuote, PrefillPlanningError> {
            candidates.push(candidate.prefill_chunk_positions);
            let context = WorkspaceContext::new(Facts::default());
            let root = placed_root(Some(64), &context);
            let registered =
                RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
            let old = view(&context, &root, 2);
            let equations = quote_inference_workspace(candidate, |span| {
                context.begin_state_span([&old])?;
                let count = match span {
                    InferenceWorkspaceSpan::Sampling(_) => {
                        unreachable!("model-only traversal fixture")
                    }
                    InferenceWorkspaceSpan::Prefill(chunk) => chunk.input.end - chunk.input.start,
                    InferenceWorkspaceSpan::Decode { .. } => 1,
                };
                let _temporary = WorkspaceTensor::full_f32(0.5, &[count as i32 * 4], &context)?;
                context.report(&[old.clone()])
            })
            .unwrap();
            Ok(ResidualInferenceQuote::compose_with_controller(
                &equations,
                state(candidate),
                contribution(&pool, candidate, &host, true, outside(candidate, 40)),
                &registered,
            )
            .unwrap()
            .into_incremental())
        };
    let probe = provider(InferenceGeometry {
        prefill_chunk_positions: 2,
        ..g
    })
    .unwrap();
    let capacity =
        payload_capacity_with_quote(&pool, &probe, &request(g), (180 + controller_controls()));
    drop(probe);
    let (reservation, quote) = plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(g),
        g,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        &mut provider,
    )
    .unwrap();
    assert_eq!(candidates, [2, 3, 2]);
    assert_eq!(quote.geometry().prefill_chunk_positions, 2);
    assert_eq!(
        reservation_payload_bytes(&reservation),
        (84 + controller_controls())
    );
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        (180 + controller_controls())
    );
    drop((quote, reservation, host, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn required_registered_inputs_preserve_the_original_controller_and_decoder_quote() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let source = pool.register_host_storage([(2u32, 24)]).unwrap();
    let quote = combined(&pool, &host);
    let joined = quote
        .clone()
        .with_registered_sources(source.clone())
        .unwrap();
    assert_eq!(joined.controller_contract(), quote.controller_contract());
    assert_eq!(joined.state(), quote.state());
    assert_eq!(joined.geometry(), quote.geometry());
    assert_eq!(
        joined.incremental_bytes().unwrap(),
        (148 + controller_controls())
    );
    let (reservation, accepted) = plan_incremental_quote(
        &pool,
        &joined,
        request(geometry()),
        (268 + controller_controls()),
    )
    .unwrap();
    drop((quote, joined, accepted, root, host, source));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        (268 + controller_controls())
    );
    drop(reservation);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn retained_span_host_and_original_terms_receive_no_controller_or_decoder_discount() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let g = geometry();
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let registered =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let equations = replacement_report(&context, &root, g);
    let full = IncrementalInferenceQuote::compose_controller(
        &equations,
        state(g),
        contribution(&pool, g, &host, false, outside(g, 40)),
    )
    .unwrap();
    let credited = ResidualInferenceQuote::compose_with_controller(
        &equations,
        state(g),
        contribution(&pool, g, &host, true, outside(g, 40)),
        &registered,
    )
    .unwrap()
    .into_incremental();
    let full_before = full.incremental_bytes().unwrap();
    let credited_before = credited.incremental_bytes().unwrap();
    assert!(credited_before < full_before);
    let h = full.span_workspace().retention_peak_bytes().unwrap();
    let full = full.with_span_workspace().unwrap();
    let credited = credited.with_span_workspace().unwrap();
    assert_eq!(full.incremental_bytes().unwrap() - full_before, h);
    assert_eq!(credited.incremental_bytes().unwrap() - credited_before, h);
    assert!(
        full.span_workspace()
            .plan()
            .same_plan(credited.span_workspace().plan())
    );
    assert_eq!(full.span_workspace().source_preparation_bytes(), Some(40));
    for i in 0..equations.span_workspace_plan().records().len() {
        assert_eq!(
            full.span_workspace().span_bytes(i),
            credited.span_workspace().span_bytes(i)
        );
    }
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        64 + host.sources[0].capacity_bytes().unwrap()
    );
    drop((full, credited, registered, original, host));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
