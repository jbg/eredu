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

fn host(pool: &WorkingMemoryPool) -> HostController {
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

fn contribution(
    pool: &WorkingMemoryPool,
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

fn combined(pool: &WorkingMemoryPool, host: &HostController) -> IncrementalInferenceQuote {
    let g = geometry();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let registered =
        RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())]).unwrap();
    let equations = replacement_report(&context, &root, g);
    let enclosing = contribution(pool, g, host, true, outside(g, 40));
    ResidualInferenceQuote::compose_with_controller(&equations, state(g), enclosing, &registered)
        .unwrap()
        .into_incremental()
}

fn plan_incremental_quote(
    pool: &WorkingMemoryPool,
    quote: &IncrementalInferenceQuote,
    request: AdmissionRequest,
    capacity: u64,
) -> Result<
    (
        WorkingMemoryReservation,
        IncrementalInferenceQuote,
    ),
    PrefillPlanningError,
> {
    plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &capabilities(),
        request,
        quote.geometry(),
        capacity,
        |_| Ok(quote.clone()),
    )
}

#[test]
fn fixed_controller_credit_preserves_full_equations_and_combines_with_decoder_identity_credit() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let g = geometry();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
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
    assert_eq!(full.incremental_bytes(), 228);
    assert_eq!(host_only.incremental_bytes(), 196);
    assert_eq!(both.incremental_bytes(), 148);
    assert_eq!(full.state(), host_only.state());
    assert_eq!(full.state(), both.state());
    assert!(both.registered_storage().pool().same_domain(&pool));
    let both = both.into_incremental();
    assert!(both.pool().same_domain(&pool));
    assert_eq!(
        both.controller_contract().unwrap().additional_host_bytes(),
        44
    );
    assert_ne!(
        both.incremental_bytes(),
        full.incremental_bytes() - 64 - 32,
        "decoder replacement overlap cannot be discounted from a scalar peak"
    );
    assert!(matches!(
        plan_incremental_quote(&pool, &both, request(g), 243),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded {
                required_bytes: 148,
                available_bytes: 147
            }
        ))
    ));
    assert_eq!(used(&pool), (96, 96));
    let (reservation, accepted) =
        plan_incremental_quote(&pool, &both, request(g), 244).unwrap();
    assert_eq!(reservation.admission().state, *full.state());
    assert_eq!(reservation.bytes(), 148);
    assert!(reservation.requires_funding_scope());
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), reservation.admission()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 244);
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
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn combined_pins_transfer_to_run_and_scopes_without_becoming_historical_metadata() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let quote = combined(&pool, &host);
    let (reservation, accepted) =
        plan_incremental_quote(&pool, &quote, request(geometry()), 244).unwrap();
    drop((quote, accepted, host, original));
    assert_eq!(pool.used_bytes().unwrap(), 244);
    let (metadata, run) = reservation.into_funding().unwrap();
    let first = run.scope().unwrap();
    let last = run.scope().unwrap();
    drop(run);
    first.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 244);
    last.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(metadata);
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn abandoned_combined_work_quarantines_both_sources_after_request_metadata_drops() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let quote = combined(&pool, &host);
    let (reservation, accepted) =
        plan_incremental_quote(&pool, &quote, request(geometry()), 244).unwrap();
    let (metadata, run) = reservation.into_funding().unwrap();
    let work = run.scope().unwrap();
    drop((metadata, run, quote, accepted, host, original));
    drop(work);
    assert_eq!(pool.used_bytes().unwrap(), 244);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn incremental_policy_keeps_full_completeness_geometry_and_safety_requirements() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let quote = combined(&pool, &host);
    let g = geometry();
    let mut req = request(g);
    req.safety_reserve_bytes = 7;
    req.application_memory_budget_bytes = Some(154);
    assert!(matches!(
        plan_incremental_quote(&pool, &quote, req, 1000),
        Err(PrefillPlanningError::Admission(
            AdmissionRejection::MemoryBudgetExceeded {
                required_bytes: 155,
                budget_bytes: 154
            }
        ))
    ));
    req.application_memory_budget_bytes = Some(155);
    assert!(matches!(
        plan_incremental_quote(&pool, &quote, req, 250),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (reservation, accepted) = plan_incremental_quote(&pool, &quote, req, 251).unwrap();
    assert_eq!(reservation.bytes(), 155);
    drop((reservation, accepted));
    let foreign = WorkingMemoryPool::new(1000, 0).unwrap();
    assert!(matches!(
        plan_incremental_quote(&foreign, &quote, request(g), 1000),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));

    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
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
    permissive.require_complete_estimate = false;
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
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn combined_credit_uses_the_shared_smaller_chunk_planner() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let g = InferenceGeometry {
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 3,
        ..geometry()
    };
    let mut candidates = Vec::new();
    let (reservation, quote) = plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(g),
        g,
        180,
        |candidate| {
            candidates.push(candidate.prefill_chunk_positions);
            let context = WorkspaceContext::new(Facts::default());
            let root = WorkspaceExistingStorage::new(Some(64), &context);
            let registered =
                RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
            let old = view(&context, &root, 2);
            let equations = quote_inference_workspace(candidate, |span| {
                context.begin_state_span([&old])?;
                let count = match span {
            InferenceWorkspaceSpan::Sampling(_) => unreachable!("model-only traversal fixture"),
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
        },
    )
    .unwrap();
    assert_eq!(candidates, [3, 2]);
    assert_eq!(quote.geometry().prefill_chunk_positions, 2);
    assert_eq!(reservation.bytes(), 84);
    assert_eq!(pool.used_bytes().unwrap(), 180);
    drop((quote, reservation, host, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn required_registered_inputs_preserve_the_original_controller_and_decoder_quote() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let source = pool.register_storage([(2u32, 24)]).unwrap();
    let quote = combined(&pool, &host);
    let joined = quote
        .clone()
        .with_registered_sources(source.clone())
        .unwrap();
    assert_eq!(joined.controller_contract(), quote.controller_contract());
    assert_eq!(joined.state(), quote.state());
    assert_eq!(joined.geometry(), quote.geometry());
    assert_eq!(joined.incremental_bytes(), 148);
    let (reservation, accepted) =
        plan_incremental_quote(&pool, &joined, request(geometry()), 268).unwrap();
    drop((quote, joined, accepted, root, host, source));
    assert_eq!(pool.used_bytes().unwrap(), 268);
    drop(reservation);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn retained_span_host_and_original_terms_receive_no_controller_or_decoder_discount() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let host = host(&pool);
    let g = geometry();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
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
    let full_before = full.incremental_bytes();
    let credited_before = credited.incremental_bytes();
    assert!(credited_before < full_before);
    let h = full.span_workspace().retention_peak_bytes().unwrap();
    let full = full.with_span_workspace().unwrap();
    let credited = credited.with_span_workspace().unwrap();
    assert_eq!(full.incremental_bytes() - full_before, h);
    assert_eq!(credited.incremental_bytes() - credited_before, h);
    assert!(full
        .span_workspace()
        .plan()
        .same_plan(credited.span_workspace().plan()));
    assert_eq!(full.span_workspace().source_preparation_bytes(), Some(40));
    for i in 0..equations.span_workspace_plan().records().len() {
        assert_eq!(
            full.span_workspace().span_bytes(i),
            credited.span_workspace().span_bytes(i)
        );
    }
    assert_eq!(
        pool.used_bytes().unwrap(),
        64 + host.sources[0].capacity_bytes().unwrap()
    );
    drop((full, credited, registered, original, host));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
