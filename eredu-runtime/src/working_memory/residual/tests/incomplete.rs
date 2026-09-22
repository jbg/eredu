use super::*;
use crate::working_memory::{ControllerStorageContract, ControllerWorkspaceContribution};
use eredu_core::{
    TextControllerStorage, TextControllerWorkspace, TokenFilter, TokenFilterController,
};
use std::error::Error as _;

#[test]
fn full_request_gap_identifies_the_required_component_without_reserving_storage() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let g = geometry();
    let before = pool.snapshot().unwrap();
    for (index, label) in [
        "full request activations",
        "full request attention",
        "full request vocabulary",
        "full request state update",
        "full request materialization",
        "full request retained storage",
        "full request physical-domain attribution",
    ]
    .into_iter()
    .enumerate()
    {
        let mut workspace = outside(g, 0);
        if index == 6 {
            workspace.physical_domains = None;
            workspace.activations = WorkspaceBound::PerDomain {
                assumptions: "missing attribution fixture".into(),
            };
        } else {
            let bounds = [
                &mut workspace.activations,
                &mut workspace.attention,
                &mut workspace.vocabulary,
                &mut workspace.state_update,
                &mut workspace.materialization,
                &mut workspace.retained,
            ];
            *bounds.into_iter().nth(index).unwrap() = WorkspaceBound::Unknown {
                reason: "selected numerical source has no bound".into(),
            };
        }
        let mut selected = state(g);
        selected.execution_workspace = Some(workspace);
        let error = full_requirement(&selected, g, &pool).unwrap_err();
        let ResidualQuoteError::IncompleteWorkspace(error) = error else {
            panic!("expected a typed numerical gap: {error:?}");
        };
        assert_eq!(error.component(), label);
        assert_eq!(error.geometry(), g);
        assert_eq!(pool.snapshot().unwrap(), before);
    }
}

#[derive(Debug)]
struct ExtentFacts {
    complete_through: u64,
}

impl WorkspaceMechanisms for ExtentFacts {
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
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if operation
            .outputs
            .iter()
            .any(|layout| layout.bytes().unwrap() > self.complete_through * 4)
        {
            return Ok(None);
        }
        Facts::default().operation_bound(operation)
    }

    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Facts::default().host_workspace_bound(operation)
    }
}

struct RunController;

impl TokenFilterController for RunController {
    type Error = std::convert::Infallible;

    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 4,
        })
    }
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwned
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("cold quote called controller")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("cold quote committed token")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("cold quote queried completion")
    }
}

fn candidate_geometry() -> InferenceGeometry {
    InferenceGeometry {
        input_positions: 4,
        max_output_tokens: 1,
        prefill_chunk_positions: 4,
        ..geometry()
    }
}

fn equations(
    pool: &MemoryLedger,
    g: InferenceGeometry,
    complete_through: u64,
) -> (InferenceWorkspaceReport, RegisteredWorkspaceStorage<u32>) {
    let context = WorkspaceContext::new(ExtentFacts { complete_through });
    let root = placed_root(Some(64), &context);
    let registered =
        RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())]).unwrap();
    let old = view(&context, &root, 2);
    let report = quote_inference_workspace(g, |span| {
        context.begin_state_span([&old])?;
        let positions = match span {
            InferenceWorkspaceSpan::Sampling(_) => unreachable!("model-only traversal fixture"),
            InferenceWorkspaceSpan::Prefill(chunk) => chunk.input.end - chunk.input.start,
            InferenceWorkspaceSpan::Decode { .. } => 1,
        };
        // A real metadata equation lacks its numerical mechanism fact only at
        // larger extents. Opening state and borrowed association stay complete.
        let _temporary = WorkspaceTensor::full_f32(0.5, &[positions as i32], &context)?;
        context.report(&[old.clone()])
    })
    .unwrap();
    (report, registered)
}

fn enclosing(pool: &MemoryLedger, g: InferenceGeometry) -> ControllerWorkspaceEstimate {
    let contract = ControllerStorageContract::inspect(&RunController).unwrap();
    ControllerWorkspaceContribution::new(
        g,
        RunController
            .inference_workspace(g.max_output_tokens)
            .unwrap(),
        3,
        &contract,
        pool,
        None,
    )
    .unwrap()
    .compose(outside(g, 0))
    .unwrap()
}

fn candidate(
    pool: &MemoryLedger,
    g: InferenceGeometry,
    complete_through: u64,
    decoder_credit: bool,
) -> Result<IncrementalInferenceQuote, PrefillPlanningError> {
    let (report, registered) = equations(pool, g, complete_through);
    let contribution = enclosing(pool, g);
    let result = if decoder_credit {
        ResidualInferenceQuote::compose_with_controller(
            &report,
            state(g),
            contribution,
            &registered,
        )
        .map(ResidualInferenceQuote::into_incremental)
    } else {
        IncrementalInferenceQuote::compose_controller(&report, state(g), contribution)
    };
    result.map_err(|error| match error {
        ResidualQuoteError::IncompleteWorkspace(incomplete) => incomplete.into(),
        ResidualQuoteError::Estimate(error) => error.into(),
        ResidualQuoteError::Storage(error) => error.into(),
    })
}

fn candidate_plan(
    pool: &MemoryLedger,
    capacity: u64,
    quote: impl FnMut(InferenceGeometry) -> Result<IncrementalInferenceQuote, PrefillPlanningError>,
) -> Result<(WorkingMemoryReservation, IncrementalInferenceQuote), PrefillPlanningError> {
    let g = candidate_geometry();
    let capacity = if pool.pin_registered_storage([(1u32, 64)]).is_ok() {
        let probe = candidate(pool, g, u64::MAX / 4, true).expect("complete metadata probe");
        payload_capacity_with_quote(pool, &probe, &request(g), capacity)
    } else {
        capacity
    };
    plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &capabilities(),
        request(g),
        g,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        quote,
    )
}

#[test]
fn incomplete_larger_equations_retry_for_full_and_registered_decoder_quotes() {
    for decoder_credit in [false, true] {
        let capacity = if decoder_credit { 76 } else { 140 };
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let original = pool.register_host_storage([(1u32, 64)]).unwrap();
        let mut attempted = Vec::new();
        let (reservation, quote) = candidate_plan(&pool, capacity, |g| {
            attempted.push(g.prefill_chunk_positions);
            assert_eq!(
                pool.payload_used_bytes().unwrap(),
                64,
                "cold retries must not reserve a candidate"
            );
            candidate(&pool, g, 2, decoder_credit)
        })
        .unwrap();
        assert_eq!(attempted, [4, 3, 2]);
        assert_eq!(quote.geometry().prefill_chunk_positions, 2);
        assert_eq!(quote.state().requested_state_bytes, 64);
        assert_eq!(
            quote
                .state()
                .execution_workspace
                .as_ref()
                .unwrap()
                .peak_bytes()
                .unwrap(),
            Some(12)
        );
        assert_eq!(
            reservation.admission().incremental_required_bytes,
            Some(capacity - 64)
        );
        assert_eq!(pool.payload_used_bytes().unwrap(), capacity);
        drop((reservation, quote, original));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn numerical_gaps_and_capacity_rejections_keep_descending_candidate_order() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let mut attempted = Vec::new();
    let (reservation, accepted) = candidate_plan(&pool, 76, |g| {
        attempted.push(g.prefill_chunk_positions);
        assert_eq!(pool.payload_used_bytes().unwrap(), 64);
        // Four is unpriced; three is complete but needs sixteen new bytes;
        // two has a complete twelve-byte requirement and fits exactly.
        candidate(&pool, g, 3, true)
    })
    .unwrap();
    assert_eq!(attempted, [4, 3, 2]);
    assert_eq!(accepted.incremental_bytes().unwrap(), 12);
    drop((reservation, accepted, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn all_incomplete_candidates_preserve_final_typed_cause_without_pinning_storage() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let mut attempted = Vec::new();
    let error = candidate_plan(&pool, 128, |g| {
        attempted.push(g.prefill_chunk_positions);
        candidate(&pool, g, 0, true)
    })
    .unwrap_err();
    assert_eq!(attempted, [4, 3, 2, 1]);
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let PrefillPlanningError::IncompleteWorkspace(incomplete) = error else {
        panic!("missing numerical cause")
    };
    assert_eq!(incomplete.geometry().prefill_chunk_positions, 1);
    assert_eq!(incomplete.component(), "residual equation workspace");
    assert!(matches!(
        incomplete
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    drop(original);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        0,
        "retained error must not pin numerical registrations"
    );
    assert!(incomplete.source().is_some());
}

#[test]
fn final_candidate_preserves_budget_or_incomplete_rejection_in_mixed_sequences() {
    for final_incomplete in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let original = pool.register_host_storage([(1u32, 64)]).unwrap();
        let mut attempted = Vec::new();
        let error = candidate_plan(&pool, 64, |g| {
            attempted.push(g.prefill_chunk_positions);
            let incomplete = (g.prefill_chunk_positions == 1) == final_incomplete;
            candidate(&pool, g, if incomplete { 0 } else { 4 }, true)
        })
        .unwrap_err();
        assert_eq!(attempted, [4, 3, 2, 1]);
        if final_incomplete {
            assert!(matches!(
                error,
                PrefillPlanningError::IncompleteWorkspace(_)
            ));
        } else {
            assert!(matches!(
                error,
                PrefillPlanningError::Reservation(capacity_error) if matches!(capacity_numbers(&capacity_error), Some((required, available)) if required == available + 8)));
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), 64);
        drop(original);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn foreign_or_wrong_geometry_incomplete_markers_are_fatal_before_retry() {
    for wrong_geometry in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let foreign = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let original = pool.register_host_storage([(1u32, 64)]).unwrap();
        let other = foreign.register_host_storage([(1u32, 64)]).unwrap();
        let calls = Cell::new(0);
        let error = candidate_plan(&pool, 128, |mut g| {
            calls.set(calls.get() + 1);
            if wrong_geometry {
                g.prefill_chunk_positions -= 1;
            }
            candidate(if wrong_geometry { &pool } else { &foreign }, g, 0, true)
        })
        .unwrap_err();
        assert_eq!(calls.get(), 1);
        assert!(matches!(
            error,
            PrefillPlanningError::Reservation(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(pool.payload_used_bytes().unwrap(), 64);
        assert_eq!(foreign.payload_used_bytes().unwrap(), 64);
        drop((original, other));
    }
}

#[test]
fn arbitrary_unknown_and_typed_fatal_estimates_do_not_retry() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    for numerical_word_in_error in [false, true] {
        let calls = Cell::new(0);
        let error = candidate_plan(&pool, 128, |_| {
            calls.set(calls.get() + 1);
            Err(if numerical_word_in_error {
                CapabilityError::InvalidConfiguration {
                    field: "selected_mechanism",
                    detail: "unknown native owner identity".into(),
                }
                .into()
            } else {
                WorkingMemoryError::UnknownBound.into()
            })
        })
        .unwrap_err();
        assert_eq!(calls.get(), 1);
        if numerical_word_in_error {
            assert!(
                matches!(error, PrefillPlanningError::Estimate(CapabilityError::InvalidConfiguration { field: "selected_mechanism", detail }) if detail == "unknown native owner identity")
            );
        } else {
            assert!(matches!(
                error,
                PrefillPlanningError::Reservation(WorkingMemoryError::UnknownBound)
            ));
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn foreign_complete_quote_is_rejected_before_retryable_application_budget_policy() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let foreign = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let other = foreign.register_host_storage([(1u32, 64)]).unwrap();
    let g = candidate_geometry();
    let mut request = request(g);
    request.memory_limits = crate::working_memory::memory_fixture::host_limits(12);
    let calls = Cell::new(0);
    let error = plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request,
        g,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, 128),
        |g| {
            calls.set(calls.get() + 1);
            // The foreign first candidate needs twenty bytes, above the
            // application allowance. A later valid twelve-byte quote would
            // fit, but domain mismatch must stop before that retry occurs.
            candidate(if calls.get() == 1 { &foreign } else { &pool }, g, 4, true)
        },
    )
    .unwrap_err();
    assert_eq!(calls.get(), 1);
    assert!(matches!(
        error,
        PrefillPlanningError::Reservation(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    assert_eq!(foreign.payload_used_bytes().unwrap(), 64);
    drop((original, other));
}

#[test]
fn structural_geometry_identity_and_overflow_errors_dominate_numerical_gaps() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let g = candidate_geometry();
    let (report, registered) = equations(&pool, g, 0);
    let (_, different_identity) = equations(&pool, g, 0);
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(g), outside(g, 0), &different_identity),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let changed = InferenceGeometry {
        max_output_tokens: 2,
        ..g
    };
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(changed), outside(g, 0), &registered),
        Err(ResidualQuoteError::Estimate(_))
    ));
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(g), outside(changed, 0), &registered),
        Err(ResidualQuoteError::Estimate(_))
    ));
    let mut overflow = outside(g, 0);
    overflow.activations = WorkspaceBound::Unknown {
        reason: "missing numerical activation fact".into(),
    };
    overflow.vocabulary = WorkspaceBound::bounded(u64::MAX, "known vocabulary capacity");
    overflow.retained = WorkspaceBound::bounded(1, "known independent retained byte");
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(g), overflow, &registered),
        Err(ResidualQuoteError::Estimate(
            CapabilityError::ArithmeticOverflow { .. }
        ))
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    drop((registered, different_identity, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn missing_state_span_and_lost_borrowed_association_remain_fatal_unknowns() {
    for missing_state in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let original = pool.register_host_storage([(1u32, 64)]).unwrap();
        let a = WorkspaceContext::new(Facts::default());
        let root = placed_root(Some(64), &a);
        let registered =
            RegisteredWorkspaceStorage::bind(&pool, &a, [(1u32, root.clone())]).unwrap();
        let old = view(&a, &root, 2);
        let b = WorkspaceContext::new(Facts::default());
        let unassociated = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap(),
            &b,
        )
        .unwrap();
        let g = candidate_geometry();
        let report = quote_inference_workspace(g, |span| {
            if matches!(span, InferenceWorkspaceSpan::Decode { .. }) {
                if missing_state {
                    a.begin_span();
                    return a.report(&[old.clone()]);
                }
                b.begin_state_span([&unassociated])?;
                return b.report(&[unassociated.clone()]);
            }
            a.begin_state_span([&old])?;
            a.report(&[old.clone()])
        })
        .unwrap();
        assert!(matches!(
            ResidualInferenceQuote::compose(&report, state(g), outside(g, 0), &registered),
            Err(ResidualQuoteError::Storage(
                WorkingMemoryError::UnknownBound
            ))
        ));
        if missing_state {
            assert!(matches!(
                IncrementalInferenceQuote::compose_controller(
                    &report,
                    state(g),
                    enclosing(&pool, g)
                ),
                Err(ResidualQuoteError::Storage(
                    WorkingMemoryError::UnknownBound
                ))
            ));
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), 64);
        drop((registered, original));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
