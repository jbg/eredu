mod quarantine_pins;
mod registered_sources;

use super::*;
use crate::working_memory::{
    plan_prefill_with_capacity, quote_inference_workspace, InferenceWorkspaceSpan,
};
use eredu_core::{
    AdmissionRejection, CacheStateStrategy, EstimationCompleteness, InputModalities,
    InputTokenCount, LayerSchedule, Observed, OutputDemand, StateMemoryLayout,
};
use eredu_nn::{workspace::*, Error, Tensor};
use std::{cell::Cell, num::NonZeroU8};

mod capacity_handoff;
mod controller;
mod incomplete;

#[derive(Debug, Default)]
struct Facts {
    missing_tensor: bool,
    missing_host: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if self.missing_tensor {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| {
                    if matches!(
                        operation.kind,
                        WorkspaceOperationKind::View(_) | WorkspaceOperationKind::Transpose(_) | WorkspaceOperationKind::Index { .. }
                    ) {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        layout.bytes().map(WorkspaceOutputStorage::Allocate)
                    }
                })
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "portable fixture: exact output allocations and storage-sharing views"
                .into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok((!self.missing_host).then(|| WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture has no disjoint host allocation".into(),
        }))
    }
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 4,
        input_positions: 1,
        max_output_tokens: 2,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn request(g: InferenceGeometry) -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(g.cached_positions + g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: g.batch_size,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    }
}
fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "portable residual fixture".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    }
}
fn state(g: InferenceGeometry) -> RuntimeStateEstimate {
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    eredu_core::estimate_runtime_state(
        &layout,
        request(g).input,
        g.max_output_tokens,
        g.batch_size,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
}
fn outside(g: InferenceGeometry, bytes: u64) -> ExecutionWorkspaceEstimate {
    let bound = |n| WorkspaceBound::bounded(n, "complete enclosing fixture allocation bound");
    ExecutionWorkspaceEstimate {
        geometry: g,
        activations: bound(bytes),
        attention: bound(0),
        vocabulary: bound(0),
        state_update: bound(0),
        materialization: bound(0),
        retained: bound(0),
    }
}
fn view(
    context: &WorkspaceContext,
    root: &WorkspaceExistingStorage,
    elements: i32,
) -> WorkspaceTensor {
    WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[elements], WorkspaceDtype::Float32).unwrap(),
        root,
        context,
    )
    .unwrap()
}
fn used(pool: &WorkingMemoryPool) -> (u64, u64) {
    (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap())
}
fn replacement_report(
    context: &WorkspaceContext,
    root: &WorkspaceExistingStorage,
    g: InferenceGeometry,
) -> InferenceWorkspaceReport {
    let mut current = view(context, root, 2);
    quote_inference_workspace(g, |span| {
        context.begin_state_span([&current])?;
        if let InferenceWorkspaceSpan::Decode { index, .. } = span {
            // A has 64 bytes of capacity despite this initial eight-byte view.
            // B=16 replaces it; the later C=80 overlaps B, not A.
            current =
                WorkspaceTensor::full_f32(0.75, &[if *index == 0 { 4 } else { 20 }], context)?;
        }
        context.report(&[current.clone()])
    })
    .unwrap()
}
fn replacement_quote(
    pool: &WorkingMemoryPool,
    g: InferenceGeometry,
    extra: u64,
) -> ResidualInferenceQuote<u32> {
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let storage = RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())]).unwrap();
    let report = replacement_report(&context, &root, g);
    ResidualInferenceQuote::compose(&report, state(g), outside(g, extra), &storage).unwrap()
}
fn plan(
    pool: &WorkingMemoryPool,
    execution: &InferenceExecutionIdentity,
    quote: &ResidualInferenceQuote<u32>,
    request: AdmissionRequest,
    capacity: u64,
) -> Result<
    (
        Admission,
        WorkingMemoryReservation,
        ResidualInferenceQuote<u32>,
    ),
    PrefillPlanningError,
> {
    plan_prefill_residual_with_capacity(
        execution,
        pool,
        &capabilities(),
        request,
        quote.geometry(),
        capacity,
        |_| Ok(quote.clone()),
    )
}

#[test]
fn replacement_residual_uses_each_span_identity_union_instead_of_scalar_subtraction() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let escaped = original.clone();
    let quote = replacement_quote(&pool, geometry(), 0);
    assert_eq!(quote.state().requested_state_bytes, 80);
    assert_eq!(
        quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap(),
        Some(64)
    );
    assert_eq!(quote.incremental_bytes(), 96);
    let full = quote.state().requested_state_bytes
        + quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .unwrap();
    assert_eq!(full, 144);
    assert!(
        quote.incremental_bytes() > full - 64,
        "the later B+C peak contains no A to subtract"
    );
    assert_eq!(used(&pool), (64, 64));
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(
        plan(&pool, &execution, &quote, request(geometry()), 159),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded {
                required_bytes: 96,
                available_bytes: 95
            }
        ))
    ));
    assert_eq!(used(&pool), (64, 64));
    let (admission, reservation, accepted) =
        plan(&pool, &execution, &quote, request(geometry()), 160).unwrap();
    assert_eq!(admission.state, *quote.state());
    assert_eq!(reservation.bytes(), 96);
    assert!(reservation.requires_funding_scope());
    assert_eq!(used(&pool), (160, 160));
    // A lower report is not permission to enter the ordinary full reservation path.
    assert!(matches!(
        pool.reserve(&execution, &admission),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop((reservation, accepted, quote, original));
    assert_eq!(pool.used_bytes().unwrap(), 64);
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn binding_requires_exact_registered_identity_bijection_and_known_capacities() {
    let pool = WorkingMemoryPool::new(200, 0).unwrap();
    let original = pool.register_storage([(1u32, 64), (2, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let other = WorkspaceContext::new(Facts::default());
    let a = WorkspaceExistingStorage::new(Some(64), &context);
    let b = WorkspaceExistingStorage::new(Some(64), &context);
    let foreign = WorkspaceExistingStorage::new(Some(64), &other);
    let mismatch = WorkspaceExistingStorage::new(Some(63), &context);
    let unknown = WorkspaceExistingStorage::new(None, &context);
    for inventory in [
        vec![(1u32, a.clone()), (1, b.clone())],
        vec![(1, a.clone()), (2, a.clone())],
        vec![(1, a.clone()), (3, b.clone())],
        vec![(1, a.clone()), (2, mismatch)],
        vec![(1, foreign)],
    ] {
        assert!(matches!(
            RegisteredWorkspaceStorage::bind(&pool, &context, inventory),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(used(&pool), (128, 128));
        assert!(context.report(&[]).unwrap().residual.is_none());
    }
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, unknown)]),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let foreign_pool = WorkingMemoryPool::new(200, 0).unwrap();
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&foreign_pool, &context, [(1u32, a.clone())]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u64, a.clone())]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let storage = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        [(1u32, a.clone()), (1, a.clone()), (2, b.clone())],
    )
    .unwrap();
    assert_eq!(storage.borrowed_storage().total_bytes(), 128);
    assert_eq!(storage.borrowed_storage().roots().len(), 2);
    assert!(storage
        .borrowed_storage()
        .same_identity(storage.clone().borrowed_storage()));
    // A second selection is rejected and its temporary registry pin rolls back.
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, a)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop((storage, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn borrowed_aliases_are_free_but_equal_sized_independent_roots_remain_incremental() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let storage =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let alias = view(&context, &root, 2);
    let independent = WorkspaceExistingStorage::new(Some(64), &context);
    let independent = view(&context, &independent, 2);
    let report = quote_inference_workspace(geometry(), |_| {
        context.begin_state_span([&alias, &alias, &independent])?;
        let reshaped = alias.reshape(&[1, 2], &context)?;
        context.report(&[alias.clone(), reshaped, independent.clone()])
    })
    .unwrap();
    let quote = ResidualInferenceQuote::compose(
        &report,
        state(geometry()),
        outside(geometry(), 0),
        &storage,
    )
    .unwrap();
    assert_eq!(quote.state().requested_state_bytes, 128);
    assert_eq!(quote.incremental_bytes(), 64);
    drop((quote, storage, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn residual_composition_and_planning_bind_geometry_selection_and_domain() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let storage =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let report = replacement_report(&context, &root, geometry());
    let other_context = WorkspaceContext::new(Facts::default());
    let other_root = WorkspaceExistingStorage::new(Some(64), &other_context);
    let other_storage =
        RegisteredWorkspaceStorage::bind(&pool, &other_context, [(1u32, other_root)]).unwrap();
    assert!(matches!(
        ResidualInferenceQuote::compose(
            &report,
            state(geometry()),
            outside(geometry(), 0),
            &other_storage
        ),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let changed = InferenceGeometry {
        max_output_tokens: 3,
        ..geometry()
    };
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(geometry()), outside(changed, 0), &storage),
        Err(ResidualQuoteError::Estimate(_))
    ));
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(changed), outside(geometry(), 0), &storage),
        Err(ResidualQuoteError::Estimate(_))
    ));
    let quote = ResidualInferenceQuote::compose(
        &report,
        state(geometry()),
        outside(geometry(), 0),
        &storage,
    )
    .unwrap();
    let foreign_pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(
        plan(&foreign_pool, &execution, &quote, request(geometry()), 1000),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let invoked = Cell::new(0);
    let mut bad_request = request(geometry());
    bad_request.max_output_tokens += 1;
    assert!(matches!(
        plan_prefill_residual_with_capacity(
            &execution,
            &pool,
            &capabilities(),
            bad_request,
            geometry(),
            1000,
            |_| {
                invoked.set(invoked.get() + 1);
                Ok(quote.clone())
            }
        ),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(invoked.get(), 0);
    assert!(matches!(
        plan_prefill_residual_with_capacity(
            &execution,
            &pool,
            &capabilities(),
            request(changed),
            changed,
            1000,
            |_| Ok(quote.clone())
        ),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(used(&pool), (64, 64));
    assert_eq!(used(&foreign_pool), (0, 0));
    drop((quote, storage, other_storage, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn missing_state_spans_tensor_host_or_full_coverage_never_become_residual_permission() {
    for gap in 0..3 {
        let pool = WorkingMemoryPool::new(1000, 0).unwrap();
        let original = pool.register_storage([(1u32, 64)]).unwrap();
        let context = WorkspaceContext::new(Facts {
            missing_tensor: gap == 1,
            missing_host: gap == 2,
        });
        let root = WorkspaceExistingStorage::new(Some(64), &context);
        let storage =
            RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
        let old = view(&context, &root, 2);
        let report = quote_inference_workspace(geometry(), |span| {
            if gap == 0 && matches!(span, InferenceWorkspaceSpan::Decode { index: 0, .. }) {
                context.begin_span();
            } else {
                context.begin_state_span([&old])?;
            }
            let _temporary = old.square(&context)?;
            context.report(&[old.clone()])
        })
        .unwrap();
        assert_eq!(report.completed_spans(), 3);
        assert!(report.first_gap().is_some());
        let result = ResidualInferenceQuote::compose(
            &report,
            state(geometry()),
            outside(geometry(), 0),
            &storage,
        );
        if gap == 0 {
            assert!(matches!(
                result,
                Err(ResidualQuoteError::Storage(
                    WorkingMemoryError::UnknownBound
                ))
            ));
        } else {
            assert!(matches!(
                result,
                Err(ResidualQuoteError::IncompleteWorkspace(_))
            ));
        }
        drop((storage, original));
        assert_eq!(used(&pool), (0, 64));
    }
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let storage =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let report = replacement_report(&context, &root, geometry());
    let mut unpriced = outside(geometry(), 0);
    unpriced.retained = WorkspaceBound::Unknown {
        reason: "controller payload unpriced".into(),
    };
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(geometry()), unpriced, &storage),
        Err(ResidualQuoteError::IncompleteWorkspace(_))
    ));
    assert!(matches!(
        ResidualInferenceQuote::compose(
            &report,
            state(geometry()),
            outside(geometry(), u64::MAX),
            &storage
        ),
        Err(ResidualQuoteError::Storage(WorkingMemoryError::Overflow))
    ));
    let mut partial_state = state(geometry());
    partial_state.persistent_state_completeness = EstimationCompleteness::PersistentStateOnly;
    let partial =
        ResidualInferenceQuote::compose(&report, partial_state, outside(geometry(), 0), &storage)
            .unwrap();
    let mut permissive = request(geometry());
    permissive.require_complete_estimate = false;
    assert!(matches!(
        plan(
            &pool,
            &InferenceExecutionIdentity::default(),
            &partial,
            permissive,
            1000
        ),
        Err(PrefillPlanningError::Admission(
            AdmissionRejection::EstimationUnsupported { .. }
        ))
    ));
    assert_eq!(used(&pool), (64, 64));
    drop((partial, storage, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn different_borrowed_tokens_between_completed_spans_reject_the_whole_trace() {
    let pool = WorkingMemoryPool::new(200, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let a = WorkspaceContext::new(Facts::default());
    let b = WorkspaceContext::new(Facts::default());
    let a_root = WorkspaceExistingStorage::new(Some(64), &a);
    let b_root = WorkspaceExistingStorage::new(Some(64), &b);
    let a_storage = RegisteredWorkspaceStorage::bind(&pool, &a, [(1u32, a_root.clone())]).unwrap();
    let b_storage = RegisteredWorkspaceStorage::bind(&pool, &b, [(1u32, b_root.clone())]).unwrap();
    let a_view = view(&a, &a_root, 2);
    let b_view = view(&b, &b_root, 2);
    let result = quote_inference_workspace(geometry(), |span| {
        let (context, root) = if matches!(span, InferenceWorkspaceSpan::Prefill(_)) {
            (&a, &a_view)
        } else {
            (&b, &b_view)
        };
        context.begin_state_span([root])?;
        context.report(&[root.clone()])
    });
    assert!(matches!(
        result,
        Err(crate::working_memory::InferenceWorkspaceError::Geometry(_))
    ));
    drop((a_storage, b_storage, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn application_budget_safety_and_domain_capacity_use_complete_incremental_cost() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 8);
    assert_eq!(quote.incremental_bytes(), 104);
    let execution = InferenceExecutionIdentity::default();
    let mut request = request(geometry());
    request.safety_reserve_bytes = 7;
    request.application_memory_budget_bytes = Some(110);
    assert!(matches!(
        plan(&pool, &execution, &quote, request, 1000),
        Err(PrefillPlanningError::Admission(
            AdmissionRejection::MemoryBudgetExceeded {
                required_bytes: 111,
                budget_bytes: 110
            }
        ))
    ));
    request.application_memory_budget_bytes = Some(111);
    assert!(matches!(
        plan(&pool, &execution, &quote, request, 174),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded {
                required_bytes: 111,
                available_bytes: 110
            }
        ))
    ));
    assert_eq!(used(&pool), (64, 64));
    let (admission, reservation, accepted) = plan(&pool, &execution, &quote, request, 175).unwrap();
    assert_eq!(admission.incremental_required_bytes, 111);
    assert_eq!(admission.state, *quote.state());
    assert_eq!(pool.used_bytes().unwrap(), 175);
    drop((reservation, accepted, quote, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

fn chunk_quote(pool: &WorkingMemoryPool, g: InferenceGeometry) -> ResidualInferenceQuote<u32> {
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let storage = RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())]).unwrap();
    let old = view(&context, &root, 2);
    let report = quote_inference_workspace(g, |span| {
        context.begin_state_span([&old])?;
        let count = match span {
            InferenceWorkspaceSpan::Prefill(chunk) => chunk.input.end - chunk.input.start,
            InferenceWorkspaceSpan::Decode { .. } => 1,
        };
        let _temporary = WorkspaceTensor::full_f32(0.5, &[count as i32 * 4], &context)?;
        context.report(&[old.clone()])
    })
    .unwrap();
    ResidualInferenceQuote::compose(&report, state(g), outside(g, 0), &storage).unwrap()
}

#[test]
fn residual_chunk_retries_share_planner_policy_without_weakening_full_reservations() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let g = InferenceGeometry {
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 3,
        ..geometry()
    };
    let execution = InferenceExecutionIdentity::default();
    let mut full_candidates = Vec::new();
    assert!(matches!(
        plan_prefill_with_capacity(
            &execution,
            &pool,
            &capabilities(),
            request(g),
            g,
            96,
            |candidate| {
                full_candidates.push(candidate.prefill_chunk_positions);
                Ok(chunk_quote(&pool, candidate).state().clone())
            }
        ),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded {
                required_bytes: 80,
                available_bytes: 32
            }
        ))
    ));
    assert_eq!(full_candidates, [3, 2, 1]);
    assert_eq!(used(&pool), (64, 64));
    let mut residual_candidates = Vec::new();
    let (admission, reservation, accepted) = plan_prefill_residual_with_capacity(
        &execution,
        &pool,
        &capabilities(),
        request(g),
        g,
        96,
        |candidate| {
            residual_candidates.push(candidate.prefill_chunk_positions);
            Ok(chunk_quote(&pool, candidate))
        },
    )
    .unwrap();
    assert_eq!(residual_candidates, [3, 2]);
    assert_eq!(reservation.geometry().prefill_chunk_positions, 2);
    assert_eq!(admission.incremental_required_bytes, 32);
    assert_eq!(accepted.state().requested_state_bytes, 64);
    assert_eq!(used(&pool), (96, 96));
    drop((reservation, accepted, original));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn residual_reservation_retains_borrowed_charge_after_quote_and_original_owner_drop() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0);
    let execution = InferenceExecutionIdentity::default();
    let (_, reservation, accepted) =
        plan(&pool, &execution, &quote, request(geometry()), 160).unwrap();
    let alias = reservation.clone();
    drop((accepted, quote, original, reservation));
    assert_eq!(used(&pool), (160, 160));
    assert!(alias.requires_funding_scope());
    alias.validate(&execution, geometry()).unwrap();
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 1000);
}

#[test]
fn conversion_moves_pins_to_live_run_and_scopes_while_metadata_remains_cold_evidence() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0);
    let execution = InferenceExecutionIdentity::default();
    let (_, reservation, accepted) =
        plan(&pool, &execution, &quote, request(geometry()), 160).unwrap();
    let (metadata, run) = reservation.into_funding().unwrap();
    let scope = run.scope().unwrap();
    let output = scope.adopt_storage_individually([(2u32, 80)]).unwrap();
    drop((accepted, quote, original));
    run.close().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 160);
    scope.certify().unwrap();
    // The old borrowed A and unused workspace expired. Only C remains charged.
    assert_eq!(pool.used_bytes().unwrap(), 80);
    assert!(metadata.requires_funding_scope());
    metadata.validate(&execution, geometry()).unwrap();
    assert_eq!(metadata.bytes(), 96);
    drop(output);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(metadata);
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn uncertified_residual_scope_keeps_borrowed_charge_and_remaining_envelope() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0);
    let execution = InferenceExecutionIdentity::default();
    let (_, reservation, accepted) =
        plan(&pool, &execution, &quote, request(geometry()), 160).unwrap();
    let (metadata, run) = reservation.into_funding().unwrap();
    let scope = run.scope().unwrap();
    drop((accepted, quote, original, metadata, run));
    drop(scope);
    assert_eq!(used(&pool), (160, 160));
    assert_eq!(pool.effective_capacity().unwrap(), 160);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

mod span_workspace;

#[test]
fn finite_binding_consumes_counted_rows_and_keeps_exact_existing_only_pins() {
    if !crate::working_memory::qualified_storage::qualified() {
        return;
    }
    let pool = WorkingMemoryPool::new(200, 0).unwrap();
    let source = pool.register_storage([(1u32, 64), (2, 32)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let a = WorkspaceExistingStorage::new(Some(64), &context);
    let b = WorkspaceExistingStorage::new(Some(32), &context);
    let initial = used(&pool);
    assert!(matches!(
        RegisteredWorkspaceStorageLayout::<u32>::new(1)
            .unwrap()
            .construct(&pool, &context, [(1, a.clone()), (2, b.clone())]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(used(&pool), initial);
    assert!(context.report(&[]).unwrap().residual.is_none());
    let layout = RegisteredWorkspaceStorageLayout::<u32>::new(3).unwrap();
    assert_eq!(layout.source_slots(), 3);
    assert!(layout.requested_bytes() > 0);
    let registered = layout
        .construct(&pool, &context, [(2, b), (1, a.clone()), (1, a.clone())])
        .unwrap();
    assert_eq!(registered.borrowed_storage().roots().len(), 2);
    assert!(registered.borrowed_storage().roots()[0].same_storage(&a));
    assert_eq!(registered.borrowed_storage().total_bytes(), 96);
    assert_eq!(used(&pool), initial);
    let shared = registered.clone();
    drop((registered, source));
    assert_eq!(pool.used_bytes().unwrap(), 96);
    drop(shared);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(matches!(
        RegisteredWorkspaceStorageLayout::<u32>::new(usize::MAX),
        Err(WorkingMemoryError::Overflow)
    ));
}

mod reservation_metadata;
