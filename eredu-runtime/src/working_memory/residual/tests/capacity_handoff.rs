use super::*;
use crate::working_memory::{
    ControllerStorageContract, ControllerWorkspaceContribution, WorkingMemoryCapacityHandoff,
    WorkingMemoryFundingRun,
};
use eredu_core::{
    TextControllerStorage, TextControllerWorkspace, TokenFilter, TokenFilterController,
};

struct Predecessor {
    metadata: WorkingMemoryReservation,
    run: Option<WorkingMemoryFundingRun>,
    handoff: WorkingMemoryCapacityHandoff,
    storage: WorkingMemoryStorage<u32>,
}
impl Predecessor {
    fn close(&mut self) {
        self.run.take().unwrap().close().unwrap();
    }
}
fn plain_reservation(
    pool: &WorkingMemoryPool,
    execution: &InferenceExecutionIdentity,
    bytes: u64,
    capacity: u64,
) -> WorkingMemoryReservation {
    let g = geometry();
    let state = state(g)
        .with_execution_workspace(outside(g, bytes))
        .unwrap();
    pool.reserve_with_capacity(
        execution,
        &Admission {
            requested_positions: g.cached_positions + g.input_positions + g.max_output_tokens,
            state,
            incremental_required_bytes: bytes,
            available_memory_bytes: None,
        },
        capacity,
    )
    .unwrap()
}
fn predecessor(pool: &WorkingMemoryPool, execution: &InferenceExecutionIdentity) -> Predecessor {
    let (metadata, mut run) = plain_reservation(pool, execution, 64, 72)
        .into_funding()
        .unwrap();
    let handoff = run.take_capacity_handoff().unwrap();
    let scope = run.scope().unwrap();
    let storage = scope
        .adopt_storage_individually([(1u32, 64)])
        .unwrap()
        .remove(&1)
        .unwrap();
    scope.certify().unwrap();
    Predecessor {
        metadata,
        run: Some(run),
        handoff,
        storage,
    }
}
fn balances(pool: &WorkingMemoryPool) -> (u64, u64, u64) {
    (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        pool.effective_capacity().unwrap(),
    )
}
fn g() -> InferenceGeometry {
    InferenceGeometry {
        input_positions: 4,
        max_output_tokens: 1,
        prefill_chunk_positions: 4,
        ..geometry()
    }
}

#[derive(Debug)]
struct ExtentFacts(u64);
impl WorkspaceMechanisms for ExtentFacts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if operation
            .outputs
            .iter()
            .any(|layout| layout.bytes().unwrap() > self.0 * 4)
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
struct Controller;
impl TokenFilterController for Controller {
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
        panic!("cold quote sampled")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("cold quote committed")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("cold quote advanced controller")
    }
}
fn candidate(
    pool: &WorkingMemoryPool,
    g: InferenceGeometry,
    complete_through: u64,
) -> Result<IncrementalInferenceQuote, PrefillPlanningError> {
    candidate_with_extra(pool, g, complete_through, 0)
}
fn candidate_with_extra(
    pool: &WorkingMemoryPool,
    g: InferenceGeometry,
    complete_through: u64,
    extra_positions: u64,
) -> Result<IncrementalInferenceQuote, PrefillPlanningError> {
    let context = WorkspaceContext::new(ExtentFacts(complete_through));
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let registered = RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())])?;
    let old = view(&context, &root, 2);
    let equations = quote_inference_workspace(g, |span| {
        context.begin_state_span([&old])?;
        let positions = match span {
            InferenceWorkspaceSpan::Sampling(_) => unreachable!("model-only traversal fixture"),
            InferenceWorkspaceSpan::Prefill(chunk) => chunk.input.end - chunk.input.start,
            InferenceWorkspaceSpan::Decode { .. } => 1,
        };
        let _temporary = WorkspaceTensor::full_f32(0.5, &[positions as i32], &context)?;
        let _additional = if extra_positions != 0 {
            Some(WorkspaceTensor::full_f32(
                0.75,
                &[extra_positions as i32],
                &context,
            )?)
        } else {
            None
        };
        context.report(&[old.clone()])
    })
    .unwrap();
    let storage = ControllerStorageContract::inspect(&Controller).unwrap();
    let contribution = ControllerWorkspaceContribution::new(
        g,
        Controller.inference_workspace(1).unwrap(),
        3,
        &storage,
        pool,
        None,
    )
    .unwrap()
    .compose(outside(g, 0))
    .unwrap();
    ResidualInferenceQuote::compose_with_controller(&equations, state(g), contribution, &registered)
        .map(ResidualInferenceQuote::into_incremental)
        .map_err(|error| match error {
            ResidualQuoteError::IncompleteWorkspace(error) => error.into(),
            ResidualQuoteError::Estimate(error) => error.into(),
            ResidualQuoteError::Storage(error) => error.into(),
        })
}
fn plan(
    pool: &WorkingMemoryPool,
    execution: &InferenceExecutionIdentity,
    capacity: u64,
    handoffs: &[WorkingMemoryCapacityHandoff],
    quote: impl FnMut(InferenceGeometry) -> Result<IncrementalInferenceQuote, PrefillPlanningError>,
) -> Result<
    (
        WorkingMemoryReservation,
        IncrementalInferenceQuote,
    ),
    PrefillPlanningError,
> {
    plan_prefill_incremental_with_capacity_handoff(
        execution,
        pool,
        &capabilities(),
        request(g()),
        g(),
        capacity,
        handoffs,
        quote,
    )
}

#[test]
fn incomplete_then_overbudget_candidates_commit_only_the_fitting_handoff() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let mut old = predecessor(&pool, &execution);
    old.close();
    let mut attempts = Vec::new();
    let (reservation, proof) = plan(
        &pool,
        &execution,
        76,
        std::slice::from_ref(&old.handoff),
        |g| {
            attempts.push(g.prefill_chunk_positions);
            assert_eq!(
                balances(&pool),
                (64, 64, 72),
                "rejected candidates cannot adopt the old ceiling"
            );
            candidate(&pool, g, 3)
        },
    )
    .unwrap();
    assert_eq!(attempts, [4, 3, 2]);
    assert_eq!(reservation.admission().incremental_required_bytes, 12);
    assert_eq!(proof.incremental_bytes(), 12);
    assert!(proof.controller_contract().is_some());
    assert_eq!(balances(&pool), (76, 76, 76));
    assert_eq!(old.metadata.bytes(), 64);
    // A later preparation failure returns B's bytes but cannot revoke the
    // already committed policy change while A's physical allocation survives.
    drop((reservation, proof));
    assert_eq!(balances(&pool), (64, 76, 76));
    drop(old.metadata);
    assert_eq!(pool.effective_capacity().unwrap(), 76);
    drop(old.storage);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 512);
    assert!(old.handoff.is_retired().unwrap());
}

#[test]
fn legacy_planner_keeps_the_predecessor_ceiling_and_chooses_a_smaller_chunk() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let mut old = predecessor(&pool, &execution);
    old.close();
    let mut attempts = Vec::new();
    let (reservation, proof) = plan_prefill_incremental_with_capacity(
        &execution,
        &pool,
        &capabilities(),
        request(g()),
        g(),
        76,
        |g| {
            attempts.push(g.prefill_chunk_positions);
            candidate(&pool, g, 4)
        },
    )
    .unwrap();
    assert_eq!(attempts, [4, 3, 2, 1]);
    assert_eq!(proof.incremental_bytes(), 8);
    assert_eq!(balances(&pool), (72, 72, 72));
    drop((reservation, proof, old));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn all_incomplete_or_overbudget_candidates_preserve_usage_peak_and_old_ceiling() {
    for incomplete in [false, true] {
        let pool = WorkingMemoryPool::new(512, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let mut old = predecessor(&pool, &execution);
        old.close();
        let mut attempts = Vec::new();
        let error = plan(
            &pool,
            &execution,
            76,
            std::slice::from_ref(&old.handoff),
            |g| {
                attempts.push(g.prefill_chunk_positions);
                assert_eq!(balances(&pool), (64, 64, 72));
                candidate_with_extra(&pool, g, if incomplete { 0 } else { 4 }, 2)
            },
        )
        .unwrap_err();
        assert_eq!(attempts, [4, 3, 2, 1]);
        if incomplete {
            assert!(matches!(
                error,
                PrefillPlanningError::IncompleteWorkspace(_)
            ));
        } else {
            assert!(matches!(
                error,
                PrefillPlanningError::Reservation(WorkingMemoryError::BudgetExceeded {
                    required_bytes: 16,
                    available_bytes: 12
                })
            ));
        }
        assert_eq!(balances(&pool), (64, 64, 72));
        drop(old);
        assert_eq!(
            pool.used_bytes().unwrap(),
            0,
            "rejected candidates must release their source pins"
        );
    }
}

#[test]
fn wrong_pool_or_execution_handoffs_reject_before_retryable_quote_callbacks() {
    for wrong_pool in [false, true] {
        let pool = WorkingMemoryPool::new(512, 0).unwrap();
        let other = WorkingMemoryPool::new(512, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let foreign_execution = InferenceExecutionIdentity::default();
        let mut old = predecessor(if wrong_pool { &other } else { &pool }, &execution);
        old.close();
        let calls = Cell::new(0);
        let error = plan(
            &pool,
            if wrong_pool {
                &execution
            } else {
                &foreign_execution
            },
            76,
            std::slice::from_ref(&old.handoff),
            |_| {
                calls.set(calls.get() + 1);
                Err(WorkingMemoryError::UnknownBound.into())
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PrefillPlanningError::Reservation(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(calls.get(), 0);
        assert_eq!(
            balances(if wrong_pool { &other } else { &pool }),
            (64, 64, 72)
        );
    }
}

#[test]
fn foreign_or_wrong_geometry_proofs_never_change_eligible_predecessor_ceilings() {
    for wrong_geometry in [false, true] {
        let pool = WorkingMemoryPool::new(512, 0).unwrap();
        let other = WorkingMemoryPool::new(512, 0).unwrap();
        let foreign_root = other.register_storage([(1u32, 64)]).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let mut old = predecessor(&pool, &execution);
        old.close();
        let calls = Cell::new(0);
        let error = plan(
            &pool,
            &execution,
            76,
            std::slice::from_ref(&old.handoff),
            |mut g| {
                calls.set(calls.get() + 1);
                if wrong_geometry {
                    g.prefill_chunk_positions -= 1;
                }
                candidate(if wrong_geometry { &pool } else { &other }, g, 4)
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PrefillPlanningError::Reservation(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(calls.get(), 1);
        assert_eq!(balances(&pool), (64, 64, 72));
        drop((foreign_root, old));
    }
}

#[test]
fn open_run_unsettled_scope_and_quarantine_are_nonretryable_handoff_failures() {
    for mode in 0..3 {
        let pool = WorkingMemoryPool::new(512, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let mut old = predecessor(&pool, &execution);
        let mut pending = (mode != 0).then(|| old.run.as_ref().unwrap().scope().unwrap());
        if mode != 0 {
            old.close();
        }
        if mode == 2 {
            drop(pending.take());
        }
        let calls = Cell::new(0);
        let error = plan(
            &pool,
            &execution,
            76,
            std::slice::from_ref(&old.handoff),
            |g| {
                calls.set(calls.get() + 1);
                candidate(&pool, g, 4)
            },
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                PrefillPlanningError::Reservation(WorkingMemoryError::ExecutionFenced)
            ),
            "{error:?}"
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(balances(&pool), (64, 64, 72));
        if let Some(scope) = pending {
            scope.certify().unwrap();
        }
        if old.run.is_some() {
            old.close();
        }
        drop(old);
        if mode != 2 {
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn unrelated_same_limit_account_and_fixed_pool_capacity_are_not_relaxed() {
    for fixed_pool in [false, true] {
        let pool = WorkingMemoryPool::new(if fixed_pool { 70 } else { 512 }, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let mut old = predecessor(&pool, &execution);
        old.close();
        let unrelated = (!fixed_pool).then(|| plain_reservation(&pool, &execution, 0, 72));
        let mut attempts = Vec::new();
        let result = plan(
            &pool,
            &execution,
            76,
            std::slice::from_ref(&old.handoff),
            |g| {
                attempts.push(g.prefill_chunk_positions);
                candidate(&pool, g, 4)
            },
        );
        assert_eq!(attempts, [4, 3, 2, 1]);
        if fixed_pool {
            assert!(matches!(
                result,
                Err(PrefillPlanningError::Reservation(
                    WorkingMemoryError::BudgetExceeded {
                        required_bytes: 8,
                        available_bytes: 6
                    }
                ))
            ));
            assert_eq!(balances(&pool), (64, 64, 70));
        } else {
            let (reservation, proof) = result.unwrap();
            assert_eq!(proof.geometry().prefill_chunk_positions, 1);
            assert_eq!(
                balances(&pool),
                (72, 72, 72),
                "unrelated identical ceiling still constrains B"
            );
            drop((reservation, proof));
            assert_eq!(pool.effective_capacity().unwrap(), 72);
            drop(unrelated);
            assert_eq!(
                pool.effective_capacity().unwrap(),
                76,
                "only the authorized account adopted B's ceiling"
            );
        }
        drop(old);
    }
}

#[test]
fn fatal_mechanism_error_is_preserved_after_earlier_incomplete_candidate() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let mut old = predecessor(&pool, &execution);
    old.close();
    let mut attempts = Vec::new();
    let error = plan(
        &pool,
        &execution,
        76,
        std::slice::from_ref(&old.handoff),
        |g| {
            attempts.push(g.prefill_chunk_positions);
            if g.prefill_chunk_positions == 4 {
                return candidate(&pool, g, 0);
            }
            Err(CapabilityError::InvalidConfiguration {
                field: "selected_native_mechanism",
                detail: "exact provider failure".into(),
            }
            .into())
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, PrefillPlanningError::Estimate(CapabilityError::InvalidConfiguration { field: "selected_native_mechanism", detail }) if detail == "exact provider failure")
    );
    assert_eq!(attempts, [4, 3]);
    assert_eq!(balances(&pool), (64, 64, 72));
}
