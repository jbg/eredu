use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    CacheStateStrategy, EstimationCompleteness, InputModalities, LayerSchedule, ModelCapabilities,
    Observed, StateMemoryLayout, TokenFilter,
};
use eredu_nn::Tensor;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceHostBound, WorkspaceMechanisms, WorkspaceOperation,
    WorkspaceOperationBound, WorkspaceOutputStorage, WorkspaceTensor,
};
use eredu_runtime::working_memory::{
    InferenceWorkspaceSpan, ResidualQuoteError, quote_inference_workspace,
};
use std::{num::NonZeroU8, sync::Arc};

#[derive(Debug)]
struct Facts {
    maximum_rows: i32,
}

impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        if operation
            .outputs
            .iter()
            .any(|output| output.shape()[0] > self.maximum_rows)
        {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 7,
            assumptions:
                "fixture provider allocates packed outputs and seven scratch bytes per operation"
                    .into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 3,
            assumptions: "fixture provider needs three disjoint host staging bytes per operation"
                .into(),
        }))
    }
}

struct Controller {
    filter: TokenFilter,
    history: Vec<u8>,
}

impl Controller {
    fn new(history_bytes: usize) -> Self {
        Self {
            filter: TokenFilter::All,
            history: vec![7; history_bytes],
        }
    }
}

impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }

    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: (&self.filter).into(),
            additional_host_bytes: self.history.capacity() as u64,
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("cold candidate planning must not invoke decisions")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("cold candidate planning must not commit tokens")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("cold candidate planning must not query mutable controller state")
    }
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 4,
        max_output_tokens: 2,
        prefill_chunk_positions: 4,
        output: OutputDemand::LastPosition,
    }
}

fn request(geometry: InferenceGeometry) -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(geometry.input_positions),
        max_output_tokens: geometry.max_output_tokens,
        batch_size: geometry.batch_size,
        additional_headroom: crate::memory_fixture::headroom(11),
        memory_limits: Default::default(),
    }
}

fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "cold candidate fixture".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    }
}

fn candidate(
    pool: &MemoryLedger,
    geometry: InferenceGeometry,
    controller: &Controller,
    maximum_rows: i32,
    output_width: usize,
) -> TextWorkspaceCandidate {
    let context = WorkspaceContext::new(Facts { maximum_rows });
    let equations = quote_inference_workspace(geometry, |span| {
        // This fixture has no decoder state. Every real invocation explicitly
        // enters that protocol; unknown operation facts remain in the trace.
        context.begin_state_span(std::iter::empty::<&WorkspaceTensor>())?;
        let rows = match span {
            InferenceWorkspaceSpan::Sampling(_) => {
                panic!("model scheduler emitted a sampling phase")
            }
            InferenceWorkspaceSpan::Prefill(chunk) => chunk.input.end - chunk.input.start,
            InferenceWorkspaceSpan::Decode { .. } => 1,
        };
        let value = WorkspaceTensor::full_f32(0.75, &[rows as i32, output_width as i32], &context)?;
        let output = value.square(&context)?;
        let report = context.report(&[]);
        drop((output, value));
        report
    })
    .unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let request = request(geometry);
    let state = eredu_core::estimate_runtime_state(
        &layout,
        request.input,
        request.max_output_tokens,
        request.batch_size,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    // All numerical operator storage is in the authentic equation trace.
    // There is no separate attention, sampler, cache, or materializer in this
    // fixture. These inapplicable terms never replace a missing trace bound.
    let absent = || {
        WorkspaceBound::bounded(
            0,
            "no separate fixture allocation outside the traced operators",
        )
    };
    let outside = crate::memory_fixture::workspace(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry,
        activations: absent(),
        attention: absent(),
        vocabulary: absent(),
        state_update: absent(),
        materialization: absent(),
        retained: absent(),
    });
    let storage = ControllerStorageContract::inspect(controller).unwrap();
    let contribution = ControllerWorkspaceContribution::new(
        geometry,
        controller
            .inference_workspace(geometry.max_output_tokens)
            .unwrap(),
        output_width,
        &storage,
        pool,
        None,
    )
    .unwrap()
    .compose(outside)
    .unwrap();
    let quote = match IncrementalInferenceQuote::compose_controller(&equations, state, contribution)
    {
        Ok(quote) => Ok(quote),
        Err(ResidualQuoteError::IncompleteWorkspace(incomplete)) => Err(incomplete),
        Err(error) => panic!("fixture must remain structurally valid: {error}"),
    };
    TextWorkspaceCandidate {
        paged_sources: None,
        native_recipe: None,
        prepared_source: None,
        execution_metadata: None,
        ordinary_publication: None,
        planning_metadata: None,
        quote,
        output_width,
    }
}

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

struct Cold {
    paths: paths::Counts,
    inputs: usize,
    resets: usize,
}

impl Cold {
    fn new() -> Self {
        Self {
            paths: paths::snapshot(),
            inputs: paths::session_input_creation_attempts(),
            resets: paths::session_reset_attempts(),
        }
    }
    fn unchanged(&self) {
        assert_eq!(paths::snapshot(), self.paths);
        assert_eq!(paths::session_input_creation_attempts(), self.inputs);
        assert_eq!(paths::session_reset_attempts(), self.resets);
    }
}

#[test]
fn first_incomplete_candidate_retries_until_a_real_smaller_quote_is_reserved() {
    let pool = crate::memory_fixture::ledger(4096, 0).unwrap();
    let original = pool.register_host_storage([(1_u32, 64)]).unwrap();
    let controller = Controller::new(16);
    let cold = Cold::new();
    let mut accepted_geometry = geometry();
    accepted_geometry.prefill_chunk_positions = 2;
    let expected = candidate(&pool, accepted_geometry, &controller, 2, 8)
        .quote
        .unwrap();
    let expected_bytes = expected
        .incremental_bytes()
        .expect("finite fixture diagnostic")
        + 11;
    let mut attempts = Vec::new();
    let (reservation, contract) = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(64 + expected_bytes),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            assert_eq!(pool.fixture_host_charge().unwrap(), 64);
            assert_eq!(pool.fixture_host_peak().unwrap(), 64);
            Ok(candidate(&pool, geometry, &controller, 2, 8))
        },
    )
    .unwrap();
    assert_eq!(attempts, [4, 3, 2]);
    assert_eq!(reservation.geometry(), accepted_geometry);
    assert_eq!(
        reservation
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        expected_bytes
    );
    assert_eq!(&reservation.admission().state, expected.state());
    assert_eq!(Some(&contract), expected.controller_contract());
    assert!(reservation.requires_funding_scope());
    assert_eq!(pool.fixture_host_charge().unwrap(), 64 + expected_bytes);
    cold.unchanged();
    drop(reservation);
    assert_eq!(pool.fixture_host_charge().unwrap(), 64);
    drop(original);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn all_incomplete_candidates_preserve_typed_last_gap_without_mutating_pool() {
    let pool = crate::memory_fixture::ledger(4096, 0).unwrap();
    let _original = pool.register_host_storage([(1_u32, 64)]).unwrap();
    let controller = Controller::new(16);
    let cold = Cold::new();
    let mut attempts = Vec::new();
    let error = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(4096),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            Ok(candidate(&pool, geometry, &controller, 0, 8))
        },
    )
    .err()
    .expect("missing provider facts cannot produce a reservation");
    assert_eq!(attempts, [4, 3, 2, 1]);
    let incomplete = cause::<IncompleteWorkspace>(&error).expect("typed final numerical gap");
    assert_eq!(incomplete.geometry().prefill_chunk_positions, 1);
    assert!(!incomplete.component().is_empty());
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert!(matches!(
        cause::<PrefillPlanningError>(&error),
        Some(PrefillPlanningError::IncompleteWorkspace(_))
    ));
    assert_eq!(
        (
            pool.fixture_host_charge().unwrap(),
            pool.fixture_host_peak().unwrap()
        ),
        (64, 64)
    );
    cold.unchanged();
}

#[test]
fn diagnostic_output_width_is_bound_even_when_both_candidates_are_incomplete() {
    let pool = crate::memory_fixture::ledger(4096, 0).unwrap();
    let controller = Controller::new(16);
    let cold = Cold::new();
    let mut attempts = Vec::new();
    let error = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(4096),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            let width = if geometry.prefill_chunk_positions == 4 {
                8
            } else {
                9
            };
            Ok(candidate(&pool, geometry, &controller, 0, width))
        },
    )
    .err()
    .expect("a smaller chunk cannot change the bound vocabulary");
    assert_eq!(attempts, [4, 3]);
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        (
            pool.fixture_host_charge().unwrap(),
            pool.fixture_host_peak().unwrap()
        ),
        (0, 0)
    );
    cold.unchanged();
}

#[test]
fn complete_candidate_with_changed_controller_contract_rejects_before_reservation() {
    let pool = crate::memory_fixture::ledger(4096, 0).unwrap();
    let controller = Controller::new(16);
    let changed = Controller::new(24);
    let cold = Cold::new();
    let mut attempts = Vec::new();
    let error = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(4096),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            let owner = if geometry.prefill_chunk_positions == 4 {
                &controller
            } else {
                &changed
            };
            Ok(candidate(&pool, geometry, owner, 3, 8))
        },
    )
    .err()
    .expect("first diagnostic contract remains binding");
    assert_eq!(attempts, [4, 3]);
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        (
            pool.fixture_host_charge().unwrap(),
            pool.fixture_host_peak().unwrap()
        ),
        (0, 0)
    );
    cold.unchanged();
}

#[derive(Debug, thiserror::Error)]
#[error("exact native candidate failure")]
struct NativeCause(Arc<()>);

#[test]
fn fatal_native_error_after_incomplete_candidate_keeps_original_source_and_stops() {
    let pool = crate::memory_fixture::ledger(4096, 0).unwrap();
    let controller = Controller::new(16);
    let identity = Arc::new(());
    // Exception construction is host metadata; no native array is allocated.
    let native = Box::new(safemlx::error::Exception::from_source(NativeCause(
        Arc::clone(&identity),
    )));
    let address = &*native as *const safemlx::error::Exception;
    let mut failure = Some(Error::Other(native));
    let cold = Cold::new();
    let mut attempts = Vec::new();
    let error = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(4096),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            if geometry.prefill_chunk_positions == 3 {
                return Err(failure.take().unwrap());
            }
            Ok(candidate(&pool, geometry, &controller, 0, 8))
        },
    )
    .err()
    .expect("fatal provider error must not retry smaller chunks");
    assert_eq!(attempts, [4, 3]);
    assert!(std::ptr::eq(
        cause::<safemlx::error::Exception>(&error).unwrap(),
        address
    ));
    assert!(Arc::ptr_eq(
        &cause::<NativeCause>(&error).unwrap().0,
        &identity
    ));
    assert!(cause::<PrefillPlanningError>(&error).is_none());
    assert_eq!(
        (
            pool.fixture_host_charge().unwrap(),
            pool.fixture_host_peak().unwrap()
        ),
        (0, 0)
    );
    cold.unchanged();
}

#[test]
fn tracking_capacity_retries_smaller_quotes_and_preserves_final_typed_refusal() {
    let pool = crate::memory_fixture::ledger(4096, 0).unwrap();
    let controller = Controller::new(16);
    let cold = Cold::new();
    let mut attempts = Vec::new();
    let (reservation, _) = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(4096),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
            if geometry.prefill_chunk_positions > 2 {
                return Err(Error::PrefillControl(
                    WorkingMemoryError::SubmissionTrackingCapacity {
                        required_bytes: geometry.prefill_chunk_positions * 1024,
                        configured_bytes: 2048,
                    },
                ));
            }
            Ok(candidate(&pool, geometry, &controller, 2, 8))
        },
    )
    .unwrap();
    assert_eq!(attempts, [4, 3, 2]);
    assert_eq!(reservation.geometry().prefill_chunk_positions, 2);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(reservation);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);

    // A decode/other constructor can impose a floor that no prefill chunk
    // removes. The last real refusal keeps its exact fields, without replacing
    // it by a generic native failure or accepting an unpriced candidate.
    attempts.clear();
    let error = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(4096),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            Err(Error::PrefillControl(
                WorkingMemoryError::SubmissionTrackingCapacity {
                    required_bytes: 2048 + geometry.prefill_chunk_positions,
                    configured_bytes: 1024,
                },
            ))
        },
    )
    .err()
    .expect("no candidate can satisfy the fixed constructor floor");
    assert_eq!(attempts, [4, 3, 2, 1]);
    assert!(matches!(
        error,
        Error::PrefillControl(WorkingMemoryError::SubmissionTrackingCapacity {
            required_bytes: 2049,
            configured_bytes: 1024,
        })
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);

    // The new quote-stage exception must not make arbitrary accounting/native
    // failures retryable; such a retry could hide a source/identity defect.
    attempts.clear();
    let error = plan_candidates(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry()),
        geometry(),
        crate::memory_fixture::resolved_limits(4096),
        controller.inference_workspace(2).unwrap(),
        |geometry| {
            attempts.push(geometry.prefill_chunk_positions);
            Err(memory(WorkingMemoryError::IdentityMismatch))
        },
    )
    .err()
    .expect("identity refusal remains terminal");
    assert_eq!(attempts, [4]);
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    cold.unchanged();
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
