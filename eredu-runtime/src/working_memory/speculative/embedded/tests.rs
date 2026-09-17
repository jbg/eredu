//! The actual neutral trace supplies nonzero obligations; no native fit claim.
use super::*;
use crate::prefill::PrefillControlPlan;
use crate::speculative::embedded_occurrence::EmbeddedPredictionShape;
use crate::*;
use eredu_core::{
    OutputDemand,
    generation::{SpeculativeConfig, SpeculativeRequestStatus, SpeculativeSchedulerOptions},
    speculative::{PredictionPrefillAlignment, SpeculativeActivationPhase as Phase},
};
use eredu_nn::Tensor;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceHostBound,
    WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
    WorkspaceOutputStorage, WorkspaceTensor,
};
use std::num::NonZeroUsize;
fn selected(class: SpeculativeStrategyClass) -> SelectedSpeculativeRealization {
    let id = |s: &str| SpeculativeIdentity::new(s).unwrap();
    let target = id("target");
    let strategy = id("prediction");
    let capture = SpeculativeCaptureSchema::new(
        id("capture"),
        [
            SpeculativeCaptureEntry::new(id("hidden"), vec![1, 2, 8], id("rank"), id("seam"))
                .unwrap(),
        ],
    )
    .unwrap();
    let state = SpeculativeStateCacheIdentityIngredients::new(
        target.clone(),
        strategy.clone(),
        None,
        None,
        id("artifact"),
        id("format"),
        id("placement"),
        0,
        id("text"),
        vec![id("target-state"), id("prediction-state")],
    )
    .unwrap();
    let requirements = SpeculativeRealizationRequirements::new(
        target.clone(),
        SpeculativeStrategyRequirements::embedded(
            class,
            strategy.clone(),
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap(),
        capture.clone(),
        SpeculativeMechanismRequirements::new([]),
        state,
    )
    .unwrap();
    let request =
        SpeculativeSelectionRequest::new(SpeculativePlacementRequest::Single, capture.clone())
            .with_architecture_proof(SpeculativeArchitectureCompatibilityProof::new(
                target,
                strategy,
                capture.identity().clone(),
            ));
    select_speculative_realization(
        &requirements,
        &request,
        &SpeculativeMechanismCapabilities::new(
            requirements.mechanisms().mechanisms().iter().copied(),
        ),
    )
    .unwrap()
}
fn plan(
    selected: &SelectedSpeculativeRealization,
    chunk: u64,
    aligned: bool,
) -> EmbeddedSchedulePlan<'_> {
    let shape = if selected.requirements().strategy().class()
        == SpeculativeStrategyClass::EmbeddedSequential
    {
        EmbeddedPredictionShape::Sequential {
            depth: NonZeroUsize::new(5).unwrap(),
        }
    } else {
        EmbeddedPredictionShape::Fused {
            depth: NonZeroUsize::new(3).unwrap(),
            maximum_proposals: NonZeroUsize::new(8).unwrap(),
        }
    };
    let prefill = PrefillControlPlan::new(
        eredu_core::InferenceGeometry {
            batch_size: 1,
            input_positions: 7,
            cached_positions: 11,
            max_output_tokens: 5,
            prefill_chunk_positions: chunk,
            output: OutputDemand::LastPosition,
        },
        true,
    )
    .unwrap();
    let config = SpeculativeConfig {
        max_tokens: 5,
        max_draft_tokens: 2,
        ..Default::default()
    };
    EmbeddedSchedulePlan::new(
        selected,
        shape,
        if aligned {
            PredictionPrefillAlignment::Aligned
        } else {
            PredictionPrefillAlignment::NextToken
        },
        29,
        prefill,
        23,
        &config,
        SpeculativeSchedulerOptions {
            lookahead_blocks: 1,
            ..Default::default()
        },
    )
    .unwrap()
}
#[derive(Debug)]
struct ScalarSquare;
impl WorkspaceMechanisms for ScalarSquare {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "fixture square allocates its scalar output only".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture square has no disjoint host payload".into(),
        }))
    }
}
fn report(geometry: eredu_core::InferenceGeometry) -> InferenceWorkspaceReport {
    quote_inference_workspace(geometry, |_| {
        let context = WorkspaceContext::new(ScalarSquare);
        let layout = WorkspaceLayout::new(&[1], WorkspaceDtype::Float32)?;
        let source = WorkspaceExistingStorage::new(Some(layout.bytes()?), &context);
        let input = WorkspaceTensor::existing_with_storage(layout, &source, &context)?;
        // The report must cover opening state as well as the actual operation.
        // begin_span alone intentionally leaves inference state unqualified.
        context.begin_state_span([&input])?;
        let output = input.square(&context)?;
        context.report(&[output])
    })
    .unwrap()
}
fn requirements(plan: &InferenceSpanWorkspacePlan) -> SpeculativeInvocationRequirements {
    let physical = plan.records().iter().try_fold(0u64, |bytes, row| {
        bytes.checked_add(row.new_tensor_allocation_bytes()?)
    });
    // This portable fixture creates no native Graph, Record, Scope or worker.
    // Account controls are priced by reserve_role itself; the real traced
    // scalar output provides its nonzero physical obligation.
    SpeculativeInvocationRequirements::new(plan, physical, Some(0), Some(0), Some(0)).unwrap()
}

#[test]
fn embedded_account_binds_full_geometry_startup_and_numerical_source_identity() {
    use crate::speculative::numerical::{SpeculativeNumericalKind, SpeculativeNumericalProgram};
    for class in [
        SpeculativeStrategyClass::EmbeddedSequential,
        SpeculativeStrategyClass::EmbeddedFused,
    ] {
        let selected = selected(class);
        let schedule = plan(&selected, 3, true);
        let (target, prediction) = schedule.prefill_invocations(0).unwrap();
        let prediction = prediction.unwrap();
        let target_replay = schedule
            .decode_invocation(Phase::TargetReplay, 18, 1)
            .unwrap();
        let prediction_replay = schedule
            .decode_invocation(Phase::PredictionReplay, 18, 1)
            .unwrap();
        let replay_workspace = EmbeddedInvocationWorkspace::target(target_replay).unwrap();
        let replay_report = report(replay_workspace.geometry());
        let target_workspace = EmbeddedInvocationWorkspace::target(target).unwrap();
        let predicted = EmbeddedInvocationWorkspace::prediction(
            prediction,
            prediction.prefill_span().unwrap().seed_start,
            OutputDemand::StateOnly,
        )
        .unwrap();
        assert_ne!(
            target_workspace.geometry().cached_positions,
            predicted.geometry().cached_positions
        );
        let target_report = report(target_workspace.geometry());
        let prediction_report = report(predicted.geometry());
        let equivalent = report(predicted.geometry());
        let capacity = 1 << 26;
        let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let request =
            OriginalSpeculativeRequest::prepare_embedded(&pool, &execution, &schedule, capacity)
                .unwrap();
        let other =
            OriginalSpeculativeRequest::prepare_embedded(&pool, &execution, &schedule, capacity)
                .unwrap();
        let target_start = request
            .reserve_embedded_startup(OriginalEmbeddedSpeculativeSource::Target, 19)
            .unwrap();
        let prediction_start = request
            .reserve_embedded_startup(OriginalEmbeddedSpeculativeSource::Prediction, 23)
            .unwrap();
        assert!(target_start.belongs_to_request(&request));
        assert!(!target_start.belongs_to_request(&other));
        assert_eq!(
            prediction_start.source(),
            OriginalEmbeddedSpeculativeSource::Prediction
        );
        assert!(
            request
                .reserve_embedded_startup(OriginalEmbeddedSpeculativeSource::Target, 19)
                .is_err()
        );
        assert!(
            request
                .reserve_startup(AutoregressiveSource::Draft, 0)
                .is_err()
        );
        let mut cursor = schedule.into_cursor();
        let before = pool.used_bytes().unwrap();
        // Matching full geometry still cannot substitute another phase/source.
        let wrong = request
            .reserve_embedded_role(
                cursor.claim(prediction_replay).unwrap(),
                replay_workspace,
                requirements(replay_report.span_workspace_plan()),
            )
            .unwrap_err();
        assert!(matches!(
            wrong.cause(),
            WorkingMemoryError::IdentityMismatch
        ));
        drop(wrong);
        // Exact invocation matches, but the report came from the target frontier.
        // Equal allocation counts cannot certify prediction-local state geometry.
        let wrong = request
            .reserve_embedded_role(
                cursor.claim(prediction).unwrap(),
                predicted,
                requirements(target_report.span_workspace_plan()),
            )
            .unwrap_err();
        assert!(matches!(
            wrong.cause(),
            WorkingMemoryError::IdentityMismatch
        ));
        drop(wrong);
        assert_eq!(pool.used_bytes().unwrap(), before);
        let role = request
            .reserve_embedded_role(
                cursor.claim(prediction).unwrap(),
                predicted,
                requirements(prediction_report.span_workspace_plan()),
            )
            .unwrap();
        role.validate_execution(&execution).unwrap();
        assert!(
            role.validate_execution(&InferenceExecutionIdentity::default())
                .is_err()
        );
        role.validate_invocation(prediction).unwrap();
        assert!(role.validate_invocation(target).is_err());
        role.validate_plan(prediction_report.span_workspace_plan())
            .unwrap();
        assert!(
            role.validate_plan(equivalent.span_workspace_plan())
                .is_err()
        );
        role.claim_neural_bank(0).unwrap();
        assert!(role.clone().claim_neural_bank(0).is_err());
        let source = role.budget_custody();
        assert!(source.belongs_to_embedded(&role));
        let retained = request.source_identity();
        assert!(SpeculativeNumericalSource::Model(&source).belongs_to_identity(&retained));
        assert!(!retained.belongs_to_request(&other));
        let program =
            SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Normalize, &[1, 1]).unwrap();
        let bytes = role.physical_bytes();
        assert!(bytes > 0);
        let required = || {
            SpeculativeNumericalRequirements::new(program, Some(bytes), Some(0), Some(0), Some(0))
                .unwrap()
        };
        let used = pool.used_bytes().unwrap();
        assert!(matches!(
            other
                .reserve_numerical(required(), &[SpeculativeNumericalSource::Model(&source)])
                .unwrap_err()
                .cause(),
            WorkingMemoryError::IdentityMismatch
        ));
        assert_eq!(pool.used_bytes().unwrap(), used);
        let numerical = request
            .reserve_numerical(required(), &[SpeculativeNumericalSource::Model(&source)])
            .unwrap()
            .begin();
        let cumulative = pool.used_bytes().unwrap();
        assert!(cumulative > used);
        drop(role);
        drop(prediction_start);
        assert_eq!(pool.used_bytes().unwrap(), cumulative);
        request.close().unwrap();
        drop(other);
        drop(request);
        assert!(pool.used_bytes().unwrap() > 0);
        drop((target_start, source, numerical, retained));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn embedded_continuation_moves_spent_roles_and_retains_actual_account_after_request_drop() {
    let selected = selected(SpeculativeStrategyClass::EmbeddedFused);
    let schedule = plan(&selected, 3, false);
    let invocation = schedule
        .decode_invocation(Phase::FusedProposal, 18, 2)
        .unwrap();
    let workspace =
        EmbeddedInvocationWorkspace::prediction(invocation, 37, OutputDemand::Sequence).unwrap();
    assert_eq!(invocation.source_positions(), 1);
    assert_eq!(workspace.geometry().input_positions, 2);
    let report = report(workspace.geometry());
    let limit = schedule.attempts(EmbeddedOccurrenceKind::FusedProposal);
    let capacity = 1 << 26;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request =
        OriginalSpeculativeRequest::prepare_embedded(&pool, &execution, &schedule, capacity)
            .unwrap();
    let funding = pool
        .prepare_workspace_metadata(&execution, capacity)
        .unwrap();
    let foreign_schedule = plan(&selected, 3, false);
    let mut foreign = foreign_schedule.into_cursor();
    let before = pool.used_bytes().unwrap();
    assert!(matches!(
        request
            .reserve_embedded_role(
                foreign.claim(invocation).unwrap(),
                workspace,
                requirements(report.span_workspace_plan())
            )
            .unwrap_err()
            .cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    let foreign_extension = foreign
        .continuation(1, SpeculativeRequestStatus::ReadyToDraft)
        .unwrap();
    assert!(matches!(
        request.prepare_embedded_continuation(&foreign_extension, &funding),
        Err(SpeculativeContinuationError::Request(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    let mut cursor = schedule.into_cursor();
    // Refusal consumes one actual occurrence, but invents no accepted charge.
    let too_large = SpeculativeInvocationRequirements::new(
        report.span_workspace_plan(),
        Some(capacity),
        Some(0),
        Some(0),
        Some(0),
    )
    .unwrap();
    assert!(
        request
            .reserve_embedded_role(cursor.claim(invocation).unwrap(), workspace, too_large)
            .is_err()
    );
    assert_eq!(pool.used_bytes().unwrap(), before);
    for _ in 1..limit {
        request
            .reserve_embedded_role(
                cursor.claim(invocation).unwrap(),
                workspace,
                requirements(report.span_workspace_plan()),
            )
            .unwrap();
    }
    assert!(cursor.claim(invocation).is_err());
    let spent = cursor.attempted();
    let cumulative = pool.used_bytes().unwrap();
    let continuation = cursor
        .continuation(1, SpeculativeRequestStatus::ReadyToDraft)
        .unwrap();
    let competitor = pool
        .prepare_workspace_metadata(&execution, capacity)
        .unwrap();
    let remaining = usize::try_from(capacity - pool.used_bytes().unwrap()).unwrap();
    competitor.reserve_metadata(remaining).unwrap();
    let destination = vec![0u8; remaining];
    assert!(matches!(
        request.prepare_embedded_continuation(&continuation, &funding),
        Err(SpeculativeContinuationError::Metadata(_))
    ));
    assert_eq!(cursor.attempted(), spent);
    assert!(cursor.claim(invocation).is_err());
    drop(destination);
    drop(competitor);
    assert_eq!(pool.used_bytes().unwrap(), cumulative);
    request
        .prepare_embedded_continuation(&continuation, &funding)
        .unwrap();
    let extended = pool.used_bytes().unwrap();
    assert!(extended > cumulative);
    assert!(matches!(
        request.prepare_embedded_continuation(&continuation, &funding),
        Err(SpeculativeContinuationError::Request(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), extended);
    cursor.install_continuation(continuation).unwrap();
    let claim = cursor.claim(invocation).unwrap();
    assert_eq!(claim.ordinal(), spent);
    let role = request
        .reserve_embedded_role(claim, workspace, requirements(report.span_workspace_plan()))
        .unwrap();
    let alias = role.budget_custody();
    role.claim_neural_bank(0).unwrap();
    assert!(role.clone().claim_neural_bank(0).is_err());
    let final_used = pool.used_bytes().unwrap();
    drop(role);
    assert_eq!(pool.used_bytes().unwrap(), final_used);
    request.close().unwrap();
    drop(request);
    drop(funding);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod capture;
