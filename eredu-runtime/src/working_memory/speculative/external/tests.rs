//! Source geometry, account identity and failed/continued attempt behavior.
use super::*;
use crate::*;
use crate::speculative::external_occurrence::*;
use crate::prefill::PrefillControlPlan;
use eredu_core::{OutputDemand, InferenceGeometry, SpeculativeRequestId,
    generation::{SpeculativeConfig, SpeculativeRequestStatus, SpeculativeSchedulerOptions},
    speculative::SpeculativeActivationOrigin};
use eredu_nn::{Tensor, workspace::{WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage,
    WorkspaceHostBound, WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation,
    WorkspaceOperationBound, WorkspaceOutputStorage, WorkspaceTensor}};
use std::num::NonZeroUsize;
fn selected() -> SelectedSpeculativeRealization {
    let id = |s: &str| SpeculativeIdentity::new(s).unwrap();
    let target = id("target"); let strategy = id("assistant");
    let capture = SpeculativeCaptureSchema::new(id("capture"), [
        SpeculativeCaptureEntry::new(id("hidden"), vec![1, 2, 8], id("rank"), id("seam")).unwrap(),
    ]).unwrap();
    let state = SpeculativeStateCacheIdentityIngredients::new(target.clone(), strategy.clone(),
        Some(id("assistant-source")), Some([7;32]), id("artifact"), id("format"), id("placement"),
        0, id("text"), vec![id("target-state"), id("assistant-state")]).unwrap();
    let requirements = SpeculativeRealizationRequirements::new(target.clone(),
        SpeculativeStrategyRequirements::external(strategy.clone(), NonZeroUsize::new(2).unwrap(), [7;32]),
        capture.clone(), SpeculativeMechanismRequirements::new([]), state).unwrap();
    let request = SpeculativeSelectionRequest::new(SpeculativePlacementRequest::Single, capture.clone())
        .with_architecture_proof(SpeculativeArchitectureCompatibilityProof::new(target, strategy,
            capture.identity().clone()))
        .with_tokenizer_proof(eredu_core::TokenizerCompatibilityProof::prove([7;32], [7;32]).unwrap());
    select_speculative_realization(&requirements, &request,
        &SpeculativeMechanismCapabilities::new(requirements.mechanisms().mechanisms().iter().copied())).unwrap()
}
fn schedule(selected: &SelectedSpeculativeRealization, shape: ExternalPredictionShape) -> ExternalSchedulePlan<'_> {
    ExternalSchedulePlan::new(selected, shape, PrefillControlPlan::new(InferenceGeometry {
        batch_size:1, cached_positions:11, input_positions:7, max_output_tokens:5,
        prefill_chunk_positions:3, output:OutputDemand::LastPosition,
    }, true).unwrap(), 23, &SpeculativeConfig { max_tokens:5, max_draft_tokens:2, ..Default::default() },
        SpeculativeSchedulerOptions { lookahead_blocks:1, ..Default::default() }).unwrap()
}
fn geometry(cached: u64, width: u64) -> InferenceGeometry {
    InferenceGeometry { batch_size:1, cached_positions:cached, input_positions:width,
        max_output_tokens:0, prefill_chunk_positions:width, output:OutputDemand::Sequence }
}
fn origin() -> SpeculativeActivationOrigin {
    descriptive_origin(&[7])
}
fn descriptive_origin(tokens: &[u32]) -> SpeculativeActivationOrigin {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"eredu.speculative.activation-prefix.v1");
    for token in tokens { digest.update(token.to_le_bytes()); }
    SpeculativeActivationOrigin { request: SpeculativeRequestId::new(3),
        committed_tokens: tokens.len(), prediction: tokens.len(),
        prefix_digest: digest.finalize().into(), optimistic: false }
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
fn external_calls_preserve_shared_chunk_frontier_and_loop_populations() {
    let source = selected();
    for shape in [ExternalPredictionShape::Sequential, ExternalPredictionShape::Fused] {
        let plan = schedule(&source, shape);
        assert_eq!(plan.attempts(ExternalInvocationKind::TargetPrefill), 3);
        assert_eq!(plan.attempts(ExternalInvocationKind::TargetVerification), 8);
        let prefill_origin = descriptive_origin(&[]);
        for (start, width) in [(0,3),(3,3),(6,1)] {
            let span = eredu_core::speculative::SpeculativePrefillSpan {
                prompt_tokens:7,input_start:start,input_end:start+width,position:11+start,
                hidden_start:start,token_start:start,sequence:width,seed_start:11+start,
            };
            let call = plan.invocation(ExternalInvocationKind::TargetPrefill, geometry(11+start,width), Some(span), prefill_origin).unwrap();
            assert_eq!(call.prefill_span(), Some(span));
        }
        assert!(plan.invocation(ExternalInvocationKind::TargetPrefill, geometry(11,3), None, prefill_origin).is_err());
        assert!(plan.invocation(ExternalInvocationKind::TargetVerification, geometry(17,2), None, origin()).is_err());
        assert!(plan.invocation(ExternalInvocationKind::TargetVerification, geometry(18,1), None, origin()).is_ok());
        assert!(plan.invocation(ExternalInvocationKind::TargetVerification, geometry(18,3), None, origin()).is_ok());
        assert!(plan.invocation(ExternalInvocationKind::TargetVerification, geometry(18,4), None, origin()).is_err());
        match shape {
            ExternalPredictionShape::Sequential => {
                assert_eq!(plan.attempts(ExternalInvocationKind::AssistantStep),16);
                assert_eq!(plan.attempts(ExternalInvocationKind::TargetTokenEmbeddings),16);
                assert_eq!(plan.attempts(ExternalInvocationKind::TargetProjectLogits),0);
            }
            ExternalPredictionShape::Fused => {
                assert_eq!(plan.attempts(ExternalInvocationKind::AssistantStep),0);
                assert_eq!(plan.attempts(ExternalInvocationKind::FusedProposal),8);
                assert_eq!(plan.attempts(ExternalInvocationKind::TargetTokenEmbeddings),8);
                assert_eq!(plan.attempts(ExternalInvocationKind::AssembleContext),7);
            }
        }
    }
}
#[test]
fn external_roles_bind_report_identity_and_keep_failed_attempts_and_continuation_custody() {
    let source = selected(); let plan = schedule(&source, ExternalPredictionShape::Sequential);
    let invocation = plan.invocation(ExternalInvocationKind::AssistantStep, geometry(18,1), None, origin()).unwrap();
    let capacity = 1 << 26; let pool = WorkingMemoryPool::new(capacity,0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare_external(&pool,&execution,&plan,capacity).unwrap();
    let startup = request.reserve_external_startup(OriginalExternalSpeculativeSource::Assistant,23).unwrap();
    assert!(startup.belongs_to_request(&request));
    assert!(request.reserve_external_startup(OriginalExternalSpeculativeSource::Assistant,23).is_err());
    assert!(request.reserve_startup(AutoregressiveSource::Draft,0).is_err());
    let exact = report(invocation.geometry()); let wrong = report(geometry(19,1));
    let equivalent = report(invocation.geometry());
    let funding = pool.prepare_workspace_metadata(&execution, capacity).unwrap();
    let mut cursor = plan.into_cursor();
    let failure = request.reserve_external_role(cursor.claim(invocation).unwrap(),requirements(wrong.span_workspace_plan())).unwrap_err();
    assert!(matches!(failure.cause(),WorkingMemoryError::IdentityMismatch)); drop(failure);
    assert_eq!(cursor.attempted(),1);
    let role = request.reserve_external_role(cursor.claim(invocation).unwrap(),requirements(exact.span_workspace_plan())).unwrap();
    assert!(role.physical_bytes()>0); role.validate_plan(exact.span_workspace_plan()).unwrap();
    assert!(role.validate_plan(equivalent.span_workspace_plan()).is_err());
    role.claim_neural_bank(0).unwrap(); assert!(role.clone().claim_neural_bank(0).is_err());
    let custody = role.budget_custody(); assert!(custody.belongs_to_external(&role));
    let before = pool.used_bytes().unwrap();
    let continuation = cursor.continuation(1,SpeculativeRequestStatus::ReadyToDraft).unwrap();
    let stale = cursor.continuation(1,SpeculativeRequestStatus::ReadyToDraft).unwrap();
    request.prepare_external_continuation(&continuation,&funding).unwrap();
    cursor.install_continuation(continuation).unwrap();
    assert!(cursor.install_continuation(stale).is_err());
    assert_eq!(cursor.attempted(),2); assert!(pool.used_bytes().unwrap()>before);
    // New future slots do not recreate an already spent role or startup.
    assert!(request.reserve_external_startup(OriginalExternalSpeculativeSource::Assistant,23).is_err());
    request.close().unwrap(); drop(request); drop(role); drop(startup); drop(funding);
    assert!(pool.used_bytes().unwrap()>0); drop(custody); assert_eq!(pool.used_bytes().unwrap(),0);
}
#[test]
fn external_population_overflow_is_refused_before_issuing_a_schedule() {
    let source=selected();
    let prefill=PrefillControlPlan::new(InferenceGeometry { batch_size:1,cached_positions:0,
        input_positions:1,max_output_tokens:5,prefill_chunk_positions:1,output:OutputDemand::LastPosition },true).unwrap();
    assert!(ExternalSchedulePlan::new(&source,ExternalPredictionShape::Sequential,prefill,6,
        &SpeculativeConfig {max_tokens:5,max_draft_tokens:2,..Default::default()},
        SpeculativeSchedulerOptions {lookahead_blocks:usize::MAX-1,..Default::default()}).is_err());
}

#[test]
fn retained_speculative_origin_accepts_closed_source_and_refuses_foreign_or_quarantined_account() {
    for quarantine in [false, true] {
        let source = selected();
        let plan = schedule(&source, ExternalPredictionShape::Sequential);
        let invocation = plan.invocation(ExternalInvocationKind::AssistantStep, geometry(18, 1), None, origin()).unwrap();
        let capacity = 1 << 26;
        let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
        let foreign = WorkingMemoryPool::new(capacity, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let request = OriginalSpeculativeRequest::prepare_external(&pool, &execution, &plan, capacity).unwrap();
        let report = report(invocation.geometry());
        let mut cursor = plan.into_cursor();
        let role = request.reserve_external_role(cursor.claim(invocation).unwrap(),
            requirements(report.span_workspace_plan())).unwrap();
        let source = role.budget_custody();
        let custody: crate::working_memory::OriginalOperationMetadataCustody = source.clone().into();
        custody.validate_retained_origin(&pool).unwrap();
        assert_eq!(custody.validate_retained_origin(&foreign), Err(WorkingMemoryError::IdentityMismatch));
        request.close().unwrap();
        if quarantine { source.quarantine(); }
        drop((request, role, source));
        let held = pool.used_bytes().unwrap();
        assert!(held > 0);
        assert_eq!(custody.validate_retained_origin(&pool),
            if quarantine { Err(WorkingMemoryError::ExecutionFenced) } else { Ok(()) });
        assert_eq!(pool.used_bytes().unwrap(), held);
        drop(custody);
        assert_eq!(pool.used_bytes().unwrap(), if quarantine { held } else { 0 });
    }
}
