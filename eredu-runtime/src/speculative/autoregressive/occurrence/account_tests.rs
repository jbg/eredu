//! Actual selected occurrence/account boundary; no native fit is fabricated.
use super::*;
use crate::working_memory::*;
use crate::*;
use eredu_core::automatic::TokenizerCompatibilityProof;
use eredu_core::generation::{SpeculativeConfig, SpeculativeSchedulerOptions};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceHostBound,
    WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
    WorkspaceOutputStorage, WorkspaceTensor,
};
use eredu_nn::Tensor;

fn selected() -> SelectedSpeculativeRealization {
    let id = |value: &str| SpeculativeIdentity::new(value).unwrap();
    let target = id("target");
    let strategy = id("independent-draft");
    let capture = SpeculativeCaptureSchema::independent(id("no-capture"));
    let state = SpeculativeStateCacheIdentityIngredients::new(
        target.clone(),
        strategy.clone(),
        Some(id("draft")),
        Some([7; 32]),
        id("artifact"),
        id("safetensors"),
        id("replicated"),
        0,
        id("text"),
        vec![id("target-state"), id("draft-state")],
    )
    .unwrap();
    let requirements = SpeculativeRealizationRequirements::new(
        target.clone(),
        SpeculativeStrategyRequirements::external(
            strategy.clone(),
            NonZeroUsize::new(1).unwrap(),
            [7; 32],
        ),
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
            ))
            .with_tokenizer_proof(TokenizerCompatibilityProof::prove([7; 32], [7; 32]).unwrap());
    select_speculative_realization(
        &requirements,
        &request,
        &SpeculativeMechanismCapabilities::new(
            requirements.mechanisms().mechanisms().iter().copied(),
        ),
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
fn speculative_role_claims_survive_rollback_and_retain_charge_through_raw_output_alias() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let plan = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let draft = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let target = AutoregressiveInvocation::decode(AutoregressivePass::TargetCommit, 1).unwrap();
    let geometry = plan
        .workspace_geometry(2, draft, NonZeroU64::new(1).unwrap())
        .unwrap();
    let first = report(geometry);
    let equivalent = report(geometry);
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare(&pool, &execution, &plan, 1 << 24).unwrap();
    let mut cursor = plan.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, draft).unwrap(),
            requirements(first.span_workspace_plan()),
        )
        .unwrap();
    assert!(role.validate_invocation(target).is_err());
    assert!(role
        .validate_plan(equivalent.span_workspace_plan())
        .is_err());
    role.validate_invocation(draft).unwrap();
    role.validate_plan(first.span_workspace_plan()).unwrap();
    role.claim_neural_bank(0).unwrap();
    assert!(role.clone().claim_neural_bank(0).is_err());
    let after_first = pool.used_bytes().unwrap();
    // A restored cache has the same frontier. Its new native attempt still
    // consumes the next occurrence and cannot recycle the old bank or charge.
    let replay = request
        .reserve_role(
            cursor.claim(2, draft).unwrap(),
            requirements(first.span_workspace_plan()),
        )
        .unwrap();
    assert!(!role.same_role(&replay));
    assert!(pool.used_bytes().unwrap() > after_first);
    assert!(matches!(
        cursor.claim(2, draft),
        Err(AutoregressiveOccurrenceError::Exhausted)
    ));
    assert_eq!(cursor.attempted(), 2);
    let raw_output_account = role.budget_custody();
    assert!(raw_output_account.belongs_to(&role));
    assert!(!raw_output_account.belongs_to(&replay));
    request.close().unwrap();
    drop(replay);
    assert_eq!(pool.used_bytes().unwrap(), after_first,
        "retired replay has no request-held workspace owner");
    drop(role);
    assert_eq!(pool.used_bytes().unwrap(), after_first,
        "raw output still owns the first operation");
    assert_eq!(cursor.attempted(), 2, "retirement cannot refund attempts");
    drop(request);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(raw_output_account);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_prefill_uses_shared_chunks_without_refunding_cancelled_claims() {
    use crate::prefill::{PrefillChunk, PrefillDriver, PrefillExecutor, PrefillProgress};
    use eredu_core::{Completion, GenerationCancellationToken, OutputDemand, Submission};
    use std::convert::Infallible;
    struct Done;
    impl Completion for Done {
        type Error = Infallible;
        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }
        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    struct Executor {
        cancellation: GenerationCancellationToken,
        claimed: Option<OriginalSpeculativePrefillSpan>,
    }
    impl PrefillExecutor<OriginalSpeculativeRole> for Executor {
        type Output = ();
        type Completion = Done;
        type Error = WorkingMemoryError;
        fn submit_chunk(
            &mut self,
            chunk: &PrefillChunk,
            role: OriginalSpeculativeRole,
        ) -> Result<Submission<Option<()>, Done>, Self::Error> {
            assert_eq!(chunk.input, 0..2);
            assert_eq!(chunk.output, OutputDemand::StateOnly);
            let claimed = role.claim_prefill_span(chunk)?;
            assert_eq!(claimed.ordinal(), 0);
            assert!(claimed.role().same_role(&role));
            assert!(role.claim_neural_bank(0).is_err());
            claimed.claim_neural_bank(0)?;
            assert!(claimed.claim_neural_bank(0).is_err());
            assert!(role.clone().claim_prefill_span(chunk).is_err());
            self.claimed = Some(claimed);
            self.cancellation.cancel();
            Ok(Submission {
                output: None,
                completion: Done,
            })
        }
    }
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 1,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let plan = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(3).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let invocation =
        AutoregressiveInvocation::prefill(AutoregressivePass::TargetPrefill, 3).unwrap();
    let geometry = plan
        .workspace_geometry(0, invocation, NonZeroU64::new(2).unwrap())
        .unwrap();
    let report = report(geometry);
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare(&pool, &execution, &plan, 1 << 24).unwrap();
    let mut cursor = plan.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(0, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let cancellation = GenerationCancellationToken::new();
    let mut driver = PrefillDriver::<(), Done, _>::new_original_speculative(
        &execution,
        role.clone(),
        geometry,
        cancellation.clone(),
    )
    .unwrap();
    let before = pool.used_bytes().unwrap();
    let mut executor = Executor {
        cancellation: cancellation.clone(),
        claimed: None,
    };
    assert!(matches!(
        driver.step(&mut executor).unwrap(),
        PrefillProgress::Cancelled
    ));
    assert_eq!(driver.completed_positions(), 2);
    assert!(matches!(
        driver.step(&mut executor).unwrap(),
        PrefillProgress::Cancelled
    ));
    drop(executor.claimed.take());
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert!(matches!(
        PrefillDriver::<(), Done, _>::new_original_speculative(
            &execution,
            role.clone(),
            geometry,
            GenerationCancellationToken::new(),
        ),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let prior = match report.span_workspace_plan().records()[0].span() {
        InferenceWorkspaceSpan::Prefill(chunk) => chunk,
        _ => panic!("actual prefill report"),
    };
    assert!(role.claim_prefill_span(prior).is_err());
    drop(driver);
    drop(executor);
    drop(role);
    drop(request);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn numerical_phases_retire_unused_custody_without_refunding_attempts_or_accepting_foreign_sources() {
    use crate::speculative::numerical::{SpeculativeNumericalKind, SpeculativeNumericalProgram};
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let plan = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let invocation = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        plan.workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let capacity = 1 << 24;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare(&pool, &execution, &plan, capacity).unwrap();
    // Two requests may borrow one schedule. Only the original header's source
    // accounts are valid numerical inputs; a schedule equality is insufficient.
    let other = OriginalSpeculativeRequest::prepare(&pool, &execution, &plan, capacity).unwrap();
    let mut cursor = plan.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let source = role.budget_custody();
    let program =
        SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Normalize, &[1, 1]).unwrap();
    // This portable account fixture has one scalar backing from the existing
    // actual mock trace. It constructs no native Graph/Record/Scope domains.
    let bytes = report.span_workspace_plan().records()[0]
        .new_tensor_allocation_bytes()
        .unwrap();
    assert!(bytes > 0);
    let required = |bytes| {
        SpeculativeNumericalRequirements::new(program, Some(bytes), Some(0), Some(0), Some(0))
            .unwrap()
    };
    let before = pool.used_bytes().unwrap();
    assert!(matches!(
        other
            .reserve_numerical(
                required(bytes),
                &[SpeculativeNumericalSource::Model(&source)]
            )
            .unwrap_err()
            .cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    let first = request
        .reserve_numerical(
            required(bytes),
            &[SpeculativeNumericalSource::Model(&source)],
        )
        .unwrap();
    assert_eq!(first.ordinal(), 0);
    let first = first.begin();
    let accepted = pool.used_bytes().unwrap();
    assert!(accepted > before);
    let second = request
        .reserve_numerical(
            required(bytes),
            &[SpeculativeNumericalSource::Numerical(&first)],
        )
        .unwrap();
    assert_eq!(second.ordinal(), 1);
    let last_alias = second.begin();
    let cumulative = pool.used_bytes().unwrap();
    assert!(cumulative > accepted);
    drop(first);
    let live = pool.used_bytes().unwrap();
    assert!(live < cumulative, "retired numerical source releases its own charge");
    assert!(live > before, "the escaped second result remains charged");
    assert!(
        request
            .reserve_numerical(
                required(capacity),
                &[SpeculativeNumericalSource::Numerical(&last_alias)]
            )
            .is_err()
    );
    assert_eq!(pool.used_bytes().unwrap(), live);
    let after_refusal = request.reserve_numerical(required(bytes),
        &[SpeculativeNumericalSource::Numerical(&last_alias)]).unwrap();
    assert_eq!(after_refusal.ordinal(), 3, "failed admission does not rewind attempts");
    drop(after_refusal);
    assert_eq!(pool.used_bytes().unwrap(), live, "unused permission retires, spent ordinal does not");
    request.close().unwrap();
    assert!(
        request
            .reserve_numerical(
                required(bytes),
                &[SpeculativeNumericalSource::Numerical(&last_alias)]
            )
            .is_err()
    );
    drop(source);
    drop(role);
    drop(other);
    drop(request);
    assert!(
        pool.used_bytes().unwrap() > 0,
        "escaped numerical backing keeps only its accepted account"
    );
    drop(last_alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn restored_occurrences_extend_paid_storage_without_recycling_spent_roles() {
    use eredu_core::generation::SpeculativeRequestStatus as Status;
    let selected = selected();
    let config = SpeculativeConfig { max_tokens: 3, max_draft_tokens: 1, ..Default::default() };
    let plan = AutoregressiveSchedulePlan::new(&selected, NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(), NonZeroU64::new(16).unwrap(), &config,
        SpeculativeSchedulerOptions::default()).unwrap();
    let invocation = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(plan.workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap()).unwrap());
    let capacity = 1 << 24;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare(&pool, &execution, &plan, capacity).unwrap();
    let funding = pool.prepare_workspace_metadata(&execution, capacity).unwrap();
    let mut cursor = plan.into_cursor();
    cursor.begin_cache().unwrap();
    for _ in 0..2 {
        request.reserve_role(cursor.claim(2, invocation).unwrap(), requirements(report.span_workspace_plan())).unwrap();
    }
    assert!(matches!(cursor.claim(2, invocation), Err(AutoregressiveOccurrenceError::Exhausted)));
    let spent = cursor.attempted();
    let before = pool.used_bytes().unwrap();
    assert!(cursor.continuation(0, Status::ReadyToDraft).is_err());
    assert!(cursor.continuation(1, Status::TargetVerificationInFlight).is_err());
    assert!(cursor.continuation(1, Status::Cancelled).is_err());
    let terminal = cursor.continuation(1, Status::Completed).unwrap();
    assert_eq!(terminal.previous_slots(), terminal.next_slots(), "early EOS cannot restore work");
    assert_eq!(pool.used_bytes().unwrap(), before);
    let continuation = cursor.continuation(1, Status::ReadyToDraft).unwrap();
    assert_eq!(continuation.next_slots() - continuation.previous_slots(), 10);
    // A competing real host destination leaves no capacity for the extension.
    // Its later retirement makes a retry possible without refunding this request.
    let competitor = pool.prepare_workspace_metadata(&execution, capacity).unwrap();
    let remaining = usize::try_from(capacity - pool.used_bytes().unwrap()).unwrap();
    competitor.reserve_metadata(remaining).unwrap();
    let competing_destination = vec![0u8; remaining];
    assert!(matches!(request.prepare_continuation(&continuation, &funding),
        Err(SpeculativeContinuationError::Metadata(eredu_nn::workspace::HostMetadataFundingError::Capacity { .. }))));
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    assert_eq!(cursor.attempted(), spent);
    assert!(matches!(cursor.claim(2, invocation), Err(AutoregressiveOccurrenceError::Exhausted)));
    drop(competing_destination);
    drop(competitor);
    assert_eq!(pool.used_bytes().unwrap(), before);
    request.prepare_continuation(&continuation, &funding).unwrap();
    let extended = pool.used_bytes().unwrap();
    assert!(extended > before);
    // A duplicate extension refuses before another destination or debit. The
    // same plan cannot replay the old limits into an already extended bank.
    assert!(matches!(request.prepare_continuation(&continuation, &funding),
        Err(SpeculativeContinuationError::Request(WorkingMemoryError::IdentityMismatch))));
    assert_eq!(pool.used_bytes().unwrap(), extended);
    cursor.install_continuation(continuation);
    assert_eq!(cursor.attempted(), spent);
    let claim = cursor.claim(2, invocation).unwrap();
    assert_eq!(claim.ordinal(), spent);
    assert_eq!(claim.pass_ordinal(), 2);
    request.reserve_role(claim, requirements(report.span_workspace_plan())).unwrap();
    request.reserve_role(cursor.claim(2, invocation).unwrap(), requirements(report.span_workspace_plan())).unwrap();
    assert!(matches!(cursor.claim(2, invocation), Err(AutoregressiveOccurrenceError::Exhausted)));
    assert_eq!(cursor.attempted(), spent + 2);
    let again = cursor.continuation(2, Status::ReadyToDraft).unwrap();
    assert_eq!(again.next_slots() - again.previous_slots(), 5);
    let prior = pool.used_bytes().unwrap();
    request.prepare_continuation(&again, &funding).unwrap();
    assert!(pool.used_bytes().unwrap() > prior);
    cursor.install_continuation(again);
    let claim = cursor.claim(2, invocation).unwrap();
    assert_eq!(claim.ordinal(), spent + 2);
    assert_eq!(claim.pass_ordinal(), 4);
    let role = request.reserve_role(claim, requirements(report.span_workspace_plan())).unwrap();
    let alias = role.budget_custody();
    assert!(matches!(cursor.claim(2, invocation), Err(AutoregressiveOccurrenceError::Exhausted)));
    assert!(pool.used_bytes().unwrap() > extended);
    drop(role);
    drop(request);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(funding);
    assert!(pool.used_bytes().unwrap() > 0, "escaped actual role retains its own account");
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod source_publication;

mod capture;

mod completed_sources;

mod registered_sources;

mod batch_cursors;

#[test]
fn media_ingress_origin_rejects_equal_geometry_foreign_roles_and_spent_ordinals() {
    use crate::media_prefill::MediaPrefillOrigin;
    let selected = selected();
    let config = SpeculativeConfig { max_tokens: 1, max_draft_tokens: 1, ..Default::default() };
    let fixture = || {
        let plan = AutoregressiveSchedulePlan::new(
            &selected, NonZeroUsize::new(1).unwrap(), NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(16).unwrap(), &config, SpeculativeSchedulerOptions::default(),
        ).unwrap();
        let invocation=AutoregressiveInvocation::prefill(AutoregressivePass::TargetPrefill,3).unwrap();
        let geometry=plan.workspace_geometry(0,invocation,NonZeroU64::new(2).unwrap()).unwrap();
        let report=report(geometry);
        let pool=WorkingMemoryPool::new(1<<24,0).unwrap();
        let execution=InferenceExecutionIdentity::default();
        let request=OriginalSpeculativeRequest::prepare(&pool,&execution,&plan,1<<24).unwrap();
        let mut cursor=plan.into_cursor();
        let role=request.reserve_role(cursor.claim(0,invocation).unwrap(),requirements(report.span_workspace_plan())).unwrap();
        (pool,execution,request,role,geometry,report)
    };
    let (pool,execution,request,role,geometry,report)=fixture();
    let (other_pool,other_execution,other_request,other,other_geometry,other_report)=fixture();
    assert_eq!(geometry,other_geometry);
    let mut wrong=geometry; wrong.prefill_chunk_positions=1;
    assert!(MediaPrefillOrigin::speculative(role.clone(),wrong).is_err());
    let mut source=MediaPrefillOrigin::speculative(role.clone(),geometry).unwrap();
    assert!(source.request().is_err(), "a media role is not an ordinary inference request");
    role.begin_prefill(&execution,geometry).unwrap();
    other.begin_prefill(&other_execution,other_geometry).unwrap();
    let chunk=|report:&InferenceWorkspaceReport,index:usize| match report.span_workspace_plan().records()[index].span() {
        InferenceWorkspaceSpan::Prefill(chunk)=>chunk.clone(), _=>panic!("actual prefill row"),
    };
    let first=role.claim_prefill_span(&chunk(&report,0)).unwrap();
    let second=role.claim_prefill_span(&chunk(&report,1)).unwrap();
    let foreign=other.claim_prefill_span(&chunk(&other_report,0)).unwrap();
    let equation = MediaPrefillOrigin::Equation;
    assert!(equation.request().is_err(), "workspace traversal has no ordinary request");
    assert!(equation.validate_span(&first).is_err(), "workspace traversal cannot consume a native media role");
    assert!(source.validate_span(&foreign).is_err());
    assert!(source.validate_span(&second).is_err(), "an already claimed future span cannot skip ingress");
    source.validate_span(&first).unwrap();
    let spent=pool.used_bytes().unwrap();
    source.committed().unwrap();
    assert!(source.validate_span(&first).is_err(), "commit cannot replay encoder ingress");
    source.validate_span(&second).unwrap();
    source.committed().unwrap();
    assert!(source.validate_span(&second).is_err());
    assert_eq!(pool.used_bytes().unwrap(),spent);
    request.close().unwrap(); other_request.close().unwrap();
    drop((first,second,foreign,role,other,request,other_request));
    assert!(pool.used_bytes().unwrap()>0, "retained ingress keeps its exact occurrence account");
    assert_eq!(other_pool.used_bytes().unwrap(),0);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(),0);
}
