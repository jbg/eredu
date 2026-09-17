//! The real numerical claim, shared logical ledger and escaped frame owner.
use super::*;
use crate::capture::{FundedSpeculativeCaptureInvocation, ScheduledCaptureBackend};
use crate::generation::{DefaultSampler, SpeculativeSampler};
use crate::speculative::numerical::{SpeculativeNumericalKind, SpeculativeNumericalProgram};
use eredu_core::{capture::*, checkpoint::TensorDtype, *};

fn capture_source() -> SharedCapturePlan {
    capture_source_with_transform(CaptureTransform::FullTensor)
}
fn capture_source_with_transform(transform: CaptureTransform) -> SharedCapturePlan {
    let point = ObservationPoint {
        path: MODEL_LOGITS_OBSERVATION_PATH.into(),
        node_id: "logits".into(),
        meaning: "one actual sampler input row".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "batch".into(),
                dimension: SymbolicDimension::Batch,
            },
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "vocabulary".into(),
                dimension: SymbolicDimension::Known(4),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: MODEL_LOGITS_OBSERVATION_PATH.into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let caps = CaptureCapabilities {
        transformations: vec![transform.kind()],
        max_histogram_bins: match &transform {
            CaptureTransform::Histogram { edges } => edges.len().saturating_sub(1) as u64,
            _ => 0,
        },
        physical_native_limit: false,
        conditions: vec![],
    };
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "row".into(),
        path: MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform,
    });
    plan.limits = CaptureLimits {
        per_step: CaptureUsage {
            captures: 1,
            retained_bytes: u64::MAX,
            host_bytes: u64::MAX,
            encoded_bytes: u64::MAX,
        },
        cumulative: CaptureUsage {
            captures: 2,
            retained_bytes: u64::MAX,
            host_bytes: u64::MAX,
            encoded_bytes: u64::MAX,
        },
        physical_native_bytes: None,
        on_limit: CaptureLimitPolicy::Skip,
    };
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 1,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    )
}
fn program(prediction: u64) -> SpeculativeNumericalProgram {
    let policy =
        <DefaultSampler as SpeculativeSampler<WorkspaceSamplingBackend>>::prepared_logit_policy(
            &DefaultSampler,
        )
        .unwrap()
        .bind(0.0, prediction as usize)
        .unwrap();
    SpeculativeNumericalProgram::new(SpeculativeNumericalKind::ProcessLogits(policy), &[1, 4])
        .unwrap()
}
fn host<'a>(source: &'a SharedCapturePlan, prediction: u64) -> SpeculativeCaptureHostPlan<'a> {
    SpeculativeCaptureHostPlan::prepare(source, prediction, program(prediction)).unwrap()
}
fn numerical(source: &SharedCapturePlan, prediction: u64) -> SpeculativeNumericalRequirements {
    // The default zero-temperature worker is identity. This portable destination
    // fixture performs no native tensor, Graph, Record or Scope construction.
    SpeculativeNumericalRequirements::new(program(prediction), Some(0), Some(0), Some(0), Some(0))
        .unwrap()
        .with_capture_destination(&host(source, prediction))
        .unwrap()
}
#[derive(Default)]
struct HostBackend {
    calls: usize,
    refuse: bool,
}
impl ScheduledCaptureBackend for HostBackend {
    type Tensor = [f32; 4];
    type Error = CaptureRunHostError;
    fn validate_source(
        &self,
        _: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error> {
        assert_eq!(geometry.source_shape(), [1, 1, 4]);
        Ok(TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: geometry.elements() as u64 * 4,
            host_bytes: geometry.elements() as u64 * 4,
            encoded_bytes: 8192,
        })
    }
    fn transform(
        &mut self,
        value: &Self::Tensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error> {
        self.calls += 1;
        if self.refuse {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let mut destination = claim.prepare()?;
        for &value in value {
            destination.push_f32(value).unwrap();
        }
        Ok(destination.finish().unwrap())
    }
}
fn invoke(
    request: &OriginalSpeculativeRequest,
    source: &SharedCapturePlan,
    model: &OriginalSpeculativeBudgetCustody,
    prediction: u64,
) -> (
    OriginalSpeculativeNumericalBudgetCustody,
    FundedSpeculativeCaptureInvocation,
) {
    request
        .reserve_numerical(
            numerical(source, prediction),
            &[SpeculativeNumericalSource::Model(model)],
        )
        .unwrap()
        .begin_with_capture(host(source, prediction))
        .unwrap()
}
#[test]
fn numerical_capture_replay_keeps_live_usage_and_escaped_shared_frame_custody() {
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let selected = selected();
    let plan = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let pass = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        plan.workspace_geometry(2, pass, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let capacity = 1 << 24;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare(&pool, &execution, &plan, capacity).unwrap();
    let mut cursor = plan.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, pass).unwrap(),
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let model = role.budget_custody();
    let source = capture_source();
    let values = [-0.5, 0.25, 1.0, -2.0];
    let mut backend = HostBackend::default();
    let (first_budget, mut first) = invoke(&request, &source, &model, 1);
    assert_eq!(first.ordinal(), 0);
    first.observe(&mut backend, &values).unwrap();
    let escaped = first.take_shared_step().unwrap().unwrap();
    assert_eq!(escaped.as_ref().outcome, CaptureStepOutcome::Untracked);
    assert_eq!(first.usage().captures, 1);
    let snapshot_alias = escaped.clone();
    let (later_budget, mut later) = invoke(&request, &source, &model, 2);
    assert_eq!(later.ordinal(), 1);
    later.observe(&mut backend, &values).unwrap();
    assert_eq!(later.usage().captures, 2);
    drop(later.take_shared_step().unwrap());
    // A replay/fork of the earlier logical position still belongs to this
    // retained request. It sees work performed after the saved frame, exactly
    // as ordinary sampler snapshots share their Rc<RefCell<LogitCapture>>.
    let (replay_budget, mut replay) = invoke(&request, &source, &model, 1);
    assert_eq!(replay.ordinal(), 2);
    replay.observe(&mut backend, &values).unwrap();
    assert_eq!(replay.usage().captures, 2);
    let skipped = replay.take_shared_step().unwrap().unwrap();
    assert!(matches!(
        skipped.as_ref().records[0].outcome,
        CaptureOutcome::Skipped { .. }
    ));
    assert_eq!(backend.calls, 2);
    assert!(
        replay.observe(&mut backend, &values).is_err(),
        "a drained bank cannot replay its phase"
    );
    assert_eq!(backend.calls, 2);
    let foreign = SharedCapturePlan::new(source.admission().clone());
    let charged = pool.used_bytes().unwrap();
    assert!(matches!(
        request
            .reserve_numerical(
                numerical(&foreign, 1),
                &[SpeculativeNumericalSource::Model(&model)]
            )
            .unwrap_err()
            .cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.used_bytes().unwrap(), charged);
    // Exact-plan refusal consumes the accepted occurrence and retains its charge.
    let phase = request
        .reserve_numerical(
            numerical(&source, 1),
            &[SpeculativeNumericalSource::Model(&model)],
        )
        .unwrap();
    assert_eq!(phase.ordinal(), 3);
    let failed = phase.begin_with_capture(host(&foreign, 1)).unwrap_err();
    assert!(matches!(
        failed.cause(),
        CaptureRunHostError::Memory(WorkingMemoryError::IdentityMismatch)
    ));
    drop(failed);
    request.close().unwrap();
    drop((
        first,
        later,
        replay,
        skipped,
        first_budget,
        later_budget,
        replay_budget,
        model,
        role,
        request,
        source,
    ));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(escaped);
    assert!(
        pool.used_bytes().unwrap() > 0,
        "shared frame alias still owns its original H"
    );
    drop(snapshot_alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_capture_source_copies_before_birth_and_shared_alias_retains_its_charge() {
    let caller = capture_source();
    let plan = PreparedCapturePlanCopy::inspect(caller.admission()).unwrap();
    let required = WorkingMemoryPool::capture_source_required_bytes(&plan).unwrap();
    let too_small = WorkingMemoryPool::new(required - 1, 0).unwrap();
    assert!(too_small.compile_capture_source(plan).is_err());
    assert_eq!(too_small.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(required, 0).unwrap();
    let source = pool
        .compile_capture_source(PreparedCapturePlanCopy::inspect(caller.admission()).unwrap())
        .unwrap();
    assert!(!source.plan().same_storage(&caller));
    assert_eq!(
        source.plan().admission().identity(),
        caller.admission().identity()
    );
    assert_eq!(source.plan().admission().plan(), caller.admission().plan());
    assert_eq!(
        source.plan().admission().points(),
        caller.admission().points()
    );
    assert!(source.validate_pool(&too_small).is_err());
    assert_eq!(pool.used_bytes().unwrap(), required);
    let escaped = source.plan().clone();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), required);
    assert_eq!(escaped.admission().plan().selections[0].id, "row");
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn numerical_capture_failure_evidence_preserves_abort_spending_and_escaped_custody() {
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
    let pass = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        plan.workspace_geometry(2, pass, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &InferenceExecutionIdentity::default(),
        &plan,
        1 << 24,
    )
    .unwrap();
    let mut cursor = plan.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, pass).unwrap(),
            requirements(report.span_workspace_plan()),
        )
        .unwrap();
    let model = role.budget_custody();
    let source = capture_source();
    let (budget, mut invocation) = invoke(&request, &source, &model, 1);
    let mut backend = HostBackend {
        calls: 0,
        refuse: true,
    };
    assert!(
        invocation
            .observe(&mut backend, &[0.5, -1.0, 2.0, 0.0])
            .is_err()
    );
    assert_eq!(backend.calls, 1);
    let frame = invocation.take_failed_evidence().unwrap().unwrap();
    assert_eq!(frame.as_ref().outcome, CaptureStepOutcome::Aborted);
    assert_eq!(frame.as_ref().cumulative_usage.captures, 1);
    assert!(invocation.take_failed_evidence().unwrap().is_none());
    request.close().unwrap();
    drop((invocation, budget, model, role, request, source));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(frame);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod readouts;

mod interventions;
