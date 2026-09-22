//! Genuine AR role/capture custody under the shared neutral host worker.
mod partition_interventions;
use super::*;
use crate::capture::{
    FundedAutoregressiveCaptureInvocation, FundedCaptureError, ScheduledCaptureBackend,
};
use eredu_core::capture::*;
use eredu_core::{
    DescriptionCompleteness, ObservationCatalog, ObservationDtype, ObservationPoint,
    ObservationPosition, ObservationRequirement, ObservationSupport, ObservationSupportReport,
    ObservationSupportStatus, ObservationValueType, SpeculativeRequestId, SymbolicDimension,
    TensorAxis, TensorObservationData,
};
fn capture_source(
    pool: &MemoryLedger,
    max_sequence: u64,
    transform: CaptureTransform,
    captures: u64,
) -> OriginalCaptureSource {
    let kind = transform.kind();
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "actual model output".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "value".into(),
            path: point.path.clone(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform,
        }],
        limits: CaptureLimits {
            per_step: unlimited,
            cumulative: CaptureUsage {
                captures,
                ..unlimited
            },
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit_invocations(
        &ObservationCatalog {
            schema_version: 1,
            points: vec![point.clone()],
            completeness: DescriptionCompleteness::Complete,
        },
        &ObservationSupportReport {
            schema_version: 1,
            capture: Default::default(),
            points: vec![ObservationSupport {
                path: point.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        },
        &CaptureCapabilities {
            transformations: vec![kind],
            max_histogram_bins: 0,
            conditions: vec![],
        },
        CaptureInvocationBounds {
            batch: 1,
            max_sequence,
            max_context: None,
            max_predictions: 5,
        },
    )
    .unwrap();
    pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&plan).unwrap())
        .unwrap()
}

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error("fixture observer failure")]
    Injected,
}
struct Values {
    data: Vec<f32>,
    rows: u64,
}
struct Backend {
    custody: OriginalSpeculativeBudgetCustody,
    calls: usize,
}
impl ScheduledCaptureBackend for Backend {
    type Tensor = Values;
    type Error = Failure;
    fn validate_source(
        &self,
        value: &Values,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<eredu_core::checkpoint::TensorDtype, Failure> {
        assert_eq!(
            geometry.source_shape(),
            &[usize::try_from(value.rows).unwrap(), 2]
        );
        Ok(eredu_core::checkpoint::TensorDtype::F32)
    }
    fn estimate(
        &self,
        value: &Values,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: value.rows * 8,
            host_bytes: value.rows * 8,
            encoded_bytes: 512,
        })
    }
    fn transform(
        &mut self,
        value: &Values,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Failure> {
        claim.validate_model_custody(&self.custody)?;
        self.calls += 1;
        let mut output = claim.prepare()?;
        for &value in &value.data {
            output.push_f32(value).unwrap();
        }
        Ok(output.finish().unwrap())
    }
}
fn forward(
    owner: &mut FundedAutoregressiveCaptureInvocation,
    backend: &mut Backend,
    values: &Values,
    epoch: u64,
    fail: bool,
) -> Result<(), FundedCaptureError<Failure>> {
    let epoch = eredu_core::DistributedCommitEpoch::new(epoch).unwrap();
    let pass = owner.role().invocation().execution_pass();
    owner
        .with_observer(backend, &|cause| cause, |observer| {
            let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
            guard.observer.prepare_transaction(epoch, pass)?;
            guard.observer.observe("block.output", values)?;
            if fail {
                return Err(FundedCaptureError::Backend(Failure::Injected));
            }
            guard.observer.complete_transaction(epoch)?;
            guard.finish(true);
            Ok(())
        })
        .unwrap()
}
fn origin() -> eredu_core::speculative::SpeculativeActivationOrigin {
    eredu_core::speculative::SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(37),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [1; 32],
        optimistic: false,
    }
}
fn frames<'a>(
    source: &'a OriginalCaptureSource,
    lineage: &'a OriginalModelCaptureLineage,
    invocation: AutoregressiveInvocation,
    plan: &InferenceSpanWorkspacePlan,
    mask: &'a [bool],
) -> Vec<AutoregressiveCaptureFrameHostPlan<'a>> {
    let prefill = invocation.execution_pass() == crate::ExpertPass::Prefill;
    plan.records()
        .iter()
        .map(|row| {
            let InferenceWorkspaceSpan::Prefill(chunk) = row.span() else {
                panic!("actual input report")
            };
            let run = CaptureRunHostPlan::prepare_invocation_window(
                source.plan(),
                if prefill {
                    CapturePhase::Prefill
                } else {
                    CapturePhase::Decode
                },
                0,
                CaptureInvocationShape {
                    batch: 1,
                    sequence: chunk.input.end - chunk.input.start,
                    context: None,
                },
                mask,
                prefill.then_some(CaptureInvocationWindow {
                    logical_sequence: plan.geometry().input_positions,
                    start: chunk.input.start,
                }),
            )
            .unwrap();
            AutoregressiveCaptureFrameHostPlan::prepare(
                source,
                run,
                invocation,
                plan.geometry(),
                row.span(),
                origin(),
            )
            .unwrap()
            .with_lineage(lineage)
            .unwrap()
            .with_quoted_usage(CaptureUsage::default())
        })
        .collect()
}
#[test]
fn autoregressive_capture_exact_spans_share_usage_failures_and_retained_accounts() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(5).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let invocation =
        AutoregressiveInvocation::prefill(AutoregressivePass::TargetPrefill, 5).unwrap();
    let mut geometry = schedule
        .workspace_geometry(0, invocation, NonZeroU64::new(2).unwrap())
        .unwrap();
    // The actual full-tensor observation worker requests every new input row.
    geometry.output = eredu_core::OutputDemand::Sequence;
    let report = report(geometry);
    let capacity = 1 << 26;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let source = capture_source(&pool, 5, CaptureTransform::FullTensor, 2);
    let baseline = pool.payload_used_bytes().unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &execution,
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
    )
    .unwrap();
    assert!(request.prepare_embedded_capture_lineage(&source).is_err());
    let lineage = request.prepare_model_capture_lineage(&source).unwrap();
    let mask = [true];
    let plans = frames(
        &source,
        &lineage,
        invocation,
        report.span_workspace_plan(),
        &mask,
    )
    .into_iter()
    .map(|plan| plan.with_transport_metadata(17).unwrap())
    .collect::<Vec<_>>();
    assert_eq!(plans.len(), 3);
    let quoted = AutoregressiveCaptureHostPlan::prepare(&plans).unwrap();
    assert!(
        quoted.initialization_peak_bytes()
            > plans
                .iter()
                .map(|row| row.initialization_peak_bytes())
                .sum()
    );
    let foreign_request = OriginalSpeculativeRequest::prepare(
        &pool,
        &execution,
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let (role, prepared) = request
        .reserve_role_with_capture(
            cursor.claim(0, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
            quoted,
        )
        .unwrap();
    request.validate_model_role(&role).unwrap();
    assert_eq!(
        foreign_request.validate_model_role(&role),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    drop(foreign_request);
    let mut bank = prepared.construct().unwrap();
    role.begin_prefill(&execution, geometry).unwrap();
    let mut backend = Backend {
        custody: role.budget_custody(),
        calls: 0,
    };
    let mut retained = Vec::new();
    let mut retained_metadata = Vec::new();
    let mut replay_error = None;
    for (ordinal, row) in report.span_workspace_plan().records().iter().enumerate() {
        let InferenceWorkspaceSpan::Prefill(chunk) = row.span() else {
            unreachable!()
        };
        let span = role.claim_prefill_span(chunk).unwrap();
        let mut owner = bank.begin_prefill(&span).unwrap();
        let metadata = owner.take_partition_metadata().unwrap();
        assert!(owner.take_partition_metadata().is_none());
        let charged = pool.payload_used_bytes().unwrap();
        metadata.reserve_metadata(17).unwrap();
        assert!(metadata.reserve_metadata(1).is_err());
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            charged,
            "protocol metadata was admitted with the model role"
        );
        retained_metadata.push(metadata);
        assert!(owner.role().same_role(&role));
        replay_error = Some(bank.begin_prefill(&span).unwrap_err());
        let rows = chunk.input.end - chunk.input.start;
        let values = Values {
            rows,
            data: (0..rows * 2)
                .map(|i| i as f32 + ordinal as f32 + 0.5)
                .collect(),
        };
        let result = forward(
            &mut owner,
            &mut backend,
            &values,
            ordinal as u64 + 1,
            ordinal == 1,
        );
        if ordinal == 0 {
            result.unwrap();
            let frame = owner.take_shared_step().unwrap().unwrap();
            assert_eq!(frame.records().len(), 1);
            let CapturePayload::SharedTensor(payload) =
                frame.records()[0].payload.as_ref().unwrap()
            else {
                panic!("funded shared payload")
            };
            let TensorObservationData::F32(actual) = payload.data() else {
                panic!("F32 capture")
            };
            assert_eq!(actual, &values.data);
            retained.push(frame);
        } else if ordinal == 1 {
            assert!(result.is_err());
            retained.push(owner.take_failed_evidence().unwrap().unwrap());
        } else {
            assert!(
                result.is_err(),
                "cumulative limit survives failed prior frame"
            );
        }
    }
    assert_eq!(backend.calls, 2);
    assert_eq!(
        request
            .inspect_model_capture_lineage_usage(&source, &lineage)
            .unwrap()
            .captures,
        2
    );
    assert!(bank.begin_decode().is_err());
    drop(plans);
    drop(bank);
    drop(backend);
    request.close().unwrap();
    assert_eq!(
        request.validate_model_role(&role),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    drop(role);
    drop(request);
    drop(lineage);
    assert!(
        pool.payload_used_bytes().unwrap() > baseline,
        "escaped frames and refusal retain the role"
    );
    drop(retained);
    drop(retained_metadata);
    assert!(
        pool.payload_used_bytes().unwrap() > baseline,
        "typed refusal is the last role owner"
    );
    drop(replay_error);
    assert_eq!(pool.payload_used_bytes().unwrap(), baseline);
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn autoregressive_decode_capture_rejects_foreign_lineage_and_spends_one_frame() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let schedule = AutoregressiveSchedulePlan::new(
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
        schedule
            .workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let capacity = 1 << 25;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let source = capture_source(&pool, 2, CaptureTransform::FullTensor, 3);
    let execution = InferenceExecutionIdentity::default();
    let make_request = || {
        OriginalSpeculativeRequest::prepare(
            &pool,
            &execution,
            &schedule,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap()
    };
    let request = make_request();
    let other = make_request();
    let lineage = request.prepare_model_capture_lineage(&source).unwrap();
    let foreign = other.prepare_model_capture_lineage(&source).unwrap();
    let mask = [true];
    let wrong = frames(
        &source,
        &foreign,
        invocation,
        report.span_workspace_plan(),
        &mask,
    );
    let plans = frames(
        &source,
        &lineage,
        invocation,
        report.span_workspace_plan(),
        &mask,
    );
    let mut cursor = schedule.into_cursor();
    let before = pool.payload_used_bytes().unwrap();
    let error = request
        .reserve_role_with_capture(
            cursor.claim(2, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
            AutoregressiveCaptureHostPlan::prepare(&wrong).unwrap(),
        )
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), before);
    let (role, prepared) = request
        .reserve_role_with_capture(
            cursor.claim(2, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
            AutoregressiveCaptureHostPlan::prepare(&plans).unwrap(),
        )
        .unwrap();
    let mut bank = prepared.construct().unwrap();
    let mut owner = bank.begin_decode().unwrap();
    assert!(bank.begin_decode().is_err());
    let mut backend = Backend {
        custody: role.budget_custody(),
        calls: 0,
    };
    forward(
        &mut owner,
        &mut backend,
        &Values {
            rows: 1,
            data: vec![1.25, -2.5],
        },
        1,
        false,
    )
    .unwrap();
    let frame = owner.take_shared_step().unwrap().unwrap();
    assert_eq!(frame.records().len(), 1);
    assert_eq!(
        request
            .inspect_model_capture_lineage_usage(&source, &lineage)
            .unwrap()
            .captures,
        1
    );
    assert_eq!(
        other
            .inspect_model_capture_lineage_usage(&source, &foreign)
            .unwrap(),
        CaptureUsage::default()
    );
}

#[test]
fn autoregressive_prefill_readout_upgrade_requires_its_active_capture_source() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let source = capture_source(&pool, 3, CaptureTransform::FullTensor, 3);
    for active in [false, true] {
        let schedule = AutoregressiveSchedulePlan::new(
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
        let mut geometry = schedule
            .workspace_geometry(0, invocation, NonZeroU64::new(2).unwrap())
            .unwrap();
        geometry.output = eredu_core::OutputDemand::Sequence;
        let report = report(geometry);
        let request = OriginalSpeculativeRequest::prepare(
            &pool,
            &InferenceExecutionIdentity::default(),
            &schedule,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 26),
        )
        .unwrap();
        let lineage = request.prepare_model_capture_lineage(&source).unwrap();
        let mask = [active];
        let frames = frames(
            &source,
            &lineage,
            invocation,
            report.span_workspace_plan(),
            &mask,
        );
        let capture = AutoregressiveCaptureHostPlan::prepare(&frames).unwrap();
        assert_eq!(capture.requires_sequence_readout(), active);
        let mut cursor = schedule.into_cursor();
        let before = pool.payload_used_bytes().unwrap();
        let accepted = request.reserve_role_with_capture(
            cursor.claim(0, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
            capture,
        );
        if active {
            let (role, prepared) = accepted.unwrap();
            role.validate_plan(report.span_workspace_plan()).unwrap();
            drop(prepared.construct().unwrap());
        } else {
            assert!(matches!(
                accepted.unwrap_err().cause(),
                WorkingMemoryError::IdentityMismatch
            ));
            assert_eq!(pool.payload_used_bytes().unwrap(), before);
        }
    }
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

mod partition;
