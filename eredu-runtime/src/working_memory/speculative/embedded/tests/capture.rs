mod lineage;
mod source;
mod usage;
use super::*;
use crate::capture::{
    FundedCaptureError, FundedEmbeddedCaptureInvocation, ScheduledCaptureBackend,
};
use eredu_core::capture::*;
use eredu_core::{
    DescriptionCompleteness, ObservationCatalog, ObservationDtype, ObservationPoint,
    ObservationPosition, ObservationRequirement, ObservationSupport, ObservationSupportReport,
    ObservationSupportStatus, ObservationValueType, SpeculativeRequestId, SymbolicDimension,
    TensorAxis, TensorObservationData,
};

fn capture_source(pool: &WorkingMemoryPool) -> OriginalCaptureSource {
    capture_source_with_sequence(pool, 3)
}
fn capture_source_with_sequence(
    pool: &WorkingMemoryPool,
    max_sequence: u64,
) -> OriginalCaptureSource {
    capture_source_with_transform(pool, max_sequence, CaptureTransform::FullTensor, 1)
}
fn capture_source_with_transform(
    pool: &WorkingMemoryPool,
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
            physical_native_bytes: None,
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
            physical_native_limit: false,
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
}
struct Backend {
    custody: OriginalSpeculativeBudgetCustody,
    calls: usize,
}
impl ScheduledCaptureBackend for Backend {
    type Tensor = [f32; 6];
    type Error = Failure;
    fn validate_source(
        &self,
        _: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<eredu_core::checkpoint::TensorDtype, Failure> {
        assert_eq!(geometry.source_shape(), &[3, 2]);
        Ok(eredu_core::checkpoint::TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 24,
            host_bytes: 24,
            encoded_bytes: 512,
        })
    }
    fn transform(
        &mut self,
        values: &Self::Tensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Failure> {
        claim.validate_model_custody(&self.custody)?;
        self.calls += 1;
        let mut tensor = claim.prepare()?;
        for &value in values {
            tensor.push_f32(value).unwrap();
        }
        Ok(tensor.finish().unwrap())
    }
}
fn forward(
    owner: &mut FundedEmbeddedCaptureInvocation,
    backend: &mut Backend,
    epoch: u64,
) -> Result<(), FundedCaptureError<Failure>> {
    let epoch = eredu_core::DistributedCommitEpoch::new(epoch).unwrap();
    owner
        .with_observer(backend, &|cause| cause, |observer| {
            let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
            guard
                .observer
                .prepare_transaction(epoch, crate::ExpertPass::Prefill)?;
            guard
                .observer
                .observe("block.output", &[0.5, -1.0, 2.25, 3.5, -4.0, 5.75])?;
            guard.observer.complete_transaction(epoch)?;
            guard.finish(true);
            Ok(())
        })
        .unwrap()
}
#[test]
fn model_capture_roles_keep_exact_source_and_cumulative_usage_through_repeated_coordinates() {
    let selected = selected(SpeculativeStrategyClass::EmbeddedSequential);
    let schedule = plan(&selected, 3, true);
    let (invocation, _) = schedule.prefill_invocations(0).unwrap();
    let workspace = EmbeddedInvocationWorkspace::target(invocation).unwrap();
    let report = report(workspace.geometry());
    let capacity = 1 << 26;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let source = capture_source(&pool);
    let source_baseline = pool.used_bytes().unwrap();
    let request = OriginalSpeculativeRequest::prepare_embedded(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        capacity,
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let shape = CaptureInvocationShape {
        batch: 1,
        sequence: 3,
        context: None,
    };
    let origin = eredu_core::speculative::SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(19),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [0; 32],
        optimistic: false,
    };
    let selected_rows = [true];
    let make = || {
        EmbeddedCaptureHostPlan::prepare(
            &source,
            CaptureRunHostPlan::prepare_invocation(
                source.plan(),
                CapturePhase::Prefill,
                0,
                shape,
                &selected_rows,
            )
            .unwrap(),
            workspace,
            origin,
        )
        .unwrap()
    };
    let (role, pending) = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            make(),
        )
        .unwrap();
    let mut owner = pending.begin().unwrap();
    assert!(owner.source().same_source(&source));
    assert!(owner.role().same_role(&role));
    let mut backend = Backend {
        custody: role.budget_custody(),
        calls: 0,
    };
    forward(&mut owner, &mut backend, 1).unwrap();
    let first = owner.take_shared_step().unwrap().unwrap();
    assert_eq!(first.as_ref().outcome, CaptureStepOutcome::Committed);
    assert_eq!(owner.usage().captures, 1);
    assert_eq!(backend.calls, 1);
    let (second_role, pending) = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            make(),
        )
        .unwrap();
    assert!(!second_role.same_role(&role));
    let mut second = pending.begin().unwrap();
    let before = pool.used_bytes().unwrap();
    let mut second_backend = Backend {
        custody: second_role.budget_custody(),
        calls: 0,
    };
    assert!(forward(&mut second, &mut second_backend, 2).is_err());
    assert_eq!(second_backend.calls, 0);
    assert_eq!(second.usage().captures, 1);
    assert_eq!(pool.used_bytes().unwrap(), before);
    let failed = second.take_shared_step().unwrap().unwrap();
    assert_eq!(failed.as_ref().outcome, CaptureStepOutcome::Aborted);
    let foreign = capture_source(&pool);
    let foreign_plan = EmbeddedCaptureHostPlan::prepare(
        &foreign,
        CaptureRunHostPlan::prepare_invocation(
            foreign.plan(),
            CapturePhase::Prefill,
            0,
            shape,
            &selected_rows,
        )
        .unwrap(),
        workspace,
        origin,
    )
    .unwrap();
    let before_refusal = pool.used_bytes().unwrap();
    let rejected = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            foreign_plan,
        )
        .unwrap_err();
    assert!(matches!(
        rejected.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.used_bytes().unwrap(), before_refusal);
    drop(rejected);
    drop(foreign);
    let CapturePayload::SharedTensor(payload) = first.records()[0].payload.as_ref().unwrap() else {
        panic!("shared tensor")
    };
    let payload = payload.clone();
    request.close().unwrap();
    drop((
        first,
        failed,
        owner,
        second,
        role,
        second_role,
        backend,
        second_backend,
        request,
    ));
    assert!(pool.used_bytes().unwrap() > source_baseline);
    let TensorObservationData::F32(values) = payload.data() else {
        panic!("F32 output")
    };
    assert_eq!(values, &[0.5, -1.0, 2.25, 3.5, -4.0, 5.75]);
    drop(payload);
    assert_eq!(pool.used_bytes().unwrap(), source_baseline);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod interventions;
