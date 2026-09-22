use super::*;
use eredu_core::{checkpoint::TensorDtype, *};
use std::{cell::Cell, convert::Infallible};

fn setup(max_predictions: u64) -> (AdmittedCapturePlan, CaptureDiscovery) {
    let point = ObservationPoint {
        path: "attention.scores".into(),
        node_id: "attention".into(),
        meaning: "cached context".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "context".into(),
                dimension: SymbolicDimension::Context,
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::FullTensor],
        ..Default::default()
    };
    let discovery = CaptureDiscovery {
        artifact_identity: "cached-source".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            points: vec![point],
            completeness: DescriptionCompleteness::Complete,
        },
        support: ObservationSupportReport {
            schema_version: 1,
            capture: capabilities,
            points: vec![ObservationSupport {
                path: "attention.scores".into(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        },
    };
    let usage = CaptureUsage {
        captures: 100,
        retained_bytes: 1_000_000,
        host_bytes: 1_000_000,
        encoded_bytes: 1_000_000,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "actual-context".into(),
            path: "attention.scores".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit_with_text_origin(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions,
        },
        CaptureTextOrigin {
            cached_positions: 7,
        },
    )
    .unwrap();
    (plan, discovery)
}
#[derive(Default)]
struct Backend {
    copies: Cell<usize>,
}
impl CaptureBackend for Backend {
    type Tensor = Vec<u64>;
    type Error = Infallible;
    fn shape(&self, value: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
        Ok(value.clone())
    }
    fn source_dtype(&self, _: &Self::Tensor) -> Option<TensorDtype> {
        Some(TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: elements(&slice.shape)? * 4,
            host_bytes: 64,
            encoded_bytes: 1024,
        })
    }
    fn transform(
        &mut self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.copies.set(self.copies.get() + 1);
        let count = elements(&slice.shape).unwrap() as usize;
        Ok(CapturePayload::Tensor(
            TensorObservation::new(
                slice.shape.iter().map(|n| *n as usize).collect(),
                TensorObservationData::F32((0..count).map(|i| i as f32 + 0.5).collect()),
            )
            .unwrap(),
        ))
    }
}
#[test]
fn existing_session_and_preflight_keep_cached_context_and_decode_one_origin() {
    let (plan, discovery) = setup(3);
    let mut estimates = vec![];
    validate_session(&plan, &discovery, |shape, _, _| {
        estimates.push(shape.to_vec());
        Ok(CaptureUsage::default())
    })
    .unwrap();
    assert_eq!(estimates, vec![vec![3, 10], vec![1, 12]]);
    let mut bad = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan.clone()));
    bad.begin_step(CapturePhase::Prefill, 0).unwrap();
    let mut backend = Backend::default();
    assert!(bad
        .observe(&mut backend, "attention.scores", &vec![3, 3])
        .is_err());
    assert_eq!(backend.copies.get(), 0);
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan));
    for (phase, prediction, shape) in [
        (CapturePhase::Prefill, 0, vec![3, 10]),
        (CapturePhase::Decode, 1, vec![1, 11]),
        (CapturePhase::Decode, 2, vec![1, 12]),
    ] {
        session.begin_step(phase, prediction).unwrap();
        session
            .observe(&mut backend, "attention.scores", &shape)
            .unwrap();
        let step = session.take_step().unwrap();
        assert_eq!(step.records[0].source_shape.as_ref(), Some(&shape));
        assert_eq!(step.prediction_index, prediction);
        assert!(matches!(step.records[0].outcome, CaptureOutcome::Captured));
    }
    assert_eq!(backend.copies.get(), 3);
    let (only_prefill, discovery) = setup(1);
    let mut estimates = vec![];
    validate_session(&only_prefill, &discovery, |shape, _, _| {
        estimates.push(shape.to_vec());
        Ok(CaptureUsage::default())
    })
    .unwrap();
    assert_eq!(estimates, vec![vec![3, 10]]);
}
#[test]
fn checkpoint_fork_readmits_origin_without_rebasing_prompt_or_local_schedule() {
    let (plan, discovery) = setup(3);
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan));
    let mut backend = Backend::default();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe(&mut backend, "attention.scores", &vec![3, 10])
        .unwrap();
    session.take_step().unwrap();
    let checkpoint = session.checkpoint(&discovery).unwrap();
    let spent = checkpoint.inherited_usage();
    let mut estimates = vec![];
    let mut child = checkpoint
        .fork(
            CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 5,
                limits: session.plan().plan().limits.clone(),
                intervention: None,
            },
            |shape, _, _| {
                estimates.push(shape.to_vec());
                Ok(CaptureUsage::default())
            },
        )
        .unwrap();
    assert_eq!(
        child.plan().text_origin(),
        Some(CaptureTextOrigin {
            cached_positions: 7
        })
    );
    assert_eq!(child.plan().request().prompt_tokens, 3);
    assert_eq!(child.plan().request().max_predictions, 5);
    assert_eq!(child.cumulative_usage(), spent);
    assert_eq!(estimates, vec![vec![1, 14]]);
    child.begin_step(CapturePhase::Decode, 1).unwrap();
    child
        .observe(&mut backend, "attention.scores", &vec![1, 11])
        .unwrap();
    let step = child.take_step().unwrap();
    assert_eq!(step.records[0].source_shape.as_deref(), Some(&[1, 11][..]));
    assert_eq!(step.prediction_index, 1);
    assert_eq!(
        step.cumulative_usage,
        spent.checked_add(step.step_usage).unwrap()
    );
    assert_eq!(session.plan().request().max_predictions, 3);
    session.restore(&checkpoint).unwrap();
    assert_eq!(session.plan().text_origin(), child.plan().text_origin());
}
