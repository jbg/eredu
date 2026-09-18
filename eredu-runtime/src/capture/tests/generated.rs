use super::*;

#[derive(Default)]
struct TypedBackend {
    transforms: usize,
}
impl CaptureBackend for TypedBackend {
    type Tensor = (Vec<u64>, TensorDtype);
    type Error = std::io::Error;
    fn shape(&self, value: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
        Ok(value.0.clone())
    }
    fn source_dtype(&self, value: &Self::Tensor) -> Option<TensorDtype> {
        Some(value.1.clone())
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 128,
            host_bytes: 2048,
            encoded_bytes: 8192,
        })
    }
    fn transform(
        &mut self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.transforms += 1;
        Ok(CapturePayload::Tensor(
            TensorObservation::new(
                slice.shape.iter().map(|value| *value as usize).collect(),
                TensorObservationData::F32(vec![1.25; elements(&slice.shape).unwrap() as usize]),
            )
            .unwrap(),
        ))
    }
}

#[test]
fn declared_empty_sources_preserve_precision_without_running_factories() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 2 },
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-1., 0., 1.],
        },
    ] {
        for dtype in [TensorDtype::F32, TensorDtype::U64, TensorDtype::Bool] {
            let (mut plan, catalog, support, caps) = fixture(transform.clone());
            plan.selections[0].slices.push(CaptureSlice {
                axis: "hidden".into(),
                start: 2,
                end: 2,
                stride: 1,
            });
            let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(admit(plan, &catalog, &support, &caps).unwrap()));
            let mut backend = TypedBackend::default();
            let source = GeneratedCaptureSource {
                creation_bytes: 4096,
                source_dtype: Some(dtype.clone()),
            };
            session.begin_step(CapturePhase::Prefill, 0).unwrap();
            session
                .observe_generated(
                    &mut backend,
                    "block.output",
                    &(vec![3, 4], TensorDtype::Bf16),
                    &source,
                    &mut || -> Result<_, CaptureExecutionError<std::io::Error>> {
                        panic!("empty source factory")
                    },
                    &|error| error,
                )
                .unwrap();
            let step = session.take_step().unwrap();
            let record = &step.records[0];
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            assert_eq!(record.source_dtype.as_ref(), Some(&dtype));
            assert_eq!(record.selected_shape.as_deref(), Some([3, 0].as_slice()));
            assert_eq!(backend.transforms, 0);
            assert!(step.cumulative_usage.retained_bytes >= source.creation_bytes);
            match record.payload.as_ref().unwrap() {
                CapturePayload::Tensor(value) => {
                    assert_eq!(
                        value.shape(),
                        if matches!(transform, CaptureTransform::Preview { .. }) {
                            &[0][..]
                        } else {
                            &[3, 0][..]
                        }
                    );
                    assert_eq!(
                        value.data(),
                        &match dtype {
                            TensorDtype::F32 => TensorObservationData::F32(vec![]),
                            TensorDtype::U64 => TensorObservationData::U64(vec![]),
                            TensorDtype::Bool => TensorObservationData::Bool(vec![]),
                            _ => unreachable!(),
                        }
                    );
                }
                CapturePayload::Summary(value) => {
                    assert_eq!(value.elements, 0);
                    assert_eq!(value.mean, None);
                }
                CapturePayload::Histogram(value) => assert_eq!(value.counts, [0, 0]),
                _ => panic!("empty transform"),
            }
        }
    }
}

#[test]
fn generated_contract_mismatch_fails_before_transform_and_keeps_reserved_cost() {
    for (actual, actual_shape) in [
        (TensorDtype::F32, vec![3, 4]),
        (TensorDtype::Bf16, vec![3, 4]),
        (TensorDtype::Bf16, vec![3, 5]),
    ] {
        let (plan, catalog, support, caps) = fixture(CaptureTransform::FullTensor);
        let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(admit(plan, &catalog, &support, &caps).unwrap()));
        let mut backend = TypedBackend::default();
        let source = GeneratedCaptureSource {
            creation_bytes: 4096,
            source_dtype: Some(TensorDtype::F32),
        };
        let mut calls = 0;
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        let result = session.observe_generated(
            &mut backend,
            "block.output",
            &(vec![3, 4], TensorDtype::U64),
            &source,
            &mut || {
                calls += 1;
                Ok((actual_shape.clone(), actual.clone()))
            },
            &|error| error,
        );
        let valid = actual == TensorDtype::F32 && actual_shape == [3, 4];
        assert_eq!(result.is_ok(), valid);
        if !valid {
            assert!(
                matches!(result, Err(CaptureExecutionError::Admission(CaptureError::Invalid(message))) if message.contains(if actual_shape == [3, 4] { "element type" } else { "geometry" }))
            );
        }
        assert_eq!(calls, 1);
        assert_eq!(backend.transforms, usize::from(valid));
        let record = session.take_step().unwrap().records.remove(0);
        assert_eq!(record.source_dtype, Some(actual));
        assert!(record.charged.retained_bytes >= 4096);
        assert_eq!(record.payload.is_some(), valid);
        assert_eq!(
            matches!(
                record.outcome,
                CaptureOutcome::Failed {
                    reason: CaptureFailureReason::Invalid,
                    ..
                }
            ),
            !valid
        );
    }
}

#[test]
fn unknown_empty_raw_source_is_unsupported_without_factory_work() {
    let (mut plan, catalog, support, caps) = fixture(CaptureTransform::Slice);
    plan.selections[0].slices.push(CaptureSlice {
        axis: "hidden".into(),
        start: 0,
        end: 0,
        stride: 1,
    });
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(admit(plan, &catalog, &support, &caps).unwrap()));
    let mut backend = TypedBackend::default();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let result = session.observe_generated(
        &mut backend,
        "block.output",
        &(vec![3, 4], TensorDtype::F32),
        &generated_source(4096),
        &mut || -> Result<_, CaptureExecutionError<std::io::Error>> {
            panic!("unknown empty factory")
        },
        &|error| error,
    );
    assert!(matches!(
        result,
        Err(CaptureExecutionError::Admission(CaptureError::Unsupported(
            _
        )))
    ));
    assert_eq!(backend.transforms, 0);
    let record = session.take_step().unwrap().records.remove(0);
    assert_eq!(
        record.source_dtype, None,
        "prototype precision is not source evidence"
    );
    assert!(record.payload.is_none());
    assert!(matches!(
        record.outcome,
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Unsupported,
            ..
        }
    ));
}
