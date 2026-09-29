use super::*;

struct Rows;
impl CaptureBackend for Rows {
    type Tensor = (u64, u64);
    type Error = Infallible;
    fn shape(&self, tensor: &Self::Tensor) -> Result<Vec<u64>, Infallible> {
        Ok(vec![tensor.1, 4])
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 0,
            host_bytes: elements(&slice.shape)? * 8,
            encoded_bytes: 2048,
        })
    }
    fn transform(
        &mut self,
        tensor: &Self::Tensor,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Infallible> {
        let values = (slice.starts[0]..slice.ends[0])
            .step_by(slice.strides[0] as usize)
            .flat_map(|row| (0..4).map(move |column| (tensor.0 + row) * 10 + column))
            .collect();
        Ok(CapturePayload::Tensor(
            TensorObservation::new(
                slice.shape.iter().map(|v| *v as usize).collect(),
                TensorObservationData::U64(values),
            )
            .unwrap(),
        ))
    }
}

fn setup() -> (CaptureSession, CaptureDiscovery) {
    let (mut plan, catalog, mut support, capabilities) = fixture(CaptureTransform::Slice);
    support.capture = capabilities.clone();
    plan.selections[0].schedule.decode = false;
    plan.selections[0].slices.push(CaptureSlice {
        axis: "sequence".into(),
        start: 2,
        end: 9,
        stride: 3,
    });
    let admitted = plan
        .admit(
            &catalog,
            &support,
            &capabilities,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 9,
                max_predictions: 4,
            },
        )
        .unwrap();
    (
        CaptureSession::new(admitted),
        CaptureDiscovery {
            artifact_identity: "prefill-fixture".into(),
            catalog,
            support,
        },
    )
}

fn advance(session: &mut CaptureSession, start: u64, end: u64) -> CapturedStep {
    let span = CapturePrefillSpan {
        start,
        end,
        total: 9,
    };
    session.set_prefill_span(span).unwrap();
    assert_eq!(session.plan().request().prompt_tokens, 9);
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session
        .observe(&mut Rows, "block.output", &(start, end - start))
        .unwrap();
    let step = session.take_step().unwrap();
    assert_eq!(step.prefill_span, Some(span));
    assert!(session.plan().prefill_span().is_none());
    step
}

#[test]
fn prefill_spans_preserve_absolute_stride_and_replay_without_refunding_capture() {
    let (mut session, discovery) = setup();
    let identity = session.plan().identity().to_owned();
    let first = advance(&mut session, 0, 2);
    assert_eq!(
        first.records[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked
        }
    );
    assert!(first.records[0].payload.is_none());
    let middle = advance(&mut session, 2, 6);
    let Some(CapturePayload::Tensor(values)) = &middle.records[0].payload else {
        panic!("captured middle");
    };
    assert_eq!(
        values.data(),
        &TensorObservationData::U64(vec![20, 21, 22, 23, 50, 51, 52, 53])
    );
    assert_eq!(middle.records[0].source_shape, Some(vec![4, 4]));
    let saved = session.checkpoint(&discovery).unwrap();
    assert_eq!(
        saved.next_prediction(),
        0,
        "prefill has not produced its first token"
    );
    let last = advance(&mut session, 6, 9);
    let Some(CapturePayload::Tensor(values)) = &last.records[0].payload else {
        panic!("captured tail");
    };
    assert_eq!(
        values.data(),
        &TensorObservationData::U64(vec![80, 81, 82, 83])
    );
    assert_eq!(session.checkpoint(&discovery).unwrap().next_prediction(), 1);
    let spent = session.cumulative_usage();
    session.restore(&saved).unwrap();
    assert_eq!(session.cumulative_usage(), spent);
    let replay = advance(&mut session, 6, 9);
    assert_eq!(replay.records, last.records);
    assert_eq!(replay.prefill_span, last.prefill_span);
    assert_eq!(
        replay.cumulative_usage,
        spent.checked_add(replay.step_usage).unwrap()
    );
    assert_eq!(session.plan().identity(), identity);
    let mut child = saved
        .fork(
            CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 4,
                limits: session.plan().plan().limits.clone(),
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default()),
        )
        .unwrap();
    assert_eq!(advance(&mut child, 6, 9).records, last.records);
}

#[test]
fn prefill_span_rejects_wrong_geometry_and_requires_drained_boundaries() {
    let (mut session, _) = setup();
    for span in [
        CapturePrefillSpan {
            start: 0,
            end: 0,
            total: 9,
        },
        CapturePrefillSpan {
            start: 0,
            end: 10,
            total: 9,
        },
        CapturePrefillSpan {
            start: 0,
            end: 2,
            total: 10,
        },
    ] {
        assert!(session.set_prefill_span(span).is_err());
    }
    let span = CapturePrefillSpan {
        start: 0,
        end: 2,
        total: 9,
    };
    session.set_prefill_span(span).unwrap();
    assert!(session.set_prefill_span(span).is_err());
    assert!(session.begin_step(CapturePhase::Decode, 1).is_err());
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session.set_prefill_span(span).is_err());
    let step = session.take_step().unwrap();
    assert_eq!(step.prefill_span, Some(span));
    session
        .set_prefill_span(CapturePrefillSpan {
            start: 2,
            end: 6,
            total: 9,
        })
        .unwrap();
}

#[test]
fn aborted_prefill_span_restores_authority_without_refunding_or_committing_progress() {
    let (mut session, discovery) = setup();
    let initial = session.checkpoint(&discovery).unwrap();
    let identity = session.plan().identity().to_owned();
    let span = CapturePrefillSpan {
        start: 2,
        end: 6,
        total: 9,
    };
    session.set_prefill_span(span).unwrap();
    let epoch = DistributedCommitEpoch::FIRST;
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    session.observe(&mut Rows, "block.output", &(2, 4)).unwrap();
    assert!(session.take_step().is_none());
    assert_eq!(session.plan().prefill_span(), Some(span));
    let spent = session.cumulative_usage();
    session.finish_transaction(epoch, false);
    let failed = session.take_step().unwrap();
    assert_eq!(failed.outcome, CaptureStepOutcome::Aborted);
    assert_eq!(failed.prefill_span, Some(span));
    assert_eq!(session.plan().identity(), identity);
    assert!(session.checkpoint(&discovery).is_err());
    session.restore(&initial).unwrap();
    assert_eq!(session.cumulative_usage(), spent);
    assert_eq!(session.checkpoint(&discovery).unwrap().next_prediction(), 0);
}

#[test]
fn generated_prediction_scores_wait_for_the_final_prefill_chunk() {
    for transform in [
        CaptureTransform::TokenScores {
            token_ids: vec![1, 2],
        },
        CaptureTransform::TopCandidates { count: 2 },
    ] {
        let (mut plan, mut catalog, mut support, mut capabilities) = fixture(transform);
        let point = &mut catalog.points[0];
        point.path = MODEL_LOGITS_OBSERVATION_PATH.into();
        point.axes.as_mut().unwrap()[1].name = "vocabulary".into();
        plan.selections[0].path = point.path.clone();
        support.points[0].path = point.path.clone();
        capabilities.transformations.extend([
            CaptureTransformKind::TokenScores,
            CaptureTransformKind::TopCandidates,
        ]);
        let mut session =
            CaptureSession::new(admit(plan, &catalog, &support, &capabilities).unwrap());
        let mut backend = ProbeBackend {
            copies: Cell::new(0),
        };
        session
            .set_prefill_span(CapturePrefillSpan {
                start: 0,
                end: 2,
                total: 3,
            })
            .unwrap();
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        session
            .observe(&mut backend, MODEL_LOGITS_OBSERVATION_PATH, &vec![2, 4])
            .unwrap();
        assert_eq!(
            session.take_step().unwrap().records[0].outcome,
            CaptureOutcome::Skipped {
                reason: CaptureSkipReason::NotInvoked
            }
        );
        assert_eq!(
            backend.copies.get(),
            0,
            "an incomplete prompt has no generated prediction"
        );
        session
            .set_prefill_span(CapturePrefillSpan {
                start: 2,
                end: 3,
                total: 3,
            })
            .unwrap();
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        assert_eq!(
            session.take_step().unwrap().records[0].outcome,
            CaptureOutcome::Missing,
            "the final chunk must supply the first generated prediction scores"
        );
    }
}
