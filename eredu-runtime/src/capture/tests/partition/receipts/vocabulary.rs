use super::*;

fn score_plan(transform: CaptureTransform, vocabulary: usize) -> AdmittedCapturePlan {
    let (mut plan, mut catalog, mut support, mut capabilities) = fixture(transform);
    capabilities.transformations.extend([
        CaptureTransformKind::TokenScores,
        CaptureTransformKind::TopCandidates,
    ]);
    plan.selections[0].path = MODEL_LOGITS_OBSERVATION_PATH.into();
    catalog.points[0].path = MODEL_LOGITS_OBSERVATION_PATH.into();
    let axis = &mut catalog.points[0].axes.as_mut().unwrap()[1];
    axis.name = "vocabulary".into();
    axis.dimension = SymbolicDimension::Known(vocabulary);
    support.points[0].path = MODEL_LOGITS_OBSERVATION_PATH.into();
    plan.limits.per_step.host_bytes = 32_000_000;
    plan.limits.cumulative.host_bytes = 32_000_000;
    admit(plan, &catalog, &support, &capabilities).unwrap()
}

fn score_authority(
    plan: &AdmittedCapturePlan,
    maps: &[ComponentCoordinateMap],
    ledger: &mut CaptureLedger,
) -> Result<PartitionCaptureReceiptPlan, PartitionCaptureMergeError> {
    let width = maps[0].global_count() as u64;
    let shape = [3, width];
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &shape).unwrap();
    PartitionCaptureReceiptPlan::new(
        plan.clone(),
        context(plan),
        maps.iter()
            .enumerate()
            .map(|(rank, map)| PartitionCaptureProducer {
                rank,
                projection: CaptureSlicePartition::new(&shape, &slice, 1, map, 16).unwrap(),
            })
            .collect(),
        4,
        PartitionCaptureReceiptLimits {
            max_producers: 4,
            max_fragments: 16,
            max_record_bytes: 16_384,
        },
        ledger,
    )
}

fn candidate(id: u32, score: f32) -> CaptureCandidate {
    CaptureCandidate {
        token_id: id,
        score,
        allowed: true,
    }
}

fn payload(transform: &CaptureTransform) -> CapturePayload {
    // Final row [4, 1, 4, -3]: tied maxima and a selected losing target.
    let log_mass = (2.0 + (-3.0f64).exp() + (-7.0f64).exp()).ln();
    match transform {
        CaptureTransform::TokenScores { .. } => CapturePayload::TokenScores(CaptureTokenScores {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: CandidateLogitsSource::Original,
            vocabulary: 4,
            log_partition: 4.0 + log_mass,
            scores: vec![
                CaptureTokenScore {
                    target: candidate(3, -3.0),
                    log_probability: -7.0 - log_mass,
                    rank: 4,
                    strongest_alternative: Some(candidate(0, 4.0)),
                },
                CaptureTokenScore {
                    target: candidate(0, 4.0),
                    log_probability: -log_mass,
                    rank: 1,
                    strongest_alternative: Some(candidate(2, 4.0)),
                },
            ],
            domain: None,
        }),
        CaptureTransform::TopCandidates { .. } => CapturePayload::Candidates(CaptureCandidates {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: CandidateLogitsSource::Original,
            candidates: vec![candidate(2, 4.0), candidate(0, 4.0), candidate(1, 1.0)],
            domain: None,
        }),
        _ => unreachable!(),
    }
}

struct ReducedBackend {
    calls: usize,
}
impl CaptureBackend for ReducedBackend {
    type Tensor = Vec<u64>;
    type Error = std::io::Error;
    fn shape(&self, shape: &Vec<u64>) -> Result<Vec<u64>, Self::Error> {
        Ok(shape.clone())
    }
    fn source_dtype(&self, _: &Vec<u64>) -> Option<TensorDtype> {
        Some(TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Vec<u64>,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 1024,
            host_bytes: 1024,
            encoded_bytes: 4096,
        })
    }
    fn transform(
        &mut self,
        _: &Vec<u64>,
        selection: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.calls += 1;
        Ok(payload(&selection.transform))
    }
}

fn encoded(plan: &AdmittedCapturePlan, receipt: &PartitionCaptureReceiptPlan) -> Vec<u8> {
    let mut backend = ReducedBackend { calls: 0 };
    let mut ledger = CaptureLedger::new(plan);
    let projection = receipt.producer(0).unwrap();
    let fragment = capture_fragment(
        &mut backend,
        &projection.local_shape().to_vec(),
        PartitionCaptureRequest {
            invocation: None,
            plan,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            projection,
            fragment_index: 0,
            producer_rank: 0,
        },
        &mut ledger,
    )
    .unwrap();
    assert_eq!(backend.calls, 1);
    receipt
        .encode_producer(0, Some(TensorDtype::F32), vec![fragment], &mut ledger)
        .unwrap()
}

#[test]
fn complete_vocabulary_receipts_preserve_reductions_and_provenance() {
    for transform in [
        CaptureTransform::TokenScores {
            token_ids: vec![3, 0],
        },
        CaptureTransform::TopCandidates { count: 3 },
    ] {
        let plan = score_plan(transform.clone(), 4);
        let maps = [ComponentCoordinateMap::range(4, 0..4).unwrap()];
        let mut ledger = CaptureLedger::new(&plan);
        let receipt = score_authority(&plan, &maps, &mut ledger).unwrap();
        let bytes = encoded(&plan, &receipt);
        let mut delivery = receipt.into_delivery();
        delivery.receive(0, &bytes, &mut ledger).unwrap();
        let capture = delivery.finish(&mut ledger).unwrap();
        assert_eq!(capture.context(), &context(&plan));
        assert_eq!(capture.producers(), [0]);
        assert_eq!(capture.capture().contributions().len(), 1);
        assert_eq!(capture.capture().record().selected_shape, Some(vec![3, 4]));
        assert_eq!(
            capture.capture().record().payload,
            Some(payload(&transform))
        );
    }
}

#[test]
fn vocabulary_admission_rejects_shards_permutations_and_replicate_export_before_work() {
    for transform in [
        CaptureTransform::TokenScores {
            token_ids: vec![3, 0],
        },
        CaptureTransform::TopCandidates { count: 3 },
    ] {
        let plan = score_plan(transform, 4);
        for maps in [
            vec![
                ComponentCoordinateMap::range(4, 0..2).unwrap(),
                ComponentCoordinateMap::range(4, 2..4).unwrap(),
            ],
            vec![ComponentCoordinateMap::indices(4, vec![1, 0, 2, 3]).unwrap()],
            vec![
                ComponentCoordinateMap::range(4, 0..4).unwrap(),
                ComponentCoordinateMap::range(4, 0..4).unwrap(),
            ],
        ] {
            let mut ledger = CaptureLedger::new(&plan);
            assert!(score_authority(&plan, &maps, &mut ledger).is_err());
            assert_eq!(ledger.total(), CaptureUsage::default());
        }
        let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 4]).unwrap();
        let projection = CaptureSlicePartition::new(
            &[3, 4],
            &slice,
            1,
            &ComponentCoordinateMap::range(4, 0..2).unwrap(),
            1,
        )
        .unwrap();
        let calls = Cell::new(0);
        let mut backend = ReducedBackend { calls: 0 };
        let mut ledger = CaptureLedger::new(&plan);
        let result = capture_generated_fragment(
            &mut backend,
            &vec![3, 2],
            &generated_source(128),
            &mut || {
                calls.set(calls.get() + 1);
                Ok::<_, String>(vec![3, 2])
            },
            PartitionCaptureRequest {
                invocation: None,
                plan: &plan,
                selection_index: 0,
                phase: CapturePhase::Prefill,
                prediction: 0,
                projection: &projection,
                fragment_index: 0,
                producer_rank: 0,
            },
            &mut ledger,
            &|error| error.to_string(),
        );
        assert!(result.is_err());
        assert_eq!((calls.get(), backend.calls), (0, 0));
        assert_eq!(ledger.total(), CaptureUsage::default());
    }
}

#[test]
fn vocabulary_receipts_reject_malformed_score_and_candidate_evidence() {
    for transform in [
        CaptureTransform::TokenScores {
            token_ids: vec![3, 0],
        },
        CaptureTransform::TopCandidates { count: 3 },
    ] {
        let plan = score_plan(transform, 4);
        let maps = [ComponentCoordinateMap::range(4, 0..4).unwrap()];
        let receipt = score_authority(&plan, &maps, &mut CaptureLedger::new(&plan)).unwrap();
        let original: PartitionCaptureProducerRecord =
            serde_json::from_slice(&encoded(&plan, &receipt)).unwrap();
        for mutation in 0..12 {
            let mut changed = original.clone();
            match changed.fragments[0].record.payload.as_mut().unwrap() {
                CapturePayload::TokenScores(value) => match mutation {
                    0 => value.scores.reverse(),
                    1 => value.vocabulary = 3,
                    2 => value.log_partition += 1.0,
                    3 => value.scores[0].rank = 0,
                    4 => value.scores[0].rank = 5,
                    5 => {
                        value.scores[0]
                            .strongest_alternative
                            .as_mut()
                            .unwrap()
                            .token_id = 3
                    }
                    6 => value.scores[0].strongest_alternative = None,
                    7 => value.source = CandidateLogitsSource::Effective,
                    8 => value.scores[0].log_probability = 0.1,
                    9 => value.scores[0].target.allowed = false,
                    10 => {
                        value.scores[0]
                            .strongest_alternative
                            .as_mut()
                            .unwrap()
                            .score = -4.0
                    }
                    _ => {
                        value.domain = Some(CandidateDomain {
                            vocabulary: 4,
                            allowed_tokens: 5,
                            constrained: false,
                        })
                    }
                },
                CapturePayload::Candidates(value) => match mutation {
                    0 => value.candidates.reverse(),
                    1 => value.candidates[0].token_id = 4,
                    2 => value.candidates[1].token_id = value.candidates[0].token_id,
                    3 => {
                        value.candidates.pop();
                    }
                    4 => value.candidates.push(candidate(3, -3.0)),
                    5 => value.candidates[2].score = 5.0,
                    6 => value.candidates[0].score = f32::INFINITY,
                    7 => value.source = CandidateLogitsSource::Effective,
                    8 => {
                        value.domain = Some(CandidateDomain {
                            vocabulary: 3,
                            allowed_tokens: 2,
                            constrained: false,
                        })
                    }
                    9 => value.candidates[0].allowed = false,
                    10 => {
                        value.domain = Some(CandidateDomain {
                            vocabulary: 4,
                            allowed_tokens: 0,
                            constrained: true,
                        })
                    }
                    _ => {
                        value.domain = Some(CandidateDomain {
                            vocabulary: 4,
                            allowed_tokens: 5,
                            constrained: false,
                        })
                    }
                },
                _ => unreachable!(),
            }
            let mut ledger = CaptureLedger::new(&plan);
            let mut delivery = score_authority(&plan, &maps, &mut ledger)
                .unwrap()
                .into_delivery();
            let before = ledger.total();
            assert!(
                delivery
                    .receive(0, &serde_json::to_vec(&changed).unwrap(), &mut ledger)
                    .is_err(),
                "mutation {mutation}"
            );
            assert!(ledger.total().host_bytes > before.host_bytes);
            assert_eq!(delivery.missing_producers().collect::<Vec<_>>(), [0]);
        }
    }
}

#[test]
fn vocabulary_delivery_reserves_bounded_payload_and_rejects_exhaustion() {
    let transform = CaptureTransform::TokenScores {
        token_ids: vec![3, 0],
    };
    let small = score_plan(transform.clone(), 4);
    let large = score_plan(transform, 1_000_000);
    let small_receipt = score_authority(
        &small,
        &[ComponentCoordinateMap::range(4, 0..4).unwrap()],
        &mut CaptureLedger::new(&small),
    )
    .unwrap();
    let large_receipt = score_authority(
        &large,
        &[ComponentCoordinateMap::range(1_000_000, 0..1_000_000).unwrap()],
        &mut CaptureLedger::new(&large),
    )
    .unwrap();
    assert_eq!(
        small_receipt.delivery_usage().unwrap(),
        large_receipt.delivery_usage().unwrap(),
        "host delivery prices bounded reductions, not a full logits export"
    );
    let bytes = encoded(&small, &small_receipt);
    let mut delivery = small_receipt.into_delivery();
    let mut ledger = CaptureLedger::new(&small);
    delivery.receive(0, &bytes, &mut ledger).unwrap();
    let mut empty = ledger.reserve_quota(CaptureUsage::default()).unwrap();
    assert!(matches!(
        delivery.finish(&mut empty),
        Err(PartitionCaptureMergeError::Capture(
            CaptureError::Limit { .. }
        ))
    ));

    let receipt = score_authority(
        &small,
        &[ComponentCoordinateMap::range(4, 0..4).unwrap()],
        &mut ledger,
    )
    .unwrap();
    let projection = receipt.producer(0).unwrap();
    let mut backend = ReducedBackend { calls: 0 };
    assert!(capture_fragment(
        &mut backend,
        &vec![3, 4],
        PartitionCaptureRequest {
            invocation: None,
            plan: &small,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            projection,
            fragment_index: 0,
            producer_rank: 0,
        },
        &mut empty
    )
    .is_err());
    assert_eq!(backend.calls, 0);
}

#[test]
fn vocabulary_receipts_preserve_stable_probabilities_at_extreme_equal_scores() {
    let plan = score_plan(
        CaptureTransform::TokenScores {
            token_ids: vec![3, 0],
        },
        4,
    );
    let maps = [ComponentCoordinateMap::range(4, 0..4).unwrap()];
    for maximum in [f32::MAX, -f32::MAX] {
        let mut ledger = CaptureLedger::new(&plan);
        let receipt = score_authority(&plan, &maps, &mut ledger).unwrap();
        let mut encoded: PartitionCaptureProducerRecord =
            serde_json::from_slice(&encoded(&plan, &receipt)).unwrap();
        let Some(CapturePayload::TokenScores(value)) = &mut encoded.fragments[0].record.payload
        else {
            panic!()
        };
        value.log_partition = f64::from(maximum) + 4.0f64.ln();
        for score in &mut value.scores {
            score.target.score = maximum;
            score.rank = 1;
            score.log_probability = -4.0f64.ln();
            score.strongest_alternative.as_mut().unwrap().score = maximum;
        }
        let mut delivery = receipt.into_delivery();
        delivery
            .receive(0, &serde_json::to_vec(&encoded).unwrap(), &mut ledger)
            .unwrap();
        let capture = delivery.finish(&mut ledger).unwrap();
        let Some(CapturePayload::TokenScores(value)) = &capture.capture().record().payload else {
            panic!()
        };
        assert_eq!(value.scores[0].log_probability, -4.0f64.ln());
    }
}

#[test]
fn vocabulary_receipts_reject_impossible_probabilities_when_log_partition_rounds() {
    let plan = score_plan(
        CaptureTransform::TokenScores {
            token_ids: vec![3, 0],
        },
        4,
    );
    let maps = [ComponentCoordinateMap::range(4, 0..4).unwrap()];
    for maximum in [f32::MAX, -f32::MAX, 1e20, -1e20] {
        for probability in [0.0, -0.1, -4.0f64.ln() - 0.01, -1000.0] {
            let mut ledger = CaptureLedger::new(&plan);
            let receipt = score_authority(&plan, &maps, &mut ledger).unwrap();
            let mut record: PartitionCaptureProducerRecord =
                serde_json::from_slice(&encoded(&plan, &receipt)).unwrap();
            let Some(CapturePayload::TokenScores(value)) = &mut record.fragments[0].record.payload
            else {
                panic!()
            };
            value.log_partition = f64::from(maximum);
            for score in &mut value.scores {
                score.target.score = maximum;
                score.rank = 1;
                score.log_probability = probability;
                score.strongest_alternative.as_mut().unwrap().score = maximum;
            }
            // Both equal maxima must contribute to the denominator. A selected
            // maximum lies between -ln(vocabulary) and -ln(2), irrespective of
            // precision lost by adding log mass to the unshifted maximum.
            let mut delivery = receipt.into_delivery();
            assert!(
                delivery
                    .receive(0, &serde_json::to_vec(&record).unwrap(), &mut ledger)
                    .is_err(),
                "accepted {probability} for tied maximum {maximum}"
            );
            assert!(delivery.finish(&mut ledger).is_err());
        }
    }
}

#[test]
fn vocabulary_receipts_preserve_probabilities_across_score_gaps() {
    let plan = score_plan(
        CaptureTransform::TokenScores {
            token_ids: vec![3, 0],
        },
        4,
    );
    let maps = [ComponentCoordinateMap::range(4, 0..4).unwrap()];
    for row in [
        [7.0f32, 1.0, -8.0, 4.0],
        [16_777_216.0, 16_777_215.0, 16_777_214.0, 16_777_213.0],
        [-f32::MAX, 0.0, 1.0, f32::MAX],
        [f32::MAX, -f32::MAX, 0.0, f32::MAX],
        [0.0, f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE],
    ] {
        let mut ledger = CaptureLedger::new(&plan);
        let receipt = score_authority(&plan, &maps, &mut ledger).unwrap();
        let mut record: PartitionCaptureProducerRecord =
            serde_json::from_slice(&encoded(&plan, &receipt)).unwrap();
        let Some(CapturePayload::TokenScores(value)) = &mut record.fragments[0].record.payload
        else {
            panic!()
        };
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mass = row
            .iter()
            .map(|x| (f64::from(*x) - maximum).exp())
            .sum::<f64>()
            .ln();
        value.log_partition = maximum + mass;
        for score in &mut value.scores {
            let id = score.target.token_id as usize;
            score.target.score = row[id];
            score.rank = 1 + row.iter().filter(|x| **x > row[id]).count() as u64;
            score.log_probability = (f64::from(row[id]) - maximum) - mass;
            let (other, &other_score) = row
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != id)
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap();
            score.strongest_alternative = Some(candidate(other as u32, other_score));
        }
        let expected = value.clone();
        let mut delivery = receipt.into_delivery();
        delivery
            .receive(0, &serde_json::to_vec(&record).unwrap(), &mut ledger)
            .unwrap();
        let capture = delivery.finish(&mut ledger).unwrap();
        let Some(CapturePayload::TokenScores(actual)) = &capture.capture().record().payload else {
            panic!()
        };
        assert_eq!(actual.stage, expected.stage);
        assert_eq!(actual.source, expected.source);
        assert_eq!(actual.vocabulary, expected.vocabulary);
        assert_eq!(actual.domain, expected.domain);
        let close = |actual: f64, expected: f64| {
            // JSON decoding can round the serialized decimal by one F64 ulp.
            assert!((actual - expected).abs() <= 4.0 * f64::EPSILON * expected.abs().max(1.0));
        };
        close(actual.log_partition, expected.log_partition);
        assert_eq!(actual.scores.len(), expected.scores.len());
        for (actual, expected) in actual.scores.iter().zip(&expected.scores) {
            assert_eq!(actual.target, expected.target);
            assert_eq!(actual.rank, expected.rank);
            assert_eq!(actual.strongest_alternative, expected.strongest_alternative);
            close(actual.log_probability, expected.log_probability);
        }
    }
}
