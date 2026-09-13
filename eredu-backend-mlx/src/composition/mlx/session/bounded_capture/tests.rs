use super::*;

fn verify_native_partition_capture_fragments(
    device: safemlx::DeviceType,
    transform: CaptureTransform,
    native_exchange: bool,
) {
    use eredu_core::{component::ComponentCoordinateMap, *};
    use eredu_runtime::capture::partition::{
        capture_fragment, PartitionCaptureExchange, PartitionCaptureProducer,
        PartitionCaptureReceiptLimits, PartitionCaptureReceiptPlan, PartitionCaptureRequest,
    };
    let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
    let transport = native_exchange.then(|| {
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let manifest = eredu_runtime::CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(2),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        crate::backend::distributed::MlxDistributedSession::from_manifest(
            &manifest, &world, &stream,
        )
        .unwrap()
    });
    let point = ObservationPoint {
        path: "test".into(),
        node_id: "component".into(),
        meaning: "native component capture".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "component".into(),
                dimension: SymbolicDimension::Known(20),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let mut selected = selection(transform.clone());
    selected.schedule.decode = false;
    selected.slices = vec![
        CaptureSlice {
            axis: "sequence".into(),
            start: 0,
            end: 3,
            stride: 2,
        },
        CaptureSlice {
            axis: "component".into(),
            start: 1,
            end: 20,
            stride: 3,
        },
    ];
    let usage = CaptureUsage {
        captures: 100,
        retained_bytes: 1_000_000,
        host_bytes: 4_000_000,
        encoded_bytes: 1_000_000,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: capabilities(),
        points: vec![ObservationSupport {
            path: "test".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![selected],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &catalog,
        &support,
        &capabilities(),
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 1,
        },
    )
    .unwrap();
    let mut global: Vec<f32> = (0..60).map(|index| (index as f32 - 30.0) * 0.25).collect();
    if !matches!(transform, CaptureTransform::Slice) {
        global[1] = f32::NAN;
        global[4] = f32::INFINITY;
        global[7] = f32::NEG_INFINITY;
    }
    let expected: Vec<f32> = [0, 2]
        .into_iter()
        .flat_map(|row| {
            (1..20)
                .step_by(3)
                .map(move |column| (row * 20 + column) as usize)
        })
        .map(|index| global[index])
        .collect();
    let slice = resolve_slice(&catalog.points[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    for bf16 in [false, true] {
        for maps in [
            vec![
                ComponentCoordinateMap::range(20, 0..8).unwrap(),
                ComponentCoordinateMap::range(20, 8..20).unwrap(),
            ],
            vec![ComponentCoordinateMap::indices(20, vec![19, 2, 16, 7, 0, 4, 1, 13, 10]).unwrap()],
        ] {
            if native_exchange && maps.len() != 1 {
                continue;
            }
            let mut native = NativeCapture {
                partition: None,
                stream: &stream,
                domain: None,
            };
            let mut ledger = CaptureLedger::new(&plan);
            let context = PartitionCaptureContext {
                invocation: None,
                artifact_identity: "native-fixture".into(),
                execution_identity: "selected-native-shards".into(),
                run_identity: "native-observed-branch".into(),
                overlay_identity: Some("native-overlay-context".into()),
                capture_plan_identity: plan.identity().into(),
                selection_index: 0,
                phase: CapturePhase::Prefill,
                prediction: 0,
                forward_epoch: 11,
            };
            let receipt_plan = PartitionCaptureReceiptPlan::new(
                plan.clone(),
                context.clone(),
                maps.iter()
                    .enumerate()
                    .map(|(rank, map)| PartitionCaptureProducer {
                        rank,
                        projection: CaptureSlicePartition::new(&[3, 20], &slice, 1, map, 16)
                            .unwrap(),
                    })
                    .collect(),
                maps.len(),
                PartitionCaptureReceiptLimits {
                    max_producers: 4,
                    max_fragments: 16,
                    max_record_bytes: 16_384,
                },
                &mut ledger,
            )
            .unwrap();
            let mut receipt_plan = Some(receipt_plan);
            let exchange = transport.as_ref().map(|transport| {
                PartitionCaptureExchange::admit(
                    transport,
                    receipt_plan.take().unwrap(),
                    &mut ledger,
                )
                .unwrap()
            });
            let mut fragments = Vec::new();
            HOST_READS.with(|reads| reads.set((0, 0)));
            for (rank, map) in maps.iter().enumerate() {
                let values: Vec<f32> = (0..3)
                    .flat_map(|row| {
                        (0..map.local_count())
                            .map(move |column| row * 20 + map.local_to_global(column).unwrap())
                    })
                    .map(|index| global[index])
                    .collect();
                let array = Array::from_slice(&values, &[3, map.local_count() as i32]);
                let array = if bf16 {
                    array.as_dtype(safemlx::Dtype::Bfloat16, &stream).unwrap()
                } else {
                    array
                };
                let tensor = MlxTensor::from_array(array);
                let projection = CaptureSlicePartition::new(&[3, 20], &slice, 1, map, 16).unwrap();
                for index in 0..projection.fragments().len() {
                    fragments.push(
                        capture_fragment(
                            &mut native,
                            &tensor,
                            PartitionCaptureRequest {
                                invocation: None,
                                plan: &plan,
                                selection_index: 0,
                                phase: CapturePhase::Prefill,
                                prediction: 0,
                                projection: &projection,
                                fragment_index: index,
                                producer_rank: rank,
                            },
                            &mut ledger,
                        )
                        .unwrap(),
                    );
                }
            }
            let (host_total, host_largest) = HOST_READS.with(std::cell::Cell::get);
            if matches!(transform, CaptureTransform::Slice) {
                assert_eq!(
                    host_total, 14,
                    "native export must contain only selected components/rows"
                );
            } else {
                assert_eq!(
                    host_largest, 1,
                    "native reductions must export only scalar statistics"
                );
                assert!(host_total <= fragments.len() * 8);
            }
            let precision = Some(if bf16 {
                checkpoint::TensorDtype::Bf16
            } else {
                checkpoint::TensorDtype::F32
            });
            let mut groups: std::collections::BTreeMap<usize, Vec<_>> =
                std::collections::BTreeMap::new();
            for fragment in fragments {
                groups
                    .entry(fragment.producer_rank())
                    .or_default()
                    .push(fragment);
            }
            let mut receipts = Vec::new();
            for rank in 0..maps.len() {
                receipts.push((
                    rank,
                    exchange
                        .as_ref()
                        .map(|exchange| exchange.receipt_plan())
                        .or(receipt_plan.as_ref())
                        .unwrap()
                        .encode_producer(
                            rank,
                            precision.clone(),
                            groups.remove(&rank).unwrap_or_default(),
                            &mut ledger,
                        )
                        .unwrap(),
                ));
            }
            let received = if let Some(exchange) = exchange {
                assert_eq!(receipts.len(), 1);
                exchange
                    .exchange(Ok(Some(receipts.pop().unwrap().1)), &mut ledger)
                    .unwrap()
            } else {
                let mut delivery = receipt_plan.unwrap().into_delivery();
                for (rank, bytes) in receipts.into_iter().rev() {
                    delivery.receive(rank, &bytes, &mut ledger).unwrap();
                }
                delivery.finish(&mut ledger).unwrap()
            };
            assert_eq!(received.context(), &context);
            assert_eq!(received.producers().len(), maps.len());
            assert!(!received.capture().contributions().is_empty());
            assert_eq!(
                HOST_READS.with(std::cell::Cell::get),
                (host_total, host_largest)
            );
            let (record, _) = received.into_parts().2.into_parts();
            assert_eq!(
                record.source_dtype,
                Some(if bf16 {
                    checkpoint::TensorDtype::Bf16
                } else {
                    checkpoint::TensorDtype::F32
                })
            );
            match record.payload.unwrap() {
                CapturePayload::Tensor(value) => {
                    assert_eq!(value.shape(), [2, 7]);
                    assert_eq!(value.data(), &TensorObservationData::F32(expected.clone()));
                }
                CapturePayload::Summary(value) => {
                    let finite: Vec<f64> = expected
                        .iter()
                        .copied()
                        .filter(|value| value.is_finite())
                        .map(f64::from)
                        .collect();
                    assert_eq!(value.elements, 14);
                    assert_eq!(value.finite, 11);
                    assert_eq!(
                        (
                            value.non_finite,
                            value.nan,
                            value.positive_infinity,
                            value.negative_infinity
                        ),
                        (3, 1, 1, 1)
                    );
                    assert_eq!(value.min, finite.iter().copied().reduce(f64::min));
                    assert_eq!(value.max, finite.iter().copied().reduce(f64::max));
                    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
                    let rms = (finite.iter().map(|value| value * value).sum::<f64>()
                        / finite.len() as f64)
                        .sqrt();
                    assert!((value.mean.unwrap() - mean).abs() < 3e-6 * rms.max(1.0));
                    assert!((value.rms.unwrap() - rms).abs() < 3e-6 * rms.max(1.0));
                }
                CapturePayload::Histogram(value) => {
                    let CaptureTransform::Histogram { edges } = &transform else {
                        panic!()
                    };
                    assert_eq!(&value.edges, edges);
                    assert_eq!(value.non_finite, 3);
                    assert_eq!(
                        value.below,
                        expected
                            .iter()
                            .filter(|value| value.is_finite() && **value < edges[0])
                            .count() as u64
                    );
                    assert_eq!(
                        value.above,
                        expected
                            .iter()
                            .filter(|value| value.is_finite() && **value > *edges.last().unwrap())
                            .count() as u64
                    );
                    for (index, bounds) in edges.windows(2).enumerate() {
                        let count = expected
                            .iter()
                            .filter(|value| {
                                value.is_finite()
                                    && **value >= bounds[0]
                                    && (**value < bounds[1]
                                        || (index + 2 == edges.len() && **value == bounds[1]))
                            })
                            .count() as u64;
                        assert_eq!(value.counts[index], count);
                    }
                }
                _ => panic!("incorrect assembled capture"),
            }
        }
    }
}

#[test]
fn native_partition_capture_fragments_cpu() {
    verify_native_partition_capture_fragments(
        safemlx::DeviceType::Cpu,
        CaptureTransform::Slice,
        false,
    );
}

#[test]
#[ignore = "requires a Metal device"]
#[cfg(feature = "metal")]
fn native_partition_capture_fragments_metal() {
    verify_native_partition_capture_fragments(
        safemlx::DeviceType::Gpu,
        CaptureTransform::Slice,
        false,
    );
}

#[test]
fn native_partition_capture_reductions_cpu() {
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 1.0, 4.0],
        },
    ] {
        verify_native_partition_capture_fragments(safemlx::DeviceType::Cpu, transform, false);
    }
}

#[test]
#[ignore = "requires a Metal device"]
#[cfg(feature = "metal")]
fn native_partition_capture_reductions_metal() {
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 1.0, 4.0],
        },
    ] {
        verify_native_partition_capture_fragments(safemlx::DeviceType::Gpu, transform, false);
    }
}

#[test]
fn native_partition_capture_exchange_cpu() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 1.0, 4.0],
        },
    ] {
        verify_native_partition_capture_fragments(safemlx::DeviceType::Cpu, transform, true);
    }
}

#[test]
#[ignore = "requires a Metal device"]
#[cfg(feature = "metal")]
fn native_partition_capture_exchange_metal() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 1.0, 4.0],
        },
    ] {
        verify_native_partition_capture_fragments(safemlx::DeviceType::Gpu, transform, true);
    }
}

fn selection(transform: CaptureTransform) -> CaptureSelection {
    CaptureSelection {
        id: "test".into(),
        path: "test".into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform,
    }
}

fn whole(shape: &[u64]) -> ResolvedCaptureSlice {
    ResolvedCaptureSlice {
        starts: vec![0; shape.len()],
        ends: shape.to_vec(),
        strides: vec![1; shape.len()],
        shape: shape.to_vec(),
    }
}

#[test]
fn partition_capture_source_has_bounded_completion_without_host_export() {
    use crate::backend::runtime::distributed::completion::{
        force_next_communication_pending, release_forced_pending_orphans,
    };
    use eredu_core::{BoundedCompletionOutcome, CompletionCancellationMode};
    use eredu_runtime::capture::partition::PartitionCaptureTransport;
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let manifest = eredu_runtime::CommunicationManifest::new(1, 0, vec![], vec![])
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_millis(250),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let transport = crate::backend::distributed::MlxDistributedSession::from_manifest(
        &manifest, &world, &stream,
    )
    .unwrap();
    let mut native = NativeCapture {
        stream: &stream,
        domain: None,
        partition: Some(&transport),
    };
    let wait = transport.capture_wait().unwrap();
    let source = Array::from_slice(&[1.0_f32, -2.0, 3.5], &[3]);
    let tensor = MlxTensor::from_array(safemlx::ops::add(&source, &source, &stream).unwrap());
    let cost = native.estimate_partition_source(&[3], wait).unwrap();
    assert!(cost.retained_bytes >= 3 * 8 && cost.host_bytes > 0);
    HOST_READS.with(|reads| reads.set((0, 0)));
    assert!(matches!(
        native.prepare_partition_source(&tensor, wait).unwrap(),
        BoundedCompletionOutcome::Completed
    ));
    assert_eq!(HOST_READS.with(std::cell::Cell::get), (0, 0));
    // Only the test reads the completed ordinary tensor; source preparation has
    // no host payload and preserves the exact input used by downstream work.
    assert_eq!(
        tensor.as_array().evaluated().unwrap().as_slice::<f32>(),
        &[2.0, -4.0, 7.0]
    );
    force_next_communication_pending();
    assert!(matches!(
        native.prepare_partition_source(&tensor, wait).unwrap(),
        BoundedCompletionOutcome::DeadlineExceeded {
            cancellation: CompletionCancellationMode::QuarantineUntilComplete
        }
    ));
    assert!(transport.ensure_capture_active().is_err());
    assert!(native.prepare_partition_source(&tensor, wait).is_err());
    assert_eq!(HOST_READS.with(std::cell::Cell::get), (0, 0));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        release_forced_pending_orphans();
        true
    });
}

#[test]
fn full_vocabulary_token_scores_use_every_logit_and_bound_host_reads() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let row: Vec<f32> = (0..3001)
        .map(|i| ((i * 17) % 101) as f32 * 0.03 - 1.5)
        .collect();
    let tensor = MlxTensor::from_array(Array::from_slice(&row, &[1, 1, 3001]));
    let mut native = NativeCapture {
        partition: None,
        stream: &stream,
        domain: None,
    };
    let request = selection(CaptureTransform::TokenScores {
        token_ids: vec![0, 73, 2000],
    });
    let cost = native
        .estimate(&tensor, &request, &whole(&[1, 1, 3001]))
        .unwrap();
    HOST_READS.with(|reads| reads.set((0, 0)));
    let CapturePayload::TokenScores(result) = native
        .transform(&tensor, &request, &whole(&[1, 1, 3001]))
        .unwrap()
    else {
        panic!("scores")
    };
    let log_partition = row.iter().map(|x| f64::from(*x).exp()).sum::<f64>().ln();
    assert!((result.log_partition - log_partition).abs() < 2e-7);
    assert_eq!(result.vocabulary, 3001);
    for selected in &result.scores {
        let expected = row[selected.target.token_id as usize];
        assert_eq!(selected.target.score, expected);
        assert!((selected.log_probability - (f64::from(expected) - log_partition)).abs() < 2e-7);
        assert_eq!(
            selected.rank,
            1 + row.iter().filter(|x| **x > expected).count() as u64
        );
        let competitor = selected.strongest_alternative.as_ref().unwrap();
        assert_ne!(competitor.token_id, selected.target.token_id);
        assert_eq!(
            competitor.score,
            row.iter()
                .enumerate()
                .filter(|(i, _)| *i != selected.target.token_id as usize)
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max)
        );
    }
    let (reads, largest) = HOST_READS.with(|reads| reads.get());
    assert_eq!(largest, 1, "only scalar reductions leave the device");
    assert!(reads <= 32);
    assert!(cost.host_bytes >= reads as u64 * 4);
    assert!(
        result.scores[0].log_probability < -7.0,
        "not a softmax over the three selected IDs"
    );
    for values in [vec![f32::MAX, f32::MAX], vec![10000.0, 10000.0]] {
        let tensor = MlxTensor::from_array(Array::from_slice(&values, &[2]));
        let request = selection(CaptureTransform::TokenScores {
            token_ids: vec![0, 1],
        });
        let CapturePayload::TokenScores(result) =
            native.transform(&tensor, &request, &whole(&[2])).unwrap()
        else {
            panic!()
        };
        for score in result.scores {
            assert_eq!(score.rank, 1);
            assert!((score.log_probability + 2f64.ln()).abs() < 1e-12);
        }
    }
}

#[test]
fn bounded_native_summary_handles_empty_nonfinite_extremes_and_chunks() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let empty = Array::from_slice(&[] as &[f32], &[0]);
    let result = summary(&empty, &stream).unwrap();
    assert_eq!(result.elements, 0);
    assert_eq!(result.mean, None);
    assert_eq!(result.rms, None);
    let values = [1.0f32, 2.0, 3.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let result = summary(&Array::from_slice(&values, &[6]), &stream).unwrap();
    assert_eq!(
        (
            result.elements,
            result.finite,
            result.nan,
            result.positive_infinity,
            result.negative_infinity
        ),
        (6, 3, 1, 1, 1)
    );
    assert_eq!(result.non_finite, 3);
    assert_eq!(result.min, Some(1.0));
    assert_eq!(result.max, Some(3.0));
    assert!((result.mean.unwrap() - 2.0).abs() < 1e-6);
    assert!((result.rms.unwrap() - (14.0f64 / 3.0).sqrt()).abs() < 1e-6);
    let extremes = Array::from_slice(&[f32::MAX, -f32::MAX], &[2]);
    let result = summary(&extremes, &stream).unwrap();
    assert_eq!(result.mean, Some(0.0));
    assert_eq!(result.rms, Some(f64::from(f32::MAX)));
    let result = summary(
        &Array::from_slice(&[f32::NAN, f32::INFINITY], &[2]),
        &stream,
    )
    .unwrap();
    assert_eq!(result.min, None);
    assert_eq!(result.mean, None);
    let values: Vec<f32> = (0..3001).map(|n| n as f32 / 3000.0).collect();
    let result = summary(&Array::from_slice(&values, &[3001]), &stream).unwrap();
    let mean = values.iter().map(|x| f64::from(*x)).sum::<f64>() / values.len() as f64;
    assert!((result.mean.unwrap() - mean).abs() < 1e-6);
}

#[test]
fn bounded_native_histogram_edge_and_nonfinite_semantics() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let values = Array::from_slice(
        &[-1.0f32, 0.0, 0.5, 1.0, 2.0, 3.0, f32::NAN, f32::INFINITY],
        &[8],
    );
    let result = histogram(&values, &[0.0, 1.0, 2.0], &stream).unwrap();
    assert_eq!(result.counts, [2, 2]);
    assert_eq!((result.below, result.above, result.non_finite), (1, 1, 2));
}

#[test]
fn bounded_native_integer_boolean_statistics_and_empty_histograms() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let integers = Array::from_slice(&[-2i32, 0, 2, 4], &[4]);
    let result = summary(&integers, &stream).unwrap();
    assert_eq!(
        (result.elements, result.finite, result.non_finite),
        (4, 4, 0)
    );
    assert_eq!(
        (result.min, result.max, result.mean),
        (Some(-2.0), Some(4.0), Some(1.0))
    );
    assert!((result.rms.unwrap() - 6.0f64.sqrt()).abs() < 1e-6);
    assert_eq!(
        histogram(&integers, &[-2.0, 0.0, 4.0], &stream)
            .unwrap()
            .counts,
        [1, 3]
    );
    let booleans = Array::from_slice(&[true, false, true, false], &[4]);
    let result = summary(&booleans, &stream).unwrap();
    assert_eq!(result.mean, Some(0.5));
    let empty = Array::from_slice(&[] as &[f32], &[0]);
    let histogram = histogram(&empty, &[0.0, 1.0], &stream).unwrap();
    assert_eq!(histogram.counts, [0]);
    assert_eq!(
        (histogram.below, histogram.above, histogram.non_finite),
        (0, 0, 0)
    );
}

#[test]
fn bounded_native_slices_and_previews_preserve_large_integer_ids() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let source = MlxTensor::from_array(Array::from_slice(
        &[u64::MAX, 1, u64::MAX - 1, 3, u64::MAX - 2, 5],
        &[2, 3],
    ));
    let mut native = NativeCapture {
        partition: None,
        stream: &stream,
        domain: None,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 0],
        ends: vec![2, 3],
        strides: vec![1, 2],
        shape: vec![2, 2],
    };
    let CapturePayload::Tensor(tensor) = native
        .transform(&source, &selection(CaptureTransform::Slice), &slice)
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(
        tensor.data(),
        &TensorObservationData::U64(vec![u64::MAX, u64::MAX - 1, 3, 5])
    );
    let CapturePayload::Tensor(tensor) = native
        .transform(
            &source,
            &selection(CaptureTransform::Preview { max_elements: 2 }),
            &whole(&[2, 3]),
        )
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(
        tensor.data(),
        &TensorObservationData::U64(vec![u64::MAX, 1])
    );
    let CapturePayload::Tensor(tensor) = native
        .transform(
            &source,
            &selection(CaptureTransform::Preview { max_elements: 0 }),
            &whole(&[2, 3]),
        )
        .unwrap()
    else {
        panic!()
    };
    assert!(tensor.data().is_empty());
}

#[test]
fn bounded_native_estimate_scales_host_transfer_with_preview_or_reduction() {
    let request = whole(&[1, 1_000_000]);
    let full = estimate_shape(
        &[1, 1_000_000],
        &selection(CaptureTransform::FullTensor),
        &request,
    )
    .unwrap();
    let preview = estimate_shape(
        &[1, 1_000_000],
        &selection(CaptureTransform::Preview { max_elements: 8 }),
        &request,
    )
    .unwrap();
    let reduced = estimate_shape(
        &[1, 1_000_000],
        &selection(CaptureTransform::Summary),
        &request,
    )
    .unwrap();
    assert_eq!(preview.host_bytes, 128);
    assert!(reduced.host_bytes < full.host_bytes / 50);
    assert!(estimate_shape(
        &[u64::MAX, 2],
        &selection(CaptureTransform::Summary),
        &request
    )
    .is_err());
}

#[test]
fn bounded_native_candidate_estimate_charges_the_row_not_the_prefill_source() {
    let vocabulary = 262_144;
    let request = selection(CaptureTransform::TopCandidates { count: 16 });
    let decode = estimate_shape(&[1, 1, vocabulary], &request, &whole(&[1, 1, vocabulary])).unwrap();
    let prefill = estimate_shape(
        &[1, 4096, vocabulary],
        &request,
        &whole(&[1, 4096, vocabulary]),
    )
    .unwrap();
    assert_eq!(prefill.retained_bytes, decode.retained_bytes);
    assert!(prefill.retained_bytes < 8 * 1024 * 1024);
}

#[test]
fn bounded_native_candidates_use_last_prediction_raw_scores() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let tensor = MlxTensor::from_array(Array::from_slice(
        &[99.0f32, 100.0, 0.0, -1.0, 4.0, 2.0],
        &[1, 2, 3],
    ));
    let mut native = NativeCapture {
        partition: None,
        stream: &stream,
        domain: None,
    };
    let request = selection(CaptureTransform::TopCandidates { count: 2 });
    let CapturePayload::Candidates(result) = native
        .transform(&tensor, &request, &whole(&[1, 2, 3]))
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(result.stage, CandidateScoreStage::RawLogitsBeforeSampling);
    assert_eq!(result.source, CandidateLogitsSource::Original);
    assert_eq!(result.domain, None);
    assert_eq!(
        result.candidates,
        [
            CaptureCandidate {
                token_id: 1,
                score: 4.0,
                allowed: true,
            },
            CaptureCandidate {
                token_id: 2,
                score: 2.0,
                allowed: true,
            }
        ]
    );
    assert_eq!(
        native
            .estimate(&tensor, &request, &whole(&[1, 2, 3]))
            .unwrap()
            .host_bytes,
        120
    );
}

#[test]
fn bounded_native_transforms_never_materialize_the_full_source_for_a_preview_or_reduction() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let source = MlxTensor::from_array(Array::from_slice(&vec![1.0f32; 16384], &[128, 128]));
    let mut native = NativeCapture {
        partition: None,
        stream: &stream,
        domain: None,
    };
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0.0, 1.0, 2.0],
        },
        CaptureTransform::Preview { max_elements: 16 },
    ] {
        HOST_READS.with(|reads| reads.set((0, 0)));
        native
            .transform(&source, &selection(transform), &whole(&[128, 128]))
            .unwrap();
        let (total, largest) = HOST_READS.with(std::cell::Cell::get);
        assert!(
            total < 1024,
            "a summary/preview must not copy a full 16384-value tensor"
        );
        assert!(largest <= 16, "hidden full-tensor host materialization");
    }
    HOST_READS.with(|reads| reads.set((0, 0)));
    native
        .transform(
            &source,
            &selection(CaptureTransform::FullTensor),
            &whole(&[128, 128]),
        )
        .unwrap();
    assert_eq!(HOST_READS.with(std::cell::Cell::get), (16384, 16384));
}

#[test]
fn original_and_effective_candidate_records_share_the_exact_domain() {
    use eredu_core::*;
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let validity = TokenFilter::allowed(vec![true, false, true, true]).unwrap();
    let filter = TokenFilter::allowed(vec![true, false, true, false]).unwrap();
    for (position, source, logits, expected_ids) in [
        (
            ObservationPosition::BeforeIntervention,
            CandidateLogitsSource::Original,
            [0.0f32, 8.0, 4.0, 1.0],
            [1, 2],
        ),
        (
            ObservationPosition::AfterIntervention,
            CandidateLogitsSource::Effective,
            [4.0f32, 0.0, 1.0, 8.0],
            [3, 0],
        ),
    ] {
        let path = MODEL_LOGITS_OBSERVATION_PATH.to_owned();
        let catalog = ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: vec![ObservationPoint {
                path: path.clone(),
                node_id: "output".into(),
                meaning: "logits".into(),
                value_type: ObservationValueType::Tensor,
                dtype: ObservationDtype::Floating,
                axes: Some(vec![TensorAxis {
                    name: "vocabulary".into(),
                    dimension: SymbolicDimension::Known(4),
                }]),
                prefill: true,
                decode: true,
                requirements: vec![],
                position,
                retained_bytes: None,
                host_bytes: None,
            }],
        };
        let support = ObservationSupportReport {
            schema_version: 1,
            capture: capabilities(),
            points: vec![ObservationSupport {
                path: path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        };
        let usage = CaptureUsage {
            captures: 4,
            retained_bytes: 1 << 20,
            host_bytes: 1 << 20,
            encoded_bytes: 1 << 20,
        };
        let mut selected = selection(CaptureTransform::TopCandidates { count: 2 });
        selected.path = path.clone();
        let plan = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: vec![selected],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &catalog,
            &support,
            &support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 1,
                max_predictions: 1,
            },
        )
        .unwrap();
        let mut capture = eredu_runtime::capture::CaptureSession::new(plan);
        capture.begin_step(CapturePhase::Prefill, 0).unwrap();
        let tensor = MlxTensor::from_array(Array::from_slice(&logits, &[4]));
        let mut native = NativeCapture {
            partition: None,
            stream: &stream,
            domain: Some(CaptureTokenDomain {
                filter: &filter,
                tokenizer_validity: &validity,
            }),
        };
        capture.observe(&mut native, &path, &tensor).unwrap();
        let step = capture.take_step().unwrap();
        let Some(CapturePayload::Candidates(result)) = &step.records[0].payload else {
            panic!()
        };
        assert_eq!(result.source, source);
        assert_eq!(
            result.domain,
            Some(CandidateDomain {
                allowed_tokens: 2,
                vocabulary: 4,
                constrained: true
            })
        );
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|c| c.token_id)
                .collect::<Vec<_>>(),
            expected_ids
        );
        assert!(!result.candidates[0].allowed);
        assert!(result.candidates[1].allowed);
        assert_eq!(result.candidates[0].score, 8.0);
    }
}
