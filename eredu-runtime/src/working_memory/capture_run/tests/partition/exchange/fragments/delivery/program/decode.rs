use super::*;
#[test]
fn scheduled_decode_projection_joins_original_host_vote_callback_and_final_delivery() {
    projected_invocation_case(CapturePhase::Decode, 1, None, false, false);
}

#[test]
fn scheduled_explicit_partition_invocations_keep_two_three_one_geometry() {
    for (phase, sequence) in [
        (CapturePhase::Prefill, 2),
        (CapturePhase::Decode, 3),
        (CapturePhase::Decode, 1),
    ] {
        projected_invocation_case(
            phase,
            4,
            Some(CaptureInvocationShape {
                batch: 1,
                sequence,
                context: None,
            }),
            false,
            false,
        );
    }
}

#[test]
fn scheduled_partition_receipt_rejects_foreign_invocation_axes() {
    projected_invocation_case(
        CapturePhase::Decode,
        4,
        Some(CaptureInvocationShape {
            batch: 1,
            sequence: 3,
            context: None,
        }),
        true,
        false,
    );
}

#[test]
fn scheduled_full_window_partition_keeps_window_custody_through_delivery() {
    projected_invocation_case(
        CapturePhase::Prefill,
        4,
        Some(CaptureInvocationShape {
            batch: 1,
            sequence: 2,
            context: None,
        }),
        false,
        true,
    );
}

#[test]
fn scheduled_partial_window_partition_preserves_logical_origin_and_physical_payload() {
    for (sequence, start) in [(1, 1), (2, 1)] {
        projected_window_case(
            CapturePhase::Prefill,
            4,
            Some(CaptureInvocationShape {
                batch: 1,
                sequence,
                context: None,
            }),
            false,
            Some(CaptureInvocationWindow {
                logical_sequence: 3,
                start,
            }),
            false,
        );
    }
}
#[test]
fn scheduled_partition_receipt_rejects_equal_width_shifted_window() {
    projected_window_case(
        CapturePhase::Prefill,
        4,
        Some(CaptureInvocationShape {
            batch: 1,
            sequence: 1,
            context: None,
        }),
        false,
        Some(CaptureInvocationWindow {
            logical_sequence: 3,
            start: 1,
        }),
        true,
    );
}

fn projected_invocation_case(
    phase: CapturePhase,
    prediction: u64,
    shape: Option<CaptureInvocationShape>,
    corrupt_axes: bool,
    full_window: bool,
) {
    let window = full_window.then(|| CaptureInvocationWindow {
        logical_sequence: shape.unwrap().sequence,
        start: 0,
    });
    projected_window_case(phase, prediction, shape, corrupt_axes, window, false);
}
fn projected_window_case(
    phase: CapturePhase,
    prediction: u64,
    shape: Option<CaptureInvocationShape>,
    corrupt_axes: bool,
    window: Option<CaptureInvocationWindow>,
    corrupt_window: bool,
) {
    use super::super::super::prefill::{
        projected_decode_receipt, projected_decode_source, projected_invocation_receipt,
        projected_invocation_source,
    };
    let sequence = shape.map_or(1, |shape| shape.sequence);
    let row_start = window.map_or(0, |window| window.start);
    for (transform, sum) in [
        (CaptureTransform::Slice, false),
        (CaptureTransform::Summary, false),
        (CaptureTransform::Summary, true),
        (
            CaptureTransform::Histogram {
                edges: vec![0.0, 100.0, 400.0],
            },
            false,
        ),
        (
            CaptureTransform::Histogram {
                edges: vec![0.0, 100.0, 400.0],
            },
            true,
        ),
    ] {
        for reject in [false, true] {
            let combination = if sum {
                PartitionCaptureCombination::SumF64ToF32
            } else {
                PartitionCaptureCombination::Disjoint
            };
            let source = if shape.is_some() {
                projected_invocation_source(transform.clone())
            } else {
                projected_decode_source(transform.clone())
            };
            let final_plan = || match shape {
                Some(shape) => CaptureRunHostPlan::prepare_invocation_window(
                    &source,
                    phase,
                    prediction,
                    shape,
                    &[true],
                    window,
                )
                .unwrap(),
                None => plan(&source),
            };
            let (metadata, _, _, _) = funding();
            let transport = ProgramRanks(Ranks {
                local: 0,
                bytes: RefCell::new(std::array::from_fn(|_| None)),
                funding: metadata.clone(),
                reject,
                calls: RefCell::new(vec![]),
            });
            let mut quote = CaptureLedger::new(source.admission());
            quote.begin_step();
            let mut prototype = match shape {
                Some(shape) => projected_invocation_receipt(
                    &source,
                    combination,
                    &metadata,
                    &mut quote,
                    phase,
                    prediction,
                    shape,
                    window,
                ),
                None => projected_decode_receipt(&source, combination, &metadata, &mut quote),
            };
            if corrupt_window {
                let mut shifted = prototype.context().clone();
                shifted.invocation_window.as_mut().unwrap().window.start += 1;
                let producers = prototype
                    .producers()
                    .map(|(rank, projection)| PartitionCaptureProducer {
                        rank,
                        projection: projection.clone(),
                    })
                    .collect();
                let mut quota = CaptureLedger::new(source.admission());
                quota.begin_step();
                let constructor = if sum {
                    PartitionCaptureReceiptPlan::new_sum
                } else {
                    PartitionCaptureReceiptPlan::new
                };
                let shifted = constructor(
                    source.clone(),
                    shifted,
                    producers,
                    4,
                    PartitionCaptureReceiptLimits {
                        max_producers: 3,
                        max_fragments: 3,
                        max_record_bytes: prototype.max_record_bytes(),
                    },
                    &mut quota,
                )
                .unwrap();
                assert_ne!(prototype.identity(), shifted.identity());
                assert!(!prototype.same_fragment_host_source(&shifted));
            }
            if !sum {
                use crate::capture::partition::PartitionInvocationReceiverSource;
                for (rank, shape) in [(1, [2, sequence, 0]), (2, [2, sequence, 4])] {
                    let source = PartitionInvocationReceiverSource::prepare_local(
                        &prototype,
                        rank,
                        TensorDtype::F16,
                        &shape,
                    )
                    .unwrap();
                    assert_eq!(source.source_shape(), shape.map(|n| n as usize));
                    assert_eq!(source.dtype(), &TensorDtype::F16);
                }
                assert!(
                    PartitionInvocationReceiverSource::prepare_local(
                        &prototype,
                        0,
                        TensorDtype::F16,
                        &[2, sequence, 4]
                    )
                    .is_err()
                );
                assert!(
                    PartitionInvocationReceiverSource::prepare_local(
                        &prototype,
                        2,
                        TensorDtype::I32,
                        &[2, sequence, 4]
                    )
                    .is_err()
                );
            }
            let geometry: Vec<_> = prototype
                .producers()
                .flat_map(|(rank, p)| {
                    p.fragments().iter().enumerate().map(move |(fragment, g)| {
                        (rank, fragment, p.local_shape().to_vec(), g.local().clone())
                    })
                })
                .collect();
            let raw = if sum {
                CaptureTransform::Slice
            } else {
                transform.clone()
            };
            let sources: Vec<_> = geometry
                .iter()
                .map(
                    |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                        producer: *rank,
                        fragment: *fragment,
                        local_shape: shape,
                        local_slice: slice,
                        transform: &raw,
                        dtype: TensorDtype::F16,
                        estimate: estimate(*rank),
                    },
                )
                .collect();
            let allowance = PreparedPartitionFragmentAllowance::prepare(
                &transport,
                &mut prototype,
                &sources,
                &metadata,
                &mut quote,
            )
            .unwrap();
            let final_charge = allowance.assembly_record_charge(&prototype).unwrap();
            let limits = PartitionCaptureReceiptLimits {
                max_producers: 3,
                max_fragments: 3,
                max_record_bytes: 64 << 10,
            };
            let producers = if sum {
                vec![
                    PartitionCaptureContiguousProducer {
                        rank: 0,
                        coordinates: 0..7,
                    },
                    PartitionCaptureContiguousProducer {
                        rank: 3,
                        coordinates: 0..7,
                    },
                ]
            } else {
                vec![
                    PartitionCaptureContiguousProducer {
                        rank: 3,
                        coordinates: 0..3,
                    },
                    PartitionCaptureContiguousProducer {
                        rank: 0,
                        coordinates: 3..7,
                    },
                    PartitionCaptureContiguousProducer {
                        rank: 1,
                        coordinates: 0..0,
                    },
                ]
            };
            let geometries: Vec<_> = geometry
                .iter()
                .map(|(rank, fragment, shape, slice)| {
                    crate::capture::partition::PartitionCaptureFragmentGeometry {
                        producer: *rank,
                        fragment: *fragment,
                        local_shape: shape,
                        local_slice: slice,
                        transform: &raw,
                        estimate: estimate(*rank),
                    }
                })
                .collect();
            let local_shape = if sum {
                [2, sequence, 7]
            } else {
                [2, sequence, 4]
            };
            let local = || {
                Some(crate::capture::partition::PartitionCaptureLocalSource {
                    producer: 0,
                    shape: &local_shape,
                    dtype: TensorDtype::F16,
                })
            };
            let retained = if let Some(shape) = shape {
                PreparedPartitionContiguousSource::new_local_shaped(
                    &source,
                    0,
                    2,
                    &producers,
                    &geometries,
                    local(),
                    combination,
                    phase,
                    prediction,
                    window
                        .map_or(Ok(shape), |window| window.validate(shape))
                        .unwrap(),
                    window.map(
                        |window| eredu_core::capture::PartitionCaptureInvocationWindow {
                            physical: shape,
                            window,
                        },
                    ),
                    &metadata,
                )
                .unwrap()
            } else {
                assert!(
                    PreparedPartitionContiguousSource::new_local_decode(
                        &source,
                        0,
                        2,
                        &producers,
                        &geometries,
                        local(),
                        combination,
                        0,
                        &metadata,
                    )
                    .is_err()
                );
                PreparedPartitionContiguousSource::new_local_decode(
                    &source,
                    0,
                    2,
                    &producers,
                    &geometries,
                    local(),
                    combination,
                    prediction,
                    &metadata,
                )
                .unwrap()
            };
            let mut oracle_quota = CaptureLedger::new(source.admission());
            oracle_quota.begin_step();
            let ordinary_rows = prototype
                .producers()
                .map(|(rank, p)| PartitionCaptureProducer {
                    rank,
                    projection: p.clone(),
                })
                .collect();
            let oracle_limits = PartitionCaptureReceiptLimits {
                max_record_bytes: prototype.max_record_bytes(),
                ..limits
            };
            let ordinary = if sum {
                PartitionCaptureReceiptPlan::new_sum(
                    source.clone(),
                    prototype.context().clone(),
                    ordinary_rows,
                    4,
                    oracle_limits,
                    &mut oracle_quota,
                )
            } else {
                PartitionCaptureReceiptPlan::new(
                    source.clone(),
                    prototype.context().clone(),
                    ordinary_rows,
                    4,
                    oracle_limits,
                    &mut oracle_quota,
                )
            }
            .unwrap();
            assert_eq!(ordinary.identity(), prototype.identity());
            let mut ordinary = ordinary.into_delivery();
            for (rank, projection) in prototype.producers() {
                let mut fragments = vec![];
                if let Some(fragment) = projection.fragments().first() {
                    let g = fragment.local();
                    let values: Vec<_> = (0..2)
                        .flat_map(|head| {
                            (0..sequence).flat_map(move |row| {
                                [1, 3, 5]
                                    .into_iter()
                                    .filter(move |column| {
                                        sum || if rank == 3 { *column < 3 } else { *column >= 3 }
                                    })
                                    .map(move |column| {
                                        head as f32 * 100.0
                                            + (row + row_start) as f32 * 10.0
                                            + column as f32
                                            + rank as f32 * 0.25
                                    })
                            })
                        })
                        .collect();
                    let selection = &source.admission().plan().selections[0];
                    let point = &source.admission().points()[0];
                    fragments.push(PartitionCaptureFragmentRecord {
                        fragment_index: 0,
                        record: CaptureRecord {
                            schema_version: CAPTURE_SCHEMA_VERSION,
                            selection_id: selection.id.clone(),
                            path: selection.path.clone(),
                            node_id: point.node_id.clone(),
                            position: point.position,
                            source_shape: Some(projection.local_shape().to_vec()),
                            source_dtype: Some(TensorDtype::F16),
                            selected_shape: Some(g.shape.clone()),
                            outcome: CaptureOutcome::Captured,
                            payload: Some(match &raw {
                                CaptureTransform::Summary => CapturePayload::Summary(
                                    crate::capture::partition::summarize_f32(&values),
                                ),
                                CaptureTransform::Histogram { edges } => {
                                    let mut histogram = eredu_core::capture::CaptureHistogram {
                                        edges: edges.clone(),
                                        counts: vec![0; edges.len() - 1],
                                        below: 0,
                                        above: 0,
                                        non_finite: 0,
                                    };
                                    crate::capture::partition::fill_histogram_f32(
                                        &values,
                                        &mut histogram,
                                    )
                                    .unwrap();
                                    CapturePayload::Histogram(histogram)
                                }
                                _ => CapturePayload::Tensor(
                                    TensorObservation::new(
                                        g.shape.iter().map(|n| *n as usize).collect(),
                                        TensorObservationData::F32(values),
                                    )
                                    .unwrap(),
                                ),
                            }),
                            charged: allowance
                                .fragment_record_charge(&prototype, rank, 0)
                                .unwrap()
                                .1,
                        },
                    });
                }
                let mut envelope = PartitionCaptureProducerRecord {
                    schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
                    combination,
                    receipt_plan_identity: prototype.identity().into(),
                    context: prototype.context().clone(),
                    producer_rank: rank,
                    source_dtype: Some(TensorDtype::F16),
                    fragments,
                };
                let bytes = serde_json::to_vec(&envelope).unwrap();
                ordinary.receive(rank, &bytes, &mut oracle_quota).unwrap();
                let bytes = if corrupt_window && rank == 3 {
                    envelope
                        .context
                        .invocation_window
                        .as_mut()
                        .unwrap()
                        .window
                        .start += 1;
                    serde_json::to_vec(&envelope).unwrap()
                } else if corrupt_axes && rank == 3 {
                    envelope.context.invocation.as_mut().unwrap().sequence += 1;
                    serde_json::to_vec(&envelope).unwrap()
                } else {
                    bytes
                };
                transport.0.bytes.borrow_mut()[rank] = Some(bytes);
            }
            let expected = ordinary.finish(&mut oracle_quota).unwrap();
            drop(allowance);
            let h = PartitionFragmentHostPlan::prepare(&prototype)
                .unwrap()
                .initialization_peak_bytes()
                + final_plan().initialization_peak_bytes();
            let pool = capture_test_ledger(h, 0).unwrap();
            let (reservation, run) = fresh(&pool, h);
            let context = prototype.context().clone();
            let host = run
                .prepare_partition_fragment_host(&reservation, prototype, None)
                .unwrap();
            let mut final_bank = run.prepare_capture_run(&reservation, final_plan()).unwrap();
            if shape.is_none() {
                drop(final_bank.begin_step(CapturePhase::Prefill, 0).unwrap());
            }
            let mut frame = final_bank
                .begin_step(phase, prediction)
                .unwrap()
                .prepare()
                .unwrap();
            let mut program = PreparedPartitionCaptureProgram::new_selected(
                &transport,
                &source,
                &context,
                &[PreparedPartitionCaptureRow::Contiguous],
                limits,
                &metadata,
            )
            .unwrap()
            .with_contiguous_row(0, retained, host)
            .unwrap();
            let mut quota = CaptureLedger::new(source.admission());
            quota.begin_step();
            let epoch = DistributedCommitEpoch::new(17).unwrap();
            program
                .prepare(&source, phase, prediction, epoch, &mut frame, &mut quota)
                .unwrap();
            assert!(program.take_invocation_projection(0).is_err());
            program.coordinate(epoch, &quota).unwrap();
            let spent = quota.total();
            let mut native = Native::default();
            assert!(program.produces(0).unwrap());
            assert!(
                program
                    .take_invocation_receiver_source(0)
                    .unwrap()
                    .is_none()
            );
            let hook = program.take_invocation_projection(0).unwrap().unwrap();
            let width = if sum { 7 } else { 4 };
            let offset = if sum { 0 } else { 3 };
            let value = Tensor {
                shape: [2, usize::try_from(sequence).unwrap(), width],
                values: (0..2)
                    .flat_map(|head| {
                        (0..sequence).flat_map(move |row| {
                            (0..width).map(move |column| {
                                head as f32 * 100.0
                                    + (row + row_start) as f32 * 10.0
                                    + (column + offset) as f32
                            })
                        })
                    })
                    .collect(),
            };
            let hook = hook.observe_invocation(&mut native, &value).unwrap();
            program.return_invocation_projection(0, hook).unwrap();
            assert!(
                program.take_invocation_projection(0).is_err(),
                "the same invocation cannot issue another native loan"
            );
            let result = program.deliver(&mut frame);
            assert_eq!(
                result.is_err(),
                reject || corrupt_axes || corrupt_window,
                "{result:?}"
            );
            if reject || corrupt_axes || corrupt_window {
                assert!(frame.records()[0].payload.is_none());
            } else {
                assert_eq!(frame.records()[0].charged, final_charge);
                assert_eq!(frame.partition_evidence().len(), 1);
                assert_eq!(
                    serde_json::to_value(frame.records()[0].payload.as_ref().unwrap()).unwrap(),
                    serde_json::to_value(expected.capture().record().payload.as_ref().unwrap())
                        .unwrap()
                );
            }
            assert_eq!(
                *transport.0.calls.borrow(),
                [
                    PartitionCaptureFrameKind::Source,
                    PartitionCaptureFrameKind::Coordination,
                    PartitionCaptureFrameKind::Preparation,
                    PartitionCaptureFrameKind::Payload,
                    PartitionCaptureFrameKind::Delivery
                ]
            );
            assert!(program.deliver(&mut frame).is_err());
            assert_eq!(quota.total(), spent);
            let escaped = if reject || corrupt_axes || corrupt_window {
                drop(frame);
                None
            } else {
                Some(
                    frame
                        .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
                        .unwrap(),
                )
            };
            drop(final_bank);
            drop(program);
            drop(run);
            drop(reservation);
            assert!(ledger(&pool).0 > 0);
            drop(escaped);
            drop(result);
            assert_eq!(ledger(&pool).0, 0);
        }
    }
}
