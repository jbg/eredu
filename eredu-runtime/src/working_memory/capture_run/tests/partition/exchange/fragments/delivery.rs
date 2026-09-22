use super::*;

struct Ranks {
    local: usize,
    bytes: RefCell<[Option<Vec<u8>>; 4]>,
    funding: HostMetadataFunding,
    reject: bool,
    calls: RefCell<Vec<PartitionCaptureFrameKind>>,
}

#[test]
fn contiguous_prefill_delivers_to_the_original_global_target_after_real_hook_progress() {
    use super::prefill::{inference, projected_receipt, projected_source};
    for sum in [false, true] {
        for (local, reject) in [(0, false), (2, false), (0, true)] {
            let transform = if sum {
                CaptureTransform::Summary
            } else {
                CaptureTransform::Slice
            };
            let combination = if sum {
                PartitionCaptureCombination::SumF64ToF32
            } else {
                PartitionCaptureCombination::Disjoint
            };
            let source = projected_source(transform.clone());
            let mut quota = CaptureLedger::new(source.admission());
            quota.begin_step();
            let (funding, _, _, _) = funding();
            let transport = Ranks {
                local,
                bytes: RefCell::new(std::array::from_fn(|_| None)),
                funding: funding.clone(),
                reject,
                calls: RefCell::new(vec![]),
            };
            let mut receipt = projected_receipt(&source, combination, &funding, &mut quota);
            let geometry: Vec<_> = receipt
                .producers()
                .flat_map(|(rank, p)| {
                    p.fragments()
                        .iter()
                        .enumerate()
                        .map(move |(i, g)| (rank, i, p.local_shape().to_vec(), g.local().clone()))
                })
                .collect();
            let sources: Vec<_> = geometry
                .iter()
                .map(
                    |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                        producer: *rank,
                        fragment: *fragment,
                        local_shape: shape,
                        local_slice: slice,
                        transform: &CaptureTransform::Slice,
                        dtype: TensorDtype::F32,
                        estimate: estimate(*rank),
                    },
                )
                .collect();
            let allowance = PreparedPartitionFragmentAllowance::prepare(
                &transport,
                &mut receipt,
                &sources,
                &funding,
                &mut quota,
            )
            .unwrap();
            let final_charge = allowance.assembly_record_charge(&receipt).unwrap();
            let ordinary_rows = receipt
                .producers()
                .map(|(rank, p)| PartitionCaptureProducer {
                    rank,
                    projection: p.clone(),
                })
                .collect();
            let limits = PartitionCaptureReceiptLimits {
                max_producers: 3,
                max_fragments: 3,
                max_record_bytes: receipt.max_record_bytes(),
            };
            let mut oq = CaptureLedger::new(source.admission());
            oq.begin_step();
            let ordinary = if sum {
                PartitionCaptureReceiptPlan::new_sum(
                    source.clone(),
                    receipt.context().clone(),
                    ordinary_rows,
                    4,
                    limits,
                    &mut oq,
                )
            } else {
                PartitionCaptureReceiptPlan::new(
                    source.clone(),
                    receipt.context().clone(),
                    ordinary_rows,
                    4,
                    limits,
                    &mut oq,
                )
            }
            .unwrap();
            assert_eq!(ordinary.identity(), receipt.identity());
            let mut ordinary = ordinary.into_delivery();
            let scalar = |rank: usize, head: usize, row: usize, column: usize| {
                head as f32 * 100.0 + row as f32 * 10.0 + column as f32 + rank as f32 * 0.25
            };
            for (rank, projection) in receipt.producers() {
                let mut fragments = vec![];
                if !projection.fragments().is_empty() {
                    let g = projection.fragments()[0].local();
                    let values: Vec<_> = (0..2)
                        .flat_map(|head| {
                            (1..3).flat_map(move |row| {
                                [1, 3, 5]
                                    .into_iter()
                                    .filter(move |column| {
                                        sum || if rank == 3 { *column < 3 } else { *column >= 3 }
                                    })
                                    .map(move |column| scalar(rank, head, row, column))
                            })
                        })
                        .collect();
                    assert_eq!(values.len() as u64, g.shape.iter().product::<u64>());
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
                            source_dtype: Some(TensorDtype::F32),
                            selected_shape: Some(g.shape.clone()),
                            outcome: CaptureOutcome::Captured,
                            payload: Some(CapturePayload::Tensor(
                                TensorObservation::new(
                                    g.shape.iter().map(|n| *n as usize).collect(),
                                    TensorObservationData::F32(values),
                                )
                                .unwrap(),
                            )),
                            charged: allowance
                                .fragment_record_charge(&receipt, rank, 0)
                                .unwrap()
                                .1,
                        },
                    });
                }
                let envelope = PartitionCaptureProducerRecord {
                    schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
                    combination,
                    receipt_plan_identity: receipt.identity().into(),
                    context: receipt.context().clone(),
                    producer_rank: rank,
                    source_dtype: Some(TensorDtype::F32),
                    fragments,
                };
                let bytes = serde_json::to_vec(&envelope).unwrap();
                ordinary.receive(rank, &bytes, &mut oq).unwrap();
                transport.bytes.borrow_mut()[rank] = Some(bytes);
            }
            let expected = ordinary.finish(&mut oq).unwrap();
            assert_eq!(expected.capture().record().charged, final_charge);
            let ranks: Vec<_> = receipt
                .producers()
                .map(|(producer, _)| PartitionCaptureRankSource {
                    producer,
                    dtype: Some(TensorDtype::F32),
                })
                .collect();
            // Local projection is retained before the delivery takes the receipt.
            let projection = receipt.producer(local).cloned();
            let host = PartitionFragmentHostPlan::prepare_prefill(&receipt, inference()).unwrap();
            let h = host.initialization_peak_bytes() + plan(&source).initialization_peak_bytes();
            let pool = capture_test_ledger(h, 0).unwrap();
            let (reservation, run) = fresh(&pool, h);
            let bank = run
                .prepare_partition_fragments(&reservation, host, allowance)
                .unwrap();
            let mut final_bank = run
                .prepare_capture_run(&reservation, plan(&source))
                .unwrap();
            let mut frame = final_bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare_prefill_with_progression(inference())
                .unwrap();
            frame.prepare_partition_evidence(&funding).unwrap();
            let mut delivery = PreparedPartitionFragmentDelivery::prepare(
                &transport, receipt, bank, &ranks, &funding,
            )
            .unwrap();
            let mut loan = None;
            let raw = if sum {
                None
            } else {
                Some(
                    CapturePrefillRowAssembly::prepare(source.admission(), 0, inference()).unwrap(),
                )
            };
            let reduced = if sum {
                Some(
                    CapturePrefillTransformPlan::prepare(source.admission(), 0, inference())
                        .unwrap(),
                )
            } else {
                None
            };
            let local_rows = projection
                .as_ref()
                .filter(|p| !p.fragments().is_empty())
                .map(|p| {
                    if sum {
                        CapturePrefillRowAssembly::prepare_additive_transform_partition(
                            source.admission(),
                            0,
                            inference(),
                            p,
                            0,
                        )
                        .unwrap()
                    } else {
                        CapturePrefillRowAssembly::prepare_partition(
                            source.admission(),
                            0,
                            inference(),
                            p,
                            0,
                            combination,
                        )
                        .unwrap()
                    }
                });
            let spent = quota.total();
            for k in 0..3 {
                let chunk = crate::prefill::PrefillChunk {
                    input: k..k + 1,
                    position: 2 + k,
                    output: inference().output.for_chunk(k == 2),
                };
                let decision = frame
                    .begin_prefill_hook(0, &chunk, &source.admission().plan().selections[0].path)
                    .unwrap();
                if decision == crate::capture::CapturePrefillHookDecision::First {
                    delivery.reserve_prefill_target(&mut frame).unwrap();
                    assert!(delivery.reserve_prefill_target(&mut frame).is_err());
                }
                assert!(
                    frame.take_assembled_prefill_destination(0).is_err(),
                    "no prefix is a global result"
                );
                // The actual owner leaves the transport-bearing program while the
                // native-style callback borrows disjoint receipt and host state.
                let (continuation, mut hook) = delivery.into_local_hook();
                if let Some(rows) = &local_rows {
                    let (receipt, bank) = hook.parts();
                    if decision == crate::capture::CapturePrefillHookDecision::First {
                        let mut native = bank
                            .begin_local_prefill(receipt, 0, &TensorDtype::F32, estimate(local))
                            .unwrap();
                        native
                            .quota_mut()
                            .reserve_quota(estimate(local).capture)
                            .unwrap();
                        loan = Some(native);
                    }
                    let fragment = rows.fragment(k).unwrap();
                    let width = projection.as_ref().unwrap().local_shape()[2] as usize;
                    let offset = if sum { 0 } else { 3 };
                    let mut writer = bank
                        .take_local_prefill_chunk(receipt, 0, &fragment)
                        .unwrap()
                        .prepare()
                        .unwrap();
                    for mapping in fragment.mappings() {
                        let i = mapping.source_index();
                        writer
                            .push_f32(scalar(local, i / width, k as usize, i % width + offset))
                            .unwrap();
                    }
                    writer.finish().unwrap();
                    bank.complete_local_prefill_chunk(receipt, 0, k).unwrap();
                }
                delivery = continuation.resume(hook).unwrap();
                if let Some(plan) = &reduced {
                    frame
                        .finish_summary_prefill_hook(0, &plan.fragment(k).unwrap())
                        .unwrap();
                } else {
                    frame
                        .finish_prefill_hook(0, &raw.as_ref().unwrap().fragment(k).unwrap())
                        .unwrap();
                }
                frame.complete_prefill_chunk(k).unwrap();
            }
            if local_rows.is_some() {
                delivery.finish_local_prefill(0).unwrap();
            }
            frame.finish_local_prefill_targets().unwrap();
            assert!(frame.finish_prefill_targets().is_err());
            let result = delivery.deliver_into_prefill_frame(&mut frame);
            assert_eq!(
                result.is_err(),
                reject,
                "sum={sum} local={local} calls={:?}",
                transport.calls.borrow()
            );
            assert!(frame.take_assembled_prefill_destination(0).is_err());
            assert_eq!(quota.total(), spent);
            assert_eq!(
                *transport.calls.borrow(),
                [
                    PartitionCaptureFrameKind::Preparation,
                    PartitionCaptureFrameKind::Payload,
                    PartitionCaptureFrameKind::Delivery
                ]
            );
            if !reject {
                frame.finish_prefill_targets().unwrap();
                assert_eq!(frame.records()[0].charged, final_charge);
                assert_eq!(
                    serde_json::to_value(frame.records()[0].payload.as_ref().unwrap()).unwrap(),
                    serde_json::to_value(expected.capture().record().payload.as_ref().unwrap())
                        .unwrap()
                );
                assert_eq!(frame.partition_evidence().len(), 1);
            } else {
                assert!(frame.records()[0].payload.is_none());
                assert!(frame.partition_evidence().is_empty());
            }
            drop(frame);
            drop(final_bank);
            drop(loan);
            drop(run);
            drop(reservation);
            assert_eq!(ledger(&pool).0 > 0, reject);
            drop(result);
            assert_eq!(ledger(&pool).0, 0);
        }
    }
}
impl ConsensusTransport for Ranks {
    type Error = Infallible;
    fn participant_count(&self) -> usize {
        4
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Infallible> {
        panic!("unbounded transport")
    }
}
impl BoundedConsensusTransport for Ranks {
    type Completion = Done;
    type GatherOutput = ();
    fn submit_all_gather_words(&self, _: &[u32]) -> Result<Submission<(), Done>, Infallible> {
        panic!("untyped frame")
    }
    fn resolve_all_gather_words(&self, _: ()) -> Result<Vec<u32>, Infallible> {
        panic!("untyped output")
    }
}
impl PartitionCaptureTransport for Ranks {
    fn capture_rank(&self) -> usize {
        self.local
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        Ok(BoundedCompletionWait::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap())
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn estimate_capture_gather(&self, _: usize) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage::default())
    }
    fn fail_capture_exchange(&self, _: &PartitionCaptureExchangeError) {
        panic!("completed rejection must retain the agreed failure")
    }
    fn capture_word_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::funded(n, &self.funding)?)
    }
    fn capture_byte_destination(
        &self,
        n: usize,
    ) -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::funded(n, &self.funding)?)
    }
    fn gather_capture_frame(
        &self,
        frame: &PartitionCaptureFrame<'_>,
        _: BoundedCompletionWait,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        self.calls.borrow_mut().push(frame.kind());
        let bytes = self.bytes.borrow();
        let mut out = PartitionCaptureBuffer::funded(frame.gathered_words(), &self.funding)?;
        match frame.kind() {
            PartitionCaptureFrameKind::Preparation | PartitionCaptureFrameKind::Delivery => {
                for rank in 0..4 {
                    let mut header: [u32; 16] = frame.words().try_into().unwrap();
                    header[3] = rank as u32;
                    if frame.kind() == PartitionCaptureFrameKind::Preparation {
                        header[13] = if bytes[rank].is_some() { 2 } else { 1 };
                        let length = bytes[rank].as_ref().map_or(0, Vec::len) as u64;
                        header[14] = length as u32;
                        header[15] = (length >> 32) as u32;
                    } else if self.reject && rank == 1 {
                        header[13] = 0;
                    }
                    out.extend_from_slice(&header)?;
                }
            }
            PartitionCaptureFrameKind::Payload => {
                for rank in 0..4 {
                    for word in 0..frame.words().len() {
                        let mut packed = [0; 4];
                        if let Some(bytes) = &bytes[rank] {
                            for (offset, byte) in packed.iter_mut().enumerate() {
                                *byte = bytes.get(word * 4 + offset).copied().unwrap_or(0);
                            }
                        }
                        out.extend_from_slice(&[u32::from_le_bytes(packed)])?;
                    }
                }
            }
            _ => panic!("unexpected receipt phase"),
        }
        Ok(out)
    }
}

#[test]
fn contiguous_delivery_preserves_local_and_empty_sources_and_releases_only_after_final_vote() {
    for combination in [
        PartitionCaptureCombination::Disjoint,
        PartitionCaptureCombination::SumF64ToF32,
    ] {
        for (local, reject) in [(0, false), (2, false), (0, true)] {
            let source = source();
            let mut quota = CaptureLedger::new(source.admission());
            quota.begin_step();
            let (funding, _, _, retired) = funding();
            let transport = Ranks {
                local,
                bytes: RefCell::new(std::array::from_fn(|_| None)),
                funding: funding.clone(),
                reject,
                calls: RefCell::new(Vec::new()),
            };
            let mut receipt = receipt(&source, combination, &funding, &mut quota);
            let geometry: Vec<_> = receipt
                .producers()
                .flat_map(|(rank, p)| {
                    p.fragments().iter().enumerate().map(move |(fragment, g)| {
                        (rank, fragment, p.local_shape().to_vec(), g.local().clone())
                    })
                })
                .collect();
            let raw = CaptureTransform::Slice;
            let transform = if combination == PartitionCaptureCombination::SumF64ToF32 {
                &raw
            } else {
                &source.admission().plan().selections[0].transform
            };
            let rows: Vec<_> = geometry
                .iter()
                .map(
                    |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                        producer: *rank,
                        fragment: *fragment,
                        local_shape: shape,
                        local_slice: slice,
                        transform,
                        dtype: TensorDtype::F16,
                        estimate: estimate(*rank),
                    },
                )
                .collect();
            let allowance = PreparedPartitionFragmentAllowance::prepare(
                &transport,
                &mut receipt,
                &rows,
                &funding,
                &mut quota,
            )
            .unwrap();
            let mut local_values = Vec::new();
            for (rank, projection) in receipt.producers() {
                let mut fragments = Vec::new();
                if !projection.fragments().is_empty() {
                    let shape = &projection.fragments()[0].local().shape;
                    let elements = shape.iter().product::<u64>();
                    let values: Vec<f32> = (0..elements)
                        .map(|i| {
                            if rank == 0 {
                                [16777216.0, -0.0, 1.25, -3.0][i as usize % 4]
                            } else {
                                [-16777216.0, 0.5, -0.75, 1.5][i as usize % 4]
                            }
                        })
                        .collect();
                    if rank == local {
                        local_values = values.clone();
                    }
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
                            selected_shape: Some(shape.clone()),
                            outcome: CaptureOutcome::Captured,
                            payload: Some(CapturePayload::Tensor(
                                TensorObservation::new(
                                    shape.iter().map(|n| *n as usize).collect(),
                                    TensorObservationData::F32(values),
                                )
                                .unwrap(),
                            )),
                            charged: allowance
                                .fragment_record_charge(&receipt, rank, 0)
                                .unwrap()
                                .1,
                        },
                    });
                }
                let envelope = PartitionCaptureProducerRecord {
                    schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
                    combination,
                    receipt_plan_identity: receipt.identity().into(),
                    context: receipt.context().clone(),
                    producer_rank: rank,
                    source_dtype: Some(TensorDtype::F16),
                    fragments,
                };
                transport.bytes.borrow_mut()[rank] = Some(serde_json::to_vec(&envelope).unwrap());
            }
            let ranks: Vec<_> = receipt
                .producers()
                .map(|(producer, _)| PartitionCaptureRankSource {
                    producer,
                    dtype: Some(TensorDtype::F16),
                })
                .collect();
            let ordinary_rows = receipt
                .producers()
                .map(|(rank, p)| PartitionCaptureProducer {
                    rank,
                    projection: p.clone(),
                })
                .collect();
            let limits = PartitionCaptureReceiptLimits {
                max_producers: 3,
                max_fragments: 3,
                max_record_bytes: receipt.max_record_bytes(),
            };
            let mut ordinary_quota = CaptureLedger::new(source.admission());
            ordinary_quota.begin_step();
            let ordinary = if combination == PartitionCaptureCombination::SumF64ToF32 {
                PartitionCaptureReceiptPlan::new_sum(
                    source.clone(),
                    receipt.context().clone(),
                    ordinary_rows,
                    4,
                    limits,
                    &mut ordinary_quota,
                )
            } else {
                PartitionCaptureReceiptPlan::new(
                    source.clone(),
                    receipt.context().clone(),
                    ordinary_rows,
                    4,
                    limits,
                    &mut ordinary_quota,
                )
            }
            .unwrap();
            assert_eq!(ordinary.identity(), receipt.identity());
            let mut ordinary = ordinary.into_delivery();
            for (rank, bytes) in transport.bytes.borrow().iter().enumerate() {
                if let Some(bytes) = bytes {
                    ordinary.receive(rank, bytes, &mut ordinary_quota).unwrap();
                }
            }
            let expected = ordinary.finish(&mut ordinary_quota).unwrap();
            let host = PartitionFragmentHostPlan::prepare(&receipt).unwrap();
            let h = host.initialization_peak_bytes() + plan(&source).initialization_peak_bytes();
            let pool = capture_test_ledger(h, 0).unwrap();
            let (reservation, run) = fresh(&pool, h);
            let bank = run
                .prepare_partition_fragments(&reservation, host, allowance)
                .unwrap();
            let mut final_bank = run
                .prepare_capture_run(&reservation, plan(&source))
                .unwrap();
            let mut frame = final_bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare()
                .unwrap();
            let spent = quota.total();
            let mut delivery = PreparedPartitionFragmentDelivery::prepare(
                &transport, receipt, bank, &ranks, &funding,
            )
            .unwrap();
            if !local_values.is_empty() {
                let (destination, mut loan) = delivery
                    .take_local(0, &TensorDtype::F16, estimate(local))
                    .unwrap()
                    .into_parts();
                loan.quota_mut()
                    .reserve_quota(estimate(local).capture)
                    .unwrap();
                let PartitionFragmentDestination::Tensor(claim) = destination else {
                    panic!("raw source")
                };
                let mut output = claim.prepare().unwrap();
                for value in local_values {
                    output.push_f32(value).unwrap();
                }
                let value = output.finish().unwrap();
                delivery
                    .record_local(
                        0,
                        &TensorDtype::F16,
                        estimate(local).capture,
                        PartitionFragmentValue::Tensor(value),
                    )
                    .unwrap();
            }
            let result = delivery.deliver(PartitionFragmentDestination::Tensor(
                frame.take_tensor(0).unwrap(),
            ));
            assert_eq!(result.is_err(), reject);
            assert_eq!(quota.total(), spent);
            assert_eq!(
                *transport.calls.borrow(),
                [
                    PartitionCaptureFrameKind::Preparation,
                    PartitionCaptureFrameKind::Payload,
                    PartitionCaptureFrameKind::Delivery
                ]
            );
            assert!(frame.take_tensor(0).is_err());
            assert!(frame.records()[0].payload.is_none());
            if let Ok(value) = &result {
                let PartitionFragmentValue::Tensor(value_tensor) = &value.value else {
                    panic!("raw result")
                };
                assert_eq!(
                    serde_json::to_value(eredu_core::capture::CapturePayloadWire::Tensor(
                        value_tensor.observation().as_observation()
                    ))
                    .unwrap(),
                    serde_json::to_value(expected.capture().record().payload.as_ref().unwrap())
                        .unwrap()
                );
                assert_eq!(value.charged, expected.capture().record().charged);
                assert_eq!(value.dtype, TensorDtype::F16);
                assert_eq!(
                    value.evidence.value.producers,
                    if combination == PartitionCaptureCombination::Disjoint {
                        vec![0, 1, 3]
                    } else {
                        vec![0, 3]
                    }
                );
            }
            drop(frame);
            drop(final_bank);
            drop(run);
            drop(reservation);
            drop(rows);
            drop(geometry);
            drop(source);
            drop(transport);
            drop(funding);
            assert!(ledger(&pool).0 > 0);
            assert!(!retired.load(Ordering::SeqCst));
            drop(result);
            assert_eq!(ledger(&pool).0, 0);
            assert!(retired.load(Ordering::SeqCst));
            assert_eq!(quota.total(), spent);
        }
    }
}

// Deliberately identical logical receipt/source identities do not authorize
// returning another request's independently paid host bank.
#[test]
fn partition_local_hook_rejects_foreign_original_owner_and_retains_both_accounts() {
    fn prepared<'t>(
        transport: &'t Ranks,
        source: &SharedCapturePlan,
        funding: &HostMetadataFunding,
    ) -> (PreparedPartitionFragmentDelivery<'t, Ranks>, MemoryLedger) {
        let mut quota = CaptureLedger::new(source.admission());
        quota.begin_step();
        let mut receipt = receipt(
            source,
            PartitionCaptureCombination::Disjoint,
            funding,
            &mut quota,
        );
        let geometry: Vec<_> = receipt
            .producers()
            .flat_map(|(rank, p)| {
                p.fragments()
                    .iter()
                    .enumerate()
                    .map(move |(i, g)| (rank, i, p.local_shape().to_vec(), g.local().clone()))
            })
            .collect();
        let transform = &source.admission().plan().selections[0].transform;
        let sources: Vec<_> = geometry
            .iter()
            .map(
                |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                    producer: *rank,
                    fragment: *fragment,
                    local_shape: shape,
                    local_slice: slice,
                    transform,
                    dtype: TensorDtype::F16,
                    estimate: estimate(*rank),
                },
            )
            .collect();
        let allowance = PreparedPartitionFragmentAllowance::prepare(
            transport,
            &mut receipt,
            &sources,
            funding,
            &mut quota,
        )
        .unwrap();
        let ranks: Vec<_> = receipt
            .producers()
            .map(|(producer, _)| PartitionCaptureRankSource {
                producer,
                dtype: Some(TensorDtype::F16),
            })
            .collect();
        let host = PartitionFragmentHostPlan::prepare(&receipt).unwrap();
        let h = host.initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let bank = run
            .prepare_partition_fragments(&reservation, host, allowance)
            .unwrap();
        let delivery =
            PreparedPartitionFragmentDelivery::prepare(transport, receipt, bank, &ranks, funding)
                .unwrap();
        drop(run);
        drop(reservation);
        (delivery, pool)
    }
    let source = source();
    let (funding, _, _, _) = funding();
    let transport = Ranks {
        local: 0,
        bytes: RefCell::new(std::array::from_fn(|_| None)),
        funding: funding.clone(),
        reject: false,
        calls: RefCell::new(vec![]),
    };
    let (first, first_pool) = prepared(&transport, &source, &funding);
    let (second, second_pool) = prepared(&transport, &source, &funding);
    let (continuation, mut original) = first.into_local_hook();
    let (other_continuation, mut foreign) = second.into_local_hook();
    assert_eq!(original.parts().0.identity(), foreign.parts().0.identity());
    drop(original);
    drop(other_continuation);
    let failure = continuation.resume(foreign).unwrap_err();
    assert!(
        transport.calls.borrow().is_empty(),
        "return validation cannot submit a protocol phase"
    );
    assert!(ledger(&first_pool).0 > 0);
    assert!(ledger(&second_pool).0 > 0);
    drop(failure);
    assert_eq!(ledger(&first_pool).0, 0);
    assert_eq!(ledger(&second_pool).0, 0);
}

mod hook_callback;

mod source_vote;

mod row_owner;

mod program;
