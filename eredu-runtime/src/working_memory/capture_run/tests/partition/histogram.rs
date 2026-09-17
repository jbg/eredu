//! Exact fixed-bin receipts share their original host claim and decoder.
use super::*;
use crate::capture::partition::PartitionCaptureRecordEncoding;

pub(super) fn histogram_source() -> SharedCapturePlan {
    histogram_source_with_sequence(false)
}
fn histogram_source_with_sequence(sequence: bool) -> SharedCapturePlan {
    let mut raw = raw();
    raw.selections.truncate(1);
    raw.selections[0].transform = CaptureTransform::Histogram {
        edges: vec![-1.0, 0.25, 2.0],
    };
    let mut point = point();
    if sequence {
        // Additive chunk reduction selects disjoint new-position rows. Context
        // includes the growing cached prefix and is intentionally not such a source.
        let axis = &mut point.axes.as_mut().unwrap()[0];
        axis.name = "sequence".into();
        axis.dimension = SymbolicDimension::Sequence;
    }
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let caps = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Histogram],
        max_histogram_bins: 2,
        physical_native_limit: false,
        conditions: vec![],
    };
    SharedCapturePlan::new(
        raw.admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: 2,
            },
        )
        .unwrap(),
    )
}
pub(super) fn histogram_record(
    mut record: CaptureRecord,
    geometry: &CaptureHistogramGeometry<'_>,
) -> CaptureRecord {
    record.source_shape = Some(geometry.source_shape().iter().map(|n| *n as u64).collect());
    record.selected_shape = Some(geometry.shape().iter().map(|n| *n as u64).collect());
    record.source_dtype = Some(TensorDtype::F16);
    record.outcome = CaptureOutcome::Captured;
    record.payload = Some(CapturePayload::Histogram(CaptureHistogram {
        edges: geometry.edges().to_vec(),
        counts: vec![3, 2],
        below: 1,
        above: 2,
        non_finite: 2,
    }));
    record.charged = record.charged.checked_add(native_usage()).unwrap();
    record
}
#[test]
fn partition_histogram_receipt_preserves_edges_counts_and_failed_payload_custody() {
    for case in [
        "valid",
        "total",
        "edges",
        "duplicate",
        "short",
        "negative",
        "identity",
        "reservation",
    ] {
        let source = histogram_source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let mut bank = run
            .prepare_capture_run(&reservation, plan(&source))
            .unwrap();
        let mut frame = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let context = context(&source, 0);
        let initial = frame.records()[0].clone();
        let claim = frame.take_histogram(0).unwrap();
        assert_eq!(claim.geometry().elements(), 10);
        let record = histogram_record(initial, claim.geometry());
        let (funding, used, refuse, retired) = funding();
        let encoded = PartitionCaptureRecordEncoding::new(
            &context,
            "retained-receipt",
            3,
            PartitionCaptureCombination::Disjoint,
            &record,
            64 << 10,
        )
        .encode(&funding)
        .unwrap();
        let text = std::str::from_utf8(&encoded).unwrap();
        let bytes = match case {
            "total" => text.replace("[3,2]", "[4,2]").into_bytes(),
            "edges" => text.replace("0.25", "0.5").into_bytes(),
            "duplicate" => text
                .replace("\"below\":1", "\"below\":1,\"below\":1")
                .into_bytes(),
            "short" => text.replace("[3,2]", "[3]").into_bytes(),
            "negative" => text.replace("[3,2]", "[3,-2]").into_bytes(),
            "identity" => text
                .replace("retained-receipt", "other-receipt")
                .into_bytes(),
            _ => text.as_bytes().to_vec(),
        };
        drop(encoded);
        if case == "reservation" {
            refuse.store(true, Ordering::SeqCst);
        }
        let result =
            claim.decode_partition_receipt(&bytes, expected(&context, record.charged), &funding);
        assert!(frame.take_histogram(0).is_err());
        assert!(used.load(Ordering::SeqCst) > 0);
        let error = if case == "valid" {
            let result = result.unwrap();
            assert_eq!(
                Some(&CapturePayload::Histogram(result.observation().clone())),
                record.payload.as_ref()
            );
            frame
                .record_histogram(result, TensorDtype::F16, native_usage())
                .unwrap();
            assert_eq!(frame.records()[0].payload, record.payload);
            None
        } else {
            Some(result.unwrap_err())
        };
        drop(frame);
        drop(bank);
        drop(run);
        drop(reservation);
        drop(funding);
        if let Some(error) = error {
            assert!(ledger(&pool).0 >= h);
            assert!(!retired.load(Ordering::SeqCst));
            drop(error);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn remote_histogram_prefill_keeps_one_paid_destination_through_all_chunks() {
    use crate::capture::CapturePrefillHookDecision;
    let source = histogram_source_with_sequence(true);
    let h = plan(&source).initialization_peak_bytes();
    let pp = WorkingMemoryPool::new(h, 0).unwrap();
    let rp = WorkingMemoryPool::new(h, 0).unwrap();
    let (pr, p_run) = fresh(&pp, h);
    let (rr, r_run) = fresh(&rp, h);
    let mut pb = p_run.prepare_capture_run(&pr, plan(&source)).unwrap();
    let mut rb = r_run.prepare_capture_run(&rr, plan(&source)).unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    let transform = CapturePrefillTransformPlan::prepare(source.admission(), 0, geometry).unwrap();
    let mut producer = pb
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill_with_progression(geometry)
        .unwrap();
    let mut receiver = rb
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill_with_progression(geometry)
        .unwrap();
    let mut pq = CaptureLedger::new(source.admission());
    pq.begin_step();
    let mut rq = CaptureLedger::new(source.admission());
    rq.begin_step();
    for k in 0..transform.chunk_count() {
        let fragment = transform.fragment(k).unwrap();
        let chunk = crate::prefill::PrefillChunk {
            input: fragment.input().clone(),
            position: fragment.position(),
            output: fragment.output_demand(),
        };
        for (frame, quota, remote) in [
            (&mut producer, &mut pq, false),
            (&mut receiver, &mut rq, true),
        ] {
            let decision = frame.begin_prefill_hook(0, &chunk, "block.output").unwrap();
            if decision == CapturePrefillHookDecision::First {
                assert!(frame
                    .reserve_prefill_hook(0, quota, TensorDtype::F16, native_usage())
                    .unwrap()
                    .is_none());
                if remote {
                    frame.mark_remote_prefill_target(0).unwrap();
                }
            }
            if remote {
                assert!(frame.take_remote_prefill_histogram(0).is_err());
            } else {
                let mut bins = frame
                    .take_prefill_histogram(0, &fragment)
                    .unwrap()
                    .prepare()
                    .unwrap();
                bins.add_bin(0, fragment.selected_elements()).unwrap();
                let receipt = bins.finish(0, 0, 0).unwrap();
                frame.record_prefill_histogram(receipt, &fragment).unwrap();
            }
            frame.finish_histogram_prefill_hook(0, &fragment).unwrap();
            frame.complete_prefill_chunk(k).unwrap();
        }
    }
    assert_eq!(pq.total(), rq.total());
    let spent = rq.total();
    producer.finish_prefill_targets().unwrap();
    receiver.finish_local_prefill_targets().unwrap();
    assert!(receiver.finish_prefill_targets().is_err());
    let context = context(&source, 0);
    let record = &producer.records()[0];
    let (funding, _, _, retired) = funding();
    let encoded = PartitionCaptureRecordEncoding::new(
        &context,
        "retained-receipt",
        3,
        PartitionCaptureCombination::Disjoint,
        record,
        64 << 10,
    )
    .encode(&funding)
    .unwrap();
    let claim = receiver.take_remote_prefill_histogram(0).unwrap();
    let decoded = claim
        .decode_partition_receipt(&encoded, expected(&context, record.charged), &funding)
        .unwrap();
    receiver
        .record_remote_prefill_histogram(decoded, TensorDtype::F16)
        .unwrap();
    assert!(receiver.take_remote_prefill_histogram(0).is_err());
    receiver.finish_prefill_targets().unwrap();
    assert_eq!(receiver.records()[0].payload, record.payload);
    assert_eq!(receiver.records()[0].charged, record.charged);
    assert_eq!(rq.total(), spent);
    let output = receiver
        .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
        .unwrap();
    let escaped = output.clone();
    drop(output);
    // The shared two-frame loop gives both short loans one common lifetime.
    // Retire its remaining frame before either borrowed bank.
    drop(producer);
    drop(pb);
    drop(p_run);
    drop(pr);
    assert_eq!(pp.used_bytes().unwrap(), 0);
    drop(rb);
    drop(r_run);
    drop(rr);
    assert_eq!(rp.used_bytes().unwrap(), h);
    drop(escaped);
    assert_eq!(rp.used_bytes().unwrap(), 0);
    drop(encoded);
    drop(funding);
    assert!(retired.load(Ordering::SeqCst));
}
