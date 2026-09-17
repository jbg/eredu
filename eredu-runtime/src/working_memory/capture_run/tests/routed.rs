use super::*;

fn source() -> SharedCapturePlan {
    source_with_bounds(3, 1)
}
fn source_with_bounds(token_end: u64, maximum: u64) -> SharedCapturePlan {
    let mut p = point();
    p.value_type = ObservationValueType::RoutedUnits {
        routing: "route".into(),
        geometry: RoutedUnitGeometry {
            experts: 7,
            units_per_expert: 5,
            routes_per_token: 3,
        },
    };
    p.axes = Some(vec![
        TensorAxis {
            name: "token".into(),
            dimension: SymbolicDimension::TokenRows,
        },
        TensorAxis {
            name: "route".into(),
            dimension: SymbolicDimension::Known(3),
        },
        TensorAxis {
            name: "unit".into(),
            dimension: SymbolicDimension::Known(5),
        },
    ]);
    let mut request = raw();
    request.selections.truncate(1);
    request.selections[0].transform = CaptureTransform::RoutedUnits;
    request.selections[0].slices = vec![
        CaptureSlice {
            axis: "token".into(),
            start: 0,
            end: token_end,
            stride: 2,
        },
        CaptureSlice {
            axis: "route".into(),
            start: 0,
            end: 3,
            stride: 2,
        },
        CaptureSlice {
            axis: "unit".into(),
            start: 1,
            end: 5,
            stride: 2,
        },
    ];
    admit(request, p, maximum, false)
}
fn fill_row(writer: &mut ScheduledCaptureRoutedUnits<'_, '_>, token: u64, slot: u64) {
    writer
        .begin_row(
            None,
            token,
            slot,
            (token + slot) % 7,
            0.25 + (slot as f32) * 0.1,
        )
        .unwrap();
    writer.push_f32(token as f32 + slot as f32 + 0.125).unwrap();
    writer.push_f32(f32::NEG_INFINITY).unwrap();
    writer.finish_row().unwrap();
}
#[test]
fn original_routed_destinations_charge_once_preserve_sparse_values_and_keep_output_custody() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, run) = fresh(&pool, h);
    let mut bank = run
        .prepare_capture_run(&reservation, plan(&source))
        .unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = step.take_routed_units(0).unwrap();
    assert_eq!(claim.geometry().rows(), 4);
    assert_eq!(claim.geometry().elements(), 8);
    let mut destination = claim.prepare().unwrap();
    for (token, slot) in [(2, 2), (0, 2), (2, 0), (0, 0)] {
        fill_row(&mut destination, token, slot);
    }
    destination.source_chunk(1, 3).unwrap();
    destination.source_chunk(0, 1).unwrap();
    let receipt = destination.finish().unwrap();
    assert_eq!(
        receipt
            .observation()
            .rows
            .iter()
            .map(|r| (r.token, r.slot))
            .collect::<Vec<_>>(),
        [(0, 0), (0, 2), (2, 0), (2, 2)]
    );
    assert_eq!(receipt.observation().rows[0].values.shape(), [2]);
    let TensorObservationData::F32(values) = receipt.observation().rows[0].values.data() else {
        panic!("float source");
    };
    let address = values.as_ptr();
    assert_eq!(values[0], 0.125);
    assert!(values[1].is_infinite());
    assert!(step.take_routed_units(0).is_err());
    step.record_routed_units(
        receipt,
        TensorDtype::F32,
        CaptureUsage {
            captures: 1,
            retained_bytes: 65536,
            host_bytes: 65536,
            encoded_bytes: 65536,
        },
    )
    .unwrap();
    let output = finish(step);
    let alias = output.clone();
    drop(output);
    drop(bank);
    drop(run);
    drop(reservation);
    assert_eq!(pool.used_bytes().unwrap(), h);
    let CapturePayload::RoutedUnits(value) = alias.records()[0].payload.as_ref().unwrap() else {
        panic!("sparse output");
    };
    let TensorObservationData::F32(values) = value.rows[0].values.data() else {
        panic!("float output");
    };
    assert_eq!(values.as_ptr(), address);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_routed_failures_retain_partial_destination_and_cannot_refund_claim() {
    for fail in 0..3 {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let mut bank = run
            .prepare_capture_run(&reservation, plan(&source))
            .unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let mut writer = step.take_routed_units(0).unwrap().prepare().unwrap();
        fill_row(&mut writer, 0, 0);
        match fail {
            0 => {
                writer.begin_row(None, 2, 0, 1, 0.5).unwrap();
                writer.push_f32(3.25).unwrap();
            }
            1 => {
                fill_row(&mut writer, 0, 0);
                fill_row(&mut writer, 2, 0);
                fill_row(&mut writer, 2, 2);
                writer.source_chunk(0, 3).unwrap();
            }
            _ => {
                writer.source_chunk(1, 3).unwrap();
                assert!(writer.source_chunk(0, 2).is_err());
            }
        }
        let error = writer.finish().unwrap_err();
        assert!(step.take_routed_units(0).is_err());
        drop(step);
        drop(bank);
        drop(run);
        drop(reservation);
        assert_eq!(pool.used_bytes().unwrap(), h);
        match error.error() {
            CaptureRoutedHostError::Geometry(
                RoutedUnitValidationError::Incomplete
                | RoutedUnitValidationError::Row
                | RoutedUnitValidationError::Chunks,
            ) => (),
            cause => panic!("unexpected error: {cause}"),
        }
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn original_routed_prefill_retains_one_destination_through_uneven_and_zero_selected_chunks(){
    for width in [1,2]{
        let source=source();let h=plan(&source).initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let mut bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
        let inference=InferenceGeometry {batch_size:1,cached_positions:2,input_positions:3,
            max_output_tokens:1,prefill_chunk_positions:width,output:OutputDemand::LastPosition};
        let sparse=CaptureRoutedPrefillPlan::prepare(source.admission(),0,inference).unwrap();
        let mut step=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill(inference).unwrap();
        let usage=CaptureUsage {captures:1,retained_bytes:65536,host_bytes:65536,encoded_bytes:65536};
        step.begin_prefill_target(0,TensorDtype::F32,usage).unwrap();
        let charged=step.records()[0].charged;
        let mut address=None;
        for index in 0..sparse.chunk_count(){
            let fragment=sparse.fragment(index).unwrap();
            for physical in 0..fragment.source_tokens(){
                let mut writer=step.take_prefill_routed_fragment(0,&fragment).unwrap();
                for slot in [0,2] {
                    if !fragment.selects(physical,slot){continue;}
                    let token=fragment.logical_token(physical).unwrap();
                    writer.begin_row(physical,slot,(token+slot)%7,0.25+slot as f32*0.1).unwrap();
                    writer.push_f32(token as f32+slot as f32+0.125).unwrap();
                    writer.push_f32(f32::NEG_INFINITY).unwrap();writer.finish_row().unwrap();
                }
                writer.source_chunk(physical,physical+1).unwrap();writer.finish().unwrap();
                if physical+1<fragment.source_tokens(){assert!(step.complete_prefill_chunk(index).is_err());}
            }
            let current=std::ptr::from_ref(step.frame.prefill.as_ref().unwrap().slots[0].routed.as_ref().unwrap());
            assert_eq!(*address.get_or_insert(current),current);
            assert!(step.take_prefill_routed_fragment(0,&fragment).is_err());
            step.complete_prefill_chunk(index).unwrap();
            assert_eq!(pool.used_bytes().unwrap(),h);
        }
        step.finish_prefill_targets().unwrap();
        let output=finish(step);drop(bank);drop(run);drop(reservation);
        assert_eq!(pool.used_bytes().unwrap(),h);
        let Some(CapturePayload::RoutedUnits(value))=&output.records()[0].payload else{panic!("sparse receipt");};
        assert_eq!(value.rows.iter().map(|r|(r.token,r.slot)).collect::<Vec<_>>(),[(0,0),(0,2),(2,0),(2,2)]);
        assert_eq!(value.source_token_ranges.first().unwrap()[0],0);
        assert_eq!(value.source_token_ranges.last().unwrap()[1],3);
        assert_eq!(output.records()[0].charged,charged);
        drop(output);assert_eq!(pool.used_bytes().unwrap(),0);
    }
}
#[test]
fn dropped_routed_prefill_writer_retains_partial_rows_and_cannot_retry(){
    let source=source();let h=plan(&source).initialization_peak_bytes();
    let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
    let mut bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
    let inference=InferenceGeometry {batch_size:1,cached_positions:2,input_positions:3,
        max_output_tokens:1,prefill_chunk_positions:2,output:OutputDemand::LastPosition};
    let sparse=CaptureRoutedPrefillPlan::prepare(source.admission(),0,inference).unwrap();
    let fragment=sparse.fragment(0).unwrap();
    let mut step=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill(inference).unwrap();
    step.begin_prefill_target(0,TensorDtype::F32,CaptureUsage {captures:1,retained_bytes:65536,host_bytes:65536,encoded_bytes:65536}).unwrap();
    let mut writer=step.take_prefill_routed_fragment(0,&fragment).unwrap();
    writer.begin_row(0,0,1,0.5).unwrap();writer.push_f32(7.25).unwrap();drop(writer);
    assert!(step.take_prefill_routed_fragment(0,&fragment).is_err());
    assert!(step.complete_prefill_chunk(0).is_err());
    assert!(step.frame.prefill.as_ref().unwrap().slots[0].routed.is_some());
    assert_eq!(pool.used_bytes().unwrap(),h);
    drop(step);drop(bank);drop(run);drop(reservation);assert_eq!(pool.used_bytes().unwrap(),0);
}

#[test]
fn original_routed_invocation_batches_share_one_destination_across_prefill_and_decode() {
    let source = source_with_bounds(1, 2);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&reservation, plan(&source)).unwrap();
    let usage = CaptureUsage { captures: 1, retained_bytes: 65536,
        host_bytes: 65536, encoded_bytes: 65536 };
    let mut outputs = Vec::new();
    for (phase, prediction, tokens) in [(CapturePhase::Prefill, 0, 3),
        (CapturePhase::Decode, 1, 1)] {
        let mut step = bank.begin_step(phase, prediction).unwrap().prepare().unwrap();
        step.begin_routed_invocation(0, TensorDtype::F32, usage).unwrap();
        assert!(step.begin_routed_invocation(0, TensorDtype::F32, usage).is_err());
        for token in 0..tokens {
            let mut writer = step.take_routed_batch(0).unwrap();
            for slot in [0, 2] {
                if writer.selects(token, slot) {
                    writer.begin_row(token, slot, slot, 0.25).unwrap();
                    writer.push_f32(prediction as f32 + slot as f32 + 0.125).unwrap();
                    writer.push_f32(-0.75).unwrap();
                    writer.finish_row().unwrap();
                }
            }
            writer.source_chunk(token, token + 1).unwrap();
            writer.finish().unwrap();
        }
        step.finish_routed_invocation(0).unwrap();
        assert!(step.take_routed_batch(0).is_err());
        assert!(step.begin_routed_invocation(0, TensorDtype::F32, usage).is_err());
        assert_eq!(step.records()[0].charged.captures, 1);
        outputs.push(finish(step));
    }
    drop((bank, run, reservation));
    assert_eq!(pool.used_bytes().unwrap(), h);
    for (prediction, output) in outputs.iter().enumerate() {
        let Some(CapturePayload::RoutedUnits(value)) = &output.records()[0].payload else {
            panic!("sparse result");
        };
        assert_eq!(value.rows.len(), 2);
        for (row, slot) in value.rows.iter().zip([0, 2]) {
            assert_eq!((row.token, row.slot), (0, slot));
            let TensorObservationData::F32(values) = row.values.data() else {
                panic!("float result");
            };
            assert_eq!(values, &[prediction as f32 + slot as f32 + 0.125, -0.75]);
        }
    }
    drop(outputs);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_routed_invocation_dropped_batch_keeps_payload_and_refuses_retry() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&reservation, plan(&source)).unwrap();
    let mut step = bank.begin_step(CapturePhase::Prefill, 0).unwrap().prepare().unwrap();
    step.begin_routed_invocation(0, TensorDtype::F32, CaptureUsage::default()).unwrap();
    {
        let mut writer = step.take_routed_batch(0).unwrap();
        writer.begin_row(0, 0, 1, 0.25).unwrap();
        writer.push_f32(1.125).unwrap();
    }
    assert!(step.take_routed_batch(0).is_err());
    assert!(step.finish_routed_invocation(0).is_err());
    assert!(step.begin_routed_invocation(0, TensorDtype::F32, CaptureUsage::default()).is_err());
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(step);
    drop((bank, run, reservation));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
