use super::*;

mod partition;
mod progression;
mod transfer;

fn rows(preview: u64) -> SharedCapturePlan {
    let mut point = point();
    point.axes = Some(vec![
        TensorAxis {
            name: "head".into(),
            dimension: SymbolicDimension::Known(2),
        },
        TensorAxis {
            name: "row".into(),
            dimension: SymbolicDimension::Sequence,
        },
        TensorAxis {
            name: "width".into(),
            dimension: SymbolicDimension::Known(2),
        },
    ]);
    let mut raw = raw();
    raw.selections.truncate(2);
    raw.selections[1].schedule = CaptureSchedule::default();
    raw.selections[1].transform = CaptureTransform::Preview {
        max_elements: preview,
    };
    admit(raw, point, 4, false)
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn charged(step: &ScheduledCaptureStep<'_>) -> CaptureUsage {
    step.records()
        .iter()
        .fold(CaptureUsage::default(), |sum, r| {
            sum.checked_add(r.charged).unwrap()
        })
}
fn charge() -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        retained_bytes: 128,
        host_bytes: 128,
        encoded_bytes: 10_000,
    }
}
fn write_fragment(
    step: &mut ScheduledCaptureStep<'_>,
    index: usize,
    assembly: &CapturePrefillRowAssembly<'_>,
    chunk: u64,
) {
    let fragment = assembly.fragment(chunk).unwrap();
    let mut writer = step
        .take_prefill_fragment(index, &fragment)
        .unwrap()
        .prepare()
        .unwrap();
    // Actual row-major values for [head, local_row, width], with different
    // values in each head and row. The expected full output is independent of
    // the fragment's destination mapping (which the worker alone applies).
    let rows = (fragment.input().end - fragment.input().start) as usize;
    for map in fragment.mappings() {
        let source = map.source_index();
        let column = source % 2;
        let row = (source / 2) % rows;
        let head = source / (2 * rows);
        let value = head * 100 + (fragment.input().start as usize + row) * 10 + column;
        writer.push_f32(value as f32).unwrap();
    }
    writer.finish().unwrap();
}
fn tensor(record: &CaptureRecord) -> &SharedTensorObservation {
    let Some(CapturePayload::SharedTensor(tensor)) = &record.payload else {
        panic!("completed shared tensor")
    };
    tensor
}
fn values(tensor: &SharedTensorObservation) -> &[f32] {
    let TensorObservationData::F32(values) = tensor.data() else {
        panic!("F32")
    };
    values
}

#[test]
fn exact_original_h_covers_all_target_controls_and_rejects_one_short_before_allocation() {
    let source = rows(5);
    let planned = plan(&source);
    let h = planned.initialization_peak_bytes();
    let full = CaptureTensorHostPlan::prepare(
        CaptureTensorGeometry::prepare(source.admission(), 0, CapturePhase::Prefill, 0, None)
            .unwrap(),
    )
    .unwrap();
    assert!(planned.tensor_peak_bytes() >= full.initialization_peak_bytes());
    assert!(
        planned.control_peak_bytes() >= 2 * size_of::<capture_tensor::prefill::TargetSlot>() as u64
    );
    for bytes in [h - 1, h] {
        let pool = capture_test_ledger(h, 0).unwrap();
        let (r, run) = fresh(&pool, bytes);
        let before = ledger(&pool);
        let allocations = CLAIM_ALLOCATIONS.get();
        let result = run.prepare_capture_run(&r, plan(&source));
        if bytes < h {
            assert!(
                matches!(result,Err(CaptureRunHostError::Memory(WorkingMemoryError::DomainAllowanceExceeded{required_bytes,available_bytes, .. })) if required_bytes==h&&available_bytes==bytes)
            );
            assert_eq!(ledger(&pool), before);
            assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
        } else {
            let mut bank = result.unwrap();
            let mut step = bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare_prefill(geometry())
                .unwrap();
            for i in 0..2 {
                step.begin_prefill_target(i, TensorDtype::F32, charge())
                    .unwrap();
            }
            let a = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
            let b = CapturePrefillRowAssembly::prepare(source.admission(), 1, geometry()).unwrap();
            let hold = ledger(&pool);
            write_fragment(&mut step, 0, &a, 0);
            write_fragment(&mut step, 1, &b, 0);
            assert_eq!(ledger(&pool), hold);
            assert_eq!(pool.payload_used_bytes().unwrap(), h);
            drop(step);
            drop(bank);
        }
        drop(r);
        drop(run);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn interleaved_targets_scatter_into_one_buffer_and_final_aliases_keep_original_h() {
    let source = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    let a = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let b = CapturePrefillRowAssembly::prepare(source.admission(), 1, geometry()).unwrap();
    for i in 0..2 {
        step.begin_prefill_target(i, TensorDtype::Bf16, charge())
            .unwrap();
    }
    write_fragment(&mut step, 0, &a, 0);
    write_fragment(&mut step, 1, &b, 0);
    let first = step.prefill_storage(0).unwrap();
    let second = step.prefill_storage(1).unwrap();
    assert_eq!((first.1, first.2, first.3), (12, 12, 4));
    assert_eq!((second.1, second.2), (5, 5));
    assert!(step.finish_prefill_targets().is_err());
    step.complete_prefill_chunk(0).unwrap();
    for chunk in 1..3 {
        write_fragment(&mut step, 1, &b, chunk);
        write_fragment(&mut step, 0, &a, chunk);
        assert_eq!(step.prefill_storage(0).unwrap().0, first.0);
        assert_eq!(step.prefill_storage(1).unwrap().0, second.0);
        step.complete_prefill_chunk(chunk).unwrap();
    }
    step.finish_prefill_targets().unwrap();
    let usage = charged(&step);
    assert_eq!(step.records()[0].charged.captures, 1);
    let full = tensor(&step.records()[0]);
    let preview = tensor(&step.records()[1]);
    assert_eq!(
        values(full),
        &[
            0., 1., 10., 11., 20., 21., 100., 101., 110., 111., 120., 121.
        ]
    );
    assert_eq!(values(preview), &[0., 1., 10., 11., 20.]);
    assert_eq!(values(full).as_ptr(), first.0);
    assert_eq!(values(preview).as_ptr(), second.0);
    let alias = full.clone();
    let delivered = step
        .finish(CaptureStepOutcome::Committed, usage, usage, 0.)
        .unwrap();
    drop(bank);
    drop(source);
    drop(r);
    drop(run);
    drop(delivered);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    assert_eq!(values(&alias)[6], 100.);
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn identity_order_duplicate_and_missing_target_reject_without_partial_row_advance() {
    let source = rows(5);
    let foreign = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    let a = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let b = CapturePrefillRowAssembly::prepare(source.admission(), 1, geometry()).unwrap();
    step.begin_prefill_target(0, TensorDtype::F32, charge())
        .unwrap();
    assert!(
        step.begin_prefill_target(0, TensorDtype::F32, charge())
            .is_err()
    );
    assert!(step.take_tensor(0).is_err());
    let wrong = CapturePrefillRowAssembly::prepare(foreign.admission(), 0, geometry()).unwrap();
    let wrong_fragment = wrong.fragment(0).unwrap();
    assert!(matches!(
        step.take_prefill_fragment(0, &wrong_fragment),
        Err(CaptureRunHostError::Prefill(
            CapturePrefillHostError::Identity
        ))
    ));
    let later = a.fragment(1).unwrap();
    assert!(matches!(
        step.take_prefill_fragment(0, &later),
        Err(CaptureRunHostError::Prefill(CapturePrefillHostError::Order))
    ));
    write_fragment(&mut step, 0, &a, 0);
    let first = a.fragment(0).unwrap();
    assert!(step.take_prefill_fragment(0, &first).is_err());
    assert!(step.complete_prefill_chunk(0).is_err());
    step.begin_prefill_target(1, TensorDtype::F16, charge())
        .unwrap();
    write_fragment(&mut step, 1, &b, 0);
    step.complete_prefill_chunk(0).unwrap();
    assert!(step.complete_prefill_chunk(0).is_err());
    assert!(step.take_prefill_fragment(0, &first).is_err());
    drop(step);
    assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
    drop(bank);
    drop(r);
    drop(run);
}

#[test]
fn incomplete_initialized_data_cannot_seal_and_abort_keeps_same_partial_payload() {
    let source = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    step.begin_prefill_target(0, TensorDtype::F32, charge())
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    let mut writer = step
        .take_prefill_fragment(0, &fragment)
        .unwrap()
        .prepare()
        .unwrap();
    writer.push_f32(9.5).unwrap();
    assert!(matches!(
        writer.finish(),
        Err(CaptureRunHostError::Prefill(
            CapturePrefillHostError::Incomplete { index: 0 }
        ))
    ));
    let partial = step.prefill_storage(0).unwrap();
    assert_eq!((partial.2, partial.3), (12, 1));
    assert!(step.take_prefill_fragment(0, &fragment).is_err());
    let usage = charged(&step);
    let error = step
        .finish(CaptureStepOutcome::Committed, usage, usage, 0.)
        .unwrap_err();
    assert!(matches!(error.error(), CaptureStepError::InvalidCompletion));
    let step = error.into_builder();
    let pending = step.into_aborted_pending(usage, usage, 0.);
    assert_eq!(pending.prefill_storage(0), Some(partial));
    drop(bank);
    // It is lifetime-free: the geometry and shared source can retire independently.
    drop(fragment);
    drop(assembly);
    drop(source);
    let delivered = pending.finish().unwrap();
    assert_eq!(delivered.as_ref().outcome, CaptureStepOutcome::Aborted);
    assert!(
        delivered
            .as_ref()
            .records
            .iter()
            .all(|r| r.payload.is_none())
    );
    drop(r);
    drop(run);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(delivered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn parent_close_and_unwind_retain_partial_payload_without_refilling_or_certifying() {
    let source = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    step.begin_prefill_target(0, TensorDtype::F32, charge())
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let mut writer = step
                .take_prefill_fragment(0, &fragment)
                .unwrap()
                .prepare()
                .unwrap();
            writer.push_f32(3.).unwrap();
            panic!("host caller unwind");
        }))
        .is_err()
    );
    let partial = step.prefill_storage(0).unwrap();
    let usage = charged(&step);
    let pending = step.into_aborted_pending(usage, usage, 0.);
    drop(bank);
    drop(run);
    let error = pending.finish().unwrap_err();
    assert!(matches!(
        error.error(),
        CaptureStepError::Memory(WorkingMemoryError::ExecutionFenced)
    ));
    let pending = error.into_pending();
    assert_eq!(pending.prefill_storage(0), Some(partial));
    drop(r);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(pending);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn global_preview_zero_needs_real_empty_claim_but_skips_all_fragment_values() {
    let source = rows(0);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    step.record_skip(
        0,
        CaptureSkipReason::NotInvoked,
        None,
        CaptureUsage::default(),
    )
    .unwrap();
    step.complete_prefill_chunk(0).unwrap();
    step.complete_prefill_chunk(1).unwrap();
    assert!(step.complete_prefill_chunk(2).is_err());
    step.begin_prefill_target(1, TensorDtype::F16, charge())
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 1, geometry()).unwrap();
    let fragment = assembly.fragment(2).unwrap();
    assert_eq!(fragment.output_elements(), 0);
    step.take_prefill_fragment(1, &fragment)
        .unwrap()
        .prepare()
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(step.prefill_storage(1).unwrap().2, 0);
    step.complete_prefill_chunk(2).unwrap();
    step.finish_prefill_targets().unwrap();
    assert!(values(tensor(&step.records()[1])).is_empty());
    let usage = charged(&step);
    let delivered = step
        .finish(CaptureStepOutcome::Committed, usage, usage, 0.)
        .unwrap();
    drop(bank);
    drop(r);
    drop(run);
    drop(delivered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn wrong_geometry_and_unsupported_axis_spend_only_the_original_frame_claim() {
    for (index, source) in [rows(5), source()].into_iter().enumerate() {
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut geometry = geometry();
        if index == 0 {
            geometry.cached_positions = 99;
        }
        assert!(
            bank.begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare_prefill(geometry)
                .is_err()
        );
        assert_eq!(bank.spent_steps(), 1);
        assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
        drop(bank);
        drop(r);
        drop(run);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn closure_or_quarantine_during_fill_poison_target_and_preserve_original_buffer() {
    for quarantine in [false, true] {
        let source = rows(5);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill(geometry())
            .unwrap();
        step.begin_prefill_target(0, TensorDtype::F32, charge())
            .unwrap();
        let assembly =
            CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
        let fragment = assembly.fragment(0).unwrap();
        let mut writer = step
            .take_prefill_fragment(0, &fragment)
            .unwrap()
            .prepare()
            .unwrap();
        writer.push_f32(17.).unwrap();
        if quarantine {
            drop(run.scope().unwrap());
        } else {
            drop(run);
        }
        assert!(matches!(
            writer.push_f32(99.),
            Err(CaptureRunHostError::Memory(
                WorkingMemoryError::ExecutionFenced
            ))
        ));
        assert!(writer.finish().is_err());
        let partial = step.prefill_storage(0).unwrap();
        assert_eq!((partial.2, partial.3), (12, 1));
        let usage = charged(&step);
        let pending = step.into_aborted_pending(usage, usage, 0.);
        drop(bank);
        drop(r);
        let pending = pending.finish().unwrap_err().into_pending();
        assert_eq!(pending.prefill_storage(0), Some(partial));
        assert_eq!(pool.payload_used_bytes().unwrap(), h);
        drop(pending);
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            if quarantine { h } else { 0 }
        );
    }
}

#[test]
fn zero_full_shape_has_zero_coverage_but_still_requires_initial_claim_and_all_chunks() {
    let mut point = point();
    point.axes = Some(vec![
        TensorAxis {
            name: "row".into(),
            dimension: SymbolicDimension::Sequence,
        },
        TensorAxis {
            name: "width".into(),
            dimension: SymbolicDimension::Known(0),
        },
    ]);
    let mut raw = raw();
    raw.selections.truncate(1);
    let source = admit(raw, point, 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    step.begin_prefill_target(0, TensorDtype::F32, charge())
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    step.take_prefill_fragment(0, &fragment)
        .unwrap()
        .prepare()
        .unwrap()
        .finish()
        .unwrap();
    assert!(step.finish_prefill_targets().is_err());
    for chunk in 0..3 {
        step.complete_prefill_chunk(chunk).unwrap();
    }
    step.finish_prefill_targets().unwrap();
    assert_eq!(tensor(&step.records()[0]).shape(), &[3, 0]);
    assert!(values(tensor(&step.records()[0])).is_empty());
    let usage = charged(&step);
    let delivered = step
        .finish(CaptureStepOutcome::Committed, usage, usage, 0.)
        .unwrap();
    drop(bank);
    drop(r);
    drop(run);
    drop(delivered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn inactive_prefill_row_checks_full_inference_frontier_before_construction() {
    for (cached, overflows) in [(u64::MAX - 4, false), (u64::MAX - 3, true)] {
        let point = point();
        let catalog = ObservationCatalog {
            schema_version: 1,
            points: vec![point.clone()],
            completeness: DescriptionCompleteness::Complete,
        };
        let support = ObservationSupportReport {
            schema_version: 1,
            capture: Default::default(),
            points: vec![ObservationSupport {
                path: point.path,
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        };
        let caps = CaptureCapabilities {
            transformations: vec![CaptureTransformKind::FullTensor],
            max_histogram_bins: 0,
            conditions: vec![],
        };
        let mut raw = raw();
        raw.selections.truncate(1);
        raw.selections[0].schedule.prefill = false;
        // This is a genuine capture admission: its last actual coordinate is
        // F + P + (M - 1), which still fits in both cases.
        let source = SharedCapturePlan::new(
            raw.admit_with_text_origin(
                &catalog,
                &support,
                &caps,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 3,
                    max_predictions: 1,
                },
                CaptureTextOrigin {
                    cached_positions: cached,
                },
            )
            .unwrap(),
        );
        let mut inference = geometry();
        inference.cached_positions = cached;
        inference.max_output_tokens = 1;
        assert_eq!(inference.validate().is_err(), overflows);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let mut bank = run
            .prepare_capture_run(&reservation, plan(&source))
            .unwrap();
        let before = ledger(&pool);
        let claim = bank.begin_step(CapturePhase::Prefill, 0).unwrap();
        assert!(
            claim.row[1..]
                .iter()
                .all(|state| *state == ClaimState::Unavailable)
        );
        let result = claim.prepare_prefill(inference);
        if overflows {
            assert!(matches!(
                result,
                Err(CaptureRunHostError::Prefill(
                    CapturePrefillHostError::Geometry(CapturePrefillGeometryError::Overflow)
                ))
            ));
        } else {
            let mut step = result.unwrap();
            for chunk in 0..3 {
                step.complete_prefill_chunk(chunk).unwrap();
            }
            step.finish_prefill_targets().unwrap();
            assert!(step.records().iter().all(|record| matches!(
                record.outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            )));
            drop(step);
        }
        assert_eq!(ledger(&pool), before);
        assert_eq!(bank.spent_steps(), 1);
        assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
        drop(bank);
        drop(reservation);
        drop(run);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

mod shared_retirement;
