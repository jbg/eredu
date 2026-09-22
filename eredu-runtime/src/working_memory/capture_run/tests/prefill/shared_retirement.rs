use super::*;

#[test]
fn original_preview_escapes_frame_and_concurrent_final_aliases_keep_same_h() {
    let source = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (reservation, run) = fresh(&pool, h);
    let mut bank = run
        .prepare_capture_run(&reservation, plan(&source))
        .unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    let a = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let b = CapturePrefillRowAssembly::prepare(source.admission(), 1, geometry()).unwrap();
    for index in 0..2 {
        step.begin_prefill_target(index, TensorDtype::F32, charge())
            .unwrap();
    }
    for chunk in 0..3 {
        write_fragment(&mut step, 0, &a, chunk);
        write_fragment(&mut step, 1, &b, chunk);
        step.complete_prefill_chunk(chunk).unwrap();
    }
    step.finish_prefill_targets().unwrap();
    let preview = tensor(&step.records()[1]).clone();
    assert_eq!(values(&preview), &[0., 1., 10., 11., 20.]);
    let pointer = values(&preview).as_ptr() as usize;
    let usage = charged(&step);
    let delivered = step
        .finish(CaptureStepOutcome::Committed, usage, usage, 0.)
        .unwrap();
    let frame_alias = delivered.clone();
    drop((bank, source, reservation, run, delivered));
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(frame_alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(9));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let alias = preview.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                assert_eq!(values(&alias).as_ptr() as usize, pointer);
                assert_eq!(values(&alias), &[0., 1., 10., 11., 20.]);
                drop(alias);
            })
        })
        .collect();
    drop(preview);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    barrier.wait();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}
