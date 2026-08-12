use std::time::{Duration, Instant};

use safemlx::{
    ops::indexing::IndexOp,
    transforms::{async_eval, async_eval_timed, async_eval_with_event},
    Array, Device, DeviceType, Stream,
};

fn stream(device_type: DeviceType) -> Stream {
    Stream::new_with_device(&Device::new(device_type, 0))
}

fn matmul(size: i32, stream: &Stream) -> Array {
    let lhs = Array::ones::<f32>(&[size, size], stream).unwrap();
    let rhs = Array::ones::<f32>(&[size, size], stream).unwrap();
    lhs.matmul(&rhs, stream).unwrap()
}

#[test]
fn cpu_timing_is_positive_and_stable() {
    let stream = stream(DeviceType::Cpu);
    let output = matmul(512, &stream);
    let timed = async_eval_timed([&output], &stream).unwrap();

    let first = timed.elapsed().unwrap();
    let second = timed.elapsed().unwrap();
    assert!(first > Duration::ZERO);
    assert_eq!(first, second);
    assert_eq!(timed.try_elapsed().unwrap(), Some(first));
}

#[test]
fn cpu_larger_workload_reports_materially_longer() {
    let stream = stream(DeviceType::Cpu);
    let small = matmul(96, &stream);
    let small_elapsed = async_eval_timed([&small], &stream)
        .unwrap()
        .elapsed()
        .unwrap();
    let large = matmul(640, &stream);
    let large_elapsed = async_eval_timed([&large], &stream)
        .unwrap()
        .elapsed()
        .unwrap();

    assert!(
        large_elapsed > small_elapsed * 2,
        "small={small_elapsed:?}, large={large_elapsed:?}"
    );
}

#[test]
fn submission_and_timing_span_creation_do_not_wait() {
    let stream = stream(DeviceType::Cpu);
    let output = matmul(1536, &stream);

    let before = Instant::now();
    let timed = async_eval_timed([&output], &stream).unwrap();
    let submit_elapsed = before.elapsed();

    assert_eq!(timed.try_elapsed().unwrap(), None);
    assert!(
        submit_elapsed < Duration::from_millis(250),
        "submission unexpectedly blocked for {submit_elapsed:?}"
    );
    assert!(timed.elapsed().unwrap() > Duration::ZERO);
}

#[test]
fn try_elapsed_is_none_then_matches_elapsed() {
    let stream = stream(DeviceType::Cpu);
    let output = matmul(2048, &stream);
    let timed = async_eval_timed([&output], &stream).unwrap();

    assert_eq!(timed.try_elapsed().unwrap(), None);
    while !timed.is_complete().unwrap() {
        std::thread::yield_now();
    }
    let queried = timed.try_elapsed().unwrap().unwrap();
    assert_eq!(queried, timed.elapsed().unwrap());
}

#[test]
fn earlier_same_stream_work_is_excluded() {
    let stream = stream(DeviceType::Cpu);

    let reference = matmul(768, &stream);
    let reference_elapsed = async_eval_timed([&reference], &stream)
        .unwrap()
        .elapsed()
        .unwrap();

    let unrelated = matmul(768, &stream);
    async_eval([&unrelated]).unwrap();
    let measured = matmul(96, &stream);
    let measured_elapsed = async_eval_timed([&measured], &stream)
        .unwrap()
        .elapsed()
        .unwrap();

    assert!(
        measured_elapsed < reference_elapsed / 2,
        "prior work leaked into interval: measured={measured_elapsed:?}, reference={reference_elapsed:?}"
    );
}

#[test]
fn cross_stream_dependency_latency_is_supported() {
    let producer = stream(DeviceType::Cpu);
    let consumer = stream(DeviceType::Cpu);
    let produced = matmul(512, &producer);
    let consumed = produced.square(&consumer).unwrap();

    let timed = async_eval_timed([&consumed], &consumer).unwrap();
    assert!(timed.elapsed().unwrap() > Duration::ZERO);
    assert_eq!(
        consumed
            .index_device((0, 0), &consumer)
            .item::<f32>(&consumer),
        512.0f32 * 512.0
    );
}

#[test]
fn mismatched_stream_is_rejected_before_submission() {
    let graph_stream = stream(DeviceType::Cpu);
    let wrong_stream = stream(DeviceType::Cpu);
    let output = matmul(128, &graph_stream);

    let error = async_eval_timed([&output], &wrong_stream).unwrap_err();
    let diagnostic = error.what().split(" at ").next().unwrap();
    assert!(diagnostic.contains("[async_eval_with_timing] Requested Stream(Device(cpu, 0)"));
    assert!(diagnostic.contains("but evaluation is rooted on Stream(Device(cpu, 0)"));

    // Rejection happened before graph submission, so the correct stream can
    // still submit the same output as a timed evaluation.
    assert!(
        async_eval_timed([&output], &graph_stream)
            .unwrap()
            .elapsed()
            .unwrap()
            > Duration::ZERO
    );
}

#[test]
fn empty_and_completed_evaluations_are_zero_duration() {
    let stream = stream(DeviceType::Cpu);
    let empty = async_eval_timed(std::iter::empty(), &stream).unwrap();
    assert_eq!(empty.try_elapsed().unwrap(), Some(Duration::ZERO));
    assert_eq!(empty.elapsed().unwrap(), Duration::ZERO);

    let available = Array::from_slice(&[1.0f32, 2.0], &[2]);
    let completed = async_eval_timed([&available], &stream).unwrap();
    assert_eq!(completed.try_elapsed().unwrap(), Some(Duration::ZERO));
    assert_eq!(completed.elapsed().unwrap(), Duration::ZERO);
}

#[test]
fn ordinary_untimed_completion_remains_correct() {
    let stream = stream(DeviceType::Cpu);
    let output = matmul(256, &stream);
    let event = async_eval_with_event([&output]).unwrap();
    event.synchronize().unwrap();
    assert!(event.is_complete().unwrap());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit Metal timestamp test; run outside the sandbox on a Metal host"]
fn metal_timestamp_is_native_nonblocking_and_plausible() {
    let stream = stream(DeviceType::Gpu);
    let output = matmul(2048, &stream);
    let before = Instant::now();
    let timed = async_eval_timed([&output], &stream).unwrap();
    let submission = before.elapsed();

    assert_eq!(timed.try_elapsed().unwrap(), None);
    assert!(submission < Duration::from_millis(250));
    let elapsed = timed.elapsed().unwrap();
    assert!(
        elapsed > Duration::from_micros(10),
        "Metal reported {elapsed:?}"
    );
    assert!(elapsed < Duration::from_secs(30));
    assert_eq!(timed.elapsed().unwrap(), elapsed);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit Metal scaling/exclusion test; run outside the sandbox"]
fn metal_scales_and_excludes_prior_work() {
    let stream = stream(DeviceType::Gpu);
    let large = matmul(2048, &stream);
    let large_elapsed = async_eval_timed([&large], &stream)
        .unwrap()
        .elapsed()
        .unwrap();

    let unrelated = matmul(2048, &stream);
    async_eval([&unrelated]).unwrap();
    let small = matmul(256, &stream);
    let small_elapsed = async_eval_timed([&small], &stream)
        .unwrap()
        .elapsed()
        .unwrap();

    assert!(
        large_elapsed > small_elapsed * 2,
        "small={small_elapsed:?}, large={large_elapsed:?}"
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit Metal multi-phase test; run outside the sandbox"]
fn metal_phases_are_all_submitted_before_resolution() {
    let stream = stream(DeviceType::Gpu);
    let weights = Array::ones::<f32>(&[1536, 1536], &stream).unwrap();
    let mut phase = Array::ones::<f32>(&[1536, 1536], &stream).unwrap();
    let before = Instant::now();
    let mut tokens = Vec::new();
    for _ in 0..4 {
        phase = phase.matmul(&weights, &stream).unwrap();
        tokens.push(async_eval_timed([&phase], &stream).unwrap());
    }
    let submission = before.elapsed();

    assert!(submission < Duration::from_millis(500));
    assert!(tokens.iter().any(|token| !token.is_complete().unwrap()));
    for token in &tokens {
        assert!(token.elapsed().unwrap() > Duration::ZERO);
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit cross-device validation test; requires Metal"]
fn metal_rejects_cross_device_timing_stream() {
    let cpu = stream(DeviceType::Cpu);
    let gpu = stream(DeviceType::Gpu);
    let output = matmul(128, &cpu);
    let error = async_eval_timed([&output], &gpu).unwrap_err();
    assert!(error.what().contains("Requested Stream(Device(gpu, 0)"));
    assert!(error.what().contains("rooted on Stream(Device(cpu, 0)"));
}

#[test]
#[ignore = "submission-overhead benchmark; run with --ignored --nocapture"]
fn timing_submission_overhead_benchmark() {
    const SPANS: usize = 200;
    let stream = stream(DeviceType::Cpu);
    let outputs = (0..SPANS).map(|_| matmul(32, &stream)).collect::<Vec<_>>();
    let before = Instant::now();
    let mut tokens = Vec::with_capacity(SPANS);
    for output in &outputs {
        tokens.push(async_eval_timed([output], &stream).unwrap());
    }
    let submission = before.elapsed();
    for token in &tokens {
        token.synchronize().unwrap();
    }
    eprintln!(
        "{SPANS} timed spans: {:?}/span submission overhead",
        submission / SPANS as u32
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Metal submission-overhead benchmark; run outside sandbox with --nocapture"]
fn metal_timing_submission_overhead_benchmark() {
    const SPANS: usize = 100;
    let untimed_stream = stream(DeviceType::Gpu);
    let timed_stream = stream(DeviceType::Gpu);
    async_eval([&matmul(64, &untimed_stream)]).unwrap();
    untimed_stream.synchronize().unwrap();
    async_eval_timed([&matmul(64, &timed_stream)], &timed_stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let untimed_outputs = (0..SPANS)
        .map(|_| matmul(64, &untimed_stream))
        .collect::<Vec<_>>();
    let timed_outputs = (0..SPANS)
        .map(|_| matmul(64, &timed_stream))
        .collect::<Vec<_>>();

    let before = Instant::now();
    for output in &untimed_outputs {
        async_eval([output]).unwrap();
    }
    let untimed_submission = before.elapsed();

    let before = Instant::now();
    let tokens = timed_outputs
        .iter()
        .map(|output| async_eval_timed([output], &timed_stream).unwrap())
        .collect::<Vec<_>>();
    let timed_submission = before.elapsed();
    untimed_stream.synchronize().unwrap();
    for token in &tokens {
        token.synchronize().unwrap();
    }

    eprintln!(
        "Metal {SPANS} warmed spans: untimed {:?}/submission, timed {:?}/submission, incremental {:?}/span",
        untimed_submission / SPANS as u32,
        timed_submission / SPANS as u32,
        timed_submission.saturating_sub(untimed_submission) / SPANS as u32,
    );
}
