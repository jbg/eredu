use std::time::{Duration, Instant};

use safemlx::{
    transforms::{async_eval, async_eval_timed},
    Array, Device, DeviceType, Stream,
};

fn matmul(size: i32, stream: &Stream) -> safemlx::error::Result<Array> {
    let lhs = Array::ones::<f32>(&[size, size], stream)?;
    let rhs = Array::ones::<f32>(&[size, size], stream)?;
    lhs.matmul(&rhs, stream)
}

fn main() -> safemlx::error::Result<()> {
    let device = Device::new(DeviceType::Gpu, 0);
    let stream = Stream::new_with_device(&device);

    // Warm up kernels and timing resource paths.
    async_eval_timed([&matmul(512, &stream)?], &stream)?.elapsed()?;

    let output = matmul(2048, &stream)?;
    let before = Instant::now();
    let timed = async_eval_timed([&output], &stream)?;
    let host_submission = before.elapsed();
    let device_execution = timed.elapsed()?;
    println!(
        "nonblocking: host submission {host_submission:?}, device execution {device_execution:?}"
    );

    let mut samples = Vec::new();
    for _ in 0..8 {
        let output = matmul(1024, &stream)?;
        samples.push(async_eval_timed([&output], &stream)?.elapsed()?);
    }
    let min = *samples.iter().min().unwrap();
    let max = *samples.iter().max().unwrap();
    let mean = samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64;
    println!(
        "stability: n={}, min={min:?}, mean={:?}, max={max:?}",
        samples.len(),
        Duration::from_secs_f64(mean)
    );

    let weights = Array::ones::<f32>(&[1536, 1536], &stream)?;
    let mut phase = Array::ones::<f32>(&[1536, 1536], &stream)?;
    let before = Instant::now();
    let mut phases = Vec::new();
    for _ in 0..4 {
        phase = phase.matmul(&weights, &stream)?;
        phases.push(async_eval_timed([&phase], &stream)?);
    }
    let phase_submission = before.elapsed();
    let phase_times = phases
        .iter()
        .map(|phase| phase.elapsed())
        .collect::<Result<Vec<_>, _>>()?;
    println!("four phases: all submitted in {phase_submission:?}; device times {phase_times:?}");

    const SPANS: usize = 100;
    let untimed_stream = Stream::new_with_device(&device);
    let timed_stream = Stream::new_with_device(&device);
    async_eval([&matmul(64, &untimed_stream)?])?;
    untimed_stream.synchronize()?;
    async_eval_timed([&matmul(64, &timed_stream)?], &timed_stream)?.synchronize()?;
    let untimed_outputs = (0..SPANS)
        .map(|_| matmul(64, &untimed_stream))
        .collect::<Result<Vec<_>, _>>()?;
    let timed_outputs = (0..SPANS)
        .map(|_| matmul(64, &timed_stream))
        .collect::<Result<Vec<_>, _>>()?;

    let before = Instant::now();
    for output in &untimed_outputs {
        async_eval([output])?;
    }
    let untimed_submission = before.elapsed();

    let before = Instant::now();
    let tokens = timed_outputs
        .iter()
        .map(|output| async_eval_timed([output], &timed_stream))
        .collect::<Result<Vec<_>, _>>()?;
    let timed_submission = before.elapsed();
    untimed_stream.synchronize()?;
    for token in &tokens {
        token.synchronize()?;
    }
    println!(
        "submission overhead: untimed={:?}/call, timed={:?}/call, incremental={:?}/span",
        untimed_submission / SPANS as u32,
        timed_submission / SPANS as u32,
        timed_submission.saturating_sub(untimed_submission) / SPANS as u32,
    );
    Ok(())
}
