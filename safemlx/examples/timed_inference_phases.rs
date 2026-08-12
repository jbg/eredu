use safemlx::{transforms::async_eval_timed, Array, Device, DeviceType, Stream, TimedEvaluation};

fn main() -> safemlx::error::Result<()> {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let hidden = Array::ones::<f32>(&[1024, 1024], &stream)?;
    let weights = Array::ones::<f32>(&[1024, 1024], &stream)?;

    let context = hidden.matmul(&weights, &stream)?;
    let context_time = async_eval_timed([&context], &stream)?;

    let transformer = context.square(&stream)?;
    let transformer_time = async_eval_timed([&transformer], &stream)?;

    let vocabulary = transformer.matmul(&weights, &stream)?;
    let vocabulary_time = async_eval_timed([&vocabulary], &stream)?;

    let verification = safemlx::ops::indexing::argmax(&vocabulary, false, &stream)?;
    let verification_time = async_eval_timed([&verification], &stream)?;

    // Every phase was submitted before the first host wait. Resolve all
    // durations only after target/draft or other host work has been queued.
    for (name, timing) in [
        ("context", context_time),
        ("transformer", transformer_time),
        ("vocabulary", vocabulary_time),
        ("verification", verification_time),
    ] {
        print_timing(name, timing)?;
    }
    Ok(())
}

fn print_timing(name: &str, timing: TimedEvaluation) -> safemlx::error::Result<()> {
    println!("{name}: {:?}", timing.elapsed()?);
    Ok(())
}
