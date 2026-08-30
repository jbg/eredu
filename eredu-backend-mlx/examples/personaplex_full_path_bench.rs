use std::{path::PathBuf, time::Instant};

use eredu_backend_mlx::native::ExecutionContext;
use eredu_backend_mlx::{
    codec::mimi::load,
    native::{MlxRealtimeBackend, MlxRealtimeInput},
    MlxTensor,
};
use eredu_codec::mimi::Mimi;
use eredu_core::{
    load_realtime_model,
    scheduler::{RequestId, SchedulerLimits},
    RealtimeModel, RealtimeSampling, RealtimeScheduler,
};
use safemlx::{transforms::eval, Array, Device, DeviceType, Stream};

const SAMPLE_RATE: f64 = 24_000.0;
const FRAME_RATE: f64 = 12.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("EREDU_PERSONAPLEX_DIR").map(PathBuf::from))
        .expect("usage: personaplex_full_path_bench <model-dir> <mimi.safetensors> [frames]");
    let mimi_path = args
        .get(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("EREDU_MIMI_PATH").map(PathBuf::from))
        .expect("usage: personaplex_full_path_bench <model-dir> <mimi.safetensors> [frames]");
    let frames = args
        .get(2)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(16);
    let frame_samples = (SAMPLE_RATE / FRAME_RATE) as i32;
    let audio_s = frames as f64 / FRAME_RATE;

    println!("model_dir={}", model_dir.display());
    println!("mimi_path={}", mimi_path.display());
    println!("frames={frames}");
    println!("frame_samples={frame_samples}");
    println!("audio_s={audio_s:.3}");

    let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = ctx.stream();
    let weights_ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let weights_stream = weights_ctx.stream();

    let load_start = Instant::now();
    let preparation = eredu_architectures::moshi::prepare_realtime_model(&model_dir)?;
    let mut model =
        load_realtime_model(MlxRealtimeBackend::new(stream, weights_stream), preparation)?;
    let config = model.speech_config();
    let input_audio_codebooks = config.input_audio_codebooks() as i32;
    let generated_audio_codebooks = config.generated_audio_codebooks() as i32;
    let depth_audio_codebooks = config.depth_audio_codebooks() as i32;
    let mut mimi = load(
        &mimi_path,
        Some(input_audio_codebooks.max(generated_audio_codebooks)),
        stream,
    )?;
    stream.synchronize()?;
    println!("load_s={:.3}", load_start.elapsed().as_secs_f64());
    println!(
        "input_codebooks={} generated_codebooks={} depth_codebooks={}",
        input_audio_codebooks, generated_audio_codebooks, depth_audio_codebooks
    );

    let pcm_frame = MlxTensor::from_array(Array::zeros::<f32>(&[1, 1, frame_samples], stream)?);
    warmup(&mut model, &mut mimi, &pcm_frame, stream)?;

    let (elapsed, encoded_frames, emitted_frames) =
        run_full_path(&mut model, &mut mimi, &pcm_frame, frames, stream)?;
    println!("encoded_frames={encoded_frames} emitted_frames={emitted_frames}");
    report("full_path_pcm_to_pcm", elapsed, audio_s, frames);

    Ok(())
}

fn warmup(
    model: &mut RealtimeModel<MlxRealtimeBackend>,
    mimi: &mut Mimi<MlxTensor>,
    pcm_frame: &MlxTensor,
    stream: &Stream,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = run_full_path(model, mimi, pcm_frame, 3, stream)?;
    Ok(())
}

fn run_full_path(
    model: &mut RealtimeModel<MlxRealtimeBackend>,
    mimi: &mut Mimi<MlxTensor>,
    pcm_frame: &MlxTensor,
    frames: i32,
    stream: &Stream,
) -> Result<(f64, i32, i32), Box<dyn std::error::Error>> {
    let request = RequestId::new(1);
    let mut scheduler = RealtimeScheduler::new(model, SchedulerLimits::new(1, 1)?)?;
    scheduler.register_request(model, request, RealtimeSampling::greedy())?;
    mimi.reset_encode_state();
    mimi.reset_decode_state();

    let start = Instant::now();
    let mut encoded_frames = 0;
    let mut emitted_frames = 0;
    for _ in 0..frames {
        let Some(input_tokens) = mimi.encode_step(pcm_frame, stream)? else {
            continue;
        };
        encoded_frames += 1;
        scheduler.enqueue(
            model,
            request,
            MlxRealtimeInput::encoded_audio(input_tokens.as_array()),
        )?;
        let output = scheduler
            .run_queued(model)?
            .pop()
            .expect("one queued realtime frame")
            .into_parts()
            .1;
        if let Some(output_tokens) = output.output_audio_tokens {
            let output_tokens = MlxTensor::from_array(output_tokens);
            let pcm = mimi.decode_step(&output_tokens, stream)?;
            eval([
                &output.text_token,
                &output.sampled_audio_tokens,
                pcm.as_array(),
            ])?;
            emitted_frames += 1;
        } else {
            eval([&output.text_token, &output.sampled_audio_tokens])?;
        }
        stream.synchronize()?;
    }
    let elapsed = start.elapsed().as_secs_f64();
    Ok((elapsed, encoded_frames, emitted_frames))
}

fn report(name: &str, elapsed_s: f64, audio_s: f64, frames: i32) {
    let realtime_factor = elapsed_s / audio_s;
    let realtime_multiple = audio_s / elapsed_s;
    let per_frame_ms = elapsed_s * 1_000.0 / frames as f64;
    println!(
        "{name}_s={elapsed_s:.6} {name}_rtf={realtime_factor:.4} {name}_x_realtime={realtime_multiple:.2} {name}_ms_per_frame={per_frame_ms:.3}"
    );
}
