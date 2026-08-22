use std::{error::Error, fs, io, io::Write, path::Path, path::PathBuf, time::Instant};

use eredu_backend_mlx::{
    codec::mimi::load,
    composition::mlx::realtime::{generate_encoded_greedy, personaplex_prompt, MlxRealtimeBackend},
    MlxTensor,
};
use eredu_codec::mimi::Mimi;
use eredu_core::load_realtime_model;
use safemlx::{Array, Device, DeviceType, Dtype, ExecutionContext, Stream};
use serde_json::json;

const SAMPLE_RATE: u32 = 24_000;
const FRAME_SAMPLES: usize = 1_920;

type EvalResult<T> = Result<T, Box<dyn Error>>;

fn main() -> EvalResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 5 {
        return Err(invalid(
            "usage: personaplex_quantization_eval <dense-model-dir> <quantized-model-dir> <mimi.safetensors> <input-mono-24khz-f32le> <output-dir> [frames]",
        ));
    }
    let dense_dir = PathBuf::from(&args[0]);
    let quantized_dir = PathBuf::from(&args[1]);
    let mimi_path = PathBuf::from(&args[2]);
    let input_path = PathBuf::from(&args[3]);
    let output_dir = PathBuf::from(&args[4]);
    let requested_frames = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if output_dir.exists() {
        return Err(invalid(format!(
            "output directory already exists: {}",
            output_dir.display()
        )));
    }

    let mut pcm = read_f32le(&input_path)?;
    let frames = requested_frames
        .unwrap_or(pcm.len() / FRAME_SAMPLES)
        .min(pcm.len() / FRAME_SAMPLES);
    if frames == 0 {
        return Err(invalid("input contains no complete 80 ms frame"));
    }
    pcm.truncate(frames * FRAME_SAMPLES);

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = gpu.stream();
    let weights_stream = cpu.stream();
    let mut mimi = load(
        &mimi_path,
        Some(personaplex_prompt::AUDIO_TOKENS_PER_STREAM),
        stream,
    )?;
    let pcm_array =
        MlxTensor::from_array(Array::from_slice(&pcm, &[1, 1, pcm.len() as i32]).copy(stream)?);
    let input_tokens = mimi.encode(&pcm_array, stream)?;
    safemlx::transforms::eval([input_tokens.as_array()])?;

    let dense_load = Instant::now();
    let mut dense =
        load_realtime_model(MlxRealtimeBackend::new(stream, weights_stream), &dense_dir)?;
    let dense_load_seconds = dense_load.elapsed().as_secs_f64();
    let dense_start = Instant::now();
    let dense_output = generate_encoded_greedy(&mut dense, input_tokens.as_array())?;
    safemlx::transforms::eval([&dense_output.text_tokens, &dense_output.audio_tokens])?;
    stream.synchronize()?;
    let dense_seconds = dense_start.elapsed().as_secs_f64();
    drop(dense);
    safemlx::transforms::compile::clear_cache()?;

    let quantized_load = Instant::now();
    let mut quantized = load_realtime_model(
        MlxRealtimeBackend::new(stream, weights_stream),
        &quantized_dir,
    )?;
    let quantized_load_seconds = quantized_load.elapsed().as_secs_f64();
    let quantized_start = Instant::now();
    let quantized_output = generate_encoded_greedy(&mut quantized, input_tokens.as_array())?;
    safemlx::transforms::eval([
        &quantized_output.text_tokens,
        &quantized_output.audio_tokens,
    ])?;
    stream.synchronize()?;
    let quantized_seconds = quantized_start.elapsed().as_secs_f64();

    let text_agreement = token_agreement(&dense_output.text_tokens, &quantized_output.text_tokens)?;
    let audio_agreement =
        token_agreement(&dense_output.audio_tokens, &quantized_output.audio_tokens)?;
    let dense_pcm = decode(&mut mimi, &dense_output.audio_tokens, stream)?;
    let quantized_pcm = decode(&mut mimi, &quantized_output.audio_tokens, stream)?;

    fs::create_dir(&output_dir)?;
    write_wav_pcm16(&output_dir.join("input.wav"), &pcm, SAMPLE_RATE)?;
    write_wav_pcm16(&output_dir.join("dense.wav"), &dense_pcm, SAMPLE_RATE)?;
    write_wav_pcm16(
        &output_dir.join("quantized.wav"),
        &quantized_pcm,
        SAMPLE_RATE,
    )?;
    fs::write(
        output_dir.join("metrics.json"),
        serde_json::to_vec_pretty(&json!({
            "format_version": 2,
            "methodology": "Both artifacts use the public realtime loader and canonical greedy scheduler over identical Mimi input tokens.",
            "input_frames": frames,
            "dense": {
                "load_seconds": dense_load_seconds,
                "generation_seconds": dense_seconds,
            },
            "quantized": {
                "load_seconds": quantized_load_seconds,
                "generation_seconds": quantized_seconds,
            },
            "agreement": {
                "text_tokens": text_agreement,
                "generated_audio_tokens": audio_agreement,
            },
        }))?,
    )?;

    println!(
        "realtime quantization evaluation complete: text agreement={text_agreement:.4}, audio agreement={audio_agreement:.4}, output={}",
        output_dir.display()
    );
    Ok(())
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn read_f32le(path: &Path) -> EvalResult<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid(format!(
            "raw f32le input length must be divisible by 4, got {} bytes",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn integer_values(array: &Array) -> EvalResult<Vec<i64>> {
    let evaluated = array.evaluated()?;
    match array.dtype() {
        Dtype::Int32 => Ok(evaluated
            .as_slice::<i32>()
            .iter()
            .map(|value| i64::from(*value))
            .collect()),
        Dtype::Uint32 => Ok(evaluated
            .as_slice::<u32>()
            .iter()
            .map(|value| i64::from(*value))
            .collect()),
        Dtype::Int64 => Ok(evaluated.as_slice::<i64>().to_vec()),
        Dtype::Uint64 => evaluated
            .as_slice::<u64>()
            .iter()
            .map(|value| i64::try_from(*value).map_err(Into::into))
            .collect(),
        dtype => Err(invalid(format!("expected integer tokens, got {dtype:?}"))),
    }
}

fn token_agreement(left: &Array, right: &Array) -> EvalResult<f64> {
    if left.shape() != right.shape() {
        return Ok(0.0);
    }
    let left = integer_values(left)?;
    let right = integer_values(right)?;
    let matches = left
        .iter()
        .zip(&right)
        .filter(|(left, right)| left == right)
        .count();
    Ok(matches as f64 / left.len().max(1) as f64)
}

fn decode(mimi: &mut Mimi<MlxTensor>, tokens: &Array, stream: &Stream) -> EvalResult<Vec<f32>> {
    mimi.reset_decode_state();
    let tokens = MlxTensor::from_array(tokens.clone());
    let decoded = mimi.decode(&tokens, stream)?.into_array();
    let decoded = if decoded.dtype() == Dtype::Float32 {
        decoded
    } else {
        decoded.as_dtype(Dtype::Float32, stream)?
    };
    Ok(decoded.evaluated()?.as_slice::<f32>().to_vec())
}

fn write_wav_pcm16(path: &Path, samples: &[f32], sample_rate: u32) -> EvalResult<()> {
    let data_bytes = u32::try_from(
        samples
            .len()
            .checked_mul(2)
            .ok_or_else(|| invalid("WAV data size overflow"))?,
    )?;
    let riff_bytes = 36u32
        .checked_add(data_bytes)
        .ok_or_else(|| invalid("WAV RIFF size overflow"))?;
    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_bytes.to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for sample in samples {
        let sample = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}
