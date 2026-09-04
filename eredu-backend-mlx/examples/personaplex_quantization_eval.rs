use std::{error::Error, path::Path, path::PathBuf};

#[path = "support/realtime.rs"]
mod realtime_support;

use eredu_backend_mlx::native::ExecutionContext;
use eredu_backend_mlx::{codec::mimi::load, native::MlxRealtimeExecutionContext};
use eredu_evaluation::{
    run_personaplex_quantization, PersonaPlexEvaluationOptions, PersonaPlexEvaluationPaths,
};
use safemlx::{Device, DeviceType};

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 7 {
        return Err(
            "usage: personaplex_quantization_eval <dense-model-dir> <quantized-model-dir> <mimi.safetensors> <text-tokenizer.model> <voice-prompt-mono-24khz-f32le> <input-mono-24khz-f32le> <output-dir> [frames] [text-prompt] [sampling-seed]"
                .into(),
        );
    }
    let paths = PersonaPlexEvaluationPaths {
        dense_model: PathBuf::from(&args[0]),
        quantized_model: PathBuf::from(&args[1]),
        text_tokenizer: PathBuf::from(&args[3]),
        voice_prompt: PathBuf::from(&args[4]),
        input: PathBuf::from(&args[5]),
        output: PathBuf::from(&args[6]),
    };
    let defaults = PersonaPlexEvaluationOptions::default();
    let options = PersonaPlexEvaluationOptions {
        frames: args
            .get(7)
            .map(|value| value.parse::<usize>())
            .transpose()?,
        text_prompt: args.get(8).cloned().unwrap_or(defaults.text_prompt),
        sampling_seed: args
            .get(9)
            .map(|value| value.parse::<u64>())
            .transpose()?
            .unwrap_or(defaults.sampling_seed),
    };

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = gpu.stream();
    let weights_stream = cpu.stream();
    let mut mimi = load(
        Path::new(&args[2]),
        Some(eredu_architectures::moshi::personaplex_prompt::AUDIO_TOKENS_PER_STREAM as i32),
        stream,
    )?;
    run_personaplex_quantization(&paths, &options, &mut mimi, stream, |artifact| {
        let preparation = eredu_architectures::moshi::prepare_realtime_model(artifact)?;
        let backend = MlxRealtimeExecutionContext::new(stream, weights_stream);
        let model = realtime_support::load(
            &backend,
            preparation,
            eredu_backend_mlx::MlxLoadRequest::default(),
        )?;
        Ok(realtime_support::SelectedRealtimeDriver::new(
            backend, model,
        ))
    })?;
    println!("evaluation={}", paths.output.display());
    Ok(())
}
