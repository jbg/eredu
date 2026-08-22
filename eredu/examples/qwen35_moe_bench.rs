use std::{path::PathBuf, time::Instant};

use eredu::{
    api::LoadedModel, backend::mlx::ModelLoadOptions, GenerationConfigOverrides,
    TextGenerationConfig, TokenOutput,
};
use eredu_backend_mlx::native::ExecutionContext;
use eredu_checkpoint::AffineQuantization;

const DEFAULT_DECODE_TOKENS: usize = 128;
const CASES: &[(&str, usize)] = &[
    ("short", 16),
    ("prefill_128", 128),
    ("prefill_512", 512),
    ("prefill_2048", 2048),
];

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quantize_on_load = args.iter().any(|value| value == "--quantize-on-load");
    let args = args
        .into_iter()
        .filter(|value| value != "--quantize-on-load")
        .collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(default_qwen35_moe_snapshot)
        .expect(
            "usage: qwen35_moe_bench [model-dir] [decode-tokens] [case-name] [--quantize-on-load]",
        );
    let decode_tokens = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_DECODE_TOKENS);
    let case_filter = args.get(2).map(String::as_str);
    let print_ids = std::env::var_os("QWEN35_MOE_BENCH_IDS").is_some();

    println!("model_dir={}", model_dir.display());
    println!("decode_tokens={decode_tokens}");
    println!("quantize_on_load={quantize_on_load}");

    let ctx = ExecutionContext::new(eredu_backend_mlx::native::Device::new(
        eredu_backend_mlx::native::DeviceType::Gpu,
        0,
    ));
    let stream = ctx.stream();
    let weights_ctx = ExecutionContext::new(eredu_backend_mlx::native::Device::new(
        eredu_backend_mlx::native::DeviceType::Cpu,
        0,
    ));
    let weights_stream = weights_ctx.stream();
    eredu_backend_mlx::native::memory::reset_peak_memory()?;
    let load_start = Instant::now();
    let options = if quantize_on_load {
        ModelLoadOptions::with_quantization(AffineQuantization::default())
    } else {
        ModelLoadOptions::default()
    };
    let mut model = LoadedModel::load(
        eredu::backend::mlx::MlxBackend::new(stream, weights_stream),
        &model_dir,
        options,
    )?;
    stream.synchronize()?;
    let load_elapsed = load_start.elapsed();
    println!("load_s={:.3}", load_elapsed.as_secs_f64());
    println!(
        "mlx_active_memory_bytes={}",
        eredu_backend_mlx::native::memory::active_memory()?
    );
    println!(
        "mlx_peak_memory_bytes={}",
        eredu_backend_mlx::native::memory::peak_memory()?
    );
    println!(
        "mlx_cache_memory_bytes={}",
        eredu_backend_mlx::native::memory::cache_memory()?
    );

    let warmup_start = Instant::now();
    let warmup_prompt = prompt_near_token_count(&mut model, 16)?;
    let _ = run_case(&mut model, &warmup_prompt, 2)?;
    println!("warmup_s={:.3}", warmup_start.elapsed().as_secs_f64());

    println!(
        "case,prompt_tokens,prefill_s,generated_tokens,decode_s,decode_tok_s,first_id,last_id"
    );
    for (name, target_tokens) in CASES {
        if case_filter.is_some_and(|filter| filter != *name) {
            continue;
        }
        let prompt = prompt_near_token_count(&mut model, *target_tokens)?;
        let result = run_case(&mut model, &prompt, decode_tokens)?;
        println!(
            "{},{},{:.6},{},{:.6},{:.3},{},{}",
            name,
            result.prompt_tokens,
            result.prefill_s,
            result.generated_tokens,
            result.decode_s,
            result.decode_tok_s,
            result.first_id,
            result.last_id
        );
        if print_ids {
            println!(
                "ids,{name},{}",
                result
                    .ids
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
    }

    Ok(())
}

struct BenchResult {
    prompt_tokens: usize,
    prefill_s: f64,
    generated_tokens: usize,
    decode_s: f64,
    decode_tok_s: f64,
    first_id: u32,
    last_id: u32,
    ids: Vec<u32>,
}

fn run_case(
    model: &mut LoadedModel<eredu::backend::mlx::MlxBackend<'static>>,
    prompt: &str,
    decode_tokens: usize,
) -> anyhow::Result<BenchResult> {
    let prompt_ids = model.encode(prompt, false)?;
    let prompt_tokens = prompt_ids.len();
    model.runtime_mut().session_mut().reset()?;
    let resolved = model.resolve_generation_config(GenerationConfigOverrides {
        temperature: Some(0.0),
        ..Default::default()
    })?;
    let mut generator = model.generate_tokens(prompt_ids, TextGenerationConfig::new(resolved))?;
    let mut ids = Vec::with_capacity(decode_tokens);

    let prefill_start = Instant::now();
    let Some(first) = generator.next() else {
        anyhow::bail!("generator produced no tokens");
    };
    let first = first?;
    let prefill_s = prefill_start.elapsed().as_secs_f64();
    ids.push(first.token_id()?);

    let decode_start = Instant::now();
    for _ in 1..decode_tokens {
        let Some(token) = generator.next() else {
            break;
        };
        ids.push(token?.token_id()?);
    }
    let decode_s = decode_start.elapsed().as_secs_f64();
    let decode_count = ids.len().saturating_sub(1);
    let decode_tok_s = if decode_count == 0 {
        0.0
    } else {
        decode_count as f64 / decode_s
    };

    let first_id = ids[0];
    let last_id = *ids.last().expect("first token was pushed");
    Ok(BenchResult {
        prompt_tokens,
        prefill_s,
        generated_tokens: ids.len(),
        decode_s,
        decode_tok_s,
        first_id,
        last_id,
        ids,
    })
}

fn prompt_near_token_count(
    model: &mut LoadedModel<eredu::backend::mlx::MlxBackend<'static>>,
    target_tokens: usize,
) -> anyhow::Result<String> {
    let base = "Discuss hybrid linear attention, sparse mixture-of-experts routing, recurrent cache updates, grouped convolution, and vocabulary projection in a text generation runtime. ";
    let mut prompt = "Summarize linear attention performance.".to_string();
    while model.encode(&prompt, false)?.len() < target_tokens {
        prompt.push_str(base);
    }
    Ok(prompt)
}

fn default_qwen35_moe_snapshot() -> Option<PathBuf> {
    let snapshots = PathBuf::from(std::env::var_os("HOME")?)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots");
    snapshots
        .read_dir()
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("config.json").exists())
}
