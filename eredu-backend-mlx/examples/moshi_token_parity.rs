//! Compare the MLX realtime backend with a native `moshi_mlx` fixture.
//!
//! ```text
//! python scripts/moshi_mlx_token_fixture.py \
//!   /models/moshi /fixtures/moshi-native.safetensors \
//!   --require-mlx-version VERSION
//! cargo run -p eredu-backend-mlx --example moshi_token_parity -- \
//!   /models/moshi /fixtures/moshi-native.safetensors
//! ```
//!
//! The generator's `--create-tiny` option is explicitly test-only; without it,
//! both commands consume the same existing native artifact directory.

use std::{collections::HashMap, path::PathBuf};

use eredu_backend_mlx::native::{Array, Device, DeviceType, ExecutionContext};
use eredu_backend_mlx::{generate_encoded_greedy, MlxRealtimeBackend};
use eredu_core::load_realtime_model;

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .expect("usage: moshi_token_parity <model-dir> <fixture.safetensors>");
    let fixture_path = args
        .get(1)
        .map(PathBuf::from)
        .expect("missing reference fixture path");

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = gpu.stream();
    let fixture = Array::load_safetensors(&fixture_path, cpu.stream())?;
    let mut model = load_realtime_model(MlxRealtimeBackend::new(stream, cpu.stream()), &model_dir)?;
    let generated =
        generate_encoded_greedy(&mut model, required(&fixture, "generation.input_audio")?)?;
    compare_tokens(
        &generated.text_tokens,
        required(&fixture, "generation.expected_text")?,
        stream,
        "generated text",
    )?;
    compare_tokens(
        &generated.audio_tokens,
        required(&fixture, "generation.expected_audio")?,
        stream,
        "generated encoded audio",
    )?;

    println!(
        "Realtime token parity passed: {} input frames, {} emitted audio frames",
        required(&fixture, "generation.input_audio")?.dim(2),
        generated.audio_tokens.dim(2),
    );
    Ok(())
}

fn compare_tokens(
    actual: &Array,
    expected: &Array,
    stream: &eredu_backend_mlx::native::Stream,
    label: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        actual.shape() == expected.shape(),
        "{label}: shape mismatch: Rust {:?}, reference {:?}",
        actual.shape(),
        expected.shape()
    );
    let expected = expected.copy(stream)?;
    let equal = actual.eq(&expected, stream)?.all(None, stream)?;
    anyhow::ensure!(equal.item::<bool>(stream), "{label}: token mismatch");
    Ok(())
}

fn required<'a>(fixture: &'a HashMap<String, Array>, key: &str) -> anyhow::Result<&'a Array> {
    fixture
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("fixture is missing tensor {key}"))
}
