//! Compare the MLX realtime backend with a native `moshi_mlx` fixture.
//!
//! ```text
//! python eredu/scripts/moshi_mlx_token_fixture.py \
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
use eredu_backend_mlx::{native::MlxRealtimeBackend, MlxTensor};
use eredu_core::{load_realtime_model, ObservationSet, ObservationValue, RealtimeSampling};
use eredu_evaluation::{
    compare_observations, encoded_audio_frames, observe_i32_tensor, run_realtime_trace,
    ParityPolicy,
};

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
    let preparation = eredu_architectures::moshi::prepare_realtime_model(&model_dir)?;
    let mut model =
        load_realtime_model(MlxRealtimeBackend::new(stream, cpu.stream()), preparation)?;
    let input = required(&fixture, "generation.input_audio")?;
    let trace = run_realtime_trace(
        &mut model,
        encoded_audio_frames(&MlxTensor::from_array(input.clone()), stream)?,
        RealtimeSampling::greedy(),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let trace_observations = trace.observations()?;
    let mut actual = ObservationSet::new();
    for path in ["trace.text_tokens", "trace.output_audio_tokens"] {
        actual.insert(
            path,
            trace_observations
                .get(path)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("trace omitted {path}"))?,
        )?;
    }
    let mut reference = ObservationSet::new();
    insert_tokens(
        &mut reference,
        "trace.text_tokens",
        required(&fixture, "generation.expected_text")?,
        stream,
    )?;
    insert_tokens(
        &mut reference,
        "trace.output_audio_tokens",
        required(&fixture, "generation.expected_audio")?,
        stream,
    )?;
    let report = compare_observations(&actual, &reference, &ParityPolicy::exact())?;
    anyhow::ensure!(report.passed, "token parity failed: {:?}", report.failures);

    println!(
        "Realtime token parity passed: {} input frames, {} emitted audio frames",
        input.dim(2),
        trace
            .frames()
            .iter()
            .filter(|frame| frame.output_audio_tokens().is_some())
            .count(),
    );
    Ok(())
}

fn insert_tokens(
    observations: &mut ObservationSet,
    path: &str,
    value: &Array,
    stream: &eredu_backend_mlx::native::Stream,
) -> anyhow::Result<()> {
    let value = MlxTensor::from_array(value.clone());
    observations.insert(
        path,
        ObservationValue::Tensor(observe_i32_tensor(&value, stream)?),
    )?;
    Ok(())
}

fn required<'a>(fixture: &'a HashMap<String, Array>, key: &str) -> anyhow::Result<&'a Array> {
    fixture
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("fixture is missing tensor {key}"))
}
