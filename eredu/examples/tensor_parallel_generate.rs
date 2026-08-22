//! Minimal generalized two-or-more-process Llama/Mistral TP generation probe.

use eredu::{
    backend::mlx::runtime::{generation::sampler::DefaultSampler, media::input},
    backend::mlx::{DeviceAssignment, MlxBackend, MlxParallelContext, ModelLoadOptions},
    core::{BackendProvider as _, BackendSession},
    load_model,
};
use eredu_backend_mlx::native::{
    distributed::{self, Backend},
    DeviceType, Stream,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: tensor_parallel_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = MlxParallelContext::for_group(
        &group,
        group.size(),
        1,
        1,
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let weights_stream = Stream::new_with_device(&topology.device.device()?);
    let backend = MlxBackend::with_distributed_world(&stream, &weights_stream, &group);
    let model = load_model(
        &backend,
        &model_dir,
        ModelLoadOptions::with_parallel(topology),
    )?;
    let mut session = backend.create_session(model)?;
    let prompt = eredu_backend_mlx::native::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let parts = [input::InputPart::text_token_ids(&prompt)];
    let mut logits = session
        .prefill(&backend, input::ModelInput::new(&parts).into())?
        .wait()?
        .into_logits()
        .ok_or("tensor-parallel session returned no logits")?;
    let mut sampler = DefaultSampler;
    for _ in 0..8 {
        let synchronized = session.sample_and_synchronize(
            Some(&logits),
            logits.dim(0),
            &mut sampler,
            0.0,
            None,
            false,
        )?;
        if group.rank() == 0 {
            eprintln!(
                "sampled token {:?}",
                synchronized.token.evaluated()?.as_slice::<u32>()
            );
        }
        if synchronized.finished {
            break;
        }
        logits = session
            .decode(&backend, synchronized.token)?
            .wait()?
            .into_logits()
            .ok_or("tensor-parallel session returned no logits")?;
    }
    Ok(())
}
