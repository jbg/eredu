//! Minimal generalized two-or-more-process Llama/Mistral TP generation probe.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    core::BackendSession,
    load_model_with_options,
    runtime::{generation::sampler::DefaultSampler, media::input},
    DeviceAssignment, MlxBackend, ModelLoadOptions, ParallelTopology,
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
    let topology = ParallelTopology::from_group(
        &group,
        group.size(),
        1,
        1,
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let weights_stream = Stream::new_with_device(&topology.device.device()?);
    let mut model = load_model_with_options(
        &model_dir,
        ModelLoadOptions::with_parallel(topology),
        &stream,
        &weights_stream,
    )?;
    let backend = MlxBackend::new(&stream);
    let mut session = backend.create_distributed_model_session(&model, topology, &group)?;
    let prompt = safemlx::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let parts = [input::InputPart::text_token_ids(&prompt)];
    let mut logits = session
        .prefill(&backend, &mut model, input::ModelInput::new(&parts).into())?
        .wait()?;
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
            .decode(&backend, &mut model, synchronized.token)?
            .wait()?;
    }
    Ok(())
}
