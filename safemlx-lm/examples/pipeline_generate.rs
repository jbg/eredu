//! Minimal two-or-more-process microbatched pipeline generation probe.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    core::{Backend as _, BackendSession as _},
    load_model,
    runtime::{generation::sampler::DefaultSampler, media::input},
    DeviceAssignment, MlxBackend, MlxParallelContext, ModelLoadOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: pipeline_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = MlxParallelContext::for_group(
        &group,
        1,
        group.size(),
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
    let prompt = safemlx::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let parts = [input::InputPart::text_token_ids(&prompt)];
    let mut logits = session
        .prefill(&backend, input::ModelInput::new(&parts).into())?
        .wait()?
        .into_logits();
    let mut sampler = DefaultSampler;
    for generation_step in 0..8 {
        let synchronized =
            session.sample_and_synchronize(logits.as_ref(), 1, &mut sampler, 0.0, None, false)?;
        if group.rank() + 1 == group.size() {
            eprintln!(
                "step {generation_step}: {:?}",
                synchronized.token.evaluated()?.as_slice::<u32>()
            );
        }
        if synchronized.finished {
            break;
        }
        logits = session
            .decode(&backend, synchronized.token)?
            .wait()?
            .into_logits();
    }
    Ok(())
}
