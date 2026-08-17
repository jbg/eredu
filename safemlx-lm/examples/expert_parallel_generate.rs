//! Minimal sparse-cache Ring expert-parallel generation probe for supported MoE models.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    api::ModelLoadOptions,
    core::{Backend as _, BackendSession as _},
    load_model_with_options,
    runtime::residency::expert_cache::ExpertCacheLoadOptions,
    runtime::{generation::sampler::DefaultSampler, media::input},
    DeviceAssignment, MlxBackend, ParallelTopology, WeightResidency,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: expert_parallel_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = ParallelTopology::from_group(
        &group,
        1,
        1,
        group.size(),
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let weights_stream = Stream::new_with_device(&topology.device.device()?);
    let options = ModelLoadOptions::with_parallel(topology).with_weight_residency(
        WeightResidency::with_expert_cache(
            safemlx_lm::NonExpertWeightResidency::LayerwiseHost(Default::default()),
            ExpertCacheLoadOptions::default(),
        ),
    );
    let model = load_model_with_options(&model_dir, options, &stream, &weights_stream)?;
    if group.rank() == 0 {
        eprintln!("loaded {} with EP={}", model.model_type(), group.size());
    }

    let backend = MlxBackend::with_distributed_world(&stream, &group);
    let mut session = backend.create_session(model)?;
    let prompt = safemlx::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let parts = [input::InputPart::text_token_ids(&prompt)];
    let mut logits = session
        .prefill(&backend, input::ModelInput::new(&parts).into())?
        .wait()?
        .into_logits()
        .ok_or("expert-parallel session returned no logits")?;
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
                synchronized.token.evaluated()?.as_slice::<u32>(),
            );
        }
        if synchronized.finished {
            break;
        }
        logits = session
            .decode(&backend, synchronized.token)?
            .wait()?
            .into_logits()
            .ok_or("expert-parallel session returned no logits")?;
    }
    Ok(())
}
