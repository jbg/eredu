//! Minimal MLX sparse-cache Ring expert-parallel generation probe.

use eredu_backend_mlx::native::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use eredu_backend_mlx::{
    DeviceAssignment, InputPart, MlxParallelContext, ModelInput, ModelLoadOptions,
};
use eredu_core::{load_model, BackendProvider as _, BackendSession as _};
use eredu_runtime::DefaultSampler;
use eredu_runtime::{ExpertCacheLoadOptions, NonExpertWeightResidency, WeightResidency};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: expert_parallel_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = MlxParallelContext::for_group(
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
            NonExpertWeightResidency::LayerwiseHost(Default::default()),
            ExpertCacheLoadOptions::default(),
        ),
    );
    let backend = eredu_backend_mlx::native::distributed_backend(&stream, &weights_stream, &group);
    let model = load_model(&backend, &model_dir, options)?;
    if group.rank() == 0 {
        eprintln!(
            "loaded {}/{} with EP={}",
            model.model_family().canonical_name(),
            model.effective_model_type(),
            group.size()
        );
    }

    let mut session = backend.create_session(model)?;
    let prompt = eredu_backend_mlx::native::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut logits = session
        .prefill(&backend, ModelInput::new(&parts).into())?
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
