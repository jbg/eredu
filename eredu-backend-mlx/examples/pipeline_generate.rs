//! Minimal MLX two-or-more-process microbatched pipeline generation probe.

use eredu_backend_mlx::backend::runtime::media::input::{token_ids_part, ModelInput};
use eredu_backend_mlx::native::{DeviceAssignment, MlxParallelContext};
use eredu_core::{load_model, BackendProvider as _, BackendSession as _};
use eredu_runtime::DefaultSampler;
use safemlx::{
    distributed::{self, Backend},
    Array, DeviceType, Stream,
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
    let stream = Stream::new_with_device(&topology.device()?);
    let weights_stream = Stream::new_with_device(&topology.device()?);
    let backend = eredu_backend_mlx::native::distributed_backend(&stream, &weights_stream, &group);
    let model = load_model(
        &backend,
        &model_dir,
        eredu_backend_mlx::native::parallel_load_options(
            topology,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
        ),
    )?;
    let mut session = backend.create_session(model)?;
    let prompt = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let parts = [token_ids_part(&prompt)?];
    let mut logits = session
        .prefill(&backend, ModelInput::new(&parts).into())?
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
