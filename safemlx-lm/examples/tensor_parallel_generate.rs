//! Minimal generalized two-or-more-process Llama/Mistral TP generation probe.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    architectures::llama::layerwise::load_llama_tensor_parallel_model,
    runtime::generation::sampler::DefaultSampler, sample_and_synchronize, DeviceAssignment,
    LayerwiseLoadOptions, ParallelBuildContext, ParallelTopology, ShardingPolicy,
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
    let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
    let mut model = load_llama_tensor_parallel_model(
        &model_dir,
        LayerwiseLoadOptions::default(),
        build,
        &stream,
        &weights_stream,
    )?;
    let mut cache = model.new_cache();
    let prompt = safemlx::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let mut logits = model.forward_tensor_parallel(&prompt, &mut cache, &group, &stream)?;
    let mut sampler = DefaultSampler;
    for _ in 0..8 {
        let synchronized = sample_and_synchronize(
            Some(&logits),
            logits.dim(0),
            &mut sampler,
            0.0,
            None,
            false,
            0,
            &group,
            &stream,
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
        logits = model.forward_tensor_parallel(&synchronized.token, &mut cache, &group, &stream)?;
    }
    Ok(())
}
