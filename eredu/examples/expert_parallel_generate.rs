//! Minimal MLX sparse-cache Ring expert-parallel generation probe.

use eredu_backend_mlx::native::DeviceAssignment;
use eredu_core::{
    load_model, GenerationConfigOverrides, ModelRuntime, TextGeneration, TextGenerationConfig,
    TokenOutput,
};
use eredu_runtime::{OrdinaryWeightResidency, ParameterBankLoadOptions, WeightResidency};
use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};

type LoadedSession<'a> = eredu_runtime::text_session_driver::PreparedTextDriver<
    eredu_backend_mlx::backend::MlxBackend<'a>,
    eredu_backend_mlx::native::MlxSessionMechanisms,
>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: expert_parallel_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, group.size(), 1)?,
        group.rank(),
    )?;
    let device = DeviceAssignment::new(DeviceType::Gpu, local_index);
    let stream = Stream::new_with_device(&device.device()?);
    let weights_stream = Stream::new_with_device(&device.device()?);
    let normalized = eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
        WeightResidency::with_independent_parameter_banks(
            OrdinaryWeightResidency::LayerwiseHost(Default::default()),
            ParameterBankLoadOptions::default(),
        ),
    );
    let parallel = eredu_runtime::ParallelLoadRequest::new(
        topology,
        eredu_runtime::PipelineWireContract::new(
            eredu_core::communication::PipelineActivationDtype::Float32,
        ),
        1,
        4096,
        eredu_core::communication::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )?,
    )?;
    let options = eredu_core::ModelLoadOptions::new(normalized.with_parallel_execution(parallel)?)
        .with_placement(device);
    let backend = eredu_backend_mlx::native::distributed_backend(&stream, &weights_stream, &group);
    let model = load_model::<
        eredu_backend_mlx::backend::MlxBackend<'_>,
        _,
        eredu_runtime::ModelLoadFacts<_>,
    >(
        &backend,
        &model_dir,
        &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
        options,
    )?;
    if group.rank() == 0 {
        eprintln!("loaded selected model with EP={}", group.size());
    }

    let mut runtime = ModelRuntime::install::<LoadedSession<'_>>(backend, model)?;
    let sampling = eredu_core::resolve_generation_config(
        None,
        GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(8),
            ..Default::default()
        },
    )?;
    let generation = TextGeneration::new(
        &mut runtime,
        vec![1, 2, 3],
        TextGenerationConfig::new(sampling),
    )?;
    for token in generation {
        let token = token?;
        if group.rank() == 0 {
            eprintln!("sampled token {}", token.token_id()?);
        }
    }
    Ok(())
}
