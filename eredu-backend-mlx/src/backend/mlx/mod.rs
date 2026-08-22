//! MLX backend adapter.

/// Prompt-cache topology conversion for MLX distributed execution.
pub mod cache;
pub mod config;
/// Session-owned MLX communicators, transfers, and collectives.
pub mod distributed;
/// Errors produced by MLX model loading and execution.
pub mod error;
#[cfg(feature = "media")]
mod media;
/// Reusable MLX neural-network building blocks.
pub mod nn;
/// MLX allocator observations for neutral residency telemetry.
pub mod residency;
/// MLX-only tensor, checkpoint, execution, and residency infrastructure.
pub mod runtime;
/// MLX process-local device binding for a canonical core rank topology.
pub mod topology;
pub use config::ensure_replicated_load_options;
pub use config::ModelLoadOptions;
pub use distributed::MlxDistributedConfig;
pub use distributed::MlxDistributedSession;
pub use runtime::distributed::topology::{PlacementPlan, RankPartition};
pub use runtime::residency::expert_cache::ExpertCacheReport;
pub use topology::{DeviceAssignment, MlxParallelContext};

use eredu_core::backend::{
    BackendCapabilities, BackendDescriptor, BackendProvider, Completion, DeviceDescriptor,
    ModelLoadingBackend, PreparedModel, Submission,
};
use std::num::NonZeroU8;

use safemlx::{transforms::async_eval_with_event, Array, DeviceType, Event, Stream};

#[cfg(feature = "media")]
use crate::composition::mlx::ModelProcessor;
use crate::{
    backend::mlx::error::Error,
    composition::mlx::{distributed::pipeline::PipelineModel, MlxModelSession, Model},
};

/// Opaque MLX executable selected for one complete model session.
///
/// Replicated, tensor-, pipeline-, and expert-parallel materializations share
/// this type. Architecture-specific rank-local executables are deliberately
/// not exposed through the public loading API.
pub struct MlxModel {
    pub inner: MlxModelKind,
    runtime_state_dtype_bytes: NonZeroU8,
    #[cfg(feature = "media")]
    pub processor: Option<ModelProcessor>,
}

pub enum MlxModelKind {
    Complete(Model),
    Pipeline(PipelineModel),
}

impl MlxModel {
    pub(crate) const fn complete(model: Model, runtime_state_dtype_bytes: NonZeroU8) -> Self {
        Self {
            inner: MlxModelKind::Complete(model),
            runtime_state_dtype_bytes,
            #[cfg(feature = "media")]
            processor: None,
        }
    }

    pub(crate) const fn pipeline(
        model: PipelineModel,
        runtime_state_dtype_bytes: NonZeroU8,
    ) -> Self {
        Self {
            inner: MlxModelKind::Pipeline(model),
            runtime_state_dtype_bytes,
            #[cfg(feature = "media")]
            processor: None,
        }
    }

    /// Wraps a directly constructed replicated model for backend integration tests.
    #[cfg(any(test, feature = "test-support"))]
    pub const fn complete_for_test(model: Model, runtime_state_dtype_bytes: NonZeroU8) -> Self {
        Self::complete(model, runtime_state_dtype_bytes)
    }

    pub(crate) const fn runtime_state_dtype_bytes(&self) -> NonZeroU8 {
        self.runtime_state_dtype_bytes
    }

    #[cfg(feature = "media")]
    pub fn with_processor(mut self, processor: Option<ModelProcessor>) -> Self {
        self.processor = processor;
        self
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn into_complete(self) -> Result<Model, Error> {
        match self.inner {
            MlxModelKind::Complete(model) => Ok(model),
            MlxModelKind::Pipeline(_) => Err(Error::Parallel(
                "the tokenizer/generation facade requires a replicated model; execute distributed models through MlxModelSession"
                    .into(),
            )),
        }
    }

    /// Returns the selected model's normalized architecture name.
    pub fn model_type(&self) -> &str {
        match &self.inner {
            MlxModelKind::Complete(model) => model.model_type(),
            MlxModelKind::Pipeline(model) => model.stage_info().model_kind.model_type_name(),
        }
    }

    /// Returns the rank-local topology for a distributed executable.
    pub fn topology(&self) -> Option<crate::backend::mlx::MlxParallelContext> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.parallel_info().map(|info| info.topology()),
            MlxModelKind::Pipeline(model) => Some(model.stage_info().topology),
        }
        .filter(|topology| !topology.is_replicated())
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.residency_report(),
            MlxModelKind::Pipeline(model) => model.parameter_residency_report(),
        }
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.dense_stream_report(),
            MlxModelKind::Pipeline(model) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<
        Option<crate::backend::mlx::runtime::residency::expert_cache::ExpertCacheReport>,
        Error,
    > {
        match &self.inner {
            MlxModelKind::Complete(model) => model.expert_cache_report(),
            MlxModelKind::Pipeline(model) => model.expert_cache_report(),
        }
    }
}

/// Request to prepare any facade-supported model on MLX.
#[derive(Debug, Clone)]
pub struct MlxModelConfig {
    /// Backend-neutral inspected artifact and materialization route.
    pub plan: eredu_core::ModelPreparationPlan,
    /// MLX materialization details for the selected neutral route.
    pub options: ModelLoadOptions,
}

/// MLX backend selected for a complete model/session.
pub struct MlxBackend<'a> {
    stream: Stream,
    weights_stream: Stream,
    planned_device_id: Option<String>,
    world: Option<&'a safemlx::distributed::Group>,
}

impl MlxBackend<'static> {
    /// Uses the selected execution and weight-materialization streams.
    pub fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            planned_device_id: None,
            world: None,
        }
    }

    pub fn for_execution_plan(stream: &Stream, weights_stream: &Stream, device_id: String) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            planned_device_id: Some(device_id),
            world: None,
        }
    }
}

impl<'a> MlxBackend<'a> {
    /// Selects MLX distributed communication for sessions created by this backend.
    pub fn with_distributed_world(
        stream: &Stream,
        weights_stream: &Stream,
        world: &'a safemlx::distributed::Group,
    ) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            planned_device_id: None,
            world: Some(world),
        }
    }
    /// Execution stream used by this backend instance.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Checkpoint materialization and transfer stream owned by this backend.
    pub const fn weights_stream(&self) -> &Stream {
        &self.weights_stream
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn communication_for_topology(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
        world: &'a safemlx::distributed::Group,
    ) -> Result<MlxDistributedSession<'a>, Error> {
        MlxDistributedSession::new(MlxDistributedConfig { topology, world }, &self.stream)
    }
}

impl<'a> BackendProvider for MlxBackend<'a> {
    type ModelConfig = MlxModelConfig;
    type Model = MlxModel;
    type Session = MlxModelSession<'a>;
    type Error = Error;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "mlx".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn devices(&self) -> Result<Vec<(DeviceDescriptor, BackendCapabilities)>, Self::Error> {
        let device = self.stream.get_device()?;
        let index = device.get_index()?;
        let inferred_family = match device.get_type()? {
            DeviceType::Cpu => "cpu",
            DeviceType::Gpu if cfg!(feature = "cuda") => "cuda",
            DeviceType::Gpu if cfg!(target_os = "macos") => "metal",
            DeviceType::Gpu => "gpu",
        };
        let id = self
            .planned_device_id
            .clone()
            .unwrap_or_else(|| format!("{inferred_family}:{index}"));
        let family = id.split_once(':').map_or_else(
            || inferred_family.to_owned(),
            |(family, _)| family.to_owned(),
        );
        Ok(vec![(
            DeviceDescriptor {
                id,
                name: format!("MLX {family} {index}"),
                family,
                memory_bytes: None,
            },
            BackendCapabilities {
                exact_completion: true,
                transfers: true,
                collectives: true,
                persistent_cache: true,
            },
        )])
    }

    fn prepare_model(
        &self,
        config: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error> {
        crate::composition::mlx::loading::materialize_model_plan(
            config.plan,
            config.options,
            &self.stream,
            &self.weights_stream,
        )
        .map(PreparedModel::new)
    }

    fn create_session(
        &self,
        model: PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error> {
        let distributed = match model.topology() {
            Some(topology) => {
                let world = self.world.ok_or_else(|| {
                    Error::Parallel(
                        "distributed model session creation requires MlxBackend::with_distributed_world"
                            .into(),
                    )
                })?;
                Some(MlxDistributedSession::new(
                    MlxDistributedConfig { topology, world },
                    &self.stream,
                )?)
            }
            None => None,
        };
        MlxModelSession::from_model(model.into_inner(), distributed)
    }
}

impl ModelLoadingBackend for MlxBackend<'_> {
    type LoadOptions = ModelLoadOptions;
    type ConfigurationResolver = eredu_architectures::configuration::ModelConfigurations;

    fn configuration_resolver(&self) -> &Self::ConfigurationResolver {
        &eredu_architectures::configuration::MODEL_CONFIGURATIONS
    }

    fn preparation_policy(
        &self,
        options: &Self::LoadOptions,
    ) -> Result<eredu_core::PreparationPolicy, Self::Error> {
        options.preparation_policy()
    }

    fn validate_preparation(
        &self,
        inspection: &eredu_core::ArtifactInspection,
        policy: eredu_core::PreparationPolicy,
    ) -> Result<(), Self::Error> {
        crate::composition::mlx::structural::validate_inspected_preparation(inspection, policy)
    }

    fn model_config(
        &self,
        plan: eredu_core::ModelPreparationPlan,
        options: Self::LoadOptions,
    ) -> Result<Self::ModelConfig, Self::Error> {
        Ok(MlxModelConfig { plan, options })
    }
}

/// Exact MLX event plus retained output arrays.
pub struct MlxCompletion {
    event: Event,
    retained: Vec<Array>,
}

impl Completion for MlxCompletion {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete().map_err(Into::into)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize().map_err(Into::into)
    }
}

impl Drop for MlxCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

impl MlxCompletion {
    pub fn submission(output: Array) -> Result<Submission<Array, Self>, Error> {
        let retained = vec![output.clone()];
        let event = async_eval_with_event(retained.iter())?;
        Ok(Submission {
            output,
            completion: Self { event, retained },
        })
    }

    /// Number of arrays held until exact completion.
    pub fn retained_resources(&self) -> usize {
        self.retained.len()
    }
}
