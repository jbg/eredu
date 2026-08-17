//! MLX backend adapter.

/// Prompt-cache topology conversion for MLX distributed execution.
pub(crate) mod cache;
/// Session-owned MLX communicators, transfers, and collectives.
pub mod distributed;
mod loading;
/// MLX allocator observations for neutral residency telemetry.
pub mod residency;
/// MLX stream assignment and exact completion for speculative sessions.
pub mod speculative;
/// Exact MLX loader binding against portable checkpoint catalogs.
pub(crate) mod structural;
pub(crate) use loading::{
    materialize_gguf_plan, validate_gguf_quantization_source, MaterializedGgufModel,
};
/// Architecture-erased model/session execution.
mod session;

pub(crate) use distributed::MlxDistributedConfig;
pub use distributed::MlxDistributedSession;
pub(crate) use session::{submit_decode_with_cache, submit_prefill_with_cache};
pub use session::{
    MlxGeneration, MlxModelInput, MlxModelOutput, MlxModelSession, MlxSessionCompletion,
    MlxTextCompletion, MlxTextGenerationState, MlxTextToken,
};

use safemlx::{transforms::async_eval_with_event, Array, DeviceType, Event, Stream};
use safemlx_lm_core::backend::{
    Backend, BackendCapabilities, BackendDescriptor, Completion, DeviceDescriptor, PreparedModel,
    Submission,
};

use crate::{
    api::{Model, ModelLoadOptions},
    architectures::distributed::{expert::ExpertParallelModel, pipeline::PipelineModel},
    error::Error,
};

/// Opaque MLX executable selected for one complete model session.
///
/// Replicated, tensor-, pipeline-, and expert-parallel materializations share
/// this type. Architecture-specific rank-local executables are deliberately
/// not exposed through the public loading API.
pub struct MlxModel {
    pub(crate) inner: MlxModelKind,
}

pub(crate) enum MlxModelKind {
    Complete(Model),
    Pipeline(PipelineModel),
    Expert(ExpertParallelModel),
}

impl MlxModel {
    pub(crate) const fn complete(model: Model) -> Self {
        Self {
            inner: MlxModelKind::Complete(model),
        }
    }

    pub(super) const fn pipeline(model: PipelineModel) -> Self {
        Self {
            inner: MlxModelKind::Pipeline(model),
        }
    }

    pub(super) const fn expert(model: ExpertParallelModel) -> Self {
        Self {
            inner: MlxModelKind::Expert(model),
        }
    }

    #[cfg(test)]
    pub(crate) fn into_complete(self) -> Result<Model, Error> {
        match self.inner {
            MlxModelKind::Complete(model) => Ok(model),
            MlxModelKind::Pipeline(_) | MlxModelKind::Expert(_) => Err(Error::Parallel(
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
            MlxModelKind::Expert(model) => model.info().model_kind.model_type_name(),
        }
    }

    /// Returns the rank-local topology for a distributed executable.
    pub fn topology(&self) -> Option<crate::ParallelTopology> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.parallel_info().map(|info| info.topology()),
            MlxModelKind::Pipeline(model) => Some(model.stage_info().topology),
            MlxModelKind::Expert(model) => Some(model.info().topology),
        }
        .filter(|topology| !topology.is_replicated())
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::manager::ResidencyReport>, Error> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.residency_report(),
            MlxModelKind::Pipeline(model) => model.parameter_residency_report(),
            MlxModelKind::Expert(_) => Ok(None),
        }
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(&self) -> Result<Option<crate::DenseDiskStreamReport>, Error> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.dense_stream_report(),
            MlxModelKind::Pipeline(model) => model.dense_stream_report(),
            MlxModelKind::Expert(model) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<crate::ExpertCacheReport>, Error> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.expert_cache_report(),
            MlxModelKind::Pipeline(model) => model.expert_cache_report(),
            MlxModelKind::Expert(model) => model.expert_cache_report(),
        }
    }
}

/// Request to prepare any facade-supported model on MLX.
#[derive(Debug, Clone)]
pub struct MlxModelConfig<'a> {
    /// Backend-neutral inspected artifact and materialization route.
    pub plan: safemlx_lm_core::ModelPreparationPlan,
    /// MLX materialization details for the selected neutral route.
    pub options: ModelLoadOptions,
    /// Stream used for checkpoint materialization and transfers.
    pub weights_stream: &'a Stream,
}

/// MLX backend selected for a complete model/session.
pub struct MlxBackend<'a> {
    stream: Stream,
    world: Option<&'a safemlx::distributed::Group>,
}

impl MlxBackend<'static> {
    /// Uses the selected session execution stream.
    pub fn new(stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            world: None,
        }
    }
}

impl<'a> MlxBackend<'a> {
    /// Selects MLX distributed communication for sessions created by this backend.
    pub fn with_distributed_world(stream: &Stream, world: &'a safemlx::distributed::Group) -> Self {
        Self {
            stream: stream.clone(),
            world: Some(world),
        }
    }
    /// Execution stream used by this backend instance.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }

    #[cfg(test)]
    pub(crate) fn communication_for_topology(
        &self,
        topology: crate::ParallelTopology,
        world: &'a safemlx::distributed::Group,
    ) -> Result<MlxDistributedSession<'a>, Error> {
        MlxDistributedSession::new(MlxDistributedConfig { topology, world }, &self.stream)
    }
}

impl<'a> Backend for MlxBackend<'a> {
    type ModelConfig = MlxModelConfig<'a>;
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
        let family = match device.get_type()? {
            DeviceType::Cpu => "cpu",
            DeviceType::Gpu => "gpu",
        };
        Ok(vec![(
            DeviceDescriptor {
                id: format!("{family}:{}", device.get_index()?),
                name: format!("MLX {family} {}", device.get_index()?),
                family: family.into(),
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
        loading::materialize_model_plan(
            config.plan,
            config.options,
            &self.stream,
            config.weights_stream,
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
    fn submission(output: Array) -> Result<Submission<Array, Self>, Error> {
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
