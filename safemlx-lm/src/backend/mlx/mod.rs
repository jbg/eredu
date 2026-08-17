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

pub use distributed::{MlxDistributedConfig, MlxDistributedSession};
pub use session::{MlxGeneration, MlxModelInput, MlxModelSession};

use safemlx::{transforms::async_eval_with_event, Array, DeviceType, Event, Stream};
use safemlx_lm_core::backend::{
    Backend, BackendCapabilities, BackendDescriptor, Completion, DeviceDescriptor,
    DistributedBackend, PreparedModel, Submission,
};

use crate::{
    api::{Model, ModelCache, ModelLoadOptions},
    error::Error,
};

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
    stream: &'a Stream,
}

impl<'a> MlxBackend<'a> {
    /// Uses the selected session execution stream.
    pub const fn new(stream: &'a Stream) -> Self {
        Self { stream }
    }
    /// Execution stream used by this backend instance.
    pub const fn stream(&self) -> &'a Stream {
        self.stream
    }

    /// Attaches topology-scoped MLX communication to this backend selection.
    pub fn distributed(
        &self,
        topology: crate::ParallelTopology,
        world: &'a safemlx::distributed::Group,
    ) -> Result<MlxDistributedSession<'a>, Error> {
        self.create_distributed_session(MlxDistributedConfig { topology, world })
    }
}

impl<'a> Backend for MlxBackend<'a> {
    type ModelConfig = MlxModelConfig<'a>;
    type Model = Model;
    type Session = ModelCache;
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
            self.stream,
            config.weights_stream,
        )
        .map(PreparedModel::new)
    }

    fn create_session(
        &self,
        model: &PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error> {
        Ok(model.get().new_cache())
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
