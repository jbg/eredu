//! MLX backend adapter.

use std::path::PathBuf;

use safemlx::{transforms::async_eval_with_event, Array, DeviceType, Event, Stream};
use safemlx_lm_core::backend::{
    Backend, BackendCapabilities, BackendDescriptor, Completion, DeviceDescriptor, PreparedModel,
    SessionExecutor, Submission,
};

use crate::{
    api::{Model, ModelCache, ModelLoadOptions},
    architectures::llama::layerwise::{LlamaCache, LlamaModel},
    error::Error,
};

/// Request to prepare any facade-supported model on MLX.
#[derive(Debug, Clone)]
pub struct MlxModelConfig {
    /// Checkpoint directory or GGUF file.
    pub model_path: PathBuf,
    /// Shared architecture-neutral load options.
    pub options: ModelLoadOptions,
}

/// MLX backend selected for a complete model/session.
pub struct MlxBackend<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl<'a> MlxBackend<'a> {
    /// Uses the established execution and weight-materialization streams.
    pub const fn new(stream: &'a Stream, weights_stream: &'a Stream) -> Self {
        Self {
            stream,
            weights_stream,
        }
    }
    /// Execution stream used by this backend instance.
    pub const fn stream(&self) -> &'a Stream {
        self.stream
    }
}

impl Backend for MlxBackend<'_> {
    type ModelConfig = MlxModelConfig;
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
        crate::api::load_model_with_options_mlx(
            config.model_path,
            config.options,
            self.stream,
            self.weights_stream,
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

impl MlxCompletion {
    /// Number of arrays held until exact completion.
    pub fn retained_resources(&self) -> usize {
        self.retained.len()
    }
}

/// Whole-session MLX Llama prefill/decode implementation.
pub struct MlxLlamaExecutor<'a> {
    model: &'a mut LlamaModel,
    cache: &'a mut LlamaCache,
    stream: &'a Stream,
}

impl<'a> MlxLlamaExecutor<'a> {
    /// Borrows one model and its matching cache for the session operation.
    pub const fn new(
        model: &'a mut LlamaModel,
        cache: &'a mut LlamaCache,
        stream: &'a Stream,
    ) -> Self {
        Self {
            model,
            cache,
            stream,
        }
    }

    fn submission(output: Array) -> Result<Submission<Array, MlxCompletion>, Error> {
        let retained = vec![output.clone()];
        let event = async_eval_with_event(retained.iter())?;
        Ok(Submission {
            output,
            completion: MlxCompletion { event, retained },
        })
    }

    pub(crate) fn prefill_retained(&mut self, input: Array) -> Result<Array, Error> {
        let submission = self.prefill(input)?;
        self.model
            .retain_backend_completion(submission.completion)?;
        Ok(submission.output)
    }

    pub(crate) fn decode_retained(&mut self, input: Array) -> Result<Array, Error> {
        let submission = self.decode(input)?;
        self.model
            .retain_backend_completion(submission.completion)?;
        Ok(submission.output)
    }
}

impl SessionExecutor for MlxLlamaExecutor<'_> {
    type PrefillInput = Array;
    type DecodeInput = Array;
    type Output = Array;
    type Completion = MlxCompletion;
    type Error = Error;

    fn prefill(&mut self, input: Array) -> Result<Submission<Array, MlxCompletion>, Error> {
        Self::submission(self.model.prefill(&input, self.cache, self.stream)?)
    }

    fn decode(&mut self, input: Array) -> Result<Submission<Array, MlxCompletion>, Error> {
        Self::submission(self.model.decode(&input, self.cache, self.stream)?)
    }
}
