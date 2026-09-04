//! Architecture-owned inspection and preparation of external draft assistants.

use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    num::NonZeroUsize,
    path::PathBuf,
};

use eredu_checkpoint::{
    schema::SafetensorsCheckpointPlan,
    validation::{resolve_gguf_plan, resolve_safetensors_plan, ResolvedCheckpointPlan},
};
use eredu_core::{
    artifact::ArtifactError, checkpoint::TensorCatalog, ArtifactFormat, LoadingProtocol,
    ModelConfiguration, ModelConfigurationResolver, ParallelRankTopology,
    ResolvedModelConfiguration, TokenizerCompatibilityProof,
};
use eredu_gguf::{Checkpoint, MetadataValue};
use serde_json::Value;

use crate::{gemma4, muse_glimmer};

use eredu_runtime::{
    SpeculativeArchitectureCompatibilityProof, SpeculativeCaptureEntry, SpeculativeCaptureEnvelope,
    SpeculativeCaptureMetadata, SpeculativeCaptureSchema, SpeculativeIdentity,
    SpeculativeMechanism, SpeculativeMechanismRequirements, SpeculativePlacementRequest,
    SpeculativeRealizationRequirements, SpeculativeSelectionRequest,
    SpeculativeStateCacheIdentityIngredients, SpeculativeStrategyRequirements,
};

/// Observation path for one assistant proposal distribution before sampling.
pub const EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH: &str =
    "external_assistant.proposal_logits";

/// Observation path for the target's complete verification logits before resolution.
pub const EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH: &str =
    "external_assistant.verification_logits";

/// Architecture-owned external target-cache envelope.
///
/// The native cache remains opaque. Prepared-input ownership, the selected realization, and the
/// committed target frontier are portable lifecycle state and therefore live above every backend.
pub struct ExternalAssistantCache<C> {
    native: C,
    selected: eredu_runtime::SelectedSpeculativeRealization,
    prepared_input: Option<SpeculativeIdentity>,
    frontier: u64,
}

impl<C> ExternalAssistantCache<C> {
    /// Couples one opaque native cache to the realization selected before materialization.
    pub fn new(native: C, selected: eredu_runtime::SelectedSpeculativeRealization) -> Self {
        Self {
            native,
            selected,
            prepared_input: None,
            frontier: 0,
        }
    }

    /// Borrows opaque backend state for one native mechanism call.
    pub const fn native(&self) -> &C {
        &self.native
    }

    /// Mutably borrows opaque backend state for one native mechanism call.
    pub const fn native_mut(&mut self) -> &mut C {
        &mut self.native
    }

    /// Binds one prepared semantic input to the reusable lane cache.
    pub fn bind_prepared_input(&mut self, identity: SpeculativeIdentity) -> Result<(), String> {
        match self.prepared_input.as_ref() {
            Some(bound) if bound != &identity => {
                Err("external speculative cache belongs to a different prepared input".into())
            }
            Some(_) => Ok(()),
            None => {
                self.prepared_input = Some(identity);
                Ok(())
            }
        }
    }

    /// Derives and binds the speculative identity of one portable prepared-input cache key.
    pub fn bind_prepared_input_cache_identity(
        &mut self,
        prepared: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<(), String> {
        let identity = SpeculativeIdentity::new(format!(
            "prepared-input/{}",
            prepared.prefix_content_fingerprint()
        ))
        .map_err(|error| error.to_string())?;
        self.bind_prepared_input(identity)
    }

    /// Advances the portable committed frontier from a backend-reported native offset.
    pub fn advance_frontier(&mut self, frontier: i32) -> Result<(), String> {
        let frontier = u64::try_from(frontier)
            .map_err(|_| "external target cache frontier is negative".to_owned())?;
        if frontier < self.frontier {
            return Err(format!(
                "external target cache frontier regressed from {} to {frontier}",
                self.frontier
            ));
        }
        self.frontier = frontier;
        Ok(())
    }

    /// Returns the architecture-owned committed frontier.
    pub fn frontier(&self) -> Result<i32, String> {
        i32::try_from(self.frontier)
            .map_err(|_| "external target cache frontier exceeds i32".to_owned())
    }

    /// Returns architecture-declared capture paths in their validated tensor order.
    pub fn capture_paths(&self) -> Vec<&str> {
        self.selected
            .requirements()
            .capture()
            .entries()
            .iter()
            .map(|entry| entry.path().as_str())
            .collect()
    }

    /// Closes and validates one ordered architecture capture at the current frontier.
    pub fn validate_capture_shapes(&self, shapes: &[Vec<usize>]) -> Result<(), String> {
        let prepared_input = self
            .prepared_input
            .clone()
            .ok_or_else(|| "external target capture precedes prepared-input binding".to_owned())?;
        let schema = self
            .selected
            .requirements()
            .capture()
            .instantiate(shapes.iter().cloned())
            .map_err(|error| error.to_string())?;
        let values = vec![(); schema.entries().len()];
        let envelope = SpeculativeCaptureEnvelope::new(
            SpeculativeCaptureMetadata::new(schema, self.frontier),
            values,
        )
        .map_err(|error| error.to_string())?;
        let lane = self.selected.lane_identity(prepared_input, self.frontier);
        self.selected
            .validate_capture(&lane, &envelope)
            .map_err(|error| error.to_string())
    }

    /// Wraps one opaque native checkpoint in the exact semantic cache boundary.
    pub fn checkpoint<N>(&self, native: N) -> ExternalAssistantCacheCheckpoint<N> {
        ExternalAssistantCacheCheckpoint {
            native,
            prepared_input: self.prepared_input.clone(),
            frontier: self.frontier,
        }
    }

    /// Restores the semantic half after its opaque native checkpoint was restored.
    pub fn restore_semantics<N>(&mut self, checkpoint: &ExternalAssistantCacheCheckpoint<N>) {
        self.prepared_input.clone_from(&checkpoint.prepared_input);
        self.frontier = checkpoint.frontier;
    }
}

/// Exact architecture-owned checkpoint envelope around opaque native cache storage.
pub struct ExternalAssistantCacheCheckpoint<C> {
    native: C,
    prepared_input: Option<SpeculativeIdentity>,
    frontier: u64,
}

impl<C> ExternalAssistantCacheCheckpoint<C> {
    /// Borrows the opaque backend checkpoint.
    pub const fn native(&self) -> &C {
        &self.native
    }

    /// Returns the committed frontier retained by this checkpoint.
    pub fn frontier(&self) -> Result<i32, String> {
        i32::try_from(self.frontier)
            .map_err(|_| "external target checkpoint frontier exceeds i32".to_owned())
    }
}

/// Explicit production observers for external-assistant activations and logits.
///
/// They are installed on a reusable materialized assistant and invoked by its architecture-owned
/// executor. The default is the neutral no-op observer used by ordinary generation.
pub struct ExternalAssistantObservers<T, L, E> {
    tensors: Box<dyn eredu_runtime::ActivationObserver<T, E>>,
    logits: Box<dyn eredu_runtime::ActivationObserver<L, E>>,
}

impl<T, L, E> Default for ExternalAssistantObservers<T, L, E>
where
    E: 'static,
{
    fn default() -> Self {
        Self {
            tensors: Box::new(eredu_runtime::NoopObserver),
            logits: Box::new(eredu_runtime::NoopObserver),
        }
    }
}

impl<T, L, E> ExternalAssistantObservers<T, L, E> {
    /// Installs explicit typed observers used by the production executor.
    pub fn new(
        tensors: impl eredu_runtime::ActivationObserver<T, E> + 'static,
        logits: impl eredu_runtime::ActivationObserver<L, E> + 'static,
    ) -> Self {
        Self {
            tensors: Box::new(tensors),
            logits: Box::new(logits),
        }
    }

    /// Observes and optionally replaces one tensor activation.
    pub fn observe_tensor(&mut self, path: &str, value: &T) -> Result<T, E>
    where
        T: Clone,
    {
        eredu_runtime::observe_and_intervene(self.tensors.as_mut(), path, value)
    }

    /// Observes and optionally replaces one logits value.
    pub fn observe_logits(&mut self, path: &str, value: &L) -> Result<L, E>
    where
        L: Clone,
    {
        eredu_runtime::observe_and_intervene(self.logits.as_mut(), path, value)
    }
}

/// Architecture-owned ordinary-target facts needed to admit an external assistant.
#[derive(Debug, Clone)]
pub enum ExternalAssistantTargetProfile {
    /// Gemma text target configuration.
    Gemma4(gemma4::FamilyConfig),
    /// Muse-Glimmer decoder target configuration.
    MuseGlimmer(muse_glimmer::DecoderConfig),
}

impl ExternalAssistantTargetProfile {
    /// Returns the architecture-owned stable target profile and geometry identity.
    pub fn speculative_identity(&self) -> Result<SpeculativeIdentity, ArtifactError> {
        let identity = match self {
            Self::Gemma4(config) => format!(
                "gemma4-target/profile={};geometry={}",
                config.model_type,
                config.architecture_fingerprint()
            ),
            Self::MuseGlimmer(config) => format!(
                "muse-glimmer-target/profile={};geometry={}",
                config.model_type,
                config.architecture_fingerprint()
            ),
        };
        external_contract_identity(identity)
    }

    /// Returns the architecture-declared upper bound for capture sequence geometry.
    pub fn maximum_capture_sequence_length(&self) -> Result<NonZeroUsize, ArtifactError> {
        let maximum = match self {
            Self::Gemma4(config) => config.text.max_position_embeddings,
            Self::MuseGlimmer(config) => config.max_position_embeddings,
        };
        usize::try_from(maximum)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| invalid_assistant("external target maximum sequence length is invalid"))
    }
}

/// Inspected checkpoint source consumed by a concrete assistant materializer.
#[derive(Debug, Clone)]
pub enum ExternalAssistantCheckpoint {
    /// Header-inspected Hugging Face SafeTensors directory.
    SafeTensors {
        /// Submitted artifact directory containing the admitted payload members.
        source: PathBuf,
        /// Exact header catalog admitted during neutral preparation.
        catalog: TensorCatalog,
        /// Strict architecture schema used to revalidate the reopened source.
        plan: SafetensorsCheckpointPlan,
        /// Exact physical layout selected during neutral admission.
        resolution: ResolvedCheckpointPlan,
    },
    /// Header-inspected and architecture-admitted GGUF checkpoint.
    Gguf {
        /// Portable checkpoint handle retained from inspection.
        checkpoint: Checkpoint,
        /// Exact architecture layout selected during admission.
        resolution: ResolvedCheckpointPlan,
        /// Canonical physical-to-logical tensor mapping resolved during admission.
        tensor_mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    },
}

impl ExternalAssistantCheckpoint {
    fn speculative_identities(
        &self,
        assistant_profile: &SpeculativeIdentity,
    ) -> Result<(SpeculativeIdentity, SpeculativeIdentity), ArtifactError> {
        match self {
            Self::SafeTensors {
                source,
                catalog,
                plan,
                resolution,
            } => {
                let keys = resolution
                    .source_keys()
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(",");
                let artifact = external_contract_identity(format!(
                    "external-assistant/profile={};source={};keys={keys};catalog={catalog:?}",
                    assistant_profile.as_str(),
                    source.display()
                ))?;
                let format = external_contract_identity(format!(
                    "safetensors/plan={plan:?};resolution={resolution:?}"
                ))?;
                Ok((artifact, format))
            }
            Self::Gguf {
                checkpoint,
                resolution,
                tensor_mapping,
            } => {
                let mapping = tensor_mapping
                    .iter()
                    .map(|tensor| {
                        format!(
                            "{}:{}:{}:{:?}:{:?}",
                            tensor.physical_name,
                            tensor.original_name,
                            tensor.layout.name,
                            tensor.layout.shape,
                            tensor.layout.dtype
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(";");
                let artifact = external_contract_identity(format!(
                    "external-assistant/profile={};checkpoint={checkpoint:?}",
                    assistant_profile.as_str()
                ))?;
                let format = external_contract_identity(format!(
                    "gguf/resolution={resolution:?};mapping={mapping}"
                ))?;
                Ok((artifact, format))
            }
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Architecture-owned type identity for one prepared external assistant.
///
/// This trait is sealed so exhaustive family dispatch remains internal to this
/// crate. Materializers receive its associated configuration through one
/// generic visitor method and never match an assistant-family enum.
#[allow(private_bounds)]
pub trait ExternalAssistantArchitecture: sealed::Sealed + Sized + 'static {
    /// Exact normalized architecture configuration.
    type Config: Clone + std::fmt::Debug;

    /// Backend-neutral assistant module specialized to a neutral neural backend.
    type Module<B>: Clone + eredu_nn::Parameterized<B::Tensor>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Architecture-owned speculative executor specialized to backend mechanisms.
    type Executor<'a, M>: eredu_core::SpeculativeExecutor<
        Input = M::Input,
        Cache = ExternalAssistantCache<M::NativeCache>,
        Logits = M::Logits,
        Context<'a> = M::Context<'a>,
        Completion = M::Completion,
        Telemetry = M::Telemetry,
        Error = M::Error,
    >
    where
        Self: 'a,
        M: ExternalAssistantExecutionMechanisms<Self> + 'static;

    /// Architecture identity whose tokenizer contract the assistant shares.
    fn tokenizer_model_kind() -> crate::configuration::ModelKind;

    /// Stable artifact identity carried by the normalized configuration.
    fn configuration_model_type(config: &Self::Config) -> &str;

    /// Builds the unloaded neutral assistant module.
    fn module<B>(
        config: Self::Config,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Module<B>, eredu_nn::Error>
    where
        Self: Sized,
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Checkpoint-declared weight encoding before optional load-time conversion.
    fn quantization(config: &Self::Config) -> Option<eredu_checkpoint::WeightQuantization>;

    /// Derives the module configuration for explicit load-time quantization.
    fn load_time_quantization(
        config: &Self::Config,
        quantization: eredu_checkpoint::WeightQuantization,
    ) -> Result<Self::Config, String>;

    /// Applies exact per-parameter formats discovered in a GGUF container.
    fn with_checkpoint_formats(
        config: &Self::Config,
        formats: HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self::Config, String>;

    /// Binds a typed assistant to the architecture-owned speculative lifecycle.
    fn executor<'a, M>(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
    ) -> Self::Executor<'a, M>
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static;

    /// Constructs the selected executor and lends it to one family-blind runtime visitor.
    fn visit_executor<'a, M, V>(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
        visitor: V,
    ) -> V::Output
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static,
        V: ExternalAssistantExecutorVisitor<Self, M>;
}

/// Family-blind continuation invoked with an architecture-selected external executor.
pub trait ExternalAssistantExecutorVisitor<
    A: ExternalAssistantArchitecture,
    M: ExternalAssistantExecutionMechanisms<A>,
>
{
    /// Result returned to the backend session.
    type Output;

    /// Runs one concrete architecture executor through a shared runtime scheduler.
    fn execute<'run, E>(self, executor: &'run mut E) -> Self::Output
    where
        Self: 'run,
        E: eredu_core::SpeculativeExecutor<
                Input = M::Input,
                Cache = ExternalAssistantCache<M::NativeCache>,
                Logits = M::Logits,
                Context<'run> = M::Context<'run>,
                Completion = M::Completion,
                Telemetry = M::Telemetry,
                Error = M::Error,
            > + 'run;
}

/// Device placement used by a family-neutral tensor primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalAssistantTensorPlacement {
    /// Ordinary-target placement.
    Target,
    /// Draft-assistant placement.
    Draft,
}

/// Direction of one explicitly ordered assistant transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalAssistantTransfer {
    /// Ordinary target to draft assistant.
    TargetToDraft,
    /// Draft assistant to ordinary target.
    DraftToTarget,
}

/// Family-neutral mechanisms supplied by a concrete backend.
///
/// Family configuration, capture interpretation, proposal equations, state geometry, and
/// executor choice remain owned by [`ExternalAssistantArchitecture`]. A backend supplies this
/// contract once for every sealed architecture it can materialize.
pub trait ExternalAssistantExecutionMechanisms<A: ExternalAssistantArchitecture>: 'static {
    /// Neutral neural backend used by the materialized assistant module.
    type NeuralBackend: eredu_nn::GroupedNeuralBackend<Tensor = Self::Tensor>
        + eredu_nn::DistributedNeuralBackend
        + Clone;
    /// Attention cache used by sequential assistant layers.
    type AttentionCache: eredu_nn::AttentionCache<Self::Tensor>;
    /// Materialized ordinary target.
    type Target: ?Sized;
    /// Materialized assistant container.
    type Assistant;
    /// Prepared ordinary-target input.
    type Input;
    /// Opaque native ordinary-target cache storage.
    type NativeCache;
    /// Opaque native target-cache checkpoint storage.
    type NativeCacheCheckpoint;
    /// Retained native tensor.
    type Tensor: eredu_nn::Tensor + Clone;
    /// Native logits consumed by sampling.
    type Logits;
    /// Selected target/draft execution assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact verification completion.
    type Completion: eredu_core::Completion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Borrows the exact normalized assistant configuration.
    fn config(assistant: &Self::Assistant) -> &A::Config;
    /// Borrows the neutral assistant module.
    fn module(assistant: &mut Self::Assistant) -> &mut A::Module<Self::NeuralBackend>;
    /// Maps a portable neural error into the backend failure type.
    fn neural_error(error: eredu_nn::Error) -> Self::Error;
    /// Constructs a stable architecture lifecycle failure.
    fn error(message: String) -> Self::Error;
    /// Exposes the portable cache identity already attached to one prepared backend prompt.
    fn prepared_input_cache_identity(
        input: &Self::Input,
    ) -> Result<eredu_runtime::PreparedInputCacheIdentity, Self::Error>;
    /// Returns the exact logical shape of one retained target-capture tensor.
    fn tensor_shape(value: &Self::Tensor) -> Result<Vec<usize>, Self::Error>;
    /// Closes and validates one architecture-ordered capture against the selected realization.
    /// Runs native ordinary-target prefill and returns its architecture-declared capture.
    fn prefill_target_native<'a>(
        target: &mut Self::Target,
        request: &crate::composite_execution::ExternalPredictionCaptureRequest,
        input: Self::Input,
        cache: &mut Self::NativeCache,
        context: Self::Context<'a>,
    ) -> Result<
        (
            Self::Tensor,
            crate::composite_execution::ExternalPredictionTargetCapture<Self::Tensor>,
        ),
        Self::Error,
    >;
    /// Runs native ordinary-target verification and returns its architecture-declared capture.
    fn verify_target_native<'a>(
        target: &mut Self::Target,
        request: &crate::composite_execution::ExternalPredictionCaptureRequest,
        tokens: &Self::Tensor,
        cache: &mut Self::NativeCache,
        context: Self::Context<'a>,
    ) -> Result<
        (
            Self::Tensor,
            crate::composite_execution::ExternalPredictionTargetCapture<Self::Tensor>,
        ),
        Self::Error,
    >;
    /// Captures an exact opaque native target-cache checkpoint.
    fn checkpoint_native(
        cache: &Self::NativeCache,
    ) -> Result<Self::NativeCacheCheckpoint, Self::Error>;
    /// Restores an exact opaque native target-cache checkpoint.
    fn restore_checkpoint_native<'a>(
        cache: &mut Self::NativeCache,
        checkpoint: &Self::NativeCacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>;
    /// Reports the native target-cache frontier after one target operation.
    fn native_cache_len(cache: &Self::NativeCache) -> Result<i32, Self::Error>;
    /// Observes and optionally replaces one retained architecture tensor.
    fn observe_tensor(
        _assistant: &mut Self::Assistant,
        _path: &str,
        value: Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(value)
    }
    /// Observes and optionally replaces one logits row.
    fn observe_logits(
        _assistant: &mut Self::Assistant,
        _path: &str,
        value: Self::Logits,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(value)
    }
    /// Returns the sequence width of a tensor.
    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error>;
    /// Selects one sequence row, optionally retaining the sequence dimension.
    fn sequence_row<'a>(
        value: &Self::Tensor,
        row: usize,
        retain_dimension: bool,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Converts a retained tensor into native sampling logits.
    fn into_logits(value: Self::Tensor) -> Self::Logits;
    /// Retains a sequence suffix.
    fn sequence_suffix<'a>(
        value: &Self::Tensor,
        maximum: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Retains a shared-attention prefix.
    fn shared_prefix<'a>(
        value: &Self::Tensor,
        cache_len: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Selects a target-token prefix.
    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Constructs exact target token ids.
    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Explicitly orders or transfers a tensor between selected placements.
    fn transfer<'a>(
        value: &Self::Tensor,
        direction: ExternalAssistantTransfer,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Applies one architecture-declared ordinary-target operation.
    fn target_operation<'a>(
        target: &mut Self::Target,
        operation: crate::composite_execution::ExternalPredictionTargetOperation<'_, Self::Tensor>,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Borrows the selected neural execution context.
    fn neural_context<'a>(
        context: Self::Context<'a>,
        placement: ExternalAssistantTensorPlacement,
    ) -> &'a <Self::Tensor as eredu_nn::Tensor>::Context;
    /// Submits all retained tensors needed to complete verification.
    fn submit_completion<'a>(
        values: impl IntoIterator<Item = &'a Self::Tensor>,
    ) -> Result<Self::Completion, Self::Error>
    where
        Self::Tensor: 'a;
}

#[derive(Debug, Clone, Copy)]
/// Gemma 4 external-assistant architecture marker used by typed dispatch.
pub struct Gemma4AssistantArchitecture;

impl sealed::Sealed for Gemma4AssistantArchitecture {}

impl ExternalAssistantArchitecture for Gemma4AssistantArchitecture {
    type Config = gemma4::AssistantConfig;
    type Module<B>
        = gemma4::Assistant<B>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    type Executor<'a, M>
        = gemma4::speculative::ExternalExecutor<
        'a,
        gemma4::speculative::ArchitectureExternalMechanisms<M>,
    >
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static;

    fn tokenizer_model_kind() -> crate::configuration::ModelKind {
        crate::configuration::ModelKind::Gemma4
    }

    fn configuration_model_type(config: &Self::Config) -> &str {
        &config.model_type
    }

    fn module<B>(
        config: Self::Config,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Module<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        gemma4::Assistant::new(config, context)
    }

    fn quantization(config: &Self::Config) -> Option<eredu_checkpoint::WeightQuantization> {
        config.quantization
    }

    fn load_time_quantization(
        config: &Self::Config,
        quantization: eredu_checkpoint::WeightQuantization,
    ) -> Result<Self::Config, String> {
        config
            .load_time_quantization(quantization)
            .map_err(|error| error.to_string())
    }

    fn with_checkpoint_formats(
        config: &Self::Config,
        formats: HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self::Config, String> {
        config
            .with_checkpoint_formats(formats)
            .map_err(|error| error.to_string())
    }

    fn executor<'a, M>(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
    ) -> Self::Executor<'a, M>
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static,
    {
        gemma4::speculative::ExternalExecutor::new(target, assistant, capture)
    }

    fn visit_executor<'a, M, V>(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
        visitor: V,
    ) -> V::Output
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static,
        V: ExternalAssistantExecutorVisitor<Self, M>,
    {
        let mut executor = gemma4::speculative::ExternalExecutor::<
            gemma4::speculative::ArchitectureExternalMechanisms<M>,
        >::new(target, assistant, capture);
        visitor.execute(&mut executor)
    }
}

#[derive(Debug, Clone, Copy)]
/// Muse-Glimmer DFlash assistant architecture marker used by typed dispatch.
pub struct MuseGlimmerAssistantArchitecture;

impl sealed::Sealed for MuseGlimmerAssistantArchitecture {}

impl ExternalAssistantArchitecture for MuseGlimmerAssistantArchitecture {
    type Config = muse_glimmer::DFlashConfig;
    type Module<B>
        = muse_glimmer::DFlash<B>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    type Executor<'a, M>
        = muse_glimmer::speculative::ExternalExecutor<
        'a,
        muse_glimmer::speculative::ArchitectureExternalMechanisms<M>,
    >
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static;

    fn tokenizer_model_kind() -> crate::configuration::ModelKind {
        crate::configuration::ModelKind::MuseGlimmer
    }

    fn configuration_model_type(config: &Self::Config) -> &str {
        &config.model_type
    }

    fn module<B>(
        config: Self::Config,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Module<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        muse_glimmer::DFlash::new(config, context)
    }

    fn quantization(config: &Self::Config) -> Option<eredu_checkpoint::WeightQuantization> {
        config.quantization
    }

    fn load_time_quantization(
        config: &Self::Config,
        quantization: eredu_checkpoint::WeightQuantization,
    ) -> Result<Self::Config, String> {
        config
            .load_time_quantization(quantization)
            .map_err(|error| error.to_string())
    }

    fn with_checkpoint_formats(
        config: &Self::Config,
        formats: HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self::Config, String> {
        config
            .with_checkpoint_formats(formats)
            .map_err(|error| error.to_string())
    }

    fn executor<'a, M>(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
    ) -> Self::Executor<'a, M>
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static,
    {
        muse_glimmer::speculative::ExternalExecutor::new(target, assistant, capture)
    }

    fn visit_executor<'a, M, V>(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
        visitor: V,
    ) -> V::Output
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static,
        V: ExternalAssistantExecutorVisitor<Self, M>,
    {
        let mut executor = muse_glimmer::speculative::ExternalExecutor::<
            muse_glimmer::speculative::ArchitectureExternalMechanisms<M>,
        >::new(target, assistant, capture);
        visitor.execute(&mut executor)
    }
}

/// Fully inspected, architecture-typed assistant materialization input.
#[derive(Debug, Clone)]
pub struct PreparedExternalAssistant<A: ExternalAssistantArchitecture> {
    checkpoint: ExternalAssistantCheckpoint,
    config: A::Config,
    _architecture: PhantomData<fn() -> A>,
}

impl<A: ExternalAssistantArchitecture> PreparedExternalAssistant<A> {
    /// Consumes the plan into its admitted checkpoint and normalized configuration.
    pub fn into_parts(self) -> (ExternalAssistantCheckpoint, A::Config) {
        (self.checkpoint, self.config)
    }

    /// Borrows the admitted checkpoint without exposing architecture dispatch.
    pub const fn checkpoint(&self) -> &ExternalAssistantCheckpoint {
        &self.checkpoint
    }

    /// Borrows the exact normalized architecture configuration.
    pub const fn config(&self) -> &A::Config {
        &self.config
    }
}

/// Family-blind visitor for one architecture-typed assistant plan.
pub trait ExternalAssistantPreparationVisitor {
    /// Materialized assistant for architecture `A`.
    type Output<A: ExternalAssistantArchitecture>;
    /// Materialization failure.
    type Error;

    /// Visits any sealed assistant architecture through one static-dispatch path.
    fn visit<A: ExternalAssistantArchitecture>(
        self,
        prepared: PreparedExternalAssistant<A>,
    ) -> Result<Self::Output<A>, Self::Error>;
}

enum DispatchedMaterializedExternalAssistant<V: ExternalAssistantPreparationVisitor> {
    Gemma4(V::Output<Gemma4AssistantArchitecture>),
    MuseGlimmer(V::Output<MuseGlimmerAssistantArchitecture>),
}

/// Architecture-owned existential wrapper around one materialized assistant.
///
/// The wrapper preserves the concrete assistant type for static dispatch while
/// preventing concrete backends from carrying a parallel family enum.
pub struct MaterializedExternalAssistant<V: ExternalAssistantPreparationVisitor> {
    dispatched: DispatchedMaterializedExternalAssistant<V>,
}

/// Family-blind visitor for an already materialized assistant.
pub trait MaterializedExternalAssistantVisitor<V: ExternalAssistantPreparationVisitor> {
    /// Result of the selected operation.
    type Output;

    /// Visits the concrete assistant selected during architecture preparation.
    fn visit<A: ExternalAssistantArchitecture>(self, assistant: &mut V::Output<A>) -> Self::Output;
}

impl<V: ExternalAssistantPreparationVisitor> MaterializedExternalAssistant<V> {
    /// Runs one generic backend visitor against the retained typed assistant.
    pub fn visit<W>(&mut self, visitor: W) -> W::Output
    where
        W: MaterializedExternalAssistantVisitor<V>,
    {
        match &mut self.dispatched {
            DispatchedMaterializedExternalAssistant::Gemma4(assistant) => {
                visitor.visit::<Gemma4AssistantArchitecture>(assistant)
            }
            DispatchedMaterializedExternalAssistant::MuseGlimmer(assistant) => {
                visitor.visit::<MuseGlimmerAssistantArchitecture>(assistant)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum DispatchedExternalAssistantPreparation {
    Gemma4(PreparedExternalAssistant<Gemma4AssistantArchitecture>),
    MuseGlimmer(PreparedExternalAssistant<MuseGlimmerAssistantArchitecture>),
}

/// Opaque architecture-dispatched external assistant preparation.
#[derive(Debug, Clone)]
pub struct ExternalAssistantPreparation {
    dispatched: DispatchedExternalAssistantPreparation,
}

/// Construction-time neutral identities and invocation bounds for an external assistant.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternalSpeculativeContractRequest {
    topology: ParallelRankTopology,
    processor: SpeculativeIdentity,
    tokenizer_proof: TokenizerCompatibilityProof,
    tokenizer_fingerprint: [u8; 32],
    maximum_draft_tokens: NonZeroUsize,
}

impl ExternalSpeculativeContractRequest {
    /// Creates exact construction-time inputs for architecture-owned external selection.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        topology: ParallelRankTopology,
        processor: SpeculativeIdentity,
        tokenizer_proof: TokenizerCompatibilityProof,
        tokenizer_fingerprint: [u8; 32],
        maximum_draft_tokens: NonZeroUsize,
    ) -> Self {
        Self {
            topology,
            processor,
            tokenizer_proof,
            tokenizer_fingerprint,
            maximum_draft_tokens,
        }
    }
}

/// Exact architecture-owned requirements and proofs for one external assistant.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternalSpeculativeContract {
    requirements: SpeculativeRealizationRequirements,
    architecture_proof: SpeculativeArchitectureCompatibilityProof,
    tokenizer_proof: TokenizerCompatibilityProof,
    target_capture: SpeculativeCaptureSchema,
}

impl ExternalSpeculativeContract {
    /// Returns complete requirements consumed by neutral speculative selection.
    pub const fn requirements(&self) -> &SpeculativeRealizationRequirements {
        &self.requirements
    }

    /// Returns the architecture proof paired with the target and capture schema.
    pub const fn architecture_proof(&self) -> &SpeculativeArchitectureCompatibilityProof {
        &self.architecture_proof
    }

    /// Returns the facade-owned tokenizer proof retained before construction.
    pub const fn tokenizer_proof(&self) -> TokenizerCompatibilityProof {
        self.tokenizer_proof
    }

    /// Returns the exact ordered capture schema advertised by the ordinary target.
    pub const fn target_capture(&self) -> &SpeculativeCaptureSchema {
        &self.target_capture
    }

    /// Creates the exact neutral selection request without reconstructing either proof.
    pub fn selection_request(
        &self,
        placement: SpeculativePlacementRequest,
    ) -> SpeculativeSelectionRequest {
        SpeculativeSelectionRequest::new(placement, self.target_capture.clone())
            .with_architecture_proof(self.architecture_proof.clone())
            .with_tokenizer_proof(self.tokenizer_proof)
    }

    /// Consumes the contract into neutral selection inputs.
    pub fn into_parts(
        self,
    ) -> (
        SpeculativeRealizationRequirements,
        SpeculativeArchitectureCompatibilityProof,
        TokenizerCompatibilityProof,
        SpeculativeCaptureSchema,
    ) {
        (
            self.requirements,
            self.architecture_proof,
            self.tokenizer_proof,
            self.target_capture,
        )
    }
}

/// Target/assistant compatibility proven before any assistant payload is opened.
pub struct CompatibleExternalAssistantPreparation {
    preparation: ExternalAssistantPreparation,
    target: ExternalAssistantTargetProfile,
    capture: crate::composite_execution::ExternalPredictionCaptureRequest,
}

impl CompatibleExternalAssistantPreparation {
    /// Returns the exact architecture-owned target capture request.
    pub const fn capture(&self) -> &crate::composite_execution::ExternalPredictionCaptureRequest {
        &self.capture
    }

    /// Derives exact architecture requirements before any assistant payload is opened.
    pub fn speculative_contract(
        &self,
        request: ExternalSpeculativeContractRequest,
    ) -> Result<ExternalSpeculativeContract, ArtifactError> {
        external_speculative_contract(self, request)
    }

    /// Materializes the already compatible assistant through one generic visitor.
    pub fn visit<V: ExternalAssistantPreparationVisitor>(
        self,
        visitor: V,
    ) -> Result<MaterializedExternalAssistant<V>, V::Error> {
        self.preparation.visit(visitor)
    }
}

impl ExternalAssistantPreparation {
    /// Architecture identity whose tokenizer contract the assistant shares.
    pub fn tokenizer_model_kind(&self) -> crate::configuration::ModelKind {
        match &self.dispatched {
            DispatchedExternalAssistantPreparation::Gemma4(_) => {
                Gemma4AssistantArchitecture::tokenizer_model_kind()
            }
            DispatchedExternalAssistantPreparation::MuseGlimmer(_) => {
                MuseGlimmerAssistantArchitecture::tokenizer_model_kind()
            }
        }
    }

    /// Proves exact target compatibility without opening assistant weight payloads.
    pub fn prove_target_compatibility(
        self,
        target: &ExternalAssistantTargetProfile,
    ) -> Result<CompatibleExternalAssistantPreparation, String> {
        let capture = match (&self.dispatched, target) {
            (
                DispatchedExternalAssistantPreparation::Gemma4(prepared),
                ExternalAssistantTargetProfile::Gemma4(target),
            ) => {
                crate::gemma4::model::external_assistant_capture_request(target, prepared.config())
            }
            (
                DispatchedExternalAssistantPreparation::MuseGlimmer(prepared),
                ExternalAssistantTargetProfile::MuseGlimmer(target),
            ) => crate::muse_glimmer::model::external_assistant_capture_request(
                target,
                prepared.config(),
            ),
            _ => Err("external assistant family does not match the selected target".into()),
        }?;
        Ok(CompatibleExternalAssistantPreparation {
            preparation: self,
            target: target.clone(),
            capture,
        })
    }

    /// Dispatches a compatibility-proven plan to one architecture-typed materializer.
    fn visit<V: ExternalAssistantPreparationVisitor>(
        self,
        visitor: V,
    ) -> Result<MaterializedExternalAssistant<V>, V::Error> {
        match self.dispatched {
            DispatchedExternalAssistantPreparation::Gemma4(prepared) => visitor
                .visit(prepared)
                .map(|assistant| MaterializedExternalAssistant {
                    dispatched: DispatchedMaterializedExternalAssistant::Gemma4(assistant),
                }),
            DispatchedExternalAssistantPreparation::MuseGlimmer(prepared) => visitor
                .visit(prepared)
                .map(|assistant| MaterializedExternalAssistant {
                    dispatched: DispatchedMaterializedExternalAssistant::MuseGlimmer(assistant),
                }),
        }
    }
}

fn external_contract_identity(
    value: impl Into<String>,
) -> Result<SpeculativeIdentity, ArtifactError> {
    SpeculativeIdentity::new(value).map_err(invalid_assistant)
}

fn external_topology_identity(
    topology: ParallelRankTopology,
) -> Result<SpeculativeIdentity, ArtifactError> {
    external_contract_identity(format!(
        "tp={}:{};pp={}:{};ep={}:{};dp={}:{}",
        topology.tensor_parallel_size(),
        topology.tensor_parallel_rank(),
        topology.pipeline_parallel_size(),
        topology.pipeline_parallel_rank(),
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        topology.data_parallel_size(),
        topology.data_parallel_rank(),
    ))
}

fn external_positive(value: i32, field: &str) -> Result<usize, ArtifactError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_assistant(format!("external assistant {field} must be positive")))
}

fn external_speculative_contract(
    compatible: &CompatibleExternalAssistantPreparation,
    request: ExternalSpeculativeContractRequest,
) -> Result<ExternalSpeculativeContract, ArtifactError> {
    if request.topology.world_size() != 1 {
        return Err(invalid_assistant(
            "external speculative targets require replicated topology",
        ));
    }
    request
        .tokenizer_proof
        .validate_target(request.tokenizer_fingerprint)
        .map_err(invalid_assistant)?;
    // Public speculative lanes are batch-one. Sequence capture geometry uses the target's
    // architecture-declared upper bound so construction never invents a narrower invocation.
    let batch = 1;
    let sequence = compatible.target.maximum_capture_sequence_length()?.get();
    let target_identity = compatible.target.speculative_identity()?;
    let topology = external_topology_identity(request.topology)?;
    let owner = external_contract_identity(format!(
        "{}/owner/rank/{}",
        topology.as_str(),
        request.topology.global_rank()
    ))?;

    let (
        family,
        architecture_capacity,
        assistant_identity,
        artifact_identity,
        format_identity,
        strategy_detail,
        capture,
        mut state,
        grouped,
    ) = match (
        &compatible.preparation.dispatched,
        &compatible.target,
        &compatible.capture,
    ) {
        (
            DispatchedExternalAssistantPreparation::Gemma4(prepared),
            ExternalAssistantTargetProfile::Gemma4(target),
            crate::composite_execution::ExternalPredictionCaptureRequest::Gemma4SharedAttention {
                final_hidden_path,
            },
        ) => {
            let assistant = prepared.config();
            let proof = assistant
                .prove_compatibility(&target.text)
                .map_err(invalid_assistant)?;
            let capacity = NonZeroUsize::new(assistant.block_size.saturating_sub(1))
                .ok_or_else(|| invalid_assistant("Gemma 4 assistant proposal capacity is zero"))?;
            let hidden = external_positive(target.text.hidden_size, "Gemma target hidden size")?;
            let mut entries = vec![SpeculativeCaptureEntry::new(
                external_contract_identity(final_hidden_path.clone())?,
                vec![batch, sequence, hidden],
                owner.clone(),
                external_contract_identity("gemma4.target.final_hidden")?,
            )
            .and_then(|entry| entry.with_bounded_dimension(1))
            .map_err(invalid_assistant)?];
            let mut state = vec![
                "target.verification_checkpoint".to_owned(),
                "assistant.target_hidden".to_owned(),
            ];
            let mut publishers = BTreeSet::new();
            for &layer in proof.target_publisher_layers() {
                if !publishers.insert(layer) {
                    continue;
                }
                let policy = target.text.layer_schedule.get(layer).ok_or_else(|| {
                    invalid_assistant("Gemma assistant publisher layer is outside target")
                })?;
                let heads =
                    usize::try_from(policy.num_key_value_heads.get()).map_err(invalid_assistant)?;
                let head_dim = usize::try_from(policy.head_dim.get()).map_err(invalid_assistant)?;
                for (component, ordinal) in [("keys", 0usize), ("values", 1usize)] {
                    entries.push(
                        SpeculativeCaptureEntry::new(
                            external_contract_identity(format!(
                                "model.language_model.layers.{layer}.shared_attention.{component}"
                            ))?,
                            vec![batch, heads, sequence, head_dim],
                            owner.clone(),
                            external_contract_identity(format!(
                                "gemma4.target.shared_attention.{layer}.{ordinal}"
                            ))?,
                        )
                        .and_then(|entry| entry.with_bounded_dimension(2))
                        .map_err(invalid_assistant)?,
                    );
                    state.push(format!("assistant.shared_attention.{layer}.{component}"));
                }
            }
            for layer in 0..assistant.text_config.num_hidden_layers() {
                state.push(format!("assistant.private_kv.{layer}.keys"));
                state.push(format!("assistant.private_kv.{layer}.values"));
            }
            let capture = SpeculativeCaptureSchema::new(
                    external_contract_identity(format!(
                        "gemma4-external/capture/batch={batch};sequence={sequence};publishers={publishers:?}"
                    ))?,
                    entries,
                )
                .map_err(invalid_assistant)?;
            let assistant_identity = external_contract_identity(format!(
                "gemma4-assistant/model={};block={};layers={};hidden={};backbone={}",
                assistant.model_type,
                assistant.block_size,
                assistant.text_config.num_hidden_layers(),
                assistant.text_config.hidden_size,
                assistant.backbone_hidden_size,
            ))?;
            let (artifact_identity, format_identity) = prepared
                .checkpoint()
                .speculative_identities(&assistant_identity)?;
            (
                "gemma4-external",
                capacity,
                assistant_identity,
                artifact_identity,
                format_identity,
                format!(
                    "block={};publishers={publishers:?};hidden={hidden}",
                    assistant.block_size
                ),
                capture,
                state,
                target.text.num_experts.is_some(),
            )
        }
        (
            DispatchedExternalAssistantPreparation::MuseGlimmer(prepared),
            ExternalAssistantTargetProfile::MuseGlimmer(target),
            crate::composite_execution::ExternalPredictionCaptureRequest::MuseGlimmerDFlash {
                target_layers,
                target_paths,
            },
        ) => {
            let assistant = prepared.config();
            let _compatibility = assistant
                .prove_compatibility(target)
                .map_err(invalid_assistant)?;
            let capacity = NonZeroUsize::new(assistant.block_size.saturating_sub(1).min(15))
                .ok_or_else(|| {
                    invalid_assistant("Muse-Glimmer assistant proposal capacity is zero")
                })?;
            let hidden = external_positive(target.hidden_size, "Muse-Glimmer target hidden size")?;
            if target_layers.len() != target_paths.len() {
                return Err(invalid_assistant(
                    "Muse-Glimmer target capture layer/path counts disagree",
                ));
            }
            let entries = target_layers
                .iter()
                .zip(target_paths.iter())
                .enumerate()
                .map(|(position, (layer, path))| {
                    SpeculativeCaptureEntry::new(
                        external_contract_identity(path.clone())?,
                        vec![batch, sequence, hidden],
                        owner.clone(),
                        external_contract_identity(format!(
                            "muse-glimmer.target_layers.{position}.layer.{layer}"
                        ))?,
                    )
                    .and_then(|entry| entry.with_bounded_dimension(1))
                    .map_err(invalid_assistant)
                })
                .collect::<Result<Vec<_>, ArtifactError>>()?;
            let capture = SpeculativeCaptureSchema::new(
                    external_contract_identity(format!(
                        "muse-glimmer-dflash/capture/batch={batch};sequence={sequence};layers={target_layers:?}"
                    ))?,
                    entries,
                )
                .map_err(invalid_assistant)?;
            let mut state = vec![
                "target.verification_checkpoint".to_owned(),
                "assistant.pending_target_context".to_owned(),
                "assistant.encoded_context".to_owned(),
                "assistant.proposal_logits".to_owned(),
                "assistant.proposal_cursor".to_owned(),
            ];
            for layer in
                0..usize::try_from(assistant.num_hidden_layers).map_err(invalid_assistant)?
            {
                state.push(format!("assistant.projected_context.{layer}.keys"));
                state.push(format!("assistant.projected_context.{layer}.values"));
            }
            let assistant_identity = external_contract_identity(format!(
                "muse-glimmer-assistant/model={};block={};layers={};hidden={};targets={target_layers:?}",
                assistant.model_type,
                assistant.block_size,
                assistant.num_hidden_layers,
                assistant.hidden_size,
            ))?;
            let (artifact_identity, format_identity) = prepared
                .checkpoint()
                .speculative_identities(&assistant_identity)?;
            (
                "muse-glimmer-dflash",
                capacity,
                assistant_identity,
                artifact_identity,
                format_identity,
                format!(
                    "block={};sliding={};mask={};layers={target_layers:?}",
                    assistant.block_size, assistant.sliding_window, assistant.mask_token_id
                ),
                capture,
                state,
                target.num_experts > 0,
            )
        }
        _ => {
            return Err(invalid_assistant(
                "compatible external assistant contract family or capture mismatch",
            ));
        }
    };

    if request.maximum_draft_tokens > architecture_capacity {
        return Err(invalid_assistant(format!(
            "requested external draft capacity {} exceeds architecture capacity {}",
            request.maximum_draft_tokens, architecture_capacity
        )));
    }
    let strategy_identity = external_contract_identity(format!(
        "{family}/{strategy_detail};batch={batch};sequence={sequence}"
    ))?;
    let strategy = SpeculativeStrategyRequirements::external(
        strategy_identity.clone(),
        request.maximum_draft_tokens,
        request.tokenizer_fingerprint,
    );
    state.push("assistant.materialized_parameters".to_owned());
    let state_components = state
        .into_iter()
        .map(|component| external_contract_identity(format!("{family}/{component}")))
        .collect::<Result<Vec<_>, _>>()?;
    let identity = SpeculativeStateCacheIdentityIngredients::new(
        target_identity.clone(),
        strategy_identity.clone(),
        Some(assistant_identity),
        Some(request.tokenizer_fingerprint),
        artifact_identity,
        format_identity,
        topology,
        request.topology.global_rank(),
        request.processor,
        state_components,
    )
    .map_err(invalid_assistant)?;
    let mechanisms = SpeculativeMechanismRequirements::new(
        grouped.then_some(SpeculativeMechanism::GroupedNeuralOperations),
    );
    let requirements = SpeculativeRealizationRequirements::new(
        target_identity.clone(),
        strategy,
        capture.clone(),
        mechanisms,
        identity,
    )
    .map_err(invalid_assistant)?;
    let architecture_proof = SpeculativeArchitectureCompatibilityProof::new(
        target_identity,
        strategy_identity,
        capture.identity().clone(),
    );
    Ok(ExternalSpeculativeContract {
        requirements,
        architecture_proof,
        tokenizer_proof: request.tokenizer_proof,
        target_capture: capture,
    })
}

/// Inspects and admits an external draft assistant without selecting a backend.
///
/// Configuration, container format, family dispatch, GGUF metadata, and the
/// strict tensor contract and resolved layout are fixed here. Concrete
/// backends receive only this plan and may open or map the selected weight
/// payloads during materialization after revalidating that admission.
pub fn prepare_external_assistant(
    source: impl AsRef<std::path::Path>,
) -> Result<ExternalAssistantPreparation, ArtifactError> {
    let inspection = eredu_core::inspect_artifact(source, &AssistantConfigurations)?;
    let family = inspection.configuration().family();
    let config_json = inspection.configuration().json();
    let metadata = inspection.gguf_checkpoint().map(gguf_metadata);

    match family {
        "gemma4_assistant" => {
            let config = match inspection.format() {
                ArtifactFormat::SafeTensors => gemma4::AssistantConfig::from_json(
                    &serde_json::to_vec(config_json.expect("SafeTensors inspection has JSON"))?,
                )
                .map_err(invalid_assistant)?,
                ArtifactFormat::Gguf => gemma4::AssistantConfig::from_gguf_metadata(
                    inspection
                        .gguf_checkpoint()
                        .expect("GGUF inspection has checkpoint"),
                    metadata.as_ref().expect("GGUF inspection has metadata"),
                )
                .map_err(invalid_assistant)?,
                _ => {
                    return Err(ArtifactError::InvalidArtifact(
                        "unsupported Gemma assistant artifact format".into(),
                    ));
                }
            };
            let checkpoint = prepared_checkpoint(
                &inspection,
                || gemma4::assistant_safetensors_plan(&config),
                || gemma4::assistant_gguf_plan(&config),
                gemma4::translate_assistant_gguf_weight_name,
            )?;
            Ok(ExternalAssistantPreparation {
                dispatched: DispatchedExternalAssistantPreparation::Gemma4(
                    PreparedExternalAssistant {
                        checkpoint,
                        config,
                        _architecture: PhantomData,
                    },
                ),
            })
        }
        "muse_glimmer_assistant" => {
            let config = match inspection.format() {
                ArtifactFormat::SafeTensors => muse_glimmer::DFlashConfig::from_hf_json(
                    &serde_json::to_vec(config_json.expect("SafeTensors inspection has JSON"))?,
                )
                .map_err(invalid_assistant)?,
                ArtifactFormat::Gguf => muse_glimmer::DFlashConfig::from_gguf_metadata(
                    metadata.as_ref().expect("GGUF inspection has metadata"),
                )
                .map_err(invalid_assistant)?,
                _ => {
                    return Err(ArtifactError::InvalidArtifact(
                        "unsupported Muse/Glimmer assistant artifact format".into(),
                    ));
                }
            };
            let checkpoint = prepared_checkpoint(
                &inspection,
                || muse_glimmer::dflash_safetensors_plan(&config),
                || muse_glimmer::dflash_gguf_plan(&config),
                muse_glimmer::translate_dflash_gguf_weight_name,
            )?;
            Ok(ExternalAssistantPreparation {
                dispatched: DispatchedExternalAssistantPreparation::MuseGlimmer(
                    PreparedExternalAssistant {
                        checkpoint,
                        config,
                        _architecture: PhantomData,
                    },
                ),
            })
        }
        other => Err(ArtifactError::UnsupportedModelType(other.into())),
    }
}

fn prepared_checkpoint(
    inspection: &eredu_core::ArtifactInspection,
    safetensors_plan: impl FnOnce() -> Result<SafetensorsCheckpointPlan, String>,
    gguf_plan: impl FnOnce() -> Result<eredu_checkpoint::schema::GgufCheckpointPlan, String>,
    gguf_translate: impl FnMut(&str) -> String,
) -> Result<ExternalAssistantCheckpoint, ArtifactError> {
    match inspection.format() {
        ArtifactFormat::SafeTensors => {
            let plan = safetensors_plan().map_err(ArtifactError::InvalidArtifact)?;
            let resolution = resolve_safetensors_plan(
                &crate::configuration::PortableSafetensorsCatalog(inspection.tensors()),
                &plan,
            )
            .map_err(|validation| {
                invalid_assistant(format!(
                    "external assistant checkpoint contract did not resolve: {validation:?}"
                ))
            })?;
            Ok(ExternalAssistantCheckpoint::SafeTensors {
                source: inspection.path().to_owned(),
                catalog: inspection.tensors().clone(),
                plan,
                resolution,
            })
        }
        ArtifactFormat::Gguf => {
            let checkpoint = inspection
                .gguf_checkpoint()
                .expect("GGUF inspection has checkpoint");
            let plan = gguf_plan().map_err(ArtifactError::InvalidArtifact)?;
            let resolution = resolve_gguf_plan(checkpoint, &plan).map_err(|validation| {
                ArtifactError::InvalidArtifact(format!(
                    "external assistant checkpoint contract did not resolve: {validation:?}"
                ))
            })?;
            let tensor_mapping = checkpoint
                .translated_outputs(gguf_translate)
                .map_err(|error| ArtifactError::InvalidArtifact(error.to_string()))?;
            Ok(ExternalAssistantCheckpoint::Gguf {
                checkpoint: checkpoint.clone(),
                resolution,
                tensor_mapping,
            })
        }
        _ => Err(ArtifactError::InvalidArtifact(
            "unsupported external-assistant artifact format".into(),
        )),
    }
}

fn invalid_assistant(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::InvalidArtifact(error.to_string())
}

fn gguf_metadata(checkpoint: &Checkpoint) -> HashMap<String, MetadataValue> {
    checkpoint
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

struct AssistantConfigurations;

impl ModelConfigurationResolver for AssistantConfigurations {
    type ArtifactPlan = ();

    fn resolve_safetensors(
        &self,
        json: &Value,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        let bytes = serde_json::to_vec(json)?;
        let model_type = json
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ArtifactError::InvalidArtifact("assistant config is missing model_type".into())
            })?;
        let family = match model_type {
            "gemma4_assistant" => {
                gemma4::AssistantConfig::from_json(&bytes).map_err(invalid_assistant)?;
                "gemma4_assistant"
            }
            "muse_glimmer_assistant" => {
                muse_glimmer::DFlashConfig::from_hf_json(&bytes).map_err(invalid_assistant)?;
                "muse_glimmer_assistant"
            }
            other => return Err(ArtifactError::UnsupportedModelType(other.into())),
        };
        Ok(ResolvedModelConfiguration::new(
            ModelConfiguration::new(
                model_type,
                model_type,
                family,
                LoadingProtocol::Model,
                Some(json.clone()),
            )?,
            (),
        ))
    }

    fn resolve_gguf(
        &self,
        architecture: &str,
        checkpoint: &Checkpoint,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        let metadata = gguf_metadata(checkpoint);
        let family = match architecture {
            "gemma4_assistant" | "gemma4-assistant" => {
                gemma4::AssistantConfig::from_gguf_metadata(checkpoint, &metadata)
                    .map_err(invalid_assistant)?;
                "gemma4_assistant"
            }
            "dflash" => {
                muse_glimmer::DFlashConfig::from_gguf_metadata(&metadata)
                    .map_err(invalid_assistant)?;
                "muse_glimmer_assistant"
            }
            other => return Err(ArtifactError::UnsupportedGgufArchitecture(other.into())),
        };
        Ok(ResolvedModelConfiguration::new(
            ModelConfiguration::new(architecture, family, family, LoadingProtocol::Model, None)?,
            (),
        ))
    }

    fn gguf_companion_requirements(
        &self,
        _architecture: &str,
        _checkpoint: &Checkpoint,
    ) -> Result<Vec<eredu_core::GgufCompanionRequirement>, ArtifactError> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::schema::StoredDtypeConstraint;
    use eredu_core::{ParallelTopology, TokenizerCompatibilityProof};
    use eredu_runtime::{
        select_speculative_realization, SpeculativeMechanismCapabilities,
        SpeculativePlacementRequest, SpeculativeStrategyClass,
    };
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::{
        convert::Infallible,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    const GEMMA_ASSISTANT: &str = r#"{
      "model_type":"gemma4_assistant","backbone_hidden_size":32,
      "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
      "text_config":{"model_type":"gemma4_text","hidden_size":32,
        "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
        "vocab_size":32,"max_position_embeddings":128,"tie_word_embeddings":false,
        "attention_k_eq_v":false,"layer_types":["full_attention"]}
    }"#;

    const MUSE_ASSISTANT: &str = r#"{
      "model_type":"muse_glimmer_assistant","hidden_size":6656,
      "intermediate_size":19968,"num_hidden_layers":5,"num_attention_heads":32,
      "num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.000001,
      "max_position_embeddings":131072,"sliding_window":2048,"block_size":16,
      "mask_token_id":201818,"target_layer_ids":[1,13,25,37,49],
      "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                     "sliding_attention","sliding_attention"],
      "hidden_act":"silu","attention_dropout":0.0,
      "rope_parameters":{"rope_theta":500000.0}
    }"#;

    type TestTensor = (String, Vec<usize>, Vec<u8>);

    fn gemma_tensors() -> Vec<TestTensor> {
        let config = gemma4::AssistantConfig::from_json(GEMMA_ASSISTANT.as_bytes()).unwrap();
        let plan = gemma4::assistant_safetensors_plan(&config).unwrap();
        assert!(plan.layout_groups.is_empty());
        plan.common_tensors
            .into_iter()
            .map(|tensor| {
                assert_eq!(tensor.dtype, StoredDtypeConstraint::Floating);
                let elements = tensor.shape.iter().product::<usize>();
                (tensor.key, tensor.shape, vec![0; elements * 4])
            })
            .collect()
    }

    fn safetensors_artifact(config: &str, tensors: Vec<TestTensor>) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), config).unwrap();
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn identity(value: &str) -> SpeculativeIdentity {
        SpeculativeIdentity::new(value).unwrap()
    }

    fn replicated_topology() -> ParallelRankTopology {
        ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap()
    }

    fn contract_request(
        capacity: usize,
        fingerprint: [u8; 32],
        proof: TokenizerCompatibilityProof,
    ) -> ExternalSpeculativeContractRequest {
        ExternalSpeculativeContractRequest::new(
            replicated_topology(),
            identity("processor-v1"),
            proof,
            fingerprint,
            NonZeroUsize::new(capacity).unwrap(),
        )
    }

    fn compatible_gemma_target() -> ExternalAssistantTargetProfile {
        let assistant = gemma4::AssistantConfig::from_json(GEMMA_ASSISTANT.as_bytes()).unwrap();
        let mut text = assistant.text_config;
        let mut publisher = *text.layer_schedule.get(0).unwrap();
        publisher.key_value = eredu_nn::AttentionStateSource::Publish {
            value: eredu_nn::AttentionValueSource::Projected,
        };
        text.layer_schedule = eredu_core::LayerSchedule::new(1, vec![publisher]).unwrap();
        ExternalAssistantTargetProfile::Gemma4(gemma4::FamilyConfig {
            model_type: "gemma4".into(),
            text,
            vision: None,
            image_token_id: None,
            video_token_id: None,
            audio: None,
            audio_token_id: None,
        })
    }

    fn compatible_muse_target() -> ExternalAssistantTargetProfile {
        let target = serde_json::json!({
            "architectures":["MuseGlimmerForConditionalGeneration"],
            "model_type":"muse_glimmer","image_token_id":22,"video_token_id":23,
            "out_hidden_size":32,"projector_hidden_size":16,
            "text_config":{
                "model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":2,
                "intermediate_size":24,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"rms_norm_eps":0.00001,"post_norm_eps":0.00001,
                "vocab_size":24,"max_position_embeddings":64,"rope_theta":10000.0,
                "layer_types":["sliding_attention","full_attention"],
                "layer_rope_theta":[10000.0,0.0],"sliding_window":8,
                "tie_word_embeddings":false,"hidden_act":"silu","attention_dropout":0.0,
                "qk_scale_factor":1.0,"output_multiplier":1.0,
                "final_logit_softcapping":30.0
            },
            "vision_config":{
                "model_type":"muse_glimmer_vision","hidden_size":8,
                "intermediate_size":12,"num_attention_heads":2,"num_hidden_layers":1,
                "patch_size":2,"patch_temporal":1,"merge_size":2,
                "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,
                "layer_norm_eps":0.00001,"hidden_act":"gelu",
                "layer_types":["full_attention"],
                "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}
            }
        });
        let mut target = muse_glimmer::DecoderConfig::from_hf_value(&target).unwrap();
        target.hidden_size = 6656;
        target.num_hidden_layers = 50;
        target.vocab_size = 201819;
        target.max_position_embeddings = 131072;
        ExternalAssistantTargetProfile::MuseGlimmer(target)
    }

    fn prepared_muse_without_payload_open() -> ExternalAssistantPreparation {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let dummy_checkpoint = match prepare_external_assistant(artifact.path())
            .unwrap()
            .dispatched
        {
            DispatchedExternalAssistantPreparation::Gemma4(prepared) => prepared.checkpoint,
            DispatchedExternalAssistantPreparation::MuseGlimmer(_) => unreachable!(),
        };
        ExternalAssistantPreparation {
            dispatched: DispatchedExternalAssistantPreparation::MuseGlimmer(
                PreparedExternalAssistant {
                    checkpoint: dummy_checkpoint,
                    config: muse_glimmer::DFlashConfig::from_hf_json(MUSE_ASSISTANT.as_bytes())
                        .unwrap(),
                    _architecture: PhantomData,
                },
            ),
        }
    }

    struct InspectedPreparation {
        checkpoint: ExternalAssistantCheckpoint,
        tokenizer_model_kind: crate::configuration::ModelKind,
        model_type: String,
    }

    struct InspectPreparation;

    impl ExternalAssistantPreparationVisitor for InspectPreparation {
        type Output<A: ExternalAssistantArchitecture> = InspectedPreparation;
        type Error = Infallible;

        fn visit<A: ExternalAssistantArchitecture>(
            self,
            prepared: PreparedExternalAssistant<A>,
        ) -> Result<Self::Output<A>, Self::Error> {
            let model_type = A::configuration_model_type(prepared.config()).to_owned();
            let (checkpoint, _) = prepared.into_parts();
            Ok(InspectedPreparation {
                checkpoint,
                tokenizer_model_kind: A::tokenizer_model_kind(),
                model_type,
            })
        }
    }

    struct TakeInspection;

    impl MaterializedExternalAssistantVisitor<InspectPreparation> for TakeInspection {
        type Output = InspectedPreparation;

        fn visit<A: ExternalAssistantArchitecture>(
            self,
            assistant: &mut InspectedPreparation,
        ) -> Self::Output {
            InspectedPreparation {
                checkpoint: assistant.checkpoint.clone(),
                tokenizer_model_kind: assistant.tokenizer_model_kind,
                model_type: assistant.model_type.clone(),
            }
        }
    }

    struct CountPayloadOpens(Arc<AtomicUsize>);

    impl ExternalAssistantPreparationVisitor for CountPayloadOpens {
        type Output<A: ExternalAssistantArchitecture> = ();
        type Error = Infallible;

        fn visit<A: ExternalAssistantArchitecture>(
            self,
            _prepared: PreparedExternalAssistant<A>,
        ) -> Result<Self::Output<A>, Self::Error> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn safetensors_assistant_visits_one_typed_family_blind_materializer() {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let preparation = prepare_external_assistant(artifact.path()).unwrap();
        assert_eq!(
            preparation.tokenizer_model_kind(),
            crate::configuration::ModelKind::Gemma4
        );
        let mut materialized = preparation.visit(InspectPreparation).unwrap();
        let inspected = materialized.visit(TakeInspection);
        assert_eq!(
            inspected.tokenizer_model_kind,
            crate::configuration::ModelKind::Gemma4
        );
        assert_eq!(inspected.model_type, "gemma4_assistant");
        assert!(matches!(
            inspected.checkpoint,
            ExternalAssistantCheckpoint::SafeTensors {
                source,
                catalog,
                resolution,
                ..
            }
                if source == artifact.path()
                    && catalog.len() == resolution.source_keys().len()
        ));
    }

    #[test]
    fn muse_assistant_visits_the_same_typed_family_blind_materializer() {
        let preparation = prepared_muse_without_payload_open();
        assert_eq!(
            preparation.tokenizer_model_kind(),
            crate::configuration::ModelKind::MuseGlimmer
        );
        let mut materialized = preparation.visit(InspectPreparation).unwrap();
        let inspected = materialized.visit(TakeInspection);
        assert_eq!(
            inspected.tokenizer_model_kind,
            crate::configuration::ModelKind::MuseGlimmer
        );
        assert_eq!(inspected.model_type, "muse_glimmer_assistant");
    }

    #[test]
    fn gemma_external_contract_preserves_capture_state_and_tokenizer_proofs() {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let target = compatible_gemma_target();
        let compatible = prepare_external_assistant(artifact.path())
            .unwrap()
            .prove_target_compatibility(&target)
            .unwrap();
        let fingerprint = [7; 32];
        let tokenizer = TokenizerCompatibilityProof::prove(fingerprint, fingerprint).unwrap();
        let contract = compatible
            .speculative_contract(contract_request(2, fingerprint, tokenizer))
            .unwrap();

        assert_eq!(
            contract.requirements().strategy().class(),
            SpeculativeStrategyClass::External
        );
        assert_eq!(
            contract.requirements().strategy().proposal_capacity().get(),
            2
        );
        assert_eq!(contract.tokenizer_proof().fingerprint(), fingerprint);
        assert_eq!(
            contract
                .target_capture()
                .entries()
                .iter()
                .map(|entry| entry.path().as_str())
                .collect::<Vec<_>>(),
            [
                "model.language_model.layers.0.output",
                "model.language_model.layers.0.shared_attention.keys",
                "model.language_model.layers.0.shared_attention.values",
            ]
        );
        assert_eq!(contract.target_capture().entries()[0].shape(), [1, 128, 32]);
        assert_eq!(
            contract.target_capture().entries()[1].shape(),
            [1, 2, 128, 8]
        );
        assert!(contract
            .requirements()
            .state()
            .state_components()
            .iter()
            .any(|component| component.as_str().contains("assistant.private_kv.0.keys")));

        let capabilities = SpeculativeMechanismCapabilities::new(
            contract
                .requirements()
                .mechanisms()
                .mechanisms()
                .iter()
                .copied(),
        );
        let selected = select_speculative_realization(
            contract.requirements(),
            &contract.selection_request(SpeculativePlacementRequest::Single),
            &capabilities,
        )
        .unwrap();
        assert_eq!(selected.requirements(), contract.requirements());
    }

    #[test]
    fn external_contract_rejects_tokenizer_and_capacity_mismatches() {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let target = compatible_gemma_target();
        let compatible = prepare_external_assistant(artifact.path())
            .unwrap()
            .prove_target_compatibility(&target)
            .unwrap();
        let proof = TokenizerCompatibilityProof::prove([3; 32], [3; 32]).unwrap();
        assert!(compatible
            .speculative_contract(contract_request(2, [4; 32], proof))
            .is_err());

        let proof = TokenizerCompatibilityProof::prove([3; 32], [3; 32]).unwrap();
        let error = compatible
            .speculative_contract(contract_request(4, [3; 32], proof))
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds architecture capacity 3"));

        assert_eq!(target.maximum_capture_sequence_length().unwrap().get(), 128);
        assert!(target
            .speculative_identity()
            .unwrap()
            .as_str()
            .contains("gemma4-target/profile=gemma4"));
    }

    #[test]
    fn external_cache_envelope_binds_identity_frontier_and_capture_causally() {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let target = compatible_gemma_target();
        let compatible = prepare_external_assistant(artifact.path())
            .unwrap()
            .prove_target_compatibility(&target)
            .unwrap();
        let fingerprint = [13; 32];
        let tokenizer = TokenizerCompatibilityProof::prove(fingerprint, fingerprint).unwrap();
        let contract = compatible
            .speculative_contract(contract_request(2, fingerprint, tokenizer))
            .unwrap();
        let capabilities = SpeculativeMechanismCapabilities::new(
            contract
                .requirements()
                .mechanisms()
                .mechanisms()
                .iter()
                .copied(),
        );
        let selected = select_speculative_realization(
            contract.requirements(),
            &contract.selection_request(SpeculativePlacementRequest::Single),
            &capabilities,
        )
        .unwrap();
        let mut cache = ExternalAssistantCache::new((), selected);
        let prepared = identity("prepared-input/content-a");
        cache.bind_prepared_input(prepared.clone()).unwrap();
        assert!(cache
            .bind_prepared_input(identity("prepared-input/content-b"))
            .unwrap_err()
            .contains("different prepared input"));
        cache.advance_frontier(3).unwrap();
        let shapes = contract
            .target_capture()
            .entries()
            .iter()
            .map(|entry| entry.shape().to_vec())
            .collect::<Vec<_>>();
        cache.validate_capture_shapes(&shapes).unwrap();

        let mut wrong = shapes.clone();
        wrong[0][2] += 1;
        assert!(cache.validate_capture_shapes(&wrong).is_err());
        let checkpoint = cache.checkpoint(());
        cache.advance_frontier(7).unwrap();
        cache.restore_semantics(&checkpoint);
        assert_eq!(cache.frontier().unwrap(), 3);
        cache.bind_prepared_input(prepared).unwrap();
    }

    #[test]
    fn external_contract_rejects_partitioned_topology_before_payload_materialization() {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let target = compatible_gemma_target();
        let compatible = prepare_external_assistant(artifact.path())
            .unwrap()
            .prove_target_compatibility(&target)
            .unwrap();
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
        let fingerprint = [5; 32];
        let tokenizer = TokenizerCompatibilityProof::prove(fingerprint, fingerprint).unwrap();
        let request = ExternalSpeculativeContractRequest::new(
            topology,
            identity("processor-v1"),
            tokenizer,
            fingerprint,
            NonZeroUsize::new(2).unwrap(),
        );
        let payload_opens = Arc::new(AtomicUsize::new(0));

        match compatible.speculative_contract(request) {
            Ok(_) => {
                compatible
                    .visit(CountPayloadOpens(payload_opens.clone()))
                    .unwrap();
                panic!("partitioned external target unexpectedly produced a contract")
            }
            Err(error) => assert!(
                error
                    .to_string()
                    .contains("external speculative targets require replicated topology"),
                "{error}"
            ),
        }
        assert_eq!(payload_opens.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn muse_external_contract_preserves_declared_capture_order_and_context_state() {
        let target = compatible_muse_target();
        let compatible = prepared_muse_without_payload_open()
            .prove_target_compatibility(&target)
            .unwrap();
        let fingerprint = [9; 32];
        let tokenizer = TokenizerCompatibilityProof::prove(fingerprint, fingerprint).unwrap();
        let contract = compatible
            .speculative_contract(contract_request(8, fingerprint, tokenizer))
            .unwrap();

        assert_eq!(
            contract
                .target_capture()
                .entries()
                .iter()
                .map(|entry| entry.path().as_str())
                .collect::<Vec<_>>(),
            [
                "model.layers.1.output",
                "model.layers.13.output",
                "model.layers.25.output",
                "model.layers.37.output",
                "model.layers.49.output",
            ]
        );
        assert!(contract
            .target_capture()
            .entries()
            .iter()
            .all(|entry| entry.shape() == [1, 131072, 6656]));
        assert_eq!(
            contract.requirements().strategy().proposal_capacity().get(),
            8
        );
        let state = contract.requirements().state().state_components();
        assert!(state
            .iter()
            .any(|component| component.as_str().contains("assistant.encoded_context")));
        assert!(state.iter().any(|component| component
            .as_str()
            .contains("assistant.projected_context.4.values")));
    }

    #[test]
    fn family_mismatch_causally_prevents_payload_materialization() {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let preparation = prepare_external_assistant(artifact.path()).unwrap();
        let target = serde_json::json!({
            "architectures":["MuseGlimmerForConditionalGeneration"],
            "model_type":"muse_glimmer",
            "image_token_id":22,"video_token_id":23,
            "out_hidden_size":32,"projector_hidden_size":16,
            "text_config":{
                "model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":2,
                "intermediate_size":24,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"rms_norm_eps":0.00001,"post_norm_eps":0.00001,
                "vocab_size":24,"max_position_embeddings":64,"rope_theta":10000.0,
                "layer_types":["sliding_attention","full_attention"],
                "layer_rope_theta":[10000.0,0.0],"sliding_window":8,
                "tie_word_embeddings":false,"hidden_act":"silu","attention_dropout":0.0,
                "qk_scale_factor":1.0,"output_multiplier":1.0,
                "final_logit_softcapping":30.0
            },
            "vision_config":{
                "model_type":"muse_glimmer_vision","hidden_size":8,
                "intermediate_size":12,"num_attention_heads":2,"num_hidden_layers":1,
                "patch_size":2,"patch_temporal":1,"merge_size":2,
                "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,
                "layer_norm_eps":0.00001,"hidden_act":"gelu",
                "layer_types":["full_attention"],
                "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}
            }
        });
        let target = ExternalAssistantTargetProfile::MuseGlimmer(
            crate::muse_glimmer::DecoderConfig::from_hf_value(&target).unwrap(),
        );
        let payload_opens = Arc::new(AtomicUsize::new(0));
        match preparation.prove_target_compatibility(&target) {
            Ok(compatible) => {
                compatible
                    .visit(CountPayloadOpens(payload_opens.clone()))
                    .unwrap();
                panic!("mismatched external assistant unexpectedly passed compatibility")
            }
            Err(error) => assert!(error.contains("family does not match")),
        }
        assert_eq!(payload_opens.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn safetensors_assistant_rejects_missing_extra_and_malformed_tensors_during_preparation() {
        let mut missing = gemma_tensors();
        missing.pop();
        let missing = safetensors_artifact(GEMMA_ASSISTANT, missing);
        assert!(matches!(
            prepare_external_assistant(missing.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));

        let mut extra = gemma_tensors();
        extra.push(("undeclared.weight".into(), vec![1], vec![0; 4]));
        let extra = safetensors_artifact(GEMMA_ASSISTANT, extra);
        assert!(matches!(
            prepare_external_assistant(extra.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));

        let mut malformed = gemma_tensors();
        malformed[0].1.push(1);
        let malformed = safetensors_artifact(GEMMA_ASSISTANT, malformed);
        assert!(matches!(
            prepare_external_assistant(malformed.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn safetensors_assistant_admission_rejects_missing_or_conflicting_identities() {
        let mut missing: Value = serde_json::from_str(GEMMA_ASSISTANT).unwrap();
        missing.as_object_mut().unwrap().remove("model_type");
        let missing = safetensors_artifact(&missing.to_string(), gemma_tensors());
        assert!(matches!(
            prepare_external_assistant(missing.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));

        let mut conflicting: Value = serde_json::from_str(GEMMA_ASSISTANT).unwrap();
        conflicting["text_config"]["model_type"] = "llama".into();
        let conflicting = safetensors_artifact(&conflicting.to_string(), gemma_tensors());
        assert!(matches!(
            prepare_external_assistant(conflicting.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn ordinary_model_cannot_cross_the_external_assistant_boundary() {
        let artifact = safetensors_artifact(
            r#"{"model_type":"llama"}"#,
            vec![("weight".into(), vec![1], vec![0; 4])],
        );
        assert!(matches!(
            prepare_external_assistant(artifact.path()),
            Err(ArtifactError::UnsupportedModelType(model_type)) if model_type == "llama"
        ));
    }
}
