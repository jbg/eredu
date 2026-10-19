//! Architecture-owned inspection and preparation of external draft assistants.

mod capture_placement;
mod construction;
use construction::NormalizedExternalAssistant;
mod static_placement;
mod target_operation;
pub use capture_placement::PreparedTargetCapturePlacement;
pub use static_placement::PreparedTargetStaticPlacement;
pub(crate) use static_placement::TargetStaticDescription;
pub use target_operation::PreparedExternalTargetOperation;
mod execution_plan;
mod parameters;
pub use execution_plan::{
    prepare_execution_plan_assistant, MaterializedExternalAssistantExecution,
    PreparedExternalAssistantExecution,
};
pub use parameters::ExternalAssistantParameterUnit;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    marker::PhantomData,
    num::NonZeroUsize,
    path::PathBuf,
    sync::Arc,
};

use eredu_checkpoint::{
    recipe::RecipeCatalog,
    schema::{SafetensorsCheckpointPlan, TensorRequirement, TensorRole},
    store::{CheckpointSource, StoreError, TensorMetadata, TensorSourceProvenance},
    validation::{resolve_gguf_plan, resolve_safetensors_plan, ResolvedCheckpointPlan},
    LinearFormat, WeightQuantization,
};
use eredu_core::{
    artifact::{ArtifactError, ArtifactIdentity, DeferredArtifactIdentity},
    checkpoint::TensorCatalog,
    ArtifactFormat, LoadingProtocol, ModelConfiguration, ModelConfigurationResolver,
    ParallelRankTopology, ResolvedModelConfiguration, TokenizerCompatibilityProof,
};
use eredu_gguf::{Checkpoint, MetadataValue};
use serde_json::Value;

use crate::{gemma4, muse};
use muse::assistant::integration::{
    MuseAssistantConfig, MuseAssistantConstruction, MuseAssistantModule,
};

use eredu_runtime::{
    ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion, ReplicatedTextParameterOwner,
    ReplicatedTextParameterRole, ReplicatedTextPhysicalSource,
    SpeculativeArchitectureCompatibilityProof, SpeculativeCaptureEntry, SpeculativeCaptureEnvelope,
    SpeculativeCaptureMetadata, SpeculativeCaptureSchema, SpeculativeIdentity,
    SpeculativeMechanism, SpeculativeMechanismRequirements, SpeculativePlacementRequest,
    SpeculativeRealizationRequirements, SpeculativeSelectionRequest,
    SpeculativeStateCacheIdentityIngredients, SpeculativeStrategyRequirements,
    WeightLoweringDescriptor, WeightLoweringKind,
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

    /// Heap and envelope storage copied with a native checkpoint.
    pub fn control_metadata_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<ExternalAssistantCacheCheckpoint<C>>() as u64).checked_add(
            self.prepared_input
                .as_ref()
                .map_or(0, |id| id.as_str().len() as u64),
        )
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
    /// Normalized Muse target configuration, retaining its distinct released identity.
    Muse(muse::DecoderConfig),
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
            Self::Muse(config) => format!(
                "{}-target/profile={};geometry={}",
                match config.identity {
                    muse::MuseIdentity::Glimmer => "muse-glimmer",
                    muse::MuseIdentity::Spark => "muse-spark",
                },
                config.identity.text_model_type(),
                config.architecture_fingerprint()
            ),
        };
        external_contract_identity(identity)
    }

    /// Returns the architecture-declared upper bound for capture sequence geometry.
    pub fn maximum_capture_sequence_length(&self) -> Result<NonZeroUsize, ArtifactError> {
        let maximum = match self {
            Self::Gemma4(config) => config.text.max_position_embeddings,
            Self::Muse(config) => config.max_position_embeddings,
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
        /// Exact canonical shard set admitted during header inspection.
        shards: eredu_checkpoint::safetensors::SafetensorsShards,
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
                source: _,
                shards: _,
                catalog: _,
                plan,
                resolution,
            } => {
                let artifact = external_contract_identity(format!(
                    "admission/safetensors/profile={};plan={plan:?};resolution={resolution:?}",
                    assistant_profile.as_str(),
                ))?;
                let format = external_contract_identity(format!(
                    "safetensors/profile={};plan={plan:?};resolution={resolution:?}",
                    assistant_profile.as_str(),
                ))?;
                Ok((artifact, format))
            }
            Self::Gguf {
                checkpoint: _,
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
                    "admission/gguf/profile={};resolution={resolution:?};mapping={mapping}",
                    assistant_profile.as_str(),
                ))?;
                let format = external_contract_identity(format!(
                    "gguf/profile={};resolution={resolution:?};mapping={mapping}",
                    assistant_profile.as_str(),
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
    /// Complete neural construction geometry after target compatibility is proven.
    type ModuleConfig: Clone + std::fmt::Debug;

    /// Retained resource equations, absent when the assistant projection is unimplemented.
    fn memory_topology(
        _config: &Self::ModuleConfig,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::ConditionedPredictionTopology>,
        eredu_nn::Error,
    > {
        Ok(None)
    }

    /// Normalizes construction without opening payloads or binding native resources.
    fn module_config(
        config: &Self::Config,
        target: Option<&ExternalAssistantTargetProfile>,
    ) -> Result<Self::ModuleConfig, String>;

    /// Backend-neutral assistant module specialized to a neutral neural backend.
    type Module<B>: Clone + eredu_nn::Parameterized<B::Tensor>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// One true independently constructed parameter unit.
    type Unit<B>: eredu_nn::Parameterized<B::Tensor> + 'static
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Execution container specialized to the selected ordinary parameter policy.
    /// It need not implement parameter traversal or cloning.
    type Execution<B, P>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Canonical unit construction, or a separately constructed resident module.
    fn execution_unit_layout(
        config: &Self::ModuleConfig,
    ) -> Result<Option<eredu_runtime::ExecutionUnitLayout>, eredu_nn::Error>;

    /// Constructs one unloaded unit from retained target-proven geometry.
    fn build_execution_unit<B>(
        config: &Self::ModuleConfig,
        ordinal: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Unit<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Consumes the construction tree once. The supplied constructor may retain
    /// its actual units or replace unloaded units with an exact-source policy.
    fn into_execution<B, P>(
        module: Self::Module<B>,
        policy: impl FnOnce(Vec<Self::Unit<B>>) -> P,
    ) -> Result<Self::Execution<B, P>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Borrows the selected policy when the runtime owns independently acquired units.
    fn parameter_policy<B, P>(execution: &Self::Execution<B, P>) -> Option<&P>
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
    fn tokenizer_model_type(config: &Self::Config) -> &'static str;

    /// Tokenizer ownership declared by the trained architecture before native construction.
    fn tokenizer_source(
        config: &Self::Config,
    ) -> crate::external_draft::ExternalDraftTokenizerSource {
        crate::external_draft::ExternalDraftTokenizerSource::Artifact {
            model_type: Self::tokenizer_model_type(config),
        }
    }

    /// Stable artifact identity carried by the normalized configuration.
    fn configuration_model_type(config: &Self::Config) -> &str;

    /// Builds the unloaded neutral assistant module.
    fn module<B>(
        config: Self::ModuleConfig,
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

    /// Describes the exact parameter topology constructed for this configuration.
    fn materialization_plan(config: &Self::Config) -> Result<SafetensorsCheckpointPlan, String>;

    /// Returns the executable format selected for one primary matrix parameter.
    fn linear_format(config: &Self::Config, name: &str) -> LinearFormat;

    /// Declares the independently acquired owner of one canonical primary parameter.
    /// Quantization companions inherit this owner rather than creating another unit.
    fn parameter_owner(
        config: &Self::Config,
        name: &str,
    ) -> Result<ReplicatedTextParameterOwner, String>;

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
///
/// Coordination defaults describe a local session. A distributed binding overrides
/// all three agreement methods with the same retained session authority. Checkpoints
/// copy model state, never that authority or its cumulative transport usage.
pub trait ExternalAssistantExecutionMechanisms<A: ExternalAssistantArchitecture>: 'static {
    /// Resolves cancellation and completion across the retained target session.
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        _context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Ok(local)
    }

    /// Agrees host preparation before any participant enters the next phase.
    fn agree_text_preparation(
        _stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        use eredu_core::run_preparation::{
            TextPreparationOutcome as O, TextPreparationStatus as S,
        };
        Ok(match status {
            S::Ready => O::Ready,
            S::Cancelled => O::Cancelled,
            S::Failed => O::Rejected { rank: 0 },
        })
    }

    /// Agrees explicit commands without executing model work or restoring credits.
    fn coordinate_decoding_control(
        local: eredu_core::decoding::DecodingCoordination,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        Ok(local)
    }

    /// Full score-row backing allowance for settled decoding handoffs.
    fn control_logits_estimate(
        _logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Neutral neural backend used by the materialized assistant module.
    type NeuralBackend: eredu_nn::GroupedNeuralBackend<Tensor = Self::Tensor>
        + eredu_nn::DistributedNeuralBackend
        + Clone;
    /// Selected complete parameter policy, independent of the target's residency.
    type ParameterPolicy: eredu_runtime::LayerwisePolicy<
        Self::NeuralBackend,
        A::Unit<Self::NeuralBackend>,
        Error: std::error::Error + Send + Sync + 'static,
    >;
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
    type Logits: Clone;
    /// Selected target/draft execution assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact verification completion.
    type Completion: eredu_core::BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        _target: &mut Self::Target,
        _assistant: &mut Self::Assistant,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Err(eredu_core::residency::ParameterConversionTrimError::Unsupported)
    }

    /// Reads participant budgets without touching caches or submitted work.
    fn parameter_conversion_retention(
        _target: &Self::Target,
        _assistant: &Self::Assistant,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

    /// Reads the installed target frontier, native capacity and parameter owners.
    fn target_memory_observation(
        _target: &Self::Target,
        _cache: &Self::NativeCache,
        _additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeModelMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

    /// Reads the current assistant parameter owner without acquiring any unit.
    fn assistant_memory_observation(
        _assistant: &Self::Assistant,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeParameterMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

    /// Complete available backing upper allowance for one borrowed tensor survey.
    /// Unknown or foreign backing stays unknown; observation never evaluates work.
    fn tensor_backing_bytes(
        _tensors: &[&Self::Tensor],
    ) -> Result<Option<u64>, eredu_core::BackendFailure> {
        Ok(None)
    }

    /// Known bound for an isolated reusable target-cache checkpoint.
    fn control_cache_estimate(
        _cache: &Self::NativeCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Complete conservative logical tensor-copy bound.
    fn control_tensor_bytes(_tensor: &Self::Tensor) -> Option<u64> {
        None
    }
    /// Isolated native checkpoint, available only with a known control estimate.
    fn control_checkpoint<'a>(
        cache: &Self::NativeCache,
        context: Self::Context<'a>,
    ) -> Result<Self::NativeCacheCheckpoint, Self::Error> {
        let _ = context;
        Self::checkpoint_native(cache)
    }
    /// Atomic isolated native restoration; never consumes saved state.
    fn control_restore<'a>(
        cache: &mut Self::NativeCache,
        saved: &Self::NativeCacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        Self::restore_checkpoint_native(cache, saved, context)
    }
    /// Copies one immutable seed tensor after budget admission.
    fn control_copy_tensor<'a>(
        tensor: &Self::Tensor,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (placement, context);
        Ok(tensor.clone())
    }

    /// Borrows the exact normalized assistant configuration.
    fn config(assistant: &Self::Assistant) -> &A::Config;
    /// Borrows the neutral assistant module.
    fn module(
        assistant: &mut Self::Assistant,
    ) -> &mut A::Execution<Self::NeuralBackend, Self::ParameterPolicy>;
    /// Validates the required operations against the retained group admission,
    /// then borrows its communicator from this invocation's target session.
    /// A caller cannot substitute an independently constructed equal-sized group.
    /// Replicated execution requires no tensor communicator.
    fn assistant_parallel_context<'a>(
        _context: Self::Context<'a>,
        _requirements: &eredu_runtime::CommunicationGroupRequirements,
    ) -> Result<
        Option<&'a <Self::NeuralBackend as eredu_nn::NeuralBackend>::ParallelContext>,
        Self::Error,
    > {
        Ok(None)
    }

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
    type ModuleConfig = gemma4::AssistantConfig;
    fn module_config(
        config: &Self::Config,
        _target: Option<&ExternalAssistantTargetProfile>,
    ) -> Result<Self::ModuleConfig, String> {
        Ok(config.clone())
    }
    type Module<B>
        = gemma4::Assistant<B>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    type Unit<B>
        = gemma4::DenseBlock<B>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    type Execution<B, P>
        = gemma4::Assistant<B>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    fn execution_unit_layout(
        _config: &Self::ModuleConfig,
    ) -> Result<Option<eredu_runtime::ExecutionUnitLayout>, eredu_nn::Error> {
        Ok(None)
    }
    fn build_execution_unit<B>(
        config: &Self::ModuleConfig,
        ordinal: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Unit<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        if ordinal >= config.text_config.num_hidden_layers() {
            return Err(eredu_nn::Error::backend("invalid assistant decoder unit"));
        }
        gemma4::DenseBlock::new_at(&config.text_config, ordinal, "model.layers", context)
    }
    fn into_execution<B, P>(
        module: Self::Module<B>,
        _policy: impl FnOnce(Vec<Self::Unit<B>>) -> P,
    ) -> Result<Self::Execution<B, P>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        Ok(module)
    }
    fn parameter_policy<B, P>(_execution: &Self::Execution<B, P>) -> Option<&P>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        None
    }
    type Executor<'a, M>
        = gemma4::speculative::ExternalExecutor<
        'a,
        gemma4::speculative::ArchitectureExternalMechanisms<M>,
    >
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static;

    fn tokenizer_model_type(_config: &Self::Config) -> &'static str {
        "gemma4"
    }

    fn configuration_model_type(config: &Self::Config) -> &str {
        &config.model_type
    }

    fn module<B>(
        config: Self::ModuleConfig,
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

    fn materialization_plan(config: &Self::Config) -> Result<SafetensorsCheckpointPlan, String> {
        gemma4::assistant_safetensors_plan(config)
    }

    fn linear_format(config: &Self::Config, name: &str) -> LinearFormat {
        config
            .quantization
            .map(LinearFormat::from)
            .unwrap_or_else(|| config.text_config.linear_format_for(name))
    }

    fn parameter_owner(
        config: &Self::Config,
        name: &str,
    ) -> Result<ReplicatedTextParameterOwner, String> {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            let (index, suffix) = rest
                .split_once('.')
                .ok_or("invalid assistant block parameter")?;
            let unit = index.parse::<usize>().map_err(|e| e.to_string())?;
            if unit >= config.text_config.num_hidden_layers() || suffix.is_empty() {
                return Err(format!("assistant parameter {name:?} has no decoder block"));
            }
            return Ok(ReplicatedTextParameterOwner::ExecutionUnit {
                group: "assistant.decoder".into(),
                unit,
            });
        }
        Ok(ReplicatedTextParameterOwner::StaticRole(
            "external_assistant".into(),
        ))
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
/// Shared Muse assistant architecture marker preserving distinct checkpoint identities.
pub struct MuseAssistantArchitecture;

impl sealed::Sealed for MuseAssistantArchitecture {}

impl ExternalAssistantArchitecture for MuseAssistantArchitecture {
    fn memory_topology(
        config: &Self::ModuleConfig,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::ConditionedPredictionTopology>,
        eredu_nn::Error,
    > {
        config.execution_definition().memory_topology().map(Some)
    }
    type Config = MuseAssistantConfig;
    type ModuleConfig = MuseAssistantConstruction;
    fn module_config(
        config: &Self::Config,
        target: Option<&ExternalAssistantTargetProfile>,
    ) -> Result<Self::ModuleConfig, String> {
        config.construction(match target {
            Some(ExternalAssistantTargetProfile::Muse(target)) => Some(target),
            _ => None,
        })
    }
    type Module<B>
        = MuseAssistantModule<B>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    type Unit<B>
        = muse::assistant::execution::MuseAssistantUnit<B>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    type Execution<B, P>
        = MuseAssistantModule<B, P>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
    fn execution_unit_layout(
        config: &Self::ModuleConfig,
    ) -> Result<Option<eredu_runtime::ExecutionUnitLayout>, eredu_nn::Error> {
        Ok(Some(config.execution_definition().unit_layout().clone()))
    }
    fn build_execution_unit<B>(
        config: &Self::ModuleConfig,
        ordinal: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Unit<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        config
            .execution_definition()
            .build_unit::<B>(ordinal, context)
    }
    fn into_execution<B, P>(
        module: Self::Module<B>,
        policy: impl FnOnce(Vec<Self::Unit<B>>) -> P,
    ) -> Result<Self::Execution<B, P>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        let (definition, units) = module.into_parts();
        Ok(MuseAssistantModule::new(
            definition,
            policy(
                units
                    .into_units()
                    .map_err(eredu_nn::Error::backend_source)?,
            ),
        ))
    }
    fn parameter_policy<B, P>(execution: &Self::Execution<B, P>) -> Option<&P>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        Some(execution.policy())
    }
    type Executor<'a, M>
        = muse::speculative::ExternalExecutor<
        'a,
        muse::speculative::ArchitectureExternalMechanisms<M>,
    >
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static;

    fn tokenizer_model_type(config: &Self::Config) -> &'static str {
        config.tokenizer_model_type()
    }

    fn tokenizer_source(
        config: &Self::Config,
    ) -> crate::external_draft::ExternalDraftTokenizerSource {
        use crate::external_draft::ExternalDraftTokenizerSource;
        match config {
            MuseAssistantConfig::Spark(_) => ExternalDraftTokenizerSource::Target,
            MuseAssistantConfig::Glimmer(_) => ExternalDraftTokenizerSource::Artifact {
                model_type: config.tokenizer_model_type(),
            },
        }
    }

    fn configuration_model_type(config: &Self::Config) -> &str {
        config.model_type()
    }

    fn module<B>(
        config: Self::ModuleConfig,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Module<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    {
        config.module(context)
    }

    fn quantization(config: &Self::Config) -> Option<eredu_checkpoint::WeightQuantization> {
        match config {
            MuseAssistantConfig::Glimmer(c) => c.quantization,
            MuseAssistantConfig::Spark(_) => None,
        }
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

    fn materialization_plan(config: &Self::Config) -> Result<SafetensorsCheckpointPlan, String> {
        config.safetensors_plan()
    }

    fn linear_format(config: &Self::Config, name: &str) -> LinearFormat {
        config.linear_format(name)
    }

    fn parameter_owner(
        config: &Self::Config,
        name: &str,
    ) -> Result<ReplicatedTextParameterOwner, String> {
        config.parameter_owner(name)
    }

    fn executor<'a, M>(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
    ) -> Self::Executor<'a, M>
    where
        M: ExternalAssistantExecutionMechanisms<Self> + 'static,
    {
        muse::speculative::ExternalExecutor::new(target, assistant, capture)
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
        let mut executor = muse::speculative::ExternalExecutor::<
            muse::speculative::ArchitectureExternalMechanisms<M>,
        >::new(target, assistant, capture);
        visitor.execute(&mut executor)
    }
}

/// Fully inspected, architecture-typed assistant materialization input.
#[derive(Debug, Clone)]
pub(crate) struct PreparedExternalAssistant<A: ExternalAssistantArchitecture> {
    checkpoint: ExternalAssistantCheckpoint,
    config: A::Config,
    _architecture: PhantomData<fn() -> A>,
}

/// Cold-selected physical materialization for one typed external assistant.
#[derive(Clone)]
pub(crate) struct SelectedExternalAssistant<A: ExternalAssistantArchitecture> {
    checkpoint: ExternalAssistantCheckpoint,
    prepared_source: Option<eredu_checkpoint::store::SharedCheckpointSource>,
    source_config: A::Config,
    config: A::Config,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    _architecture: PhantomData<fn() -> A>,
}

/// Exact neutral source role paired with one selected external assistant.
pub struct PreparedExternalAssistantSource<A: ExternalAssistantArchitecture> {
    checkpoint: ExternalAssistantCheckpoint,
    artifact_identity: DeferredArtifactIdentity,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    source_module_config: A::ModuleConfig,
    module_config: A::ModuleConfig,
    config: A::Config,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    parameter_units: Vec<ExternalAssistantParameterUnit>,
    residency: eredu_runtime::LayerWeightResidency,
    execution_layout: Option<eredu_runtime::ExecutionUnitLayout>,
    parameter_layout: Option<eredu_runtime::LocalModelLayout>,
    conversion_retention: Option<eredu_core::residency::ParameterConversionRetentionPolicy>,
}

impl<A: ExternalAssistantArchitecture> PreparedExternalAssistantSource<A> {
    /// Cold immutable-parameter payload for the selected policy. This excludes
    /// conversion retention, loading, operator workspace, state and source caches.
    pub fn parameter_residency_requirement(
        &self,
    ) -> Result<eredu_core::BoundedResidencyRequirement, eredu_runtime::BoundedResidencySizingError>
    {
        let bytes = self
            .parameter_units
            .iter()
            .map(|unit| unit.parameter_bytes())
            .collect::<Vec<_>>();
        let (static_bytes, window_bytes, depth) = match &self.execution_layout {
            Some(layout) => (
                0,
                eredu_runtime::execution_unit_parameter_window_bytes(
                    layout,
                    &bytes,
                    self.residency,
                )?,
                self.residency.device_depth(layout.len()),
            ),
            None => {
                let total = bytes
                    .iter()
                    .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
                    .ok_or(
                        eredu_runtime::BoundedResidencySizingError::ArithmeticOverflow(
                            "resident assistant parameter bytes",
                        ),
                    )?;
                (total, 0, 0)
            }
        };
        Ok(eredu_core::BoundedResidencyRequirement {
            static_bytes,
            window_bytes,
            required_bytes: static_bytes.checked_add(window_bytes).ok_or(
                eredu_runtime::BoundedResidencySizingError::ArithmeticOverflow(
                    "assistant parameter requirement",
                ),
            )?,
            depth,
        })
    }

    /// Requested per-execution retention policy; absence selects the managed default.
    pub const fn parameter_conversion_retention(
        &self,
    ) -> Option<eredu_core::residency::ParameterConversionRetentionPolicy> {
        self.conversion_retention
    }

    /// Retains conversion policy without allocating caches or changing source admission.
    pub fn with_parameter_conversion_retention(
        mut self,
        policy: Option<eredu_core::residency::ParameterConversionRetentionPolicy>,
    ) -> Self {
        self.conversion_retention = policy;
        self
    }

    /// Canonical execution addresses frozen together with exact source owners.
    pub fn execution_unit_layout(&self) -> Option<&eredu_runtime::ExecutionUnitLayout> {
        self.execution_layout.as_ref()
    }

    /// Complete assistant policy retained independently of target residency.
    pub const fn residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.residency
    }

    /// Freezes an explicitly selected parameter policy before native construction.
    pub fn with_residency(
        mut self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<Self, ArtifactError> {
        if self.execution_layout.is_none() && !residency.is_fully_resident() {
            return Err(ArtifactError::UnsupportedResidencyPolicy("selected assistant uses resident module construction; a bounded parameter-unit driver is not implemented".into()));
        }
        self.residency = residency;
        Ok(self)
    }

    /// Exact rank-local parameter selections retained independently of the source catalog.
    pub fn parameter_layout(&self) -> Option<&eredu_runtime::LocalModelLayout> {
        self.parameter_layout.as_ref()
    }

    /// Complete disjoint physical acquisition units, validated before opening payloads.
    pub fn parameter_units(&self) -> &[ExternalAssistantParameterUnit] {
        &self.parameter_units
    }

    /// Exact admitted checkpoint provenance paired with this source.
    pub const fn checkpoint(&self) -> &ExternalAssistantCheckpoint {
        &self.checkpoint
    }

    /// Resolves the content-exact identity only when a caller needs it.
    pub fn artifact_identity(&self) -> Result<ArtifactIdentity, std::sync::Arc<ArtifactError>> {
        self.artifact_identity.resolve()
    }

    /// Selected executable configuration paired with this source.
    pub const fn config(&self) -> &A::Config {
        &self.config
    }

    /// Selected, target-bound neural geometry; no artifact needs reopening.
    pub const fn module_config(&self) -> &A::ModuleConfig {
        &self.module_config
    }

    /// Consumes the exact source, source/selected neural geometry, selected config and tasks.
    pub fn into_parts(
        self,
    ) -> (
        eredu_checkpoint::store::SharedCheckpointSource,
        ExternalAssistantCheckpoint,
        DeferredArtifactIdentity,
        A::ModuleConfig,
        A::Config,
        A::ModuleConfig,
        Vec<ReplicatedTextMaterializationTask>,
    ) {
        (
            self.source,
            self.checkpoint,
            self.artifact_identity,
            self.source_module_config,
            self.config,
            self.module_config,
            self.tasks,
        )
    }
}

impl PreparedExternalAssistantSource<MuseAssistantArchitecture> {
    /// Selects local DSpark construction and binding from this exact already-open source.
    /// This does not admit a distributed speculative session.
    pub fn partition_tensor_parallel(
        mut self,
        rank: ParallelRankTopology,
    ) -> Result<Self, ArtifactError> {
        let (source, selected, layout) = construction::partition_muse_construction(
            &self.source_module_config,
            &self.module_config,
            &self.tasks,
            rank,
        )?;
        let units = parameters::local_parameter_units(&self.tasks, &layout, self.source.as_ref())
            .map_err(invalid_assistant)?;
        self.parameter_units =
            parameters::canonical_unit_order(units, selected.execution_definition().unit_layout())
                .map_err(invalid_assistant)?;
        self.execution_layout = Some(selected.execution_definition().unit_layout().clone());
        self.source_module_config = source;
        self.module_config = selected;
        self.parameter_layout = Some(layout);
        Ok(self)
    }
}

impl<A: ExternalAssistantArchitecture> SelectedExternalAssistant<A> {
    /// Prepares the admitted physical source once before backend materialization.
    fn prepare_source(
        self,
        max_cached_sources: usize,
        source_module_config: A::ModuleConfig,
        module_config: A::ModuleConfig,
        parameter_layout: Option<eredu_runtime::LocalModelLayout>,
    ) -> Result<PreparedExternalAssistantSource<A>, ArtifactError> {
        let Self {
            checkpoint,
            prepared_source,
            source_config: _,
            config,
            tasks,
            _architecture: _,
        } = self;
        let execution_layout =
            A::execution_unit_layout(&module_config).map_err(invalid_assistant)?;
        let retained_checkpoint = checkpoint.clone();
        let artifact_identity = match &checkpoint {
            ExternalAssistantCheckpoint::SafeTensors { shards, .. } => {
                DeferredArtifactIdentity::safetensors(
                    "eredu.external-assistant.safetensors.v1",
                    shards,
                )?
            }
            ExternalAssistantCheckpoint::Gguf { checkpoint, .. } => {
                DeferredArtifactIdentity::gguf("eredu.external-assistant.gguf.v1", checkpoint)?
            }
        };
        let source = match checkpoint {
            ExternalAssistantCheckpoint::SafeTensors {
                source: _,
                shards,
                catalog,
                plan: _,
                resolution,
            } => eredu_core::artifact::open_prepared_safetensors_artifact(
                &catalog,
                shards,
                resolution,
                max_cached_sources,
            )?,
            ExternalAssistantCheckpoint::Gguf {
                checkpoint: _,
                resolution: _,
                tensor_mapping: _,
            } => prepared_source.ok_or_else(|| {
                ArtifactError::InvalidArtifact(
                    "selected GGUF assistant omitted its singular prepared source".into(),
                )
            })?,
        };
        let parameter_units = match &parameter_layout {
            Some(layout) => parameters::local_parameter_units(&tasks, layout, source.as_ref()),
            None => parameters::parameter_units(&tasks),
        }
        .map_err(invalid_assistant)?;
        let parameter_units = if let Some(layout) = &execution_layout {
            parameters::canonical_unit_order(parameter_units, layout).map_err(invalid_assistant)?
        } else {
            parameter_units
        };
        Ok(PreparedExternalAssistantSource {
            source,
            checkpoint: retained_checkpoint,
            artifact_identity,
            source_module_config,
            module_config,
            config,
            tasks,
            parameter_units,
            residency: eredu_runtime::LayerWeightResidency::FullyResident,
            execution_layout,
            parameter_layout,
            conversion_retention: None,
        })
    }

    /// Borrows the exact selected module configuration.
    const fn config(&self) -> &A::Config {
        &self.config
    }

    /// Borrows the admitted checkpoint behind this selection.
    const fn checkpoint(&self) -> &ExternalAssistantCheckpoint {
        &self.checkpoint
    }
}

impl<A: ExternalAssistantArchitecture> PreparedExternalAssistant<A> {
    /// Consumes the plan into its admitted checkpoint and normalized configuration.
    fn into_parts(self) -> (ExternalAssistantCheckpoint, A::Config) {
        (self.checkpoint, self.config)
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
        prepared: PreparedExternalAssistantSource<A>,
    ) -> Result<Self::Output<A>, Self::Error>;
}

enum DispatchedMaterializedExternalAssistant<V: ExternalAssistantPreparationVisitor> {
    Gemma4(V::Output<Gemma4AssistantArchitecture>),
    Muse(V::Output<MuseAssistantArchitecture>),
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

/// Read-only operation on an architecture-dispatched materialized assistant.
pub trait MaterializedExternalAssistantRefVisitor<V: ExternalAssistantPreparationVisitor> {
    /// Result of the observation.
    type Output;

    /// Observes retained resources without obtaining mutation authority.
    fn visit<A: ExternalAssistantArchitecture>(self, assistant: &V::Output<A>) -> Self::Output;
}

impl<V: ExternalAssistantPreparationVisitor> MaterializedExternalAssistant<V> {
    /// Observes the retained typed assistant without changing execution state.
    pub fn visit_ref<W: MaterializedExternalAssistantRefVisitor<V>>(
        &self,
        visitor: W,
    ) -> W::Output {
        match &self.dispatched {
            DispatchedMaterializedExternalAssistant::Gemma4(assistant) => {
                visitor.visit::<Gemma4AssistantArchitecture>(assistant)
            }
            DispatchedMaterializedExternalAssistant::Muse(assistant) => {
                visitor.visit::<MuseAssistantArchitecture>(assistant)
            }
        }
    }

    /// Runs one generic backend visitor against the retained typed assistant.
    pub fn visit<W>(&mut self, visitor: W) -> W::Output
    where
        W: MaterializedExternalAssistantVisitor<V>,
    {
        match &mut self.dispatched {
            DispatchedMaterializedExternalAssistant::Gemma4(assistant) => {
                visitor.visit::<Gemma4AssistantArchitecture>(assistant)
            }
            DispatchedMaterializedExternalAssistant::Muse(assistant) => {
                visitor.visit::<MuseAssistantArchitecture>(assistant)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum DispatchedExternalAssistantPreparation {
    Gemma4(PreparedExternalAssistant<Gemma4AssistantArchitecture>),
    Muse(PreparedExternalAssistant<MuseAssistantArchitecture>),
}

#[derive(Clone)]
enum DispatchedSelectedExternalAssistant {
    Gemma4(SelectedExternalAssistant<Gemma4AssistantArchitecture>),
    Muse(SelectedExternalAssistant<MuseAssistantArchitecture>),
}

/// Opaque architecture-dispatched external assistant preparation.
#[derive(Debug, Clone)]
pub struct ExternalAssistantPreparation {
    dispatched: DispatchedExternalAssistantPreparation,
}

/// Opaque architecture-dispatched materialization selected before native resources exist.
#[derive(Clone)]
pub struct SelectedExternalAssistantPreparation {
    dispatched: DispatchedSelectedExternalAssistant,
}

/// Construction-time neutral identities and invocation bounds for an external assistant.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternalSpeculativeContractRequest {
    topology: ParallelRankTopology,
    processor: SpeculativeIdentity,
    tokenizer_proof: TokenizerCompatibilityProof,
    tokenizer_fingerprint: [u8; 32],
    maximum_draft_tokens: NonZeroUsize,
    maximum_capture_sequence_length: NonZeroUsize,
}

impl ExternalSpeculativeContractRequest {
    /// Creates exact construction-time inputs for architecture-owned external selection.
    /// The capture sequence bound describes one decoder invocation, independently
    /// of cumulative committed context and media-tower sequence geometry.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        topology: ParallelRankTopology,
        processor: SpeculativeIdentity,
        tokenizer_proof: TokenizerCompatibilityProof,
        tokenizer_fingerprint: [u8; 32],
        maximum_draft_tokens: NonZeroUsize,
        maximum_capture_sequence_length: NonZeroUsize,
    ) -> Self {
        Self {
            topology,
            processor,
            tokenizer_proof,
            tokenizer_fingerprint,
            maximum_draft_tokens,
            maximum_capture_sequence_length,
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
    preparation: DispatchedNormalizedExternalAssistant,
    target: ExternalAssistantTargetProfile,
    capture: crate::composite_execution::ExternalPredictionCaptureRequest,
    placement: Option<ExternalAssistantSourcePlacement>,
    target_selection: Option<construction::AssistantTargetSelection>,
}

enum DispatchedNormalizedExternalAssistant {
    Gemma4(NormalizedExternalAssistant<Gemma4AssistantArchitecture>),
    Muse(NormalizedExternalAssistant<MuseAssistantArchitecture>),
}

enum DispatchedPreparedExternalAssistant {
    Gemma4(PreparedExternalAssistantSource<Gemma4AssistantArchitecture>),
    Muse(PreparedExternalAssistantSource<MuseAssistantArchitecture>),
}

/// Compatibility-proven assistant whose exact neutral source is already open.
///
/// Concrete backends receive this value only after all artifact policy and
/// source construction has completed, so native resources cannot precede it.
pub struct PreparedCompatibleExternalAssistant {
    preparation: DispatchedPreparedExternalAssistant,
    capture: crate::composite_execution::ExternalPredictionCaptureRequest,
    placement: Option<ExternalAssistantSourcePlacement>,
}

/// Immutable parameter placement selected against an ordinary target rank.
/// It describes source construction, not admission of a speculative session.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExternalAssistantSourcePlacement {
    /// Complete assistant parameters on each participating target rank.
    Replicated(ParallelRankTopology),
    /// Sharded body projections and complete replicated scoring parameters.
    TensorParallel(ParallelRankTopology),
}

impl CompatibleExternalAssistantPreparation {
    /// Returns the exact architecture-owned target capture request.
    pub const fn capture(&self) -> &crate::composite_execution::ExternalPredictionCaptureRequest {
        &self.capture
    }

    /// Derives exact architecture requirements before any assistant payload is opened.
    /// Selected placement requires successful target invocation admission.
    /// Profile-only singleton contracts describe requirements without a bound target.
    pub fn speculative_contract(
        &self,
        request: ExternalSpeculativeContractRequest,
    ) -> Result<ExternalSpeculativeContract, ArtifactError> {
        external_speculative_contract(self, request)
    }

    /// Opens the exact admitted assistant source before native materialization.
    pub fn prepare_source(
        self,
        max_cached_sources: usize,
    ) -> Result<PreparedCompatibleExternalAssistant, ArtifactError> {
        let preparation = match self.preparation {
            DispatchedNormalizedExternalAssistant::Gemma4(selected) => {
                DispatchedPreparedExternalAssistant::Gemma4(
                    selected.prepare_source(max_cached_sources)?,
                )
            }
            DispatchedNormalizedExternalAssistant::Muse(selected) => {
                DispatchedPreparedExternalAssistant::Muse(
                    selected.prepare_source(max_cached_sources)?,
                )
            }
        };
        Ok(PreparedCompatibleExternalAssistant {
            preparation,
            capture: self.capture,
            placement: self.placement,
        })
    }
}

impl PreparedCompatibleExternalAssistant {
    /// Selected local source policy. Explicit unplaced preparation has no policy.
    pub const fn source_placement(&self) -> Option<ExternalAssistantSourcePlacement> {
        self.placement
    }

    /// Retains exact assistant TP selections without reopening or reselecting its source.
    /// Complete public distributed assistant admission remains a separate contract.
    pub fn partition_tensor_parallel(
        mut self,
        rank: ParallelRankTopology,
    ) -> Result<Self, ArtifactError> {
        if self.placement.is_some() {
            return Err(invalid_assistant(
                "assistant source placement is already selected",
            ));
        }
        self.preparation = match self.preparation {
            DispatchedPreparedExternalAssistant::Muse(source) => {
                DispatchedPreparedExternalAssistant::Muse(source.partition_tensor_parallel(rank)?)
            }
            DispatchedPreparedExternalAssistant::Gemma4(_) => {
                return Err(invalid_assistant(
                    "tensor-parallel assistant source construction is implemented for Muse DSpark",
                ))
            }
        };
        self.placement = Some(ExternalAssistantSourcePlacement::TensorParallel(rank));
        Ok(self)
    }

    /// Exact selected immutable device-parameter payload, excluding other resources.
    pub fn parameter_residency_requirement(
        &self,
    ) -> Result<eredu_core::BoundedResidencyRequirement, eredu_runtime::BoundedResidencySizingError>
    {
        match &self.preparation {
            DispatchedPreparedExternalAssistant::Gemma4(source) => {
                source.parameter_residency_requirement()
            }
            DispatchedPreparedExternalAssistant::Muse(source) => {
                source.parameter_residency_requirement()
            }
        }
    }

    /// Applies the load plan's independent per-execution conversion allowance.
    pub fn with_parameter_conversion_retention(
        mut self,
        policy: Option<eredu_core::residency::ParameterConversionRetentionPolicy>,
    ) -> Self {
        self.preparation = match self.preparation {
            DispatchedPreparedExternalAssistant::Gemma4(source) => {
                DispatchedPreparedExternalAssistant::Gemma4(
                    source.with_parameter_conversion_retention(policy),
                )
            }
            DispatchedPreparedExternalAssistant::Muse(source) => {
                DispatchedPreparedExternalAssistant::Muse(
                    source.with_parameter_conversion_retention(policy),
                )
            }
        };
        self
    }

    /// Assistant parameter policy retained on the prepared source.
    pub fn residency(&self) -> eredu_runtime::LayerWeightResidency {
        match &self.preparation {
            DispatchedPreparedExternalAssistant::Gemma4(source) => source.residency(),
            DispatchedPreparedExternalAssistant::Muse(source) => source.residency(),
        }
    }
    /// Retains one explicit assistant policy on the exact already-prepared source.
    pub fn with_residency(
        mut self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<Self, ArtifactError> {
        self.preparation = match self.preparation {
            DispatchedPreparedExternalAssistant::Gemma4(source) => {
                DispatchedPreparedExternalAssistant::Gemma4(source.with_residency(residency)?)
            }
            DispatchedPreparedExternalAssistant::Muse(source) => {
                DispatchedPreparedExternalAssistant::Muse(source.with_residency(residency)?)
            }
        };
        Ok(self)
    }
    /// Exact architecture-owned target capture request.
    pub const fn capture(&self) -> &crate::composite_execution::ExternalPredictionCaptureRequest {
        &self.capture
    }

    /// Dispatches an already prepared neutral source to native materialization.
    pub fn visit<V: ExternalAssistantPreparationVisitor>(
        self,
        visitor: V,
    ) -> Result<MaterializedExternalAssistant<V>, V::Error> {
        match self.preparation {
            DispatchedPreparedExternalAssistant::Gemma4(prepared) => {
                visitor
                    .visit(prepared)
                    .map(|assistant| MaterializedExternalAssistant {
                        dispatched: DispatchedMaterializedExternalAssistant::Gemma4(assistant),
                    })
            }
            DispatchedPreparedExternalAssistant::Muse(prepared) => {
                visitor
                    .visit(prepared)
                    .map(|assistant| MaterializedExternalAssistant {
                        dispatched: DispatchedMaterializedExternalAssistant::Muse(assistant),
                    })
            }
        }
    }
}

impl ExternalAssistantPreparation {
    /// Returns the exact tokenizer ownership declared by the selected assistant architecture.
    pub fn tokenizer_source(&self) -> crate::external_draft::ExternalDraftTokenizerSource {
        match &self.dispatched {
            DispatchedExternalAssistantPreparation::Gemma4(p) => {
                Gemma4AssistantArchitecture::tokenizer_source(&p.config)
            }
            DispatchedExternalAssistantPreparation::Muse(p) => {
                MuseAssistantArchitecture::tokenizer_source(&p.config)
            }
        }
    }

    /// Architecture identity whose tokenizer contract the assistant shares.
    pub fn tokenizer_model_type(&self) -> &'static str {
        match &self.dispatched {
            DispatchedExternalAssistantPreparation::Gemma4(p) => {
                Gemma4AssistantArchitecture::tokenizer_model_type(&p.config)
            }
            DispatchedExternalAssistantPreparation::Muse(p) => {
                MuseAssistantArchitecture::tokenizer_model_type(&p.config)
            }
        }
    }

    /// Proves exact target compatibility without opening assistant weight payloads.
    pub fn select_materialization<F>(
        self,
        quantization: Option<eredu_core::QuantizationRequest>,
        max_cached_sources: usize,
        lowering: F,
    ) -> Result<SelectedExternalAssistantPreparation, String>
    where
        F: Fn(&WeightLoweringDescriptor, bool) -> Option<WeightLoweringKind>,
    {
        let dispatched = match self.dispatched {
            DispatchedExternalAssistantPreparation::Gemma4(prepared) => {
                DispatchedSelectedExternalAssistant::Gemma4(select_external_materialization(
                    prepared,
                    quantization,
                    max_cached_sources,
                    &lowering,
                )?)
            }
            DispatchedExternalAssistantPreparation::Muse(prepared) => {
                DispatchedSelectedExternalAssistant::Muse(select_external_materialization(
                    prepared,
                    quantization,
                    max_cached_sources,
                    &lowering,
                )?)
            }
        };
        Ok(SelectedExternalAssistantPreparation { dispatched })
    }
}

impl SelectedExternalAssistantPreparation {
    /// Proves exact target compatibility without opening assistant weight payloads.
    pub fn prove_target_compatibility(
        self,
        target: &ExternalAssistantTargetProfile,
    ) -> Result<CompatibleExternalAssistantPreparation, String> {
        let capture = match (&self.dispatched, target) {
            (
                DispatchedSelectedExternalAssistant::Gemma4(prepared),
                ExternalAssistantTargetProfile::Gemma4(target),
            ) => {
                crate::gemma4::model::external_assistant_capture_request(target, prepared.config())
            }
            (
                DispatchedSelectedExternalAssistant::Muse(prepared),
                ExternalAssistantTargetProfile::Muse(target),
            ) => prepared.config().capture_request(target),
            _ => Err("external assistant family does not match the selected target".into()),
        }?;
        let preparation = match self.dispatched {
            DispatchedSelectedExternalAssistant::Gemma4(selected) => {
                DispatchedNormalizedExternalAssistant::Gemma4(NormalizedExternalAssistant::new(
                    selected,
                    Some(target),
                )?)
            }
            DispatchedSelectedExternalAssistant::Muse(selected) => {
                DispatchedNormalizedExternalAssistant::Muse(NormalizedExternalAssistant::new(
                    selected,
                    Some(target),
                )?)
            }
        };
        Ok(CompatibleExternalAssistantPreparation {
            preparation,
            target: target.clone(),
            capture,
            placement: None,
            target_selection: None,
        })
    }
}

enum SelectedCheckpointCatalog<'a> {
    SafeTensors(&'a TensorCatalog),
    Source(Arc<dyn CheckpointSource>),
}

trait SelectedCatalogMetadata {
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError>;
}

impl<T: CheckpointSource + ?Sized> SelectedCatalogMetadata for T {
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        CheckpointSource::source_metadata(self, key)
    }
}

impl SelectedCatalogMetadata for SelectedCheckpointCatalog<'_> {
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        match self {
            Self::SafeTensors(catalog) => {
                crate::configuration::PortableSafetensorsCatalog(catalog).tensor_metadata(key)
            }
            Self::Source(source) => source.source_metadata(key),
        }
    }
}

impl SelectedCheckpointCatalog<'_> {
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        match self {
            Self::SafeTensors(catalog) => {
                let metadata = crate::configuration::PortableSafetensorsCatalog(catalog)
                    .tensor_metadata(key)?;
                Ok(TensorSourceProvenance {
                    catalog_key: key.to_owned(),
                    physical_tensor: key.to_owned(),
                    output: key.to_owned(),
                    backing_shard: metadata.backing_shard,
                    source_encoding: eredu_checkpoint::SourceTensorEncoding::Safetensors(
                        metadata.stored_dtype,
                    ),
                })
            }
            Self::Source(source) => source.source_provenance(key),
        }
    }
}

fn selected_checkpoint_catalog(
    checkpoint: &ExternalAssistantCheckpoint,
    max_cached_sources: usize,
) -> Result<SelectedCheckpointCatalog<'_>, String> {
    match checkpoint {
        ExternalAssistantCheckpoint::SafeTensors {
            catalog,
            plan,
            resolution,
            ..
        } => {
            let actual = resolve_safetensors_plan(
                &crate::configuration::PortableSafetensorsCatalog(catalog),
                plan,
            )
            .map_err(|failure| {
                format!("external assistant checkpoint no longer matches admission: {failure:?}")
            })?;
            if &actual != resolution {
                return Err(
                    "external assistant checkpoint resolution changed after admission".into(),
                );
            }
            Ok(SelectedCheckpointCatalog::SafeTensors(catalog))
        }
        ExternalAssistantCheckpoint::Gguf {
            checkpoint,
            resolution,
            tensor_mapping,
        } => Ok(SelectedCheckpointCatalog::Source(Arc::new(
            eredu_checkpoint::gguf_store::GgufWeightStore::builder()
                .max_cached_readers(max_cached_sources)
                .map_err(|error| error.to_string())?
                .add_resolved_checkpoint(checkpoint.clone(), resolution, tensor_mapping)
                .map_err(|error| error.to_string())?
                .build()
                .map_err(|error| error.to_string())?,
        ))),
    }
}

fn gguf_checkpoint_formats(
    checkpoint: &ExternalAssistantCheckpoint,
) -> Result<HashMap<String, WeightQuantization>, String> {
    let ExternalAssistantCheckpoint::Gguf {
        checkpoint,
        tensor_mapping,
        ..
    } = checkpoint
    else {
        return Ok(HashMap::new());
    };
    let mapped = tensor_mapping
        .iter()
        .map(|mapping| {
            (
                (
                    mapping.physical_name.as_str(),
                    mapping.original_name.as_str(),
                ),
                mapping.layout.name.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut formats = HashMap::new();
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            let descriptor = tensor.descriptor();
            let Some(name) = tensor
                .outputs()
                .first()
                .and_then(|output| mapped.get(&(descriptor.name.as_str(), output.name.as_str())))
            else {
                continue;
            };
            let format = if let Some((bits, group_size)) = tensor.affine() {
                let group_size = i32::try_from(group_size)
                    .map_err(|_| "GGUF affine group size exceeds i32".to_owned())?;
                Some(WeightQuantization::Affine(
                    eredu_checkpoint::AffineQuantization::new(group_size, i32::from(bits))
                        .map_err(|error| error.to_string())?,
                ))
            } else if tensor.is_mxfp4() {
                Some(WeightQuantization::MxFp4)
            } else if !matches!(
                descriptor.ggml_type,
                eredu_gguf::GgmlType::F32
                    | eredu_gguf::GgmlType::F16
                    | eredu_gguf::GgmlType::Bf16
                    | eredu_gguf::GgmlType::I32
            ) {
                Some(WeightQuantization::GgufIQuant {
                    ggml_type: descriptor.ggml_type,
                    endian: shard.endian(),
                })
            } else {
                None
            };
            if let Some(format) = format {
                if formats.insert((*name).to_owned(), format).is_some() {
                    return Err(format!(
                        "external assistant GGUF formats collide at {name:?}"
                    ));
                }
            }
        }
    }
    Ok(formats)
}

fn selected_plan_constraints<'a>(
    plan: &'a SafetensorsCheckpointPlan,
    store: &(impl SelectedCatalogMetadata + ?Sized),
) -> Result<Vec<&'a eredu_checkpoint::schema::SafetensorsTensorConstraint>, String> {
    let mut selected = plan.common_tensors.iter().collect::<Vec<_>>();
    for group in &plan.layout_groups {
        let candidates = group
            .variants
            .iter()
            .filter(|variant| {
                variant.tensors.iter().all(|constraint| {
                    constraint.requirement == TensorRequirement::Optional
                        || constraint.role == TensorRole::Companion
                        || store.source_metadata(&constraint.key).is_ok()
                        || constraint
                            .aliases
                            .iter()
                            .any(|alias| store.source_metadata(alias).is_ok())
                })
            })
            .collect::<Vec<_>>();
        match candidates.as_slice() {
            [variant] => selected.extend(variant.tensors.iter()),
            [] if !group.required => {}
            [] => {
                return Err(format!(
                    "external assistant materialization cannot resolve layout group {:?}",
                    group.id
                ));
            }
            _ => {
                return Err(format!(
                    "external assistant materialization ambiguously resolves layout group {:?}",
                    group.id
                ));
            }
        }
    }
    Ok(selected)
}

fn exact_physical_source(
    store: &SelectedCheckpointCatalog<'_>,
    key: &str,
) -> Result<ReplicatedTextPhysicalSource, String> {
    let provenance = store
        .source_provenance(key)
        .map_err(|error| error.to_string())?;
    let metadata = store
        .source_metadata(key)
        .map_err(|error| error.to_string())?;
    ReplicatedTextPhysicalSource::new(
        provenance.catalog_key,
        provenance.physical_tensor,
        provenance
            .backing_shard
            .ok_or_else(|| format!("external assistant source {key:?} has no backing shard"))?,
        provenance.output,
        provenance.source_encoding,
        metadata.encoded_byte_len,
    )
    .map_err(|error| error.to_string())
}

fn select_external_materialization<A, F>(
    prepared: PreparedExternalAssistant<A>,
    quantization: Option<eredu_core::QuantizationRequest>,
    max_cached_sources: usize,
    lowering: &F,
) -> Result<SelectedExternalAssistant<A>, String>
where
    A: ExternalAssistantArchitecture,
    F: Fn(&WeightLoweringDescriptor, bool) -> Option<WeightLoweringKind>,
{
    let (checkpoint, source_config) = prepared.into_parts();
    let store = selected_checkpoint_catalog(&checkpoint, max_cached_sources)?;
    let prepared_source = match &store {
        SelectedCheckpointCatalog::Source(source) => Some(Arc::clone(source)),
        SelectedCheckpointCatalog::SafeTensors(_) => None,
    };
    let source_config = if matches!(checkpoint, ExternalAssistantCheckpoint::Gguf { .. }) {
        A::with_checkpoint_formats(&source_config, gguf_checkpoint_formats(&checkpoint)?)?
    } else {
        source_config
    };
    let target_config = match quantization {
        Some(eredu_core::QuantizationRequest::Affine { group_size, bits }) => {
            let group_size = i32::try_from(group_size)
                .map_err(|_| "assistant affine group size exceeds i32".to_owned())?;
            A::load_time_quantization(
                &source_config,
                WeightQuantization::Affine(
                    eredu_checkpoint::AffineQuantization::new(group_size, i32::from(bits))
                        .map_err(|error| error.to_string())?,
                ),
            )?
        }
        Some(eredu_core::QuantizationRequest::MxFp4) => {
            A::load_time_quantization(&source_config, WeightQuantization::MxFp4)?
        }
        Some(_) => return Err("unsupported external assistant transform request".into()),
        None => source_config.clone(),
    };
    let target_plan = A::materialization_plan(&target_config)?;
    let constraints = selected_plan_constraints(&target_plan, &store)?;
    let companions = constraints
        .iter()
        .filter_map(|constraint| {
            constraint.linear_companion.as_ref().map(|companion| {
                let role = match companion.kind {
                    eredu_checkpoint::schema::LinearCompanionKind::GlobalScale => {
                        eredu_nn::LinearCompanionRole::GlobalScale
                    }
                    eredu_checkpoint::schema::LinearCompanionKind::Scale => {
                        eredu_nn::LinearCompanionRole::Scale
                    }
                    eredu_checkpoint::schema::LinearCompanionKind::AffineBias => {
                        eredu_nn::LinearCompanionRole::AffineBias
                    }
                };
                (companion.primary.clone(), *constraint, role)
            })
        })
        .fold(
            BTreeMap::<String, Vec<_>>::new(),
            |mut groups, (primary, value, role)| {
                groups.entry(primary).or_default().push((value, role));
                groups
            },
        );
    let mut tasks = Vec::new();
    for constraint in constraints {
        if constraint.linear_companion.is_some() {
            continue;
        }
        let source = match store.source_metadata(&constraint.key) {
            Ok(metadata) => metadata,
            Err(_) if constraint.requirement == TensorRequirement::Optional => continue,
            Err(error) => return Err(error.to_string()),
        };
        let is_matrix = source.logical_shape.len() == 2
            || (constraint.key.ends_with(".weight") && source.logical_shape.len() > 2);
        let packed_output = companions.contains_key(&constraint.key);
        let executable = if quantization.is_some() && !packed_output {
            LinearFormat::Dense
        } else {
            A::linear_format(&target_config, &constraint.key)
        };
        let transforms = quantization.is_some() && packed_output;
        let descriptor = WeightLoweringDescriptor::new(
            store
                .source_provenance(&constraint.key)
                .map_err(|error| error.to_string())?
                .source_encoding,
            executable,
            source.physical_shape.clone(),
            source.logical_shape.clone(),
            (is_matrix && executable != LinearFormat::Dense)
                .then(|| source.logical_shape.len() - 1),
        )
        .map_err(|error| error.to_string())?;
        let kind = lowering(&descriptor, transforms).ok_or_else(|| {
            format!(
                "backend has no exact external assistant lowering for {:?}",
                constraint.key
            )
        })?;
        let owner = A::parameter_owner(&target_config, &constraint.key)?;
        let companion_owner = owner
            .parameter_group_owner()
            .map_err(|error| error.to_string())?;
        let mut task = ReplicatedTextMaterializationTask::from_exact_source(
            constraint.key.clone(),
            exact_physical_source(&store, &constraint.key)?,
            constraint.aliases.clone(),
            source.physical_shape,
            source.logical_shape,
            if is_matrix {
                ReplicatedTextParameterRole::LinearWeight
            } else {
                ReplicatedTextParameterRole::Other
            },
            owner,
            executable,
            kind,
            descriptor,
        )
        .map_err(|error| error.to_string())?;
        let outputs = companions
            .get(&constraint.key)
            .into_iter()
            .flatten()
            .map(|(companion, role)| {
                let mut output = ReplicatedTextOutputCompanion::new(
                    companion.key.clone(),
                    *role,
                    companion.shape.clone(),
                    companion_owner.clone(),
                )
                .map_err(|error| error.to_string())?;
                if store.source_metadata(&companion.key).is_ok() {
                    output =
                        output.with_catalog_source(exact_physical_source(&store, &companion.key)?);
                }
                Ok(output)
            })
            .collect::<Result<Vec<_>, String>>()?;
        task = task
            .with_output_companions(outputs)
            .map_err(|error| error.to_string())?;
        tasks.push(task);
    }
    if tasks.is_empty() {
        return Err("external assistant selection produced no exact materialization tasks".into());
    }
    Ok(SelectedExternalAssistant {
        checkpoint,
        prepared_source,
        source_config,
        config: target_config,
        tasks,
        _architecture: PhantomData,
    })
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
    if let Some(placement) = compatible.placement {
        let retained_rank = match placement {
            ExternalAssistantSourcePlacement::Replicated(rank)
            | ExternalAssistantSourcePlacement::TensorParallel(rank) => rank,
        };
        if retained_rank != request.topology {
            return Err(invalid_assistant(
                "external speculative topology differs from retained source placement",
            ));
        }
    }
    if let Some(selected) = &compatible.target_selection {
        selected.validate_request(&request, &compatible.target)?;
    } else if request.topology.world_size() != 1 {
        return Err(invalid_assistant(
            "partitioned external speculation requires retained target invocation admission",
        ));
    }
    request
        .tokenizer_proof
        .validate_target(request.tokenizer_fingerprint)
        .map_err(invalid_assistant)?;
    // Public speculative lanes are batch-one. Capture cannot exceed either the
    // architecture context or the selected decoder invocation.
    let batch = 1;
    let sequence = compatible
        .target
        .maximum_capture_sequence_length()?
        .get()
        .min(request.maximum_capture_sequence_length.get());
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
        &compatible.preparation,
        &compatible.target,
        &compatible.capture,
    ) {
        (
            DispatchedNormalizedExternalAssistant::Gemma4(prepared),
            ExternalAssistantTargetProfile::Gemma4(target),
            crate::composite_execution::ExternalPredictionCaptureRequest::Gemma4SharedAttention {
                final_hidden_path,
            },
        ) => {
            let assistant = prepared.selected.config();
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
                .selected
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
            DispatchedNormalizedExternalAssistant::Muse(prepared),
            ExternalAssistantTargetProfile::Muse(target),
            crate::composite_execution::ExternalPredictionCaptureRequest::Muse {
                target_layers,
                target_paths,
            },
        ) => {
            let assistant = prepared.selected.config();
            assistant
                .capture_request(target)
                .map_err(invalid_assistant)?;
            let family = assistant.strategy_name();
            let capacity = NonZeroUsize::new(assistant.proposal_capacity()).ok_or_else(|| {
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
                            "{family}.target_layers.{position}.layer.{layer}"
                        ))?,
                    )
                    .and_then(|entry| entry.with_bounded_dimension(1))
                    .map_err(invalid_assistant)
                })
                .collect::<Result<Vec<_>, ArtifactError>>()?;
            let capture = SpeculativeCaptureSchema::new(
                external_contract_identity(format!(
                    "{family}/capture/batch={batch};sequence={sequence};layers={target_layers:?}"
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
                0..usize::try_from(assistant.num_hidden_layers()).map_err(invalid_assistant)?
            {
                state.push(format!("assistant.projected_context.{layer}.keys"));
                state.push(format!("assistant.projected_context.{layer}.values"));
            }
            match assistant {
                MuseAssistantConfig::Spark(c) => match c.released().identity() {
                    muse::assistant::spark::SparkAssistantIdentity::DSpark => state.extend(
                        [
                            "assistant.proposal_hidden",
                            "assistant.markov_logits",
                            "assistant.target_columns",
                            "assistant.confidence",
                        ]
                        .map(str::to_owned),
                    ),
                    muse::assistant::spark::SparkAssistantIdentity::DFlash2 => state.extend(
                        [
                            "assistant.proposal_hidden",
                            "assistant.candidate_ids",
                            "assistant.unary_logits",
                        ]
                        .map(str::to_owned),
                    ),
                },
                MuseAssistantConfig::Glimmer(_) => {}
            }
            let assistant_identity = external_contract_identity(format!(
                "{family}/model={};block={};layers={};hidden={};targets={target_layers:?}",
                assistant.model_type(),
                assistant.block_size(),
                assistant.num_hidden_layers(),
                assistant.hidden_size(),
            ))?;
            let (artifact_identity, format_identity) = prepared
                .selected
                .checkpoint()
                .speculative_identities(&assistant_identity)?;
            (
                family,
                capacity,
                assistant_identity,
                artifact_identity,
                format_identity,
                format!(
                    "block={};sliding={};mask={};layers={target_layers:?}",
                    assistant.block_size(),
                    assistant.sliding_window(),
                    assistant.mask_token_id()
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
                ArtifactFormat::SafeTensors => muse::DFlashConfig::from_hf_json(
                    &serde_json::to_vec(config_json.expect("SafeTensors inspection has JSON"))?,
                )
                .map_err(invalid_assistant)?,
                ArtifactFormat::Gguf => muse::DFlashConfig::from_gguf_metadata(
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
                || muse::dflash_safetensors_plan(&config),
                || muse::dflash_gguf_plan(&config),
                muse::translate_dflash_gguf_weight_name,
            )?;
            Ok(ExternalAssistantPreparation {
                dispatched: DispatchedExternalAssistantPreparation::Muse(
                    PreparedExternalAssistant {
                        checkpoint,
                        config: MuseAssistantConfig::Glimmer(config),
                        _architecture: PhantomData,
                    },
                ),
            })
        }
        "muse_spark_assistant" => {
            let released = muse::assistant::spark::SparkAssistantConfig::from_hf_value(
                config_json.ok_or_else(|| {
                    invalid_assistant(
                        "Spark assistants require their released SafeTensors configuration",
                    )
                })?,
            )
            .map_err(invalid_assistant)?;
            let config = MuseAssistantConfig::Spark(
                muse::assistant::spark::SparkAssistantLoadConfig::new(released),
            );
            let checkpoint = prepared_checkpoint(
                &inspection,
                || config.safetensors_plan(),
                || Err("no released Spark assistant GGUF format is declared".into()),
                str::to_owned,
            )?;
            Ok(ExternalAssistantPreparation {
                dispatched: DispatchedExternalAssistantPreparation::Muse(
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
                shards: inspection
                    .safetensors_shards()
                    .expect("SafeTensors inspection retains admitted shards")
                    .clone(),
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
        let family = if muse::assistant::spark::SparkAssistantConfig::declared_by(json) {
            muse::assistant::spark::SparkAssistantConfig::from_hf_value(json)
                .map_err(invalid_assistant)?;
            "muse_spark_assistant"
        } else {
            match model_type {
                "gemma4_assistant" => {
                    gemma4::AssistantConfig::from_json(&bytes).map_err(invalid_assistant)?;
                    "gemma4_assistant"
                }
                "muse_glimmer_assistant" => {
                    muse::DFlashConfig::from_hf_json(&bytes).map_err(invalid_assistant)?;
                    "muse_glimmer_assistant"
                }
                other => return Err(ArtifactError::UnsupportedModelType(other.into())),
            }
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
                muse::DFlashConfig::from_gguf_metadata(&metadata).map_err(invalid_assistant)?;
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
