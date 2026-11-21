//! Execution types and native mechanisms for conditioned prediction.

use eredu_core::checkpoint_mechanisms::unit_binding::ParameterStorageBinding;
use eredu_core::speculative::assistant::{ExternalAssistantBinder, ExternalAssistantOperators};

use crate::capture::SpeculativeCaptureMechanisms as _;

use super::ExternalAssistantCache;
use eredu_core::residency::parameter_binding::preflight_bindings;
use eredu_core::residency::parameter_binding::WeightBinding;

use eredu_checkpoint::store::CheckpointSource;
use eredu_core::{
    capability::NativeAllocatorObservations, residency::ParameterConversionRetentionEligibility,
    BackendFailure,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    memory_estimation::MemoryBytes,
    memory_forecast::{
        ConditionedPredictionTopology, ParameterLoadingMemoryProfile,
        ParameterResidencyMemoryBounds,
    },
    LayerWeightResidency,
};

/// Execution types supplied by an assistant implementation.
pub trait ExternalAssistantModel: Sized + 'static {
    /// Exact normalized architecture configuration.
    type Config: Clone + std::fmt::Debug;

    /// One true independently constructed parameter unit.
    type Unit<B>: eredu_nn::Parameterized<B::Tensor> + 'static
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Execution container specialized to the selected ordinary parameter policy.
    /// It need not implement parameter traversal or cloning.
    type Execution<B, P>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Borrows the selected policy when the runtime owns independently acquired units.
    fn parameter_policy<B, P>(execution: &Self::Execution<B, P>) -> Option<&P>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
}

/// Unloaded module construction supplied by the portable model implementation.
pub trait ExternalAssistantConstruction: ExternalAssistantModel {
    /// Prepared construction geometry, including any selected local layout.
    type Geometry: Clone;

    /// Unloaded assistant module with exact parameter traversal.
    type Module<B>: Clone + eredu_nn::Parameterized<B::Tensor>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Constructs the unloaded module from prepared geometry.
    fn module<B>(
        geometry: Self::Geometry,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Module<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Constructs exactly one independently bound unit.
    fn build_execution_unit<B>(
        geometry: &Self::Geometry,
        ordinal: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Unit<B>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;

    /// Converts the module to execution form under the selected parameter policy.
    fn into_execution<B, P>(
        module: Self::Module<B>,
        policy: impl FnOnce(Vec<Self::Unit<B>>) -> P,
    ) -> Result<Self::Execution<B, P>, eredu_nn::Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone;
}

/// Exact binding inputs for an already selected assistant. The source is restricted
/// to the admitted parameter tasks and identifies the original encodings.
pub struct PreparedAssistantBinding<A: ExternalAssistantConstruction> {
    /// Restricted checkpoint source for the admitted tasks.
    pub source: eredu_checkpoint::store::SharedCheckpointSource,
    /// Unpartitioned module geometry used for parameter transforms.
    pub source_geometry: A::Geometry,
    /// Selected module geometry for this execution.
    pub geometry: A::Geometry,
    /// Exact normalized assistant configuration.
    pub config: A::Config,
    /// Selected parameter materialization tasks.
    pub tasks: Vec<crate::ReplicatedTextMaterializationTask>,
    /// Exact grouping of tasks into acquisition units.
    pub parameter_units: Vec<crate::ParameterAcquisitionUnit>,
    /// Selected parameter residency policy.
    pub residency: crate::LayerWeightResidency,
    /// Independently acquired execution units, when applicable.
    pub execution_layout: Option<crate::ExecutionUnitLayout>,
    /// Selected parameter transform and sharding layout, when applicable.
    pub parameter_layout: Option<crate::LocalModelLayout>,
    /// Admitted retention policy for optional parameter conversions.
    pub conversion_retention: Option<eredu_core::residency::ParameterConversionRetentionPolicy>,
    /// Portable memory equations for conditioned prediction, when available.
    pub memory_topology: Option<crate::memory_forecast::ConditionedPredictionTopology>,
}

type AssistantUnitPolicy<B, U> =
    <<B as ExternalAssistantBinder>::Storage as crate::ParameterUnitMechanisms<
        <B as ExternalAssistantBinder>::Backend,
    >>::UnitPolicy<U>;

/// Installed assistant configuration, portable executable, and request-independent resources.
pub struct InstalledExternalAssistant<B: ExternalAssistantBinder, A: ExternalAssistantModel> {
    /// Exact normalized assistant configuration.
    pub config: A::Config,
    /// Architecture equations over the selected parameter policy and native operators.
    pub module: A::Execution<
        B::Backend,
        super::AssistantParameterPolicy<AssistantUnitPolicy<B, A::Unit<B::Backend>>>,
    >,
    /// Explicit typed observations from the portable executor.
    pub observers: super::ExternalAssistantObservers<
        <B::Backend as NeuralBackend>::Tensor,
        B::Logits,
        B::ObservationError,
    >,
    memory_topology: Option<ConditionedPredictionTopology>,
    parameter_memory_bounds: Option<ParameterResidencyMemoryBounds>,
    parameter_loading: ParameterLoadingMemoryProfile,
}

impl<B: ExternalAssistantBinder, A: ExternalAssistantModel> InstalledExternalAssistant<B, A> {
    /// Projects retained portable geometry together with native storage observations.
    pub fn memory_profile(
        &self,
        host_execution: bool,
        residency: LayerWeightResidency,
        observe: impl FnOnce() -> Result<
            Option<eredu_core::speculative::SpeculativeParameterMemoryObservation>,
            BackendFailure,
        >,
        native_workspace: MemoryBytes,
    ) -> Result<Option<crate::memory_forecast::ConditionedAssistantMemoryProfile>, BackendFailure>
    {
        let Some(topology) = &self.memory_topology else {
            return Ok(None);
        };
        let Some(parameters) = observe()? else {
            return Ok(None);
        };
        Ok(Some(
            crate::memory_forecast::ConditionedAssistantMemoryProfile {
                topology: topology.clone(),
                host_execution,
                residency,
                parameter_bounds: self.parameter_memory_bounds.clone(),
                parameter_loading: Some(self.parameter_loading.clone()),
                parameters,
                native_workspace,
            },
        ))
    }
}

/// Assistant container selected by portable construction for the native binding.
pub type InstalledAssistantFor<M, A> =
    InstalledExternalAssistant<<M as ExternalAssistantExecutionMechanisms>::Binder, A>;

impl<B: ExternalAssistantBinder, A: ExternalAssistantModel> InstalledExternalAssistant<B, A> {
    fn unit_policy(&self) -> Option<&AssistantUnitPolicy<B, A::Unit<B::Backend>>> {
        A::parameter_policy::<B::Backend, _>(&self.module).and_then(|policy| policy.units())
    }

    /// Reads the selected parameter owner and combines its ledger with native counters.
    pub fn parameter_memory_observation<N: NativeAllocatorObservations>(
        &self,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeParameterMemoryObservation>,
        BackendFailure,
    > {
        let Some(policy) = self.unit_policy() else {
            return Ok(None);
        };
        let report = crate::UnitPolicyResidency::residency(policy)?;
        let parameter_conversions = report.device_parameter_conversions().map(<[_]>::to_vec);
        let parameters =
            crate::text_session_driver::capability::static_memory_from_residency_and_components(
                Some(report),
                std::iter::empty(),
                [],
                N::native_memory_facts(),
            )
            .map_err(BackendFailure::from_error)?;
        Ok(Some(
            eredu_core::speculative::SpeculativeParameterMemoryObservation {
                parameters,
                parameter_conversions,
                available: N::available_memory()?,
                allocator_cache_limit: N::allocator_cache_limit(),
            },
        ))
    }

    /// Reads the ledger without touching caches or submitted work.
    pub fn parameter_conversion_retention(
        &self,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        BackendFailure,
    > {
        match self.unit_policy() {
            Some(policy) => Ok(crate::UnitPolicyResidency::residency(policy)?
                .offload()
                .parameter_conversion_retention()
                .clone()),
            None => Ok(eredu_core::Observed::unsupported(
                "resident module construction has no conversion-retention ledger",
            )),
        }
    }

    /// Releases optional conversions at an already settled boundary.
    pub fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        let policy = self
            .unit_policy()
            .ok_or(eredu_core::residency::ParameterConversionTrimError::Unsupported)?;
        crate::LocalParameterRetention::trim_local_conversions(policy)
    }
}

/// Trims the two admitted participant ledgers at a settled boundary.
pub fn trim_external_assistant_conversions<
    M: ExternalAssistantExecutionMechanisms,
    A: ExternalAssistantModel,
>(
    target: &mut ExternalAssistantTarget<M>,
    assistant: &mut InstalledAssistantFor<M, A>,
) -> Result<
    eredu_core::residency::ExecutionConversionRetentionTrimReport,
    eredu_core::residency::ParameterConversionTrimError,
> {
    Ok(
        eredu_core::residency::ExecutionConversionRetentionTrimReport {
            target: target.trim_parameter_conversions()?,
            external_drafter: Some(assistant.trim_parameter_conversions()?),
        },
    )
}

/// Reads the two admitted participant ledgers without submitting work.
pub fn external_assistant_conversion_retention<
    M: ExternalAssistantExecutionMechanisms,
    A: ExternalAssistantModel,
>(
    target: &ExternalAssistantTarget<M>,
    assistant: &InstalledAssistantFor<M, A>,
) -> Result<Option<eredu_core::residency::ExecutionConversionRetentionReport>, BackendFailure> {
    Ok(Some(
        eredu_core::residency::ExecutionConversionRetentionReport {
            target: target.parameter_conversion_retention()?,
            external_drafter: Some(assistant.parameter_conversion_retention()?),
        },
    ))
}

/// Constructs the assistant executable from its retained source and native operations.
pub fn bind_external_assistant<A, M>(
    prepared: PreparedAssistantBinding<A>,
    mechanisms: M,
) -> Result<InstalledExternalAssistant<M, A>, M::Error>
where
    A: ExternalAssistantConstruction,
    M: ExternalAssistantBinder,
{
    let PreparedAssistantBinding {
        source: mut store,
        source_geometry,
        geometry,
        config,
        tasks,
        parameter_units,
        residency,
        execution_layout,
        parameter_layout,
        conversion_retention,
        memory_topology,
    } = prepared;
    if execution_layout.is_none() && !residency.is_fully_resident() {
        return Err(
            contract("resident module construction received a bounded assistant policy").into(),
        );
    }

    let context = mechanisms.context();
    let transform_groups =
        crate::group_replicated_text_transform_tasks(&tasks).map_err(BackendFailure::from_error)?;
    let mut module =
        A::module::<M::Backend>(geometry.clone(), context).map_err(BackendFailure::from_error)?;
    if !transform_groups.is_empty() {
        let source_module = A::module::<M::Backend>(source_geometry, context)
            .map_err(BackendFailure::from_error)?;
        crate::exact_parameter_materialization::materialize_exact_transforms::<
            M::Backend,
            M::Parameters,
            _,
            A::Module<M::Backend>,
        >(
            &mut store,
            &source_module,
            &module,
            &[],
            &[],
            parameter_layout.as_ref(),
            &tasks,
            &transform_groups,
            context,
        )?;
    }
    let task_refs = tasks.iter().collect::<Vec<_>>();
    let bindings =
        crate::exact_parameter_materialization::exact_bindings::<M::Backend, M::Parameters, _>(
            &module,
            store.as_ref(),
            &task_refs,
            &std::collections::BTreeSet::new(),
            parameter_layout.as_ref(),
        )?;
    preflight_bindings::<M::Backend>(store.as_ref(), &bindings)
        .map_err(BackendFailure::from_error)?;
    let units = crate::partition_parameter_acquisition_bindings(&parameter_units, bindings)
        .map_err(BackendFailure::from_error)?;
    let storage_mechanisms = mechanisms.storage();
    let (memory_bounds, loading) = assistant_parameter_memory::<M>(
        &storage_mechanisms,
        store.as_ref(),
        &units,
        residency,
        execution_layout.is_some(),
    )?;

    let policy = if let Some(layout) = execution_layout {
        let addresses = crate::ExecutionUnitAddressMap::identity(&layout);
        let (storage, _) = crate::prepare_parameter_unit_storage(
            &storage_mechanisms,
            store,
            &mut crate::EmptyParameterModule,
            &addresses,
            residency,
            Vec::new(),
            units,
            |_| false,
            Some((
                conversion_retention,
                ParameterConversionRetentionEligibility::Eligible,
            )),
        )?;
        if residency.is_fully_resident() {
            let units = (0..layout.len())
                .map(|ordinal| {
                    A::build_execution_unit::<M::Backend>(&geometry, ordinal, context)
                        .map_err(BackendFailure::from_error)
                })
                .collect::<Result<Vec<_>, _>>()?;
            use crate::bounded_execution::units::BindResidentPolicy as _;
            Some(
                AssistantUnitPolicy::<M, A::Unit<M::Backend>>::bind_resident(
                    storage, units, context,
                )?,
            )
        } else {
            use crate::bounded_execution::units::BindBoundedPolicy as _;
            Some(AssistantUnitPolicy::<M, A::Unit<M::Backend>>::bind_bounded(
                storage,
            ))
        }
    } else {
        for bindings in &units {
            let values = crate::materialize_checkpoint_bindings::<
                <M::Storage as ParameterStorageBinding<M::Backend>>::Parameters,
                _,
                _,
            >(store.as_ref(), bindings, |owners| {
                crate::materialize_parameter_owners(
                    store.as_ref(),
                    owners.iter().copied(),
                    &mut mechanisms.parameter_reads(),
                )
            })?;
            let unit = crate::MaterializedUnit::<
                <M::Storage as ParameterStorageBinding<M::Backend>>::Parameters,
            >::try_from_named_weights(values)
            .map_err(BackendFailure::from_error)?;
            let selected = bindings
                .iter()
                .map(WeightBinding::name)
                .collect::<std::collections::BTreeSet<_>>();
            crate::bind_materialized_unit_excluding(&mut module, unit, |id| {
                !selected.contains(id.as_str())
            })
            .map_err(BackendFailure::from_error)?;
        }
        None
    };
    let execution = A::into_execution::<M::Backend, _>(module, |_| {
        super::AssistantParameterPolicy { units: policy }
    })
    .map_err(BackendFailure::from_error)?;
    Ok(InstalledExternalAssistant {
        config,
        module: execution,
        observers: Default::default(),
        memory_topology,
        parameter_memory_bounds: memory_bounds,
        parameter_loading: loading,
    })
}

fn contract(error: &str) -> BackendFailure {
    BackendFailure::from_error(eredu_nn::Error::backend(error.to_owned()))
}

/// Combines metadata-only native allocation bounds according to the selected residency.
fn assistant_parameter_memory<M: ExternalAssistantBinder>(
    mechanisms: &M::Storage,
    source: &dyn CheckpointSource,
    units: &[Vec<WeightBinding>],
    residency: LayerWeightResidency,
    independent_units: bool,
) -> Result<
    (
        Option<ParameterResidencyMemoryBounds>,
        ParameterLoadingMemoryProfile,
    ),
    M::Error,
> {
    let source_payload = match source.retained_payload_bytes_upper_bound() {
        Some(0) => MemoryBytes::exact(0),
        Some(bytes) => MemoryBytes::estimated(0, bytes, "retained checkpoint-source payload backing; shared child owners can be conservatively counted more than once"),
        None => MemoryBytes::unknown("retained checkpoint-source payload ownership unavailable"),
    };
    let unknown = if !source.has_direct_encoded_leases() {
        Some("source has no direct encoded-lease workspace contract")
    } else if units.is_empty() {
        Some("parameter loading has no retained atomic-unit bindings")
    } else {
        None
    };
    let mut workspaces = Vec::new();
    let mut capacities = Vec::new();
    if independent_units || unknown.is_none() {
        for bindings in units {
            let device = eredu_core::residency::parameter_binding::physical_binding_bytes(bindings)
                .map_err(BackendFailure::from_error)?;
            let host = mechanisms.unit_host_capacity(bindings)?;
            if independent_units {
                capacities.push((device, host));
            }
            if unknown.is_none() {
                let mut workspace = 0u64;
                for binding in bindings.iter().filter(|binding| !binding.is_alias()) {
                    let bytes = M::recipe_workspace(&binding.recipe_view(), source)?;
                    workspace = workspace
                        .checked_add(bytes)
                        .ok_or_else(|| contract("native loading workspace overflow"))?;
                }
                workspaces.push((device, host, workspace, workspace));
            }
        }
    }
    let memory_bounds = independent_units
        .then(|| {
            ParameterResidencyMemoryBounds::from_unit_capacities(capacities, residency)
                .map_err(BackendFailure::from_error)
        })
        .transpose()?;
    let loading = if let Some(detail) = unknown {
        ParameterLoadingMemoryProfile {
            source_payload,
            host_workspace: MemoryBytes::unknown(detail),
            execution_workspace: MemoryBytes::unknown(detail),
        }
    } else {
        ParameterLoadingMemoryProfile::from_unit_workspaces(workspaces, residency, source_payload)
            .map_err(BackendFailure::from_error)?
    };
    Ok((memory_bounds, loading))
}

/// Observes the retained resources of a materialized assistant.
pub trait BoundAssistantRefVisitor<B: ExternalAssistantBinder> {
    /// Observation result.
    type Output;

    /// Receives a shared borrow of the materialized assistant.
    fn visit<A: ExternalAssistantConstruction>(
        self,
        assistant: &InstalledExternalAssistant<B, A>,
    ) -> Self::Output;
}

/// Mutates native resources of a materialized assistant.
pub trait BoundAssistantVisitor<B: ExternalAssistantBinder> {
    /// Operation result.
    type Output;

    /// Receives the selected assistant under exclusive ownership.
    fn visit<A: ExternalAssistantConstruction>(
        self,
        assistant: &mut InstalledExternalAssistant<B, A>,
    ) -> Self::Output;
}

/// Dispatch authority supplied by the portable implementation for the selected assistant.
pub trait SelectedAssistantExecutor<A: ExternalAssistantModel> {
    /// Runs the continuation with the selected executor.
    fn execute<M, V>(
        self,
        target: &mut ExternalAssistantTarget<M>,
        assistant: &mut InstalledAssistantFor<M, A>,
        capture: crate::composite_execution::ExternalPredictionCaptureRequest,
        visitor: V,
    ) -> V::Output
    where
        M: ExternalAssistantExecutionMechanisms,
        V: ExternalAssistantExecutorVisitor<A, M>;
}

/// Supplies native resources and a continuation for the selected assistant executor.
pub trait SelectedAssistantExecutionVisitor<B: ExternalAssistantBinder> {
    /// Result of one executor invocation.
    type Output;

    /// Receives the retained assistant and the authority selecting its executor.
    fn visit<A, Dispatch>(
        self,
        assistant: &mut InstalledExternalAssistant<B, A>,
        dispatch: Dispatch,
    ) -> Self::Output
    where
        A: ExternalAssistantConstruction,
        Dispatch: SelectedAssistantExecutor<A>;
}

/// Family-blind continuation invoked with an architecture-selected external executor.
pub trait ExternalAssistantExecutorVisitor<
    A: ExternalAssistantModel,
    M: ExternalAssistantExecutionMechanisms,
>
{
    /// Result returned to the backend session.
    type Output;

    /// Runs one concrete architecture executor through a shared runtime scheduler.
    fn execute<'run, E>(self, executor: &'run mut E) -> Self::Output
    where
        Self: 'run,
        E: eredu_core::speculative::SpeculativeCacheFactory<
                crate::SelectedSpeculativeRealization,
                Input = ExternalAssistantPrompt<M>,
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

/// Prompt carried by the portable assistant invocation.
pub type ExternalAssistantPrompt<M> = crate::text_session_driver::PreparedTextPrompt<
    <<M as ExternalAssistantExecutionMechanisms>::TargetTypes as eredu_core::backend::TextExecutableTypes>::InputTensor,
>;

/// Reads the prepared cache identity required by an external assistant.
pub fn external_assistant_input_identity<M: ExternalAssistantExecutionMechanisms>(
    input: &ExternalAssistantPrompt<M>,
) -> Result<&crate::PreparedInputCacheIdentity, M::Error> {
    input.cache_identity().ok_or_else(|| {
        M::error("external speculative input is missing its prepared-input cache identity".into())
    })
}

/// Installed ordinary target borrowed by the portable assistant.
pub type ExternalAssistantTarget<M> =
    dyn crate::executable_binding::ErasedExternalPredictionExecutable<
        <M as ExternalAssistantExecutionMechanisms>::TargetTypes,
    >;

/// Family-neutral mechanisms supplied by a concrete backend.
///
/// Family configuration, capture interpretation, proposal equations, state geometry, and
/// executor choice remain owned by the architecture. A backend supplies this contract over the
/// execution types provided by the model.
///
/// Coordination defaults describe a local session. A distributed binding overrides
/// all three agreement methods with the same retained session authority. Checkpoints
/// copy model state, never that authority or its cumulative transport usage.
pub trait ExternalAssistantExecutionMechanisms: 'static {
    /// Resolves cancellation and completion across the retained target session.
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Self::Capture::coordinate_speculative_step(&context, local)
    }

    /// Agrees host preparation before any participant enters the next phase.
    fn agree_text_preparation(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        Self::Capture::agree_text_preparation(&context, stage, status)
    }

    /// Agrees explicit commands without executing model work or restoring credits.
    fn coordinate_decoding_control(
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        Self::Capture::coordinate_decoding_control(&context, local)
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
    /// Native tensor and parameter binding for the portable assistant owner.
    type Binder: ExternalAssistantBinder<
        Backend = Self::NeuralBackend,
        Logits = Self::Logits,
        ObservationError = Self::Error,
    >;
    /// Attention cache used by sequential assistant layers.
    type AttentionCache: eredu_nn::AttentionCache<Self::Tensor>;
    /// Physical values carried by the installed ordinary target.
    type TargetTypes: eredu_core::backend::TextExecutableTypes<
        Tensor = Self::Tensor,
        PredictionState = Self::NativeCache,
    >;
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
    /// Native transport and collector facts retained by the portable scheduler.
    type Capture: for<'a> crate::capture::SpeculativeCaptureMechanisms<
            Context<'a> = Self::Context<'a>,
            Provider: crate::capture::CaptureBackendProvider<Tensor = Self::Tensor>,
            Error = Self::Error,
        > + 'static;
    /// Exact verification completion.
    type Completion: eredu_core::BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Creates empty native state for the selected ordinary target.
    fn new_target_cache(
        target: &mut ExternalAssistantTarget<Self>,
    ) -> Result<Self::NativeCache, Self::Error> {
        target
            .prepare_external_prediction_target_cache()
            .map_err(Self::target_error)
    }

    /// Translates the original failure from the installed target.
    fn target_error(
        error: <Self::TargetTypes as eredu_core::backend::TextExecutableTypes>::Error,
    ) -> Self::Error;

    /// Materializes an input part over the already produced native tokens.
    fn target_token_part(
        tokens: &Self::Tensor,
    ) -> Result<
        eredu_core::PreparedInputPart<
            <Self::TargetTypes as eredu_core::backend::TextExecutableTypes>::InputTensor,
        >,
        <Self::TargetTypes as eredu_core::backend::TextExecutableTypes>::Error,
    >;

    /// Reads the installed target frontier, native capacity and parameter owners.
    fn target_memory_observation(
        target: &ExternalAssistantTarget<Self>,
        cache: &Self::NativeCache,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeModelMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        target
            .continuation_memory_observation(cache, additional)
            .map(Some)
    }

    /// Exact native memory telemetry, independent of the selected assistant.
    type Memory: NativeAllocatorObservations;

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

    /// Validates the required operations against the retained group admission,
    /// then borrows its communicator from this invocation's target session.
    /// A caller cannot substitute an independently constructed equal-sized group.
    /// Replicated execution requires no tensor communicator.
    fn assistant_parallel_context<'a>(
        context: Self::Context<'a>,
        requirements: &eredu_core::communication::CommunicationGroupRequirements,
    ) -> Result<Option<&'a <Self::NeuralBackend as NeuralBackend>::ParallelContext>, Self::Error>
    {
        let Some(binding) = Self::Capture::partition_binding(&context) else {
            return Ok(None);
        };
        let Some((id, manifest)) = binding
            .assistant_group(requirements)
            .map_err(Self::transport_error)?
        else {
            return Ok(None);
        };
        Self::bound_assistant_group(binding.transport(), id, manifest).map(Some)
    }
    /// Borrows the native group for the exact admitted manifest and group ID.
    fn bound_assistant_group<'a>(
        transport: &'a <Self::Capture as crate::capture::SpeculativeCaptureMechanisms>::Transport,
        group: eredu_core::CollectiveGroupId,
        manifest: &eredu_core::communication::CommunicationManifest,
    ) -> Result<&'a <Self::NeuralBackend as NeuralBackend>::ParallelContext, Self::Error>;
    /// Projects a portable transport failure to the invocation's error type.
    fn transport_error(error: BackendFailure) -> Self::Error;

    /// Maps a portable neural error into the backend failure type.
    fn neural_error(error: eredu_nn::Error) -> Self::Error;
    /// Constructs a stable architecture lifecycle failure.
    fn error(message: String) -> Self::Error;
    /// Returns the exact logical shape of one retained target-capture tensor.
    fn tensor_shape(value: &Self::Tensor) -> Result<Vec<usize>, Self::Error>;
    /// Runs the installed target prefill and returns its architecture-declared capture.
    fn prefill_target<'a>(
        target: &mut ExternalAssistantTarget<Self>,
        request: &crate::composite_execution::ExternalPredictionCaptureRequest,
        input: ExternalAssistantPrompt<Self>,
        cache: &mut Self::NativeCache,
        _context: Self::Context<'a>,
    ) -> Result<
        (
            Self::Tensor,
            crate::composite_execution::ExternalPredictionTargetCapture<Self::Tensor>,
        ),
        Self::Error,
    > {
        use crate::text_session_driver::TextPrefillInput as _;
        let input = Ok(input.borrowed());
        target
            .capture_external_prediction_target(input, crate::ExpertPass::Prefill, request, cache)
            .map_err(Self::target_error)
    }
    /// Invokes portable ordinary-target verification and obtains its selected capture.
    fn verify_target<'a>(
        target: &mut ExternalAssistantTarget<Self>,
        request: &crate::composite_execution::ExternalPredictionCaptureRequest,
        tokens: &Self::Tensor,
        cache: &mut Self::NativeCache,
        _context: Self::Context<'a>,
    ) -> Result<
        (
            Self::Tensor,
            crate::composite_execution::ExternalPredictionTargetCapture<Self::Tensor>,
        ),
        Self::Error,
    > {
        let part;
        let input = match Self::target_token_part(tokens) {
            Ok(value) => {
                part = value;
                Ok(crate::text_session_driver::TextPromptView::new(
                    std::slice::from_ref(&part),
                ))
            }
            Err(error) => Err(error),
        };
        target
            .capture_external_prediction_target(input, crate::ExpertPass::Decode, request, cache)
            .map_err(Self::target_error)
    }
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
        target: &mut ExternalAssistantTarget<Self>,
        operation: crate::composite_execution::ExternalPredictionTargetOperation<'_, Self::Tensor>,
        _context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        target
            .apply_external_prediction_target_operation(operation)
            .map_err(Self::target_error)
    }
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

impl<N> ExternalAssistantExecutionMechanisms for N
where
    N: ExternalAssistantOperators,
    N::Binder: ExternalAssistantBinder<
        Backend = N::NeuralBackend,
        Logits = N::Logits,
        ObservationError = N::Error,
    >,
    N::Capture: for<'a> crate::capture::SpeculativeCaptureMechanisms<
        Context<'a> = crate::capture::SpeculativeContextFor<'a, N::Capture>,
        Queue = <N::Tensor as Tensor>::Context,
        Provider: crate::capture::CaptureBackendProvider<Tensor = N::Tensor>,
        Transport = <N::Capture as eredu_core::capture::SpeculativeCapturePrimitives>::Transport,
        Error = N::Error,
    >,
{
    type NeuralBackend = N::NeuralBackend;
    type Binder = N::Binder;
    type AttentionCache = crate::cache::ResidentKeyValueCache<N::Cache>;
    type TargetTypes = crate::executable_binding::PortableTextTypes<N::Text>;
    type NativeCache =
        <Self::TargetTypes as eredu_core::backend::TextExecutableTypes>::PredictionState;
    type NativeCacheCheckpoint = Self::NativeCache;
    type Tensor = N::Tensor;
    type Logits = N::Logits;
    type Context<'a> = crate::capture::SpeculativeContextFor<'a, N::Capture>;
    type Capture = N::Capture;
    type Completion = N::Completion;
    type Telemetry = N::Telemetry;
    type Error = N::Error;
    type Memory = N::Memory;

    fn control_logits_estimate(
        logits: &N::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        N::control_logits_estimate(logits)
    }

    fn target_error(error: <N::Text as eredu_core::backend::NativeTextValues>::Error) -> N::Error {
        N::failure(error)
    }

    fn target_token_part(
        tokens: &N::Tensor,
    ) -> Result<
        eredu_core::PreparedInputPart<
            <N::Text as eredu_core::backend::NativeTextValues>::InputTensor,
        >,
        <N::Text as eredu_core::backend::NativeTextValues>::Error,
    > {
        eredu_core::PreparedInputPart::new_with_extents(
            eredu_core::InputModality::Text,
            eredu_core::PreparedInputPayload::TokenIds(N::target_input(tokens)),
            [],
            [],
        )
        .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn tensor_backing_bytes(values: &[&N::Tensor]) -> Result<Option<u64>, BackendFailure> {
        N::tensor_backing_bytes(values)
    }

    fn control_cache_estimate(
        cache: &Self::NativeCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        cache.control_estimate()
    }

    fn control_tensor_bytes(tensor: &N::Tensor) -> Option<u64> {
        N::control_tensor_bytes(tensor)
    }

    fn control_checkpoint<'a>(
        cache: &Self::NativeCache,
        context: Self::Context<'a>,
    ) -> Result<Self::NativeCacheCheckpoint, N::Error> {
        cache.control_copy(context.target()).map_err(N::failure)
    }

    fn control_restore<'a>(
        cache: &mut Self::NativeCache,
        saved: &Self::NativeCacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), N::Error> {
        let replacement = saved.control_copy(context.target()).map_err(N::failure)?;
        *cache = replacement;
        Ok(())
    }

    fn control_copy_tensor<'a>(
        tensor: &N::Tensor,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<N::Tensor, N::Error> {
        N::copy_tensor(tensor, Self::neural_context(context, placement))
    }

    fn bound_assistant_group<'a>(
        transport: &'a <N::Capture as crate::capture::SpeculativeCaptureMechanisms>::Transport,
        group: eredu_core::CollectiveGroupId,
        manifest: &eredu_core::communication::CommunicationManifest,
    ) -> Result<&'a <N::NeuralBackend as NeuralBackend>::ParallelContext, N::Error> {
        N::bound_assistant_group(transport, group, manifest)
    }

    fn transport_error(error: BackendFailure) -> N::Error {
        N::failure(error)
    }

    fn neural_error(error: eredu_nn::Error) -> N::Error {
        N::invalid(error.to_string())
    }

    fn error(message: String) -> N::Error {
        N::invalid(message)
    }

    fn tensor_shape(value: &N::Tensor) -> Result<Vec<usize>, N::Error> {
        N::tensor_shape(value)
    }

    fn checkpoint_native(
        cache: &Self::NativeCache,
    ) -> Result<Self::NativeCacheCheckpoint, N::Error> {
        cache.deep_clone().map_err(N::failure)
    }

    fn restore_checkpoint_native<'a>(
        cache: &mut Self::NativeCache,
        checkpoint: &Self::NativeCacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), N::Error> {
        cache
            .restore(checkpoint, context.target())
            .map_err(N::failure)
    }

    fn native_cache_len(cache: &Self::NativeCache) -> Result<i32, N::Error> {
        i32::try_from(
            cache
                .generation()
                .map_err(|error| N::invalid(error.to_string()))?,
        )
        .map_err(|_| N::invalid("external target cache frontier exceeds i32".into()))
    }

    fn sequence_len(value: &N::Tensor) -> Result<usize, N::Error> {
        N::sequence_len(value)
    }

    fn sequence_row<'a>(
        value: &N::Tensor,
        row: usize,
        retain_dimension: bool,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<N::Tensor, N::Error> {
        N::sequence_row(
            value,
            row,
            retain_dimension,
            Self::neural_context(context, placement),
        )
    }

    fn into_logits(value: N::Tensor) -> N::Logits {
        N::into_logits(value)
    }

    fn sequence_suffix<'a>(
        value: &N::Tensor,
        maximum: i32,
        context: Self::Context<'a>,
    ) -> Result<N::Tensor, N::Error> {
        N::sequence_suffix(value, maximum, context.target())
    }

    fn shared_prefix<'a>(
        value: &N::Tensor,
        cache_len: i32,
        context: Self::Context<'a>,
    ) -> Result<N::Tensor, N::Error> {
        N::shared_prefix(value, cache_len, context.target())
    }

    fn token_prefix<'a>(
        value: &N::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<N::Tensor, N::Error> {
        N::token_prefix(value, end, context.target())
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<N::Tensor, N::Error> {
        N::target_tokens(tokens, context.crosses_devices().then(|| context.target()))
    }

    fn transfer<'a>(
        value: &N::Tensor,
        direction: ExternalAssistantTransfer,
        context: Self::Context<'a>,
    ) -> Result<N::Tensor, N::Error> {
        if !context.is_split() {
            return Ok(value.clone());
        }
        let destination = match direction {
            ExternalAssistantTransfer::TargetToDraft => context.draft(),
            ExternalAssistantTransfer::DraftToTarget => context.target(),
        };
        if context.crosses_devices() {
            return N::copy_tensor(value, destination);
        }
        let root = N::tensor_root(value);
        let _handoff = match direction {
            ExternalAssistantTransfer::TargetToDraft => context.wait_for_target_outputs([root])?,
            ExternalAssistantTransfer::DraftToTarget => context.wait_for_draft_outputs([root])?,
        };
        Ok(value.clone())
    }

    fn neural_context<'a>(
        context: Self::Context<'a>,
        placement: ExternalAssistantTensorPlacement,
    ) -> &'a <N::Tensor as Tensor>::Context {
        match placement {
            ExternalAssistantTensorPlacement::Target => context.target(),
            ExternalAssistantTensorPlacement::Draft => context.draft(),
        }
    }

    fn submit_completion<'a>(
        values: impl IntoIterator<Item = &'a N::Tensor>,
    ) -> Result<N::Completion, N::Error>
    where
        N::Tensor: 'a,
    {
        N::submit_completion(values)
    }
}
