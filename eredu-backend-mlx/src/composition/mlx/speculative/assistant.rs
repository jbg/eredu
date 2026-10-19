use super::parameter_policy::MlxAssistantParameterPolicy;
use super::*;
use crate::backend::ordinary_retirement::{self, OrdinaryRetirement};
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use eredu_runtime::WeightBinding;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// Reports only generic mechanisms available to neutral speculative selection.
///
/// Architecture identity, proposal mode, capture paths, and assistant family
/// never enter this report.
pub(crate) fn speculative_mechanism_capabilities() -> SpeculativeMechanismCapabilities {
    SpeculativeMechanismCapabilities::new([
        SpeculativeMechanism::TensorOperations,
        SpeculativeMechanism::NeuralOperations,
        SpeculativeMechanism::GroupedNeuralOperations,
        SpeculativeMechanism::HyperNeuralOperations,
        SpeculativeMechanism::PayloadMaterialization,
        SpeculativeMechanism::LogitsProcessing,
        SpeculativeMechanism::Sampling,
        SpeculativeMechanism::Randomness,
        SpeculativeMechanism::StateStorage,
        SpeculativeMechanism::StorageResidency,
        SpeculativeMechanism::ExactCompletion,
        SpeculativeMechanism::Observation,
        SpeculativeMechanism::Timing,
        SpeculativeMechanism::QueueBinding,
        SpeculativeMechanism::Communication,
        SpeculativeMechanism::Agreement,
        SpeculativeMechanism::Publication,
        SpeculativeMechanism::SameDeviceHandoff,
        SpeculativeMechanism::CrossDeviceTransfer,
    ])
}

pub(crate) struct MlxExternalAssistant<A: eredu_architectures::ExternalAssistantArchitecture> {
    memory_topology: Option<eredu_runtime::memory_forecast::ConditionedPredictionTopology>,
    parameter_memory_bounds: Option<eredu_runtime::memory_forecast::ParameterResidencyMemoryBounds>,
    parameter_loading: eredu_runtime::memory_forecast::ParameterLoadingMemoryProfile,
    pub(crate) config: A::Config,
    pub(crate) module: MlxModule<
        A::Execution<MlxNeuralBackend, MlxAssistantParameterPolicy<A::Unit<MlxNeuralBackend>>>,
    >,
    pub(crate) observers: eredu_architectures::external_assistant::ExternalAssistantObservers<
        MlxTensor,
        Array,
        Exception,
    >,
}

pub(crate) struct MlxAssistantPreparationVisitor {
    pub(super) stream: Stream,
    pub(super) weights_stream: Stream,
}

impl<A: eredu_architectures::ExternalAssistantArchitecture> MlxExternalAssistant<A> {
    pub(super) fn memory_profile(
        &self,
        host_execution: bool,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::ConditionedAssistantMemoryProfile>,
        eredu_runtime::memory_forecast::GenerationForecastError,
    > {
        let Some(topology) = &self.memory_topology else {
            return Ok(None);
        };
        let Some(parameters) = self.parameter_memory_observation()? else {
            return Ok(None);
        };
        Ok(Some(eredu_runtime::memory_forecast::ConditionedAssistantMemoryProfile {
            topology: topology.clone(), host_execution, residency,
            parameter_bounds: self.parameter_memory_bounds.clone(),
            parameter_loading: Some(self.parameter_loading.clone()),
            parameters,
            native_workspace: eredu_runtime::memory_estimation::MemoryBytes::unknown("MLX conditioned projection/dequantization and operator scratch is not yet projected"),
        }))
    }

    pub(super) fn parameter_memory_observation(
        &self,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeParameterMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        let Some(policy) = self.parameter_policy() else {
            return Ok(None);
        };
        let report = policy
            .residency_report()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let parameter_conversions = report.device_parameter_conversions().map(<[_]>::to_vec);
        Ok(Some(
            eredu_core::speculative::SpeculativeParameterMemoryObservation {
                parameters: crate::composition::mlx::capability::static_memory_from_residency(
                    Some(report),
                    None,
                )
                .map_err(eredu_core::BackendFailure::from_error)?,
                parameter_conversions,
                available: crate::composition::mlx::capability::available_memory()
                    .map_err(eredu_core::BackendFailure::from_error)?,
                allocator_cache_limit: match crate::allocator_cache_policy() {
                    Ok(policy) => eredu_core::Observed::exact(
                        policy.limit_bytes,
                        "MLX process allocator-cache policy",
                    ),
                    Err(error) => eredu_core::Observed::unavailable(error.to_string()),
                },
            },
        ))
    }

    fn parameter_policy(
        &self,
    ) -> Option<
        &crate::backend::runtime::execution::generic::MlxSelectedLayerwisePolicy<
            A::Unit<MlxNeuralBackend>,
            (),
        >,
    > {
        A::parameter_policy::<MlxNeuralBackend, _>(&self.module.inner)
            .and_then(|policy| policy.0.as_ref())
    }

    pub(super) fn parameter_conversion_retention(
        &self,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    > {
        match self.parameter_policy() {
            Some(policy) => Ok(policy
                .residency_report()
                .map_err(eredu_core::BackendFailure::from_error)?
                .offload()
                .parameter_conversion_retention()
                .clone()),
            None => Ok(eredu_core::Observed::unsupported(
                "resident module construction has no conversion-retention ledger",
            )),
        }
    }

    pub(super) fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.parameter_policy()
            .ok_or(eredu_core::residency::ParameterConversionTrimError::Unsupported)?
            .trim_parameter_conversions()
    }
}

struct ObserveAssistantConversions;
struct ObserveAssistantMemory {
    host_execution: bool,
    residency: eredu_runtime::LayerWeightResidency,
}
impl eredu_architectures::MaterializedExternalAssistantRefVisitor<MlxAssistantPreparationVisitor>
    for ObserveAssistantMemory
{
    type Output = Result<
        Option<eredu_runtime::memory_forecast::ConditionedAssistantMemoryProfile>,
        eredu_runtime::memory_forecast::GenerationForecastError,
    >;
    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        assistant: &MlxExternalAssistant<A>,
    ) -> Self::Output {
        assistant.memory_profile(self.host_execution, self.residency)
    }
}

impl eredu_architectures::MaterializedExternalAssistantRefVisitor<MlxAssistantPreparationVisitor>
    for ObserveAssistantConversions
{
    type Output = Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    >;
    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        assistant: &MlxExternalAssistant<A>,
    ) -> Self::Output {
        assistant.parameter_conversion_retention()
    }
}

struct TrimAssistantConversions;
impl eredu_architectures::MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor>
    for TrimAssistantConversions
{
    type Output = Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    >;
    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        assistant: &mut MlxExternalAssistant<A>,
    ) -> Self::Output {
        assistant.trim_parameter_conversions()
    }
}

impl eredu_architectures::ExternalAssistantPreparationVisitor for MlxAssistantPreparationVisitor {
    type Output<A: eredu_architectures::ExternalAssistantArchitecture> = MlxExternalAssistant<A>;
    type Error = Error;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    ) -> Result<Self::Output<A>, Self::Error> {
        materialize_external_assistant::<A>(prepared, &self.stream, &self.weights_stream)
    }
}

pub(super) struct PreparedAssistantParameters<A: eredu_architectures::ExternalAssistantArchitecture>
{
    store: eredu_checkpoint::store::SharedCheckpointSource,
    config: A::Config,
    module_config: A::ModuleConfig,
    pub(super) module: MlxModule<A::Module<MlxNeuralBackend>>,
    units: Vec<Vec<WeightBinding>>,
    layout: Option<eredu_runtime::ExecutionUnitLayout>,
    residency: eredu_runtime::LayerWeightResidency,
    conversion_retention: Option<eredu_core::residency::ParameterConversionRetentionPolicy>,
}

pub(super) fn prepare_external_assistant_parameters<
    A: eredu_architectures::ExternalAssistantArchitecture,
>(
    prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    stream: &Stream,
) -> Result<PreparedAssistantParameters<A>, Error> {
    use crate::backend::runtime::{
        checkpoint::binding::{build_mlx_exact_replicated_text_bindings, ModuleBindingError},
        execution::layerwise::quantize_exact_replicated_text_tasks,
    };
    let parameter_units = prepared.parameter_units().to_vec();
    let parameter_layout = prepared.parameter_layout().cloned();
    let residency = prepared.residency();
    let conversion_retention = prepared.parameter_conversion_retention();
    let layout = prepared.execution_unit_layout().cloned();
    let (
        store,
        _checkpoint,
        _artifact_identity,
        source_module_config,
        config,
        module_config,
        tasks,
    ) = prepared.into_parts();
    let mut store = store;
    let transformed = tasks
        .iter()
        .filter(|task| {
            matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            )
        })
        .collect::<Vec<_>>();
    if !transformed.is_empty() {
        let source = A::module::<MlxNeuralBackend>(source_module_config, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let target = A::module::<MlxNeuralBackend>(module_config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut groups = Vec::<(eredu_checkpoint::WeightQuantization, Vec<_>)>::new();
        for task in transformed {
            let format = task.executable().weight_quantization().ok_or_else(|| {
                Error::Quantization(format!(
                    "selected external assistant transform {:?} has no packed format",
                    task.name()
                ))
            })?;
            if let Some((_, grouped)) = groups.iter_mut().find(|(selected, _)| *selected == format)
            {
                grouped.push(task);
            } else {
                groups.push((format, vec![task]));
            }
        }
        for (format, grouped) in groups {
            store = quantize_exact_replicated_text_tasks(
                store,
                &source,
                &target,
                &[] as &[A::Module<MlxNeuralBackend>],
                &[],
                parameter_layout.as_ref(),
                format,
                &grouped,
                stream,
            )?
            .0;
        }
    }
    let module = MlxModule::new(
        A::module::<MlxNeuralBackend>(module_config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
    );
    let task_refs = tasks.iter().collect::<Vec<_>>();
    let bindings = build_mlx_exact_replicated_text_bindings(
        &module,
        store.as_ref(),
        &task_refs,
        &std::collections::BTreeSet::new(),
        parameter_layout.as_ref(),
    )?;
    eredu_runtime::preflight_bindings::<MlxNeuralBackend>(store.as_ref(), &bindings)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
    // Validate the complete destination partition before any payload is read.
    // Architecture-declared units include every packed companion; native code
    // neither parses checkpoint names nor chooses an assistant's boundaries.
    let mut units = vec![Vec::new(); parameter_units.len()];
    let mut assigned = std::collections::BTreeSet::new();
    for binding in bindings {
        let name = binding.logical_target().unwrap_or_else(|| binding.name());
        let index = parameter_units
            .iter()
            .position(|unit| unit.outputs().contains(name))
            .ok_or_else(|| {
                Error::ArchitectureModel(format!(
                    "assistant binding {name:?} has no acquisition unit"
                ))
            })?;
        if !assigned.insert(name.to_owned()) {
            return Err(Error::ArchitectureModel(format!(
                "assistant binding {name:?} has duplicate destinations"
            )));
        }
        units[index].push(binding);
    }
    if parameter_units
        .iter()
        .any(|unit| !unit.outputs().is_subset(&assigned))
    {
        return Err(Error::ArchitectureModel(
            "assistant acquisition units have missing destinations".into(),
        ));
    }
    for (definition, bindings) in parameter_units.iter().zip(&units) {
        let actual = crate::backend::runtime::checkpoint::binding::binding_bytes(bindings)?;
        if actual != definition.parameter_bytes() {
            return Err(Error::ArchitectureModel(
                "assistant native parameter extent differs from the retained cold unit".into(),
            ));
        }
    }
    Ok(PreparedAssistantParameters {
        store,
        config,
        module_config,
        module,
        units,
        layout,
        residency,
        conversion_retention,
    })
}

pub(super) fn bind_resident_assistant_parameters<
    A: eredu_architectures::ExternalAssistantArchitecture,
>(
    prepared: &mut PreparedAssistantParameters<A>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    use crate::backend::runtime::checkpoint::binding::{
        materialize_module_bindings, populate_module_from_arrays_excluding,
    };
    for bindings in &prepared.units {
        let arrays =
            materialize_module_bindings(prepared.store.as_ref(), bindings, weights_stream, stream)?;
        populate_module_from_arrays_excluding(&mut prepared.module, &arrays, |name| {
            !arrays.contains_key(name)
        })?;
    }
    Ok(())
}

fn materialize_external_assistant<A: eredu_architectures::ExternalAssistantArchitecture>(
    prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxExternalAssistant<A>, Error> {
    let mut prepared = prepare_external_assistant_parameters::<A>(prepared, stream)?;
    let memory_topology = A::memory_topology(&prepared.module_config)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    // Query exact binding extents and the native host-buffer capacity mechanism
    // before units are consumed. No artifact reopening or payload materialization.
    let parameter_memory_bounds = if prepared.layout.is_some() {
        use crate::backend::runtime::{
            checkpoint::binding::binding_bytes,
            residency::manager::host_capacity_upper_bound_for_bindings,
        };
        let capacities = prepared
            .units
            .iter()
            .map(|bindings| {
                Ok((
                    binding_bytes(bindings)?,
                    host_capacity_upper_bound_for_bindings(bindings)?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Some(
            eredu_runtime::memory_forecast::ParameterResidencyMemoryBounds::from_unit_capacities(
                capacities,
                prepared.residency,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        )
    } else {
        None
    };
    let parameter_loading = crate::backend::runtime::residency::manager::parameter_loading_memory_profile(
        prepared.store.as_ref(), &prepared.units, prepared.residency,
    )?;
    let policy = if let Some(layout) = prepared.layout.take() {
        use crate::backend::runtime::execution::generic::MlxSelectedLayerwisePolicy;
        let addresses = (0..layout.len())
            .map(|ordinal| layout.address(ordinal).expect("validated assistant layout"))
            .collect::<Vec<_>>();
        let bindings = std::mem::take(&mut prepared.units);
        let (policy, _) =
            crate::backend::runtime::execution::generic::prepare_parameter_unit_policy::<
                A::Unit<MlxNeuralBackend>,
                _,
                _,
                _,
            >(
                prepared.store,
                &mut Vec::<A::Unit<MlxNeuralBackend>>::new(),
                (),
                prepared.residency,
                stream,
                weights_stream,
                |_| false,
                layout.clone(),
                Vec::new(),
                bindings,
                Vec::new(),
            )?;
        policy
            .residency_manager()
            .configure_parameter_conversion_retention(
                prepared.conversion_retention,
                eredu_core::residency::ParameterConversionRetentionEligibility::Eligible,
            )?;
        let policy = if prepared.residency.is_fully_resident() {
            let units = (0..layout.len())
                .map(|ordinal| {
                    A::build_execution_unit::<MlxNeuralBackend>(
                        &prepared.module_config,
                        ordinal,
                        stream,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            MlxSelectedLayerwisePolicy::resident(
                policy.into_resident_units(units, stream)?,
                &layout,
                &addresses,
            )?
        } else {
            MlxSelectedLayerwisePolicy::bounded(policy, &layout, &addresses)?
        };
        MlxAssistantParameterPolicy(Some(policy))
    } else {
        if !prepared.residency.is_fully_resident() {
            return Err(Error::ArchitectureModel(
                "resident module construction received a bounded assistant policy".into(),
            ));
        }
        bind_resident_assistant_parameters(&mut prepared, stream, weights_stream)?;
        MlxAssistantParameterPolicy(None)
    };
    let module = A::into_execution::<MlxNeuralBackend, _>(prepared.module.inner, |_| policy)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok(MlxExternalAssistant {
        memory_topology,
        parameter_memory_bounds,
        parameter_loading,
        config: prepared.config,
        module: MlxModule::new(module),
        observers: Default::default(),
    })
}

/// Architecture-dispatched MLX draft model with its fixed execution placement.
pub struct MlxDrafter {
    payload: Rc<OrdinaryRetirement<DrafterPayload>>,
    poisoned: Rc<Cell<bool>>,
}

struct DrafterPayload {
    execution: DrafterExecution,
    stream: Stream,
    target_session: String,
}

enum DrafterExecution {
    Assistant(
        eredu_architectures::MaterializedExternalAssistantExecution<MlxAssistantPreparationVisitor>,
    ),
    Autoregressive {
        model: crate::backend::MlxModel,
        selected: SelectedSpeculativeRealization,
        tokenizer: TokenizerCompatibilityProof,
    },
}
impl DrafterExecution {
    fn selected(&self) -> &SelectedSpeculativeRealization {
        match self {
            Self::Assistant(a) => a.selected(),
            Self::Autoregressive { selected, .. } => selected,
        }
    }
    fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        match self {
            Self::Assistant(a) => a.tokenizer_compatibility(),
            Self::Autoregressive { tokenizer, .. } => *tokenizer,
        }
    }
    fn visit<
        W: eredu_architectures::MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor>,
    >(
        &mut self,
        visitor: W,
    ) -> W::Output {
        match self {
            Self::Assistant(a) => a.visit(visitor),
            Self::Autoregressive { .. } => {
                unreachable!("independent draft uses its selected ordinary executor")
            }
        }
    }
    fn capture(
        &self,
    ) -> &eredu_architectures::composite_execution::ExternalPredictionCaptureRequest {
        match self {
            Self::Assistant(a) => a.capture(),
            Self::Autoregressive { .. } => {
                unreachable!("independent draft consumes no target features")
            }
        }
    }
}

struct DrafterRetention {
    _payload: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload>>>>>,
    poisoned: Rc<Cell<bool>>,
}

impl Retention for DrafterRetention {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.poisoned.set(true);
        }
    }
}

pub(in crate::composition::mlx) struct DrafterOperation<'a> {
    drafter: &'a mut MlxDrafter,
    retained: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload>>>>>,
    recovery: Recovery<DrafterRetention>,
    completed: bool,
}

impl DrafterOperation<'_> {
    pub(in crate::composition::mlx) fn visit<W, T>(mut self, visitor: W) -> Result<T, Error>
    where
        W: eredu_architectures::MaterializedExternalAssistantVisitor<
            MlxAssistantPreparationVisitor,
            Output = Result<T, Error>,
        >,
    {
        let result = self.payload().execution.visit(visitor);
        self.finish(result)
    }

    pub(in crate::composition::mlx) fn with_autoregressive<T>(
        mut self,
        run: impl FnOnce(&mut crate::backend::MlxModel) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let result = match &mut self.payload().execution {
            DrafterExecution::Autoregressive { model, .. } => run(model),
            _ => Err(Error::Speculative(
                "draft requires target-feature execution".into(),
            )),
        };
        self.finish(result)
    }

    fn payload(&mut self) -> &mut DrafterPayload {
        Rc::get_mut(&mut self.drafter.payload).expect("healthy drafter owns its native payload")
    }

    fn finish<T>(mut self, result: Result<T, Error>) -> Result<T, Error> {
        self.retained
            .replace(Some(Rc::clone(&self.drafter.payload)));
        self.recovery.seal();
        // One nonblocking poll may leave healthy CPU work or native recovery
        // records pending. A successful synchronous operation must wait for
        // completion; failures retain their nonblocking recovery path.
        let status = if result.is_ok() {
            self.recovery.wait()
        } else {
            self.recovery.progress()
        };
        if !status.settled || status.failed || status.blocked {
            self.drafter.poisoned.set(true);
            let message = match &result {
                Ok(_) => "native drafter work failed or remains unresolved".to_owned(),
                Err(error) => format!("native drafter work failed or remains unresolved: {error}"),
            };
            return Err(Error::Speculative(message));
        }
        let value = result?;
        self.completed = true;
        self.retained.take();
        Ok(value)
    }
}

impl Drop for DrafterOperation<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        self.drafter.poisoned.set(true);
        self.retained
            .replace(Some(Rc::clone(&self.drafter.payload)));
        self.recovery.seal();
        let status = self.recovery.progress();
        if status.failed || status.blocked || std::thread::panicking() {
            self.drafter.poisoned.set(true);
        }
    }
}

impl MlxDrafter {
    /// Preserves the target ownership established by core's drafting handoff.
    /// Equal checkpoints, geometry or communicators do not identify a session.
    pub(crate) fn validate_target(
        &self,
        target: &super::super::session::MlxModelSession,
    ) -> Result<(), Error> {
        if self.payload.target_session != target.native_session_identity() {
            return Err(Error::Speculative(
                "external drafter belongs to a different target session".into(),
            ));
        }
        Ok(())
    }

    /// Immutable conditioned resource projection over the actual selected parameter owner.
    pub(crate) fn conditioned_memory_profile(
        &self,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::ConditionedAssistantMemoryProfile>,
        eredu_runtime::memory_forecast::GenerationForecastError,
    > {
        let DrafterExecution::Assistant(assistant) = &self.payload.execution else {
            return Ok(None);
        };
        let host_execution = self
            .stream()
            .get_device()
            .and_then(|d| d.get_type())
            .map_err(eredu_core::BackendFailure::from_error)?
            == safemlx::DeviceType::Cpu;
        assistant.visit_ref(ObserveAssistantMemory {
            host_execution,
            residency: assistant.residency(),
        })
    }

    /// Immutable facts for independent drafting; observing this does not acquire
    /// mutation authority, synchronize work or allocate an execution state.
    pub(crate) fn autoregressive_memory_profile(
        &self,
        mut profile: eredu_runtime::memory_forecast::LoadedMemoryProfile,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::LoadedMemoryProfile>,
        eredu_runtime::memory_forecast::GenerationForecastError,
    > {
        let DrafterExecution::Autoregressive { model, .. } = &self.payload.execution else {
            return Ok(None);
        };
        profile.geometry = model.memory_geometry()?;
        profile.parameter_conversions = model
            .residency_report()
            .map_err(eredu_core::BackendFailure::from_error)?
            .and_then(|report| report.device_parameter_conversions().map(<[_]>::to_vec));
        profile.parameters = crate::composition::mlx::capability::static_memory_from_residency(
            model
                .residency_report()
                .map_err(eredu_core::BackendFailure::from_error)?,
            model
                .parameter_bank_report()
                .map_err(eredu_core::BackendFailure::from_error)?,
        )?;
        profile.host_execution = self
            .stream()
            .get_device()
            .and_then(|d| d.get_type())
            .map_err(eredu_core::BackendFailure::from_error)?
            == safemlx::DeviceType::Cpu;
        Ok(Some(profile))
    }

    fn begin_operation(&mut self) -> Result<DrafterOperation<'_>, Error> {
        crate::backend::submission_recovery::reap();
        ordinary_retirement::reclaim();
        if self.poisoned.get() {
            return Err(Error::Speculative(
                "native drafter is poisoned by failed or unresolved work".into(),
            ));
        }
        // A healthy terminal scope can still retain its payload while another
        // thread owns the native runtime. Terminal evidence is not exclusive
        // mutation authority: reject until retirement releases that owner.
        if Rc::get_mut(&mut self.payload).is_none() {
            return Err(Error::Speculative(
                "native drafter payload is still retained by prior work".into(),
            ));
        }
        let retained = Rc::new(RefCell::new(None));
        let recovery = Recovery::begin(DrafterRetention {
            _payload: Rc::clone(&retained),
            poisoned: Rc::clone(&self.poisoned),
        })?;
        Ok(DrafterOperation {
            drafter: self,
            retained,
            recovery,
            completed: false,
        })
    }

    /// Installs typed production observers for external-assistant tensors and logits.
    pub fn install_external_observers<TensorObserver, LogitsObserver>(
        &mut self,
        tensors: TensorObserver,
        logits: LogitsObserver,
    ) -> Result<(), Error>
    where
        TensorObserver: eredu_runtime::ActivationObserver<MlxTensor, Exception> + 'static,
        LogitsObserver: eredu_runtime::ActivationObserver<Array, Exception> + 'static,
    {
        if self.is_autoregressive() {
            return Err(Error::Speculative("independent drafts expose ordinary model observations, not target-feature assistant observations".into()));
        }
        let mut operation = self.begin_operation()?;
        operation
            .payload()
            .execution
            .visit(InstallExternalObservers {
                observers: Some(
                    eredu_architectures::external_assistant::ExternalAssistantObservers::new(
                        tensors, logits,
                    ),
                ),
            });
        operation.finish(Ok(()))
    }

    /// Materializes an architecture-inspected drafter with proven tokenizer compatibility.
    pub(crate) fn materialize(
        preparation: eredu_architectures::PreparedExternalDraft,
        target: &super::super::session::MlxModelSession,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        ordinary_retirement::reclaim();
        let poisoned = Rc::new(Cell::new(false));
        let retained = Rc::new(RefCell::new(None));
        let mut recovery = Recovery::begin(DrafterRetention {
            _payload: Rc::clone(&retained),
            poisoned: Rc::clone(&poisoned),
        })?;
        let execution = match preparation {
            eredu_architectures::PreparedExternalDraft::Assistant(p) => {
                DrafterExecution::Assistant(p.materialize(MlxAssistantPreparationVisitor {
                    stream: stream.clone(),
                    weights_stream: weights_stream.clone(),
                })?)
            }
            eredu_architectures::PreparedExternalDraft::Autoregressive(p) => {
                let (sources, selected, tokenizer) = p.into_parts();
                let model = crate::composition::mlx::loading::materialize_model_plan(
                    sources,
                    None,
                    stream,
                    weights_stream,
                )?;
                DrafterExecution::Autoregressive {
                    model,
                    selected,
                    tokenizer,
                }
            }
        };
        let payload = Rc::new(OrdinaryRetirement::new(DrafterPayload {
            execution,
            stream: stream.clone(),
            target_session: target.native_session_identity().to_owned(),
        }));
        retained.replace(Some(Rc::clone(&payload)));
        recovery.seal();
        let status = recovery.wait();
        if !status.settled || status.failed || status.blocked {
            return Err(Error::Speculative(
                "native drafter materialization failed or remains unresolved".into(),
            ));
        }
        drop(recovery);
        retained.borrow_mut().take();
        Ok(Self { payload, poisoned })
    }

    pub(in crate::composition::mlx) fn prepare_operation(
        &mut self,
        prepare: impl FnOnce(Result<(), Error>) -> Result<(), Error>,
    ) -> Result<DrafterOperation<'_>, Error> {
        let (operation, local) = match self.begin_operation() {
            Ok(operation) => (Some(operation), Ok(())),
            Err(error) => (None, Err(error)),
        };
        let prepared = prepare(local);
        let Some(operation) = operation else {
            return prepared.and_then(|()| {
                Err(Error::Speculative(
                    "drafter preparation accepted absent mutation authority".into(),
                ))
            });
        };
        if let Err(error) = prepared {
            // No assistant state or parameter visitor has run. A peer refusal
            // releases this unused scope only after native completion evidence.
            return Err(operation.finish(Ok(error))?);
        }
        Ok(operation)
    }

    pub(crate) fn is_autoregressive(&self) -> bool {
        matches!(
            self.payload.execution,
            DrafterExecution::Autoregressive { .. }
        )
    }
    /// Returns the portable proof established before this assistant was materialized.
    pub fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.payload.execution.tokenizer_compatibility()
    }

    /// Execution stream selected when this drafter was loaded.
    pub fn stream(&self) -> &Stream {
        &self.payload.stream
    }

    /// Returns topology selected by the portable execution plan before queue construction.
    pub fn topology(&self) -> SpeculativeExecutionTopology {
        self.payload.execution.selected().placement().topology()
    }

    /// Returns the one neutral realization retained from preconstruction selection.
    pub fn selected(&self) -> &SelectedSpeculativeRealization {
        self.payload.execution.selected()
    }

    pub(crate) fn capture(
        &self,
    ) -> &eredu_architectures::composite_execution::ExternalPredictionCaptureRequest {
        self.payload.execution.capture()
    }
}

struct InstallExternalObservers {
    observers: Option<
        eredu_architectures::external_assistant::ExternalAssistantObservers<
            MlxTensor,
            Array,
            Exception,
        >,
    >,
}

impl eredu_architectures::MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor>
    for InstallExternalObservers
{
    type Output = ();

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        mut self,
        assistant: &mut MlxExternalAssistant<A>,
    ) -> Self::Output {
        assistant.observers = self
            .observers
            .take()
            .expect("external observers are installed exactly once");
    }
}

impl eredu_core::residency::ParameterConversionRetentionObserver for MlxDrafter {
    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    > {
        match &self.payload.execution {
            DrafterExecution::Autoregressive { model, .. } => {
                model.parameter_conversion_retention()
            }
            DrafterExecution::Assistant(assistant) => {
                assistant.visit_ref(ObserveAssistantConversions)
            }
        }
    }
}

#[cfg(test)]
#[path = "assistant_recovery_tests.rs"]
mod recovery_tests;

impl eredu_core::residency::ParameterConversionRetentionTrimmer for MlxDrafter {
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        let mut operation = self
            .begin_operation()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let result = match &mut operation.payload().execution {
            DrafterExecution::Autoregressive { model, .. } => model.trim_parameter_conversions(),
            DrafterExecution::Assistant(assistant) => assistant.visit(TrimAssistantConversions),
        };
        operation
            .finish(Ok(result))
            .map_err(eredu_core::BackendFailure::from_error)?
    }
}
