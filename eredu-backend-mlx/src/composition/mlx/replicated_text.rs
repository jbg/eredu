//! MLX mechanisms and generic binding for replicated text composition.

use std::{marker::PhantomData, path::Path, sync::Arc};

use eredu_checkpoint::{store::CheckpointSource, LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    StateComponentRole,
};
use eredu_nn::NeuralBackend;
use eredu_runtime::{
    ArchitectureStateFactory, BackendMechanismCapabilities, CacheResidencyPolicy,
    CacheResidencyReport, DenseDiskStreamReport, GroupedOperationRequirement, LayerRuntimeState,
    LayerwiseRuntime, ReplicatedTextArchitecture, ReplicatedTextRequirements,
    ReplicatedTextSelectionRequest, ResidencyReport, SelectedReplicatedTextRealization,
    SelectedStateRealization, StateComponentMechanism, StateComponentPlacement,
    StateMechanismCapabilities, WeightLoweringCapability, WeightLoweringDescriptor,
    WeightLoweringKind, WeightResidencyMechanism,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

#[cfg(test)]
use eredu_runtime::PagedCacheOptions;

use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::{
                residency::{
                    load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
                },
                state::{MlxHybridState, MlxKeyValueState},
            },
            execution::generic::{MlxLayerwisePolicy, MlxResidentPolicy},
            media::input,
        },
    },
    native_quantization::NativeQuantizationFormat,
    MlxTensor,
};

use crate::backend::nn::shared::MlxModule;
use crate::backend::runtime::checkpoint::binding::{
    build_module_bindings_with_recipes_excluding,
    build_neutral_module_bindings_with_recipes_excluding,
};
use crate::backend::runtime::execution::{
    generic::{
        architecture_execution_layout, construct_architecture_unit,
        prepare_layerwise_policy_with_bindings,
    },
    layerwise::quantize_module_store_with_bindings,
};

use eredu_architectures::replicated_text::{
    PreparedReplicatedTextArchitecture, ReplicatedTextArchitectureVisitor,
};

/// Reports the exact MLX mechanisms applicable to one neutral requirement set.
///
/// The report is derived only from source encodings, executable formats, and
/// implemented backend facilities. It does not receive architecture identity.
pub(crate) const GROUPED_OPERATION_CAPABILITIES: [GroupedOperationRequirement; 4] = [
    GroupedOperationRequirement::GatedProduct,
    GroupedOperationRequirement::GatedProductTensorParallelPartial,
    GroupedOperationRequirement::Relu2,
    GroupedOperationRequirement::Relu2TensorParallelPartial,
];

pub(crate) fn capabilities(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
) -> BackendMechanismCapabilities {
    let mut weight_lowerings = Vec::new();
    for parameter in requirements.parameters() {
        if !parameter.has_lowering_source() {
            continue;
        }
        let requested = request
            .quantization()
            .and_then(|requested| parameter.transform_target(requested).ok().flatten())
            .map(|target| target.executable());
        for executable in std::iter::once(parameter.native_executable()).chain(requested) {
            let descriptor = parameter
                .lowering_descriptor(executable)
                .expect("validated replicated parameter forms a lowering query");
            let kind =
                if executable == parameter.native_executable() && supports_direct(&descriptor) {
                    Some(WeightLoweringKind::Direct)
                } else if supports_transform(&descriptor) {
                    Some(WeightLoweringKind::Transform)
                } else {
                    None
                };
            if let Some(kind) = kind {
                let capability = WeightLoweringCapability::new(descriptor, kind);
                if !weight_lowerings.contains(&capability) {
                    weight_lowerings.push(capability);
                }
            }
        }
    }
    let state =
        StateMechanismCapabilities::new((0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .expect("validated state layout exposes every layer")
                .iter()
                .filter_map(move |component| {
                    let paged = match component.role() {
                        StateComponentRole::AttentionKeys
                        | StateComponentRole::AttentionValues
                        | StateComponentRole::CompressedLatent
                        | StateComponentRole::RotaryKeys => StateComponentPlacement::Paged,
                        StateComponentRole::Fixed(_) => StateComponentPlacement::Device,
                    };
                    mlx_supports_state_component(component).then(|| {
                        StateComponentMechanism::new(
                            layer,
                            component.clone(),
                            Some(StateComponentPlacement::Device),
                            Some(paged),
                        )
                    })
                })
        }))
        .with_transactions(true, true)
        .with_reset(true)
        .with_prompt_cache(matches!(request.state(), CacheResidencyPolicy::Paged(_)))
        .with_observation_retention(true);
    BackendMechanismCapabilities::new(
        MlxNeuralBackend::OPERATOR_CAPABILITIES,
        weight_lowerings,
        vec![
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
            WeightResidencyMechanism::DiskStreamed,
        ],
        state,
    )
    .with_session(eredu_core::SessionCapabilities::new(true, true, true))
    .with_grouped_operations(GROUPED_OPERATION_CAPABILITIES)
    .with_prompt_cache(true)
    .with_exact_completion(true)
}

fn mlx_supports_state_component(component: &eredu_core::cache::StateComponentPolicy) -> bool {
    use eredu_core::cache::{StateTensorDimension, StateTensorDtype};

    !component.shape().is_empty()
        && component.shape().iter().all(|dimension| match dimension {
            StateTensorDimension::Fixed(value)
            | StateTensorDimension::PrefixTokensDiv(value)
            | StateTensorDimension::PrefixTokensRem(value) => value.get() > 0,
            StateTensorDimension::Batch | StateTensorDimension::PrefixTokens => true,
            StateTensorDimension::Scalar => component.shape().len() == 1,
        })
        && matches!(
            component.dtype(),
            StateTensorDtype::Floating
                | StateTensorDtype::Float32
                | StateTensorDtype::Int32
                | StateTensorDtype::Uint32
        )
}

#[cfg(test)]
type StatePresenceSnapshot = Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;

#[cfg(test)]
type FixedNumericStateSnapshot = Vec<(
    usize,
    eredu_core::cache::StateTensorRole,
    Vec<i32>,
    Vec<f32>,
)>;

#[cfg(test)]
type RetainedNumericStateSnapshot = Vec<(Vec<i32>, Vec<f32>)>;

#[cfg(test)]
type CheckpointRestoreProbe = (
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    Vec<f32>,
);

/// Backend-private erased operations for a paired architecture and mutable state.
pub trait ErasedReplicatedTextExecutable {
    fn effective_model_type(&self) -> &str;
    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate;
    fn selected_session_binding(&self) -> &SelectedSessionBinding;
    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency;
    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot;
    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception>;
    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error>;
    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport>;
    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity;
    fn reset_cache(&mut self) -> Result<(), Exception>;
    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error>;
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error>;
    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error>;
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error>;
}

/// Session mechanisms admitted by the authoritative realization.
pub struct SelectedSessionBinding {
    capabilities: eredu_core::SessionCapabilities,
    prompt_cache: bool,
    exact_completion: bool,
}

impl SelectedSessionBinding {
    fn from_selected(selected: &SelectedReplicatedTextRealization) -> Self {
        Self {
            capabilities: selected.session(),
            prompt_cache: selected.prompt_cache(),
            exact_completion: selected.exact_completion(),
        }
    }

    pub(crate) fn validate_bound_mechanisms(
        &self,
        capabilities: eredu_core::SessionCapabilities,
        prompt_cache: bool,
        exact_completion: bool,
    ) -> Result<(), String> {
        self.capabilities
            .validate(&capabilities)
            .map_err(|error| error.to_string())?;
        if self.prompt_cache && !prompt_cache {
            return Err("selected prompt-cache persistence was not bound by the session".into());
        }
        if self.exact_completion && !exact_completion {
            return Err("selected exact completion was not bound by the session".into());
        }
        Ok(())
    }

    fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }
}

struct MlxKeyValueStateFactory<'a> {
    selected: &'a SelectedStateRealization,
}

impl ArchitectureStateFactory<MlxNeuralBackend> for MlxKeyValueStateFactory<'_> {
    type State = MlxKeyValueState;
    type Error = Error;

    fn realize(&mut self, layout: &eredu_runtime::StateLayout) -> Result<Self::State, Self::Error> {
        validate_selected_state(self.selected, layout)?;
        #[cfg(test)]
        super::path_instrumentation::state_allocation();
        match self.selected.policy() {
            CacheResidencyPolicy::Device => {
                MlxKeyValueState::device(layout.clone()).map_err(Into::into)
            }
            CacheResidencyPolicy::Paged(options) => MlxKeyValueState::paged(
                layout.clone(),
                CacheResidencyManager::new(options.clone())
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )
            .map_err(Into::into),
        }
    }
}

struct MlxHybridStateFactory<'a> {
    selected: &'a SelectedStateRealization,
}

impl ArchitectureStateFactory<MlxNeuralBackend> for MlxHybridStateFactory<'_> {
    type State = MlxHybridState;
    type Error = Error;

    fn realize(&mut self, layout: &eredu_runtime::StateLayout) -> Result<Self::State, Self::Error> {
        validate_selected_state(self.selected, layout)?;
        #[cfg(test)]
        super::path_instrumentation::state_allocation();
        match self.selected.policy() {
            CacheResidencyPolicy::Device => {
                MlxHybridState::device(layout.clone()).map_err(Into::into)
            }
            CacheResidencyPolicy::Paged(options) => MlxHybridState::paged(
                layout.clone(),
                CacheResidencyManager::new(options.clone())
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )
            .map_err(Into::into),
        }
    }
}

fn validate_selected_state(
    selected: &SelectedStateRealization,
    layout: &eredu_runtime::StateLayout,
) -> Result<(), Error> {
    if selected.layout() != layout {
        return Err(Error::ArchitectureModel(
            "selected state layout differs from constructed architecture".into(),
        ));
    }
    let expected = (0..layout.len())
        .flat_map(|layer| {
            layout
                .components(layer)
                .expect("validated state layout exposes every layer")
                .iter()
                .map(move |component| {
                    let placement = match (selected.policy(), component.role()) {
                        (CacheResidencyPolicy::Device, _) => StateComponentPlacement::Device,
                        (
                            CacheResidencyPolicy::Paged(_),
                            StateComponentRole::AttentionKeys
                            | StateComponentRole::AttentionValues
                            | StateComponentRole::CompressedLatent
                            | StateComponentRole::RotaryKeys,
                        ) => StateComponentPlacement::Paged,
                        (CacheResidencyPolicy::Paged(_), StateComponentRole::Fixed(_)) => {
                            StateComponentPlacement::Device
                        }
                    };
                    (layer, component, placement)
                })
        })
        .collect::<Vec<_>>();
    let exact = selected.components().len() == expected.len()
        && selected
            .components()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| {
                actual.layer() == expected.0
                    && actual.component() == expected.1
                    && actual.placement() == expected.2
            });
    if !exact {
        return Err(Error::ArchitectureModel(
            "selected state component realization differs from MLX mechanisms".into(),
        ));
    }
    if !selected.checkpoint() || !selected.rollback() || !selected.reset() {
        return Err(Error::ArchitectureModel(
            "selected state lifecycle omits required replicated facilities".into(),
        ));
    }
    Ok(())
}

trait MlxReplicatedState: LayerRuntimeState<MlxNeuralBackend> + Sized {
    fn realize(selected: &SelectedStateRealization) -> Result<Self, Error>;
    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error>;
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    #[cfg(test)]
    fn deep_checkpoint(&self) -> Result<Self, Exception>;
    #[cfg(test)]
    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception>;
    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;
    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    >;
    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception>;
}

impl MlxReplicatedState for MlxKeyValueState {
    fn realize(selected: &SelectedStateRealization) -> Result<Self, Error> {
        MlxKeyValueStateFactory { selected }.realize(selected.layout())
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        _stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = MlxKeyValueState::paged(selected.layout().clone(), manager, None)?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxKeyValueState::save_prompt_cache(
            self,
            destination,
            descriptor,
            prefix_token_ids,
            options,
        )
        .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxKeyValueState::residency_report(self)
    }

    #[cfg(test)]
    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    #[cfg(test)]
    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxKeyValueState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.as_ref()
            .iter()
            .map(|layer| (eredu_nn::AttentionCache::offset(layer), Vec::new()))
            .collect()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        Ok(Vec::new())
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        Ok(Vec::new())
    }
}

impl MlxReplicatedState for MlxHybridState {
    fn realize(selected: &SelectedStateRealization) -> Result<Self, Error> {
        MlxHybridStateFactory { selected }.realize(selected.layout())
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut state = MlxHybridState::paged(selected.layout().clone(), manager, None)?;
        state.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Error::Parallel("prompt-cache prefix exceeds i32".into()))?,
            identity.layer_prefix_offsets(),
        )?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxHybridState::save_prompt_cache(self, destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxHybridState::residency_report(self)
    }

    #[cfg(test)]
    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    #[cfg(test)]
    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxHybridState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.semantic_snapshot()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        self.fixed_numeric_snapshot()
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        self.retained_numeric_snapshot()
    }
}

type ResidentRuntime<A, S> = LayerwiseRuntime<
    A,
    MlxNeuralBackend,
    S,
    MlxResidentPolicy<<A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>>::Unit>,
>;
type BoundedRuntime<A, S> = LayerwiseRuntime<
    A,
    MlxNeuralBackend,
    S,
    MlxLayerwisePolicy<<A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>>::Unit>,
>;

enum Execution<A, S>
where
    S: MlxReplicatedState,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    Resident(ResidentRuntime<A, S>),
    Bounded(BoundedRuntime<A, S>),
}

struct BoundReplicatedText<A, S>
where
    S: MlxReplicatedState,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    execution: Execution<A, S>,
    state_layout: eredu_runtime::StateLayout,
    state: S,
    prompt_cache_identity: PromptCacheModelIdentity,
    capability_estimate: eredu_architectures::capability::CapabilityEstimate,
    effective_model_type: String,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    #[cfg(test)]
    selected_residency: eredu_runtime::LayerWeightResidency,
    selected_session: SelectedSessionBinding,
    selected_state: SelectedStateRealization,
    stream: Stream,
}

impl<A, S> BoundReplicatedText<A, S>
where
    S: MlxReplicatedState,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn new(
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let requirements = prepared.requirements().clone();
        let selected = prepared.selected().clone();
        let selected_session = SelectedSessionBinding::from_selected(&selected);
        let selected_state = selected.state().clone();
        let capability_estimate = prepared.capability_estimate().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let residency = selected.residency();
        let mut modules = prepared.into_modules();
        let mut architecture = modules.take_architecture();
        let source_architecture = modules.take_source_architecture();
        let static_recipes = modules.take_static_recipes();
        let unit_recipes = modules.take_unit_recipes();
        validate_architecture_contract(&architecture, &requirements)?;
        validate_parameter_contract(
            source_architecture.as_ref().unwrap_or(&architecture),
            &requirements,
            stream,
        )?;
        let (store, materialization) = match source_architecture {
            Some(source) => {
                let quantization = selected_transform_quantization(&selected)?;
                let source_layout = architecture_execution_layout::<_, S>(&source)?;
                let target_layout = architecture_execution_layout::<_, S>(&architecture)?;
                if source_layout != target_layout {
                    return Err(Error::Quantization(
                        "selected weight transform changed the execution-unit layout".into(),
                    ));
                }
                let unit_count = source_layout.len();
                let source_static = source.static_modules().clone();
                let target_static = architecture.static_modules().clone();
                let quantization_static_recipes = static_recipes.clone();
                let quantization_unit_recipes = unit_recipes.clone();
                let (store, report) = quantize_module_store_with_bindings(
                    store,
                    &source_static,
                    &target_static,
                    |ordinal, context| {
                        construct_architecture_unit(
                            &source,
                            &source_layout,
                            ordinal,
                            context,
                            PhantomData::<S>,
                        )
                    },
                    |ordinal, context| {
                        construct_architecture_unit(
                            &architecture,
                            &target_layout,
                            ordinal,
                            context,
                            PhantomData::<S>,
                        )
                    },
                    unit_count,
                    quantization,
                    stream,
                    move |modules, store| {
                        let mut recipes = quantization_static_recipes;
                        build_neutral_module_bindings_with_recipes_excluding(
                            modules,
                            store,
                            &mut recipes,
                            |_| false,
                        )
                        .map_err(Into::into)
                    },
                    move |ordinal, unit, store| {
                        let mut recipes = quantization_unit_recipes
                            .get(ordinal)
                            .cloned()
                            .unwrap_or_default();
                        build_neutral_module_bindings_with_recipes_excluding(
                            unit,
                            store,
                            &mut recipes,
                            |_| false,
                        )
                        .map_err(Into::into)
                    },
                )?;
                (store, Some(report))
            }
            None => (store, None),
        };
        let state_layout = architecture
            .state_layout()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let prompt_cache_identity = crate::composition::replicated_prompt_cache_identity(
            &architecture,
            eredu_core::cache::PromptCacheTopology::default(),
        )?;
        let (policy, _) = prepare_layerwise_policy_with_bindings(
            store,
            &mut architecture,
            (),
            PhantomData::<S>,
            residency,
            stream,
            weights_stream,
            |_| false,
            move |modules, store| {
                build_module_bindings_with_recipes_excluding(
                    &MlxModule::new(modules.clone()),
                    "",
                    store,
                    static_recipes,
                    |_| false,
                )
                .map_err(Into::into)
            },
            move |ordinal, _address, _path, unit, store, _stream| {
                let recipes = unit_recipes.get(ordinal).cloned().unwrap_or_default();
                build_module_bindings_with_recipes_excluding(
                    &MlxModule::new(unit),
                    "",
                    store,
                    recipes,
                    |_| false,
                )
                .map_err(Into::into)
            },
        )?;
        let execution = if residency.is_fully_resident() {
            Execution::Resident(LayerwiseRuntime::new_policy_first(
                policy.into_resident(&architecture, stream, PhantomData::<S>)?,
                architecture,
            ))
        } else {
            Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
        };
        let state = S::realize(&selected_state)?;
        Ok(Self {
            execution,
            state_layout,
            state,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            materialization,
            #[cfg(test)]
            selected_residency: residency,
            selected_session,
            selected_state,
            stream: stream.clone(),
        })
    }

    fn forward(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        self.validate_state()?;
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let input = A::text_input(&tokens, mask.as_ref());
        let output = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward(input, &mut self.state, stream),
            Execution::Bounded(runtime) => runtime.forward(input, &mut self.state, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(output.into_array())
    }

    fn validate_state(&self) -> Result<(), Error> {
        if self.state.layout() != &self.state_layout {
            return Err(Error::ArchitectureModel(
                "replicated text state layout does not match its paired architecture".into(),
            ));
        }
        Ok(())
    }

    fn new_state(&self) -> Result<S, Error> {
        S::realize(&self.selected_state)
    }
}

fn validate_architecture_contract<A, S>(
    architecture: &A,
    requirements: &ReplicatedTextRequirements,
) -> Result<(), Error>
where
    S: MlxReplicatedState,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if &graph != requirements.execution_graph() {
        return Err(Error::ArchitectureModel(
            "constructed execution graph differs from selected replicated requirements".into(),
        ));
    }
    let units = architecture_execution_layout::<_, S>(architecture)?;
    if &units != requirements.execution_units() {
        return Err(Error::ArchitectureModel(
            "constructed execution-unit geometry differs from selected replicated requirements"
                .into(),
        ));
    }
    let transports = (0..graph.groups().len())
        .map(|group| architecture.group_transport(group))
        .collect::<Vec<_>>();
    if transports != requirements.group_transports() {
        return Err(Error::ArchitectureModel(
            "constructed group transport differs from selected replicated requirements".into(),
        ));
    }
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if &state_layout != requirements.state_layout() {
        return Err(Error::ArchitectureModel(
            "constructed mutable-state layout differs from selected replicated requirements".into(),
        ));
    }
    Ok(())
}

fn validate_parameter_contract<A, S>(
    architecture: &A,
    requirements: &ReplicatedTextRequirements,
    context: &Stream,
) -> Result<(), Error>
where
    S: MlxReplicatedState,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    use std::collections::{BTreeMap, BTreeSet};

    use eredu_runtime::{ParameterGroupOwner, ReplicatedTextParameterOwner};

    let description = architecture
        .parameter_description(context)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut actual = BTreeMap::new();
    for group in description.groups() {
        for member in group.group().members() {
            actual.insert(member.target(), (member.global_shape(), group.owner()));
        }
    }
    let expected = requirements
        .parameters()
        .iter()
        .filter(|parameter| {
            !matches!(
                parameter.presence(),
                eredu_runtime::ReplicatedTextParameterPresence::OptionalAbsent
                    | eredu_runtime::ReplicatedTextParameterPresence::Tied { .. }
            )
        })
        .map(|parameter| parameter.name())
        .collect::<BTreeSet<_>>();
    let actual_names = actual.keys().copied().collect::<BTreeSet<_>>();
    if expected != actual_names {
        return Err(Error::ArchitectureModel(format!(
            "replicated parameter requirements differ from architecture topology: missing {:?}, unexpected {:?}",
            expected.difference(&actual_names).collect::<Vec<_>>(),
            actual_names.difference(&expected).collect::<Vec<_>>()
        )));
    }
    for parameter in requirements.parameters().iter().filter(|parameter| {
        !matches!(
            parameter.presence(),
            eredu_runtime::ReplicatedTextParameterPresence::OptionalAbsent
                | eredu_runtime::ReplicatedTextParameterPresence::Tied { .. }
        )
    }) {
        let (shape, owner) = actual
            .get(parameter.name())
            .expect("equal parameter-name sets contain every requirement");
        let owner_matches = match (owner, parameter.owner()) {
            (
                ParameterGroupOwner::StaticRole(actual),
                ReplicatedTextParameterOwner::StaticRole(expected),
            ) => actual == expected,
            (
                ParameterGroupOwner::StaticAnyOf(actual),
                ReplicatedTextParameterOwner::StaticRole(expected),
            ) => actual.iter().any(|role| role == expected),
            (
                ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                },
                ReplicatedTextParameterOwner::ExecutionUnit {
                    group: expected_group,
                    unit: expected_unit,
                },
            ) => group.as_str() == expected_group && global_unit == expected_unit,
            _ => false,
        };
        let mut expected_shape = parameter.logical_shape().to_vec();
        if let Some(last) = expected_shape.last_mut() {
            match parameter.native_executable() {
                LinearFormat::Affine(config) => {
                    *last = last.saturating_mul(config.bits as usize) / 32;
                }
                LinearFormat::MxFp4 => *last = last.saturating_mul(4) / 32,
                LinearFormat::GgufIQuant { ggml_type, .. } => {
                    if let Ok((block, bytes)) = ggml_type.block_and_bytes() {
                        if let (Ok(block), Ok(bytes)) =
                            (usize::try_from(block), usize::try_from(bytes))
                        {
                            *last = last.saturating_mul(bytes) / block;
                        }
                    }
                }
                LinearFormat::Dense | LinearFormat::E4M3BlockFp8(_) => {}
            }
        }
        if *shape != expected_shape || !owner_matches {
            return Err(Error::ArchitectureModel(format!(
                "replicated parameter requirement {:?} differs from architecture shape/owner: requirement {:?} {:?}, architecture {:?} {:?}",
                parameter.name(),
                expected_shape,
                parameter.owner(),
                shape,
                owner,
            )));
        }
    }
    Ok(())
}

impl<A, S> ErasedReplicatedTextExecutable for BoundReplicatedText<A, S>
where
    S: MlxReplicatedState + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    fn selected_session_binding(&self) -> &SelectedSessionBinding {
        &self.selected_session
    }

    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.selected_residency
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot {
        MlxReplicatedState::state_snapshot(&self.state)
    }

    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception> {
        MlxReplicatedState::fixed_numeric_snapshot(&self.state)
    }

    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error> {
        let before = MlxReplicatedState::state_snapshot(&self.state);
        let before_numeric = MlxReplicatedState::fixed_numeric_snapshot(&self.state)?;
        let before_retained = MlxReplicatedState::retained_numeric_snapshot(&self.state)?;
        let checkpoint = self.state.deep_checkpoint()?;
        let continuation = self
            .forward(tokens, None, stream)?
            .evaluated()?
            .as_slice::<f32>()
            .to_vec();
        let advanced = MlxReplicatedState::state_snapshot(&self.state);
        let advanced_numeric = MlxReplicatedState::fixed_numeric_snapshot(&self.state)?;
        self.state.restore_checkpoint(&checkpoint, stream)?;
        let restored = MlxReplicatedState::state_snapshot(&self.state);
        let restored_numeric = MlxReplicatedState::fixed_numeric_snapshot(&self.state)?;
        let restored_retained = MlxReplicatedState::retained_numeric_snapshot(&self.state)?;
        assert_eq!(restored_retained, before_retained);
        Ok((
            before,
            advanced,
            restored,
            before_numeric,
            advanced_numeric,
            restored_numeric,
            continuation,
        ))
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report()?,
            Execution::Bounded(runtime) => runtime.policy().residency_report()?,
        };
        Ok(Some(report))
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.materialization.as_ref()
    }

    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    fn reset_cache(&mut self) -> Result<(), Exception> {
        self.state = self
            .new_state()
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(())
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        if !self.selected_session.prompt_cache() {
            return Err(Error::ArchitectureModel(
                "prompt-cache persistence was not selected for this state realization".into(),
            ));
        }
        eredu_core::cache::validate_prompt_cache_model_identity(
            expected,
            &self.prompt_cache_identity,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let (state, manifest) = S::load_prompt_cache(
            &self.selected_state,
            directory,
            expected,
            &self.prompt_cache_identity,
            prefix_token_ids,
            &self.stream,
        )?;
        self.state = state;
        Ok(manifest)
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        if !self.selected_session.prompt_cache() {
            return Err(Error::ArchitectureModel(
                "prompt-cache persistence was not selected for this state realization".into(),
            ));
        }
        eredu_core::cache::validate_prompt_cache_model_identity(
            &descriptor,
            &self.prompt_cache_identity,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        MlxReplicatedState::save_prompt_cache(
            &mut self.state,
            destination,
            descriptor,
            prefix_token_ids,
            options,
        )
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxReplicatedState::residency_report(&self.state)
    }

    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, None, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        self.forward(tokens, None, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        self.validate_state()?;
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let input = A::text_input(&tokens, mask.as_ref());
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_observer(input, &mut self.state, stream, &mut observer)
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_observer(input, &mut self.state, stream, &mut observer)
            }
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        eredu_runtime::observe_model_logits(&mut observer, &output)
            .map(MlxTensor::into_array)
            .map_err(Into::into)
    }
}

fn selected_transform_quantization(
    selected: &SelectedReplicatedTextRealization,
) -> Result<eredu_checkpoint::WeightQuantization, Error> {
    let mut quantization = None;
    for parameter in selected.parameters() {
        if !matches!(
            parameter.lowering(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        ) {
            continue;
        }
        let current = parameter
            .executable()
            .weight_quantization()
            .ok_or_else(|| {
                Error::Quantization(format!(
                    "selected transform for {:?} has no materializable packed format",
                    parameter.name()
                ))
            })?;
        if quantization
            .replace(current)
            .is_some_and(|prior| prior != current)
        {
            return Err(Error::Quantization(
                "one replicated text realization selected multiple transform formats".into(),
            ));
        }
    }
    quantization.ok_or_else(|| {
        Error::Quantization("selected transform contains no transformed parameters".into())
    })
}

/// Family-agnostic MLX visitor that binds neutral parameter topology.
pub(crate) struct BindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
}

impl ReplicatedTextArchitectureVisitor<MlxNeuralBackend, MlxKeyValueState> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        BoundReplicatedText::new(prepared, store, self.stream, self.weights_stream)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl ReplicatedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        BoundReplicatedText::new(prepared, store, self.stream, self.weights_stream)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

fn valid_packed_geometry(descriptor: &WeightLoweringDescriptor) -> bool {
    let Some(extent) = descriptor.packed_extent() else {
        return false;
    };
    match descriptor.executable() {
        LinearFormat::Affine(format) => usize::try_from(format.group_size)
            .ok()
            .is_some_and(|group| group != 0 && group <= extent && extent.is_multiple_of(group)),
        LinearFormat::MxFp4 => extent.is_multiple_of(32),
        LinearFormat::GgufIQuant { ggml_type, .. } => ggml_type
            .block_and_bytes()
            .ok()
            .and_then(|(block, _)| usize::try_from(block).ok())
            .is_some_and(|block| extent.is_multiple_of(block)),
        LinearFormat::E4M3BlockFp8(_) => true,
        LinearFormat::Dense => true,
    }
}

fn valid_direct_source_geometry(descriptor: &WeightLoweringDescriptor) -> bool {
    let same_unpacked_dimensions = |packed_axis: usize| {
        descriptor
            .physical_shape()
            .iter()
            .zip(descriptor.logical_shape())
            .enumerate()
            .all(|(axis, (physical, logical))| axis == packed_axis || physical == logical)
    };
    match descriptor.source() {
        SourceTensorEncoding::Gguf { ggml_type, .. } => ggml_type
            .block_and_bytes()
            .ok()
            .and_then(|(block, _)| usize::try_from(block).ok())
            .is_some_and(|block| match descriptor.packed_axis() {
                Some(axis) if same_unpacked_dimensions(axis) => {
                    let physical = descriptor.physical_shape()[axis];
                    let logical = descriptor.logical_shape()[axis];
                    physical >= logical
                        && physical.is_multiple_of(block)
                        && physical - logical < block
                }
                Some(_) => false,
                None => descriptor.physical_shape() == descriptor.logical_shape(),
            }),
        SourceTensorEncoding::Safetensors(StoredDtype::U32) => {
            let Some(axis) = descriptor.packed_axis() else {
                return false;
            };
            if !same_unpacked_dimensions(axis) {
                return false;
            }
            let bits = match descriptor.executable() {
                LinearFormat::Affine(format) => usize::try_from(format.bits).ok(),
                LinearFormat::MxFp4 => Some(4),
                _ => None,
            };
            bits.is_some_and(|bits| {
                descriptor.physical_shape()[axis].checked_mul(32)
                    == descriptor.logical_shape()[axis].checked_mul(bits)
            }) && valid_packed_geometry(descriptor)
        }
        _ => {
            descriptor.physical_shape() == descriptor.logical_shape()
                && (descriptor.executable() == LinearFormat::Dense
                    || valid_packed_geometry(descriptor))
        }
    }
}

fn supports_direct(descriptor: &WeightLoweringDescriptor) -> bool {
    let source = descriptor.source();
    let executable = descriptor.executable();
    let supported = match (source, executable) {
        (
            SourceTensorEncoding::Safetensors(
                StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32,
            ),
            LinearFormat::Dense,
        ) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::MxFp4) => true,
        (
            SourceTensorEncoding::Safetensors(StoredDtype::F8E4M3),
            LinearFormat::E4M3BlockFp8(format),
        ) => format.validate().is_ok(),
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Dense) => matches!(
            ggml_type,
            eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
        ),
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Affine(format)) => {
            gguf_affine(*ggml_type).is_some_and(|native| native == format)
        }
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::MxFp4) => {
            *ggml_type == eredu_gguf::GgmlType::MxFp4
        }
        (
            SourceTensorEncoding::Gguf { ggml_type, endian },
            LinearFormat::GgufIQuant {
                ggml_type: executable,
                endian: executable_endian,
            },
        ) => {
            *ggml_type == executable
                && *endian == executable_endian
                && NativeQuantizationFormat::from_ggml_type(executable).is_some()
        }
        _ => false,
    };
    supported && valid_direct_source_geometry(descriptor)
}

fn supports_transform(descriptor: &WeightLoweringDescriptor) -> bool {
    let source = descriptor.source();
    let executable = descriptor.executable();
    let decodable = match source {
        SourceTensorEncoding::Safetensors(dtype) => matches!(
            dtype,
            StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32 | StoredDtype::F64
        ),
        SourceTensorEncoding::Gguf { ggml_type, .. } => matches!(
            ggml_type,
            eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
        ),
        _ => false,
    };
    decodable
        && descriptor.physical_shape() == descriptor.logical_shape()
        && valid_packed_geometry(descriptor)
        && match executable {
            LinearFormat::Affine(format) => format.validate().is_ok(),
            LinearFormat::MxFp4 => true,
            LinearFormat::Dense
            | LinearFormat::GgufIQuant { .. }
            | LinearFormat::E4M3BlockFp8(_) => false,
        }
}

fn gguf_affine(ggml_type: eredu_gguf::GgmlType) -> Option<eredu_checkpoint::AffineQuantization> {
    let (bits, group_size) = match ggml_type {
        eredu_gguf::GgmlType::Q2K => (2, 16),
        eredu_gguf::GgmlType::Q3K => (3, 16),
        eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::Q4_1 | eredu_gguf::GgmlType::Q4K => {
            (4, 32)
        }
        eredu_gguf::GgmlType::Q5_0 | eredu_gguf::GgmlType::Q5_1 | eredu_gguf::GgmlType::Q5K => {
            (5, 32)
        }
        eredu_gguf::GgmlType::Q6K => (6, 16),
        eredu_gguf::GgmlType::Q8_0 => (8, 32),
        _ => return None,
    };
    eredu_checkpoint::AffineQuantization::new(group_size, bits).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::SourceTensorEncoding;
    use eredu_core::{
        cache::LayerCachePolicy, AttentionPolicy, LayerSchedule, ModelConfigurationResolver,
    };
    use eredu_runtime::{
        ParameterTransformConstraint, ReplicatedTextParameterRequirement,
        ReplicatedTextStateAccess, StateLayout, WeightLoweringDescriptor,
    };

    struct ForgedShapeSource {
        inner: eredu_checkpoint::store::SharedCheckpointSource,
        key: String,
    }

    impl eredu_checkpoint::store::CheckpointSource for ForgedShapeSource {
        fn source_keys(&self) -> Vec<String> {
            self.inner.source_keys()
        }

        fn source_metadata(
            &self,
            key: &str,
        ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError>
        {
            let mut metadata = self.inner.source_metadata(key)?;
            if key == self.key {
                metadata.physical_shape.push(2);
            }
            Ok(metadata)
        }

        fn acquire_lease(
            &self,
            request: eredu_checkpoint::store::TensorReadRequest,
        ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError>
        {
            self.inner.acquire_lease(request)
        }

        fn source_diagnostics(
            &self,
        ) -> Result<
            eredu_checkpoint::store::WeightStoreDiagnostics,
            eredu_checkpoint::store::StoreError,
        > {
            self.inner.source_diagnostics()
        }

        fn source_provenance(
            &self,
            key: &str,
        ) -> Result<
            eredu_checkpoint::store::TensorSourceProvenance,
            eredu_checkpoint::store::StoreError,
        > {
            self.inner.source_provenance(key)
        }
    }

    fn materialize_model_plan(
        plan: eredu_core::ModelPreparationPlan<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        options: crate::MlxLoadRequest,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<crate::backend::MlxModel, crate::backend::error::Error> {
        let selected =
            super::super::loading::select_preparation(plan.inspection(), options, plan.policy())?;
        super::super::loading::materialize_model_plan(plan, selected, stream, weights_stream)
    }

    #[test]
    fn exact_lowering_rejects_unsupported_encodings_and_incoherent_physical_geometry() {
        let affine =
            LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
        let gguf = |physical_shape| {
            WeightLoweringDescriptor::new(
                SourceTensorEncoding::Gguf {
                    ggml_type: eredu_gguf::GgmlType::Q4_0,
                    endian: eredu_gguf::Endian::Little,
                },
                affine,
                physical_shape,
                vec![64, 8],
                Some(1),
            )
            .unwrap()
        };
        assert!(supports_direct(&gguf(vec![64, 32])));
        assert!(!supports_direct(&gguf(vec![63, 32])));
        assert!(!supports_direct(&gguf(vec![64, 31])));
        assert!(!supports_direct(&gguf(vec![64, 64])));

        let safetensors = |source, physical_shape| {
            WeightLoweringDescriptor::new(source, affine, physical_shape, vec![64, 64], Some(1))
                .unwrap()
        };
        assert!(supports_direct(&safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::U32),
            vec![64, 8],
        )));
        assert!(!supports_direct(&safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::U32),
            vec![64, 7],
        )));
        assert!(!supports_direct(&safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::U8),
            vec![64, 64],
        )));

        let transform = safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::F16),
            vec![64, 32],
        );
        assert!(!supports_transform(&transform));
    }

    #[test]
    fn selected_store_geometry_mismatch_rejects_before_construction_or_payload_open() {
        super::super::path_instrumentation::reset();
        let artifact = tiny_heterogeneous_artifact(lfm2_config());
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        );
        let selected = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request,
            &capabilities(&requirements, &request),
        )
        .unwrap();
        let source: eredu_checkpoint::store::SharedCheckpointSource = Arc::new(
            eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
        );
        let key = requirements
            .parameters()
            .iter()
            .find_map(|parameter| parameter.sources().first())
            .unwrap()
            .clone();
        let forged: eredu_checkpoint::store::SharedCheckpointSource =
            Arc::new(ForgedShapeSource { inner: source, key });
        let (stream, weights_stream) = execution_streams();
        let error = super::super::loading::bind_replicated_text(
            inspection.architecture_plan(),
            selected,
            forged,
            &stream,
            &weights_stream,
        )
        .err()
        .expect("forged source geometry must fail");
        assert!(error.to_string().contains("physical geometry"));
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn report_distinguishes_native_and_transforming_lowerings() {
        let parameter = ReplicatedTextParameterRequirement::new(
            "projection.weight",
            vec!["projection.weight".into()],
            vec![eredu_runtime::ReplicatedTextPhysicalSource::new(
                "projection.weight",
                "/checkpoint/model.safetensors",
                "projection.weight",
            )
            .unwrap()],
            Vec::new(),
            Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
            Some(vec![64, 64]),
            vec![64, 64],
            LinearFormat::Dense,
            eredu_runtime::ReplicatedTextParameterRole::LinearWeight,
            eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit {
                group: "decoder".into(),
                unit: 0,
            },
            eredu_runtime::ReplicatedTextParameterPresence::Required,
            ParameterTransformConstraint::Linear { packed_axis: 1 },
        )
        .unwrap();
        let graph = eredu_runtime::ExecutionGraph::chain(["decoder"]).unwrap();
        let requirements = ReplicatedTextRequirements::new(
            "test.generic-binding",
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            eredu_runtime::ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            vec![eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec!["embedding".into()],
                last_owner_static_roles: vec!["output".into()],
                merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            eredu_runtime::ReplicatedTextStateAccess::KeyValue,
            vec![parameter],
        )
        .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_quantization(eredu_core::QuantizationRequest::Affine {
            group_size: 64,
            bits: 4,
        });
        let report = capabilities(&requirements, &request);
        assert!(report.weight_lowerings().iter().any(|lowering| {
            lowering.executable() == LinearFormat::Dense
                && lowering.kind() == WeightLoweringKind::Direct
        }));
        assert!(report.weight_lowerings().iter().any(|lowering| {
            matches!(lowering.executable(), LinearFormat::Affine(_))
                && lowering.kind() == WeightLoweringKind::Transform
        }));
    }

    fn tiny_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
        tiny_safetensors_artifact(model_type, tied, false)
    }

    fn tiny_sharded_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
        tiny_safetensors_artifact(model_type, tied, true)
    }

    fn tiny_safetensors_artifact(model_type: &str, tied: bool, sharded: bool) -> tempfile::TempDir {
        use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

        let root = tempfile::tempdir().unwrap();
        let architecture = match model_type {
            "llama" => "LlamaForCausalLM",
            "mistral" => "MistralForCausalLM",
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "qwen3_moe" => "Qwen3MoeForCausalLM",
            _ => unreachable!(),
        };
        let mut config = serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": tied
        });
        if model_type == "qwen3_moe" {
            config["num_experts"] = 2.into();
            config["num_experts_per_tok"] = 1.into();
            config["moe_intermediate_size"] = 16.into();
        }
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let plan = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        );
        let tensors = constraints
            .into_iter()
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let elements = constraint.shape.iter().product::<usize>();
                let bytes: Vec<u8> = if constraint.key.ends_with(".A_log") {
                    (-1.0_f32)
                        .to_le_bytes()
                        .into_iter()
                        .cycle()
                        .take(elements * 4)
                        .collect()
                } else if constraint.key.contains("norm") && constraint.key.ends_with(".weight") {
                    (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect()
                } else {
                    let seed = constraint.key.bytes().fold(1_u32, |value, byte| {
                        value.wrapping_mul(31) ^ u32::from(byte)
                    });
                    (0..elements)
                        .flat_map(|index| {
                            let signed = i32::try_from((seed as usize + index) % 29).unwrap() - 14;
                            (signed as f32 * 0.001).to_le_bytes()
                        })
                        .collect()
                };
                (constraint.key.clone(), constraint.shape.clone(), bytes)
            })
            .collect::<Vec<_>>();
        let write = |path: &Path, tensors: &[(String, Vec<usize>, Vec<u8>)]| {
            let views = tensors
                .iter()
                .map(|(name, shape, bytes)| {
                    (
                        name.as_str(),
                        TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            serialize_to_file(views, None, path).unwrap();
        };
        if sharded {
            let split = tensors.len() / 2;
            let first = "model-00001-of-00002.safetensors";
            let second = "model-00002-of-00002.safetensors";
            write(&root.path().join(first), &tensors[..split]);
            write(&root.path().join(second), &tensors[split..]);
            let weight_map = tensors
                .iter()
                .enumerate()
                .map(|(index, (name, _, _))| {
                    (name.clone(), if index < split { first } else { second })
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            std::fs::write(
                root.path().join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
            )
            .unwrap();
        } else {
            write(&root.path().join("model.safetensors"), &tensors);
        }
        root
    }

    fn tiny_heterogeneous_artifact(config: serde_json::Value) -> tempfile::TempDir {
        tiny_heterogeneous_artifact_with_layout(config, false)
    }

    fn tiny_heterogeneous_artifact_with_layout(
        config: serde_json::Value,
        fused_qwen_next: bool,
    ) -> tempfile::TempDir {
        use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let plan = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| {
                    if fused_qwen_next && group.variants.iter().any(|variant| variant.id == "fused")
                    {
                        group.variants.iter().find(|variant| variant.id == "fused")
                    } else {
                        group.variants.first()
                    }
                })
                .flat_map(|variant| variant.tensors.iter()),
        );
        let tensors = constraints
            .into_iter()
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let elements = constraint.shape.iter().product::<usize>();
                let bytes: Vec<u8> = if constraint.key.ends_with(".A_log") {
                    (-1.0_f32)
                        .to_le_bytes()
                        .into_iter()
                        .cycle()
                        .take(elements * 4)
                        .collect()
                } else if constraint.key.contains("norm") && constraint.key.ends_with(".weight") {
                    (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect()
                } else {
                    let seed = constraint.key.bytes().fold(1_u32, |value, byte| {
                        value.wrapping_mul(31) ^ u32::from(byte)
                    });
                    (0..elements)
                        .flat_map(|index| {
                            let signed = i32::try_from((seed as usize + index) % 29).unwrap() - 14;
                            (signed as f32 * 0.001).to_le_bytes()
                        })
                        .collect()
                };
                (constraint.key.clone(), constraint.shape.clone(), bytes)
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
        root
    }

    fn tiny_heterogeneous_gguf(family: &str, stream: &Stream) -> crate::test_utils::SyntheticGguf {
        tiny_heterogeneous_gguf_with_packed_qwen_next(family, None, stream)
    }

    fn tiny_heterogeneous_gguf_with_packed_qwen_next(
        family: &str,
        packed_qkvz: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::{MetadataArray, MetadataValue};

        let (plan, metadata) = match family {
            "lfm2" => {
                let args = eredu_architectures::lfm2::model_args_from_config_value(&lfm2_config())
                    .unwrap();
                let key = |suffix: &str| format!("lfm2.{suffix}");
                (
                    eredu_architectures::lfm2::gguf_plan(&args).unwrap(),
                    HashMap::from([
                        (
                            "general.architecture".into(),
                            MetadataValue::String("lfm2".into()),
                        ),
                        ("general.file_type".into(), MetadataValue::Uint32(0)),
                        (key("block_count"), MetadataValue::Uint32(2)),
                        (key("embedding_length"), MetadataValue::Uint32(16)),
                        (
                            key("feed_forward_length"),
                            MetadataValue::Uint32(args.dense_intermediate_size as u32),
                        ),
                        (key("attention.head_count"), MetadataValue::Uint32(4)),
                        (
                            key("attention.head_count_kv"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 2])),
                        ),
                        (
                            key("attention.layer_norm_rms_epsilon"),
                            MetadataValue::Float32(args.norm_eps),
                        ),
                        (key("context_length"), MetadataValue::Uint32(64)),
                        (key("shortconv.l_cache"), MetadataValue::Uint32(3)),
                        (
                            key("rope.freq_base"),
                            MetadataValue::Float32(args.rope.theta),
                        ),
                        (key("vocab_size"), MetadataValue::Uint32(64)),
                    ]),
                )
            }
            "kimi_linear" => {
                let args = eredu_architectures::kimi_linear::model_args_from_config_value(
                    &kimi_linear_config(),
                )
                .unwrap();
                let key = |suffix: &str| format!("kimi-linear.{suffix}");
                (
                    eredu_architectures::kimi_linear::gguf_plan(&args).unwrap(),
                    HashMap::from([
                        (
                            "general.architecture".into(),
                            MetadataValue::String("kimi-linear".into()),
                        ),
                        ("general.file_type".into(), MetadataValue::Uint32(0)),
                        (key("block_count"), MetadataValue::Uint32(2)),
                        (key("embedding_length"), MetadataValue::Uint32(12)),
                        (key("attention.head_count"), MetadataValue::Uint32(3)),
                        (
                            key("attention.head_count_kv"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 1])),
                        ),
                        (key("rope.dimension_count"), MetadataValue::Uint32(2)),
                        (key("attention.key_length_mla"), MetadataValue::Uint32(6)),
                        (key("vocab_size"), MetadataValue::Uint32(64)),
                        (key("feed_forward_length"), MetadataValue::Uint32(16)),
                        (key("context_length"), MetadataValue::Uint32(64)),
                        (
                            key("attention.layer_norm_rms_epsilon"),
                            MetadataValue::Float32(args.rms_norm_eps),
                        ),
                        (key("kda.head_dim"), MetadataValue::Uint32(4)),
                        (key("ssm.conv_kernel"), MetadataValue::Uint32(3)),
                        (key("expert_count"), MetadataValue::Uint32(2)),
                        (key("expert_feed_forward_length"), MetadataValue::Uint32(8)),
                        (key("attention.kv_lora_rank"), MetadataValue::Uint32(6)),
                        (key("attention.value_length_mla"), MetadataValue::Uint32(4)),
                        (key("leading_dense_block_count"), MetadataValue::Uint32(2)),
                        (key("expert_used_count"), MetadataValue::Uint32(1)),
                        (key("expert_shared_count"), MetadataValue::Uint32(1)),
                    ]),
                )
            }
            "nemotron_h" => {
                let args = eredu_architectures::nemotron_h::model_args_from_config_value(
                    &nemotron_h_config(),
                )
                .unwrap();
                let key = |suffix: &str| format!("nemotron_h.{suffix}");
                (
                    eredu_architectures::nemotron_h::gguf_plan(&args).unwrap(),
                    HashMap::from([
                        (
                            "general.architecture".into(),
                            MetadataValue::String("nemotron_h".into()),
                        ),
                        ("general.file_type".into(), MetadataValue::Uint32(0)),
                        (key("block_count"), MetadataValue::Uint32(4)),
                        (key("embedding_length"), MetadataValue::Uint32(16)),
                        (
                            key("feed_forward_length"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 0, 24, 0])),
                        ),
                        (
                            key("attention.head_count_kv"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 2, 0, 0])),
                        ),
                        (key("attention.head_count"), MetadataValue::Uint32(4)),
                        (key("attention.key_length"), MetadataValue::Uint32(4)),
                        (
                            key("attention.layer_norm_rms_epsilon"),
                            MetadataValue::Float32(args.norm_eps),
                        ),
                        (key("context_length"), MetadataValue::Uint32(64)),
                        (key("ssm.inner_size"), MetadataValue::Uint32(16)),
                        (key("ssm.time_step_rank"), MetadataValue::Uint32(4)),
                        (key("ssm.state_size"), MetadataValue::Uint32(3)),
                        (key("ssm.group_count"), MetadataValue::Uint32(2)),
                        (key("ssm.conv_kernel"), MetadataValue::Uint32(3)),
                        (key("vocab_size"), MetadataValue::Uint32(64)),
                    ]),
                )
            }
            "qwen35" | "qwen3next" => {
                let (config, architecture) = if family == "qwen35" {
                    (qwen_hybrid_config(), "qwen35")
                } else {
                    (qwen_next_config(), "qwen3next")
                };
                let args = eredu_architectures::qwen::hybrid::model_args_from_config_value(&config)
                    .unwrap()
                    .text;
                let key = |suffix: &str| format!("{architecture}.{suffix}");
                let plan = eredu_architectures::qwen::hybrid::gguf_plan(&args).unwrap();
                let metadata = HashMap::from([
                    (
                        "general.architecture".into(),
                        MetadataValue::String(architecture.into()),
                    ),
                    ("general.file_type".into(), MetadataValue::Uint32(0)),
                    (key("block_count"), MetadataValue::Uint32(2)),
                    (key("embedding_length"), MetadataValue::Uint32(32)),
                    (key("attention.head_count"), MetadataValue::Uint32(4)),
                    (key("attention.head_count_kv"), MetadataValue::Uint32(2)),
                    (key("attention.key_length"), MetadataValue::Uint32(8)),
                    (key("rope.dimension_count"), MetadataValue::Uint32(2)),
                    (key("full_attention_interval"), MetadataValue::Uint32(2)),
                    (key("vocab_size"), MetadataValue::Uint32(64)),
                    (key("context_length"), MetadataValue::Uint32(128)),
                    (
                        key("attention.layer_norm_rms_epsilon"),
                        MetadataValue::Float32(args.rms_norm_eps),
                    ),
                    (key("feed_forward_length"), MetadataValue::Uint32(48)),
                    (key("ssm.conv_kernel"), MetadataValue::Uint32(4)),
                    (key("ssm.state_size"), MetadataValue::Uint32(8)),
                    (key("ssm.group_count"), MetadataValue::Uint32(2)),
                    (key("ssm.time_step_rank"), MetadataValue::Uint32(4)),
                ]);
                (plan, metadata)
            }
            _ => unreachable!("heterogeneous GGUF fixture family"),
        };
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| {
                    if family == "qwen3next"
                        && group.variants.iter().any(|variant| variant.id == "fused")
                    {
                        group.variants.iter().find(|variant| variant.id == "fused")
                    } else {
                        group.variants.first()
                    }
                })
                .flat_map(|variant| variant.tensors.iter()),
        );
        let arrays = constraints
            .into_iter()
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let shape = constraint
                    .shape
                    .iter()
                    .map(|dimension| i32::try_from(*dimension).unwrap())
                    .collect::<Vec<_>>();
                let array = if constraint.key.ends_with("ssm_a") {
                    Array::full::<f32>(&shape, Array::from_f32(-1.0), stream).unwrap()
                } else {
                    let seed = constraint
                        .key
                        .bytes()
                        .fold(0_usize, |sum, byte| sum.wrapping_add(usize::from(byte)));
                    let values = (0..shape.iter().map(|dimension| *dimension as usize).product())
                        .map(|index| ((index + seed) % 29 + 1) as f32 / 100.0)
                        .collect::<Vec<_>>();
                    Array::from_slice(&values, &shape)
                };
                (constraint.key.clone(), array)
            })
            .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&arrays, &metadata, |name, _| {
            packed_qkvz.filter(|_| name.contains("attn_qkvz.weight"))
        })
    }

    fn lfm2_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "lfm2", "vocab_size": 64, "hidden_size": 16,
            "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "layer_types": ["conv", "full_attention"], "conv_L_cache": 3,
            "block_multiple_of": 8, "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true, "tie_word_embeddings": false
        })
    }

    fn kimi_linear_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"kimi_linear","vocab_size":64,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":16,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":8,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":2,"num_expert_group":1,"topk_group":1
        })
    }

    fn nemotron_h_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":64, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-M", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
            "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
            "num_experts_per_tok":2, "n_group":2, "topk_group":1,
            "num_nextn_predict_layers":0
        })
    }

    fn qwen_hybrid_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_5_text", "vocab_size": 64, "hidden_size": 32,
            "num_hidden_layers": 2, "mtp_num_hidden_layers": 0,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128, "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "intermediate_size": 48, "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24, "num_experts_per_tok": 0,
            "num_experts": 0, "layer_types": ["linear_attention", "full_attention"]
        })
    }

    fn qwen_next_config() -> serde_json::Value {
        let mut config = qwen_hybrid_config();
        config["model_type"] = "qwen3_next".into();
        config
    }

    fn execution_streams() -> (Stream, Stream) {
        let execution_device = if cfg!(feature = "metal") {
            safemlx::Device::new(safemlx::DeviceType::Gpu, 0)
        } else {
            safemlx::Device::new(safemlx::DeviceType::Cpu, 0)
        };
        let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        (
            Stream::new_with_device(&execution_device),
            Stream::new_with_device(&weights_device),
        )
    }

    fn complete_state_capabilities(
        components: impl IntoIterator<Item = StateComponentMechanism>,
    ) -> StateMechanismCapabilities {
        StateMechanismCapabilities::new(components)
            .with_transactions(true, true)
            .with_reset(true)
            .with_prompt_cache(true)
            .with_observation_retention(true)
    }

    fn capabilities_with(
        full: &BackendMechanismCapabilities,
        operators: eredu_nn::NeuralOperatorCapabilities,
        state: StateMechanismCapabilities,
    ) -> BackendMechanismCapabilities {
        BackendMechanismCapabilities::new(
            operators,
            full.weight_lowerings().to_vec(),
            full.weight_residencies().to_vec(),
            state,
        )
        .with_session(full.session())
        .with_prompt_cache(full.prompt_cache())
        .with_exact_completion(full.exact_completion())
        .with_grouped_operations(full.grouped_operations().iter().copied())
    }

    fn tiny_llama_gguf(
        architecture: &str,
        packed: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::MetadataValue;

        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(1)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(4)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("feed_forward_length"), MetadataValue::Uint32(64)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-5),
            ),
            (key("vocab_size"), MetadataValue::Uint32(64)),
            (key("context_length"), MetadataValue::Uint32(32)),
            (key("rope.freq_base"), MetadataValue::Float32(10_000.0)),
        ]);
        let tensors = [
            ("token_embd.weight", vec![64, 32]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![8, 32]),
            ("blk.0.attn_v.weight", vec![8, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_gate.weight", vec![64, 32]),
            ("blk.0.ffn_up.weight", vec![64, 32]),
            ("blk.0.ffn_down.weight", vec![32, 64]),
        ]
        .into_iter()
        .map(|(name, shape)| {
            (
                name.to_string(),
                Array::zeros::<f32>(&shape, stream).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&tensors, &metadata, |name, array| {
            packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2)
        })
    }

    fn tiny_qwen_gguf(
        architecture: &str,
        packed: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::MetadataValue;

        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(1)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(4)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("feed_forward_length"), MetadataValue::Uint32(64)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-5),
            ),
            (key("vocab_size"), MetadataValue::Uint32(64)),
            (key("context_length"), MetadataValue::Uint32(32)),
            (key("rope.freq_base"), MetadataValue::Float32(1_000_000.0)),
        ]);
        let mut tensors = vec![
            ("token_embd.weight", vec![64, 32]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![8, 32]),
            ("blk.0.attn_v.weight", vec![8, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_gate.weight", vec![64, 32]),
            ("blk.0.ffn_up.weight", vec![64, 32]),
            ("blk.0.ffn_down.weight", vec![32, 64]),
        ];
        if architecture == "qwen2" {
            tensors.extend([
                ("blk.0.attn_q.bias", vec![32]),
                ("blk.0.attn_k.bias", vec![8]),
                ("blk.0.attn_v.bias", vec![8]),
            ]);
        } else {
            tensors.extend([
                ("blk.0.attn_q_norm.weight", vec![8]),
                ("blk.0.attn_k_norm.weight", vec![8]),
            ]);
        }
        let tensors = tensors
            .into_iter()
            .map(|(name, shape)| {
                (
                    name.to_string(),
                    Array::zeros::<f32>(&shape, stream).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&tensors, &metadata, |name, array| {
            packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2)
        })
    }

    #[test]
    fn gguf_requirements_retain_shard_and_multi_output_provenance() {
        let (stream, _) = execution_streams();
        let gguf = tiny_llama_gguf("llama", Some(eredu_gguf::GgmlType::MxFp4), &stream);
        let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
        let shard = inspection.gguf_checkpoint().unwrap().shards()[0]
            .path()
            .to_path_buf();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let output = requirements
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == "lm_head.weight")
            .expect("tied GGUF output remains explicit in the logical topology");
        assert!(matches!(
            output.presence(),
            eredu_runtime::ReplicatedTextParameterPresence::Tied { target }
                if target == "model.embed_tokens.weight"
        ));
        let derived = requirements
            .parameters()
            .iter()
            .find(|parameter| {
                matches!(
                    parameter.presence(),
                    eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
                ) && parameter
                    .physical_sources()
                    .iter()
                    .any(|source| source.output().ends_with(".scales"))
            })
            .expect("MXFP4 requirements include a derived scales output");
        let source = &derived.physical_sources()[0];
        assert_eq!(source.shard(), shard);
        assert!(source.tensor().ends_with(".weight"));
        assert!(source.output().ends_with(".scales"));
        let direct = requirements
            .parameters()
            .iter()
            .find(|parameter| {
                parameter
                    .physical_sources()
                    .iter()
                    .any(|candidate| candidate.tensor() == source.tensor())
                    && parameter.presence().has_physical_source()
            })
            .expect("the same MXFP4 tensor includes its direct weight output");
        assert_eq!(direct.physical_sources()[0].shard(), source.shard());
        assert_ne!(direct.physical_sources()[0].output(), source.output());
    }

    #[test]
    fn public_handoff_executes_llama_and_dense_qwen_with_repeated_decode() {
        super::super::path_instrumentation::reset();
        let (stream, weights_stream) = execution_streams();
        for (model_type, tied) in [
            ("llama", true),
            ("mistral", false),
            ("qwen2", false),
            ("qwen3", true),
        ] {
            let root = tiny_artifact(model_type, tied);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let policy = eredu_core::PreparationPolicy::default();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary replicated text must use the generic executable")
            };
            for token in [1_u32, 2] {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap();
                assert_eq!(logits.shape(), &[1, 64]);
                logits.evaluated().unwrap();
            }
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts {
                architecture_constructions: 4,
                state_allocations: 4,
                payload_opens: 4,
                forwards: 8,
            }
        );
    }

    #[test]
    fn generic_handoff_executes_every_replicated_state_profile() {
        super::super::path_instrumentation::reset();
        let (stream, weights_stream) = execution_streams();
        for (name, config) in [
            ("lfm2", lfm2_config()),
            ("kimi_linear", kimi_linear_config()),
            ("nemotron_h", nemotron_h_config()),
            ("qwen3_next", qwen_next_config()),
            ("qwen3_5_text", qwen_hybrid_config()),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap_or_else(|error| panic!("{name} requirements: {error}"));
            assert!(requirements
                .state_layout()
                .layers()
                .iter()
                .any(|layer| !layer.fixed_state().is_empty()));
            let stateful_layers = requirements
                .state_layout()
                .layers()
                .iter()
                .map(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty())
                .collect::<Vec<_>>();
            let policy = eredu_core::PreparationPolicy::default();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("{name} did not use generic replicated composition")
            };
            let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            let logits = executable
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap_or_else(|error| panic!("{name} prefill: {error}"));
            assert_eq!(logits.shape(), &[1, 64], "{name}");
            logits.evaluated().unwrap();
            let snapshot = executable.state_snapshot();
            assert!(
                snapshot
                    .iter()
                    .zip(&stateful_layers)
                    .all(|((position, _), stateful)| *position == if *stateful { 2 } else { 0 }),
                "{name} prefill: {snapshot:?}"
            );
            for (step, token) in [3_u32, 4].into_iter().enumerate() {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap_or_else(|error| panic!("{name} decode {step}: {error}"));
                assert_eq!(logits.shape(), &[1, 64], "{name}");
                logits.evaluated().unwrap();
                let snapshot = executable.state_snapshot();
                assert!(
                    snapshot
                        .iter()
                        .zip(&stateful_layers)
                        .all(|((position, _), stateful)| *position
                            == if *stateful { step as i32 + 3 } else { 0 }),
                    "{name} step {step}: {snapshot:?}"
                );
                assert!(snapshot
                    .iter()
                    .flat_map(|(_, fixed)| fixed)
                    .all(|(_, present)| *present));
            }
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts {
                architecture_constructions: 5,
                state_allocations: 5,
                payload_opens: 5,
                forwards: 15,
            }
        );
    }

    #[test]
    fn homogeneous_state_schedules_execute_with_only_their_exact_mechanisms() {
        let (stream, weights_stream) = execution_streams();
        let mut cases = Vec::new();
        let mut lfm = lfm2_config();
        lfm["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
        cases.push((
            "lfm_attention",
            lfm,
            ReplicatedTextStateAccess::KeyValue,
            None,
        ));
        let mut lfm = lfm2_config();
        lfm["layer_types"] = serde_json::json!(["conv", "conv"]);
        cases.push(("lfm_fixed", lfm, ReplicatedTextStateAccess::Fixed, None));

        let mut kimi = kimi_linear_config();
        kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([1, 2]);
        kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([]);
        cases.push((
            "kimi_kda",
            kimi,
            ReplicatedTextStateAccess::Fixed,
            Some(eredu_nn::NeuralOperatorCapabilities::GATED_DELTA_SCAN),
        ));
        let mut kimi = kimi_linear_config();
        kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([]);
        kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([1, 2]);
        cases.push((
            "kimi_mla",
            kimi,
            ReplicatedTextStateAccess::CompressedAttention,
            None,
        ));

        for (name, pattern, access, operator) in [
            (
                "nemotron_attention",
                "****",
                ReplicatedTextStateAccess::KeyValue,
                None,
            ),
            (
                "nemotron_mamba",
                "MMMM",
                ReplicatedTextStateAccess::Fixed,
                Some(eredu_nn::NeuralOperatorCapabilities::SELECTIVE_STATE_SPACE_SCAN),
            ),
            (
                "nemotron_stateless",
                "----",
                ReplicatedTextStateAccess::Stateless,
                None,
            ),
        ] {
            let mut nemo = nemotron_h_config();
            nemo["hybrid_override_pattern"] = pattern.into();
            cases.push((name, nemo, access, operator));
        }
        let mut qwen = qwen_hybrid_config();
        qwen["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
        cases.push((
            "qwen_attention",
            qwen,
            ReplicatedTextStateAccess::KeyValue,
            None,
        ));
        let mut qwen = qwen_hybrid_config();
        qwen["layer_types"] = serde_json::json!(["linear_attention", "linear_attention"]);
        cases.push((
            "qwen_fixed",
            qwen,
            ReplicatedTextStateAccess::Fixed,
            Some(eredu_nn::NeuralOperatorCapabilities::GATED_DELTA_SCAN),
        ));

        for (name, config, access, operator) in cases {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap_or_else(|error| panic!("{name} requirements: {error}"));
            assert_eq!(requirements.state_access(), access, "{name}");
            if let Some(operator) = operator {
                assert!(requirements.operators().contains(operator), "{name}");
            } else {
                assert_eq!(
                    requirements.operators(),
                    eredu_nn::NeuralOperatorCapabilities::NONE,
                    "{name}"
                );
            }
            let stateful = requirements
                .state_layout()
                .layers()
                .iter()
                .map(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty())
                .collect::<Vec<_>>();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
                panic!("{name} did not use replicated composition")
            };
            let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            generic
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let logits = generic
                .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            assert!(
                logits.iter().all(|value| value.is_finite())
                    && logits.iter().any(|value| value.abs() > 1e-12),
                "{name}: {logits:?}"
            );
            assert!(generic
                .state_snapshot()
                .iter()
                .zip(&stateful)
                .all(|((position, _), stateful)| *position == if *stateful { 3 } else { 0 }));
        }
    }

    #[test]
    fn heterogeneous_gguf_artifacts_use_the_same_generic_state_contract() {
        super::super::path_instrumentation::reset();
        let (stream, weights_stream) = execution_streams();
        for (name, gguf_name, config) in [
            ("lfm2", "lfm2", lfm2_config()),
            ("kimi_linear", "kimi_linear", kimi_linear_config()),
            ("nemotron_h", "nemotron_h", nemotron_h_config()),
            ("qwen3_next", "qwen3next", qwen_next_config()),
            ("qwen3_5_text", "qwen35", qwen_hybrid_config()),
        ] {
            let gguf = tiny_heterogeneous_gguf(gguf_name, &stream);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap_or_else(|error| panic!("{name} GGUF requirements: {error}"));
            let stateful = (0..requirements.state_layout().len())
                .map(|layer| {
                    !requirements
                        .state_layout()
                        .components(layer)
                        .unwrap()
                        .is_empty()
                })
                .collect::<Vec<_>>();
            let safe = tiny_heterogeneous_artifact(config);
            let safe_inspection =
                eredu_architectures::configuration::inspect_artifact(safe.path()).unwrap();
            let safe_requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(
                    &safe_inspection,
                )
                .unwrap();
            assert_eq!(
                requirements.state_access(),
                safe_requirements.state_access()
            );
            assert_eq!(
                requirements.state_layout(),
                safe_requirements.state_layout()
            );
            assert_eq!(requirements.operators(), safe_requirements.operators());
            assert_eq!(
                requirements.execution_graph(),
                safe_requirements.execution_graph()
            );

            let execute = |token| {
                let fresh =
                    eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
                let plan = eredu_core::plan_model_preparation(
                    fresh,
                    eredu_core::PreparationPolicy::default(),
                    eredu_core::SessionCapabilities::default(),
                )
                .unwrap();
                let model = materialize_model_plan(
                    plan,
                    crate::MlxLoadRequest::default(),
                    &stream,
                    &weights_stream,
                )
                .unwrap_or_else(|error| panic!("{name} GGUF: {error}"));
                let mut executable = model.into_complete().unwrap();
                let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
                    panic!("{name} GGUF did not use generic replicated composition")
                };
                let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
                let parts = [input::token_ids_part(&prompt).unwrap()];
                generic
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap();
                let fixed_before = generic.fixed_numeric_state_snapshot().unwrap();
                let logits = generic
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec();
                let fixed_after = generic.fixed_numeric_state_snapshot().unwrap();
                (logits, generic.state_snapshot(), fixed_before, fixed_after)
            };
            let token_three = execute(3_u32);
            let token_four = execute(4_u32);
            for (logits, snapshot, fixed_before, fixed_after) in [&token_three, &token_four] {
                assert!(
                    logits.iter().all(|value| value.is_finite())
                        && logits.iter().any(|value| value.abs() > 1e-12),
                    "{name} GGUF produced invalid logits: {logits:?}"
                );
                assert!(snapshot
                    .iter()
                    .zip(&stateful)
                    .all(|((position, fixed), stateful)| {
                        *position == if *stateful { 3 } else { 0 }
                            && fixed.iter().all(|(_, present)| *present)
                    }));
                assert_ne!(
                    fixed_before, fixed_after,
                    "{name} fixed state did not consume decode input"
                );
            }
            assert_ne!(
                token_three.0, token_four.0,
                "{name} ignored token identity at an identical state frontier"
            );
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts {
                architecture_constructions: 10,
                state_allocations: 10,
                payload_opens: 10,
                forwards: 20,
            }
        );
    }

    #[test]
    fn packed_fused_qwen_next_gguf_format_reaches_split_projection_execution() {
        let (stream, weights_stream) = execution_streams();
        let gguf = tiny_heterogeneous_gguf_with_packed_qwen_next(
            "qwen3next",
            Some(eredu_gguf::GgmlType::MxFp4),
            &stream,
        );
        let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let fused_targets = requirements
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.name().contains("linear_attn.in_proj_qkv.weight")
                    || parameter.name().contains("linear_attn.in_proj_z.weight")
            })
            .collect::<Vec<_>>();
        assert_eq!(fused_targets.len(), 2);
        assert!(fused_targets.iter().all(|parameter| {
            matches!(
                parameter.presence(),
                eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
            ) && parameter.has_lowering_source()
                && parameter.native_executable() == eredu_checkpoint::LinearFormat::MxFp4
        }));
        let selection_request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        );
        let selected = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &selection_request,
            &capabilities(&requirements, &selection_request),
        )
        .unwrap();
        assert!(selected
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.name().contains("linear_attn.in_proj_qkv.weight")
                    || parameter.name().contains("linear_attn.in_proj_z.weight")
            })
            .all(|parameter| parameter.lowering() == eredu_runtime::WeightLoweringKind::Derived));

        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        let mut executable = model.into_complete().unwrap();
        let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
            panic!("packed Qwen3-Next did not use generic replicated composition")
        };
        let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let logits = generic
            .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(
            logits.iter().all(|value| value.is_finite())
                && logits.iter().any(|value| value.abs() > 1e-12)
        );
        assert!(generic
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 3));
    }

    #[test]
    fn fused_qwen_next_safetensors_preserves_both_source_families_into_execution() {
        let (stream, weights_stream) = execution_streams();
        let root = tiny_heterogeneous_artifact_with_layout(qwen_next_config(), true);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let fused = requirements
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.name().contains("linear_attn.in_proj_")
                    && matches!(
                        parameter.presence(),
                        eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
                    )
            })
            .collect::<Vec<_>>();
        assert_eq!(fused.len(), 4);
        assert!(fused.iter().all(|parameter| {
            parameter.has_lowering_source()
                && matches!(
                    parameter.source_encoding(),
                    Some(eredu_checkpoint::SourceTensorEncoding::Safetensors(
                        eredu_checkpoint::StoredDtype::F32
                    ))
                )
                && parameter.physical_shape().is_some()
                && parameter.physical_sources().len() == 1
        }));
        assert_eq!(
            fused
                .iter()
                .filter(|parameter| parameter.physical_sources()[0].tensor().contains("qkvz"))
                .count(),
            2
        );
        assert_eq!(
            fused
                .iter()
                .filter(|parameter| parameter.physical_sources()[0].tensor().contains("ba"))
                .count(),
            2
        );
        assert!(fused
            .iter()
            .filter(|parameter| parameter.physical_sources()[0].tensor().contains("ba"))
            .all(
                |parameter| parameter.native_executable() == eredu_checkpoint::LinearFormat::Dense
            ));

        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        let mut executable = model.into_complete().unwrap();
        let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
            panic!("fused SafeTensors Qwen3-Next did not use replicated composition")
        };
        let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let logits = generic
            .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(
            logits.iter().all(|value| value.is_finite())
                && logits.iter().any(|value| value.abs() > 1e-12)
        );
        assert!(generic
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 3));
    }

    #[test]
    fn admitted_gguf_media_projector_keeps_qwen_hybrid_out_of_text_binding() {
        let (stream, _) = execution_streams();
        let gguf = tiny_heterogeneous_gguf("qwen35", &stream);
        let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
        let plan = inspection.architecture_plan().gguf_plan().unwrap();
        assert!(super::super::loading::gguf_uses_replicated_text_binding(
            plan, false
        ));
        assert!(!super::super::loading::gguf_uses_replicated_text_binding(
            plan, true
        ));
    }

    #[test]
    fn both_text_architectures_retain_exact_sharded_admission_with_one_cached_payload() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["mistral", "qwen3"] {
            let root = tiny_sharded_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let streaming =
                eredu_runtime::DenseDiskStreamLoadOptions::default().with_max_cached_shards(1);
            let options = crate::MlxLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::dense_disk_stream(streaming),
            );
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            std::fs::remove_file(root.path().join("model.safetensors.index.json")).unwrap();

            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary sharded text must use the generic executable")
            };
            executable
                .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let report = executable.dense_stream_report().unwrap().unwrap();
            assert!(report.residency().weight_store().currently_cached_shards <= 1);
            assert!(!report
                .residency()
                .weight_store()
                .payload_shard_paths
                .is_empty());
        }
    }

    #[test]
    fn unsupported_topology_fails_before_checkpoint_payload_or_module_construction() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("llama", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let topology = crate::composition::mlx::distributed::topology::MlxParallelPlan::for_rank(
            0,
            2,
            1,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let request = request.with_topology(topology.topology());
        std::fs::remove_file(root.path().join("model.safetensors")).unwrap();

        let error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request,
            &capabilities(&requirements, &request),
        )
        .expect_err("unsupported topology was admitted");
        let message = error.to_string();
        assert!(
            message.contains("replicated execution topology"),
            "{message}"
        );
        assert!(!message.contains("No such file"), "{message}");
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn heterogeneous_state_and_operator_gaps_reject_before_any_production_path() {
        use eredu_core::cache::{
            LayerCachePolicy, StateComponentRole, StateTensorDtype, StateTensorPolicy,
        };

        super::super::path_instrumentation::reset();
        let paged = CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(4, 4096, 4096, 1)
                .unwrap()
                .with_full_attention(true),
        );
        for (name, config) in [
            ("lfm2", lfm2_config()),
            ("kimi_linear", kimi_linear_config()),
            ("nemotron_h", nemotron_h_config()),
            ("qwen3_next", qwen_next_config()),
            ("qwen3_5_text", qwen_hybrid_config()),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                CacheResidencyPolicy::Device,
            );
            let full = capabilities(&requirements, &request);

            let fixed_components = full
                .state()
                .components()
                .iter()
                .filter(|mechanism| {
                    matches!(mechanism.component().role(), StateComponentRole::Fixed(_))
                })
                .cloned()
                .collect::<Vec<_>>();
            assert!(
                !fixed_components.is_empty(),
                "{name} fixture has no fixed state"
            );
            for fixed in &fixed_components {
                let missing_fixed = complete_state_capabilities(
                    full.state()
                        .components()
                        .iter()
                        .filter(|mechanism| *mechanism != fixed)
                        .cloned(),
                );
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(&full, full.operators(), missing_fixed),
                )
                .expect_err("missing fixed state was admitted");
                assert!(
                    error.issues().iter().any(|issue| {
                        issue.contains(&fixed.component().role().stable_name())
                            && issue.contains("state component")
                    }),
                    "{name}: {error}"
                );
            }
            if name == "kimi_linear" {
                let without_compressed = complete_state_capabilities(
                    full.state()
                        .components()
                        .iter()
                        .filter(|mechanism| {
                            mechanism.component().role() != StateComponentRole::CompressedLatent
                        })
                        .cloned(),
                );
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(&full, full.operators(), without_compressed),
                )
                .expect_err("missing compressed attention was admitted");
                assert!(error
                    .issues()
                    .iter()
                    .any(|issue| issue.contains("attention.compressed_latent")));
            }

            for fixed in &fixed_components {
                let StateComponentRole::Fixed(role) = fixed.component().role() else {
                    unreachable!("fixed component filter changed")
                };
                let wrong_shape =
                    LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new_with_residency(
                        role,
                        vec![eredu_core::cache::StateTensorDimension::fixed(999).unwrap()],
                        fixed.component().dtype(),
                        fixed.component().residency(),
                    )
                    .unwrap()])
                    .unwrap()
                    .components()
                    .pop()
                    .unwrap();
                let components = full.state().components().iter().map(|mechanism| {
                    if mechanism == fixed {
                        StateComponentMechanism::new(
                            mechanism.layer(),
                            wrong_shape.clone(),
                            Some(StateComponentPlacement::Device),
                            Some(StateComponentPlacement::Device),
                        )
                    } else {
                        mechanism.clone()
                    }
                });
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(
                        &full,
                        full.operators(),
                        complete_state_capabilities(components),
                    ),
                )
                .expect_err("wrong fixed-state shape was admitted");
                assert!(error
                    .issues()
                    .iter()
                    .any(|issue| issue.contains("shape") && issue.contains("dtype")));

                let alternate_dtype = match fixed.component().dtype() {
                    StateTensorDtype::Float32 => StateTensorDtype::Floating,
                    _ => StateTensorDtype::Float32,
                };
                let wrong_dtype =
                    LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new_with_residency(
                        role,
                        fixed.component().shape().to_vec(),
                        alternate_dtype,
                        fixed.component().residency(),
                    )
                    .unwrap()])
                    .unwrap()
                    .components()
                    .pop()
                    .unwrap();
                let components = full.state().components().iter().map(|mechanism| {
                    if mechanism == fixed {
                        StateComponentMechanism::new(
                            mechanism.layer(),
                            wrong_dtype.clone(),
                            Some(StateComponentPlacement::Device),
                            Some(StateComponentPlacement::Device),
                        )
                    } else {
                        mechanism.clone()
                    }
                });
                assert!(
                    eredu_runtime::select_replicated_text_realization(
                        &requirements,
                        &request,
                        &capabilities_with(
                            &full,
                            full.operators(),
                            complete_state_capabilities(components),
                        ),
                    )
                    .is_err(),
                    "{name} admitted an incompatible fixed-state dtype"
                );
            }

            let paged_components = full.state().components().iter().map(|mechanism| {
                StateComponentMechanism::new(
                    mechanism.layer(),
                    mechanism.component().clone(),
                    Some(StateComponentPlacement::Device),
                    Some(StateComponentPlacement::Paged),
                )
            });
            let paged_request = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                paged.clone(),
            )
            .with_prompt_cache(true);
            assert!(
                eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &paged_request,
                    &capabilities_with(
                        &full,
                        full.operators(),
                        complete_state_capabilities(paged_components),
                    ),
                )
                .is_err(),
                "{name} admitted incompatible paged fixed-component placement"
            );

            if requirements.operators() != eredu_nn::NeuralOperatorCapabilities::NONE {
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(
                        &full,
                        eredu_nn::NeuralOperatorCapabilities::NONE,
                        full.state().clone(),
                    ),
                )
                .expect_err("missing semantic neural operations were admitted");
                let operation = match name {
                    "nemotron_h" => "selective_state_space_scan",
                    "kimi_linear" | "qwen3_next" | "qwen3_5_text" => "gated_delta_scan",
                    _ => unreachable!(),
                };
                assert!(
                    error.issues().iter().any(|issue| issue.contains(operation)),
                    "{name}: {error}"
                );
            }

            let paged_full = capabilities(&requirements, &paged_request);
            for (facility, state) in [
                (
                    "checkpoint",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(false, true)
                    .with_reset(true)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
                ),
                (
                    "rollback",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(true, false)
                    .with_reset(true)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
                ),
                (
                    "reset",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(true, true)
                    .with_reset(false)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
                ),
                (
                    "prompt-cache",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(true, true)
                    .with_reset(true)
                    .with_prompt_cache(false)
                    .with_observation_retention(true),
                ),
            ] {
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &paged_request,
                    &capabilities_with(&paged_full, paged_full.operators(), state),
                )
                .unwrap_err();
                assert!(
                    error.issues().iter().any(|issue| issue.contains(facility)),
                    "{name}: missing {facility} diagnostic: {error}"
                );
            }

            std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn routed_prediction_and_media_graphs_are_ineligible_without_production_work() {
        use eredu_architectures::replicated_text::ReplicatedTextIneligibility;

        super::super::path_instrumentation::reset();
        let mut routed = qwen_hybrid_config();
        routed["model_type"] = "qwen3_next".into();
        routed["num_experts"] = 2.into();
        routed["num_experts_per_tok"] = 1.into();

        let mut nemotron_prediction = nemotron_h_config();
        nemotron_prediction["num_nextn_predict_layers"] = 1.into();
        nemotron_prediction["mtp_hybrid_override_pattern"] = "*E".into();

        let mut qwen_prediction = qwen_hybrid_config();
        qwen_prediction["mtp_num_hidden_layers"] = 1.into();

        let text = qwen_hybrid_config();
        let media = serde_json::json!({
            "model_type": "qwen3_5",
            "image_token_id": 60,
            "video_token_id": 61,
            "text_config": text,
            "vision_config": {
                "depth": 1, "hidden_size": 8, "intermediate_size": 16,
                "num_heads": 2, "num_position_embeddings": 16,
                "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
                "temporal_patch_size": 2, "out_hidden_size": 32
            }
        });

        for (name, config, expected) in [
            ("routed", routed, ReplicatedTextIneligibility::Routed),
            (
                "nemotron prediction",
                nemotron_prediction,
                ReplicatedTextIneligibility::EmbeddedPrediction,
            ),
            (
                "qwen prediction",
                qwen_prediction,
                ReplicatedTextIneligibility::EmbeddedPrediction,
            ),
            ("media", media, ReplicatedTextIneligibility::CompositeInput),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
            let error =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .expect_err("excluded graph entered replicated text admission");
            assert!(
                matches!(
                    error,
                    eredu_architectures::replicated_text::ReplicatedTextRequirementsError::Ineligible(actual)
                        if actual == expected
                ),
                "{name}: {error}"
            );
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn invalid_source_and_missing_grouped_mechanism_never_reach_production_paths() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("llama", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );

        let first = &requirements.parameters()[0];
        let invalid = ReplicatedTextParameterRequirement::new(
            first.name(),
            first.sources().to_vec(),
            first.physical_sources().to_vec(),
            first.aliases().to_vec(),
            Some(SourceTensorEncoding::Safetensors(StoredDtype::U8)),
            first.physical_shape().map(<[usize]>::to_vec),
            first.logical_shape().to_vec(),
            first.native_executable(),
            first.role(),
            first.owner().clone(),
            first.presence().clone(),
            first.transform_constraint(),
        )
        .unwrap();
        let mut parameters = requirements.parameters().to_vec();
        parameters[0] = invalid;
        let invalid_requirements = ReplicatedTextRequirements::new(
            requirements.architecture_identity().to_owned(),
            requirements.operators(),
            requirements.execution_graph().clone(),
            requirements.execution_units().clone(),
            requirements.group_transports().to_vec(),
            requirements.state_layout().clone(),
            requirements.state_access(),
            parameters,
        )
        .unwrap();
        let error = eredu_runtime::select_replicated_text_realization(
            &invalid_requirements,
            &request,
            &capabilities(&invalid_requirements, &request),
        )
        .unwrap_err();
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("weight lowering")));

        let routed_root = tiny_artifact("qwen3_moe", false);
        let routed =
            eredu_architectures::configuration::inspect_artifact(routed_root.path()).unwrap();
        let topology = crate::composition::mlx::distributed::topology::MlxParallelPlan::for_rank(
            0,
            2,
            1,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let routed_options = crate::MlxLoadRequest::with_parallel(
            topology,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
        );
        let routed_policy = routed_options.preparation_policy().unwrap();
        let error = super::super::loading::select_preparation_with_grouped_capabilities(
            &routed,
            routed_options,
            routed_policy,
            &[GroupedOperationRequirement::GatedProduct],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("GatedProductTensorParallelPartial"));

        let affine_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request
                .clone()
                .with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 128,
                    bits: 4,
                }),
            &capabilities(&requirements, &request),
        )
        .unwrap_err();
        assert!(affine_error
            .issues()
            .iter()
            .any(|issue| issue.contains("affine group size")));

        let linear_index = requirements
            .parameters()
            .iter()
            .position(|parameter| {
                matches!(
                    parameter.transform_constraint(),
                    ParameterTransformConstraint::Linear { .. }
                )
            })
            .unwrap();
        let linear = &requirements.parameters()[linear_index];
        let ParameterTransformConstraint::Linear { packed_axis } = linear.transform_constraint()
        else {
            unreachable!()
        };
        let mut logical_shape = linear.logical_shape().to_vec();
        logical_shape[packed_axis] = 48;
        let invalid_mxfp4 = ReplicatedTextParameterRequirement::new(
            linear.name(),
            linear.sources().to_vec(),
            linear.physical_sources().to_vec(),
            linear.aliases().to_vec(),
            linear.source_encoding().cloned(),
            Some(logical_shape.clone()),
            logical_shape,
            linear.native_executable(),
            linear.role(),
            linear.owner().clone(),
            linear.presence().clone(),
            linear.transform_constraint(),
        )
        .unwrap();
        let mut parameters = requirements.parameters().to_vec();
        parameters[linear_index] = invalid_mxfp4;
        let invalid_mxfp4_requirements = ReplicatedTextRequirements::new(
            requirements.architecture_identity().to_owned(),
            requirements.operators(),
            requirements.execution_graph().clone(),
            requirements.execution_units().clone(),
            requirements.group_transports().to_vec(),
            requirements.state_layout().clone(),
            requirements.state_access(),
            parameters,
        )
        .unwrap();
        let mxfp4_request = request
            .clone()
            .with_quantization(eredu_core::QuantizationRequest::MxFp4);
        let mxfp4_error = eredu_runtime::select_replicated_text_realization(
            &invalid_mxfp4_requirements,
            &mxfp4_request,
            &capabilities(&invalid_mxfp4_requirements, &mxfp4_request),
        )
        .unwrap_err();
        assert!(mxfp4_error
            .issues()
            .iter()
            .any(|issue| issue.contains("MXFP4 packed extent 48")));

        let full = capabilities(&requirements, &request);
        let only_basic = BackendMechanismCapabilities::new(
            full.operators(),
            full.weight_lowerings().to_vec(),
            vec![WeightResidencyMechanism::Resident],
            StateMechanismCapabilities::new(full.state().components().iter().map(|mechanism| {
                StateComponentMechanism::new(
                    mechanism.layer(),
                    mechanism.component().clone(),
                    Some(StateComponentPlacement::Device),
                    None,
                )
            })),
        );
        let paged = CacheResidencyPolicy::Paged(PagedCacheOptions::new(4, 4096, 4096, 1).unwrap());
        let state_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                paged,
            ),
            &only_basic,
        )
        .unwrap_err();
        assert!(state_error
            .issues()
            .iter()
            .any(|issue| issue.contains("state component")));
        let session_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request
                .clone()
                .with_session(eredu_core::SessionCapabilities::new(true, false, false)),
            &only_basic,
        )
        .unwrap_err();
        assert!(session_error
            .issues()
            .iter()
            .any(|issue| issue.contains("session capability")));
        let residency_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::DenseDiskStream(
                    eredu_runtime::DenseDiskStreamLoadOptions::default(),
                ),
                CacheResidencyPolicy::Device,
            ),
            &only_basic,
        )
        .unwrap_err();
        assert!(residency_error
            .issues()
            .iter()
            .any(|issue| issue.contains("weight residency")));
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn selected_paged_state_controls_generic_construction() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["llama", "qwen3"] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let policy = eredu_core::PreparationPolicy::default();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            let state = CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            );
            let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                state.clone(),
            );
            let selected = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &request,
                &capabilities(&requirements, &request),
            )
            .unwrap();
            assert_eq!(selected.state().policy(), &state);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let architecture_plan = plan.inspection().architecture_plan().clone();
            let artifact = plan.into_artifact();
            let eredu_core::ModelArtifact::SafeTensors {
                configuration,
                tensors,
                shards,
                ..
            } = artifact
            else {
                panic!("expected SafeTensors fixture")
            };
            let prepared = super::super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                super::super::loading::prepared_safetensors_architecture(&architecture_plan)
                    .unwrap()
                    .clone(),
                tensors,
                shards,
                1,
            )
            .unwrap();
            let executable =
                eredu_architectures::replicated_text::visit_replicated_text_architecture::<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                    _,
                >(
                    &architecture_plan,
                    selected,
                    prepared.store(),
                    &stream,
                    BindingVisitor {
                        stream: &stream,
                        weights_stream: &weights_stream,
                    },
                )
                .unwrap();
            assert!(executable.cache_residency_report().unwrap().is_some());
        }
    }

    #[test]
    fn heterogeneous_requirements_are_invariant_across_caller_policies() {
        for config in [
            lfm2_config(),
            kimi_linear_config(),
            nemotron_h_config(),
            qwen_next_config(),
            qwen_hybrid_config(),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let expected =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            let requests = [
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::FullyResident,
                    CacheResidencyPolicy::Device,
                ),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(
                        eredu_runtime::LayerwiseLoadOptions::default(),
                    ),
                    CacheResidencyPolicy::Paged(
                        PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                            .unwrap()
                            .with_full_attention(true),
                    ),
                )
                .with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 16,
                    bits: 4,
                })
                .with_session(eredu_core::SessionCapabilities::new(true, true, true))
                .with_prompt_cache(true)
                .with_exact_completion(true),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::DenseDiskStream(
                        eredu_runtime::DenseDiskStreamLoadOptions::default(),
                    ),
                    CacheResidencyPolicy::Device,
                )
                .with_quantization(eredu_core::QuantizationRequest::MxFp4),
            ];
            for request in requests {
                assert!(matches!(
                    request.residency(),
                    eredu_runtime::LayerWeightResidency::FullyResident
                        | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                        | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                ));
                assert_eq!(
                    expected,
                    eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn included_safetensors_configs_cannot_enter_family_materialization() {
        let (stream, weights_stream) = execution_streams();
        for config in [
            lfm2_config(),
            kimi_linear_config(),
            nemotron_h_config(),
            qwen_next_config(),
            qwen_hybrid_config(),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let architecture_plan = plan.inspection().architecture_plan().clone();
            let eredu_core::ModelArtifact::SafeTensors {
                configuration,
                tensors,
                shards,
                ..
            } = plan.into_artifact()
            else {
                panic!("expected SafeTensors fixture")
            };
            let prepared = super::super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                super::super::loading::prepared_safetensors_architecture(&architecture_plan)
                    .unwrap()
                    .clone(),
                tensors,
                shards,
                1,
            )
            .unwrap();
            let options = super::super::loading::SelectedMlxConstruction::from_request(
                crate::MlxLoadRequest::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            super::super::path_instrumentation::reset();
            let error = match super::super::loading::materialize_safetensors(
                &prepared,
                None,
                options,
                &stream,
                &weights_stream,
            ) {
                Ok(_) => panic!("included configuration entered a family materializer"),
                Err(error) => error,
            };
            assert!(error
                .to_string()
                .contains("architecture-owned eligibility requires replicated text composition"));
            assert_eq!(
                super::super::path_instrumentation::snapshot(),
                super::super::path_instrumentation::Counts::default()
            );
        }
    }

    #[test]
    fn public_handoff_executes_selected_load_time_transform() {
        let (stream, weights_stream) = execution_streams();
        for (model_type, request) in [
            (
                "llama",
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            ("llama", eredu_core::QuantizationRequest::MxFp4),
            (
                "qwen3",
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            ("qwen3", eredu_core::QuantizationRequest::MxFp4),
        ] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let options = crate::MlxLoadRequest::with_quantization(request);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
            assert!(model.materialization_report().is_some());
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary replicated text must use the generic executable")
            };
            executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn heterogeneous_generic_handoff_executes_selected_load_time_transforms() {
        let (stream, weights_stream) = execution_streams();
        let mut lfm_affine = lfm2_config();
        lfm_affine["hidden_size"] = 32.into();
        lfm_affine["intermediate_size"] = 32.into();
        lfm_affine["num_key_value_heads"] = 1.into();
        lfm_affine["block_auto_adjust_ff_dim"] = false.into();
        let mut kimi_affine = kimi_linear_config();
        kimi_affine["hidden_size"] = 32.into();
        kimi_affine["intermediate_size"] = 32.into();
        kimi_affine["kv_lora_rank"] = 32.into();
        kimi_affine["moe_intermediate_size"] = 32.into();
        kimi_affine["linear_attn_config"]["num_heads"] = 4.into();
        kimi_affine["linear_attn_config"]["head_dim"] = 32.into();
        kimi_affine["num_attention_heads"] = 4.into();
        kimi_affine["qk_nope_head_dim"] = 24.into();
        kimi_affine["qk_rope_head_dim"] = 8.into();
        kimi_affine["v_head_dim"] = 8.into();
        let mut nemotron_affine = nemotron_h_config();
        nemotron_affine["hidden_size"] = 32.into();
        nemotron_affine["intermediate_size"] = 32.into();
        nemotron_affine["num_attention_heads"] = 8.into();
        nemotron_affine["num_key_value_heads"] = 4.into();
        nemotron_affine["mamba_num_heads"] = 8.into();
        nemotron_affine["moe_intermediate_size"] = 32.into();
        nemotron_affine["moe_shared_expert_intermediate_size"] = 32.into();
        let mut lfm_mxfp4 = lfm2_config();
        lfm_mxfp4["hidden_size"] = 32.into();
        lfm_mxfp4["intermediate_size"] = 64.into();
        lfm_mxfp4["num_key_value_heads"] = 1.into();
        lfm_mxfp4["block_auto_adjust_ff_dim"] = false.into();
        let mut qwen_mxfp4 = qwen_hybrid_config();
        qwen_mxfp4["intermediate_size"] = 64.into();
        for (name, config, request) in [
            (
                "lfm2-affine",
                lfm_affine,
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            (
                "kimi-affine",
                kimi_affine,
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            (
                "nemotron-affine",
                nemotron_affine,
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            (
                "lfm2-mxfp4",
                lfm_mxfp4,
                eredu_core::QuantizationRequest::MxFp4,
            ),
            (
                "qwen-mxfp4",
                qwen_mxfp4,
                eredu_core::QuantizationRequest::MxFp4,
            ),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let options = crate::MlxLoadRequest::with_quantization(request);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let report = model
                .materialization_report()
                .unwrap_or_else(|| panic!("{name}: no materialization report"));
            assert!(report.transformed_weights > 0, "{name}");
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
                panic!("{name} did not use generic replicated composition")
            };
            generic
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn public_handoff_executes_admitted_gguf_mapping() {
        let (stream, weights_stream) = execution_streams();
        let artifacts = [
            tiny_llama_gguf("llama", None, &stream),
            tiny_llama_gguf("mistral", None, &stream),
            tiny_qwen_gguf("qwen2", None, &stream),
            tiny_qwen_gguf("qwen3", None, &stream),
        ];
        for artifact in artifacts {
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap();
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary replicated GGUF text must use the generic executable")
            };
            let logits = executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap();
            assert_eq!(logits.shape(), &[1, 64]);
            logits.evaluated().unwrap();
        }
    }

    #[test]
    fn both_text_architectures_execute_checkpoint_native_packed_gguf_formats() {
        let (stream, weights_stream) = execution_streams();
        for (architecture, format) in [
            ("llama", eredu_gguf::GgmlType::Q4_0),
            ("llama", eredu_gguf::GgmlType::MxFp4),
            ("llama", eredu_gguf::GgmlType::IQ4NL),
            ("qwen2", eredu_gguf::GgmlType::Q4_0),
            ("qwen2", eredu_gguf::GgmlType::MxFp4),
            ("qwen2", eredu_gguf::GgmlType::IQ4NL),
        ] {
            let artifact = if architecture == "llama" {
                tiny_llama_gguf(architecture, Some(format), &stream)
            } else {
                tiny_qwen_gguf(architecture, Some(format), &stream)
            };
            let checkpoint = eredu_gguf::Checkpoint::open(artifact.path()).unwrap();
            let translated = if architecture == "llama" {
                checkpoint
                    .translated_outputs(eredu_architectures::llama::translate_gguf_weight_name)
                    .unwrap()
            } else {
                checkpoint
                    .translated_outputs(|name| {
                        eredu_architectures::qwen::translate_gguf_weight_name(name, false)
                    })
                    .unwrap()
            };
            if matches!(
                format,
                eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::MxFp4
            ) {
                assert!(translated.iter().any(|mapping| {
                    mapping.original_name.ends_with(".scales")
                        && mapping.layout.name.ends_with(".scales")
                        && mapping.layout.name.starts_with("model.")
                }));
            }
            if format == eredu_gguf::GgmlType::Q4_0 {
                assert!(translated.iter().any(|mapping| {
                    mapping.original_name.ends_with(".biases")
                        && mapping.layout.name.ends_with(".biases")
                }));
            }
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            assert!(requirements.parameters().iter().any(|parameter| {
                matches!(
                    (format, parameter.native_executable()),
                    (eredu_gguf::GgmlType::Q4_0, LinearFormat::Affine(_))
                        | (eredu_gguf::GgmlType::MxFp4, LinearFormat::MxFp4)
                        | (eredu_gguf::GgmlType::IQ4NL, LinearFormat::GgufIQuant { .. })
                )
            }));
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{architecture} {format:?}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary packed GGUF text must use the generic executable")
            };
            executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn generic_controls_cover_residency_cache_persistence_and_observation() {
        struct Observer {
            activation: bool,
            logits: bool,
            intervened: bool,
            stream: Stream,
        }
        impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
            fn observe(&mut self, path: &str, _value: &Array) -> Result<(), Exception> {
                self.logits |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                self.activation |= path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
                if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                    self.intervened = true;
                    Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
                } else {
                    Ok(None)
                }
            }
        }

        let (stream, weights_stream) = execution_streams();
        let mut host = eredu_runtime::LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 7).unwrap(),
        );
        host = host.with_max_cached_shards(3);
        let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 4).unwrap();
        for (model_type, residency) in ["llama", "qwen2"].into_iter().flat_map(|family| {
            [
                eredu_runtime::WeightResidency::fully_resident(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
            ]
            .into_iter()
            .map(move |residency| (family, residency))
        }) {
            let root = tiny_artifact(model_type, false);
            let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true);
            let options = crate::MlxLoadRequest::default()
                .with_weight_residency(residency)
                .with_state_residency(CacheResidencyPolicy::Paged(paged.clone()));
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
            assert!(model.residency_report().unwrap().is_some());
            assert_eq!(
                model.dense_stream_report().unwrap().is_some(),
                matches!(
                    residency,
                    eredu_runtime::WeightResidency::Layers(
                        eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                    )
                )
            );
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
                panic!("ordinary replicated text must use the generic executable")
            };
            assert_eq!(generic.selected_residency(), residency.layers());
            generic
                .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();

            let mut observer = Observer {
                activation: false,
                logits: false,
                intervened: false,
                stream: stream.clone(),
            };
            let replacement = generic
                .forward_with_observer(
                    &Array::from_slice(&[3_u32], &[1, 1]),
                    None,
                    &stream,
                    &mut observer,
                )
                .unwrap();
            let replacement = replacement.evaluated().unwrap();
            assert!(observer.logits);
            assert!(observer.activation);
            assert!(observer.intervened);
            assert!(replacement
                .as_slice::<f32>()
                .iter()
                .all(|value| *value == 0.0));

            let identity = generic.prompt_cache_model_identity().clone();
            let descriptor = PromptCacheDescriptor::from_model_identity(
                identity,
                "tiny-checkpoint",
                "tokens:1,2,3",
                1,
            )
            .unwrap();
            let cache_root = tempfile::tempdir().unwrap();
            let destination = cache_root.path().join("cache");
            let prefix = [1_u32, 2, 3];
            generic.reset_cache().unwrap();
            generic
                .decode(&Array::from_slice(&prefix, &[1, 3]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let manifest = generic
                .save_prompt_cache(
                    &destination,
                    descriptor.clone(),
                    &prefix,
                    &PromptCacheOptions::default(),
                )
                .unwrap();
            assert_eq!(manifest.block_size_tokens, paged.block_size_tokens());
            let incompatible = descriptor
                .clone()
                .with_architecture_fingerprint(format!(
                    "{}-different",
                    descriptor.architecture_fingerprint()
                ))
                .unwrap();
            assert!(generic
                .load_prompt_cache(&destination, &incompatible, &prefix)
                .is_err());
            generic
                .load_prompt_cache(&destination, &descriptor, &prefix)
                .unwrap();
            assert!(generic.cache_residency_report().unwrap().is_some());
            generic
                .decode(&Array::from_slice(&[4_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn heterogeneous_generic_sessions_preserve_every_state_component_across_controls() {
        struct Observer {
            activation: bool,
            logits: bool,
            intervened: bool,
            stream: Stream,
        }
        impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
            fn observe(&mut self, path: &str, _value: &Array) -> Result<(), Exception> {
                self.logits |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                self.activation |= path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
                if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                    self.intervened = true;
                    Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
                } else {
                    Ok(None)
                }
            }
        }

        let (stream, weights_stream) = execution_streams();
        let host = eredu_runtime::LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 3).unwrap(),
        )
        .with_max_cached_shards(2);
        let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 2).unwrap();
        let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        let cases = vec![
            (
                "lfm2",
                lfm2_config(),
                eredu_runtime::WeightResidency::fully_resident(),
                CacheResidencyPolicy::Device,
            ),
            (
                "lfm2-paged",
                lfm2_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "kimi_linear",
                kimi_linear_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Device,
            ),
            (
                "kimi_linear-paged",
                kimi_linear_config(),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "nemotron_h",
                nemotron_h_config(),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
                CacheResidencyPolicy::Device,
            ),
            (
                "nemotron_h-paged",
                nemotron_h_config(),
                eredu_runtime::WeightResidency::fully_resident(),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "qwen3_next",
                qwen_next_config(),
                eredu_runtime::WeightResidency::fully_resident(),
                CacheResidencyPolicy::Device,
            ),
            (
                "qwen3_next-paged",
                qwen_next_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "qwen3_5_text",
                qwen_hybrid_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Device,
            ),
            (
                "qwen3_5_text-paged",
                qwen_hybrid_config(),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
                CacheResidencyPolicy::Paged(paged),
            ),
        ];

        for (name, config, residency, state_policy) in cases {
            let root = tiny_heterogeneous_artifact(config);
            let options = crate::MlxLoadRequest::default()
                .with_weight_residency(residency)
                .with_state_residency(state_policy.clone());
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(model.residency_report().unwrap().is_some());
            assert_eq!(
                model.dense_stream_report().unwrap().is_some(),
                matches!(
                    residency,
                    eredu_runtime::WeightResidency::Layers(
                        eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                    )
                ),
                "{name}"
            );
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
                panic!("{name} did not use generic replicated composition")
            };
            assert_eq!(generic.selected_residency(), residency.layers(), "{name}");
            assert_eq!(
                generic.cache_residency_report().unwrap().is_some(),
                matches!(state_policy, CacheResidencyPolicy::Paged(_)),
                "{name}"
            );

            let prefix = [1_u32, 2, 3];
            let prompt = Array::from_slice(&prefix, &[1, 3]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            generic
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let saved_snapshot = generic.state_snapshot();
            let saved_numeric = generic
                .fixed_numeric_state_snapshot()
                .unwrap_or_else(|error| panic!("{name} numeric snapshot: {error}"));
            assert!(!saved_numeric.is_empty(), "{name}");
            assert!(saved_snapshot
                .iter()
                .flat_map(|(_, fixed)| fixed)
                .all(|(_, present)| *present));

            let persisted = matches!(state_policy, CacheResidencyPolicy::Paged(_)).then(|| {
                let identity = generic.prompt_cache_model_identity().clone();
                let descriptor = PromptCacheDescriptor::from_model_identity(
                    identity,
                    format!("{name}-checkpoint"),
                    "tokens:1,2,3",
                    1,
                )
                .unwrap();
                let cache_root = tempfile::tempdir().unwrap();
                let destination = cache_root.path().join("cache");
                generic
                    .save_prompt_cache(
                        &destination,
                        descriptor.clone(),
                        &prefix,
                        &PromptCacheOptions::default(),
                    )
                    .unwrap();
                (cache_root, destination, descriptor)
            });
            let continuation_token = Array::from_slice(&[4_u32], &[1, 1]);
            let persistence_baseline = persisted
                .as_ref()
                .map(|_| generic.checkpoint_restore_probe(&continuation_token, &stream))
                .transpose()
                .unwrap_or_else(|error| panic!("{name} persistence baseline: {error}"));
            generic.reset_cache().unwrap();
            assert!(generic.state_snapshot().iter().all(|(position, fixed)| {
                *position == 0 && fixed.iter().all(|(_, present)| !present)
            }));
            if let Some((_cache_root, destination, descriptor)) = persisted {
                let incompatible = descriptor
                    .clone()
                    .with_architecture_fingerprint(format!(
                        "{}-different",
                        descriptor.architecture_fingerprint()
                    ))
                    .unwrap();
                assert!(generic
                    .load_prompt_cache(&destination, &incompatible, &prefix)
                    .is_err());
                generic
                    .load_prompt_cache(&destination, &descriptor, &prefix)
                    .unwrap();
            } else {
                assert!(generic
                    .save_prompt_cache(
                        tempfile::tempdir().unwrap().path(),
                        PromptCacheDescriptor::from_model_identity(
                            generic.prompt_cache_model_identity().clone(),
                            format!("{name}-checkpoint"),
                            "tokens:1,2,3",
                            1,
                        )
                        .unwrap(),
                        &prefix,
                        &PromptCacheOptions::default(),
                    )
                    .is_err());
                generic
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap();
            }
            assert_eq!(generic.state_snapshot(), saved_snapshot, "{name}");
            assert_eq!(
                generic.fixed_numeric_state_snapshot().unwrap(),
                saved_numeric,
                "{name} fixed tensors changed across prompt-cache restoration"
            );
            let probe = generic
                .checkpoint_restore_probe(&continuation_token, &stream)
                .unwrap_or_else(|error| panic!("{name} checkpoint/restore: {error}"));
            if let Some(baseline) = persistence_baseline {
                assert_eq!(
                    probe, baseline,
                    "{name} continuation changed after prompt-cache restoration"
                );
            }
            let (
                before,
                advanced,
                restored,
                before_numeric,
                advanced_numeric,
                restored_numeric,
                continuation,
            ) = probe;
            assert_eq!(before, saved_snapshot, "{name}");
            assert_ne!(advanced, before, "{name}");
            assert_eq!(restored, before, "{name}");
            assert_eq!(before_numeric, saved_numeric, "{name}");
            assert_ne!(advanced_numeric, before_numeric, "{name}");
            assert_eq!(restored_numeric, before_numeric, "{name}");
            let replayed = generic.decode(&continuation_token, &stream).unwrap();
            let replayed = replayed.evaluated().unwrap();
            assert_eq!(
                replayed.as_slice::<f32>(),
                continuation.as_slice(),
                "{name}"
            );
            assert_eq!(generic.state_snapshot(), advanced, "{name}");
            assert_eq!(
                generic.fixed_numeric_state_snapshot().unwrap(),
                advanced_numeric,
                "{name}"
            );

            let mut observer = Observer {
                activation: false,
                logits: false,
                intervened: false,
                stream: stream.clone(),
            };
            let replacement = generic
                .forward_with_observer(
                    &Array::from_slice(&[5_u32], &[1, 1]),
                    None,
                    &stream,
                    &mut observer,
                )
                .unwrap();
            let replacement = replacement.evaluated().unwrap();
            assert!(observer.activation && observer.logits, "{name}");
            assert!(observer.intervened, "{name}");
            assert!(
                replacement
                    .as_slice::<f32>()
                    .iter()
                    .all(|value| *value == 0.0),
                "{name}"
            );
        }
    }

    #[test]
    fn heterogeneous_logits_and_fixed_state_match_across_weight_and_state_residency() {
        let (stream, weights_stream) = execution_streams();
        let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 2).unwrap();
        let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        for (name, config) in [
            ("lfm2", lfm2_config()),
            ("kimi_linear", kimi_linear_config()),
            ("nemotron_h", nemotron_h_config()),
            ("qwen3_next", qwen_next_config()),
            ("qwen3_5_text", qwen_hybrid_config()),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let mut results = Vec::new();
            for (residency, state) in [
                (
                    eredu_runtime::WeightResidency::fully_resident(),
                    CacheResidencyPolicy::Device,
                ),
                (
                    eredu_runtime::WeightResidency::dense_disk_stream(disk),
                    CacheResidencyPolicy::Paged(paged.clone()),
                ),
            ] {
                let options = crate::MlxLoadRequest::default()
                    .with_weight_residency(residency)
                    .with_state_residency(state);
                let inspection =
                    eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
                let plan = eredu_core::plan_model_preparation(
                    inspection,
                    options.preparation_policy().unwrap(),
                    eredu_core::SessionCapabilities::default(),
                )
                .unwrap();
                let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let mut executable = model.into_complete().unwrap();
                let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
                    panic!("{name} did not use generic replicated composition")
                };
                let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
                let parts = [input::token_ids_part(&prompt).unwrap()];
                generic
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap();
                let logits = generic
                    .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec();
                assert!(
                    logits.iter().all(|value| value.is_finite())
                        && logits.iter().any(|value| value.abs() > 1e-12),
                    "{name}: {logits:?}"
                );
                results.push((
                    logits,
                    generic.state_snapshot(),
                    generic.fixed_numeric_state_snapshot().unwrap(),
                ));
            }
            let (resident_logits, resident_semantics, resident_fixed) = &results[0];
            let (bounded_logits, bounded_semantics, bounded_fixed) = &results[1];
            assert_eq!(resident_semantics, bounded_semantics, "{name}");
            assert_eq!(resident_fixed.len(), bounded_fixed.len(), "{name}");
            for (left, right) in resident_logits.iter().zip(bounded_logits) {
                assert!((left - right).abs() <= 1e-5, "{name}: {left} != {right}");
            }
            for (left, right) in resident_fixed.iter().zip(bounded_fixed) {
                assert_eq!((&left.0, &left.1, &left.2), (&right.0, &right.1, &right.2));
                for (left, right) in left.3.iter().zip(&right.3) {
                    assert!((left - right).abs() <= 1e-5, "{name}: {left} != {right}");
                }
            }
        }
    }
}
