//! Unified neutral DeepSeek-V3/V4 loading across layer-residency policies.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use eredu_architectures::deepseek::{self, LayerPolicy, V3Args, V4Args, V4AttentionPolicy};
use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, DeviceState, ExecutionUnitLayout, LayerWeightResidency,
    LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions, ParameterRole, RuntimeState,
    WeightResidency,
};
use safemlx::{
    distributed::Group,
    error::Exception,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

use crate::backend::mlx::runtime::{
    distributed::parallel::ParallelBuildContext,
    execution::layerwise::{open_safetensors_weight_store, shard_layer_bindings},
    media::input,
};
use crate::backend::mlx::{
    error::Error,
    nn::shared::{MlxBackend, MlxModule},
    runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
            },
            state::MlxPoolingAttentionCache,
            CompressedLatentCache,
        },
        checkpoint::binding::{
            build_module_bindings, build_module_bindings_with_recipes_excluding,
            parameter_name_in_targets, parameter_role_targets,
            populate_module_from_lease_excluding,
        },
        checkpoint::load::gguf_quantization_configs,
        checkpoint::store::open_gguf_checkpoint_source,
        execution::generic::{
            prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            MlxUnitPopulator,
        },
        execution::layerwise::quantize_module_store_with_bindings,
        residency::expert_cache::{ExpertCache, ExpertCacheReport},
        residency::manager::ResidentUnitLease,
    },
};
use eredu_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
};

type V3Architecture = deepseek::v3::Model<MlxBackend>;
type V3Unit = deepseek::v3::Unit<MlxBackend>;
type V3State = DeviceState<MlxBackend, CompressedLatentCache>;
type V3Resident = LayerwiseRuntime<V3Architecture, MlxBackend, V3State, MlxResidentPolicy<V3Unit>>;
type V3Layerwise = LayerwiseRuntime<
    V3Architecture,
    MlxBackend,
    V3State,
    MlxLayerwisePolicy<V3Unit, V3UnitPopulator>,
>;

type V4Architecture = deepseek::v4::Model<MlxBackend>;
type V4Unit = deepseek::v4::Unit<MlxBackend>;
type V4State = DeviceState<MlxBackend, MlxPoolingAttentionCache>;
type V4Resident = LayerwiseRuntime<V4Architecture, MlxBackend, V4State, MlxResidentPolicy<V4Unit>>;
type V4Layerwise = LayerwiseRuntime<
    V4Architecture,
    MlxBackend,
    V4State,
    MlxLayerwisePolicy<V4Unit, V4UnitPopulator>,
>;

fn construct_v3_unit(
    architecture: &V3Architecture,
    ordinal: usize,
    stream: &Stream,
) -> Result<V3Unit, Error> {
    let layout = execution_layout::<V3Architecture, V3State>(architecture)?;
    let address = layout
        .address(ordinal)
        .ok_or_else(|| unsupported(format!("V3 unit ordinal {ordinal} is out of range")))?;
    <V3Architecture as LayeredArchitecture<MlxBackend, V3State>>::build_unit(
        architecture,
        address.group(),
        address.index(),
        stream,
    )
    .map_err(neutral_error)
}

fn construct_v4_unit(
    architecture: &V4Architecture,
    ordinal: usize,
    stream: &Stream,
) -> Result<V4Unit, Error> {
    let layout = execution_layout::<V4Architecture, V4State>(architecture)?;
    let address = layout
        .address(ordinal)
        .ok_or_else(|| unsupported(format!("V4 unit ordinal {ordinal} is out of range")))?;
    <V4Architecture as LayeredArchitecture<MlxBackend, V4State>>::build_unit(
        architecture,
        address.group(),
        address.index(),
        stream,
    )
    .map_err(neutral_error)
}

struct NeutralDeepSeekObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
}

impl eredu_runtime::ActivationObserver<crate::MlxTensor, eredu_nn::Error>
    for NeutralDeepSeekObserver<'_>
{
    fn observe(&mut self, path: &str, value: &crate::MlxTensor) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe(path, value.as_array())
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &crate::MlxTensor,
    ) -> Result<Option<crate::MlxTensor>, eredu_nn::Error> {
        self.inner
            .intervene(path, value.as_array())
            .map(|value| value.map(crate::MlxTensor::from_array))
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        let raw = eredu_runtime::RoutingObservation {
            path: routing.path,
            selected_experts: routing.selected_experts.as_array(),
            selected_scores: routing.selected_scores.as_array(),
            route_weights: routing.route_weights.as_array(),
            routed_output: routing.routed_output.as_array(),
            local_routed_output: routing.local_routed_output.map(crate::MlxTensor::as_array),
            reduced_routed_output: routing
                .reduced_routed_output
                .map(crate::MlxTensor::as_array),
            shared_output: routing.shared_output.map(crate::MlxTensor::as_array),
            combined_output: routing.combined_output.map(crate::MlxTensor::as_array),
            expert_count: routing.expert_count,
        };
        self.inner
            .observe_routing(raw)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

fn neutral_embedded_input<'a>(
    input: deepseek::mtp::EmbeddedInput<'a, Array>,
) -> deepseek::mtp::EmbeddedInput<'a, crate::MlxTensor> {
    match input {
        deepseek::mtp::EmbeddedInput::Target { tokens, mask } => {
            deepseek::mtp::EmbeddedInput::target(
                crate::composition::tensor_ref(tokens),
                crate::composition::tensor_opt(mask),
            )
        }
        deepseek::mtp::EmbeddedInput::Draft {
            tokens,
            hidden,
            depth,
        } => deepseek::mtp::EmbeddedInput::draft(
            crate::composition::tensor_ref(tokens),
            crate::composition::tensor_ref(hidden),
            depth,
        ),
        deepseek::mtp::EmbeddedInput::DsparkContext { captures } => {
            deepseek::mtp::EmbeddedInput::dspark_context(crate::composition::tensor_ref(captures))
        }
        deepseek::mtp::EmbeddedInput::DsparkProposal { anchor, capacity } => {
            deepseek::mtp::EmbeddedInput::dspark_proposal(
                crate::composition::tensor_ref(anchor),
                capacity,
            )
        }
    }
}

#[derive(Clone)]
struct V3UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<std::collections::BTreeSet<String>>,
}

impl MlxUnitPopulator<V3Unit> for V3UnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<V3Unit>,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

#[derive(Clone)]
struct V4UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<std::collections::BTreeSet<String>>,
}

impl MlxUnitPopulator<V4Unit> for V4UnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<V4Unit>,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

enum V3Execution {
    Resident(Box<V3Resident>),
    Layerwise(Box<V3Layerwise>),
}

enum V4Execution {
    Resident(Box<V4Resident>),
    Layerwise(Box<V4Layerwise>),
}

/// One neutral DeepSeek model whose architecture is independent of residency.
pub struct DeepSeekModel {
    inner: DeepSeekModelInner,
}

enum DeepSeekModelInner {
    /// DeepSeek-V3/R1 target and embedded MTP graph.
    V3 {
        /// Validated neutral configuration.
        args: V3Args,
        execution: V3Execution,
        expert_cache: Option<ExpertCache>,
        materialization: Option<eredu_runtime::WeightMaterializationReport>,
        tensor_parallel: bool,
    },
    /// DeepSeek-V4 target and MTP/DSpark graph.
    V4 {
        /// Validated neutral configuration.
        args: V4Args,
        execution: V4Execution,
        expert_cache: Option<ExpertCache>,
        materialization: Option<eredu_runtime::WeightMaterializationReport>,
        tensor_parallel: bool,
    },
}

/// Matching architecture-declared mutable state.
#[derive(Debug, Clone)]
pub struct DeepSeekState {
    inner: DeepSeekStateInner,
    target_layers: usize,
}

#[derive(Debug, Clone)]
enum DeepSeekStateInner {
    /// V3 compressed latent/rotary state.
    V3(V3State),
    /// V4 local and pooled state.
    V4(V4State),
}

impl DeepSeekState {
    pub fn clear(&mut self) -> Result<(), Exception> {
        match &mut self.inner {
            DeepSeekStateInner::V3(state) => {
                for cache in state.as_mut() {
                    cache.clear()?;
                }
            }
            DeepSeekStateInner::V4(state) => {
                for cache in state.as_mut() {
                    cache.clear()?;
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn offset(&self) -> i32 {
        match &self.inner {
            DeepSeekStateInner::V3(state) => {
                state.as_ref().first().map_or(0, |cache| cache.offset())
            }
            DeepSeekStateInner::V4(state) => {
                state.as_ref().first().map_or(0, |cache| cache.offset())
            }
        }
    }

    fn commit_prediction_layers_from(
        &mut self,
        draft: &Self,
        target_layers: usize,
    ) -> Result<(), Exception> {
        match (&mut self.inner, &draft.inner) {
            (DeepSeekStateInner::V3(current), DeepSeekStateInner::V3(draft)) => {
                let current = current.as_mut();
                let draft = draft.as_ref();
                if current.len() != draft.len() || target_layers > current.len() {
                    return Err(Exception::custom("V3 draft state layout mismatch"));
                }
                current[target_layers..].clone_from_slice(&draft[target_layers..]);
            }
            (DeepSeekStateInner::V4(current), DeepSeekStateInner::V4(draft)) => {
                let current = current.as_mut();
                let draft = draft.as_ref();
                if current.len() != draft.len() || target_layers > current.len() {
                    return Err(Exception::custom("V4 draft state layout mismatch"));
                }
                current[target_layers..].clone_from_slice(&draft[target_layers..]);
            }
            _ => return Err(Exception::custom("DeepSeek draft state family mismatch")),
        }
        Ok(())
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        match (&mut self.inner, &checkpoint.inner) {
            (DeepSeekStateInner::V3(current), DeepSeekStateInner::V3(previous)) => {
                if current.as_ref().len() != previous.as_ref().len() {
                    return Err(Exception::custom("V3 checkpoint state layout mismatch"));
                }
                for (current, previous) in current.as_mut().iter_mut().zip(previous.as_ref()) {
                    eredu_nn::CompressedAttentionCache::restore(current, previous, stream)
                        .map_err(|error| Exception::custom(error.to_string()))?;
                }
            }
            (DeepSeekStateInner::V4(current), DeepSeekStateInner::V4(previous)) => {
                if current.as_ref().len() != previous.as_ref().len() {
                    return Err(Exception::custom("V4 checkpoint state layout mismatch"));
                }
                for (current, previous) in current.as_mut().iter_mut().zip(previous.as_ref()) {
                    eredu_nn::PoolingAttentionCache::restore(current, previous, stream)
                        .map_err(|error| Exception::custom(error.to_string()))?;
                }
            }
            _ => {
                return Err(Exception::custom(
                    "DeepSeek checkpoint state family mismatch",
                ))
            }
        }
        Ok(())
    }

    pub fn residency_report(
        &self,
    ) -> Result<Option<eredu_runtime::CacheResidencyReport>, Exception> {
        match &self.inner {
            DeepSeekStateInner::V3(state) => state
                .as_ref()
                .iter()
                .find_map(|cache| cache.residency_manager().map(CacheResidencyManager::report))
                .transpose()
                .map_err(|error| Exception::custom(error.to_string())),
            DeepSeekStateInner::V4(state) => state
                .as_ref()
                .iter()
                .find_map(|cache| cache.residency_manager().map(CacheResidencyManager::report))
                .transpose()
                .map_err(|error| Exception::custom(error.to_string())),
        }
    }
}

impl DeepSeekModel {
    /// Loads a validated V3 architecture from one resolved neutral store.
    pub fn load_v3(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        args: V3Args,
        options: LayerWeightResidency,
        stream: &Stream,
        weights_stream: &Stream,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let mut architecture = V3Architecture::new(args.clone(), stream).map_err(neutral_error)?;
        let layout = execution_layout::<V3Architecture, V3State>(&architecture)?;
        let expert_targets = Arc::new(
            deepseek::parallel::v3_parameter_description(&args)?
                .targets_for_role(ParameterRole::ExpertIntermediate),
        );
        let binding_args = args.clone();
        let factory = V3UnitPopulator {
            external_experts,
            expert_targets: Arc::clone(&expert_targets),
        };
        let excluded_expert_targets = Arc::clone(&expert_targets);
        let binding_expert_targets = Arc::clone(&expert_targets);
        let (policy, _metadata) = prepare_layerwise_policy_with_bindings(
            store,
            &mut architecture,
            factory,
            std::marker::PhantomData::<V3State>,
            layout,
            options,
            stream,
            weights_stream,
            move |key| {
                key.starts_with("rope_freqs.")
                    || key.ends_with("rotary_emb.inv_freq")
                    || (external_experts
                        && parameter_name_in_targets(key, &excluded_expert_targets))
            },
            |modules, store| {
                build_module_bindings(&MlxModule::new(modules.clone()), "", store)
                    .map_err(Into::into)
            },
            move |ordinal, unit, store, _| {
                let recipes = v3_unit_recipes(store, &binding_args, ordinal, external_experts)?;
                build_module_bindings_with_recipes_excluding(
                    &MlxModule::new(unit),
                    "",
                    store,
                    recipes,
                    |name| {
                        external_experts && parameter_name_in_targets(name, &binding_expert_targets)
                    },
                )
                .map_err(Into::into)
            },
        )?;
        let execution = if options.is_fully_resident() {
            V3Execution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
                policy.into_resident(&architecture, stream, std::marker::PhantomData::<V3State>)?,
                architecture,
            )))
        } else {
            V3Execution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
        };
        Ok(Self {
            inner: DeepSeekModelInner::V3 {
                args,
                execution,
                expert_cache: None,
                materialization: None,
                tensor_parallel: false,
            },
        })
    }

    /// Loads a validated V4 architecture from one resolved neutral store.
    pub fn load_v4(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        args: V4Args,
        options: LayerWeightResidency,
        stream: &Stream,
        weights_stream: &Stream,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let mut architecture = V4Architecture::new(args.clone(), stream).map_err(neutral_error)?;
        let layout = execution_layout::<V4Architecture, V4State>(&architecture)?;
        let expert_targets = Arc::new(
            deepseek::parallel::v4_parameter_description(&args)?
                .targets_for_role(ParameterRole::ExpertIntermediate),
        );
        let binding_args = args.clone();
        let factory = V4UnitPopulator {
            external_experts,
            expert_targets: Arc::clone(&expert_targets),
        };
        let excluded_expert_targets = Arc::clone(&expert_targets);
        let binding_expert_targets = Arc::clone(&expert_targets);
        let (policy, _metadata) = prepare_layerwise_policy_with_bindings(
            store,
            &mut architecture,
            factory,
            std::marker::PhantomData::<V4State>,
            layout,
            options,
            stream,
            weights_stream,
            move |key| {
                key.ends_with("rotary_emb.inv_freq")
                    || (external_experts
                        && parameter_name_in_targets(key, &excluded_expert_targets))
            },
            |modules, store| {
                build_module_bindings(&MlxModule::new(modules.clone()), "", store)
                    .map_err(Into::into)
            },
            move |ordinal, unit, store, _| {
                let recipes = if external_experts {
                    BTreeMap::new()
                } else {
                    v4_unit_recipes(store, &binding_args, ordinal)?
                };
                build_module_bindings_with_recipes_excluding(
                    &MlxModule::new(unit),
                    "",
                    store,
                    recipes,
                    |name| {
                        external_experts && parameter_name_in_targets(name, &binding_expert_targets)
                    },
                )
                .map_err(Into::into)
            },
        )?;
        let execution = if options.is_fully_resident() {
            V4Execution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
                policy.into_resident(&architecture, stream, std::marker::PhantomData::<V4State>)?,
                architecture,
            )))
        } else {
            V4Execution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
        };
        Ok(Self {
            inner: DeepSeekModelInner::V4 {
                args,
                execution,
                expert_cache: None,
                materialization: None,
                tensor_parallel: false,
            },
        })
    }

    /// Loads tensor-partitioned V3 nonexpert weights while leaving routed
    /// experts to an external EP provider.
    pub fn load_v3_external_expert_parallel(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        args: V3Args,
        options: LayerWeightResidency,
        build: ParallelBuildContext,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
            .map_err(|_| unsupported("invalid V3 unit count"))?;
        let mut planner = build.planner();
        for group in deepseek::parallel::v3_static_parameter_groups(&args)? {
            planner.register(group)?;
        }
        for layer in 0..total {
            for group in deepseek::parallel::v3_layer_parameter_groups(&args, layer)? {
                planner.register(group)?;
            }
        }
        let (_, layout) = planner.finish()?;
        let geometry = deepseek::parallel::v3_local_geometry(&args, &layout)?;
        let global = V3Architecture::new(args.clone(), stream).map_err(neutral_error)?;
        let expert_targets = Arc::new(
            deepseek::parallel::v3_parameter_description(&args)?
                .targets_for_role(ParameterRole::ExpertIntermediate),
        );
        let global_static = global.static_modules().clone();
        let mut architecture =
            V3Architecture::new_parallel(args.clone(), geometry, stream).map_err(neutral_error)?;
        let factory = V3UnitPopulator {
            external_experts: true,
            expert_targets: Arc::clone(&expert_targets),
        };
        let static_layout = Arc::new(layout);
        let unit_layout = Arc::clone(&static_layout);
        let binding_args = args.clone();
        let runtime_layout = execution_layout::<V3Architecture, V3State>(&architecture)?;
        let (policy, _metadata) = prepare_layerwise_policy_with_bindings(
            store,
            &mut architecture,
            factory,
            std::marker::PhantomData::<V3State>,
            runtime_layout,
            options,
            stream,
            weights_stream,
            {
                let expert_targets = Arc::clone(&expert_targets);
                move |key| {
                    key.starts_with("rope_freqs.")
                        || key.ends_with("rotary_emb.inv_freq")
                        || parameter_name_in_targets(key, &expert_targets)
                }
            },
            move |_modules, store| {
                let bindings =
                    build_module_bindings(&MlxModule::new(global_static.clone()), "", store)?;
                shard_layer_bindings(bindings, "", store, &static_layout)
            },
            move |ordinal, _unit, store, stream| {
                let probe = new_v3_unit(&binding_args, ordinal, true, stream)?;
                let bindings = v3_unit_bindings(&binding_args, ordinal, &probe, store, true)?;
                shard_layer_bindings(bindings, "", store, &unit_layout)
            },
        )?;
        let execution = if options.is_fully_resident() {
            V3Execution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
                policy.into_resident(&architecture, stream, std::marker::PhantomData::<V3State>)?,
                architecture,
            )))
        } else {
            V3Execution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
        };
        Ok(Self {
            inner: DeepSeekModelInner::V3 {
                args,
                execution,
                expert_cache: None,
                materialization: None,
                tensor_parallel: true,
            },
        })
    }

    /// Loads tensor-partitioned V4 nonexpert weights while leaving routed
    /// experts to an external EP provider.
    pub fn load_v4_external_expert_parallel(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        args: V4Args,
        options: LayerWeightResidency,
        build: ParallelBuildContext,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
            .map_err(|_| unsupported("invalid V4 unit count"))?;
        let mut planner = build.planner();
        for group in deepseek::parallel::v4_static_parameter_groups(&args)? {
            planner.register(group)?;
        }
        for layer in 0..total {
            for group in deepseek::parallel::v4_layer_parameter_groups(&args, layer)? {
                planner.register(group)?;
            }
        }
        let (_, layout) = planner.finish()?;
        let geometry = deepseek::parallel::v4_local_geometry(&args, &layout)?;
        let global = V4Architecture::new(args.clone(), stream).map_err(neutral_error)?;
        let expert_targets = Arc::new(
            deepseek::parallel::v4_parameter_description(&args)?
                .targets_for_role(ParameterRole::ExpertIntermediate),
        );
        let global_static = global.static_modules().clone();
        let mut architecture =
            V4Architecture::new_parallel(args.clone(), geometry, stream).map_err(neutral_error)?;
        let factory = V4UnitPopulator {
            external_experts: true,
            expert_targets: Arc::clone(&expert_targets),
        };
        let static_layout = Arc::new(layout);
        let unit_layout = Arc::clone(&static_layout);
        let binding_args = args.clone();
        let runtime_layout = execution_layout::<V4Architecture, V4State>(&architecture)?;
        let (policy, _metadata) = prepare_layerwise_policy_with_bindings(
            store,
            &mut architecture,
            factory,
            std::marker::PhantomData::<V4State>,
            runtime_layout,
            options,
            stream,
            weights_stream,
            {
                let expert_targets = Arc::clone(&expert_targets);
                move |key| {
                    key.ends_with("rotary_emb.inv_freq")
                        || parameter_name_in_targets(key, &expert_targets)
                }
            },
            move |_modules, store| {
                let bindings =
                    build_module_bindings(&MlxModule::new(global_static.clone()), "", store)?;
                shard_layer_bindings(bindings, "", store, &static_layout)
            },
            move |ordinal, _unit, store, stream| {
                let probe = new_v4_unit(&binding_args, ordinal, true, stream)?;
                let bindings = v4_unit_bindings(&binding_args, ordinal, &probe, store, true)?;
                shard_layer_bindings(bindings, "", store, &unit_layout)
            },
        )?;
        let execution = if options.is_fully_resident() {
            V4Execution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
                policy.into_resident(&architecture, stream, std::marker::PhantomData::<V4State>)?,
                architecture,
            )))
        } else {
            V4Execution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
        };
        Ok(Self {
            inner: DeepSeekModelInner::V4 {
                args,
                execution,
                expert_cache: None,
                materialization: None,
                tensor_parallel: true,
            },
        })
    }

    /// Allocates resident state directly from the architecture layout.
    pub fn new_state(&self) -> Result<DeepSeekState, Error> {
        let layout = self.state_layout()?;
        match &self.inner {
            DeepSeekModelInner::V3 { args, .. } => Ok(DeepSeekState {
                inner: DeepSeekStateInner::V3(DeviceState::create(layout, |_, _| {
                    Ok::<_, Error>(CompressedLatentCache::new())
                })?),
                target_layers: args.num_hidden_layers as usize,
            }),
            DeepSeekModelInner::V4 { args, .. } => {
                let state_args = args.clone();
                Ok(DeepSeekState {
                    inner: DeepSeekStateInner::V4(DeviceState::create(layout, move |layer, _| {
                        let ratio = match state_args.attention_policy(layer) {
                            Some(V4AttentionPolicy::Local) => 0,
                            Some(V4AttentionPolicy::Compressed { ratio }) => ratio,
                            None => return Err(unsupported("missing V4 state attention policy")),
                        };
                        MlxPoolingAttentionCache::resident(ratio, state_args.sliding_window)
                            .map_err(Into::into)
                    })?),
                    target_layers: args.num_hidden_layers as usize,
                })
            }
        }
    }

    pub fn new_state_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<DeepSeekState, Error> {
        match policy {
            CacheResidencyPolicy::Device => self.new_state(),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| unsupported(error.to_string()))?;
                self.paged_state(manager, 0, None)
            }
        }
    }

    fn paged_state(
        &self,
        manager: CacheResidencyManager,
        prefix_tokens: i32,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
    ) -> Result<DeepSeekState, Error> {
        let layout = self.state_layout()?;
        match &self.inner {
            DeepSeekModelInner::V3 { args, .. } => Ok(DeepSeekState {
                inner: DeepSeekStateInner::V3(DeviceState::create(layout, |layer, _| {
                    CompressedLatentCache::new_paged(manager.clone(), layer, rank)
                        .map_err(Error::from)
                })?),
                target_layers: args.num_hidden_layers as usize,
            }),
            DeepSeekModelInner::V4 { args, .. } => {
                let state_args = args.clone();
                Ok(DeepSeekState {
                    inner: DeepSeekStateInner::V4(DeviceState::create(layout, move |layer, _| {
                        let ratio = match state_args.attention_policy(layer) {
                            Some(V4AttentionPolicy::Local) => 0,
                            Some(V4AttentionPolicy::Compressed { ratio }) => ratio,
                            None => return Err(unsupported("missing V4 state attention policy")),
                        };
                        MlxPoolingAttentionCache::paged(
                            ratio,
                            state_args.sliding_window,
                            manager.clone(),
                            layer,
                            prefix_tokens,
                            rank,
                        )
                        .map_err(Error::from)
                    })?),
                    target_layers: args.num_hidden_layers as usize,
                })
            }
        }
    }

    fn run_v3<'a>(
        args: &V3Args,
        execution: &mut V3Execution,
        expert_cache: Option<&ExpertCache>,
        input: deepseek::mtp::EmbeddedInput<'a, crate::MlxTensor>,
        state: &mut V3State,
        pass: eredu_runtime::ExpertPass,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            deepseek::v3::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        if let Some(cache) = expert_cache {
            let mut provider = crate::composition::deepseek_expert::v3_provider(cache, args);
            match execution {
                V3Execution::Resident(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        state,
                        stream,
                        |architecture: &mut V3Architecture,
                         group,
                         index,
                         unit: &mut V3Unit,
                         hidden,
                         state: &mut V3State,
                         forward,
                         context| {
                            architecture.forward_unit_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                            )
                        },
                        |_, _, _| Ok(()),
                    )
                    .map_err(runtime_error),
                V3Execution::Layerwise(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        state,
                        stream,
                        |architecture: &mut V3Architecture,
                         group,
                         index,
                         unit: &mut V3Unit,
                         hidden,
                         state: &mut V3State,
                         forward,
                         context| {
                            architecture.forward_unit_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                            )
                        },
                        |_, _, _| Ok(()),
                    )
                    .map_err(runtime_error),
            }
        } else {
            match execution {
                V3Execution::Resident(runtime) => runtime
                    .forward_with_context_hook(input, state, stream, |_, _, _| Ok(()))
                    .map_err(runtime_error),
                V3Execution::Layerwise(runtime) => runtime
                    .forward_with_context_hook(input, state, stream, |_, _, _| Ok(()))
                    .map_err(runtime_error),
            }
        }
    }

    fn run_v4<'a>(
        args: &V4Args,
        execution: &mut V4Execution,
        expert_cache: Option<&ExpertCache>,
        input: deepseek::mtp::EmbeddedInput<'a, crate::MlxTensor>,
        state: &mut V4State,
        pass: eredu_runtime::ExpertPass,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            deepseek::v4::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        if let Some(cache) = expert_cache {
            let mut provider = crate::composition::deepseek_expert::v4_provider(cache, args);
            match execution {
                V4Execution::Resident(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        state,
                        stream,
                        |architecture: &mut V4Architecture,
                         group,
                         index,
                         unit: &mut V4Unit,
                         hidden,
                         state: &mut V4State,
                         forward,
                         context| {
                            architecture.forward_unit_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                            )
                        },
                        |_, _, _| Ok(()),
                    )
                    .map_err(runtime_error),
                V4Execution::Layerwise(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        state,
                        stream,
                        |architecture: &mut V4Architecture,
                         group,
                         index,
                         unit: &mut V4Unit,
                         hidden,
                         state: &mut V4State,
                         forward,
                         context| {
                            architecture.forward_unit_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                            )
                        },
                        |_, _, _| Ok(()),
                    )
                    .map_err(runtime_error),
            }
        } else {
            match execution {
                V4Execution::Resident(runtime) => runtime
                    .forward_with_context_hook(input, state, stream, |_, _, _| Ok(()))
                    .map_err(runtime_error),
                V4Execution::Layerwise(runtime) => runtime
                    .forward_with_context_hook(input, state, stream, |_, _, _| Ok(()))
                    .map_err(runtime_error),
            }
        }
    }

    fn run_v3_with_provider<'a, P>(
        execution: &mut V3Execution,
        input: deepseek::mtp::EmbeddedInput<'a, crate::MlxTensor>,
        state: &mut V3State,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            deepseek::v3::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    >
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        match execution {
            V3Execution::Resident(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    state,
                    stream,
                    |architecture: &mut V3Architecture,
                     group,
                     index,
                     unit: &mut V3Unit,
                     hidden,
                     state: &mut V3State,
                     forward,
                     context| {
                        architecture.forward_unit_with_provider(
                            group, index, unit, hidden, state, forward, pass, provider, context,
                        )
                    },
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
            V3Execution::Layerwise(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    state,
                    stream,
                    |architecture: &mut V3Architecture,
                     group,
                     index,
                     unit: &mut V3Unit,
                     hidden,
                     state: &mut V3State,
                     forward,
                     context| {
                        architecture.forward_unit_with_provider(
                            group, index, unit, hidden, state, forward, pass, provider, context,
                        )
                    },
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
        }
    }

    fn run_v4_with_provider<'a, P>(
        execution: &mut V4Execution,
        input: deepseek::mtp::EmbeddedInput<'a, crate::MlxTensor>,
        state: &mut V4State,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            deepseek::v4::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    >
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        match execution {
            V4Execution::Resident(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    state,
                    stream,
                    |architecture: &mut V4Architecture,
                     group,
                     index,
                     unit: &mut V4Unit,
                     hidden,
                     state: &mut V4State,
                     forward,
                     context| {
                        architecture.forward_unit_with_provider(
                            group, index, unit, hidden, state, forward, pass, provider, context,
                        )
                    },
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
            V4Execution::Layerwise(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    state,
                    stream,
                    |architecture: &mut V4Architecture,
                     group,
                     index,
                     unit: &mut V4Unit,
                     hidden,
                     state: &mut V4State,
                     forward,
                     context| {
                        architecture.forward_unit_with_provider(
                            group, index, unit, hidden, state, forward, pass, provider, context,
                        )
                    },
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
        }
    }

    fn run_v3_parallel_with_provider<'a, P>(
        execution: &mut V3Execution,
        input: deepseek::mtp::EmbeddedInput<'a, crate::MlxTensor>,
        state: &mut V3State,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        group: &Group,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            deepseek::v3::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    >
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        let hook = |architecture: &mut V3Architecture,
                    group_index,
                    index,
                    unit: &mut V3Unit,
                    hidden: &crate::MlxTensor,
                    state: &mut V3State,
                    forward: &mut deepseek::v3::ForwardContext<crate::MlxTensor>,
                    parallel: &Group,
                    context: &Stream| {
            architecture.forward_unit_parallel_with_provider(
                group_index,
                index,
                unit,
                hidden,
                state,
                forward,
                pass,
                parallel,
                provider,
                context,
            )
        };
        match execution {
            V3Execution::Resident(runtime) => runtime
                .forward_parallel_with_unit_executor_and_context_hook(
                    input,
                    state,
                    group,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
            V3Execution::Layerwise(runtime) => runtime
                .forward_parallel_with_unit_executor_and_context_hook(
                    input,
                    state,
                    group,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
        }
    }

    fn run_v4_parallel_with_provider<'a, P>(
        execution: &mut V4Execution,
        input: deepseek::mtp::EmbeddedInput<'a, crate::MlxTensor>,
        state: &mut V4State,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        group: &Group,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            deepseek::v4::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    >
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        let hook = |architecture: &mut V4Architecture,
                    group_index,
                    index,
                    unit: &mut V4Unit,
                    hidden: &crate::MlxTensor,
                    state: &mut V4State,
                    forward: &mut deepseek::v4::ForwardContext<crate::MlxTensor>,
                    parallel: &Group,
                    context: &Stream| {
            architecture.forward_unit_parallel_with_provider(
                group_index,
                index,
                unit,
                hidden,
                state,
                forward,
                pass,
                parallel,
                provider,
                context,
            )
        };
        match execution {
            V4Execution::Resident(runtime) => runtime
                .forward_parallel_with_unit_executor_and_context_hook(
                    input,
                    state,
                    group,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
            V4Execution::Layerwise(runtime) => runtime
                .forward_parallel_with_unit_executor_and_context_hook(
                    input,
                    state,
                    group,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map_err(runtime_error),
        }
    }

    pub fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        state: &mut DeepSeekState,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                execute,
            );
        match (&mut self.inner, &mut state.inner) {
            (DeepSeekModelInner::V3 { execution, .. }, DeepSeekStateInner::V3(state)) => {
                Self::run_v3_with_provider(
                    execution,
                    deepseek::mtp::EmbeddedInput::target(
                        crate::composition::tensor_ref(tokens),
                        None,
                    ),
                    state,
                    pass,
                    &mut provider,
                    stream,
                )
                .map(|(logits, _)| logits.into_array())
            }
            (DeepSeekModelInner::V4 { execution, .. }, DeepSeekStateInner::V4(state)) => {
                Self::run_v4_with_provider(
                    execution,
                    deepseek::mtp::EmbeddedInput::target(
                        crate::composition::tensor_ref(tokens),
                        None,
                    ),
                    state,
                    pass,
                    &mut provider,
                    stream,
                )
                .map(|(logits, _)| logits.into_array())
            }
            _ => Err(unsupported(
                "DeepSeek model and state families do not match",
            )),
        }
    }

    pub fn forward_tensor_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        state: &mut DeepSeekState,
        tensor_group: &Group,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                execute,
            );
        match (&mut self.inner, &mut state.inner) {
            (
                DeepSeekModelInner::V3 {
                    execution,
                    tensor_parallel: true,
                    ..
                },
                DeepSeekStateInner::V3(state),
            ) => Self::run_v3_parallel_with_provider(
                execution,
                deepseek::mtp::EmbeddedInput::target(crate::composition::tensor_ref(tokens), None),
                state,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )
            .map(|(logits, _)| logits.into_array()),
            (
                DeepSeekModelInner::V4 {
                    execution,
                    tensor_parallel: true,
                    ..
                },
                DeepSeekStateInner::V4(state),
            ) => Self::run_v4_parallel_with_provider(
                execution,
                deepseek::mtp::EmbeddedInput::target(crate::composition::tensor_ref(tokens), None),
                state,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )
            .map(|(logits, _)| logits.into_array()),
            _ => Err(Error::Parallel(
                "DeepSeek model/state was not loaded for tensor plus expert parallelism".into(),
            )),
        }
    }

    /// Runs one target pass and returns complete sequence logits.
    pub fn forward(
        &mut self,
        tokens: &Array,
        state: &mut DeepSeekState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        match (&mut self.inner, &mut state.inner) {
            (
                DeepSeekModelInner::V3 {
                    args,
                    execution,
                    expert_cache,
                    ..
                },
                DeepSeekStateInner::V3(state),
            ) => Self::run_v3(
                args,
                execution,
                expert_cache.as_ref(),
                deepseek::mtp::EmbeddedInput::target(crate::composition::tensor_ref(tokens), None),
                state,
                pass,
                stream,
            )
            .map(|(logits, _)| logits.into_array()),
            (
                DeepSeekModelInner::V4 {
                    args,
                    execution,
                    expert_cache,
                    ..
                },
                DeepSeekStateInner::V4(state),
            ) => Self::run_v4(
                args,
                execution,
                expert_cache.as_ref(),
                deepseek::mtp::EmbeddedInput::target(crate::composition::tensor_ref(tokens), None),
                state,
                pass,
                stream,
            )
            .map(|(logits, _)| logits.into_array()),
            _ => Err(unsupported(
                "DeepSeek model and state families do not match",
            )),
        }
    }

    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        state: &mut DeepSeekState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let mut observer = NeutralDeepSeekObserver { inner: observer };
        let output = match (&mut self.inner, &mut state.inner) {
            (
                DeepSeekModelInner::V3 {
                    args,
                    execution,
                    expert_cache,
                    ..
                },
                DeepSeekStateInner::V3(state),
            ) => if let Some(cache) = expert_cache {
                let mut provider = crate::composition::deepseek_expert::v3_provider(cache, args);
                match execution {
                    V3Execution::Resident(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V3Architecture,
                         group,
                         index,
                         unit: &mut V3Unit,
                         hidden,
                         state: &mut V3State,
                         forward,
                         context| {
                            architecture.forward_unit_observed_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                    V3Execution::Layerwise(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V3Architecture,
                         group,
                         index,
                         unit: &mut V3Unit,
                         hidden,
                         state: &mut V3State,
                         forward,
                         context| {
                            architecture.forward_unit_observed_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                }
            } else {
                match execution {
                    V3Execution::Resident(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V3Architecture,
                         group,
                         index,
                         unit: &mut V3Unit,
                         hidden,
                         state: &mut V3State,
                         forward,
                         context| {
                            architecture.forward_unit_observed(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                    V3Execution::Layerwise(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V3Architecture,
                         group,
                         index,
                         unit: &mut V3Unit,
                         hidden,
                         state: &mut V3State,
                         forward,
                         context| {
                            architecture.forward_unit_observed(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                }
            }
            .map_err(runtime_error)?,
            (
                DeepSeekModelInner::V4 {
                    args,
                    execution,
                    expert_cache,
                    ..
                },
                DeepSeekStateInner::V4(state),
            ) => if let Some(cache) = expert_cache {
                let mut provider = crate::composition::deepseek_expert::v4_provider(cache, args);
                match execution {
                    V4Execution::Resident(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V4Architecture,
                         group,
                         index,
                         unit: &mut V4Unit,
                         hidden,
                         state: &mut V4State,
                         forward,
                         context| {
                            architecture.forward_unit_observed_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                    V4Execution::Layerwise(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V4Architecture,
                         group,
                         index,
                         unit: &mut V4Unit,
                         hidden,
                         state: &mut V4State,
                         forward,
                         context| {
                            architecture.forward_unit_observed_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                pass,
                                &mut provider,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                }
            } else {
                match execution {
                    V4Execution::Resident(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V4Architecture,
                         group,
                         index,
                         unit: &mut V4Unit,
                         hidden,
                         state: &mut V4State,
                         forward,
                         context| {
                            architecture.forward_unit_observed(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                    V4Execution::Layerwise(runtime) => runtime.forward_with_unit_executor(
                        deepseek::mtp::EmbeddedInput::target(
                            crate::composition::tensor_ref(tokens),
                            crate::composition::tensor_opt(mask),
                        ),
                        state,
                        stream,
                        |architecture: &mut V4Architecture,
                         group,
                         index,
                         unit: &mut V4Unit,
                         hidden,
                         state: &mut V4State,
                         forward,
                         context| {
                            architecture.forward_unit_observed(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                context,
                                &mut observer,
                            )
                        },
                    ),
                }
            }
            .map_err(runtime_error)?,
            _ => {
                return Err(unsupported(
                    "DeepSeek model and state families do not match",
                ))
            }
        };
        observer
            .inner
            .observe("model.logits", output.as_array())
            .map_err(Error::from)?;
        Ok(output.into_array())
    }

    /// Returns last-token logits for prefill or decode.
    pub fn next_logits(
        &mut self,
        tokens: &Array,
        state: &mut DeepSeekState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, state, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    fn forward_embedded<'a>(
        &mut self,
        input: deepseek::mtp::EmbeddedInput<'a, Array>,
        state: &mut DeepSeekState,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let pass = match &input {
            deepseek::mtp::EmbeddedInput::Target { tokens, .. }
            | deepseek::mtp::EmbeddedInput::Draft { tokens, .. } => {
                if tokens.dim(1) > 1 {
                    eredu_runtime::ExpertPass::Prefill
                } else {
                    eredu_runtime::ExpertPass::Decode
                }
            }
            deepseek::mtp::EmbeddedInput::DsparkContext { .. } => {
                eredu_runtime::ExpertPass::Prefill
            }
            deepseek::mtp::EmbeddedInput::DsparkProposal { capacity, .. } => {
                if *capacity > 1 {
                    eredu_runtime::ExpertPass::Prefill
                } else {
                    eredu_runtime::ExpertPass::Decode
                }
            }
        };
        let input = neutral_embedded_input(input);
        match (&mut self.inner, &mut state.inner) {
            (
                DeepSeekModelInner::V3 {
                    args,
                    execution,
                    expert_cache,
                    ..
                },
                DeepSeekStateInner::V3(state),
            ) => {
                let (logits, context) = Self::run_v3(
                    args,
                    execution,
                    expert_cache.as_ref(),
                    input,
                    state,
                    pass,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let hidden = context
                    .target_capture()
                    .or_else(|| context.draft_hidden())
                    .cloned()
                    .unwrap_or_else(|| logits.clone());
                Ok((logits.into_array(), hidden.into_array()))
            }
            (
                DeepSeekModelInner::V4 {
                    args,
                    execution,
                    expert_cache,
                    ..
                },
                DeepSeekStateInner::V4(state),
            ) => {
                let (logits, context) = Self::run_v4(
                    args,
                    execution,
                    expert_cache.as_ref(),
                    input,
                    state,
                    pass,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let hidden = context
                    .target_capture()
                    .or_else(|| context.draft_hidden())
                    .cloned()
                    .unwrap_or_else(|| logits.clone());
                Ok((logits.into_array(), hidden.into_array()))
            }
            _ => Err(Exception::custom(
                "DeepSeek embedded model and state families do not match",
            )),
        }
    }

    pub fn forward_embedded_with_expert_executor<'a, F>(
        &mut self,
        input: deepseek::mtp::EmbeddedInput<'a, Array>,
        state: &mut DeepSeekState,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let pass = match &input {
            deepseek::mtp::EmbeddedInput::Target { tokens, .. }
            | deepseek::mtp::EmbeddedInput::Draft { tokens, .. } => {
                if tokens.dim(1) > 1 {
                    eredu_runtime::ExpertPass::Prefill
                } else {
                    eredu_runtime::ExpertPass::Decode
                }
            }
            deepseek::mtp::EmbeddedInput::DsparkContext { .. } => {
                eredu_runtime::ExpertPass::Prefill
            }
            deepseek::mtp::EmbeddedInput::DsparkProposal { capacity, .. } => {
                if *capacity > 1 {
                    eredu_runtime::ExpertPass::Prefill
                } else {
                    eredu_runtime::ExpertPass::Decode
                }
            }
        };
        let input = neutral_embedded_input(input);
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                execute,
            );
        match (&mut self.inner, &mut state.inner) {
            (DeepSeekModelInner::V3 { execution, .. }, DeepSeekStateInner::V3(state)) => {
                let (logits, context) = Self::run_v3_with_provider(
                    execution,
                    input,
                    state,
                    pass,
                    &mut provider,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let hidden = context
                    .target_capture()
                    .or_else(|| context.draft_hidden())
                    .cloned()
                    .unwrap_or_else(|| logits.clone());
                Ok((logits.into_array(), hidden.into_array()))
            }
            (DeepSeekModelInner::V4 { execution, .. }, DeepSeekStateInner::V4(state)) => {
                let (logits, context) = Self::run_v4_with_provider(
                    execution,
                    input,
                    state,
                    pass,
                    &mut provider,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let hidden = context
                    .target_capture()
                    .or_else(|| context.draft_hidden())
                    .cloned()
                    .unwrap_or_else(|| logits.clone());
                Ok((logits.into_array(), hidden.into_array()))
            }
            _ => Err(Exception::custom(
                "DeepSeek embedded model and state families do not match",
            )),
        }
    }

    pub fn forward_embedded_tensor_expert_parallel<'a, F>(
        &mut self,
        input: deepseek::mtp::EmbeddedInput<'a, Array>,
        state: &mut DeepSeekState,
        tensor_group: &Group,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let pass = match &input {
            deepseek::mtp::EmbeddedInput::Target { tokens, .. }
            | deepseek::mtp::EmbeddedInput::Draft { tokens, .. } => {
                if tokens.dim(1) > 1 {
                    eredu_runtime::ExpertPass::Prefill
                } else {
                    eredu_runtime::ExpertPass::Decode
                }
            }
            deepseek::mtp::EmbeddedInput::DsparkContext { .. } => {
                eredu_runtime::ExpertPass::Prefill
            }
            deepseek::mtp::EmbeddedInput::DsparkProposal { capacity, .. } => {
                if *capacity > 1 {
                    eredu_runtime::ExpertPass::Prefill
                } else {
                    eredu_runtime::ExpertPass::Decode
                }
            }
        };
        let input = neutral_embedded_input(input);
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                execute,
            );
        match (&mut self.inner, &mut state.inner) {
            (
                DeepSeekModelInner::V3 {
                    execution,
                    tensor_parallel: true,
                    ..
                },
                DeepSeekStateInner::V3(state),
            ) => {
                let (logits, context) = Self::run_v3_parallel_with_provider(
                    execution,
                    input,
                    state,
                    pass,
                    &mut provider,
                    tensor_group,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let hidden = context
                    .target_capture()
                    .or_else(|| context.draft_hidden())
                    .cloned()
                    .unwrap_or_else(|| logits.clone());
                Ok((logits.into_array(), hidden.into_array()))
            }
            (
                DeepSeekModelInner::V4 {
                    execution,
                    tensor_parallel: true,
                    ..
                },
                DeepSeekStateInner::V4(state),
            ) => {
                let (logits, context) = Self::run_v4_parallel_with_provider(
                    execution,
                    input,
                    state,
                    pass,
                    &mut provider,
                    tensor_group,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let hidden = context
                    .target_capture()
                    .or_else(|| context.draft_hidden())
                    .cloned()
                    .unwrap_or_else(|| logits.clone());
                Ok((logits.into_array(), hidden.into_array()))
            }
            _ => Err(Exception::custom(
                "DeepSeek embedded model/state was not loaded for TP+EP",
            )),
        }
    }

    /// Returns the normalized family model type.
    pub fn model_type(&self) -> &str {
        match &self.inner {
            DeepSeekModelInner::V3 { args, .. } => &args.model_type,
            DeepSeekModelInner::V4 { args, .. } => &args.model_type,
        }
    }

    /// Returns the number of embedded prediction units.
    pub fn mtp_len(&self) -> usize {
        match &self.inner {
            DeepSeekModelInner::V3 { args, .. } => args.num_nextn_predict_layers.max(0) as usize,
            DeepSeekModelInner::V4 { args, .. } => args.num_nextn_predict_layers.max(0) as usize,
        }
    }

    pub fn v3_args(&self) -> Option<&V3Args> {
        match &self.inner {
            DeepSeekModelInner::V3 { args, .. } => Some(args),
            _ => None,
        }
    }

    pub fn v4_args(&self) -> Option<&V4Args> {
        match &self.inner {
            DeepSeekModelInner::V4 { args, .. } => Some(args),
            _ => None,
        }
    }

    pub fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        match &self.inner {
            DeepSeekModelInner::V3 { execution, .. } => match execution {
                V3Execution::Resident(runtime) => runtime.architecture().runtime_state_layout(),
                V3Execution::Layerwise(runtime) => runtime.architecture().runtime_state_layout(),
            }
            .map_err(neutral_error),
            DeepSeekModelInner::V4 { execution, .. } => match execution {
                V4Execution::Resident(runtime) => runtime.architecture().runtime_state_layout(),
                V4Execution::Layerwise(runtime) => runtime.architecture().runtime_state_layout(),
            }
            .map_err(neutral_error),
        }
    }

    pub fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        let (report, materialization) = match &self.inner {
            DeepSeekModelInner::V3 {
                execution,
                materialization,
                ..
            } => (
                match execution {
                    V3Execution::Resident(runtime) => runtime.policy().residency_report()?,
                    V3Execution::Layerwise(runtime) => runtime.policy().residency_report()?,
                },
                materialization,
            ),
            DeepSeekModelInner::V4 {
                execution,
                materialization,
                ..
            } => (
                match execution {
                    V4Execution::Resident(runtime) => runtime.policy().residency_report()?,
                    V4Execution::Layerwise(runtime) => runtime.policy().residency_report()?,
                },
                materialization,
            ),
        };
        Ok(report.with_materialization(materialization.clone()))
    }

    fn set_materialization(&mut self, report: eredu_runtime::WeightMaterializationReport) {
        match &mut self.inner {
            DeepSeekModelInner::V3 {
                materialization, ..
            }
            | DeepSeekModelInner::V4 {
                materialization, ..
            } => *materialization = Some(report),
        }
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.inner {
            DeepSeekModelInner::V3 {
                execution: V3Execution::Layerwise(runtime),
                ..
            } => runtime.policy().dense_stream_report(),
            DeepSeekModelInner::V4 {
                execution: V4Execution::Layerwise(runtime),
                ..
            } => runtime.policy().dense_stream_report(),
            _ => Ok(None),
        }
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.inner {
            DeepSeekModelInner::V3 { execution, .. } => match execution {
                V3Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
                V3Execution::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
            },
            DeepSeekModelInner::V4 { execution, .. } => match execution {
                V4Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
                V4Execution::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
            },
        }
    }

    fn attach_expert_cache(
        &mut self,
        options: eredu_runtime::ExpertCacheLoadOptions,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<(), Error> {
        let store = self.checkpoint_store_arc();
        match &mut self.inner {
            DeepSeekModelInner::V3 {
                args, expert_cache, ..
            } => {
                let entries =
                    crate::composition::deepseek_expert::v3_catalog(args, store.as_ref())?;
                *expert_cache = Some(ExpertCache::new_shared(
                    store,
                    entries,
                    options,
                    weights_stream.clone(),
                    stream.clone(),
                )?);
            }
            DeepSeekModelInner::V4 {
                args, expert_cache, ..
            } => {
                let entries =
                    crate::composition::deepseek_expert::v4_catalog(args, store.as_ref())?;
                *expert_cache = Some(ExpertCache::new_shared(
                    store,
                    entries,
                    options,
                    weights_stream.clone(),
                    stream.clone(),
                )?);
            }
        }
        Ok(())
    }

    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        let cache = match &self.inner {
            DeepSeekModelInner::V3 { expert_cache, .. }
            | DeepSeekModelInner::V4 { expert_cache, .. } => expert_cache,
        };
        cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    pub fn architecture_fingerprint(&self) -> String {
        match &self.inner {
            DeepSeekModelInner::V3 { args, .. } => deepseek::v3_architecture_fingerprint(args),
            DeepSeekModelInner::V4 { args, .. } => deepseek::v4_architecture_fingerprint(args),
        }
    }

    pub fn prompt_cache_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let layout = self.state_layout()?;
        let target = match &self.inner {
            DeepSeekModelInner::V3 { args, .. } => args.num_hidden_layers as usize,
            DeepSeekModelInner::V4 { args, .. } => args.num_hidden_layers as usize,
        };
        let dspark =
            matches!(&self.inner, DeepSeekModelInner::V4 { args, .. } if args.dspark.is_some());
        let offsets = (0..layout.len())
            .map(|layer| if layer >= target && !dspark { -1 } else { 0 })
            .collect();
        let identity = PromptCacheModelIdentity {
            model_family: self.model_type().into(),
            effective_model_type: self.model_type().into(),
            architecture_fingerprint: self.architecture_fingerprint(),
            layer_count: layout.len(),
            global_layer_start: 0,
            global_layer_end: layout.len(),
            sink_tokens: 0,
            layer_prefix_offsets: offsets,
            topology: PromptCacheTopology::default(),
            layer_layout: layout.layers().clone(),
        };
        identity
            .validate()
            .map_err(|error| unsupported(error.to_string()))?;
        Ok(identity)
    }

    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Error> {
        Ok(self.prompt_cache_identity()?.layer_prefix_offsets)
    }

    pub fn save_prompt_cache(
        &self,
        state: &mut DeepSeekState,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        validate_prompt_cache_model_identity(&descriptor, &self.prompt_cache_identity()?)
            .map_err(|error| unsupported(error.to_string()))?;
        match &mut state.inner {
            DeepSeekStateInner::V3(state) => {
                let mut manager = None;
                for layer in state.as_mut() {
                    layer.finalize().map_err(Error::from)?;
                    manager.get_or_insert_with(|| layer.residency_manager().cloned());
                }
                manager
                    .flatten()
                    .ok_or_else(|| unsupported("prompt-cache persistence requires paged V3 state"))?
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
                    .map_err(|error| unsupported(error.to_string()))
            }
            DeepSeekStateInner::V4(state) => {
                let mut manager = None;
                for layer in state.as_mut() {
                    layer.finalize().map_err(Error::from)?;
                    manager.get_or_insert_with(|| layer.residency_manager().cloned());
                }
                let fixed = state
                    .as_ref()
                    .iter()
                    .enumerate()
                    .flat_map(|(layer, cache)| cache.prompt_cache_state_arrays(layer))
                    .collect::<Vec<_>>();
                manager
                    .flatten()
                    .ok_or_else(|| unsupported("prompt-cache persistence requires paged V4 state"))?
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, &fixed, options)
                    .map_err(|error| unsupported(error.to_string()))
            }
        }
    }

    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(DeepSeekState, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_identity()?;
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| unsupported(error.to_string()))?;
        let prefix = i32::try_from(prefix_token_ids.len())
            .map_err(|_| unsupported("prompt length exceeds i32"))?;
        let mut state = self.paged_state(manager, prefix, None)?;
        if let DeepSeekStateInner::V4(state) = &mut state.inner {
            let mut fixed = load_prompt_cache_state_tensors(directory, &manifest, stream)
                .map_err(|error| unsupported(error.to_string()))?
                .into_iter()
                .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
                .collect::<BTreeMap<_, _>>();
            for (layer, cache) in state.as_mut().iter_mut().enumerate() {
                let processed = prefix
                    .checked_add(identity.layer_prefix_offsets[layer])
                    .ok_or_else(|| unsupported("prompt layer offset overflow"))?;
                cache
                    .restore_prompt_cache_state(layer, &mut fixed, processed)
                    .map_err(Error::from)?;
            }
            if !fixed.is_empty() {
                return Err(unsupported(
                    "prompt cache contains unexpected state tensors",
                ));
            }
        }
        Ok((state, manifest))
    }
}

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget for DeepSeekModel {
    type Cache = DeepSeekState;
    type DraftCache = DeepSeekState;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        cache.clear()?;
        let (logits, hidden) = self.forward_embedded(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            cache,
            stream,
        )?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: crate::MlxTensor::from_array(logits),
                hidden: crate::MlxTensor::from_array(hidden),
                tokens: crate::MlxTensor::from_array(tokens),
            },
        )
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let (logits, hidden) = self.forward_embedded(
            deepseek::mtp::EmbeddedInput::target(tokens.as_array(), None),
            cache,
            stream,
        )?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: crate::MlxTensor::from_array(logits),
                hidden: crate::MlxTensor::from_array(hidden),
                tokens: tokens.clone(),
            },
        )
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if matches!(&self.inner, DeepSeekModelInner::V4 { args, .. } if args.dspark.is_some()) {
            let _ = self.forward_embedded(
                deepseek::mtp::EmbeddedInput::dspark_context(output.hidden.as_array()),
                cache,
                stream,
            )?;
            return Ok(());
        }
        let sequence = tokens.as_array().dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = match &self.inner {
            DeepSeekModelInner::V3 { .. } => output
                .hidden
                .as_array()
                .try_index_device((.., ..sequence - 1, ..), stream)?,
            DeepSeekModelInner::V4 { .. } => output
                .hidden
                .as_array()
                .try_index_device((.., ..sequence - 1, .., ..), stream)?,
        };
        let next = tokens.as_array().try_index_device((.., 1..), stream)?;
        for depth in 0..self.mtp_len() {
            let _ = self.forward_embedded(
                deepseek::mtp::EmbeddedInput::draft(&next, &hidden, depth),
                cache,
                stream,
            )?;
        }
        Ok(())
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        cache.clone()
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache
            .commit_prediction_layers_from(draft, draft.target_layers)
            .expect("validated DeepSeek draft and target layouts match");
    }

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_checkpoint(checkpoint, stream)
    }

    fn draft_logits(
        &mut self,
        hidden: &crate::MlxTensor,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, crate::MlxTensor), Exception> {
        if matches!(&self.inner, DeepSeekModelInner::V4 { args, .. } if args.dspark.is_some()) {
            return Err(Exception::custom(
                "DSpark uses fused proposal execution, not sequential prediction layers",
            ));
        }
        let token = Array::from_slice(&[last_token], &[1, 1]);
        self.forward_embedded(
            deepseek::mtp::EmbeddedInput::draft(&token, hidden.as_array(), draft_index),
            cache,
            stream,
        )
        .map(|(logits, hidden)| {
            (
                crate::MlxTensor::from_array(logits),
                crate::MlxTensor::from_array(hidden),
            )
        })
    }

    fn fused_draft_logits(
        &mut self,
        _hidden: &crate::MlxTensor,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<crate::MlxTensor>, Exception> {
        if !matches!(&self.inner, DeepSeekModelInner::V4 { args, .. } if args.dspark.is_some()) {
            return Ok(None);
        }
        let anchor = Array::from_slice(&[last_token], &[1, 1]);
        let mut proposal = cache.clone();
        self.forward_embedded(
            deepseek::mtp::EmbeddedInput::dspark_proposal(&anchor, proposal_capacity),
            &mut proposal,
            stream,
        )
        .map(|(logits, _)| Some(crate::MlxTensor::from_array(logits)))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if matches!(&self.inner, DeepSeekModelInner::V4 { args, .. } if args.dspark.is_some()) {
            let _ = self.forward_embedded(
                deepseek::mtp::EmbeddedInput::dspark_context(hidden.as_array()),
                cache,
                stream,
            )?;
        } else {
            for depth in 0..self.mtp_len() {
                let _ = self.forward_embedded(
                    deepseek::mtp::EmbeddedInput::draft(
                        tokens.as_array(),
                        hidden.as_array(),
                        depth,
                    ),
                    cache,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

impl CausalModel<DeepSeekState> for DeepSeekModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        state: &mut DeepSeekState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.next_logits(&tokens, state, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        state: &mut DeepSeekState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.next_logits(input_tokens.as_array(), state, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

pub fn quantize_v3_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &V3Args,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        V3Args,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    quantization.validate()?;
    let mut target_args = source_args.clone();
    target_args.linear_format = quantization.into();
    target_args.linear_formats.clear();
    target_args
        .validate()
        .map_err(|error| unsupported(error.to_string()))?;
    let source = V3Architecture::new(source_args.clone(), stream).map_err(neutral_error)?;
    let target = V3Architecture::new(target_args.clone(), stream).map_err(neutral_error)?;
    let count = usize::try_from(
        source_args
            .num_hidden_layers
            .checked_add(source_args.num_nextn_predict_layers)
            .ok_or_else(|| unsupported("V3 quantization unit count overflowed"))?,
    )
    .map_err(|_| unsupported("invalid V3 quantization unit count"))?;
    let binding_args = source_args.clone();
    let source_static = MlxModule::new(source.static_modules().clone());
    let target_static = MlxModule::new(target.static_modules().clone());
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |index, stream| construct_v3_unit(&source, index, stream).map(MlxModule::new),
        move |index, stream| construct_v3_unit(&target, index, stream).map(MlxModule::new),
        count,
        quantization,
        stream,
        |modules, store| build_module_bindings(modules, "", store).map_err(Into::into),
        move |index, unit, store| {
            build_module_bindings_with_recipes_excluding(
                unit,
                "",
                store,
                v3_unit_recipes(store, &binding_args, index, false)?,
                |_| false,
            )
            .map_err(Into::into)
        },
    )?;
    Ok((store, target_args, report))
}

pub fn quantize_v4_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &V4Args,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        V4Args,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    quantization.validate()?;
    let mut target_args = source_args.clone();
    target_args.linear_format = quantization.into();
    target_args.linear_formats.clear();
    let total = usize::try_from(
        target_args
            .num_hidden_layers
            .checked_add(target_args.num_nextn_predict_layers)
            .ok_or_else(|| unsupported("V4 quantization unit count overflowed"))?,
    )
    .map_err(|_| unsupported("invalid V4 quantization unit count"))?;
    for layer in 0..total {
        let root = if layer < target_args.num_hidden_layers as usize {
            format!("layers.{layer}.ffn.switch_mlp")
        } else {
            format!(
                "mtp.{}.ffn.switch_mlp",
                layer - target_args.num_hidden_layers as usize
            )
        };
        target_args
            .linear_formats
            .insert(format!("{root}.gate_up_proj"), quantization.into());
        target_args
            .linear_formats
            .insert(format!("{root}.down_proj"), quantization.into());
    }
    target_args
        .validate()
        .map_err(|error| unsupported(error.to_string()))?;
    let source = V4Architecture::new(source_args.clone(), stream).map_err(neutral_error)?;
    let target = V4Architecture::new(target_args.clone(), stream).map_err(neutral_error)?;
    let binding_args = source_args.clone();
    let source_static = MlxModule::new(source.static_modules().clone());
    let target_static = MlxModule::new(target.static_modules().clone());
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |index, stream| construct_v4_unit(&source, index, stream).map(MlxModule::new),
        move |index, stream| construct_v4_unit(&target, index, stream).map(MlxModule::new),
        total,
        quantization,
        stream,
        |modules, store| build_module_bindings(modules, "", store).map_err(Into::into),
        move |index, unit, store| {
            build_module_bindings_with_recipes_excluding(
                unit,
                "",
                store,
                v4_unit_recipes(store, &binding_args, index)?,
                |_| false,
            )
            .map_err(Into::into)
        },
    )?;
    Ok((store, target_args, report))
}

/// Loads a SafeTensors DeepSeek family through the neutral architecture.
pub fn load_safetensors(
    model_dir: &Path,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekModel, Error> {
    load_safetensors_internal(
        model_dir,
        residency,
        quantization,
        false,
        None,
        stream,
        weights_stream,
    )
}

pub fn load_safetensors_external_experts(
    model_dir: &Path,
    residency: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekModel, Error> {
    load_safetensors_internal(
        model_dir,
        WeightResidency::with_layers(residency),
        quantization,
        true,
        None,
        stream,
        weights_stream,
    )
}

pub fn load_safetensors_external_experts_parallel(
    model_dir: &Path,
    residency: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    build: ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekModel, Error> {
    load_safetensors_internal(
        model_dir,
        WeightResidency::with_layers(residency),
        quantization,
        true,
        Some(build),
        stream,
        weights_stream,
    )
}

fn load_safetensors_internal(
    model_dir: &Path,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    force_external_experts: bool,
    parallel: Option<ParallelBuildContext>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekModel, Error> {
    let expert_options = residency.expert_cache();
    let value: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let store = open_safetensors_weight_store(model_dir, residency.max_mapped_shards())?;
    match value.get("model_type").and_then(serde_json::Value::as_str) {
        Some("deepseek_v3") => {
            let args = deepseek::parse_v3_config(&value)
                .map_err(|error| unsupported(error.to_string()))?;
            let plan = deepseek::v3_safetensors_plan(&args, true).map_err(unsupported)?;
            let store = resolve_safetensors_store(store, &plan, &args.model_type)?;
            let (store, args, materialization) = match quantization {
                Some(quantization) => {
                    let (store, args, report) =
                        quantize_v3_store(store, &args, quantization, stream)?;
                    (store, args, Some(report))
                }
                None => (store, args, None),
            };
            let mut model = match parallel {
                Some(build) => DeepSeekModel::load_v3_external_expert_parallel(
                    store,
                    args,
                    residency.layers(),
                    build,
                    stream,
                    weights_stream,
                )?,
                None => DeepSeekModel::load_v3(
                    store,
                    args,
                    residency.layers(),
                    stream,
                    weights_stream,
                    force_external_experts || expert_options.is_some(),
                )?,
            };
            if !force_external_experts {
                if let Some(options) = expert_options {
                    model.attach_expert_cache(options, stream, weights_stream)?;
                }
            }
            if let Some(report) = materialization {
                model.set_materialization(report);
            }
            Ok(model)
        }
        Some("deepseek_v4") => {
            let args = deepseek::parse_v4_config(&value)
                .map_err(|error| unsupported(error.to_string()))?;
            let plan = deepseek::v4_safetensors_plan(&args).map_err(unsupported)?;
            let store = resolve_safetensors_store(store, &plan, &args.model_type)?;
            let (store, args, materialization) = match quantization {
                Some(quantization) => {
                    let (store, args, report) =
                        quantize_v4_store(store, &args, quantization, stream)?;
                    (store, args, Some(report))
                }
                None => (store, args, None),
            };
            let mut model = match parallel {
                Some(build) => DeepSeekModel::load_v4_external_expert_parallel(
                    store,
                    args,
                    residency.layers(),
                    build,
                    stream,
                    weights_stream,
                )?,
                None => DeepSeekModel::load_v4(
                    store,
                    args,
                    residency.layers(),
                    stream,
                    weights_stream,
                    force_external_experts || expert_options.is_some(),
                )?,
            };
            if !force_external_experts {
                if let Some(options) = expert_options {
                    model.attach_expert_cache(options, stream, weights_stream)?;
                }
            }
            if let Some(report) = materialization {
                model.set_materialization(report);
            }
            Ok(model)
        }
        other => Err(unsupported(format!(
            "neutral DeepSeek loader received model_type {other:?}"
        ))),
    }
}

/// Loads a GGUF DeepSeek family through the neutral architecture.
pub fn load_gguf(
    checkpoint: &GgufCheckpoint,
    _metadata: &HashMap<String, GgufMetadataValue>,
    family_v4: bool,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekModel, Vec<u32>), Error> {
    load_gguf_internal(
        checkpoint,
        _metadata,
        family_v4,
        residency,
        false,
        None,
        stream,
        weights_stream,
    )
}

pub fn load_gguf_external_experts(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    family_v4: bool,
    residency: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekModel, Vec<u32>), Error> {
    load_gguf_internal(
        checkpoint,
        metadata,
        family_v4,
        WeightResidency::with_layers(residency),
        true,
        None,
        stream,
        weights_stream,
    )
}

pub fn load_gguf_external_experts_parallel(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    family_v4: bool,
    residency: LayerWeightResidency,
    build: ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekModel, Vec<u32>), Error> {
    load_gguf_internal(
        checkpoint,
        metadata,
        family_v4,
        WeightResidency::with_layers(residency),
        true,
        Some(build),
        stream,
        weights_stream,
    )
}

fn load_gguf_internal(
    checkpoint: &GgufCheckpoint,
    _metadata: &HashMap<String, GgufMetadataValue>,
    family_v4: bool,
    residency: WeightResidency,
    force_external_experts: bool,
    parallel: Option<ParallelBuildContext>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekModel, Vec<u32>), Error> {
    let expert_options = residency.expert_cache();
    let portable_metadata = checkpoint
        .catalog()
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<HashMap<_, _>>();
    let eos = crate::composition::mlx::loading::gguf_eos_token_ids(&portable_metadata)?;
    let options = residency.layers();
    let model = if family_v4 {
        let mut args = deepseek::parse_v4_gguf(&portable_metadata)
            .map_err(|error| unsupported(error.to_string()))?;
        args.linear_formats =
            gguf_quantization_configs(checkpoint, deepseek::translate_v4_gguf_weight_name)?
                .into_iter()
                .map(|(name, format)| (name, format.into()))
                .collect();
        for layer in 0..args.attention_schedule.len() {
            let root = format!("layers.{layer}.ffn");
            if let Some(format) = args
                .linear_formats
                .get(&format!("{root}.expert_banks.w1.weight"))
                .or_else(|| args.linear_formats.get(&format!("{root}.expert_banks.w1")))
                .copied()
            {
                args.linear_formats
                    .insert(format!("{root}.switch_mlp.gate_up_proj"), format);
            }
            if let Some(format) = args
                .linear_formats
                .get(&format!("{root}.expert_banks.w2.weight"))
                .or_else(|| args.linear_formats.get(&format!("{root}.expert_banks.w2")))
                .copied()
            {
                args.linear_formats
                    .insert(format!("{root}.switch_mlp.down_proj"), format);
            }
        }
        let plan = deepseek::v4_gguf_plan(&args).map_err(unsupported)?;
        let store = Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &plan,
            deepseek::translate_v4_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
        match parallel {
            Some(build) => DeepSeekModel::load_v4_external_expert_parallel(
                store,
                args,
                options,
                build,
                stream,
                weights_stream,
            )?,
            None => DeepSeekModel::load_v4(
                store,
                args,
                options,
                stream,
                weights_stream,
                force_external_experts || expert_options.is_some(),
            )?,
        }
    } else {
        let catalog = PortableCatalog(checkpoint.catalog());
        let mut args = deepseek::parse_v3_gguf(&catalog, &portable_metadata)
            .map_err(|error| unsupported(error.to_string()))?;
        args.linear_formats =
            gguf_quantization_configs(checkpoint, deepseek::translate_v3_gguf_weight_name)?
                .into_iter()
                .map(|(name, format)| (name, format.into()))
                .collect();
        for layer in 0..args.layer_schedule.len() {
            let root = format!("model.layers.{layer}.mlp");
            if let Some(format) = args
                .linear_formats
                .get(&format!("{root}.experts.gate_proj"))
                .copied()
            {
                args.linear_formats
                    .insert(format!("{root}.experts.gate_up_proj"), format);
            }
        }
        let plan = deepseek::v3_gguf_plan(&args).map_err(unsupported)?;
        let store = Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &plan,
            deepseek::translate_v3_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
        match parallel {
            Some(build) => DeepSeekModel::load_v3_external_expert_parallel(
                store,
                args,
                options,
                build,
                stream,
                weights_stream,
            )?,
            None => DeepSeekModel::load_v3(
                store,
                args,
                options,
                stream,
                weights_stream,
                force_external_experts || expert_options.is_some(),
            )?,
        }
    };
    let mut model = model;
    if !force_external_experts {
        if let Some(options) = expert_options {
            model.attach_expert_cache(options, stream, weights_stream)?;
        }
    }
    Ok((model, eos))
}

struct PortableCatalog<'a>(&'a eredu_gguf::Checkpoint);

impl deepseek::GgufTensorCatalog for PortableCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0
            .tensors()
            .any(|tensor| tensor.descriptor().name == name)
    }
}

fn resolve_safetensors_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    plan: &eredu_checkpoint::schema::SafetensorsCheckpointPlan,
    identity: &str,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), plan)
        .map_err(|validation| {
            unsupported(format!(
                "{identity} checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn execution_layout<A, S>(architecture: &A) -> Result<ExecutionUnitLayout, Error>
where
    A: LayeredArchitecture<MlxBackend, S, Error = eredu_nn::Error>,
    S: RuntimeState<MlxBackend>,
{
    let graph = architecture.execution_graph().map_err(neutral_error)?;
    let counts = (0..graph.groups().len())
        .map(|group| architecture.group_unit_count(group).map_err(neutral_error))
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts).map_err(|error| unsupported(error.to_string()))
}

fn v3_unit_recipes(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    args: &V3Args,
    layer: usize,
    external_experts: bool,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    let mut recipes = BTreeMap::new();
    let physical = format!("blk.{layer}");
    let logical = format!("model.layers.{layer}.self_attn");
    if store
        .source_metadata(&format!("{logical}.k_b_proj.weight"))
        .is_ok()
    {
        let heads = usize::try_from(args.num_attention_heads)
            .map_err(|_| unsupported("invalid V3 attention-head count"))?;
        let nope = usize::try_from(args.qk_nope_head_dim)
            .map_err(|_| unsupported("invalid V3 non-rotary head width"))?;
        let value = usize::try_from(args.v_head_dim)
            .map_err(|_| unsupported("invalid V3 value-head width"))?;
        let rank =
            usize::try_from(args.kv_lora_rank).map_err(|_| unsupported("invalid V3 KV rank"))?;
        let width = heads
            .checked_mul(
                nope.checked_add(value)
                    .ok_or_else(|| unsupported("V3 KV-B head width overflowed"))?,
            )
            .ok_or_else(|| unsupported("V3 KV-B width overflowed"))?;
        recipes.insert(
            format!("{logical}.kv_b_proj.weight"),
            eredu_checkpoint::recipe::DerivedWeightRecipe::Reshape {
                input: Box::new(eredu_checkpoint::recipe::DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        eredu_checkpoint::recipe::DerivedWeightRecipe::Transpose {
                            input: Box::new(eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                                format!("{logical}.k_b_proj.weight"),
                                eredu_checkpoint::store::TensorSelection::Full,
                            )),
                            axes: vec![0, 2, 1],
                        },
                        eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                            format!("{logical}.v_b_proj.weight"),
                            eredu_checkpoint::store::TensorSelection::Full,
                        ),
                    ],
                }),
                shape: vec![width, rank],
            },
        );
    } else if store
        .source_metadata(&format!("{physical}.attn_k_b.weight"))
        .is_ok()
    {
        recipes.insert(
            format!("model.layers.{layer}.self_attn.kv_b_proj.weight"),
            deepseek::v3_gguf_kv_b_recipe(args, layer, true).map_err(unsupported)?,
        );
    }
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| unsupported("invalid V3 layer count"))?;
    if !external_experts
        && (layer >= target || args.layer_schedule.get(layer) == Some(&LayerPolicy::SparseMoe))
    {
        let expert = deepseek::v3_expert_recipes(store, args, layer).map_err(unsupported)?;
        recipes.insert(expert.target_gate_up, expert.gate_up);
        recipes.insert(expert.target_down, expert.down);
    }
    Ok(recipes)
}

fn v4_unit_recipes(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    args: &V4Args,
    layer: usize,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    let expert = deepseek::v4_expert_recipes(store, args, layer).map_err(unsupported)?;
    Ok(BTreeMap::from([
        (expert.target_gate_up, expert.gate_up),
        (expert.target_down, expert.down),
    ]))
}

/// Builds one unloaded neutral V3 target or prediction unit for a placement
/// owned by a higher-level composition.
pub fn new_v3_unit(
    args: &V3Args,
    ordinal: usize,
    _external_experts: bool,
    stream: &Stream,
) -> Result<MlxModule<V3Unit>, Error> {
    let architecture = V3Architecture::new(args.clone(), stream).map_err(neutral_error)?;
    construct_v3_unit(&architecture, ordinal, stream).map(MlxModule::new)
}

/// Builds exact checkpoint bindings for one neutral V3 execution unit.
pub fn v3_unit_bindings(
    args: &V3Args,
    ordinal: usize,
    unit: &MlxModule<V3Unit>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
) -> Result<Vec<eredu_runtime::WeightBinding>, Error> {
    let expert_targets = parameter_role_targets(
        &deepseek::parallel::v3_layer_parameter_groups(args, ordinal)?,
        ParameterRole::ExpertIntermediate,
    );
    let recipes = v3_unit_recipes(store, args, ordinal, external_experts)?;
    build_module_bindings_with_recipes_excluding(unit, "", store, recipes, |name| {
        external_experts && parameter_name_in_targets(name, &expert_targets)
    })
    .map_err(Into::into)
}

/// Builds one unloaded neutral V4 target or prediction unit for a placement
/// owned by a higher-level composition.
pub fn new_v4_unit(
    args: &V4Args,
    ordinal: usize,
    _external_experts: bool,
    stream: &Stream,
) -> Result<MlxModule<V4Unit>, Error> {
    let architecture = V4Architecture::new(args.clone(), stream).map_err(neutral_error)?;
    construct_v4_unit(&architecture, ordinal, stream).map(MlxModule::new)
}

/// Builds exact checkpoint bindings for one neutral V4 execution unit.
pub fn v4_unit_bindings(
    args: &V4Args,
    ordinal: usize,
    unit: &MlxModule<V4Unit>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
) -> Result<Vec<eredu_runtime::WeightBinding>, Error> {
    let expert_targets = parameter_role_targets(
        &deepseek::parallel::v4_layer_parameter_groups(args, ordinal)?,
        ParameterRole::ExpertIntermediate,
    );
    let recipes = if external_experts {
        BTreeMap::new()
    } else {
        v4_unit_recipes(store, args, ordinal)?
    };
    build_module_bindings_with_recipes_excluding(unit, "", store, recipes, |name| {
        external_experts && parameter_name_in_targets(name, &expert_targets)
    })
    .map_err(Into::into)
}

fn neutral_error(error: eredu_nn::Error) -> Error {
    unsupported(error.to_string())
}

fn runtime_error<A: std::fmt::Display, P: std::fmt::Display>(
    error: eredu_runtime::LayerwiseRuntimeError<A, P>,
) -> Error {
    unsupported(error.to_string())
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::UnsupportedArchitecture(message.into())
}
