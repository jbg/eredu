//! Neutral Gemma 4 binding to MLX storage, state, and residency policy.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_architectures::gemma4::{
    AudioIngressBatchPlan, AudioIngressPartPlan, AudioInput, DecoderInputPart, FamilyConfig,
    LayeredModel as Architecture, ModelInput, Unit, VisionIngressBatchPlan, VisionIngressPartPlan,
    VisionInput,
};
use eredu_architectures::media_plan::Gemma4InputPartPlan;
use eredu_checkpoint::{
    store::{CheckpointSource, SharedCheckpointSource},
    WeightQuantization,
};
use eredu_core::InputMetadataKey;
use eredu_nn::Tensor;
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, ExecutionUnitLayout,
    LayerWeightResidency, LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions,
    ParallelModelInfo, ParameterRole, RuntimeState, StaticUnitBindings, WeightBinding,
    WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{NewAxis, TryIndexOp},
        maximum, pad, GgufCheckpoint, PadWidth,
    },
    Array, Stream,
};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
            },
            state::MlxHybridState,
        },
        checkpoint::{
            binding::{
                binding_bytes, build_module_bindings, build_module_bindings_with_recipes_excluding,
                materialize_module_bindings, parameter_name_in_targets, parameter_role_targets,
                populate_module_from_arrays_excluding, populate_module_from_lease_excluding,
            },
            load::{gguf_metadata, gguf_quantization_configs},
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                construct_architecture_unit, prepare_layerwise_policy_with_bindings,
                MlxLayerwisePolicy, MlxResidentPolicy, MlxUnitPopulator,
            },
            layerwise::{
                open_safetensors_weight_store, quantize_parameterized_module_store,
                quantize_parameterized_store, shard_layer_bindings,
            },
        },
        media::input,
        residency::expert_cache::{ExpertCache, ExpertCacheReport},
    },
};

type NeutralArchitecture = Architecture<MlxNeuralBackend>;
type NeutralUnit = Unit<MlxNeuralBackend>;
type NeutralAssistant = eredu_architectures::gemma4::Assistant<MlxNeuralBackend>;
pub type Gemma4PipelineUnit = MlxModule<NeutralUnit>;

fn group_kind(
    architecture: &NeutralArchitecture,
    group: usize,
) -> eredu_runtime::ArchitectureGroupKind {
    <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_transport(
        architecture,
        group,
    )
    .kind
}
/// Binding-only helper for Gemma 4 pipeline checkpoint materialization.
pub struct Gemma4Bindings {
    external_experts: bool,
}

impl Gemma4Bindings {
    pub const fn new(external_experts: bool) -> Self {
        Self { external_experts }
    }

    pub fn model_type<'a>(&self, architecture: &'a NeutralArchitecture) -> &'a str {
        &architecture.args().model_type
    }

    pub fn static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub fn layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        layer: &Gemma4PipelineUnit,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let is_decoder =
            group_kind(architecture, group) == eredu_runtime::ArchitectureGroupKind::Decoder;
        let expert_targets = if is_decoder {
            parameter_role_targets(
                &eredu_architectures::gemma4::layer_parameter_groups(
                    &architecture.args().text,
                    index,
                )?,
                ParameterRole::ExpertIntermediate,
            )
        } else {
            Default::default()
        };
        let recipes = if !self.external_experts {
            eredu_architectures::gemma4::unit_recipes(
                store,
                architecture.args(),
                execution_ordinal(architecture, group, index)?,
            )
            .map_err(Error::ArchitectureModel)?
        } else {
            BTreeMap::new()
        };
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && parameter_name_in_targets(name, &expert_targets)
        })
        .map_err(Into::into)
    }

    pub fn cartesian_layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        global_layer: &Gemma4PipelineUnit,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Vec<WeightBinding>, Error> {
        let is_decoder =
            group_kind(architecture, group) == eredu_runtime::ArchitectureGroupKind::Decoder;
        let expert_targets = if is_decoder {
            parameter_role_targets(
                &eredu_architectures::gemma4::layer_parameter_groups(
                    &architecture.args().text,
                    index,
                )?,
                ParameterRole::ExpertIntermediate,
            )
        } else {
            Default::default()
        };
        let recipes = if !self.external_experts {
            eredu_architectures::gemma4::unit_recipes(
                store,
                architecture.args(),
                execution_ordinal(architecture, group, index)?,
            )
            .map_err(Error::ArchitectureModel)?
        } else {
            BTreeMap::new()
        };
        let bindings = build_module_bindings_with_recipes_excluding(
            global_layer,
            "",
            store,
            recipes,
            |name| self.external_experts && parameter_name_in_targets(name, &expert_targets),
        )?;
        match (is_decoder, layout) {
            (true, Some(layout)) => {
                let root = <NeutralArchitecture as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::unit_path(architecture, group, index)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                shard_layer_bindings(bindings, &root, store, layout)
            }
            _ => Ok(bindings),
        }
    }
}
type Resident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
>;
type Bounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralUnit, UnitPopulator>,
>;
type ParallelResident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
>;
type ParallelBounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralUnit, UnitPopulator>,
>;

#[derive(Clone)]
struct UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<std::collections::BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralUnit> for UnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralUnit>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

enum Execution {
    Resident(Resident),
    Bounded(Bounded),
    ParallelResident(Box<ParallelResident>),
    ParallelBounded(Box<ParallelBounded>),
}

impl Execution {
    fn output_group(&self) -> Result<usize, Error> {
        let architecture = match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Bounded(runtime) => runtime.architecture(),
            Self::ParallelResident(runtime) => runtime.architecture(),
            Self::ParallelBounded(runtime) => runtime.architecture(),
        };
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::execution_graph(
            architecture,
        )
        .map(|graph| graph.output())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_external_experts<P>(
    architecture: &mut NeutralArchitecture,
    group: usize,
    index: usize,
    unit: &mut NeutralUnit,
    hidden: &crate::MlxTensor,
    state: &mut MlxHybridState,
    forward: &mut eredu_architectures::gemma4::ForwardContext<crate::MlxTensor>,
    stream: &Stream,
    provider: &mut P,
) -> Result<crate::MlxTensor, eredu_nn::Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::forward_unit_with_provider(
        architecture,
        group,
        index,
        unit,
        hidden,
        state,
        forward,
        if hidden.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        },
        provider,
        stream,
    )
}

/// One neutral Gemma 4 object shared by resident and bounded execution.
pub struct Gemma4Model {
    args: FamilyConfig,
    state_layout: eredu_runtime::StateLayout,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    execution: Execution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::MlxParallelContext>>,
}

/// Fully resident external assistant built from the neutral Gemma equations.
pub struct Gemma4AssistantModel {
    pub config: eredu_architectures::gemma4::AssistantConfig,
    module: MlxModule<NeutralAssistant>,
}

impl Gemma4AssistantModel {
    pub fn max_proposals(&self) -> usize {
        self.module.max_proposals()
    }

    pub fn begin_round(
        &self,
        shared_kv: eredu_architectures::gemma4::SharedAttentionStates<crate::MlxTensor>,
        kv_offset: i32,
        hidden: crate::MlxTensor,
    ) -> eredu_architectures::gemma4::AssistantState<crate::MlxTensor> {
        self.module.begin_round(shared_kv, kv_offset, hidden)
    }

    pub fn draft_step(
        &mut self,
        embedding: &crate::MlxTensor,
        state: &mut eredu_architectures::gemma4::AssistantState<crate::MlxTensor>,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.module
            .draft_step::<crate::backend::runtime::cache::kv::ConcatKeyValueCache>(
                embedding, state, stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

/// Loads the released SafeTensors assistant into the backend-neutral module.
pub fn load_assistant_safetensors(
    model_dir: &Path,
    source_config: eredu_architectures::gemma4::AssistantConfig,
    options: crate::backend::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::ArchitectureModel(
            "Gemma 4 assistant loading supports fully resident weights only".into(),
        ));
    }
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Gemma 4 assistant loading requires replicated placement".into(),
        ));
    }
    let requested = options
        .quantization
        .map(|requested| {
            should_quantize_on_load("Gemma 4 assistant", source_config.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let config = requested
        .map(|requested| {
            source_config
                .load_time_quantization(requested)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .transpose()?
        .unwrap_or_else(|| source_config.clone());
    let store =
        open_safetensors_weight_store(model_dir, options.weight_residency.max_mapped_shards())?;
    let store = if let Some(requested) = requested {
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0
    } else {
        store
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(Gemma4AssistantModel { config, module })
}

pub fn load_assistant_gguf(
    checkpoint: eredu_gguf::Checkpoint,
    resolution: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    source_config: eredu_architectures::gemma4::AssistantConfig,
    options: crate::backend::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::ArchitectureModel(
            "Gemma 4 assistant loading supports fully resident weights only".into(),
        ));
    }
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Gemma 4 assistant loading requires replicated placement".into(),
        ));
    }
    let mlx_checkpoint = GgufCheckpoint::from_portable(checkpoint.clone());
    let metadata = gguf_metadata(&mlx_checkpoint);
    let formats = gguf_quantization_configs(
        &mlx_checkpoint,
        eredu_architectures::gemma4::translate_assistant_gguf_weight_name,
    )?;
    let source_config = source_config
        .with_checkpoint_formats(formats)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    crate::composition::mlx::validate_gguf_quantization_source(
        &mlx_checkpoint,
        &metadata,
        options.quantization,
    )?;
    let store: SharedCheckpointSource = Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(options.weight_residency.max_mapped_shards())?
            .add_resolved_checkpoint(checkpoint, &resolution, |name| {
                eredu_architectures::gemma4::translate_assistant_gguf_weight_name(name)
            })?
            .build()?,
    );
    let (store, config) = if let Some(requested) = options.quantization {
        let config = source_config
            .load_time_quantization(requested)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        (
            quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0,
            config,
        )
    } else {
        (store, source_config)
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(Gemma4AssistantModel { config, module })
}

/// Ordinary target outputs retained by the neutral speculative adapter.
pub struct Gemma4MtpOutput {
    pub logits: crate::MlxTensor,
    pub hidden: crate::MlxTensor,
    pub shared_kv: eredu_architectures::gemma4::SharedAttentionStates<crate::MlxTensor>,
}

impl Gemma4Model {
    pub fn args(&self) -> &FamilyConfig {
        &self.args
    }

    pub fn metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    pub fn parallel_info(&self) -> Option<&ParallelModelInfo<crate::backend::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone())
            .expect("validated Gemma 4 state must be realizable")
    }

    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxHybridState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let rank = self.parallel_info.as_ref().and_then(|info| {
                    crate::backend::cache::prompt_cache_topology(info.topology())
                        .cache_rank_identity()
                });
                MlxHybridState::paged(
                    self.state_layout.clone(),
                    CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?,
                    rank,
                )
                .map_err(Into::into)
            }
        }
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let topology = self
            .parallel_info
            .as_ref()
            .map_or_else(eredu_core::cache::PromptCacheTopology::default, |info| {
                crate::backend::cache::prompt_cache_topology(info.topology())
            });
        eredu_architectures::gemma4::state_identity(&self.args, &self.state_layout, 0, topology)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .prompt_cache_identity(&self.state_layout)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MlxHybridState, eredu_core::cache::PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        let rank = identity.topology.cache_rank_identity();
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory.as_ref(), &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut state = MlxHybridState::paged(self.state_layout.clone(), manager, rank)?;
        let processed = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("prompt-cache prefix length exceeds i32".into()))?;
        state.restore_prompt_cache_state(tensors, processed, &identity.layer_prefix_offsets)?;
        Ok((state, manifest))
    }

    pub fn save_prompt_cache(
        &self,
        state: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &eredu_core::cache::PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<eredu_core::cache::PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(
            &descriptor,
            &self.prompt_cache_model_identity()?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        state
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        let report = match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report()?,
            Execution::Bounded(runtime) => runtime.policy().residency_report()?,
            Execution::ParallelResident(runtime) => runtime.policy().residency_report()?,
            Execution::ParallelBounded(runtime) => runtime.policy().residency_report()?,
        };
        Ok(Some(report))
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) | Execution::ParallelResident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
            Execution::ParallelBounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ParallelResident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ParallelBounded(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    fn forward_with_capture(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            eredu_architectures::gemma4::ForwardContext<crate::MlxTensor>,
            crate::MlxTensor,
        ),
        Error,
    > {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "Gemma 4 tensor-parallel execution requires a collective session".into(),
            ));
        }
        if state.layout() != &self.state_layout {
            return Err(Error::ArchitectureModel(
                "Gemma 4 cache layout mismatch".into(),
            ));
        }
        let output_group = self.execution.output_group()?;
        let mut final_text_hidden = None;
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.text.clone();
            let mut provider =
                crate::composition::gemma4_expert::cached_provider(&expert_cache, &args);
            let result = match &mut self.execution {
                Execution::Resident(runtime) => runtime
                    .forward_with_unit_executor_and_activation_hook(
                        input,
                        state,
                        stream,
                        |architecture, group, index, unit, hidden, state, forward, stream| {
                            forward_external_experts(
                                architecture,
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut provider,
                            )
                        },
                        |group, _index, hidden, _forward| {
                            if group == output_group {
                                final_text_hidden = Some(hidden.clone());
                            }
                            Ok(())
                        },
                    ),
                Execution::Bounded(runtime) => runtime
                    .forward_with_unit_executor_and_activation_hook(
                        input,
                        state,
                        stream,
                        |architecture, group, index, unit, hidden, state, forward, stream| {
                            forward_external_experts(
                                architecture,
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut provider,
                            )
                        },
                        |group, _index, hidden, _forward| {
                            if group == output_group {
                                final_text_hidden = Some(hidden.clone());
                            }
                            Ok(())
                        },
                    ),
                Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
            };
            drop(provider);
            self.expert_cache = Some(expert_cache);
            let (logits, forward) =
                result.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let hidden = final_text_hidden.ok_or_else(|| {
                Error::ArchitectureModel("Gemma 4 text graph produced no activation".into())
            })?;
            return Ok((logits, forward, hidden));
        }
        let result = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward_with_unit_executor_and_activation_hook(
                input,
                state,
                stream,
                |architecture, group, index, unit, hidden, state, forward, stream| {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
                },
                |group, _index, hidden, _forward| {
                    if group == output_group {
                        final_text_hidden = Some(hidden.clone());
                    }
                    Ok(())
                },
            ),
            Execution::Bounded(runtime) => runtime.forward_with_unit_executor_and_activation_hook(
                input,
                state,
                stream,
                |architecture, group, index, unit, hidden, state, forward, stream| {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
                },
                |group, _index, hidden, _forward| {
                    if group == output_group {
                        final_text_hidden = Some(hidden.clone());
                    }
                    Ok(())
                },
            ),
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let hidden = final_text_hidden.ok_or_else(|| {
            Error::ArchitectureModel("Gemma 4 text graph produced no activation".into())
        })?;
        Ok((result.0, result.1, hidden))
    }

    fn forward(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.forward_with_capture(input, state, stream)
            .map(|(logits, _, _)| logits)
    }

    pub fn embed_mtp_token(
        &mut self,
        token: u32,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "Gemma 4 assistant embedding is unavailable in tensor-parallel execution".into(),
            ));
        }
        let tokens = crate::MlxTensor::from_array(Array::from_slice(&[token], &[1, 1]));
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.architecture_mut().token_embeddings(&tokens, stream)
            }
            Execution::Bounded(runtime) => {
                runtime.architecture_mut().token_embeddings(&tokens, stream)
            }
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn mtp_output(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Gemma4MtpOutput, Error> {
        let (logits, forward, hidden) = self.forward_with_capture(input, state, stream)?;
        Ok(Gemma4MtpOutput {
            logits,
            hidden,
            shared_kv: forward.shared_attention_states().clone(),
        })
    }

    pub fn prefill_mtp(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Gemma4MtpOutput, Error> {
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        self.mtp_output(
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }

    pub fn verify_mtp(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Gemma4MtpOutput, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.mtp_output(
            ModelInput {
                parts: &parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }

    pub fn forward_tokens(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward(
            ModelInput {
                parts: &parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }

    pub fn forward_tensor_parallel(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Gemma 4 tensor-parallel cache layout mismatch".into(),
            ));
        }
        let parts = [DecoderInputPart::Text(tokens)];
        let input = ModelInput {
            parts: &parts,
            vision: None,
            audio: None,
            per_layer_tokens: None,
            mask: None,
        };
        self.forward_parallel_input(input, state, group, stream)
    }

    pub fn prefill_tensor_parallel(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Gemma 4 tensor-parallel cache layout mismatch".into(),
            ));
        }
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        let input = ModelInput {
            parts: &parts,
            vision: prepared.vision_input(),
            audio: prepared.audio_input(),
            per_layer_tokens: None,
            mask: None,
        };
        self.forward_parallel_input(input, state, group, stream)
    }

    fn forward_parallel_input(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.text.clone();
            let mut provider =
                crate::composition::gemma4_expert::cached_provider(&expert_cache, &args);
            let result = match &mut self.execution {
                Execution::ParallelResident(runtime) => runtime
                    .forward_parallel_with_unit_executor(
                        input,
                        state,
                        group,
                        stream,
                        |architecture,
                         execution_group,
                         index,
                         unit,
                         hidden,
                         state,
                         forward,
                         parallel,
                         stream| {
                            <NeutralArchitecture as eredu_runtime::ParallelRoutedLayeredArchitecture<
                                MlxNeuralBackend,
                                MlxHybridState,
                            >>::forward_unit_parallel_with_provider(
                                architecture,
                                execution_group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                if hidden.dim(1) > 1 {
                                    eredu_runtime::ExpertPass::Prefill
                                } else {
                                    eredu_runtime::ExpertPass::Decode
                                },
                                &mut provider,
                                parallel,
                                stream,
                            )
                        },
                    ),
                Execution::ParallelBounded(runtime) => runtime.forward_parallel_with_unit_executor(
                    input,
                    state,
                    group,
                    stream,
                    |architecture,
                     execution_group,
                     index,
                     unit,
                     hidden,
                     state,
                     forward,
                     parallel,
                     stream| {
                        <NeutralArchitecture as eredu_runtime::ParallelRoutedLayeredArchitecture<
                            MlxNeuralBackend,
                            MlxHybridState,
                        >>::forward_unit_parallel_with_provider(
                            architecture,
                            execution_group,
                            index,
                            unit,
                            hidden,
                            state,
                            forward,
                            if hidden.dim(1) > 1 {
                                eredu_runtime::ExpertPass::Prefill
                            } else {
                                eredu_runtime::ExpertPass::Decode
                            },
                            &mut provider,
                            parallel,
                            stream,
                        )
                    },
                ),
                Execution::Resident(_) | Execution::Bounded(_) => {
                    drop(provider);
                    self.expert_cache = Some(expert_cache);
                    return Err(Error::Parallel(
                        "Gemma 4 model was not loaded for tensor parallelism".into(),
                    ));
                }
            };
            drop(provider);
            self.expert_cache = Some(expert_cache);
            return result.map_err(|error| Error::Parallel(error.to_string()));
        }
        match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel(input, state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel(input, state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::Resident(_) | Execution::Bounded(_) => Err(Error::Parallel(
                "Gemma 4 model was not loaded for tensor parallelism".into(),
            )),
        }
    }

    pub fn forward_input(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        self.forward(
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }
}

struct PreparedVision {
    patches: crate::MlxTensor,
    positions: crate::MlxTensor,
    valid: crate::MlxTensor,
    key_mask: crate::MlxTensor,
    grid_extents: Vec<(i32, i32)>,
}

struct PreparedVisionPart {
    patches: crate::MlxTensor,
    positions: crate::MlxTensor,
    plan: VisionIngressPartPlan,
}

struct PreparedAudio {
    features: crate::MlxTensor,
    input_mask: crate::MlxTensor,
    first_stage_mask: crate::MlxTensor,
    valid: Vec<i32>,
}

struct PreparedAudioPart {
    features: crate::MlxTensor,
    mask: crate::MlxTensor,
    plan: AudioIngressPartPlan,
}

pub struct PreparedParts {
    tokens: Vec<crate::MlxTensor>,
    modalities: Vec<input::Modality>,
    projected: Vec<Option<crate::MlxTensor>>,
    vision_parts: Vec<PreparedVisionPart>,
    vision: Option<PreparedVision>,
    audio_parts: Vec<PreparedAudioPart>,
    audio: Option<PreparedAudio>,
}

impl PreparedParts {
    pub fn new(
        args: &FamilyConfig,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut prepared = Self {
            tokens: Vec::with_capacity(typed.parts.len()),
            modalities: Vec::with_capacity(typed.parts.len()),
            projected: Vec::with_capacity(typed.parts.len()),
            vision_parts: Vec::new(),
            vision: None,
            audio_parts: Vec::new(),
            audio: None,
        };
        for part in typed.parts {
            let plan = eredu_architectures::media_plan::gemma4_input_part(
                args,
                part,
                &input::MlxInputInspector,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            match plan {
                Gemma4InputPartPlan::TextTokens { .. } => {
                    let input::InputPayload::TokenIds(tokens) = part.payload() else {
                        unreachable!("architecture-admitted Gemma text payload")
                    };
                    prepared
                        .tokens
                        .push(crate::MlxTensor::from_array(tokens.clone()));
                    prepared.modalities.push(input::Modality::Text);
                    prepared.projected.push(None);
                }
                Gemma4InputPartPlan::Projected {
                    placeholder_token_id,
                    positions,
                    ..
                } => {
                    let input::InputPayload::Embeddings(embeddings) = part.payload() else {
                        unreachable!("architecture-admitted Gemma projected payload")
                    };
                    let positions = usize::try_from(positions).map_err(|_| {
                        Error::ArchitectureModel(
                            "Gemma projected sequence exceeds host capacity".into(),
                        )
                    })?;
                    prepared
                        .tokens
                        .push(crate::MlxTensor::from_array(Array::from_slice(
                            &vec![placeholder_token_id; positions],
                            &[1, embeddings.dim(1)],
                        )));
                    prepared.modalities.push(part.modality());
                    prepared
                        .projected
                        .push(Some(crate::MlxTensor::from_array(embeddings.clone())));
                }
                Gemma4InputPartPlan::Vision {
                    placeholder_token_id,
                    ingress,
                    ..
                } => {
                    let input::InputPayload::Tensor(patches) = part.payload() else {
                        unreachable!("architecture-admitted Gemma vision payload")
                    };
                    let positions = part
                        .metadata_value(InputMetadataKey::PatchPositions)
                        .expect("architecture-admitted Gemma patch positions");
                    prepared.push_vision(
                        part.modality(),
                        patches,
                        positions,
                        ingress,
                        placeholder_token_id,
                    )?;
                }
                Gemma4InputPartPlan::Audio {
                    placeholder_token_id,
                    ingress,
                    ..
                } => {
                    let input::InputPayload::Tensor(features) = part.payload() else {
                        unreachable!("architecture-admitted Gemma audio payload")
                    };
                    let mask = part
                        .metadata_value(InputMetadataKey::AudioMask)
                        .expect("architecture-admitted Gemma audio mask");
                    prepared.push_audio(features, mask, ingress, placeholder_token_id)?;
                }
            }
        }
        prepared.finish_vision(stream)?;
        prepared.finish_audio(stream)?;
        Ok(prepared)
    }

    fn push_vision(
        &mut self,
        modality: input::Modality,
        patches: &Array,
        positions: &Array,
        plan: VisionIngressPartPlan,
        placeholder_token_id: u32,
    ) -> Result<(), Error> {
        self.tokens
            .push(crate::MlxTensor::from_array(Array::from_slice(
                &vec![placeholder_token_id; plan.decoder_positions as usize],
                &[1, plan.decoder_positions],
            )));
        self.modalities.push(modality);
        self.projected.push(None);

        self.vision_parts.push(PreparedVisionPart {
            patches: crate::MlxTensor::from_array(patches.clone()),
            positions: crate::MlxTensor::from_array(positions.clone()),
            plan,
        });
        Ok(())
    }

    fn finish_vision(&mut self, stream: &Stream) -> Result<(), Error> {
        if self.vision_parts.is_empty() {
            return Ok(());
        }
        let plan = VisionIngressBatchPlan::new(
            &self
                .vision_parts
                .iter()
                .map(|part| part.plan.clone())
                .collect::<Vec<_>>(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let max_patches = plan.padded_patches;
        let patches = self
            .vision_parts
            .iter()
            .map(|part| pad_sequence(part.patches.as_array(), max_patches, 0, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let positions = self
            .vision_parts
            .iter()
            .map(|part| pad_sequence(part.positions.as_array(), max_patches, -1, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let patch_refs = patches.iter().collect::<Vec<_>>();
        let position_refs = positions.iter().collect::<Vec<_>>();
        let patches = concatenate_axis(&patch_refs, 0, stream)?;
        let positions = concatenate_axis(&position_refs, 0, stream)?;
        let sanitized = maximum(positions.clone(), Array::from_int(0), stream)?;
        self.vision = Some(PreparedVision {
            patches: crate::MlxTensor::from_array(patches),
            positions: crate::MlxTensor::from_array(sanitized),
            valid: crate::MlxTensor::from_array(Array::from_slice(
                &plan.position_valid_values,
                &plan.position_valid_shape(),
            )),
            key_mask: crate::MlxTensor::from_array(Array::from_slice(
                &plan.key_mask_values,
                &plan.key_mask_shape(),
            )),
            grid_extents: plan.grid_extents,
        });
        Ok(())
    }

    fn push_audio(
        &mut self,
        features: &Array,
        mask: &Array,
        plan: AudioIngressPartPlan,
        placeholder_token_id: u32,
    ) -> Result<(), Error> {
        self.tokens
            .push(crate::MlxTensor::from_array(Array::from_slice(
                &vec![placeholder_token_id; plan.decoder_positions as usize],
                &[1, plan.decoder_positions],
            )));
        self.modalities.push(input::Modality::Audio);
        self.projected.push(None);

        self.audio_parts.push(PreparedAudioPart {
            features: crate::MlxTensor::from_array(features.clone()),
            mask: crate::MlxTensor::from_array(mask.clone()),
            plan,
        });
        Ok(())
    }

    fn finish_audio(&mut self, stream: &Stream) -> Result<(), Error> {
        if self.audio_parts.is_empty() {
            return Ok(());
        }
        let plan = AudioIngressBatchPlan::new(
            &self
                .audio_parts
                .iter()
                .map(|part| part.plan.clone())
                .collect::<Vec<_>>(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let max_frames = plan.padded_frames;
        let features = self
            .audio_parts
            .iter()
            .map(|part| pad_sequence(part.features.as_array(), max_frames, 0, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let masks = self
            .audio_parts
            .iter()
            .map(|part| {
                let extra = max_frames - part.mask.dim(1);
                if extra == 0 {
                    Ok(part.mask.as_array().clone())
                } else {
                    Ok(pad(
                        part.mask.as_array(),
                        PadWidth::from(&[(0, 0), (0, extra)][..]),
                        Array::from_bool(false),
                        None,
                        stream,
                    )?)
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let feature_refs = features.iter().collect::<Vec<_>>();
        let mask_refs = masks.iter().collect::<Vec<_>>();
        let features = concatenate_axis(&feature_refs, 0, stream)?;
        let mask = concatenate_axis(&mask_refs, 0, stream)?;
        let input_mask = mask
            .as_dtype(features.dtype(), stream)?
            .try_index_device((.., .., NewAxis), stream)?;
        self.audio = Some(PreparedAudio {
            features: crate::MlxTensor::from_array(features),
            input_mask: crate::MlxTensor::from_array(input_mask),
            first_stage_mask: crate::MlxTensor::from_array(Array::from_slice(
                &plan.first_stage_mask_values,
                &plan.first_stage_mask_shape(),
            )),
            valid: plan.valid_subsampled_frames,
        });
        Ok(())
    }

    pub fn decoder_parts(&self) -> Vec<DecoderInputPart<'_, crate::MlxTensor>> {
        self.tokens
            .iter()
            .zip(&self.modalities)
            .zip(&self.projected)
            .map(|((tokens, modality), embeddings)| {
                if let Some(embeddings) = embeddings {
                    DecoderInputPart::Projected { tokens, embeddings }
                } else {
                    match modality {
                        input::Modality::Text => DecoderInputPart::Text(tokens),
                        input::Modality::Image => DecoderInputPart::Image(tokens),
                        input::Modality::Video => DecoderInputPart::Video(tokens),
                        input::Modality::Audio => DecoderInputPart::Audio(tokens),
                    }
                }
            })
            .collect()
    }

    pub fn vision_input(&self) -> Option<VisionInput<'_, crate::MlxTensor>> {
        self.vision.as_ref().map(|vision| VisionInput {
            patches: &vision.patches,
            position_ids: &vision.positions,
            position_valid: &vision.valid,
            key_mask: &vision.key_mask,
            grid_extents: &vision.grid_extents,
        })
    }

    pub fn audio_input(&self) -> Option<AudioInput<'_, crate::MlxTensor>> {
        self.audio.as_ref().map(|audio| AudioInput {
            features: &audio.features,
            input_mask: &audio.input_mask,
            first_stage_mask: &audio.first_stage_mask,
            valid_subsampled_frames: &audio.valid,
        })
    }
}

fn pad_sequence(value: &Array, sequence: i32, fill: i32, stream: &Stream) -> Result<Array, Error> {
    let extra = sequence - value.dim(1);
    if extra < 0 {
        return Err(Error::ArchitectureModel(
            "prepared media sequence exceeds its batch padding extent".into(),
        ));
    }
    if extra == 0 {
        return Ok(value.clone());
    }
    Ok(pad(
        value,
        PadWidth::from(&[(0, 0), (0, extra), (0, 0)][..]),
        Array::from_int(fill).as_dtype(value.dtype(), stream)?,
        None,
        stream,
    )?)
}

impl CausalModel<MlxHybridState> for Gemma4Model {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let logits = self
            .forward_input(input, state, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        logits
            .as_array()
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }

    fn decode_logits(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let logits = self
            .forward_tokens(tokens, state, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        logits
            .as_array()
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }
}

fn execution_layout(architecture: &NeutralArchitecture) -> Result<ExecutionUnitLayout, Error> {
    let graph =
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::execution_graph(
            architecture,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_unit_count(
                architecture,
                group,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

fn execution_ordinal(
    architecture: &NeutralArchitecture,
    group: usize,
    index: usize,
) -> Result<usize, Error> {
    execution_layout(architecture)?
        .ordinal(group, index)
        .ok_or_else(|| Error::Parallel(format!("Gemma 4 has no unit {index} in group {group}")))
}

fn quantize_store(
    store: SharedCheckpointSource,
    source: &FamilyConfig,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        SharedCheckpointSource,
        FamilyConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target = eredu_architectures::gemma4::load_time_quantization(source, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = execution_layout(&source_architecture)?;
    let target_layout = execution_layout(&target_architecture)?;
    if source_layout.len() != target_layout.len() {
        return Err(Error::Quantization(
            "Gemma 4 quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = <NeutralArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::static_modules(&source_architecture)
    .clone();
    let target_static = <NeutralArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::static_modules(&target_architecture)
    .clone();
    let (store, report) = quantize_parameterized_store(
        store,
        &source_static,
        &target_static,
        move |index, stream| {
            construct_architecture_unit(
                &source_architecture,
                &source_layout,
                index,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        move |index, stream| {
            construct_architecture_unit(
                &target_architecture,
                &target_layout,
                index,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

fn load_store(
    store: SharedCheckpointSource,
    args: FamilyConfig,
    residency: eredu_runtime::LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<Gemma4Model, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let external_expert_source_keys = if external_experts {
        let mut keys = BTreeSet::new();
        for layer in 0..args.text.num_hidden_layers() {
            if args.text.layer_policy(layer).is_none_or(|policy| {
                policy.feed_forward
                    != eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
            }) {
                continue;
            }
            let resolved =
                eredu_architectures::gemma4::expert_recipes(store.as_ref(), &args.text, layer)
                    .map_err(Error::ArchitectureModel)?;
            keys.extend(
                resolved
                    .gate_up
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned),
            );
            keys.extend(resolved.down.source_keys().into_iter().map(str::to_owned));
        }
        keys
    } else {
        BTreeSet::new()
    };
    let binding_args = args.clone();
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        UnitPopulator {
            external_experts,
            expert_targets: Arc::clone(&expert_targets),
        },
        std::marker::PhantomData::<MlxHybridState>,
        residency,
        stream,
        weights_stream,
        move |key| external_expert_source_keys.contains(key),
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |ordinal, _address, _path, unit, store, _stream| {
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                eredu_architectures::gemma4::unit_recipes(store, &binding_args, ordinal)
                    .map_err(Error::ArchitectureModel)?
            };
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.text.weight_quantization);
    metadata.set_materialization(materialization);
    let execution = if residency.is_fully_resident() {
        Execution::Resident(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        ))
    } else {
        Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(Gemma4Model {
        state_layout: eredu_architectures::gemma4::state_layout(&args.text)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        args,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: None,
    })
}

fn load_parallel_store(
    store: SharedCheckpointSource,
    args: FamilyConfig,
    residency: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let layer_count = args.text.num_hidden_layers();
    let mut planner = build.planner();
    for group in eredu_architectures::gemma4::static_parameter_groups(&args.text)? {
        planner.register(group)?;
    }
    for index in 0..layer_count {
        for group in eredu_architectures::gemma4::layer_parameter_groups(&args.text, index)? {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Gemma 4 declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = Arc::new(
        eredu_architectures::gemma4::local_geometry(&args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let mut architecture =
        NeutralArchitecture::new_parallel(args.clone(), geometry.as_ref().clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let global_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &global_architecture,
        )
        .clone(),
    );
    let global_static_bindings = build_module_bindings(&global_static, "", store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    let global_layout = execution_layout(&global_architecture)?;
    let decoder_groups = (0..global_layout.group_count())
        .filter(|&group| {
            group_kind(&global_architecture, group) == eredu_runtime::ArchitectureGroupKind::Decoder
        })
        .collect::<Vec<_>>();
    let [decoder_group] = decoder_groups.as_slice() else {
        return Err(Error::Parallel(
            "Gemma 4 must declare exactly one decoder execution group".into(),
        ));
    };
    let decoder_group = *decoder_group;
    for ordinal in 0..global_layout.len() {
        let unit = MlxModule::new(construct_architecture_unit(
            &global_architecture,
            &global_layout,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxHybridState>,
        )?);
        let recipes = eredu_architectures::gemma4::unit_recipes(store.as_ref(), &args, ordinal)
            .map_err(Error::ArchitectureModel)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(
                &build_module_bindings_with_recipes_excluding(
                    &unit,
                    "",
                    store.as_ref(),
                    recipes,
                    |_| false,
                )?,
            )?)
            .ok_or_else(|| Error::Parallel("Gemma 4 global parameter bytes overflowed".into()))?;
    }

    let static_layout = Arc::new(layout);
    let unit_sharding = Arc::clone(&static_layout);
    let report_layout = Arc::clone(&static_layout);
    let binding_family = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        UnitPopulator {
            external_experts: false,
            expert_targets: Arc::new(Default::default()),
        },
        std::marker::PhantomData::<MlxHybridState>,
        residency,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| {
            shard_layer_bindings(global_static_bindings, "", store, &static_layout)
        },
        move |ordinal, address, path, local, store, stream| {
            if address.group() != decoder_group {
                return build_module_bindings(&MlxModule::new(local.clone()), "", store)
                    .map_err(Into::into);
            }
            let layer = address.index();
            let global = MlxModule::new(NeutralUnit::Text(
                eredu_architectures::gemma4::DenseBlock::<MlxNeuralBackend>::new(
                    &binding_family.text,
                    layer,
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            ));
            let recipes =
                eredu_architectures::gemma4::unit_recipes(store, &binding_family, ordinal)
                    .map_err(Error::ArchitectureModel)?;
            let bindings =
                build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |_| {
                    false
                })?;
            shard_layer_bindings(bindings, path, store, &unit_sharding)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.text.weight_quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("Gemma 4 local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("Gemma 4 device parameter bytes overflowed".into()))?;
    let parallel_info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        report_layout
            .tensors()
            .map(|(target, _)| target.to_owned())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if residency.is_fully_resident() {
            local_parameter_bytes
        } else {
            metadata.static_device_bytes()
        },
        maximum_device_parameter_bytes,
    );
    let execution = if residency.is_fully_resident() {
        Execution::ParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        Execution::ParallelBounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(Gemma4Model {
        args,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: Some(parallel_info),
    })
}

pub fn load_safetensors_tensor_parallel(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let eredu_architectures::configuration::SafetensorsModelConfig::Gemma4(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Gemma 4 loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    load_parallel_store(store, args, residency, build, stream, weights_stream)
}

pub fn load_gguf_tensor_parallel(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    residency: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let (store, args) = open_pipeline_gguf_store(source, projector, residency.max_mapped_shards())?;
    load_parallel_store(store, args, residency, build, stream, weights_stream)
}

fn attach_expert_cache(
    model: &mut Gemma4Model,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries =
        crate::composition::gemma4_expert::expert_catalog(&model.args.text, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors through one neutral family object and residency policy.
pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.expert_cache();
    let eredu_architectures::configuration::SafetensorsModelConfig::Gemma4(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Gemma 4 loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Gemma 4", args.text.weight_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let (store, args, materialization) = match requested {
        Some(quantization) => {
            let (store, args, report) = quantize_store(store, &args, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut model = load_store(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        materialization,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads a Gemma 4 decoder and optional sibling media projector through the
/// same neutral family object.
pub fn load_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.expert_cache();
    let (store, args) = open_pipeline_gguf_store(source, projector, residency.max_mapped_shards())?;
    let mut model = load_store(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        None,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

pub fn open_pipeline_gguf_store(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, FamilyConfig), Error> {
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::Gemma4(family) = source.model() else {
        return Err(Error::ArchitectureModel(
            "Gemma 4 GGUF loader received a different prepared model".into(),
        ));
    };
    let args = match projector {
        Some(projector) => {
            let eredu_architectures::gguf_companion::GgufMediaProjectorConfig::Gemma4(family) =
                projector.model()
            else {
                return Err(Error::ArchitectureModel(
                    "Gemma 4 GGUF loader received a mismatched media-projector plan".into(),
                ));
            };
            family.clone()
        }
        None => family.clone(),
    };
    let formats = gguf_quantization_configs(
        checkpoint,
        eredu_architectures::gemma4::translate_gguf_weight_name,
    )?;
    let args = eredu_architectures::gemma4::with_checkpoint_formats(&args, formats)
        .map_err(Error::ArchitectureModel)?;
    let builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_cached_readers)?
        .add_checkpoint(
            checkpoint.catalog().clone(),
            source.plan().checkpoint(),
            |name| eredu_architectures::gemma4::translate_gguf_weight_name(name),
        )?;
    let builder = if let Some(projector) = projector {
        builder.add_checkpoint(
            projector.checkpoint().catalog().clone(),
            projector.plan().checkpoint(),
            |name| eredu_architectures::gemma4::translate_mmproj_weight_name(name),
        )?
    } else {
        builder
    };
    let store: SharedCheckpointSource = Arc::new(builder.build()?);
    Ok((store, args))
}
