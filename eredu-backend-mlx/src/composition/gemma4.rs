//! Neutral Gemma 4 binding to MLX storage, state, and residency policy.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
    sync::Arc,
};

use eredu_architectures::gemma4::{
    AudioInput, DecoderInputPart, FamilyConfig, LayeredModel as Architecture, ModelInput, Unit,
    VisionInput,
};
use eredu_checkpoint::{
    store::{CheckpointSource, SharedCheckpointSource},
    WeightQuantization,
};
use eredu_nn::Tensor;
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, ExecutionUnitLayout, LayerWeightResidency,
    LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions, ParallelModelInfo, ParameterRole,
    RuntimeState, StaticUnitBindings, WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{NewAxis, TryIndexOp},
        maximum, pad, GgufCheckpoint, GgufMetadataValue, PadWidth,
    },
    Array, Dtype, Stream,
};

use crate::backend::mlx::{
    error::Error,
    nn::shared::{MlxBackend, MlxModule},
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
            load::{gguf_metadata, gguf_quantization_configs, GgufTensorNames},
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
use crate::composition::mlx::artifact::find_sibling_mmproj;

type NeutralArchitecture = Architecture<MlxBackend>;
type NeutralUnit = Unit<MlxBackend>;
type NeutralAssistant = eredu_architectures::gemma4::Assistant<MlxBackend>;
pub type Gemma4PipelineUnit = MlxModule<NeutralUnit>;

fn group_kind(
    architecture: &NeutralArchitecture,
    group: usize,
) -> eredu_runtime::ArchitectureGroupKind {
    <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::group_transport(
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

    fn static_modules(
        architecture: &NeutralArchitecture,
    ) -> &eredu_architectures::gemma4::StaticModules<MlxBackend> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            architecture,
        )
    }

    pub fn model_type<'a>(&self, architecture: &'a NeutralArchitecture) -> &'a str {
        &architecture.args().model_type
    }

    pub fn selected_static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let args = architecture.args();
        let modules = Self::static_modules(architecture);
        let mut units = Vec::new();
        macro_rules! push_leaf {
            ($role:literal, $module:expr, $prefix:literal, $packed:expr) => {
                if select(concat!("gemma4.static.", $role)) {
                    let prefix = concat!($prefix, ".");
                    let bindings = build_module_bindings(
                        &MlxModule::new($module.clone()),
                        "",
                        store,
                    )?
                    .into_iter()
                    .map(|binding| {
                        let local = binding
                            .name()
                            .strip_prefix(prefix)
                            .ok_or_else(|| {
                                Error::Parallel(format!(
                                    "Gemma 4 static binding {:?} does not start with {prefix:?}",
                                    binding.name()
                                ))
                            })?
                            .to_string();
                        let local = if $packed && local == "weight" {
                            "inner.weight"
                        } else {
                            local.as_str()
                        };
                        binding.with_name(local).map_err(Error::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                    units.push(StaticUnitBindings::new(
                        concat!("gemma4.static.", $role),
                        bindings,
                    )?);
                }
            };
        }
        macro_rules! push {
            ($role:literal, $module:expr) => {
                if select(concat!("gemma4.static.", $role)) {
                    units.push(StaticUnitBindings::new(
                        concat!("gemma4.static.", $role),
                        build_module_bindings(&MlxModule::new($module.clone()), "", store)?,
                    )?);
                }
            };
        }
        push_leaf!(
            "embedding",
            modules.text.embeddings,
            "model.language_model.embed_tokens",
            args.text
                .linear_format_for("model.language_model.embed_tokens.weight")
                .weight_quantization()
                .is_some()
        );
        if let Some(module) = &modules.text.per_layer_embeddings {
            push_leaf!(
                "per_layer_embedding",
                module,
                "model.language_model.embed_tokens_per_layer",
                args.text
                    .linear_format_for("model.language_model.embed_tokens_per_layer.weight")
                    .weight_quantization()
                    .is_some()
            );
        }
        if let Some(module) = &modules.text.per_layer_projection {
            push_leaf!(
                "per_layer_projection",
                module,
                "model.language_model.per_layer_model_projection",
                args.text
                    .linear_format_for("model.language_model.per_layer_model_projection.weight")
                    .weight_quantization()
                    .is_some()
            );
        }
        if let Some(module) = &modules.text.per_layer_norm {
            push_leaf!(
                "per_layer_norm",
                module,
                "model.language_model.per_layer_projection_norm",
                false
            );
        }
        push_leaf!(
            "norm",
            modules.text.norm,
            "model.language_model.norm",
            false
        );
        if let Some(module) = &modules.text.head {
            push_leaf!(
                "output",
                module,
                "lm_head",
                args.text
                    .linear_format_for("lm_head.weight")
                    .weight_quantization()
                    .is_some()
            );
        }
        if let Some(module) = &modules.vision {
            push!("vision", module);
        }
        if let Some(module) = &modules.vision_projection {
            push!("vision_projection", module);
        }
        if let Some(module) = &modules.audio {
            push!("audio", module);
        }
        if let Some(module) = &modules.audio_projection {
            push!("audio_projection", module);
        }
        Ok(units)
    }

    pub fn static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(architecture, store, &|_| true)
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
        let recipes = if is_decoder && !self.external_experts {
            gemma4_unit_recipes(&architecture.args().text, index, store)?
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
        let recipes = if is_decoder && !self.external_experts {
            gemma4_unit_recipes(&architecture.args().text, index, store)?
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
            (true, Some(layout)) => shard_layer_bindings(
                bindings,
                &format!("model.language_model.layers.{index}"),
                store,
                layout,
            ),
            _ => Ok(bindings),
        }
    }
}
type Resident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
>;
type Bounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralUnit, UnitPopulator>,
>;
type ParallelResident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
>;
type ParallelBounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
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
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
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
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::execution_graph(
            architecture,
        )
        .map(|graph| graph.output())
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
    P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
    P::Error: std::fmt::Display,
{
    <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
        MlxBackend,
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
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
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
            .draft_step::<crate::backend::mlx::runtime::cache::kv::ConcatKeyValueCache>(
                embedding, state, stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

/// Loads the released SafeTensors assistant into the backend-neutral module.
pub fn load_assistant_safetensors(
    model_dir: &Path,
    options: crate::backend::mlx::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::UnsupportedArchitecture(
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
    let bytes = std::fs::read(model_dir.join("config.json"))?;
    let source_config = eredu_architectures::gemma4::AssistantConfig::from_json(&bytes)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let requested = options
        .quantization
        .map(|requested| {
            if source_config.use_ordered_embeddings {
                return Err(Error::Quantization(
                    "Gemma 4 ordered assistant heads cannot be quantized".into(),
                ));
            }
            should_quantize_on_load("Gemma 4 assistant", source_config.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut config = source_config.clone();
    if let Some(requested) = requested {
        if config.use_ordered_embeddings {
            return Err(Error::Quantization(
                "Gemma 4 ordered assistant heads cannot be quantized".into(),
            ));
        }
        config.quantization = Some(requested);
        config.text_config.weight_quantization = Some(requested);
    }
    let store =
        open_safetensors_weight_store(model_dir, options.weight_residency.max_mapped_shards())?;
    let store = if let Some(requested) = requested {
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0
    } else {
        store
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(Gemma4AssistantModel { config, module })
}

pub fn load_assistant_gguf(
    gguf_file: &Path,
    options: crate::backend::mlx::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::UnsupportedArchitecture(
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
    struct Catalog<'a>(&'a GgufCheckpoint);
    impl eredu_architectures::gemma4::GgufTensorCatalog for Catalog<'_> {
        fn contains(&self, name: &str) -> bool {
            self.0.contains_gguf_tensor(name)
        }
    }
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    let mut config = eredu_architectures::gemma4::AssistantConfig::from_gguf_metadata(
        &Catalog(&checkpoint),
        &metadata,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let formats = gguf_quantization_configs(
        &checkpoint,
        eredu_architectures::gemma4::translate_assistant_gguf_weight_name,
    )?;
    if !formats.is_empty() {
        config.text_config.quantized_weight_configs = Some(formats);
    }
    if config.use_ordered_embeddings && options.quantization.is_some() {
        return Err(Error::Quantization(
            "Gemma 4 ordered assistant heads cannot be quantized".into(),
        ));
    }
    crate::composition::mlx::validate_gguf_quantization_source(
        &checkpoint,
        &metadata,
        options.quantization,
    )?;
    let plan = eredu_architectures::gemma4::assistant_gguf_plan(&config)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: SharedCheckpointSource = Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(options.weight_residency.max_mapped_shards())?
            .add_checkpoint(checkpoint.catalog().clone(), &plan, |name| {
                eredu_architectures::gemma4::translate_assistant_gguf_weight_name(name)
            })?
            .build()?,
    );
    let source_config = config.clone();
    let (store, config) = if let Some(requested) = options.quantization {
        config.quantization = Some(requested);
        config.text_config.weight_quantization = Some(requested);
        config.text_config.quantized_weight_configs = None;
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        (
            quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0,
            config,
        )
    } else {
        (store, config)
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
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

    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
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
                    crate::backend::mlx::cache::prompt_cache_topology(info.topology())
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

    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.state_layout.layers().clone())
    }

    fn prompt_identity(&self) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let topology = self
            .parallel_info
            .as_ref()
            .map_or_else(eredu_core::cache::PromptCacheTopology::default, |info| {
                crate::backend::mlx::cache::prompt_cache_topology(info.topology())
            });
        eredu_runtime::ModelStateIdentity {
            model_family: "gemma4".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: self.args.architecture_fingerprint(),
            layer_count: self.state_layout.len(),
            global_layer_start: 0,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; self.state_layout.len()],
            topology,
        }
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
        let identity = self.prompt_identity()?;
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
            &self.prompt_identity()?,
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
            return Err(Error::UnsupportedArchitecture(
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
                result.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let hidden = final_text_hidden.ok_or_else(|| {
                Error::UnsupportedArchitecture("Gemma 4 text graph produced no activation".into())
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
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let hidden = final_text_hidden.ok_or_else(|| {
            Error::UnsupportedArchitecture("Gemma 4 text graph produced no activation".into())
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
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
                                MlxBackend,
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
                            MlxBackend,
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
    height: i32,
    width: i32,
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
    valid_frames: i32,
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
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    prepared
                        .tokens
                        .push(crate::MlxTensor::from_array(tokens.clone()));
                    prepared.modalities.push(input::Modality::Text);
                    prepared.projected.push(None);
                }
                (
                    modality @ (input::Modality::Image | input::Modality::Video),
                    input::InputPayload::Tensor(patches),
                ) => prepared.push_vision(args, modality, patches, part.metadata)?,
                (input::Modality::Audio, input::InputPayload::Tensor(features)) => {
                    prepared.push_audio(args, features, part.metadata)?
                }
                (modality, input::InputPayload::Embeddings(embeddings)) => {
                    input::ensure_hidden_size(
                        embeddings,
                        args.text.hidden_size,
                        "Gemma 4 projected embeddings",
                    )?;
                    let token = modality_token(args, modality)?;
                    prepared
                        .tokens
                        .push(crate::MlxTensor::from_array(Array::from_slice(
                            &vec![token; embeddings.dim(1) as usize],
                            &[1, embeddings.dim(1)],
                        )));
                    prepared.modalities.push(modality);
                    prepared
                        .projected
                        .push(Some(crate::MlxTensor::from_array(embeddings.clone())));
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Gemma 4 does not accept this {} payload",
                        modality.as_str()
                    )))
                }
            }
        }
        prepared.finish_vision(stream)?;
        prepared.finish_audio(stream)?;
        Ok(prepared)
    }

    fn push_vision(
        &mut self,
        args: &FamilyConfig,
        modality: input::Modality,
        patches: &Array,
        metadata: input::InputMetadata<'_>,
    ) -> Result<(), Error> {
        let positions = metadata.patch_positions.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} input requires patch positions",
                modality.as_str()
            ))
        })?;
        metadata.patch_grid.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} input requires a prepared patch grid",
                modality.as_str()
            ))
        })?;
        let [time, height, width] = metadata.patch_extent.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} input requires a host-known patch extent",
                modality.as_str()
            ))
        })?;
        if time != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} parts must contain one prepared frame; got {time}",
                modality.as_str()
            )));
        }
        let pool = args
            .vision
            .as_ref()
            .ok_or_else(|| Error::UnsupportedArchitecture("Gemma 4 has no vision tower".into()))?
            .pooling_kernel_size;
        let count = (height / pool) * (width / pool);
        let token = modality_token(args, modality)?;
        self.tokens
            .push(crate::MlxTensor::from_array(Array::from_slice(
                &vec![token; count as usize],
                &[1, count],
            )));
        self.modalities.push(modality);
        self.projected.push(None);

        self.vision_parts.push(PreparedVisionPart {
            patches: crate::MlxTensor::from_array(patches.clone()),
            positions: crate::MlxTensor::from_array(positions.clone()),
            height,
            width,
        });
        Ok(())
    }

    fn finish_vision(&mut self, stream: &Stream) -> Result<(), Error> {
        if self.vision_parts.is_empty() {
            return Ok(());
        }
        let max_patches = self
            .vision_parts
            .iter()
            .map(|part| part.patches.dim(1))
            .max()
            .unwrap_or(0);
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
        let x = positions.try_index_device((.., .., 0), stream)?;
        let y = positions.try_index_device((.., .., 1), stream)?;
        let padding = x
            .eq(Array::from_int(-1), stream)?
            .logical_and(&y.eq(Array::from_int(-1), stream)?, stream)?;
        let sanitized = maximum(positions.clone(), Array::from_int(0), stream)?;
        let valid = padding
            .logical_not(stream)?
            .as_dtype(Dtype::Float32, stream)?
            .try_index_device((.., .., NewAxis), stream)?;
        let key_mask = padding
            .try_index_device((.., NewAxis, NewAxis, ..), stream)?
            .as_dtype(Dtype::Float32, stream)?
            .multiply(Array::from_f32(-1.0e9), stream)?;
        self.vision = Some(PreparedVision {
            patches: crate::MlxTensor::from_array(patches),
            positions: crate::MlxTensor::from_array(sanitized),
            valid: crate::MlxTensor::from_array(valid),
            key_mask: crate::MlxTensor::from_array(key_mask),
            grid_extents: self
                .vision_parts
                .iter()
                .map(|part| (part.height, part.width))
                .collect(),
        });
        Ok(())
    }

    fn push_audio(
        &mut self,
        args: &FamilyConfig,
        features: &Array,
        metadata: input::InputMetadata<'_>,
    ) -> Result<(), Error> {
        let mask = metadata.audio_mask.ok_or_else(|| {
            Error::UnsupportedArchitecture("Gemma 4 audio input requires an audio mask".into())
        })?;
        let valid_frames = metadata.audio_valid_frames.ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "Gemma 4 audio input requires a host-known valid-frame extent".into(),
            )
        })?;
        let valid = (valid_frames + 3) / 4;
        let token = modality_token(args, input::Modality::Audio)?;
        self.tokens
            .push(crate::MlxTensor::from_array(Array::from_slice(
                &vec![token; valid as usize],
                &[1, valid],
            )));
        self.modalities.push(input::Modality::Audio);
        self.projected.push(None);

        self.audio_parts.push(PreparedAudioPart {
            features: crate::MlxTensor::from_array(features.clone()),
            mask: crate::MlxTensor::from_array(mask.clone()),
            valid_frames,
        });
        Ok(())
    }

    fn finish_audio(&mut self, stream: &Stream) -> Result<(), Error> {
        if self.audio_parts.is_empty() {
            return Ok(());
        }
        let max_frames = self
            .audio_parts
            .iter()
            .map(|part| part.features.dim(1))
            .max()
            .unwrap_or(0);
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
        let first_frames = (max_frames + 1) / 2;
        let first_stage = self
            .audio_parts
            .iter()
            .map(|part| {
                let first_valid = (part.valid_frames + 1) / 2;
                let mut mask = vec![0.0f32; first_frames as usize];
                mask[..first_valid as usize].fill(1.0);
                Array::from_slice(&mask, &[1, first_frames, 1, 1])
            })
            .collect::<Vec<_>>();
        let first_refs = first_stage.iter().collect::<Vec<_>>();
        self.audio = Some(PreparedAudio {
            features: crate::MlxTensor::from_array(features),
            input_mask: crate::MlxTensor::from_array(input_mask),
            first_stage_mask: crate::MlxTensor::from_array(concatenate_axis(
                &first_refs,
                0,
                stream,
            )?),
            valid: self
                .audio_parts
                .iter()
                .map(|part| (part.valid_frames + 3) / 4)
                .collect(),
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
        return Err(Error::UnsupportedArchitecture(
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

fn modality_token(args: &FamilyConfig, modality: input::Modality) -> Result<u32, Error> {
    match modality {
        input::Modality::Text => Some(args.text.pad_token_id),
        input::Modality::Image => args.image_token_id,
        input::Modality::Video => args.video_token_id,
        input::Modality::Audio => args.audio_token_id,
    }
    .and_then(|token| u32::try_from(token).ok())
    .ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Gemma 4 has no valid {} placeholder",
            modality.as_str()
        ))
    })
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

fn execution_layout(args: &FamilyConfig) -> Result<ExecutionUnitLayout, Error> {
    let graph = eredu_runtime::ExecutionGraph::new(
        vec![
            eredu_runtime::ExecutionGroupSpec::root("vision"),
            eredu_runtime::ExecutionGroupSpec::root("audio"),
            eredu_runtime::ExecutionGroupSpec::with_dependencies(
                "text_decoder",
                ["vision", "audio"],
            ),
        ],
        "text_decoder",
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    ExecutionUnitLayout::new(
        &graph,
        [
            args.vision
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            args.audio
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            args.text.num_hidden_layers(),
        ],
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub fn resolve_pipeline_store(
    store: SharedCheckpointSource,
    args: &FamilyConfig,
) -> Result<SharedCheckpointSource, Error> {
    let plan = eredu_architectures::gemma4::safetensors_plan(args)
        .map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|error| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 checkpoint contract did not resolve: {error:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

pub fn load_pipeline_config(model_dir: &Path) -> Result<FamilyConfig, Error> {
    FamilyConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
    let mut target = source.clone();
    target.text.weight_quantization = Some(quantization);
    target.text.quantized_weights = None;
    target.text.quantized_weight_configs = None;
    if let Some(vision) = target.vision.as_mut() {
        vision.weight_quantization = Some(quantization);
        vision.quantized_weights = None;
        vision.quantized_weight_configs = None;
    }
    if let Some(audio) = target.audio.as_mut() {
        audio.weight_quantization = Some(quantization);
        audio.quantized_weights = None;
        audio.quantized_weight_configs = None;
    }
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let unit_count = source
        .vision
        .as_ref()
        .map_or(0, |config| config.num_hidden_layers as usize)
        .checked_add(
            source
                .audio
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
        )
        .and_then(|count| count.checked_add(source.text.num_hidden_layers()))
        .ok_or_else(|| Error::Quantization("Gemma 4 unit count overflowed".into()))?;
    let source_layout = execution_layout(source)?;
    let target_layout = execution_layout(&target)?;
    let source_static =
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &source_architecture,
        )
        .clone();
    let target_static =
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &target_architecture,
        )
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
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
            let resolved = eredu_architectures::gemma4::expert_recipes(
                store.as_ref(),
                &args.text,
                "model.language_model.layers",
                layer,
            )
            .map_err(Error::UnsupportedArchitecture)?;
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
    let vision_layers = args
        .vision
        .as_ref()
        .map_or(0, |config| config.num_hidden_layers as usize);
    let audio_layers = args
        .audio
        .as_ref()
        .map_or(0, |config| config.num_hidden_layers as usize);
    let binding_args = args.text.clone();
    let text_start = vision_layers + audio_layers;
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
        move |ordinal, unit, store, _stream| {
            let recipes = if !external_experts && ordinal >= text_start {
                let layer = ordinal - text_start;
                if binding_args.layer_policy(layer).is_some_and(|policy| {
                    policy.feed_forward
                        == eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
                }) {
                    let resolved = eredu_architectures::gemma4::expert_recipes(
                        store,
                        &binding_args,
                        "model.language_model.layers",
                        layer,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                    BTreeMap::from([
                        (resolved.target_gate_up, resolved.gate_up),
                        (resolved.target_down, resolved.down),
                    ])
                } else {
                    BTreeMap::new()
                }
            } else {
                BTreeMap::new()
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
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        args,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: None,
    })
}

fn gemma4_unit_recipes(
    args: &eredu_architectures::gemma4::ModelArgs,
    layer: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    if args.layer_policy(layer).is_some_and(|policy| {
        policy.feed_forward == eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
    }) {
        let resolved = eredu_architectures::gemma4::expert_recipes(
            store,
            args,
            "model.language_model.layers",
            layer,
        )
        .map_err(Error::UnsupportedArchitecture)?;
        Ok(BTreeMap::from([
            (resolved.target_gate_up, resolved.gate_up),
            (resolved.target_down, resolved.down),
        ]))
    } else {
        Ok(BTreeMap::new())
    }
}

fn load_parallel_store(
    store: SharedCheckpointSource,
    args: FamilyConfig,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
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
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let state_layout = architecture
        .runtime_state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let vision_layers = geometry.vision_layers();
    let audio_layers = geometry.audio_layers();
    let text_start = vision_layers + audio_layers;

    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let global_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &global_architecture,
        )
        .clone(),
    );
    let global_static_bindings = build_module_bindings(&global_static, "", store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    let total_units = text_start
        .checked_add(layer_count)
        .ok_or_else(|| Error::Parallel("Gemma 4 unit count overflowed".into()))?;
    let global_layout = execution_layout(&args)?;
    for ordinal in 0..total_units {
        let unit = MlxModule::new(construct_architecture_unit(
            &global_architecture,
            &global_layout,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxHybridState>,
        )?);
        let recipes = if ordinal >= text_start {
            gemma4_unit_recipes(&args.text, ordinal - text_start, store.as_ref())?
        } else {
            BTreeMap::new()
        };
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
        move |ordinal, local, store, stream| {
            if ordinal < text_start {
                return build_module_bindings(&MlxModule::new(local.clone()), "", store)
                    .map_err(Into::into);
            }
            let layer = ordinal - text_start;
            let global = MlxModule::new(NeutralUnit::Text(
                eredu_architectures::gemma4::DenseBlock::<MlxBackend>::new(
                    &binding_family.text,
                    layer,
                    stream,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            ));
            let recipes = gemma4_unit_recipes(&binding_family.text, layer, store)?;
            let bindings =
                build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |_| {
                    false
                })?;
            shard_layer_bindings(
                bindings,
                &format!("model.language_model.layers.{layer}"),
                store,
                &unit_sharding,
            )
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
    model_dir: impl AsRef<Path>,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = FamilyConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = open_safetensors_weight_store(model_dir, residency.max_mapped_shards())?;
    let store = resolve_pipeline_store(store, &args)?;
    load_parallel_store(store, args, residency, build, stream, weights_stream)
}

pub fn load_gguf_tensor_parallel(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Gemma4Model, Vec<u32>), Error> {
    let (store, args) = open_pipeline_gguf_store(
        gguf_file,
        checkpoint,
        metadata,
        residency.max_mapped_shards(),
    )?;
    let eos = crate::composition::mlx::gguf_eos_token_ids(metadata)?;
    Ok((
        load_parallel_store(store, args, residency, build, stream, weights_stream)?,
        eos,
    ))
}

fn attach_expert_cache(
    model: &mut Gemma4Model,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = crate::composition::gemma4_expert::expert_catalog(
        &model.args.text,
        store.as_ref(),
        stream,
    )?;
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
    model_dir: impl AsRef<Path>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.expert_cache();
    let model_dir = model_dir.as_ref();
    let args = FamilyConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = open_safetensors_weight_store(model_dir, residency.max_mapped_shards())?;
    let store = resolve_pipeline_store(store, &args)?;
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
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.expert_cache();
    let (store, args) = open_pipeline_gguf_store(
        gguf_file,
        checkpoint,
        metadata,
        residency.max_mapped_shards(),
    )?;
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
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, FamilyConfig), Error> {
    let projector = find_sibling_mmproj(gguf_file, "gemma4")?
        .map(GgufCheckpoint::open)
        .transpose()?;
    let projector_metadata = projector
        .as_ref()
        .map(crate::backend::mlx::runtime::checkpoint::load::gguf_metadata);
    if let Some(metadata) = projector_metadata.as_ref() {
        eredu_architectures::gemma4::validate_projector_identity(metadata)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    }
    let names = checkpoint
        .catalog()
        .tensors()
        .flat_map(|tensor| tensor.outputs())
        .map(|output| output.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut text = eredu_architectures::gemma4::ModelArgs::from_gguf_metadata(&names, metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    text.quantized_weight_configs = Some(gguf_quantization_configs(
        checkpoint,
        eredu_architectures::gemma4::translate_gguf_weight_name,
    )?);
    let args = eredu_architectures::gemma4::family_from_gguf_metadata(
        text,
        metadata,
        projector_metadata.as_ref(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    if let Some(projector) = projector.as_ref() {
        let quantized = gguf_quantization_configs(
            projector,
            eredu_architectures::gemma4::translate_mmproj_weight_name,
        )?;
        if !quantized.is_empty() {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 projector GGUF admits only dense F16, BF16, or F32 tensors".into(),
            ));
        }
        let tokens = [
            args.image_token_id,
            args.video_token_id,
            args.audio_token_id,
        ]
        .into_iter()
        .flatten()
        .filter_map(|token| u32::try_from(token).ok())
        .collect::<std::collections::BTreeSet<_>>();
        eredu_architectures::gemma4::Gemma4ArtifactConfig {
            unified: args.model_type == "gemma4_unified",
            hidden_size: args.text.hidden_size as usize,
            image_token_id: args.image_token_id.and_then(|token| token.try_into().ok()),
            video_token_id: args.video_token_id.and_then(|token| token.try_into().ok()),
            audio_token_id: args.audio_token_id.and_then(|token| token.try_into().ok()),
            projector: true,
            assistant: false,
        }
        .projector_compatibility(
            args.model_type.clone(),
            args.text.hidden_size as usize,
            tokens,
        )
        .validate()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    }
    let plan = eredu_architectures::gemma4::gguf_plan(&args.text)
        .map_err(Error::UnsupportedArchitecture)?;
    let builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_cached_readers)?
        .add_checkpoint(checkpoint.catalog().clone(), &plan, |name| {
            eredu_architectures::gemma4::translate_gguf_weight_name(name)
        })?;
    let builder = if let Some(projector) = projector.as_ref() {
        let plan = eredu_architectures::gemma4::mmproj_gguf_plan(&args)
            .map_err(Error::UnsupportedArchitecture)?;
        builder.add_checkpoint(projector.catalog().clone(), &plan, |name| {
            eredu_architectures::gemma4::translate_mmproj_weight_name(name)
        })?
    } else {
        builder
    };
    let store: SharedCheckpointSource = Arc::new(builder.build()?);
    Ok((store, args))
}
