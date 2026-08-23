//! Neutral Muse-Glimmer binding to MLX storage and execution policy.

use std::{collections::HashMap, path::Path, sync::Arc};

use eredu_architectures::muse_glimmer::{
    DecoderConfig, DecoderInputPart, LayeredModel as Architecture, ModelInput, Unit, VisionInput,
};
use eredu_checkpoint::{
    store::{CheckpointSource, SharedCheckpointSource},
    WeightQuantization,
};
use eredu_nn::Tensor;
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, LayerWeightResidency, LayeredArchitecture,
    LayeredForwardState, LayerwiseRuntime, PagedCacheOptions, ParallelModelInfo, ParameterRole,
    RuntimeState, StaticUnitBindings, WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::residency::{open_prompt_cache, CacheResidencyManager},
        cache::state::MlxKeyValueState,
        checkpoint::{
            binding::{
                binding_bytes, build_module_bindings, build_module_bindings_with_recipes_excluding,
                materialize_module_bindings, parameter_name_in_targets,
                populate_module_from_arrays_excluding, populate_module_from_lease_excluding,
            },
            load::{gguf_metadata, gguf_quantization_configs, GgufTensorNames},
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                architecture_execution_layout, construct_architecture_unit,
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitPopulator,
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

type NeutralArchitecture = Architecture<MlxNeuralBackend>;
type NeutralUnit = Unit<MlxNeuralBackend>;
type NeutralDFlash = eredu_architectures::muse_glimmer::DFlash<MlxNeuralBackend>;
pub type MuseGlimmerPipelineUnit = MlxModule<NeutralUnit>;

fn group_kind(
    architecture: &NeutralArchitecture,
    group: usize,
) -> eredu_runtime::ArchitectureGroupKind {
    <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
    >>::group_transport(architecture, group)
    .kind
}
type Resident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralUnit>,
>;
type Bounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralUnit, UnitPopulator>,
>;
type ParallelResident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralUnit>,
>;
type ParallelBounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
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
        <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::execution_graph(architecture)
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
    state: &mut MlxKeyValueState,
    forward: &mut eredu_architectures::muse_glimmer::ForwardContext<crate::MlxTensor>,
    stream: &Stream,
    provider: &mut P,
) -> Result<crate::MlxTensor, eredu_nn::Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
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

/// One family object shared by resident and bounded execution.
pub struct MuseGlimmerModel {
    args: DecoderConfig,
    state_layout: eredu_runtime::StateLayout,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    execution: Execution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::MlxParallelContext>>,
}

/// Fully resident DFlash assistant built from neutral equations.
pub struct MuseGlimmerDFlashModel {
    pub config: eredu_architectures::muse_glimmer::DFlashConfig,
    module: MlxModule<NeutralDFlash>,
}

impl MuseGlimmerDFlashModel {
    pub fn target_layer_ids(&self) -> &[usize] {
        self.module.target_layer_ids()
    }

    pub fn assemble_target_states(
        &self,
        states: &[crate::MlxTensor],
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.module
            .assemble_target_states(states, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub fn update_context(
        &mut self,
        previous: Option<eredu_architectures::muse_glimmer::DFlashContext<crate::MlxTensor>>,
        states: &crate::MlxTensor,
        absolute_end: i32,
        stream: &Stream,
    ) -> Result<eredu_architectures::muse_glimmer::DFlashContext<crate::MlxTensor>, Error> {
        self.module
            .update_context(previous, states, absolute_end, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub fn proposal_states(
        &mut self,
        embeddings: &crate::MlxTensor,
        committed: &eredu_architectures::muse_glimmer::DFlashContext<crate::MlxTensor>,
        absolute_end: i32,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.module
            .proposal_states(embeddings, committed, absolute_end, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

pub fn load_dflash_safetensors(
    model_dir: &Path,
    options: crate::backend::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerDFlashModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer DFlash requires fully resident assistant weights".into(),
        ));
    }
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Muse-Glimmer DFlash requires replicated placement".into(),
        ));
    }
    let bytes = std::fs::read(model_dir.join("config.json"))?;
    let source_config = eredu_architectures::muse_glimmer::DFlashConfig::from_hf_json(&bytes)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let requested = options
        .quantization
        .map(|requested| {
            should_quantize_on_load("Muse-Glimmer DFlash", source_config.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut config = source_config.clone();
    if let Some(requested) = requested {
        config.quantization = Some(requested);
    }
    let store =
        open_safetensors_weight_store(model_dir, options.weight_residency.max_mapped_shards())?;
    let store = if let Some(requested) = requested {
        let source = NeutralDFlash::new(source_config, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target = NeutralDFlash::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0
    } else {
        store
    };
    let mut module = MlxModule::new(
        NeutralDFlash::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(MuseGlimmerDFlashModel { config, module })
}

pub fn load_dflash_gguf(
    gguf_file: &Path,
    options: crate::backend::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerDFlashModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer DFlash requires fully resident assistant weights".into(),
        ));
    }
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Muse-Glimmer DFlash requires replicated placement".into(),
        ));
    }
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    let mut config = eredu_architectures::muse_glimmer::DFlashConfig::from_gguf_metadata(&metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    config.quantized_weights = gguf_quantization_configs(
        &checkpoint,
        eredu_architectures::muse_glimmer::translate_dflash_gguf_weight_name,
    )?;
    crate::composition::mlx::validate_gguf_quantization_source(
        &checkpoint,
        &metadata,
        options.quantization,
    )?;
    let plan = eredu_architectures::muse_glimmer::dflash_gguf_plan(&config)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: SharedCheckpointSource = Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(options.weight_residency.max_mapped_shards())?
            .add_checkpoint(checkpoint.catalog().clone(), &plan, |name| {
                eredu_architectures::muse_glimmer::translate_dflash_gguf_weight_name(name)
            })?
            .build()?,
    );
    let source_config = config.clone();
    let (store, config) = if let Some(requested) = options.quantization {
        config.quantization = Some(requested);
        config.quantized_weights.clear();
        let source = NeutralDFlash::new(source_config, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target = NeutralDFlash::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        (
            quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0,
            config,
        )
    } else {
        (store, config)
    };
    let mut module = MlxModule::new(
        NeutralDFlash::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(MuseGlimmerDFlashModel { config, module })
}

pub struct MuseGlimmerMtpOutput {
    pub logits: crate::MlxTensor,
    pub target_states: Vec<crate::MlxTensor>,
}

pub struct PreparedMuseInput {
    pub tokens: Vec<crate::MlxTensor>,
    pub media: Vec<bool>,
    pub pixels: Option<crate::MlxTensor>,
    pub grid: Vec<(i32, i32, i32)>,
}

/// Transportable neutral ingress state used while a pipeline placement walks
/// the native vision group. The architecture forward context is the same one
/// used by resident and bounded execution; only ownership of its tensors moves.
pub struct MuseGlimmerPlacedState {
    pub forward: LayeredForwardState<
        crate::MlxTensor,
        eredu_architectures::muse_glimmer::ForwardContext<crate::MlxTensor>,
    >,
    pub state: MlxKeyValueState,
}

impl MuseGlimmerPlacedState {
    pub fn new(
        forward: LayeredForwardState<
            crate::MlxTensor,
            eredu_architectures::muse_glimmer::ForwardContext<crate::MlxTensor>,
        >,
        state: MlxKeyValueState,
    ) -> Self {
        Self { forward, state }
    }
}

impl MuseGlimmerPlacedState {
    pub fn hidden(&self) -> &crate::MlxTensor {
        &self.forward.hidden
    }

    pub fn replace_hidden(&mut self, hidden: crate::MlxTensor) {
        self.forward.hidden = hidden;
    }
}

/// Cold-path Muse-Glimmer checkpoint binding templates.
///
/// Forward execution and mutable state remain on the placed neutral model.
#[derive(Default)]
pub struct MuseGlimmerPipelineBindings {
    external_experts: bool,
}

impl MuseGlimmerPipelineBindings {
    pub const fn new() -> Self {
        Self {
            external_experts: false,
        }
    }

    pub const fn new_external_experts() -> Self {
        Self {
            external_experts: true,
        }
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

    pub fn layer_count(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
    ) -> Result<usize, Error> {
        <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::group_unit_count(architecture, group)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub fn cartesian_layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        global_layer: &MuseGlimmerPipelineUnit,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::runtime::distributed::expert::ExpertAssignment>,
    ) -> Result<Vec<WeightBinding>, Error> {
        let expert_targets = architecture
            .parameter_description()
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate);
        let recipes = crate::composition::muse_glimmer_expert::module_recipes(
            global_layer,
            architecture.args(),
            store,
        )?;
        let bindings = build_module_bindings_with_recipes_excluding(
            global_layer,
            "",
            store,
            recipes,
            |name| self.external_experts && parameter_name_in_targets(name, &expert_targets),
        )?;
        match layout {
            Some(layout) => {
                let root = <NeutralArchitecture as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                >>::unit_path(architecture, group, index)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                shard_layer_bindings(bindings, &root, store, layout)
            }
            None => Ok(bindings),
        }
    }

    pub fn layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        _index: usize,
        layer: &MuseGlimmerPipelineUnit,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.layer_count(architecture, group)?;
        let expert_targets = architecture
            .parameter_description()
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate);
        let recipes = crate::composition::muse_glimmer_expert::module_recipes(
            layer,
            architecture.args(),
            store,
        )?;
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && parameter_name_in_targets(name, &expert_targets)
        })
        .map_err(Into::into)
    }
}

pub fn prepare_muse_input(
    args: &DecoderConfig,
    typed: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<PreparedMuseInput, Error> {
    input::validate(typed)?;
    let mut tokens = Vec::with_capacity(typed.parts.len());
    let mut media = Vec::with_capacity(typed.parts.len());
    let mut pixels = Vec::new();
    let mut grid = Vec::new();
    let merge = args.vision_config.merge_size;
    for part in typed.parts {
        match (part.modality, part.payload) {
            (input::Modality::Text, input::InputPayload::TokenIds(value)) => {
                tokens.push(crate::MlxTensor::from_array(value.clone()));
                media.push(false);
            }
            (
                modality @ (input::Modality::Image | input::Modality::Video),
                input::InputPayload::Tensor(value),
            ) => {
                if modality == input::Modality::Video
                    && args.weight_convention
                        == eredu_architectures::muse_glimmer::WeightConvention::Gguf
                {
                    return Err(Error::UnsupportedArchitecture(
                        "the released Muse-Glimmer GGUF projector is image-only".into(),
                    ));
                }
                let entries = input::patch_grid_from_array(
                    part.metadata.patch_grid.ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Muse-Glimmer media requires patch_grid metadata".into(),
                        )
                    })?,
                    stream,
                )?;
                let count = entries
                    .iter()
                    .map(|(t, h, w)| t * (h / merge) * (w / merge))
                    .sum::<i32>();
                let id = if modality == input::Modality::Image {
                    args.image_token_id
                } else {
                    args.video_token_id
                };
                tokens.push(crate::MlxTensor::from_array(Array::from_slice(
                    &vec![id; count as usize],
                    &[1, count],
                )));
                media.push(true);
                pixels.push(value.clone());
                grid.extend(entries);
            }
            (modality, _) => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Muse-Glimmer does not accept this {} payload",
                    modality.as_str()
                )))
            }
        }
    }
    let pixels = if pixels.is_empty() {
        None
    } else {
        Some(concatenate_axis(
            &pixels.iter().collect::<Vec<_>>(),
            0,
            stream,
        )?)
    };
    Ok(PreparedMuseInput {
        tokens,
        media,
        pixels: pixels.map(crate::MlxTensor::from_array),
        grid,
    })
}

impl MuseGlimmerModel {
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
    }

    /// Returns canonical parameter/residency metadata.
    pub fn metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    pub fn parallel_info(&self) -> Option<&ParallelModelInfo<crate::backend::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    pub fn new_cache(&self) -> MlxKeyValueState {
        MlxKeyValueState::device(self.state_layout.clone())
            .expect("validated neutral state must be realizable")
    }

    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxKeyValueState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let rank = self.parallel_info.as_ref().and_then(|info| {
                    crate::backend::cache::prompt_cache_topology(info.topology())
                        .cache_rank_identity()
                });
                MlxKeyValueState::paged(
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
        eredu_architectures::muse_glimmer::state_identity(
            &self.args,
            &self.state_layout,
            0,
            topology,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
        .prompt_cache_identity(&self.state_layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(MlxKeyValueState, eredu_core::cache::PromptCacheManifest), Error> {
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
        let state = MlxKeyValueState::paged(self.state_layout.clone(), manager, rank)?;
        Ok((state, manifest))
    }

    pub fn save_prompt_cache(
        &self,
        state: &mut MlxKeyValueState,
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

    fn forward_with_taps(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxKeyValueState,
        target_layers: &[usize],
        stream: &Stream,
    ) -> Result<MuseGlimmerMtpOutput, Error> {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "Muse-Glimmer tensor-parallel execution requires a collective session".into(),
            ));
        }
        if state.layout() != &self.state_layout {
            return Err(Error::UnsupportedArchitecture(
                "Muse-Glimmer cache layout mismatch".into(),
            ));
        }
        let mut capture = (!target_layers.is_empty())
            .then(|| eredu_runtime::TargetStateCapture::new(target_layers.iter().copied()))
            .transpose()
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let output_group = self.execution.output_group()?;
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let mut provider =
                crate::composition::muse_glimmer_expert::cached_provider(&expert_cache, &args);
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
                        |group, index, hidden, _forward| {
                            if group == output_group
                                && capture.as_ref().is_some_and(|capture| capture.wants(index))
                            {
                                capture
                                    .as_mut()
                                    .expect("capture was present")
                                    .capture(eredu_runtime::TargetStateTap {
                                        layer: index,
                                        value: hidden,
                                    })
                                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
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
                        |group, index, hidden, _forward| {
                            if group == output_group
                                && capture.as_ref().is_some_and(|capture| capture.wants(index))
                            {
                                capture
                                    .as_mut()
                                    .expect("capture was present")
                                    .capture(eredu_runtime::TargetStateTap {
                                        layer: index,
                                        value: hidden,
                                    })
                                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            }
                            Ok(())
                        },
                    ),
                Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
            };
            drop(provider);
            self.expert_cache = Some(expert_cache);
            let (logits, _) =
                result.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let target_states = capture
                .map(eredu_runtime::TargetStateCapture::into_ordered)
                .transpose()
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
                .unwrap_or_default();
            return Ok(MuseGlimmerMtpOutput {
                logits,
                target_states,
            });
        }
        let result = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward_with_unit_executor_and_activation_hook(
                input,
                state,
                stream,
                |architecture, group, index, unit, hidden, state, forward, stream| {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
                },
                |group, index, hidden, _forward| {
                    if group == output_group
                        && capture.as_ref().is_some_and(|capture| capture.wants(index))
                    {
                        capture
                            .as_mut()
                            .expect("capture was present")
                            .capture(eredu_runtime::TargetStateTap {
                                layer: index,
                                value: hidden,
                            })
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
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
                |group, index, hidden, _forward| {
                    if group == output_group
                        && capture.as_ref().is_some_and(|capture| capture.wants(index))
                    {
                        capture
                            .as_mut()
                            .expect("capture was present")
                            .capture(eredu_runtime::TargetStateTap {
                                layer: index,
                                value: hidden,
                            })
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    }
                    Ok(())
                },
            ),
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target_states = capture
            .map(eredu_runtime::TargetStateCapture::into_ordered)
            .transpose()
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
            .unwrap_or_default();
        Ok(MuseGlimmerMtpOutput {
            logits: result.0,
            target_states,
        })
    }

    fn forward(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.forward_with_taps(input, state, &[], stream)
            .map(|output| output.logits)
    }

    pub fn embed_dflash_tokens(
        &mut self,
        tokens: &crate::MlxTensor,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "Muse-Glimmer DFlash embedding is unavailable in tensor parallelism".into(),
            ));
        }
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.architecture_mut().token_embeddings(tokens, stream)
            }
            Execution::Bounded(runtime) => {
                runtime.architecture_mut().token_embeddings(tokens, stream)
            }
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub fn project_dflash_logits(
        &mut self,
        hidden: &crate::MlxTensor,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "Muse-Glimmer DFlash projection is unavailable in tensor parallelism".into(),
            ));
        }
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.architecture_mut().project_logits(hidden, stream)
            }
            Execution::Bounded(runtime) => {
                runtime.architecture_mut().project_logits(hidden, stream)
            }
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub fn forward_tokens(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward(
            ModelInput {
                parts: &parts,
                vision: None,
                mask: None,
            },
            state,
            stream,
        )
    }

    pub fn forward_tensor_parallel(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Muse-Glimmer tensor-parallel cache layout mismatch".into(),
            ));
        }
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward_parallel_input(
            ModelInput {
                parts: &parts,
                vision: None,
                mask: None,
            },
            state,
            group,
            stream,
        )
    }

    pub fn prefill_tensor_parallel(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        let prepared = prepare_muse_input(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(&prepared.media)
            .map(|(value, media)| {
                if *media {
                    DecoderInputPart::Media(value)
                } else {
                    DecoderInputPart::Text(value)
                }
            })
            .collect::<Vec<_>>();
        let input = ModelInput {
            parts: &parts,
            vision: prepared.pixels.as_ref().map(|pixels| VisionInput {
                pixels,
                grid: &prepared.grid,
            }),
            mask: None,
        };
        self.forward_parallel_input(input, state, group, stream)
    }

    fn forward_parallel_input(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Muse-Glimmer tensor-parallel cache layout mismatch".into(),
            ));
        }
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let mut provider =
                crate::composition::muse_glimmer_expert::cached_provider(&expert_cache, &args);
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
                                MlxKeyValueState,
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
                            MlxKeyValueState,
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
                        "Muse-Glimmer model was not loaded for tensor parallelism".into(),
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
                "Muse-Glimmer model was not loaded for tensor parallelism".into(),
            )),
        }
    }

    pub fn verify_dflash(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxKeyValueState,
        target_layers: &[usize],
        stream: &Stream,
    ) -> Result<MuseGlimmerMtpOutput, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward_with_taps(
            ModelInput {
                parts: &parts,
                vision: None,
                mask: None,
            },
            state,
            target_layers,
            stream,
        )
    }

    pub fn forward_input(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.forward_input_with_taps(typed, state, &[], stream)
            .map(|output| output.logits)
    }

    pub fn forward_input_with_taps(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        target_layers: &[usize],
        stream: &Stream,
    ) -> Result<MuseGlimmerMtpOutput, Error> {
        let prepared = prepare_muse_input(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(&prepared.media)
            .map(|(value, media)| {
                if *media {
                    DecoderInputPart::Media(value)
                } else {
                    DecoderInputPart::Text(value)
                }
            })
            .collect::<Vec<_>>();
        self.forward_with_taps(
            ModelInput {
                parts: &parts,
                vision: prepared.pixels.as_ref().map(|pixels| VisionInput {
                    pixels,
                    grid: &prepared.grid,
                }),
                mask: None,
            },
            state,
            target_layers,
            stream,
        )
    }
}

impl CausalModel<MlxKeyValueState> for MuseGlimmerModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
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
        state: &mut MlxKeyValueState,
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

fn resolve_store(
    store: SharedCheckpointSource,
    args: &DecoderConfig,
) -> Result<SharedCheckpointSource, Error> {
    let plan = eredu_architectures::muse_glimmer::safetensors_plan(args)
        .map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|error| {
            Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer checkpoint contract did not resolve: {error:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_store(
    store: SharedCheckpointSource,
    source: &DecoderConfig,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        SharedCheckpointSource,
        DecoderConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target = source.clone();
    target.quantization = Some(quantization);
    target.quantization_config = None;
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    target.vision_config.weight_quantization = Some(quantization);
    target.vision_config.quantized_weight_configs.clear();
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let source_layout = architecture_execution_layout::<_, MlxKeyValueState>(&source_architecture)?;
    let target_layout = architecture_execution_layout::<_, MlxKeyValueState>(&target_architecture)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Muse-Glimmer quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = <NeutralArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
    >>::static_modules(&source_architecture)
    .clone();
    let target_static = <NeutralArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
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
                std::marker::PhantomData::<MlxKeyValueState>,
            )
        },
        move |index, stream| {
            construct_architecture_unit(
                &target_architecture,
                &target_layout,
                index,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
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
    args: DecoderConfig,
    residency: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<MuseGlimmerModel, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description()
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let static_args = args.clone();
    let unit_args = args.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        UnitPopulator {
            external_experts,
            expert_targets: Arc::clone(&expert_targets),
        },
        std::marker::PhantomData::<MlxKeyValueState>,
        residency,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        move |modules, store| {
            let module = MlxModule::new(modules.clone());
            let recipes = crate::composition::muse_glimmer_expert::module_recipes(
                &module,
                &static_args,
                store,
            )?;
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |_| false)
                .map_err(Into::into)
        },
        move |_address, _path, unit, store, _stream| {
            let module = MlxModule::new(unit);
            let recipes = crate::composition::muse_glimmer_expert::module_recipes(
                &module, &unit_args, store,
            )?;
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |name| {
                external_experts && parameter_name_in_targets(name, &binding_expert_targets)
            })
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization.or(args.quantization_config));
    metadata.set_materialization(materialization);
    let execution = if residency.is_fully_resident() {
        Execution::Resident(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        ))
    } else {
        Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(MuseGlimmerModel {
        state_layout: eredu_architectures::muse_glimmer::state_layout(&args)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        args,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: None,
    })
}

fn load_parallel_store(
    store: SharedCheckpointSource,
    args: DecoderConfig,
    residency: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerModel, Error> {
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let global_execution =
        architecture_execution_layout::<_, MlxKeyValueState>(&global_architecture)?;
    let decoder_groups = (0..global_execution.group_count())
        .filter(|&group| {
            group_kind(&global_architecture, group) == eredu_runtime::ArchitectureGroupKind::Decoder
        })
        .collect::<Vec<_>>();
    let [decoder_group] = decoder_groups.as_slice() else {
        return Err(Error::Parallel(format!(
            "Muse-Glimmer architecture declared {} decoder execution groups; expected one",
            decoder_groups.len()
        )));
    };
    let layer_count = global_execution
        .group_range(*decoder_group)
        .expect("validated execution group")
        .len();
    let mut planner = build.planner();
    for group in eredu_architectures::muse_glimmer::static_parameter_groups(&args)? {
        planner.register(group)?;
    }
    for index in 0..layer_count {
        for group in eredu_architectures::muse_glimmer::layer_parameter_groups(&args, index)? {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Muse-Glimmer declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = eredu_architectures::muse_glimmer::local_geometry(&args, &layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut architecture = NeutralArchitecture::new_parallel(args.clone(), geometry, stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let state_layout = architecture
        .runtime_state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::static_modules(
            &global_architecture,
        )
        .clone(),
    );
    let static_recipes = crate::composition::muse_glimmer_expert::module_recipes(
        &global_static,
        &args,
        store.as_ref(),
    )?;
    let global_static_bindings = build_module_bindings_with_recipes_excluding(
        &global_static,
        "",
        store.as_ref(),
        static_recipes,
        |_| false,
    )?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for ordinal in 0..global_execution.len() {
        let unit = MlxModule::new(construct_architecture_unit(
            &global_architecture,
            &global_execution,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxKeyValueState>,
        )?);
        let recipes =
            crate::composition::muse_glimmer_expert::module_recipes(&unit, &args, store.as_ref())?;
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
            .ok_or_else(|| {
                Error::Parallel("Muse-Glimmer global parameter bytes overflowed".into())
            })?;
    }

    let static_layout = Arc::new(layout);
    let unit_sharding = Arc::clone(&static_layout);
    let report_layout = Arc::clone(&static_layout);
    let binding_args = args.clone();
    let binding_architecture = global_architecture;
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        UnitPopulator {
            external_experts: false,
            expert_targets: Arc::new(Default::default()),
        },
        std::marker::PhantomData::<MlxKeyValueState>,
        residency,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| {
            shard_layer_bindings(global_static_bindings, "", store, &static_layout)
        },
        move |address, path, _local, store, stream| {
            let global =
                MlxModule::new(
                    <NeutralArchitecture as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::build_unit(
                        &binding_architecture,
                        address.group(),
                        address.index(),
                        stream,
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                );
            let recipes = crate::composition::muse_glimmer_expert::module_recipes(
                &global,
                &binding_args,
                store,
            )?;
            let bindings =
                build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |_| {
                    false
                })?;
            if group_kind(&binding_architecture, address.group())
                == eredu_runtime::ArchitectureGroupKind::Decoder
            {
                shard_layer_bindings(bindings, path, store, &unit_sharding)
            } else {
                Ok(bindings)
            }
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization.or(args.quantization_config));
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("Muse-Glimmer local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("Muse-Glimmer device parameter bytes overflowed".into()))?;
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
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        Execution::ParallelBounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(MuseGlimmerModel {
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
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = DecoderConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = open_safetensors_weight_store(model_dir, residency.max_mapped_shards())?;
    let store = resolve_store(store, &args)?;
    load_parallel_store(store, args, residency, build, stream, weights_stream)
}

pub fn load_gguf_tensor_parallel(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(MuseGlimmerModel, Vec<u32>), Error> {
    let (store, args) = open_gguf_store(
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
    model: &mut MuseGlimmerModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries =
        crate::composition::muse_glimmer_expert::expert_catalog(&model.args, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors through one neutral family model and one residency policy.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerModel, Error> {
    let expert_options = residency.expert_cache();
    let model_dir = model_dir.as_ref();
    let args = DecoderConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = open_safetensors_weight_store(model_dir, residency.max_mapped_shards())?;
    let store = resolve_store(store, &args)?;
    let current = args.quantization.or(args.quantization_config);
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Muse-Glimmer", current, requested)
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

/// Loads split text/projector GGUF through the same neutral family object.
pub fn load_gguf(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerModel, Error> {
    let expert_options = residency.expert_cache();
    let (store, args) = open_gguf_store(
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

fn open_gguf_store(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, DecoderConfig), Error> {
    let projector_path = find_sibling_mmproj(gguf_file, "muse-glimmer")?.ok_or_else(|| {
        Error::UnsupportedArchitecture(
            "Muse-Glimmer GGUF requires its validated sibling vision projector".into(),
        )
    })?;
    let projector = GgufCheckpoint::open(projector_path)?;
    let projector_metadata = gguf_metadata(&projector);
    let projector_quantization = gguf_quantization_configs(
        &projector,
        eredu_architectures::muse_glimmer::translate_projector_gguf_name,
    )?;
    let output_head_present = checkpoint.contains_gguf_tensor("output.weight");
    let args = DecoderConfig::from_gguf_metadata(metadata, output_head_present)
        .and_then(|args| {
            args.with_gguf_projector_metadata(&projector_metadata, projector_quantization)
        })
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let text_plan = eredu_architectures::muse_glimmer::gguf_plan(&args)
        .map_err(Error::UnsupportedArchitecture)?;
    let projector_plan = eredu_architectures::muse_glimmer::projector_gguf_plan(&args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: SharedCheckpointSource = Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(max_cached_readers)?
            .add_checkpoint(checkpoint.catalog().clone(), &text_plan, |name| {
                eredu_architectures::muse_glimmer::translate_text_gguf_name(name)
            })?
            .add_checkpoint(projector.catalog().clone(), &projector_plan, |name| {
                eredu_architectures::muse_glimmer::translate_projector_gguf_name(name)
            })?
            .build()?,
    );
    Ok((store, args))
}

pub fn prepare_gguf_pipeline_source(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_cached_readers: usize,
) -> Result<(DecoderConfig, SharedCheckpointSource), Error> {
    let path = checkpoint
        .catalog()
        .shards()
        .first()
        .map(|shard| shard.path())
        .ok_or_else(|| Error::UnsupportedArchitecture("Muse-Glimmer GGUF has no shards".into()))?;
    let (store, args) = open_gguf_store(path, checkpoint, metadata, max_cached_readers)?;
    Ok((args, store))
}
