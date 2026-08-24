// MLX artifact and residency binding for the neutral Qwen hybrid graph.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_architectures::qwen::{
    hybrid::{
        self, ConditionalInput, ConditionalLayeredModel, ConditionalUnit, EmbeddedInput,
        HybridConfig, ParsedHybridConfig, Unit,
    },
    vision,
    vl::InputPart,
};
use eredu_checkpoint::{
    store::{CheckpointSource, CompositeCheckpointSource},
    WeightQuantization,
};
use eredu_nn::Tensor;
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, ExecutionResidency, ExecutionUnitLayout,
    LayerWeightResidency, LayeredArchitecture, LayerwiseModelMetadata, LayerwiseRuntime,
    PagedCacheOptions, ParameterRole, ResidencyReport, StaticUnitBindings, WeightBinding,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

fn neutral_input_parts<'a>(
    parts: &'a [InputPart<'a, Array>],
) -> Vec<InputPart<'a, crate::MlxTensor>> {
    parts
        .iter()
        .map(|part| match part {
            InputPart::Text(tokens) => InputPart::Text(crate::composition::tensor_ref(tokens)),
            InputPart::Image { tokens, grid } => InputPart::Image {
                tokens: crate::composition::tensor_ref(tokens),
                grid,
            },
            InputPart::Video { tokens, grid } => InputPart::Video {
                tokens: crate::composition::tensor_ref(tokens),
                grid,
            },
            InputPart::Projected { tokens, embeddings } => InputPart::Projected {
                tokens: crate::composition::tensor_ref(tokens),
                embeddings: crate::composition::tensor_ref(embeddings),
            },
        })
        .collect()
}

use crate::backend::{
    error::Error,
    nn::shared::{MlxNeuralBackend, MlxModule},
    runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
            },
            state::MlxHybridState,
        },
        checkpoint::binding::{
            build_module_bindings_with_recipes, build_module_bindings_with_recipes_excluding,
            parameter_name_in_targets, parameter_role_targets, populate_module_from_lease_excluding,
        },
        checkpoint::{
            load::{gguf_quantization_configs, GgufTensorNames},
            quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::generic::{
            construct_architecture_unit, prepare_layerwise_policy_with_bindings,
            MlxLayerwisePolicy, MlxResidentPolicy, MlxUnitPopulator,
        },
        execution::layerwise::{
            quantize_parameterized_store, shard_layer_bindings,
        },
        media::input,
        residency::{
            expert_cache::{ExpertCache, ExpertCatalogEntry},
            expert_provider::CachedGatedProductExpertProvider,
        },
    },
};

type Architecture = hybrid::LayeredModel<MlxNeuralBackend>;
type Block = Unit<MlxNeuralBackend>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub struct QwenHybridCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxNeuralBackend>,
    pub units: Vec<Block>,
}

#[cfg(any(test, feature = "test-support"))]
impl QwenHybridCheckpointTemplate {
    pub fn new(config: HybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = Architecture::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target_units = config.num_hidden_layers as usize;
        let total = target_units + config.mtp_num_hidden_layers as usize;
        let units = (0..total)
            .map(|flat| {
                let (group, index) = if flat < target_units {
                    (0, flat)
                } else {
                    (flat - target_units + 1, 0)
                };
                architecture
                    .construct_unit(group, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: architecture.into_static_modules(),
            units,
        })
    }
}

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub struct QwenConditionalCheckpointTemplate {
    pub static_modules: hybrid::ConditionalStaticModules<MlxNeuralBackend>,
    pub units: Vec<hybrid::ConditionalUnit<MlxNeuralBackend>>,
}

#[cfg(any(test, feature = "test-support"))]
impl QwenConditionalCheckpointTemplate {
    pub fn new(parsed: ParsedHybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = ConditionalArchitecture::new(parsed, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let graph = <ConditionalArchitecture as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::execution_graph(&architecture)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let mut units = Vec::new();
        for group in 0..graph.groups().len() {
            let count = <ConditionalArchitecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::group_unit_count(&architecture, group)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for index in 0..count {
                units.push(
                    <ConditionalArchitecture as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(&architecture, group, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                );
            }
        }
        Ok(Self {
            static_modules: <ConditionalArchitecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::static_modules(&architecture)
            .clone(),
            units,
        })
    }
}

#[derive(Clone)]
struct UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

#[derive(Clone)]
struct ConditionalUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<ConditionalUnit<MlxNeuralBackend>> for ConditionalUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<ConditionalUnit<MlxNeuralBackend>>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

/// Pipeline/loading adapter over the neutral Qwen hybrid units.
#[derive(Default)]
pub struct QwenHybridPipelineBindings {
    external_experts: bool,
}

/// Pipeline/loading adapter over the neutral conditional Qwen3.5 graph.
#[derive(Default)]
pub struct QwenConditionalPipelineBindings {
    external_experts: bool,
}

fn conditional_expert_targets(
    architecture: &ConditionalArchitecture,
    group: usize,
    index: usize,
    unit: &MlxModule<ConditionalUnit<MlxNeuralBackend>>,
) -> Result<BTreeSet<String>, Error> {
    let groups = match &unit.inner {
        ConditionalUnit::Vision(_) => return Ok(BTreeSet::new()),
        ConditionalUnit::Target(block) => hybrid::unit_parallel_parameter_groups(
            &Unit::Target(block.clone()),
            &architecture.parsed().text,
            0,
            index,
        ),
        ConditionalUnit::Prediction(prediction) => hybrid::unit_parallel_parameter_groups(
            &Unit::Prediction(prediction.clone()),
            &architecture.parsed().text,
            group.checked_sub(1).ok_or_else(|| {
                Error::Parallel("conditional Qwen prediction unit has no text group".into())
            })?,
            index,
        ),
    }?;
    Ok(parameter_role_targets(
        &groups,
        ParameterRole::ExpertIntermediate,
    ))
}

impl QwenConditionalPipelineBindings {
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

    pub fn model_type<'a>(&self, architecture: &'a ConditionalArchitecture) -> &'a str {
        &architecture.parsed().text.model_type
    }

    pub fn static_units(
        &self,
        architecture: &ConditionalArchitecture,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
    }

    pub fn begin_pipeline_ingress(
        &self,
        architecture: &mut ConditionalArchitecture,
        typed: input::ModelInput<'_>,
        offset: i32,
        parallel: Option<&safemlx::distributed::Group>,
        stream: &Stream,
    ) -> Result<hybrid::ConditionalPipelineVisionState<crate::MlxTensor>, Error> {
        let parsed = architecture.parsed().clone();
        input::validate(typed)?;
        let mut token_storage = Vec::new();
        let mut grids = Vec::new();
        let mut pixels = Vec::new();
        enum Kind {
            Text(usize),
            Projected(usize, usize),
            Image(usize, usize),
            Video(usize, usize),
        }
        let mut kinds = Vec::new();
        for (original, part) in typed.parts.iter().enumerate() {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    token_storage.push(tokens.clone());
                    kinds.push(Kind::Text(token_storage.len() - 1));
                }
                (input::Modality::Text, input::InputPayload::Embeddings(embeddings)) => {
                    token_storage.push(input::token_ids_array(
                        &vec![0; usize::try_from(embeddings.dim(1)).unwrap_or_default()],
                        stream,
                    )?);
                    kinds.push(Kind::Projected(token_storage.len() - 1, original));
                }
                (
                    modality @ (input::Modality::Image | input::Modality::Video),
                    input::InputPayload::Tensor(tensor),
                ) => {
                    let ingress = super::qwen_media_ingress(
                        modality,
                        tensor,
                        part.metadata,
                        stream,
                        |input| {
                            eredu_architectures::media_plan::qwen_hybrid_ingress(&parsed, input)
                        },
                    )?;
                    token_storage.push(ingress.tokens);
                    grids.push(ingress.patch_grid);
                    pixels.push(tensor.clone());
                    let token = token_storage.len() - 1;
                    let grid = grids.len() - 1;
                    kinds.push(if modality == input::Modality::Image {
                        Kind::Image(token, grid)
                    } else {
                        Kind::Video(token, grid)
                    });
                }
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Embeddings(embeddings),
                ) => {
                    token_storage.push(input::token_ids_array(
                        &vec![0; usize::try_from(embeddings.dim(1)).unwrap_or_default()],
                        stream,
                    )?);
                    kinds.push(Kind::Projected(token_storage.len() - 1, original));
                }
                (modality, _) => {
                    return Err(Error::Parallel(format!(
                        "Qwen3.5 does not support this {} payload",
                        modality.as_str()
                    )))
                }
            }
        }
        let mut parts = Vec::with_capacity(kinds.len());
        for kind in kinds {
            parts.push(match kind {
                Kind::Text(token) => InputPart::Text(&token_storage[token]),
                Kind::Projected(token, original) => {
                    let input::InputPayload::Embeddings(embeddings) = typed.parts[original].payload
                    else {
                        unreachable!()
                    };
                    InputPart::Projected {
                        tokens: &token_storage[token],
                        embeddings,
                    }
                }
                Kind::Image(token, grid) => InputPart::Image {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
                Kind::Video(token, grid) => InputPart::Video {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
            });
        }
        let pixel_refs = pixels.iter().collect::<Vec<_>>();
        let pixels = if pixel_refs.is_empty() {
            None
        } else {
            Some(safemlx::ops::concatenate_axis(&pixel_refs, 0, stream)?)
        };
        let parts = neutral_input_parts(&parts);
        let pixels = crate::composition::tensor_opt(pixels.as_ref());
        match parallel {
            Some(parallel) => architecture
                .begin_pipeline_target_parallel(&parts, pixels, None, offset, parallel, stream),
            None => architecture.begin_pipeline_target(&parts, pixels, None, offset, stream),
        }
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub fn layer_bindings(
        &self,
        architecture: &ConditionalArchitecture,
        group: usize,
        index: usize,
        layer: &MlxModule<ConditionalUnit<MlxNeuralBackend>>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let expert_targets = conditional_expert_targets(architecture, group, index, layer)?;
        let is_vision = <ConditionalArchitecture as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::group_transport(architecture, group)
        .kind
            == eredu_runtime::ArchitectureGroupKind::VisionEncoder;
        let recipes = if is_vision {
            BTreeMap::new()
        } else {
            let layout = conditional_unit_layout(architecture)?;
            let ordinal = layout.ordinal(group, index).ok_or_else(|| {
                Error::Parallel(format!(
                    "conditional Qwen has no unit {index} in group {group}"
                ))
            })?;
            let vision_units = (0..layout.group_count())
                .filter(|&slot| {
                    <ConditionalArchitecture as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::group_transport(architecture, slot)
                    .kind
                        == eredu_runtime::ArchitectureGroupKind::VisionEncoder
                })
                .filter_map(|slot| layout.group_range(slot))
                .map(|range| range.len())
                .sum::<usize>();
            let flat = ordinal.checked_sub(vision_units).ok_or_else(|| {
                Error::Parallel("conditional Qwen text unit precedes its vision graph".into())
            })?;
            hybrid::unit_recipes(store, &architecture.parsed().text, flat)
                .map_err(Error::UnsupportedArchitecture)?
        };
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && parameter_name_in_targets(name, &expert_targets)
        })
        .map_err(Into::into)
    }

    pub fn cartesian_layer_bindings(
        &self,
        architecture: &ConditionalArchitecture,
        group: usize,
        index: usize,
        global_layer: &MlxModule<ConditionalUnit<MlxNeuralBackend>>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Vec<WeightBinding>, Error> {
        let expert_targets = conditional_expert_targets(architecture, group, index, global_layer)?;
        let is_vision = <ConditionalArchitecture as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::group_transport(architecture, group)
        .kind
            == eredu_runtime::ArchitectureGroupKind::VisionEncoder;
        let recipes = if is_vision {
            BTreeMap::new()
        } else {
            let layout = conditional_unit_layout(architecture)?;
            let ordinal = layout.ordinal(group, index).ok_or_else(|| {
                Error::Parallel(format!(
                    "conditional Qwen has no unit {index} in group {group}"
                ))
            })?;
            let vision_units = (0..layout.group_count())
                .filter(|&slot| {
                    <ConditionalArchitecture as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::group_transport(architecture, slot)
                    .kind
                        == eredu_runtime::ArchitectureGroupKind::VisionEncoder
                })
                .filter_map(|slot| layout.group_range(slot))
                .map(|range| range.len())
                .sum::<usize>();
            let flat = ordinal.checked_sub(vision_units).ok_or_else(|| {
                Error::Parallel("conditional Qwen text unit precedes its vision graph".into())
            })?;
            hybrid::unit_recipes(store, &architecture.parsed().text, flat)
                .map_err(Error::UnsupportedArchitecture)?
        };
        let bindings = build_module_bindings_with_recipes_excluding(
            global_layer,
            "",
            store,
            recipes,
            |name| self.external_experts && parameter_name_in_targets(name, &expert_targets),
        )?;
        match layout {
            Some(layout) => {
                let root = <ConditionalArchitecture as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::unit_path(architecture, group, index)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                shard_layer_bindings(bindings, &root, store, layout)
            }
            None => Ok(bindings),
        }
    }

    pub fn expert_parallel_assignment(
        &self,
        architecture: &ConditionalArchitecture,
        topology: crate::backend::MlxParallelContext,
    ) -> Result<Option<crate::backend::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        let parsed = architecture.parsed();
        if !parsed.text.is_moe() {
            return Err(Error::Parallel(
                "conditional Qwen3.5 PP+EP requires a routed text checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::runtime::distributed::expert::ExpertAssignment::balanced(
                parsed.text.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }
}

impl QwenHybridPipelineBindings {
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

    pub fn model_type<'a>(&self, architecture: &'a Architecture) -> &'a str {
        &architecture.config().model_type
    }

    pub fn embedded_mtp_len(&self, architecture: &Architecture) -> usize {
        architecture.config().mtp_num_hidden_layers.max(0) as usize
    }

    pub fn static_units(
        &self,
        architecture: &Architecture,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
    }

    pub fn layer_count(&self, architecture: &Architecture, group: usize) -> Result<usize, Error> {
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_unit_count(
            architecture,
            group,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn flat_index(
        &self,
        architecture: &Architecture,
        group: usize,
        index: usize,
    ) -> Result<usize, Error> {
        if index >= self.layer_count(architecture, group)? {
            return Err(Error::Parallel(format!(
                "Qwen hybrid has no unit {index} in group {group}"
            )));
        }
        unit_layout(architecture)?
            .ordinal(group, index)
            .ok_or_else(|| {
                Error::Parallel(format!("Qwen hybrid has no unit {index} in group {group}"))
            })
    }

    pub fn layer_bindings(
        &self,
        architecture: &Architecture,
        group: usize,
        index: usize,
        layer: &MlxModule<Block>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let expert_targets = parameter_role_targets(
            &hybrid::unit_parallel_parameter_groups(layer, architecture.config(), group, index)?,
            ParameterRole::ExpertIntermediate,
        );
        let recipes = hybrid::unit_recipes(
            store,
            architecture.config(),
            self.flat_index(architecture, group, index)?,
        )
        .map_err(Error::UnsupportedArchitecture)?;
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && parameter_name_in_targets(name, &expert_targets)
        })
        .map_err(Into::into)
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub fn expert_parallel_assignment(
        &self,
        architecture: &Architecture,
        topology: crate::backend::MlxParallelContext,
    ) -> Result<Option<crate::backend::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        let config = architecture.config();
        if !config.is_moe() {
            return Err(Error::Parallel(
                "Qwen hybrid PP+EP requires a sparse-MoE checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::runtime::distributed::expert::ExpertAssignment::balanced(
                config.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cartesian_layer_bindings(
        &self,
        architecture: &Architecture,
        group: usize,
        index: usize,
        global_layer: &MlxModule<Block>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::runtime::distributed::expert::ExpertAssignment>,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings = self.layer_bindings(architecture, group, index, global_layer, store)?;
        match layout {
            Some(layout) => {
                let root =
                    <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::unit_path(
                        architecture,
                        group,
                        index,
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                shard_layer_bindings(bindings, &root, store, layout)
            }
            None => Ok(bindings),
        }
    }
}

/// Canonical independent-expert catalog for selected architecture-owned units.
pub fn expert_catalog_selected(
    config: &HybridConfig,
    store: &dyn CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = hybrid::expert_residency_catalog(store, config)
        .map_err(Error::UnsupportedArchitecture)?;
    let units = catalog.into_units_selected_by_owner(owns_unit);
    crate::composition::architecture_expert_units(units, store, layout)
}

const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _config: &HybridConfig,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

struct HybridGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::qwen::GgufTensorCatalog for HybridGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        GgufTensorNames::contains_gguf_tensor(self.0, name)
    }
}

struct HybridVisionGgufCatalog<'a>(&'a GgufCheckpoint);

impl vision::VisionGgufCatalog for HybridVisionGgufCatalog<'_> {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.0
            .catalog()
            .tensors()
            .find(|tensor| tensor.descriptor().name == name)
            .map(|tensor| tensor.descriptor().row_major_shape())
            .and_then(|shape| {
                shape
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
            })
    }
}

fn prepare_hybrid_gguf_store(
    checkpoint: &GgufCheckpoint,
    projector: Option<&GgufCheckpoint>,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
) -> Result<(ParsedHybridConfig, Arc<dyn CheckpointSource>), Error> {
    let mut parsed = hybrid::model_args_from_gguf_catalog(&HybridGgufCatalog(checkpoint), metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    checkpoint
        .catalog()
        .translated_outputs(hybrid::translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    parsed.text.linear_formats =
        gguf_quantization_configs(checkpoint, hybrid::translate_gguf_weight_name)?
            .into_iter()
            .map(|(name, config)| (name, config.into()))
            .collect();
    let text_plan = hybrid::gguf_plan(&parsed.text).map_err(Error::UnsupportedArchitecture)?;
    let text: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        &text_plan,
        hybrid::translate_gguf_weight_name,
        max_mapped_shards,
    )?);
    if parsed.text.variant == hybrid::HybridVariant::Qwen3Next {
        return Ok((parsed, text));
    }
    let Some(projector) = projector else {
        return Ok((parsed, text));
    };
    let projector_metadata =
        crate::backend::runtime::checkpoint::load::gguf_metadata(projector);
    let mut vision =
        vision::config_from_gguf_catalog(&HybridVisionGgufCatalog(projector), &projector_metadata)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let deepstack = vision.deepstack_layers();
    let translate = |name: &str| hybrid::translate_vision_gguf_weight_name(name, &deepstack);
    projector
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    vision.linear_formats = gguf_quantization_configs(&projector, translate)?
        .into_iter()
        .map(|(name, format)| (name, format.into()))
        .collect();
    let vision_plan = vision::gguf_plan(&vision, parsed.text.hidden_size)
        .map_err(Error::UnsupportedArchitecture)?;
    let vision_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        projector.clone(),
        &vision_plan,
        translate,
        max_mapped_shards,
    )?);
    parsed = hybrid::with_gguf_vision_projector(parsed, metadata, vision)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok((
        parsed,
        Arc::new(CompositeCheckpointSource::new([text, vision_source])?),
    ))
}

pub fn prepare_gguf_pipeline(
    checkpoint: &GgufCheckpoint,
    projector: Option<&GgufCheckpoint>,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
) -> Result<(ParsedHybridConfig, Arc<dyn CheckpointSource>), Error> {
    prepare_hybrid_gguf_store(checkpoint, projector, metadata, max_mapped_shards)
}

/// Loads a llama.cpp Qwen3-Next/Qwen3.5 text artifact through the same
/// neutral resident/bounded execution graph as SafeTensors.
pub(crate) fn load_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&GgufCheckpoint>,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(QwenHybridModel, Vec<u32>), Error> {
    if !matches!(
        source.architecture(),
        eredu_architectures::GgufArchitecture::Qwen35
            | eredu_architectures::GgufArchitecture::Qwen35Moe
            | eredu_architectures::GgufArchitecture::Qwen3Next
    ) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen hybrid GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    let (mut parsed, store) = prepare_hybrid_gguf_store(
        checkpoint,
        projector,
        metadata,
        options.max_mapped_shards(),
    )?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen hybrid GGUF", None, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let (store, materialization) = if let Some(quantization) = quantize_on_load {
        if parsed.vision.is_some() {
            let (store, target, report) =
                quantize_conditional_store(store, &parsed, quantization, stream)?;
            parsed = target;
            (store, Some(report))
        } else {
            let (store, target, report) =
                quantize_store(store, &parsed.text, quantization, stream)?;
            parsed.text = target;
            (store, Some(report))
        }
    } else {
        (store, None)
    };
    let mut model = if parsed.vision.is_some() {
        load_conditional_store(
            store,
            parsed,
            options,
            materialization,
            expert_options.is_some(),
            stream,
            weights_stream,
        )?
    } else {
        load_store(
            store,
            parsed,
            options,
            materialization,
            expert_options.is_some(),
            stream,
            weights_stream,
        )?
    };
    if let Some(expert_options) = expert_options {
        attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok((
        model,
        crate::composition::mlx::gguf_eos_token_ids(metadata)?,
    ))
}

impl MlxUnitPopulator<Block> for UnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<Block>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

type Resident =
    LayerwiseRuntime<Architecture, MlxNeuralBackend, MlxHybridState, MlxResidentPolicy<Block>>;
type Bounded = LayerwiseRuntime<
    Architecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<Block, UnitPopulator>,
>;
type ConditionalArchitecture = ConditionalLayeredModel<MlxNeuralBackend>;
type ConditionalResident = LayerwiseRuntime<
    ConditionalArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<ConditionalUnit<MlxNeuralBackend>>,
>;
type ConditionalBounded = LayerwiseRuntime<
    ConditionalArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<ConditionalUnit<MlxNeuralBackend>, ConditionalUnitPopulator>,
>;

enum Execution {
    Resident(Box<Resident>),
    Bounded(Box<Bounded>),
    ConditionalResident(Box<ConditionalResident>),
    ConditionalBounded(Box<ConditionalBounded>),
}

struct PreparedConditionalOutput {
    logits: crate::MlxTensor,
    hidden: Option<crate::MlxTensor>,
    tokens: crate::MlxTensor,
}

/// Neutral Qwen3-Next/Qwen3.5 text model bound to MLX storage policy.
pub struct QwenHybridModel {
    parsed: ParsedHybridConfig,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    execution: Execution,
    expert_cache: Option<ExpertCache>,
}

impl QwenHybridModel {
    /// Complete validated text and optional vision architecture policy.
    pub fn parsed_args(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    /// Validated neutral text policy.
    pub fn args(&self) -> &HybridConfig {
        &self.parsed.text
    }

    /// Conditional vision is handled by the composite binder.
    pub fn vision_config(&self) -> Option<&eredu_architectures::qwen::vision::VisionConfig> {
        self.parsed.vision.as_ref()
    }

    pub fn vision_spatial_merge_size(&self) -> Option<i32> {
        self.parsed
            .vision
            .as_ref()
            .map(|vision| vision.spatial_merge_size)
    }
    /// Effective model type selected by strict neutral configuration parsing.
    pub fn model_type(&self) -> &str {
        &self.parsed.text.model_type
    }
    /// Actual configured embedded prediction depth.
    pub fn mtp_len(&self) -> usize {
        self.parsed.text.mtp_num_hidden_layers.max(0) as usize
    }
    /// This initial binder is replicated; distributed construction installs topology separately.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::MlxParallelContext>> {
        None
    }
    /// Canonical residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }
    pub fn residency_metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }
    /// Whether all execution units are pinned on device.
    pub fn is_fully_resident(&self) -> bool {
        self.metadata.residency() == ExecutionResidency::FullyResident
    }
    /// Rank-local cache policy.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.state_layout.layers().clone())
    }
    /// Stable architecture fingerprint.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        hybrid::prompt_cache_architecture_fingerprint(&self.parsed.text)
    }
    /// Allocates the declared recurrent, convolution, KV, and MTP state.
    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone()).expect("validated Qwen hybrid state")
    }

    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxHybridState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                MlxHybridState::paged(self.state_layout.clone(), manager, None).map_err(Into::into)
            }
        }
    }

    pub fn expert_cache_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::expert_cache::ExpertCacheReport>,
        Error,
    > {
        Ok(self
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()?)
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ConditionalResident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ConditionalBounded(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        hybrid::state_identity(
            &self.parsed.text,
            &self.state_layout,
            0,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
        .prompt_cache_identity(&self.state_layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn save_prompt_cache(
        &self,
        cache: &mut MlxHybridState,
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
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
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
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut cache = MlxHybridState::paged(self.state_layout.clone(), manager, None)?;
        cache.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Exception::custom("prompt-cache prefix exceeds i32"))?,
            &identity.layer_prefix_offsets,
        )?;
        Ok((cache, manifest))
    }
    /// Logical residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report(),
            Execution::Bounded(runtime) => runtime.policy().residency_report(),
            Execution::ConditionalResident(runtime) => runtime.policy().residency_report(),
            Execution::ConditionalBounded(runtime) => runtime.policy().residency_report(),
        }
    }
    /// Dense disk-stream telemetry when selected.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) | Execution::ConditionalResident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
            Execution::ConditionalBounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }
    /// Executes the one neutral target group.
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let config = self.parsed.text.clone();
            let result = {
                let mut provider = cached_provider(&expert_cache, &config);
                self.forward_with_provider(tokens, cache, &mut provider, stream)
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(crate::composition::tensor_ref(tokens))];
            let input = ConditionalInput::Target {
                parts: &parts,
                pixels: None,
                mask: None,
            };
            return match &mut self.execution {
                Execution::ConditionalResident(runtime) => runtime.forward(input, cache, stream),
                Execution::ConditionalBounded(runtime) => runtime.forward(input, cache, stream),
                _ => unreachable!("conditional policy uses conditional execution"),
            }
            .map(crate::MlxTensor::into_array)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
        }
        let input = EmbeddedInput::target(crate::composition::tensor_ref(tokens), None);
        match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward(input, cache, stream),
            Execution::Bounded(runtime) => runtime.forward(input, cache, stream),
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => {
                unreachable!("text policy uses text execution")
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn forward_with_provider<P>(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(crate::composition::tensor_ref(tokens))];
            let input = ConditionalInput::Target {
                parts: &parts,
                pixels: None,
                mask: None,
            };
            let hook = |architecture: &mut ConditionalArchitecture,
                        group: usize,
                        index: usize,
                        unit: &mut ConditionalUnit<MlxNeuralBackend>,
                        hidden: &crate::MlxTensor,
                        state: &mut MlxHybridState,
                        forward: &mut hybrid::ConditionalForwardContext<crate::MlxTensor>,
                        context: &Stream| {
                architecture.forward_unit_with_provider(
                    group, index, unit, hidden, state, forward, provider, context,
                )
            };
            return match &mut self.execution {
                Execution::ConditionalResident(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        cache,
                        stream,
                        hook,
                        |_, _, _| Ok(()),
                    )
                    .map(|(output, _)| output),
                Execution::ConditionalBounded(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        cache,
                        stream,
                        hook,
                        |_, _, _| Ok(()),
                    )
                    .map(|(output, _)| output),
                _ => unreachable!("conditional policy uses conditional execution"),
            }
            .map(crate::MlxTensor::into_array)
            .map_err(|error| Error::Parallel(error.to_string()));
        }
        let input = EmbeddedInput::target(crate::composition::tensor_ref(tokens), None);
        let hook = |architecture: &mut Architecture,
                    group: usize,
                    index: usize,
                    unit: &mut Block,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxHybridState,
                    forward: &mut hybrid::ForwardContext<crate::MlxTensor>,
                    context: &Stream| {
            architecture.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, provider, context,
            )
        };
        match &mut self.execution {
            Execution::Resident(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map(|(output, _)| output),
            Execution::Bounded(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map(|(output, _)| output),
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => {
                unreachable!("text policy uses text execution")
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let hook = |path: &str, input: &crate::MlxTensor, output: &crate::MlxTensor| {
            observer
                .observe(&format!("{path}.input"), input.as_array())
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            observer
                .observe(&format!("{path}.output"), output.as_array())
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            observer
                .intervene(&format!("{path}.output"), output.as_array())
                .map(|replacement| replacement.map(crate::MlxTensor::from_array))
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
        };
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(crate::composition::tensor_ref(tokens))];
            let input = ConditionalInput::Target {
                parts: &parts,
                pixels: None,
                mask: None,
            };
            return match &mut self.execution {
                Execution::ConditionalResident(runtime) => {
                    runtime.forward_with_unit_hook(input, cache, stream, hook)
                }
                Execution::ConditionalBounded(runtime) => {
                    runtime.forward_with_unit_hook(input, cache, stream, hook)
                }
                _ => unreachable!("conditional policy uses conditional execution"),
            }
            .map(crate::MlxTensor::into_array)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
        }
        let input = EmbeddedInput::target(crate::composition::tensor_ref(tokens), None);
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_unit_hook(input, cache, stream, hook)
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_unit_hook(input, cache, stream, hook)
            }
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => unreachable!(),
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        if self.parsed.vision.is_some() {
            return self
                .prepared_conditional_forward(input, cache, stream, Some(observer))
                .map(|output| output.logits.into_array())
                .map_err(Error::Exception);
        }
        let tokens = input::text_token_ids(input, stream)?;
        self.forward_with_observer(&tokens, cache, stream, observer)
    }

    fn prepared_conditional_forward(
        &mut self,
        typed: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PreparedConditionalOutput, Exception> {
        input::validate(typed)?;
        let mut token_storage = Vec::new();
        let mut grids = Vec::new();
        let mut pixels = Vec::new();
        enum Kind {
            Text(usize),
            Projected(usize, usize),
            Image(usize, usize),
            Video(usize, usize),
        }
        let mut kinds = Vec::new();
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    token_storage.push(tokens.clone());
                    kinds.push(Kind::Text(token_storage.len() - 1));
                }
                (input::Modality::Text, input::InputPayload::Embeddings(embeddings)) => {
                    token_storage.push(input::token_ids_array(
                        &vec![0; usize::try_from(embeddings.dim(1)).unwrap_or_default()],
                        stream,
                    )?);
                    kinds.push(Kind::Projected(token_storage.len() - 1, kinds.len()));
                }
                (
                    modality @ (input::Modality::Image | input::Modality::Video),
                    input::InputPayload::Tensor(tensor),
                ) => {
                    let ingress = super::qwen_media_ingress(
                        modality,
                        tensor,
                        part.metadata,
                        stream,
                        |input| {
                            eredu_architectures::media_plan::qwen_hybrid_ingress(
                                &self.parsed,
                                input,
                            )
                        },
                    )?;
                    token_storage.push(ingress.tokens);
                    grids.push(ingress.patch_grid);
                    pixels.push(tensor.clone());
                    let token = token_storage.len() - 1;
                    let grid = grids.len() - 1;
                    kinds.push(if modality == input::Modality::Image {
                        Kind::Image(token, grid)
                    } else {
                        Kind::Video(token, grid)
                    });
                }
                (modality, _) => {
                    return Err(Exception::custom(format!(
                        "Qwen3.5 does not support this {} payload",
                        modality.as_str()
                    )))
                }
            }
        }
        let mut parts = Vec::with_capacity(kinds.len());
        for kind in kinds {
            parts.push(match kind {
                Kind::Text(token) => InputPart::Text(&token_storage[token]),
                Kind::Projected(token, original) => {
                    let input::InputPayload::Embeddings(embeddings) = typed.parts[original].payload
                    else {
                        unreachable!()
                    };
                    InputPart::Projected {
                        tokens: &token_storage[token],
                        embeddings,
                    }
                }
                Kind::Image(token, grid) => InputPart::Image {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
                Kind::Video(token, grid) => InputPart::Video {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
            });
        }
        let refs = pixels.iter().collect::<Vec<_>>();
        let pixels = if refs.is_empty() {
            None
        } else {
            Some(safemlx::ops::concatenate_axis(&refs, 0, stream)?)
        };
        let parts = neutral_input_parts(&parts);
        let model_input = ConditionalInput::Target {
            parts: &parts,
            pixels: crate::composition::tensor_opt(pixels.as_ref()),
            mask: None,
        };
        let token_refs = token_storage.iter().collect::<Vec<_>>();
        let tokens =
            crate::MlxTensor::from_array(safemlx::ops::concatenate_axis(&token_refs, 1, stream)?);
        if let Some(observer) = observer {
            let hook = |path: &str, input: &crate::MlxTensor, output: &crate::MlxTensor| {
                observer
                    .observe(&format!("{path}.input"), input.as_array())
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                observer
                    .observe(&format!("{path}.output"), output.as_array())
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                observer
                    .intervene(&format!("{path}.output"), output.as_array())
                    .map(|replacement| replacement.map(crate::MlxTensor::from_array))
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))
            };
            let logits = match &mut self.execution {
                Execution::ConditionalResident(runtime) => {
                    runtime.forward_with_unit_hook(model_input, cache, stream, hook)
                }
                Execution::ConditionalBounded(runtime) => {
                    runtime.forward_with_unit_hook(model_input, cache, stream, hook)
                }
                _ => return Err(Exception::custom("Qwen3.5 model is not conditional")),
            }
            .map_err(|error| Exception::custom(error.to_string()))?;
            return Ok(PreparedConditionalOutput {
                logits,
                hidden: None,
                tokens,
            });
        }
        let (logits, forward) = match &mut self.execution {
            Execution::ConditionalResident(runtime) => {
                runtime.forward_with_context_hook(model_input, cache, stream, |_, _, _| Ok(()))
            }
            Execution::ConditionalBounded(runtime) => {
                runtime.forward_with_context_hook(model_input, cache, stream, |_, _, _| Ok(()))
            }
            _ => return Err(Exception::custom("Qwen3.5 model is not conditional")),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(PreparedConditionalOutput {
            logits,
            hidden: forward.target_hidden().cloned(),
            tokens,
        })
    }

    pub fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, cache, stream)
    }

    pub fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
    }

    fn forward_mtp(
        &mut self,
        input: EmbeddedInput<'_, crate::MlxTensor>,
        tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let config = self.parsed.text.clone();
            let result = {
                let mut provider = cached_provider(&expert_cache, &config);
                self.forward_mtp_with_provider(input, tokens, cache, &mut provider, stream)
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        if self.parsed.vision.is_some() {
            let result = match input {
                EmbeddedInput::Target { tokens, mask } => {
                    let parts = [InputPart::Text(tokens)];
                    let input = ConditionalInput::Target {
                        parts: &parts,
                        pixels: None,
                        mask,
                    };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
                EmbeddedInput::Draft {
                    tokens,
                    hidden,
                    depth,
                } => {
                    let input = ConditionalInput::Draft {
                        tokens,
                        hidden,
                        depth,
                    };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
            }
            .map_err(|error| Exception::custom(error.to_string()))?;
            let hidden =
                result.1.target_hidden().cloned().ok_or_else(|| {
                    Exception::custom("conditional Qwen3.5 retained no hidden state")
                })?;
            return Ok(
                crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                    logits: result.0,
                    hidden,
                    tokens: tokens.clone(),
                },
            );
        }
        let result = match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            }
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => {
                unreachable!("text policy uses text execution")
            }
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = result
            .1
            .target_hidden()
            .cloned()
            .ok_or_else(|| Exception::custom("Qwen hybrid pass retained no hidden state"))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: result.0,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    fn forward_mtp_with_provider<P>(
        &mut self,
        input: EmbeddedInput<'_, crate::MlxTensor>,
        tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        if self.parsed.vision.is_some() {
            let conditional = match input {
                EmbeddedInput::Target { tokens, mask } => {
                    let parts = [InputPart::Text(tokens)];
                    let input = ConditionalInput::Target {
                        parts: &parts,
                        pixels: None,
                        mask,
                    };
                    let hook =
                        |architecture: &mut ConditionalArchitecture,
                         group: usize,
                         index: usize,
                         unit: &mut ConditionalUnit<MlxNeuralBackend>,
                         hidden: &crate::MlxTensor,
                         state: &mut MlxHybridState,
                         forward: &mut hybrid::ConditionalForwardContext<crate::MlxTensor>,
                         context: &Stream| {
                            architecture.forward_unit_with_provider(
                                group, index, unit, hidden, state, forward, provider, context,
                            )
                        };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
                EmbeddedInput::Draft {
                    tokens,
                    hidden,
                    depth,
                } => {
                    let input = ConditionalInput::Draft {
                        tokens,
                        hidden,
                        depth,
                    };
                    let hook =
                        |architecture: &mut ConditionalArchitecture,
                         group: usize,
                         index: usize,
                         unit: &mut ConditionalUnit<MlxNeuralBackend>,
                         hidden: &crate::MlxTensor,
                         state: &mut MlxHybridState,
                         forward: &mut hybrid::ConditionalForwardContext<crate::MlxTensor>,
                         context: &Stream| {
                            architecture.forward_unit_with_provider(
                                group, index, unit, hidden, state, forward, provider, context,
                            )
                        };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
            }
            .map_err(|error| Exception::custom(error.to_string()))?;
            let hidden =
                conditional.1.target_hidden().cloned().ok_or_else(|| {
                    Exception::custom("conditional Qwen3.5 retained no hidden state")
                })?;
            return Ok(
                crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                    logits: conditional.0,
                    hidden,
                    tokens: tokens.clone(),
                },
            );
        }
        let hook = |architecture: &mut Architecture,
                    group: usize,
                    index: usize,
                    unit: &mut Block,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxHybridState,
                    forward: &mut hybrid::ForwardContext<crate::MlxTensor>,
                    context: &Stream| {
            architecture.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, provider, context,
            )
        };
        let result = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward_with_unit_executor_and_context_hook(
                input,
                cache,
                stream,
                hook,
                |_, _, _| Ok(()),
            ),
            Execution::Bounded(runtime) => runtime.forward_with_unit_executor_and_context_hook(
                input,
                cache,
                stream,
                hook,
                |_, _, _| Ok(()),
            ),
            _ => unreachable!("text policy uses text execution"),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = result
            .1
            .target_hidden()
            .cloned()
            .ok_or_else(|| Exception::custom("Qwen hybrid pass retained no hidden state"))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: result.0,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }
}

impl CausalModel<MlxHybridState> for QwenHybridModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let output = if self.parsed.vision.is_some() {
            self.prepared_conditional_forward(input, cache, stream, None)?
                .logits
        } else {
            let tokens = input::text_token_ids(input, stream)?;
            crate::MlxTensor::from_array(
                self.forward(&tokens, cache, stream)
                    .map_err(|error| Exception::custom(error.to_string()))?,
            )
        };
        output
            .as_array()
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let output = self
            .forward(input_tokens.as_array(), cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        output
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }
}

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget for QwenHybridModel {
    type Cache = MlxHybridState;
    type DraftCache = MlxHybridState;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        cache.clear()?;
        if self.parsed.vision.is_some() {
            let prepared = self.prepared_conditional_forward(input, cache, stream, None)?;
            let hidden = prepared.hidden.ok_or_else(|| {
                Exception::custom("conditional Qwen3.5 prefill retained no target hidden state")
            })?;
            return Ok(
                crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                    logits: prepared.logits,
                    hidden,
                    tokens: prepared.tokens,
                },
            );
        }
        let tokens = input::text_token_ids(input, stream)?;
        let tokens = crate::MlxTensor::from_array(tokens);
        self.forward_mtp(EmbeddedInput::target(&tokens, None), &tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.forward_mtp(EmbeddedInput::target(tokens, None), tokens, cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = crate::MlxTensor::from_array(
            output
                .hidden
                .as_array()
                .try_index_device((.., ..sequence - 1, ..), stream)?,
        );
        let next =
            crate::MlxTensor::from_array(tokens.as_array().try_index_device((.., 1..), stream)?);
        for depth in 0..self.mtp_len() {
            let _ = self.forward_mtp(
                EmbeddedInput::draft(&next, &hidden, depth),
                &next,
                cache,
                stream,
            )?;
        }
        Ok(())
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        cache
            .deep_clone_state()
            .expect("evaluated Qwen hybrid state must be forkable")
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache
            .commit_segment_from(draft, hybrid::PREDICTION_STATE_SEGMENT)
            .expect("validated Qwen hybrid prediction state segment")
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
        let token = crate::MlxTensor::from_array(Array::from_slice(&[last_token], &[1, 1]));
        let output = self.forward_mtp(
            EmbeddedInput::draft(&token, hidden, draft_index),
            &token,
            cache,
            stream,
        )?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..self.mtp_len() {
            let _ = self.forward_mtp(
                EmbeddedInput::draft(tokens, hidden, depth),
                tokens,
                cache,
                stream,
            )?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

fn unit_layout(architecture: &Architecture) -> Result<ExecutionUnitLayout, Error> {
    let graph = <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::execution_graph(
        architecture,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_unit_count(
                architecture,
                group,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn conditional_unit_layout(
    architecture: &ConditionalArchitecture,
) -> Result<ExecutionUnitLayout, Error> {
    let graph = <ConditionalArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::execution_graph(architecture)
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <ConditionalArchitecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::group_unit_count(architecture, group)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn resolve_store(
    store: Arc<dyn CheckpointSource>,
    config: &HybridConfig,
) -> Result<Arc<dyn CheckpointSource>, Error> {
    let plan = hybrid::safetensors_plan(config).map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "{} checkpoint contract did not resolve: {validation:?}",
                config.model_type
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_store(
    store: Arc<dyn CheckpointSource>,
    source: &HybridConfig,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        HybridConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target = source.clone();
    target.fp8 = None;
    target.quantization = Some(quantization);
    target.linear_formats.clear();
    let source_architecture = Architecture::new(source.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_architecture = Architecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_layers = usize::try_from(source.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen hybrid layer count".into()))?;
    let total = target_layers
        .checked_add(usize::try_from(source.mtp_num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture("invalid Qwen hybrid MTP layer count".into())
        })?)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen hybrid layer count overflowed".into())
        })?;
    let source_layout = unit_layout(&source_architecture)?;
    let target_layout = unit_layout(&target_architecture)?;
    let source_static =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &source_architecture,
        )
        .clone();
    let target_static =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &target_architecture,
        )
        .clone();
    let (store, report) = quantize_parameterized_store(
        store,
        &source_static,
        &target_static,
        move |flat, stream| {
            construct_architecture_unit(
                &source_architecture,
                &source_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        move |flat, stream| {
            construct_architecture_unit(
                &target_architecture,
                &target_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        total,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

fn quantize_conditional_store(
    store: Arc<dyn CheckpointSource>,
    source: &ParsedHybridConfig,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        ParsedHybridConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target = source.clone();
    target.text.fp8 = None;
    target.text.quantization = Some(quantization);
    target.text.linear_formats.clear();
    let source_architecture = ConditionalArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_architecture = ConditionalArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let vision_layers = source
        .vision
        .as_ref()
        .expect("conditional configuration has vision")
        .layer_count();
    let target_layers = usize::try_from(source.text.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen hybrid layer count".into()))?;
    let prediction_layers = usize::try_from(source.text.mtp_num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen hybrid MTP depth".into()))?;
    let total = vision_layers
        .checked_add(target_layers)
        .and_then(|count| count.checked_add(prediction_layers))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("conditional unit count overflowed".into())
        })?;
    let source_layout = conditional_unit_layout(&source_architecture)?;
    let target_layout = conditional_unit_layout(&target_architecture)?;
    let source_static = <ConditionalArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::static_modules(&source_architecture)
    .clone();
    let target_static = <ConditionalArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::static_modules(&target_architecture)
    .clone();
    let (store, report) = quantize_parameterized_store(
        store,
        &source_static,
        &target_static,
        move |flat, stream| {
            construct_architecture_unit(
                &source_architecture,
                &source_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        move |flat, stream| {
            construct_architecture_unit(
                &target_architecture,
                &target_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        total,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

/// Loads SafeTensors through the generic component residency engine.
pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    load_safetensors_with_residency(
        artifact,
        eredu_runtime::WeightResidency::with_layers(options.into()),
        quantization,
        stream,
        weights_stream,
    )
}

pub fn load_safetensors_with_residency(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    let mut parsed = hybrid::model_args_from_config_value(artifact.config()?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen hybrid", parsed.text.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    let store = artifact.store();
    if parsed.vision.is_some() {
        let (store, materialization) = if let Some(quantization) = quantize_on_load {
            let (store, target, report) =
                quantize_conditional_store(store, &parsed, quantization, stream)?;
            parsed = target;
            (store, Some(report))
        } else {
            (store, None)
        };
        let mut model = load_conditional_store(
            store,
            parsed,
            options,
            materialization,
            expert_options.is_some(),
            stream,
            weights_stream,
        )?;
        if let Some(expert_options) = expert_options {
            attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let store = resolve_store(store, &parsed.text)?;
    let (store, materialization) = if let Some(quantization) = quantize_on_load {
        let (store, target, report) = quantize_store(store, &parsed.text, quantization, stream)?;
        parsed.text = target;
        (store, Some(report))
    } else {
        (store, None)
    };
    let mut model = load_store(
        store,
        parsed,
        options,
        materialization,
        expert_options.is_some(),
        stream,
        weights_stream,
    )?;
    if let Some(expert_options) = expert_options {
        attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}

fn load_conditional_store(
    store: Arc<dyn CheckpointSource>,
    parsed: ParsedHybridConfig,
    options: LayerWeightResidency,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    let mut architecture = ConditionalArchitecture::new(parsed.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let factory = ConditionalUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let binding = parsed.text.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| {
            external_experts && parameter_name_in_targets(key, &excluded_expert_targets)
        },
        |modules, store| {
            build_module_bindings_with_recipes(
                &MlxModule::new(modules.clone()),
                "",
                store,
                hybrid::static_recipes(store).map_err(Error::UnsupportedArchitecture)?,
            )
            .map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _| {
            let recipes = if address.group() == 0 {
                BTreeMap::new()
            } else {
                let flat = if address.group() == 1 {
                    address.index()
                } else {
                    binding.num_hidden_layers as usize + address.group() - 2
                };
                hybrid::unit_recipes(store, &binding, flat)
                    .map_err(Error::UnsupportedArchitecture)?
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
    metadata.set_model_type(parsed.text.model_type.clone());
    metadata.set_quantization(parsed.text.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        Execution::ConditionalResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        Execution::ConditionalBounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    let state_layout = hybrid::state_layout(&parsed.text)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(QwenHybridModel {
        parsed,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
    })
}

fn load_store(
    store: Arc<dyn CheckpointSource>,
    parsed: ParsedHybridConfig,
    options: LayerWeightResidency,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    let mut architecture = Architecture::new(parsed.text.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let factory = UnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let binding_config = parsed.text.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| {
            external_experts && parameter_name_in_targets(key, &excluded_expert_targets)
        },
        |modules, store| {
            let recipes =
                hybrid::static_recipes(store).map_err(Error::UnsupportedArchitecture)?;
            build_module_bindings_with_recipes(&MlxModule::new(modules.clone()), "", store, recipes)
                .map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _| {
            let flat = if address.group() == 0 {
                address.index()
            } else {
                binding_config.num_hidden_layers as usize + address.group() - 1
            };
            let recipes = hybrid::unit_recipes(store, &binding_config, flat)
                .map_err(Error::UnsupportedArchitecture)?;
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
    metadata.set_model_type(parsed.text.model_type.clone());
    metadata.set_quantization(parsed.text.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        Execution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        Execution::Bounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    let state_layout = hybrid::state_layout(&parsed.text)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(QwenHybridModel {
        parsed,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
    })
}

fn attach_expert_cache(
    model: &mut QwenHybridModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = expert_catalog_selected(model.args(), store.as_ref(), None, |_, _| true)?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}
