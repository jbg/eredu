// MLX artifact and residency binding for the neutral Qwen hybrid graph.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::qwen::{
    hybrid::{
        self, ConditionalInput, ConditionalLayeredModel, ConditionalUnit, EmbeddedInput,
        HybridConfig, HybridLayerPolicy, ParsedHybridConfig, Unit,
    },
    vision,
    vl::InputPart,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    store::{CheckpointSource, CompositeCheckpointSource, TensorSelection, WeightStoreBackend},
    WeightQuantization,
};
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, ExecutionResidency, ExecutionUnitLayout, ExpertIdentity,
    LayerWeightResidency, LayeredArchitecture, LayerwiseModelMetadata, LayerwiseRuntime,
    OffloadUnit, PagedCacheOptions, ResidencyReport, StaticUnitBindings, WeightBinding,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
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
        checkpoint::binding::{
            build_module_bindings, build_module_bindings_with_recipes,
            build_module_bindings_with_recipes_excluding, populate_module_from_lease_excluding,
        },
        checkpoint::binding_plan::{BindingPlan, PlannedBinding},
        checkpoint::{
            load::{gguf_quantization_configs, GgufTensorNames},
            quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::generic::{
            prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            MlxUnitFactory,
        },
        execution::layerwise::{
            open_safetensors_weight_store, quantize_parameterized_store, shard_layer_bindings,
        },
        media::input,
        residency::{
            expert_cache::{ExpertCache, ExpertCatalogEntry},
            expert_provider::{CachedGatedProductBankSpec, CachedGatedProductExpertProvider},
        },
    },
};

type Architecture = hybrid::LayeredModel<MlxBackend>;
type Block = Unit<MlxBackend>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "Array")]
#[doc(hidden)]
#[cfg(test)]
pub struct QwenHybridCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxBackend>,
    pub units: Vec<Block>,
}

#[cfg(test)]
impl QwenHybridCheckpointTemplate {
    pub fn new(config: HybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = Architecture::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let total = config.num_hidden_layers as usize + config.mtp_num_hidden_layers as usize;
        let units = (0..total)
            .map(|index| build_unit(&config, index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: architecture.into_static_modules(),
            units,
        })
    }
}

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "Array")]
#[doc(hidden)]
#[cfg(test)]
pub struct QwenConditionalCheckpointTemplate {
    pub static_modules: hybrid::ConditionalStaticModules<MlxBackend>,
    pub units: Vec<hybrid::ConditionalUnit<MlxBackend>>,
}

#[cfg(test)]
impl QwenConditionalCheckpointTemplate {
    pub fn new(parsed: ParsedHybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = ConditionalArchitecture::new(parsed, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let graph = <ConditionalArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::execution_graph(&architecture)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let mut units = Vec::new();
        for group in 0..graph.groups().len() {
            let count = <ConditionalArchitecture as LayeredArchitecture<
                MlxBackend,
                MlxHybridState,
            >>::group_unit_count(&architecture, group)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for index in 0..count {
                units.push(
                    <ConditionalArchitecture as LayeredArchitecture<
                        MlxBackend,
                        MlxHybridState,
                    >>::build_unit(&architecture, group, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                );
            }
        }
        Ok(Self {
            static_modules: <ConditionalArchitecture as LayeredArchitecture<
                MlxBackend,
                MlxHybridState,
            >>::static_modules(&architecture)
            .clone(),
            units,
        })
    }
}

#[derive(Clone)]
struct UnitFactory {
    config: HybridConfig,
    target_layers: usize,
    external_experts: bool,
}

#[derive(Clone)]
struct ConditionalUnitFactory {
    parsed: ParsedHybridConfig,
    vision_layers: usize,
    target_layers: usize,
    external_experts: bool,
}

impl MlxUnitFactory<ConditionalUnit<MlxBackend>> for ConditionalUnitFactory {
    fn build(
        &mut self,
        flat: usize,
        stream: &Stream,
    ) -> Result<ConditionalUnit<MlxBackend>, Error> {
        if flat < self.vision_layers {
            return vision::VisionBlock::new_with_root(
                self.parsed.vision.as_ref().expect("validated vision"),
                "model.visual",
                flat,
                stream,
            )
            .map(ConditionalUnit::Vision)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
        }
        let text = flat - self.vision_layers;
        if text < self.target_layers {
            hybrid::Block::new(&self.parsed.text, text, stream)
                .map(ConditionalUnit::Target)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        } else {
            hybrid::PredictionUnit::new(&self.parsed.text, text - self.target_layers, stream)
                .map(ConditionalUnit::Prediction)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
    }

    fn populate(
        &mut self,
        unit: &mut MlxModule<ConditionalUnit<MlxBackend>>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".experts.")
        })?;
        Ok(())
    }
}

/// Pipeline/loading adapter over the neutral Qwen hybrid units.
pub(crate) struct QwenHybridPipelineAdapter {
    config: HybridConfig,
    static_modules: eredu_architectures::decoder::StaticModules<MlxBackend>,
    external_experts: bool,
}

/// Pipeline/loading adapter over the neutral conditional Qwen3.5 graph.
pub(crate) struct QwenConditionalPipelineAdapter {
    parsed: ParsedHybridConfig,
    architecture: ConditionalArchitecture,
    external_experts: bool,
}

impl QwenConditionalPipelineAdapter {
    pub(crate) fn new(parsed: ParsedHybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = ConditionalArchitecture::new(parsed.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(Self {
            parsed,
            architecture,
            external_experts: false,
        })
    }

    pub(crate) fn new_external_experts(
        parsed: ParsedHybridConfig,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new(parsed, stream)?;
        adapter.external_experts = true;
        Ok(adapter)
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.parsed.text.model_type
    }

    pub(crate) fn static_units(
        &self,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    pub(crate) fn architecture_mut(&mut self) -> &mut ConditionalArchitecture {
        &mut self.architecture
    }

    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        typed: input::ModelInput<'_>,
        offset: i32,
        stream: &Stream,
    ) -> Result<hybrid::ConditionalPipelineVisionState<Array>, Error> {
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
                    let grid = part.metadata.patch_grid.ok_or_else(|| {
                        Error::Parallel(format!(
                            "Qwen3.5 {} input requires patch_grid metadata",
                            modality.as_str()
                        ))
                    })?;
                    let grid = input::patch_grid_from_array(grid, stream)?;
                    let merge = self
                        .parsed
                        .vision
                        .as_ref()
                        .expect("validated conditional vision")
                        .spatial_merge_size;
                    let merged = grid
                        .iter()
                        .try_fold(0_i32, |total, &(time, height, width)| {
                            if time <= 0
                                || height <= 0
                                || width <= 0
                                || height % merge != 0
                                || width % merge != 0
                            {
                                return Err(Error::Parallel(
                                    "Qwen3.5 patch grid is not divisible by spatial merge size"
                                        .into(),
                                ));
                            }
                            total
                                .checked_add(time * (height / merge) * (width / merge))
                                .ok_or_else(|| {
                                    Error::Parallel("Qwen3.5 placeholder count overflowed".into())
                                })
                        })?;
                    let token_id = if modality == input::Modality::Image {
                        self.parsed.image_token_id
                    } else {
                        self.parsed.video_token_id
                    }
                    .ok_or_else(|| Error::Parallel("Qwen3.5 media token ID is absent".into()))?;
                    token_storage.push(input::token_ids_array(
                        &vec![
                            u32::try_from(token_id).map_err(|_| Error::Parallel(
                                "negative Qwen3.5 media token".into()
                            ))?;
                            usize::try_from(merged).map_err(|_| Error::Parallel(
                                "invalid Qwen3.5 placeholder count".into()
                            ))?
                        ],
                        stream,
                    )?);
                    grids.push(grid);
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
        self.architecture
            .begin_pipeline_target(&parts, pixels.as_ref(), None, offset, stream)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn pipeline_ingress_active(
        &self,
        state: &hybrid::ConditionalPipelineVisionState<Array>,
    ) -> bool {
        ConditionalArchitecture::pipeline_vision_active(state)
    }

    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &hybrid::ConditionalPipelineVisionState<Array>,
    ) -> Vec<Array> {
        ConditionalArchitecture::pipeline_retained_values(state)
    }

    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut hybrid::ConditionalPipelineVisionState<Array>,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        ConditionalArchitecture::replace_pipeline_retained_values(state, arrays)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn forward_pipeline_vision_layer(
        &mut self,
        index: usize,
        layer: &mut MlxModule<ConditionalUnit<MlxBackend>>,
        state: &mut hybrid::ConditionalPipelineVisionState<Array>,
        group: Option<&safemlx::distributed::Group>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let ConditionalUnit::Vision(block) = &mut layer.inner else {
            return Err(Error::Parallel(format!(
                "conditional Qwen3.5 vision range contains text unit {index}"
            )));
        };
        self.architecture
            .forward_pipeline_vision(index, block, state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        state: hybrid::ConditionalPipelineVisionState<Array>,
        group: Option<&safemlx::distributed::Group>,
        stream: &Stream,
    ) -> Result<hybrid::ConditionalPipelinePrepared<Array>, Error> {
        self.architecture
            .finish_pipeline_target(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let modules = <ConditionalArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules(&self.architecture);
        let recipes = static_transform_recipes(store)?;
        let mut units = Vec::new();
        for (role, module) in [("vision", MlxModule::new(modules.vision.clone()))] {
            let id = format!("qwen_conditional.static.{role}");
            if select(&id) {
                units.push(StaticUnitBindings::new(
                    id,
                    build_module_bindings_with_recipes(&module, "", store, recipes.clone())?,
                )?);
            }
        }
        if select("qwen_conditional.static.embedding") {
            units.push(StaticUnitBindings::new(
                "qwen_conditional.static.embedding",
                build_module_bindings_with_recipes(
                    &MlxModule::new(modules.text.embeddings.clone()),
                    "",
                    store,
                    recipes.clone(),
                )?,
            )?);
        }
        if select("qwen_conditional.static.norm") {
            units.push(StaticUnitBindings::new(
                "qwen_conditional.static.norm",
                build_module_bindings_with_recipes(
                    &MlxModule::new(modules.text.norm.clone()),
                    "",
                    store,
                    recipes,
                )?,
            )?);
        }
        if select("qwen_conditional.static.output") {
            if let Some(head) = &modules.text.lm_head {
                units.push(StaticUnitBindings::new(
                    "qwen_conditional.static.output",
                    build_module_bindings(&MlxModule::new(head.clone()), "", store)?,
                )?);
            }
        }
        Ok(units)
    }

    pub(crate) fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<ConditionalUnit<MlxBackend>>, Error> {
        self.new_cartesian_layer(group, index, None, stream)
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MlxModule<ConditionalUnit<MlxBackend>>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let recipes = if group == 0 {
            BTreeMap::new()
        } else {
            let flat = if group == 1 {
                index
            } else {
                self.parsed.text.num_hidden_layers as usize + group - 2
            };
            unit_recipes(store, &self.parsed.text, flat)?
        };
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && name.contains(".experts.")
        })
        .map_err(Into::into)
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let modules = <ConditionalArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules(&self.architecture);
        for group in eredu_architectures::decoder::static_parallel_parameter_groups::<MlxBackend>(
            &modules.text.embeddings,
            &modules.text.norm,
            modules.text.lm_head.as_ref(),
            "model",
        )? {
            planner.register(group)?;
        }
        for group in vision::static_parallel_parameter_groups::<MlxBackend>(
            &modules.vision,
            self.parsed.vision.as_ref().expect("validated vision"),
            "model.visual",
        )? {
            planner.register(group)?;
        }
        for index in 0..self
            .parsed
            .vision
            .as_ref()
            .expect("validated vision")
            .layer_count()
        {
            let unit = self.new_cartesian_layer(0, index, None, stream)?;
            let ConditionalUnit::Vision(block) = &unit.inner else {
                unreachable!()
            };
            for group in vision::block_parallel_parameter_groups(
                block,
                self.parsed.vision.as_ref().expect("validated vision"),
                "model.visual",
                index,
            )? {
                planner.register(group)?;
            }
        }
        for index in 0..self.parsed.text.num_hidden_layers as usize {
            let unit = hybrid::Block::<MlxBackend>::new(&self.parsed.text, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for group in hybrid::unit_parallel_parameter_groups(
                &hybrid::Unit::Target(unit),
                &self.parsed.text,
                0,
                index,
            )? {
                planner.register(group)?;
            }
        }
        for depth in 0..self.parsed.text.mtp_num_hidden_layers.max(0) as usize {
            let unit = hybrid::PredictionUnit::<MlxBackend>::new(&self.parsed.text, depth, stream)
                .map(hybrid::Unit::Prediction)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for group in
                hybrid::unit_parallel_parameter_groups(&unit, &self.parsed.text, depth + 1, 0)?
            {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn configure_parallel_static(
        &mut self,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let config = self.parsed.vision.as_ref().expect("validated vision");
        let widths = vision::local_merger_widths(config, "model.visual", layout)?;
        let replacement = vision::VisionStatic::<MlxBackend>::new_parallel_with_root(
            config.clone(),
            vision::VisionMode::WindowScheduled,
            "model.visual",
            &widths,
            stream,
        )?;
        <ConditionalArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(&mut self.architecture)
        .vision = replacement;
        Ok(())
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<MlxModule<ConditionalUnit<MlxBackend>>, Error> {
        let unit = match group {
            0 => {
                let config = self.parsed.vision.as_ref().expect("validated vision");
                match layout {
                    Some(layout) => {
                        let (heads, intermediate) =
                            vision::local_block_geometry(config, "model.visual", index, layout)?;
                        vision::VisionBlock::new_parallel_with_root(
                            config,
                            "model.visual",
                            index,
                            heads,
                            intermediate,
                            stream,
                        )
                    }
                    None => {
                        vision::VisionBlock::new_with_root(config, "model.visual", index, stream)
                    }
                }
                .map(ConditionalUnit::Vision)
            }
            1 => {
                let config = match layout {
                    Some(layout) => hybrid::local_unit_config(&self.parsed.text, 0, index, layout)?,
                    None => self.parsed.text.clone(),
                };
                hybrid::Block::new(&config, index, stream).map(ConditionalUnit::Target)
            }
            _ => {
                let config = match layout {
                    Some(layout) => {
                        hybrid::local_unit_config(&self.parsed.text, group - 1, 0, layout)?
                    }
                    None => self.parsed.text.clone(),
                };
                hybrid::PredictionUnit::new(&config, group - 2, stream)
                    .map(ConditionalUnit::Prediction)
            }
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(MlxModule::new(unit))
    }

    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MlxModule<ConditionalUnit<MlxBackend>>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let recipes = if group == 0 {
            BTreeMap::new()
        } else {
            let flat = if group == 1 {
                index
            } else {
                self.parsed.text.num_hidden_layers as usize + group - 2
            };
            unit_recipes(store, &self.parsed.text, flat)?
        };
        let global = self.new_cartesian_layer(group, index, None, stream)?;
        let bindings =
            build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |name| {
                self.external_experts && name.contains(".experts.")
            })?;
        match layout {
            Some(layout) => {
                let root = match group {
                    0 => format!("model.visual.blocks.{index}"),
                    1 => format!("model.layers.{index}"),
                    _ => format!("mtp.layers.{}", group - 2),
                };
                shard_layer_bindings(bindings, &root, store, layout)
            }
            None => {
                let _ = layer;
                Ok(bindings)
            }
        }
    }

    pub(crate) fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        if !self.parsed.text.is_moe() {
            return Err(Error::Parallel(
                "conditional Qwen3.5 PP+EP requires a routed text checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.parsed.text.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub(crate) fn local_state_geometry(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Vec<hybrid::HybridStateGeometry>, Error> {
        let config = &self.parsed.text;
        let total = config.num_hidden_layers as usize + config.mtp_num_hidden_layers as usize;
        let mut geometry = Vec::with_capacity(total);
        for flat in 0..total {
            let (group, index, policy) = if flat < config.num_hidden_layers as usize {
                (
                    0,
                    flat,
                    *config.layer_schedule.get(flat).ok_or_else(|| {
                        Error::Parallel(format!("Qwen hybrid has no layer {flat}"))
                    })?,
                )
            } else {
                (
                    flat - config.num_hidden_layers as usize + 1,
                    0,
                    HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full),
                )
            };
            let local = hybrid::local_unit_config(config, group, index, layout)?;
            geometry.push(match policy {
                HybridLayerPolicy::LinearAttention => {
                    hybrid::HybridStateGeometry::LinearAttention {
                        key_heads: local.linear_num_key_heads,
                        value_heads: local.linear_num_value_heads,
                    }
                }
                HybridLayerPolicy::SelfAttention(_) => hybrid::HybridStateGeometry::FullAttention {
                    key_value_heads: local.num_key_value_heads,
                },
            });
        }
        Ok(geometry)
    }
}

impl QwenHybridPipelineAdapter {
    pub(crate) fn new(config: HybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = Architecture::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(Self {
            config,
            static_modules: architecture.static_modules().clone(),
            external_experts: false,
        })
    }

    pub(crate) fn new_external_experts(
        config: HybridConfig,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new(config, stream)?;
        adapter.external_experts = true;
        Ok(adapter)
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.config.model_type
    }

    pub(crate) fn embedded_mtp_len(&self) -> usize {
        self.config.mtp_num_hidden_layers.max(0) as usize
    }

    pub(crate) fn static_units(
        &self,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let recipes = static_transform_recipes(store)?;
        let mut units = Vec::new();
        if select("qwen_hybrid.static.embedding") {
            units.push(StaticUnitBindings::new(
                "qwen_hybrid.static.embedding",
                build_module_bindings_with_recipes(
                    &MlxModule::new(self.static_modules.embeddings.clone()),
                    "",
                    store,
                    recipes.clone(),
                )?,
            )?);
        }
        if select("qwen_hybrid.static.norm") {
            units.push(StaticUnitBindings::new(
                "qwen_hybrid.static.norm",
                build_module_bindings_with_recipes(
                    &MlxModule::new(self.static_modules.norm.clone()),
                    "",
                    store,
                    recipes,
                )?,
            )?);
        }
        if select("qwen_hybrid.static.output") {
            if let Some(head) = &self.static_modules.lm_head {
                units.push(StaticUnitBindings::new(
                    "qwen_hybrid.static.output",
                    build_module_bindings(&MlxModule::new(head.clone()), "", store)?,
                )?);
            }
        }
        Ok(units)
    }

    pub(crate) fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.config.num_hidden_layers.max(0) as usize)
        } else if group <= self.embedded_mtp_len() {
            Ok(1)
        } else {
            Err(Error::Parallel(format!(
                "Qwen hybrid has no execution group {group}"
            )))
        }
    }

    fn flat_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        if index >= self.layer_count(group)? {
            return Err(Error::Parallel(format!(
                "Qwen hybrid has no unit {index} in group {group}"
            )));
        }
        Ok(if group == 0 {
            index
        } else {
            self.config.num_hidden_layers as usize + group - 1
        })
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<Block>, Error> {
        let flat = self.flat_index(group, index)?;
        build_unit(&self.config, flat, stream).map(MlxModule::new)
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MlxModule<Block>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let recipes = unit_recipes(store, &self.config, self.flat_index(group, index)?)?;
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && name.contains(".experts.")
        })
        .map_err(Into::into)
    }

    pub(crate) fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub(crate) fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        if !self.config.is_moe() {
            return Err(Error::Parallel(
                "Qwen hybrid PP+EP requires a sparse-MoE checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.config.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        _build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        for group in eredu_architectures::decoder::static_parallel_parameter_groups::<MlxBackend>(
            &self.static_modules.embeddings,
            &self.static_modules.norm,
            self.static_modules.lm_head.as_ref(),
            "model",
        )? {
            planner.register(group)?;
        }
        for group in 0..=self.embedded_mtp_len() {
            for index in 0..self.layer_count(group)? {
                let unit = self.new_layer(group, index, stream)?;
                for parameters in
                    hybrid::unit_parallel_parameter_groups(&unit, &self.config, group, index)?
                {
                    planner.register(parameters)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn local_state_geometry(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Vec<hybrid::HybridStateGeometry>, Error> {
        let mut geometry =
            Vec::with_capacity(self.config.num_hidden_layers as usize + self.embedded_mtp_len());
        for group in 0..=self.embedded_mtp_len() {
            for index in 0..self.layer_count(group)? {
                let local = hybrid::local_unit_config(&self.config, group, index, layout)?;
                let policy = if group == 0 {
                    self.config
                        .layer_schedule
                        .get(index)
                        .copied()
                        .ok_or_else(|| {
                            Error::Parallel(format!("Qwen hybrid has no layer {index}"))
                        })?
                } else {
                    HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full)
                };
                geometry.push(match policy {
                    HybridLayerPolicy::LinearAttention => {
                        hybrid::HybridStateGeometry::LinearAttention {
                            key_heads: local.linear_num_key_heads,
                            value_heads: local.linear_num_value_heads,
                        }
                    }
                    HybridLayerPolicy::SelfAttention(_) => {
                        hybrid::HybridStateGeometry::FullAttention {
                            key_value_heads: local.num_key_value_heads,
                        }
                    }
                });
            }
        }
        Ok(geometry)
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<MlxModule<Block>, Error> {
        let Some(layout) = layout else {
            return self.new_layer(group, index, stream);
        };
        self.flat_index(group, index)?;
        let local = hybrid::local_unit_config(&self.config, group, index, layout)?;
        let unit = if group == 0 {
            hybrid::Block::new(&local, index, stream).map(Block::Target)
        } else {
            hybrid::PredictionUnit::new(&local, group - 1, stream).map(Block::Prediction)
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(MlxModule::new(unit))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &MlxModule<Block>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        let bindings = self.layer_bindings(group, index, &global, store)?;
        match layout {
            Some(layout) => {
                let root = if group == 0 {
                    format!("model.layers.{index}")
                } else {
                    format!("mtp.layers.{}", group - 1)
                };
                shard_layer_bindings(bindings, &root, store, layout)
            }
            None => Ok(bindings),
        }
    }
}

fn build_unit(config: &HybridConfig, flat: usize, stream: &Stream) -> Result<Block, Error> {
    let target = config.num_hidden_layers as usize;
    if flat < target {
        hybrid::Block::new(config, flat, stream)
            .map(Block::Target)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    } else {
        hybrid::PredictionUnit::new(config, flat - target, stream)
            .map(Block::Prediction)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

/// Canonical independent-expert catalog for selected target/MTP units.
pub(crate) fn expert_catalog_selected(
    config: &HybridConfig,
    store: &dyn CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    mut include_layer: impl FnMut(usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    if !config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching requires a routed Qwen hybrid model".into(),
        ));
    }
    let target = config.num_hidden_layers as usize;
    let total = target + config.mtp_num_hidden_layers as usize;
    let mut entries = Vec::new();
    for layer in 0..total {
        if !include_layer(layer) {
            continue;
        }
        let root = if layer < target {
            format!("model.layers.{layer}.mlp.experts")
        } else {
            format!("mtp.layers.{}.mlp.experts", layer - target)
        };
        let packed = store
            .tensor_metadata(&format!("{root}.gate_up_proj"))
            .is_ok();
        let split_banks = ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .all(|name| store.tensor_metadata(&format!("{root}.{name}")).is_ok());
        for expert in 0..config.num_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let selection = TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            };
            let mut planned = Vec::new();
            if packed {
                for (target_name, required) in [
                    ("gate_up_proj", true),
                    ("gate_up_proj_scale_inv", false),
                    ("gate_up_proj_scales", false),
                    ("gate_up_proj_biases", false),
                    ("down_proj", true),
                    ("down_proj_scale_inv", false),
                    ("down_proj_scales", false),
                    ("down_proj_biases", false),
                ] {
                    let source = format!("{root}.{target_name}");
                    if store.tensor_metadata(&source).is_err() {
                        if required {
                            return Err(Error::UnsupportedArchitecture(format!(
                                "missing packed Qwen hybrid expert tensor {source}"
                            )));
                        }
                        continue;
                    }
                    planned.push(planned_expert_binding(
                        target_name,
                        DerivedWeightRecipe::source(source, selection.clone()),
                        store,
                    )?);
                }
            } else if split_banks {
                planned.push(planned_expert_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: ["gate_proj", "up_proj"]
                            .into_iter()
                            .map(|name| {
                                DerivedWeightRecipe::source(
                                    format!("{root}.{name}"),
                                    selection.clone(),
                                )
                            })
                            .collect(),
                    },
                    store,
                )?);
                planned.push(planned_expert_binding(
                    "down_proj",
                    DerivedWeightRecipe::source(format!("{root}.down_proj"), selection.clone()),
                    store,
                )?);
                add_split_bank_companions(&mut planned, &root, &selection, store)?;
            } else {
                let projection = |names: &[&str], suffix: &str| {
                    names
                        .iter()
                        .map(|name| format!("{root}.{expert}.{name}.{suffix}"))
                        .find(|name| store.tensor_metadata(name).is_ok())
                        .map(|name| DerivedWeightRecipe::source(name, TensorSelection::Full))
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "missing split Qwen hybrid expert {expert} tensor under {root}"
                            ))
                        })
                };
                let gate = projection(&["gate_proj", "w1"], "weight")?;
                let up = projection(&["up_proj", "w3"], "weight")?;
                let down = projection(&["down_proj", "w2"], "weight")?;
                planned.push(planned_expert_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![gate, up],
                        }],
                    },
                    store,
                )?);
                planned.push(planned_expert_binding(
                    "down_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![down],
                    },
                    store,
                )?);
                if let (Ok(gate), Ok(up), Ok(down)) = (
                    projection(&["gate_proj", "w1"], "weight_scale_inv"),
                    projection(&["up_proj", "w3"], "weight_scale_inv"),
                    projection(&["down_proj", "w2"], "weight_scale_inv"),
                ) {
                    planned.push(planned_expert_binding(
                        "gate_up_proj_scale_inv",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![gate, up],
                            }],
                        },
                        store,
                    )?);
                    planned.push(planned_expert_binding(
                        "down_proj_scale_inv",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![down],
                        },
                        store,
                    )?);
                }
            }
            let bindings = BindingPlan::new(planned)
                .and_then(|plan| plan.build_bindings(store))
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bindings = match layout {
                Some(layout) => shard_layer_bindings(bindings, &root, store, layout)?,
                None => bindings,
            };
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Qwen hybrid expert bytes overflowed".into())
                })
            })?;
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

fn add_split_bank_companions(
    planned: &mut Vec<PlannedBinding>,
    root: &str,
    selection: &TensorSelection,
    store: &dyn CheckpointSource,
) -> Result<(), Error> {
    for suffix in ["scales", "biases", "scale_inv"] {
        let gate = format!("{root}.gate_proj_{suffix}");
        let up = format!("{root}.up_proj_{suffix}");
        if store.tensor_metadata(&gate).is_ok() && store.tensor_metadata(&up).is_ok() {
            planned.push(planned_expert_binding(
                &format!("gate_up_proj_{suffix}"),
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate, selection.clone()),
                        DerivedWeightRecipe::source(up, selection.clone()),
                    ],
                },
                store,
            )?);
        }
        let down = format!("{root}.down_proj_{suffix}");
        if store.tensor_metadata(&down).is_ok() {
            planned.push(planned_expert_binding(
                &format!("down_proj_{suffix}"),
                DerivedWeightRecipe::source(down, selection.clone()),
                store,
            )?);
        }
    }
    Ok(())
}

fn planned_expert_binding(
    name: impl Into<String>,
    recipe: DerivedWeightRecipe,
    store: &dyn CheckpointSource,
) -> Result<PlannedBinding, Error> {
    let metadata = recipe.infer(store)?;
    Ok(PlannedBinding {
        target_name: name.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    })
}

pub(crate) fn cached_expert_spec(
    config: &HybridConfig,
    layer: usize,
) -> crate::backend::mlx::runtime::residency::expert_provider::CachedGatedProductBankSpec {
    let target = config.num_hidden_layers as usize;
    let root = if layer < target {
        format!("model.layers.{layer}.mlp.experts")
    } else {
        format!("mtp.layers.{}.mlp.experts", layer - target)
    };
    crate::backend::mlx::runtime::residency::expert_provider::CachedGatedProductBankSpec {
        hidden_dimensions: config.hidden_size,
        intermediate_dimensions: config.moe_intermediate_size,
        gate_up_quantization: config
            .linear_format(&format!("{root}.gate_up_proj"))
            .weight_quantization(),
        down_quantization: config
            .linear_format(&format!("{root}.down_proj"))
            .weight_quantization(),
        gate_up_bias: false,
        down_bias: false,
        policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
    }
}

fn cached_provider<'a>(
    cache: &'a ExpertCache,
    config: &'a HybridConfig,
) -> CachedGatedProductExpertProvider<'a, impl FnMut(usize) -> CachedGatedProductBankSpec + 'a> {
    CachedGatedProductExpertProvider::new(cache, move |layer| cached_expert_spec(config, layer))
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
            .map(|tensor| tensor.descriptor().mlx_shape())
            .and_then(|shape| {
                shape
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
            })
    }
}

fn translate_hybrid_vision_gguf(name: &str, deepstack: &[i32]) -> String {
    match name {
        "v.patch_embd.weight" => "model.visual.patch_embed.proj.weight.0".into(),
        "v.patch_embd.weight.1" => "model.visual.patch_embed.proj.weight.1".into(),
        _ => vision::translate_gguf_weight_name(name, deepstack),
    }
}

fn prepare_hybrid_gguf_store(
    model_path: &Path,
    checkpoint: &GgufCheckpoint,
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
    let Some(projector_path) =
        crate::composition::mlx::artifact::find_sibling_mmproj(model_path, "qwen35")?
    else {
        return Ok((parsed, text));
    };
    let projector = GgufCheckpoint::open(projector_path)?;
    let projector_metadata =
        crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&projector);
    let mut vision =
        vision::config_from_gguf_catalog(&HybridVisionGgufCatalog(&projector), &projector_metadata)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let deepstack = vision.deepstack_layers();
    let translate = |name: &str| translate_hybrid_vision_gguf(name, &deepstack);
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
        projector,
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

pub(crate) fn prepare_gguf_pipeline(
    model_path: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
) -> Result<(ParsedHybridConfig, Arc<dyn CheckpointSource>), Error> {
    prepare_hybrid_gguf_store(model_path, checkpoint, metadata, max_mapped_shards)
}

/// Loads a llama.cpp Qwen3-Next/Qwen3.5 text artifact through the same
/// neutral resident/bounded execution graph as SafeTensors.
pub fn load_gguf(
    model_path: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(QwenHybridModel, Vec<u32>), Error> {
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let (mut parsed, store) = prepare_hybrid_gguf_store(
        model_path,
        checkpoint,
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
    Ok((model, crate::backend::mlx::gguf_eos_token_ids(metadata)?))
}

impl MlxUnitFactory<Block> for UnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<Block, Error> {
        if index < self.target_layers {
            hybrid::Block::new(&self.config, index, stream)
                .map(Block::Target)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        } else {
            hybrid::PredictionUnit::new(&self.config, index - self.target_layers, stream)
                .map(Block::Prediction)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
    }

    fn populate(
        &mut self,
        unit: &mut MlxModule<Block>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".experts.")
        })?;
        Ok(())
    }
}

type Resident =
    LayerwiseRuntime<Architecture, MlxBackend, MlxHybridState, MlxResidentPolicy<Block>>;
type Bounded = LayerwiseRuntime<
    Architecture,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<Block, UnitFactory>,
>;
type ConditionalArchitecture = ConditionalLayeredModel<MlxBackend>;
type ConditionalResident = LayerwiseRuntime<
    ConditionalArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxResidentPolicy<ConditionalUnit<MlxBackend>>,
>;
type ConditionalBounded = LayerwiseRuntime<
    ConditionalArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<ConditionalUnit<MlxBackend>, ConditionalUnitFactory>,
>;

enum Execution {
    Resident(Box<Resident>),
    Bounded(Box<Bounded>),
    ConditionalResident(Box<ConditionalResident>),
    ConditionalBounded(Box<ConditionalBounded>),
}

struct PreparedConditionalOutput {
    logits: Array,
    hidden: Option<Array>,
    tokens: Array,
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
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
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
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
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
        Option<crate::backend::mlx::runtime::residency::expert_cache::ExpertCacheReport>,
        Error,
    > {
        Ok(self
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()?)
    }

    pub(crate) fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ConditionalResident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ConditionalBounded(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    fn prompt_cache_model_identity(
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
            let parts = [InputPart::Text(tokens)];
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
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
        }
        let input = EmbeddedInput::target(tokens, None);
        match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward(input, cache, stream),
            Execution::Bounded(runtime) => runtime.forward(input, cache, stream),
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => {
                unreachable!("text policy uses text execution")
            }
        }
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
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(tokens)];
            let input = ConditionalInput::Target {
                parts: &parts,
                pixels: None,
                mask: None,
            };
            let hook = |architecture: &mut ConditionalArchitecture,
                        group: usize,
                        index: usize,
                        unit: &mut ConditionalUnit<MlxBackend>,
                        hidden: &Array,
                        state: &mut MlxHybridState,
                        forward: &mut hybrid::ConditionalForwardContext<Array>,
                        context: &Stream| {
                if group == 0 {
                    <ConditionalArchitecture as LayeredArchitecture<
                        MlxBackend,
                        MlxHybridState,
                    >>::forward_unit(
                        architecture,
                        group,
                        index,
                        unit,
                        hidden,
                        state,
                        forward,
                        context,
                    )
                } else {
                    architecture.forward_unit_with_provider(
                        group, index, unit, hidden, state, forward, provider, context,
                    )
                }
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
            .map_err(|error| Error::Parallel(error.to_string()));
        }
        let input = EmbeddedInput::target(tokens, None);
        let hook = |architecture: &mut Architecture,
                    group: usize,
                    index: usize,
                    unit: &mut Block,
                    hidden: &Array,
                    state: &mut MlxHybridState,
                    forward: &mut hybrid::ForwardContext<Array>,
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
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let hook = |path: &str, input: &Array, output: &Array| {
            observer
                .observe(&format!("{path}.input"), input)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            observer
                .observe(&format!("{path}.output"), output)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            observer
                .intervene(&format!("{path}.output"), output)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
        };
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(tokens)];
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
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
        }
        let input = EmbeddedInput::target(tokens, None);
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_unit_hook(input, cache, stream, hook)
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_unit_hook(input, cache, stream, hook)
            }
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => unreachable!(),
        }
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
                .map(|output| output.logits)
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
                    let grid = part.metadata.patch_grid.ok_or_else(|| {
                        Exception::custom(format!(
                            "Qwen3.5 {} input requires patch_grid metadata",
                            modality.as_str()
                        ))
                    })?;
                    let grid = input::patch_grid_from_array(grid, stream)?;
                    let merge = self
                        .parsed
                        .vision
                        .as_ref()
                        .expect("conditional vision")
                        .spatial_merge_size;
                    let merged = grid
                        .iter()
                        .try_fold(0_i32, |total, &(time, height, width)| {
                            if time <= 0
                                || height <= 0
                                || width <= 0
                                || height % merge != 0
                                || width % merge != 0
                            {
                                return Err(Exception::custom(
                                    "Qwen3.5 patch grid is not divisible by merge size",
                                ));
                            }
                            total
                                .checked_add(time * (height / merge) * (width / merge))
                                .ok_or_else(|| Exception::custom("Qwen3.5 media length overflowed"))
                        })?;
                    let token_id = if modality == input::Modality::Image {
                        self.parsed.image_token_id
                    } else {
                        self.parsed.video_token_id
                    }
                    .expect("validated media token");
                    token_storage.push(input::token_ids_array(
                        &vec![
                            u32::try_from(token_id)
                                .map_err(|_| Exception::custom("negative Qwen3.5 media token"))?;
                            usize::try_from(merged)
                                .map_err(|_| Exception::custom("invalid Qwen3.5 media length"))?
                        ],
                        stream,
                    )?);
                    grids.push(grid);
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
        let model_input = ConditionalInput::Target {
            parts: &parts,
            pixels: pixels.as_ref(),
            mask: None,
        };
        let token_refs = token_storage.iter().collect::<Vec<_>>();
        let tokens = safemlx::ops::concatenate_axis(&token_refs, 1, stream)?;
        if let Some(observer) = observer {
            let hook = |path: &str, input: &Array, output: &Array| {
                observer
                    .observe(&format!("{path}.input"), input)
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                observer
                    .observe(&format!("{path}.output"), output)
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                observer
                    .intervene(&format!("{path}.output"), output)
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

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, cache, stream)
    }

    pub(crate) fn prefill_tensor_parallel(
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
        input: EmbeddedInput<'_, Array>,
        tokens: &Array,
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
        input: EmbeddedInput<'_, Array>,
        tokens: &Array,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
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
                    let hook = |architecture: &mut ConditionalArchitecture,
                                group: usize,
                                index: usize,
                                unit: &mut ConditionalUnit<MlxBackend>,
                                hidden: &Array,
                                state: &mut MlxHybridState,
                                forward: &mut hybrid::ConditionalForwardContext<Array>,
                                context: &Stream| {
                        if group == 0 {
                            <ConditionalArchitecture as LayeredArchitecture<
                                MlxBackend,
                                MlxHybridState,
                            >>::forward_unit(
                                architecture,
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                context,
                            )
                        } else {
                            architecture.forward_unit_with_provider(
                                group, index, unit, hidden, state, forward, provider, context,
                            )
                        }
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
                    let hook = |architecture: &mut ConditionalArchitecture,
                                group: usize,
                                index: usize,
                                unit: &mut ConditionalUnit<MlxBackend>,
                                hidden: &Array,
                                state: &mut MlxHybridState,
                                forward: &mut hybrid::ConditionalForwardContext<Array>,
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
                    hidden: &Array,
                    state: &mut MlxHybridState,
                    forward: &mut hybrid::ForwardContext<Array>,
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
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let output = if self.parsed.vision.is_some() {
            self.prepared_conditional_forward(input, cache, stream, None)?
                .logits
        } else {
            let tokens = input::text_token_ids(input, stream)?;
            self.forward(&tokens, cache, stream)
                .map_err(|error| Exception::custom(error.to_string()))?
        };
        output.try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
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
        self.forward_mtp(EmbeddedInput::target(&tokens, None), &tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.forward_mtp(EmbeddedInput::target(tokens, None), tokens, cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
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
            .commit_layer_range_from(draft, self.parsed.text.num_hidden_layers as usize)
            .expect("validated Qwen hybrid draft range")
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
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
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
        hidden: &Array,
        tokens: &Array,
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
    let graph = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::execution_graph(
        architecture,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::group_unit_count(
                architecture,
                group,
            )
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

fn unit_recipes(
    store: &dyn CheckpointSource,
    config: &HybridConfig,
    flat: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let target_layers = usize::try_from(config.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen hybrid layer count".into()))?;
    let mut recipes = BTreeMap::new();
    if flat < target_layers
        && config.variant == hybrid::HybridVariant::Qwen3Next
        && matches!(
            config.layer_schedule.get(flat),
            Some(HybridLayerPolicy::LinearAttention)
        )
    {
        let fused = format!("model.layers.{flat}.linear_attn.in_proj_qkvz.weight");
        if store.tensor_metadata(&fused).is_ok() {
            recipes.extend(
                hybrid::qwen3_next_fused_recipes(store, config, flat)
                    .map_err(Error::UnsupportedArchitecture)?
                    .iter()
                    .map(|(name, recipe)| (name.to_owned(), recipe.clone())),
            );
        }
    }
    if store.source_diagnostics()?.backend == WeightStoreBackend::Gguf {
        add_gguf_unit_transforms(&mut recipes, store, config, flat)?;
    }
    if !config.is_moe() {
        return Ok(recipes);
    }
    let root = if flat < target_layers {
        format!("model.layers.{flat}.mlp.experts")
    } else {
        format!("mtp.layers.{}.mlp.experts", flat - target_layers)
    };
    let gate_up = format!("{root}.gate_up_proj");
    if store.tensor_metadata(&gate_up).is_err() {
        let gate = format!("{root}.gate_proj");
        let up = format!("{root}.up_proj");
        if store.tensor_metadata(&gate).is_ok() && store.tensor_metadata(&up).is_ok() {
            recipes.insert(
                gate_up,
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate, TensorSelection::Full),
                        DerivedWeightRecipe::source(up, TensorSelection::Full),
                    ],
                },
            );
        } else {
            let inputs = (0..config.num_experts)
                .map(|expert| {
                    Ok(DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                format!("{root}.{expert}.gate_proj.weight"),
                                TensorSelection::Full,
                            ),
                            DerivedWeightRecipe::source(
                                format!("{root}.{expert}.up_proj.weight"),
                                TensorSelection::Full,
                            ),
                        ],
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            recipes.insert(gate_up, DerivedWeightRecipe::Stack { axis: 0, inputs });
        }
    }
    let down = format!("{root}.down_proj");
    if store.tensor_metadata(&down).is_err() {
        recipes.insert(
            down,
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: (0..config.num_experts)
                    .map(|expert| {
                        DerivedWeightRecipe::source(
                            format!("{root}.{expert}.down_proj.weight"),
                            TensorSelection::Full,
                        )
                    })
                    .collect(),
            },
        );
    }
    Ok(recipes)
}

fn static_transform_recipes(
    store: &dyn CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let mut recipes = BTreeMap::new();
    let patch0 = "model.visual.patch_embed.proj.weight.0";
    let patch1 = "model.visual.patch_embed.proj.weight.1";
    if store.source_metadata(patch0).is_ok() && store.source_metadata(patch1).is_ok() {
        recipes.insert(
            "model.visual.patch_embed.proj.weight".into(),
            DerivedWeightRecipe::Stack {
                axis: 2,
                inputs: vec![
                    DerivedWeightRecipe::source(patch0, TensorSelection::Full),
                    DerivedWeightRecipe::source(patch1, TensorSelection::Full),
                ],
            },
        );
    }
    if store.source_diagnostics()?.backend == WeightStoreBackend::Gguf {
        let name = "model.norm.weight";
        if store.source_metadata(name).is_ok() {
            recipes.insert(
                name.into(),
                DerivedWeightRecipe::SubtractOne {
                    input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                },
            );
        }
    }
    Ok(recipes)
}

fn add_gguf_unit_transforms(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    store: &dyn CheckpointSource,
    config: &HybridConfig,
    flat: usize,
) -> Result<(), Error> {
    let target_layers = usize::try_from(config.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen hybrid layer count".into()))?;
    let root = if flat < target_layers {
        format!("model.layers.{flat}")
    } else {
        format!("mtp.layers.{}", flat - target_layers)
    };
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ] {
        let name = format!("{root}.{suffix}");
        if store.source_metadata(&name).is_ok() {
            recipes.insert(
                name.clone(),
                DerivedWeightRecipe::SubtractOne {
                    input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                },
            );
        }
    }
    let a_log = format!("{root}.linear_attn.A_log");
    if store.source_metadata(&a_log).is_ok() {
        recipes.insert(
            a_log.clone(),
            DerivedWeightRecipe::NegLog {
                input: Box::new(DerivedWeightRecipe::source(
                    a_log.clone(),
                    TensorSelection::Full,
                )),
            },
        );
    }
    let shared_gate = format!("{root}.mlp.shared_expert_gate.weight");
    if let Ok(metadata) = store.source_metadata(&shared_gate) {
        if metadata.logical_shape.len() == 1 {
            recipes.insert(
                shared_gate.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(
                        shared_gate,
                        TensorSelection::Full,
                    )),
                    shape: vec![1, metadata.logical_shape[0]],
                },
            );
        }
    }
    let conv = format!("{root}.linear_attn.conv1d.weight");
    if let Ok(metadata) = store.source_metadata(&conv) {
        if metadata.logical_shape.len() == 2 {
            let source = DerivedWeightRecipe::source(conv.clone(), TensorSelection::Full);
            let source = if config.variant == hybrid::HybridVariant::Qwen3Next {
                source
            } else {
                qwen35_value_head_recipe(
                    "linear_attn.in_proj_qkv.weight",
                    source,
                    &metadata.logical_shape,
                    config,
                )?
                .unwrap_or_else(|| DerivedWeightRecipe::source(conv.clone(), TensorSelection::Full))
            };
            recipes.insert(
                conv.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(source),
                    shape: vec![metadata.logical_shape[0], 1, metadata.logical_shape[1]],
                },
            );
        }
    }
    if config.variant == hybrid::HybridVariant::Qwen3Next {
        return Ok(());
    }
    for suffix in [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.dt_bias",
        "linear_attn.A_log",
        "linear_attn.out_proj.weight",
    ] {
        let name = format!("{root}.{suffix}");
        let Ok(metadata) = store.source_metadata(&name) else {
            continue;
        };
        let base = recipes
            .remove(&name)
            .unwrap_or_else(|| DerivedWeightRecipe::source(name.clone(), TensorSelection::Full));
        if let Some(recipe) =
            qwen35_value_head_recipe(suffix, base, &metadata.logical_shape, config)?
        {
            recipes.insert(name, recipe);
        }
    }
    Ok(())
}

fn qwen35_value_head_recipe(
    suffix: &str,
    recipe: DerivedWeightRecipe,
    shape: &[usize],
    config: &HybridConfig,
) -> Result<Option<DerivedWeightRecipe>, Error> {
    let num_k = usize::try_from(config.linear_num_key_heads)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen3.5 key-head count".into()))?;
    let num_v = usize::try_from(config.linear_num_value_heads)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen3.5 value-head count".into()))?;
    if num_k == 0 || num_v == 0 || num_v % num_k != 0 {
        return Err(Error::UnsupportedArchitecture(
            "invalid Qwen3.5 value-head grouping".into(),
        ));
    }
    let repeats = num_v / num_k;
    let reorder =
        |input: DerivedWeightRecipe, axis: usize, head_width: usize, original: Vec<usize>| {
            let mut expanded = original.clone();
            expanded.splice(axis..=axis, [repeats, num_k, head_width]);
            let mut axes = (0..expanded.len()).collect::<Vec<_>>();
            axes.swap(axis, axis + 1);
            DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::Transpose {
                    input: Box::new(DerivedWeightRecipe::Reshape {
                        input: Box::new(input),
                        shape: expanded,
                    }),
                    axes,
                }),
                shape: original,
            }
        };
    if suffix.ends_with("in_proj_qkv.weight") {
        if shape.len() != 2 {
            return Ok(None);
        }
        let prefix = 2usize
            .checked_mul(num_k)
            .and_then(|value| value.checked_mul(config.linear_key_head_dim as usize))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture("Qwen3.5 value-tail width overflow".into())
            })?;
        if prefix >= shape[0] || (shape[0] - prefix) % num_v != 0 {
            return Ok(None);
        }
        let leading = DerivedWeightRecipe::Select {
            input: Box::new(recipe.clone()),
            selection: TensorSelection::Range {
                axis: 0,
                start: 0,
                end: prefix,
            },
        };
        let tail = DerivedWeightRecipe::Select {
            input: Box::new(recipe),
            selection: TensorSelection::Range {
                axis: 0,
                start: prefix,
                end: shape[0],
            },
        };
        return Ok(Some(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                leading,
                reorder(
                    tail,
                    0,
                    (shape[0] - prefix) / num_v,
                    vec![shape[0] - prefix, shape[1]],
                ),
            ],
        }));
    }
    let axis = if suffix.ends_with("out_proj.weight") {
        1
    } else {
        0
    };
    if axis >= shape.len() || shape[axis] % num_v != 0 {
        return Ok(None);
    }
    let admitted = suffix.ends_with("in_proj_z.weight")
        || suffix.ends_with("in_proj_a.weight")
        || suffix.ends_with("in_proj_b.weight")
        || suffix.ends_with("dt_bias")
        || suffix.ends_with("A_log")
        || suffix.ends_with("out_proj.weight");
    Ok(admitted.then(|| reorder(recipe, axis, shape[axis] / num_v, shape.to_vec())))
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
    let source_config = source.clone();
    let target_config = target.clone();
    let (store, report) = quantize_parameterized_store(
        store,
        <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &source_architecture,
        ),
        <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &target_architecture,
        ),
        move |flat, stream| {
            let mut factory = UnitFactory {
                config: source_config.clone(),
                target_layers,
                external_experts: false,
            };
            factory.build(flat, stream)
        },
        move |flat, stream| {
            let mut factory = UnitFactory {
                config: target_config.clone(),
                target_layers,
                external_experts: false,
            };
            factory.build(flat, stream)
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
    let source_parsed = source.clone();
    let target_parsed = target.clone();
    let (store, report) = quantize_parameterized_store(
        store,
        <ConditionalArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &source_architecture,
        ),
        <ConditionalArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &target_architecture,
        ),
        move |flat, stream| {
            let mut factory = ConditionalUnitFactory {
                parsed: source_parsed.clone(),
                vision_layers,
                target_layers,
                external_experts: false,
            };
            factory.build(flat, stream)
        },
        move |flat, stream| {
            let mut factory = ConditionalUnitFactory {
                parsed: target_parsed.clone(),
                vision_layers,
                target_layers,
                external_experts: false,
            };
            factory.build(flat, stream)
        },
        total,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

/// Loads SafeTensors through the generic component residency engine.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    load_safetensors_with_residency(
        model_dir,
        eredu_runtime::WeightResidency::with_layers(options.into()),
        quantization,
        stream,
        weights_stream,
    )
}

pub fn load_safetensors_with_residency(
    model_dir: impl AsRef<Path>,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    let model_dir = model_dir.as_ref();
    let value: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let mut parsed = hybrid::model_args_from_config_value(&value)
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
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
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
    let graph = <ConditionalArchitecture as LayeredArchitecture<
        MlxBackend,
        MlxHybridState,
    >>::execution_graph(&architecture)
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <ConditionalArchitecture as LayeredArchitecture<
                MlxBackend,
                MlxHybridState,
            >>::group_unit_count(&architecture, group)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let layout = ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let vision_layers = parsed
        .vision
        .as_ref()
        .expect("validated vision")
        .layer_count();
    let target_layers = parsed.text.num_hidden_layers as usize;
    let factory = ConditionalUnitFactory {
        parsed: parsed.clone(),
        vision_layers,
        target_layers,
        external_experts,
    };
    let binding = parsed.text.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        <ConditionalArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(&mut architecture),
        factory,
        layout,
        options,
        stream,
        weights_stream,
        move |key| {
            key.ends_with("rotary_emb.inv_freq")
                || (external_experts && key.contains(".experts."))
        },
        |modules, store| {
            build_module_bindings_with_recipes(
                &MlxModule::new(modules.clone()),
                "",
                store,
                static_transform_recipes(store)?,
            )
            .map_err(Into::into)
        },
        move |flat, unit, store, _| {
            let recipes = if flat < vision_layers {
                BTreeMap::new()
            } else {
                unit_recipes(store, &binding, flat - vision_layers)?
            };
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && name.contains(".experts."),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(parsed.text.model_type.clone());
    metadata.set_quantization(parsed.text.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        Execution::ConditionalResident(Box::new(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
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

pub(crate) fn load_parsed_config(model_dir: impl AsRef<Path>) -> Result<ParsedHybridConfig, Error> {
    let value: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    hybrid::model_args_from_config_value(&value)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
    let layout = unit_layout(&architecture)?;
    let target_layers = usize::try_from(parsed.text.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen hybrid layer count".into()))?;
    let factory = UnitFactory {
        config: parsed.text.clone(),
        target_layers,
        external_experts,
    };
    let binding_config = parsed.text.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        architecture.static_modules_mut(),
        factory,
        layout,
        options,
        stream,
        weights_stream,
        move |key| {
            key.ends_with("rotary_emb.inv_freq") || (external_experts && key.contains(".experts."))
        },
        |modules, store| {
            let recipes = static_transform_recipes(store)?;
            build_module_bindings_with_recipes(&MlxModule::new(modules.clone()), "", store, recipes)
                .map_err(Into::into)
        },
        move |flat, unit, store, _| {
            let recipes = unit_recipes(store, &binding_config, flat)?;
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && name.contains(".experts."),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(parsed.text.model_type.clone());
    metadata.set_quantization(parsed.text.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        Execution::Resident(Box::new(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
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
    let entries = expert_catalog_selected(model.args(), store.as_ref(), None, |_| true)?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}
