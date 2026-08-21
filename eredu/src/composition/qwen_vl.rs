// MLX artifact and residency binding for the neutral Qwen3-VL graph.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::qwen::{self, vision, vl};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    store::{CheckpointSource, CompositeCheckpointSource, TensorSelection},
    WeightQuantization,
};
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, ExecutionResidency, ExecutionUnitLayout,
    LayerWeightResidency, LayeredArchitecture, LayerwiseModelMetadata, LayerwiseRuntime,
    PagedCacheOptions, ResidencyReport, StaticUnitBindings, WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
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
            build_module_bindings_with_recipes, build_module_bindings_with_recipes_excluding,
            populate_module_from_lease_excluding,
        },
        checkpoint::{
            load::{gguf_quantization_configs, GgufTensorNames},
            quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitFactory,
            },
            layerwise::{open_safetensors_weight_store, quantize_parameterized_store},
        },
        media::input,
        residency::expert_cache::ExpertCache,
    },
};

type Architecture = vl::LayeredModel<MlxBackend>;
type Unit = vl::Unit<MlxBackend>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "Array")]
#[doc(hidden)]
#[cfg(test)]
pub struct QwenVlCheckpointTemplate {
    pub static_modules: vl::StaticModules<MlxBackend>,
    pub units: Vec<Unit>,
}

#[cfg(test)]
impl QwenVlCheckpointTemplate {
    pub fn new(args: vl::ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = Architecture::new(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let static_modules =
            <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
                &architecture,
            )
            .clone();
        let vision_layers = args.vision.layer_count();
        let mut factory = UnitFactory {
            args: args.clone(),
            vision_layers,
            external_experts: false,
        };
        let units = (0..vision_layers + args.text.num_hidden_layers as usize)
            .map(|index| factory.build(index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules,
            units,
        })
    }
}

struct TextGgufCatalog<'a>(&'a GgufCheckpoint);

impl qwen::GgufTensorCatalog for TextGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        GgufTensorNames::contains_gguf_tensor(self.0, name)
    }
}

struct VisionGgufCatalog<'a>(&'a GgufCheckpoint);

impl vision::VisionGgufCatalog for VisionGgufCatalog<'_> {
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

fn translate_vision_gguf(name: &str, deepstack: &[i32]) -> String {
    match name {
        "v.patch_embd.weight" => "model.visual.patch_embed.proj.weight.0".into(),
        "v.patch_embd.weight.1" => "model.visual.patch_embed.proj.weight.1".into(),
        _ => vision::translate_gguf_weight_name(name, deepstack),
    }
}

fn static_recipes(store: &dyn CheckpointSource) -> BTreeMap<String, DerivedWeightRecipe> {
    let mut recipes = BTreeMap::new();
    let first = "model.visual.patch_embed.proj.weight.0";
    let second = "model.visual.patch_embed.proj.weight.1";
    if store.tensor_metadata(first).is_ok() && store.tensor_metadata(second).is_ok() {
        recipes.insert(
            "model.visual.patch_embed.proj.weight".into(),
            DerivedWeightRecipe::Stack {
                axis: 2,
                inputs: vec![
                    DerivedWeightRecipe::source(first, TensorSelection::Full),
                    DerivedWeightRecipe::source(second, TensorSelection::Full),
                ],
            },
        );
    }
    recipes
}

fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &vl::ModelArgs,
    flat: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let vision_layers = args.vision.layer_count();
    if flat < vision_layers || !args.text.is_moe() {
        return Ok(BTreeMap::new());
    }
    let resolved = qwen::expert_recipes(
        store,
        &args.text,
        &args.text.parameter_root,
        flat - vision_layers,
    )
    .map_err(Error::UnsupportedArchitecture)?;
    Ok(BTreeMap::from([
        (resolved.target_gate_up, resolved.gate_up),
        (resolved.target_down, resolved.down),
    ]))
}

#[derive(Clone)]
struct UnitFactory {
    args: vl::ModelArgs,
    vision_layers: usize,
    external_experts: bool,
}

/// Pipeline and Cartesian-parallel binder for the same neutral Qwen3-VL
/// architecture used by resident and bounded execution.
pub(crate) struct QwenVlPipelineAdapter {
    architecture: Architecture,
    text: crate::composition::qwen::QwenParallelComposition,
    external_experts: bool,
}

impl QwenVlPipelineAdapter {
    pub(crate) fn new(args: vl::ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            architecture: Architecture::new(args.clone(), stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            text: crate::composition::qwen::QwenParallelComposition::new(args.text, stream)?,
            external_experts: false,
        })
    }

    pub(crate) fn new_external_experts(
        args: vl::ModelArgs,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.external_experts = true;
        adapter.text = crate::composition::qwen::QwenParallelComposition::new_external_experts(
            adapter.architecture.args().text.clone(),
            stream,
        )?;
        Ok(adapter)
    }

    pub(crate) fn args(&self) -> &vl::ModelArgs {
        self.architecture.args()
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.args().model_type
    }

    pub(crate) fn architecture_mut(&mut self) -> &mut Architecture {
        &mut self.architecture
    }

    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        typed: input::ModelInput<'_>,
        offset: i32,
        delta: Option<&Array>,
        stream: &Stream,
    ) -> Result<vl::PipelineVisionState<Array>, Error> {
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
                        Error::Parallel(format!(
                            "Qwen3-VL {} input requires patch_grid metadata",
                            modality.as_str()
                        ))
                    })?;
                    let grid = input::patch_grid_from_array(grid, stream)?;
                    let merge = self.args().vision.spatial_merge_size;
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
                                    "Qwen3-VL patch grid is not divisible by spatial merge size"
                                        .into(),
                                ));
                            }
                            total
                                .checked_add(time * (height / merge) * (width / merge))
                                .ok_or_else(|| {
                                    Error::Parallel("Qwen3-VL placeholder count overflowed".into())
                                })
                        })?;
                    let token_id = if modality == input::Modality::Image {
                        self.args().image_token_id
                    } else {
                        self.args().video_token_id
                    };
                    token_storage.push(input::token_ids_array(
                        &vec![
                            u32::try_from(token_id).map_err(|_| Error::Parallel(
                                "negative Qwen3-VL media token".into()
                            ))?;
                            usize::try_from(merged).map_err(|_| Error::Parallel(
                                "invalid Qwen3-VL placeholder count".into()
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
                (modality, _) => {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL does not support this {} payload",
                        modality.as_str()
                    )))
                }
            }
        }
        let mut parts = Vec::with_capacity(kinds.len());
        for kind in kinds {
            parts.push(match kind {
                Kind::Text(token) => vl::InputPart::Text(&token_storage[token]),
                Kind::Projected(token, original) => {
                    let input::InputPayload::Embeddings(embeddings) = typed.parts[original].payload
                    else {
                        unreachable!()
                    };
                    vl::InputPart::Projected {
                        tokens: &token_storage[token],
                        embeddings,
                    }
                }
                Kind::Image(token, grid) => vl::InputPart::Image {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
                Kind::Video(token, grid) => vl::InputPart::Video {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
            });
        }
        let pixel_refs = pixels.iter().collect::<Vec<_>>();
        let pixels = if pixel_refs.is_empty() {
            None
        } else {
            Some(concatenate_axis(&pixel_refs, 0, stream)?)
        };
        self.architecture
            .begin_pipeline(
                vl::ModelInput {
                    parts: &parts,
                    pixels: pixels.as_ref(),
                    mask: None,
                },
                offset,
                delta,
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn pipeline_ingress_active(&self, state: &vl::PipelineVisionState<Array>) -> bool {
        Architecture::pipeline_vision_active(state)
    }

    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &vl::PipelineVisionState<Array>,
    ) -> Vec<Array> {
        Architecture::pipeline_retained_values(state)
    }

    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut vl::PipelineVisionState<Array>,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        Architecture::replace_pipeline_retained_values(state, arrays)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn forward_pipeline_vision_layer(
        &mut self,
        index: usize,
        layer: &mut MlxModule<Unit>,
        state: &mut vl::PipelineVisionState<Array>,
        group: Option<&safemlx::distributed::Group>,
        stream: &Stream,
    ) -> Result<Vec<Array>, Error> {
        let Unit::Vision(block) = &mut layer.inner else {
            return Err(Error::Parallel(format!(
                "Qwen3-VL vision range contains text unit {index}"
            )));
        };
        self.architecture
            .forward_pipeline_vision(index, block, state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(self.pipeline_ingress_arrays(state))
    }

    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        state: vl::PipelineVisionState<Array>,
        group: Option<&safemlx::distributed::Group>,
        stream: &Stream,
    ) -> Result<vl::PipelinePrepared<Array>, Error> {
        self.architecture
            .finish_pipeline(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn static_units(
        &self,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    pub(crate) fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<Unit>, Error> {
        self.new_cartesian_layer(group, index, None, None, stream)
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MlxModule<Unit>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let recipes = if group == 1 {
            unit_recipes(store, self.args(), self.args().vision.layer_count() + index)?
        } else {
            BTreeMap::new()
        };
        Ok(build_module_bindings_with_recipes_excluding(
            layer,
            "",
            store,
            recipes,
            |name| self.external_experts && name.contains(".experts."),
        )?)
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let modules =
            <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
                &self.architecture,
            );
        let recipes = static_recipes(store);
        let mut units = Vec::new();
        if select("qwen_vl.static.vision") {
            units.push(StaticUnitBindings::new(
                "qwen_vl.static.vision",
                build_module_bindings_with_recipes(
                    &MlxModule::new(modules.vision.clone()),
                    "",
                    store,
                    recipes,
                )?,
            )?);
        }
        if select("qwen_vl.static.embedding") {
            units.push(StaticUnitBindings::new(
                "qwen_vl.static.embedding",
                build_module_bindings_with_recipes(
                    &MlxModule::new(modules.text.embeddings.clone()),
                    "",
                    store,
                    BTreeMap::new(),
                )?,
            )?);
        }
        if select("qwen_vl.static.norm") {
            units.push(StaticUnitBindings::new(
                "qwen_vl.static.norm",
                build_module_bindings_with_recipes(
                    &MlxModule::new(modules.text.norm.clone()),
                    "",
                    store,
                    BTreeMap::new(),
                )?,
            )?);
        }
        if select("qwen_vl.static.output") {
            if let Some(head) = &modules.text.lm_head {
                units.push(StaticUnitBindings::new(
                    "qwen_vl.static.output",
                    build_module_bindings_with_recipes(
                        &MlxModule::new(head.clone()),
                        "",
                        store,
                        BTreeMap::new(),
                    )?,
                )?);
            }
        }
        Ok(units)
    }

    pub(crate) fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        if !self.args().text.is_moe() {
            return Err(Error::Parallel(
                "Qwen3-VL PP+EP requires a routed text checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args().text.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let modules =
            <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
                &self.architecture,
            );
        for group in vision::static_parallel_parameter_groups::<MlxBackend>(
            &modules.vision,
            &self.args().vision,
            "model.visual",
        )? {
            planner.register(group)?;
        }
        for index in 0..self.args().vision.layer_count() {
            let block = vision::VisionBlock::<MlxBackend>::new_with_root(
                &self.args().vision,
                "model.visual",
                index,
                stream,
            )?;
            for group in vision::block_parallel_parameter_groups(
                &block,
                &self.args().vision,
                "model.visual",
                index,
            )? {
                planner.register(group)?;
            }
        }
        self.text.register_parallel_parameters(planner, stream)
    }

    pub(crate) fn configure_parallel_static(
        &mut self,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let widths = vision::local_merger_widths(&self.args().vision, "model.visual", layout)?;
        let replacement = vision::VisionStatic::<MlxBackend>::new_parallel_with_root(
            self.args().vision.clone(),
            vision::VisionMode::DeepStack,
            "model.visual",
            &widths,
            stream,
        )?;
        <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules_mut(
            &mut self.architecture,
        )
        .vision = replacement;
        Ok(())
    }

    pub(crate) fn local_key_value_heads(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Vec<i32>, Error> {
        eredu_architectures::qwen::local_key_value_heads(&self.args().text, layout)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<MlxModule<Unit>, Error> {
        match group {
            0 => {
                let block = match layout {
                    Some(layout) => {
                        let (heads, intermediate) = vision::local_block_geometry(
                            &self.args().vision,
                            "model.visual",
                            index,
                            layout,
                        )?;
                        vision::VisionBlock::new_parallel_with_root(
                            &self.args().vision,
                            "model.visual",
                            index,
                            heads,
                            intermediate,
                            stream,
                        )?
                    }
                    None => vision::VisionBlock::new_with_root(
                        &self.args().vision,
                        "model.visual",
                        index,
                        stream,
                    )?,
                };
                Ok(MlxModule::new(Unit::Vision(block)))
            }
            1 => {
                let block = self
                    .text
                    .new_cartesian_layer(0, index, layout, assignment, stream)?;
                Ok(MlxModule::new(Unit::Text(block.inner)))
            }
            _ => Err(Error::Parallel(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MlxModule<Unit>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        match &layer.inner {
            Unit::Vision(block) if group == 0 => {
                let global = MlxModule::new(Unit::Vision(vision::VisionBlock::new_with_root(
                    &self.args().vision,
                    "model.visual",
                    index,
                    stream,
                )?));
                let bindings = build_module_bindings_with_recipes(
                    if layout.is_some() { &global } else { layer },
                    "",
                    store,
                    BTreeMap::new(),
                )?;
                if let Some(layout) = layout {
                    crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
                        bindings, "", store, layout,
                    )
                } else {
                    let _ = block;
                    Ok(bindings)
                }
            }
            Unit::Text(block) if group == 1 => self.text.cartesian_layer_bindings(
                0,
                index,
                &MlxModule::new(block.clone()),
                store,
                layout,
                assignment,
                stream,
            ),
            _ => Err(Error::Parallel("Qwen3-VL unit/group mismatch".into())),
        }
    }
}

impl MlxUnitFactory<Unit> for UnitFactory {
    fn build(&mut self, flat: usize, stream: &Stream) -> Result<Unit, Error> {
        if flat < self.vision_layers {
            vision::VisionBlock::new_with_root(&self.args.vision, "model.visual", flat, stream)
                .map(Unit::Vision)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        } else {
            let layer = flat - self.vision_layers;
            qwen::new_block::<MlxBackend>(&self.args.text, layer, stream)
                .map(Unit::Text)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
    }

    fn populate(
        &mut self,
        unit: &mut MlxModule<Unit>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".experts.")
        })?;
        Ok(())
    }
}

type Resident = LayerwiseRuntime<Architecture, MlxBackend, MlxHybridState, MlxResidentPolicy<Unit>>;
type Bounded = LayerwiseRuntime<
    Architecture,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<Unit, UnitFactory>,
>;

enum Execution {
    Resident(Box<Resident>),
    Bounded(Box<Bounded>),
}

/// Neutral Qwen3-VL dense-or-MoE model bound to MLX storage policy.
pub struct QwenVlModel {
    args: vl::ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    execution: Execution,
    expert_cache: Option<ExpertCache>,
}

impl QwenVlModel {
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub fn args(&self) -> &vl::ModelArgs {
        &self.args
    }

    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        None
    }

    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }
    pub fn residency_metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    pub fn is_fully_resident(&self) -> bool {
        self.metadata.residency() == ExecutionResidency::FullyResident
    }

    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.state_layout.layers().clone())
    }

    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        vl::prompt_cache_architecture_fingerprint(&self.args)
    }

    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone()).expect("validated Qwen3-VL state layout")
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

    fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        vl::state_identity(
            &self.args,
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

    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report(),
            Execution::Bounded(runtime) => runtime.policy().residency_report(),
        }
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
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

    fn forward(
        &mut self,
        input: vl::ModelInput<'_, Array>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.text.clone();
            let result = {
                let mut provider =
                    crate::composition::qwen::expert::cached_provider(&expert_cache, &args);
                self.forward_with_provider(input, cache, &mut provider, stream)
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward(input, cache, stream),
            Execution::Bounded(runtime) => runtime.forward(input, cache, stream),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn forward_with_provider<P>(
        &mut self,
        input: vl::ModelInput<'_, Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        let hook = |architecture: &mut Architecture,
                    group: usize,
                    index: usize,
                    unit: &mut Unit,
                    hidden: &Array,
                    state: &mut MlxHybridState,
                    forward: &mut vl::ForwardContext<Array>,
                    context: &Stream| {
            architecture.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, provider, context,
            )
        };
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn prepared_forward(
        &mut self,
        typed: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
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
                    let token = input::token_ids_array(
                        &vec![0; usize::try_from(embeddings.dim(1)).unwrap_or_default()],
                        stream,
                    )?;
                    token_storage.push(token);
                    kinds.push(Kind::Projected(token_storage.len() - 1, kinds.len()));
                }
                (
                    modality @ (input::Modality::Image | input::Modality::Video),
                    input::InputPayload::Tensor(tensor),
                ) => {
                    let grid = part.metadata.patch_grid.ok_or_else(|| {
                        Exception::custom(format!(
                            "Qwen3-VL {} input requires patch_grid metadata",
                            modality.as_str()
                        ))
                    })?;
                    let grid = input::patch_grid_from_array(grid, stream)?;
                    let merge = self.args.vision.spatial_merge_size;
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
                                    "Qwen3-VL patch grid is not divisible by spatial merge size",
                                ));
                            }
                            total
                                .checked_add(time * (height / merge) * (width / merge))
                                .ok_or_else(|| {
                                    Exception::custom("Qwen3-VL placeholder count overflowed")
                                })
                        })?;
                    let token_id = if modality == input::Modality::Image {
                        self.args.image_token_id
                    } else {
                        self.args.video_token_id
                    };
                    let ids = vec![
                        u32::try_from(token_id).map_err(|_| Exception::custom(
                            "negative Qwen3-VL media token"
                        ))?;
                        usize::try_from(merged).map_err(|_| Exception::custom(
                            "invalid Qwen3-VL placeholder count"
                        ))?
                    ];
                    token_storage.push(input::token_ids_array(&ids, stream)?);
                    grids.push(grid);
                    pixels.push(tensor.clone());
                    let token_index = token_storage.len() - 1;
                    let grid_index = grids.len() - 1;
                    kinds.push(if modality == input::Modality::Image {
                        Kind::Image(token_index, grid_index)
                    } else {
                        Kind::Video(token_index, grid_index)
                    });
                }
                (modality, _) => {
                    return Err(Exception::custom(format!(
                        "Qwen3-VL does not support this {} payload",
                        modality.as_str()
                    )))
                }
            }
        }
        let mut parts = Vec::with_capacity(kinds.len());
        for kind in kinds {
            parts.push(match kind {
                Kind::Text(token) => vl::InputPart::Text(&token_storage[token]),
                Kind::Projected(token, original) => {
                    let input::InputPayload::Embeddings(embeddings) = typed.parts[original].payload
                    else {
                        unreachable!()
                    };
                    vl::InputPart::Projected {
                        tokens: &token_storage[token],
                        embeddings,
                    }
                }
                Kind::Image(token, grid) => vl::InputPart::Image {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
                Kind::Video(token, grid) => vl::InputPart::Video {
                    tokens: &token_storage[token],
                    grid: &grids[grid],
                },
            });
        }
        let pixel_refs = pixels.iter().collect::<Vec<_>>();
        let pixels = if pixel_refs.is_empty() {
            None
        } else {
            Some(concatenate_axis(&pixel_refs, 0, stream)?)
        };
        self.forward(
            vl::ModelInput {
                parts: &parts,
                pixels: pixels.as_ref(),
                mask: None,
            },
            cache,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Runs a complete multimodal prefill and returns logits for every input
    /// position. Generation-facing callers normally use `CausalModel`, which
    /// selects only the final position.
    pub fn prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.prepared_forward(input, cache, stream)
    }

    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prepared_forward(input, cache, stream)
            .map_err(Error::Exception)
    }

    pub(crate) fn decode_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let parts = [vl::InputPart::Text(tokens)];
        self.forward(
            vl::ModelInput {
                parts: &parts,
                pixels: None,
                mask: None,
            },
            cache,
            stream,
        )
    }
}

impl CausalModel<MlxHybridState> for QwenVlModel {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.prepared_forward(input, cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let parts = [vl::InputPart::Text(input_tokens)];
        self.forward(
            vl::ModelInput {
                parts: &parts,
                pixels: None,
                mask: None,
            },
            cache,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?
        .try_index_device((.., -1, ..), stream)
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
    args: &vl::ModelArgs,
) -> Result<Arc<dyn CheckpointSource>, Error> {
    let plan = vl::safetensors_plan(args).map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "{} checkpoint contract did not resolve: {validation:?}",
                args.model_type
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_store(
    store: Arc<dyn CheckpointSource>,
    source: &vl::ModelArgs,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        vl::ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target = source.clone();
    target.text.quantization = Some(quantization);
    target.text.quantization_config = None;
    target.text.quantized_weights = None;
    target.text.quantized_weight_configs = None;
    target.vision.apply_load_time_quantization(quantization);
    let source_architecture = Architecture::new(source.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_architecture = Architecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let source_vision = source.vision.layer_count();
    let total = source_vision
        .checked_add(usize::try_from(source.text.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture("invalid Qwen3-VL text layer count".into())
        })?)
        .ok_or_else(|| Error::UnsupportedArchitecture("Qwen3-VL unit count overflowed".into()))?;
    let source_args = source.clone();
    let target_args = target.clone();
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
                args: source_args.clone(),
                vision_layers: source_vision,
                external_experts: false,
            };
            factory.build(flat, stream)
        },
        move |flat, stream| {
            let mut factory = UnitFactory {
                args: target_args.clone(),
                vision_layers: source_vision,
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

pub(crate) fn prepare_gguf_pipeline(
    model_path: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
) -> Result<(vl::ModelArgs, Arc<dyn CheckpointSource>), Error> {
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) => value.as_str(),
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3-VL GGUF general.architecture must be a string".into(),
            ))
        }
    };
    let context = match architecture {
        "qwen3vl" => qwen::TextConfigContext::Qwen3Vl,
        "qwen3vlmoe" => qwen::TextConfigContext::Qwen3VlMoe,
        other => {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3-VL GGUF architecture {other:?}"
            )))
        }
    };
    let mut text = qwen::model_args_from_gguf_catalog_with_context(
        &TextGgufCatalog(checkpoint),
        metadata,
        context,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let is_moe = text.is_moe();
    let translate_text = |name: &str| {
        let name = qwen::translate_gguf_weight_name(name, is_moe);
        name.strip_prefix("model.")
            .map(|name| format!("model.language_model.{name}"))
            .unwrap_or(name)
    };
    checkpoint
        .catalog()
        .translated_outputs(translate_text)
        .map_err(safemlx::error::IoError::from)?;
    let mut text_formats = gguf_quantization_configs(checkpoint, translate_text)?;
    if is_moe {
        for layer in 0..text.num_hidden_layers {
            let root = format!("model.language_model.layers.{layer}.mlp.experts");
            if let Some(format) = text_formats.remove(&format!("{root}.gate_proj.weight")) {
                text_formats.remove(&format!("{root}.up_proj.weight"));
                text_formats.insert(format!("{root}.gate_up_proj"), format);
            }
        }
    }
    text.quantized_weights = Some(text_formats.keys().cloned().collect());
    text.quantized_weight_configs = Some(text_formats);
    text.quantization = None;
    let projector_path = crate::composition::mlx::artifact::find_sibling_mmproj(
        model_path, "qwen3vl",
    )?
    .ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Qwen3-VL GGUF requires a sibling mmproj relative to {}",
            model_path.display()
        ))
    })?;
    let projector = GgufCheckpoint::open(projector_path)?;
    let projector_metadata =
        crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&projector);
    let mut vision =
        vision::config_from_gguf_catalog(&VisionGgufCatalog(&projector), &projector_metadata)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let deepstack = vision.deepstack_layers();
    let translate_vision = |name: &str| translate_vision_gguf(name, &deepstack);
    projector
        .catalog()
        .translated_outputs(translate_vision)
        .map_err(safemlx::error::IoError::from)?;
    vision.linear_formats = gguf_quantization_configs(&projector, translate_vision)?
        .into_iter()
        .map(|(name, format)| (name, format.into()))
        .collect();
    let args = vl::model_args_from_gguf_parts(text, metadata, vision)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let text_plan = qwen::gguf_plan(&args.text).map_err(Error::UnsupportedArchitecture)?;
    let vision_plan = vision::gguf_plan(&args.vision, args.text.hidden_size)
        .map_err(Error::UnsupportedArchitecture)?;
    let text_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        &text_plan,
        translate_text,
        max_mapped_shards,
    )?);
    let vision_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        projector,
        &vision_plan,
        translate_vision,
        max_mapped_shards,
    )?);
    Ok((
        args,
        Arc::new(CompositeCheckpointSource::new([
            text_source,
            vision_source,
        ])?),
    ))
}

/// Loads split Qwen3-VL text plus projector GGUF artifacts through one
/// composite neutral checkpoint source.
pub fn load_gguf(
    model_path: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(QwenVlModel, Vec<u32>), Error> {
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) => value.as_str(),
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3-VL GGUF general.architecture must be a string".into(),
            ))
        }
    };
    let context = match architecture {
        "qwen3vl" => qwen::TextConfigContext::Qwen3Vl,
        "qwen3vlmoe" => qwen::TextConfigContext::Qwen3VlMoe,
        other => {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3-VL GGUF architecture {other:?}"
            )))
        }
    };
    let mut text = qwen::model_args_from_gguf_catalog_with_context(
        &TextGgufCatalog(checkpoint),
        metadata,
        context,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let is_moe = text.is_moe();
    let translate_text = |name: &str| {
        let name = qwen::translate_gguf_weight_name(name, is_moe);
        name.strip_prefix("model.")
            .map(|name| format!("model.language_model.{name}"))
            .unwrap_or(name)
    };
    checkpoint
        .catalog()
        .translated_outputs(translate_text)
        .map_err(safemlx::error::IoError::from)?;
    let mut text_formats = gguf_quantization_configs(checkpoint, translate_text)?;
    if is_moe {
        for layer in 0..text.num_hidden_layers {
            let root = format!("model.language_model.layers.{layer}.mlp.experts");
            if let Some(format) = text_formats.remove(&format!("{root}.gate_proj.weight")) {
                text_formats.remove(&format!("{root}.up_proj.weight"));
                text_formats.insert(format!("{root}.gate_up_proj"), format);
            }
        }
    }
    text.quantized_weights = Some(text_formats.keys().cloned().collect());
    text.quantized_weight_configs = Some(text_formats);
    text.quantization = None;

    let projector_path = crate::composition::mlx::artifact::find_sibling_mmproj(
        model_path, "qwen3vl",
    )?
    .ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Qwen3-VL GGUF requires a sibling mmproj relative to {}",
            model_path.display()
        ))
    })?;
    let projector = GgufCheckpoint::open(projector_path)?;
    let projector_metadata =
        crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&projector);
    let mut vision =
        vision::config_from_gguf_catalog(&VisionGgufCatalog(&projector), &projector_metadata)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let deepstack = vision.deepstack_layers();
    let translate_vision = |name: &str| translate_vision_gguf(name, &deepstack);
    projector
        .catalog()
        .translated_outputs(translate_vision)
        .map_err(safemlx::error::IoError::from)?;
    vision.linear_formats = gguf_quantization_configs(&projector, translate_vision)?
        .into_iter()
        .map(|(name, format)| (name, format.into()))
        .collect();
    let mut args = vl::model_args_from_gguf_parts(text, metadata, vision)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    let text_plan = qwen::gguf_plan(&args.text).map_err(Error::UnsupportedArchitecture)?;
    let vision_plan = vision::gguf_plan(&args.vision, args.text.hidden_size)
        .map_err(Error::UnsupportedArchitecture)?;
    let text_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        &text_plan,
        translate_text,
        options.max_mapped_shards(),
    )?);
    let vision_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        projector,
        &vision_plan,
        translate_vision,
        options.max_mapped_shards(),
    )?);
    let store: Arc<dyn CheckpointSource> = Arc::new(CompositeCheckpointSource::new([
        text_source,
        vision_source,
    ])?);
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen3-VL GGUF", None, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let (store, materialization) = if let Some(quantization) = quantize_on_load {
        let (store, target, report) = quantize_store(store, &args, quantization, stream)?;
        args = target;
        (store, Some(report))
    } else {
        (store, None)
    };
    let eos = crate::backend::mlx::gguf_eos_token_ids(metadata)?;
    let mut model = load_store(
        store,
        args,
        options,
        materialization,
        expert_options.is_some(),
        stream,
        weights_stream,
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok((model, eos))
}

/// Loads a Qwen3-VL SafeTensors artifact through the generic component engine.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenVlModel, Error> {
    load_safetensors_with_residency(
        model_dir,
        WeightResidency::with_layers(options.into()),
        quantization,
        stream,
        weights_stream,
    )
}

pub fn load_safetensors_with_residency(
    model_dir: impl AsRef<Path>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenVlModel, Error> {
    let model_dir = model_dir.as_ref();
    let value: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let mut args = vl::model_args_from_config_value(&value)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen3-VL", args.text.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    let store = resolve_store(store, &args)?;
    let (store, materialization) = if let Some(quantization) = quantize_on_load {
        let (store, target, report) = quantize_store(store, &args, quantization, stream)?;
        args = target;
        (store, Some(report))
    } else {
        (store, None)
    };
    let mut model = load_store(
        store,
        args,
        options,
        materialization,
        expert_options.is_some(),
        stream,
        weights_stream,
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_expert_cache(
    model: &mut QwenVlModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = match &model.execution {
        Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
        Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
    };
    let entries =
        crate::composition::qwen::expert::expert_catalog(&model.args.text, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

fn load_store(
    store: Arc<dyn CheckpointSource>,
    args: vl::ModelArgs,
    options: LayerWeightResidency,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenVlModel, Error> {
    let mut architecture = Architecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let layout = unit_layout(&architecture)?;
    let factory = UnitFactory {
        vision_layers: args.vision.layer_count(),
        args: args.clone(),
        external_experts,
    };
    let binding_args = args.clone();
    let static_modules =
        <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules_mut(
            &mut architecture,
        );
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        static_modules,
        factory,
        layout,
        options,
        stream,
        weights_stream,
        move |key| {
            key.ends_with("rotary_emb.inv_freq") || (external_experts && key.contains(".experts."))
        },
        |modules, store| {
            build_module_bindings_with_recipes(
                &MlxModule::new(modules.clone()),
                "",
                store,
                static_recipes(store),
            )
            .map_err(Into::into)
        },
        move |flat, unit, store, _| {
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                unit_recipes(store, &binding_args, flat)?,
                |name| external_experts && name.contains(".experts."),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.text.weight_quantization());
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        Execution::Resident(Box::new(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        )))
    } else {
        Execution::Bounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    let state_layout = vl::state_layout(&args)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(QwenVlModel {
        args,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
    })
}
