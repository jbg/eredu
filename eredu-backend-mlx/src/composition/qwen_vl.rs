// MLX artifact and residency binding for the neutral Qwen3-VL graph.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_architectures::qwen::{self, vision, vl};
use eredu_checkpoint::{
    store::{CheckpointSource, CompositeCheckpointSource, TensorSelection},
    WeightQuantization,
};
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, ExecutionResidency, ExecutionUnitLayout,
    LayerWeightResidency, LayeredArchitecture, LayerwiseModelMetadata, LayerwiseRuntime,
    PagedCacheOptions, ParameterRole, ResidencyReport, StaticUnitBindings, WeightBinding,
    WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

fn neutral_input_parts<'a>(
    parts: &'a [vl::InputPart<'a, Array>],
) -> Vec<vl::InputPart<'a, crate::MlxTensor>> {
    parts
        .iter()
        .map(|part| match part {
            vl::InputPart::Text(tokens) => {
                vl::InputPart::Text(crate::composition::tensor_ref(tokens))
            }
            vl::InputPart::Image { tokens, grid } => vl::InputPart::Image {
                tokens: crate::composition::tensor_ref(tokens),
                grid,
            },
            vl::InputPart::Video { tokens, grid } => vl::InputPart::Video {
                tokens: crate::composition::tensor_ref(tokens),
                grid,
            },
            vl::InputPart::Projected { tokens, embeddings } => vl::InputPart::Projected {
                tokens: crate::composition::tensor_ref(tokens),
                embeddings: crate::composition::tensor_ref(embeddings),
            },
        })
        .collect()
}

use crate::backend::mlx::{
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
            parameter_name_in_targets, parameter_role_targets,
            populate_module_from_lease_excluding,
        },
        checkpoint::{
            load::{gguf_quantization_configs, GgufTensorNames},
            quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::{
            generic::{
                construct_architecture_unit, prepare_layerwise_policy_with_bindings,
                MlxLayerwisePolicy, MlxResidentPolicy, MlxUnitPopulator,
            },
            layerwise::{open_safetensors_weight_store, quantize_parameterized_store},
        },
        media::input,
        residency::expert_cache::ExpertCache,
    },
};

type Architecture = vl::LayeredModel<MlxNeuralBackend>;
type Unit = vl::Unit<MlxNeuralBackend>;

fn group_kind(architecture: &Architecture, group: usize) -> eredu_runtime::ArchitectureGroupKind {
    <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_transport(
        architecture,
        group,
    )
    .kind
}

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub struct QwenVlCheckpointTemplate {
    pub static_modules: vl::StaticModules<MlxNeuralBackend>,
    pub units: Vec<Unit>,
}

#[cfg(any(test, feature = "test-support"))]
impl QwenVlCheckpointTemplate {
    pub fn new(args: vl::ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = Architecture::new(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let static_modules =
            <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
                &architecture,
            )
            .clone();
        let layout = unit_layout(&architecture)?;
        let units = (0..layout.len())
            .map(|index| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    index,
                    stream,
                    std::marker::PhantomData::<MlxHybridState>,
                )
            })
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

#[derive(Clone)]
struct UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

fn unit_expert_targets(
    architecture: &Architecture,
    index: usize,
    unit: &MlxModule<Unit>,
) -> Result<BTreeSet<String>, Error> {
    let Unit::Text(block) = &unit.inner else {
        return Ok(BTreeSet::new());
    };
    Ok(parameter_role_targets(
        &qwen::layer_parallel_parameter_groups(block, &architecture.args().text, index)?,
        ParameterRole::ExpertIntermediate,
    ))
}

/// Pipeline and Cartesian-parallel binder for the same neutral Qwen3-VL
/// architecture used by resident and bounded execution.
#[derive(Default)]
pub struct QwenVlPipelineBindings {
    external_experts: bool,
}

impl QwenVlPipelineBindings {
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
        &architecture.args().model_type
    }

    pub fn begin_pipeline_ingress(
        &self,
        architecture: &mut Architecture,
        typed: input::ModelInput<'_>,
        offset: i32,
        delta: Option<&Array>,
        parallel: Option<&safemlx::distributed::Group>,
        stream: &Stream,
    ) -> Result<vl::PipelineVisionState<crate::MlxTensor>, Error> {
        let args = architecture.args().clone();
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
                    let merge = args.vision.spatial_merge_size;
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
                        args.image_token_id
                    } else {
                        args.video_token_id
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
        let neutral_parts = neutral_input_parts(&parts);
        let input = vl::ModelInput {
            parts: &neutral_parts,
            pixels: crate::composition::tensor_opt(pixels.as_ref()),
            mask: None,
        };
        let delta = crate::composition::tensor_opt(delta);
        match parallel {
            Some(parallel) => {
                architecture.begin_pipeline_parallel(input, offset, delta, parallel, stream)
            }
            None => architecture.begin_pipeline(input, offset, delta, stream),
        }
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn static_units(
        &self,
        architecture: &Architecture,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub fn layer_bindings(
        &self,
        architecture: &Architecture,
        group: usize,
        index: usize,
        layer: &MlxModule<Unit>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let expert_targets = unit_expert_targets(architecture, index, layer)?;
        let recipes =
            if group_kind(architecture, group) == eredu_runtime::ArchitectureGroupKind::Decoder {
                let args = architecture.args();
                vl::unit_recipes(store, args, args.vision.layer_count() + index)
                    .map_err(Error::UnsupportedArchitecture)?
            } else {
                BTreeMap::new()
            };
        Ok(build_module_bindings_with_recipes_excluding(
            layer,
            "",
            store,
            recipes,
            |name| self.external_experts && parameter_name_in_targets(name, &expert_targets),
        )?)
    }

    pub fn expert_parallel_assignment(
        &self,
        architecture: &Architecture,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        let args = architecture.args();
        if !args.text.is_moe() {
            return Err(Error::Parallel(
                "Qwen3-VL PP+EP requires a routed text checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                args.text.num_experts as usize,
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
        global_layer: &MlxModule<Unit>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
    ) -> Result<Vec<WeightBinding>, Error> {
        match (&global_layer.inner, group_kind(architecture, group)) {
            (Unit::Vision(_), eredu_runtime::ArchitectureGroupKind::VisionEncoder) => {
                let bindings =
                    build_module_bindings_with_recipes(global_layer, "", store, BTreeMap::new())?;
                if let Some(layout) = layout {
                    crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
                        bindings, "", store, layout,
                    )
                } else {
                    Ok(bindings)
                }
            }
            (Unit::Text(_), eredu_runtime::ArchitectureGroupKind::Decoder) => {
                let args = architecture.args();
                let expert_targets = unit_expert_targets(architecture, index, global_layer)?;
                let recipes = if self.external_experts {
                    BTreeMap::new()
                } else {
                    vl::unit_recipes(store, args, args.vision.layer_count() + index)
                        .map_err(Error::UnsupportedArchitecture)?
                };
                let mut bindings = build_module_bindings_with_recipes_excluding(
                    global_layer,
                    "",
                    store,
                    recipes,
                    |name| {
                        self.external_experts && parameter_name_in_targets(name, &expert_targets)
                    },
                )?;
                if let Some(assignment) = assignment {
                    let indices = assignment.local_global_expert_ids().to_vec();
                    bindings = bindings
                        .into_iter()
                        .map(|binding| {
                            let target = binding.logical_target().unwrap_or_else(|| binding.name());
                            if parameter_name_in_targets(target, &expert_targets) {
                                binding
                                    .select_bounded_output(
                                        store,
                                        TensorSelection::Indices {
                                            axis: 0,
                                            indices: indices.clone(),
                                        },
                                    )
                                    .map_err(Error::from)
                            } else {
                                Ok(binding)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                match layout {
                    Some(layout) => {
                        crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
                            bindings,
                            &format!("model.layers.{index}"),
                            store,
                            layout,
                        )
                    }
                    None => Ok(bindings),
                }
            }
            _ => Err(Error::Parallel("Qwen3-VL unit/group mismatch".into())),
        }
    }
}

impl MlxUnitPopulator<Unit> for UnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<Unit>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

type Resident = LayerwiseRuntime<Architecture, MlxNeuralBackend, MlxHybridState, MlxResidentPolicy<Unit>>;
type Bounded = LayerwiseRuntime<
    Architecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<Unit, UnitPopulator>,
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
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
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
        let parts = neutral_input_parts(input.parts);
        let input = vl::ModelInput {
            parts: &parts,
            pixels: crate::composition::tensor_opt(input.pixels),
            mask: crate::composition::tensor_opt(input.mask),
        };
        match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward(input, cache, stream),
            Execution::Bounded(runtime) => runtime.forward(input, cache, stream),
        }
        .map(crate::MlxTensor::into_array)
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
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let parts = neutral_input_parts(input.parts);
        let input = vl::ModelInput {
            parts: &parts,
            pixels: crate::composition::tensor_opt(input.pixels),
            mask: crate::composition::tensor_opt(input.mask),
        };
        let hook = |architecture: &mut Architecture,
                    group: usize,
                    index: usize,
                    unit: &mut Unit,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxHybridState,
                    forward: &mut vl::ForwardContext<crate::MlxTensor>,
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
        .map(crate::MlxTensor::into_array)
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

    pub fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prepared_forward(input, cache, stream)
            .map_err(Error::Exception)
    }

    pub fn decode_tensor_parallel(
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
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.prepared_forward(input, cache, stream)?
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let parts = [vl::InputPart::Text(input_tokens.as_array())];
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
        .map(crate::MlxTensor::from_array)
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

pub fn prepare_gguf_pipeline(
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
    let translate_text = |name: &str| vl::translate_text_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate_text)
        .map_err(safemlx::error::IoError::from)?;
    let mut text_formats = gguf_quantization_configs(checkpoint, translate_text)?;
    vl::normalize_text_weight_formats(&text, &mut text_formats);
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
    let translate_vision = |name: &str| vl::translate_vision_gguf_weight_name(name, &deepstack);
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
    let translate_text = |name: &str| vl::translate_text_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate_text)
        .map_err(safemlx::error::IoError::from)?;
    let mut text_formats = gguf_quantization_configs(checkpoint, translate_text)?;
    vl::normalize_text_weight_formats(&text, &mut text_formats);
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
    let translate_vision = |name: &str| vl::translate_vision_gguf_weight_name(name, &deepstack);
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
    let eos = crate::composition::mlx::gguf_eos_token_ids(metadata)?;
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
    let entries = crate::composition::qwen::expert::expert_catalog(
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
    let binding_args = args.clone();
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
                vl::static_recipes(store),
            )
            .map_err(Into::into)
        },
        move |address, _path, unit, store, _| {
            let flat = if address.group() == 0 {
                address.index()
            } else {
                binding_args.vision.layer_count() + address.index()
            };
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                vl::unit_recipes(store, &binding_args, flat)
                    .map_err(Error::UnsupportedArchitecture)?,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.text.weight_quantization());
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
