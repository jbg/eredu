//! Shared bounded layer execution for dense and MoE Qwen3-VL models.

use eredu_runtime::{
    ExecutionGraph, ExecutionUnitLayout, ExpertCacheLoadOptions, ExpertPass, LayerWeightResidency,
    LayeredArchitecture, LayeredForwardState, LayerwiseRuntime, NonExpertWeightResidency,
    ParallelLayeredArchitecture, StaticUnitBindings, WeightResidency,
};

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;
use eredu_runtime::WeightBinding;

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters},
    nn,
    ops::{
        concatenate_axis,
        indexing::{masked_scatter, TryIndexOp},
        zeros_dtype, GgufCheckpoint, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Stream,
};

use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::{
        self as common,
        parallel::{
            planned_kv_head_layout, vocab_embedding_parameter_group, vocab_lm_head_parameter_group,
            VocabParallelEmbedding, VocabParallelLmHead,
        },
        shared::{MlxBackend, MlxEmbedding, MlxLinear, MlxModule, MlxParameterTree, MlxRmsNorm},
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::KeyValueCache,
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_binding_plan_with_recipes,
        build_module_binding_plan_with_recipes_excluding,
    },
    backend::mlx::runtime::checkpoint::store::{TensorSelection, WeightStoreBackend},
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, recipe::DerivedWeightRecipe,
    },
    backend::mlx::runtime::execution::generic::{
        prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        MlxUnitFactory,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_module_store_with_bindings, shard_layer_bindings,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{ExpertCache, ExpertCacheReport},
    composition::mlx_architectures::qwen::{
        vl::model::{self as resident, Cache, ModelArgs},
        vl::vision::{
            configure_vision_parallel_static, grid_thw_from_array, new_parallel_vision_block,
            vision_parallel_parameter_groups, QwenVisionBlock, QwenVisionLayerwiseState,
            QwenVisionLayerwiseStatic, QwenVisionTransformer,
        },
    },
};
use eredu_architectures::qwen::{self as neutral_qwen, FeedForward as NeutralFeedForward};
use eredu_nn::{
    EmbeddingOperator, LinearOperator, ParameterSpec, RoutedNeuralBackend, SwiGluExpertBankSpec,
    SwiGluExpertLayout, SwiGluExpertProjection,
};
use eredu_runtime::PagedCacheOptions;

use eredu_runtime::ResidencyReport;

const VISION_STATIC_UNIT: &str = "qwen3_vl.static.vision";
const EMBEDDING_UNIT: &str = "qwen3_vl.static.embedding";
const NORM_UNIT: &str = "qwen3_vl.static.norm";
const HEAD_UNIT: &str = "qwen3_vl.static.output";

type NeutralTextBlock = MlxModule<neutral_qwen::TransformerBlock<MlxBackend>>;

fn qwen_vl_text_layer_bindings(
    layer: &NeutralTextBlock,
    args: &eredu_architectures::qwen::ModelArgs,
    index: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
) -> Result<Vec<WeightBinding>, Error> {
    let mut recipes = BTreeMap::new();
    if args.is_moe() && !external_experts {
        let resolved =
            eredu_architectures::qwen::expert_recipes(store, args, &args.parameter_root, index)
                .map_err(Error::UnsupportedArchitecture)?;
        recipes.insert(resolved.target_gate_up, resolved.gate_up);
        recipes.insert(resolved.target_down, resolved.down);
    }
    Ok(
        build_module_binding_plan_with_recipes_excluding(layer, "", store, recipes, |name| {
            external_experts && name.contains(".mlp.experts.")
        })?
        .build_bindings(store)?,
    )
}

fn unloaded_expert_bank(
    args: &eredu_architectures::qwen::ModelArgs,
    layer: usize,
    expert_count: i32,
    intermediate: i32,
    stream: &Stream,
) -> Result<<MlxBackend as RoutedNeuralBackend>::SwiGluExpertBank, Error> {
    let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    let gate_up = format!("{prefix}.gate_up_proj");
    let down = format!("{prefix}.down_proj");
    <MlxBackend as RoutedNeuralBackend>::swiglu_expert_bank(
        SwiGluExpertBankSpec {
            expert_count,
            input_dimensions: args.hidden_size,
            intermediate_dimensions: intermediate,
            output_dimensions: args.hidden_size,
            activation: eredu_nn::GatedExpertActivation::Silu,
            limit: None,
            layout: SwiGluExpertLayout::Packed {
                gate_up: SwiGluExpertProjection {
                    weight: ParameterSpec::trainable(&gate_up)
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                    format: args.weight_quantization_for(&gate_up).into(),
                },
                down: SwiGluExpertProjection {
                    weight: ParameterSpec::trainable(&down)
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                    format: args.weight_quantization_for(&down).into(),
                },
            },
        },
        stream,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

type Qwen3VlUnit = MlxParameterTree<Qwen3VlLayer>;
type Qwen3VlStatic = MlxParameterTree<Qwen3VlStaticModules>;
type Qwen3VlResidentRuntime =
    LayerwiseRuntime<Qwen3VlArchitecture, MlxBackend, Cache, MlxResidentPolicy<Qwen3VlUnit>>;
type Qwen3VlBoundedRuntime = LayerwiseRuntime<
    Qwen3VlArchitecture,
    MlxBackend,
    Cache,
    MlxLayerwisePolicy<Qwen3VlUnit, Qwen3VlUnitFactory>,
>;

enum Qwen3VlRuntime {
    Resident(Box<Qwen3VlResidentRuntime>),
    Layerwise(Box<Qwen3VlBoundedRuntime>),
}

struct Qwen3VlExecution {
    runtime: Qwen3VlRuntime,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    parallel_info:
        Option<eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl Qwen3VlExecution {
    fn architecture(&self) -> &Qwen3VlArchitecture {
        match &self.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime.architecture(),
            Qwen3VlRuntime::Layerwise(runtime) => runtime.architecture(),
        }
    }

    fn architecture_mut(&mut self) -> &mut Qwen3VlArchitecture {
        match &mut self.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime.architecture_mut(),
            Qwen3VlRuntime::Layerwise(runtime) => runtime.architecture_mut(),
        }
    }

    fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime.policy().checkpoint_store(),
            Qwen3VlRuntime::Layerwise(runtime) => runtime.policy().checkpoint_store(),
        }
    }

    fn checkpoint_store_arc(&self) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            Qwen3VlRuntime::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime.policy().residency_report(),
            Qwen3VlRuntime::Layerwise(runtime) => runtime.policy().residency_report(),
        }
    }

    fn dense_stream_report(&self) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.runtime {
            Qwen3VlRuntime::Resident(_) => Ok(None),
            Qwen3VlRuntime::Layerwise(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    fn forward(
        &mut self,
        input: Qwen3VlInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime
                .forward(input, cache, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Qwen3VlRuntime::Layerwise(runtime) => runtime
                .forward(input, cache, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    fn forward_parallel(
        &mut self,
        input: Qwen3VlInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Qwen3VlRuntime::Layerwise(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3VlReplicatedStatic {
    #[param]
    vision: QwenVisionLayerwiseStatic,
    #[param]
    embedding: MlxEmbedding,
    #[param]
    norm: MlxRmsNorm,
    #[param]
    lm_head: Option<MlxLinear>,
}

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3VlParallelStatic {
    #[param]
    vision: QwenVisionLayerwiseStatic,
    #[param]
    embedding: VocabParallelEmbedding,
    #[param]
    norm: MlxRmsNorm,
    #[param]
    lm_head: Option<VocabParallelLmHead>,
}

#[derive(Debug, Clone)]
enum Qwen3VlStaticModules {
    Replicated(Qwen3VlReplicatedStatic),
    Parallel(Qwen3VlParallelStatic),
}

macro_rules! qwen3_vl_static_parameters {
    ($self:ident, $method:ident $(, $arg:expr)?) => {
        match $self {
            Qwen3VlStaticModules::Replicated(module) => module.$method($($arg)?),
            Qwen3VlStaticModules::Parallel(module) => module.$method($($arg)?),
        }
    };
}

impl ModuleParameters for Qwen3VlStaticModules {
    fn num_parameters(&self) -> usize {
        qwen3_vl_static_parameters!(self, num_parameters)
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        qwen3_vl_static_parameters!(self, parameters)
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        qwen3_vl_static_parameters!(self, parameters_mut)
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        qwen3_vl_static_parameters!(self, trainable_parameters)
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        qwen3_vl_static_parameters!(self, freeze_parameters, recursive)
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        qwen3_vl_static_parameters!(self, unfreeze_parameters, recursive)
    }
    fn all_frozen(&self) -> Option<bool> {
        qwen3_vl_static_parameters!(self, all_frozen)
    }
    fn any_frozen(&self) -> Option<bool> {
        qwen3_vl_static_parameters!(self, any_frozen)
    }
}

impl Qwen3VlStaticModules {
    fn replicated(args: &ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let visual = QwenVisionTransformer::new_deepstack(args.vision_config.clone(), stream)?;
        let text = neutral_qwen::StaticModules::<MlxBackend>::new(&args.text_config, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(Self::Replicated(Qwen3VlReplicatedStatic {
            vision: QwenVisionLayerwiseStatic::from_transformer(visual),
            embedding: text.embeddings.clone(),
            norm: text.norm.clone(),
            lm_head: text.lm_head,
        }))
    }
}

#[derive(Clone)]
struct Qwen3VlUnitFactory {
    args: ModelArgs,
    vision_units: usize,
    parallel_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    sparse_experts: bool,
}

impl MlxUnitFactory<Qwen3VlUnit> for Qwen3VlUnitFactory {
    fn build(&mut self, ordinal: usize, stream: &Stream) -> Result<Qwen3VlUnit, Error> {
        let (group, index) = if ordinal < self.vision_units {
            (0, ordinal)
        } else {
            (1, ordinal - self.vision_units)
        };
        build_qwen3_vl_unit(
            &self.args,
            group,
            index,
            self.parallel_layout.as_deref(),
            self.sparse_experts,
            stream,
        )
    }
}

struct Qwen3VlArchitecture {
    args: ModelArgs,
    static_modules: Qwen3VlStatic,
    sparse_experts: bool,
    expert_cache: Option<ExpertCache>,
    parallel_kv_heads: Option<Vec<i32>>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl Qwen3VlArchitecture {
    fn new(args: ModelArgs, sparse_experts: bool, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            static_modules: MlxParameterTree::new(
                Qwen3VlStaticModules::replicated(&args, stream)?,
                "",
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            args,
            sparse_experts,
            expert_cache: None,
            parallel_kv_heads: None,
            parallel_topology: None,
        })
    }

    fn replicated_static_mut(&mut self) -> Result<&mut Qwen3VlReplicatedStatic, Error> {
        match &mut *self.static_modules {
            Qwen3VlStaticModules::Replicated(modules) => Ok(modules),
            Qwen3VlStaticModules::Parallel(_) => Err(Error::Parallel(
                "Qwen3-VL replicated execution received parallel static modules".into(),
            )),
        }
    }

    fn parallel_static_mut(&mut self) -> Result<&mut Qwen3VlParallelStatic, Error> {
        match &mut *self.static_modules {
            Qwen3VlStaticModules::Parallel(modules) => Ok(modules),
            Qwen3VlStaticModules::Replicated(_) => Err(Error::Parallel(
                "Qwen3-VL parallel execution received replicated static modules".into(),
            )),
        }
    }

    fn validate_cache(&self, cache: &Cache) -> Result<(), Error> {
        cache.validate(&self.args)
    }
}

fn build_qwen3_vl_unit(
    args: &ModelArgs,
    group: usize,
    index: usize,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    sparse_experts: bool,
    stream: &Stream,
) -> Result<Qwen3VlUnit, Error> {
    let layer = match (group, layout) {
        (0, Some(layout)) => Qwen3VlLayer::Vision(Box::new(new_parallel_vision_block(
            &args.vision_config,
            "model.visual",
            index,
            layout,
            stream,
        )?)),
        (0, None) => Qwen3VlLayer::Vision(Box::new(QwenVisionBlock::new(
            &args.vision_config,
            index,
            stream,
        )?)),
        (1, layout) => {
            let mut config = args.text_config.clone();
            if let Some(layout) = layout {
                let prefix = format!("model.language_model.layers.{index}");
                let planned = |name: &str| {
                    layout
                        .tensor(&format!("{prefix}.{name}.weight"))
                        .or_else(|| layout.tensor(&format!("{prefix}.{name}.inner.weight")))
                };
                let query = planned("self_attn.q_proj")
                    .ok_or_else(|| Error::Parallel(format!("missing TP query for {prefix}")))?;
                let key = planned("self_attn.k_proj")
                    .ok_or_else(|| Error::Parallel(format!("missing TP key for {prefix}")))?;
                config.num_attention_heads = i32::try_from(query.local_shape()[0])
                    .map_err(|_| Error::Parallel("Qwen local query width exceeds i32".into()))?
                    / config.head_dim;
                config.num_key_value_heads = i32::try_from(key.local_shape()[0])
                    .map_err(|_| Error::Parallel("Qwen local key width exceeds i32".into()))?
                    / config.head_dim;
                if config.is_moe() {
                    let experts = layout
                        .tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))
                        .ok_or_else(|| {
                            Error::Parallel(format!("missing TP experts for {prefix}"))
                        })?;
                    config.moe_intermediate_size = i32::try_from(experts.local_shape()[1] / 2)
                        .map_err(|_| {
                            Error::Parallel("Qwen local expert width exceeds i32".into())
                        })?;
                } else {
                    let gate = planned("mlp.gate_proj")
                        .ok_or_else(|| Error::Parallel(format!("missing TP MLP for {prefix}")))?;
                    config.intermediate_size = i32::try_from(gate.local_shape()[0])
                        .map_err(|_| Error::Parallel("Qwen local MLP width exceeds i32".into()))?;
                }
            }
            Qwen3VlLayer::Text(Box::new(MlxModule::new(
                neutral_qwen::new_block::<MlxBackend>(&config, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            )))
        }
        _ => {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            )))
        }
    };
    MlxParameterTree::new_filtered(layer, "", |name| {
        !sparse_experts || !name.starts_with("mlp.experts.")
    })
    .map_err(|error| Error::Parallel(error.to_string()))
}

fn prepare_qwen3_vl_prefill<F>(
    args: &ModelArgs,
    vision: &mut QwenVisionLayerwiseStatic,
    typed: input::ModelInput<'_>,
    cache: &mut Cache,
    stream: &Stream,
    mut embed: F,
) -> Result<LayeredForwardState<Array, Qwen3VlForwardContext>, Error>
where
    F: FnMut(&Array) -> Result<Array, Error>,
{
    input::validate(typed)?;
    let mut token_parts = Vec::new();
    let mut prepared_parts = Vec::new();
    let mut pixels = Vec::new();
    let mut grids = Vec::new();
    let merge = args.vision_config.spatial_merge_size;
    for part in typed.parts {
        match (part.modality, part.payload) {
            (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                token_parts.push(tokens.clone());
                prepared_parts.push(PreparedPart::Text(embed(tokens)?));
            }
            (
                input::Modality::Image | input::Modality::Video,
                input::InputPayload::Tensor(tensor),
            ) => {
                let grid = part.metadata.patch_grid.ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Qwen3-VL {} input requires patch_grid metadata",
                        part.modality.as_str()
                    ))
                })?;
                let merged = grid_thw_from_array(grid, stream)?
                    .into_iter()
                    .map(|(t, h, w)| t * (h / merge) * (w / merge))
                    .sum::<i32>();
                let token_id = if part.modality == input::Modality::Image {
                    args.image_token_id
                } else {
                    args.video_token_id
                };
                token_parts.push(input::token_ids_array(
                    &vec![token_id; merged as usize],
                    stream,
                )?);
                prepared_parts.push(PreparedPart::Visual(merged));
                pixels.push(tensor.clone());
                grids.push(grid.clone());
            }
            (modality, _) => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3-VL input does not support {} payloads of this kind",
                    modality.as_str()
                )))
            }
        }
    }
    let tokens = concatenate_axis(&token_parts.iter().collect::<Vec<_>>(), 1, stream)?;
    let (position_ids, rope_delta) =
        resident::multimodal_position_ids(typed, merge, tokens.dim(1), stream)?;
    cache.rope_delta = rope_delta;
    let (cos, sin) = resident::mrope_embeddings(
        &position_ids,
        args.text_config.head_dim,
        args.text_config.rope_theta,
        &args.mrope_section,
    );
    let (hidden, vision_state) = if pixels.is_empty() {
        let hidden = prepared_parts
            .iter()
            .filter_map(|part| match part {
                PreparedPart::Text(value) => Some(value),
                PreparedPart::Visual(_) => None,
            })
            .collect::<Vec<_>>();
        (concatenate_axis(&hidden, 1, stream)?, None)
    } else {
        let pixels = concatenate_axis(&pixels.iter().collect::<Vec<_>>(), 0, stream)?;
        let grids = concatenate_axis(&grids.iter().collect::<Vec<_>>(), 0, stream)?;
        let (hidden, state) = vision.begin(&pixels, &grids, stream)?;
        (hidden, Some(state))
    };
    Ok(LayeredForwardState {
        hidden,
        context: Qwen3VlForwardContext {
            tokens,
            parts: prepared_parts,
            vision: vision_state,
            mask: None,
            cos,
            sin,
            visual_mask: None,
            deepstack_features: Vec::new(),
        },
    })
}

fn prepare_qwen3_vl_decode<F>(
    args: &ModelArgs,
    tokens: &Array,
    cache: &Cache,
    _stream: &Stream,
    mut embed: F,
) -> Result<LayeredForwardState<Array, Qwen3VlForwardContext>, Error>
where
    F: FnMut(&Array) -> Result<Array, Error>,
{
    let hidden = embed(tokens)?;
    let start = cache
        .kv
        .first()
        .and_then(Option::as_ref)
        .map(KeyValueCache::offset)
        .unwrap_or(0)
        + cache.rope_delta;
    let positions = [
        (start..start + tokens.dim(1)).collect(),
        (start..start + tokens.dim(1)).collect(),
        (start..start + tokens.dim(1)).collect(),
    ];
    let (cos, sin) = resident::mrope_embeddings(
        &positions,
        args.text_config.head_dim,
        args.text_config.rope_theta,
        &args.mrope_section,
    );
    Ok(LayeredForwardState {
        hidden,
        context: Qwen3VlForwardContext {
            tokens: tokens.clone(),
            parts: Vec::new(),
            vision: None,
            mask: None,
            cos,
            sin,
            visual_mask: None,
            deepstack_features: Vec::new(),
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn begin_qwen3_vl_text_group(
    args: &ModelArgs,
    vision: &mut QwenVisionLayerwiseStatic,
    initial: &Array,
    dependencies: &[&Array],
    cache: &Cache,
    forward: &mut Qwen3VlForwardContext,
    parallel: Option<&safemlx::distributed::Group>,
    stream: &Stream,
) -> Result<Array, Error> {
    let hidden = dependencies.first().copied().unwrap_or(initial);
    let hidden =
        if let Some(mut state) = forward.vision.take() {
            let output = match parallel {
                Some(group) => vision.finish_tensor_parallel(hidden, &mut state, group, stream)?,
                None => vision.finish(hidden, &mut state, stream)?,
            };
            forward.deepstack_features = output.deepstack_features;
            let mut visual_offset = 0;
            let mut assembled = Vec::with_capacity(forward.parts.len());
            for part in &forward.parts {
                match part {
                    PreparedPart::Text(embedding) => assembled.push(embedding.clone()),
                    PreparedPart::Visual(len) => {
                        assembled.push(output.embeddings.try_index_device(
                            (.., visual_offset..visual_offset + *len, ..),
                            stream,
                        )?);
                        visual_offset += *len;
                    }
                }
            }
            concatenate_axis(&assembled.iter().collect::<Vec<_>>(), 1, stream)?
        } else {
            hidden.clone()
        };
    forward.mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
        Some(AttentionMask::Array(mask)) => Some(mask),
        Some(AttentionMask::Causal) => {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3-VL execution requires an explicit causal mask".into(),
            ))
        }
        None => None,
    };
    forward.visual_mask = if forward.deepstack_features.is_empty() {
        None
    } else {
        Some(
            forward
                .tokens
                .eq(Array::from_int(args.image_token_id as i32), stream)?
                .logical_or(
                    &forward
                        .tokens
                        .eq(Array::from_int(args.video_token_id as i32), stream)?,
                    stream,
                )?,
        )
    };
    Ok(hidden)
}

fn add_qwen3_vl_deepstack(
    mut output: Array,
    index: usize,
    forward: &Qwen3VlForwardContext,
    stream: &Stream,
) -> Result<Array, Error> {
    if let Some(features) = forward.deepstack_features.get(index) {
        let base = zeros_dtype(output.shape(), output.dtype(), stream)?;
        let features = features.try_index_device((0, .., ..), stream)?;
        let aligned = masked_scatter(
            &base,
            forward.visual_mask.as_ref().expect("DeepStack visual mask"),
            features,
            stream,
        )?;
        output = output.add(aligned, stream)?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_qwen3_vl_text<C>(
    block: &mut NeutralTextBlock,
    hidden: &Array,
    mask: Option<&Array>,
    cache: Option<&mut C>,
    cosine: &Array,
    sine: &Array,
    parallel: Option<&safemlx::distributed::Group>,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
{
    let input = neutral_qwen::AttentionInput {
        hidden,
        mask,
        cache,
        allow_sliding_prefill: false,
        rotary_position: Some(eredu_nn::RotaryPosition::Embeddings { cosine, sine }),
    };
    match parallel {
        Some(group) => block.forward_tensor_parallel(input, group, stream),
        None => block.forward(input, stream),
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_qwen3_vl_text_external_experts<C, P>(
    block: &mut NeutralTextBlock,
    hidden: &Array,
    mask: Option<&Array>,
    cache: Option<&mut C>,
    cosine: &Array,
    sine: &Array,
    layer: usize,
    pass: ExpertPass,
    parallel: Option<&safemlx::distributed::Group>,
    stream: &Stream,
    provider: &mut P,
) -> Result<Array, Error>
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
    P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
    P::Error: std::fmt::Display,
{
    let input = neutral_qwen::AttentionInput {
        hidden,
        mask,
        cache,
        allow_sliding_prefill: false,
        rotary_position: Some(eredu_nn::RotaryPosition::Embeddings { cosine, sine }),
    };
    let feed_forward =
        |policy: &mut NeutralFeedForward<MlxBackend>, normalized: &Array, context: &Stream| {
            let shape = normalized.shape().to_vec();
            let flat = normalized
                .reshape(&[-1, normalized.dim(-1)], context)
                .map_err(eredu_nn::Error::backend)?;
            policy
                .forward_with_provider(layer, pass, &flat, context, provider)?
                .reshape(&shape, context)
                .map_err(eredu_nn::Error::backend)
        };
    match parallel {
        Some(group) => {
            block.forward_tensor_parallel_with_feed_forward(input, group, stream, feed_forward)
        }
        None => block.forward_with_feed_forward(input, stream, feed_forward),
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

impl LayeredArchitecture<MlxBackend, Cache> for Qwen3VlArchitecture {
    type Input<'a> = Qwen3VlInput<'a>;
    type StaticModules = Qwen3VlStatic;
    type Unit = Qwen3VlUnit;
    type ForwardContext = Qwen3VlForwardContext;
    type RetainedContextValues<'a> = std::vec::IntoIter<&'a Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        if self.args.text_config.is_moe() {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::chain(["vision_encoder", "text_decoder"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.args.vision_config.layer_count()),
            1 => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL group {group} has no unit {index}"
            )));
        }
        Ok(if group == 0 {
            format!("model.visual.blocks.{index}")
        } else {
            format!("model.language_model.layers.{index}")
        })
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Unit, Error> {
        build_qwen3_vl_unit(&self.args, group, index, None, self.sparse_experts, stream)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.validate_cache(cache)?;
        let args = self.args.clone();
        let modules = self.replicated_static_mut()?;
        match input {
            Qwen3VlInput::Prefill(input) => prepare_qwen3_vl_prefill(
                &args,
                &mut modules.vision,
                input,
                cache,
                stream,
                |tokens| {
                    modules
                        .embedding
                        .forward(tokens, stream)
                        .map_err(Into::into)
                },
            ),
            Qwen3VlInput::Decode(tokens) => {
                prepare_qwen3_vl_decode(&args, tokens, cache, stream, |tokens| {
                    modules
                        .embedding
                        .forward(tokens, stream)
                        .map_err(Into::into)
                })
            }
        }
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        cache: &mut Cache,
        forward: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match group {
            0 => Ok(initial.clone()),
            1 => {
                let args = self.args.clone();
                let modules = self.replicated_static_mut()?;
                begin_qwen3_vl_text_group(
                    &args,
                    &mut modules.vision,
                    initial,
                    dependencies,
                    cache,
                    forward,
                    None,
                    stream,
                )
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group != 0 || forward.vision.is_some()
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut Cache,
        forward: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, &mut **layer) {
            (0, Qwen3VlLayer::Vision(block)) => {
                let Some(state) = forward.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let modules = self.replicated_static_mut()?;
                let output =
                    modules
                        .vision
                        .forward_block(block, index, hidden.clone(), state, stream)?;
                modules
                    .vision
                    .capture_deepstack(index, &output, state, stream)?;
                Ok(output)
            }
            (1, Qwen3VlLayer::Text(block)) => {
                let mut output = if self.sparse_experts {
                    let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Qwen3-VL sparse expert cache was not initialized".into(),
                        )
                    })?;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    let mut provider = crate::composition::qwen_expert::cached_provider(
                        expert_cache,
                        &self.args.text_config,
                    );
                    forward_qwen3_vl_text_external_experts(
                        block,
                        hidden,
                        forward.mask.as_ref(),
                        cache.kv[index].as_mut(),
                        &forward.cos,
                        &forward.sin,
                        index,
                        pass,
                        None,
                        stream,
                        &mut provider,
                    )?
                } else {
                    forward_qwen3_vl_text(
                        block,
                        hidden,
                        forward.mask.as_ref(),
                        cache.kv[index].as_mut(),
                        &forward.cos,
                        &forward.sin,
                        None,
                        stream,
                    )?
                };
                output = add_qwen3_vl_deepstack(output, index, forward, stream)?;
                Ok(output)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL execution unit does not match group {group}"
            ))),
        }
    }

    fn finish_forward(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _forward: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let modules = self.replicated_static_mut()?;
        let hidden = modules.norm.forward(hidden, stream)?;
        match &mut modules.lm_head {
            Some(head) => head
                .forward(&hidden, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            None => modules
                .embedding
                .as_linear(&hidden, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        }
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward
            .vision
            .as_ref()
            .map(QwenVisionLayerwiseState::retained_arrays)
            .unwrap_or_default()
            .into_iter()
            .chain(forward.deepstack_features.iter())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl ParallelLayeredArchitecture<MlxBackend, Cache> for Qwen3VlArchitecture {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.validate_cache(cache)?;
        let args = self.args.clone();
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("Qwen3-VL parallel topology was not configured".into())
        })?;
        let modules = self.parallel_static_mut()?;
        match input {
            Qwen3VlInput::Prefill(input) => prepare_qwen3_vl_prefill(
                &args,
                &mut modules.vision,
                input,
                cache,
                stream,
                |tokens| {
                    let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
                    modules.embedding.forward(tokens, &execution)
                },
            ),
            Qwen3VlInput::Decode(tokens) => {
                prepare_qwen3_vl_decode(&args, tokens, cache, stream, |tokens| {
                    let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
                    modules.embedding.forward(tokens, &execution)
                })
            }
        }
    }

    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut Cache,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (group_index, &mut **layer) {
            (0, Qwen3VlLayer::Vision(block)) => {
                let Some(state) = forward.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let modules = self.parallel_static_mut()?;
                let output = modules.vision.forward_block_tensor_parallel(
                    block,
                    index,
                    hidden.clone(),
                    state,
                    group,
                    stream,
                )?;
                modules
                    .vision
                    .capture_deepstack_tensor_parallel(index, &output, state, group, stream)?;
                Ok(output)
            }
            (1, Qwen3VlLayer::Text(block)) => {
                let output = forward_qwen3_vl_text(
                    block,
                    hidden,
                    forward.mask.as_ref(),
                    cache.kv[index].as_mut(),
                    &forward.cos,
                    &forward.sin,
                    Some(group),
                    stream,
                )?;
                add_qwen3_vl_deepstack(output, index, forward, stream)
            }
            _ => Err(Error::Parallel(format!(
                "Qwen3-VL parallel unit does not match group {group_index}"
            ))),
        }
    }

    fn begin_execution_group_parallel(
        &mut self,
        group_index: usize,
        initial: &Array,
        dependencies: &[&Array],
        cache: &mut Cache,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match group_index {
            0 => Ok(initial.clone()),
            1 => {
                let args = self.args.clone();
                let modules = self.parallel_static_mut()?;
                begin_qwen3_vl_text_group(
                    &args,
                    &mut modules.vision,
                    initial,
                    dependencies,
                    cache,
                    forward,
                    Some(group),
                    stream,
                )
            }
            _ => Err(Error::Parallel(format!(
                "Qwen3-VL has no execution group {group_index}"
            ))),
        }
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("Qwen3-VL parallel topology was not configured".into())
        })?;
        let modules = self.parallel_static_mut()?;
        let hidden = modules.norm.forward(hidden, stream)?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        let logits = if let Some(head) = &mut modules.lm_head {
            head.forward(&hidden, &execution)?
        } else {
            modules.embedding.project_logits(&hidden, &execution)?
        };
        logits.all_gather(&execution)
    }
}

/// Dense or MoE Qwen3-VL with independent vision and text residency windows.
pub struct Qwen3VlLayerwiseModel {
    execution: Qwen3VlExecution,
}

impl Qwen3VlLayerwiseModel {
    /// Returns the parsed multimodal model arguments.
    pub fn args(&self) -> &ModelArgs {
        &self.execution.architecture().args
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.topology = Some(topology);
        self.execution.architecture_mut().parallel_topology = Some(topology);
    }

    /// Returns sparse expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .architecture()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the public architecture type.
    pub fn model_type(&self) -> &'static str {
        if self.args().text_config.is_moe() {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
    }

    /// Creates empty KV and multimodal position state.
    pub fn new_cache(&self) -> Cache {
        Cache::new(self.args())
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.execution.parallel_info.as_ref()
    }

    /// Returns generalized parameter-residency and encoding metadata.
    pub fn residency_metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.execution.metadata
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns the complete rank-local multimodal prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        qwen3_vl_prompt_cache_identity(
            self.args(),
            self.execution.topology,
            self.execution.architecture().parallel_kv_heads.as_deref(),
        )
    }

    /// Persists a compatible multimodal prefix cache.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        resident::Model::save_prompt_cache(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
        .map_err(Into::into)
    }

    pub(crate) fn save_prompt_cache_with_validated_identity(
        &self,
        cache: &mut Cache,
        directory: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.save_prompt_cache(
            cache,
            directory,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Restores a compatible multimodal prefix cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        let _ = options;
        resident::Model::load_prompt_cache_with_identity(
            self.args(),
            directory,
            expected,
            prefix_token_ids,
            &self.prompt_cache_model_identity()?,
            stream,
        )
        .map_err(Into::into)
    }

    pub(crate) fn load_prompt_cache_with_validated_identity(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        let _ = options;
        resident::Model::load_prompt_cache_with_identity(
            self.args(),
            directory,
            expected,
            prefix_token_ids,
            identity,
            stream,
        )
        .map_err(Into::into)
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.execution.checkpoint_store()
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        self.execution.checkpoint_store_arc()
    }

    /// Runs typed multimodal prefill through vision and text execution groups.
    pub fn prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Qwen3VlInput::Prefill(input), cache, stream)
    }

    /// Runs a text decode step using cached multimodal RoPE state.
    pub fn decode(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Qwen3VlInput::Decode(tokens), cache, stream)
    }

    /// Runs a multimodal prefill through rank-local vision execution units.
    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_parallel(Qwen3VlInput::Prefill(input), cache, group, stream)
    }

    /// Runs text decode on a TP-loaded multimodal model.
    pub(crate) fn decode_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_parallel(Qwen3VlInput::Decode(tokens), cache, group, stream)
    }

    /// Runs streamed text layers while delegating routed experts to a caller.
    pub(crate) fn decode_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                &mut execute,
            );
        let hook = |architecture: &mut Qwen3VlArchitecture,
                    group,
                    index,
                    layer: &mut Qwen3VlUnit,
                    hidden: &Array,
                    cache: &mut Cache,
                    context: &mut Qwen3VlForwardContext,
                    stream: &Stream| match (group, &mut **layer) {
            (0, Qwen3VlLayer::Vision(block)) => {
                let Some(state) = context.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let modules = architecture.replicated_static_mut()?;
                let output =
                    modules
                        .vision
                        .forward_block(block, index, hidden.clone(), state, stream)?;
                modules
                    .vision
                    .capture_deepstack(index, &output, state, stream)?;
                Ok(output)
            }
            (1, Qwen3VlLayer::Text(block)) => {
                let mut output = forward_qwen3_vl_text_external_experts(
                    block,
                    hidden,
                    context.mask.as_ref(),
                    cache.kv[index].as_mut(),
                    &context.cos,
                    &context.sin,
                    index,
                    ExpertPass::Decode,
                    None,
                    stream,
                    &mut provider,
                )?;
                if let Some(features) = context.deepstack_features.get(index) {
                    let base = zeros_dtype(output.shape(), output.dtype(), stream)?;
                    let features = features.try_index_device((0, .., ..), stream)?;
                    let aligned = masked_scatter(
                        &base,
                        context.visual_mask.as_ref().expect("DeepStack visual mask"),
                        features,
                        stream,
                    )?;
                    output = output.add(aligned, stream)?;
                }
                Ok(output)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL execution unit does not match group {group}"
            ))),
        };
        match &mut self.execution.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime
                .forward_with_unit_executor(Qwen3VlInput::Decode(tokens), cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            Qwen3VlRuntime::Layerwise(runtime) => runtime
                .forward_with_unit_executor(Qwen3VlInput::Decode(tokens), cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    /// Runs the TP-sharded MRoPE decoder while delegating routed experts to
    /// the matching-coordinate EP exchange group.
    pub(crate) fn decode_tensor_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                &mut execute,
            );
        let hook = |_architecture: &mut Qwen3VlArchitecture,
                    group: usize,
                    index: usize,
                    layer: &mut Qwen3VlUnit,
                    hidden: &Array,
                    cache: &mut Cache,
                    context: &mut Qwen3VlForwardContext,
                    tp_group: &safemlx::distributed::Group,
                    stream: &Stream| {
            if group != 1 {
                return Err(Error::Parallel(format!(
                    "Qwen3-VL TP+EP decoder received non-text execution group {group}"
                )));
            }
            let Qwen3VlLayer::Text(block) = &mut **layer else {
                return Err(Error::Parallel(format!(
                    "Qwen3-VL TP+EP decoder received a vision unit at text layer {index}"
                )));
            };
            let mut output = forward_qwen3_vl_text_external_experts(
                block,
                hidden,
                context.mask.as_ref(),
                cache.kv[index].as_mut(),
                &context.cos,
                &context.sin,
                index,
                ExpertPass::Decode,
                Some(tp_group),
                stream,
                &mut provider,
            )?;
            if let Some(features) = context.deepstack_features.get(index) {
                let base = zeros_dtype(output.shape(), output.dtype(), stream)?;
                let features = features.try_index_device((0, .., ..), stream)?;
                let aligned = masked_scatter(
                    &base,
                    context.visual_mask.as_ref().expect("DeepStack visual mask"),
                    features,
                    stream,
                )?;
                output = output.add(aligned, stream)?;
            }
            Ok(output)
        };
        match &mut self.execution.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime
                .forward_parallel_with_unit_executor(
                    Qwen3VlInput::Decode(tokens),
                    cache,
                    tensor_group,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            Qwen3VlRuntime::Layerwise(runtime) => runtime
                .forward_parallel_with_unit_executor(
                    Qwen3VlInput::Decode(tokens),
                    cache,
                    tensor_group,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    /// Clears temporary copies for one execution group.
    pub fn clear_device_group(&self, group: &str) -> Result<(), Error> {
        match &self.execution.runtime {
            Qwen3VlRuntime::Resident(runtime) => runtime.policy().clear_device_group(group),
            Qwen3VlRuntime::Layerwise(runtime) => runtime.policy().clear_device_group(group),
        }
    }
}

impl CausalModel<Cache> for Qwen3VlLayerwiseModel {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.prefill(input, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

fn qwen3_vl_execution_layout(args: &ModelArgs) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["vision_encoder", "text_decoder"])?;
    ExecutionUnitLayout::new(
        &graph,
        [
            args.vision_config.layer_count(),
            args.text_config.num_hidden_layers as usize,
        ],
    )
    .map_err(|error| Error::Parallel(error.to_string()))
}

fn qwen3_vl_prompt_cache_identity(
    args: &ModelArgs,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
    parallel_kv_heads: Option<&[i32]>,
) -> Result<PromptCacheModelIdentity, Error> {
    let layer_count = args.text_config.num_hidden_layers as usize;
    let kv_heads = match topology {
        Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => parallel_kv_heads
            .ok_or_else(|| {
                Error::Parallel(
                    "Qwen3-VL parallel cache identity requested before local layout configuration"
                        .into(),
                )
            })?
            .to_vec(),
        _ => vec![args.text_config.num_key_value_heads; layer_count],
    };
    Ok(PromptCacheModelIdentity {
        model_family: "qwen3_vl".into(),
        effective_model_type: if args.text_config.is_moe() {
            "qwen3_vl_moe".into()
        } else {
            "qwen3_vl".into()
        },
        architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        sink_tokens: 0,
        layer_prefix_offsets: vec![0; layer_count],
        topology: topology.map_or_else(
            PromptCacheTopology::default,
            crate::backend::mlx::cache::prompt_cache_topology,
        ),
        layer_layout: resident::prompt_cache_layer_layout_with_kv_heads(args, &kv_heads)?,
    })
}

fn name_qwen3_vl_binding(binding: WeightBinding, name: String) -> Result<WeightBinding, Error> {
    binding.with_name(name).map_err(Into::into)
}

fn qwen3_vl_static_bindings(
    modules: &Qwen3VlStaticModules,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let Qwen3VlStaticModules::Replicated(modules) = modules else {
        return Err(Error::Parallel(
            "Qwen3-VL global static modules are sharded".into(),
        ));
    };
    let patch = "model.visual.patch_embed.proj.weight";
    let vision_recipes = if store.source_keys().contains(&format!("{patch}.1")) {
        BTreeMap::from([(
            "patch_embed.proj.weight".to_string(),
            DerivedWeightRecipe::Stack {
                axis: 2,
                inputs: vec![
                    DerivedWeightRecipe::source(patch, TensorSelection::Full),
                    DerivedWeightRecipe::source(format!("{patch}.1"), TensorSelection::Full),
                ],
            },
        )])
    } else {
        BTreeMap::new()
    };
    let mut bindings = build_module_binding_plan_with_recipes(
        &modules.vision,
        "model.visual",
        store,
        vision_recipes,
    )?
    .build_bindings(store)?
    .into_iter()
    .map(|binding| {
        let name = format!("vision.{}", binding.name());
        name_qwen3_vl_binding(binding, name)
    })
    .collect::<Result<Vec<_>, _>>()?;
    bindings.extend(
        build_module_binding_plan_with_recipes(
            &modules.embedding,
            "model.language_model.embed_tokens",
            store,
            BTreeMap::new(),
        )?
        .build_bindings(store)?
        .into_iter()
        .map(|binding| {
            let name = format!("embedding.{}", binding.name());
            name_qwen3_vl_binding(binding, name)
        })
        .collect::<Result<Vec<_>, _>>()?,
    );
    bindings.extend(
        build_module_binding_plan_with_recipes(
            &modules.norm,
            "model.language_model.norm",
            store,
            BTreeMap::new(),
        )?
        .build_bindings(store)?
        .into_iter()
        .map(|binding| {
            let name = format!("norm.{}", binding.name());
            name_qwen3_vl_binding(binding, name)
        })
        .collect::<Result<Vec<_>, _>>()?,
    );
    if let Some(head) = &modules.lm_head {
        bindings.extend(
            build_module_binding_plan_with_recipes(head, "lm_head", store, BTreeMap::new())?
                .build_bindings(store)?
                .into_iter()
                .map(|binding| {
                    let name = format!("lm_head.{}", binding.name());
                    name_qwen3_vl_binding(binding, name)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(bindings)
}

fn qwen3_vl_unit_bindings(
    args: &ModelArgs,
    group: usize,
    index: usize,
    layer: &Qwen3VlLayer,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    sparse_experts: bool,
) -> Result<Vec<WeightBinding>, Error> {
    if group == 1 {
        let Qwen3VlLayer::Text(layer) = layer else {
            return Err(Error::Parallel(
                "Qwen3-VL text group contains a vision unit".into(),
            ));
        };
        qwen_vl_text_layer_bindings(layer, &args.text_config, index, store, sparse_experts)
    } else {
        Ok(build_module_binding_plan_with_recipes(
            layer,
            &format!("model.visual.blocks.{index}"),
            store,
            BTreeMap::new(),
        )?
        .build_bindings(store)?)
    }
}

fn resolve_qwen3_vl_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: &ModelArgs,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend != WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan =
        super::checkpoint::safetensors_plan(args, true).map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3-VL checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn qwen3_vl_ordinal(args: &ModelArgs, ordinal: usize) -> (usize, usize) {
    let vision = args.vision_config.layer_count();
    if ordinal < vision {
        (0, ordinal)
    } else {
        (1, ordinal - vision)
    }
}

fn quantize_qwen3_vl_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &ModelArgs,
    sparse_experts: bool,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target_args = source_args.clone();
    target_args.text_config.quantization = Some(quantization);
    target_args.text_config.quantization_config = None;
    target_args.text_config.quantized_weight_configs = None;
    target_args
        .vision_config
        .apply_load_time_quantization(quantization);
    let source_static = Qwen3VlStaticModules::replicated(source_args, stream)?;
    let target_static = Qwen3VlStaticModules::replicated(&target_args, stream)?;
    let source_units = source_args.clone();
    let target_units = target_args.clone();
    let binding_args = source_args.clone();
    let unit_count = source_args.vision_config.layer_count()
        + source_args.text_config.num_hidden_layers as usize;
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |ordinal, stream| {
            let (group, index) = qwen3_vl_ordinal(&source_units, ordinal);
            build_qwen3_vl_unit(&source_units, group, index, None, sparse_experts, stream)
                .map(|unit| (*unit).clone())
        },
        move |ordinal, stream| {
            let (group, index) = qwen3_vl_ordinal(&target_units, ordinal);
            build_qwen3_vl_unit(&target_units, group, index, None, sparse_experts, stream)
                .map(|unit| (*unit).clone())
        },
        unit_count,
        quantization,
        stream,
        |modules, store| qwen3_vl_static_bindings(modules, store),
        move |ordinal, layer, store| {
            let (group, index) = qwen3_vl_ordinal(&binding_args, ordinal);
            qwen3_vl_unit_bindings(&binding_args, group, index, layer, store, sparse_experts)
        },
    )?;
    Ok((store, target_args, report))
}

fn load_qwen3_vl_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlExecution, Error> {
    let store = resolve_qwen3_vl_store(store, &args)?;
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_qwen3_vl_store(store, &args, sparse_experts, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut architecture = Qwen3VlArchitecture::new(args.clone(), sparse_experts, stream)?;
    let factory = Qwen3VlUnitFactory {
        args: args.clone(),
        vision_units: args.vision_config.layer_count(),
        parallel_layout: None,
        sparse_experts,
    };
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        qwen3_vl_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        |modules, store| qwen3_vl_static_bindings(modules, store),
        move |ordinal, unit, store, _| {
            let (group, index) = qwen3_vl_ordinal(&binding_args, ordinal);
            qwen3_vl_unit_bindings(&binding_args, group, index, &unit, store, sparse_experts)
        },
    )?;
    metadata.set_model_type(if args.text_config.is_moe() {
        "qwen3_vl_moe"
    } else {
        "qwen3_vl"
    });
    metadata.set_quantization(args.text_config.weight_quantization());
    metadata.set_materialization(materialization);
    let runtime = if options.is_fully_resident() {
        Qwen3VlRuntime::Resident(Box::new(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        )))
    } else {
        Qwen3VlRuntime::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(Qwen3VlExecution {
        runtime,
        metadata,
        parallel_info: None,
        topology: None,
    })
}

fn register_qwen3_vl_parallel_parameters(
    planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(), Error> {
    let Qwen3VlStaticModules::Replicated(modules) = Qwen3VlStaticModules::replicated(args, stream)?
    else {
        unreachable!()
    };
    for group in neutral_qwen::static_parallel_parameter_groups::<MlxBackend>(
        &modules.embedding,
        &modules.norm,
        modules.lm_head.as_ref(),
        "model.language_model",
    )? {
        planner.register(group)?;
    }
    for group in vision_parallel_parameter_groups(&args.vision_config, "model.visual", stream)? {
        planner.register(group)?;
    }
    for index in 0..args.text_config.num_hidden_layers as usize {
        let layer = neutral_qwen::new_block::<MlxBackend>(&args.text_config, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        for group in neutral_qwen::layer_parallel_parameter_groups::<MlxBackend>(
            &layer,
            &args.text_config,
            index,
        )? {
            planner.register(group)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_qwen3_vl_parallel_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlExecution, Error> {
    let store = resolve_qwen3_vl_store(store, &args)?;
    let mut planner = build.planner();
    register_qwen3_vl_parallel_parameters(&mut planner, &args, stream)?;
    let (_, local_layout) = planner.finish()?;
    let Qwen3VlStaticModules::Replicated(global_modules) =
        Qwen3VlStaticModules::replicated(&args, stream)?
    else {
        unreachable!()
    };
    let mut vision = global_modules.vision.clone();
    configure_vision_parallel_static(&mut vision, "model.visual", &local_layout, stream)?;
    let parallel_static = Qwen3VlStaticModules::Parallel(Qwen3VlParallelStatic {
        vision,
        embedding: VocabParallelEmbedding::unloaded(
            args.text_config.vocab_size as usize,
            args.text_config.hidden_size,
            args.text_config
                .quantization
                .or(args.text_config.quantization_config),
            build,
            stream,
        )?,
        norm: global_modules.norm.clone(),
        lm_head: if global_modules.lm_head.is_some() {
            Some(VocabParallelLmHead::unloaded(
                args.text_config.hidden_size,
                args.text_config.vocab_size as usize,
                args.text_config
                    .quantization
                    .or(args.text_config.quantization_config),
                build,
                stream,
            )?)
        } else {
            None
        },
    });
    let mut architecture = Qwen3VlArchitecture::new(args.clone(), sparse_experts, stream)?;
    architecture.static_modules = MlxParameterTree::new(parallel_static, "")
        .map_err(|error| Error::Parallel(error.to_string()))?;
    architecture.parallel_topology = Some(build.topology());
    architecture.parallel_kv_heads = Some(planned_kv_head_layout(
        &local_layout,
        args.text_config.num_hidden_layers as usize,
        args.text_config.head_dim,
        "model.language_model.layers",
    )?);

    let global_static = Qwen3VlStaticModules::replicated(&args, stream)?;
    let static_bindings = qwen3_vl_static_bindings(&global_static, store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&static_bindings)?;
    for group in 0..2 {
        let count = if group == 0 {
            args.vision_config.layer_count()
        } else {
            args.text_config.num_hidden_layers as usize
        };
        for index in 0..count {
            let global = build_qwen3_vl_unit(&args, group, index, None, sparse_experts, stream)?;
            global_parameter_bytes = global_parameter_bytes
                .checked_add(binding_bytes(&qwen3_vl_unit_bindings(
                    &args,
                    group,
                    index,
                    &global,
                    store.as_ref(),
                    sparse_experts,
                )?)?)
                .ok_or_else(|| {
                    Error::Parallel("Qwen3-VL global parameter bytes overflowed".into())
                })?;
        }
    }
    let shared_layout = Arc::new(local_layout);
    let factory = Qwen3VlUnitFactory {
        args: args.clone(),
        vision_units: args.vision_config.layer_count(),
        parallel_layout: Some(Arc::clone(&shared_layout)),
        sparse_experts,
    };
    let static_layout = Arc::clone(&shared_layout);
    let unit_layout = Arc::clone(&shared_layout);
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        qwen3_vl_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        move |_, store| shard_layer_bindings(static_bindings, "", store, &static_layout),
        move |ordinal, _local, store, stream| {
            let (group, index) = qwen3_vl_ordinal(&binding_args, ordinal);
            let global =
                build_qwen3_vl_unit(&binding_args, group, index, None, sparse_experts, stream)?;
            let prefix = if group == 0 {
                format!("model.visual.blocks.{index}")
            } else {
                format!("model.language_model.layers.{index}")
            };
            shard_layer_bindings(
                qwen3_vl_unit_bindings(
                    &binding_args,
                    group,
                    index,
                    &global,
                    store,
                    sparse_experts,
                )?,
                &prefix,
                store,
                &unit_layout,
            )
        },
    )?;
    metadata.set_model_type(if args.text_config.is_moe() {
        "qwen3_vl_moe"
    } else {
        "qwen3_vl"
    });
    metadata.set_quantization(args.text_config.weight_quantization());
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("Qwen3-VL local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("Qwen3-VL device parameter bytes overflowed".into()))?;
    let info = eredu_runtime::ParallelModelInfo::new(
        build.topology(),
        if args.text_config.is_moe() {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        },
        shared_layout
            .tensors()
            .map(|(target, _)| target.to_string())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if options.is_fully_resident() {
            local_parameter_bytes
        } else {
            metadata.static_device_bytes()
        },
        maximum_device_parameter_bytes,
    );
    let runtime = if options.is_fully_resident() {
        Qwen3VlRuntime::Resident(Box::new(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        )))
    } else {
        Qwen3VlRuntime::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(Qwen3VlExecution {
        runtime,
        metadata,
        parallel_info: Some(info),
        topology: Some(build.topology()),
    })
}

/// Loads either Qwen3-VL architecture through shared generalized residency.
pub fn load_qwen3_vl_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Qwen3-VL",
                args.text_config.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(Qwen3VlLayerwiseModel {
        execution: load_qwen3_vl_with_store(
            store,
            args,
            options,
            quantize_on_load,
            false,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Qwen3-VL with rank-local vision execution groups.
pub(crate) fn load_qwen3_vl_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let vision_path = resident::find_qwen3_vl_mmproj(model_dir)?;
        let vision_checkpoint = GgufCheckpoint::open(vision_path)?;
        let vision_metadata =
            crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
        return load_qwen3_vl_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            (&vision_checkpoint, &vision_metadata),
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    Ok(Qwen3VlLayerwiseModel {
        execution: load_qwen3_vl_parallel_with_store(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            args,
            options,
            build,
            false,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_qwen3_vl_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    vision: (&GgufCheckpoint, &HashMap<String, GgufMetadataValue>),
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Qwen3VlLayerwiseModel, Vec<u32>), Error> {
    let (vision_checkpoint, vision_metadata) = vision;
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared = resident::prepare_qwen3_vl_gguf_checkpoint(
        checkpoint,
        metadata,
        vision_checkpoint,
        vision_metadata,
    )?;
    let store = qwen3_vl_gguf_store(
        checkpoint,
        vision_checkpoint,
        &prepared.args,
        options.max_mapped_shards(),
    )?;
    let execution = load_qwen3_vl_parallel_with_store(
        store,
        prepared.args,
        options,
        build,
        false,
        stream,
        weights_stream,
    )?;
    Ok((Qwen3VlLayerwiseModel { execution }, prepared.eos_token_ids))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_qwen3_vl_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    vision_checkpoint: &GgufCheckpoint,
    vision_metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Qwen3VlLayerwiseModel, Vec<u32>), Error> {
    let prepared = resident::prepare_qwen3_vl_gguf_checkpoint(
        checkpoint,
        metadata,
        vision_checkpoint,
        vision_metadata,
    )?;
    let store = qwen3_vl_gguf_store(
        checkpoint,
        vision_checkpoint,
        &prepared.args,
        residency.max_mapped_shards(),
    )?;
    let args = prepared.args;
    let execution = if let Some(expert_options) = residency.expert_cache() {
        load_qwen3_vl_gguf_sparse_execution(
            store,
            args,
            expert_options,
            residency.layers(),
            quantization,
            stream,
            weights_stream,
        )?
    } else {
        load_qwen3_vl_with_store(
            store,
            args,
            residency.layers(),
            quantization,
            false,
            stream,
            weights_stream,
        )?
    };
    Ok((Qwen3VlLayerwiseModel { execution }, prepared.eos_token_ids))
}

fn load_qwen3_vl_gguf_sparse_execution(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: resident::ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlExecution, Error> {
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires qwen3vlmoe GGUF".into(),
        ));
    }
    let mut execution = load_qwen3_vl_with_store(
        store,
        args.clone(),
        non_expert,
        quantization,
        true,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries =
        crate::composition::qwen_expert::expert_catalog(&args.text_config, store.as_ref())?;
    execution.architecture_mut().expert_cache = Some(match quantization {
        Some(quantization) => ExpertCache::new_quantized_shared(
            store,
            entries,
            options,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        None => ExpertCache::new_shared(
            store,
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
        )?,
    });
    Ok(execution)
}

pub(crate) fn qwen3_vl_gguf_store(
    checkpoint: &GgufCheckpoint,
    vision_checkpoint: &GgufCheckpoint,
    args: &resident::ModelArgs,
    max_mapped_shards: usize,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    let deepstack = args.vision_config.deepstack_layers();
    let is_moe = args.text_config.is_moe();
    let text_plan = eredu_architectures::qwen::gguf_plan(&args.text_config)
        .map_err(Error::UnsupportedArchitecture)?;
    let vision_plan =
        super::checkpoint::projector_gguf_plan(&args.vision_config, args.text_config.hidden_size)
            .map_err(Error::UnsupportedArchitecture)?;
    Ok(Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(max_mapped_shards)?
            .add_checkpoint(checkpoint.catalog().clone(), &text_plan, move |name| {
                let name = eredu_architectures::qwen::translate_gguf_weight_name(name, is_moe);
                name.strip_prefix("model.")
                    .map(|name| format!("model.language_model.{name}"))
                    .unwrap_or(name)
            })?
            .add_checkpoint(
                vision_checkpoint.catalog().clone(),
                &vision_plan,
                move |name| resident::translate_qwen3_vl_mmproj_name(name, &deepstack),
            )?
            .build()?,
    ))
}

/// Loads Qwen3-VL-MoE with independently cached experts and bounded non-expert units.
pub fn load_qwen3_vl_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3-VL-MoE checkpoint".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Qwen3-VL independent expert cache",
                args.text_config.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut execution = load_qwen3_vl_with_store(
        store,
        args.clone(),
        non_expert.layers(),
        quantize_on_load,
        true,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries =
        crate::composition::qwen_expert::expert_catalog(&args.text_config, store.as_ref())?;
    execution.architecture_mut().expert_cache = Some(match quantize_on_load {
        Some(quantization) => ExpertCache::new_quantized_shared(
            store,
            entries,
            options,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        None => ExpertCache::new_shared(
            store,
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
        )?,
    });
    Ok(Qwen3VlLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Qwen3-VL execution base used by distributed EP.
pub(crate) fn load_qwen3_vl_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "streamed sparse expert parallelism requires Qwen3-VL-MoE".into(),
        ));
    }
    let execution = load_qwen3_vl_with_store(
        store,
        args,
        non_expert.into(),
        None,
        true,
        stream,
        weights_stream,
    )?;
    Ok(Qwen3VlLayerwiseModel { execution })
}

/// Builds the TP-sharded nonexpert Qwen3-VL-MoE base used by combined TP+EP.
pub(crate) fn load_qwen3_vl_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "combined tensor/expert parallelism requires Qwen3-VL-MoE".into(),
        ));
    }
    let execution = load_qwen3_vl_parallel_with_store(
        store,
        args,
        non_expert.into(),
        build,
        true,
        stream,
        weights_stream,
    )?;
    Ok(Qwen3VlLayerwiseModel { execution })
}

/// Family-specific input distinguishing typed prefill from cached decode.
pub enum Qwen3VlInput<'a> {
    /// Ordered text and visual prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text token IDs for a cached decode step.
    Decode(&'a Array),
}

enum PreparedPart {
    Text(Array),
    Visual(i32),
}

/// Per-forward vision preparation and multimodal text state.
pub struct Qwen3VlForwardContext {
    tokens: Array,
    parts: Vec<PreparedPart>,
    vision: Option<QwenVisionLayerwiseState>,
    mask: Option<Array>,
    cos: Array,
    sin: Array,
    visual_mask: Option<Array>,
    deepstack_features: Vec<Array>,
}

/// Architecture-authored immutable state relayed with pipeline activations.
///
/// DeepStack features are already aligned to the assembled decoder sequence,
/// so the transport layer only needs stable `[batch, sequence, hidden]`
/// tensors and never needs to understand visual placeholder semantics.
pub(crate) struct Qwen3VlPipelinePrepared {
    pub(crate) hidden: Array,
    pub(crate) cos: Array,
    pub(crate) sin: Array,
    pub(crate) rope_delta: i32,
    pub(crate) deepstack_features: Vec<Array>,
}

/// Opaque prepared state routed between placed Qwen3-VL vision owners.
pub(crate) struct Qwen3VlPipelineIngressState {
    cache: Cache,
    forward: eredu_runtime::LayeredForwardState<Array, Qwen3VlForwardContext>,
}

/// One temporary unit from either the vision or text group.
#[derive(Clone)]
pub enum Qwen3VlLayer {
    /// Vision transformer block.
    Vision(Box<QwenVisionBlock>),
    /// Dense or sparse-MoE Qwen3 decoder block.
    Text(Box<NeutralTextBlock>),
}

impl ModuleParameters for Qwen3VlLayer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Vision(layer) => layer.num_parameters(),
            Self::Text(layer) => layer.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.parameters(),
            Self::Text(layer) => layer.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Vision(layer) => layer.parameters_mut(),
            Self::Text(layer) => layer.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.trainable_parameters(),
            Self::Text(layer) => layer.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.freeze_parameters(recursive),
            Self::Text(layer) => layer.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.unfreeze_parameters(recursive),
            Self::Text(layer) => layer.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.all_frozen(),
            Self::Text(layer) => layer.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.any_frozen(),
            Self::Text(layer) => layer.any_frozen(),
        }
    }
}

/// Pipeline-staging legacy adapter retained until the old MLX runtime is deleted.
/// Production model loading and inference use `Qwen3VlArchitecture` above.
pub struct Qwen3VlLayerwiseAdapter {
    args: ModelArgs,
    vision: QwenVisionLayerwiseStatic,
    embedding: MaybeQuantized<nn::Embedding>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl Qwen3VlLayerwiseAdapter {
    /// Starts placed Qwen3-VL ingress before any vision block executes.
    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        typed: input::ModelInput<'_>,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelineIngressState, Error> {
        let mut cache = Cache::new(&self.args);
        let forward = self.prepare_prefill(typed, &mut cache, execution, stream)?;
        Ok(Qwen3VlPipelineIngressState { cache, forward })
    }

    /// Rebuilds only parameter-free vision scheduling state on a downstream
    /// PP owner. Patch/text embeddings and projectors stay on their declared
    /// static owners; the previous encoder owner supplies the hidden tensor.
    pub(crate) fn begin_pipeline_continuation(
        &self,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelineIngressState, Error> {
        input::validate(typed)?;
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        for part in typed.parts {
            if matches!(
                part.modality,
                input::Modality::Image | input::Modality::Video
            ) {
                let input::InputPayload::Tensor(tensor) = part.payload else {
                    return Err(Error::Parallel(
                        "Qwen3-VL continuation requires tensor media payloads".into(),
                    ));
                };
                let grid = part.metadata.patch_grid.ok_or_else(|| {
                    Error::Parallel("Qwen3-VL continuation omitted patch_grid".into())
                })?;
                pixels.push(tensor.clone());
                grids.push(grid.clone());
            }
        }
        if pixels.is_empty() {
            return Err(Error::Parallel(
                "Qwen3-VL continuation requires at least one visual payload".into(),
            ));
        }
        let pixel_refs = pixels.iter().collect::<Vec<_>>();
        let grid_refs = grids.iter().collect::<Vec<_>>();
        let pixels = concatenate_axis(&pixel_refs, 0, stream)?;
        let grids = concatenate_axis(&grid_refs, 0, stream)?;
        let vision = self.vision.continuation_state(&pixels, &grids, stream)?;
        let hidden = zeros_dtype(
            &[pixels.dim(0), self.args.vision_config.hidden_size],
            pixels.dtype(),
            stream,
        )?;
        let empty = zeros_dtype(
            &[1, 0, self.args.text_config.head_dim],
            pixels.dtype(),
            stream,
        )?;
        Ok(Qwen3VlPipelineIngressState {
            cache: Cache::new(&self.args),
            forward: eredu_runtime::LayeredForwardState {
                hidden,
                context: Qwen3VlForwardContext {
                    tokens: Array::from_slice(&[] as &[u32], &[1, 0]),
                    parts: Vec::new(),
                    vision: Some(vision),
                    mask: None,
                    cos: empty.clone(),
                    sin: empty,
                    visual_mask: None,
                    deepstack_features: Vec::new(),
                },
            },
        })
    }

    /// Returns whether this request contains model-native visual work.
    pub(crate) fn pipeline_ingress_active(&self, state: &Qwen3VlPipelineIngressState) -> bool {
        state.forward.context.vision.is_some()
    }

    /// Exports variable DeepStack state and the evolving vision activation.
    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &Qwen3VlPipelineIngressState,
    ) -> Vec<Array> {
        std::iter::once(state.forward.hidden.clone())
            .chain(
                state
                    .forward
                    .context
                    .vision
                    .iter()
                    .flat_map(QwenVisionLayerwiseState::retained_arrays)
                    .cloned(),
            )
            .collect()
    }

    /// Imports activation and DeepStack features from the previous PP owner.
    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut Qwen3VlPipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let (hidden, features) = arrays.split_first().ok_or_else(|| {
            Error::Parallel("Qwen3-VL placed ingress omitted vision hidden state".into())
        })?;
        state.forward.hidden = hidden.clone();
        if let Some(vision) = &mut state.forward.context.vision {
            vision.replace_deepstack_features(features.to_vec());
        } else if !features.is_empty() {
            return Err(Error::Parallel(
                "Qwen3-VL text-only ingress received DeepStack state".into(),
            ));
        }
        Ok(())
    }

    /// Executes one placed vision block.
    pub(crate) fn forward_pipeline_vision_layer(
        &mut self,
        index: usize,
        layer: &mut Qwen3VlLayer,
        state: &mut Qwen3VlPipelineIngressState,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Vec<Array>, Error> {
        state.forward.hidden = match execution {
            Some(execution) => self.forward_layer_with_execution(
                0,
                index,
                layer,
                &state.forward.hidden,
                &mut state.cache,
                &mut state.forward.context,
                execution,
            )?,
            None => self.forward_layer(
                0,
                index,
                layer,
                &state.forward.hidden,
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        Ok(self.pipeline_ingress_arrays(state))
    }

    /// Finalizes merger/projector state into decoder-facing payload tensors.
    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: Qwen3VlPipelineIngressState,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelinePrepared, Error> {
        let hidden = match execution {
            Some(execution) => self.begin_execution_group_with_execution(
                1,
                &state.forward.hidden,
                &[state.forward.hidden.clone()],
                &mut state.cache,
                &mut state.forward.context,
                execution,
            )?,
            None => self.begin_execution_group(
                1,
                &state.forward.hidden,
                &[state.forward.hidden.clone()],
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        let deepstack_features = state
            .forward
            .context
            .deepstack_features
            .iter()
            .map(|features| {
                let base = zeros_dtype(hidden.shape(), hidden.dtype(), stream)?;
                masked_scatter(
                    &base,
                    state
                        .forward
                        .context
                        .visual_mask
                        .as_ref()
                        .expect("DeepStack visual mask"),
                    features.try_index_device((0, .., ..), stream)?,
                    stream,
                )
                .map_err(Error::from)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Qwen3VlPipelinePrepared {
            hidden,
            cos: state.forward.context.cos,
            sin: state.forward.context.sin,
            rope_delta: state.cache.rope_delta,
            deepstack_features,
        })
    }

    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let visual = QwenVisionTransformer::new_deepstack(args.vision_config.clone(), stream)?;
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.text_config.vocab_size,
            args.text_config.hidden_size,
            args.text_config.weight_quantization_for(&format!(
                "{}.embed_tokens.weight",
                args.text_config.parameter_root
            )),
            stream,
        )?;
        let norm = nn::RmsNorm::unloaded(
            args.text_config.hidden_size,
            args.text_config.rms_norm_eps,
            safemlx::Dtype::Float32,
            stream,
        )?;
        let lm_head = if args.text_config.tie_word_embeddings {
            None
        } else {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.text_config.hidden_size,
                    args.text_config.vocab_size,
                    args.text_config
                        .quantization
                        .or(args.text_config.quantization_config),
                    stream,
                )?,
            )
        };
        Ok(Self {
            args,
            vision: QwenVisionLayerwiseStatic::from_transformer(visual),
            embedding,
            parallel_embedding: None,
            norm,
            lm_head,
            parallel_lm_head: None,
            parallel_kv_heads: None,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns parsed multimodal arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    pub(crate) fn vision_mut(&mut self) -> &mut QwenVisionLayerwiseStatic {
        &mut self.vision
    }

    pub(crate) fn embedding_mut(&mut self) -> &mut MaybeQuantized<nn::Embedding> {
        &mut self.embedding
    }

    pub(crate) fn parallel_embedding_mut(&mut self) -> Option<&mut VocabParallelEmbedding> {
        self.parallel_embedding.as_mut()
    }

    pub(crate) fn norm_mut(&mut self) -> &mut nn::RmsNorm {
        &mut self.norm
    }

    pub(crate) fn lm_head_mut(&mut self) -> Option<&mut MaybeQuantized<nn::Linear>> {
        self.lm_head.as_mut()
    }

    pub(crate) fn parallel_lm_head_mut(&mut self) -> Option<&mut VocabParallelLmHead> {
        self.parallel_lm_head.as_mut()
    }

    /// Executes the local multimodal reference path and returns decoder-facing
    /// state. Distributed execution uses the same architecture-owned methods
    /// through the generic placement and transport plan.
    pub(crate) fn prepare_pipeline_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        vision_layers: &mut [Qwen3VlLayer],
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelinePrepared, Error> {
        let mut cache = Cache::new(&self.args);
        let mut state = self.prepare_prefill(typed, &mut cache, execution, stream)?;
        if state.context.vision.is_some() {
            if vision_layers.len() != self.args.vision_config.layer_count() {
                return Err(Error::Parallel(format!(
                    "Qwen3-VL local reference owns {} vision blocks, expected {}",
                    vision_layers.len(),
                    self.args.vision_config.layer_count()
                )));
            }
            for (index, layer) in vision_layers.iter_mut().enumerate() {
                state.hidden = if let Some(execution) = execution {
                    self.forward_layer_with_execution(
                        0,
                        index,
                        layer,
                        &state.hidden,
                        &mut cache,
                        &mut state.context,
                        execution,
                    )?
                } else {
                    self.forward_layer(
                        0,
                        index,
                        layer,
                        &state.hidden,
                        &mut cache,
                        &mut state.context,
                        stream,
                    )?
                };
            }
        }
        let hidden = if let Some(execution) = execution {
            self.begin_execution_group_with_execution(
                1,
                &state.hidden,
                &[state.hidden.clone()],
                &mut cache,
                &mut state.context,
                execution,
            )?
        } else {
            self.begin_execution_group(
                1,
                &state.hidden,
                &[state.hidden.clone()],
                &mut cache,
                &mut state.context,
                stream,
            )?
        };
        let deepstack_features = state
            .context
            .deepstack_features
            .iter()
            .map(|features| {
                let base = zeros_dtype(hidden.shape(), hidden.dtype(), stream)?;
                masked_scatter(
                    &base,
                    state
                        .context
                        .visual_mask
                        .as_ref()
                        .expect("DeepStack visual mask"),
                    features.try_index_device((0, .., ..), stream)?,
                    stream,
                )
                .map_err(Error::from)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Qwen3VlPipelinePrepared {
            hidden,
            cos: state.context.cos,
            sin: state.context.sin,
            rope_delta: cache.rope_delta,
            deepstack_features,
        })
    }

    /// Embeds a text-only pipeline step and reconstructs MRoPE from the
    /// persisted multimodal position delta.
    pub(crate) fn prepare_pipeline_tokens(
        &mut self,
        tokens: &Array,
        offset: i32,
        rope_delta: i32,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelinePrepared, Error> {
        let hidden = match (&mut self.parallel_embedding, execution) {
            (Some(embedding), Some(execution)) => embedding.forward(tokens, execution)?,
            _ => self.embedding.forward(tokens, stream)?,
        };
        let start = offset + rope_delta;
        let positions = [
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
        ];
        let (cos, sin) = resident::mrope_embeddings(
            &positions,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        let deepstack_features = (0..self.args.vision_config.deepstack_layer_count())
            .map(|_| zeros_dtype(hidden.shape(), hidden.dtype(), stream).map_err(Error::from))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Qwen3VlPipelinePrepared {
            hidden,
            cos,
            sin,
            rope_delta,
            deepstack_features,
        })
    }

    fn prepare_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        cache: &mut Cache,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, Qwen3VlForwardContext>, Error> {
        input::validate(typed)?;
        let mut token_parts = Vec::new();
        let mut prepared_parts = Vec::new();
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        let merge = self.args.vision_config.spatial_merge_size;
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    token_parts.push(tokens.clone());
                    let embedding = match (&mut self.parallel_embedding, execution) {
                        (Some(embedding), Some(execution)) => {
                            embedding.forward(tokens, execution)?
                        }
                        _ => self.embedding.forward(tokens, stream)?,
                    };
                    prepared_parts.push(PreparedPart::Text(embedding));
                }
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Tensor(tensor),
                ) => {
                    let grid = part.metadata.patch_grid.ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Qwen3-VL {} input requires patch_grid metadata",
                            part.modality.as_str()
                        ))
                    })?;
                    let merged = grid_thw_from_array(grid, stream)?
                        .into_iter()
                        .map(|(t, h, w)| t * (h / merge) * (w / merge))
                        .sum::<i32>();
                    let token_id = if part.modality == input::Modality::Image {
                        self.args.image_token_id
                    } else {
                        self.args.video_token_id
                    };
                    token_parts.push(input::token_ids_array(
                        &vec![token_id; merged as usize],
                        stream,
                    )?);
                    prepared_parts.push(PreparedPart::Visual(merged));
                    pixels.push(tensor.clone());
                    grids.push(grid.clone());
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Qwen3-VL layerwise input does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        let token_refs = token_parts.iter().collect::<Vec<_>>();
        let tokens = concatenate_axis(&token_refs, 1, stream)?;
        let (position_ids, rope_delta) =
            resident::multimodal_position_ids(typed, merge, tokens.dim(1), stream)?;
        cache.rope_delta = rope_delta;
        let (cos, sin) = resident::mrope_embeddings(
            &position_ids,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        let (hidden, vision) = if pixels.is_empty() {
            let hidden = prepared_parts
                .iter()
                .filter_map(|part| match part {
                    PreparedPart::Text(value) => Some(value),
                    PreparedPart::Visual(_) => None,
                })
                .collect::<Vec<_>>();
            (concatenate_axis(&hidden, 1, stream)?, None)
        } else {
            let pixel_refs = pixels.iter().collect::<Vec<_>>();
            let grid_refs = grids.iter().collect::<Vec<_>>();
            let pixels = concatenate_axis(&pixel_refs, 0, stream)?;
            let grids = concatenate_axis(&grid_refs, 0, stream)?;
            let (hidden, state) = self.vision.begin(&pixels, &grids, stream)?;
            (hidden, Some(state))
        };
        Ok(eredu_runtime::LayeredForwardState {
            hidden,
            context: Qwen3VlForwardContext {
                tokens,
                parts: prepared_parts,
                vision,
                mask: None,
                cos,
                sin,
                visual_mask: None,
                deepstack_features: Vec::new(),
            },
        })
    }
}

impl Qwen3VlLayerwiseAdapter {}

impl Qwen3VlLayerwiseAdapter {
    pub(crate) fn model_type(&self) -> &str {
        if self.args.text_config.is_moe() {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
    }

    pub(crate) fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        let vision = target
            .strip_prefix("model.visual.")
            .or_else(|| target.split_once(".visual.").map(|(_, suffix)| suffix));
        match vision {
            Some(target) => self
                .args
                .vision_config
                .quantized_weight_configs
                .contains_key(target),
            None => true,
        }
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = self.args.text_config.num_hidden_layers as usize;
        let local_kv_heads = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                self.parallel_kv_heads.clone().ok_or_else(|| {
                    Error::Parallel(
                    "Qwen3-VL parallel cache identity requested before local layout configuration"
                        .into(),
                )
                })?
            }
            _ => vec![self.args.text_config.num_key_value_heads; layer_count],
        };
        Ok(PromptCacheModelIdentity {
            model_family: "qwen3_vl".into(),
            effective_model_type: self.model_type().into(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
            layer_layout: resident::prompt_cache_layer_layout_with_kv_heads(
                &self.args,
                &local_kv_heads,
            )?,
        })
    }

    pub(crate) fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let patch = "model.visual.patch_embed.proj.weight";
        let vision_recipes = if store.source_keys().contains(&format!("{patch}.1")) {
            BTreeMap::from([(
                "patch_embed.proj.weight".to_string(),
                DerivedWeightRecipe::Stack {
                    axis: 2,
                    inputs: vec![
                        DerivedWeightRecipe::source(patch, TensorSelection::Full),
                        DerivedWeightRecipe::source(format!("{patch}.1"), TensorSelection::Full),
                    ],
                },
            )])
        } else {
            BTreeMap::new()
        };
        let mut units = vec![
            StaticUnitBindings::new(
                VISION_STATIC_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.vision,
                    "model.visual",
                    store,
                    vision_recipes,
                )?
                .build_bindings(store)?,
            )?,
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.embedding,
                    "model.language_model.embed_tokens",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.norm,
                    "model.language_model.norm",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?,
        ];
        if let Some(head) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_binding_plan_with_recipes(head, "lm_head", store, BTreeMap::new())?
                    .build_bindings(store)?,
            )?);
        }
        Ok(units)
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Qwen3VlLayer, Error> {
        match group {
            0 => Ok(Qwen3VlLayer::Vision(Box::new(QwenVisionBlock::new(
                &self.args.vision_config,
                index,
                stream,
            )?))),
            1 => Ok(Qwen3VlLayer::Text(Box::new(MlxModule::new(
                neutral_qwen::new_block::<MlxBackend>(&self.args.text_config, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            )))),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    pub(crate) fn parallel_parameter_groups(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, Error> {
        let mut groups = vec![vocab_embedding_parameter_group(
            &self.embedding,
            "model.language_model.embed_tokens",
            self.args.text_config.vocab_size as usize,
            self.args.text_config.hidden_size,
            false,
        )?];
        if let Some(head) = &self.lm_head {
            groups.push(vocab_lm_head_parameter_group(
                head,
                "lm_head",
                self.args.text_config.hidden_size,
                self.args.text_config.vocab_size as usize,
                false,
            )?);
        }
        Ok(groups)
    }

    pub(crate) fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let config = &self.args.text_config;
        self.parallel_kv_heads = Some(planned_kv_head_layout(
            layout,
            config.num_hidden_layers as usize,
            config.head_dim,
            "model.language_model.layers",
        )?);
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            config.vocab_size as usize,
            config.hidden_size,
            config.quantization.or(config.quantization_config),
            context,
            stream,
        )?);
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                config.hidden_size,
                config.vocab_size as usize,
                config.quantization.or(config.quantization_config),
                context,
                stream,
            )?);
        }
        configure_vision_parallel_static(&mut self.vision, "model.visual", layout, stream)?;
        Ok(())
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        for group in self.parallel_parameter_groups(context)? {
            planner.register(group)?;
        }
        for group in
            vision_parallel_parameter_groups(&self.args.vision_config, "model.visual", stream)?
        {
            planner.register(group)?;
        }
        for index in 0..self.args.text_config.num_hidden_layers as usize {
            let layer =
                neutral_qwen::new_block::<MlxBackend>(&self.args.text_config, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for group in neutral_qwen::layer_parallel_parameter_groups::<MlxBackend>(
                &layer,
                &self.args.text_config,
                index,
            )? {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Qwen3VlLayer, Error> {
        if group != 0 {
            let prefix = format!("model.language_model.layers.{index}");
            let planned = |name: &str| {
                layout
                    .tensor(&format!("{prefix}.{name}.weight"))
                    .or_else(|| layout.tensor(&format!("{prefix}.{name}.inner.weight")))
            };
            let mut config = self.args.text_config.clone();
            let query = planned("self_attn.q_proj")
                .ok_or_else(|| Error::Parallel(format!("missing TP query for {prefix}")))?;
            let key = planned("self_attn.k_proj")
                .ok_or_else(|| Error::Parallel(format!("missing TP key for {prefix}")))?;
            config.num_attention_heads = i32::try_from(query.local_shape()[0])
                .map_err(|_| Error::Parallel("Qwen local query width exceeds i32".into()))?
                / config.head_dim;
            config.num_key_value_heads = i32::try_from(key.local_shape()[0])
                .map_err(|_| Error::Parallel("Qwen local key width exceeds i32".into()))?
                / config.head_dim;
            if config.is_moe() {
                let experts = layout
                    .tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))
                    .ok_or_else(|| Error::Parallel(format!("missing TP experts for {prefix}")))?;
                config.moe_intermediate_size = i32::try_from(experts.local_shape()[1] / 2)
                    .map_err(|_| Error::Parallel("Qwen local expert width exceeds i32".into()))?;
            } else {
                let gate = planned("mlp.gate_proj")
                    .ok_or_else(|| Error::Parallel(format!("missing TP MLP for {prefix}")))?;
                config.intermediate_size = i32::try_from(gate.local_shape()[0])
                    .map_err(|_| Error::Parallel("Qwen local MLP width exceeds i32".into()))?;
            }
            return Ok(Qwen3VlLayer::Text(Box::new(MlxModule::new(
                neutral_qwen::new_block::<MlxBackend>(&config, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            ))));
        }
        let config = &self.args.vision_config;
        Ok(Qwen3VlLayer::Vision(Box::new(new_parallel_vision_block(
            config,
            "model.visual",
            index,
            layout,
            stream,
        )?)))
    }

    pub(crate) fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Qwen3VlLayer, Error> {
        if group != 1 || !self.args.text_config.is_moe() {
            return Err(Error::Parallel(format!(
                "Qwen3-VL expert-local construction requires an MoE text layer, got group {group}"
            )));
        }
        let mut layer = self.new_layer(group, index, stream)?;
        let Qwen3VlLayer::Text(block) = &mut layer else {
            unreachable!("validated text group")
        };
        let NeutralFeedForward::Routed(moe) = &mut block.mlp else {
            return Err(Error::Parallel(format!(
                "Qwen3-VL text layer {index} is not an MoE layer"
            )));
        };
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen3-VL expert count exceeds i32".into()))?
        };
        if local_experts > 0 {
            moe.experts = unloaded_expert_bank(
                &self.args.text_config,
                index,
                local_experts,
                self.args.text_config.moe_intermediate_size,
                stream,
            )?;
        }
        Ok(layer)
    }

    pub(crate) fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Qwen3VlLayer, Error> {
        if group != 1 || !self.args.text_config.is_moe() {
            return Err(Error::Parallel(format!(
                "Qwen3-VL combined tensor/expert construction requires an MoE text layer, got group {group}"
            )));
        }
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        let Qwen3VlLayer::Text(block) = &mut layer else {
            unreachable!("validated text group")
        };
        let NeutralFeedForward::Routed(moe) = &mut block.mlp else {
            return Err(Error::Parallel(format!(
                "Qwen3-VL text layer {index} is not an MoE layer"
            )));
        };
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen3-VL expert count exceeds i32".into()))?
        };
        if local_experts > 0 {
            let local = neutral_qwen::local_block_args(&self.args.text_config, index, layout)?;
            moe.experts = unloaded_expert_bank(
                &self.args.text_config,
                index,
                local_experts,
                local.moe_intermediate_size,
                stream,
            )?;
        }
        Ok(layer)
    }

    pub(crate) fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        if !self.args.text_config.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen3-VL has no routed experts for expert-parallel ownership".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.text_config.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub(crate) fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("model.visual.blocks.{index}")
        } else {
            format!("model.language_model.layers.{index}")
        }
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Qwen3VlLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = self.layer_checkpoint_prefix(group, index);
        if group == 1 {
            let Qwen3VlLayer::Text(layer) = layer else {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3-VL text execution group contains a vision layer at {index}"
                )));
            };
            qwen_vl_text_layer_bindings(
                layer,
                &self.args.text_config,
                index,
                store,
                self.sparse_expert_cache,
            )
        } else {
            Ok(
                build_module_binding_plan_with_recipes(layer, &prefix, store, BTreeMap::new())?
                    .build_bindings(store)?,
            )
        }
    }

    pub(crate) fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Qwen3VlLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
            self.layer_bindings(group, index, &global, store)?,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    pub(crate) fn expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Qwen3VlLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        if group != 1 {
            return Err(Error::Parallel(format!(
                "Qwen3-VL expert-local bindings require text group 1, got {group}"
            )));
        }
        let global = self.new_layer(group, index, stream)?;
        let indices = assignment.local_global_expert_ids().to_vec();
        self.layer_bindings(group, index, &global, store)?
            .into_iter()
            .map(|binding| {
                let target = binding.logical_target().unwrap_or_else(|| binding.name());
                if target.contains(".experts.") {
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
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Qwen3VlLayer,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut Qwen3VlForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, layer) {
            (0, Qwen3VlLayer::Vision(block)) => {
                let Some(state) = context.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let output =
                    self.vision
                        .forward_block(block, index, hidden.clone(), state, stream)?;
                self.vision
                    .capture_deepstack(index, &output, state, stream)?;
                Ok(output)
            }
            (1, Qwen3VlLayer::Text(block)) => {
                let mut output = if self.sparse_expert_cache {
                    let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Qwen3-VL sparse expert cache was not initialized".into(),
                        )
                    })?;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    let mut provider = crate::composition::qwen_expert::cached_provider(
                        expert_cache,
                        &self.args.text_config,
                    );
                    forward_qwen3_vl_text_external_experts(
                        block,
                        hidden,
                        context.mask.as_ref(),
                        cache.kv[index].as_mut(),
                        &context.cos,
                        &context.sin,
                        index,
                        pass,
                        None,
                        stream,
                        &mut provider,
                    )?
                } else {
                    forward_qwen3_vl_text(
                        block,
                        hidden,
                        context.mask.as_ref(),
                        cache.kv[index].as_mut(),
                        &context.cos,
                        &context.sin,
                        None,
                        stream,
                    )?
                };
                if let Some(features) = context.deepstack_features.get(index) {
                    let base = zeros_dtype(output.shape(), output.dtype(), stream)?;
                    let features = features.try_index_device((0, .., ..), stream)?;
                    let aligned = masked_scatter(
                        &base,
                        context.visual_mask.as_ref().expect("DeepStack visual mask"),
                        features,
                        stream,
                    )?;
                    output = output.add(aligned, stream)?;
                }
                Ok(output)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL execution unit does not match group {group}"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Qwen3VlLayer,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut Qwen3VlForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        let Some(tp_group) = execution.group() else {
            return self.forward_layer(
                group,
                index,
                layer,
                hidden,
                cache,
                context,
                execution.stream(),
            );
        };
        if group == 0 {
            if let Qwen3VlLayer::Vision(block) = layer {
                let Some(state) = context.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let output = self.vision.forward_block_tensor_parallel(
                    block,
                    index,
                    hidden.clone(),
                    state,
                    tp_group,
                    execution.stream(),
                )?;
                self.vision.capture_deepstack_tensor_parallel(
                    index,
                    &output,
                    state,
                    tp_group,
                    execution.stream(),
                )?;
                return Ok(output);
            }
        } else if let Qwen3VlLayer::Text(block) = layer {
            return forward_qwen3_vl_text(
                block,
                hidden,
                context.mask.as_ref(),
                cache.kv[index].as_mut(),
                &context.cos,
                &context.sin,
                Some(tp_group),
                execution.stream(),
            );
        }
        self.forward_layer(
            group,
            index,
            layer,
            hidden,
            cache,
            context,
            execution.stream(),
        )
    }

    pub(crate) fn begin_execution_group(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut Cache,
        context: &mut Qwen3VlForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if group == 0 {
            return Ok(initial_hidden.clone());
        }
        if group != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            )));
        }
        let hidden = dependency_outputs.first().unwrap_or(initial_hidden);
        let hidden = if let Some(mut state) = context.vision.take() {
            let output = self.vision.finish(hidden, &mut state, stream)?;
            context.deepstack_features = output.deepstack_features;
            let mut visual_offset = 0;
            let mut assembled = Vec::with_capacity(context.parts.len());
            for part in &context.parts {
                match part {
                    PreparedPart::Text(embedding) => assembled.push(embedding.clone()),
                    PreparedPart::Visual(len) => {
                        assembled.push(output.embeddings.try_index_device(
                            (.., visual_offset..visual_offset + *len, ..),
                            stream,
                        )?);
                        visual_offset += *len;
                    }
                }
            }
            let refs = assembled.iter().collect::<Vec<_>>();
            concatenate_axis(&refs, 1, stream)?
        } else {
            hidden.clone()
        };
        context.mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Some(mask),
            Some(AttentionMask::Causal) => {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen3-VL layerwise execution requires an explicit causal mask".into(),
                ));
            }
            None => None,
        };
        context.visual_mask = if context.deepstack_features.is_empty() {
            None
        } else {
            Some(
                context
                    .tokens
                    .eq(Array::from_int(self.args.image_token_id as i32), stream)?
                    .logical_or(
                        &context
                            .tokens
                            .eq(Array::from_int(self.args.video_token_id as i32), stream)?,
                        stream,
                    )?,
            )
        };
        Ok(hidden)
    }

    pub(crate) fn begin_execution_group_with_execution(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut Cache,
        context: &mut Qwen3VlForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        let Some(tp_group) = execution.group() else {
            return self.begin_execution_group(
                group,
                initial_hidden,
                dependency_outputs,
                cache,
                context,
                execution.stream(),
            );
        };
        if group == 0 {
            return Ok(initial_hidden.clone());
        }
        if group != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            )));
        }
        let hidden = dependency_outputs.first().unwrap_or(initial_hidden);
        let stream = execution.stream();
        let hidden = if let Some(mut state) = context.vision.take() {
            let output = self
                .vision
                .finish_tensor_parallel(hidden, &mut state, tp_group, stream)?;
            context.deepstack_features = output.deepstack_features;
            let mut visual_offset = 0;
            let mut assembled = Vec::with_capacity(context.parts.len());
            for part in &context.parts {
                match part {
                    PreparedPart::Text(embedding) => assembled.push(embedding.clone()),
                    PreparedPart::Visual(len) => {
                        assembled.push(output.embeddings.try_index_device(
                            (.., visual_offset..visual_offset + *len, ..),
                            stream,
                        )?);
                        visual_offset += *len;
                    }
                }
            }
            concatenate_axis(&assembled.iter().collect::<Vec<_>>(), 1, stream)?
        } else {
            hidden.clone()
        };
        context.mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Some(mask),
            Some(AttentionMask::Causal) => {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen3-VL layerwise execution requires an explicit causal mask".into(),
                ));
            }
            None => None,
        };
        context.visual_mask = if context.deepstack_features.is_empty() {
            None
        } else {
            Some(
                context
                    .tokens
                    .eq(Array::from_int(self.args.image_token_id as i32), stream)?
                    .logical_or(
                        &context
                            .tokens
                            .eq(Array::from_int(self.args.video_token_id as i32), stream)?,
                        stream,
                    )?,
            )
        };
        Ok(hidden)
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        Ok(self
            .static_units(store)?
            .into_iter()
            .filter(|unit| select(unit.id().as_str()))
            .collect())
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Qwen3VlLayer, Error> {
        match (layout, assignment) {
            (None, None) => self.new_layer(group, index, stream),
            (Some(layout), None) => self.new_parallel_layer(group, index, layout, stream),
            (None, Some(assignment)) => {
                self.new_expert_parallel_layer(group, index, assignment, stream)
            }
            (Some(layout), Some(assignment)) => {
                self.new_tensor_expert_parallel_layer(group, index, layout, assignment, stream)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tensor_expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Qwen3VlLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &eredu_runtime::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings =
            self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)?;
        shard_layer_bindings(
            bindings,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Qwen3VlLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        match (layout, assignment) {
            (None, None) => {
                // The execution layer can have transformed target geometry
                // (for example load-time affine quantization). Bindings must
                // continue to describe the adapter's source checkpoint
                // geometry and are transformed only during population.
                let source = self.new_layer(group, index, stream)?;
                self.layer_bindings(group, index, &source, store)
            }
            (Some(layout), None) => {
                self.parallel_layer_bindings(group, index, layer, store, layout, stream)
            }
            (None, Some(assignment)) => {
                self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)
            }
            (Some(layout), Some(assignment)) => self.tensor_expert_parallel_layer_bindings(
                group, index, layer, store, layout, assignment, stream,
            ),
        }
    }
}

/// Qwen3-VL generation using shared vision/text bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Qwen3VlLayerwiseModel, Cache, S>;

#[cfg(test)]
mod neutral_runtime_tests {
    #[test]
    fn production_model_and_loaders_use_the_neutral_layerwise_runtime() {
        let source = include_str!("layerwise.rs");
        let wrapper_start = source
            .find("pub struct Qwen3VlLayerwiseModel")
            .expect("Qwen3-VL production wrapper");
        let adapter_start = source
            .find("/// Pipeline-staging legacy adapter retained")
            .expect("pipeline-only legacy adapter marker");
        let production = &source[wrapper_start..adapter_start];
        assert!(production.contains("Qwen3VlExecution"));
        for legacy in ["LayerwiseModel<", ".adapter()", ".adapter_mut()"] {
            assert!(
                !production.contains(legacy),
                "production Qwen3-VL wrapper still references {legacy}"
            );
        }
        let loaders_start = source
            .find("fn resolve_qwen3_vl_store")
            .expect("neutral Qwen3-VL loader");
        let loaders = &source[loaders_start..adapter_start];
        for legacy in [
            "load_layerwise_model(",
            "load_layerwise_model_with_quantization(",
            "load_tensor_parallel_layerwise_model(",
            "Qwen3VlLayerwiseAdapter::new(",
        ] {
            assert!(
                !loaders.contains(legacy),
                "production Qwen3-VL loaders still reference {legacy}"
            );
        }
    }
}
