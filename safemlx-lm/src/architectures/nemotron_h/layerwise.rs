//! Unified fully resident and bounded layer execution for Nemotron-H.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    api::{
        common::{self, generation::CausalLm, linear::project_logits_maybe_quantized},
        input,
        nemotron_h::{
            self as resident, BlockInput, Cache, Experts, LayerCache, LayerPolicy, ModelArgs,
            TransformerBlock,
        },
    },
    error::Error,
    nn::{
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        tensor::{create_attention_mask, AttentionMask},
    },
    runtime::cache::{
        residency::{
            PagedCacheOptions, PromptCacheDescriptor, PromptCacheManifest,
            PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
        },
        KeyValueCache,
    },
    runtime::checkpoint::binding::{
        build_module_bindings_with_recipes, canonical_checkpoint_name, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    runtime::checkpoint::recipe::DerivedWeightRecipe,
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::distributed::parallel::{
        array_parameter_member, exact_parallel_division, module_parameter_group,
        register_projection_module, register_replicated_module, MemberSharding,
        ParallelPlanBuilder, ParameterGroupSpec, ParameterRole, ProjectionSharding,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_safetensors_layerwise_model,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        LayerExecutionLoadOptions, LayerwiseForwardState, LayerwiseModel, StaticUnitBindings,
        WeightResidency,
    },
    runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry, ExpertIdentity,
        ExpertPass,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "nemotron_h.static.embedding";
const NORM_UNIT: &str = "nemotron_h.static.norm";
const HEAD_UNIT: &str = "nemotron_h.static.output";

fn register_nemotron_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    register_replicated_module(planner, &layer.norm, &format!("{prefix}.norm"))?;
    match layer.policy {
        LayerPolicy::Mamba => {
            let mamba = layer.mamba.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its Mamba mixer"
                ))
            })?;
            let intermediate = usize::try_from(mamba.intermediate_size)
                .map_err(|_| Error::Parallel("Nemotron Mamba width exceeds usize".into()))?;
            let grouped = usize::try_from(mamba.n_groups * mamba.ssm_state_size)
                .map_err(|_| Error::Parallel("Nemotron Mamba group width exceeds usize".into()))?;
            let heads = usize::try_from(mamba.num_heads)
                .map_err(|_| Error::Parallel("Nemotron Mamba heads exceed usize".into()))?;
            let in_segments = vec![
                0..intermediate,
                intermediate..2 * intermediate,
                2 * intermediate..2 * intermediate + grouped,
                2 * intermediate + grouped..2 * intermediate + 2 * grouped,
                2 * intermediate + 2 * grouped..2 * intermediate + 2 * grouped + heads,
            ];
            planner.register(module_parameter_group(
                &format!("{prefix}.mamba.in_proj"),
                ParameterRole::Segmented,
                &mamba.in_proj,
                &format!("{prefix}.mamba.in_proj"),
                |_, _| {
                    Ok(MemberSharding::Segmented {
                        axis: 0,
                        segments: in_segments.clone(),
                    })
                },
            )?)?;
            let conv_segments = vec![
                0..intermediate,
                intermediate..intermediate + grouped,
                intermediate + grouped..intermediate + 2 * grouped,
            ];
            let mut convolution = vec![array_parameter_member(
                format!("{prefix}.mamba.conv1d.weight"),
                mamba.conv1d.weight.as_ref(),
                MemberSharding::Segmented {
                    axis: 0,
                    segments: conv_segments.clone(),
                },
            )?];
            if let Some(bias) = mamba.conv1d.bias.as_ref().as_ref() {
                convolution.push(array_parameter_member(
                    format!("{prefix}.mamba.conv1d.bias"),
                    bias,
                    MemberSharding::Segmented {
                        axis: 0,
                        segments: conv_segments,
                    },
                )?);
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.mamba.conv1d"),
                ParameterRole::Channels,
                convolution,
            )?)?;
            for (name, value) in [
                ("dt_bias", mamba.dt_bias.as_ref()),
                ("A_log", mamba.A_log.as_ref()),
                ("D", mamba.D.as_ref()),
            ] {
                planner.register(ParameterGroupSpec::new(
                    format!("{prefix}.mamba.{name}"),
                    ParameterRole::Channels,
                    [array_parameter_member(
                        format!("{prefix}.mamba.{name}"),
                        value,
                        MemberSharding::Equal { axis: 0 },
                    )?],
                )?)?;
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.mamba.norm"),
                ParameterRole::Channels,
                [array_parameter_member(
                    format!("{prefix}.mamba.norm.weight"),
                    mamba.norm.weight.as_ref(),
                    MemberSharding::Equal { axis: 0 },
                )?],
            )?)?;
            register_projection_module(
                planner,
                &mamba.out_proj,
                &format!("{prefix}.mamba.out_proj"),
                ProjectionSharding::Row,
            )?;
        }
        LayerPolicy::SelfAttention(_) => {
            let attention = layer.attention.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its attention mixer"
                ))
            })?;
            for (name, projection, placement) in [
                ("q_proj", &attention.q_proj, ProjectionSharding::Column),
                ("k_proj", &attention.k_proj, ProjectionSharding::Column),
                ("v_proj", &attention.v_proj, ProjectionSharding::Column),
                ("o_proj", &attention.o_proj, ProjectionSharding::Row),
            ] {
                register_projection_module(
                    planner,
                    projection,
                    &format!("{prefix}.attention.{name}"),
                    placement,
                )?;
            }
        }
        LayerPolicy::DenseMlp => {
            let mlp = layer.mlp.as_ref().ok_or_else(|| {
                Error::Parallel(format!("Nemotron-H layer {index} is missing its dense MLP"))
            })?;
            register_projection_module(
                planner,
                &mlp.up_proj,
                &format!("{prefix}.mlp.up_proj"),
                ProjectionSharding::Column,
            )?;
            register_projection_module(
                planner,
                &mlp.down_proj,
                &format!("{prefix}.mlp.down_proj"),
                ProjectionSharding::Row,
            )?;
        }
        LayerPolicy::SparseMoe => {
            let moe = layer.moe.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its sparse MoE"
                ))
            })?;
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.moe.gate"))?;
            for (name, projection, placement) in [
                (
                    "up_proj",
                    &moe.shared_experts.up_proj,
                    ProjectionSharding::Column,
                ),
                (
                    "down_proj",
                    &moe.shared_experts.down_proj,
                    ProjectionSharding::Row,
                ),
            ] {
                register_projection_module(
                    planner,
                    projection,
                    &format!("{prefix}.moe.shared_experts.{name}"),
                    placement,
                )?;
            }
            let experts = &moe.experts;
            let mut up = vec![array_parameter_member(
                format!("{prefix}.moe.experts.up_proj"),
                experts.up_proj.as_ref(),
                MemberSharding::Equal { axis: 1 },
            )?];
            for (name, value) in [
                ("up_proj_scales", experts.up_proj_scales.as_ref().as_ref()),
                ("up_proj_biases", experts.up_proj_biases.as_ref().as_ref()),
            ] {
                if let Some(value) = value {
                    up.push(array_parameter_member(
                        format!("{prefix}.moe.experts.{name}"),
                        value,
                        MemberSharding::Equal { axis: 1 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.moe.experts.up"),
                ParameterRole::ExpertIntermediate,
                up,
            )?)?;
            let mut down = vec![array_parameter_member(
                format!("{prefix}.moe.experts.down_proj"),
                experts.down_proj.as_ref(),
                MemberSharding::Equal { axis: 2 },
            )?];
            for (name, value) in [
                (
                    "down_proj_scales",
                    experts.down_proj_scales.as_ref().as_ref(),
                ),
                (
                    "down_proj_biases",
                    experts.down_proj_biases.as_ref().as_ref(),
                ),
            ] {
                if let Some(value) = value {
                    down.push(array_parameter_member(
                        format!("{prefix}.moe.experts.{name}"),
                        value,
                        MemberSharding::Equal { axis: 2 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.moe.experts.down"),
                ParameterRole::ExpertIntermediate,
                down,
            )?)?;
        }
    }
    Ok(())
}

/// Nemotron-H causal LM using bounded residency for hybrid blocks.
pub struct NemotronHLayerwiseModel {
    execution: LayerwiseModel<NemotronHLayerwiseAdapter>,
}

impl NemotronHLayerwiseModel {
    /// Returns validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates cache/state matching the hybrid block pattern.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(&self) -> Option<&crate::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Persists a compatible prefix cache.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.execution.save_prompt_cache(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Restores a compatible prefix cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        self.execution
            .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns sparse expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
    }

    /// Backward-compatible alias for [`Self::checkpoint_store`].
    pub fn weight_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.checkpoint_store()
    }

    /// Runs the hybrid decoder while preserving KV and Mamba state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }
    /// Runs a rank-local tensor-parallel hybrid forward pass.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(inputs, cache, group, stream)
    }

    /// Runs streamed layers while delegating routed experts to a caller.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_with_layer_executor(
            inputs,
            cache,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, stream| {
                Ok(layer.forward_sparse_experts(
                    BlockInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(&mut cache.layers[index]),
                    },
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Clears temporary hybrid blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalLm<Cache> for NemotronHLayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads Nemotron-H through the generalized execution engine.
pub fn load_nemotron_h_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::NemotronH,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_nemotron_h_model_args(model_dir)?;
    let adapter = NemotronHLayerwiseAdapter::new(args, stream)?;
    Ok(NemotronHLayerwiseModel {
        execution: load_safetensors_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}
/// Loads Nemotron-H through the generalized tensor-parallel engine.
pub fn load_nemotron_h_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_nemotron_h_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::NemotronH,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    Ok(NemotronHLayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            NemotronHLayerwiseAdapter::new(
                resident::get_nemotron_h_model_args(model_dir)?,
                stream,
            )?,
            options,
            build,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_nemotron_h_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerExecutionLoadOptions,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(NemotronHLayerwiseModel, Vec<u32>), Error> {
    crate::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared =
        resident::prepare_nemotron_h_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let execution = load_tensor_parallel_layerwise_model(
        store,
        NemotronHLayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((
        NemotronHLayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

pub(crate) fn load_nemotron_h_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(NemotronHLayerwiseModel, Vec<u32>), Error> {
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(name)) if name == "nemotron_h" => {
            crate::api::GgufArchitecture::NemotronH
        }
        Some(GgufMetadataValue::String(name)) if name == "nemotron_h_moe" => {
            crate::api::GgufArchitecture::NemotronHMoe
        }
        Some(GgufMetadataValue::String(name)) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF architecture {name:?}; this loader supports nemotron_h and nemotron_h_moe"
            )));
        }
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"general.architecture\" has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata is missing required key \"general.architecture\"".into(),
            ));
        }
    };
    crate::api::structural::validate_gguf(
        architecture,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared =
        resident::prepare_nemotron_h_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let args = prepared.args;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => load_layerwise_model(
            store,
            NemotronHLayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::DenseDiskStream(options) => load_layerwise_model(
            store,
            NemotronHLayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::SparseExpertCache(options) => {
            return Ok((
                load_nemotron_h_gguf_sparse_with_store(
                    store,
                    args,
                    options,
                    options.non_expert,
                    stream,
                    weights_stream,
                )?,
                prepared.eos_token_ids,
            ));
        }
        WeightResidency::SparseExpertCacheWithDenseLayers(options) => {
            return Ok((
                load_nemotron_h_gguf_sparse_with_store(
                    store,
                    args,
                    options.expert_cache,
                    options.non_expert,
                    stream,
                    weights_stream,
                )?,
                prepared.eos_token_ids,
            ));
        }
        WeightResidency::FullyResident => {
            return Err(Error::UnsupportedArchitecture(
                "the bounded GGUF Nemotron-H loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((
        NemotronHLayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

fn load_nemotron_h_gguf_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    if !args
        .layer_schedule
        .iter()
        .any(|policy| *policy == LayerPolicy::SparseMoe)
    {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Nemotron-H MoE GGUF checkpoint".into(),
        ));
    }
    let mut adapter = NemotronHLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    let checkpoint_store = execution.weight_store_arc();
    let entries = nemotron_h_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        checkpoint_store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(NemotronHLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Nemotron-H execution base used by distributed EP.
pub(crate) fn load_nemotron_h_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let mut adapter = NemotronHLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(NemotronHLayerwiseModel { execution })
}

/// Loads Nemotron-H with expert-granular sparse caching.
pub fn load_nemotron_h_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    load_nemotron_h_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Nemotron-H with expert caching and disk-streamed non-expert units.
pub fn load_nemotron_h_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    load_nemotron_h_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

fn load_nemotron_h_sparse_expert_cache_model_with_non_expert(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::NemotronH,
        model_dir,
        crate::api::ModelLoadOptions::default()
            .with_weight_residency(WeightResidency::SparseExpertCache(options)),
    )?;
    let args = resident::get_nemotron_h_model_args(model_dir)?;
    if !args
        .layer_schedule
        .iter()
        .any(|policy| *policy == LayerPolicy::SparseMoe)
    {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Nemotron-H MoE checkpoint".into(),
        ));
    }
    let mut adapter = NemotronHLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution =
        load_safetensors_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.weight_store_arc();
    let entries = nemotron_h_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(NemotronHLayerwiseModel { execution })
}

/// Adapter shared by Nemotron-H Mamba, attention, dense, and MoE blocks.
pub struct NemotronHLayerwiseAdapter {
    args: ModelArgs,
    embeddings: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl NemotronHLayerwiseAdapter {
    /// Creates metadata-only pinned modules.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        resident::validate_model_args(&args)?;
        let embeddings = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embeddings.weight"),
            stream,
        )?;
        let norm = nn::RmsNorm::unloaded(args.hidden_size, args.norm_eps, Dtype::Float32, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
            args,
            embeddings,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_lm_head: None,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    /// Returns validated model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }

    fn recipes_for_module(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
        layer_index: Option<usize>,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let normalized = normalized_checkpoint_keys(store, &self.args)?;
        let keys = store.keys();
        let mut recipes = BTreeMap::new();

        if let Some(index) = layer_index
            .filter(|index| self.args.layer_schedule.get(*index) == Some(&LayerPolicy::SparseMoe))
        {
            let packed_prefix = format!("model.layers.{index}.moe.experts");
            if !keys.contains(&format!("{packed_prefix}.up_proj"))
                && !normalized.contains_key(&format!("{packed_prefix}.up_proj"))
            {
                let mut up = Vec::with_capacity(self.args.n_routed_experts as usize);
                let mut down = Vec::with_capacity(self.args.n_routed_experts as usize);
                for expert in 0..self.args.n_routed_experts {
                    up.push(source_for_normalized(
                        &normalized,
                        &format!("{packed_prefix}.{expert}.up_proj.weight"),
                    )?);
                    down.push(source_for_normalized(
                        &normalized,
                        &format!("{packed_prefix}.{expert}.down_proj.weight"),
                    )?);
                }
                recipes.insert(
                    "moe.experts.up_proj".into(),
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: up,
                    },
                );
                recipes.insert(
                    "moe.experts.down_proj".into(),
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: down,
                    },
                );
            }
        }

        if layer_index
            .is_some_and(|index| self.args.layer_schedule.get(index) == Some(&LayerPolicy::Mamba))
        {
            let parameters = module.parameters().flatten();
            for local_name in [
                "mamba.conv1d.weight",
                "mamba.A_log",
                "mamba.D",
                "mamba.norm.weight",
            ] {
                let source = format!("{prefix}.{local_name}");
                let Some(parameter) = parameters.get(local_name) else {
                    continue;
                };
                if !keys.contains(&source) {
                    continue;
                }
                let mut recipe = DerivedWeightRecipe::source(source, TensorSelection::Full);
                if local_name == "mamba.A_log" {
                    recipe = DerivedWeightRecipe::NegLog {
                        input: Box::new(recipe),
                    };
                }
                let expected = parameter
                    .shape()
                    .iter()
                    .map(|dimension| *dimension as usize)
                    .collect::<Vec<_>>();
                if recipe.infer(store)?.shape() != expected {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: expected,
                    };
                }
                recipes.insert(local_name.to_string(), recipe);
            }
        }

        for local_name in module.parameters().flatten().keys() {
            if recipes.contains_key(local_name.as_ref()) {
                continue;
            }
            let destination = if prefix.is_empty() {
                local_name.to_string()
            } else {
                format!("{prefix}.{local_name}")
            };
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Nemotron-H checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(recipes)
    }
}

fn normalized_checkpoint_keys(
    store: &dyn WeightStore,
    args: &ModelArgs,
) -> Result<BTreeMap<String, String>, Error> {
    let mut normalized = BTreeMap::new();
    for raw in store.keys() {
        let rewritten = resident::rewrite_nemotron_h_weight_key(&raw, args)?;
        let runtime = if let Some(rest) = rewritten.strip_prefix("model.backbone.") {
            format!("model.{rest}")
        } else if let Some(rest) = rewritten.strip_prefix("backbone.") {
            format!("model.{rest}")
        } else {
            rewritten
        };
        normalized.insert(runtime, raw);
    }
    Ok(normalized)
}

fn source_for_normalized(
    normalized: &BTreeMap<String, String>,
    runtime: &str,
) -> Result<DerivedWeightRecipe, Error> {
    normalized
        .get(runtime)
        .cloned()
        .map(|key| DerivedWeightRecipe::source(key, TensorSelection::Full))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Nemotron-H checkpoint is missing split expert tensor {runtime}"
            ))
        })
}

/// Per-forward causal mask shared by attention blocks.
pub struct NemotronHForwardContext {
    mask: Option<Array>,
}

struct OffsetOnlyCache(i32);

impl KeyValueCache for OffsetOnlyCache {
    fn offset(&self) -> i32 {
        self.0
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        _stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        Ok((keys, values))
    }
}

impl ArchitectureAdapter for NemotronHLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = TransformerBlock;
    type ForwardContext = NemotronHForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let mut local = self.args.clone();
        if let Some(topology) = topology {
            local.num_attention_heads = exact_parallel_division(
                "Nemotron-H prompt-cache attention heads",
                local.num_attention_heads,
                topology.tensor_parallel_size,
            )?;
            local.num_key_value_heads = exact_parallel_division(
                "Nemotron-H prompt-cache KV heads",
                local.num_key_value_heads,
                topology.tensor_parallel_size,
            )?;
            local.mamba_num_heads = exact_parallel_division(
                "Nemotron-H prompt-cache Mamba heads",
                local.mamba_num_heads,
                topology.tensor_parallel_size,
            )?;
            local.n_groups = exact_parallel_division(
                "Nemotron-H prompt-cache Mamba groups",
                local.n_groups,
                topology.tensor_parallel_size,
            )?;
        }
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Nemotron-H cache layer count"))?;
        Ok(PromptCacheModelIdentity {
            model_family: "nemotron_h".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                PromptCacheTopology::for_parallel_topology,
            ),
            layer_layout: resident::prompt_cache_layer_layout(&local)
                .map_err(|error| Exception::custom(error.to_string()))?,
        })
    }

    fn save_prompt_cache(
        &self,
        cache: &mut Self::Cache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let rank = descriptor.topology.cache_rank_identity();
        resident::Model::save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            rank,
            stream,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        _options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        resident::Model::load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity.clone(),
            stream,
        )
        .map_err(Into::into)
    }

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.embeddings,
                    "model.embeddings",
                    store,
                    self.recipes_for_module(&self.embeddings, "model.embeddings", store, None)?,
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm_f",
                    store,
                    self.recipes_for_module(&self.norm, "model.norm_f", store, None)?,
                )?,
            )?,
        ];
        if let Some(head) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(
                    head,
                    "lm_head",
                    store,
                    self.recipes_for_module(head, "lm_head", store, None)?,
                )?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        if let Some(v) = &mut self.parallel_embedding {
            populate_module_from_lease(v.inner_mut(), &leases[0])?;
        } else {
            populate_module_from_lease(&mut self.embeddings, &leases[0])?;
        }
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(v) = &mut self.parallel_lm_head {
            populate_module_from_lease(v.inner_mut(), &leases[2])?;
        } else if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.args.layer_schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.layer_schedule.len()
            )));
        }
        for (index, (policy, cache)) in self
            .args
            .layer_schedule
            .iter()
            .zip(&cache.layers)
            .enumerate()
        {
            let matches = match (policy, cache) {
                (LayerPolicy::Mamba, LayerCache::Mamba(_))
                | (LayerPolicy::DenseMlp, LayerCache::Mlp)
                | (LayerPolicy::SparseMoe, LayerCache::Moe) => true,
                (LayerPolicy::SelfAttention(expected), LayerCache::Attention(cache)) => {
                    *expected == cache.policy()
                }
                _ => false,
            };
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Nemotron-H cache kind does not match layer schedule at layer {index}"
                )));
            }
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embeddings.forward(input, stream)?;
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Nemotron-H requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: NemotronHForwardContext { mask },
        })
    }
    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(v) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let hidden = v.forward(input, execution)?;
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), execution.stream())? {
                Some(AttentionMask::Array(v)) => Some(v),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Nemotron-H requires an array causal mask".into(),
                    ))
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: NemotronHForwardContext { mask },
        })
    }

    fn execution_group_count(&self) -> usize {
        1
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        if group == 0 {
            Ok("text_decoder".into())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H has no execution group {group}"
            )))
        }
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        TransformerBlock::new(&self.args, index, stream)
    }
    fn register_parallel_parameters(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        planner.register(crate::nn::parallel::vocab_embedding_parameter_group(
            &self.embeddings,
            "model.embeddings",
            self.args.vocab_size as usize,
            self.args.hidden_size,
            false,
        )?)?;
        crate::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            "model.norm_f",
        )?;
        if let Some(head) = &self.lm_head {
            planner.register(crate::nn::parallel::vocab_lm_head_parameter_group(
                head,
                "lm_head",
                self.args.hidden_size,
                self.args.vocab_size as usize,
                false,
            )?)?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = TransformerBlock::new(&self.args, index, stream)?;
            register_nemotron_layer_parallel_plan(planner, &layer, index)?;
        }
        Ok(())
    }
    fn configure_parallel_static(
        &mut self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args.weight_quantization_for("model.embeddings.weight"),
            context,
            stream,
        )?);
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                self.args.hidden_size,
                self.args.vocab_size as usize,
                self.args.weight_quantization_for("lm_head.weight"),
                context,
                stream,
            )?);
        }
        Ok(())
    }
    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        let prefix = format!("model.layers.{index}.");
        let parts = layout
            .tensors()
            .filter(|(name, _)| name.starts_with(&prefix))
            .flat_map(|(_, v)| {
                v.global_shape()
                    .iter()
                    .zip(v.local_shape())
                    .filter_map(|(g, l)| (*l > 0 && g % l == 0).then_some(g / l))
            })
            .max()
            .unwrap_or(1) as i32;
        let mut args = self.args.clone();
        args.num_attention_heads /= parts;
        args.num_key_value_heads /= parts;
        args.mamba_num_heads /= parts;
        args.n_groups /= parts;
        args.intermediate_size /= parts;
        args.moe_intermediate_size /= parts;
        args.moe_shared_expert_intermediate_size /= parts;
        TransformerBlock::new(&args, index, stream)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("nemotron_h.layer.{index:05}")
    }

    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if self.sparse_expert_cache {
            Ok(populate_module_from_lease_excluding(
                layer,
                lease,
                |name| name.starts_with("moe.experts."),
            )?)
        } else {
            Ok(populate_module_from_lease(layer, lease)?)
        }
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = format!("model.layers.{index}");
        let bindings = build_module_bindings_with_recipes(
            layer,
            &prefix,
            store,
            self.recipes_for_module(layer, &prefix, store, Some(index))?,
        )?;
        Ok(if self.sparse_expert_cache {
            bindings
                .into_iter()
                .filter(|binding| !binding.name().starts_with("moe.experts."))
                .collect()
        } else {
            bindings
        })
    }
    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn WeightStore,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        crate::runtime::execution::layerwise::shard_layer_bindings(
            self.layer_bindings(group, index, &global, store)?,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
                .into_iter()
                .filter(|key| key.contains(".experts."))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.layer_count(group)?;
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Nemotron-H sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_sparse_experts(
                BlockInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(&mut cache.layers[index]),
                },
                stream,
                |flat, indices, weights, stream| {
                    expert_cache
                        .execute_routes_bounded(
                            index,
                            flat,
                            indices,
                            weights,
                            pass,
                            stream,
                            |flat, acquired, weights, stream| {
                                let started = Instant::now();
                                let prefix = format!("model.layers.{index}.moe.experts");
                                let mut bank = Experts::new(
                                    acquired.identities().len() as i32,
                                    self.args.hidden_size,
                                    self.args.moe_intermediate_size,
                                    [
                                        self.args
                                            .weight_quantization_for(&format!("{prefix}.up_proj")),
                                        self.args.weight_quantization_for(&format!(
                                            "{prefix}.down_proj"
                                        )),
                                    ],
                                    stream,
                                )?;
                                bank.up_proj = Param::new(
                                    acquired
                                        .compact_binding("up_proj", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.up_proj_scales = Param::new(
                                    acquired
                                        .optional_compact_binding("up_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.up_proj_biases = Param::new(
                                    acquired
                                        .optional_compact_binding("up_proj_biases", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj = Param::new(
                                    acquired
                                        .compact_binding("down_proj", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_scales = Param::new(
                                    acquired
                                        .optional_compact_binding("down_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_biases = Param::new(
                                    acquired
                                        .optional_compact_binding("down_proj_biases", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                expert_cache.record_compact_bank(
                                    pass,
                                    acquired.scratch_bytes(),
                                    started.elapsed(),
                                )?;
                                Ok(bank.forward(
                                    flat,
                                    acquired.compact_routes(),
                                    weights,
                                    stream,
                                )?)
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))
                },
            )?);
        }
        Ok(layer.forward(
            BlockInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(&mut cache.layers[index]),
            },
            stream,
        )?)
    }
    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
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
        self.layer_count(group)?;
        Ok(layer.forward_tensor_parallel(
            BlockInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(&mut cache.layers[index]),
            },
            tp_group,
            execution.stream(),
        )?)
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        _group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        cache.layers[index].retained_arrays()
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embeddings,
            &hidden,
            stream,
        )?)
    }
    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        let logits = match &mut self.parallel_lm_head {
            Some(head) => head.forward(&hidden, execution)?,
            None => embedding.project_logits(&hidden, execution)?,
        };
        logits.all_gather(execution)
    }
}

pub(crate) fn nemotron_h_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store, args)?;
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        if args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        let prefix = format!("model.layers.{layer}.moe.experts");
        let packed = normalized.contains_key(&format!("{prefix}.up_proj"));
        for expert in 0..args.n_routed_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            for projection in ["up_proj", "down_proj"] {
                let weight_recipe = if packed {
                    DerivedWeightRecipe::source(
                        normalized[&format!("{prefix}.{projection}")].clone(),
                        TensorSelection::Range {
                            axis: 0,
                            start: expert,
                            end: expert + 1,
                        },
                    )
                } else {
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![source_for_normalized(
                            &normalized,
                            &format!("{prefix}.{expert}.{projection}.weight"),
                        )?],
                    }
                };
                bindings.push(nemotron_recipe_binding(projection, weight_recipe, store)?);
                if packed {
                    for suffix in ["scales", "biases"] {
                        let runtime = format!("{prefix}.{projection}_{suffix}");
                        if let Some(raw) = normalized.get(&runtime) {
                            bindings.push(nemotron_recipe_binding(
                                &format!("{projection}_{suffix}"),
                                DerivedWeightRecipe::source(
                                    raw.clone(),
                                    TensorSelection::Range {
                                        axis: 0,
                                        start: expert,
                                        end: expert + 1,
                                    },
                                ),
                                store,
                            )?);
                        }
                    }
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Nemotron-H expert byte total overflowed".into())
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

fn nemotron_recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn WeightStore,
) -> Result<WeightBinding, Error> {
    let bytes = recipe.infer(store)?.byte_len();
    Ok(WeightBinding::from_recipe(name, recipe, bytes)?)
}

/// Nemotron-H token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, NemotronHLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, ones_dtype, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::{
        load_nemotron_h_layerwise_model, load_nemotron_h_sparse_expert_cache_model,
        NemotronHLayerwiseAdapter,
    };
    use crate::{
        architectures::nemotron_h::model::{
            self as resident, Cache, LayerCache, LayerPolicy, Model, ModelArgs, ModelInput,
        },
        runtime::residency::expert_cache::ExpertCacheLoadOptions,
        runtime::residency::policy::{OffloadConfig, ResidencyPolicy},
        runtime::{
            cache::KeyValueCache,
            distributed::{
                parallel::{ParallelBuildContext, ShardingPolicy},
                topology::{DeviceAssignment, ParallelTopology},
            },
            execution::layerwise::{ArchitectureAdapter, LayerwiseLoadOptions},
        },
    };

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "nemotron_h",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 4,
            "hybrid_override_pattern": "M-E*",
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "sliding_window": 3,
            "mamba_num_heads": 2,
            "mamba_head_dim": 4,
            "n_groups": 1,
            "ssm_state_size": 4,
            "conv_kernel": 3,
            "chunk_size": 2,
            "moe_intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 10,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": false,
            "torch_dtype": "float32"
        })
    }

    fn args() -> ModelArgs {
        resident::model_args_from_config_value(&config()).unwrap()
    }

    #[test]
    fn tensor_parallel_plan_shards_every_hybrid_operator() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let adapter = NemotronHLayerwiseAdapter::new(args(), execution.stream()).unwrap();
        let context = ParallelBuildContext::new(
            ParallelTopology::from_rank(2, 0, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap(),
            ShardingPolicy::Require,
        );
        let mut planner = context.planner();
        adapter
            .register_parallel_parameters(context, &mut planner, execution.stream())
            .unwrap();
        let (_, layout) = planner.finish().unwrap();
        assert_eq!(
            layout
                .tensor("model.layers.0.mamba.in_proj.weight")
                .unwrap()
                .local_shape(),
            &[13, 8]
        );
        assert_eq!(
            layout
                .tensor("model.layers.3.attention.q_proj.weight")
                .unwrap()
                .local_shape(),
            &[4, 8]
        );
        assert_eq!(
            layout
                .tensor("model.layers.2.moe.experts.up_proj")
                .unwrap()
                .local_shape(),
            &[2, 3, 8]
        );
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if name.ends_with("norm.weight") || name.as_ref() == "model.norm_f.weight"
            {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
    }

    fn public_name(runtime: &str, args: &ModelArgs) -> String {
        if let Some(rest) = runtime.strip_prefix("model.embeddings.") {
            return format!("backbone.embeddings.{rest}");
        }
        if let Some(rest) = runtime.strip_prefix("model.norm_f.") {
            return format!("backbone.norm_f.{rest}");
        }
        for index in 0..args.num_hidden_layers as usize {
            let prefix = format!("model.layers.{index}.");
            let Some(rest) = runtime.strip_prefix(&prefix) else {
                continue;
            };
            let field = match args.layer_schedule.get(index).unwrap() {
                LayerPolicy::Mamba => "mamba",
                LayerPolicy::SelfAttention(_) => "attention",
                LayerPolicy::DenseMlp => "mlp",
                LayerPolicy::SparseMoe => "moe",
            };
            if let Some(mixer_rest) = rest.strip_prefix(&format!("{field}.")) {
                return format!("backbone.layers.{index}.mixer.{mixer_rest}");
            }
            return format!("backbone.layers.{index}.{rest}");
        }
        runtime.to_string()
    }

    fn write_fixture(dir: &Path, model: &Model, stream: &Stream) {
        let params = model.parameters().flatten();
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in params {
            let runtime = crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
            if runtime.ends_with("moe.experts.up_proj") {
                let prefix = public_name(runtime.trim_end_matches(".up_proj"), &model.args);
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.up_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
            } else if runtime.ends_with("moe.experts.down_proj") {
                let prefix = public_name(runtime.trim_end_matches(".down_proj"), &model.args);
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.down_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
            } else {
                arrays.push((public_name(&runtime, &model.args), value.clone()));
            }
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config()).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 3e-5, "{left} != {right}");
        }
    }

    fn parity(depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());

        let mut resident =
            resident::load_nemotron_h_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_nemotron_h_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                let expected_offset = match expected {
                    LayerCache::Mamba(cache) => Some(cache.offset),
                    LayerCache::Attention(cache) => {
                        Some(crate::runtime::cache::KeyValueCache::offset(cache))
                    }
                    LayerCache::Mlp | LayerCache::Moe => None,
                };
                let actual_offset = match actual {
                    LayerCache::Mamba(cache) => Some(cache.offset),
                    LayerCache::Attention(cache) => {
                        Some(crate::runtime::cache::KeyValueCache::offset(cache))
                    }
                    LayerCache::Mlp | LayerCache::Moe => None,
                };
                assert_eq!(expected_offset, actual_offset);
                if let (LayerCache::Attention(expected), LayerCache::Attention(actual)) =
                    (expected, actual)
                {
                    let expected_lengths = expected
                        .retained_arrays()
                        .iter()
                        .map(|array| array.dim(-2))
                        .collect::<Vec<_>>();
                    let actual_lengths = actual
                        .retained_arrays()
                        .iter()
                        .map(|array| array.dim(-2))
                        .collect::<Vec<_>>();
                    assert_eq!(actual_lengths, expected_lengths);
                    assert!(actual_lengths.iter().all(|length| *length <= 2));
                }
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("nemotron_h.layer."))
                .collect::<Vec<_>>();
            assert!(layers.iter().all(|unit| unit.host_resident()));
            assert!(layers.iter().filter(|unit| unit.device_resident()).count() <= depth);
            assert!(report
                .units()
                .iter()
                .filter(|unit| unit.device_resident() && !layers.contains(unit))
                .all(|unit| unit.policy() == ResidencyPolicy::Pinned));
        }
    }

    #[test]
    fn nemotron_h_public_split_moe_hybrid_layerwise_parity() {
        parity(1);
        parity(2);
    }

    #[test]
    fn nemotron_h_sparse_expert_cache_prefill_and_decode_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());
        let mut resident =
            resident::load_nemotron_h_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
            1,
        )
        .unwrap();
        let mut cached = load_nemotron_h_sparse_expert_cache_model(
            dir.path(),
            options,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = cached
                .forward(&tokens, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 2);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        assert!(report.prefill.compact_banks > 1);
        crate::architectures::distributed::expert::assert_rank_owned_sparse_ep_load(
            dir.path(),
            options,
            crate::api::ModelKind::NemotronH,
            report.owned_experts / 2,
            gpu.stream(),
            cpu.stream(),
        );
    }
}
