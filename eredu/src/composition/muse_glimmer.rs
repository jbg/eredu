//! Neutral Muse-Glimmer binding to MLX storage and execution policy.

use std::{collections::HashMap, path::Path, sync::Arc};

use eredu_architectures::muse_glimmer::{
    DecoderConfig, DecoderInputPart, LayeredModel as Architecture, ModelInput, Unit, VisionInput,
};
use eredu_checkpoint::{
    store::{CheckpointSource, SharedCheckpointSource},
    WeightQuantization,
};
use eredu_nn::{ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized};
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, ExecutionGraph, ExecutionUnitLayout, LayerWeightResidency,
    LayeredArchitecture, LayeredForwardState, LayerwiseRuntime, PagedCacheOptions,
    ParallelLayeredArchitecture, ParallelModelInfo, RuntimeState, StaticUnitBindings,
    WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

use crate::backend::mlx::{
    error::Error,
    nn::{
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        shared::{MlxBackend, MlxModule, MlxNamedModule},
    },
    runtime::{
        cache::residency::{open_prompt_cache, CacheResidencyManager},
        cache::state::MlxKeyValueState,
        checkpoint::{
            binding::{
                binding_bytes, build_module_bindings, build_module_bindings_with_recipes_excluding,
                materialize_module_bindings, populate_module_from_arrays_excluding,
                populate_module_from_lease_excluding,
            },
            load::{gguf_metadata, gguf_quantization_configs, GgufTensorNames},
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitFactory,
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
type NeutralDFlash = eredu_architectures::muse_glimmer::DFlash<MlxBackend>;
pub(crate) type MuseGlimmerPipelineUnit = MlxModule<NeutralUnit>;
type Resident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralUnit>,
>;
type Bounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralUnit, UnitFactory>,
>;
type ParallelResident = LayerwiseRuntime<
    MuseGlimmerParallelComposition,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<eredu_architectures::muse_glimmer::TransformerBlock<MlxBackend>>,
>;
type ParallelBounded = LayerwiseRuntime<
    MuseGlimmerParallelComposition,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<
        eredu_architectures::muse_glimmer::TransformerBlock<MlxBackend>,
        ParallelUnitFactory,
    >,
>;

#[derive(Clone)]
struct UnitFactory {
    args: DecoderConfig,
    vision_layers: usize,
    external_experts: bool,
}

impl MlxUnitFactory<NeutralUnit> for UnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<NeutralUnit, Error> {
        if index < self.vision_layers {
            eredu_architectures::muse_glimmer::VisionBlock::new(
                &self.args.vision_config,
                index,
                stream,
            )
            .map(NeutralUnit::Vision)
        } else {
            eredu_architectures::muse_glimmer::TransformerBlock::new(
                &self.args,
                index - self.vision_layers,
                stream,
            )
            .map(NeutralUnit::Text)
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralUnit>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".mlp.experts.")
        })?;
        Ok(())
    }
}

#[derive(Clone)]
struct ParallelUnitFactory {
    args: Arc<Vec<DecoderConfig>>,
}

impl MlxUnitFactory<eredu_architectures::muse_glimmer::TransformerBlock<MlxBackend>>
    for ParallelUnitFactory
{
    fn build(
        &mut self,
        index: usize,
        stream: &Stream,
    ) -> Result<eredu_architectures::muse_glimmer::TransformerBlock<MlxBackend>, Error> {
        let args = self.args.get(index).ok_or_else(|| {
            Error::Parallel(format!(
                "parallel Muse-Glimmer layer {index} is outside {} local layouts",
                self.args.len()
            ))
        })?;
        eredu_architectures::muse_glimmer::TransformerBlock::new(args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

pub struct MuseGlimmerParallelComposition {
    architecture: NeutralArchitecture,
    embedding: MlxNamedModule<VocabParallelEmbedding>,
    output: Option<MlxNamedModule<VocabParallelLmHead>>,
    vision_blocks: Vec<eredu_architectures::muse_glimmer::VisionBlock<MlxBackend>>,
    local_args: Arc<Vec<DecoderConfig>>,
    topology: crate::backend::mlx::MlxParallelContext,
}

/// Borrowed ingress accepted by the neutral Muse-Glimmer TP composition.
pub enum MuseGlimmerParallelInput<'a> {
    Text(&'a Array),
    Prepared(ModelInput<'a, Array>),
}

impl MuseGlimmerParallelComposition {
    fn new(
        args: DecoderConfig,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let local_args = Arc::new(
            (0..args.num_hidden_layers as usize)
                .map(|index| {
                    eredu_architectures::muse_glimmer::local_decoder_config(&args, index, layout)
                        .map_err(|error| Error::Parallel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let embedding_name = "model.embed_tokens.weight";
        let output_name = "lm_head.weight";
        let embedding = MlxNamedModule::new(
            VocabParallelEmbedding::unloaded(
                args.vocab_size as usize,
                args.hidden_size,
                args.weight_quantization_for(embedding_name),
                build,
                stream,
            )?,
            ParameterSpec::trainable(embedding_name)
                .map_err(|error| Error::Parallel(error.to_string()))?,
            None,
        )?;
        let output = (!args.tie_word_embeddings)
            .then(|| {
                Ok::<_, Error>(MlxNamedModule::new(
                    VocabParallelLmHead::unloaded(
                        args.hidden_size,
                        args.vocab_size as usize,
                        args.weight_quantization_for(output_name),
                        build,
                        stream,
                    )?,
                    ParameterSpec::trainable(output_name)
                        .map_err(|error| Error::Parallel(error.to_string()))?,
                    None,
                )?)
            })
            .transpose()?;
        let vision_blocks = (0..args.vision_config.layer_count())
            .map(|index| {
                eredu_architectures::muse_glimmer::VisionBlock::new(
                    &args.vision_config,
                    index,
                    stream,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            architecture,
            embedding,
            output,
            vision_blocks,
            local_args,
            topology: build.topology(),
        })
    }

    fn execution_context<'a>(
        &self,
        group: &'a safemlx::distributed::Group,
        stream: &'a Stream,
    ) -> Result<
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'a>,
        Error,
    > {
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
            self.topology,
            group,
            stream,
        )
    }
}

impl Parameterized<Array> for MuseGlimmerParallelComposition {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        self.embedding.visit_parameters(visitor);
        let modules = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxKeyValueState,
        >>::static_modules(&self.architecture);
        modules.text.final_norm.visit_parameters(visitor);
        modules.vision.visit_parameters(visitor);
        self.vision_blocks.visit_parameters(visitor);
        if let Some(output) = &self.output {
            output.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        self.embedding.visit_parameters_mut(visitor);
        let modules = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxKeyValueState,
        >>::static_modules_mut(&mut self.architecture);
        modules.text.final_norm.visit_parameters_mut(visitor);
        modules.vision.visit_parameters_mut(visitor);
        self.vision_blocks.visit_parameters_mut(visitor);
        if let Some(output) = &mut self.output {
            output.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.embedding.set_trainable(trainable);
        let modules = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxKeyValueState,
        >>::static_modules_mut(&mut self.architecture);
        modules.text.final_norm.set_trainable(trainable);
        modules.vision.set_trainable(trainable);
        self.vision_blocks.set_trainable(trainable);
        if let Some(output) = &mut self.output {
            output.set_trainable(trainable);
        }
    }
}

impl LayeredArchitecture<MlxBackend, MlxKeyValueState> for MuseGlimmerParallelComposition {
    type Input<'a> = MuseGlimmerParallelInput<'a>;
    type StaticModules = Self;
    type Unit = eredu_architectures::muse_glimmer::TransformerBlock<MlxBackend>;
    type ForwardContext = eredu_architectures::muse_glimmer::ForwardContext<Array>;
    type RetainedContextValues<'a> = std::vec::IntoIter<&'a Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.architecture.args().model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        if group != 0 {
            return Err(Error::Parallel(format!(
                "parallel Muse-Glimmer decoder has no execution group {group}"
            )));
        }
        Ok(self.local_args.len())
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::Parallel(format!(
                "parallel Muse-Glimmer layer {index} is out of range"
            )));
        }
        Ok(format!("model.layers.{index}"))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Self::Unit, Self::Error> {
        if group != 0 {
            return Err(Error::Parallel(
                "invalid Muse-Glimmer TP execution group".into(),
            ));
        }
        eredu_architectures::muse_glimmer::TransformerBlock::new(
            self.local_args
                .get(index)
                .ok_or_else(|| Error::Parallel("missing Muse-Glimmer local args".into()))?,
            index,
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn begin_forward<'a>(
        &mut self,
        _input: Self::Input<'a>,
        _state: &mut MlxKeyValueState,
        _stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        Err(Error::Parallel(
            "parallel Muse-Glimmer composition requires a collective context".into(),
        ))
    }

    fn forward_unit(
        &mut self,
        _group: usize,
        _index: usize,
        _unit: &mut Self::Unit,
        _hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Muse-Glimmer composition requires a collective context".into(),
        ))
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        _state: &mut MlxKeyValueState,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::Parallel(format!(
                "parallel Muse-Glimmer group {group} received {} dependencies",
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }

    fn finish_forward(
        &mut self,
        _hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Muse-Glimmer composition requires a collective context".into(),
        ))
    }

    fn retained_context_values<'a>(
        &'a self,
        _forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        Vec::new().into_iter()
    }
}

impl ParallelLayeredArchitecture<MlxBackend, MlxKeyValueState> for MuseGlimmerParallelComposition {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        match input {
            MuseGlimmerParallelInput::Text(tokens) => {
                let embeddings = self.embedding.forward(tokens, &execution)?;
                self.architecture
                    .begin_parallel_text(tokens, embeddings, state, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            }
            MuseGlimmerParallelInput::Prepared(input) => {
                let embeddings = input
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        DecoderInputPart::Text(tokens) => Some(*tokens),
                        DecoderInputPart::Media(_) => None,
                    })
                    .map(|tokens| self.embedding.forward(tokens, &execution))
                    .collect::<Result<Vec<_>, _>>()?;
                self.architecture
                    .begin_parallel_input(input, embeddings, &mut self.vision_blocks, state, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            }
        }
    }

    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &Array,
        state: &mut MlxKeyValueState,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        if group_index != 0 {
            return Err(Error::Parallel(
                "invalid Muse-Glimmer TP execution group".into(),
            ));
        }
        self.architecture
            .forward_text_unit_parallel(index, unit, hidden, state, forward, group, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .architecture
            .final_parallel_hidden(hidden, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let logits = match &mut self.output {
            Some(output) => output.forward(&hidden, &execution)?,
            None => self.embedding.project_logits(&hidden, &execution)?,
        }
        .all_gather(&execution)?;
        self.architecture
            .finish_parallel_logits(logits, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

enum Execution {
    Resident(Resident),
    Bounded(Bounded),
    ParallelResident(Box<ParallelResident>),
    ParallelBounded(Box<ParallelBounded>),
}

#[allow(clippy::too_many_arguments)]
fn forward_external_experts<P>(
    architecture: &mut NeutralArchitecture,
    group: usize,
    index: usize,
    unit: &mut NeutralUnit,
    hidden: &Array,
    state: &mut MlxKeyValueState,
    forward: &mut eredu_architectures::muse_glimmer::ForwardContext<Array>,
    stream: &Stream,
    provider: &mut P,
) -> Result<Array, eredu_nn::Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
    P::Error: std::fmt::Display,
{
    if group == 1 {
        let NeutralUnit::Text(block) = unit else {
            return Err(eredu_nn::Error::backend(
                "Muse-Glimmer text execution group received a vision unit",
            ));
        };
        return architecture.forward_text_unit_with_provider(
            index,
            block,
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
        );
    }
    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
}

/// One family object shared by resident and bounded execution.
pub struct MuseGlimmerModel {
    args: DecoderConfig,
    state_layout: eredu_runtime::StateLayout,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    execution: Execution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
}

/// Fully resident DFlash assistant built from neutral equations.
pub(crate) struct MuseGlimmerDFlashModel {
    pub(crate) config: eredu_architectures::muse_glimmer::DFlashConfig,
    module: MlxModule<NeutralDFlash>,
}

impl MuseGlimmerDFlashModel {
    pub(crate) fn target_layer_ids(&self) -> &[usize] {
        self.module.target_layer_ids()
    }

    pub(crate) fn assemble_target_states(
        &self,
        states: &[Array],
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.module
            .assemble_target_states(states, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub(crate) fn update_context(
        &mut self,
        previous: Option<eredu_architectures::muse_glimmer::DFlashContext<Array>>,
        states: &Array,
        absolute_end: i32,
        stream: &Stream,
    ) -> Result<eredu_architectures::muse_glimmer::DFlashContext<Array>, Error> {
        self.module
            .update_context(previous, states, absolute_end, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub(crate) fn proposal_states(
        &mut self,
        embeddings: &Array,
        committed: &eredu_architectures::muse_glimmer::DFlashContext<Array>,
        absolute_end: i32,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.module
            .proposal_states(embeddings, committed, absolute_end, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

pub(crate) fn load_dflash_safetensors(
    model_dir: &Path,
    options: crate::backend::mlx::ModelLoadOptions,
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

pub(crate) fn load_dflash_gguf(
    gguf_file: &Path,
    options: crate::backend::mlx::ModelLoadOptions,
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

pub(crate) struct MuseGlimmerMtpOutput {
    pub(crate) logits: Array,
    pub(crate) target_states: Vec<Array>,
}

struct PreparedMuseInput {
    tokens: Vec<Array>,
    media: Vec<bool>,
    pixels: Option<Array>,
    grid: Vec<(i32, i32, i32)>,
}

/// Transportable neutral ingress state used while a pipeline placement walks
/// the native vision group. The architecture forward context is the same one
/// used by resident and bounded execution; only ownership of its tensors moves.
pub(crate) struct MuseGlimmerPipelineIngressState {
    forward: LayeredForwardState<Array, eredu_architectures::muse_glimmer::ForwardContext<Array>>,
    state: MlxKeyValueState,
}

impl MuseGlimmerPipelineIngressState {
    pub(crate) fn hidden(&self) -> &Array {
        &self.forward.hidden
    }

    pub(crate) fn replace_hidden(&mut self, hidden: Array) {
        self.forward.hidden = hidden;
    }
}

/// Pipeline/loading binder over the same neutral Muse-Glimmer model and units
/// used by every other residency policy.
pub(crate) struct MuseGlimmerPipelineAdapter {
    args: DecoderConfig,
    architecture: NeutralArchitecture,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    external_experts: bool,
}

impl MuseGlimmerPipelineAdapter {
    pub(crate) fn new(args: DecoderConfig, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            architecture: NeutralArchitecture::new(args.clone(), stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            args,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            external_experts: false,
        })
    }

    pub(crate) fn new_external_experts(
        args: DecoderConfig,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.external_experts = true;
        Ok(adapter)
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn static_modules(
        &self,
    ) -> &eredu_architectures::muse_glimmer::model::StaticModules<MlxBackend> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::static_modules(
            &self.architecture,
        )
    }

    fn static_modules_mut(
        &mut self,
    ) -> &mut eredu_architectures::muse_glimmer::model::StaticModules<MlxBackend> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::static_modules_mut(
            &mut self.architecture,
        )
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let modules = self.static_modules();
        let mut units = Vec::new();
        macro_rules! push_leaf {
            ($role:literal, $module:expr, $prefix:literal, $packed:expr) => {
                if select(concat!("muse_glimmer.static.", $role)) {
                    let module = MlxModule::new($module.clone());
                    let recipes = crate::composition::muse_glimmer_expert::module_recipes(
                        &module,
                        &self.args,
                        store,
                    )?;
                    let prefix = concat!($prefix, ".");
                    let bindings = build_module_bindings_with_recipes_excluding(
                        &module,
                        "",
                        store,
                        recipes,
                        |_| false,
                    )?
                    .into_iter()
                    .map(|binding| {
                        let local = binding
                            .name()
                            .strip_prefix(prefix)
                            .ok_or_else(|| {
                                Error::Parallel(format!(
                                    "Muse-Glimmer static binding {:?} does not start with {prefix:?}",
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
                        concat!("muse_glimmer.static.", $role),
                        bindings,
                    )?);
                }
            };
        }
        macro_rules! push {
            ($role:literal, $module:expr) => {
                if select(concat!("muse_glimmer.static.", $role)) {
                    let module = MlxModule::new($module.clone());
                    let recipes = crate::composition::muse_glimmer_expert::module_recipes(
                        &module, &self.args, store,
                    )?;
                    units.push(StaticUnitBindings::new(
                        concat!("muse_glimmer.static.", $role),
                        build_module_bindings_with_recipes_excluding(
                            &module,
                            "",
                            store,
                            recipes,
                            |_| false,
                        )?,
                    )?);
                }
            };
        }
        push!("vision", modules.vision);
        push_leaf!(
            "embedding",
            modules.text.embeddings,
            "model.embed_tokens",
            self.args
                .weight_quantization_for("model.embed_tokens.weight")
                .is_some()
        );
        push_leaf!("norm", modules.text.final_norm, "model.norm", false);
        if let Some(head) = &modules.text.head {
            push_leaf!(
                "output",
                head,
                "lm_head",
                self.args
                    .weight_quantization_for("lm_head.weight")
                    .is_some()
            );
        }
        Ok(units)
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

    pub(crate) fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.args.vision_config.layer_count()),
            1 => Ok(self.args.num_hidden_layers as usize),
            _ => Err(Error::Parallel(format!(
                "Muse-Glimmer has no execution group {group}"
            ))),
        }
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelineUnit, Error> {
        self.new_cartesian_layer(group, index, None, None, stream)
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        _build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        _stream: &Stream,
    ) -> Result<(), Error> {
        for group in eredu_architectures::muse_glimmer::static_parameter_groups(&self.args)? {
            planner.register(group)?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            for group in
                eredu_architectures::muse_glimmer::layer_parameter_groups(&self.args, index)?
            {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn configure_parallel_static(
        &mut self,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_layout = Some(_layout.clone());
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args
                .weight_quantization_for("model.embed_tokens.weight"),
            build,
            stream,
        )?);
        self.parallel_lm_head = (!self.args.tie_word_embeddings)
            .then(|| {
                VocabParallelLmHead::unloaded(
                    self.args.hidden_size,
                    self.args.vocab_size as usize,
                    self.args.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        Ok(())
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelineUnit, Error> {
        let unit = match group {
            0 => NeutralUnit::Vision(
                eredu_architectures::muse_glimmer::VisionBlock::new(
                    &self.args.vision_config,
                    index,
                    stream,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            ),
            1 => {
                let args = match layout {
                    Some(layout) => eredu_architectures::muse_glimmer::local_decoder_config(
                        &self.args, index, layout,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                    None => self.args.clone(),
                };
                NeutralUnit::Text(
                    eredu_architectures::muse_glimmer::TransformerBlock::new(&args, index, stream)
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                )
            }
            _ => {
                return Err(Error::Parallel(format!(
                    "Muse-Glimmer has no execution group {group}"
                )))
            }
        };
        Ok(MlxModule::new(unit))
    }

    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &MuseGlimmerPipelineUnit,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_cartesian_layer(group, index, None, None, stream)?;
        let recipes =
            crate::composition::muse_glimmer_expert::module_recipes(&global, &self.args, store)?;
        let bindings =
            build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |name| {
                self.external_experts && name.contains(".mlp.experts.")
            })?;
        match layout {
            Some(layout) => {
                let prefix = match group {
                    0 => format!("model.vision_tower.layers.{index}"),
                    1 => format!("model.layers.{index}"),
                    _ => unreachable!(),
                };
                shard_layer_bindings(bindings, &prefix, store, layout)
            }
            None => Ok(bindings),
        }
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        _index: usize,
        layer: &MuseGlimmerPipelineUnit,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.layer_count(group)?;
        let recipes =
            crate::composition::muse_glimmer_expert::module_recipes(layer, &self.args, store)?;
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && name.contains(".mlp.experts.")
        })
        .map_err(Into::into)
    }

    pub(crate) fn vision_module_mut(
        &mut self,
    ) -> crate::backend::mlx::nn::shared::MlxModuleRef<
        '_,
        eredu_architectures::muse_glimmer::VisionStatic<MlxBackend>,
    > {
        crate::backend::mlx::nn::shared::MlxModuleRef::new(&mut self.static_modules_mut().vision)
    }

    pub(crate) fn embedding_mut(&mut self) -> &mut crate::backend::mlx::nn::shared::MlxEmbedding {
        &mut self.static_modules_mut().text.embeddings
    }

    pub(crate) fn parallel_embedding_mut(&mut self) -> Option<&mut VocabParallelEmbedding> {
        self.parallel_embedding.as_mut()
    }

    pub(crate) fn norm_mut(&mut self) -> &mut crate::backend::mlx::nn::shared::MlxRmsNorm {
        &mut self.static_modules_mut().text.final_norm
    }

    pub(crate) fn lm_head_mut(
        &mut self,
    ) -> Option<&mut crate::backend::mlx::nn::shared::MlxLinear> {
        self.static_modules_mut().text.head.as_mut()
    }

    pub(crate) fn parallel_lm_head_mut(&mut self) -> Option<&mut VocabParallelLmHead> {
        self.parallel_lm_head.as_mut()
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let state_args = match &self.parallel_layout {
            Some(layout) => {
                eredu_architectures::muse_glimmer::local_decoder_config(&self.args, 0, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?
            }
            None => self.args.clone(),
        };
        let layout = eredu_architectures::muse_glimmer::state_layout(&state_args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut identity = eredu_runtime::ModelStateIdentity {
            model_family: "muse_glimmer".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: self.args.architecture_fingerprint(),
            layer_count: layout.len(),
            global_layer_start: 0,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layout.len()],
            topology: Default::default(),
        }
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
        if let Some(topology) = topology {
            identity.topology = crate::backend::mlx::cache::prompt_cache_topology(topology);
        }
        Ok(identity)
    }

    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        typed: input::ModelInput<'_>,
        _execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelineIngressState, Error> {
        let prepared = prepare_muse_input(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(&prepared.media)
            .map(|(tokens, media)| {
                if *media {
                    DecoderInputPart::Media(tokens)
                } else {
                    DecoderInputPart::Text(tokens)
                }
            })
            .collect::<Vec<_>>();
        let mut state = MlxKeyValueState::device(
            eredu_architectures::muse_glimmer::state_layout(&self.args)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        )?;
        let forward = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxKeyValueState,
        >>::begin_forward(
            &mut self.architecture,
            ModelInput {
                parts: &parts,
                vision: prepared.pixels.as_ref().map(|pixels| VisionInput {
                    pixels,
                    grid: &prepared.grid,
                }),
                mask: None,
            },
            &mut state,
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(MuseGlimmerPipelineIngressState { forward, state })
    }

    pub(crate) fn begin_pipeline_continuation(
        &mut self,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelineIngressState, Error> {
        self.begin_pipeline_ingress(typed, None, stream)
    }

    pub(crate) fn pipeline_ingress_active(&self, state: &MuseGlimmerPipelineIngressState) -> bool {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::should_execute_group(
            &self.architecture,
            0,
            &state.forward.context,
        )
    }

    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &MuseGlimmerPipelineIngressState,
    ) -> Vec<Array> {
        vec![state.hidden().clone()]
    }

    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut MuseGlimmerPipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Muse-Glimmer placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.replace_hidden(hidden);
        Ok(())
    }

    pub(crate) fn forward_pipeline_vision_layer(
        &mut self,
        index: usize,
        layer: &mut MuseGlimmerPipelineUnit,
        state: &mut MuseGlimmerPipelineIngressState,
        _execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<(), Error> {
        state.forward.hidden = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxKeyValueState,
        >>::forward_unit(
            &mut self.architecture,
            0,
            index,
            &mut **layer,
            &state.forward.hidden,
            &mut state.state,
            &mut state.forward.context,
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(())
    }

    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: MuseGlimmerPipelineIngressState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if self.pipeline_ingress_active(&state) {
            state.forward.hidden = <NeutralArchitecture as LayeredArchitecture<
                MlxBackend,
                MlxKeyValueState,
            >>::complete_execution_group(
                &mut self.architecture,
                0,
                &state.forward.hidden,
                &mut state.state,
                &mut state.forward.context,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        }
        Ok(state.forward.hidden)
    }

    pub(crate) fn prepare_pipeline_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        vision_layers: &mut [MuseGlimmerPipelineUnit],
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut state = self.begin_pipeline_ingress(typed, execution, stream)?;
        if self.pipeline_ingress_active(&state) {
            if vision_layers.len() != self.args.vision_config.layer_count() {
                return Err(Error::Parallel(format!(
                    "Muse-Glimmer local ingress owns {} vision blocks, expected {}",
                    vision_layers.len(),
                    self.args.vision_config.layer_count()
                )));
            }
            for (index, layer) in vision_layers.iter_mut().enumerate() {
                self.forward_pipeline_vision_layer(index, layer, &mut state, execution, stream)?;
            }
        }
        self.finish_pipeline_ingress(state, stream)
    }

    pub(crate) fn prepare_pipeline_tokens(
        &mut self,
        tokens: &Array,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (self.parallel_embedding.as_mut(), execution) {
            (Some(embedding), Some(execution)) => {
                let hidden = embedding.forward(tokens, execution)?;
                self.static_modules()
                    .text
                    .normalize_embeddings(&hidden, execution.stream())
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            }
            _ => self
                .architecture
                .token_embeddings(tokens, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        }
    }

    pub(crate) fn finish_pipeline_text(
        &mut self,
        hidden: &Array,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let (Some(embedding), Some(execution)) = (self.parallel_embedding.as_mut(), execution) {
            let hidden = self
                .architecture
                .final_parallel_hidden(hidden, execution.stream())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let sharded = match self.parallel_lm_head.as_mut() {
                Some(head) => head.forward(&hidden, execution)?,
                None => embedding.project_logits(&hidden, execution)?,
            };
            let logits = sharded.all_gather(execution)?;
            return self
                .architecture
                .finish_parallel_logits(logits, execution.stream())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
        }
        self.architecture
            .project_logits(hidden, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

fn prepare_muse_input(
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
                tokens.push(value.clone());
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
                tokens.push(Array::from_slice(&vec![id; count as usize], &[1, count]));
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
        pixels,
        grid,
    })
}

impl MuseGlimmerModel {
    pub(crate) const fn args(&self) -> &DecoderConfig {
        &self.args
    }

    /// Returns canonical parameter/residency metadata.
    pub fn metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    pub(crate) fn new_cache(&self) -> MlxKeyValueState {
        MlxKeyValueState::device(self.state_layout.clone())
            .expect("validated neutral state must be realizable")
    }

    pub(crate) fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxKeyValueState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let rank = self.parallel_info.as_ref().and_then(|info| {
                    crate::backend::mlx::cache::prompt_cache_topology(info.topology())
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

    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
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
            model_family: "muse_glimmer".into(),
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

    pub(crate) fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(MlxKeyValueState, eredu_core::cache::PromptCacheManifest), Error> {
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
        let state = MlxKeyValueState::paged(self.state_layout.clone(), manager, rank)?;
        Ok((state, manifest))
    }

    pub(crate) fn save_prompt_cache(
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
            &self.prompt_identity()?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        state
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    pub(crate) fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        let report = match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report()?,
            Execution::Bounded(runtime) => runtime.policy().residency_report()?,
            Execution::ParallelResident(runtime) => runtime.policy().residency_report()?,
            Execution::ParallelBounded(runtime) => runtime.policy().residency_report()?,
        };
        Ok(Some(report))
    }

    pub(crate) fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) | Execution::ParallelResident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
            Execution::ParallelBounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    pub(crate) fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
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
        input: ModelInput<'_, Array>,
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
                            if group == 1
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
                            if group == 1
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
                    if group == 1 && capture.as_ref().is_some_and(|capture| capture.wants(index)) {
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
                    if group == 1 && capture.as_ref().is_some_and(|capture| capture.wants(index)) {
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
        input: ModelInput<'_, Array>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_with_taps(input, state, &[], stream)
            .map(|output| output.logits)
    }

    pub(crate) fn embed_dflash_tokens(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
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

    pub(crate) fn project_dflash_logits(
        &mut self,
        hidden: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
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

    pub(crate) fn forward_tokens(
        &mut self,
        tokens: &Array,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Error> {
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

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Muse-Glimmer tensor-parallel cache layout mismatch".into(),
            ));
        }
        match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel(MuseGlimmerParallelInput::Text(tokens), state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel(MuseGlimmerParallelInput::Text(tokens), state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::Resident(_) | Execution::Bounded(_) => Err(Error::Parallel(
                "Muse-Glimmer model was not loaded for tensor parallelism".into(),
            )),
        }
    }

    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
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
        let input = MuseGlimmerParallelInput::Prepared(ModelInput {
            parts: &parts,
            vision: prepared.pixels.as_ref().map(|pixels| VisionInput {
                pixels,
                grid: &prepared.grid,
            }),
            mask: None,
        });
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

    pub(crate) fn verify_dflash(
        &mut self,
        tokens: &Array,
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

    pub(crate) fn forward_input(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_input_with_taps(typed, state, &[], stream)
            .map(|output| output.logits)
    }

    pub(crate) fn forward_input_with_taps(
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
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_input(input, state, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        tokens: &Array,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_tokens(tokens, state, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

fn layout(args: &DecoderConfig) -> Result<ExecutionUnitLayout, Error> {
    let graph = eredu_runtime::ExecutionGraph::chain(["vision", "text_decoder"])
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    ExecutionUnitLayout::new(
        &graph,
        [
            args.vision_config.layer_count(),
            args.num_hidden_layers as usize,
        ],
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
    let mut source_factory = UnitFactory {
        args: source.clone(),
        vision_layers: source.vision_config.layer_count(),
        external_experts: false,
    };
    let mut target_factory = UnitFactory {
        args: target.clone(),
        vision_layers: target.vision_config.layer_count(),
        external_experts: false,
    };
    let unit_count = source_factory
        .vision_layers
        .checked_add(source.num_hidden_layers as usize)
        .ok_or_else(|| Error::Quantization("Muse-Glimmer unit count overflowed".into()))?;
    let (store, report) = quantize_parameterized_store(
        store,
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::static_modules(
            &source_architecture,
        ),
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::static_modules(
            &target_architecture,
        ),
        move |index, stream| source_factory.build(index, stream),
        move |index, stream| target_factory.build(index, stream),
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
    let vision_layers = args.vision_config.layer_count();
    let static_args = args.clone();
    let unit_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::static_modules_mut(
            &mut architecture,
        ),
        UnitFactory {
            args: args.clone(),
            vision_layers,
            external_experts,
        },
        layout(&args)?,
        residency,
        stream,
        weights_stream,
        move |key| external_experts && key.contains(".mlp.experts."),
        move |modules, store| {
            let module = MlxModule::new(modules.clone());
            let recipes = crate::composition::muse_glimmer_expert::module_recipes(
                &module,
                &static_args,
                store,
            )?;
            build_module_bindings_with_recipes_excluding(
                &module,
                "",
                store,
                recipes,
                |_| false,
            )
            .map_err(Into::into)
        },
        move |_ordinal, unit, store, _stream| {
            let module = MlxModule::new(unit);
            let recipes = crate::composition::muse_glimmer_expert::module_recipes(
                &module,
                &unit_args,
                store,
            )?;
            build_module_bindings_with_recipes_excluding(
                &module,
                "",
                store,
                recipes,
                |name| external_experts && name.contains(".mlp.experts."),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization.or(args.quantization_config));
    metadata.set_materialization(materialization);
    let execution = if residency.is_fully_resident() {
        Execution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
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
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerModel, Error> {
    let layer_count = args.num_hidden_layers as usize;
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
    let mut composition =
        MuseGlimmerParallelComposition::new(args.clone(), build, &layout, stream)?;
    let local_config = composition
        .local_args
        .first()
        .ok_or_else(|| Error::Parallel("Muse-Glimmer has no decoder layers".into()))?;
    let state_layout = eredu_architectures::muse_glimmer::state_layout(local_config)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_args = Arc::clone(&composition.local_args);
    let unit_layout = ExecutionUnitLayout::new(
        &ExecutionGraph::chain(["text_decoder"])
            .map_err(|error| Error::Parallel(error.to_string()))?,
        [layer_count],
    )
    .map_err(|error| Error::Parallel(error.to_string()))?;

    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let global_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::static_modules(
            &global_architecture,
        )
        .clone(),
    );
    let static_recipes = crate::composition::muse_glimmer_expert::module_recipes(
        &global_static,
        &args,
        store.as_ref(),
    )?;
    let mut global_static_bindings = build_module_bindings_with_recipes_excluding(
        &global_static,
        "",
        store.as_ref(),
        static_recipes,
        |_| false,
    )?;
    for index in 0..args.vision_config.layer_count() {
        let block = MlxModule::new(
            eredu_architectures::muse_glimmer::VisionBlock::<MlxBackend>::new(
                &args.vision_config,
                index,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        );
        let recipes =
            crate::composition::muse_glimmer_expert::module_recipes(&block, &args, store.as_ref())?;
        global_static_bindings.extend(build_module_bindings_with_recipes_excluding(
            &block,
            "",
            store.as_ref(),
            recipes,
            |_| false,
        )?);
    }
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for index in 0..layer_count {
        let unit = MlxModule::new(
            eredu_architectures::muse_glimmer::TransformerBlock::<MlxBackend>::new(
                &args, index, stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        );
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
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut composition,
        ParallelUnitFactory { args: local_args },
        unit_layout,
        residency,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| {
            shard_layer_bindings(global_static_bindings, "", store, &static_layout)
        },
        move |index, _local, store, stream| {
            let global = MlxModule::new(
                eredu_architectures::muse_glimmer::TransformerBlock::<MlxBackend>::new(
                    &binding_args,
                    index,
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
            shard_layer_bindings(
                bindings,
                &format!("model.layers.{index}"),
                store,
                &unit_sharding,
            )
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
        Execution::ParallelResident(Box::new(LayerwiseRuntime::new(
            composition,
            policy.into_resident(stream)?,
        )))
    } else {
        Execution::ParallelBounded(Box::new(LayerwiseRuntime::new(composition, policy)))
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

pub(crate) fn load_safetensors_tensor_parallel(
    model_dir: impl AsRef<Path>,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
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

pub(crate) fn load_gguf_tensor_parallel(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
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
    let entries = crate::composition::muse_glimmer_expert::expert_catalog(
        &model.args,
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

/// Loads SafeTensors through one neutral family model and one residency policy.
pub(crate) fn load_safetensors(
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
pub(crate) fn load_gguf(
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

pub(crate) fn load_pipeline_config(model_dir: &Path) -> Result<DecoderConfig, Error> {
    DecoderConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn prepare_gguf_pipeline_source(
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
