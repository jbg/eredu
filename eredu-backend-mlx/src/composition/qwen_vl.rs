// MLX artifact and residency binding for the neutral Qwen3-VL graph.

use std::{collections::BTreeSet, path::Path, sync::Arc};

use eredu_architectures::qwen::vl;
use eredu_checkpoint::{
    store::{CheckpointSource, CompositeCheckpointSource},
    WeightQuantization,
};
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, ExecutionUnitLayout,
    LayerWeightResidency, LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions, ParameterRole,
    ResidencyReport, WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

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

fn native_input_parts<'a>(
    parts: &'a [vl::InputPart<'a, crate::MlxTensor>],
) -> Vec<vl::InputPart<'a, Array>> {
    parts
        .iter()
        .map(|part| match part {
            vl::InputPart::Text(tokens) => vl::InputPart::Text(tokens.as_array()),
            vl::InputPart::Image { tokens, grid } => vl::InputPart::Image {
                tokens: tokens.as_array(),
                grid,
            },
            vl::InputPart::Video { tokens, grid } => vl::InputPart::Video {
                tokens: tokens.as_array(),
                grid,
            },
            vl::InputPart::Projected { tokens, embeddings } => vl::InputPart::Projected {
                tokens: tokens.as_array(),
                embeddings: embeddings.as_array(),
            },
        })
        .collect()
}

fn prepare_model_input(
    args: &vl::ModelArgs,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<vl::PreparedInput<crate::MlxTensor>, safemlx::error::Exception> {
    let prepared = crate::composition::mlx::replicated_text::prepared_composite_input(input)
        .map_err(|error| safemlx::error::Exception::custom(error.to_string()))?;
    let admitted = eredu_architectures::media_plan::admit_qwen_vl_input(
        args,
        &prepared,
        &input::MlxTensorInputInspector,
    )
    .map_err(|error| safemlx::error::Exception::custom(error.to_string()))?;
    let input =
        eredu_architectures::composite_execution::PreparedCompositeInput::new(&prepared, &admitted)
            .map_err(safemlx::error::Exception::custom)?;
    vl::prepare_input(input, stream)
        .map_err(|error| safemlx::error::Exception::custom(error.to_string()))
}

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
        checkpoint::binding::{
            build_module_bindings_with_recipes, build_module_bindings_with_recipes_excluding,
            parameter_name_in_targets, populate_module_from_lease_excluding,
        },
        checkpoint::{
            load::gguf_quantization_configs, quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::{
            generic::{
                construct_architecture_unit, prepare_layerwise_policy_with_bindings,
                MlxLayerwisePolicy, MlxResidentPolicy, MlxUnitPopulator,
            },
            layerwise::quantize_parameterized_store,
        },
        media::input,
        residency::parameter_bank::AddressableParameterBank,
    },
};

type Architecture = vl::LayeredModel<MlxNeuralBackend>;
type Unit = vl::Unit<MlxNeuralBackend>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(test)]
pub struct QwenVlCheckpointTemplate {
    pub static_modules: vl::StaticModules<MlxNeuralBackend>,
    pub units: Vec<Unit>,
}

#[cfg(test)]
impl QwenVlCheckpointTemplate {
    pub fn new(args: vl::ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = Architecture::new(args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let static_modules = <Architecture as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::static_modules(&architecture)
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

#[derive(Clone)]
struct UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<Unit> for UnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<Unit>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

type Resident =
    LayerwiseRuntime<Architecture, MlxNeuralBackend, MlxHybridState, MlxResidentPolicy<Unit>>;
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

impl Execution {
    fn architecture(&self) -> &Architecture {
        match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Bounded(runtime) => runtime.architecture(),
        }
    }
}

/// Neutral Qwen3-VL dense-or-MoE model bound to MLX storage policy.
pub struct QwenVlModel {
    args: vl::ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    execution: Execution,
    parameter_bank: Option<AddressableParameterBank>,
}

impl QwenVlModel {
    pub fn effective_model_type(&self) -> &str {
        self.args.effective_model_type()
    }

    pub fn args(&self) -> &vl::ModelArgs {
        &self.args
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

    pub(crate) fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        crate::composition::replicated_prompt_cache_identity(
            self.execution.architecture(),
            eredu_core::cache::PromptCacheTopology::default(),
        )
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
            identity.layer_prefix_offsets(),
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

    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(self
            .parameter_bank
            .as_ref()
            .map(AddressableParameterBank::report)
            .transpose()?)
    }

    fn forward(
        &mut self,
        input: vl::ModelInput<'_, Array>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let args = self.args.text.clone();
            let result = {
                let mut provider =
                    crate::composition::qwen::expert::cached_provider(&parameter_bank, &args);
                self.forward_with_provider(input, cache, &mut provider, stream)
            };
            self.parameter_bank = Some(parameter_bank);
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
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
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
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn forward_with_observer(
        &mut self,
        input: vl::ModelInput<'_, Array>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let parts = neutral_input_parts(input.parts);
        let input = vl::ModelInput {
            parts: &parts,
            pixels: crate::composition::tensor_opt(input.pixels),
            mask: crate::composition::tensor_opt(input.mask),
        };
        let parameter_bank = self.parameter_bank.take();
        let result = {
            let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
            match parameter_bank.as_ref() {
                Some(parameter_bank) => {
                    let args = self.args.text.clone();
                    let mut provider =
                        crate::composition::qwen::expert::cached_provider(parameter_bank, &args);
                    match &mut self.execution {
                        Execution::Resident(runtime) => runtime
                            .forward_with_inferred_provider_and_observer(
                                input,
                                cache,
                                &mut provider,
                                stream,
                                &mut neutral,
                            ),
                        Execution::Bounded(runtime) => runtime
                            .forward_with_inferred_provider_and_observer(
                                input,
                                cache,
                                &mut provider,
                                stream,
                                &mut neutral,
                            ),
                    }
                }
                None => match &mut self.execution {
                    Execution::Resident(runtime) => {
                        runtime.forward_with_observer(input, cache, stream, &mut neutral)
                    }
                    Execution::Bounded(runtime) => {
                        runtime.forward_with_observer(input, cache, stream, &mut neutral)
                    }
                },
            }
        };
        self.parameter_bank = parameter_bank;
        let logits = result
            .map(crate::MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        eredu_runtime::observe_model_logits(observer, &logits).map_err(Error::from)
    }

    fn prepared_forward(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<Array, Exception> {
        prepare_model_input(&self.args, input, stream)?.with_model_input(|input| {
            let parts = native_input_parts(input.parts);
            let input = vl::ModelInput {
                parts: &parts,
                pixels: input.pixels.map(crate::MlxTensor::as_array),
                mask: input.mask.map(crate::MlxTensor::as_array),
            };
            match observer {
                Some(observer) => self.forward_with_observer(input, cache, stream, observer),
                None => self.forward(input, cache, stream),
            }
            .map_err(|error| Exception::custom(error.to_string()))
        })
    }

    pub fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Exception> {
        self.prepared_forward(input, cache, stream, Some(observer))
    }

    pub fn forward_tokens_with_observer(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let parts = [vl::InputPart::Text(tokens)];
        self.forward_with_observer(
            vl::ModelInput {
                parts: &parts,
                pixels: None,
                mask: None,
            },
            cache,
            stream,
            observer,
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
        self.prepared_forward(input, cache, stream, None)?
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
    let graph =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::execution_graph(
            architecture,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_unit_count(
                architecture,
                group,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
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
    let target =
        vl::load_time_quantization(source, quantization).map_err(Error::ArchitectureModel)?;
    let source_architecture = Architecture::new(source.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_architecture = Architecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = unit_layout(&source_architecture)?;
    let target_layout = unit_layout(&target_architecture)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Qwen3-VL quantization changed the architecture execution layout".into(),
        ));
    }
    let total = source_layout.len();
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
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: &crate::composition::mlx::structural::AdmittedGgufProjector,
    max_cached_shards: usize,
) -> Result<(vl::ModelArgs, Arc<dyn CheckpointSource>), Error> {
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::Qwen(_) = source.model() else {
        return Err(Error::ArchitectureModel(
            "Qwen3-VL GGUF loader received a different prepared model".into(),
        ));
    };
    let eredu_architectures::gguf_companion::GgufMediaProjectorConfig::Qwen3Vl(args) =
        projector.model()
    else {
        return Err(Error::ArchitectureModel(
            "Qwen3-VL GGUF loader received a mismatched media-projector plan".into(),
        ));
    };
    let args = args.clone();
    let text_formats = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let vision_formats =
        gguf_quantization_configs(projector.checkpoint(), projector.plan().tensor_mapping())?;
    let args = vl::with_checkpoint_formats(&args, text_formats, vision_formats)
        .map_err(Error::ArchitectureModel)?;
    let text_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        source.plan().tensor_mapping(),
        max_cached_shards,
    )?);
    let vision_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        projector.checkpoint().clone(),
        projector.plan().checkpoint(),
        projector.plan().tensor_mapping(),
        max_cached_shards,
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
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: &crate::composition::mlx::structural::AdmittedGgufProjector,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenVlModel, Error> {
    let expert_options = residency.parameter_bank_cache();
    let options = residency.layers();
    let (mut args, store) = prepare_gguf_pipeline(source, projector, options.max_cached_shards())?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen3-VL GGUF", None, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = if let Some(quantization) = quantize_on_load {
        let (store, target, _) = quantize_store(store, &args, quantization, stream)?;
        args = target;
        store
    } else {
        store
    };
    let mut model = load_store(
        store,
        args,
        options,
        expert_options.is_some(),
        stream,
        weights_stream,
    )?;
    if let Some(options) = expert_options {
        attach_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads a Qwen3-VL SafeTensors artifact through the generic component engine.
pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenVlModel, Error> {
    load_safetensors_with_residency(
        artifact,
        WeightResidency::with_layers(options.into()),
        quantization,
        stream,
        weights_stream,
    )
}

pub fn load_safetensors_with_residency(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenVlModel, Error> {
    let eredu_architectures::configuration::SafetensorsModelConfig::QwenVl(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Qwen3-VL loader received a different prepared architecture".into(),
        ));
    };
    let mut args = args.clone();
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen3-VL", args.text.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_options = residency.parameter_bank_cache();
    let options = residency.layers();
    let store = artifact.store();
    let store = if let Some(quantization) = quantize_on_load {
        let (store, target, _) = quantize_store(store, &args, quantization, stream)?;
        args = target;
        store
    } else {
        store
    };
    let mut model = load_store(
        store,
        args,
        options,
        expert_options.is_some(),
        stream,
        weights_stream,
    )?;
    if let Some(options) = expert_options {
        attach_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_parameter_bank(
    model: &mut QwenVlModel,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = match &model.execution {
        Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
        Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
    };
    let entries =
        crate::composition::qwen::expert::expert_catalog(&model.args.text, store.as_ref())?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
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
    external_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenVlModel, Error> {
    let mut architecture = Architecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
    let (policy, _) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        |modules, store| {
            build_module_bindings_with_recipes(
                &MlxModule::new(modules.clone()),
                "",
                store,
                vl::static_recipes(store),
            )
            .map_err(Into::into)
        },
        move |ordinal, _address, _path, unit, store, _| {
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                vl::unit_recipes(store, &binding_args, ordinal)
                    .map_err(Error::ArchitectureModel)?,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
    Ok(QwenVlModel {
        args,
        state_layout,
        execution,
        parameter_bank: None,
    })
}
