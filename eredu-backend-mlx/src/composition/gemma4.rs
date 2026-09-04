//! Neutral Gemma 4 binding to MLX storage, state, and residency policy.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    gemma4::{DecoderInputPart, FamilyConfig, LayeredModel as Architecture, ModelInput, Unit},
};
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, ExecutionUnitLayout,
    LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions, ParameterRole, RuntimeState,
    WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::runtime::checkpoint::gguf::GgufCheckpoint;
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
        checkpoint::{
            binding::{
                build_module_bindings, build_module_bindings_with_recipes_excluding,
                materialize_module_bindings, parameter_name_in_targets,
                populate_module_from_arrays_excluding, populate_module_from_lease_excluding,
            },
            load::{gguf_metadata, gguf_quantization_configs},
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                construct_architecture_unit, prepare_layerwise_policy_with_bindings,
                MlxLayerwisePolicy, MlxResidentPolicy, MlxUnitPopulator,
            },
            layerwise::{quantize_parameterized_module_store, quantize_parameterized_store},
        },
        media::input,
        residency::parameter_bank::{AddressableParameterBank, ParameterBankResidencyReport},
    },
};

type NeutralArchitecture = Architecture<MlxNeuralBackend>;
type NeutralUnit = Unit<MlxNeuralBackend>;
type NeutralAssistant = eredu_architectures::gemma4::Assistant<MlxNeuralBackend>;

type Resident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
>;
type Bounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
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
}

impl Execution {
    fn architecture(&self) -> &NeutralArchitecture {
        match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Bounded(runtime) => runtime.architecture(),
        }
    }

    fn output_group(&self) -> Result<usize, Error> {
        let architecture = self.architecture();
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::execution_graph(
            architecture,
        )
        .map(|graph| graph.output())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
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
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::forward_unit_with_inferred_provider(
        architecture,
        group,
        index,
        unit,
        hidden,
        state,
        forward,
        provider,
        stream,
    )
}

/// One neutral Gemma 4 object shared by resident and bounded execution.
pub struct Gemma4Model {
    args: FamilyConfig,
    state_layout: eredu_runtime::StateLayout,
    execution: Execution,
    parameter_bank: Option<AddressableParameterBank>,
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
            .draft_step::<crate::backend::runtime::cache::kv::ConcatKeyValueCache>(
                embedding, state, stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

/// Loads the released SafeTensors assistant into the backend-neutral module.
pub fn load_assistant_safetensors(
    store: SharedCheckpointSource,
    source_config: eredu_architectures::gemma4::AssistantConfig,
    options: crate::MlxLoadRequest,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    let quantization = options.weight_quantization()?;
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::ArchitectureModel(
            "Gemma 4 assistant loading supports fully resident weights only".into(),
        ));
    }
    if options
        .parallel_topology()
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Gemma 4 assistant loading requires replicated placement".into(),
        ));
    }
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Gemma 4 assistant", source_config.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let config = requested
        .map(|requested| {
            source_config
                .load_time_quantization(requested)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .transpose()?
        .unwrap_or_else(|| source_config.clone());
    let store = if let Some(requested) = requested {
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0
    } else {
        store
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(Gemma4AssistantModel { config, module })
}

pub fn load_assistant_gguf(
    checkpoint: eredu_gguf::Checkpoint,
    resolution: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    tensor_mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    source_config: eredu_architectures::gemma4::AssistantConfig,
    options: crate::MlxLoadRequest,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    let quantization = options.weight_quantization()?;
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::ArchitectureModel(
            "Gemma 4 assistant loading supports fully resident weights only".into(),
        ));
    }
    if options
        .parallel_topology()
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Gemma 4 assistant loading requires replicated placement".into(),
        ));
    }
    let mlx_checkpoint = GgufCheckpoint::from_portable(checkpoint.clone());
    let metadata = gguf_metadata(&mlx_checkpoint);
    let formats = gguf_quantization_configs(&mlx_checkpoint, &tensor_mapping)?;
    let source_config = source_config
        .with_checkpoint_formats(formats)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    crate::composition::mlx::validate_gguf_quantization_source(
        &mlx_checkpoint,
        &metadata,
        quantization,
    )?;
    let store: SharedCheckpointSource = Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(options.weight_residency.max_cached_shards())?
            .add_resolved_checkpoint(checkpoint, &resolution, &tensor_mapping)?
            .build()?,
    );
    let (store, config) = if let Some(requested) = quantization {
        let config = source_config
            .load_time_quantization(requested)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        (
            quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0,
            config,
        )
    } else {
        (store, source_config)
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(Gemma4AssistantModel { config, module })
}

impl Gemma4Model {
    pub fn args(&self) -> &FamilyConfig {
        &self.args
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
            CacheResidencyPolicy::Paged(options) => MlxHybridState::paged(
                self.state_layout.clone(),
                CacheResidencyManager::new(options)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )
            .map_err(Into::into),
        }
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let topology = eredu_core::cache::PromptCacheTopology::default();
        crate::composition::replicated_prompt_cache_identity(
            self.execution.architecture(),
            topology,
        )
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
        let rank = identity.topology().cache_rank_identity();
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
        state.restore_prompt_cache_state(tensors, processed, identity.layer_prefix_offsets())?;
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
        };
        Ok(Some(report))
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    pub fn parameter_bank_report(&self) -> Result<Option<ParameterBankResidencyReport>, Error> {
        self.parameter_bank
            .as_ref()
            .map(AddressableParameterBank::report)
            .transpose()
            .map_err(Into::into)
    }

    fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
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
        if state.layout() != &self.state_layout {
            return Err(Error::ArchitectureModel(
                "Gemma 4 cache layout mismatch".into(),
            ));
        }
        let output_group = self.execution.output_group()?;
        let mut final_text_hidden = None;
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let args = self.args.text.clone();
            let mut provider =
                crate::composition::gemma4_expert::cached_provider(&parameter_bank, &args);
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
            };
            drop(provider);
            self.parameter_bank = Some(parameter_bank);
            let (logits, forward) =
                result.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let hidden = final_text_hidden.ok_or_else(|| {
                Error::ArchitectureModel("Gemma 4 text graph produced no activation".into())
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
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let hidden = final_text_hidden.ok_or_else(|| {
            Error::ArchitectureModel("Gemma 4 text graph produced no activation".into())
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

    fn forward_with_observer(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::ArchitectureModel(
                "Gemma 4 cache layout mismatch".into(),
            ));
        }
        let parameter_bank = self.parameter_bank.take();
        let result = {
            let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
            match parameter_bank.as_ref() {
                Some(parameter_bank) => {
                    let args = self.args.text.clone();
                    let mut provider =
                        crate::composition::gemma4_expert::cached_provider(parameter_bank, &args);
                    match &mut self.execution {
                        Execution::Resident(runtime) => runtime
                            .forward_with_inferred_provider_and_observer(
                                input,
                                state,
                                &mut provider,
                                stream,
                                &mut neutral,
                            ),
                        Execution::Bounded(runtime) => runtime
                            .forward_with_inferred_provider_and_observer(
                                input,
                                state,
                                &mut provider,
                                stream,
                                &mut neutral,
                            ),
                    }
                }
                None => match &mut self.execution {
                    Execution::Resident(runtime) => {
                        runtime.forward_with_observer(input, state, stream, &mut neutral)
                    }
                    Execution::Bounded(runtime) => {
                        runtime.forward_with_observer(input, state, stream, &mut neutral)
                    }
                },
            }
        };
        self.parameter_bank = parameter_bank;
        let logits = result.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        eredu_runtime::observe_model_logits(observer, logits.as_array())
            .map(crate::MlxTensor::from_array)
            .map_err(Error::from)
    }

    pub fn forward_tokens_with_observer(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward_with_observer(
            ModelInput {
                parts: &parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
            observer,
        )
    }

    pub fn prefill_with_observer(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        input::validate(typed)?;
        let prepared = prepare_parts(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        self.forward_with_observer(
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
            observer,
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

    pub fn forward_input(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        input::validate(typed)?;
        let prepared = prepare_parts(&self.args, typed, stream)?;
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

pub type PreparedParts =
    eredu_architectures::gemma4::model::PreparedCompositeIngress<crate::MlxTensor>;

pub fn prepare_parts(
    args: &FamilyConfig,
    typed: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<PreparedParts, Error> {
    let prepared = crate::composition::mlx::replicated_text::prepared_composite_input(typed)?;
    let admitted = <NeutralArchitecture as CompositeArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::admit_prepared_input(args, &prepared, &input::MlxTensorInputInspector)
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let paired =
        PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
    eredu_architectures::gemma4::model::prepare_composite_ingress::<MlxNeuralBackend>(
        paired, stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
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

fn execution_layout(architecture: &NeutralArchitecture) -> Result<ExecutionUnitLayout, Error> {
    let graph =
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::execution_graph(
            architecture,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_unit_count(
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
    let target = eredu_architectures::gemma4::load_time_quantization(source, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = execution_layout(&source_architecture)?;
    let target_layout = execution_layout(&target_architecture)?;
    if source_layout.len() != target_layout.len() {
        return Err(Error::Quantization(
            "Gemma 4 quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = <NeutralArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::static_modules(&source_architecture)
    .clone();
    let target_static = <NeutralArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
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
    external_experts: bool,
) -> Result<Gemma4Model, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let external_expert_source_keys = if external_experts {
        crate::composition::gemma4_expert::checkpoint_keys(&args.text, store.as_ref())?
    } else {
        Default::default()
    };
    let binding_args = args.clone();
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, _) = prepare_layerwise_policy_with_bindings(
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
        move |ordinal, _address, _path, unit, store, _stream| {
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                eredu_architectures::gemma4::unit_recipes(store, &binding_args, ordinal)
                    .map_err(Error::ArchitectureModel)?
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
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
        state_layout,
        args,
        execution,
        parameter_bank: None,
    })
}

fn attach_parameter_bank(
    model: &mut Gemma4Model,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries =
        crate::composition::gemma4_expert::expert_catalog(&model.args.text, store.as_ref())?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
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
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.parameter_bank_cache();
    let eredu_architectures::configuration::SafetensorsModelConfig::Gemma4(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Gemma 4 loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Gemma 4", args.text.weight_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let (store, args) = match requested {
        Some(quantization) => {
            let (store, args, _) = quantize_store(store, &args, quantization, stream)?;
            (store, args)
        }
        None => (store, args),
    };
    let mut model = load_store(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads a Gemma 4 decoder and optional sibling media projector through the
/// same neutral family object.
pub fn load_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.parameter_bank_cache();
    let (store, args) = open_pipeline_gguf_store(source, projector, residency.max_cached_shards())?;
    let mut model = load_store(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

pub fn open_pipeline_gguf_store(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, FamilyConfig), Error> {
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::Gemma4(family) = source.model() else {
        return Err(Error::ArchitectureModel(
            "Gemma 4 GGUF loader received a different prepared model".into(),
        ));
    };
    let args = match projector {
        Some(projector) => {
            let eredu_architectures::gguf_companion::GgufMediaProjectorConfig::Gemma4(family) =
                projector.model()
            else {
                return Err(Error::ArchitectureModel(
                    "Gemma 4 GGUF loader received a mismatched media-projector plan".into(),
                ));
            };
            family.clone()
        }
        None => family.clone(),
    };
    let formats = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let args = eredu_architectures::gemma4::with_checkpoint_formats(&args, formats)
        .map_err(Error::ArchitectureModel)?;
    let builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_cached_readers)?
        .add_checkpoint(
            checkpoint.catalog().clone(),
            source.plan().checkpoint(),
            source.plan().tensor_mapping(),
        )?;
    let builder = if let Some(projector) = projector {
        builder.add_checkpoint(
            projector.checkpoint().catalog().clone(),
            projector.plan().checkpoint(),
            projector.plan().tensor_mapping(),
        )?
    } else {
        builder
    };
    let store: SharedCheckpointSource = Arc::new(builder.build()?);
    Ok((store, args))
}
