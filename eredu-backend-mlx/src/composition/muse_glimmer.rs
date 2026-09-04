//! Neutral Muse-Glimmer binding to MLX storage and execution policy.

use std::{path::Path, sync::Arc};

use eredu_architectures::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    muse_glimmer::{
        DecoderConfig, DecoderInputPart, LayeredModel as Architecture, ModelInput, Unit,
    },
};
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, LayerWeightResidency,
    LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions, ParameterRole, RuntimeState,
    WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::residency::{open_prompt_cache, CacheResidencyManager},
        cache::state::MlxKeyValueState,
        checkpoint::{
            binding::{
                build_module_bindings_with_recipes_excluding, parameter_name_in_targets,
                populate_module_from_lease_excluding,
            },
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                architecture_execution_layout, construct_architecture_unit,
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitPopulator,
            },
            layerwise::quantize_parameterized_store,
        },
        media::input,
        residency::parameter_bank::{AddressableParameterBank, ParameterBankResidencyReport},
    },
};

type NeutralArchitecture = Architecture<MlxNeuralBackend>;
type NeutralUnit = Unit<MlxNeuralBackend>;

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
        <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::execution_graph(architecture)
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

/// One family object shared by resident and bounded execution.
pub struct MuseGlimmerModel {
    args: DecoderConfig,
    state_layout: eredu_runtime::StateLayout,
    execution: Execution,
    parameter_bank: Option<AddressableParameterBank>,
}

pub struct MuseGlimmerSpeculativeOutput {
    pub logits: crate::MlxTensor,
    pub target_states: Vec<crate::MlxTensor>,
}

pub fn prepare_muse_input(
    args: &DecoderConfig,
    typed: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<eredu_architectures::muse_glimmer::PreparedCompositeIngress<crate::MlxTensor>, Error> {
    let prepared = crate::composition::mlx::replicated_text::prepared_composite_input(typed)?;
    let admitted = <NeutralArchitecture as CompositeArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
    >>::admit_prepared_input(args, &prepared, &input::MlxTensorInputInspector)
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let paired =
        PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
    eredu_architectures::muse_glimmer::prepare_composite_ingress::<MlxNeuralBackend>(paired, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

impl MuseGlimmerModel {
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
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
            CacheResidencyPolicy::Paged(options) => MlxKeyValueState::paged(
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
        _stream: &Stream,
    ) -> Result<(MlxKeyValueState, eredu_core::cache::PromptCacheManifest), Error> {
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

    fn forward_with_taps(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxKeyValueState,
        target_layers: &[usize],
        stream: &Stream,
    ) -> Result<MuseGlimmerSpeculativeOutput, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::ArchitectureModel(
                "Muse-Glimmer cache layout mismatch".into(),
            ));
        }
        let mut capture = (!target_layers.is_empty())
            .then(|| eredu_runtime::TargetStateCapture::new(target_layers.iter().copied()))
            .transpose()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let output_group = self.execution.output_group()?;
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let args = self.args.clone();
            let mut provider =
                crate::composition::muse_glimmer_expert::cached_provider(&parameter_bank, &args);
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
            };
            drop(provider);
            self.parameter_bank = Some(parameter_bank);
            let (logits, _) =
                result.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let target_states = capture
                .map(eredu_runtime::TargetStateCapture::into_ordered)
                .transpose()
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?
                .unwrap_or_default();
            return Ok(MuseGlimmerSpeculativeOutput {
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
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let target_states = capture
            .map(eredu_runtime::TargetStateCapture::into_ordered)
            .transpose()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .unwrap_or_default();
        Ok(MuseGlimmerSpeculativeOutput {
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
            .map(|output| {
                debug_assert!(output.target_states.is_empty());
                output.logits
            })
    }

    fn forward_with_observer(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<safemlx::Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::ArchitectureModel(
                "Muse-Glimmer cache layout mismatch".into(),
            ));
        }
        let parameter_bank = self.parameter_bank.take();
        let result = {
            let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
            match parameter_bank.as_ref() {
                Some(parameter_bank) => {
                    let args = self.args.clone();
                    let mut provider = crate::composition::muse_glimmer_expert::cached_provider(
                        parameter_bank,
                        &args,
                    );
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
        state: &mut MlxKeyValueState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<safemlx::Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward_with_observer(
            ModelInput {
                parts: &parts,
                vision: None,
                mask: None,
            },
            state,
            stream,
            observer,
        )
    }

    pub fn forward_input_with_observer(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<safemlx::Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        let prepared = prepare_muse_input(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        self.forward_with_observer(
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
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

    pub fn forward_input(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        let prepared = prepare_muse_input(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        self.forward_with_taps(
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
                mask: None,
            },
            state,
            &[],
            stream,
        )
        .map(|output| {
            debug_assert!(output.target_states.is_empty());
            output.logits
        })
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
    let target = eredu_architectures::muse_glimmer::load_time_quantization(source, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
    external_experts: bool,
) -> Result<MuseGlimmerModel, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let static_args = args.clone();
    let unit_args = args.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, _) = prepare_layerwise_policy_with_bindings(
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
            let recipes =
                eredu_architectures::muse_glimmer::static_safetensors_recipes(&static_args, store)
                    .map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |_| false)
                .map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _stream| {
            let module = MlxModule::new(unit);
            let recipes = eredu_architectures::muse_glimmer::unit_safetensors_recipes(
                &unit_args,
                store,
                address.group(),
                address.index(),
            )
            .map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |name| {
                external_experts && parameter_name_in_targets(name, &binding_expert_targets)
            })
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
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        ))
    } else {
        Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(MuseGlimmerModel {
        state_layout,
        args,
        execution,
        parameter_bank: None,
    })
}

fn attach_parameter_bank(
    model: &mut MuseGlimmerModel,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries =
        crate::composition::muse_glimmer_expert::expert_catalog(&model.args, store.as_ref())?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
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
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerModel, Error> {
    let expert_options = residency.parameter_bank_cache();
    let eredu_architectures::configuration::SafetensorsModelConfig::MuseGlimmer(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Muse-Glimmer loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    let current = args.quantization;
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Muse-Glimmer", current, requested)
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

/// Loads split text/projector GGUF through the same neutral family object.
pub fn load_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerModel, Error> {
    let expert_options = residency.parameter_bank_cache();
    let (store, args) = open_gguf_store(source, projector, residency.max_cached_shards())?;
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

fn open_gguf_store(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, DecoderConfig), Error> {
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::MuseGlimmer(primary_args) =
        source.model()
    else {
        return Err(Error::ArchitectureModel(
            "Muse-Glimmer GGUF loader received a different prepared model".into(),
        ));
    };
    let args = match projector {
        Some(projector) => {
            let eredu_architectures::gguf_companion::GgufMediaProjectorConfig::MuseGlimmer(args) =
                projector.model()
            else {
                return Err(Error::ArchitectureModel(
                    "Muse-Glimmer GGUF loader received a mismatched media-projector plan".into(),
                ));
            };
            args.clone()
        }
        None => primary_args.clone(),
    };
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
