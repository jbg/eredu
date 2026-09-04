//! Neutral Inkling binding to MLX storage and heterogeneous state.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::inkling::{
    DecoderInputPart, LayeredModel as Architecture, ModelArgs, ModelInput, Unit,
};
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_nn::Tensor;
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, LayeredArchitecture,
    LayerwiseRuntime, PagedCacheOptions, ParameterRole, RuntimeState, WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

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
                build_module_bindings_with_recipes_excluding, parameter_name_in_targets,
                populate_module_from_lease_excluding,
            },
            load::gguf_quantization_configs,
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                architecture_execution_layout, construct_architecture_unit,
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitPopulator,
            },
            layerwise::quantize_module_store_with_bindings,
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
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
>;
type Bounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralUnit, UnitPopulator>,
>;

/// Complete neutral Inkling generation state, including checkpoint-embedded MTP.
#[derive(Debug, Clone)]
pub struct InklingState {
    target: MlxHybridState,
    mtp: Option<MlxHybridState>,
}

impl InklingState {
    pub fn target(&self) -> &MlxHybridState {
        &self.target
    }

    fn clear(&mut self) -> Result<(), Exception> {
        self.target.clear()?;
        if let Some(mtp) = &mut self.mtp {
            mtp.clear()?;
        }
        Ok(())
    }

    fn restore_target_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.target.restore_checkpoint(&checkpoint.target, stream)
    }
}

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
            MlxHybridState,
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
    state: &mut MlxHybridState,
    forward: &mut eredu_architectures::inkling::ForwardContext<crate::MlxTensor>,
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

/// One neutral Inkling object shared by resident and bounded execution.
pub struct InklingModel {
    args: ModelArgs,
    state_layouts: eredu_architectures::inkling::InklingStateLayouts,
    execution: Execution,
    parameter_bank: Option<AddressableParameterBank>,
}

impl InklingModel {
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn mtp_state(&self) -> Result<Option<MlxHybridState>, Error> {
        self.state_layouts
            .prediction()
            .cloned()
            .map(|state| {
                MlxHybridState::device_with_global_layer_start(
                    state.layout().clone(),
                    state.global_layer_offset(),
                )
                .map_err(Into::into)
            })
            .transpose()
    }

    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Error> {
        Ok(self.state_layouts.composite().layer_prefix_offsets())
    }

    pub fn new_cache(&self) -> InklingState {
        InklingState {
            target: MlxHybridState::device(self.state_layouts.target().clone())
                .expect("validated Inkling state must be realizable"),
            mtp: self
                .mtp_state()
                .expect("validated Inkling MTP state must be realizable"),
        }
    }

    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<InklingState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                let rank = None;
                Ok(InklingState {
                    target: MlxHybridState::paged(
                        self.state_layouts.target().clone(),
                        manager.clone(),
                        rank,
                    )?,
                    mtp: self
                        .state_layouts
                        .prediction()
                        .cloned()
                        .map(|state| {
                            MlxHybridState::paged_with_global_layer_start(
                                state.layout().clone(),
                                manager.clone(),
                                rank,
                                state.global_layer_offset(),
                            )
                        })
                        .transpose()?,
                })
            }
        }
    }

    pub(crate) fn prompt_identity(
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
    ) -> Result<(InklingState, eredu_core::cache::PromptCacheManifest), Error> {
        let identity = self.prompt_identity()?;
        let rank = identity.topology().cache_rank_identity();
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let loaded = load_prompt_cache_state_tensors(directory.as_ref(), &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut tensors = loaded
            .into_iter()
            .map(|state| ((state.owner, state.role), state.array))
            .collect::<BTreeMap<_, _>>();
        let offsets = self.prompt_cache_layer_prefix_offsets()?;
        let processed = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("prompt-cache prefix length exceeds i32".into()))?;
        let target_len = self.state_layouts.target().len();
        let mut target =
            MlxHybridState::paged(self.state_layouts.target().clone(), manager.clone(), rank)?;
        target.restore_prompt_cache_state_range(
            &mut tensors,
            0..target_len,
            processed,
            &offsets[..target_len],
        )?;
        let mtp = self
            .state_layouts
            .prediction()
            .cloned()
            .map(|prediction| {
                let len = prediction.layout().len();
                let mut state = MlxHybridState::paged_with_global_layer_start(
                    prediction.layout().clone(),
                    manager.clone(),
                    rank,
                    prediction.global_layer_offset(),
                )?;
                state.restore_prompt_cache_state_range(
                    &mut tensors,
                    0..len,
                    processed,
                    &offsets[target_len..],
                )?;
                Ok::<_, Exception>(state)
            })
            .transpose()?;
        if let Some(((owner, role), _)) = tensors.into_iter().next() {
            return Err(Error::Parallel(format!(
                "prompt cache contains undeclared Inkling state {owner:?}/{role:?}"
            )));
        }
        Ok((InklingState { target, mtp }, manifest))
    }

    pub fn save_prompt_cache(
        &self,
        state: &mut InklingState,
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
        let processed = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("prompt-cache prefix length exceeds i32".into()))?;
        let offsets = self.prompt_cache_layer_prefix_offsets()?;
        let target_len = self.state_layouts.target().len();
        let manager = state
            .target
            .residency_manager()
            .cloned()
            .ok_or_else(|| Error::Parallel("prompt persistence requires paged state".into()))?;
        let mut arrays = state.target.prompt_cache_state_arrays_range(
            0..target_len,
            processed,
            &offsets[..target_len],
        )?;
        if let Some(mtp) = &mut state.mtp {
            arrays.extend(mtp.prompt_cache_state_arrays_range(
                0..mtp.layout().len(),
                processed,
                &offsets[target_len..],
            )?);
        }
        manager
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &arrays, options)
            .map_err(|error| Error::Parallel(error.to_string()))
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
        state: &mut InklingState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        if state.target.layout() != self.state_layouts.target() {
            return Err(Error::ArchitectureModel(
                "Inkling cache layout mismatch".into(),
            ));
        }
        let mut final_text_hidden = None;
        let mut ordered_tokens = None;
        let output_group = self.execution.output_group()?;
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let args = self.args.clone();
            let mut provider =
                crate::composition::inkling_expert::cached_provider(&parameter_bank, &args);
            let result = match &mut self.execution {
                Execution::Resident(runtime) => runtime
                    .forward_with_unit_executor_and_activation_hook(
                        input,
                        &mut state.target,
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
                        |group, _index, hidden, forward| {
                            if group == output_group {
                                final_text_hidden = Some(hidden.clone());
                                ordered_tokens = Some(forward.tokens().clone());
                            }
                            Ok(())
                        },
                    ),
                Execution::Bounded(runtime) => runtime
                    .forward_with_unit_executor_and_activation_hook(
                        input,
                        &mut state.target,
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
                        |group, _index, hidden, forward| {
                            if group == output_group {
                                final_text_hidden = Some(hidden.clone());
                                ordered_tokens = Some(forward.tokens().clone());
                            }
                            Ok(())
                        },
                    ),
            };
            drop(provider);
            self.parameter_bank = Some(parameter_bank);
            let (logits, _) =
                result.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            return Ok(
                crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                    logits,
                    hidden: final_text_hidden.ok_or_else(|| {
                        Error::ArchitectureModel(
                            "Inkling text graph produced no target activation".into(),
                        )
                    })?,
                    tokens: ordered_tokens.ok_or_else(|| {
                        Error::ArchitectureModel(
                            "Inkling text graph retained no ordered token identity".into(),
                        )
                    })?,
                },
            );
        }
        let result = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward_with_unit_executor_and_activation_hook(
                input,
                &mut state.target,
                stream,
                |architecture, group, index, unit, hidden, state, forward, stream| {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
                },
                |group, _index, hidden, forward| {
                    if group == output_group {
                        final_text_hidden = Some(hidden.clone());
                        ordered_tokens = Some(forward.tokens().clone());
                    }
                    Ok(())
                },
            ),
            Execution::Bounded(runtime) => runtime.forward_with_unit_executor_and_activation_hook(
                input,
                &mut state.target,
                stream,
                |architecture, group, index, unit, hidden, state, forward, stream| {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
                },
                |group, _index, hidden, forward| {
                    if group == output_group {
                        final_text_hidden = Some(hidden.clone());
                        ordered_tokens = Some(forward.tokens().clone());
                    }
                    Ok(())
                },
            ),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: result.0,
                hidden: final_text_hidden.ok_or_else(|| {
                    Error::ArchitectureModel(
                        "Inkling text graph produced no target activation".into(),
                    )
                })?,
                tokens: ordered_tokens.ok_or_else(|| {
                    Error::ArchitectureModel(
                        "Inkling text graph retained no ordered token identity".into(),
                    )
                })?,
            },
        )
    }

    fn forward(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut InklingState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.forward_with_capture(input, state, stream)
            .map(|output| output.logits)
    }

    fn forward_with_observer(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut InklingState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        if state.target.layout() != self.state_layouts.target() {
            return Err(Error::ArchitectureModel(
                "Inkling cache layout mismatch".into(),
            ));
        }
        let parameter_bank = self.parameter_bank.take();
        let result = {
            let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
            match parameter_bank.as_ref() {
                Some(parameter_bank) => {
                    let args = self.args.clone();
                    let mut provider =
                        crate::composition::inkling_expert::cached_provider(parameter_bank, &args);
                    match &mut self.execution {
                        Execution::Resident(runtime) => runtime
                            .forward_with_inferred_provider_and_observer(
                                input,
                                &mut state.target,
                                &mut provider,
                                stream,
                                &mut neutral,
                            ),
                        Execution::Bounded(runtime) => runtime
                            .forward_with_inferred_provider_and_observer(
                                input,
                                &mut state.target,
                                &mut provider,
                                stream,
                                &mut neutral,
                            ),
                    }
                }
                None => match &mut self.execution {
                    Execution::Resident(runtime) => runtime.forward_with_observer(
                        input,
                        &mut state.target,
                        stream,
                        &mut neutral,
                    ),
                    Execution::Bounded(runtime) => runtime.forward_with_observer(
                        input,
                        &mut state.target,
                        stream,
                        &mut neutral,
                    ),
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
        state: &mut InklingState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward_with_observer(
            ModelInput {
                parts: &parts,
                vision_patches: None,
                audio: None,
            },
            state,
            stream,
            observer,
        )
    }

    pub fn forward_input_with_observer(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        prepare_input(&self.args, typed, stream)?
            .with_model_input(|input| self.forward_with_observer(input, state, stream, observer))
    }

    pub fn forward_tokens(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut InklingState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward(
            ModelInput {
                parts: &parts,
                vision_patches: None,
                audio: None,
            },
            state,
            stream,
        )
    }

    /// Runs a rank-local text pass through the configured tensor communicator.
    fn forward_input_with_capture(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        prepare_input(&self.args, typed, stream)?
            .with_model_input(|input| self.forward_with_capture(input, state, stream))
    }

    pub fn forward_input(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        self.forward_input_with_capture(typed, state, stream)
            .map(|output| output.logits)
    }

    pub fn mtp_len(&self) -> usize {
        match &self.execution {
            Execution::Resident(runtime) => runtime.architecture().mtp_len(),
            Execution::Bounded(runtime) => runtime.architecture().mtp_len(),
        }
    }

    fn forward_mtp_draft(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        depth: usize,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let architecture = match &mut self.execution {
            Execution::Resident(runtime) => runtime.architecture_mut(),
            Execution::Bounded(runtime) => runtime.architecture_mut(),
        };
        let embeddings = architecture
            .mtp_token_embeddings(tokens, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let output = architecture
            .forward_mtp_step(
                hidden,
                &embeddings,
                tokens,
                depth,
                state.layers_mut(),
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let logits = architecture
            .project_mtp_logits(&output.hidden, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden: output.hidden,
                tokens: output.tokens,
            },
        )
    }
}

pub(crate) fn prepare_input(
    args: &ModelArgs,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<eredu_architectures::inkling::PreparedInput<crate::MlxTensor>, Error> {
    let prepared = crate::composition::mlx::replicated_text::prepared_composite_input(input)?;
    let admitted = eredu_architectures::media_plan::admit_inkling_input(
        args,
        &prepared,
        &input::MlxTensorInputInspector,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let input =
        eredu_architectures::composite_execution::PreparedCompositeInput::new(&prepared, &admitted)
            .map_err(Error::ArchitectureModel)?;
    eredu_architectures::inkling::prepare_input(input, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

impl CausalModel<InklingState> for InklingModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        state: &mut InklingState,
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
        state: &mut InklingState,
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

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget for InklingModel {
    type Cache = InklingState;
    type DraftCache = MlxHybridState;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        cache.clear()?;
        self.forward_input_with_capture(input, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward_with_capture(
            ModelInput {
                parts: &parts,
                vision_patches: None,
                audio: None,
            },
            cache,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))
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
        let depth_count = self.mtp_len();
        let mtp = cache
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling checkpoint has no MTP state"))?;
        for depth in 0..depth_count {
            let _ = self.forward_mtp_draft(&hidden, &next, depth, mtp, stream)?;
        }
        Ok(())
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        cache
            .mtp
            .clone()
            .expect("MTP target is invoked only when Inkling has embedded predictor state")
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache.mtp = Some(draft.clone());
    }

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_target_checkpoint(checkpoint, stream)
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
        let output = self.forward_mtp_draft(hidden, &token, draft_index, cache, stream)?;
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
            let _ = self.forward_mtp_draft(hidden, tokens, depth, cache, stream)?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

fn quantize_store(
    store: SharedCheckpointSource,
    source: &ModelArgs,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        SharedCheckpointSource,
        ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target = eredu_architectures::inkling::load_time_quantization(source, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = architecture_execution_layout::<_, MlxHybridState>(&source_architecture)?;
    let target_layout = architecture_execution_layout::<_, MlxHybridState>(&target_architecture)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Inkling quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &source_architecture,
        )
        .clone(),
    );
    let target_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &target_architecture,
        )
        .clone(),
    );
    let static_args = source.clone();
    let unit_args = source.clone();
    let recipe_layout = source_layout.clone();
    let (store, report) = quantize_module_store_with_bindings(
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
            .map(MlxModule::new)
        },
        move |index, stream| {
            construct_architecture_unit(
                &target_architecture,
                &target_layout,
                index,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
            .map(MlxModule::new)
        },
        unit_count,
        quantization,
        stream,
        move |module, store| {
            let recipes =
                eredu_architectures::inkling::static_safetensors_recipes(&static_args, store)
                    .map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes_excluding(module, "", store, recipes, |_| false)
                .map_err(Into::into)
        },
        move |index, module, store| {
            let address = recipe_layout
                .address(index)
                .expect("validated Inkling recipe layout covers every unit");
            let recipes = eredu_architectures::inkling::unit_safetensors_recipes(
                &unit_args,
                store,
                address.group(),
                address.index(),
            )
            .map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes_excluding(module, "", store, recipes, |_| false)
                .map_err(Into::into)
        },
    )?;
    Ok((store, target, report))
}

fn load_store(
    store: SharedCheckpointSource,
    args: ModelArgs,
    layer_policy: eredu_runtime::LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<InklingModel, Error> {
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
        std::marker::PhantomData::<MlxHybridState>,
        layer_policy,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        move |modules, store| {
            let module = MlxModule::new(modules.clone());
            let recipes =
                eredu_architectures::inkling::static_safetensors_recipes(&static_args, store)
                    .map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |_| false)
                .map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _stream| {
            let module = MlxModule::new(unit);
            let recipes = eredu_architectures::inkling::unit_safetensors_recipes(
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
    let state_layouts = architecture
        .state_layouts()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let execution = if layer_policy.is_fully_resident() {
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
    Ok(InklingModel {
        state_layouts,
        args,
        execution,
        parameter_bank: None,
    })
}

fn attach_parameter_bank(
    model: &mut InklingModel,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = crate::composition::inkling_expert::expert_catalog(&model.args, store.as_ref())?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors into one neutral model across resident/bounded policies.
pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let expert_options = residency.parameter_bank_cache();
    let eredu_architectures::configuration::SafetensorsModelConfig::Inkling(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Inkling loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Inkling", args.text_config.weight_quantization, requested)
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

/// Loads the text GGUF and optional sibling media artifact into one neutral model.
pub fn load_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let expert_options = residency.parameter_bank_cache();
    let (store, args) = open_gguf_store(source, projector, residency.max_cached_shards())?;
    let layer_policy = residency.layers();
    let mut model = load_store(
        store,
        args,
        layer_policy,
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
) -> Result<(SharedCheckpointSource, ModelArgs), Error> {
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::Inkling(primary_args) = source.model()
    else {
        return Err(Error::ArchitectureModel(
            "Inkling GGUF loader received a different prepared model".into(),
        ));
    };
    let args = match projector {
        Some(projector) => {
            let eredu_architectures::gguf_companion::GgufMediaProjectorConfig::Inkling(args) =
                projector.model()
            else {
                return Err(Error::ArchitectureModel(
                    "Inkling GGUF loader received a mismatched media-projector plan".into(),
                ));
            };
            args.clone()
        }
        None => primary_args.clone(),
    };
    let text_formats = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let args = eredu_architectures::inkling::with_checkpoint_formats(&args, text_formats)
        .map_err(Error::ArchitectureModel)?;
    let mut builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_cached_readers)?
        .add_checkpoint(
            checkpoint.catalog().clone(),
            source.plan().checkpoint(),
            source.plan().tensor_mapping(),
        )?;
    if let Some(projector) = projector {
        builder = builder.add_checkpoint(
            projector.checkpoint().catalog().clone(),
            projector.plan().checkpoint(),
            projector.plan().tensor_mapping(),
        )?;
    }
    let store: SharedCheckpointSource = Arc::new(builder.build()?);
    Ok((store, args))
}
