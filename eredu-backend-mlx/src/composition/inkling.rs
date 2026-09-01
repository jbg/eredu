//! Neutral Inkling binding to MLX storage and heterogeneous state.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::{
    inkling::{
        AudioInput, DecoderInputPart, LayeredModel as Architecture, ModelArgs, ModelInput, Unit,
    },
    media_plan,
};
use eredu_checkpoint::{
    store::{CheckpointSource, SharedCheckpointSource},
    WeightQuantization,
};
use eredu_core::InputModality;
use eredu_nn::Tensor;
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, LayerWeightResidency,
    LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions, ParallelModelInfo, ParameterRole,
    RuntimeState, WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp},
    transforms::async_eval_with_event,
    Array, Stream,
};

use crate::backend::{
    error::Error,
    nn::{
        shared::{MlxModule, MlxNeuralBackend},
        tensor::TokenValidationScope,
    },
    runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
            },
            state::MlxHybridState,
        },
        checkpoint::{
            binding::{
                binding_bytes, build_module_bindings_with_recipes_excluding,
                parameter_name_in_targets, parameter_role_targets,
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
            layerwise::{quantize_module_store_with_bindings, shard_layer_bindings},
        },
        media::input,
        residency::parameter_bank::{AddressableParameterBank, ParameterBankResidencyReport},
    },
};

type NeutralArchitecture = Architecture<MlxNeuralBackend>;
type NeutralUnit = Unit<MlxNeuralBackend>;
pub type InklingPipelineUnit = MlxModule<NeutralUnit>;

fn with_token_validation_scope<T>(
    operation: impl FnOnce() -> Result<T, Exception>,
) -> Result<T, Exception> {
    let scope = TokenValidationScope::begin()?;
    let output = operation()?;
    let validations = scope.finish();
    if !validations.is_empty() {
        async_eval_with_event(validations.arrays())?.synchronize()?;
        validations.validate_completed()?;
    }
    Ok(output)
}

fn group_kind(
    architecture: &NeutralArchitecture,
    group: usize,
) -> eredu_runtime::ArchitectureGroupKind {
    <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::group_transport(architecture, group)
    .kind
}
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

type ParallelBounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralUnit, ParallelUnitPopulator>,
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

/// Binding-only helper for Inkling pipeline checkpoint materialization.
#[derive(Default)]
pub struct InklingBindings {
    external_experts: bool,
}

impl InklingBindings {
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

    pub fn model_type<'a>(&self, architecture: &'a NeutralArchitecture) -> &'a str {
        &architecture.args().model_type
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub fn layer_count(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
    ) -> Result<usize, Error> {
        <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::group_unit_count(architecture, group)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    pub fn layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        layer: &InklingPipelineUnit,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.layer_count(architecture, group)?;
        let expert_targets = if group_kind(architecture, group)
            == eredu_runtime::ArchitectureGroupKind::Decoder
        {
            parameter_role_targets(
                &eredu_architectures::inkling::layer_parameter_groups(architecture.args(), index)?,
                ParameterRole::ExpertIntermediate,
            )
        } else {
            Default::default()
        };
        let recipes = eredu_architectures::inkling::unit_safetensors_recipes(
            architecture.args(),
            store,
            group,
            index,
        )
        .map_err(Error::ArchitectureModel)?;
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && parameter_name_in_targets(name, &expert_targets)
        })
        .map_err(Into::into)
    }

    pub fn cartesian_layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global_layer = MlxModule::new(
            <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::build_unit(architecture, group, index, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        let bindings = self.layer_bindings(architecture, group, index, &global_layer, store)?;
        if let Some(layout) = layout {
            shard_layer_bindings(bindings, store, layout)
        } else {
            Ok(bindings)
        }
    }
}

#[derive(Clone)]
struct UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<std::collections::BTreeSet<String>>,
}

#[derive(Clone)]
struct ParallelUnitPopulator {
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

impl MlxUnitPopulator<NeutralUnit> for ParallelUnitPopulator {
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
    ParallelResident(Box<Resident>),
    ParallelBounded(Box<ParallelBounded>),
}

impl Execution {
    fn architecture(&self) -> &NeutralArchitecture {
        match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Bounded(runtime) => runtime.architecture(),
            Self::ParallelResident(runtime) => runtime.architecture(),
            Self::ParallelBounded(runtime) => runtime.architecture(),
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
    >>::forward_unit_with_provider(
        architecture,
        group,
        index,
        unit,
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
    )
}

fn forward_mtp_draft_parallel_architecture(
    architecture: &mut NeutralArchitecture,
    hidden: &crate::MlxTensor,
    tokens: &crate::MlxTensor,
    depth: usize,
    state: &mut MlxHybridState,
    group: &crate::backend::runtime::distributed::Group,
    stream: &Stream,
) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
    let embeddings = architecture
        .mtp_token_embeddings_parallel(tokens, group, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let output = architecture
        .forward_mtp_step(
            hidden,
            &embeddings,
            tokens,
            depth,
            state.layers_mut(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let logits = architecture
        .project_mtp_logits_parallel(&output.hidden, group, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok(
        crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
            logits,
            hidden: output.hidden,
            tokens: output.tokens,
        },
    )
}

/// One neutral Inkling object shared by resident and bounded execution.
pub struct InklingModel {
    args: ModelArgs,
    state_layouts: eredu_architectures::inkling::InklingStateLayouts,
    execution: Execution,
    parameter_bank: Option<AddressableParameterBank>,
    parallel_info:
        Option<ParallelModelInfo<crate::composition::mlx::distributed::topology::MlxParallelPlan>>,
}

/// Collective context adapter for the neutral tensor-parallel MTP target.
pub struct InklingTensorMtpTarget<'a> {
    model: &'a mut InklingModel,
    group: &'a crate::backend::runtime::distributed::Group,
}

impl<'a> InklingTensorMtpTarget<'a> {
    pub const fn new(
        model: &'a mut InklingModel,
        group: &'a crate::backend::runtime::distributed::Group,
    ) -> Self {
        Self { model, group }
    }
}

impl InklingModel {
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::composition::mlx::distributed::topology::MlxParallelPlan>>
    {
        self.parallel_info.as_ref()
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
                let rank = self.parallel_info.as_ref().and_then(|info| {
                    crate::composition::mlx::distributed::topology::prompt_cache_topology(
                        info.topology(),
                    )
                    .cache_rank_identity()
                });
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
        let topology = self.parallel_info.as_ref().map_or_else(
            eredu_core::cache::PromptCacheTopology::default,
            |info| {
                crate::composition::mlx::distributed::topology::prompt_cache_topology(
                    info.topology(),
                )
            },
        );
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
            Execution::ParallelResident(runtime) => runtime.policy().residency_report()?,
            Execution::ParallelBounded(runtime) => runtime.policy().residency_report()?,
        };
        Ok(Some(report))
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) | Execution::ParallelResident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
            Execution::ParallelBounded(runtime) => runtime.policy().dense_stream_report(),
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
            Execution::ParallelResident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ParallelBounded(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    fn forward_with_capture(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut InklingState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "ordinary Inkling forward cannot execute a tensor-parallel model without its communicator"
                    .into(),
            ));
        }
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
                Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
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
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
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
        let positions = input
            .parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens)
                | DecoderInputPart::Image(tokens)
                | DecoderInputPart::Audio(tokens) => tokens.dim(1),
                DecoderInputPart::Projected { tokens, .. } => tokens.dim(1),
            })
            .sum::<i32>();
        let pass = if positions > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let parameter_bank = self.parameter_bank.take();
        let result = {
            let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
            match parameter_bank.as_ref() {
                Some(parameter_bank) => {
                    let args = self.args.clone();
                    let mut provider =
                        crate::composition::inkling_expert::cached_provider(parameter_bank, &args);
                    match &mut self.execution {
                        Execution::Resident(runtime) => runtime.forward_with_provider_and_observer(
                            input,
                            &mut state.target,
                            pass,
                            &mut provider,
                            stream,
                            &mut neutral,
                        ),
                        Execution::Bounded(runtime) => runtime.forward_with_provider_and_observer(
                            input,
                            &mut state.target,
                            pass,
                            &mut provider,
                            stream,
                            &mut neutral,
                        ),
                        Execution::ParallelResident(_) | Execution::ParallelBounded(_) => {
                            return Err(Error::Parallel(
                                "Inkling tensor-parallel observation requires its communicator"
                                    .into(),
                            ));
                        }
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
                    Execution::ParallelResident(_) | Execution::ParallelBounded(_) => {
                        return Err(Error::Parallel(
                            "Inkling tensor-parallel observation requires its communicator".into(),
                        ));
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
        let prepared = PreparedInklingInput::new(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(prepared.kinds.iter().copied())
            .zip(&prepared.projected)
            .map(|((value, kind), projected)| match projected {
                Some(embeddings) => DecoderInputPart::Projected {
                    tokens: value,
                    embeddings,
                },
                None => match kind {
                    InputModality::Text => DecoderInputPart::Text(value),
                    InputModality::Image => DecoderInputPart::Image(value),
                    InputModality::Audio => DecoderInputPart::Audio(value),
                    InputModality::Video => unreachable!(),
                    _ => unreachable!("validated Inkling input modality"),
                },
            })
            .collect::<Vec<_>>();
        let audio = prepared.audio.as_ref().map(|code_ids| AudioInput {
            code_ids,
            valid_frames: code_ids.dim(1),
        });
        self.forward_with_observer(
            ModelInput {
                parts: &parts,
                vision_patches: prepared.images.as_ref(),
                audio,
            },
            state,
            stream,
            observer,
        )
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
    pub fn forward_tensor_parallel(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut InklingState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.target.layout() != self.state_layouts.target() {
            return Err(Error::Parallel(
                "Inkling tensor-parallel cache layout mismatch".into(),
            ));
        }
        let parts = [DecoderInputPart::Text(tokens)];
        let input = ModelInput {
            parts: &parts,
            vision_patches: None,
            audio: None,
        };
        let logits = match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel(input, &mut state.target, group, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?,
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel(input, &mut state.target, group, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?,
            Execution::Resident(_) | Execution::Bounded(_) => {
                return Err(Error::Parallel(
                    "Inkling model was not loaded for tensor parallelism".into(),
                ))
            }
        };
        Ok(logits)
    }

    pub fn prefill_tensor_parallel(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.target.layout() != self.state_layouts.target() {
            return Err(Error::Parallel(
                "Inkling tensor-parallel cache layout mismatch".into(),
            ));
        }
        let prepared = PreparedInklingInput::new(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(prepared.kinds.iter().copied())
            .zip(&prepared.projected)
            .map(|((value, kind), projected)| match projected {
                Some(embeddings) => DecoderInputPart::Projected {
                    tokens: value,
                    embeddings,
                },
                None => match kind {
                    InputModality::Text => DecoderInputPart::Text(value),
                    InputModality::Image => DecoderInputPart::Image(value),
                    InputModality::Audio => DecoderInputPart::Audio(value),
                    InputModality::Video => unreachable!(),
                    _ => unreachable!("validated Inkling input modality"),
                },
            })
            .collect::<Vec<_>>();
        let audio = prepared.audio.as_ref().map(|code_ids| AudioInput {
            code_ids,
            valid_frames: code_ids.dim(1),
        });
        let input = ModelInput {
            parts: &parts,
            vision_patches: prepared.images.as_ref(),
            audio,
        };
        let logits = match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel(input, &mut state.target, group, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?,
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel(input, &mut state.target, group, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?,
            Execution::Resident(_) | Execution::Bounded(_) => {
                return Err(Error::Parallel(
                    "Inkling model was not loaded for tensor parallelism".into(),
                ))
            }
        };
        Ok(logits)
    }

    pub fn forward_tensor_parallel_with_observer(
        &mut self,
        tokens: &crate::MlxTensor,
        state: &mut InklingState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward_parallel_input_with_observer(
            ModelInput {
                parts: &parts,
                vision_patches: None,
                audio: None,
            },
            state,
            group,
            stream,
            observer,
        )
    }

    pub fn prefill_tensor_parallel_with_observer(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        let prepared = PreparedInklingInput::new(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(prepared.kinds.iter().copied())
            .zip(&prepared.projected)
            .map(|((value, kind), projected)| match projected {
                Some(embeddings) => DecoderInputPart::Projected {
                    tokens: value,
                    embeddings,
                },
                None => match kind {
                    InputModality::Text => DecoderInputPart::Text(value),
                    InputModality::Image => DecoderInputPart::Image(value),
                    InputModality::Audio => DecoderInputPart::Audio(value),
                    InputModality::Video => unreachable!(),
                    _ => unreachable!("validated Inkling input modality"),
                },
            })
            .collect::<Vec<_>>();
        let audio = prepared.audio.as_ref().map(|code_ids| AudioInput {
            code_ids,
            valid_frames: code_ids.dim(1),
        });
        self.forward_parallel_input_with_observer(
            ModelInput {
                parts: &parts,
                vision_patches: prepared.images.as_ref(),
                audio,
            },
            state,
            group,
            stream,
            observer,
        )
    }

    fn forward_parallel_input_with_observer(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut InklingState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<crate::MlxTensor, Error> {
        if state.target.layout() != self.state_layouts.target() {
            return Err(Error::Parallel(
                "Inkling tensor-parallel cache layout mismatch".into(),
            ));
        }
        let positions = input
            .parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens)
                | DecoderInputPart::Image(tokens)
                | DecoderInputPart::Audio(tokens) => tokens.dim(1),
                DecoderInputPart::Projected { tokens, .. } => tokens.dim(1),
            })
            .sum::<i32>();
        let pass = if positions > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let parameter_bank = self.parameter_bank.take();
        let result = {
            let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
            let output = match parameter_bank.as_ref() {
                Some(parameter_bank) => {
                    let args = self.args.clone();
                    let mut provider =
                        crate::composition::inkling_expert::cached_provider(parameter_bank, &args);
                    match &mut self.execution {
                        Execution::ParallelResident(runtime) => runtime
                            .forward_parallel_with_provider_and_observer(
                                input,
                                &mut state.target,
                                pass,
                                &mut provider,
                                group,
                                stream,
                                &mut neutral,
                            ),
                        Execution::ParallelBounded(runtime) => runtime
                            .forward_parallel_with_provider_and_observer(
                                input,
                                &mut state.target,
                                pass,
                                &mut provider,
                                group,
                                stream,
                                &mut neutral,
                            ),
                        _ => {
                            return Err(Error::Parallel(
                                "Inkling was not loaded for tensor parallelism".into(),
                            ))
                        }
                    }
                }
                None => match &mut self.execution {
                    Execution::ParallelResident(runtime) => runtime.forward_parallel_with_observer(
                        input,
                        &mut state.target,
                        group,
                        stream,
                        &mut neutral,
                    ),
                    Execution::ParallelBounded(runtime) => runtime.forward_parallel_with_observer(
                        input,
                        &mut state.target,
                        group,
                        stream,
                        &mut neutral,
                    ),
                    _ => {
                        return Err(Error::Parallel(
                            "Inkling was not loaded for tensor parallelism".into(),
                        ))
                    }
                },
            }
            .map_err(|error| Error::Parallel(error.to_string()))?;
            eredu_runtime::observe_model_logits(&mut neutral, &output).map_err(Error::from)
        };
        self.parameter_bank = parameter_bank;
        result
    }

    fn forward_input_with_capture(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        let prepared = PreparedInklingInput::new(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(prepared.kinds.iter().copied())
            .zip(&prepared.projected)
            .map(|((value, kind), projected)| match projected {
                Some(embeddings) => DecoderInputPart::Projected {
                    tokens: value,
                    embeddings,
                },
                None => match kind {
                    InputModality::Text => DecoderInputPart::Text(value),
                    InputModality::Image => DecoderInputPart::Image(value),
                    InputModality::Audio => DecoderInputPart::Audio(value),
                    InputModality::Video => unreachable!(),
                    _ => unreachable!("validated Inkling input modality"),
                },
            })
            .collect::<Vec<_>>();
        let audio_input = prepared.audio.as_ref().map(|code_ids| AudioInput {
            code_ids,
            valid_frames: code_ids.dim(1),
        });
        self.forward_with_capture(
            ModelInput {
                parts: &parts,
                vision_patches: prepared.images.as_ref(),
                audio: audio_input,
            },
            state,
            stream,
        )
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
            Execution::ParallelResident(runtime) => runtime.architecture().mtp_len(),
            Execution::ParallelBounded(runtime) => runtime.architecture().mtp_len(),
        }
    }

    fn forward_parallel_with_capture(
        &mut self,
        input: ModelInput<'_, crate::MlxTensor>,
        state: &mut InklingState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        if state.target.layout() != self.state_layouts.target() {
            return Err(Error::Parallel(
                "Inkling tensor-parallel cache layout mismatch".into(),
            ));
        }
        let (logits, context) = match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel_with_context_hook(
                    input,
                    &mut state.target,
                    group,
                    stream,
                    |_, _, _| Ok(()),
                )
                .map_err(|error| Error::Parallel(error.to_string()))?,
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel_with_context_hook(
                    input,
                    &mut state.target,
                    group,
                    stream,
                    |_, _, _| Ok(()),
                )
                .map_err(|error| Error::Parallel(error.to_string()))?,
            Execution::Resident(_) | Execution::Bounded(_) => {
                return Err(Error::Parallel(
                    "Inkling model was not loaded for tensor parallelism".into(),
                ))
            }
        };
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden: context.target_hidden().cloned().ok_or_else(|| {
                    Error::Parallel("Inkling TP target pass retained no hidden state".into())
                })?,
                tokens: context.tokens().clone(),
            },
        )
    }

    fn forward_input_parallel_with_capture(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        let prepared = PreparedInklingInput::new(&self.args, typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(prepared.kinds.iter().copied())
            .zip(&prepared.projected)
            .map(|((value, kind), projected)| match projected {
                Some(embeddings) => DecoderInputPart::Projected {
                    tokens: value,
                    embeddings,
                },
                None => match kind {
                    InputModality::Text => DecoderInputPart::Text(value),
                    InputModality::Image => DecoderInputPart::Image(value),
                    InputModality::Audio => DecoderInputPart::Audio(value),
                    InputModality::Video => unreachable!(),
                    _ => unreachable!("validated Inkling input modality"),
                },
            })
            .collect::<Vec<_>>();
        let audio = prepared.audio.as_ref().map(|code_ids| AudioInput {
            code_ids,
            valid_frames: code_ids.dim(1),
        });
        self.forward_parallel_with_capture(
            ModelInput {
                parts: &parts,
                vision_patches: prepared.images.as_ref(),
                audio,
            },
            state,
            group,
            stream,
        )
    }

    fn forward_mtp_draft_parallel(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        depth: usize,
        state: &mut MlxHybridState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        match &mut self.execution {
            Execution::ParallelResident(runtime) => forward_mtp_draft_parallel_architecture(
                runtime.architecture_mut(),
                hidden,
                tokens,
                depth,
                state,
                group,
                stream,
            ),
            Execution::ParallelBounded(runtime) => forward_mtp_draft_parallel_architecture(
                runtime.architecture_mut(),
                hidden,
                tokens,
                depth,
                state,
                group,
                stream,
            ),
            Execution::Resident(_) | Execution::Bounded(_) => Err(Error::Parallel(
                "Inkling was not loaded for tensor-parallel MTP".into(),
            )),
        }
        .map_err(|error| Exception::custom(error.to_string()))
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
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => {
                return Err(Exception::custom(
                    "tensor-parallel Inkling embedded MTP is not configured",
                ))
            }
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

pub struct PreparedInklingInput {
    pub tokens: Vec<crate::MlxTensor>,
    pub kinds: Vec<InputModality>,
    pub projected: Vec<Option<crate::MlxTensor>>,
    pub images: Option<crate::MlxTensor>,
    pub audio: Option<crate::MlxTensor>,
}

impl PreparedInklingInput {
    pub fn new(
        args: &ModelArgs,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<PreparedInklingInput, Error> {
        input::validate(typed)?;
        let mut tokens = Vec::new();
        let mut kinds = Vec::new();
        let mut projected = Vec::new();
        let mut images = Vec::new();
        let mut audio = Vec::new();
        for part in typed.parts {
            let plan = media_plan::inkling_input_part(args, part, &input::MlxInputInspector)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            match (plan, part.payload()) {
                (
                    media_plan::InklingInputPartPlan::TextTokens { .. },
                    input::InputPayload::TokenIds(value),
                ) => {
                    tokens.push(crate::MlxTensor::from_array(value.clone()));
                    kinds.push(InputModality::Text);
                    projected.push(None);
                }
                (
                    media_plan::InklingInputPartPlan::Media {
                        modality: InputModality::Image,
                        ingress,
                        ..
                    },
                    input::InputPayload::Tensor(value),
                ) => {
                    let count = usize::try_from(ingress.placeholder_count).map_err(|_| {
                        Error::ArchitectureModel(
                            "Inkling image placeholder span exceeds host capacity".into(),
                        )
                    })?;
                    tokens.push(crate::MlxTensor::from_array(input::token_ids_array(
                        &vec![ingress.placeholder_token_id; count],
                        stream,
                    )?));
                    kinds.push(InputModality::Image);
                    projected.push(None);
                    images.push(value.clone());
                }
                (
                    media_plan::InklingInputPartPlan::Media {
                        modality: InputModality::Audio,
                        ingress,
                        ..
                    },
                    input::InputPayload::Tensor(value),
                ) => {
                    let count = usize::try_from(ingress.placeholder_count).map_err(|_| {
                        Error::ArchitectureModel(
                            "Inkling audio placeholder span exceeds host capacity".into(),
                        )
                    })?;
                    let retained_frames =
                        i32::try_from(ingress.placeholder_count).map_err(|_| {
                            Error::ArchitectureModel(
                                "Inkling audio placeholder span exceeds tensor capacity".into(),
                            )
                        })?;
                    tokens.push(crate::MlxTensor::from_array(input::token_ids_array(
                        &vec![ingress.placeholder_token_id; count],
                        stream,
                    )?));
                    kinds.push(InputModality::Audio);
                    projected.push(None);
                    audio.push(value.try_index_device((.., ..retained_frames, ..), stream)?);
                }
                (
                    media_plan::InklingInputPartPlan::Projected {
                        modality,
                        placeholder_token_id,
                        positions,
                    },
                    input::InputPayload::Embeddings(value),
                ) => {
                    let count = usize::try_from(positions).map_err(|_| {
                        Error::ArchitectureModel(
                            "Inkling projected placeholder span exceeds host capacity".into(),
                        )
                    })?;
                    tokens.push(crate::MlxTensor::from_array(input::token_ids_array(
                        &vec![placeholder_token_id; count],
                        stream,
                    )?));
                    kinds.push(match modality {
                        InputModality::Image => InputModality::Image,
                        InputModality::Audio => InputModality::Audio,
                        InputModality::Text | InputModality::Video => unreachable!(),
                        _ => unreachable!("validated Inkling input modality"),
                    });
                    projected.push(Some(crate::MlxTensor::from_array(value.clone())));
                }
                _ => {
                    return Err(Error::ArchitectureModel(format!(
                        "Inkling input plan disagrees with the prepared {} payload",
                        part.modality().as_str()
                    )))
                }
            }
        }
        Ok(PreparedInklingInput {
            tokens,
            kinds,
            projected,
            images: concatenate(&images, 0, stream)?.map(crate::MlxTensor::from_array),
            audio: concatenate(&audio, 1, stream)?.map(crate::MlxTensor::from_array),
        })
    }
}

fn concatenate(values: &[Array], axis: i32, stream: &Stream) -> Result<Option<Array>, Error> {
    match values {
        [] => Ok(None),
        [value] => Ok(Some(value.clone())),
        _ => Ok(Some(concatenate_axis(
            &values.iter().collect::<Vec<_>>(),
            axis,
            stream,
        )?)),
    }
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

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget
    for InklingTensorMtpTarget<'_>
{
    type Cache = InklingState;
    type DraftCache = MlxHybridState;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        with_token_validation_scope(|| {
            cache.clear()?;
            self.model
                .forward_input_parallel_with_capture(input, cache, self.group, stream)
                .map_err(|error| Exception::custom(error.to_string()))
        })
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        with_token_validation_scope(|| {
            let parts = [DecoderInputPart::Text(tokens)];
            self.model
                .forward_parallel_with_capture(
                    ModelInput {
                        parts: &parts,
                        vision_patches: None,
                        audio: None,
                    },
                    cache,
                    self.group,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))
        })
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
        let mtp = cache
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling checkpoint has no MTP state"))?;
        with_token_validation_scope(|| {
            for depth in 0..self.model.mtp_len() {
                let _ = self
                    .model
                    .forward_mtp_draft_parallel(&hidden, &next, depth, mtp, self.group, stream)?;
            }
            Ok(())
        })
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
        with_token_validation_scope(|| {
            let token = crate::MlxTensor::from_array(Array::from_slice(&[last_token], &[1, 1]));
            let output = self.model.forward_mtp_draft_parallel(
                hidden,
                &token,
                draft_index,
                cache,
                self.group,
                stream,
            )?;
            Ok((output.logits, output.hidden))
        })
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        with_token_validation_scope(|| {
            for depth in 0..self.model.mtp_len() {
                let _ = self
                    .model
                    .forward_mtp_draft_parallel(hidden, tokens, depth, cache, self.group, stream)?;
            }
            Ok(())
        })
    }

    fn max_draft_tokens(&self) -> usize {
        self.model.mtp_len()
    }
}

pub fn prepare_gguf_pipeline_source(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, ModelArgs), Error> {
    open_gguf_store(source, projector, max_cached_readers)
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
        parallel_info: None,
    })
}

fn load_parallel_store(
    store: SharedCheckpointSource,
    args: ModelArgs,
    layer_policy: LayerWeightResidency,
    build: crate::composition::mlx::distributed::topology::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_execution =
        architecture_execution_layout::<_, MlxHybridState>(&global_architecture)?;
    let layout =
        crate::composition::parallel_layout_from_description(build, &parameter_description)?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Inkling declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = Arc::new(
        eredu_architectures::inkling::local_geometry(&args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let mut architecture =
        NeutralArchitecture::new_parallel(args.clone(), Arc::clone(&geometry), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let state_layouts = architecture
        .state_layouts()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let global_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &global_architecture,
        )
        .clone(),
    );
    let static_recipes =
        eredu_architectures::inkling::static_safetensors_recipes(&args, store.as_ref())
            .map_err(Error::ArchitectureModel)?;
    let global_static_bindings = build_module_bindings_with_recipes_excluding(
        &global_static,
        "",
        store.as_ref(),
        static_recipes,
        |_| false,
    )?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for ordinal in 0..global_execution.len() {
        let address = global_execution
            .address(ordinal)
            .expect("validated Inkling layout covers every global unit");
        let unit = MlxModule::new(construct_architecture_unit(
            &global_architecture,
            &global_execution,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxHybridState>,
        )?);
        let recipes = eredu_architectures::inkling::unit_safetensors_recipes(
            &args,
            store.as_ref(),
            address.group(),
            address.index(),
        )
        .map_err(Error::ArchitectureModel)?;
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
            .ok_or_else(|| Error::Parallel("Inkling global parameter bytes overflowed".into()))?;
    }

    let static_layout = Arc::new(layout);
    let unit_sharding = Arc::clone(&static_layout);
    let report_layout = Arc::clone(&static_layout);
    let binding_args = args.clone();
    let binding_architecture = global_architecture;
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        ParallelUnitPopulator {
            external_experts: false,
            expert_targets: Arc::new(Default::default()),
        },
        std::marker::PhantomData::<MlxHybridState>,
        layer_policy,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| shard_layer_bindings(global_static_bindings, store, &static_layout),
        move |_ordinal, address, _path, _local, store, stream| {
            let global =
                MlxModule::new(
                    <NeutralArchitecture as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(
                        &binding_architecture,
                        address.group(),
                        address.index(),
                        stream,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
                );
            let recipes = eredu_architectures::inkling::unit_safetensors_recipes(
                &binding_args,
                store,
                address.group(),
                address.index(),
            )
            .map_err(Error::ArchitectureModel)?;
            let bindings =
                build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |_| {
                    false
                })?;
            shard_layer_bindings(bindings, store, &unit_sharding)
        },
    )?;
    metadata.set_effective_model_type(args.model_type.clone());
    metadata.set_quantization(args.text_config.weight_quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("Inkling local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("Inkling device parameter bytes overflowed".into()))?;
    let parallel_info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        report_layout
            .tensors()
            .map(|(target, _)| target.to_owned())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if layer_policy.is_fully_resident() {
            local_parameter_bytes
        } else {
            metadata.static_device_bytes()
        },
        maximum_device_parameter_bytes,
    );
    let execution = if layer_policy.is_fully_resident() {
        Execution::ParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        Execution::ParallelBounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(InklingModel {
        args,
        state_layouts,
        execution,
        parameter_bank: None,
        parallel_info: Some(parallel_info),
    })
}

/// Loads an Inkling SafeTensors checkpoint for pure tensor parallelism.
pub fn load_safetensors_tensor_parallel(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    layer_policy: LayerWeightResidency,
    build: crate::composition::mlx::distributed::topology::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let eredu_architectures::configuration::SafetensorsModelConfig::Inkling(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Inkling loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    load_parallel_store(store, args, layer_policy, build, stream, weights_stream)
}

/// Loads an Inkling GGUF checkpoint through the same neutral TP binder.
pub fn load_gguf_tensor_parallel(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    layer_policy: LayerWeightResidency,
    build: crate::composition::mlx::distributed::topology::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let (store, args) = open_gguf_store(source, projector, layer_policy.max_cached_shards())?;
    load_parallel_store(store, args, layer_policy, build, stream, weights_stream)
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
