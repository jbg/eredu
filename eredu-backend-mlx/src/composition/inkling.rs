//! Neutral Inkling binding to MLX storage and heterogeneous state.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

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
use eredu_nn::Tensor;
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, LayerWeightResidency, LayeredArchitecture, LayerwiseRuntime,
    PagedCacheOptions, ParallelModelInfo, ParameterRole, RuntimeState, StaticUnitBindings,
    WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

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
                binding_bytes, build_module_bindings_with_recipes_excluding,
                parameter_name_in_targets, parameter_role_targets,
                populate_module_from_lease_excluding,
            },
            load::{gguf_metadata, gguf_quantization_configs},
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
        residency::expert_cache::{ExpertCache, ExpertCacheReport},
    },
};

type NeutralArchitecture = Architecture<MlxNeuralBackend>;
type NeutralUnit = Unit<MlxNeuralBackend>;
pub type InklingPipelineUnit = MlxModule<NeutralUnit>;

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

type ParallelResident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
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

    pub fn static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
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
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
        let recipes =
            crate::composition::inkling_expert::module_recipes(layer, architecture.args(), store)?;
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
        global_layer: &InklingPipelineUnit,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings = self.layer_bindings(architecture, group, index, global_layer, store)?;
        if let Some(layout) = layout {
            let root = <NeutralArchitecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::unit_path(architecture, group, index)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            shard_layer_bindings(bindings, &root, store, layout)
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
    ParallelResident(Box<ParallelResident>),
    ParallelBounded(Box<ParallelBounded>),
}

impl Execution {
    fn output_group(&self) -> Result<usize, Error> {
        let architecture = match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Bounded(runtime) => runtime.architecture(),
            Self::ParallelResident(runtime) => runtime.architecture(),
            Self::ParallelBounded(runtime) => runtime.architecture(),
        };
        <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::execution_graph(architecture)
        .map(|graph| graph.output())
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
    group: &safemlx::distributed::Group,
    stream: &Stream,
) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
    let embeddings = architecture
        .mtp_token_embeddings_parallel(tokens, group, stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let output = architecture
        .forward_mtp_step(
            hidden,
            &embeddings,
            tokens,
            depth,
            state.layers_mut(),
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let logits = architecture
        .project_mtp_logits_parallel(&output.hidden, group, stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
    state_layout: eredu_runtime::StateLayout,
    mtp_state_layout: Option<eredu_runtime::StateLayout>,
    prompt_state_layout: eredu_runtime::StateLayout,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    execution: Execution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::MlxParallelContext>>,
}

/// Collective context adapter for the neutral tensor-parallel MTP target.
pub struct InklingTensorMtpTarget<'a> {
    model: &'a mut InklingModel,
    group: &'a safemlx::distributed::Group,
}

impl<'a> InklingTensorMtpTarget<'a> {
    pub const fn new(model: &'a mut InklingModel, group: &'a safemlx::distributed::Group) -> Self {
        Self { model, group }
    }
}

impl InklingModel {
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    pub fn metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    pub fn parallel_info(&self) -> Option<&ParallelModelInfo<crate::backend::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    fn mtp_state(&self) -> Result<Option<MlxHybridState>, Error> {
        self.mtp_state_layout
            .clone()
            .map(|layout| {
                MlxHybridState::device_with_global_layer_start(layout, self.state_layout.len())
                    .map_err(Into::into)
            })
            .transpose()
    }

    fn prompt_state_identity(
        &self,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        eredu_architectures::inkling::state_identity(
            &self.args,
            &self.prompt_state_layout,
            0,
            topology,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Error> {
        Ok(self.prompt_state_layout.layer_prefix_offsets())
    }

    pub fn new_cache(&self) -> InklingState {
        InklingState {
            target: MlxHybridState::device(self.state_layout.clone())
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
                    crate::backend::cache::prompt_cache_topology(info.topology())
                        .cache_rank_identity()
                });
                Ok(InklingState {
                    target: MlxHybridState::paged(
                        self.state_layout.clone(),
                        manager.clone(),
                        rank,
                    )?,
                    mtp: self
                        .mtp_state_layout
                        .clone()
                        .map(|layout| {
                            MlxHybridState::paged_with_global_layer_start(
                                layout,
                                manager.clone(),
                                rank,
                                self.state_layout.len(),
                            )
                        })
                        .transpose()?,
                })
            }
        }
    }

    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.prompt_state_layout.layers().clone())
    }

    fn prompt_identity(&self) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let topology = self
            .parallel_info
            .as_ref()
            .map_or_else(eredu_core::cache::PromptCacheTopology::default, |info| {
                crate::backend::cache::prompt_cache_topology(info.topology())
            });
        self.prompt_state_identity(topology)?
            .prompt_cache_identity(&self.prompt_state_layout)
            .map_err(|error| Error::Parallel(error.to_string()))
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
        let rank = identity.topology.cache_rank_identity();
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
        let target_len = self.state_layout.len();
        let mut target = MlxHybridState::paged(self.state_layout.clone(), manager.clone(), rank)?;
        target.restore_prompt_cache_state_range(
            &mut tensors,
            0..target_len,
            processed,
            &offsets[..target_len],
        )?;
        let mtp = self
            .mtp_state_layout
            .clone()
            .map(|layout| {
                let len = layout.len();
                let mut state = MlxHybridState::paged_with_global_layer_start(
                    layout,
                    manager.clone(),
                    rank,
                    target_len,
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
        let target_len = self.state_layout.len();
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

    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
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
        if state.target.layout() != &self.state_layout {
            return Err(Error::UnsupportedArchitecture(
                "Inkling cache layout mismatch".into(),
            ));
        }
        let mut final_text_hidden = None;
        let mut ordered_tokens = None;
        let output_group = self.execution.output_group()?;
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let mut provider =
                crate::composition::inkling_expert::cached_provider(&expert_cache, &args);
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
            self.expert_cache = Some(expert_cache);
            let (logits, _) =
                result.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            return Ok(
                crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                    logits,
                    hidden: final_text_hidden.ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Inkling text graph produced no target activation".into(),
                        )
                    })?,
                    tokens: ordered_tokens.ok_or_else(|| {
                        Error::UnsupportedArchitecture(
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
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: result.0,
                hidden: final_text_hidden.ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Inkling text graph produced no target activation".into(),
                    )
                })?,
                tokens: ordered_tokens.ok_or_else(|| {
                    Error::UnsupportedArchitecture(
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
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.target.layout() != &self.state_layout {
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
        let vocabulary = self
            .args
            .text_config
            .unpadded_vocab_size
            .unwrap_or(self.args.text_config.vocab_size);
        if vocabulary == self.args.text_config.vocab_size {
            Ok(logits)
        } else {
            logits
                .as_array()
                .try_index_device((.., .., ..vocabulary), stream)
                .map(crate::MlxTensor::from_array)
                .map_err(Into::into)
        }
    }

    pub fn prefill_tensor_parallel(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut InklingState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Error> {
        if state.target.layout() != &self.state_layout {
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
                    input::Modality::Text => DecoderInputPart::Text(value),
                    input::Modality::Image => DecoderInputPart::Image(value),
                    input::Modality::Audio => DecoderInputPart::Audio(value),
                    input::Modality::Video => unreachable!(),
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
        let vocabulary = self
            .args
            .text_config
            .unpadded_vocab_size
            .unwrap_or(self.args.text_config.vocab_size);
        if vocabulary == self.args.text_config.vocab_size {
            Ok(logits)
        } else {
            logits
                .as_array()
                .try_index_device((.., .., ..vocabulary), stream)
                .map(crate::MlxTensor::from_array)
                .map_err(Into::into)
        }
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
                    input::Modality::Text => DecoderInputPart::Text(value),
                    input::Modality::Image => DecoderInputPart::Image(value),
                    input::Modality::Audio => DecoderInputPart::Audio(value),
                    input::Modality::Video => unreachable!(),
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
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        if state.target.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Inkling tensor-parallel cache layout mismatch".into(),
            ));
        }
        let (mut logits, context) = match &mut self.execution {
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
        let vocabulary = self
            .args
            .text_config
            .unpadded_vocab_size
            .unwrap_or(self.args.text_config.vocab_size);
        if vocabulary != self.args.text_config.vocab_size {
            logits = crate::MlxTensor::from_array(
                logits
                    .as_array()
                    .try_index_device((.., .., ..vocabulary), stream)?,
            );
        }
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
        group: &safemlx::distributed::Group,
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
                    input::Modality::Text => DecoderInputPart::Text(value),
                    input::Modality::Image => DecoderInputPart::Image(value),
                    input::Modality::Audio => DecoderInputPart::Audio(value),
                    input::Modality::Video => unreachable!(),
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
        group: &safemlx::distributed::Group,
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
    pub kinds: Vec<input::Modality>,
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
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(value)) => {
                    tokens.push(crate::MlxTensor::from_array(value.clone()));
                    kinds.push(input::Modality::Text);
                    projected.push(None);
                }
                (input::Modality::Image, input::InputPayload::Tensor(value)) => {
                    let architecture_input =
                        input::prepared_media_input(part.modality, value, part.metadata)?;
                    let plan = media_plan::inkling_ingress(args, &architecture_input)
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                    let count = usize::try_from(plan.placeholder_count).map_err(|_| {
                        Error::UnsupportedArchitecture(
                            "Inkling image placeholder span exceeds host capacity".into(),
                        )
                    })?;
                    tokens.push(crate::MlxTensor::from_array(input::token_ids_array(
                        &vec![plan.placeholder_token_id; count],
                        stream,
                    )?));
                    kinds.push(input::Modality::Image);
                    projected.push(None);
                    images.push(value.clone());
                }
                (input::Modality::Audio, input::InputPayload::Tensor(value)) => {
                    let architecture_input =
                        input::prepared_media_input(part.modality, value, part.metadata)?;
                    let plan = media_plan::inkling_ingress(args, &architecture_input)
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                    let count = usize::try_from(plan.placeholder_count).map_err(|_| {
                        Error::UnsupportedArchitecture(
                            "Inkling audio placeholder span exceeds host capacity".into(),
                        )
                    })?;
                    let retained_frames = i32::try_from(plan.placeholder_count).map_err(|_| {
                        Error::UnsupportedArchitecture(
                            "Inkling audio placeholder span exceeds tensor capacity".into(),
                        )
                    })?;
                    tokens.push(crate::MlxTensor::from_array(input::token_ids_array(
                        &vec![plan.placeholder_token_id; count],
                        stream,
                    )?));
                    kinds.push(input::Modality::Audio);
                    projected.push(None);
                    audio.push(value.try_index_device((.., ..retained_frames, ..), stream)?);
                }
                (
                    modality @ (input::Modality::Image | input::Modality::Audio),
                    input::InputPayload::Embeddings(value),
                ) => {
                    let count = value.dim(1);
                    let token = if modality == input::Modality::Image {
                        args.image_token_id
                    } else {
                        args.audio_token_id
                    };
                    tokens.push(crate::MlxTensor::from_array(Array::from_slice(
                        &vec![token; count as usize],
                        &[1, count],
                    )));
                    kinds.push(modality);
                    projected.push(Some(crate::MlxTensor::from_array(value.clone())));
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling does not accept this {} payload",
                        modality.as_str()
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
        cache.clear()?;
        self.model
            .forward_input_parallel_with_capture(input, cache, self.group, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
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
        for depth in 0..self.model.mtp_len() {
            let _ = self
                .model
                .forward_mtp_draft_parallel(&hidden, &next, depth, mtp, self.group, stream)?;
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
        let output = self.model.forward_mtp_draft_parallel(
            hidden,
            &token,
            draft_index,
            cache,
            self.group,
            stream,
        )?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..self.model.mtp_len() {
            let _ = self
                .model
                .forward_mtp_draft_parallel(hidden, tokens, depth, cache, self.group, stream)?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.model.mtp_len()
    }
}

fn resolve_store(
    store: SharedCheckpointSource,
    args: &ModelArgs,
) -> Result<SharedCheckpointSource, Error> {
    let plan = eredu_architectures::inkling::safetensors_plan(args)
        .map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|error| {
            Error::UnsupportedArchitecture(format!(
                "Inkling checkpoint contract did not resolve: {error:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

pub fn resolve_pipeline_store(
    store: SharedCheckpointSource,
    args: &ModelArgs,
) -> Result<SharedCheckpointSource, Error> {
    resolve_store(store, args)
}

pub fn prepare_gguf_pipeline_source(
    checkpoint: &GgufCheckpoint,
    projector: Option<&GgufCheckpoint>,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, ModelArgs), Error> {
    open_gguf_store(checkpoint, projector, metadata, max_cached_readers)
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
    let mut target = source.clone();
    target.text_config.weight_quantization = Some(quantization);
    target.text_config.quantized_weight_configs = None;
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
                crate::composition::inkling_expert::module_recipes(module, &static_args, store)?;
            build_module_bindings_with_recipes_excluding(module, "", store, recipes, |_| false)
                .map_err(Into::into)
        },
        move |_index, module, store| {
            let recipes =
                crate::composition::inkling_expert::module_recipes(module, &unit_args, store)?;
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
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<InklingModel, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
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
                crate::composition::inkling_expert::module_recipes(&module, &static_args, store)?;
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |_| false)
                .map_err(Into::into)
        },
        move |_ordinal, _address, _path, unit, store, _stream| {
            let module = MlxModule::new(unit);
            let recipes =
                crate::composition::inkling_expert::module_recipes(&module, &unit_args, store)?;
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |name| {
                external_experts && parameter_name_in_targets(name, &binding_expert_targets)
            })
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.text_config.weight_quantization);
    metadata.set_materialization(materialization);
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
    let state_layout = eredu_architectures::inkling::state_layout(&args)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let mtp_state_layout = eredu_architectures::inkling::mtp_state_layout(&args)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let prompt_state_layout = eredu_architectures::inkling::composite_state_layout(
        &state_layout,
        mtp_state_layout.as_ref(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(InklingModel {
        state_layout,
        mtp_state_layout,
        prompt_state_layout,
        args,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: None,
    })
}

fn load_parallel_store(
    store: SharedCheckpointSource,
    args: ModelArgs,
    layer_policy: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let global_execution =
        architecture_execution_layout::<_, MlxHybridState>(&global_architecture)?;
    let decoder_groups = (0..global_execution.group_count())
        .filter(|&group| {
            group_kind(&global_architecture, group) == eredu_runtime::ArchitectureGroupKind::Decoder
        })
        .collect::<Vec<_>>();
    let [decoder_group] = decoder_groups.as_slice() else {
        return Err(Error::Parallel(format!(
            "Inkling architecture declared {} decoder execution groups; expected one",
            decoder_groups.len()
        )));
    };
    let layer_count = global_execution
        .group_range(*decoder_group)
        .expect("validated execution group")
        .len();
    let mut planner = build.planner();
    for group in eredu_architectures::inkling::static_parameter_groups(&args)? {
        planner.register(group)?;
    }
    for group in eredu_architectures::inkling::mtp_parameter_groups(&args)? {
        planner.register(group)?;
    }
    for index in 0..layer_count {
        for group in eredu_architectures::inkling::layer_parameter_groups(&args, index)? {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
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
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let state_layout = architecture
        .runtime_state_layout()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let global_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &global_architecture,
        )
        .clone(),
    );
    let static_recipes =
        crate::composition::inkling_expert::module_recipes(&global_static, &args, store.as_ref())?;
    let global_static_bindings = build_module_bindings_with_recipes_excluding(
        &global_static,
        "",
        store.as_ref(),
        static_recipes,
        |_| false,
    )?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for ordinal in 0..global_execution.len() {
        let unit = MlxModule::new(construct_architecture_unit(
            &global_architecture,
            &global_execution,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxHybridState>,
        )?);
        let recipes =
            crate::composition::inkling_expert::module_recipes(&unit, &args, store.as_ref())?;
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
        move |_modules, store| {
            shard_layer_bindings(global_static_bindings, "", store, &static_layout)
        },
        move |_ordinal, address, path, _local, store, stream| {
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
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                );
            let recipes =
                crate::composition::inkling_expert::module_recipes(&global, &binding_args, store)?;
            let bindings =
                build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |_| {
                    false
                })?;
            shard_layer_bindings(bindings, path, store, &unit_sharding)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
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
    let mtp_state_layout = eredu_architectures::inkling::mtp_state_layout(&args)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let prompt_state_layout = eredu_architectures::inkling::composite_state_layout(
        &state_layout,
        mtp_state_layout.as_ref(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(InklingModel {
        args,
        state_layout,
        mtp_state_layout,
        prompt_state_layout,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: Some(parallel_info),
    })
}

/// Loads an Inkling SafeTensors checkpoint for pure tensor parallelism.
pub fn load_safetensors_tensor_parallel(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    layer_policy: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let args = ModelArgs::from_hf_json(&serde_json::to_vec(artifact.config()?)?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = artifact.store();
    let store = resolve_store(store, &args)?;
    load_parallel_store(store, args, layer_policy, build, stream, weights_stream)
}

/// Loads an Inkling GGUF checkpoint through the same neutral TP binder.
pub fn load_gguf_tensor_parallel(
    checkpoint: &GgufCheckpoint,
    projector: Option<&GgufCheckpoint>,
    metadata: &HashMap<String, GgufMetadataValue>,
    layer_policy: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let (store, args) = open_gguf_store(
        checkpoint,
        projector,
        metadata,
        layer_policy.max_mapped_shards(),
    )?;
    load_parallel_store(store, args, layer_policy, build, stream, weights_stream)
}

fn attach_expert_cache(
    model: &mut InklingModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = crate::composition::inkling_expert::expert_catalog(&model.args, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
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
    let expert_options = residency.expert_cache();
    let args = ModelArgs::from_hf_json(&serde_json::to_vec(artifact.config()?)?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = artifact.store();
    let store = resolve_store(store, &args)?;
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Inkling", args.text_config.weight_quantization, requested)
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

/// Loads the text GGUF and optional sibling media artifact into one neutral model.
pub fn load_gguf(
    checkpoint: &GgufCheckpoint,
    projector: Option<&GgufCheckpoint>,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingModel, Error> {
    let expert_options = residency.expert_cache();
    let (store, args) = open_gguf_store(
        checkpoint,
        projector,
        metadata,
        residency.max_mapped_shards(),
    )?;
    let layer_policy = residency.layers();
    let mut model = load_store(
        store,
        args,
        layer_policy,
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
    checkpoint: &GgufCheckpoint,
    projector: Option<&GgufCheckpoint>,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, ModelArgs), Error> {
    let mut args = ModelArgs::from_gguf_metadata(metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let translation_args = args.clone();
    let mut text_formats = gguf_quantization_configs(checkpoint, |name| {
        eredu_architectures::inkling::translate_gguf_weight_name_for_model(name, &translation_args)
    })?;
    eredu_architectures::inkling::normalize_gguf_weight_formats(&args, &mut text_formats)
        .map_err(Error::Quantization)?;
    args.text_config.quantized_weight_configs = (!text_formats.is_empty()).then_some(text_formats);
    let projector_metadata = projector.map(gguf_metadata);
    if let (Some(projector), Some(projector_metadata)) = (projector, &projector_metadata) {
        let formats = gguf_quantization_configs(
            projector,
            eredu_architectures::inkling::translate_mmproj_weight_name,
        )?;
        args = args
            .with_gguf_projector_metadata(metadata, projector_metadata, formats)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    }
    let text_plan =
        eredu_architectures::inkling::gguf_plan(&args).map_err(Error::UnsupportedArchitecture)?;
    let translation_args = args.clone();
    let mut builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_cached_readers)?
        .add_checkpoint(checkpoint.catalog().clone(), &text_plan, move |name| {
            eredu_architectures::inkling::translate_gguf_weight_name_for_model(
                name,
                &translation_args,
            )
        })?;
    if let Some(projector) = projector {
        let plan = eredu_architectures::inkling::mmproj_gguf_plan(&args)
            .map_err(Error::UnsupportedArchitecture)?;
        builder = builder.add_checkpoint(projector.catalog().clone(), &plan, |name| {
            eredu_architectures::inkling::translate_mmproj_weight_name(name)
        })?;
    }
    let store: SharedCheckpointSource = Arc::new(builder.build()?);
    Ok((store, args))
}
