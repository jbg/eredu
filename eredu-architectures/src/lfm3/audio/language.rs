//! Shared language traversal for speech-conditioned prompts.

use super::{artifact::checkpoint, ArtifactError, AudioPrompt, BindingSource};
use crate::lfm3::{Block, CompleteLfm3Kernel, Declaration, LayeredModel};
use eredu_core::{BackendFailure, ParallelRankTopology};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Tensor,
    TensorParallelGroupedNeuralBackend,
};
use eredu_runtime::{
    ComponentDefinition, LayerRuntimeState, LayerwisePolicy, LayerwiseRuntime, PreparedComponent,
    RuntimeStateComponents, StateLayout, SubmissionBackend,
};
use std::sync::Arc;

type Context<B> = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context;

/// Exact language resources for the Audio checkpoint's F32 profile.
pub struct PreparedAudioLanguage {
    declaration: Arc<Declaration>,
    parameters: PreparedComponent<Arc<ComponentDefinition>>,
}

impl PreparedAudioLanguage {
    pub(super) fn new(
        declaration: Arc<Declaration>,
        weights: &BindingSource,
    ) -> Result<Self, ArtifactError> {
        let parameters = PreparedComponent::prepare(
            "lfm3/audio/language",
            declaration.definition().clone(),
            weights.source().clone(),
            |member| {
                weights
                    .recipes()
                    .get(member.target())
                    .cloned()
                    .ok_or_else(|| {
                        eredu_runtime::ComponentPreparationError::ArchitectureMismatch(format!(
                            "LFM3 Audio language parameter {}",
                            member.target()
                        ))
                    })
            },
        )
        .map_err(checkpoint)?;
        Ok(Self {
            declaration,
            parameters,
        })
    }

    /// Constructs the retained language declaration and applies native storage preferences.
    pub fn into_layerwise<B, S>(
        self,
        context: &Context<B>,
    ) -> Result<(LayeredModel<B>, PreparedComponent<Arc<ComponentDefinition>>), BackendFailure>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.into_selected::<B, S>(None, context)
    }

    /// Applies the retained backbone's tensor placement and constructs rank-local modules.
    pub fn into_tensor_parallel<B, S>(
        self,
        topology: ParallelRankTopology,
        context: &Context<B>,
    ) -> Result<(LayeredModel<B>, PreparedComponent<Arc<ComponentDefinition>>), BackendFailure>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.into_selected::<B, S>(Some(topology), context)
    }

    fn into_selected<B, S>(
        self,
        topology: Option<ParallelRankTopology>,
        context: &Context<B>,
    ) -> Result<(LayeredModel<B>, PreparedComponent<Arc<ComponentDefinition>>), BackendFailure>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let (parameters, local) = match topology {
            Some(rank) => {
                let layout = self
                    .parameters
                    .tensor_parallel_layout(rank)
                    .map_err(BackendFailure::from_error)?;
                let geometry = crate::lfm3::local_geometry(&self.declaration, &layout)
                    .map_err(BackendFailure::from_error)?;
                let parameters = self
                    .parameters
                    .select(&layout)
                    .map_err(BackendFailure::from_error)?;
                (parameters, Some(geometry))
            }
            None => (self.parameters, None),
        };
        let model = CompleteLfm3Kernel::<B>::from_selected_declaration(
            self.declaration,
            parameters.declaration().clone(),
            local,
            context,
        )
        .map_err(BackendFailure::from_error)?;
        let parameters = parameters
            .with_storage_preferences::<B, S, _>(&model, context)
            .map_err(BackendFailure::from_error)?;
        Ok((model, parameters))
    }
}

/// Language operations used by ordinary and controlled Audio execution.
pub trait AudioLanguage<B: GroupedNeuralBackend + DistributedNeuralBackend, S> {
    /// Whether the selected language owner commits a distributed generation step.
    fn uses_collective_commit(&self) -> bool {
        false
    }

    /// Agrees the phase result and decision identity before another rank enters transport.
    fn agree_audio(&mut self, _descriptor: [u64; 8], _success: bool) -> Result<(), BackendFailure> {
        Ok(())
    }

    /// Fixes the all-rank decision for a completed candidate generation step.
    fn commit_audio(&mut self) -> Result<(), BackendFailure> {
        Ok(())
    }

    /// Layout of the selected language state, including rank-local KV and convolution carry.
    fn state_layout(&self) -> &StateLayout;

    /// Runs the shared causal traversal with speech rows substituted at ingress.
    fn forward_prompt(
        &mut self,
        prompt: &AudioPrompt<'_, B::Tensor>,
        state: &mut S,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure>;

    /// Applies the selected vocabulary projection to normalized language states.
    fn project_normalized(
        &mut self,
        hidden: &B::Tensor,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure>;
}

impl<B, S, P> AudioLanguage<B, S> for LayerwiseRuntime<LayeredModel<B>, B, S, P>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: LayerwisePolicy<B, Block<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn state_layout(&self) -> &StateLayout {
        self.architecture().definition().state_layout()
    }

    fn forward_prompt(
        &mut self,
        prompt: &AudioPrompt<'_, B::Tensor>,
        state: &mut S,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        prompt
            .forward_layerwise(self, state, context)
            .map_err(BackendFailure::from_error)
    }

    fn project_normalized(
        &mut self,
        hidden: &B::Tensor,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        self.architecture_mut()
            .kernel_mut()
            .static_modules_mut()
            .project_normalized(hidden, context)
            .map_err(BackendFailure::from_error)
    }
}

/// Rank-local language traversal. The enclosing distributed session owns
/// collective ordering and agreement across the cohort.
pub struct TensorParallelAudioLanguage<B, S, P>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: LayerwisePolicy<B, Block<B>>,
{
    runtime: LayerwiseRuntime<LayeredModel<B>, B, S, P>,
    collective: B::ParallelContext,
}

impl<B, S, P> TensorParallelAudioLanguage<B, S, P>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: LayerwisePolicy<B, Block<B>>,
    P::Error: std::fmt::Display,
{
    /// Attaches a collective supplied by the enclosing rank session.
    pub fn new(
        runtime: LayerwiseRuntime<LayeredModel<B>, B, S, P>,
        collective: B::ParallelContext,
    ) -> Result<Self, BackendFailure> {
        if runtime
            .architecture()
            .kernel()
            .parallel_geometry()
            .is_none()
        {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "Audio tensor traversal requires rank-local language geometry",
            )));
        }
        Ok(Self {
            runtime,
            collective,
        })
    }
}

impl<B, S, P> AudioLanguage<B, S> for TensorParallelAudioLanguage<B, S, P>
where
    B: TensorParallelGroupedNeuralBackend
        + DistributedNeuralBackend
        + SubmissionBackend<Executor = Context<B>>,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: LayerwisePolicy<B, Block<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn state_layout(&self) -> &StateLayout {
        self.runtime.architecture().definition().state_layout()
    }

    fn forward_prompt(
        &mut self,
        prompt: &AudioPrompt<'_, B::Tensor>,
        state: &mut S,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        prompt
            .forward_tensor_parallel(&mut self.runtime, state, &self.collective, context)
            .map_err(BackendFailure::from_error)
    }

    fn project_normalized(
        &mut self,
        hidden: &B::Tensor,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        self.runtime
            .architecture_mut()
            .kernel_mut()
            .static_modules_mut()
            .project_normalized_parallel(hidden, &self.collective, context)
            .map_err(BackendFailure::from_error)
    }
}

/// The retained source and exact resource owners for one language pipeline stage.
pub struct PreparedAudioLanguagePartition {
    rank: ParallelRankTopology,
    declaration: Arc<Declaration>,
    partition: eredu_runtime::ArchitecturePartition<
        crate::lfm3::PartitionLocalGeometry,
        eredu_runtime::NoAuxiliaryBoundarySchema,
    >,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    units: Vec<Vec<eredu_runtime::WeightBinding>>,
    pinned: Vec<eredu_runtime::WeightBinding>,
}

impl PreparedAudioLanguage {
    /// Selects language-only pipeline placement with optional tensor sharding.
    /// This retains no authority for speech components or inter-rank transport.
    pub fn select_pipeline_rank(
        self,
        rank: ParallelRankTopology,
    ) -> Result<PreparedAudioLanguagePartition, BackendFailure> {
        use eredu_runtime::{ArchitecturePartition, NoAuxiliaryBoundarySchema, PartitionOwnership};

        if rank.expert_parallel_size() != 1 || rank.data_parallel_size() != 1 {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "Audio language partitioning requires tensor and pipeline axes only",
            )));
        }
        let definition = self.declaration.definition();
        let range = rank
            .layer_range(definition.state_layout().len())
            .map_err(BackendFailure::from_error)?;
        let layout = self
            .parameters
            .parallel_layout(rank)
            .map_err(BackendFailure::from_error)?;
        let geometry =
            crate::lfm3::partition_local_geometry(&self.declaration, &layout, range.clone())
                .map_err(BackendFailure::from_error)?;
        let transport = crate::transport::decoder();
        let mut static_roles = Vec::new();
        if rank.owns_embedding() {
            static_roles.extend(transport.first_owner_static_roles);
        }
        if rank.owns_output_head() {
            static_roles.extend(transport.last_owner_static_roles);
        }
        let ownership =
            PartitionOwnership::new(rank.owns_embedding(), rank.owns_output_head(), static_roles)
                .map_err(BackendFailure::from_error)?;
        let state = geometry.complete_state_layout().clone();
        let partition = ArchitecturePartition::from_description(
            definition.parameters(),
            [(crate::decoder::TARGET_EXECUTION_GROUP, range)],
            ownership,
            &state,
            &crate::transport::pipeline_state(0, &state),
            geometry,
            NoAuxiliaryBoundarySchema::new(self.declaration.args().hidden_size),
        )
        .map_err(BackendFailure::from_error)?;
        let parameters = self
            .parameters
            .select(&layout)
            .map_err(BackendFailure::from_error)?
            .select_units(partition.units())
            .map_err(BackendFailure::from_error)?;
        let (_, invocations, source, units, pinned) = parameters.into_parts();
        if &invocations != partition.invocations() {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "retained Audio language invocations differ from the selected pipeline stage",
            )));
        }
        let targets = partition
            .parameter_bindings()
            .iter()
            .flat_map(|group| group.members())
            .map(|member| member.target())
            .collect::<std::collections::BTreeSet<_>>();
        let pinned = pinned
            .into_values()
            .flatten()
            .filter(|binding| targets.contains(binding.logical_target().unwrap_or(binding.name())))
            .collect::<Vec<_>>();
        let retained_targets = units
            .iter()
            .flatten()
            .chain(&pinned)
            .map(|binding| binding.logical_target().unwrap_or(binding.name()))
            .collect::<std::collections::BTreeSet<_>>();
        if retained_targets != targets {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "retained Audio language parameters differ from the selected pipeline stage",
            )));
        }
        let keys = units
            .iter()
            .flatten()
            .chain(&pinned)
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned))
            .collect();
        let source = Arc::new(
            eredu_checkpoint::store::RestrictedCheckpointSource::including(
                source,
                "lfm3/audio/language/pipeline",
                keys,
            )
            .map_err(BackendFailure::from_error)?,
        );
        Ok(PreparedAudioLanguagePartition {
            rank,
            declaration: self.declaration,
            partition,
            source,
            units,
            pinned,
        })
    }
}

impl PreparedAudioLanguagePartition {
    pub(super) fn rank(&self) -> ParallelRankTopology {
        self.rank
    }

    pub(super) fn args(&self) -> &crate::lfm3::ModelArgs {
        self.declaration.args()
    }

    /// The retained semantic and state ownership for the local language stage.
    pub fn partition(
        &self,
    ) -> &eredu_runtime::ArchitecturePartition<
        crate::lfm3::PartitionLocalGeometry,
        eredu_runtime::NoAuxiliaryBoundarySchema,
    > {
        &self.partition
    }

    /// Builds the stage-local modules and applies their physical storage preferences.
    pub fn into_layerwise<B, S>(
        self,
        context: &Context<B>,
    ) -> Result<
        (
            crate::lfm3::PartitionedLayeredModel<B>,
            eredu_checkpoint::store::SharedCheckpointSource,
            eredu_runtime::ExecutionUnitAddressMap,
            Vec<eredu_runtime::WeightBinding>,
            Vec<Vec<eredu_runtime::WeightBinding>>,
        ),
        BackendFailure,
    >
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        use eredu_runtime::LayeredArchitecture;

        let definition = self.declaration.definition().clone();
        let model = crate::lfm3::Lfm3Kernel::<B>::from_partition(
            self.declaration,
            definition,
            &self.partition,
            context,
        )
        .map_err(BackendFailure::from_error)?;
        let pinned = eredu_runtime::apply_parameter_storage_layouts(
            <_ as LayeredArchitecture<B, S>>::static_modules(&model),
            self.pinned,
        );
        let invocations = self.partition.invocations().clone();
        let units = self
            .units
            .into_iter()
            .enumerate()
            .map(|(ordinal, bindings)| {
                let location = invocations
                    .storage_location(ordinal)
                    .expect("retained owner");
                let module =
                    <_ as LayeredArchitecture<B, S>>::build_storage_unit(&model, location, context)
                        .map_err(BackendFailure::from_error)?;
                Ok(eredu_runtime::apply_parameter_storage_layouts(
                    &module, bindings,
                ))
            })
            .collect::<Result<_, BackendFailure>>()?;
        Ok((model, self.source, invocations, pinned, units))
    }
}
