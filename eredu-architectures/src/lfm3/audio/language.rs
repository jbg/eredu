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

pub use crate::lfm3::PreparedLanguagePartition as PreparedAudioLanguagePartition;

impl PreparedAudioLanguage {
    /// Selects the shared LFM3 pipeline for normalized Audio hidden output.
    pub fn select_pipeline_rank(
        self,
        rank: ParallelRankTopology,
    ) -> Result<PreparedAudioLanguagePartition, BackendFailure> {
        crate::lfm3::PreparedLanguagePartition::select(
            self.declaration,
            self.parameters,
            rank,
            None,
        )
    }
}
