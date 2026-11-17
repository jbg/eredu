//! Shared language traversal for speech-conditioned prompts.

use super::{artifact::checkpoint, ArtifactError, AudioPlacement, AudioPrompt, BindingSource};
use crate::lfm3::{CompleteLfm3Kernel, Declaration, LayeredModel};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Tensor,
    TensorParallelGroupedNeuralBackend,
};
use eredu_runtime::{
    BoundComponentPipeline, BoundLocalComponent, ComponentDefinition, ComponentRankMechanisms,
    LayerRuntimeState, LayerWeightResidency, LocalComponentMechanisms, NoopObserver,
    PreparedComponent, RuntimeStateComponents, StateLayout,
};
use std::sync::Arc;

type Context<B> = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context;

/// Exact cold language plan consumed by a binding for the selected route.
pub struct PreparedAudioLanguage(AudioLanguagePlan);

enum AudioLanguagePlan {
    Local(LocalLanguage),
    Pipeline(PipelineLanguage),
}

struct LocalLanguage {
    declaration: Arc<Declaration>,
    parameters: PreparedComponent<Arc<ComponentDefinition>>,
    geometry: Option<crate::lfm3::LocalGeometry>,
}

struct PipelineLanguage {
    stage: crate::lfm3::PreparedLanguagePartition,
    communication: crate::lfm3::LanguagePipelineCommunication,
    layout: StateLayout,
}

impl PreparedAudioLanguage {
    pub(super) fn new(
        declaration: Arc<Declaration>,
        weights: &BindingSource,
        dtype: &eredu_checkpoint::recipe::RecipeDtype,
        placement: AudioPlacement,
    ) -> Result<Self, ArtifactError> {
        let parameters = PreparedComponent::prepare(
            "lfm3/audio/language",
            declaration.definition().clone(),
            weights.source().clone(),
            |member| {
                Ok::<_, eredu_runtime::ComponentPreparationError>(
                    super::placement::audio_storage_recipe(weights, member, dtype),
                )
            },
        )
        .map_err(checkpoint)?;
        if let AudioPlacement::Pipeline(rank) = placement {
            let stage =
                crate::lfm3::PreparedLanguagePartition::select(declaration, parameters, rank, None)
                    .map_err(checkpoint)?;
            let communication = stage.pipeline_communication().map_err(checkpoint)?;
            let layout = stage
                .partition()
                .state()
                .ok_or_else(|| checkpoint("Audio pipeline stage has no cache layout"))?
                .layout()
                .clone();
            return Ok(Self(AudioLanguagePlan::Pipeline(PipelineLanguage {
                stage,
                communication,
                layout,
            })));
        }

        let (parameters, geometry) = if let AudioPlacement::Tensor(rank) = placement {
            let layout = parameters
                .tensor_parallel_layout(rank)
                .map_err(checkpoint)?;
            let geometry =
                crate::lfm3::local_geometry(&declaration, &layout).map_err(checkpoint)?;
            let parameters = parameters
                .select(&layout)
                .map_err(checkpoint)?
                .select_state_layout(geometry.state_layout())
                .map_err(checkpoint)?;
            (parameters, Some(geometry))
        } else {
            (parameters, None)
        };
        Ok(Self(AudioLanguagePlan::Local(LocalLanguage {
            declaration,
            parameters,
            geometry,
        })))
    }

    /// Exact language state layout selected without native allocation.
    pub(super) fn state_layout(&self) -> &StateLayout {
        match &self.0 {
            AudioLanguagePlan::Local(local) => local.parameters.declaration().state_layout(),
            AudioLanguagePlan::Pipeline(pipeline) => &pipeline.layout,
        }
    }

    /// Sizes the rank's exact pinned and leased weights and admits the device window.
    pub(super) fn parameter_bytes(
        &self,
        residency: LayerWeightResidency,
    ) -> Result<u64, super::ResidentAudioError> {
        use eredu_runtime::physical_binding_bytes;
        let (invocations, units, pinned) = match &self.0 {
            AudioLanguagePlan::Local(local) => {
                let pinned = local.parameters.pinned_bindings().values().try_fold(
                    0u64,
                    |sum, bindings| {
                        let bytes = physical_binding_bytes(bindings).map_err(checkpoint)?;
                        sum.checked_add(bytes)
                            .ok_or_else(|| checkpoint("Audio pinned byte count overflow"))
                    },
                )?;
                (
                    local.parameters.invocations(),
                    local.parameters.unit_bindings(),
                    pinned,
                )
            }
            AudioLanguagePlan::Pipeline(pipeline) => {
                let (units, pinned) = pipeline.stage.bindings();
                (
                    pipeline.stage.partition().invocations(),
                    units,
                    physical_binding_bytes(pinned).map_err(checkpoint)?,
                )
            }
        };
        super::placement::admit_audio_weights("language", invocations, units, pinned, residency)
    }

    /// Constructs the selected local or tensor-parallel component and its exact parameter policy.
    fn prepare_layerwise<B, S>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
        context: &Context<B>,
    ) -> Result<eredu_runtime::PreparedLocalComponent<LayeredModel<B>>, BackendFailure>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        use eredu_core::residency::ParameterConversionRetentionEligibility as Eligibility;
        use eredu_runtime::LayerWeightResidency;

        let (architecture, parameters) = self.into_selected::<B, S>(context)?;
        let conversion_eligibility = match residency {
            LayerWeightResidency::LayerwiseHost(_) => Eligibility::HostLayerwise,
            LayerWeightResidency::DenseDiskStream(_) => Eligibility::DiskStreamed,
            _ => Eligibility::Eligible,
        };
        eredu_runtime::PreparedLocalComponent::new(
            architecture,
            parameters,
            eredu_runtime::LocalComponentPolicy {
                residency,
                state: None,
                conversion_retention: None,
                conversion_eligibility,
            },
        )
        .map_err(BackendFailure::from_error)
    }

    fn into_selected<B, S>(
        self,
        context: &Context<B>,
    ) -> Result<(LayeredModel<B>, PreparedComponent<Arc<ComponentDefinition>>), BackendFailure>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let AudioLanguagePlan::Local(LocalLanguage {
            declaration,
            parameters,
            geometry,
        }) = self.0
        else {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "Audio local binding requires a retained local language plan",
            )));
        };
        let model = CompleteLfm3Kernel::<B>::from_selected_declaration(
            declaration,
            parameters.declaration().clone(),
            geometry,
            context,
        )
        .map_err(BackendFailure::from_error)?;
        let parameters = parameters
            .with_storage_preferences::<B, S, _>(&model, context)
            .map_err(BackendFailure::from_error)?;
        Ok((model, parameters))
    }
}

/// A speech-generation invocation with audio-only or joint Omni conditioning.
pub enum SpeechPrompt<'a, T: Tensor> {
    /// Audio input and speech feedback.
    Audio(&'a AudioPrompt<'a, T>),
    /// Vision and speech input with normalized hidden readout.
    Omni(&'a crate::lfm3::omni::OmniPrompt<'a, T>),
}

impl<'a, T: Tensor> SpeechPrompt<'a, T> {
    /// Borrows this prompt as a shared language invocation.
    pub fn input(&self) -> crate::lfm3::Lfm3Input<'a, T> {
        match self {
            Self::Audio(prompt) => crate::lfm3::Lfm3Input::Audio(prompt),
            Self::Omni(prompt) => crate::lfm3::Lfm3Input::Omni(prompt),
        }
    }

    /// Returns the common invocation identity for rank agreement.
    pub fn collective_descriptor(&self, context: &T::Context) -> Result<[u64; 8], eredu_nn::Error> {
        match self {
            Self::Audio(prompt) => prompt.collective_descriptor(context),
            Self::Omni(prompt) => prompt.collective_descriptor(context),
        }
    }

    /// Visits the tensor roots retained by the selected invocation.
    pub fn retained_tensors(&self) -> impl Iterator<Item = &T> {
        let (audio, omni) = match self {
            Self::Audio(prompt) => (Some(*prompt), None),
            Self::Omni(prompt) => (None, Some(*prompt)),
        };
        audio
            .into_iter()
            .flat_map(AudioPrompt::retained_tensors)
            .chain(
                omni.into_iter()
                    .flat_map(crate::lfm3::omni::OmniPrompt::retained_tensors),
            )
    }
}

/// Language operations used by ordinary and controlled Audio execution.
pub trait AudioLanguage<B: GroupedNeuralBackend + DistributedNeuralBackend, S> {
    /// Whether the selected language owner commits a distributed generation step.
    fn uses_collective_commit(&self) -> bool {
        false
    }

    /// Agrees the phase result and decision identity before another rank enters transport.
    fn agree_audio(
        &mut self,
        _descriptor: [u64; 8],
        _success: bool,
        _context: &Context<B>,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }

    /// Fixes the all-rank decision for a completed candidate generation step.
    fn commit_audio(&mut self, _context: &Context<B>) -> Result<(), BackendFailure> {
        Ok(())
    }

    /// Layout of the selected language state, including rank-local KV and convolution carry.
    fn state_layout(&self) -> &StateLayout;

    /// Runs the shared causal traversal with speech rows substituted at ingress.
    fn forward_prompt(
        &mut self,
        prompt: SpeechPrompt<'_, B::Tensor>,
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

/// Replicated traversal over the shared local component.
pub struct LocalAudioLanguage<L>(L);

impl<B, S, L> AudioLanguage<B, S> for LocalAudioLanguage<L>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    L: BoundLocalComponent<LayeredModel<B>, B, S>,
{
    fn state_layout(&self) -> &StateLayout {
        self.0.state_layout()
    }

    fn forward_prompt(
        &mut self,
        prompt: SpeechPrompt<'_, B::Tensor>,
        state: &mut S,
        _: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        self.0
            .execute(prompt.input(), state, &mut NoopObserver)
            .map_err(BackendFailure::from_error)
    }

    fn project_normalized(
        &mut self,
        hidden: &B::Tensor,
        _: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        self.0
            .readout(|architecture, context| {
                architecture
                    .kernel_mut()
                    .static_modules_mut()
                    .project_normalized(hidden, context)
            })
            .map_err(BackendFailure::from_error)
    }
}

/// Rank-local traversal over a shared local component and its tensor collective.
pub struct TensorParallelAudioLanguage<L, G> {
    component: L,
    collective: G,
}

impl<B, S, L, G> AudioLanguage<B, S> for TensorParallelAudioLanguage<L, G>
where
    B: TensorParallelGroupedNeuralBackend<ParallelContext = G> + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    L: BoundLocalComponent<LayeredModel<B>, B, S>,
{
    fn state_layout(&self) -> &StateLayout {
        self.component.state_layout()
    }

    fn forward_prompt(
        &mut self,
        prompt: SpeechPrompt<'_, B::Tensor>,
        state: &mut S,
        _: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        self.component
            .execute_parallel(prompt.input(), state, &self.collective)
            .map_err(BackendFailure::from_error)
    }

    fn project_normalized(
        &mut self,
        hidden: &B::Tensor,
        _: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        self.component
            .readout(|architecture, context| {
                architecture
                    .kernel_mut()
                    .static_modules_mut()
                    .project_normalized_parallel(hidden, &self.collective, context)
            })
            .map_err(BackendFailure::from_error)
    }
}

/// Built Audio rank. The same layout and output facts accompany the prepared and bound stages.
struct AudioLanguageStage<P> {
    /// Portable pipeline plan or its bound native realization.
    pipeline: P,
    /// Cache geometry for this stage.
    layout: StateLayout,
    /// Whether this rank's stage owns the language readout.
    owns_output: bool,
    /// Width of the selected vocabulary readout.
    vocabulary: i32,
}

impl<P> AudioLanguageStage<P> {
    /// Realizes the retained plan while preserving the stage's execution facts.
    fn bind<Q>(
        self,
        bind: impl FnOnce(P) -> Result<Q, BackendFailure>,
    ) -> Result<AudioLanguageStage<Q>, BackendFailure> {
        Ok(AudioLanguageStage {
            pipeline: bind(self.pipeline)?,
            layout: self.layout,
            owns_output: self.owns_output,
            vocabulary: self.vocabulary,
        })
    }
}

impl PreparedAudioLanguage {
    /// Constructs the retained pipeline for a normalized Audio hidden readout.
    fn prepare_pipeline<B, S>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
        context: &Context<B>,
    ) -> Result<
        AudioLanguageStage<
            eredu_runtime::PreparedComponentPipeline<crate::lfm3::PartitionedLayeredModel<B>>,
        >,
        BackendFailure,
    >
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let AudioLanguagePlan::Pipeline(PipelineLanguage {
            stage: prepared,
            communication,
            layout,
        }) = self.0
        else {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "Audio pipeline binding requires a retained pipeline plan",
            )));
        };
        let owns_output = prepared.partition().ownership().owns_output();
        let (architecture, source, invocations, pinned, units) =
            prepared.into_layerwise::<B, S>(context)?;
        let vocabulary = architecture.kernel().args().vocab_size;
        let units = units
            .into_iter()
            .enumerate()
            .map(|(ordinal, unit)| {
                let address = invocations
                    .storage_layout()
                    .address(ordinal)
                    .ok_or_else(|| {
                        BackendFailure::from_error(eredu_nn::Error::backend(
                            "Audio pipeline binding has no retained storage owner",
                        ))
                    })?;
                Ok((address, unit))
            })
            .collect::<Result<_, BackendFailure>>()?;
        Ok(AudioLanguageStage {
            pipeline: eredu_runtime::PreparedComponentPipeline {
                architecture,
                source,
                invocations,
                units,
                pinned,
                manifest: communication.manifest,
                plan: communication.plan,
                tensor_group: communication.tensor_group,
                activation_dtype: eredu_runtime::PipelineActivationDtype::Float32,
                residency,
            },
            layout,
            owns_output,
            vocabulary,
        })
    }
}

/// Audio language execution on a selected pipeline rank.
pub struct PipelineAudioLanguage<P> {
    stage: AudioLanguageStage<P>,
    epoch: DistributedCommitEpoch,
}

impl<P> PipelineAudioLanguage<P> {
    /// Installs a bound stage at the first commit epoch.
    fn new(stage: AudioLanguageStage<P>) -> Self {
        Self {
            stage,
            epoch: DistributedCommitEpoch::FIRST,
        }
    }

    fn next_epoch(&mut self) -> Result<DistributedCommitEpoch, BackendFailure> {
        let epoch = self.epoch;
        self.epoch = epoch.next().ok_or_else(|| {
            BackendFailure::from_error(eredu_nn::Error::backend(
                "Audio distributed commit epoch exhausted",
            ))
        })?;
        Ok(epoch)
    }
}

impl<B, S, P> AudioLanguage<B, S> for PipelineAudioLanguage<P>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B> + eredu_nn::RetainedTensorRoots<B::Tensor>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: BoundComponentPipeline<crate::lfm3::PartitionedLayeredModel<B>, B, S>,
{
    fn state_layout(&self) -> &StateLayout {
        &self.stage.layout
    }

    fn forward_prompt(
        &mut self,
        prompt: SpeechPrompt<'_, B::Tensor>,
        state: &mut S,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        let descriptor = prompt
            .collective_descriptor(context)
            .map_err(BackendFailure::from_error);
        let agreed = self.stage.pipeline.agree(
            [0x415544494f505245, 0, 0, 0, 0, 0, 0, 0],
            descriptor.is_ok(),
            context,
        );
        let descriptor = descriptor?;
        agreed?;
        let epoch = self.next_epoch()?;
        let (hidden, outcome) = self.stage.pipeline.execute(
            prompt.input(),
            prompt.retained_tensors().cloned().collect(),
            state,
            descriptor,
            epoch,
            context,
        )?;
        if outcome != DistributedCommitOutcome::Committed(epoch) {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "Audio language pipeline did not settle",
            )));
        }
        Ok(hidden)
    }

    fn project_normalized(
        &mut self,
        hidden: &B::Tensor,
        context: &Context<B>,
    ) -> Result<B::Tensor, BackendFailure> {
        let owns_output = self.stage.owns_output;
        let vocabulary = self.stage.vocabulary;
        self.stage.pipeline.publish_readout(
            vec![hidden.clone()],
            context,
            |architecture, parallel| {
                if owns_output {
                    architecture
                        .kernel_mut()
                        .project_audio_position(hidden, parallel, context)
                } else {
                    B::Tensor::full_f32(0.0, &[hidden.dim(0), vocabulary], context)
                }
                .map_err(BackendFailure::from_error)
            },
        )
    }

    fn uses_collective_commit(&self) -> bool {
        true
    }

    fn agree_audio(
        &mut self,
        descriptor: [u64; 8],
        success: bool,
        context: &Context<B>,
    ) -> Result<(), BackendFailure> {
        self.stage.pipeline.agree(descriptor, success, context)
    }

    fn commit_audio(&mut self, context: &Context<B>) -> Result<(), BackendFailure> {
        let epoch = self.next_epoch()?;
        self.stage.pipeline.commit_candidate(epoch, context)
    }
}

mod placement_seal {
    pub trait Sealed {}
    impl Sealed for super::ReplicatedAudio {}
    impl<G> Sealed for super::TensorParallelAudio<G> {}
    impl<C> Sealed for super::PipelineAudio<C> {}
}

/// Portable construction of the language traversal for a selected placement.
pub trait AudioLanguagePlacement<B, S, H>: placement_seal::Sealed
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
{
    /// Whether this placement can restore or fork without cohort coordination.
    const INDEPENDENT_SNAPSHOTS: bool = false;

    /// Language traversal installed in the resident Audio driver.
    type Execution: AudioLanguage<B, S>;

    /// Whether the mechanism binds this retained route.
    fn accepts(placement: AudioPlacement) -> bool;

    /// Prepares the exact architecture component and binds its native mechanisms.
    fn bind(
        &self,
        prepared: PreparedAudioLanguage,
        residency: LayerWeightResidency,
        context: &Context<B>,
        mechanisms: &H,
    ) -> Result<Self::Execution, BackendFailure>;
}

/// Replicated language execution.
pub struct ReplicatedAudio;

impl<B, S, H> AudioLanguagePlacement<B, S, H> for ReplicatedAudio
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    H: LocalComponentMechanisms<B, S>,
{
    const INDEPENDENT_SNAPSHOTS: bool = true;

    type Execution = LocalAudioLanguage<H::Local<LayeredModel<B>>>;

    fn accepts(placement: AudioPlacement) -> bool {
        matches!(placement, AudioPlacement::Local)
    }

    fn bind(
        &self,
        prepared: PreparedAudioLanguage,
        residency: LayerWeightResidency,
        context: &Context<B>,
        mechanisms: &H,
    ) -> Result<Self::Execution, BackendFailure> {
        mechanisms
            .bind_local(|| prepared.prepare_layerwise::<B, S>(residency, context))
            .map(LocalAudioLanguage)
    }
}

/// Tensor-sharded language execution and its matching collective.
pub struct TensorParallelAudio<G> {
    collective: G,
}

impl<G> TensorParallelAudio<G> {
    /// Binds the native collective for a retained tensor rank.
    pub fn new(collective: G) -> Self {
        Self { collective }
    }
}

impl<B, S, H, G> AudioLanguagePlacement<B, S, H> for TensorParallelAudio<G>
where
    B: TensorParallelGroupedNeuralBackend<ParallelContext = G> + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    H: LocalComponentMechanisms<B, S>,
    G: Clone,
{
    type Execution = TensorParallelAudioLanguage<H::Local<LayeredModel<B>>, G>;

    fn accepts(placement: AudioPlacement) -> bool {
        matches!(placement, AudioPlacement::Tensor(_))
    }

    fn bind(
        &self,
        prepared: PreparedAudioLanguage,
        residency: LayerWeightResidency,
        context: &Context<B>,
        mechanisms: &H,
    ) -> Result<Self::Execution, BackendFailure> {
        let component =
            mechanisms.bind_local(|| prepared.prepare_layerwise::<B, S>(residency, context))?;
        Ok(TensorParallelAudioLanguage {
            component,
            collective: self.collective.clone(),
        })
    }
}

/// Pipeline or combined parallel language placement with its communication owner.
pub struct PipelineAudio<C> {
    communication: C,
}

impl<C> PipelineAudio<C> {
    /// Binds the communication owner for a retained stage rank.
    pub fn new(communication: C) -> Self {
        Self { communication }
    }
}

impl<B, S, H, C> AudioLanguagePlacement<B, S, H> for PipelineAudio<C>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B> + eredu_nn::RetainedTensorRoots<B::Tensor>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    H: ComponentRankMechanisms<B, S, C>,
{
    type Execution = PipelineAudioLanguage<H::Pipeline<crate::lfm3::PartitionedLayeredModel<B>>>;

    fn accepts(placement: AudioPlacement) -> bool {
        matches!(placement, AudioPlacement::Pipeline(_))
    }

    fn bind(
        &self,
        prepared: PreparedAudioLanguage,
        residency: LayerWeightResidency,
        context: &Context<B>,
        mechanisms: &H,
    ) -> Result<Self::Execution, BackendFailure> {
        let stage = prepared
            .prepare_pipeline::<B, S>(residency, context)?
            .bind(|pipeline| {
                mechanisms
                    .bind_pipeline(pipeline, &self.communication)
                    .map(|(execution, _)| execution)
            })?;
        Ok(PipelineAudioLanguage::new(stage))
    }
}
