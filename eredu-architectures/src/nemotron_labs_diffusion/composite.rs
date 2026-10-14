//! Pixtral context preparation and traversal of the singular text backbone.
/// Exact owned image/text construction and pipeline invocation transport.
pub mod partition;

use super::{
    FamilyConfig,
    input::{InputPartPlan, admit_input},
    model::{ForwardContext as TextForward, TextLayer, TextModel},
    vision::{ImageEmbedding, ImageInput, ImageLayer, ImageProjector, ImageState},
};
use crate::{
    composite_execution::{
        CompositeArchitecture, ParallelCompositeArchitecture, PreparedCompositeInput,
        SingleCompositePrefillRequest,
    },
    decoder::{Config, PartitionedConfig},
    media_plan::AdmittedCompositeInput,
};
use eredu_core::decoding::AttentionInvocation;
use eredu_nn::{
    DistributedNeuralBackend, EmbeddingLookupPolicy, Error, Index, Parameterized,
    ReadOnlyAttentionCache, Tensor,
};
use eredu_runtime::{
    ArchitectureParameters, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    ParallelLayeredArchitecture,
};

/// Pinned roles have distinct first/last pipeline owners.
#[derive(Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: DistributedNeuralBackend> {
    /// One text embedding, normalization and readout catalog.
    pub text: TextModel<B>,
    /// First image-owner parameters.
    pub image_embedding: ImageEmbedding<B>,
}
/// Each bounded unit contains exactly one image or text residual block.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: DistributedNeuralBackend> {
    /// Independently materialized Pixtral block.
    Vision(ImageLayer<B>),
    /// Independent learned image merger and spatial projector.
    Projector(ImageProjector<B>),
    /// Shared causal/refinement/draft block and optional separate LoRA.
    Text(TextLayer<B>),
}
struct Segment<T> {
    tokens: T,
    projected: Option<T>,
    image: Option<(i32, i32, bool)>,
}
/// Retained invocation products; image processing does not modify the text cache.
pub struct ForwardContext<T, C = TextForward<T>> {
    tokens: T,
    segments: Vec<Segment<T>>,
    image_initial: Option<T>,
    image_state: Option<ImageState<T>>,
    image_output: Option<T>,
    text: Option<C>,
    invocation: AttentionInvocation,
}
/// Backend-neutral composite using the ordinary layered residency lifecycle.
pub struct Model<B: DistributedNeuralBackend> {
    config: FamilyConfig,
    statics: StaticModules<B>,
    local_layout: Option<eredu_runtime::LocalModelLayout>,
}
impl<B: DistributedNeuralBackend> Model<B> {
    /// Declares each pinned role and no resident residual layers.
    pub fn new(
        config: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let image_embedding = ImageEmbedding::new(&config, context)?;
        let text = TextModel::new(config.clone(), context)?;
        Ok(Self {
            config,
            statics: StaticModules {
                text,
                image_embedding,
            },
            local_layout: None,
        })
    }
    /// Declares exact text vocabulary/cache shards and Pixtral head/channel shards.
    pub fn new_parallel(
        config: FamilyConfig,
        layout: eredu_runtime::LocalModelLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let geometry = crate::decoder::local_geometry(&config, &layout, |config, index, layout| {
            config
                .local_block_config(index, layout)
                .map_err(|e| eredu_runtime::ParallelPlanError::InvalidGroup(e.to_string()))
        })
        .map_err(Error::backend)?;
        let image_embedding = ImageEmbedding::new(&config, context)?;
        let text = TextModel::new_parallel(config.clone(), geometry, context)?;
        Ok(Self {
            config,
            statics: StaticModules {
                text,
                image_embedding,
            },
            local_layout: Some(layout),
        })
    }
    /// Shared family and checkpoint identity.
    pub fn config(&self) -> &FamilyConfig {
        &self.config
    }

    /// Borrows all independently bound pinned roles.
    pub fn static_modules(&self) -> &StaticModules<B> {
        &self.statics
    }

    /// Binds the same pinned roles before execution or reload.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.statics
    }

    fn prepare(
        &mut self,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let PreparedSegments {
            tokens,
            segments,
            patches,
            grids,
            invocation,
        } = prepare_segments(&self.config, input, context)?;
        let (image_initial, image_state) = if patches.is_empty() {
            (None, None)
        } else {
            let patches = B::Tensor::concatenate(&patches, 0, context)?;
            let (hidden, state) = self.statics.image_embedding.begin(
                ImageInput {
                    patches: &patches,
                    grids: &grids,
                },
                context,
            )?;
            (Some(hidden), Some(state))
        };
        let hidden =
            B::Tensor::full_f32(0.0, &[1, tokens.dim(1), self.config.hidden_size()], context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                tokens,
                segments,
                image_initial,
                image_state,
                image_output: None,
                text: None,
                invocation,
            },
        })
    }
}
impl<B: DistributedNeuralBackend> ArchitectureParameters<B> for Model<B> {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        self.statics.text.state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.statics.text.state_identity(state, topology)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        super::parallel::composite_parameter_description(&self.config).map_err(Error::backend)
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        super::checkpoint::module_source_recipes(&self.statics, source)
    }
    fn visit_static_parameters<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.statics.text.visit_static_parameters(visitor)?;
        visitor.visit("vision_embedding", &self.statics.image_embedding)
    }
    fn visit_static_parameters_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.statics.text.visit_static_parameters_mut(visitor)?;
        visitor.visit_mut("vision_embedding", &mut self.statics.image_embedding)
    }
}
impl<B, S> LayeredArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Input<'a>
        = PreparedCompositeInput<'a, B::Tensor, InputPartPlan>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <TextModel<B> as LayeredArchitecture<B, S>>::observation_hooks(&self.statics.text)
    }
    fn output_score_kind(
        forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        Some(forward.invocation.scores)
    }
    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        crate::nemotron_labs_diffusion::parallel::group_transport(group)
    }
    fn primary_execution_group(&self) -> &str {
        crate::decoder::TEXT_DECODER_EXECUTION_GROUP
    }
    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(2, 0..layout.len()),
        ])
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        Ok(super::parallel::composite_graph(&self.config)
            .map_err(Error::backend)?
            .0)
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.config.vision.as_ref().unwrap().num_hidden_layers as usize),
            1 => Ok(1),
            2 => Ok(self.config.num_hidden_layers() as usize),
            _ => Err(Error::backend("invalid image/text group")),
        }
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= <Self as LayeredArchitecture<B, S>>::group_unit_count(self, group)? {
            return Err(Error::backend("unit outside image/text stack"));
        }
        Ok(if group == 0 {
            format!("model.vision_tower.transformer.layers.{index}")
        } else if group == 1 {
            "model.multi_modal_projector".to_owned()
        } else {
            format!("model.layers.{index}")
        })
    }
    fn static_modules(&self) -> &Self::StaticModules {
        &self.statics
    }
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.statics
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index)?;
        Ok(if group == 0 {
            Unit::Vision(match &self.local_layout {
                Some(layout) => ImageLayer::new_parallel(
                    self.config.vision.as_ref().unwrap(),
                    index,
                    layout,
                    context,
                )?,
                None => ImageLayer::new(self.config.vision.as_ref().unwrap(), index, context)?,
            })
        } else if group == 1 {
            Unit::Projector(ImageProjector::new(&self.config, context)?)
        } else {
            Unit::Text(self.statics.text.construct_unit(index, context)?)
        })
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.prepare(input, context)
    }
    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let forward = self.prepare(input, context)?;
        if let Some(hidden) = &forward.context.image_initial {
            observer.observe("model.vision_tower.ln_pre.output", hidden)?;
        }
        Ok(forward)
    }
    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 2 || ((group == 0 || group == 1) && forward.image_state.is_some())
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match group {
            0 => Ok(forward.image_initial.as_ref().unwrap_or(initial).clone()),
            1 => Ok(dependencies
                .first()
                .map_or(initial, |hidden| *hidden)
                .clone()),
            2 => {
                if let Some(output) = dependencies.first() {
                    forward.image_output = Some((*output).clone());
                }
                let embedded = self
                    .statics
                    .text
                    .static_modules_mut()
                    .embed(&forward.tokens, context)?;
                let hidden = assemble_embeddings(forward, &embedded, context)?;
                let prepared = self.statics.text.begin_embedded(
                    hidden,
                    None,
                    forward.invocation,
                    state,
                    context,
                )?;
                forward.text = Some(prepared.context);
                Ok(prepared.hidden)
            }
            _ => {
                let _ = initial;
                Err(Error::backend("invalid image/text group"))
            }
        }
    }
    fn state_ordinal(&self, group: usize, index: usize, ordinal: usize) -> usize {
        if group == 2 { index } else { ordinal }
    }
    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _: usize,
    ) -> std::ops::Range<usize> {
        if group == 2 { index..index + 1 } else { 0..0 }
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_observed(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn forward_unit_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        match (group, unit) {
            (0, Unit::Vision(layer)) => layer.forward_observed(
                hidden,
                forward
                    .image_state
                    .as_ref()
                    .ok_or_else(|| Error::backend("image layer has no positions"))?,
                None,
                context,
                observer,
            ),
            (1, Unit::Projector(projector)) => projector.finish_observed(
                hidden,
                forward
                    .image_state
                    .as_ref()
                    .ok_or_else(|| Error::backend("image projection has no positions"))?,
                context,
                observer,
            ),
            (2, Unit::Text(layer)) => self.statics.text.forward_unit_observed(
                0,
                index,
                layer,
                hidden,
                state,
                forward
                    .text
                    .as_mut()
                    .ok_or_else(|| Error::backend("text ingress has not completed"))?,
                context,
                observer,
            ),
            _ => Err(Error::backend("unit has a different execution group")),
        }
    }
    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 1 && forward.image_state.is_some() {
            forward.image_output = Some(hidden.clone());
        }
        let _ = context;
        Ok(hidden.clone())
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics.text.finish_forward(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text context is absent"))?,
            context,
        )
    }
    fn finish_forward_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.statics.text.finish_forward_observed(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text context is absent"))?,
            context,
            observer,
        )
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = vec![&forward.tokens];
        values.extend(forward.image_initial.iter());
        values.extend(forward.image_output.iter());
        if let Some(image) = &forward.image_state {
            values.extend(image.retained_values());
        }
        if let Some(text) = &forward.text {
            values.extend(text.mask.iter());
        }
        for segment in &forward.segments {
            values.push(&segment.tokens);
            values.extend(segment.projected.iter());
        }
        values.into_iter()
    }
}
impl<B, S> ParallelLayeredArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as LayeredArchitecture<B, S>>::observation_hooks(self)
    }
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut S,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.prepare(input, context)
    }
    fn begin_execution_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 2 {
            return self.begin_execution_group(
                group,
                initial,
                dependencies,
                state,
                forward,
                context,
            );
        }
        if let Some(output) = dependencies.first() {
            forward.image_output = Some((*output).clone());
        }
        let embedded = self.statics.text.static_modules_mut().embed_parallel(
            &forward.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let hidden = assemble_embeddings(forward, &embedded, context)?;
        let prepared =
            self.statics
                .text
                .begin_embedded(hidden, None, forward.invocation, state, context)?;
        forward.text = Some(prepared.context);
        Ok(prepared.hidden)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_parallel_observed(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn forward_unit_parallel_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        match (group, unit) {
            (0, Unit::Vision(layer)) => layer.forward_observed(
                hidden,
                forward
                    .image_state
                    .as_ref()
                    .ok_or_else(|| Error::backend("image layer has no positions"))?,
                Some(parallel),
                context,
                observer,
            ),
            (1, Unit::Projector(projector)) => projector.finish_observed(
                hidden,
                forward
                    .image_state
                    .as_ref()
                    .ok_or_else(|| Error::backend("image projection has no positions"))?,
                context,
                observer,
            ),
            (2, Unit::Text(layer)) => self.statics.text.forward_unit_parallel_observed(
                0,
                index,
                layer,
                hidden,
                state,
                forward
                    .text
                    .as_mut()
                    .ok_or_else(|| Error::backend("text ingress has not completed"))?,
                parallel,
                context,
                observer,
            ),
            _ => Err(Error::backend("unit has a different execution group")),
        }
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics.text.finish_forward_parallel(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text context is absent"))?,
            parallel,
            context,
        )
    }
    fn finish_forward_parallel_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.statics.text.finish_forward_parallel_observed(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text context is absent"))?,
            parallel,
            context,
            observer,
        )
    }
}
impl<B, S> CompositeArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type InputPartPlan = InputPartPlan;
    type PrefillRequest = SingleCompositePrefillRequest<B::Tensor, InputPartPlan>;
    type AdmissionConfig = FamilyConfig;
    fn supports_attention_invocation(_: &FamilyConfig, invocation: AttentionInvocation) -> bool {
        [
            AttentionInvocation::CAUSAL,
            AttentionInvocation::DIFFUSION,
            AttentionInvocation::DRAFT,
        ]
        .contains(&invocation)
    }
    fn supports_chunked_text_prefill() -> bool {
        true
    }
    fn block_refinement_geometry(
        config: &FamilyConfig,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        Some(
            eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
                canvas_length: config.block_size,
                vocabulary_size: std::num::NonZeroU32::new(config.vocabulary_size() as u32)
                    .unwrap(),
            },
        )
    }
    fn prepare_prefill_request(
        prepared: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<InputPartPlan>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Error> {
        SingleCompositePrefillRequest::new(prepared, admitted)
    }
    fn admission_config(&self) -> FamilyConfig {
        self.config.clone()
    }
    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let values = input
            .prepared()
            .parts()
            .iter()
            .zip(input.admitted().parts())
            .map(|(part, plan)| plan.token_ids(part, context))
            .collect::<Result<Vec<_>, _>>()?;
        B::Tensor::concatenate(&values, 1, context)
    }
    fn admit_prepared_input(
        config: &FamilyConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<InputPartPlan>, eredu_core::CapabilityError> {
        admit_input(config, input, inspector)
    }
    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
    ) -> bool {
        group == 2
            || ((group == 0 || group == 1)
                && input
                    .admitted()
                    .parts()
                    .iter()
                    .any(|part| matches!(part, InputPartPlan::Image { native: true, .. })))
    }
    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_forward(input, state, context)
    }
    fn begin_composite_forward_observed<'a, O>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.begin_forward_observed(input, state, context, observer)
    }
    fn primary_ingress_collectives_pending(&self, forward: &Self::ForwardContext) -> bool {
        forward.text.is_none()
    }
    fn prepared_primary_ingress_collectives(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        Ok(Some(if tensor_partitions > 1 {
            vec![crate::composite_execution::CompositeTensorCollective::Sum {
                shape: vec![
                    1,
                    i32::try_from(input.admitted().decoder_positions())
                        .map_err(|_| "decoder extent exceeds i32")?,
                    self.config.hidden_size(),
                ],
            }]
        } else {
            vec![]
        }))
    }
    fn prepared_group_collective_waves(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        let pipeline_stages = stage_units.len();
        if tensor_partitions <= 1 {
            return Ok(None);
        }
        if group == 2 {
            let sequence = i32::try_from(input.admitted().decoder_positions())
                .map_err(|_| "decoder sequence exceeds i32")?;
            let width = self.config.hidden_size();
            let vocabulary = self.config.vocabulary_size() as usize;
            let counts = (0..tensor_partitions)
                .map(|rank| {
                    eredu_core::balanced_contiguous_range(
                        vocabulary,
                        tensor_partitions,
                        rank,
                        false,
                    )
                    .map(|range| range.len())
                    .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let last = self.config.num_hidden_layers() as usize;
            return Ok(Some(stage_units.iter().map(|range| {
                let mut operations = Vec::new();
                if !range.is_empty() && range.start == 0 {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                }
                for _ in range.clone() {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                    if input.attention_invocation().is_some_and(|invocation| invocation.adapters == eredu_core::decoding::AdapterScope::Draft) {
                        if let Some(adapter) = self.config.draft_adapter() {
                            operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, adapter.rank()] });
                        }
                    }
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                }
                if !range.is_empty() && range.end == last {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Gather { shape: vec![1, sequence, vocabulary as i32], counts: counts.clone(), axis: 2 });
                }
                operations
            }).collect()));
        }
        if group != 0 {
            return Ok(None);
        }
        if pipeline_stages == 0 {
            return Err("Pixtral collective schedule requires a positive stage count".into());
        }
        let patches = native_patch_positions(input)?;
        let vision = self
            .config
            .vision
            .as_ref()
            .ok_or("Pixtral schedule has no vision geometry")?;
        let mut stages = Vec::with_capacity(pipeline_stages);
        for stage in 0..pipeline_stages {
            let layers = stage_units[stage].clone();
            let operations = if patches == 0 {
                Vec::new()
            } else {
                vec![
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: vec![1, patches, vision.hidden_size],
                    };
                    2 * layers.len()
                ]
            };
            stages.push(operations);
        }
        Ok(Some(stages))
    }
    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group == 0 {
            return native_patch_positions(input);
        } else if group == 1 {
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    InputPartPlan::Image {
                        grid, native: true, ..
                    } => Some(
                        (grid.0 as u64) * (grid.1 as u64)
                            / u64::from(self.config.spatial_merge_size).pow(2),
                    ),
                    _ => None,
                })
                .sum()
        } else {
            input.admitted().decoder_positions()
        };
        i32::try_from(positions).map_err(|_| "group boundary extent exceeds i32".into())
    }
    fn prepared_group_continuation_batched(&self, _: usize) -> bool {
        false
    }
    fn partition_boundary_values(
        &self,
        source: usize,
        _: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        hidden: &B::Tensor,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>, Error> {
        encode_image_boundary(source, schema, hidden, context)
    }
    fn accept_partition_boundary(
        &mut self,
        source: usize,
        _: usize,
        _: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        _: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        accept_image_boundary(source, values, context)
    }
    fn partition_boundary_schema(
        &self,
        source: usize,
        destination: usize,
        selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        batch: i32,
        sequence: i32,
        _: &[i32],
        continuation: Option<(i32, i32)>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Error> {
        resolve_group_boundary(
            &self.config,
            source,
            destination,
            selected,
            batch,
            sequence,
            continuation,
        )
    }
    fn prepared_group_continuation_geometry(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
    ) -> Result<Option<(i32, i32)>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = native_patch_positions(input)?;
        Ok((patches > 0).then_some((patches, self.config.vision.as_ref().unwrap().hidden_size)))
    }
}
impl<B, S> ParallelCompositeArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_forward_parallel(input, state, parallel, context)
    }
    fn begin_composite_forward_parallel_observed<'a, O>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, InputPartPlan>,
        state: &mut S,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.begin_forward_observed(input, state, context, observer)
    }
}

fn native_patch_positions<T: eredu_nn::Tensor>(
    input: PreparedCompositeInput<'_, T, InputPartPlan>,
) -> Result<i32, String> {
    let patches = input
        .admitted()
        .parts()
        .iter()
        .filter_map(|part| match part {
            InputPartPlan::Image {
                grid, native: true, ..
            } => Some((grid.0 as u64) * (grid.1 as u64)),
            _ => None,
        })
        .try_fold(0u64, |total, count| total.checked_add(count))
        .ok_or("image patch count overflow")?;
    i32::try_from(patches).map_err(|_| "image continuation exceeds i32".into())
}

struct PreparedSegments<T> {
    tokens: T,
    segments: Vec<Segment<T>>,
    patches: Vec<T>,
    grids: Vec<(i32, i32)>,
    invocation: AttentionInvocation,
}
fn prepare_segments<T: Tensor>(
    config: &FamilyConfig,
    input: PreparedCompositeInput<'_, T, InputPartPlan>,
    context: &T::Context,
) -> Result<PreparedSegments<T>, Error> {
    let invocation = input
        .attention_invocation()
        .unwrap_or(if input.refinement().is_some() {
            AttentionInvocation::DIFFUSION
        } else {
            AttentionInvocation::CAUSAL
        });
    if ![
        AttentionInvocation::CAUSAL,
        AttentionInvocation::DIFFUSION,
        AttentionInvocation::DRAFT,
    ]
    .contains(&invocation)
        || input.refinement().flatten().is_some()
        || (input.refinement().is_some() && invocation == AttentionInvocation::CAUSAL)
    {
        return Err(Error::backend("invalid masked composite invocation"));
    }
    let mut segments = Vec::new();
    let mut patches = Vec::new();
    let mut grids = Vec::new();
    for (part, plan) in input
        .prepared()
        .parts()
        .iter()
        .zip(input.admitted().parts())
    {
        let tokens = plan.token_ids(part, context)?;
        let segment = match plan {
            InputPartPlan::TextTokens { .. } => Segment {
                tokens,
                projected: None,
                image: None,
            },
            InputPartPlan::TextEmbeddings { .. } => Segment {
                tokens,
                projected: Some(part.payload().value().clone()),
                image: None,
            },
            InputPartPlan::Image { grid, native, .. } => {
                if invocation != AttentionInvocation::CAUSAL {
                    return Err(Error::backend(
                        "image preparation requires a causal context invocation",
                    ));
                }
                if *native {
                    patches.push(part.payload().value().clone());
                    grids.push(*grid);
                }
                Segment {
                    tokens,
                    projected: (!native).then(|| part.payload().value().clone()),
                    image: Some((
                        grid.0 / config.spatial_merge_size as i32,
                        grid.1 / config.spatial_merge_size as i32,
                        *native,
                    )),
                }
            }
        };
        segments.push(segment);
    }
    let tokens = T::concatenate(
        &segments
            .iter()
            .map(|segment| segment.tokens.clone())
            .collect::<Vec<_>>(),
        1,
        context,
    )?;
    Ok(PreparedSegments {
        tokens,
        segments,
        patches,
        grids,
        invocation,
    })
}

fn assemble_embeddings<T: Tensor, C>(
    forward: &ForwardContext<T, C>,
    embedded: &T,
    context: &T::Context,
) -> Result<T, Error> {
    let slice = |tensor: &T, start, end| {
        tensor.index(
            &[Index::Full, Index::Range(start, end), Index::Full],
            context,
        )
    };
    let mut token_offset = 0;
    let mut image_offset = 0;
    let mut values = Vec::new();
    let text_element = embedded
        .element_type()
        .ok_or_else(|| Error::backend("unknown text embedding precision during image placement"))?;
    for segment in &forward.segments {
        let positions = segment.tokens.dim(1);
        if let Some((rows, columns, native)) = segment.image {
            let projected = if native {
                forward.image_output.as_ref().ok_or_else(|| {
                    Error::backend("image group has no completed projector output")
                })?
            } else {
                segment
                    .projected
                    .as_ref()
                    .ok_or_else(|| Error::backend("projected image has no payload"))?
            };
            let start = if native { image_offset } else { 0 };
            values.push(slice(embedded, token_offset, token_offset + 1)?);
            for row in 0..rows {
                values.push(
                    slice(
                        projected,
                        start + row * columns,
                        start + (row + 1) * columns,
                    )?
                    .cast_float(text_element, context)?,
                );
                let marker = token_offset + 1 + row * (columns + 1) + columns;
                values.push(slice(embedded, marker, marker + 1)?);
            }
            if native {
                image_offset += rows * columns;
            }
        } else if let Some(projected) = &segment.projected {
            values.push(projected.cast_float(text_element, context)?);
        } else {
            values.push(slice(embedded, token_offset, token_offset + positions)?);
        }
        token_offset += positions;
    }
    T::concatenate(&values, 1, context)
}

/// Cold transport geometry across the tower, projector and shared text boundaries.
pub fn group_boundary_schema(
    config: &FamilyConfig,
    source: usize,
    destination: usize,
) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
    use eredu_runtime::{BoundaryTensorSpec, BoundaryWireSchema};
    let (identity, width) = match (source, destination) {
        (0, 0) => (
            "nemotron_labs_diffusion.vision_continuation",
            config
                .vision
                .as_ref()
                .ok_or_else(|| Error::backend("vision boundary has no tower"))?
                .hidden_size,
        ),
        (0, 1) => (
            "nemotron_labs_diffusion.vision_to_projection",
            config
                .vision
                .as_ref()
                .ok_or_else(|| Error::backend("vision boundary has no tower"))?
                .hidden_size,
        ),
        (1, 2) => (
            "nemotron_labs_diffusion.projection_to_text",
            config.hidden_size(),
        ),
        _ => return Err(Error::backend("invalid image pipeline boundary")),
    };
    BoundaryWireSchema::new(
        identity,
        BoundaryTensorSpec::primary_activation(width),
        [BoundaryTensorSpec::new(
            "activation_precision",
            [eredu_runtime::BoundaryTensorDimension::Fixed(1)],
            eredu_runtime::BoundaryTensorDtype::Int32,
        )],
    )
    .map_err(Error::backend)
}

fn encode_image_boundary<T: Tensor>(
    source: usize,
    schema: &eredu_runtime::ResolvedBoundaryWireSchema,
    hidden: &T,
    context: &T::Context,
) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<T>>>, Error> {
    if source == 2 {
        return Ok(None);
    }
    let precision = T::from_i32_slice(
        &[super::partitioned::precision_code(hidden)?],
        &[1],
        context,
    )?;
    Ok(Some(vec![
        eredu_runtime::ArchitectureBoundaryValue::new(schema.primary().role(), hidden.clone())
            .map_err(Error::backend)?,
        eredu_runtime::ArchitectureBoundaryValue::new("activation_precision", precision)
            .map_err(Error::backend)?,
    ]))
}
fn accept_image_boundary<T: Tensor>(
    source: usize,
    values: Vec<T>,
    context: &T::Context,
) -> Result<Option<T>, Error> {
    if source == 2 {
        return Ok(None);
    }
    let mut values = values.into_iter();
    let hidden = values
        .next()
        .ok_or_else(|| Error::backend("image boundary has no activation"))?;
    let precision = values
        .next()
        .ok_or_else(|| Error::backend("image boundary has no precision"))?;
    if values.next().is_some()
        || precision.shape() != [1]
        || precision.element_type() != Some(eredu_nn::TensorElementType::I32)
    {
        return Err(Error::backend(
            "image boundary has invalid precision geometry",
        ));
    }
    let precision = precision.to_i32_vec(context)?;
    let element = super::partitioned::precision_from_code(
        *precision
            .first()
            .ok_or_else(|| Error::backend("image boundary has empty precision"))?,
    )?;
    hidden.cast_float(element, context).map(Some)
}
fn resolve_group_boundary(
    config: &FamilyConfig,
    source: usize,
    destination: usize,
    selected: &eredu_runtime::ResolvedBoundaryWireSchema,
    batch: i32,
    sequence: i32,
    continuation: Option<(i32, i32)>,
) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Error> {
    if source == 2 {
        return Ok(None);
    }
    let schema = group_boundary_schema(config, source, destination)?;
    if schema.identity() != selected.identity() {
        return Err(Error::backend(
            "image boundary differs from selected transport authority",
        ));
    }
    let sequence = continuation.map_or(sequence, |(length, _)| length);
    schema
        .resolve(batch, sequence)
        .map(Some)
        .map_err(Error::backend)
}

impl<B, S> eredu_runtime::RoutedLayeredArchitecture<B, S> for Model<B>
where
    B: eredu_nn::GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _: eredu_runtime::ExpertPass,
        _: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_unit(group, index, unit, hidden, state, forward, context)
    }
}
