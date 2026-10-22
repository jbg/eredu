//! Pixtral context preparation and traversal of the singular text backbone.
/// Exact owned image/text construction and pipeline invocation transport.
pub mod partition;

use super::{
    input::{admit_input, InputPartPlan},
    model::{ForwardContext as TextForward, TextKernel, TextLayer},
    vision::{ImageEmbedding, ImageInput, ImageLayer, ImageProjector, ImageState},
    FamilyConfig,
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
    DistributedNeuralBackend, Error, Index, Parameterized, ReadOnlyAttentionCache, Tensor,
};
use eredu_runtime::{ComponentEquations as _, LayerRuntimeState, LayeredForwardState};

/// Pinned roles have distinct first/last pipeline owners.
#[derive(Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: DistributedNeuralBackend> {
    /// One text embedding, normalization and readout catalog.
    pub text: TextKernel<B>,
    /// First image-owner parameters.
    pub image_embedding: ImageEmbedding<B>,
}
/// Typed access to physically retained composite text and image resources.
pub trait CompositeResources<B: DistributedNeuralBackend>:
    Parameterized<B::Tensor> + Sized + 'static
{
    /// Internal observation coverage of ordinary low-level parallel readout.
    const OBSERVED_PARALLEL_READOUT: bool;
    /// Selected decoder pinned storage.
    type TextModules: crate::decoder::DecoderStaticResources<B>;
    /// Selected decoder invocation/state geometry.
    type TextGeometry: crate::decoder::DecoderGeometry<FamilyConfig>;
    /// Text resources, absent on stateless media owners.
    fn text(&self) -> Option<&CompositeTextKernel<B, Self>>;
    /// Mutable access to those same resources.
    fn text_mut(&mut self) -> Option<&mut CompositeTextKernel<B, Self>>;
    /// Image ingress, only on its physical owner.
    fn image(&self) -> Option<&ImageEmbedding<B>>;
    /// Mutable access to the retained image ingress.
    fn image_mut(&mut self) -> Option<&mut ImageEmbedding<B>>;
}
/// Text equation specialization selected by the composite's physical aggregate.
pub type CompositeTextKernel<B, R> = TextKernel<
    B,
    <R as CompositeResources<B>>::TextModules,
    <R as CompositeResources<B>>::TextGeometry,
>;
impl<B: DistributedNeuralBackend> CompositeResources<B> for StaticModules<B> {
    const OBSERVED_PARALLEL_READOUT: bool = true;
    type TextModules = crate::decoder::StaticModules<B>;
    type TextGeometry = crate::decoder::CompleteDecoderGeometry<FamilyConfig>;
    fn text(&self) -> Option<&CompositeTextKernel<B, Self>> {
        Some(&self.text)
    }
    fn text_mut(&mut self) -> Option<&mut CompositeTextKernel<B, Self>> {
        Some(&mut self.text)
    }
    fn image(&self) -> Option<&ImageEmbedding<B>> {
        Some(&self.image_embedding)
    }
    fn image_mut(&mut self) -> Option<&mut ImageEmbedding<B>> {
        Some(&mut self.image_embedding)
    }
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
/// Complete composite execution through the shared validated runtime adapter.
pub type Model<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompositeKernel<B>,
    eredu_runtime::KernelParameterModules,
>;
/// Family-owned media assembly and context/refinement equations.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CompositeKernel<B: DistributedNeuralBackend, R: CompositeResources<B> = StaticModules<B>>
{
    #[parameter(skip)]
    config: FamilyConfig,
    #[parameter(skip)]
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    statics: R,
    #[parameter(skip)]
    local_layout: Option<eredu_runtime::LocalModelLayout>,
    #[parameter(skip)]
    geometry: Option<partition::Geometry>,
    #[parameter(skip)]
    backend: std::marker::PhantomData<fn() -> B>,
}
impl<B: DistributedNeuralBackend, R: CompositeResources<B>> CompositeKernel<B, R> {
    /// Retained semantic declaration used by execution and inspection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    /// Shared family and checkpoint identity.
    pub fn config(&self) -> &FamilyConfig {
        &self.config
    }

    /// Borrows all independently bound pinned roles.
    pub fn static_modules(&self) -> &R {
        &self.statics
    }

    /// Binds the same pinned roles before execution or reload.
    pub fn static_modules_mut(&mut self) -> &mut R {
        &mut self.statics
    }

    fn text(&mut self) -> Result<&mut CompositeTextKernel<B, R>, Error> {
        self.statics
            .text_mut()
            .ok_or_else(|| Error::backend("partition owns no shared text stack"))
    }
    /// Constructs one declared resource within the retained local ownership.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.definition.unit_address(group, index)?;
        if let Some(g) = &self.geometry {
            let owned = match group {
                0 => g.vision.contains(&index),
                1 => g.projection && index == 0,
                2 => g
                    .text
                    .as_ref()
                    .is_some_and(|g| g.owned_units().contains(&index)),
                _ => false,
            };
            if !owned {
                return Err(Error::backend(
                    "partition attempted an unowned image/text unit",
                ));
            }
        }
        let layout = self
            .geometry
            .as_ref()
            .map(|g| &g.layout)
            .or(self.local_layout.as_ref());
        Ok(match group {
            0 => Unit::Vision(match layout {
                Some(layout) => ImageLayer::new_parallel(
                    self.config.vision.as_ref().expect("validated tower"),
                    index,
                    layout,
                    context,
                )?,
                None => ImageLayer::new(
                    self.config.vision.as_ref().expect("validated tower"),
                    index,
                    context,
                )?,
            }),
            1 => Unit::Projector(ImageProjector::new(&self.config, context)?),
            _ => Unit::Text(
                self.statics
                    .text()
                    .ok_or_else(|| Error::backend("partition owns no text layers"))?
                    .construct_unit(index, context)?,
            ),
        })
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
        } else if let Some(embedding) = self.statics.image_mut() {
            let patches = B::Tensor::concatenate(&patches, 0, context)?;
            let (hidden, state) = embedding.begin(
                ImageInput {
                    patches: &patches,
                    grids: &grids,
                },
                context,
            )?;
            (Some(hidden), Some(state))
        } else {
            (
                None,
                Some(ImageState::prepare(&self.config, &grids, context)?),
            )
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
    fn run_unit<S, O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        let index = address.index();
        match (address.group(), unit) {
            (0, Unit::Vision(layer)) => layer.forward_observed(
                hidden,
                forward
                    .image_state
                    .as_ref()
                    .ok_or_else(|| Error::backend("image layer has no positions"))?,
                parallel,
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
            (2, Unit::Text(layer)) => {
                let text = self.text()?;
                let unit_address = text.definition().unit_address(0, index)?;
                text.run_unit(
                    unit_address,
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
                )
            }
            _ => Err(Error::backend("unit has a different execution group")),
        }
    }
    fn begin_group_inner<S>(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
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
                    .text()?
                    .embed_tokens(&forward.tokens, parallel, context)?;
                let hidden = assemble_embeddings(forward, &embedded, context)?;
                let prepared = self.text()?.begin_embedded(
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
}
impl<B: DistributedNeuralBackend, R: CompositeResources<B>>
    eredu_runtime::ComponentPinnedResources<B> for CompositeKernel<B, R>
{
    type Modules = R;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.statics
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.statics
    }
    fn visit_pinned<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(text) = self.statics.text() {
            text.visit_pinned(visitor)?;
        }
        if let Some(image) = self.statics.image() {
            visitor.visit("vision_embedding", image)?;
        }
        Ok(())
    }
    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(text) = self.statics.text_mut() {
            text.visit_pinned_mut(visitor)?;
        }
        if let Some(image) = self.statics.image_mut() {
            visitor.visit_mut("vision_embedding", image)?;
        }
        Ok(())
    }
}
impl<B: DistributedNeuralBackend, R: CompositeResources<B>> eredu_runtime::ComponentKernel<B>
    for CompositeKernel<B, R>
{
    type Input<'a>
        = PreparedCompositeInput<'a, B::Tensor, InputPartPlan>
    where
        Self: 'a;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.statics
            .text()
            .ok_or_else(|| Error::backend("stateless vision owner has no text cache identity"))?
            .state_identity(state, topology)
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
    fn output_score_kind(
        forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        Some(forward.invocation.scores)
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.construct_unit(address.group(), address.index(), context)
    }
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 2 || ((group == 0 || group == 1) && forward.image_state.is_some())
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: eredu_runtime::ExecutionUnitAddress,
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
impl<B, S, R> eredu_runtime::ComponentEquations<B, S> for CompositeKernel<B, R>
where
    R: CompositeResources<B>,
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.prepare(input, context)
    }
    fn begin_observed<'a, O>(
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
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.begin_group_inner(group, initial, dependencies, state, forward, None, context)
    }
    fn complete_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 1 && (self.geometry.is_some() || forward.image_state.is_some()) {
            forward.image_output = Some(hidden.clone());
        }
        let _ = context;
        Ok(hidden.clone())
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.text()?.finish(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text context is absent"))?,
            context,
        )
    }
    fn finish_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.text()?.finish_observed(
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
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.run_unit(
            address,
            unit,
            hidden,
            state,
            forward,
            None,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn execute_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.run_unit(
            address, unit, hidden, state, forward, None, context, observer,
        )
    }
}
impl<B, S, R> eredu_runtime::ParallelComponentKernel<B, S> for CompositeKernel<B, R>
where
    R: CompositeResources<B>,
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as eredu_runtime::ComponentEquations<B, S>>::observation_hooks(self)
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut S,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.prepare(input, context)
    }
    fn begin_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.begin_observed(input, state, context, observer)
    }

    fn begin_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.begin_group_inner(
            group,
            initial,
            dependencies,
            state,
            forward,
            Some(parallel),
            context,
        )
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.text()?.finish_parallel(
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
    fn finish_parallel_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        if !R::OBSERVED_PARALLEL_READOUT {
            return self.finish_parallel(hidden, state, forward, parallel, context);
        }
        self.text()?.finish_parallel_observed(
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
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.run_unit(
            address,
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn execute_parallel_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.run_unit(
            address,
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
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
    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a> {
        input
    }
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<
        crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    > {
        Some(*input)
    }

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
        self.kernel().config.clone()
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
                    self.kernel().config.hidden_size(),
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
            let width = self.kernel().config.hidden_size();
            let vocabulary = self.kernel().config.vocabulary_size() as usize;
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
            let last = self.kernel().config.num_hidden_layers() as usize;
            return Ok(Some(stage_units.iter().map(|range| {
                let mut operations = Vec::new();
                if !range.is_empty() && range.start == 0 {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                }
                for _ in range.clone() {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                    if input.attention_invocation().is_some_and(|invocation| invocation.adapters == eredu_core::decoding::AdapterScope::Draft) {
                        if let Some(adapter) = self.kernel().config.draft_adapter() {
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
            .kernel()
            .config
            .vision
            .as_ref()
            .ok_or("Pixtral schedule has no vision geometry")?;
        let mut stages = Vec::with_capacity(pipeline_stages);
        for layers in &stage_units[..pipeline_stages] {
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
                            / u64::from(self.kernel().config.spatial_merge_size).pow(2),
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
            &self.kernel().config,
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
    ) -> Result<Option<crate::composite_execution::CompositeContinuationGeometry>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = native_patch_positions(input)?;
        Ok((patches > 0).then_some(
            (
                patches,
                self.kernel().config.vision.as_ref().unwrap().hidden_size,
            )
                .into(),
        ))
    }
}
impl<B, S> ParallelCompositeArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
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

impl<B, S, R> eredu_runtime::RoutedComponentKernel<B, S> for CompositeKernel<B, R>
where
    R: CompositeResources<B>,
    B: eredu_nn::GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        self.execute(address, unit, hidden, state, forward, context)
    }
}

impl<B: DistributedNeuralBackend> CompositeKernel<B> {
    /// Materializes declared complete composite resources.
    pub fn new(
        config: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Model<B>, Error> {
        let definition = std::sync::Arc::new(super::definition::composite(&config, None)?);
        let image_embedding = ImageEmbedding::new(&config, context)?;
        let text = TextKernel::new(config.clone(), context)?.into_parts().1;
        let kernel = Self {
            config,
            definition: definition.clone(),
            statics: StaticModules {
                text,
                image_embedding,
            },
            local_layout: None,
            geometry: None,
            backend: std::marker::PhantomData,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel)
            .map_err(Error::from)
    }
    /// Materializes declared complete composite resources.
    pub fn new_parallel(
        config: FamilyConfig,
        layout: eredu_runtime::LocalModelLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Model<B>, Error> {
        let geometry = crate::decoder::local_geometry(&config, &layout, |config, index, layout| {
            config
                .local_block_config(index, layout)
                .map_err(|e| eredu_runtime::ParallelPlanError::InvalidGroup(e.to_string()))
        })
        .map_err(Error::backend)?;
        let definition =
            std::sync::Arc::new(super::definition::composite(&config, Some(&geometry))?);
        let image_embedding = ImageEmbedding::new(&config, context)?;
        let text = TextKernel::new_parallel(config.clone(), geometry, context)?
            .into_parts()
            .1;
        let kernel = Self {
            config,
            definition: definition.clone(),
            statics: StaticModules {
                text,
                image_embedding,
            },
            local_layout: Some(layout),
            geometry: None,
            backend: std::marker::PhantomData,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel)
            .map_err(Error::from)
    }
}
