//! Retained composite ownership; each stage constructs only selected parameters.
use super::*;
use crate::nemotron_labs_diffusion::partitioned::{
    PartitionContext, PartitionedTextModel, TextBoundary, TextBoundarySchema,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ArchitecturePartition, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, LayeredPartitionInput, LayeredPartitionOutput,
    LocalModelLayout, PartitionOwnership, PartitionedLayeredArchitecture, StateLayout,
};
use std::ops::Range;

/// Backend-free local geometry and independently selected image/text ranges.
#[derive(Debug, Clone)]
pub struct Geometry {
    layout: LocalModelLayout,
    vision: Range<usize>,
    projection: bool,
    text: Option<crate::decoder::PartitionLocalGeometry<FamilyConfig>>,
    complete_state: StateLayout,
}
impl Geometry {
    /// Complete tensor-local decoder state before pipeline slicing.
    pub fn complete_state(&self) -> &StateLayout {
        &self.complete_state
    }
}
/// Validates physical tensor shards and cache geometry without native allocations.
pub fn geometry(
    config: &FamilyConfig,
    layout: LocalModelLayout,
    vision: Range<usize>,
    projection: bool,
    text: Option<Range<usize>>,
) -> Result<Geometry, Error> {
    config.validate_config()?;
    let image = config
        .vision
        .as_ref()
        .ok_or_else(|| Error::backend("composite partition requires Pixtral"))?;
    if vision.start > vision.end || vision.end > image.num_hidden_layers as usize {
        return Err(Error::backend(
            "vision ownership exceeds the admitted tower",
        ));
    }
    for index in vision.clone() {
        crate::nemotron_labs_diffusion::parallel::vision_local_widths(image, index, &layout)
            .map_err(Error::backend)?;
    }
    let complete = crate::decoder::partition_local_geometry(
        config,
        &layout,
        0..config.num_hidden_layers() as usize,
    )
    .map_err(Error::backend)?;
    let complete_state = complete.complete_state_layout().clone();
    let text = text
        .map(|range| {
            crate::decoder::partition_local_geometry(config, &layout, range).map_err(Error::backend)
        })
        .transpose()?;
    Ok(Geometry {
        layout,
        vision,
        projection,
        text,
        complete_state,
    })
}
/// Pinned modules reflect exact first/last owners, including stateless vision stages.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: DistributedNeuralBackend> {
    /// Shared text embedding/readout roles, only where text units are selected.
    pub text: Option<PartitionedTextModel<B>>,
    /// Present only on the first image owner.
    pub image_embedding: Option<ImageEmbedding<B>>,
}
/// One retained pipeline stage over the shared image/text parameter catalog.
pub struct Model<B: DistributedNeuralBackend> {
    config: FamilyConfig,
    geometry: Geometry,
    parameters: ArchitectureParameterDescription,
    statics: StaticModules<B>,
}
impl<B: DistributedNeuralBackend> Model<B> {
    /// Consumes an exact neutral partition and creates no unowned static storage.
    pub fn new(
        config: FamilyConfig,
        parameters: &ArchitectureParameterDescription,
        partition: &ArchitecturePartition<Geometry, TextBoundarySchema>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let expected =
            crate::nemotron_labs_diffusion::parallel::composite_parameter_description(&config)
                .map_err(Error::backend)?;
        if parameters.graph() != expected.graph()
            || parameters.unit_layout() != expected.unit_layout()
            || parameters.groups() != expected.groups()
            || partition.graph() != parameters.graph()
            || partition.unit_layout() != parameters.unit_layout()
        {
            return Err(Error::backend(
                "composite partition differs from the admitted parameter topology",
            ));
        }
        let geometry = partition.local_geometry().clone();
        let vision = partition
            .groups()
            .iter()
            .find(|group| group.group().as_str() == "vision")
            .map(|group| group.global_units().clone())
            .unwrap_or(0..0);
        let text_range = partition
            .groups()
            .iter()
            .find(|group| group.group().as_str() == crate::decoder::TEXT_DECODER_EXECUTION_GROUP)
            .map(|group| group.global_units().clone());
        let projection = partition
            .groups()
            .iter()
            .find(|group| group.group().as_str() == "vision_projection")
            .map(|group| group.global_units().clone());
        if projection.as_ref().is_some_and(|range| *range != (0..1))
            || projection.is_some() != geometry.projection
        {
            return Err(Error::backend(
                "projector unit differs from selected ownership",
            ));
        }
        if vision != geometry.vision
            || text_range.as_ref() != geometry.text.as_ref().map(|g| g.owned_units()).as_ref()
        {
            return Err(Error::backend(
                "composite parameter and execution ownership disagree",
            ));
        }
        let ownership = partition.ownership();
        if ownership.static_roles().iter().any(|role| {
            !["embedding", "norm", "output", "vision_embedding"].contains(&role.as_str())
        }) || !ownership.replicated_static_roles().is_empty()
        {
            return Err(Error::backend(
                "composite partition has undeclared static replicas",
            ));
        }
        if ownership.owns_static_role("norm") != ownership.owns_static_role("output")
            || ownership.owns_output() != ownership.owns_static_role("output")
        {
            return Err(Error::backend(
                "text output ownership requires its normalization and untied head",
            ));
        }
        let text = if let Some(local) = &geometry.text {
            let description =
                crate::nemotron_labs_diffusion::parallel::parameter_description(&config)
                    .map_err(Error::backend)?;
            let text_ownership = PartitionOwnership::new(
                ownership.owns_static_role("embedding"),
                ownership.owns_static_role("output"),
                ownership
                    .static_roles()
                    .iter()
                    .filter(|role| ["embedding", "norm", "output"].contains(&role.as_str()))
                    .cloned(),
            )
            .map_err(Error::backend)?;
            let text_partition = ArchitecturePartition::from_description(
                &description,
                [(
                    crate::decoder::TEXT_DECODER_EXECUTION_GROUP,
                    local.owned_units(),
                )],
                text_ownership,
                &geometry.complete_state,
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(
                        0,
                        0..geometry.complete_state.len(),
                    ),
                ]),
                local.clone(),
                TextBoundarySchema::new(config.hidden_size()),
            )
            .map_err(Error::backend)?;
            if partition.state() != text_partition.state() {
                return Err(Error::backend(
                    "composite text state differs from retained local ownership",
                ));
            }
            Some(PartitionedTextModel::new(
                config.clone(),
                &description,
                &text_partition,
                context,
            )?)
        } else {
            if partition.state().is_some()
                || ownership.owns_static_role("embedding")
                || ownership.owns_static_role("norm")
                || ownership.owns_static_role("output")
            {
                return Err(Error::backend(
                    "stateless image owner retains text parameters or state",
                ));
            }
            None
        };
        let image_embedding = ownership
            .owns_static_role("vision_embedding")
            .then(|| ImageEmbedding::new(&config, context))
            .transpose()?;
        Ok(Self {
            config,
            geometry,
            parameters: parameters.clone(),
            statics: StaticModules {
                text,
                image_embedding,
            },
        })
    }
    /// Exact retained family geometry.
    pub fn config(&self) -> &FamilyConfig {
        &self.config
    }
    /// Borrows only locally retained pinned storage.
    pub fn static_modules(&self) -> &StaticModules<B> {
        &self.statics
    }
    /// Binds only locally retained pinned storage.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.statics
    }
    /// Constructs an owned image or shared text layer, rejecting unowned indices.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        match group {
            0 if self.geometry.vision.contains(&index) => {
                Ok(Unit::Vision(ImageLayer::new_parallel(
                    self.config.vision.as_ref().expect("validated tower"),
                    index,
                    &self.geometry.layout,
                    context,
                )?))
            }
            1 if index == 0 && self.geometry.projection => {
                Ok(Unit::Projector(ImageProjector::new(&self.config, context)?))
            }
            2 => Ok(Unit::Text(
                self.statics
                    .text
                    .as_ref()
                    .ok_or_else(|| Error::backend("partition owns no text layers"))?
                    .construct_unit(index, context)?,
            )),
            _ => Err(Error::backend(
                "partition attempted an unowned image/text unit",
            )),
        }
    }
    fn prepare(
        &mut self,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        LayeredForwardState<B::Tensor, ForwardContext<B::Tensor, PartitionContext<B::Tensor>>>,
        Error,
    > {
        let PreparedSegments {
            tokens,
            segments,
            patches,
            grids,
            invocation,
        } = prepare_segments(&self.config, input, context)?;
        let (image_initial, image_state) = if patches.is_empty() {
            (None, None)
        } else if let Some(embedding) = &mut self.statics.image_embedding {
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
}
impl<B: DistributedNeuralBackend> ArchitectureParameters<B> for Model<B> {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        Ok(self.geometry.complete_state.clone())
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.statics
            .text
            .as_ref()
            .ok_or_else(|| Error::backend("stateless vision owner has no text cache identity"))?
            .state_identity(state, topology)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        Ok(self.parameters.clone())
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        crate::nemotron_labs_diffusion::checkpoint::module_source_recipes(&self.statics, source)
    }
    fn visit_static_parameters<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(text) = &self.statics.text {
            text.visit_static_parameters(visitor)?;
        }
        if let Some(embedding) = &self.statics.image_embedding {
            visitor.visit("vision_embedding", embedding)?;
        }
        Ok(())
    }
    fn visit_static_parameters_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(text) = &mut self.statics.text {
            text.visit_static_parameters_mut(visitor)?;
        }
        if let Some(embedding) = &mut self.statics.image_embedding {
            visitor.visit_mut("vision_embedding", embedding)?;
        }
        Ok(())
    }
}
impl<B: DistributedNeuralBackend> Model<B> {
    fn text(&mut self) -> Result<&mut PartitionedTextModel<B>, Error> {
        self.statics
            .text
            .as_mut()
            .ok_or_else(|| Error::backend("partition owns no shared text stack"))
    }
    fn begin_text<S: LayerRuntimeState<B>>(
        &mut self,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor, PartitionContext<B::Tensor>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        if let Some(output) = dependencies.first() {
            forward.image_output = Some((*output).clone());
        }
        let prepared = self.text()?.prepare_partition_input(
            crate::nemotron_labs_diffusion::model::TextInput {
                tokens: &forward.tokens,
                mask: None,
                invocation: forward.invocation,
                feedback: None,
            },
            state,
            parallel,
            context,
        )?;
        let LayeredPartitionInput::Hidden { hidden, auxiliary } = prepared.input else {
            return Err(Error::backend(
                "prepared text ingress does not own embeddings",
            ));
        };
        let hidden = assemble_embeddings(forward, &hidden, context)?;
        let expected = state.layout().clone();
        let result = self.text()?.begin_partition_observed(
            LayeredPartitionInput::Hidden { hidden, auxiliary },
            prepared.mask.as_ref(),
            state,
            &expected,
            0,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )?;
        forward.text = Some(result.context);
        Ok(result.hidden)
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
    type ForwardContext = ForwardContext<B::Tensor, PartitionContext<B::Tensor>>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
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
    fn state_partition_plan(&self, layout: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            2,
            0..layout.len(),
        )])
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        Ok(self.parameters.graph().clone())
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self
                .config
                .vision
                .as_ref()
                .expect("validated tower")
                .num_hidden_layers as usize),
            1 => Ok(1),
            2 => Ok(self.config.num_hidden_layers() as usize),
            _ => Err(Error::backend("invalid composite group")),
        }
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= <Self as LayeredArchitecture<B, S>>::group_unit_count(self, group)? {
            return Err(Error::backend("unit exceeds composite stack"));
        }
        Ok(match group {
            0 => format!("model.vision_tower.transformer.layers.{index}"),
            1 => "model.multi_modal_projector".into(),
            _ => format!("model.layers.{index}"),
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
        self.construct_unit(group, index, context)
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
            2 => self.begin_text(dependencies, state, forward, None, context),
            _ => Err(Error::backend("invalid composite group")),
        }
    }
    fn state_ordinal(&self, group: usize, index: usize, ordinal: usize) -> usize {
        if group == 2 {
            index
        } else {
            ordinal
        }
    }
    fn retained_state_ordinals(&self, group: usize, index: usize, _: usize) -> Range<usize> {
        if group == 2 {
            index..index + 1
        } else {
            0..0
        }
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
            (0, Unit::Vision(layer)) if self.geometry.vision.contains(&index) => layer
                .forward_observed(
                    hidden,
                    forward
                        .image_state
                        .as_ref()
                        .ok_or_else(|| Error::backend("vision continuation has no positions"))?,
                    None,
                    context,
                    observer,
                ),
            (1, Unit::Projector(projector)) if index == 0 && self.geometry.projection => projector
                .finish_observed(
                    hidden,
                    forward
                        .image_state
                        .as_ref()
                        .ok_or_else(|| Error::backend("projector continuation has no positions"))?,
                    context,
                    observer,
                ),
            (2, Unit::Text(layer)) => self.text()?.forward_unit_observed(
                0,
                index,
                layer,
                hidden,
                state,
                forward
                    .text
                    .as_mut()
                    .ok_or_else(|| Error::backend("text continuation has no invocation"))?,
                context,
                observer,
            ),
            _ => Err(Error::backend(
                "partition attempted an unowned execution unit",
            )),
        }
    }
    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 1 {
            forward.image_output = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.text()?.finish_forward(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text output has no invocation"))?,
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
        self.text()?.finish_forward_observed(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text output has no invocation"))?,
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
        if group == 2 {
            self.begin_text(dependencies, state, forward, Some(parallel), context)
        } else {
            self.begin_execution_group(group, initial, dependencies, state, forward, context)
        }
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
            (0, Unit::Vision(layer)) if self.geometry.vision.contains(&index) => layer
                .forward_observed(
                    hidden,
                    forward
                        .image_state
                        .as_ref()
                        .ok_or_else(|| Error::backend("vision continuation has no positions"))?,
                    Some(parallel),
                    context,
                    observer,
                ),
            (1, Unit::Projector(projector)) if index == 0 && self.geometry.projection => projector
                .finish_observed(
                    hidden,
                    forward
                        .image_state
                        .as_ref()
                        .ok_or_else(|| Error::backend("projector continuation has no positions"))?,
                    context,
                    observer,
                ),
            (2, Unit::Text(layer)) => self.text()?.forward_unit_parallel_observed(
                0,
                index,
                layer,
                hidden,
                state,
                forward
                    .text
                    .as_mut()
                    .ok_or_else(|| Error::backend("text continuation has no invocation"))?,
                parallel,
                context,
                observer,
            ),
            _ => Err(Error::backend(
                "partition attempted an unowned execution unit",
            )),
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
        self.text()?.finish_forward_parallel(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text output has no invocation"))?,
            parallel,
            context,
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
    fn supports_chunked_text_prefill() -> bool {
        true
    }
    fn supports_attention_invocation(_: &FamilyConfig, invocation: AttentionInvocation) -> bool {
        [
            AttentionInvocation::CAUSAL,
            AttentionInvocation::DIFFUSION,
            AttentionInvocation::DRAFT,
        ]
        .contains(&invocation)
    }
    fn block_refinement_geometry(
        config: &FamilyConfig,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        <super::Model<B> as CompositeArchitecture<B, S>>::block_refinement_geometry(config)
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
        <super::Model<B> as CompositeArchitecture<B, S>>::prepared_prediction_token_ids(
            input, context,
        )
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
        self.statics.text.is_some() && forward.text.is_none()
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
            Vec::new()
        }))
    }
    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
    ) -> Result<i32, String> {
        if group == 0 {
            return native_patch_positions(input);
        }
        let positions = if group == 1 {
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
                .try_fold(0u64, |n, count| n.checked_add(count))
                .ok_or("projector extent overflow")?
        } else {
            input.admitted().decoder_positions()
        };
        i32::try_from(positions).map_err(|_| "group extent exceeds i32".into())
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
    ) -> Result<Option<crate::composite_execution::CompositeContinuationGeometry>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = native_patch_positions(input)?;
        Ok((patches > 0).then_some(
            (
                patches,
                self.config
                    .vision
                    .as_ref()
                    .expect("validated tower")
                    .hidden_size,
            )
                .into(),
        ))
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
        let vision = self.config.vision.as_ref().expect("validated tower");
        let mut stages = Vec::with_capacity(pipeline_stages);
        for range in &stage_units[..pipeline_stages] {
            stages.push(if patches == 0 {
                Vec::new()
            } else {
                vec![
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: vec![1, patches, vision.hidden_size]
                    };
                    2 * range.len()
                ]
            });
        }
        Ok(Some(stages))
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
impl<B, S> PartitionedLayeredArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Boundary = TextBoundarySchema;
    fn boundary_schema(&self) -> Result<TextBoundarySchema, Error> {
        Ok(TextBoundarySchema::new(self.config.hidden_size()))
    }
    fn partition_observation_hooks(
        &self,
        _: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as LayeredArchitecture<B, S>>::observation_hooks(self)
    }
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_partition_observed(
            input,
            mask,
            state,
            expected,
            first,
            None,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_partition_observed(
            input,
            mask,
            state,
            expected,
            first,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_partition_observed<
        'a,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let origin = self
            .geometry
            .text
            .as_ref()
            .ok_or_else(|| {
                Error::backend("stateless image owner cannot resume a decoder boundary")
            })?
            .owned_units()
            .start;
        if first != origin {
            return Err(Error::backend(
                "decoder boundary origin differs from retained local ownership",
            ));
        }
        let result = self.text()?.begin_partition_observed(
            input, mask, state, expected, 0, parallel, context, observer,
        )?;
        let tokens = B::Tensor::from_i32_slice(
            &vec![0; result.hidden.dim(0) as usize * result.hidden.dim(1) as usize],
            &[result.hidden.dim(0), result.hidden.dim(1)],
            context,
        )?;
        let invocation = result.context.invocation;
        Ok(LayeredForwardState {
            hidden: result.hidden,
            context: ForwardContext {
                tokens,
                segments: Vec::new(),
                image_initial: None,
                image_state: None,
                image_output: None,
                text: Some(result.context),
                invocation,
            },
        })
    }
    fn enter_partition_group(
        &mut self,
        _: usize,
        initial: &B::Tensor,
        _: &mut S,
        _: &mut Self::ForwardContext,
        _: Option<&B::ParallelContext>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error> {
        self.finish_partition_observed(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn finish_partition_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error> {
        self.text()?.finish_partition_observed(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text boundary has no settled invocation"))?,
            owns_output,
            parallel,
            context,
            observer,
        )
    }
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

impl<B, S> eredu_runtime::ParallelRoutedLayeredArchitecture<B, S> for Model<B>
where
    B: eredu_nn::GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _: eredu_runtime::ExpertPass,
        _: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_unit_parallel(
            group, index, unit, hidden, state, forward, parallel, context,
        )
    }
}
