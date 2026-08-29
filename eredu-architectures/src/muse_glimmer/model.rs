//! One neutral Muse-Glimmer multimodal model for resident and bounded runtimes.

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, Error, Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, ExpertPass,
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, OwnedParameterGroupSpec, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, ParameterGroupOwner, PartitionedLayeredArchitecture,
    RoutedExpertProvider, RoutedLayeredArchitecture, StateLayout,
};

use super::{
    layer_parameter_groups, state_layout, static_parameter_groups, vision_layer_parameter_groups,
    vision_static_parameter_groups, DecoderConfig, LocalGeometry,
    StaticModules as TextStaticModules, TransformerBlock, VisionBlock, VisionInput, VisionState,
    VisionStatic,
};

/// One ordered decoder-ingress segment.
pub enum DecoderInputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image/video placeholder token IDs matching the next projected media span.
    Media(&'a T),
}

/// Prepared ordered text/media request.
pub struct ModelInput<'a, T> {
    /// Ordered token segments at decoder ingress.
    pub parts: &'a [DecoderInputPart<'a, T>],
    /// Optional packed raw image/video patches and host grid metadata.
    pub vision: Option<VisionInput<'a, T>>,
    /// Optional explicit decoder attention mask.
    pub mask: Option<&'a T>,
}

/// Typed decoder input for one pipeline partition.
pub enum TextPartitionInput<'a, T> {
    /// Token identities owned by the first decoder partition.
    Tokens(&'a T),
    /// Embedded activation received from an upstream decoder partition.
    Hidden(T),
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Media { tokens: T },
}

/// Forward-pass values retained across streamed unit submissions.
pub struct ForwardContext<T> {
    mask: Option<T>,
    parts: Vec<PreparedPart<T>>,
    vision: Option<VisionState<T>>,
}

/// Pinned text and media modules shared by every storage policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: RoutedNeuralBackend> {
    /// Text embedding, final norm, and output head.
    pub text: TextStaticModules<B>,
    /// Optional patch/position modules, merge adapter, and language projection.
    pub vision: Option<VisionStatic<B>>,
}

/// A streamable native-vision block or decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend> {
    /// Vision encoder block.
    Vision(VisionBlock<B>),
    /// Text decoder block.
    Text(TransformerBlock<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (1, Unit::Text(unit)) => self.forward_text_unit_with_provider(
                index, unit, hidden, state, forward, pass, provider, context,
            ),
            (_, unit) => <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            ),
        }
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (1, Unit::Text(unit)) => self.forward_text_unit_parallel_with_provider(
                index, unit, hidden, state, forward, pass, provider, parallel, context,
            ),
            (_, unit) => <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            ),
        }
    }
}

/// The same architecture object used by resident and bounded runtimes.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: DecoderConfig,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
}

impl<B: RoutedNeuralBackend> eredu_runtime::ArchitectureParameters<B> for LayeredModel<B> {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl()
    }

    fn parameter_description(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        self.parameter_description_impl()
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        super::static_safetensors_recipes(&self.args, source)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        if let Some(vision) = &self.static_modules.vision {
            visitor.visit("vision", vision)?;
        }
        visitor.visit("embedding", &self.static_modules.text.embeddings)?;
        visitor.visit("norm", &self.static_modules.text.final_norm)?;
        if let Some(head) = &self.static_modules.text.head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(vision) = &mut self.static_modules.vision {
            visitor.visit_mut("vision", vision)?;
        }
        visitor.visit_mut("embedding", &mut self.static_modules.text.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.text.final_norm)?;
        if let Some(head) = &mut self.static_modules.text.head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    #[allow(clippy::too_many_arguments)]
    fn begin_text_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout() != expected {
            return Err(Error::backend(
                "Muse-Glimmer partition state layout mismatch",
            ));
        }
        let (input, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                (TextPartitionInput::Tokens(tokens), tokens.dim(1))
            }
            LayeredPartitionInput::Hidden { hidden, .. } => {
                let sequence = hidden.dim(1);
                (TextPartitionInput::Hidden(hidden), sequence)
            }
        };
        let offset = state
            .layer(first_state_ordinal)
            .map_err(Error::backend)?
            .offset();
        self.begin_routed_text_partition(input, mask, sequence, offset, parallel, context)
    }

    /// Builds unloaded pinned modules.
    pub fn new(
        args: DecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Muse-Glimmer",
            crate::operator_requirements::MUSE_GLIMMER,
        )?;
        let static_modules = StaticModules {
            text: TextStaticModules::new(&args, context)?,
            vision: args
                .vision_config
                .clone()
                .map(|vision| VisionStatic::new(vision, context))
                .transpose()?,
        };
        Ok(Self {
            args,
            static_modules,
            parallel_geometry: None,
        })
    }

    /// Builds the canonical multimodal graph with planner-derived text geometry.
    pub fn new_parallel(
        args: DecoderConfig,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Muse-Glimmer",
            crate::operator_requirements::MUSE_GLIMMER,
        )?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let static_modules = StaticModules {
            text: TextStaticModules::new_parallel(&args, &geometry, context)?,
            vision: args
                .vision_config
                .clone()
                .map(|vision| VisionStatic::new(vision, context))
                .transpose()?,
        };
        Ok(Self {
            args,
            static_modules,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
        })
    }

    /// Returns normalized configuration.
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
    }

    /// Describes pinned multimodal modules and each vision/text graph unit with
    /// explicit neutral ownership.
    fn parameter_description_impl(&self) -> Result<ArchitectureParameterDescription, Error> {
        let graph =
            ExecutionGraph::chain(["vision_encoder", "text_decoder"]).map_err(Error::backend)?;
        let counts = [
            self.args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.layer_count()),
            self.args.num_hidden_layers as usize,
        ];
        let layout = ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)?;
        let text_static = static_parameter_groups(&self.args).map_err(Error::backend)?;
        let vision_static = vision_static_parameter_groups(&self.args).map_err(Error::backend)?;
        let mut expected = text_static
            .iter()
            .chain(&vision_static)
            .cloned()
            .collect::<Vec<_>>();
        let mut owned = text_static
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                OwnedParameterGroupSpec::new(
                    if index == 0 && self.args.tie_word_embeddings {
                        ParameterGroupOwner::static_any_of(["embedding", "output"])
                    } else {
                        ParameterGroupOwner::static_role(match index {
                            0 => "embedding",
                            1 => "norm",
                            _ => "output",
                        })
                    },
                    group,
                )
            })
            .chain(vision_static.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("vision"), group)
            }))
            .collect::<Vec<_>>();
        for (group_index, &count) in counts.iter().enumerate() {
            let owner_group = layout
                .group_id(group_index)
                .expect("Muse layout group")
                .clone();
            for index in 0..count {
                let groups = if group_index == 0 {
                    vision_layer_parameter_groups(&self.args, index)
                } else {
                    layer_parameter_groups(&self.args, index)
                }
                .map_err(Error::backend)?;
                expected.extend(groups.iter().cloned());
                owned.extend(groups.into_iter().map(|group| {
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::execution_unit(owner_group.clone(), index),
                        group,
                    )
                }));
            }
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }

    /// Returns the replicated or planner-derived mutable-state layout.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| state_layout(&self.args).map_err(Error::backend), Ok)
    }

    /// Applies the architecture-owned tensor-parallel token embedding boundary.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            &mut self.static_modules.text.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )
    }

    /// Applies final normalization and architecture-owned tensor-parallel
    /// vocabulary projection, including released logit transforms.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.final_hidden(hidden, context)?;
        let logits = match &mut self.static_modules.text.head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context)?,
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            )?,
        };
        self.static_modules.text.finish_logits(logits, context)
    }

    /// Enters the serial decoder partition through the family embedding boundary.
    pub fn begin_partition_text(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.token_embeddings(tokens, context)
    }

    /// Enters the tensor-parallel decoder partition through lookup and input normalization.
    pub fn begin_partition_text_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.pipeline_embed_parallel(tokens, parallel, context)?;
        self.static_modules
            .text
            .normalize_embeddings(&hidden, context)
    }

    /// Resumes an already embedded decoder partition in the canonical layered context.
    pub fn resume_partition_text(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
    ) -> LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>> {
        LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                parts: Vec::new(),
                vision: None,
            },
        }
    }

    /// Enters or resumes a routed text partition with architecture-owned
    /// causal-mask construction.
    pub fn begin_routed_text_partition(
        &mut self,
        input: TextPartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        sequence: i32,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let hidden = match input {
            TextPartitionInput::Tokens(tokens) => match parallel {
                Some(parallel) => self.begin_partition_text_parallel(tokens, parallel, context)?,
                None => self.begin_partition_text(tokens, context)?,
            },
            TextPartitionInput::Hidden(hidden) => hidden,
        };
        let mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => None,
        };
        Ok(self.resume_partition_text(hidden, mask))
    }

    /// Finishes the serial decoder partition through the family output boundary.
    pub fn finish_partition_text(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_logits(hidden, context)
    }

    /// Finishes the tensor-parallel decoder partition through the family output boundary.
    pub fn finish_partition_text_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
    }

    /// Returns planner-derived geometry for a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_deref()
    }

    /// Shares authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Starts a text-only pass from a rank-local vocabulary embedding shard.
    pub fn begin_parallel_text<S: LayerRuntimeState<B>>(
        &mut self,
        tokens: &B::Tensor,
        embeddings: B::Tensor,
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout().len() != self.args.num_hidden_layers as usize {
            return Err(Error::backend(
                "Muse-Glimmer rank-local state layout mismatch",
            ));
        }
        let hidden = self
            .static_modules
            .text
            .normalize_embeddings(&embeddings, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: None,
                parts: vec![PreparedPart::Text {
                    tokens: tokens.clone(),
                    embeddings,
                }],
                vision: None,
            },
        })
    }

    /// Runs replicated media ingress and assembles rank-local text embeddings
    /// before the tensor-parallel decoder traversal.
    pub fn begin_parallel_input<S: LayerRuntimeState<B>>(
        &mut self,
        input: ModelInput<'_, B::Tensor>,
        text_embeddings: Vec<B::Tensor>,
        vision_blocks: &mut [VisionBlock<B>],
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout().len() != self.args.num_hidden_layers as usize {
            return Err(Error::backend(
                "Muse-Glimmer rank-local state layout mismatch",
            ));
        }
        let mut embeddings = text_embeddings.into_iter();
        let parts = input
            .parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self.static_modules.text.normalize_embeddings(
                        &embeddings.next().ok_or_else(|| {
                            Error::backend("Muse-Glimmer parallel text embedding is missing")
                        })?,
                        context,
                    )?,
                }),
                DecoderInputPart::Media(tokens) => Ok(PreparedPart::Media {
                    tokens: (*tokens).clone(),
                }),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if embeddings.next().is_some() {
            return Err(Error::backend(
                "Muse-Glimmer parallel input has excess text embeddings",
            ));
        }
        let hidden =
            match input.vision {
                Some(vision) => {
                    let config = self.args.vision_config.as_ref().ok_or_else(|| {
                        Error::backend("Muse-Glimmer model has no vision projector")
                    })?;
                    if vision_blocks.len() != config.layer_count() {
                        return Err(Error::backend(
                            "Muse-Glimmer parallel vision block count mismatch",
                        ));
                    }
                    let vision_static = self.static_modules.vision.as_mut().ok_or_else(|| {
                        Error::backend("Muse-Glimmer model has no vision modules")
                    })?;
                    let (mut hidden, vision_state) = vision_static.begin(vision, context)?;
                    for (index, block) in vision_blocks.iter_mut().enumerate() {
                        hidden = block.forward_scheduled(
                            &hidden,
                            config.schedule[index],
                            &vision_state,
                            context,
                        )?;
                    }
                    let media = vision_static.finish(&hidden, &vision_state, context)?;
                    self.assemble(&parts, Some(&media), context)?
                }
                None => self.assemble(&parts, None, context)?,
            };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                parts,
                vision: None,
            },
        })
    }

    /// Executes one decoder block with rank-local projections and collectives.
    pub fn forward_text_unit_parallel<S: LayerRuntimeState<B>>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        unit.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            Some(state.layer(index).map_err(Error::backend)?),
            parallel,
            context,
        )
    }

    /// Executes one rank-local decoder block with a runtime-owned routed bank.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_unit_parallel_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_parallel_with_provider(
            hidden,
            forward.mask.as_ref(),
            Some(state.layer(index).map_err(Error::backend)?),
            pass,
            provider,
            parallel,
            context,
        )
    }

    /// Applies the replicated final normalization before a sharded output head.
    pub fn final_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.final_hidden(hidden, context)
    }

    /// Applies released output scaling and softcapping after vocab gather.
    pub fn finish_parallel_logits(
        &self,
        logits: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.finish_logits(logits, context)
    }

    /// Applies the target's ordinary token embedding and input normalization.
    pub fn token_embeddings(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.embed(tokens, context)
    }

    /// Applies the target-owned final normalization and vocabulary head.
    pub fn project_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.logits(hidden, context)
    }

    /// Executes one text unit while delegating its routed bank to runtime residency.
    pub fn forward_text_unit_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_with_provider(
            hidden,
            forward.mask.as_ref(),
            Some(state.layer(index).map_err(Error::backend)?),
            pass,
            provider,
            context,
        )
    }

    fn prepare_parts(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        if parts.is_empty() {
            return Err(Error::backend("Muse-Glimmer input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self.static_modules.text.embed(tokens, context)?,
                }),
                DecoderInputPart::Media(tokens) => {
                    if tokens.shape().len() != 2 {
                        return Err(Error::backend(
                            "Muse-Glimmer media token IDs must have rank two",
                        ));
                    }
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            })
            .collect()
    }

    fn prepare_parts_parallel(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        if parts.is_empty() {
            return Err(Error::backend("Muse-Glimmer input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => {
                    let embeddings = B::vocabulary_parallel_lookup(
                        &mut self.static_modules.text.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?;
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: self
                            .static_modules
                            .text
                            .normalize_embeddings(&embeddings, context)?,
                    })
                }
                DecoderInputPart::Media(tokens) => {
                    if tokens.shape().len() != 2 {
                        return Err(Error::backend(
                            "Muse-Glimmer media token IDs must have rank two",
                        ));
                    }
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            })
            .collect()
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let media_tokens = parts
            .iter()
            .filter_map(|part| match part {
                PreparedPart::Media { tokens } => Some(tokens.dim(1)),
                PreparedPart::Text { .. } => None,
            })
            .sum::<i32>();
        match vision {
            Some(vision) if vision.shape() != [media_tokens, self.args.hidden_size] => {
                return Err(Error::backend(format!(
                    "Muse-Glimmer projected media has shape {:?}, expected [{media_tokens}, {}]",
                    vision.shape(),
                    self.args.hidden_size
                )))
            }
            None if media_tokens != 0 => {
                return Err(Error::backend(
                    "Muse-Glimmer media placeholders require projected media",
                ))
            }
            _ => {}
        }
        let mut owned_embeddings = Vec::with_capacity(parts.len());
        let mut offset = 0;
        for part in parts {
            match part {
                PreparedPart::Text { embeddings, .. } => {
                    owned_embeddings.push(embeddings.clone());
                }
                PreparedPart::Media { tokens } => {
                    let length = tokens.dim(1);
                    let media = vision
                        .expect("validated media exists")
                        .index(
                            &[
                                eredu_nn::Index::Range(offset, offset + length),
                                eredu_nn::Index::Full,
                            ],
                            context,
                        )?
                        .expand_dims(0, context)?;
                    owned_embeddings.push(media);
                    offset += length;
                }
            }
        }
        let ordered = parts
            .iter()
            .zip(&owned_embeddings)
            .map(|(part, embeddings)| OrderedInputPart {
                token_ids: match part {
                    PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens,
                },
                embeddings,
            })
            .collect::<Vec<_>>();
        Ok(assemble_ordered_inputs(&ordered, self.args.hidden_size, context)?.embeddings)
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Input<'a> = ModelInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        if group == 0 {
            eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
                first_owner_static_roles: vec!["vision".into()],
                last_owner_static_roles: Vec::new(),
                merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
                request_optional: true,
            }
        } else {
            crate::transport::decoder()
        }
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(1, layout)
    }

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::chain(["vision_encoder", "text_decoder"]).map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        match group {
            0 => Ok(self.parallel_geometry.as_ref().map_or_else(
                || {
                    self.args
                        .vision_config
                        .as_ref()
                        .map_or(0, |vision| vision.layer_count())
                },
                |geometry| geometry.vision_layers(),
            )),
            1 => Ok(self.args.num_hidden_layers as usize),
            _ => Err(Error::backend("Muse-Glimmer has two execution groups")),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        let count = match group {
            0 => self
                .args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.layer_count()),
            1 => self.args.num_hidden_layers as usize,
            _ => return Err(Error::backend("Muse-Glimmer has two execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Muse-Glimmer unit is outside its group"));
        }
        match group {
            0 => Ok(format!("model.vision_tower.layers.{index}")),
            1 => Ok(format!("model.layers.{index}")),
            _ => unreachable!(),
        }
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        match group {
            0 => Ok(Unit::Vision(VisionBlock::new(
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?,
                index,
                context,
            )?)),
            1 => {
                let args = self
                    .parallel_geometry
                    .as_ref()
                    .map(|geometry| {
                        geometry.text_block(index).ok_or_else(|| {
                            Error::backend(format!(
                                "Muse-Glimmer text unit {index} has no rank-local geometry"
                            ))
                        })
                    })
                    .transpose()?
                    .unwrap_or(&self.args);
                Ok(Unit::Text(TransformerBlock::new(args, index, context)?))
            }
            _ => Err(Error::backend("Muse-Glimmer has two execution groups")),
        }
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            1 => index,
            _ => index,
        }
    }

    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _ordinal: usize,
    ) -> std::ops::Range<usize> {
        match group {
            0 => 0..0,
            1 => index..index + 1,
            _ => 0..0,
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        if state.layout() != &state_layout(&self.args).map_err(Error::backend)? {
            return Err(Error::backend("Muse-Glimmer runtime state layout mismatch"));
        }
        let parts = self.prepare_parts(input.parts, context)?;
        let (hidden, vision) = match input.vision {
            Some(vision) => {
                let (hidden, state) = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .begin(vision, context)?;
                (hidden, Some(state))
            }
            None => (self.assemble(&parts, None, context)?, None),
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                parts,
                vision,
            },
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1, [vision_or_assembled]) => Ok((*vision_or_assembled).clone()),
            _ => Err(Error::backend(
                "invalid Muse-Glimmer execution dependencies",
            )),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group != 0 || forward.vision.is_some()
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
    ) -> Result<B::Tensor, Self::Error> {
        match (group, unit) {
            (0, Unit::Vision(unit)) => unit.forward_scheduled(
                hidden,
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .schedule[index],
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer vision state is missing"))?,
                context,
            ),
            (1, Unit::Text(unit)) => unit.forward(
                hidden,
                forward.mask.as_ref(),
                Some(state.layer(index).map_err(Error::backend)?),
                context,
            ),
            _ => Err(Error::backend("Muse-Glimmer unit/group mismatch")),
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match (group, forward.vision.as_ref()) {
            (0, Some(vision)) => {
                let media = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .finish(hidden, vision, context)?;
                self.assemble(&forward.parts, Some(&media), context)
            }
            (0, None) | (1, _) => Ok(hidden.clone()),
            _ => Err(Error::backend("invalid Muse-Glimmer execution group")),
        }
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.static_modules.text.logits(hidden, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(forward.mask.iter());
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    values.extend([tokens, embeddings]);
                }
                PreparedPart::Media { tokens } => values.push(tokens),
            }
        }
        if let Some(vision) = &forward.vision {
            values.extend(vision.retained_values());
        }
        values.into_iter()
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| {
                Error::backend("Muse-Glimmer model was not built with rank-local geometry")
            })?
            .state_layout();
        if state.layout() != expected {
            return Err(Error::backend(
                "Muse-Glimmer rank-local state layout mismatch",
            ));
        }
        let parts = self.prepare_parts_parallel(input.parts, parallel, context)?;
        let (hidden, vision) = match input.vision {
            Some(vision) => {
                let (hidden, state) = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .begin(vision, context)?;
                (hidden, Some(state))
            }
            None => (self.assemble(&parts, None, context)?, None),
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                parts,
                vision,
            },
        })
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
    ) -> Result<B::Tensor, Self::Error> {
        match (group, unit) {
            (0, Unit::Vision(unit)) => unit.forward_scheduled(
                hidden,
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .schedule[index],
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer vision state is missing"))?,
                context,
            ),
            (1, Unit::Text(unit)) => self
                .forward_text_unit_parallel(index, unit, hidden, state, forward, parallel, context),
            _ => Err(Error::backend("Muse-Glimmer parallel unit/group mismatch")),
        }
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Muse-Glimmer model was not built with rank-local geometry",
            ));
        }
        let hidden = self.static_modules.text.final_hidden(hidden, context)?;
        let logits = match &mut self.static_modules.text.head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context)?,
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            )?,
        };
        self.static_modules.text.finish_logits(logits, context)
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args().hidden_size,
        ))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_text_partition(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            None,
            context,
        )
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_text_partition(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            Some(parallel),
            context,
        )
    }

    fn enter_partition_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
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
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
}
