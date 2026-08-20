//! One neutral Muse-Glimmer multimodal model for resident and bounded runtimes.

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, Error, Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ExecutionGraph, ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    RoutedExpertProvider,
};

use super::{
    state_layout, DecoderConfig, StaticModules as TextStaticModules, TransformerBlock, VisionBlock,
    VisionInput, VisionState, VisionStatic,
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
    /// Patch/position modules, merge adapter, and language projection.
    pub vision: VisionStatic<B>,
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

/// The same architecture object used by resident and bounded runtimes.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: DecoderConfig,
    static_modules: StaticModules<B>,
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded pinned modules.
    pub fn new(
        args: DecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let static_modules = StaticModules {
            text: TextStaticModules::new(&args, context)?,
            vision: VisionStatic::new(args.vision_config.clone(), context)?,
        };
        Ok(Self {
            args,
            static_modules,
        })
    }

    /// Returns normalized configuration.
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
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
        let hidden = match input.vision {
            Some(vision) => {
                if vision_blocks.len() != self.args.vision_config.layer_count() {
                    return Err(Error::backend(
                        "Muse-Glimmer parallel vision block count mismatch",
                    ));
                }
                let (mut hidden, vision_state) =
                    self.static_modules.vision.begin(vision, context)?;
                for (index, block) in vision_blocks.iter_mut().enumerate() {
                    hidden = block.forward_scheduled(
                        &hidden,
                        self.args.vision_config.schedule[index],
                        &vision_state,
                        context,
                    )?;
                }
                let media = self
                    .static_modules
                    .vision
                    .finish(&hidden, &vision_state, context)?;
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

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::chain(["vision", "text_decoder"]).map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        match group {
            0 => Ok(self.args.vision_config.layer_count()),
            1 => Ok(self.args.num_hidden_layers as usize),
            _ => Err(Error::backend("Muse-Glimmer has two execution groups")),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        let count = match group {
            0 => self.args.vision_config.layer_count(),
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
                &self.args.vision_config,
                index,
                context,
            )?)),
            1 => Ok(Unit::Text(TransformerBlock::new(
                &self.args, index, context,
            )?)),
            _ => Err(Error::backend("Muse-Glimmer has two execution groups")),
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
                let (hidden, state) = self.static_modules.vision.begin(vision, context)?;
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
                self.args.vision_config.schedule[index],
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
                let media = self.static_modules.vision.finish(hidden, vision, context)?;
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
