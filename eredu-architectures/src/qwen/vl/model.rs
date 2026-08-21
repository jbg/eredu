//! Composite Qwen3-VL lifecycle over the shared vision tower and ordinary Qwen decoder.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingOperator, Error, Index, LinearOperator, NormalizationOperator,
    Parameterized, RotaryPosition, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ExecutionGraph, ExecutionGroupSpec, ExpertPass, LayerRuntimeState, LayeredArchitecture,
    LayeredForwardState, RoutedExpertProvider, RuntimeStateComponents,
};

use crate::qwen::vision::{VisionBlock, VisionInput, VisionMode, VisionState, VisionStatic};
use crate::qwen::{self, AttentionInput};

use super::{
    mrope_embeddings, multimodal_position_ids, position_ids_tensor, ModelArgs, PositionPart,
};

/// One semantic segment in decoder order.
pub enum InputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image placeholders and their unmerged patch grids.
    Image {
        /// Placeholder IDs shaped `[1, merged_patches]`.
        tokens: &'a T,
        /// One or more `(time, height, width)` patch grids.
        grid: &'a [(i32, i32, i32)],
    },
    /// Video placeholders and their unmerged patch grids.
    Video {
        /// Placeholder IDs shaped `[1, merged_patches]`.
        tokens: &'a T,
        /// One or more `(time, height, width)` patch grids.
        grid: &'a [(i32, i32, i32)],
    },
    /// Already projected decoder-width embeddings.
    Projected {
        /// Semantic token identities.
        tokens: &'a T,
        /// Embeddings shaped `[1, sequence, hidden]`.
        embeddings: &'a T,
    },
}

/// Prepared text and optional model-native visual input.
pub struct ModelInput<'a, T> {
    /// Ordered text and media segments.
    pub parts: &'a [InputPart<'a, T>],
    /// Flattened patches for every image/video part in order.
    pub pixels: Option<&'a T>,
    /// Optional explicit text attention mask.
    pub mask: Option<&'a T>,
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Media { tokens: T },
}

/// Pinned text and shared-vision modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: RoutedNeuralBackend> {
    /// Ordinary Qwen embeddings, final norm, and vocabulary head.
    pub text: qwen::StaticModules<B>,
    /// Qwen shared vision patch, position, and merger modules.
    pub vision: VisionStatic<B>,
}

/// One streamable vision or ordinary Qwen text unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend> {
    /// Shared vision transformer block.
    Vision(VisionBlock<B>),
    /// Existing neutral ordinary Qwen dense-or-MoE block.
    Text(qwen::TransformerBlock<B>),
}

/// Architecture-owned values retained for one complete pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    tokens: Option<T>,
    parts: Vec<PreparedPart<T>>,
    rotary: (T, T),
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
    visual_mask: Option<T>,
}

/// Request-scoped state transported while pipeline owners execute the shared
/// vision tower.
pub struct PipelineVisionState<T> {
    /// Current vision activation.
    pub hidden: T,
    parts: Vec<PreparedPart<T>>,
    rotary: (T, T),
    delta: T,
    mask: Option<T>,
    vision: Option<VisionState<T>>,
    vision_output: Option<T>,
}

/// Decoder-facing values produced after the placed vision group completes.
pub struct PipelinePrepared<T> {
    /// Assembled text-width activation.
    pub hidden: T,
    /// Text mRoPE cosine.
    pub cosine: T,
    /// Text mRoPE sine.
    pub sine: T,
    /// Persisted decode position delta.
    pub position_delta: T,
    /// Optional explicit or causal mask.
    pub mask: Option<T>,
    /// Selected raw DeepStack features.
    pub deepstack: Vec<T>,
    /// Image-or-video placeholder mask used for DeepStack scatter.
    pub visual_mask: Option<T>,
}

/// One neutral composite model for dense and MoE Qwen3-VL.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: ModelArgs,
    static_modules: StaticModules<B>,
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded modules with their canonical checkpoint identities.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let text = qwen::StaticModules::new(&args.text, context)?;
        let vision = VisionStatic::new_with_root(
            args.vision.clone(),
            VisionMode::DeepStack,
            "model.visual",
            context,
        )?;
        Ok(Self {
            args,
            static_modules: StaticModules { text, vision },
        })
    }

    /// Returns normalized nested text and vision policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn prepare_parts(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        if parts.is_empty() {
            return Err(Error::backend("Qwen3-VL input has no ordered parts"));
        }
        let mut grids = Vec::new();
        let prepared = parts
            .iter()
            .map(|part| match part {
                InputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self
                        .static_modules
                        .text
                        .embeddings
                        .forward(tokens, context)?,
                }),
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(Error::backend("Qwen3-VL projected input geometry mismatch"));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok((prepared, grids))
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        let media_tokens = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Media { tokens } => tokens.dim(1),
                _ => 0,
            })
            .sum::<i32>();
        match vision {
            Some(value) if value.shape() == [1, media_tokens, self.args.text.hidden_size] => {}
            None if media_tokens == 0 => {}
            Some(value) => {
                return Err(Error::backend(format!(
                    "Qwen3-VL vision output {:?} does not match {media_tokens} placeholders",
                    value.shape()
                )))
            }
            None => {
                return Err(Error::backend(
                    "Qwen3-VL media placeholders require vision output",
                ))
            }
        }
        let mut offset = 0;
        let mut embeddings = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
                PreparedPart::Media { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(vision.expect("validated vision output").index(
                        &[
                            Index::Full,
                            Index::Range(offset, offset + length),
                            Index::Full,
                        ],
                        context,
                    )?);
                    offset += length;
                }
            }
        }
        let ordered = parts
            .iter()
            .zip(&embeddings)
            .map(|(part, embeddings)| OrderedInputPart {
                token_ids: match part {
                    PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens,
                },
                embeddings,
            })
            .collect::<Vec<_>>();
        assemble_ordered_inputs(&ordered, self.args.text.hidden_size, context)
    }

    fn finish_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
        match &mut self.static_modules.text.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self
                .static_modules
                .text
                .embeddings
                .as_linear(&hidden, context),
        }
    }

    /// Prepares a pipeline request without binding it to a concrete cache
    /// container. The caller supplies the current token offset and persisted
    /// multimodal delta from the generic state slots.
    pub fn begin_pipeline<'a>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        offset: i32,
        persisted_delta: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelineVisionState<B::Tensor>, Error> {
        let (parts, grids) = self.prepare_parts(input.parts, context)?;
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(Error::backend(
                "Qwen3-VL pixels and media metadata must appear together",
            ));
        }
        if media && offset != 0 {
            return Err(Error::backend(
                "Qwen3-VL media input cannot append to a populated cache",
            ));
        }
        let (vision_initial, vision) = match input.pixels {
            Some(pixels) => {
                let (hidden, state) = self.static_modules.vision.begin(
                    VisionInput {
                        pixels,
                        grid: &grids,
                    },
                    context,
                )?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let position_parts = input
            .parts
            .iter()
            .map(|part| match part {
                InputPart::Text(tokens) | InputPart::Projected { tokens, .. } => {
                    PositionPart::Text(tokens.dim(1))
                }
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => {
                    PositionPart::Media(grid)
                }
            })
            .collect::<Vec<_>>();
        let (mut positions, computed_delta) = multimodal_position_ids(
            &position_parts,
            self.args.vision.spatial_merge_size,
            sequence,
        )
        .map_err(Error::backend)?;
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = position_ids_tensor::<B::Tensor>(&positions, context)?;
        let delta = if media || persisted_delta.is_none() {
            B::Tensor::full_i32(computed_delta, &[1], context)?
        } else {
            persisted_delta.expect("checked persisted delta").clone()
        };
        if !media {
            positions = positions.add(&delta, context)?;
        }
        let rotary = mrope_embeddings(
            &positions,
            self.args.text.head_dim,
            self.args.text.rope_theta,
            &self.args.mrope_section,
            context,
        )?;
        let mask = if let Some(mask) = input.mask {
            Some(mask.clone())
        } else if sequence > 1 {
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let assembled = self.assemble(&parts, None, context);
        let hidden = match vision_initial {
            Some(hidden) => hidden,
            None => assembled?.embeddings,
        };
        Ok(PipelineVisionState {
            hidden,
            parts,
            rotary,
            delta,
            mask,
            vision,
            vision_output: None,
        })
    }

    /// Whether this request owns visual component work.
    pub fn pipeline_vision_active(state: &PipelineVisionState<B::Tensor>) -> bool {
        state.vision.is_some()
    }

    /// Exports all request tensors needed by a downstream vision owner.
    pub fn pipeline_retained_values(state: &PipelineVisionState<B::Tensor>) -> Vec<B::Tensor> {
        let mut values = vec![
            state.hidden.clone(),
            state.rotary.0.clone(),
            state.rotary.1.clone(),
            state.delta.clone(),
        ];
        values.extend(state.mask.iter().cloned());
        for part in &state.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    values.extend([tokens.clone(), embeddings.clone()]);
                }
                PreparedPart::Media { tokens } => values.push(tokens.clone()),
            }
        }
        if let Some(vision) = &state.vision {
            values.extend(vision.retained_values().cloned());
        }
        values
    }

    /// Replaces request tensors with values transported from the previous
    /// component owner. Structure is validated against the locally rebuilt
    /// parameter-free request state.
    pub fn replace_pipeline_retained_values(
        state: &mut PipelineVisionState<B::Tensor>,
        values: Vec<B::Tensor>,
    ) -> Result<(), Error> {
        let fixed = 4
            + usize::from(state.mask.is_some())
            + state
                .parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Text { .. } => 2,
                    PreparedPart::Media { .. } => 1,
                })
                .sum::<usize>();
        let minimum = fixed + usize::from(state.vision.is_some()) * 2;
        if values.len() < minimum || (state.vision.is_none() && values.len() != fixed) {
            return Err(Error::backend(format!(
                "Qwen3-VL pipeline continuation received {} tensors, expected at least {minimum}",
                values.len(),
            )));
        }
        let mut values = values.into_iter();
        state.hidden = values.next().expect("validated hidden");
        state.rotary.0 = values.next().expect("validated cosine");
        state.rotary.1 = values.next().expect("validated sine");
        state.delta = values.next().expect("validated delta");
        if state.mask.is_some() {
            state.mask = Some(values.next().expect("validated mask"));
        }
        for part in &mut state.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    *tokens = values.next().expect("validated tokens");
                    *embeddings = values.next().expect("validated embeddings");
                }
                PreparedPart::Media { tokens } => {
                    *tokens = values.next().expect("validated media tokens");
                }
            }
        }
        if let Some(vision) = &mut state.vision {
            vision.replace_retained_values(values.collect())?;
        }
        Ok(())
    }

    /// Executes one placed shared-vision block.
    pub fn forward_pipeline_vision(
        &mut self,
        index: usize,
        block: &mut VisionBlock<B>,
        state: &mut PipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        let vision = state
            .vision
            .as_mut()
            .ok_or_else(|| Error::backend("missing Qwen3-VL pipeline vision state"))?;
        state.hidden = match parallel {
            Some(parallel) => self.static_modules.vision.forward_block_parallel(
                block,
                index,
                &state.hidden,
                vision,
                parallel,
                context,
            )?,
            None => self.static_modules.vision.forward_block(
                block,
                index,
                &state.hidden,
                vision,
                context,
            )?,
        };
        Ok(())
    }

    /// Finishes the vision merger and assembles decoder-width input.
    pub fn finish_pipeline(
        &mut self,
        mut state: PipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelinePrepared<B::Tensor>, Error> {
        let deepstack = if let Some(vision) = &mut state.vision {
            let output = match parallel {
                Some(parallel) => self.static_modules.vision.finish_parallel(
                    &state.hidden,
                    vision,
                    parallel,
                    context,
                )?,
                None => self
                    .static_modules
                    .vision
                    .finish(&state.hidden, vision, context)?,
            };
            state.vision_output = Some(output.embeddings);
            output.deepstack_features
        } else {
            Vec::new()
        };
        let assembled = self.assemble(&state.parts, state.vision_output.as_ref(), context)?;
        let visual_mask = if deepstack.is_empty() {
            None
        } else {
            Some(
                assembled
                    .token_ids
                    .equal_i32(self.args.image_token_id, context)?
                    .logical_or(
                        &assembled
                            .token_ids
                            .equal_i32(self.args.video_token_id, context)?,
                        context,
                    )?,
            )
        };
        let deepstack = match visual_mask.as_ref() {
            Some(mask) => deepstack
                .into_iter()
                .map(|features| {
                    assembled.embeddings.zeros_like(context)?.masked_scatter(
                        mask,
                        &features.index(&[Index::At(0), Index::Full, Index::Full], context)?,
                        context,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?,
            None => deepstack,
        };
        Ok(PipelinePrepared {
            hidden: assembled.embeddings,
            cosine: state.rotary.0,
            sine: state.rotary.1,
            position_delta: state.delta,
            mask: state.mask,
            deepstack,
            visual_mask: None,
        })
    }

    /// Executes one ordinary-Qwen text block from a pipeline payload.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pipeline_text<S, P>(
        &mut self,
        index: usize,
        block: &mut qwen::TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        prepared: &PipelinePrepared<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let feed_forward = |policy: &mut qwen::FeedForward<B>,
                            normalized: &B::Tensor,
                            context: &<B::Tensor as Tensor>::Context| {
            let shape = normalized.shape().to_vec();
            let flat =
                normalized.reshape(&[-1, normalized.dim(normalized.shape().len() - 1)], context)?;
            let forwarded = match parallel {
                Some(parallel) => policy.forward_with_provider_parallel(
                    index, pass, &flat, parallel, context, provider,
                )?,
                None => policy.forward_with_provider(index, pass, &flat, context, provider)?,
            };
            forwarded.reshape(&shape, context)
        };
        let input = AttentionInput {
            hidden,
            mask: prepared.mask.as_ref(),
            cache: Some(state),
            allow_sliding_prefill: true,
            rotary_position: Some(RotaryPosition::Embeddings {
                cosine: &prepared.cosine,
                sine: &prepared.sine,
            }),
        };
        let mut output = match parallel {
            Some(parallel) => block.forward_tensor_parallel_with_feed_forward(
                input,
                parallel,
                context,
                feed_forward,
            )?,
            None => block.forward_with_feed_forward(input, context, feed_forward)?,
        };
        if let Some(features) = prepared.deepstack.get(index) {
            output = if features.shape() == output.shape() {
                output.add(features, context)?
            } else {
                let source = features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                output.add(
                    &output.zeros_like(context)?.masked_scatter(
                        prepared
                            .visual_mask
                            .as_ref()
                            .ok_or_else(|| Error::backend("missing Qwen3-VL visual mask"))?,
                        &source,
                        context,
                    )?,
                    context,
                )?
            };
        }
        Ok(output)
    }

    /// Applies the shared final norm and vocabulary projection.
    pub fn finish_pipeline_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_logits(hidden, context)
    }

    /// Executes one unit while routing ordinary-Qwen MoE banks through a
    /// runtime-owned provider. Vision units retain the shared vision path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                context,
            ),
            (1, Unit::Text(block)) => {
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let mask = forward.mask.as_ref();
                let cosine = &forward.rotary.0;
                let sine = &forward.rotary.1;
                let mut output = block.forward_with_feed_forward(
                    AttentionInput {
                        hidden,
                        mask,
                        cache: Some(state.layer(index).map_err(Error::backend)?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings { cosine, sine }),
                    },
                    context,
                    |policy, normalized, context| {
                        let shape = normalized.shape().to_vec();
                        let flat = normalized.reshape(
                            &[-1, normalized.dim(normalized.shape().len() - 1)],
                            context,
                        )?;
                        policy
                            .forward_with_provider(index, pass, &flat, context, provider)?
                            .reshape(&shape, context)
                    },
                )?;
                if let Some(features) = forward.deepstack.get(index) {
                    let source =
                        features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                    output = output.add(
                        &output.zeros_like(context)?.masked_scatter(
                            forward
                                .visual_mask
                                .as_ref()
                                .ok_or_else(|| Error::backend("missing Qwen3-VL visual mask"))?,
                            &source,
                            context,
                        )?,
                        context,
                    )?;
                }
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL unit/group mismatch")),
        }
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
        ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::with_dependencies("text_decoder", ["vision"]),
            ],
            "text_decoder",
        )
        .map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        match group {
            0 => Ok(self.args.vision.layer_count()),
            1 => usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend),
            _ => Err(Error::backend("Qwen3-VL has two execution groups")),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        let count = match group {
            0 => self.args.vision.layer_count(),
            1 => usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend)?,
            _ => return Err(Error::backend("Qwen3-VL has two execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Qwen3-VL unit is outside its group"));
        }
        match group {
            0 => Ok(format!("model.visual.blocks.{index}")),
            1 => Ok(format!("{}.layers.{index}", self.args.text.parameter_root)),
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
        let count = match group {
            0 => self.args.vision.layer_count(),
            1 => usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend)?,
            _ => return Err(Error::backend("Qwen3-VL has two execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Qwen3-VL unit is outside its group"));
        }
        match group {
            0 => Ok(Unit::Vision(VisionBlock::new_with_root(
                &self.args.vision,
                "model.visual",
                index,
                context,
            )?)),
            1 => Ok(Unit::Text(qwen::new_block(
                &self.args.text,
                index,
                context,
            )?)),
            _ => unreachable!(),
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = super::state_layout(&self.args).map_err(Error::backend)?;
        if state.layout() != &expected {
            return Err(Error::backend("Qwen3-VL runtime state layout mismatch"));
        }
        let (parts, grids) = self.prepare_parts(input.parts, context)?;
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(Error::backend(
                "Qwen3-VL pixels and media metadata must appear together",
            ));
        }
        let (vision_initial, vision_state) = match input.pixels {
            Some(pixels) => {
                let (hidden, state) = self.static_modules.vision.begin(
                    VisionInput {
                        pixels,
                        grid: &grids,
                    },
                    context,
                )?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let assembled = self.assemble(&parts, None, context);
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let state_layer = state.layer(0).map_err(Error::backend)?;
        let offset = state_layer.position();
        let mut position_parts = Vec::with_capacity(input.parts.len());
        for part in input.parts {
            match part {
                InputPart::Text(tokens) | InputPart::Projected { tokens, .. } => {
                    position_parts.push(PositionPart::Text(tokens.dim(1)))
                }
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => {
                    position_parts.push(PositionPart::Media(grid))
                }
            }
        }
        let (mut positions, computed_delta) = multimodal_position_ids(
            &position_parts,
            self.args.vision.spatial_merge_size,
            sequence,
        )
        .map_err(Error::backend)?;
        if media && offset != 0 {
            return Err(Error::backend(
                "Qwen3-VL media input cannot append to a populated cache",
            ));
        }
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = position_ids_tensor::<B::Tensor>(&positions, context)?;
        let delta = state_layer
            .fixed_component(StateTensorRole::PositionDelta)
            .map_err(Error::backend)?;
        if media || delta.is_none() {
            *delta = Some(B::Tensor::full_i32(computed_delta, &[1], context)?);
        } else if let Some(delta) = delta.as_ref() {
            positions = positions.add(delta, context)?;
        }
        let rotary = mrope_embeddings(
            &positions,
            self.args.text.head_dim,
            self.args.text.rope_theta,
            &self.args.mrope_section,
            context,
        )?;
        let mask = if let Some(mask) = input.mask {
            Some(mask.clone())
        } else if sequence > 1 {
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let (assembled_tokens, assembled_hidden, assembled_error) = match assembled {
            Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
            Err(error) => (None, None, Some(error)),
        };
        let hidden = vision_initial
            .as_ref()
            .cloned()
            .or(assembled_hidden)
            .ok_or_else(|| {
                assembled_error.unwrap_or_else(|| Error::backend("empty Qwen3-VL input"))
            })?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                tokens: assembled_tokens,
                parts,
                rotary,
                vision_state,
                vision_initial,
                vision_output: None,
                deepstack: Vec::new(),
                visual_mask: None,
            },
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        _dependencies: &[&B::Tensor],
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match group {
            0 => Ok(forward.vision_initial.as_ref().unwrap_or(initial).clone()),
            1 => {
                let assembled =
                    self.assemble(&forward.parts, forward.vision_output.as_ref(), context)?;
                forward.visual_mask = if forward.deepstack.is_empty() {
                    None
                } else {
                    Some(
                        assembled
                            .token_ids
                            .equal_i32(self.args.image_token_id, context)?
                            .logical_or(
                                &assembled
                                    .token_ids
                                    .equal_i32(self.args.video_token_id, context)?,
                                context,
                            )?,
                    )
                };
                forward.tokens = Some(assembled.token_ids);
                Ok(assembled.embeddings)
            }
            _ => Err(Error::backend("invalid Qwen3-VL execution group")),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 1 || (group == 0 && forward.vision_state.is_some())
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            1 => index,
            _ => index,
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
    ) -> Result<B::Tensor, Self::Error> {
        match (group, unit) {
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                context,
            ),
            (1, Unit::Text(block)) => {
                let mut output = block.forward(
                    AttentionInput {
                        hidden,
                        mask: forward.mask.as_ref(),
                        cache: Some(state.layer(index).map_err(|error| {
                            Error::backend(format!(
                                "Qwen3-VL text group {group} unit {index} cache: {error}"
                            ))
                        })?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings {
                            cosine: &forward.rotary.0,
                            sine: &forward.rotary.1,
                        }),
                    },
                    context,
                )?;
                if let Some(features) = forward.deepstack.get(index) {
                    let source =
                        features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                    output = output.add(
                        &output.zeros_like(context)?.masked_scatter(
                            forward
                                .visual_mask
                                .as_ref()
                                .ok_or_else(|| Error::backend("missing Qwen3-VL visual mask"))?,
                            &source,
                            context,
                        )?,
                        context,
                    )?;
                }
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL unit/group mismatch")),
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
        if group == 0 && forward.vision_state.is_some() {
            let output = self.static_modules.vision.finish(
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .expect("validated vision state"),
                context,
            )?;
            forward.deepstack = output.deepstack_features;
            forward.vision_output = Some(output.embeddings);
            return Ok(forward
                .vision_output
                .as_ref()
                .expect("installed vision output")
                .clone());
        }
        Ok(hidden.clone())
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_logits(hidden, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(forward.mask.iter());
        values.extend(forward.tokens.iter());
        values.extend([&forward.rotary.0, &forward.rotary.1]);
        values.extend(forward.vision_initial.iter());
        values.extend(forward.vision_output.iter());
        values.extend(forward.deepstack.iter());
        values.extend(forward.visual_mask.iter());
        if let Some(state) = &forward.vision_state {
            values.extend(state.retained_values());
        }
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => values.extend([tokens, embeddings]),
                PreparedPart::Media { tokens } => values.push(tokens),
            }
        }
        values.into_iter()
    }
}
