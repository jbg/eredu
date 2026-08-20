//! One neutral Inkling multimodal model for resident and bounded runtimes.

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AuxiliaryConvolutionState, EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator,
    LinearSpec, NormalizationOperator, NormalizationSpec, ParameterSpec, Parameterized,
    RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ExecutionGraph, ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    RoutedExpertProvider,
};

use super::{
    state_layout, AudioInput, AudioTower, DecoderLayer, ModelArgs, MtpModel, MtpOutput,
    VisionLayer, VisionStatic,
};

/// Pinned text, audio, and image modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: RoutedNeuralBackend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Required normalization after complete text/media assembly.
    pub embedding_norm: B::Normalization,
    /// Final decoder normalization.
    pub final_norm: B::Normalization,
    /// Untied output projection.
    pub output: B::Linear,
    /// Optional checkpoint-embedded multi-token predictor.
    pub mtp: Option<MtpModel<B>>,
    /// Optional pinned dMel projector.
    pub audio: Option<AudioTower<B>>,
    /// Optional pinned hMLP final normalization.
    pub vision: Option<VisionStatic<B>>,
}

impl<B: RoutedNeuralBackend> StaticModules<B> {
    fn new(args: &ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let text = &args.text_config;
        let norm = |name: &str| {
            B::rms_norm(
                NormalizationSpec {
                    dimensions: text.hidden_size,
                    epsilon: text.rms_norm_eps,
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                },
                context,
            )
        };
        Ok(Self {
            embeddings: B::embedding(
                EmbeddingSpec {
                    vocabulary: text.vocab_size,
                    dimensions: text.hidden_size,
                    weight: ParameterSpec::trainable("model.embed_tokens.weight")
                        .map_err(Error::backend)?,
                    quantization: text
                        .linear_format_for("model.embed_tokens.weight")
                        .weight_quantization(),
                },
                context,
            )?,
            embedding_norm: norm("model.embed_norm.weight")?,
            final_norm: norm("model.norm.weight")?,
            output: B::linear(
                LinearSpec {
                    input: text.hidden_size,
                    output: text.vocab_size,
                    weight: ParameterSpec::trainable("lm_head.weight").map_err(Error::backend)?,
                    bias: None,
                    format: text.linear_format_for("lm_head.weight"),
                },
                context,
            )?,
            mtp: MtpModel::new(args, context)?,
            audio: args
                .audio_config
                .as_ref()
                .map(|audio| AudioTower::new(audio, context))
                .transpose()?,
            vision: args
                .vision_config
                .as_ref()
                .map(|vision| VisionStatic::new(vision, context))
                .transpose()?,
        })
    }
}

/// One ordered decoder-ingress segment.
pub enum DecoderInputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image placeholder IDs matching projected hMLP output.
    Image(&'a T),
    /// Audio placeholder IDs matching projected dMel frames.
    Audio(&'a T),
    /// Caller-supplied decoder-width embeddings with explicit token identity.
    Projected {
        /// Semantic token IDs used for cache and position identity.
        tokens: &'a T,
        /// Decoder-width embeddings that bypass native media towers.
        embeddings: &'a T,
    },
}

/// Prepared token and optional native media input.
pub struct ModelInput<'a, T> {
    /// Ordered text/image/audio segments.
    pub parts: &'a [DecoderInputPart<'a, T>],
    /// Optional hMLP patches shaped `[patches, 2, 40, 40, 3]`.
    pub vision_patches: Option<&'a T>,
    /// Optional prepared dMel code IDs and valid-frame extent.
    pub audio: Option<AudioInput<'a, T>>,
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Image { tokens: T },
    Audio { tokens: T },
    Projected { tokens: T, embeddings: T },
}

/// A streamable hMLP stage or text decoder layer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend> {
    /// Folded hMLP image stage.
    Vision(VisionLayer<B>),
    /// Stateful text decoder layer.
    Text(DecoderLayer<B>),
}

/// Transient values retained across component groups.
pub struct ForwardContext<T> {
    parts: Vec<PreparedPart<T>>,
    tokens: T,
    audio: Option<T>,
    has_vision: bool,
    target_hidden: Option<T>,
}

/// Inkling architecture shared by resident, layerwise, and streamed runtimes.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: ModelArgs,
    static_modules: StaticModules<B>,
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded pinned modules from normalized family configuration.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let static_modules = StaticModules::new(&args, context)?;
        Ok(Self {
            args,
            static_modules,
        })
    }

    /// Returns normalized family configuration.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Starts a text-only pass from an embedding produced by a backend-owned
    /// vocabulary-parallel table.
    pub fn begin_parallel_text<S: LayerRuntimeState<B>>(
        &mut self,
        tokens: &B::Tensor,
        embeddings: B::Tensor,
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
    {
        if state.layout().len() != self.args.text_config.num_hidden_layers as usize {
            return Err(Error::backend("Inkling rank-local state layout mismatch"));
        }
        let hidden = self
            .static_modules
            .embedding_norm
            .forward(&embeddings, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts: vec![PreparedPart::Text {
                    tokens: tokens.clone(),
                    embeddings,
                }],
                tokens: tokens.clone(),
                audio: None,
                has_vision: false,
                target_hidden: None,
            },
        })
    }

    /// Starts a multimodal pass from rank-local text embeddings while keeping
    /// hMLP, dMel, normalization, and ordered assembly in the neutral model.
    pub fn begin_parallel_input<S: LayerRuntimeState<B>>(
        &mut self,
        input: ModelInput<'_, B::Tensor>,
        text_embeddings: &[B::Tensor],
        vision_layers: &mut [VisionLayer<B>],
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
    {
        if state.layout().len() != self.args.text_config.num_hidden_layers as usize {
            return Err(Error::backend("Inkling rank-local state layout mismatch"));
        }
        let mut next_embedding = text_embeddings.iter();
        let parts = input
            .parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: next_embedding
                        .next()
                        .ok_or_else(|| Error::backend("missing Inkling rank-local text embedding"))?
                        .clone(),
                }),
                DecoderInputPart::Image(tokens) => Ok(PreparedPart::Image {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => Ok(PreparedPart::Projected {
                    tokens: (*tokens).clone(),
                    embeddings: (*embeddings).clone(),
                }),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if next_embedding.next().is_some() {
            return Err(Error::backend("unused Inkling rank-local text embedding"));
        }
        let tokens = ordered_tokens(&parts, context)?;
        let audio = match input.audio {
            Some(audio) => Some(
                self.static_modules
                    .audio
                    .as_mut()
                    .ok_or_else(|| Error::backend("Inkling has no audio tower"))?
                    .forward(audio, context)?,
            ),
            None => None,
        };
        let vision = match input.vision_patches {
            Some(patches) => {
                if patches.shape().len() != 5 || patches.shape()[1..] != [2, 40, 40, 3] {
                    return Err(Error::backend("invalid Inkling hMLP patch geometry"));
                }
                let expected = self
                    .args
                    .vision_config
                    .as_ref()
                    .map_or(0, |vision| vision.num_hidden_layers as usize);
                if vision_layers.len() != expected {
                    return Err(Error::backend("Inkling hMLP layer count mismatch"));
                }
                let mut hidden = patches.clone();
                for layer in vision_layers {
                    hidden = layer.forward(&hidden, context)?;
                }
                Some(
                    self.static_modules
                        .vision
                        .as_mut()
                        .ok_or_else(|| Error::backend("Inkling has no vision tower"))?
                        .finish(&hidden, context)?,
                )
            }
            None => None,
        };
        let hidden = self.assemble(&parts, vision.as_ref(), audio.as_ref(), context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts,
                tokens,
                audio,
                has_vision: vision.is_some(),
                target_hidden: None,
            },
        })
    }

    /// Executes one ordinary text layer with rank-local projections.
    pub fn forward_text_unit_parallel<S: LayerRuntimeState<B>>(
        &mut self,
        index: usize,
        unit: &mut DecoderLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
    {
        unit.forward_parallel(
            hidden,
            Some(state.layer(index).map_err(Error::backend)?),
            parallel,
            context,
        )
    }

    /// Executes one TP text layer while runtime owns routed expert residency.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_unit_parallel_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut DecoderLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_parallel_with_provider(
            hidden,
            Some(state.layer(index).map_err(Error::backend)?),
            pass,
            provider,
            parallel,
            context,
        )
    }

    /// Applies the replicated final normalization and muP scaling before a
    /// backend-owned vocabulary-parallel output head.
    pub fn final_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .final_norm
            .forward(hidden, context)?
            .multiply_scalar(
                self.args.text_config.logits_mup_width_multiplier.recip(),
                context,
            )
    }

    /// Returns the checkpoint-embedded prediction depth count.
    pub fn mtp_len(&self) -> usize {
        self.static_modules.mtp.as_ref().map_or(0, MtpModel::len)
    }

    /// Returns one embedded prediction depth's attention policy.
    pub fn mtp_policy(&self, depth: usize) -> Option<eredu_core::AttentionPolicy> {
        self.static_modules
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.policy(depth))
    }

    /// Embeds predictor token identities through the ordinary shared ingress.
    pub fn mtp_token_embeddings(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embeddings = self.static_modules.embeddings.forward(tokens, context)?;
        self.static_modules
            .embedding_norm
            .forward(&embeddings, context)
    }

    /// Normalizes predictor embeddings supplied by a backend-owned sharded
    /// vocabulary table.
    pub fn normalize_mtp_embeddings(
        &mut self,
        embeddings: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .embedding_norm
            .forward(embeddings, context)
    }

    /// Applies the target muP scale before a backend-owned sharded MTP head.
    pub fn final_mtp_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        hidden.multiply_scalar(
            self.args.text_config.logits_mup_width_multiplier.recip(),
            context,
        )
    }

    /// Executes one checkpoint-embedded prediction depth.
    pub fn forward_mtp_step<S>(
        &mut self,
        hidden: &B::Tensor,
        embeddings: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut [S],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MtpOutput<B::Tensor>, Error>
    where
        S: eredu_nn::AttentionCache<B::Tensor> + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    {
        self.static_modules
            .mtp
            .as_mut()
            .ok_or_else(|| Error::backend("Inkling checkpoint has no MTP predictor"))?
            .forward_step(hidden, embeddings, tokens, depth, state, context)
    }

    /// Applies the target-owned muP-scaled vocabulary projection to MTP hidden state.
    pub fn project_mtp_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.final_mtp_parallel_hidden(hidden, context)?;
        let logits = self.static_modules.output.forward(&hidden, context)?;
        let vocabulary = self
            .args
            .text_config
            .unpadded_vocab_size
            .unwrap_or(self.args.text_config.vocab_size);
        if vocabulary == self.args.text_config.vocab_size {
            return Ok(logits);
        }
        let mut indexes = vec![Index::Full; logits.shape().len()];
        *indexes.last_mut().expect("logits have vocabulary axis") = Index::Range(0, vocabulary);
        logits.index(&indexes, context)
    }

    /// Executes one text unit while delegating routed and shared banks to runtime residency.
    pub fn forward_text_unit_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut DecoderLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_with_provider(
            hidden,
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
            return Err(Error::backend("Inkling input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self.static_modules.embeddings.forward(tokens, context)?,
                }),
                DecoderInputPart::Image(tokens) => Ok(PreparedPart::Image {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => Ok(PreparedPart::Projected {
                    tokens: (*tokens).clone(),
                    embeddings: (*embeddings).clone(),
                }),
            })
            .collect()
    }

    fn assemble(
        &mut self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        audio: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let image_tokens =
            part_token_count(parts, |part| matches!(part, PreparedPart::Image { .. }));
        let audio_tokens =
            part_token_count(parts, |part| matches!(part, PreparedPart::Audio { .. }));
        validate_component(
            "image",
            vision,
            image_tokens,
            self.args.text_config.hidden_size,
        )?;
        validate_component(
            "audio",
            audio,
            audio_tokens,
            self.args.text_config.hidden_size,
        )?;
        let mut image_offset = 0;
        let mut audio_offset = 0;
        let mut embeddings = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
                PreparedPart::Image { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(slice_component(
                        vision.expect("validated image component"),
                        image_offset,
                        length,
                        context,
                    )?);
                    image_offset += length;
                }
                PreparedPart::Audio { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(slice_component(
                        audio.expect("validated audio component"),
                        audio_offset,
                        length,
                        context,
                    )?);
                    audio_offset += length;
                }
                PreparedPart::Projected {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
            }
        }
        let ordered = parts
            .iter()
            .zip(&embeddings)
            .map(|(part, embeddings)| OrderedInputPart {
                token_ids: match part {
                    PreparedPart::Text { tokens, .. }
                    | PreparedPart::Image { tokens }
                    | PreparedPart::Audio { tokens }
                    | PreparedPart::Projected { tokens, .. } => tokens,
                },
                embeddings,
            })
            .collect::<Vec<_>>();
        let hidden = assemble_ordered_inputs(&ordered, self.args.text_config.hidden_size, context)?
            .embeddings;
        self.static_modules.embedding_norm.forward(&hidden, context)
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
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
            0 => Ok(self
                .args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.num_hidden_layers as usize)),
            1 => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => Err(Error::backend("Inkling has two execution groups")),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        let count = match group {
            0 => self
                .args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.num_hidden_layers as usize),
            1 => self.args.text_config.num_hidden_layers as usize,
            _ => return Err(Error::backend("Inkling has two execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Inkling unit is outside its group"));
        }
        match group {
            0 => Ok(format!("visual.layers.{index}")),
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
            0 => {
                let vision = self
                    .args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Inkling has no vision config"))?;
                Ok(Unit::Vision(VisionLayer::new(
                    vision,
                    index,
                    vision.layer_specs()[index],
                    context,
                )?))
            }
            1 => Ok(Unit::Text(DecoderLayer::new(
                &self.args.text_config,
                index,
                context,
            )?)),
            _ => Err(Error::backend("Inkling has two execution groups")),
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        if state.layout() != &state_layout(&self.args).map_err(Error::backend)? {
            return Err(Error::backend("Inkling runtime state layout mismatch"));
        }
        let parts = self.prepare_parts(input.parts, context)?;
        let tokens = ordered_tokens(&parts, context)?;
        let audio = match input.audio {
            Some(audio) => Some(
                self.static_modules
                    .audio
                    .as_mut()
                    .ok_or_else(|| Error::backend("Inkling has no audio tower"))?
                    .forward(audio, context)?,
            ),
            None => None,
        };
        let has_vision = input.vision_patches.is_some();
        let hidden = match input.vision_patches {
            Some(patches) => {
                if patches.shape().len() != 5 || patches.shape()[1..] != [2, 40, 40, 3] {
                    return Err(Error::backend("invalid Inkling hMLP patch geometry"));
                }
                patches.clone()
            }
            None => self.assemble(&parts, None, audio.as_ref(), context)?,
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts,
                tokens,
                audio,
                has_vision,
                target_hidden: None,
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
            (1, [assembled]) => Ok((*assembled).clone()),
            _ => Err(Error::backend("invalid Inkling execution dependencies")),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group != 0 || forward.has_vision
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        _forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match (group, unit) {
            (0, Unit::Vision(unit)) => unit.forward(hidden, context),
            (1, Unit::Text(unit)) => unit.forward(
                hidden,
                Some(state.layer(index).map_err(Error::backend)?),
                context,
            ),
            _ => Err(Error::backend("Inkling unit/group mismatch")),
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
        match group {
            0 if forward.has_vision => {
                let vision = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Inkling vision static modules are missing"))?
                    .finish(hidden, context)?;
                self.assemble(
                    &forward.parts,
                    Some(&vision),
                    forward.audio.as_ref(),
                    context,
                )
            }
            0 => Ok(hidden.clone()),
            1 => Ok(hidden.clone()),
            _ => Err(Error::backend("invalid Inkling execution group")),
        }
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let hidden = self.static_modules.final_norm.forward(hidden, context)?;
        let hidden = hidden.multiply_scalar(
            self.args.text_config.logits_mup_width_multiplier.recip(),
            context,
        )?;
        let logits = self.static_modules.output.forward(&hidden, context)?;
        let vocabulary = self
            .args
            .text_config
            .unpadded_vocab_size
            .unwrap_or(self.args.text_config.vocab_size);
        if vocabulary == self.args.text_config.vocab_size {
            return Ok(logits);
        }
        let mut indexes = vec![Index::Full; logits.shape().len()];
        *indexes.last_mut().expect("logits have vocabulary axis") = Index::Range(0, vocabulary);
        logits.index(&indexes, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.push(&forward.tokens);
        values.extend(forward.audio.iter());
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => values.extend([tokens, embeddings]),
                PreparedPart::Image { tokens } | PreparedPart::Audio { tokens } => {
                    values.push(tokens)
                }
                PreparedPart::Projected { tokens, embeddings } => {
                    values.extend([tokens, embeddings])
                }
            }
        }
        values.into_iter()
    }
}

impl<T> ForwardContext<T> {
    /// Returns complete ordered token identity after text/media placeholder assembly.
    pub const fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Retains the complete post-decoder activation for embedded prediction.
    pub fn capture_target_hidden(&mut self, hidden: T) {
        self.target_hidden = Some(hidden);
    }

    /// Returns the post-decoder activation retained by a TP target pass.
    pub const fn target_hidden(&self) -> Option<&T> {
        self.target_hidden.as_ref()
    }
}

fn ordered_tokens<T: Tensor>(parts: &[PreparedPart<T>], context: &T::Context) -> Result<T, Error> {
    let tokens = parts
        .iter()
        .map(|part| match part {
            PreparedPart::Text { tokens, .. }
            | PreparedPart::Image { tokens }
            | PreparedPart::Audio { tokens }
            | PreparedPart::Projected { tokens, .. } => tokens.clone(),
        })
        .collect::<Vec<_>>();
    T::concatenate(&tokens, 1, context)
}

fn part_token_count<T: Tensor>(
    parts: &[PreparedPart<T>],
    select: impl Fn(&PreparedPart<T>) -> bool,
) -> i32 {
    parts
        .iter()
        .filter(|part| select(part))
        .map(|part| match part {
            PreparedPart::Text { tokens, .. }
            | PreparedPart::Image { tokens }
            | PreparedPart::Audio { tokens }
            | PreparedPart::Projected { tokens, .. } => tokens.dim(1),
        })
        .sum()
}

fn validate_component<T: Tensor>(
    name: &str,
    value: Option<&T>,
    tokens: i32,
    hidden: i32,
) -> Result<(), Error> {
    match value {
        Some(value) if value.shape() == [1, tokens, hidden] => Ok(()),
        None if tokens == 0 => Ok(()),
        Some(value) => Err(Error::backend(format!(
            "Inkling {name} output has shape {:?}, expected [1, {tokens}, {hidden}]",
            value.shape()
        ))),
        None => Err(Error::backend(format!(
            "Inkling {name} placeholders require projected media"
        ))),
    }
}

fn slice_component<T: Tensor>(
    value: &T,
    offset: i32,
    length: i32,
    context: &T::Context,
) -> Result<T, Error> {
    value.index(
        &[
            Index::Full,
            Index::Range(offset, offset + length),
            Index::Full,
        ],
        context,
    )
}
