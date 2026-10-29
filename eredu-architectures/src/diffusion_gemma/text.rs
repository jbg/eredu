//! Shared text parameters with independent context and refinement invocations.

use eredu_nn::{
    AttentionCache, EmbeddingOperator, EmbeddingSpec, Error, GroupedNeuralBackend, LinearOperator,
    LinearSpec, NormalizationOperator, Parameter, ParameterSpec, Parameterized,
    ReadOnlyAttentionCache, ReadOnlyCache, Tensor, TensorElementType,
};
use eredu_runtime::{ExpertPass, ParameterProvider, VocabularyModulePlacement};

use super::FamilyConfig;
use crate::{
    decoder::{ComponentInstrumentation, PartitionStaticModules, StaticModuleConstruction},
    gemma4::{BlockInput, DenseBlock, ModelArgs},
};

/// Diffusion feedback equations over soft token embeddings.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SelfConditioning<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Learned normalization of the previous iteration's soft embedding.
    pub pre_norm: B::Normalization,
    /// GELU-tanh gate projection.
    pub gate: B::Linear,
    /// Multiplicative up projection.
    pub up: B::Linear,
    /// Projection to the shared embedding width.
    pub down: B::Linear,
    #[parameter(skip)]
    epsilon: f32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Clone for SelfConditioning<B> {
    fn clone(&self) -> Self {
        Self {
            pre_norm: self.pre_norm.clone(),
            gate: self.gate.clone(),
            up: self.up.clone(),
            down: self.down.clone(),
            epsilon: self.epsilon,
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SelfConditioning<B> {
    /// Declares the released decoder-only parameters without allocating weights.
    pub fn new(args: &ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let width = args
            .layer_policy(0)
            .ok_or_else(|| Error::backend("empty diffusion stack"))?
            .intermediate_size
            .get() as i32;
        Self::new_with_width(args, width, context)
    }

    fn new_with_width(
        args: &ModelArgs,
        width: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if width <= 0 {
            return Err(Error::backend(
                "feedback construction requires nonempty local units",
            ));
        }
        let prefix = "model.decoder.self_conditioning";
        let linear = |field: &str, input, output| {
            let key = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&key).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &key,
                        args.linear_format_for(&key),
                    )?,
                },
                context,
            )
        };
        Ok(Self {
            pre_norm: B::normalization(
                args.normalization(
                    args.hidden_size,
                    ParameterSpec::trainable(format!("{prefix}.pre_norm.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )?,
            gate: linear("gate_proj", args.hidden_size, width)?,
            up: linear("up_proj", args.hidden_size, width)?,
            down: linear("down_proj", width, args.hidden_size)?,
            epsilon: args.rms_norm_eps,
        })
    }

    /// Adds transformed feedback before scale-free normalization. First-step
    /// feedback is a zero embedding, including the ordinary learned pre-norm.
    pub fn forward(
        &mut self,
        embeddings: &B::Tensor,
        feedback: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_inner(
            embeddings,
            feedback,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Computes rank-local gate/up units and sums the down-projection contribution.
    pub fn forward_parallel(
        &mut self,
        embeddings: &B::Tensor,
        feedback: Option<&B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_inner(
            embeddings,
            feedback,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn forward_inner(
        &mut self,
        embeddings: &B::Tensor,
        feedback: Option<&B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let zeros;
        let feedback = match feedback {
            Some(value) if value.shape() == embeddings.shape() => value,
            Some(_) => {
                return Err(Error::backend(
                    "self-conditioning embedding geometry differs from the canvas",
                ))
            }
            None => {
                zeros = embeddings.zeros_like(context)?;
                &zeros
            }
        };
        instrumentation.observe("embedding", embeddings)?;
        instrumentation.observe("feedback", feedback)?;
        let normalized = self.pre_norm.forward(feedback, context)?;
        let gate = B::gelu_approximate(self.gate.forward(&normalized, context)?, context)?;
        let up = self.up.forward(&normalized, context)?;
        let units = gate.multiply(&up, context)?;
        let signal = match parallel {
            Some(parallel) => B::row_parallel_linear(&mut self.down, &units, parallel, context)?,
            None => self.down.forward(&units, context)?,
        };
        let output =
            B::rms_norm_without_weight(&embeddings.add(&signal, context)?, self.epsilon, context)?;
        instrumentation.apply("conditioning", output)
    }
}

/// One materialization of shared matrices and two independently stored scales.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TextLayer<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Shared attention, dense branch, routed bank and normalization weights.
    pub block: DenseBlock<B>,
    /// Encoder invocation's independent released residual scale.
    pub encoder_scalar: Parameter<B::Tensor>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TextLayer<B> {
    /// Builds one independently materializable layer unit.
    pub fn new(
        args: &ModelArgs,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_experts(args, index, None, context)
    }

    pub(crate) fn new_with_experts(
        args: &ModelArgs,
        index: usize,
        experts: Option<eredu_nn::GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            block: DenseBlock::new_at_with_routed_spec(
                args,
                index,
                "model.decoder.layers",
                experts,
                context,
            )?,
            encoder_scalar: Parameter::unloaded(
                ParameterSpec::trainable(format!(
                    "model.encoder.language_model.layers.{index}.layer_scalar"
                ))
                .map_err(Error::backend)?,
                &[1],
                context,
            )?,
        })
    }

    /// Encodes authoritative positions into the owner's ordinary mutable cache.
    pub fn encode<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.block.forward_with_layer_scalar(
            input,
            self.encoder_scalar.as_ref(),
            pass,
            provider,
            context,
            instrumentation,
        )
    }

    /// Refines a canvas without publishing projected key/value state. The
    /// caller supplies the full retained-prefix and bidirectional canvas mask.
    pub fn refine<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: ReadOnlyAttentionCache<B::Tensor>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let owner = input
            .cache
            .ok_or_else(|| Error::backend("refinement requires an authoritative context cache"))?;
        let mut read_only = ReadOnlyCache::new(owner);
        self.block.forward_with_provider_observed(
            BlockInput {
                hidden: input.hidden,
                mask: input.mask,
                cache: Some(&mut read_only),
                shared: input.shared,
                per_layer_input: None,
                rotary_position: input.rotary_position,
            },
            pass,
            provider,
            context,
            instrumentation,
        )
    }

    /// Encodes with rank-local projections and the selected distributed provider.
    pub fn encode_parallel<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
        C: AttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.block.forward_parallel_with_layer_scalar(
            input,
            self.encoder_scalar.as_ref(),
            pass,
            provider,
            parallel,
            context,
            instrumentation,
        )
    }

    /// Executes the same read-only canvas invocation with collective projections.
    pub fn refine_parallel<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
        C: ReadOnlyAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let owner = input
            .cache
            .ok_or_else(|| Error::backend("refinement requires an authoritative context cache"))?;
        let mut read_only = ReadOnlyCache::new(owner);
        self.block.forward_parallel_with_provider_observed(
            BlockInput {
                hidden: input.hidden,
                mask: input.mask,
                cache: Some(&mut read_only),
                shared: input.shared,
                per_layer_input: None,
                rotary_position: input.rotary_position,
            },
            pass,
            provider,
            parallel,
            context,
            instrumentation,
        )
    }
}

/// Shared vocabulary and readout plus decoder-only feedback parameters.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TextStatic<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Shared encoder/decoder token table and tied output projection.
    pub embeddings: Option<B::Embedding>,
    /// Shared final learned normalization.
    pub norm: Option<B::Normalization>,
    /// Optional independent vocabulary projection.
    pub head: Option<B::Linear>,
    /// Decoder-only self-conditioning module.
    pub conditioning: Option<SelfConditioning<B>>,
    #[parameter(skip)]
    embedding_scale: f32,
    #[parameter(skip)]
    softcap: Option<f32>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Clone for TextStatic<B> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            norm: self.norm.clone(),
            head: self.head.clone(),
            conditioning: self.conditioning.clone(),
            embedding_scale: self.embedding_scale,
            softcap: self.softcap,
        }
    }
}

/// Backend-neutral pinned specifications shared by cold catalogs and materialization.
pub(super) fn static_construction_specs(
    config: &FamilyConfig,
) -> Result<StaticModuleConstruction, Error> {
    let args = &config.text;
    let key = "model.decoder.embed_tokens.weight";
    let embedding = EmbeddingSpec {
        vocabulary: args.vocab_size,
        dimensions: args.hidden_size,
        weight: ParameterSpec::trainable(key).map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(key, args.linear_format_for(key))?,
    };
    let head = (!args.tie_word_embeddings)
        .then(|| -> Result<_, Error> {
            Ok(LinearSpec {
                input: args.hidden_size,
                output: args.vocab_size,
                weight: ParameterSpec::trainable("lm_head.weight").map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    "lm_head.weight",
                    args.linear_format_for("lm_head.weight"),
                )?,
            })
        })
        .transpose()?;
    let normalization = args.normalization(
        args.hidden_size,
        ParameterSpec::trainable("model.decoder.norm.weight").map_err(Error::backend)?,
    );
    StaticModuleConstruction::new(embedding, head, normalization)
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TextStatic<B> {
    /// Declares pinned text parameters independently from layer residency.
    pub fn new(
        config: &FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::materialize(config, None, context)
    }

    /// Declares vocabulary-owned tables and feedback units from exact local geometry.
    pub fn new_parallel(
        config: &FamilyConfig,
        geometry: &super::parallel::LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        geometry.validate_for(config).map_err(Error::backend)?;
        Self::materialize(config, Some(geometry), context)
    }

    fn materialize(
        config: &FamilyConfig,
        geometry: Option<&super::parallel::LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let args = &config.text;
        let placement = |role| match geometry {
            None => VocabularyModulePlacement::Local,
            Some(geometry) => {
                let range = match role {
                    "embedding" => geometry.embedding_range(),
                    "output" => geometry.output_range(),
                    _ => unreachable!("declared vocabulary role"),
                };
                range.cloned().map_or(
                    VocabularyModulePlacement::Omitted,
                    VocabularyModulePlacement::VocabularyParallel,
                )
            }
        };
        let modules = PartitionStaticModules::<B>::from_construction(
            static_construction_specs(config)?,
            placement("embedding"),
            if args.tie_word_embeddings {
                VocabularyModulePlacement::Omitted
            } else {
                placement("output")
            },
            geometry.is_none_or(|geometry| geometry.owns_norm()),
            context,
        )?;
        let conditioning = match geometry {
            None => Some(SelfConditioning::new(args, context)?),
            Some(geometry) => geometry
                .conditioning_width()
                .map(|width| SelfConditioning::new_with_width(args, width, context))
                .transpose()?,
        };
        Ok(Self {
            embeddings: modules.embeddings,
            norm: modules.norm,
            head: modules.lm_head,
            conditioning,
            embedding_scale: (args.hidden_size as f32).sqrt(),
            softcap: args.final_logit_softcapping,
        })
    }

    /// Shared scaled lookup used when encoding authoritative token identities.
    pub fn encode_input(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embeddings = self
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("this stage does not own token ingress"))?
            .forward(tokens, context)?;
        self.scale_embedding(&embeddings, context)
    }

    /// Sums owned vocabulary-row lookups before applying embedding scaling.
    pub fn encode_input_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embeddings = B::vocabulary_parallel_lookup(
            self.embeddings
                .as_mut()
                .ok_or_else(|| Error::backend("this stage does not own token ingress"))?,
            tokens,
            eredu_nn::EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.scale_embedding(&embeddings, context)
    }

    fn scale_embedding(
        &self,
        embeddings: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let format = embeddings
            .element_type()
            .ok_or_else(|| Error::backend("embedding arithmetic requires a known floating type"))?;
        let scale = B::Tensor::from_f32_slice(&[self.embedding_scale], &[1], context)?
            .cast_float(format, context)?;
        embeddings.multiply(&scale, context)
    }

    /// Softmax uses FP32; weighted row access rounds probabilities to embedding
    /// arithmetic. Feedback scores retain their selected dtype before softmax.
    pub fn refine_input(
        &mut self,
        tokens: &B::Tensor,
        preceding_logits: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.refine_input_observed(
            tokens,
            preceding_logits,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes the actual embedding and self-conditioning invocation boundaries.
    pub fn refine_input_observed(
        &mut self,
        tokens: &B::Tensor,
        preceding_logits: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let embeddings = self.encode_input(tokens, context)?;
        let feedback = preceding_logits
            .map(|scores| {
                let probabilities = scores
                    .cast_float(TensorElementType::F32, context)?
                    .softmax_axis(-1, true, context)?;
                let embedding = self
                    .embeddings
                    .as_mut()
                    .ok_or_else(|| Error::backend("this stage does not own soft token ingress"))?
                    .weighted_sum(&probabilities, context)?;
                self.scale_embedding(&embedding, context)
            })
            .transpose()?;
        self.conditioning
            .as_mut()
            .ok_or_else(|| Error::backend("this stage does not own conditioning ingress"))?
            .forward_inner(
                &embeddings,
                feedback.as_ref(),
                None,
                context,
                instrumentation,
            )
    }

    /// Combines vocabulary-owned soft embeddings and local conditioning units.
    pub fn refine_input_parallel(
        &mut self,
        tokens: &B::Tensor,
        preceding_logits: Option<&B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embeddings = self.encode_input_parallel(tokens, parallel, context)?;
        let feedback = preceding_logits
            .map(|scores| {
                let probabilities = scores
                    .cast_float(TensorElementType::F32, context)?
                    .softmax_axis(-1, true, context)?;
                let embedding = B::vocabulary_parallel_weighted_sum(
                    self.embeddings.as_mut().ok_or_else(|| {
                        Error::backend("this stage does not own soft token ingress")
                    })?,
                    &probabilities,
                    parallel,
                    context,
                )?;
                self.scale_embedding(&embedding, context)
            })
            .transpose()?;
        self.conditioning
            .as_mut()
            .ok_or_else(|| Error::backend("this stage does not own conditioning ingress"))?
            .forward_parallel(&embeddings, feedback.as_ref(), parallel, context)
    }

    /// Applies the learned shared final norm on its owning stage.
    pub fn normalize(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.norm
            .as_mut()
            .ok_or_else(|| Error::backend("this stage does not own final normalization"))?
            .forward(hidden, context)
    }

    /// Projects every canvas position; final readout does not collapse the
    /// sequence axis to the final causal token.
    pub fn readout(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.readout_observed(
            hidden,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Gathers vocabulary-owned scores for every canvas position.
    pub fn readout_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.readout_observed(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn normalize_observed(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        instrumentation.normalize_readout(
            hidden,
            self.norm
                .as_mut()
                .ok_or_else(|| Error::backend("this stage does not own final normalization"))?,
            context,
        )
    }

    /// Emits the actual full-canvas normalization, projection-input and score seams.
    pub fn readout_observed(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.normalize_observed(hidden, context, instrumentation)?;
        let scores = match (parallel, self.head.as_mut()) {
            (Some(parallel), Some(head)) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                head,
                &hidden,
                parallel,
                context,
            )?,
            (None, Some(head)) => {
                instrumentation.project::<B>("projection_input", head, &hidden, None, context)?
            }
            (parallel, None) => {
                let embeddings = self
                    .embeddings
                    .as_mut()
                    .ok_or_else(|| Error::backend("this stage does not own tied readout"))?;
                match parallel {
                    Some(parallel) => instrumentation.project_vocabulary_embedding::<B>(
                        "projection_input",
                        embeddings,
                        &hidden,
                        parallel,
                        context,
                    )?,
                    None => instrumentation.project_embedding(
                        "projection_input",
                        embeddings,
                        &hidden,
                        context,
                    )?,
                }
            }
        };
        let scores = instrumentation.apply("linear", scores)?;
        let scores = self.cap_scores(scores, context)?;
        instrumentation.apply("scores", scores)
    }

    fn cap_scores(
        &self,
        scores: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let scores = scores.cast_float(TensorElementType::F32, context)?;
        match self.softcap {
            Some(cap) => scores
                .divide(&B::Tensor::from_f32_slice(&[cap], &[1], context)?, context)?
                .tanh(context)?
                .multiply_scalar(cap, context),
            None => Ok(scores),
        }
    }
}
