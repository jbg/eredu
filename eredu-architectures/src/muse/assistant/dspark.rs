//! DSpark's causal proposal decoder with independently encoded committed context.

use eredu_nn::{
    AttentionCache, AttentionRequest, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    Error, GroupedNeuralBackend, LinearOperator, LinearSpec, NeuralBackend, Parameter,
    ParameterSpec, Parameterized, Tensor, TensorElementType,
};

use super::super::DecoderConfig;
use super::{DFlashContext, DFlashLayerContext};

mod parallel;

/// Causal dense assistant body with independently acquired context/block/norm units.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DsparkDecoder<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    units: eredu_runtime::ResidentUnitWindow<super::body::DsparkBodyUnit<B>>,
    #[parameter(skip)]
    definition: super::body::DsparkBodyDefinition,
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DsparkDecoder<B> {
    /// Constructs a causal body from normalized geometry, without checkpoint admission.
    pub fn from_decoder_config(
        args: DecoderConfig,
        captures: usize,
        block_size: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = super::body::DsparkBodyDefinition::new(args, captures, block_size)?;
        let units = super::body::AssistantBodyDefinition::<B>::build_units(&definition, context)?;
        Ok(Self {
            units: eredu_runtime::ResidentUnitWindow::new(units),
            definition,
        })
    }
    /// Separates validated geometry from resident units for ordinary parameter policies.
    pub fn into_body_parts(
        self,
    ) -> (
        super::body::DsparkBodyDefinition,
        eredu_runtime::ResidentUnitWindow<super::body::DsparkBodyUnit<B>>,
    ) {
        (self.definition, self.units)
    }
    /// Concatenates ordered target publications into the trained context encoder.
    pub fn assemble_target_states(
        &self,
        states: &[B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        super::body::AssistantBodyDefinition::<B>::geometry(&self.definition)
            .assemble_target_states(states, context)
    }
    /// Appends only committed target taps through each layer's trained input norm.
    pub fn update_context(
        &mut self,
        previous: Option<DFlashContext<B::Tensor>>,
        pending: &B::Tensor,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DFlashContext<B::Tensor>, Error> {
        super::body::update_context::<B, _, _>(
            &self.definition,
            &mut self.units,
            previous,
            pending,
            end,
            context,
        )
    }
    /// Evaluates every sampled slot without mutating committed context.
    pub fn proposal_states(
        &mut self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.proposal_states_observed(
            embeddings,
            committed,
            end,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
    /// Runs the same unit traversal with admitted observations and interventions.
    pub fn proposal_states_observed(
        &mut self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        super::body::proposal_states::<B, _, _>(
            &self.definition,
            &mut self.units,
            embeddings,
            committed,
            end,
            context,
            instrumentation,
        )
    }
}

pub(super) struct ProposalContext<B: GroupedNeuralBackend> {
    pub(super) prior: Option<DFlashLayerContext<B::Tensor>>,
    pub(super) end: i32,
    pub(super) window: i32,
}

/// Full-vocabulary DSpark output and transition geometry.
#[derive(Debug, Clone, Copy)]
pub struct DsparkHeadGeometry {
    /// Decoder width.
    pub hidden: i32,
    /// Shared draft/target vocabulary cardinality.
    pub vocabulary: i32,
    /// Markov table rank.
    pub rank: i32,
    /// Multiplier applied to conditional logits in FP32.
    pub output_multiplier: f32,
    /// Positive tanh cap.
    pub soft_cap: f32,
}

/// One conditional distribution in the target vocabulary, with trained acceptance confidence.
#[derive(Debug, Clone)]
pub struct DsparkConditional<T> {
    /// Full target-vocabulary scores; unmapped columns remain negative infinity.
    pub logits: T,
    /// Acceptance estimate conditioned on the same actual predecessor token.
    pub confidence: T,
}

/// Validated DSpark head geometry and independently constructed parameter units.
#[derive(Debug, Clone)]
pub struct DsparkHeadDefinition {
    geometry: DsparkHeadGeometry,
    formats: std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    layout: eredu_runtime::ExecutionUnitLayout,
}

/// One vocabulary, transition, confidence or exact mapping parameter unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DsparkHeadUnit<B: NeuralBackend> {
    module: DsparkHeadModule<B>,
}
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum DsparkHeadModule<B: NeuralBackend> {
    Projection(B::Linear),
    Embedding(B::Embedding),
    Confidence(B::Linear),
    Mapping(Parameter<B::Tensor>),
}

impl DsparkHeadDefinition {
    /// Validates geometry without creating native parameters or allocating a tensor.
    pub fn new(
        geometry: DsparkHeadGeometry,
        formats: std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self, Error> {
        let g = geometry;
        if g.hidden <= 0
            || g.vocabulary <= 0
            || g.rank <= 0
            || g.hidden.checked_add(g.rank).is_none()
            || !g.output_multiplier.is_finite()
            || g.output_multiplier <= 0.0
            || !g.soft_cap.is_finite()
            || g.soft_cap <= 0.0
        {
            return Err(Error::backend("invalid DSpark head geometry"));
        }
        let graph = eredu_runtime::ExecutionGraph::chain([
            "assistant.vocabulary",
            "assistant.predecessor_table",
            "assistant.successor_table",
            "assistant.confidence",
            "assistant.token_mapping",
        ])
        .map_err(Error::backend)?;
        let layout =
            eredu_runtime::ExecutionUnitLayout::new(&graph, [1; 5]).map_err(Error::backend)?;
        Ok(Self {
            geometry,
            formats,
            layout,
        })
    }
    /// Cold head geometry without parameter construction.
    pub const fn geometry(&self) -> DsparkHeadGeometry {
        self.geometry
    }
    /// Canonical per-head acquisition addresses.
    pub fn unit_layout(&self) -> &eredu_runtime::ExecutionUnitLayout {
        &self.layout
    }
    /// Shared draft vocabulary cardinality.
    pub fn vocabulary_size(&self) -> i32 {
        self.geometry.vocabulary
    }
    /// Constructs exactly one unloaded parameter unit.
    pub fn build_unit<B: NeuralBackend>(
        &self,
        ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DsparkHeadUnit<B>, Error> {
        let g = self.geometry;
        let formats = &self.formats;
        let format = |name: &str| {
            crate::linear_format::standard_linear_format(
                name,
                formats
                    .get(name)
                    .copied()
                    .map(Into::into)
                    .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
            )
        };
        let projection = |name: &str, input, output, bias, compute| {
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                    bias,
                    format: format(name)?.with_compute_element(compute)?,
                },
                context,
            )
        };
        let module = match ordinal {
            0 => DsparkHeadModule::Projection(projection(
                "lm_head.weight",
                g.hidden,
                g.vocabulary,
                None,
                None,
            )?),
            1 => DsparkHeadModule::Embedding(B::embedding(
                EmbeddingSpec {
                    vocabulary: g.vocabulary,
                    dimensions: g.rank,
                    weight: ParameterSpec::trainable("markov_head.markov_w1.weight")
                        .map_err(Error::backend)?,
                    format: format("markov_head.markov_w1.weight")?,
                },
                context,
            )?),
            2 => DsparkHeadModule::Projection(projection(
                "markov_head.markov_w2.weight",
                g.rank,
                g.vocabulary,
                None,
                None,
            )?),
            3 => DsparkHeadModule::Confidence(projection(
                "confidence_head.proj.weight",
                g.hidden + g.rank,
                1,
                Some(
                    ParameterSpec::trainable("confidence_head.proj.bias")
                        .map_err(Error::backend)?,
                ),
                Some(TensorElementType::F32),
            )?),
            4 => {
                B::require_operator_capabilities(
                    "DSpark checkpoint mapping",
                    eredu_nn::NeuralOperatorCapabilities::UNLOADED_I64,
                )?;
                let mut mapping = ParameterSpec::trainable("d2t").map_err(Error::backend)?;
                mapping.trainable = false;
                DsparkHeadModule::Mapping(Parameter::unloaded_i64(
                    mapping,
                    &[g.vocabulary],
                    context,
                )?)
            }
            _ => return Err(Error::backend("invalid DSpark head unit ordinal")),
        };
        Ok(DsparkHeadUnit { module })
    }
    /// Constructs resident heads through the same individual unit constructors.
    pub fn build_units<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<DsparkHeadUnit<B>>, Error> {
        (0..self.layout.len())
            .map(|i| self.build_unit::<B>(i, context))
            .collect()
    }
    /// Raw draft-vocabulary logits before transition bias and output transforms.
    pub fn logits<B: NeuralBackend, P: eredu_runtime::LayerwisePolicy<B, DsparkHeadUnit<B>>>(
        &self,
        policy: &mut P,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        super::heads::execute::<B, _, _, _>(
            policy,
            &self.layout,
            0,
            hidden,
            &[],
            |ctx| self.build_unit::<B>(0, ctx),
            |unit| match &mut unit.module {
                DsparkHeadModule::Projection(projection) => {
                    super::body::parameter_projection::<B>(projection, hidden, context)
                }
                _ => Err(Error::backend("incorrect DSpark vocabulary unit")),
            },
            context,
        )
    }

    /// Replicated transition embedding for already selected global target IDs.
    pub fn markov_embed<B: NeuralBackend, P: eredu_runtime::LayerwisePolicy<B, DsparkHeadUnit<B>>>(
        &self,
        policy: &mut P,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        super::heads::execute::<B, _, _, _>(
            policy,
            &self.layout,
            1,
            tokens,
            &[],
            |ctx| self.build_unit::<B>(1, ctx),
            |unit| match &mut unit.module {
                DsparkHeadModule::Embedding(embedding) => {
                    embedding.lookup(tokens, EmbeddingLookupPolicy::Strict, context)
                }
                _ => Err(Error::backend("incorrect DSpark Markov embedding unit")),
            },
            context,
        )
    }

    /// Raw full-vocabulary transition bias, before output transforms.
    pub fn markov_bias<B: NeuralBackend, P: eredu_runtime::LayerwisePolicy<B, DsparkHeadUnit<B>>>(
        &self,
        policy: &mut P,
        embedding: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        super::heads::execute::<B, _, _, _>(
            policy,
            &self.layout,
            2,
            embedding,
            &[],
            |ctx| self.build_unit::<B>(2, ctx),
            |unit| match &mut unit.module {
                DsparkHeadModule::Projection(projection) => {
                    super::body::parameter_projection::<B>(projection, embedding, context)
                }
                _ => Err(Error::backend("incorrect DSpark Markov output unit")),
            },
            context,
        )
    }

    /// Applies the architecture's FP32 multiplier and tanh cap to conditional logits.
    pub fn transform_logits<B: NeuralBackend>(
        &self,
        logits: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        logits
            .cast_float(TensorElementType::F32, context)?
            .multiply_scalar(self.geometry.output_multiplier, context)?
            .divide(
                &B::Tensor::full_f32(self.geometry.soft_cap, &[], context)?,
                context,
            )?
            .tanh(context)?
            .multiply_scalar(self.geometry.soft_cap, context)
    }

    /// Conditions one row on actual target IDs and maps its complete distribution
    /// before the shared sampler or likelihood-ratio verifier consumes it.
    pub fn conditional_target_scores<
        B: NeuralBackend,
        P: eredu_runtime::LayerwisePolicy<B, DsparkHeadUnit<B>>,
    >(
        &self,
        policy: &mut P,
        hidden: &B::Tensor,
        raw_logits: &B::Tensor,
        previous_target_ids: &B::Tensor,
        target_columns: &B::Tensor,
        target_vocabulary: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DsparkConditional<B::Tensor>, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        let batch = hidden.shape().first().copied().unwrap_or(0);
        if hidden.shape() != [batch, self.geometry.hidden]
            || batch <= 0
            || raw_logits.shape() != [batch, self.geometry.vocabulary]
            || previous_target_ids.shape() != [batch]
            || target_columns.shape() != [self.geometry.vocabulary]
            || target_columns.element_type() != Some(TensorElementType::I64)
            || target_vocabulary <= 0
        {
            return Err(Error::backend("invalid conditional DSpark score geometry"));
        }
        B::require_operator_capabilities(
            "conditional DSpark vocabulary mapping",
            eredu_nn::NeuralOperatorCapabilities::SCATTER_LAST_AXIS
                .union(eredu_nn::NeuralOperatorCapabilities::FULL_F32),
        )?;
        let markov = self.markov_embed::<B, P>(policy, previous_target_ids, context)?;
        let bias = self.markov_bias::<B, P>(policy, &markov, context)?;
        let scores = self.transform_logits::<B>(&raw_logits.add(&bias, context)?, context)?;
        let columns = target_columns
            .reshape(&[1, self.geometry.vocabulary], context)?
            .broadcast_to(&[batch, self.geometry.vocabulary], context)?;
        let logits = B::Tensor::full_f32(f32::NEG_INFINITY, &[batch, target_vocabulary], context)?
            .scatter_last_axis(&columns, &scores, context)?;
        Ok(DsparkConditional {
            logits,
            confidence: self.confidence::<B, P>(policy, hidden, &markov, context)?,
        })
    }

    /// Acceptance confidence uses concatenated hidden/Markov features in FP32,
    /// including FP32 projection and bias arithmetic.
    pub fn confidence<B: NeuralBackend, P: eredu_runtime::LayerwisePolicy<B, DsparkHeadUnit<B>>>(
        &self,
        policy: &mut P,
        hidden: &B::Tensor,
        markov: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        if hidden.shape().is_empty()
            || hidden.shape().last() != Some(&self.geometry.hidden)
            || markov.shape().last() != Some(&self.geometry.rank)
            || hidden.shape()[..hidden.shape().len() - 1]
                != markov.shape()[..markov.shape().len().saturating_sub(1)]
        {
            return Err(Error::backend("invalid DSpark confidence features"));
        }
        let input = B::Tensor::concatenate(&[hidden.clone(), markov.clone()], -1, context)?
            .cast_float(TensorElementType::F32, context)?;
        super::heads::execute::<B, _, _, _>(
            policy,
            &self.layout,
            3,
            &input,
            &[hidden, markov],
            |ctx| self.build_unit::<B>(3, ctx),
            |unit| match &mut unit.module {
                DsparkHeadModule::Confidence(projection) => B::sigmoid(
                    projection
                        .forward(&input, context)?
                        .squeeze_axes(&[-1], context)?,
                    context,
                ),
                _ => Err(Error::backend("incorrect DSpark confidence unit")),
            },
            context,
        )
    }

    /// Maps canonical sampled draft IDs by adding the checkpoint's exact signed
    /// offsets. No floating representation or I32 narrowing is introduced.
    pub fn map_draft_to_target<
        B: NeuralBackend,
        P: eredu_runtime::LayerwisePolicy<B, DsparkHeadUnit<B>>,
    >(
        &self,
        policy: &mut P,
        ids: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        if !matches!(
            ids.element_type(),
            Some(TensorElementType::I32 | TensorElementType::I64)
        ) {
            return Err(Error::backend(
                "DSpark mapping requires integral IDs and I64 offsets",
            ));
        }
        super::heads::execute::<B, _, _, _>(
            policy,
            &self.layout,
            4,
            ids,
            &[],
            |ctx| self.build_unit::<B>(4, ctx),
            |unit| match &unit.module {
                DsparkHeadModule::Mapping(mapping)
                    if mapping.as_ref().element_type() == Some(TensorElementType::I64) =>
                {
                    mapping
                        .as_ref()
                        .take_axis(ids, 0, context)?
                        .add(ids, context)
                }
                _ => Err(Error::backend("DSpark mapping requires I64 offsets")),
            },
            context,
        )
    }
}

/// Resident DSpark heads using the ordinary completion-safe parameter policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DsparkHeads<B: NeuralBackend> {
    units: eredu_runtime::ResidentUnitWindow<DsparkHeadUnit<B>>,
    #[parameter(skip)]
    definition: DsparkHeadDefinition,
}
impl<B: NeuralBackend> DsparkHeads<B> {
    /// Draft vocabulary cardinality.
    pub fn vocabulary_size(&self) -> i32 {
        self.definition.vocabulary_size()
    }
    /// Constructs dense resident heads.
    pub fn new(
        geometry: DsparkHeadGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_formats(geometry, &std::collections::HashMap::new(), context)
    }
    /// Constructs resident units with the selected physical encodings.
    pub fn new_with_formats(
        geometry: DsparkHeadGeometry,
        formats: &std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = DsparkHeadDefinition::new(geometry, formats.clone())?;
        let units = eredu_runtime::ResidentUnitWindow::new(definition.build_units::<B>(context)?);
        Ok(Self { units, definition })
    }
    /// Separates geometry from units for a selected ordinary parameter policy.
    pub fn into_head_parts(
        self,
    ) -> (
        DsparkHeadDefinition,
        eredu_runtime::ResidentUnitWindow<DsparkHeadUnit<B>>,
    ) {
        (self.definition, self.units)
    }
    /// Uses the shared per-unit head equation.
    pub fn logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition
            .logits::<B, _>(&mut self.units, hidden, context)
    }
    /// Uses the shared per-unit head equation.
    pub fn markov_embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition
            .markov_embed::<B, _>(&mut self.units, tokens, context)
    }
    /// Uses the shared per-unit head equation.
    pub fn markov_bias(
        &mut self,
        embedding: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition
            .markov_bias::<B, _>(&mut self.units, embedding, context)
    }
    /// Uses the shared per-unit head equation.
    pub fn conditional_target_scores(
        &mut self,
        hidden: &B::Tensor,
        raw_logits: &B::Tensor,
        previous_target_ids: &B::Tensor,
        target_columns: &B::Tensor,
        target_vocabulary: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DsparkConditional<B::Tensor>, Error> {
        self.definition.conditional_target_scores::<B, _>(
            &mut self.units,
            hidden,
            raw_logits,
            previous_target_ids,
            target_columns,
            target_vocabulary,
            context,
        )
    }
    /// Uses the shared per-unit head equation.
    pub fn confidence(
        &mut self,
        hidden: &B::Tensor,
        markov: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition
            .confidence::<B, _>(&mut self.units, hidden, markov, context)
    }
    /// Uses the shared per-unit head equation.
    pub fn map_draft_to_target(
        &mut self,
        ids: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition
            .map_draft_to_target::<B, _>(&mut self.units, ids, context)
    }
    /// Uses the shared per-unit head equation.
    pub fn transform_logits(
        &self,
        logits: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition.transform_logits::<B>(logits, context)
    }
}

impl<B: GroupedNeuralBackend> AttentionCache<B::Tensor> for ProposalContext<B> {
    fn offset(&self) -> i32 {
        self.end
    }
    fn max_size(&self) -> Option<i32> {
        Some(self.window)
    }
    fn update_for_attention(
        &mut self,
        keys: B::Tensor,
        values: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        let prior = self
            .prior
            .take()
            .ok_or_else(|| Error::backend("DSpark proposal context is single-use"))?;
        Ok((
            B::Tensor::concatenate(&[prior.keys, keys], 2, context)?,
            B::Tensor::concatenate(&[prior.values, values], 2, context)?,
        ))
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::attention_with_sinks(request, context)
    }
}
