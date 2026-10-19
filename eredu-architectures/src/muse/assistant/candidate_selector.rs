//! Conditional transition lattice for DFlash2 proposal candidates.

use eredu_nn::{
    EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator,
    LinearSpec, NeuralBackend, ParameterSpec, Parameterized, Tensor, TensorElementType,
};

/// Replicated candidate-selector geometry, independent of target head sharding.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CandidateSelectorGeometry {
    /// Assistant hidden width.
    pub hidden: i32,
    /// Global target vocabulary size.
    pub vocabulary: i32,
    /// Transition table and hidden-projection rank.
    pub rank: i32,
    /// Candidate count per proposal position.
    pub top_k: i32,
}

/// Validated selector geometry and individually constructed table/projection units.
#[derive(Debug, Clone)]
pub struct CandidateSelectorDefinition {
    geometry: CandidateSelectorGeometry,
    formats: std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    layout: eredu_runtime::ExecutionUnitLayout,
}
/// Exactly one selector table or projection, independent of body and vocabulary weights.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CandidateSelectorUnit<B: NeuralBackend> {
    module: CandidateSelectorModule<B>,
}
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum CandidateSelectorModule<B: NeuralBackend> {
    Table(B::Embedding),
    Projection(B::Linear),
}
impl CandidateSelectorDefinition {
    /// Validates selected geometry without allocating a native tensor.
    pub fn new(
        geometry: CandidateSelectorGeometry,
        formats: std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self, Error> {
        let g = geometry;
        if g.hidden <= 0
            || g.vocabulary <= 0
            || g.rank <= 0
            || g.top_k <= 0
            || g.top_k > g.vocabulary
        {
            return Err(Error::backend(
                "invalid assistant candidate-selector geometry",
            ));
        }
        let graph = eredu_runtime::ExecutionGraph::chain([
            "assistant.predecessor_table",
            "assistant.successor_table",
            "assistant.conditional_projection",
        ])
        .map_err(Error::backend)?;
        let layout =
            eredu_runtime::ExecutionUnitLayout::new(&graph, [1; 3]).map_err(Error::backend)?;
        Ok(Self {
            geometry,
            formats,
            layout,
        })
    }
    /// Cold selector geometry without parameter construction.
    pub const fn geometry(&self) -> CandidateSelectorGeometry {
        self.geometry
    }
    /// Canonical selector parameter addresses.
    pub fn unit_layout(&self) -> &eredu_runtime::ExecutionUnitLayout {
        &self.layout
    }
    /// Constructs exactly one unloaded selector parameter unit.
    pub fn build_unit<B: NeuralBackend>(
        &self,
        ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<CandidateSelectorUnit<B>, Error> {
        let g = self.geometry;
        let formats = &self.formats;
        let format = |name: &str| {
            crate::linear_format::matrix_linear_format(
                name,
                name.strip_suffix(".weight").unwrap_or(name),
                formats
                    .get(name)
                    .copied()
                    .map(Into::into)
                    .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
            )
        };
        let table = |name: &str| {
            B::embedding(
                EmbeddingSpec {
                    vocabulary: g.vocabulary,
                    dimensions: g.rank,
                    weight: ParameterSpec::trainable(format!("candidate_selector.{name}"))
                        .map_err(Error::backend)?,
                    format: format(&format!("candidate_selector.{name}"))?,
                },
                context,
            )
        };
        let module = match ordinal {
            0 => CandidateSelectorModule::Table(table("predecessor_codebook")?),
            1 => CandidateSelectorModule::Table(table("successor_codebook")?),
            2 => CandidateSelectorModule::Projection(B::linear(
                LinearSpec {
                    input: g.hidden,
                    output: g.rank,
                    weight: ParameterSpec::trainable("candidate_selector.hidden_projection.weight")
                        .map_err(Error::backend)?,
                    bias: None,
                    format: format("candidate_selector.hidden_projection.weight")?,
                },
                context,
            )?),
            _ => return Err(Error::backend("invalid candidate-selector unit ordinal")),
        };
        Ok(CandidateSelectorUnit { module })
    }
    /// Constructs resident units through the individual constructors.
    pub fn build_units<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<CandidateSelectorUnit<B>>, Error> {
        (0..self.layout.len())
            .map(|i| self.build_unit::<B>(i, context))
            .collect()
    }
    fn lookup<B: NeuralBackend, P: eredu_runtime::LayerwisePolicy<B, CandidateSelectorUnit<B>>>(
        &self,
        policy: &mut P,
        ordinal: usize,
        ids: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        super::heads::execute::<B, _, _, _>(
            policy,
            &self.layout,
            ordinal,
            ids,
            &[],
            |ctx| self.build_unit::<B>(ordinal, ctx),
            |unit| match &mut unit.module {
                CandidateSelectorModule::Table(table) => {
                    table.lookup(ids, EmbeddingLookupPolicy::Strict, context)
                }
                _ => Err(Error::backend("incorrect candidate-selector table unit")),
            },
            context,
        )
    }
    fn project<B: NeuralBackend, P: eredu_runtime::LayerwisePolicy<B, CandidateSelectorUnit<B>>>(
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
            2,
            hidden,
            &[],
            |ctx| self.build_unit::<B>(2, ctx),
            |unit| match &mut unit.module {
                CandidateSelectorModule::Projection(projection) => {
                    projection.forward(hidden, context)
                }
                _ => Err(Error::backend(
                    "incorrect candidate-selector projection unit",
                )),
            },
            context,
        )
    }
    /// Returns `[batch, slot, predecessor, candidate]` transition scores. IDs are
    /// I32 global vocabulary rows; unary scores already include the architecture's
    /// output multiplier and soft cap. The first predecessor axis repeats the
    /// anchor so the same conditional walk consumes every slot.
    pub fn build_lattice<
        B: NeuralBackend,
        P: eredu_runtime::LayerwisePolicy<B, CandidateSelectorUnit<B>>,
    >(
        &self,
        policy: &mut P,
        candidate_ids: &B::Tensor,
        unary_logits: &B::Tensor,
        hidden_states: &B::Tensor,
        anchor_ids: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        let g = self.geometry;
        if candidate_ids.shape().len() != 3
            || candidate_ids.shape().iter().any(|&n| n <= 0)
            || candidate_ids.dim(2) != g.top_k
            || candidate_ids.element_type() != Some(TensorElementType::I32)
            || unary_logits.shape() != candidate_ids.shape()
            || hidden_states.shape() != [candidate_ids.dim(0), candidate_ids.dim(1), g.hidden]
            || anchor_ids.shape() != [candidate_ids.dim(0)]
            || anchor_ids.element_type() != Some(TensorElementType::I32)
        {
            return Err(Error::backend("invalid DFlash candidate lattice inputs"));
        }
        let batch = candidate_ids.dim(0);
        let slots = candidate_ids.dim(1);
        let anchors = anchor_ids
            .reshape(&[batch, 1, 1], context)?
            .broadcast_to(&[batch, 1, g.top_k], context)?;
        let predecessor_ids = if slots == 1 {
            anchors
        } else {
            B::Tensor::concatenate(
                &[
                    anchors,
                    candidate_ids.index(
                        &[Index::Full, Index::Range(0, slots - 1), Index::Full],
                        context,
                    )?,
                ],
                1,
                context,
            )?
        };
        let predecessors = self.lookup::<B, P>(policy, 0, &predecessor_ids, context)?;
        let successors = self.lookup::<B, P>(policy, 1, candidate_ids, context)?;
        let hidden = self.project::<B, P>(policy, hidden_states, context)?;
        let conditioned = predecessors.multiply(&hidden.expand_dims(-2, context)?, context)?;
        B::Tensor::matmul(
            &conditioned,
            &successors.transpose_axes(&[0, 1, 3, 2], context)?,
            context,
        )?
        .add(&unary_logits.expand_dims(-2, context)?, context)
    }

    /// Selects global candidates from the complete raw vocabulary projection before
    /// applying FP32 output scaling and softcapping. The target owns any TP gather.
    pub fn prepare_candidates<B: NeuralBackend>(
        &self,
        raw_logits: &B::Tensor,
        output_multiplier: f32,
        soft_cap: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        if raw_logits.shape().last().copied() != Some(self.geometry.vocabulary)
            || !output_multiplier.is_finite()
            || output_multiplier <= 0.0
            || !soft_cap.is_finite()
            || soft_cap <= 0.0
        {
            return Err(Error::backend("invalid candidate vocabulary projection"));
        }
        B::require_operator_capabilities(
            "assistant candidate selection",
            eredu_nn::NeuralOperatorCapabilities::RANKED_TOPK
                .union(eredu_nn::NeuralOperatorCapabilities::FULL_F32),
        )?;
        let (scores, ids) = raw_logits.ranked_topk(self.geometry.top_k, context)?;
        let scores = scores
            .cast_float(TensorElementType::F32, context)?
            .multiply_scalar(output_multiplier, context)?
            .divide(&B::Tensor::full_f32(soft_cap, &[], context)?, context)?
            .tanh(context)?
            .multiply_scalar(soft_cap, context)?;
        Ok((ids, scores))
    }

    /// Computes the realized conditional row directly from the actual predecessor,
    /// without sampling or retaining a second candidate-walk state machine.
    pub fn conditional_target_scores<
        B: NeuralBackend,
        P: eredu_runtime::LayerwisePolicy<B, CandidateSelectorUnit<B>>,
    >(
        &self,
        policy: &mut P,
        candidate_ids: &B::Tensor,
        unary_logits: &B::Tensor,
        hidden_states: &B::Tensor,
        previous_target_ids: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        let g = self.geometry;
        let batch = candidate_ids.shape().first().copied().unwrap_or(0);
        if batch <= 0
            || candidate_ids.shape() != [batch, g.top_k]
            || candidate_ids.element_type() != Some(TensorElementType::I32)
            || unary_logits.shape() != candidate_ids.shape()
            || hidden_states.shape() != [batch, g.hidden]
            || previous_target_ids.shape() != [batch]
            || previous_target_ids.element_type() != Some(TensorElementType::I32)
        {
            return Err(Error::backend(
                "invalid conditional candidate-selector inputs",
            ));
        }
        B::require_operator_capabilities(
            "conditional candidate vocabulary mapping",
            eredu_nn::NeuralOperatorCapabilities::SCATTER_LAST_AXIS
                .union(eredu_nn::NeuralOperatorCapabilities::FULL_F32),
        )?;
        let predecessor = self.lookup::<B, P>(policy, 0, previous_target_ids, context)?;
        let successors = self.lookup::<B, P>(policy, 1, candidate_ids, context)?;
        let hidden = self.project::<B, P>(policy, hidden_states, context)?;
        let conditioned = predecessor
            .multiply(&hidden, context)?
            .expand_dims(-1, context)?;
        let scores = B::Tensor::matmul(&successors, &conditioned, context)?
            .squeeze_axes(&[-1], context)?
            .add(unary_logits, context)?
            .cast_float(TensorElementType::F32, context)?;
        B::Tensor::full_f32(f32::NEG_INFINITY, &[batch, g.vocabulary], context)?.scatter_last_axis(
            candidate_ids,
            &scores,
            context,
        )
    }
}

/// Resident selector using the same individual table/projection lease operations.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CandidateSelector<B: NeuralBackend> {
    units: eredu_runtime::ResidentUnitWindow<CandidateSelectorUnit<B>>,
    #[parameter(skip)]
    definition: CandidateSelectorDefinition,
}
impl<B: NeuralBackend> CandidateSelector<B> {
    /// Constructs dense resident selector units.
    pub fn new(
        geometry: CandidateSelectorGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_formats(geometry, &std::collections::HashMap::new(), context)
    }
    /// Constructs resident units using selected per-weight physical formats.
    pub fn new_with_formats(
        geometry: CandidateSelectorGeometry,
        formats: &std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = CandidateSelectorDefinition::new(geometry, formats.clone())?;
        let units = eredu_runtime::ResidentUnitWindow::new(definition.build_units::<B>(context)?);
        Ok(Self { units, definition })
    }
    /// Separates validated geometry from parameters for an ordinary unit policy.
    pub fn into_selector_parts(
        self,
    ) -> (
        CandidateSelectorDefinition,
        eredu_runtime::ResidentUnitWindow<CandidateSelectorUnit<B>>,
    ) {
        (self.definition, self.units)
    }
    /// Uses the shared selector equation.
    pub fn build_lattice(
        &mut self,
        candidate_ids: &B::Tensor,
        unary_logits: &B::Tensor,
        hidden_states: &B::Tensor,
        anchor_ids: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition.build_lattice::<B, _>(
            &mut self.units,
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_ids,
            context,
        )
    }
    /// Uses the shared selector equation.
    pub fn conditional_target_scores(
        &mut self,
        candidate_ids: &B::Tensor,
        unary_logits: &B::Tensor,
        hidden_states: &B::Tensor,
        previous_target_ids: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition.conditional_target_scores::<B, _>(
            &mut self.units,
            candidate_ids,
            unary_logits,
            hidden_states,
            previous_target_ids,
            context,
        )
    }
    /// Uses the shared selector equation.
    pub fn prepare_candidates(
        &self,
        raw_logits: &B::Tensor,
        output_multiplier: f32,
        soft_cap: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        self.definition
            .prepare_candidates::<B>(raw_logits, output_multiplier, soft_cap, context)
    }
}
