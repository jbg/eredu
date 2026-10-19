//! Typed Muse assistant construction; shared speculation owns every sampling decision.

use super::{
    body,
    execution::{
        self, DFlashBodyProjection, DsparkBodyProjection, DsparkHeadProjection, Equation,
        MuseAssistantDefinition, MuseAssistantUnit, SelectorProjection,
    },
};
use super::{
    spark::{SparkAssistantIdentity, SparkAssistantLoadConfig},
    DFlash, DFlashConfig, DFlashContext, ProposalLayout,
};
use crate::muse::{DecoderConfig, MuseIdentity};
use eredu_checkpoint::{schema::SafetensorsCheckpointPlan, LinearFormat, WeightQuantization};
use eredu_nn::{
    DistributedNeuralBackend, Error, GroupedNeuralBackend, Index, Parameterized, Tensor,
};
use eredu_runtime::{LayerwisePolicy, ResidentUnitWindow};
use std::collections::HashMap;

/// Released identity and physical encoding retained by the common Muse assistant path.
#[derive(Debug, Clone)]
pub enum MuseAssistantConfig {
    /// Glimmer's direct DFlash decoder.
    Glimmer(DFlashConfig),
    /// Spark's causal DSpark or convolved DFlash2 decoder.
    Spark(SparkAssistantLoadConfig),
}

impl MuseAssistantConfig {
    /// Source configuration identity; architecture markers distinguish DFlash2 from Qwen.
    pub fn model_type(&self) -> &str {
        match self {
            Self::Glimmer(c) => &c.model_type,
            Self::Spark(c) => match c.released().identity() {
                SparkAssistantIdentity::DSpark => "muse_spark_dspark",
                SparkAssistantIdentity::DFlash2 => "qwen3",
            },
        }
    }
    /// Target tokenizer identity, independent of target executable registration.
    pub const fn tokenizer_model_type(&self) -> &'static str {
        match self {
            Self::Glimmer(_) => "muse_glimmer",
            Self::Spark(_) => "muse_spark",
        }
    }
    /// Distinct architecture strategy identity retained in cache and capture contracts.
    pub fn strategy_name(&self) -> &'static str {
        match self {
            Self::Glimmer(_) => "muse-glimmer-dflash",
            Self::Spark(c) => match c.released().identity() {
                SparkAssistantIdentity::DSpark => "muse-spark-dspark",
                SparkAssistantIdentity::DFlash2 => "muse-spark-dflash2",
            },
        }
    }
    /// Number and meaning of sampled positions.
    pub fn proposal_layout(&self) -> ProposalLayout {
        match self {
            Self::Glimmer(_) => ProposalLayout::AnchorThenMasks,
            Self::Spark(c) => c.released().proposal_layout(),
        }
    }
    /// Maximum candidate count for the released block.
    pub fn proposal_capacity(&self) -> usize {
        match self {
            Self::Glimmer(c) => c.block_size.saturating_sub(1).min(15),
            Self::Spark(c) => c.released().proposal_capacity(),
        }
    }
    /// Total evaluated positions per complete proposal block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::Glimmer(c) => c.block_size,
            Self::Spark(c) => c.released().block_size(),
        }
    }
    /// Exact checkpoint-local mask ID in the target vocabulary.
    pub fn mask_token_id(&self) -> u32 {
        match self {
            Self::Glimmer(c) => c.mask_token_id,
            Self::Spark(c) => c.released().mask_token_id(),
        }
    }
    /// Committed context window.
    pub fn sliding_window(&self) -> i32 {
        match self {
            Self::Glimmer(c) => c.sliding_window,
            Self::Spark(c) => c.released().attention_window() as i32,
        }
    }
    /// Exact number of independently projected decoder context banks.
    pub fn num_hidden_layers(&self) -> usize {
        match self {
            Self::Glimmer(c) => c.num_hidden_layers as usize,
            Self::Spark(c) => c.released().num_hidden_layers(),
        }
    }
    /// Encoder/decoder hidden width.
    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Glimmer(c) => c.hidden_size,
            Self::Spark(c) => c.released().hidden_size() as i32,
        }
    }
    /// Exact projection format selected independently for each matrix.
    pub fn linear_format(&self, name: &str) -> LinearFormat {
        match self {
            Self::Glimmer(c) => c.linear_format_for(name),
            Self::Spark(c) => c.linear_format(name),
        }
    }
    /// Physical acquisition owner for an admitted primary parameter. Decoder
    /// blocks and auxiliary vocabulary tables remain separate bounded units.
    pub fn parameter_owner(
        &self,
        name: &str,
    ) -> Result<eredu_runtime::ReplicatedTextParameterOwner, String> {
        let unit = |group: &str, unit| {
            Ok(eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit {
                group: group.into(),
                unit,
            })
        };
        if let Some(rest) = name.strip_prefix("layers.") {
            let (index, suffix) = rest
                .split_once('.')
                .ok_or("invalid assistant block parameter")?;
            let index = index.parse::<usize>().map_err(|e| e.to_string())?;
            if index >= self.num_hidden_layers() || suffix.is_empty() {
                return Err(format!("assistant parameter {name:?} has no decoder block"));
            }
            return unit("assistant.decoder", index);
        }
        match (self, name) {
            (Self::Glimmer(_), "encoder.fc.weight" | "encoder.output_norm_enc.weight")
            | (Self::Spark(_), "fc.weight" | "hidden_norm.weight") => unit("assistant.context", 0),
            (_, "norm.weight") => unit("assistant.final_norm", 0),
            (Self::Spark(config), name) => match (config.released().identity(), name) {
                (SparkAssistantIdentity::DSpark, "lm_head.weight") => {
                    unit("assistant.vocabulary", 0)
                }
                (SparkAssistantIdentity::DSpark, "d2t") => unit("assistant.token_mapping", 0),
                (SparkAssistantIdentity::DSpark, "markov_head.markov_w1.weight")
                | (SparkAssistantIdentity::DFlash2, "candidate_selector.predecessor_codebook") => {
                    unit("assistant.predecessor_table", 0)
                }
                (SparkAssistantIdentity::DSpark, "markov_head.markov_w2.weight")
                | (SparkAssistantIdentity::DFlash2, "candidate_selector.successor_codebook") => {
                    unit("assistant.successor_table", 0)
                }
                (
                    SparkAssistantIdentity::DSpark,
                    "confidence_head.proj.weight" | "confidence_head.proj.bias",
                ) => unit("assistant.confidence", 0),
                (
                    SparkAssistantIdentity::DFlash2,
                    "candidate_selector.hidden_projection.weight",
                ) => unit("assistant.conditional_projection", 0),
                _ => Err(format!(
                    "assistant parameter {name:?} has no acquisition owner"
                )),
            },
            _ => Err(format!(
                "assistant parameter {name:?} has no acquisition owner"
            )),
        }
    }
    /// Complete materialization schema for the selected physical encodings.
    pub fn safetensors_plan(&self) -> Result<SafetensorsCheckpointPlan, String> {
        match self {
            Self::Glimmer(c) => super::dflash_safetensors_plan(c),
            Self::Spark(c) => c.safetensors_plan().map_err(|e| e.to_string()),
        }
    }
    /// Selects every supported matrix's encoding for an explicit load-time conversion.
    pub fn load_time_quantization(&self, format: WeightQuantization) -> Result<Self, String> {
        match self {
            Self::Glimmer(c) => c
                .load_time_quantization(format)
                .map(Self::Glimmer)
                .map_err(|e| e.to_string()),
            Self::Spark(c) => c
                .load_time_quantization(format)
                .map(Self::Spark)
                .map_err(|e| e.to_string()),
        }
    }
    /// Retains exact per-weight checkpoint formats.
    pub fn with_checkpoint_formats(
        &self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, String> {
        match self {
            Self::Glimmer(c) => c
                .with_checkpoint_formats(formats)
                .map(Self::Glimmer)
                .map_err(|e| e.to_string()),
            Self::Spark(c) => c
                .clone()
                .with_checkpoint_formats(formats)
                .map(Self::Spark)
                .map_err(|e| e.to_string()),
        }
    }
    /// Proves target identity and ordered captures before source preparation.
    pub fn capture_request(
        &self,
        target: &DecoderConfig,
    ) -> Result<crate::composite_execution::ExternalPredictionCaptureRequest, String> {
        match self {
            Self::Glimmer(c) if target.identity == MuseIdentity::Glimmer => {
                crate::muse::model::external_assistant_capture_request(target, c)
            }
            Self::Glimmer(_) => Err("Glimmer assistant requires a Glimmer target".into()),
            Self::Spark(c) => c
                .released()
                .capture_request(target)
                .map_err(|e| e.to_string()),
        }
    }
    /// Binds complete neural geometry after compatibility proof, before payload access.
    pub fn construction(
        &self,
        target: Option<&DecoderConfig>,
    ) -> Result<MuseAssistantConstruction, String> {
        let definition = match self {
            Self::Glimmer(c) => {
                c.validate_released().map_err(|e| e.to_string())?;
                MuseAssistantDefinition::glimmer(
                    body::DFlashBodyDefinition::new(c.clone().into()).map_err(|e| e.to_string())?,
                )
            }
            Self::Spark(c) => c
                .construction(
                    target.ok_or("Spark assistant construction requires a proven Muse target")?,
                )
                .map_err(|e| e.to_string())?
                .execution_definition(),
        }
        .map_err(|e| e.to_string())?;
        Ok(MuseAssistantConstruction { definition })
    }

    /// Target-owned embedding equation selected by the trained assistant.
    pub fn embedding_operation<'a, T>(
        &self,
        ids: &'a T,
    ) -> crate::composite_execution::ExternalPredictionTargetOperation<'a, T> {
        use crate::composite_execution::ExternalPredictionTargetOperation as Op;
        match self {
            Self::Glimmer(_) => Op::TokenEmbeddings(ids),
            Self::Spark(_) => Op::RawTokenEmbeddings(ids),
        }
    }
}

/// Target-bound module geometry retained independently for source and selected encodings.
#[derive(Debug, Clone)]
pub struct MuseAssistantConstruction {
    definition: MuseAssistantDefinition,
}

/// Shared assistant equations over one resident or bounded parameter policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MuseAssistantModule<
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    P = ResidentUnitWindow<MuseAssistantUnit<B>>,
> {
    policy: P,
    #[parameter(skip)]
    definition: MuseAssistantDefinition,
    #[parameter(skip)]
    backend: std::marker::PhantomData<B>,
}

/// Immutable neural proposal data. The shared driver owns its cursor and sampled predecessor.
#[derive(Debug, Clone)]
pub enum MuseAssistantProposal<T> {
    /// Complete Glimmer target-projected score block.
    Scores(T),
    /// DSpark body states, raw vocabulary scores, and exact global target columns.
    DSpark {
        /// Final assistant body states, one row per sampled slot.
        hidden: T,
        /// Untransformed draft-vocabulary projection for every slot.
        raw: T,
        /// I64 global target column for every draft row.
        target_columns: T,
        /// Complete target vocabulary extent.
        vocabulary: i32,
    },
    /// DFlash2 body states, global candidates, and already-transformed unary scores.
    DFlash2 {
        /// Final assistant body states, one row per sampled slot.
        hidden: T,
        /// Global I32 candidate IDs, selected before capping.
        ids: T,
        /// FP32 scaled and capped base scores for those candidates.
        unary: T,
    },
}

/// One realized conditional distribution and the separately trained confidence estimate.
pub struct MuseAssistantRow<T> {
    /// Full target-vocabulary causal scores consumed by the shared sampler and verifier.
    pub logits: T,
    /// Optional FP32 confidence for diagnostics under fixed-capacity drafting.
    pub confidence: Option<T>,
}

impl MuseAssistantConstruction {
    pub(crate) fn with_materialization_precisions(
        mut self,
        tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
    ) -> Self {
        self.definition.compute_precisions =
            crate::linear_format::floating_compute_precisions(tasks);
        self
    }

    /// Constructs from retained geometry without reopening artifacts or inspecting a native target.
    pub fn module<B: GroupedNeuralBackend + DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MuseAssistantModule<B>, Error> {
        MuseAssistantModule::resident(self.definition.clone(), context)
    }
    /// Borrows the normalized construction selected before native resources exist.
    pub const fn execution_definition(&self) -> &MuseAssistantDefinition {
        &self.definition
    }
    pub(crate) fn partition_tensor_parallel(
        &self,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, eredu_runtime::ArchitectureParameterDescription), Error> {
        let (definition, parameters) = self.definition.partition_tensor_parallel(rank)?;
        Ok((Self { definition }, parameters))
    }
    pub(crate) fn source_for_selected(&self, selected: &Self) -> Result<Self, Error> {
        Ok(Self {
            definition: self.definition.source_for_selected(&selected.definition)?,
        })
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> MuseAssistantModule<B> {
    /// Constructs resident units through the ordinary per-unit definition.
    pub fn resident(
        definition: MuseAssistantDefinition,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let units = definition.build_units::<B>(context)?;
        Ok(Self::new(definition, ResidentUnitWindow::new(units)))
    }
    /// Joins existing independently bound causal body and heads without copying weights.
    pub fn from_dspark_parts(
        decoder: super::dspark::DsparkDecoder<B>,
        heads: super::dspark::DsparkHeads<B>,
    ) -> Result<Self, Error> {
        let (body, body_units) = decoder.into_body_parts();
        let (heads, head_units) = heads.into_head_parts();
        let definition = MuseAssistantDefinition::dspark(body, heads)?;
        let mut units =
            execution::wrap_dspark_body(body_units.into_units().map_err(Error::backend_source)?);
        units.extend(execution::wrap_dspark_heads(
            head_units.into_units().map_err(Error::backend_source)?,
        ));
        Ok(Self::new(definition, ResidentUnitWindow::new(units)))
    }
    /// Joins an existing bound convolved body and selector without copying weights.
    pub fn from_dflash2_parts(
        decoder: DFlash<B>,
        selector: super::candidate_selector::CandidateSelector<B>,
    ) -> Result<Self, Error> {
        let (body, body_units) = decoder.into_body_parts();
        let (selector, head_units) = selector.into_selector_parts();
        let definition = MuseAssistantDefinition::dflash2(body, selector)?;
        let mut units =
            execution::wrap_dflash_body(body_units.into_units().map_err(Error::backend_source)?);
        units.extend(execution::wrap_selector(
            head_units.into_units().map_err(Error::backend_source)?,
        ));
        Ok(Self::new(definition, ResidentUnitWindow::new(units)))
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend, P> MuseAssistantModule<B, P> {
    /// Attaches one policy and its single parameter budget to the complete topology.
    pub fn new(definition: MuseAssistantDefinition, policy: P) -> Self {
        Self {
            definition,
            policy,
            backend: std::marker::PhantomData,
        }
    }
    /// Borrows the retained backend-independent definition.
    pub fn definition(&self) -> &MuseAssistantDefinition {
        &self.definition
    }
    /// Borrows the shared body/head parameter policy for telemetry.
    pub fn policy(&self) -> &P {
        &self.policy
    }
    /// Mutably borrows the policy at a settled execution boundary.
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }
    /// Separates retained geometry and policy without cloning parameters.
    pub fn into_parts(self) -> (MuseAssistantDefinition, P) {
        (self.definition, self.policy)
    }
}

impl<B, P> MuseAssistantModule<B, P>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    P: LayerwisePolicy<B, MuseAssistantUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Declares the target-owned projection required by this proposal equation.
    /// DSpark owns its head, DFlash2 borrows raw projection, and Glimmer uses normal readout.
    pub fn target_projection<'a>(
        &self,
        states: &'a B::Tensor,
    ) -> Option<crate::composite_execution::ExternalPredictionTargetOperation<'a, B::Tensor>> {
        use crate::composite_execution::ExternalPredictionTargetOperation as Op;
        match &self.definition.equation {
            Equation::Glimmer(_) => Some(Op::ProjectLogits(states)),
            Equation::DSpark { .. } => None,
            Equation::DFlash2 { .. } => Some(Op::RawVocabularyProjection(states)),
        }
    }

    /// Prepares immutable neural proposal data on draft placement after any target head transfer.
    /// This performs no token draws and does not advance the shared proposal cursor.
    pub fn prepare_proposal(
        &mut self,
        states: &B::Tensor,
        target_projection: Option<&B::Tensor>,
        target_vocabulary: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MuseAssistantProposal<B::Tensor>, Error> {
        if states.shape().len() != 3 || states.dim(0) != 1 || states.dim(1) <= 0 {
            return Err(Error::backend("invalid Muse proposal state geometry"));
        }
        match &self.definition.equation {
            Equation::Glimmer(_) => {
                let scores = target_projection
                    .ok_or_else(|| Error::backend("Glimmer proposal requires target readout"))?;
                if scores.shape().len() != 3 || scores.shape()[..2] != states.shape()[..2] {
                    return Err(Error::backend("invalid Glimmer proposal score geometry"));
                }
                Ok(MuseAssistantProposal::Scores(scores.clone()))
            }
            Equation::DSpark { heads, .. } => {
                if target_projection.is_some() || target_vocabulary <= 0 {
                    return Err(Error::backend("invalid DSpark target-head contract"));
                }
                let mut policy = self.definition.view::<B, _, _, DsparkHeadProjection>(
                    &mut self.policy,
                    heads.unit_layout(),
                    self.definition.body_geometry().unit_layout().len(),
                )?;
                let draft_vocabulary = heads.vocabulary_size();
                let ids = B::Tensor::from_i32_slice(
                    &(0..draft_vocabulary).collect::<Vec<_>>(),
                    &[draft_vocabulary],
                    context,
                )?;
                let target_columns =
                    heads.map_draft_to_target::<B, _>(&mut policy, &ids, context)?;
                let raw = heads.logits::<B, _>(&mut policy, states, context)?;
                Ok(MuseAssistantProposal::DSpark {
                    hidden: states.clone(),
                    raw,
                    target_columns,
                    vocabulary: target_vocabulary,
                })
            }
            Equation::DFlash2 { selector, .. } => {
                let raw = target_projection.ok_or_else(|| {
                    Error::backend("DFlash2 proposal requires raw target projection")
                })?;
                if raw.shape().len() != 3 || raw.shape()[..2] != states.shape()[..2] {
                    return Err(Error::backend("invalid DFlash2 raw projection geometry"));
                }
                let (ids, unary) =
                    selector.prepare_candidates::<B>(raw, 0.1767766952966369, 20., context)?;
                Ok(MuseAssistantProposal::DFlash2 {
                    hidden: states.clone(),
                    ids,
                    unary,
                })
            }
        }
    }

    /// Scores one slot using the actual preceding global token, preserving batch dimension.
    pub fn proposal_row(
        &mut self,
        proposal: &MuseAssistantProposal<B::Tensor>,
        row: usize,
        previous_target_id: u32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MuseAssistantRow<B::Tensor>, Error> {
        let row = i32::try_from(row).map_err(Error::backend)?;
        let take_row = |tensor: &B::Tensor| {
            if tensor.shape().len() != 3 || row >= tensor.dim(1) {
                return Err(Error::backend(
                    "Muse proposal row is outside the prepared block",
                ));
            }
            tensor.index(&[Index::Full, Index::At(row), Index::Full], context)
        };
        let previous = || {
            let previous = i32::try_from(previous_target_id).map_err(Error::backend)?;
            B::Tensor::from_i32_slice(&[previous], &[1], context)
        };
        match (&self.definition.equation, proposal) {
            (Equation::Glimmer(_), MuseAssistantProposal::Scores(scores)) => Ok(MuseAssistantRow {
                logits: take_row(scores)?,
                confidence: None,
            }),
            (
                Equation::DSpark { heads, .. },
                MuseAssistantProposal::DSpark {
                    hidden,
                    raw,
                    target_columns,
                    vocabulary,
                },
            ) => {
                let mut policy = self.definition.view::<B, _, _, DsparkHeadProjection>(
                    &mut self.policy,
                    heads.unit_layout(),
                    self.definition.body_geometry().unit_layout().len(),
                )?;
                let result = heads.conditional_target_scores::<B, _>(
                    &mut policy,
                    &take_row(hidden)?,
                    &take_row(raw)?,
                    &previous()?,
                    target_columns,
                    *vocabulary,
                    context,
                )?;
                Ok(MuseAssistantRow {
                    logits: result.logits,
                    confidence: Some(result.confidence),
                })
            }
            (
                Equation::DFlash2 { selector, .. },
                MuseAssistantProposal::DFlash2 { hidden, ids, unary },
            ) => {
                let mut policy = self.definition.view::<B, _, _, SelectorProjection>(
                    &mut self.policy,
                    selector.unit_layout(),
                    self.definition.body_geometry().unit_layout().len(),
                )?;
                let logits = selector.conditional_target_scores::<B, _>(
                    &mut policy,
                    &take_row(ids)?,
                    &take_row(unary)?,
                    &take_row(hidden)?,
                    &previous()?,
                    context,
                )?;
                Ok(MuseAssistantRow {
                    logits,
                    confidence: None,
                })
            }
            _ => Err(Error::backend(
                "Muse proposal payload does not match its materialized assistant",
            )),
        }
    }

    /// Concatenates the exact ordered target publications for the selected encoder.
    pub fn assemble_target_states(
        &self,
        states: &[B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.definition
            .body_geometry()
            .assemble_target_states(states, context)
    }
    /// Appends committed context through the same body equations and shared policy.
    pub fn update_context(
        &mut self,
        previous: Option<DFlashContext<B::Tensor>>,
        pending: &B::Tensor,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DFlashContext<B::Tensor>, Error> {
        match &self.definition.equation {
            Equation::Glimmer(body) | Equation::DFlash2 { body, .. } => {
                let mut policy = self.definition.view::<B, _, _, DFlashBodyProjection>(
                    &mut self.policy,
                    body.geometry().unit_layout(),
                    0,
                )?;
                body::update_context::<B, _, _>(body, &mut policy, previous, pending, end, context)
            }
            Equation::DSpark { body, .. } => {
                let mut policy = self.definition.view::<B, _, _, DsparkBodyProjection>(
                    &mut self.policy,
                    body.geometry().unit_layout(),
                    0,
                )?;
                body::update_context::<B, _, _>(body, &mut policy, previous, pending, end, context)
            }
        }
    }
    /// Runs the retained placement through the ordinary body traversal.
    /// A partitioned definition requires its invocation's selected tensor group.
    pub fn proposal_states_selected(
        &mut self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self.definition.placement() {
            execution::MuseAssistantPlacement::Replicated => {
                self.proposal_states(embeddings, committed, end, context)
            }
            execution::MuseAssistantPlacement::TensorParallel(_) => {
                let parallel = parallel.ok_or_else(|| {
                    Error::backend(
                    "partitioned assistant proposals require their selected tensor-parallel group",
                )
                })?;
                self.proposal_states_parallel_observed(
                    embeddings,
                    committed,
                    end,
                    parallel,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::disabled(),
                )
            }
        }
    }

    /// Runs the neural body without sampling or changing committed context.
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
    /// Uses the same traversal with ordinary admitted component instrumentation.
    pub fn proposal_states_observed(
        &mut self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        if self.definition.placement() != execution::MuseAssistantPlacement::Replicated {
            return Err(Error::backend(
                "partitioned assistant proposals require their selected tensor-parallel group",
            ));
        }
        match &self.definition.equation {
            Equation::Glimmer(body) | Equation::DFlash2 { body, .. } => {
                let mut policy = self.definition.view::<B, _, _, DFlashBodyProjection>(
                    &mut self.policy,
                    body.geometry().unit_layout(),
                    0,
                )?;
                body::proposal_states::<B, _, _>(
                    body,
                    &mut policy,
                    embeddings,
                    committed,
                    end,
                    context,
                    instrumentation,
                )
            }
            Equation::DSpark { body, .. } => {
                let mut policy = self.definition.view::<B, _, _, DsparkBodyProjection>(
                    &mut self.policy,
                    body.geometry().unit_layout(),
                    0,
                )?;
                body::proposal_states::<B, _, _>(
                    body,
                    &mut policy,
                    embeddings,
                    committed,
                    end,
                    context,
                    instrumentation,
                )
            }
        }
    }
    /// Runs the selected local body and complete scoring policy with its TP group.
    /// Sampling and verification remain in the shared speculative driver.
    pub fn proposal_states_parallel_observed(
        &mut self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let execution::MuseAssistantPlacement::TensorParallel(rank) = self.definition.placement()
        else {
            return Err(Error::backend(
                "assistant definition has no selected tensor-parallel placement",
            ));
        };
        if B::parallel_size(parallel) != rank.tensor_parallel_size()
            || B::parallel_rank(parallel) != rank.tensor_parallel_rank()
        {
            return Err(Error::backend(
                "assistant proposal group differs from selected tensor-parallel rank",
            ));
        }
        let Equation::DSpark { body, .. } = &self.definition.equation else {
            return Err(Error::backend(
                "selected parallel assistant equation has no block operation",
            ));
        };
        let mut policy = self.definition.view::<B, _, _, DsparkBodyProjection>(
            &mut self.policy,
            body.geometry().unit_layout(),
            0,
        )?;
        body::proposal_states_parallel::<B, _, _>(
            body,
            &mut policy,
            embeddings,
            committed,
            end,
            parallel,
            context,
            instrumentation,
        )
    }
}
