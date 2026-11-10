//! DSpark body units preserve context/input normalization and residual precision.

use super::super::{
    dspark::ProposalContext, proposal_attention_mask, DFlashContext, DFlashLayerContext,
    ProposalAttentionPolicy, ProposalLayout,
};
use super::{
    parameter_projection, AssistantBodyDefinition, AssistantBodyGeometry, AssistantBodyState,
};
use crate::{
    decoder::ComponentInstrumentation,
    muse::{AttentionEquation, DecoderPolicy, LearnedRmsNorm, TransformerBlock, WeightConvention},
};
use eredu_nn::{
    DistributedNeuralBackend, Error, GroupedNeuralBackend, Parameterized, Tensor, TensorElementType,
};

mod parallel;
mod parameters;
pub(crate) use parameters::DsparkBodyParameters;

/// Validated causal dense body geometry, separate from acquired parameters.
#[derive(Debug, Clone)]
pub struct DsparkBodyDefinition {
    args: DecoderPolicy,
    parameters: std::sync::Arc<DsparkBodyParameters>,
    local_parameters: Option<DsparkBodyParameters>,
    geometry: AssistantBodyGeometry,
}
impl DsparkBodyDefinition {
    /// Validates the normalized DSpark equations before any neural construction.
    pub fn new(args: DecoderPolicy, captures: usize, block: usize) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let parameters = DsparkBodyParameters::from_args(&args, captures)?;
        Self::from_parameters(args, std::sync::Arc::new(parameters), block)
    }

    pub(in crate::muse::assistant) fn from_parameters(
        args: DecoderPolicy,
        parameters: std::sync::Arc<DsparkBodyParameters>,
        block: usize,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        if args.is_moe()
            || args.attention_equation != AttentionEquation::NormalizedHeadsScoreScaled
            || args.weight_convention != WeightConvention::HuggingFace
            || args.vision_config.is_some()
            || args.audio_config.is_some()
            || args.attention_schedule.len() != args.num_hidden_layers as usize
            || args.layer_uses_rope.len() != args.num_hidden_layers as usize
            || args.layer_uses_rope.iter().any(|uses| !uses)
            || args.attention_schedule.iter().any(|p| p.window().is_none())
            || args
                .attention_schedule
                .iter()
                .any(|p| Some(p) != args.attention_schedule.iter().next())
        {
            return Err(Error::backend("invalid normalized DSpark decoder equation"));
        }
        let captures = parameters.validate_binding(&args)?;
        let window = args
            .attention_schedule
            .iter()
            .next()
            .unwrap()
            .window()
            .unwrap()
            .get() as i32;
        let geometry = AssistantBodyGeometry::new(
            args.hidden_size,
            captures,
            args.num_hidden_layers as usize,
            args.num_key_value_heads,
            args.head_dim,
            block,
            window,
            ProposalLayout::SampleFromAnchor,
        )?;
        Ok(Self {
            args,
            parameters,
            local_parameters: None,
            geometry,
        })
    }
    fn selected_parameters(&self) -> &DsparkBodyParameters {
        self.local_parameters.as_ref().unwrap_or(&self.parameters)
    }

    /// Cold unit layout and context geometry, without a backend specialization.
    pub const fn geometry(&self) -> &AssistantBodyGeometry {
        &self.geometry
    }

    /// Keeps original source formats while using the selected block's logical dimensions.
    pub(in crate::muse::assistant) fn source_for_selected(
        &self,
        selected: &Self,
    ) -> Result<Self, Error> {
        let global = |a: &DecoderPolicy| {
            (
                a.hidden_size,
                a.num_hidden_layers,
                a.num_attention_heads,
                a.num_key_value_heads,
                a.head_dim,
                a.intermediate_size,
            )
        };
        if global(&self.args) != global(&selected.args)
            || self.geometry.captures != selected.geometry.captures
            || self.geometry.block != selected.geometry.block
            || self.geometry.window != selected.geometry.window
        {
            return Err(Error::backend(
                "assistant source and selected neural geometries differ",
            ));
        }
        let mut source = self.clone();
        let mut parameters = self.parameters.as_ref().clone();
        for (source, selected) in parameters
            .blocks
            .iter_mut()
            .zip(&selected.selected_parameters().blocks)
        {
            let heads = selected.attention.projection.geometry;
            *source = source.select_dimensions(
                heads.query_heads(),
                heads.key_value_heads(),
                selected.dense.gate.output,
            )?;
        }
        parameters.validate()?;
        source.local_parameters = Some(parameters);
        source.geometry.key_value_heads = selected.geometry.key_value_heads;
        Ok(source)
    }

    pub(in crate::muse::assistant) fn memory_body(
        &self,
    ) -> eredu_runtime::memory_forecast::ConditionedPredictionBody {
        let g = &self.geometry;
        let blocks = &self.selected_parameters().blocks;
        let heads = blocks[0].attention.projection.geometry;
        eredu_runtime::memory_forecast::ConditionedPredictionBody {
            hidden: g.hidden as u64,
            captures: g.captures as u64,
            layers: g.layers as u64,
            query_heads: blocks
                .iter()
                .map(|b| b.attention.projection.geometry.query_heads() as u64)
                .max()
                .unwrap(),
            head_width: heads.key_dimensions() as u64,
            key_value_width: heads.key_width() as u64,
            intermediate: blocks
                .iter()
                .map(|b| b.dense.gate.output as u64)
                .max()
                .unwrap(),
            context_window: g.window as u64,
            block_positions: g.block as u64,
            convolution: None,
        }
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct ContextEncoder<B: GroupedNeuralBackend> {
    projection: B::Linear,
    normalization: LearnedRmsNorm<B>,
}
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum Module<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    Context(ContextEncoder<B>),
    Decoder(TransformerBlock<B>),
    Final(LearnedRmsNorm<B>),
}
/// One independently populated DSpark context, dense block or final norm.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DsparkBodyUnit<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    module: Module<B>,
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> AssistantBodyDefinition<B>
    for DsparkBodyDefinition
{
    type Unit = DsparkBodyUnit<B>;
    fn geometry(&self) -> &AssistantBodyGeometry {
        &self.geometry
    }
    fn build_unit(
        &self,
        ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        let edge = self.selected_parameters().edge.clone();
        let [context_norm, final_norm] = edge.norms;
        let module = if ordinal == 0 {
            Module::Context(ContextEncoder {
                projection: B::linear(edge.projection, context)?,
                normalization: LearnedRmsNorm::new(context_norm, context)?,
            })
        } else if ordinal <= self.geometry.layers {
            Module::Decoder(TransformerBlock::from_dspark_parameters(
                &self.args,
                self.selected_parameters().blocks[ordinal - 1].clone(),
                ordinal - 1,
                context,
            )?)
        } else if ordinal == self.geometry.layers + 1 {
            Module::Final(LearnedRmsNorm::new(final_norm, context)?)
        } else {
            return Err(Error::backend("unknown DSpark parameter unit"));
        };
        Ok(DsparkBodyUnit { module })
    }
    fn encode(
        &self,
        unit: &mut Self::Unit,
        pending: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let Module::Context(encoder) = &mut unit.module else {
            return Err(Error::backend("DSpark context unit required"));
        };
        let encoded = parameter_projection::<B>(&mut encoder.projection, pending, context)?;
        encoder.normalization.forward(&encoded, context)
    }
    fn project_context(
        &self,
        unit: &mut Self::Unit,
        encoded: &B::Tensor,
        start: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DFlashLayerContext<B::Tensor>, Error> {
        let Module::Decoder(block) = &mut unit.module else {
            return Err(Error::backend("DSpark decoder unit required"));
        };
        let input = block.input_norm.forward(encoded, context)?;
        block.attention.project_context(&input, start, context)
    }
    fn begin_proposal(
        &self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(AssistantBodyState<B::Tensor>, Option<B::Tensor>), Error> {
        let mask = proposal_attention_mask::<B::Tensor>(
            committed.retained_len(),
            embeddings.dim(1),
            end,
            self.geometry.window,
            ProposalAttentionPolicy::Causal,
            context,
        )?;
        Ok((
            AssistantBodyState {
                hidden: embeddings.cast_float(TensorElementType::F32, context)?,
                residual: None,
            },
            Some(mask),
        ))
    }
    fn forward_block(
        &self,
        unit: &mut Self::Unit,
        state: &AssistantBodyState<B::Tensor>,
        prior: &DFlashLayerContext<B::Tensor>,
        _: i32,
        end: i32,
        mask: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observed: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<AssistantBodyState<B::Tensor>, Error> {
        let Module::Decoder(block) = &mut unit.module else {
            return Err(Error::backend("DSpark decoder unit required"));
        };
        let mut cache = ProposalContext::<B> {
            prior: Some(prior.clone()),
            end,
            window: self.geometry.window,
        };
        Ok(AssistantBodyState {
            hidden: block.forward_observed(
                &state.hidden,
                mask,
                Some(&mut cache),
                context,
                observed,
            )?,
            residual: None,
        })
    }
    fn normalize(
        &self,
        unit: &mut Self::Unit,
        state: &AssistantBodyState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let Module::Final(norm) = &mut unit.module else {
            return Err(Error::backend("DSpark final norm unit required"));
        };
        norm.forward(&state.hidden, context)
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend>
    super::TensorParallelAssistantBodyDefinition<B> for DsparkBodyDefinition
{
    fn forward_block_parallel(
        &self,
        unit: &mut DsparkBodyUnit<B>,
        state: &AssistantBodyState<B::Tensor>,
        prior: &DFlashLayerContext<B::Tensor>,
        _: i32,
        end: i32,
        mask: Option<&B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observed: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<AssistantBodyState<B::Tensor>, Error> {
        let Module::Decoder(block) = &mut unit.module else {
            return Err(Error::backend("DSpark decoder unit required"));
        };
        let mut cache = ProposalContext::<B> {
            prior: Some(prior.clone()),
            end,
            window: self.geometry.window,
        };
        Ok(AssistantBodyState {
            hidden: block.forward_dense_parallel_observed(
                &state.hidden,
                mask,
                Some(&mut cache),
                parallel,
                context,
                observed,
            )?,
            residual: None,
        })
    }
}
