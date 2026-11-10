//! Direct/convolved DFlash equations over true context, block and norm units.

use super::super::{
    DFlashBlock, DFlashContext, DFlashDecoderConfig, DFlashLayerContext, ProposalLayout,
};
use super::{
    parameter_projection, AssistantBodyDefinition, AssistantBodyGeometry, AssistantBodyState,
};
use crate::decoder::ComponentInstrumentation;
use eredu_nn::{
    DeclareLogicalParameters, DistributedNeuralBackend, Error, LogicalParameterDeclaration,
    NeuralBackend, NormalizationOperator, Parameterized, Tensor,
};

/// Validated direct/convolved DFlash body geometry, without resident parameters.
#[derive(Debug, Clone)]
pub struct DFlashBodyDefinition {
    config: DFlashDecoderConfig,
    geometry: AssistantBodyGeometry,
    parameters: super::super::dflash_specs::Parameters,
}
impl DFlashBodyDefinition {
    pub(in crate::muse::assistant) fn project_formats(
        &self,
        formats: &std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
        uniform: Option<eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self, Error> {
        let mut projected = self.clone();
        projected.config.quantization = uniform;
        projected.config.quantized_weights = formats.clone();
        projected.parameters = self
            .parameters
            .project_formats(&|name| projected.config.linear_format_for(name))?;
        Ok(projected)
    }
    /// Validates architecture geometry before constructing any parameter unit.
    pub fn new(config: DFlashDecoderConfig) -> Result<Self, Error> {
        let parameters = super::super::dflash_specs::Parameters::new(&config)?;
        let attention = &parameters.blocks[0].attention;
        let geometry = AssistantBodyGeometry::new(
            parameters.edge.projection.output,
            config.target_layer_ids.len(),
            parameters.blocks.len(),
            attention.projection.geometry.key_value_heads(),
            attention.projection.geometry.key_dimensions(),
            config.block_size,
            config.sliding_window,
            ProposalLayout::AnchorThenMasks,
        )?;
        Ok(Self {
            config,
            geometry,
            parameters,
        })
    }
    pub(in crate::muse::assistant) fn append_parameters(
        &self,
        declaration: &mut crate::external_assistant::AssistantParameterDeclarations,
    ) -> Result<(), Error> {
        self.parameters.append_parameters(declaration)
    }

    /// Ordered target publications selected by the trained encoder.
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.config.target_layer_ids
    }
    /// Cold unit layout and context geometry, without a backend specialization.
    pub const fn geometry(&self) -> &AssistantBodyGeometry {
        &self.geometry
    }

    pub(in crate::muse::assistant) fn memory_body(
        &self,
    ) -> eredu_runtime::memory_forecast::ConditionedPredictionBody {
        use eredu_runtime::memory_forecast::{
            ConditionedBlockConvolution, ConditionedPredictionBody,
        };
        let g = &self.geometry;
        let attention = &self.parameters.blocks[0].attention;
        ConditionedPredictionBody {
            hidden: g.hidden as u64,
            captures: g.captures as u64,
            layers: g.layers as u64,
            query_heads: attention.projection.geometry.query_heads() as u64,
            head_width: attention.projection.geometry.key_dimensions() as u64,
            key_value_width: attention.projection.key.output as u64,
            intermediate: self.parameters.blocks[0].mlp.gate.output as u64,
            context_window: g.window as u64,
            block_positions: g.block as u64,
            convolution: self.parameters.blocks[0].convolutions[0]
                .as_ref()
                .map(|spec| {
                    let geometry = spec.geometry();
                    ConditionedBlockConvolution {
                        taps: geometry.taps as u64,
                        group_size: geometry.group_size as u64,
                    }
                }),
        }
    }
}

impl DeclareLogicalParameters for DFlashBodyDefinition {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.parameters.logical_parameters()
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct ContextEncoder<B: NeuralBackend> {
    projection: B::Linear,
    normalization: B::Normalization,
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum Module<B: NeuralBackend + DistributedNeuralBackend> {
    Context(ContextEncoder<B>),
    Decoder(DFlashBlock<B>),
    Final(B::Normalization),
}

/// One independently populated DFlash context, decoder block or final norm.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DFlashBodyUnit<B: NeuralBackend + DistributedNeuralBackend> {
    module: Module<B>,
}

impl<B: NeuralBackend + DistributedNeuralBackend> AssistantBodyDefinition<B>
    for DFlashBodyDefinition
{
    type Unit = DFlashBodyUnit<B>;
    fn geometry(&self) -> &AssistantBodyGeometry {
        &self.geometry
    }
    fn build_unit(
        &self,
        ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        let config = &self.config;
        let module = if ordinal == 0 {
            Module::Context(ContextEncoder {
                projection: B::linear(self.parameters.edge.projection.clone(), context)?,
                normalization: B::normalization(
                    self.parameters.edge.context_norm.clone(),
                    context,
                )?,
            })
        } else if ordinal <= self.parameters.blocks.len() {
            Module::Decoder(DFlashBlock::new(
                config,
                &self.parameters.blocks[ordinal - 1],
                context,
            )?)
        } else if ordinal == self.parameters.blocks.len() + 1 {
            Module::Final(B::normalization(
                self.parameters.edge.final_norm.clone(),
                context,
            )?)
        } else {
            return Err(Error::backend("unknown DFlash parameter unit"));
        };
        Ok(DFlashBodyUnit { module })
    }
    fn encode(
        &self,
        unit: &mut Self::Unit,
        pending: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let Module::Context(encoder) = &mut unit.module else {
            return Err(Error::backend("DFlash context unit required"));
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
            return Err(Error::backend("DFlash decoder unit required"));
        };
        block.attention.project_context(encoded, start, context)
    }
    fn begin_proposal(
        &self,
        embeddings: &B::Tensor,
        _: &DFlashContext<B::Tensor>,
        _: i32,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<(AssistantBodyState<B::Tensor>, Option<B::Tensor>), Error> {
        Ok((
            AssistantBodyState {
                hidden: embeddings.clone(),
                residual: None,
            },
            None,
        ))
    }
    fn forward_block(
        &self,
        unit: &mut Self::Unit,
        state: &AssistantBodyState<B::Tensor>,
        prior: &DFlashLayerContext<B::Tensor>,
        retained: i32,
        end: i32,
        _: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<AssistantBodyState<B::Tensor>, Error> {
        let Module::Decoder(block) = &mut unit.module else {
            return Err(Error::backend("DFlash decoder unit required"));
        };
        block.forward_observed(
            state,
            prior,
            retained,
            end,
            self.config.sliding_window,
            context,
            instrumentation,
        )
    }
    fn normalize(
        &self,
        unit: &mut Self::Unit,
        state: &AssistantBodyState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let Module::Final(norm) = &mut unit.module else {
            return Err(Error::backend("DFlash final norm unit required"));
        };
        let dtype = state
            .hidden
            .element_type()
            .ok_or_else(|| Error::backend("assistant state has no dtype"))?;
        let input = match &state.residual {
            Some(residual) => state
                .hidden
                .cast_float(eredu_nn::TensorElementType::F32, context)?
                .add(
                    &residual.cast_float(eredu_nn::TensorElementType::F32, context)?,
                    context,
                )?,
            None => state.hidden.clone(),
        };
        norm.forward(&input, context)?.cast_float(dtype, context)
    }
}
