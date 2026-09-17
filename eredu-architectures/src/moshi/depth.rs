//! Ordered depth-prediction slice units.

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, LinearOperator,
    LinearSpec, NeuralBackend, Tensor, VocabularyParallelRange,
};
use eredu_runtime::LayerRuntimeState;

use super::{block, MoshiConfig, MoshiTransformerConfig};
use crate::decoder::ModuleMetadata;

/// One ordered codebook slice with its own projections and shared decoder blocks.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DepthSlice<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Previous-decision embedding.
    pub embedding: B::Embedding,
    /// Temporal-to-depth projection.
    pub input: B::Linear,
    /// Audio-vocabulary projection.
    pub output: B::Linear,
    /// Shared decoder blocks reusing the frame-local depth state slots.
    pub blocks: Vec<block::Block<B>>,
    #[parameter(skip, metadata)]
    index: usize,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DepthSlice<B> {
    /// Builds one canonical depth slice.
    pub fn new(
        config: &MoshiConfig,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata=ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self,MoshiTransformerConfig,Vec<block::Block<B>>,usize)>()?;
        let transformer=match B::construction_metadata(context) {
            Some(context)=>config.depth_transformer_workspace(index,context)?,
            None=>config.depth_transformer(index).map_err(Error::backend)?,
        };
        let prefix=metadata.text(format_args!("depformer.slices.{index}"))?;
        let input_vocabulary = if index == 0 {
            config.text_vocabulary_size()
        } else {
            config.audio_vocabulary_size()
        }
        .checked_add(1)
        .ok_or_else(|| metadata.error(format_args!("Moshi depth input vocabulary overflowed")))?;
        let embedding_name = metadata.text(format_args!("{prefix}.emb.weight"))?;
        let input_name = metadata.text(format_args!("{prefix}.linear_in.weight"))?;
        let output_name = metadata.text(format_args!("{prefix}.linear_out.weight"))?;
        let embedding = B::embedding(
            EmbeddingSpec {
                vocabulary: input_vocabulary,
                dimensions: transformer.hidden_size(),
                weight: metadata.plain_parameter(&embedding_name)?,
                format: metadata.format(
                    &embedding_name,
                    config.native_quantization().into(),
                )?,
            },
            context,
        )?;
        let input = B::linear(
            LinearSpec {
                input: config.temporal().hidden_size(),
                output: transformer.hidden_size(),
                weight: metadata.plain_parameter(&input_name)?,
                bias: None,
                format: metadata.format(
                    &input_name,
                    config.native_quantization().into(),
                )?,
            },
            context,
        )?;
        let output = B::linear(
            LinearSpec {
                input: transformer.hidden_size(),
                output: config.audio_vocabulary_size(),
                weight: metadata.plain_parameter(&output_name)?,
                bias: None,
                format: metadata.format(
                    &output_name,
                    config.native_quantization().into(),
                )?,
            },
            context,
        )?;
        let mut blocks=metadata.vector(transformer.num_hidden_layers() as usize)?;
        for layer in 0..transformer.num_hidden_layers() as usize {
            blocks.push(block::build::<B>(&transformer,layer,context)?);
        }
        Ok(Self {
            embedding,
            input,
            output,
            blocks,
            index,
        })
    }

    /// Builds one rank-local depth slice with vocabulary-parallel edge modules.
    pub fn new_parallel(
        config: &MoshiConfig,
        index: usize,
        geometry: &super::LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata=ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self,MoshiTransformerConfig,Vec<block::Block<B>>,usize)>()?;
        let transformer=match B::construction_metadata(context) {
            Some(context)=>config.depth_transformer_workspace(index,context)?,
            None=>config.depth_transformer(index).map_err(Error::backend)?,
        };
        let prefix=metadata.text(format_args!("depformer.slices.{index}"))?;
        let input_vocabulary = if index == 0 {
            config.text_vocabulary_size()
        } else {
            config.audio_vocabulary_size()
        }
        .checked_add(1)
        .ok_or_else(|| metadata.error(format_args!("Moshi depth input vocabulary overflowed")))?;
        let embedding_name = metadata.text(format_args!("{prefix}.emb.weight"))?;
        let output_name = metadata.text(format_args!("{prefix}.linear_out.weight"))?;
        let embedding = B::vocabulary_parallel_embedding(
            EmbeddingSpec {
                vocabulary: input_vocabulary,
                dimensions: transformer.hidden_size(),
                weight: metadata.plain_parameter(&embedding_name)?,
                format: metadata.format(
                    &embedding_name,
                    config.native_quantization().into(),
                )?,
            },
            VocabularyParallelRange {
                global_vocabulary: input_vocabulary as usize,
                local: geometry
                    .vocabulary_range(&embedding_name)
                    .cloned()
                    .ok_or_else(|| metadata.error(format_args!("missing depth embedding range")))?,
            },
            context,
        )?;
        let input = B::linear(
            LinearSpec {
                input: config.temporal().hidden_size(),
                output: transformer.hidden_size(),
                weight: metadata.named_parameter(format_args!("{prefix}.linear_in.weight"))?,
                bias: None,
                format: metadata.format(
                    &metadata.text(format_args!("{prefix}.linear_in.weight"))?,
                    config.native_quantization().into(),
                )?,
            },
            context,
        )?;
        let output = B::vocabulary_parallel_linear(
            LinearSpec {
                input: transformer.hidden_size(),
                output: config.audio_vocabulary_size(),
                weight: metadata.plain_parameter(&output_name)?,
                bias: None,
                format: metadata.format(
                    &output_name,
                    config.native_quantization().into(),
                )?,
            },
            VocabularyParallelRange {
                global_vocabulary: config.audio_vocabulary_size() as usize,
                local: geometry
                    .vocabulary_range(&output_name)
                    .cloned()
                    .ok_or_else(|| metadata.error(format_args!("missing depth output range")))?,
            },
            context,
        )?;
        let mut blocks=metadata.vector(transformer.num_hidden_layers() as usize)?;
        for layer in 0..transformer.num_hidden_layers() as usize {
            let local=match B::construction_metadata(context) {
                Some(context)=>geometry.depth_config_workspace(&transformer,index,layer,context)?,
                None=>geometry.depth_config(&transformer,index,layer).map_err(Error::backend)?,
            };
            blocks.push(block::build::<B>(&local,layer,context)?);
        }
        Ok(Self {
            embedding,
            input,
            output,
            blocks,
            index,
        })
    }

    /// Zero-based codebook prediction index.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Executes this slice from the normalized temporal state and prior token.
    pub fn forward<S>(
        &mut self,
        transformer: &MoshiTransformerConfig,
        temporal: &B::Tensor,
        previous: &B::Tensor,
        state_offset: usize,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.forward_with_readout(
            transformer,
            temporal,
            previous,
            state_offset,
            state,
            eredu_core::OutputDemand::Sequence,
            context,
        )
    }

    /// Executes the complete depth body and projects only demanded positions.
    /// State-only execution returns the genuine body hidden value to traversal.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_readout<S>(
        &mut self,
        transformer: &MoshiTransformerConfig,
        temporal: &B::Tensor,
        previous: &B::Tensor,
        state_offset: usize,
        state: &mut S,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if self.blocks.len() != transformer.num_hidden_layers() as usize {
            return Err(Error::backend("Moshi depth block count drifted"));
        }
        let embedded =
            self.embedding
                .lookup(previous, EmbeddingLookupPolicy::ZeroSentinel(-1), context)?;
        let mut hidden = self
            .input
            .forward(temporal, context)?
            .add(&embedded, context)?;
        let mask = if hidden.dim(1) > 1 {
            let offset = state.layer(state_offset).map_err(Error::backend)?.offset();
            Some(B::causal_mask(hidden.dim(1), offset, None, context)?)
        } else {
            None
        };
        for (layer, block) in self.blocks.iter_mut().enumerate() {
            hidden = block::forward(
                block,
                state_offset + layer,
                &hidden,
                mask.as_ref(),
                true,
                state,
                context,
            )?;
        }
        match crate::readout::select_readout_positions(&hidden, demand, 1, context)? {
            Some(selected) => self.output.forward(&selected, context),
            None => Ok(hidden),
        }
    }

    /// Executes the shared depth body from a caller-produced vocabulary
    /// embedding, leaving vocabulary-parallel output projection to the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_embedded_parallel<S>(
        &mut self,
        transformer: &MoshiTransformerConfig,
        temporal: &B::Tensor,
        embedded: &B::Tensor,
        state_offset: usize,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if self.blocks.len() != transformer.num_hidden_layers() as usize {
            return Err(Error::backend("Moshi depth block count drifted"));
        }
        let mut hidden = self
            .input
            .forward(temporal, context)?
            .add(embedded, context)?;
        let mask = if hidden.dim(1) > 1 {
            let offset = state.layer(state_offset).map_err(Error::backend)?.offset();
            Some(B::causal_mask(hidden.dim(1), offset, None, context)?)
        } else {
            None
        };
        for (layer, block) in self.blocks.iter_mut().enumerate() {
            hidden = block::forward_parallel(
                block,
                state_offset + layer,
                &hidden,
                mask.as_ref(),
                true,
                state,
                parallel,
                context,
            )?;
        }
        Ok(hidden)
    }

    /// Executes a complete rank-local vocabulary-parallel depth slice.
    pub fn forward_parallel<S>(
        &mut self,
        transformer: &MoshiTransformerConfig,
        temporal: &B::Tensor,
        previous: &B::Tensor,
        state_offset: usize,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.forward_parallel_with_readout(
            transformer,
            temporal,
            previous,
            state_offset,
            state,
            eredu_core::OutputDemand::Sequence,
            parallel,
            context,
        )
    }

    /// Parallel counterpart preserving the complete body before row selection.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_readout<S>(
        &mut self,
        transformer: &MoshiTransformerConfig,
        temporal: &B::Tensor,
        previous: &B::Tensor,
        state_offset: usize,
        state: &mut S,
        demand: eredu_core::OutputDemand,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.embedding,
            previous,
            EmbeddingLookupPolicy::ZeroSentinel(-1),
            parallel,
            context,
        )?;
        let hidden = self.forward_embedded_parallel(
            transformer,
            temporal,
            &embedded,
            state_offset,
            state,
            parallel,
            context,
        )?;
        match crate::readout::select_readout_positions(&hidden, demand, 1, context)? {
            Some(selected) => {
                B::vocabulary_parallel_project(&mut self.output, &selected, parallel, context)
            }
            None => Ok(hidden),
        }
    }
}
