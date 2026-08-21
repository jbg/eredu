//! Ordered depth-prediction slice units.

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, LinearOperator,
    LinearSpec, NeuralBackend, ParameterSpec, Tensor, VocabularyParallelRange,
};
use eredu_runtime::LayerRuntimeState;

use super::{block, MoshiConfig, MoshiTransformerConfig};

/// One ordered codebook slice with its own projections and shared decoder blocks.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DepthSlice<B: NeuralBackend> {
    /// Previous-decision embedding.
    pub embedding: B::Embedding,
    /// Temporal-to-depth projection.
    pub input: B::Linear,
    /// Audio-vocabulary projection.
    pub output: B::Linear,
    /// Shared decoder blocks reusing the frame-local depth state slots.
    pub blocks: Vec<block::Block<B>>,
    #[parameter(skip)]
    index: usize,
}

impl<B: NeuralBackend> DepthSlice<B> {
    /// Builds one canonical depth slice.
    pub fn new(
        config: &MoshiConfig,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let transformer = config.depth_transformer(index).map_err(Error::backend)?;
        let prefix = format!("depformer.slices.{index}");
        let input_vocabulary = if index == 0 {
            config.text_vocabulary_size()
        } else {
            config.audio_vocabulary_size()
        }
        .checked_add(1)
        .ok_or_else(|| Error::backend("Moshi depth input vocabulary overflowed"))?;
        let embedding_name = format!("{prefix}.emb.weight");
        let input_name = format!("{prefix}.linear_in.weight");
        let output_name = format!("{prefix}.linear_out.weight");
        let embedding = B::embedding(
            EmbeddingSpec {
                vocabulary: input_vocabulary,
                dimensions: transformer.hidden_size(),
                weight: ParameterSpec::trainable(&embedding_name).map_err(Error::backend)?,
                quantization: config.native_quantization(),
            },
            context,
        )?;
        let input = B::linear(
            LinearSpec {
                input: config.temporal().hidden_size(),
                output: transformer.hidden_size(),
                weight: ParameterSpec::trainable(&input_name).map_err(Error::backend)?,
                bias: None,
                format: config.native_quantization().into(),
            },
            context,
        )?;
        let output = B::linear(
            LinearSpec {
                input: transformer.hidden_size(),
                output: config.audio_vocabulary_size(),
                weight: ParameterSpec::trainable(&output_name).map_err(Error::backend)?,
                bias: None,
                format: config.native_quantization().into(),
            },
            context,
        )?;
        let blocks = (0..transformer.num_hidden_layers() as usize)
            .map(|layer| block::build::<B>(&transformer, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
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
        let transformer = config.depth_transformer(index).map_err(Error::backend)?;
        let prefix = format!("depformer.slices.{index}");
        let input_vocabulary = if index == 0 {
            config.text_vocabulary_size()
        } else {
            config.audio_vocabulary_size()
        }
        .checked_add(1)
        .ok_or_else(|| Error::backend("Moshi depth input vocabulary overflowed"))?;
        let embedding_name = format!("{prefix}.emb.weight");
        let output_name = format!("{prefix}.linear_out.weight");
        let embedding = B::vocabulary_parallel_embedding(
            EmbeddingSpec {
                vocabulary: input_vocabulary,
                dimensions: transformer.hidden_size(),
                weight: ParameterSpec::trainable(&embedding_name).map_err(Error::backend)?,
                quantization: config.native_quantization(),
            },
            VocabularyParallelRange {
                global_vocabulary: input_vocabulary as usize,
                local: geometry
                    .vocabulary_range(&embedding_name)
                    .cloned()
                    .ok_or_else(|| Error::backend("missing depth embedding range"))?,
            },
            context,
        )?;
        let input = B::linear(
            LinearSpec {
                input: config.temporal().hidden_size(),
                output: transformer.hidden_size(),
                weight: ParameterSpec::trainable(format!("{prefix}.linear_in.weight"))
                    .map_err(Error::backend)?,
                bias: None,
                format: config.native_quantization().into(),
            },
            context,
        )?;
        let output = B::vocabulary_parallel_linear(
            LinearSpec {
                input: transformer.hidden_size(),
                output: config.audio_vocabulary_size(),
                weight: ParameterSpec::trainable(&output_name).map_err(Error::backend)?,
                bias: None,
                format: config.native_quantization().into(),
            },
            VocabularyParallelRange {
                global_vocabulary: config.audio_vocabulary_size() as usize,
                local: geometry
                    .vocabulary_range(&output_name)
                    .cloned()
                    .ok_or_else(|| Error::backend("missing depth output range"))?,
            },
            context,
        )?;
        let blocks = (0..transformer.num_hidden_layers() as usize)
            .map(|layer| {
                let local = geometry
                    .depth_config(&transformer, index, layer)
                    .map_err(Error::backend)?;
                block::build::<B>(&local, layer, context)
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        self.output.forward(&hidden, context)
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
        B::vocabulary_parallel_project(&mut self.output, &hidden, parallel, context)
    }
}
