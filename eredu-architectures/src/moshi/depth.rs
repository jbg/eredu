//! Ordered depth-prediction slice units.

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, LinearOperator,
    LinearSpec, NeuralBackend, ParameterSpec, Tensor, VocabularyParallelRange,
};
use eredu_runtime::{LayerRuntimeState, VocabularyModulePlacement};

use super::{block, MoshiConfig, MoshiTransformerConfig};

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
    #[parameter(skip)]
    index: usize,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DepthSlice<B> {
    /// Builds one canonical depth slice.
    pub fn new(
        config: &MoshiConfig,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_geometry(config, index, None, context)
    }

    /// Builds one rank-local depth slice with vocabulary-parallel edge modules.
    pub fn new_parallel(
        config: &MoshiConfig,
        index: usize,
        geometry: &super::LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_geometry(config, index, Some(geometry), context)
    }

    fn from_geometry(
        config: &MoshiConfig,
        index: usize,
        geometry: Option<&super::LocalGeometry>,
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
        let embedding_spec = EmbeddingSpec {
            vocabulary: input_vocabulary,
            dimensions: transformer.hidden_size(),
            weight: ParameterSpec::trainable(&embedding_name).map_err(Error::backend)?,
            format: crate::linear_format::standard_linear_format(
                &embedding_name,
                config.native_quantization().into(),
            )?,
        };
        let input_spec = LinearSpec {
            input: config.temporal().hidden_size(),
            output: transformer.hidden_size(),
            weight: ParameterSpec::trainable(&input_name).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(
                &input_name,
                config.native_quantization().into(),
            )?,
        };
        let output_spec = LinearSpec {
            input: transformer.hidden_size(),
            output: config.audio_vocabulary_size(),
            weight: ParameterSpec::trainable(&output_name).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(
                &output_name,
                config.native_quantization().into(),
            )?,
        };
        let placement = |name: &str, vocabulary, missing| -> Result<_, Error> {
            Ok(match geometry {
                None => VocabularyModulePlacement::Local,
                Some(geometry) => {
                    VocabularyModulePlacement::VocabularyParallel(VocabularyParallelRange {
                        global_vocabulary: vocabulary as usize,
                        local: geometry
                            .vocabulary_range(name)
                            .cloned()
                            .ok_or_else(|| Error::backend(missing))?,
                    })
                }
            })
        };
        let embedding = placement(
            &embedding_name,
            embedding_spec.vocabulary,
            "missing depth embedding range",
        )?;
        let output = placement(
            &output_name,
            output_spec.output,
            "missing depth output range",
        )?;
        embedding.validate_rows(embedding_spec.vocabulary)?;
        output.validate_rows(output_spec.output)?;
        let embedding = embedding
            .materialize_embedding::<B>(embedding_spec, context)?
            .expect("depth slice owns its embedding");
        let input = B::linear(input_spec, context)?;
        let output = output
            .materialize_linear::<B>(output_spec, context)?
            .expect("depth slice owns its output");
        let blocks = (0..transformer.num_hidden_layers() as usize)
            .map(|layer| match geometry {
                None => block::build::<B>(&transformer, layer, context),
                Some(geometry) => {
                    let local = geometry
                        .depth_config(&transformer, index, layer)
                        .map_err(Error::backend)?;
                    block::build::<B>(&local, layer, context)
                }
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
