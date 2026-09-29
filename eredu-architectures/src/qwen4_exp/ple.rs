//! Per-layer lexical injection into complete residual streams.
use super::{
    config::Config,
    ngram::{NGramEmbedding, NGramEmbeddingError, NGramEmbeddingSpec, NGramError, NGramHashSpec},
};
use crate::decoder::ComponentInstrumentation;
use eredu_core::cache::{
    MutableStateResidency, StateTensorDimension as D, StateTensorDtype, StateTensorPolicy,
    StateTensorRole,
};
use eredu_nn::residual_streams::ResidualStreamGeometry;
use eredu_nn::{
    CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error,
    GroupedNeuralBackend, LinearFormatSpec, LinearOperator, LinearSpec,
    NeuralOperatorCapabilities as C, NormalizationConstructionSpec, NormalizationOperator,
    NormalizationScale, ParameterSpec, Tensor,
};
use eredu_runtime::{ParameterBankAccess, ParameterProvider, RuntimeStateComponents};

/// Exact identities and geometry for the released PLE equation.
#[derive(Debug, Clone)]
pub struct LexicalInjectionSpec {
    /// Complete residual stream geometry.
    pub geometry: ResidualStreamGeometry,
    /// Hashing, bounded table lookup and retained original token IDs.
    pub embedding: NGramEmbeddingSpec,
    /// Concatenated n-gram feature width.
    pub embedding_width: i32,
    /// Shared n-gram features to one key per residual stream.
    pub key: LinearSpec,
    /// Shared n-gram features to the common injected value.
    pub value: LinearSpec,
    /// Grouped RMS normalization with explicit checkpoint scale policy.
    pub key_norm: NormalizationConstructionSpec,
    /// Grouped normalization of original residual streams.
    pub query_norm: NormalizationConstructionSpec,
    /// Grouped normalization before the local convolution.
    pub convolution_norm: NormalizationConstructionSpec,
    /// Bias-free dilated convolution followed by SiLU.
    pub convolution: CausalDepthwiseConvolutionSpec,
    /// Explicit history identity, distinct from the recurrent mixer's convolution.
    pub convolution_slot: u32,
}
impl LexicalInjectionSpec {
    /// Declares the lexical equation from configuration and header-only row geometry.
    /// Hash literals are bound separately when constructing the executable injection.
    pub fn from_header(
        config: &Config,
        layer: usize,
        root: &str,
        table: &eredu_runtime::RowLookupSpec,
        max_rows: usize,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, NGramEmbeddingError> {
        let eos = config.eos.first().ok_or(NGramError::Constants)?;
        if !config.ngram.layers.contains(&layer) {
            return Err(NGramError::Constants.into());
        }
        let geometry = ResidualStreamGeometry::new(config.residual.streams, config.hidden_size)?;
        let root = format!("{root}.ple");
        let parameter = |suffix: &str| {
            ParameterSpec::trainable(format!("{root}.{suffix}.weight")).map_err(Error::backend)
        };
        let mut linear = |suffix: &str, output| -> Result<LinearSpec, Error> {
            let weight = parameter(suffix)?;
            Ok(LinearSpec {
                input: config.ngram.embedding_dim,
                output,
                format: format(weight.id.as_str())?,
                weight,
                bias: None,
            })
        };
        let norm = |suffix: &str| -> Result<NormalizationConstructionSpec, Error> {
            Ok(NormalizationConstructionSpec {
                dimensions: geometry.flattened_width(),
                groups: Some(geometry.streams()),
                epsilon: config.norm_epsilon,
                scale: NormalizationScale::LearnedOffset {
                    weight: parameter(suffix)?,
                    offset: 1.,
                },
            })
        };
        let spec = Self {
            geometry,
            embedding: NGramEmbeddingSpec::new(
                config.vocabulary as u64,
                *eos as u64,
                config.ngram.order as usize,
                config.ngram.heads as usize,
                table.clone(),
                0,
                max_rows,
            )?,
            embedding_width: config.ngram.embedding_dim,
            key: linear("key_proj", geometry.flattened_width())?,
            value: linear("value_proj", geometry.hidden_size())?,
            key_norm: norm("norm_key")?,
            query_norm: norm("norm_query")?,
            convolution_norm: norm("norm_conv")?,
            convolution: CausalDepthwiseConvolutionSpec {
                channels: geometry.flattened_width(),
                kernel_size: config.ngram.kernel,
                dilation: config.ngram.order,
                weight: parameter("conv1d")?,
                bias: None,
                activation: ConvolutionActivation::Silu,
            },
            convolution_slot: 0,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Rejects malformed construction before allocating any backend parameters.
    pub fn validate(&self) -> Result<(), Error> {
        let width = self.geometry.flattened_width();
        if self.embedding_width != self.embedding.output_width()
            || self.key.input != self.embedding_width
            || self.key.output != width
            || self.value.input != self.embedding_width
            || self.value.output != self.geometry.hidden_size()
        {
            return Err(Error::backend(
                "lexical projection geometry differs from declared streams and embedding",
            ));
        }
        for linear in [&self.key, &self.value] {
            if linear.bias.is_some() {
                return Err(Error::backend("lexical projections must be bias-free"));
            }
            linear.format.validate_for_weight(&linear.weight)?;
        }
        for norm in [&self.key_norm, &self.query_norm, &self.convolution_norm] {
            norm.validate()?;
            if norm.dimensions != width || norm.groups != Some(self.geometry.streams()) {
                return Err(Error::backend(
                    "lexical normalization must reduce within each residual stream",
                ));
            }
        }
        self.convolution.validate()?;
        if self.convolution.channels != width
            || self.convolution.bias.is_some()
            || self.convolution.activation != ConvolutionActivation::Silu
            || self.convolution.dilation != self.embedding.order()
        {
            return Err(Error::backend(
                "lexical convolution must use n-gram order as dilation, no bias and SiLU",
            ));
        }
        Ok(())
    }
    /// Both mutable histories belong to the injection unit and ordinary snapshots.
    pub fn state_policies(&self) -> Result<Vec<StateTensorPolicy>, Error> {
        self.validate()?;
        let mut result = vec![self.embedding.history_policy()];
        let history = self.convolution.history_len()?;
        if history > 0 {
            result.push(
                StateTensorPolicy::new(
                    StateTensorRole::Convolution {
                        slot: self.convolution_slot,
                    },
                    vec![
                        D::Batch,
                        D::fixed(history).map_err(Error::backend)?,
                        D::fixed(self.geometry.flattened_width()).map_err(Error::backend)?,
                    ],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map_err(Error::backend)?,
            );
        }
        Ok(result)
    }
}

/// Independently stateful lexical execution unit before a decoder sublayer.
/// Complete residuals and original IDs cross its boundaries without reconstruction.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LexicalSublayer<B: GroupedNeuralBackend> {
    injection: LexicalInjection<B>,
}

/// Original input provenance and causal position owned by the shared driver.
pub struct LexicalSublayerInput<'a, T> {
    /// All residual streams, including replaced media embeddings.
    pub residual: &'a T,
    /// Original IDs before any embedding or media transformation.
    pub ids: Option<&'a [u64]>,
    /// Current zero/one padding mask `[batch,tokens]`.
    pub padding: Option<&'a T>,
    /// Absolute prefix position of this independently resumable unit.
    pub offset: i32,
}
impl<B: GroupedNeuralBackend> LexicalSublayer<B> {
    /// Uses the same lexical equation and generic parameter provider as ordinary execution.
    pub fn new(
        spec: LexicalInjectionSpec,
        hash: NGramHashSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            injection: LexicalInjection::new(spec, hash, context)?,
        })
    }
    /// Adds the contribution and advances explicit fixed-state position only after
    /// successful output observation. The shared transaction restores all histories
    /// on subsequent failure or cancellation; provider budgets are not rolled back.
    pub fn forward<P: ParameterProvider<B>, S: RuntimeStateComponents<B>>(
        &mut self,
        input: LexicalSublayerInput<'_, B::Tensor>,
        state: &mut S,
        provider: &mut P,
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, NGramEmbeddingError> {
        self.injection
            .geometry
            .validate_streams(input.residual.shape())?;
        let tokens = input.residual.dim(1);
        if input.offset < 0
            || state.position() != input.offset
            || input.offset.checked_add(tokens).is_none()
        {
            return Err(NGramError::Shape.into());
        }
        // Missing IDs fail before observations, lookup or mutable history publication.
        let ids = input.ids.ok_or(NGramError::MissingTokenIds)?;
        let residual = instrumentation.apply("lexical.input", input.residual.clone())?;
        let contribution = self.injection.forward(
            &residual,
            Some(ids),
            input.padding,
            state,
            provider,
            access,
            context,
        )?;
        let contribution = instrumentation.apply("lexical.write", contribution)?;
        let output =
            instrumentation.apply("lexical.residual", residual.add(&contribution, context)?)?;
        state.advance_fixed(tokens)?;
        Ok(output)
    }
}
/// Architecture-owned PLE computation; providers and state are supplied by the shared driver.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LexicalInjection<B: GroupedNeuralBackend> {
    #[parameter(skip)]
    geometry: ResidualStreamGeometry,
    #[parameter(skip)]
    embedding: NGramEmbedding,
    #[parameter(skip)]
    convolution_slot: u32,
    key: B::Linear,
    value: B::Linear,
    key_norm: B::Normalization,
    query_norm: B::Normalization,
    convolution_norm: B::Normalization,
    convolution: CausalDepthwiseConvolution<B>,
}
impl<B: GroupedNeuralBackend> LexicalInjection<B> {
    /// Builds from architecture-authored identities and explicitly admitted operations.
    pub fn new(
        spec: LexicalInjectionSpec,
        hash: NGramHashSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        let embedding = NGramEmbedding::new(spec.embedding, hash).map_err(Error::backend_source)?;
        B::require_operator_capabilities(
            "qwen4_exp lexical injection",
            C::SIGMOID
                .union(C::ABS)
                .union(C::SIGN)
                .union(C::SQRT)
                .union(C::BROADCAST_TO)
                .union(C::FROM_I32_SLICE)
                .union(C::TO_I32_VEC)
                .union(C::TO_F32_VEC),
        )?;
        Ok(Self {
            geometry: spec.geometry,
            embedding,
            convolution_slot: spec.convolution_slot,
            key: B::linear(spec.key, context)?,
            value: B::linear(spec.value, context)?,
            key_norm: B::normalization(spec.key_norm, context)?,
            query_norm: B::normalization(spec.query_norm, context)?,
            convolution_norm: B::normalization(spec.convolution_norm, context)?,
            convolution: CausalDepthwiseConvolution::new(spec.convolution, context)?,
        })
    }
    /// Returns the additive PLE contribution as `[batch,tokens,streams,hidden]`.
    /// IDs remain the original pre-media-replacement IDs. A zero/one padding mask
    /// has shape `[batch,tokens]`. Masked positions use EOS in lexical history,
    /// matching the reference text ingress, while request IDs remain unchanged.
    /// The shared layer transaction owns rollback of both state components.
    pub fn forward<P: ParameterProvider<B>, S: RuntimeStateComponents<B>>(
        &mut self,
        residual: &B::Tensor,
        ids: Option<&[u64]>,
        padding: Option<&B::Tensor>,
        state: &mut S,
        provider: &mut P,
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, NGramEmbeddingError> {
        self.geometry.validate_streams(residual.shape())?;
        let batch = residual.dim(0);
        let tokens = residual.dim(1);
        if padding.is_some_and(|mask| mask.shape() != [batch, tokens]) {
            return Err(
                Error::backend("lexical padding mask must have [batch,tokens] shape").into(),
            );
        }
        let ids = ids.ok_or(super::ngram::NGramError::MissingTokenIds)?;
        if ids.len() != batch as usize * tokens as usize
            || ids.len() > self.embedding.specification().max_tokens()
        {
            return Err(super::ngram::NGramError::Shape.into());
        }
        let masked_ids = padding
            .map(|mask| -> Result<Vec<u64>, NGramEmbeddingError> {
                let mask = mask.to_f32_vec(context)?;
                if mask.len() != ids.len() || mask.iter().any(|v| *v != 0. && *v != 1.) {
                    return Err(Error::backend(
                        "lexical padding mask must contain only zero or one",
                    )
                    .into());
                }
                Ok(ids
                    .iter()
                    .zip(mask)
                    .map(|(&id, valid)| {
                        if valid == 0. {
                            self.embedding.specification().padding_token()
                        } else {
                            id
                        }
                    })
                    .collect())
            })
            .transpose()?;
        let ids = masked_ids.as_deref().unwrap_or(ids);
        let history = if self.convolution.history_len() > 0 {
            state
                .fixed_component(StateTensorRole::Convolution {
                    slot: self.convolution_slot,
                })
                .map_err(Error::backend)?
                .clone()
        } else {
            None
        };
        let embeddings = self.embedding.forward::<B, _, _>(
            Some(ids),
            batch as usize,
            tokens as usize,
            state,
            provider,
            access,
            context,
        )?;
        let key = self.key.forward(&embeddings, context)?;
        let key = self
            .geometry
            .unflatten(&self.key_norm.forward(&key, context)?, context)?;
        let input = self.geometry.flatten(residual, context)?;
        let query = self
            .geometry
            .unflatten(&self.query_norm.forward(&input, context)?, context)?;
        let gate = B::Tensor::sum_axis(&key.multiply(&query, context)?, -1, true, context)?
            .multiply_scalar(1. / (self.geometry.hidden_size() as f32).sqrt(), context)?;
        let signed_root = gate
            .abs(context)?
            .maximum_scalar(1e-6, context)?
            .sqrt(context)?
            .multiply(&gate.sign(context)?, context)?;
        let gates = B::sigmoid(signed_root, context)?.broadcast_to(residual.shape(), context)?;
        let values = self
            .geometry
            .expand(&self.value.forward(&embeddings, context)?, context)?;
        let mut values = self
            .geometry
            .flatten(&values.multiply(&gates, context)?, context)?;
        let mut normalized = self.convolution_norm.forward(&values, context)?;
        if let Some(mask) = padding {
            let mask = mask
                .expand_dims(-1, context)?
                .broadcast_to(values.shape(), context)?;
            values = values.multiply(&mask, context)?;
            normalized = normalized.multiply(&mask, context)?;
        }
        let convolved = self
            .convolution
            .forward(&normalized, history.as_ref(), context)?;
        let output = self
            .geometry
            .unflatten(&values.add(&convolved.output, context)?, context)?;
        if self.convolution.history_len() > 0 {
            *state
                .fixed_component(StateTensorRole::Convolution {
                    slot: self.convolution_slot,
                })
                .map_err(Error::backend)? = convolved.history;
        }
        Ok(output)
    }
}
