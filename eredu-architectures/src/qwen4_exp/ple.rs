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
    CausalDepthwiseConvolution, DeclareLogicalParameters, Error, GroupedNeuralBackend,
    LinearOperator, LogicalParameterDeclaration, NeuralOperatorCapabilities as C,
    NormalizationOperator, Tensor,
};
use eredu_runtime::{ParameterBankAccess, ParameterProvider, RuntimeStateComponents};

/// Shared logical lexical parameter declarations.
pub mod specs;
pub use specs::LexicalParameters;

/// Exact neural slots, row provider and independent history for one lexical invocation.
#[derive(Debug, Clone)]
pub struct LexicalInjectionSpec {
    /// Owned affine, normalization and convolution declarations.
    pub parameters: LexicalParameters,
    /// Hashing, bounded table lookup and retained original token IDs.
    pub embedding: NGramEmbeddingSpec,
    /// History identity independent of the recurrent mixer's convolution.
    pub convolution_slot: u32,
}
impl LexicalInjectionSpec {
    /// Attaches header-only row geometry to the declared lexical parameters.
    /// Hash literals bind separately when constructing the executable injection.
    pub fn from_parameters(
        config: &Config,
        parameters: LexicalParameters,
        table: &eredu_runtime::RowLookupSpec,
        max_rows: usize,
    ) -> Result<Self, NGramEmbeddingError> {
        let eos = config.eos.first().ok_or(NGramError::Constants)?;
        let spec = Self {
            parameters,
            embedding: NGramEmbeddingSpec::new(
                config.vocabulary as u64,
                *eos as u64,
                config.ngram.order as usize,
                config.ngram.heads as usize,
                table.clone(),
                0,
                max_rows,
            )?,
            convolution_slot: 0,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Validates parameter geometry and the independent row/history contracts before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        self.parameters.validate()?;
        if self.parameters.key.input != self.embedding.output_width()
            || self.parameters.convolution.dilation != self.embedding.order()
        {
            return Err(Error::backend(
                "lexical parameters differ from declared n-gram feature/history geometry",
            ));
        }
        Ok(())
    }
    /// Both mutable histories belong to the injection unit and ordinary snapshots.
    pub fn state_policies(&self) -> Result<Vec<StateTensorPolicy>, Error> {
        self.validate()?;
        let mut result = vec![self.embedding.history_policy()];
        let history = self.parameters.convolution.history_len()?;
        if history > 0 {
            result.push(
                StateTensorPolicy::new(
                    StateTensorRole::Convolution {
                        slot: self.convolution_slot,
                    },
                    vec![
                        D::Batch,
                        D::fixed(history).map_err(Error::backend)?,
                        D::fixed(self.parameters.geometry.flattened_width())
                            .map_err(Error::backend)?,
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

impl DeclareLogicalParameters for LexicalInjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.parameters.logical_parameters()
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
            geometry: spec.parameters.geometry,
            embedding,
            convolution_slot: spec.convolution_slot,
            key: B::linear(spec.parameters.key, context)?,
            value: B::linear(spec.parameters.value, context)?,
            key_norm: B::normalization(spec.parameters.key_norm, context)?,
            query_norm: B::normalization(spec.parameters.query_norm, context)?,
            convolution_norm: B::normalization(spec.parameters.convolution_norm, context)?,
            convolution: CausalDepthwiseConvolution::new(spec.parameters.convolution, context)?,
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
