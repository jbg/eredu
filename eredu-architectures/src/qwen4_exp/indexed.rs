//! Released selected-position attention with gated residual transport and shared state.
use super::{
    config::Config,
    qsa::{
        CachedQsaSelector, QsaError, QsaExecutionLimits, QsaIndexerSpec, QsaSelectionInput,
        QsaSelectionSpec, QsaStreamStateSpec,
    },
    residual::mixer_spec,
};
use crate::decoder::{Attention, AttentionInput, ComponentInstrumentation};
use eredu_nn::{
    residual_streams::{GatedResidual, GatedResidualSpec},
    AttentionArithmetic, AttentionCache, Error, LinearFormatSpec, LinearSpec, NeuralBackend,
    NeuralOperatorCapabilities as C, NormalizationConstructionSpec, NormalizationScale,
    ParameterSpec, RotaryPosition, RotarySpec, Tensor, TensorElementType,
};
use eredu_runtime::{RuntimeAppendStreams, RuntimeStateComponents};

/// Complete architecture-owned parameters, state and workspace for an indexed sublayer.
#[derive(Debug, Clone)]
pub struct IndexedSublayerSpec {
    /// Replicated small residual mixer, including the injection projection.
    pub residual: GatedResidualSpec,
    /// Query heads, optionally localized after global artifact validation.
    pub heads: i32,
    /// K/V heads; may be replicated for tensor partitions smaller than query groups.
    pub kv_heads: i32,
    /// Per-head attention width.
    pub head_dim: i32,
    /// Exact query (with gate), key, value and output projections.
    pub projections: [LinearSpec; 4],
    /// Per-head query normalization.
    pub query_norm: NormalizationConstructionSpec,
    /// Per-head key normalization.
    pub key_norm: NormalizationConstructionSpec,
    /// Rotary policy shared with the indexer.
    pub rotary: RotarySpec,
    /// Query/key index projections and tiled selection geometry.
    pub indexer: QsaIndexerSpec,
    /// Exact fixed and named-stream state ownership.
    pub state: QsaStreamStateSpec,
    /// Finite coordinator workspace and invocation bounds.
    pub limits: QsaExecutionLimits,
    /// Explicit attention rounding, independent of storage or selected native kernel.
    pub arithmetic: AttentionArithmetic,
}
impl IndexedSublayerSpec {
    /// Constructs the released target namespace and equations; prediction callers
    /// supply their architecture-selected rotary policy before construction.
    #[allow(clippy::too_many_arguments)]
    pub fn from_config(
        config: &Config,
        root: &str,
        tile_blocks: i32,
        selection_workspace: u64,
        limits: QsaExecutionLimits,
        element: TensorElementType,
        rotary_element: TensorElementType,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, QsaError> {
        let a = &config.attention;
        let query = a.heads.checked_mul(a.head_dim).ok_or(QsaError::Geometry)?;
        let kv = a
            .kv_heads
            .checked_mul(a.head_dim)
            .ok_or(QsaError::Geometry)?;
        let index = a
            .index_heads
            .checked_add(1)
            .and_then(|heads| heads.checked_mul(a.index_head_dim))
            .ok_or(QsaError::Geometry)?;
        let residual = mixer_spec(
            config,
            &format!("{root}.attn_hyper_connection"),
            true,
            &mut format,
        )?;
        let root = format!("{root}.self_attn");
        let parameter = |suffix: &str| {
            ParameterSpec::trainable(format!("{root}.{suffix}")).map_err(Error::backend)
        };
        let mut projection = |suffix: &str, input, output, bias| -> Result<LinearSpec, Error> {
            let name = format!("{root}.{suffix}.weight");
            Ok(LinearSpec {
                input,
                output,
                weight: parameter(&format!("{suffix}.weight"))?,
                bias: if bias {
                    Some(parameter(&format!("{suffix}.bias"))?)
                } else {
                    None
                },
                format: format(&name)?,
            })
        };
        let projections = [
            projection(
                "q_proj",
                config.hidden_size,
                query.checked_mul(2).ok_or(QsaError::Geometry)?,
                a.bias,
            )?,
            projection("k_proj", config.hidden_size, kv, a.bias)?,
            projection("v_proj", config.hidden_size, kv, a.bias)?,
            projection("o_proj", query, config.hidden_size, false)?,
        ];
        let norm = |suffix: &str, dimensions| -> Result<NormalizationConstructionSpec, Error> {
            Ok(NormalizationConstructionSpec {
                dimensions,
                groups: None,
                epsilon: config.norm_epsilon,
                scale: NormalizationScale::LearnedOffset {
                    weight: parameter(&format!("{suffix}.weight"))?,
                    offset: 1.,
                },
            })
        };
        let selection = QsaSelectionSpec {
            heads: a.index_heads,
            dimensions: a.index_head_dim,
            ratio: a.ratio,
            token_budget: a.budget,
            tile_blocks,
            workspace_bytes: selection_workspace,
        };
        let spec = Self {
            residual,
            heads: a.heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            projections,
            query_norm: norm("q_norm", a.head_dim)?,
            key_norm: norm("k_norm", a.head_dim)?,
            rotary: a.rotary.clone(),
            indexer: QsaIndexerSpec {
                selection,
                query_key: projection("indexer.index_qk_proj", config.hidden_size, index, false)?,
                query_norm: norm("indexer.q_layernorm", a.index_head_dim)?,
                key_norm: norm("indexer.k_layernorm", a.index_head_dim)?,
                rotary: a.rotary.clone(),
            },
            state: QsaStreamStateSpec::new(
                selection,
                a.rotary.dimensions,
                element,
                rotary_element,
            )?,
            limits,
            arithmetic: AttentionArithmetic::InputScores,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Validates composition geometry without touching any native device or parameter.
    pub fn validate(&self) -> Result<(), QsaError> {
        self.residual.validate()?;
        self.state.validate_indexer(&self.indexer)?;
        if self.heads <= 0
            || self.kv_heads <= 0
            || self.heads % self.kv_heads != 0
            || self.head_dim <= 0
            || self.rotary.dimensions <= 0
            || self.rotary.dimensions > self.head_dim
            || self.rotary != self.indexer.rotary
            || self.residual.injection.is_none()
        {
            return Err(QsaError::Geometry);
        }
        let query = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or(QsaError::Geometry)?;
        let kv = self
            .kv_heads
            .checked_mul(self.head_dim)
            .ok_or(QsaError::Geometry)?;
        let h = self.residual.geometry.hidden_size();
        if self.indexer.query_key.input != h {
            return Err(QsaError::Geometry);
        }
        for (p, i, o) in [
            (
                &self.projections[0],
                h,
                query.checked_mul(2).ok_or(QsaError::Geometry)?,
            ),
            (&self.projections[1], h, kv),
            (&self.projections[2], h, kv),
            (&self.projections[3], query, h),
        ] {
            if p.input != i || p.output != o {
                return Err(QsaError::Geometry);
            }
            p.format.validate_for_weight(&p.weight)?;
        }
        for n in [&self.query_norm, &self.key_norm] {
            n.validate()?;
            if n.dimensions != self.head_dim || n.groups.is_some() {
                return Err(QsaError::Geometry);
            }
        }
        Ok(())
    }
}

/// QSA and full K/V state execute in the same surrounding unit transaction.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct IndexedSublayer<B: NeuralBackend> {
    residual: GatedResidual<B>,
    selector: CachedQsaSelector<B>,
    attention: Attention<B>,
}
/// Complete residual input plus original causal visibility and rotary provenance.
pub struct IndexedSublayerInput<'a, T> {
    /// `[batch,tokens,streams,hidden]`, including every residual stream.
    pub residual: &'a T,
    /// Optional row-major key visibility for the current chunk.
    pub visible: Option<&'a [bool]>,
    /// Current rotary values, or ordinary positions beginning at the cache offset.
    pub rotary: Option<RotaryPosition<'a, T>>,
}
impl<B: NeuralBackend> IndexedSublayer<B> {
    /// Allocates only declared parameters; prepared construction supplies mutable state.
    pub fn new(
        spec: IndexedSublayerSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, QsaError> {
        spec.validate()?;
        spec.limits.validate::<B::Tensor>(spec.state)?;
        B::require_operator_capabilities(
            "qwen4_exp indexed sublayer",
            C::INDEXED_ATTENTION
                .union(C::BROADCAST_TO)
                .union(C::SIGMOID),
        )?;
        let selector = CachedQsaSelector::new(spec.indexer, spec.state, spec.limits, context)?;
        let [q, k, v, o] = spec.projections;
        let mut attention = Attention::from_gated_parts(
            spec.heads,
            spec.kv_heads,
            spec.head_dim,
            B::linear(q, context)?,
            B::linear(k, context)?,
            B::linear(v, context)?,
            B::linear(o, context)?,
            Some(B::normalization(spec.query_norm, context)?),
            Some(B::normalization(spec.key_norm, context)?),
            Some(B::rotary(spec.rotary, context)?),
            None,
        )?;
        attention.arithmetic = spec.arithmetic;
        Ok(Self {
            residual: GatedResidual::new(spec.residual, context)?,
            selector,
            attention,
        })
    }
    /// Uses cached selected-position reads, never a dense causal or sequence mask.
    /// `parallel` selects the shared output reduction after localized projections;
    /// the indexer and residual mixer are initially replicated across those ranks.
    pub fn forward<S>(
        &mut self,
        input: IndexedSublayerInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, QsaError>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        let offset = AttentionCache::offset(state);
        if offset != RuntimeStateComponents::<B>::position(state) {
            return Err(QsaError::Geometry);
        }
        let residual = self.residual.forward(input.residual, context)?;
        let hidden = instrumentation.apply("attention.input", residual.mixed.clone())?;
        let rotary = input.rotary.unwrap_or(RotaryPosition::Offset(offset));
        let selected = self.selector.select(
            QsaSelectionInput {
                hidden: &hidden,
                offset,
                visible: input.visible,
                rotary,
            },
            state,
            context,
        )?;
        instrumentation.observe("attention.selected_positions", &selected)?;
        let output = self.attention.forward_instrumented(
            AttentionInput {
                hidden: &hidden,
                selected_positions: Some(&selected),
                mask: None,
                cache: Some(state),
                allow_sliding_prefill: false,
                rotary_position: Some(rotary),
            },
            parallel,
            context,
            instrumentation,
        )?;
        let output = instrumentation.apply("attention.write", output)?;
        let output = instrumentation.apply("attention.output", output)?;
        Ok(instrumentation.apply("attention.residual", residual.inject(&output, context)?)?)
    }
}
