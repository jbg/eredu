//! Released selected-position attention with gated residual transport and shared state.
use super::{
    config::Config,
    qsa::{
        CachedQsaSelector, QsaError, QsaExecutionLimits, QsaIndexerSpec, QsaSelectionInput,
        QsaSelectionSpec, QsaStreamStateSpec,
    },
};
use crate::decoder::{Attention, AttentionInput, ComponentInstrumentation};
use eredu_nn::{
    residual_streams::{GatedResidual, GatedResidualSpec},
    AttentionArithmetic, AttentionCache, DeclareLogicalParameters, Error, NeuralBackend,
    NeuralOperatorCapabilities as C, RotaryPosition, RotarySpec, Tensor, TensorElementType,
};
use eredu_runtime::{RuntimeAppendStreams, RuntimeStateComponents};

mod parameters;
pub use parameters::IndexedParameters;

/// Complete architecture-owned parameters, state and workspace for an indexed sublayer.
#[derive(Debug, Clone)]
pub struct IndexedSublayerSpec {
    /// Replicated small residual mixer, including the injection projection.
    pub residual: GatedResidualSpec,
    /// Validated geometry and ownership shared with cold consumers.
    pub parameters: IndexedParameters,
    /// Rotary policy shared with the indexer.
    pub rotary: RotarySpec,
    /// Tiled selection geometry and retained scratch limits.
    pub selection: QsaSelectionSpec,
    /// Exact fixed and named-stream state ownership.
    pub state: QsaStreamStateSpec,
    /// Finite coordinator workspace and invocation bounds.
    pub limits: QsaExecutionLimits,
    /// Explicit attention rounding, independent of storage or selected native kernel.
    pub arithmetic: AttentionArithmetic,
}
impl IndexedSublayerSpec {
    /// Attaches state and workspace to declared parameters; prediction callers
    /// supply their architecture-selected rotary policy before construction.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parameters(
        config: &Config,
        residual: GatedResidualSpec,
        parameters: IndexedParameters,
        tile_blocks: i32,
        selection_workspace: u64,
        limits: QsaExecutionLimits,
        element: TensorElementType,
        rotary_element: TensorElementType,
    ) -> Result<Self, QsaError> {
        let a = &config.attention;
        let selection = QsaSelectionSpec {
            heads: parameters.index_geometry.query_heads(),
            dimensions: parameters.index_geometry.key_dimensions(),
            ratio: a.ratio,
            token_budget: a.budget,
            tile_blocks,
            workspace_bytes: selection_workspace,
        };
        let spec = Self {
            residual,
            parameters,
            rotary: a.rotary,
            selection,
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
    /// Projects the indexer invocation from its authoritative logical slots.
    pub fn indexer_spec(&self) -> QsaIndexerSpec {
        QsaIndexerSpec {
            selection: self.selection,
            query_key: self.parameters.index_projection.clone(),
            query_norm: self.parameters.index_query_norm.clone(),
            key_norm: self.parameters.index_key_norm.clone(),
            rotary: self.rotary,
        }
    }
    /// Validates composition geometry without touching any native device or parameter.
    pub fn validate(&self) -> Result<(), QsaError> {
        self.validate_geometry()?;
        self.declarations()?;
        Ok(())
    }
    fn declarations(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.residual.logical_parameters()?;
        parameters.extend(self.parameters.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    fn validate_geometry(&self) -> Result<(), QsaError> {
        self.residual.validate()?;
        self.parameters.validate()?;
        self.state.validate_indexer(&self.indexer_spec())?;
        if self.rotary.dimensions <= 0
            || self.rotary.dimensions > self.parameters.projection.geometry.key_dimensions()
            || self.residual.injection.is_none()
            || self.parameters.projection.query.input != self.residual.geometry.hidden_size()
            || self.selection.heads != self.parameters.index_geometry.query_heads()
            || self.selection.dimensions != self.parameters.index_geometry.key_dimensions()
        {
            return Err(QsaError::Geometry);
        }
        Ok(())
    }
}

impl DeclareLogicalParameters for IndexedSublayerSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry().map_err(Error::backend_source)?;
        self.declarations()
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
        let selector =
            CachedQsaSelector::new(spec.indexer_spec(), spec.state, spec.limits, context)?;
        let parameters = spec.parameters;
        let geometry = parameters.projection.geometry;
        let eredu_nn::SelfAttentionProjectionSpec {
            query: q,
            key: k,
            value: v,
            output: o,
            ..
        } = parameters.projection;
        let mut attention = Attention::from_gated_parts(
            geometry.query_heads(),
            geometry.key_value_heads(),
            geometry.key_dimensions(),
            B::linear(q, context)?,
            B::linear(k, context)?,
            B::linear(v, context)?,
            B::linear(o, context)?,
            Some(B::normalization(parameters.query_norm, context)?),
            Some(B::normalization(parameters.key_norm, context)?),
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
