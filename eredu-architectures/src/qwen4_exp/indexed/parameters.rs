//! Parameter ownership and geometry for selected-position attention, independent of invocation bounds.
use super::super::{config::Config, qsa::QsaError};
use eredu_nn::{
    AttentionHeadGeometry, AttentionQueryLayout, DeclareLogicalParameters, Error, LinearFormatSpec,
    LinearSpec, LogicalParameterDeclaration, NormalizationConstructionSpec, NormalizationScale,
    ParameterSpec, SelfAttentionProjectionSpec,
};

/// Parameter declarations for target and prediction attention invocations.
#[derive(Debug, Clone)]
pub struct IndexedParameters {
    /// Gated Q/K/V/output slots and selected or replicated attention head axes.
    pub projection: SelfAttentionProjectionSpec,
    /// Replicated selection-query heads and one shared selection key.
    pub index_geometry: AttentionHeadGeometry,
    /// Per-head attention query scale.
    pub query_norm: NormalizationConstructionSpec,
    /// Per-head attention key scale.
    pub key_norm: NormalizationConstructionSpec,
    /// Component-major index queries followed by one shared index key.
    pub index_projection: LinearSpec,
    /// Per-head index query scale.
    pub index_query_norm: NormalizationConstructionSpec,
    /// Shared index key scale.
    pub index_key_norm: NormalizationConstructionSpec,
}
impl IndexedParameters {
    /// Released query rows interleave each head's query and attended-value gate.
    pub const QUERY_LAYOUT: AttentionQueryLayout = AttentionQueryLayout::InterleavedGate;
    /// Declares slots without selecting state, scratch limits, rotary positions or a backend.
    pub fn from_config(
        config: &Config,
        root: &str,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, QsaError> {
        let a = &config.attention;
        if a.index_kv_heads != 1 {
            return Err(QsaError::Geometry);
        }
        let geometry = AttentionHeadGeometry::new(a.heads, a.kv_heads, a.head_dim, a.head_dim)?;
        let index_geometry = AttentionHeadGeometry::new(
            a.index_heads,
            a.index_kv_heads,
            a.index_head_dim,
            a.index_head_dim,
        )?;
        let index = index_geometry
            .query_width()
            .checked_add(index_geometry.key_width())
            .ok_or(QsaError::Geometry)?;
        let root = format!("{root}.self_attn");
        let parameter = |suffix: &str| {
            let mut parameter =
                ParameterSpec::trainable(format!("{root}.{suffix}")).map_err(Error::backend)?;
            parameter.group = Some(root.clone());
            Ok::<_, Error>(parameter)
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
        let attention = SelfAttentionProjectionSpec::new(
            geometry,
            Self::QUERY_LAYOUT,
            projection(
                "q_proj",
                config.hidden_size,
                Self::QUERY_LAYOUT.projected_width(geometry)?,
                a.bias,
            )?,
            projection("k_proj", config.hidden_size, geometry.key_width(), a.bias)?,
            projection("v_proj", config.hidden_size, geometry.value_width(), a.bias)?,
            projection(
                "o_proj",
                geometry.attended_width(),
                config.hidden_size,
                false,
            )?,
        )?;
        let norm = |suffix: &str, dimensions| -> Result<NormalizationConstructionSpec, Error> {
            Ok(NormalizationConstructionSpec {
                reduction: eredu_nn::NormalizationReduction::Native,
                dimensions,
                groups: None,
                epsilon: config.norm_epsilon,
                scale: NormalizationScale::LearnedOffset {
                    weight: parameter(&format!("{suffix}.weight"))?,
                    offset: 1.,
                },
            })
        };
        let spec = Self {
            projection: attention,
            index_geometry,
            query_norm: norm("q_norm", geometry.key_dimensions())?,
            key_norm: norm("k_norm", geometry.key_dimensions())?,
            index_projection: projection(
                "indexer.index_qk_proj",
                config.hidden_size,
                index,
                false,
            )?,
            index_query_norm: norm("indexer.q_layernorm", index_geometry.key_dimensions())?,
            index_key_norm: norm("indexer.k_layernorm", index_geometry.key_dimensions())?,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Checks compatible geometry, exact identities and authoritative sharing.
    pub fn validate(&self) -> Result<(), QsaError> {
        self.validate_geometry()?;
        self.declarations()?;
        Ok(())
    }
    fn declarations(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.index_projection.logical_parameters()?;
        parameters.extend(self.index_query_norm.logical_parameters()?);
        parameters.extend(self.index_key_norm.logical_parameters()?);
        parameters.extend(self.projection.logical_parameters()?);
        parameters.extend(self.query_norm.logical_parameters()?);
        parameters.extend(self.key_norm.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    fn validate_geometry(&self) -> Result<(), QsaError> {
        self.projection.validate_geometry()?;
        let geometry = self.projection.geometry;
        let index_geometry = self.index_geometry;
        let index = index_geometry
            .query_width()
            .checked_add(index_geometry.key_width())
            .ok_or(QsaError::Geometry)?;
        let hidden = self.projection.query.input;
        if self.projection.query_layout != Self::QUERY_LAYOUT
            || geometry.key_dimensions() != geometry.value_dimensions()
            || index_geometry.key_value_heads() != 1
            || index_geometry.key_dimensions() != index_geometry.value_dimensions()
            || self.projection.output.output != hidden
            || self.index_projection.input != hidden
            || self.index_projection.output != index
            || self.index_projection.bias.is_some()
            || self.projection.output.bias.is_some()
        {
            return Err(QsaError::Geometry);
        }
        for (norm, dimensions) in [
            (&self.query_norm, geometry.key_dimensions()),
            (&self.key_norm, geometry.key_dimensions()),
            (&self.index_query_norm, index_geometry.key_dimensions()),
            (&self.index_key_norm, index_geometry.key_dimensions()),
        ] {
            norm.validate()?;
            if norm.dimensions != dimensions || norm.groups.is_some() {
                return Err(QsaError::Geometry);
            }
        }
        Ok(())
    }
}
impl DeclareLogicalParameters for IndexedParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry().map_err(Error::backend_source)?;
        self.declarations()
    }
}
