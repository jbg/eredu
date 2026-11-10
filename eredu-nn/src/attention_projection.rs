//! Affine slots and head geometry for self-attention parameter declarations.

use crate::{
    ClippedLinearSpec, DeclareLogicalParameters, Error, LinearClippingSpec, LinearSpec,
    LogicalParameterDeclaration,
};

/// Validated grouped-query head axes, with independent key and value widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionHeadGeometry {
    query_heads: i32,
    key_value_heads: i32,
    key_dimensions: i32,
    value_dimensions: i32,
}
impl AttentionHeadGeometry {
    /// Checks complete head groups and every projected width before construction.
    pub fn new(
        query_heads: i32,
        key_value_heads: i32,
        key_dimensions: i32,
        value_dimensions: i32,
    ) -> Result<Self, Error> {
        if query_heads <= 0
            || key_value_heads <= 0
            || key_dimensions <= 0
            || value_dimensions <= 0
            || query_heads % key_value_heads != 0
            || [
                (query_heads, key_dimensions),
                (key_value_heads, key_dimensions),
                (key_value_heads, value_dimensions),
                (query_heads, value_dimensions),
            ]
            .into_iter()
            .any(|(heads, width)| heads.checked_mul(width).is_none())
        {
            return Err(Error::backend(
                "invalid attention head or projection geometry",
            ));
        }
        Ok(Self {
            query_heads,
            key_value_heads,
            key_dimensions,
            value_dimensions,
        })
    }
    /// Selects complete K/V head groups and their associated query heads.
    /// The invocation keeps its own state; this does not select parameter storage.
    pub fn partition_key_value_heads(self, range: std::ops::Range<usize>) -> Result<Self, Error> {
        if range.start >= range.end || range.end > self.key_value_heads as usize {
            return Err(Error::backend(
                "attention head interval is outside the declared groups",
            ));
        }
        let heads = i32::try_from(range.len()).map_err(Error::backend)?;
        Self::new(
            heads * (self.query_heads / self.key_value_heads),
            heads,
            self.key_dimensions,
            self.value_dimensions,
        )
    }

    /// Query heads before grouping against the retained K/V heads.
    pub const fn query_heads(self) -> i32 {
        self.query_heads
    }
    /// Independent key/value heads, including replicated local K/V heads.
    pub const fn key_value_heads(self) -> i32 {
        self.key_value_heads
    }
    /// Key and query feature width per head.
    pub const fn key_dimensions(self) -> i32 {
        self.key_dimensions
    }
    /// Value and attended feature width per query head.
    pub const fn value_dimensions(self) -> i32 {
        self.value_dimensions
    }
    /// Complete query projection rows.
    pub const fn query_width(self) -> i32 {
        self.query_heads * self.key_dimensions
    }
    /// Complete key projection rows.
    pub const fn key_width(self) -> i32 {
        self.key_value_heads * self.key_dimensions
    }
    /// Complete value projection rows.
    pub const fn value_width(self) -> i32 {
        self.key_value_heads * self.value_dimensions
    }
    /// Complete input columns of the attention output projection.
    pub const fn attended_width(self) -> i32 {
        self.query_heads * self.value_dimensions
    }
}

/// Logical row order within each projected query head.
///
/// This describes parameter geometry only; architectures declare the gate's
/// activation, execution equation and observation contracts separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionQueryLayout {
    /// Query features without an output gate in the same affine.
    QueryOnly,
    /// Each head stores query features followed by attended-value gate features.
    InterleavedGate,
}
impl AttentionQueryLayout {
    /// Complete row stride of one head in the query affine.
    pub fn head_width(self, geometry: AttentionHeadGeometry) -> Result<i32, Error> {
        match self {
            Self::QueryOnly => Ok(geometry.key_dimensions()),
            Self::InterleavedGate => geometry
                .key_dimensions()
                .checked_add(geometry.value_dimensions())
                .ok_or_else(|| Error::backend("gated attention head width overflow")),
        }
    }
    /// Complete output rows of the query affine, checked before allocation.
    pub fn projected_width(self, geometry: AttentionHeadGeometry) -> Result<i32, Error> {
        self.head_width(geometry)?
            .checked_mul(geometry.query_heads())
            .ok_or_else(|| Error::backend("attention query projection width overflow"))
    }
}

/// Exact Q/K/V and output affine slots with authoritative head geometry.
///
/// Q/K/V consume the same logical input. The output may have an independent
/// width; enclosing residual equations validate their own boundary. Architecture
/// declarations retain score arithmetic, normalization, position/state policy
/// and observation semantics alongside these parameters.
#[derive(Debug, Clone)]
pub struct SelfAttentionProjectionSpec {
    /// Complete query and K/V head axes.
    pub geometry: AttentionHeadGeometry,
    /// Query-only or per-head interleaved query/gate row order.
    pub query_layout: AttentionQueryLayout,
    /// Query projection, retaining biases, sharing and encoding companions.
    pub query: LinearSpec,
    /// Key projection.
    pub key: LinearSpec,
    /// Value projection.
    pub value: LinearSpec,
    /// Projection of concatenated attended value heads.
    pub output: LinearSpec,
}
impl SelfAttentionProjectionSpec {
    /// Validates intrinsic geometry and identities; enclosing scopes resolve external sharing.
    pub fn new(
        geometry: AttentionHeadGeometry,
        query_layout: AttentionQueryLayout,
        query: LinearSpec,
        key: LinearSpec,
        value: LinearSpec,
        output: LinearSpec,
    ) -> Result<Self, Error> {
        let spec = Self {
            geometry,
            query_layout,
            query,
            key,
            value,
            output,
        };
        crate::validate_logical_parameter_scope(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    /// Affine slots in query, key, value, output order.
    pub fn projections(&self) -> [&LinearSpec; 4] {
        [&self.query, &self.key, &self.value, &self.output]
    }
    /// Checks affine connectivity against the retained head axes.
    pub fn validate_geometry(&self) -> Result<(), Error> {
        let query_width = self.query_layout.projected_width(self.geometry)?;
        if self.query.input != self.key.input
            || self.query.input != self.value.input
            || self.query.output != query_width
            || self.key.output != self.geometry.key_width()
            || self.value.output != self.geometry.value_width()
            || self.output.input != self.geometry.attended_width()
        {
            return Err(Error::backend(
                "incompatible self-attention projection geometry",
            ));
        }
        Ok(())
    }
    /// Resolves the complete parameter scope before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for SelfAttentionProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = Vec::new();
        for projection in self.projections() {
            parameters.extend(projection.logical_parameters()?);
        }
        Ok(parameters)
    }
}

/// Shared attention affine scope with independently optional learned clipping.
///
/// Bounds extend the named affine slots without repeating their geometry,
/// encoding, identity or sharing. Architecture declarations retain attention,
/// normalization and residual equations independently of the learned clamps.
#[derive(Debug, Clone)]
pub struct ClippedSelfAttentionProjectionSpec {
    /// Authoritative affines, head geometry and query row layout.
    pub projection: SelfAttentionProjectionSpec,
    /// Input/output scalar bounds around the query affine.
    pub query_bounds: Option<LinearClippingSpec>,
    /// Input/output scalar bounds around the key affine.
    pub key_bounds: Option<LinearClippingSpec>,
    /// Input/output scalar bounds around the value affine.
    pub value_bounds: Option<LinearClippingSpec>,
    /// Input/output scalar bounds around the output affine.
    pub output_bounds: Option<LinearClippingSpec>,
}
impl ClippedSelfAttentionProjectionSpec {
    /// Validates geometry and local identities; enclosing scopes resolve external sharing.
    pub fn new(
        geometry: AttentionHeadGeometry,
        query_layout: AttentionQueryLayout,
        query: ClippedLinearSpec,
        key: ClippedLinearSpec,
        value: ClippedLinearSpec,
        output: ClippedLinearSpec,
    ) -> Result<Self, Error> {
        let spec = Self {
            projection: SelfAttentionProjectionSpec::new(
                geometry,
                query_layout,
                query.linear,
                key.linear,
                value.linear,
                output.linear,
            )?,
            query_bounds: query.bounds,
            key_bounds: key.bounds,
            value_bounds: value.bounds,
            output_bounds: output.bounds,
        };
        crate::validate_logical_parameter_scope(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    /// Named affine and clipping pairs in query, key, value, output order.
    pub fn clipped_projections(&self) -> [(&LinearSpec, Option<&LinearClippingSpec>); 4] {
        [
            (&self.projection.query, self.query_bounds.as_ref()),
            (&self.projection.key, self.key_bounds.as_ref()),
            (&self.projection.value, self.value_bounds.as_ref()),
            (&self.projection.output, self.output_bounds.as_ref()),
        ]
    }
    /// Resolves the complete clipped parameter scope before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
    /// Binds each affine to its learned bounds in query, key, value, output order.
    pub fn into_projections(self) -> Result<[ClippedLinearSpec; 4], Error> {
        self.validate()?;
        Ok([
            ClippedLinearSpec {
                linear: self.projection.query,
                bounds: self.query_bounds,
            },
            ClippedLinearSpec {
                linear: self.projection.key,
                bounds: self.key_bounds,
            },
            ClippedLinearSpec {
                linear: self.projection.value,
                bounds: self.value_bounds,
            },
            ClippedLinearSpec {
                linear: self.projection.output,
                bounds: self.output_bounds,
            },
        ])
    }
}
impl DeclareLogicalParameters for ClippedSelfAttentionProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.projection.validate_geometry()?;
        let mut parameters = Vec::new();
        for (projection, bounds) in self.clipped_projections() {
            parameters.extend(projection.logical_parameters()?);
            if let Some(bounds) = bounds {
                parameters.extend(bounds.logical_parameters()?);
            }
        }
        Ok(parameters)
    }
}
