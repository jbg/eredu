//! Routed affine reads that feed a scalar component's aggregation.
use super::*;
use crate::parameters::ParameterDiscovery;

/// A selected mixture of per-expert activated projections. Routing decisions
/// depend on the current normalized input; no single fixed affine row replaces
/// this equation. `routing` joins the architecture's selection/coefficient policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentRoutedRead {
    /// Owning mixture node, including routing semantics and parameter groups.
    pub node_id: String,
    /// Route observation identity for selected experts, scores and coefficients.
    pub routing: String,
    /// Global expert count, independent of selected route cardinality.
    pub expert_count: usize,
    /// Width of each expert's normalized input.
    pub input_width: usize,
    /// Exact per-expert parameter rows and their semantic role.
    pub read: RoutedComponentRead,
    /// Nonlinearity applied separately to each expert before coefficient weighting.
    pub activation: ComponentNonlinearity,
    /// Router projection weight. Correction bias affects selection only, as
    /// declared by the routing policy; it is not an expert output bias.
    pub selector_weight: String,
    /// Optional selection-only correction bias.
    pub selector_correction_bias: Option<String>,
}

impl ComponentRoutedRead {
    /// Joins a component's per-expert read row to actual loaded effective geometry.
    /// This is descriptive selection, not query or editing authority.
    pub fn read_weight<'a>(
        &self,
        group: &ComponentGroup,
        component: &ComponentId,
        expert: usize,
        loaded: &'a ParameterDiscovery,
    ) -> Result<RoutedComponentSelection<'a>, RoutedComponentError> {
        if !group.contains(component) || !group.routed_reads.iter().any(|read| read == self) {
            return Err(RoutedComponentError::Geometry);
        }
        let rows = self
            .read
            .rows
            .row_range(component.index)
            .ok_or(RoutedComponentError::Geometry)?;
        super::routed::select_parameter(
            self.expert_count,
            &self.read.weight,
            expert,
            &[self.read.projection_rows, self.input_width],
            &[rows.start, 0],
            &[rows.len(), self.input_width],
            loaded,
        )?
        .ok_or(RoutedComponentError::Geometry)
    }

    /// Joins an optional expert output bias with the same channel mapping.
    pub fn read_bias<'a>(
        &self,
        group: &ComponentGroup,
        component: &ComponentId,
        expert: usize,
        loaded: &'a ParameterDiscovery,
    ) -> Result<Option<RoutedComponentSelection<'a>>, RoutedComponentError> {
        if !group.contains(component)
            || expert >= self.expert_count
            || !group.routed_reads.iter().any(|read| read == self)
        {
            return Err(RoutedComponentError::Geometry);
        }
        let Some(bias) = &self.read.bias else {
            return Ok(None);
        };
        let rows = self
            .read
            .rows
            .row_range(component.index)
            .ok_or(RoutedComponentError::Geometry)?;
        super::routed::select_parameter(
            self.expert_count,
            bias,
            expert,
            &[self.read.projection_rows],
            &[rows.start],
            &[rows.len()],
            loaded,
        )
    }
}
