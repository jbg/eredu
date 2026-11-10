//! Learned routing and independently localized gated-product parameter banks.

use crate::{
    DeclareLogicalParameters, Error, GroupedGatedProductSpec, LogicalParameterDeclaration,
    TopKGroupSelectorSpec,
};

/// Complete learned selector and expert-bank parameter scope.
///
/// Input widths agree; the bank may have an independent output width.
/// The selector retains its global routing axis when the bank is localized by
/// placement. Invocation identity, residency ownership and mutable state belong
/// to the architecture and execution plan, independently of these parameters.
#[derive(Debug, Clone)]
pub struct RoutedGatedProductSpec {
    /// Global learned routing projection and selection equation.
    pub selector: TopKGroupSelectorSpec,
    /// Provider-owned bank with global or prepared rank-local geometry.
    pub experts: GroupedGatedProductSpec,
}

impl RoutedGatedProductSpec {
    /// Builds and validates the complete scope before backend allocation.
    pub fn new(
        selector: TopKGroupSelectorSpec,
        experts: GroupedGatedProductSpec,
    ) -> Result<Self, Error> {
        let spec = Self { selector, experts };
        spec.validate()?;
        Ok(spec)
    }

    /// Rejects incompatible geometry, conflicting identities and unresolved sharing.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }

    /// Selects local experts and channels while retaining the global selector.
    /// A local bank may contain fewer experts than the global top-k.
    pub fn with_group_geometry(mut self, groups: i32, intermediate: i32) -> Result<Self, Error> {
        if groups > self.experts.group_count()
            || intermediate > self.experts.intermediate_dimensions()
        {
            return Err(Error::backend(
                "local expert geometry exceeds its declared bank",
            ));
        }
        self.experts = self.experts.with_group_geometry(groups, intermediate)?;
        self.validate()?;
        Ok(self)
    }
}

impl DeclareLogicalParameters for RoutedGatedProductSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.selector.input_dimensions() != self.experts.input_dimensions()
            || self.experts.group_count() > self.selector.selection().group_count()
        {
            return Err(Error::backend("incompatible routed gated-product geometry"));
        }
        let mut parameters = self.selector.logical_parameters()?;
        parameters.extend(self.experts.logical_parameters()?);
        Ok(parameters)
    }
}
