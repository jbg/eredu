//! Selected affine groups whose pointwise activation precedes mixture reduction.

use crate::{Error, GroupSelection, GroupedProjectionSpec, Parameterized, Tensor};
use std::ops::Range;

/// Activation applied independently to each selected projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupedLinearActivation {
    /// Ordinary affine projection.
    Identity,
    /// SiLU applied after the complete input-width dot product.
    Silu,
}

/// Exact construction geometry for a packed selected-linear bank.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedLinearSpec {
    groups: i32,
    input: i32,
    global_output: i32,
    output: Range<i32>,
    activation: GroupedLinearActivation,
    reduction: crate::GroupReduction,
    projection: GroupedProjectionSpec,
}

impl GroupedLinearSpec {
    /// Declares a complete bank with packed weights `[groups, output, input]`.
    pub fn new(
        groups: i32,
        input: i32,
        output: i32,
        activation: GroupedLinearActivation,
        projection: GroupedProjectionSpec,
    ) -> Result<Self, Error> {
        if groups <= 0 {
            return Err(Error::backend(
                "a global grouped-linear bank must contain experts",
            ));
        }
        let spec = Self {
            groups,
            input,
            global_output: output,
            output: 0..output,
            activation,
            reduction: crate::GroupReduction::Sum,
            projection,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Projects storage without changing bank geometry, activation or output ownership.
    pub fn with_projection_format(
        mut self,
        format: crate::LinearFormatSpec,
    ) -> Result<Self, Error> {
        self.projection = self.projection.with_format(format)?;
        self.validate()?;
        Ok(self)
    }
    /// Selects output rows while retaining the complete input width. The
    /// activated result is an owned output shard and must never be all-summed.
    pub fn partition_output(mut self, output: Range<i32>) -> Result<Self, Error> {
        self.output = output;
        self.validate()?;
        Ok(self)
    }
    /// Substitutes a local bank cardinality, including an empty expert owner.
    pub fn with_group_count(mut self, groups: i32) -> Result<Self, Error> {
        self.groups = groups;
        self.validate()?;
        Ok(self)
    }
    /// Number of independently addressable groups.
    pub const fn group_count(&self) -> i32 {
        self.groups
    }
    /// Complete input width, including in tensor-parallel execution.
    pub const fn input_dimensions(&self) -> i32 {
        self.input
    }
    /// Locally owned projected output width.
    pub fn output_dimensions(&self) -> i32 {
        self.output.end - self.output.start
    }
    /// Global output width before head partitioning.
    pub const fn global_output_dimensions(&self) -> i32 {
        self.global_output
    }
    /// Owned output rows in global projection coordinates.
    pub fn output_range(&self) -> Range<i32> {
        self.output.clone()
    }
    /// Pointwise activation before route weighting and summation.
    pub const fn activation(&self) -> GroupedLinearActivation {
        self.activation
    }
    /// Selects the weighted-output accumulation equation.
    pub const fn with_reduction(mut self, reduction: crate::GroupReduction) -> Self {
        self.reduction = reduction;
        self
    }
    /// Weighted-output accumulation equation.
    pub const fn reduction(&self) -> crate::GroupReduction {
        self.reduction
    }
    /// Exact parameter identities and physical encoding.
    pub const fn projection(&self) -> &GroupedProjectionSpec {
        &self.projection
    }
    /// Validates dimensions and output ownership without native resources.
    pub fn validate(&self) -> Result<(), Error> {
        if self.groups < 0
            || self.input <= 0
            || self.global_output <= 0
            || self.output.start < 0
            || self.output.end > self.global_output
            || self.output.start >= self.output.end
        {
            return Err(Error::backend(
                "invalid grouped-linear bank or output partition",
            ));
        }
        self.projection.validate()
    }
}

/// Selected grouped projection. Activation is applied to every complete expert
/// projection before weighting; applying it to the mixed result is incorrect.
pub trait GroupedLinearOperator<T: Tensor>: Clone + std::fmt::Debug + Parameterized<T> {
    /// Retained exact geometry and equation policy.
    fn spec(&self) -> &GroupedLinearSpec;
    /// Evaluates `sum(coefficients * activation(projected_rows))`.
    fn forward_grouped(
        &mut self,
        input: &T,
        selections: &GroupSelection<T>,
        context: &T::Context,
    ) -> Result<T, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_keeps_complete_input_and_cannot_claim_a_reducible_partial() {
        let projection = GroupedProjectionSpec::new(
            crate::ParameterSpec::trainable("bank.weight").unwrap(),
            None,
            crate::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
        )
        .unwrap();
        let global =
            GroupedLinearSpec::new(64, 2560, 1024, GroupedLinearActivation::Silu, projection)
                .unwrap();
        let local = global
            .clone()
            .partition_output(256..512)
            .unwrap()
            .with_group_count(16)
            .unwrap();
        assert_eq!(local.input_dimensions(), 2560);
        assert_eq!(local.output_dimensions(), 256);
        assert_eq!(local.output_range(), 256..512);
        assert_eq!(local.group_count(), 16);
        assert!(global.clone().partition_output(1024..1025).is_err());
        assert!(global.clone().partition_output(5..5).is_err());
        assert_eq!(global.clone().with_group_count(0).unwrap().group_count(), 0);
        assert!(global.with_group_count(-1).is_err());
    }
}

/// Runs a packed block-diagonal projection and selects each group's own output
/// slice. The underlying projection owns input observation and precision.
pub fn project_group_diagonal<T: crate::Tensor>(
    linear: &mut impl crate::LinearOperator<T>,
    input: &T,
    groups: i32,
    output_per_group: i32,
    context: &T::Context,
    observer: Option<&mut dyn crate::ProjectionInputObserver<T>>,
) -> Result<T, Error> {
    use crate::Index;
    if groups <= 0 || output_per_group <= 0 || input.shape().len() != 4 || input.dim(1) != groups {
        return Err(Error::backend("invalid grouped projection geometry"));
    }
    let width = groups
        .checked_mul(output_per_group)
        .ok_or_else(|| Error::backend("grouped projection width overflows"))?;
    let projected = linear.forward_with_input_observer(input, context, observer)?;
    if projected.shape().last() != Some(&width) {
        return Err(Error::backend("invalid grouped projection output width"));
    }
    let pieces = (0..groups)
        .map(|group| {
            let start = group * output_per_group;
            projected.index(
                &[
                    Index::Full,
                    Index::Range(group, group + 1),
                    Index::Full,
                    Index::Range(start, start + output_per_group),
                ],
                context,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    T::concatenate(&pieces, 1, context)
}
