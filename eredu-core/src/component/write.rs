//! Bounded joins for direct and grouped factorized component writes.
use super::{ComponentGroup, ComponentId};
use crate::parameters::{LoadedParameter, ParameterDiscovery, ParameterRegion};
use serde::{Deserialize, Serialize};

/// A grouped linear stage before the component group's final output projection.
/// For `C` components, group `g` consumes `C / groups` consecutive components
/// using rows `g * rank..(g + 1) * rank` of this matrix. Concatenated group
/// outputs enter the final matrix. Biases are separate terms, never repeated
/// once per scalar component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentGroupedWriteProjection {
    /// Actual effective matrix, logically `[groups * rank, C / groups]`.
    pub weight: String,
    /// Architecture group containing that matrix.
    pub parameter_group: String,
    /// Shared physical parameter identity across logical invocations.
    pub shared_weight: String,
    /// Optional vector added after the grouped projection, `[groups * rank]`.
    pub bias: Option<String>,
    /// Number of disjoint consecutive component groups.
    pub groups: usize,
    /// Output rows produced by each group.
    pub rank: usize,
    /// Actual first-matrix multiplication input after its selected precision
    /// transform, in `[batch, group, sequence, C / groups]` geometry. Keeping
    /// the native grouped shape permits admission before diagnostic copies.
    pub input: String,
    /// Actual grouped projection output before the final matrix's input transform.
    pub output: String,
    /// Actual multiplication input of the final matrix, after its selected
    /// precision/quantization input transform. A fixed matrix product alone does
    /// not describe any rounding difference between these two observations.
    pub final_input: String,
}

/// A checked region in an actual loaded parameter. This grants no native-work
/// authority; the ordinary parameter query/projection/edit API still admits it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentParameterSelection<'a> {
    /// Loaded shape, dtype, alias identity and operation support.
    pub parameter: &'a LoadedParameter,
    /// Unsqueezed exact region, suitable for an ordinary parameter operation.
    pub region: ParameterRegion,
}

/// Parameter factors producing one scalar component's write direction.
/// A direct write has one output column and no input factor. A grouped write
/// is the matrix product `output * input`, with shapes `[hidden, rank]` and
/// `[rank, 1]`. A selected readout direction can be contracted through these
/// factors with two bounded native projections without exporting their product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentWriteColumn<'a> {
    /// Columns of the final write matrix.
    pub output: ComponentParameterSelection<'a>,
    /// Column of the selected group's input matrix, when factorized.
    pub input: Option<ComponentParameterSelection<'a>>,
}

/// Invalid component-to-parameter join, distinct from a measured zero.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ComponentWriteError {
    /// Scalar identity is outside the declared component group.
    #[error("invalid component identity")]
    Identity,
    /// Malformed/overflowed factors or incompatible loaded parameter geometry.
    #[error("component write geometry differs from loaded parameters")]
    Geometry,
    /// The declared effective parameter is absent from loaded discovery.
    #[error("component write parameter {0:?} is absent from loaded discovery")]
    MissingParameter(String),
}

fn parameter<'a>(
    loaded: &'a ParameterDiscovery,
    name: &str,
) -> Result<&'a LoadedParameter, ComponentWriteError> {
    let mut matches = loaded
        .parameters
        .iter()
        .filter(|parameter| parameter.id == name);
    let parameter = matches
        .next()
        .ok_or_else(|| ComponentWriteError::MissingParameter(name.into()))?;
    if matches.next().is_some() {
        return Err(ComponentWriteError::Geometry);
    }
    Ok(parameter)
}

impl ComponentGroup {
    /// Joins a scalar's direct column or exact grouped factors to actual loaded
    /// geometry. No checkpoint names are parsed, no full product is built, and
    /// loaded operation support remains attached to each returned selection.
    pub fn write_column<'a>(
        &self,
        id: &ComponentId,
        loaded: &'a ParameterDiscovery,
    ) -> Result<ComponentWriteColumn<'a>, ComponentWriteError> {
        if !self.contains(id) || self.count == 0 {
            return Err(ComponentWriteError::Identity);
        }
        let output = parameter(loaded, &self.write_weight)?;
        let [hidden, columns] = output.shape.as_slice() else {
            return Err(ComponentWriteError::Geometry);
        };
        if *hidden == 0 {
            return Err(ComponentWriteError::Geometry);
        }
        let convert = |n| u64::try_from(n).map_err(|_| ComponentWriteError::Geometry);
        let (start, width, input) = match &self.write_input_projection {
            None if *columns == convert(self.count)? => (id.index, 1, None),
            None => return Err(ComponentWriteError::Geometry),
            Some(stage) => {
                if stage.groups == 0 || stage.rank == 0 || self.count % stage.groups != 0 {
                    return Err(ComponentWriteError::Geometry);
                }
                let rows = stage
                    .groups
                    .checked_mul(stage.rank)
                    .ok_or(ComponentWriteError::Geometry)?;
                let per_group = self.count / stage.groups;
                let start = (id.index / per_group)
                    .checked_mul(stage.rank)
                    .ok_or(ComponentWriteError::Geometry)?;
                let input = parameter(loaded, &stage.weight)?;
                if *columns != convert(rows)?
                    || input.shape != [convert(rows)?, convert(per_group)?]
                {
                    return Err(ComponentWriteError::Geometry);
                }
                let region = ParameterRegion {
                    starts: vec![convert(start)?, convert(id.index % per_group)?],
                    shape: vec![convert(stage.rank)?, 1],
                };
                region
                    .validate(&input.shape)
                    .map_err(|_| ComponentWriteError::Geometry)?;
                (
                    start,
                    stage.rank,
                    Some(ComponentParameterSelection {
                        parameter: input,
                        region,
                    }),
                )
            }
        };
        let region = ParameterRegion {
            starts: vec![0, convert(start)?],
            shape: vec![*hidden, convert(width)?],
        };
        region
            .validate(&output.shape)
            .map_err(|_| ComponentWriteError::Geometry)?;
        Ok(ComponentWriteColumn {
            output: ComponentParameterSelection {
                parameter: output,
                region,
            },
            input,
        })
    }
}

#[cfg(test)]
mod tests;
