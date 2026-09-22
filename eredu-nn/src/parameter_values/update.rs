//! One rectangular effective-parameter update and its completion boundaries.
use crate::{
    workspace::{WorkspaceContext, WorkspaceFloatingType, WorkspaceMetadataError, WorkspaceTensor},
    Error, Tensor,
};
use eredu_core::parameters::{ParameterRegion, ParameterUpdate};

/// The selected numerical and metadata realizations follow this same update
/// sequence. The caller owns source, destination and host-readback admission.
pub trait ParameterUpdateMechanism {
    /// A native value or its descriptive workspace counterpart.
    type Value;
    /// The selected mechanism's typed refusal or execution failure.
    type Error;
    /// Construct the edit values through the selected host source.
    fn values(&self) -> Result<Self::Value, Self::Error>;
    /// Borrow the selected region through its actual slice operation.
    fn select(&self, source: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Convert the selected region to the edit arithmetic representation.
    fn cast_f32(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// Add the selected source region and supplied edit values.
    fn add(&self, left: &Self::Value, right: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Convert the edited region to the source parameter's representation.
    fn cast_like(
        &self,
        value: Self::Value,
        source: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    /// Complete the actual cast result and validate every value. A metadata
    /// realization records that completion; it never asserts numerical facts.
    fn finite_completed(&self, value: &Self::Value) -> Result<bool, Self::Error>;
    /// Create the resulting parameter with only the selected region replaced.
    fn replace(
        &self,
        source: &Self::Value,
        value: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    /// Establish completion of the resulting parameter before publication.
    fn complete(&self, value: &Self::Value) -> Result<(), Self::Error>;
}

/// Returns no destination when the selected dtype overflows. The original
/// parameter remains borrowed and unchanged on rejection or any failure.
pub fn update_parameter<M: ParameterUpdateMechanism>(
    mechanism: M,
    source: &M::Value,
    update: &ParameterUpdate,
) -> Result<Option<M::Value>, M::Error> {
    let values = mechanism.values()?;
    let values = match update {
        ParameterUpdate::Replace { .. } => values,
        ParameterUpdate::Add { .. } => {
            let selected = mechanism.cast_f32(mechanism.select(source)?)?;
            mechanism.add(&selected, &values)?
        }
    };
    let values = mechanism.cast_like(values, source)?;
    if !mechanism.finite_completed(&values)? {
        return Ok(None);
    }
    let output = mechanism.replace(source, &values)?;
    mechanism.complete(&output)?;
    Ok(Some(output))
}

/// Metadata realization using the retained source's actual representation and
/// the backend's descriptive eager host constructor.
pub struct WorkspaceParameterUpdate<'a> {
    region: &'a ParameterRegion,
    context: &'a WorkspaceContext,
    construct: fn(&[i32], &WorkspaceContext) -> Result<WorkspaceTensor, Error>,
    starts: Vec<i32>,
    ends: Vec<i32>,
    strides: Vec<i32>,
}
impl<'a> WorkspaceParameterUpdate<'a> {
    /// Retain the actual region and backend host constructor under the metadata payer.
    pub fn new(
        region: &'a ParameterRegion,
        context: &'a WorkspaceContext,
        construct: fn(&[i32], &WorkspaceContext) -> Result<WorkspaceTensor, Error>,
    ) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(Self, Result<Self, Error>)>())?;
        if region.starts.len() != region.shape.len() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut starts = context.metadata_vec(region.shape.len())?;
        let mut ends = context.metadata_vec(region.shape.len())?;
        let mut strides = context.metadata_vec(region.shape.len())?;
        for (&start, &count) in region.starts.iter().zip(&region.shape) {
            starts.push(i32::try_from(start).map_err(|_| WorkspaceMetadataError::Overflow)?);
            ends.push(
                i32::try_from(
                    start
                        .checked_add(count)
                        .ok_or(WorkspaceMetadataError::Overflow)?,
                )
                .map_err(|_| WorkspaceMetadataError::Overflow)?,
            );
            strides.push(1);
        }
        Ok(Self {
            region,
            context,
            construct,
            starts,
            ends,
            strides,
        })
    }
}
impl ParameterUpdateMechanism for WorkspaceParameterUpdate<'_> {
    type Value = WorkspaceTensor;
    type Error = Error;
    fn values(&self) -> Result<WorkspaceTensor, Error> {
        let mut shape = self.context.metadata_vec(self.region.shape.len())?;
        for &n in &self.region.shape {
            shape.push(i32::try_from(n).map_err(|_| WorkspaceMetadataError::Overflow)?);
        }
        (self.construct)(&shape, self.context)
    }
    fn select(&self, source: &WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        source.static_slice(&self.starts, &self.ends, &self.strides, self.context)
    }
    fn cast_f32(&self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        value.cast_floating(WorkspaceFloatingType::Float32, self.context)
    }
    fn add(
        &self,
        left: &WorkspaceTensor,
        right: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        left.add(right, self.context)
    }
    fn cast_like(
        &self,
        value: WorkspaceTensor,
        source: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        let dtype = source
            .layout()
            .representation()
            .ok_or(WorkspaceMetadataError::Unqualified)?
            .dtype();
        value.cast_floating(dtype, self.context)
    }
    fn finite_completed(&self, value: &WorkspaceTensor) -> Result<bool, Error> {
        let checked = value
            .cast_floating(WorkspaceFloatingType::Float32, self.context)?
            .contiguous(self.context)?;
        self.context.complete_values(&[&checked])?;
        Ok(true)
    }
    fn replace(
        &self,
        source: &WorkspaceTensor,
        value: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        source.update_static_slice(value, &self.starts, &self.ends, &self.strides, self.context)
    }
    fn complete(&self, value: &WorkspaceTensor) -> Result<(), Error> {
        self.context.complete_values(&[value])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    struct Matrix {
        rows: usize,
        columns: usize,
        data: Vec<f32>,
    }
    struct Worker<'a> {
        region: &'a ParameterRegion,
        update: &'a ParameterUpdate,
        completions: &'a Cell<usize>,
        writes: &'a Cell<usize>,
    }
    impl ParameterUpdateMechanism for Worker<'_> {
        type Value = Matrix;
        type Error = ();
        fn values(&self) -> Result<Matrix, ()> {
            Ok(Matrix {
                rows: self.region.shape[0] as usize,
                columns: self.region.shape[1] as usize,
                data: self.update.values().to_vec(),
            })
        }
        fn select(&self, source: &Matrix) -> Result<Matrix, ()> {
            let rows = self.region.shape[0] as usize;
            let columns = self.region.shape[1] as usize;
            let mut data = Vec::with_capacity(rows * columns);
            for row in 0..rows {
                for column in 0..columns {
                    data.push(
                        source.data[(row + self.region.starts[0] as usize) * source.columns
                            + column
                            + self.region.starts[1] as usize],
                    );
                }
            }
            Ok(Matrix {
                rows,
                columns,
                data,
            })
        }
        fn cast_f32(&self, value: Matrix) -> Result<Matrix, ()> {
            Ok(value)
        }
        fn add(&self, left: &Matrix, right: &Matrix) -> Result<Matrix, ()> {
            Ok(Matrix {
                rows: left.rows,
                columns: left.columns,
                data: left
                    .data
                    .iter()
                    .zip(&right.data)
                    .map(|(a, b)| a + b)
                    .collect(),
            })
        }
        fn cast_like(&self, value: Matrix, _: &Matrix) -> Result<Matrix, ()> {
            Ok(value)
        }
        fn finite_completed(&self, value: &Matrix) -> Result<bool, ()> {
            self.completions.set(self.completions.get() + 1);
            Ok(value.data.iter().all(|v| v.is_finite()))
        }
        fn replace(&self, source: &Matrix, value: &Matrix) -> Result<Matrix, ()> {
            self.writes.set(self.writes.get() + 1);
            let mut data = source.data.clone();
            for row in 0..value.rows {
                for column in 0..value.columns {
                    data[(row + self.region.starts[0] as usize) * source.columns
                        + column
                        + self.region.starts[1] as usize] =
                        value.data[row * value.columns + column];
                }
            }
            Ok(Matrix {
                rows: source.rows,
                columns: source.columns,
                data,
            })
        }
        fn complete(&self, _: &Matrix) -> Result<(), ()> {
            self.completions.set(self.completions.get() + 1);
            Ok(())
        }
    }
    #[test]
    fn additive_and_replacement_regions_preserve_unselected_values_and_source() {
        let original = Matrix {
            rows: 2,
            columns: 3,
            data: vec![1., 2., 3., 4., 5., 6.],
        };
        let region = ParameterRegion {
            starts: vec![0, 1],
            shape: vec![2, 2],
        };
        for (update, expected) in [
            (
                ParameterUpdate::Add {
                    values: vec![2., -1., -3., 4.],
                },
                vec![1., 4., 2., 4., 2., 10.],
            ),
            (
                ParameterUpdate::Replace {
                    values: vec![2., -1., -3., 4.],
                },
                vec![1., 2., -1., 4., -3., 4.],
            ),
        ] {
            let completions = Cell::new(0);
            let writes = Cell::new(0);
            let result = update_parameter(
                Worker {
                    region: &region,
                    update: &update,
                    completions: &completions,
                    writes: &writes,
                },
                &original,
                &update,
            )
            .unwrap()
            .unwrap();
            assert_eq!(result.data, expected);
            assert_eq!(original.data, [1., 2., 3., 4., 5., 6.]);
            assert_eq!(completions.get(), 2);
            assert_eq!(writes.get(), 1);
        }
    }
    #[test]
    fn overflow_is_completed_and_rejected_before_destination_or_source_change() {
        let source = Matrix {
            rows: 1,
            columns: 2,
            data: vec![f32::MAX, 7.],
        };
        let region = ParameterRegion {
            starts: vec![0, 0],
            shape: vec![1, 1],
        };
        let update = ParameterUpdate::Add {
            values: vec![f32::MAX],
        };
        let completions = Cell::new(0);
        let writes = Cell::new(0);
        assert!(update_parameter(
            Worker {
                region: &region,
                update: &update,
                completions: &completions,
                writes: &writes
            },
            &source,
            &update
        )
        .unwrap()
        .is_none());
        assert_eq!(source.data, [f32::MAX, 7.]);
        assert_eq!(completions.get(), 1);
        assert_eq!(writes.get(), 0);
    }
}
