//! Architecture-authored shapes for inactive participants in routed pipeline waves.
use eredu_nn::TensorElementType;

/// One axis of a tensor sum, resolved from the current invocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RoutedTensorDimension {
    /// A fixed positive extent.
    Fixed(i32),
    /// Current sequence lanes.
    Batch,
    /// Current tokens per lane.
    Sequence,
    /// Flattened token rows, multiplied by this positive factor.
    Tokens(i32),
}

/// One sum which active and inactive tensor groups must submit in the same order.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RoutedTensorReduction {
    /// Provider failure status over all participants in the pipeline wave.
    Status,
    /// Ordinary floating hidden output with shape `[batch, sequence, hidden_width]`.
    Hidden,
    /// An exact architecture-declared shape and scalar representation.
    Tensor {
        /// Invocation-relative axes.
        shape: Vec<RoutedTensorDimension>,
        /// Exact scalar representation, independent of the pipeline wire dtype.
        dtype: TensorElementType,
    },
}

impl RoutedTensorReduction {
    pub(super) fn resolve(
        &self,
        batch: i32,
        sequence: i32,
        hidden_width: usize,
    ) -> Result<(Vec<i32>, Option<TensorElementType>), eredu_nn::Error> {
        let invalid = || eredu_nn::Error::backend("invalid routed tensor collective geometry");
        if batch <= 0 || sequence <= 0 {
            return Err(invalid());
        }
        let (shape, dtype) = match self {
            Self::Status => (vec![1], Some(TensorElementType::I32)),
            Self::Hidden => (
                vec![
                    batch,
                    sequence,
                    i32::try_from(hidden_width).map_err(|_| invalid())?,
                ],
                None,
            ),
            Self::Tensor { shape, dtype } => (
                shape
                    .iter()
                    .map(|axis| match axis {
                        RoutedTensorDimension::Fixed(size) => Ok(*size),
                        RoutedTensorDimension::Batch => Ok(batch),
                        RoutedTensorDimension::Sequence => Ok(sequence),
                        RoutedTensorDimension::Tokens(factor) => batch
                            .checked_mul(sequence)
                            .and_then(|rows| rows.checked_mul(*factor))
                            .ok_or_else(invalid),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                Some(*dtype),
            ),
        };
        if shape.is_empty() || shape.iter().any(|size| *size <= 0) {
            return Err(invalid());
        }
        Ok((shape, dtype))
    }
}

/// Ordered tensor sums before and after a routed provider invocation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoutedTensorReductions {
    /// Sums preceding the provider exchange, or all sums of an ordinary unit.
    pub before: Vec<RoutedTensorReduction>,
    /// Sums following the provider exchange.
    pub after: Vec<RoutedTensorReduction>,
}
impl RoutedTensorReductions {
    /// Repeated ordinary hidden-output reductions.
    pub fn hidden(before: usize, after: usize) -> Self {
        Self {
            before: vec![RoutedTensorReduction::Hidden; before],
            after: vec![RoutedTensorReduction::Hidden; after],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn status_and_flattened_rows_preserve_exact_shapes_and_integer_dtype() {
        let status = RoutedTensorReduction::Status;
        let rows = RoutedTensorReduction::Tensor {
            shape: vec![
                RoutedTensorDimension::Tokens(3),
                RoutedTensorDimension::Fixed(8),
            ],
            dtype: TensorElementType::F32,
        };
        assert_eq!(
            status.resolve(2, 5, 64).unwrap(),
            (vec![1], Some(TensorElementType::I32))
        );
        assert_eq!(
            rows.resolve(2, 5, 64).unwrap(),
            (vec![30, 8], Some(TensorElementType::F32))
        );
        assert!(rows.resolve(i32::MAX, 2, 64).is_err());
        assert!(rows.resolve(0, 1, 64).is_err());
    }
}
