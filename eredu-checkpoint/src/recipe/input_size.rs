//! Logical recipe input size; allocator and traversal overhead are caller policy.
use super::{DerivedWeightRecipe, RecipeDtype, TensorSelection};
use std::mem::{size_of, size_of_val};

fn selection(value: &TensorSelection) -> usize {
    match value {
        TensorSelection::Indices { indices, .. } => size_of_val(indices.as_slice()),
        TensorSelection::Contiguous { shape, .. } => size_of_val(shape.as_slice()),
        TensorSelection::Full | TensorSelection::Range { .. } => 0,
    }
}
fn dtype(value: &RecipeDtype) -> usize {
    match value {
        RecipeDtype::Other(name) => name.len(),
        _ => 0,
    }
}
impl DerivedWeightRecipe {
    /// Logical declaration bytes, including every node, name, index and shape.
    /// Borrowed traversal allocates nothing. This is input to configurable host
    /// estimates, not a measurement of retained storage or an allocator ceiling.
    pub fn metadata_input_bytes(&self) -> Option<usize> {
        let extra = match self {
            Self::Source {
                key,
                selection: value,
            } => key.len().checked_add(selection(value))?,
            Self::Select {
                input,
                selection: value,
            } => input
                .metadata_input_bytes()?
                .checked_add(selection(value))?,
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                inputs.iter().try_fold(0usize, |n, input| {
                    n.checked_add(input.metadata_input_bytes()?)
                })?
            }
            Self::Reshape { input, shape } => input
                .metadata_input_bytes()?
                .checked_add(size_of_val(shape.as_slice()))?,
            Self::Transpose { input, axes } => input
                .metadata_input_bytes()?
                .checked_add(size_of_val(axes.as_slice()))?,
            Self::Cast {
                input,
                dtype: value,
            } => input.metadata_input_bytes()?.checked_add(dtype(value))?,
            Self::View {
                input,
                dtype: value,
                shape,
            } => input
                .metadata_input_bytes()?
                .checked_add(dtype(value))?
                .checked_add(size_of_val(shape.as_slice()))?,
            Self::NegLog { input } | Self::SubtractOne { input } => input.metadata_input_bytes()?,
        };
        size_of::<Self>().checked_add(extra)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_input_size_includes_nested_indices_shapes_axes_and_dtype_names() {
        let source = DerivedWeightRecipe::source("weight", TensorSelection::Full);
        let selected = DerivedWeightRecipe::Select {
            input: Box::new(source.clone()),
            selection: TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 7, 11],
            },
        };
        let recipe = DerivedWeightRecipe::View {
            input: Box::new(DerivedWeightRecipe::Transpose {
                input: Box::new(DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: vec![selected, source.clone()],
                }),
                axes: vec![1, 0],
            }),
            dtype: RecipeDtype::Other("extension".into()),
            shape: vec![2, 3],
        };
        // Six declarations, two source names, seven index/axis/shape values and
        // one extension dtype. Payload size and inferred tensor values are absent.
        assert_eq!(
            recipe.metadata_input_bytes(),
            Some(6 * size_of::<DerivedWeightRecipe>() + 12 + 7 * size_of::<usize>() + 9)
        );
        assert_eq!(
            recipe.clone().metadata_input_bytes(),
            recipe.metadata_input_bytes()
        );
        assert!(recipe.metadata_input_bytes().unwrap() > source.metadata_input_bytes().unwrap());
    }
}
