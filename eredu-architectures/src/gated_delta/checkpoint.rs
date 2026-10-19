//! Portable head-layout recipes. Parameter names and encoding choices belong to callers.
use eredu_checkpoint::{recipe::DerivedWeightRecipe as Recipe, store::TensorSelection};

/// Inverts tiled value-head storage into grouped key/value heads. `prefix` units
/// on `axis` are unchanged (e.g. leading Q/K rows); every remaining head must be
/// represented by complete storage units. Callers ensure encoded blocks do not
/// straddle head boundaries before applying this byte-preserving permutation.
pub(crate) fn grouped_value_heads(
    input: Recipe,
    shape: &[usize],
    axis: usize,
    prefix: usize,
    key_heads: usize,
    value_heads: usize,
) -> Result<Recipe, String> {
    if key_heads == 0
        || value_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || axis >= shape.len()
        || prefix >= shape[axis]
    {
        return Err("invalid grouped value-head storage geometry".into());
    }
    // These groupings are identities even when a packed storage unit spans
    // several heads. Do not impose scalar head alignment on unchanged bytes.
    if key_heads == 1 || key_heads == value_heads {
        return Ok(input);
    }
    if !(shape[axis] - prefix).is_multiple_of(value_heads) {
        return Err("invalid grouped value-head storage geometry".into());
    }
    let mut tail_shape = shape.to_vec();
    tail_shape[axis] -= prefix;
    let mut expanded = tail_shape.clone();
    expanded.splice(
        axis..=axis,
        [
            value_heads / key_heads,
            key_heads,
            tail_shape[axis] / value_heads,
        ],
    );
    let mut axes: Vec<_> = (0..expanded.len()).collect();
    axes.swap(axis, axis + 1);
    let tail = if prefix == 0 {
        input.clone()
    } else {
        Recipe::Select {
            input: Box::new(input.clone()),
            selection: TensorSelection::Range {
                axis,
                start: prefix,
                end: shape[axis],
            },
        }
    };
    let tail = Recipe::Reshape {
        input: Box::new(Recipe::Transpose {
            input: Box::new(Recipe::Reshape {
                input: Box::new(tail),
                shape: expanded,
            }),
            axes,
        }),
        shape: tail_shape,
    };
    Ok(if prefix == 0 {
        tail
    } else {
        Recipe::Concatenate {
            axis,
            inputs: vec![
                Recipe::Select {
                    input: Box::new(input),
                    selection: TensorSelection::Range {
                        axis,
                        start: 0,
                        end: prefix,
                    },
                },
                tail,
            ],
        }
    })
}
