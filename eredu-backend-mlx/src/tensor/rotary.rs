//! One native rotary equation, with generated or borrowed host frequencies.
//! Prepared original execution consumes its admitted native row bank.
mod original;
mod rows;
pub(crate) use original::PreparedRotaryProfile;
const INLINE_VALUES: usize = 4;
type ArrayOwners = SmallVec<[Array; INLINE_VALUES]>;

use super::*;
use eredu_nn::multimodal::{fill_rotary_axis_frequencies, MultiAxisRotarySpecRef};

#[cfg(test)]
thread_local! {
    static PREPARED_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PREPARED_TWO_AXIS_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn reset_prepared_calls() {
    PREPARED_CALLS.set(0);
    PREPARED_TWO_AXIS_CALLS.set(0);
}
#[cfg(test)]
pub(crate) fn prepared_calls() -> usize {
    PREPARED_CALLS.get()
}
#[cfg(test)]
pub(crate) fn prepared_two_axis_calls() -> usize {
    PREPARED_TWO_AXIS_CALLS.get()
}

pub(super) fn execute(
    position_ids: &MlxTensor,
    spec: MultiAxisRotarySpecRef<'_>,
    prepared_frequencies: Option<&[f32]>,
    context: &Stream,
) -> Result<(MlxTensor, MlxTensor), Error> {
    let original = original::Execution::current()?;
    let backend = |result| original.native(result);
    let dimensions = spec.dimensions().map_err(|cause| original.source(cause))?;
    let position_shape = position_ids.shape();
    let axes = spec.axes.len() as i32;
    if position_shape.len() < 2 || position_shape.last().copied() != Some(axes) {
        return Err(original.geometry_error());
    }
    #[cfg(test)]
    if prepared_frequencies.is_some() {
        PREPARED_CALLS.set(PREPARED_CALLS.get() + 1);
        if axes == 2 {
            PREPARED_TWO_AXIS_CALLS.set(PREPARED_TWO_AXIS_CALLS.get() + 1);
        }
    }
    let rows =
        position_shape[..position_shape.len() - 1]
            .iter()
            .try_fold(1_i32, |rows, dimension| {
                rows.checked_mul(*dimension)
                    .ok_or_else(|| original.overflow_error())
            })?;
    original.require_profile(spec, position_shape.len(), prepared_frequencies.is_some())?;
    let positions = backend(position_ids.as_array().reshape(&[rows, axes], context))?;
    let signed = positions.dtype() == Dtype::Int32;
    let wide = signed || positions.dtype() == Dtype::Uint32;
    let positions = if wide {
        backend(positions.as_dtype(Dtype::Int64, context))?
    } else {
        positions
    };
    let column = |axis: usize| -> Result<Array, Error> {
        let positions = backend(positions.try_index_device((.., axis as i32), context))?;
        let offset = if wide {
            Array::try_from_slice(&[i64::from(spec.axes[axis].position_offset)], &[])
                .map_err(|cause| original.source(cause))?
        } else {
            Array::try_from_int(spec.axes[axis].position_offset)
                .map_err(|cause| original.source(cause))?
        };
        let positions = backend(positions.add(offset, context))?;
        // The neutral signed-position reference uses saturating addition.
        // Widen before adding so native I32 overflow cannot wrap padding
        // or a large positive coordinate into another semantic position.
        let positions = if signed {
            backend(safemlx::ops::clip(
                &positions,
                (
                    Array::try_from_slice(&[i64::from(i32::MIN)], &[])
                        .map_err(|cause| original.source(cause))?,
                    Array::try_from_slice(&[i64::from(i32::MAX)], &[])
                        .map_err(|cause| original.source(cause))?,
                ),
                context,
            ))?
        } else {
            positions
        };
        let minimum = if wide {
            Array::try_from_slice(&[i64::from(spec.minimum_position)], &[])
                .map_err(|cause| original.source(cause))?
        } else {
            Array::try_from_int(spec.minimum_position).map_err(|cause| original.source(cause))?
        };
        backend(maximum(positions, minimum, context))
    };
    let mut frequency_at = 0;
    let angles = if spec.layout != MultiAxisRotaryLayout::RoundRobinSections {
        let mut axis_angles = rows::Rows::new(spec.axes.len(), position_shape.len(), &original)?;
        for (axis_index, axis) in spec.axes.iter().enumerate() {
            let count = axis.dimensions as usize / 2;
            let inv = if let Some(source) = prepared_frequencies {
                let values = &source[frequency_at..frequency_at + count];
                frequency_at += count;
                std::borrow::Cow::Borrowed(values)
            } else {
                let mut values = vec![0.0; count];
                fill_rotary_axis_frequencies(spec.base, axis.dimensions, &mut values)
                    .map_err(|cause| original.source(cause))?;
                std::borrow::Cow::Owned(values)
            };
            let inv = backend(
                Array::try_from_slice(&inv, &[1, inv.len() as i32])
                    .map_err(|cause| original.source(cause))?
                    .copy(context),
            )?;
            let positions = column(axis_index)?;
            let positions = backend(positions.as_dtype(Dtype::Float32, context))?;
            let positions = backend(positions.expand_dims(-1, context))?;
            let angles = backend(positions.multiply(inv, context))?;
            // Expand each independent row before appending it to the same
            // sink, so the equation needs only one owning row population.
            let angles = if spec.layout == MultiAxisRotaryLayout::IndependentAxes {
                backend(concatenate_axis(&[angles.clone(), angles], -1, context))?
            } else {
                angles
            };
            axis_angles.push(angles, &original)?;
        }
        let half = axis_angles.concatenate(context, &original)?;
        if spec.layout == MultiAxisRotaryLayout::SplitHalves {
            backend(concatenate_axis(&[half.clone(), half], -1, context))?
        } else {
            half
        }
    } else {
        let half = dimensions / 2;
        let axis_count = spec.axes.len();
        let mut selected = rows::Rows::new(half as usize, position_shape.len(), &original)?;
        for frequency in 0..half {
            let candidate = frequency as usize % axis_count;
            let section = spec.axes[candidate].dimensions / 2;
            let axis = if candidate != 0 && (frequency as u64) < section as u64 * axis_count as u64
            {
                candidate
            } else {
                0
            };
            let positions = column(axis)?;
            selected.push(backend(positions.expand_dims(-1, context))?, &original)?;
        }
        let selected = selected.concatenate(context, &original)?;
        let inv = if let Some(source) = prepared_frequencies {
            std::borrow::Cow::Borrowed(source)
        } else {
            let mut values = vec![0.0; half as usize];
            spec.fill_frequencies(&mut values)
                .map_err(|cause| original.source(cause))?;
            std::borrow::Cow::Owned(values)
        };
        let inv = backend(
            Array::try_from_slice(&inv, &[1, half])
                .map_err(|cause| original.source(cause))?
                .copy(context),
        )?;
        let selected = backend(selected.as_dtype(Dtype::Float32, context))?;
        let half = backend(selected.multiply(inv, context))?;
        backend(concatenate_axis(&[half.clone(), half], -1, context))?
    };
    let cosine = backend(angles.cos(context))?;
    let cosine = backend(safemlx::ops::reshape_like_prefix(
        &cosine,
        position_ids.as_array(),
        dimensions,
        context,
    ))?;
    let sine = backend(angles.sin(context))?;
    let sine = backend(safemlx::ops::reshape_like_prefix(
        &sine,
        position_ids.as_array(),
        dimensions,
        context,
    ))?;
    Ok((MlxTensor::from_array(cosine), MlxTensor::from_array(sine)))
}

#[cfg(test)]
mod tests;
