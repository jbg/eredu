//! The same mask selection with caller-owned finite host destinations.
use super::{ComponentCoordinateError, ComponentCoordinateMap};
use std::mem::{size_of, size_of_val};

/// Source geometry or exact caller-owned scratch/output storage differs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ComponentIndexProjectionError {
    /// The original global set or native local index is invalid.
    #[error(transparent)]
    Coordinates(#[from] ComponentCoordinateError),
    /// The supplied slice does not have the source-derived exact length.
    #[error("component index projection requires {expected} entries, received {actual}")]
    Storage { /// Source-derived length.
        expected: usize, /// Actual supplied length.
        actual: usize },
}
/// Borrowed coordinate and index source. It grants no storage or edit authority.
#[derive(Debug)]
pub struct ComponentIndexProjectionPlan<'a> {
    coordinates: &'a ComponentCoordinateMap,
    indices: &'a [u32],
    local: usize,
}
impl<'a> ComponentIndexProjectionPlan<'a> {
    /// Validate the full source using exactly one scratch entry per input index.
    /// The scratch is temporary; the plan keeps the original order and source.
    pub fn prepare(coordinates: &'a ComponentCoordinateMap, indices: &'a [u32],
        scratch: &mut [(u32, usize)]) -> Result<Self, ComponentIndexProjectionError> {
        if scratch.len() != indices.len() {
            return Err(ComponentIndexProjectionError::Storage { expected: indices.len(), actual: scratch.len() });
        }
        for (position, (&global, slot)) in indices.iter().zip(scratch.iter_mut()).enumerate() {
            *slot = (global, position);
        }
        scratch.sort_unstable();
        // Preserve ordinary first-failure order even with several duplicates or
        // a later out-of-range ID. Sorting does not reorder the emitted mask.
        let duplicate = scratch.windows(2).filter(|pair| pair[0].0 == pair[1].0)
            .map(|pair| pair[1].1).min();
        let mut local = 0usize;
        for (position, &global) in indices.iter().enumerate() {
            let global = global as usize;
            if global >= coordinates.global_count() {
                return Err(ComponentCoordinateError::OutOfRange { index: global,
                    count: coordinates.global_count() }.into());
            }
            if duplicate == Some(position) {
                return Err(ComponentCoordinateError::Duplicate(global).into());
            }
            if let Some(index) = coordinates.global_to_local(global) {
                u32::try_from(index).map_err(|_| ComponentCoordinateError::LocalIndexOverflow(index))?;
                local += 1;
            }
        }
        Ok(Self { coordinates, indices, local })
    }
    /// Exact number of output identities, not the global set's maximum.
    pub const fn local_count(&self) -> usize { self.local }
    /// Fill the exact preallocated output in the original global-set order.
    pub fn write(&self, output: &mut [u32]) -> Result<(), ComponentIndexProjectionError> {
        if output.len() != self.local {
            return Err(ComponentIndexProjectionError::Storage { expected: self.local, actual: output.len() });
        }
        let mut next = 0;
        for &global in self.indices {
            if let Some(local) = self.coordinates.global_to_local(global as usize) {
                output[next] = u32::try_from(local).expect("validated local source index");
                next += 1;
            }
        }
        Ok(())
    }
    /// Fixed validation/sort/copy controls; caller storage is counted separately.
    pub fn control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>(), size_of::<Result<Self, ComponentIndexProjectionError>>(),
            size_of::<Result<(), ComponentIndexProjectionError>>(), size_of::<(&ComponentCoordinateMap, &[u32], &mut [(u32, usize)])>(),
            size_of::<(&Self, &mut [u32])>(), size_of::<[usize; 6]>(), size_of::<Option<usize>>(),
            size_of::<std::slice::Windows<'_, (u32, usize)>>(), size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<ComponentCoordinateError>()];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_index_projection_preserves_order_empty_owners_and_first_refusal() {
        for map in [ComponentCoordinateMap::indices(10, vec![8, 2, 6]).unwrap(),
            ComponentCoordinateMap::range(10, 2..7).unwrap(), ComponentCoordinateMap::range(10, 0..0).unwrap()] {
            let input = [6, 1, 8, 2];
            let mut scratch = [(0, 0); 4];
            let plan = ComponentIndexProjectionPlan::prepare(&map, &input, &mut scratch).unwrap();
            let mut output = vec![0; plan.local_count()];
            plan.write(&mut output).unwrap();
            let expected: Vec<_> = input.iter().filter_map(|value| map.global_to_local(*value as usize))
                .map(|value| value as u32).collect();
            assert_eq!(output, expected);
            assert_eq!(map.localize_indices(&input).unwrap(), output);
            output.push(0);
            assert!(plan.write(&mut output).is_err());
        }
        let map = ComponentCoordinateMap::range(10, 0..10).unwrap();
        for (input, expected) in [(vec![8, 2, 8, 12], ComponentCoordinateError::Duplicate(8)),
            (vec![12, 8, 8], ComponentCoordinateError::OutOfRange { index: 12, count: 10 }),
            (vec![8, 2, 2, 8], ComponentCoordinateError::Duplicate(2))] {
            let mut scratch = vec![(0, 0); input.len()];
            assert_eq!(ComponentIndexProjectionPlan::prepare(&map, &input, &mut scratch).unwrap_err(),
                ComponentIndexProjectionError::Coordinates(expected));
        }
        assert!(ComponentIndexProjectionPlan::prepare(&map, &[1], &mut []).is_err());
    }
}
