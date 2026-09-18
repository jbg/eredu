//! Validated global identities for a local component axis.

use crate::{HostMetadataFunding, HostMetadataFundingError};
use eredu_collections::ordered_map::{Map, TryInsertError};
use std::ops::Range;

use serde::{Deserialize, Serialize};
mod index_projection;
pub use index_projection::{ComponentIndexProjectionError, ComponentIndexProjectionPlan};

/// A local scalar axis in global component coordinates. This map describes
/// coordinates only; it does not authorize observation, intervention or storage.
/// Empty local selections are permitted and do not imply missing global values.
#[derive(Debug, Serialize, Deserialize)]
#[serde(try_from = "WireMap", into = "WireMap")]
pub struct ComponentCoordinateMap {
    global_count: usize,
    selection: Selection,
    inverse: Map<usize, usize>,
}

impl PartialEq for ComponentCoordinateMap {
    fn eq(&self, other: &Self) -> bool {
        // Both constructors derive the inverse from this exact validated source.
        // Storage policy cannot change coordinate or serialized identity.
        self.global_count == other.global_count && self.selection == other.selection
    }
}
impl Eq for ComponentCoordinateMap {}

/// A coordinate source constructor's fixed refusal or semantic validation error.
#[derive(Debug, thiserror::Error)]
pub enum ComponentCoordinateConstructionError {
    /// The original coordinate declaration is invalid.
    #[error("{0}")]
    Coordinates(#[from] ComponentCoordinateError),
    /// The actual source account refused a producer before allocation.
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    /// The selected host allocator refused the actual vector request.
    #[error("coordinate source allocation: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
    /// The selected allocator returned an unqualified capacity.
    #[error("coordinate source capacity differs from its request")]
    Capacity,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Selection {
    Range { start: usize, end: usize },
    Indices { indices: Vec<usize> },
}

#[derive(Serialize, Deserialize)]
struct WireMap {
    global_count: usize,
    selection: Selection,
}

impl TryFrom<WireMap> for ComponentCoordinateMap {
    type Error = ComponentCoordinateError;

    fn try_from(value: WireMap) -> Result<Self, Self::Error> {
        match value.selection {
            Selection::Range { start, end } => Self::range(value.global_count, start..end),
            Selection::Indices { indices } => Self::indices(value.global_count, indices),
        }
    }
}

impl From<ComponentCoordinateMap> for WireMap {
    fn from(value: ComponentCoordinateMap) -> Self {
        Self {
            global_count: value.global_count,
            selection: value.selection,
        }
    }
}

impl ComponentCoordinateMap {
    /// Maps one contiguous local axis, including a complete replicated axis.
    pub fn range(
        global_count: usize,
        range: Range<usize>,
    ) -> Result<Self, ComponentCoordinateError> {
        if global_count == 0 {
            return Err(ComponentCoordinateError::EmptyGlobalAxis);
        }
        if range.start > range.end || range.end > global_count {
            return Err(ComponentCoordinateError::InvalidRange {
                start: range.start,
                end: range.end,
                count: global_count,
            });
        }
        Ok(Self {
            global_count,
            selection: Selection::Range {
                start: range.start,
                end: range.end,
            },
            inverse: Map::new(),
        })
    }

    /// Maps explicitly ordered global indices, preserving permutations and gaps.
    pub fn indices(
        global_count: usize,
        indices: Vec<usize>,
    ) -> Result<Self, ComponentCoordinateError> {
        match Self::indices_worker(global_count, indices, None) {
            Ok(value) => Ok(value),
            Err(ComponentCoordinateConstructionError::Coordinates(error)) => Err(error),
            Err(_) => {
                unreachable!("ordinary coordinate allocation")
            }
        }
    }

    /// Builds the same inverse with exact prospective node and control funding.
    /// The caller must retain this account with the resulting source or error;
    /// the supplied indices must already have their own construction custody.
    pub fn indices_with_funding(
        global_count: usize,
        indices: Vec<usize>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, ComponentCoordinateConstructionError> {
        Self::indices_worker(global_count, indices, Some(funding))
    }

    fn indices_worker(
        global_count: usize,
        indices: Vec<usize>,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<Self, ComponentCoordinateConstructionError> {
        controls(funding, Self::copy_control_bytes())?;
        if global_count == 0 {
            return Err(ComponentCoordinateError::EmptyGlobalAxis.into());
        }
        let mut inverse = Map::new();
        for (local, global) in indices.iter().copied().enumerate() {
            if global >= global_count {
                return Err(ComponentCoordinateError::OutOfRange {
                    index: global,
                    count: global_count,
                }
                .into());
            }
            if insert(&mut inverse, global, local, funding)?.is_some() {
                return Err(ComponentCoordinateError::Duplicate(global).into());
            }
        }
        Ok(Self {
            global_count,
            selection: Selection::Indices { indices },
            inverse,
        })
    }

    /// Expands architecture-declared partition units into scalar coordinates.
    /// Physical packed matrix dimensions are deliberately not an input.
    pub fn partition_units(
        global_count: usize,
        units: usize,
        range: Range<usize>,
    ) -> Result<Self, ComponentCoordinateError> {
        if units == 0 || global_count == 0 || !global_count.is_multiple_of(units) {
            return Err(ComponentCoordinateError::InvalidUnits {
                count: global_count,
                units,
            });
        }
        if range.start > range.end || range.end > units {
            return Err(ComponentCoordinateError::InvalidRange {
                start: range.start,
                end: range.end,
                count: units,
            });
        }
        // Bounds above prove both products are at most global_count.
        let width = global_count / units;
        Self::range(global_count, range.start * width..range.end * width)
    }

    /// Fixed source/copy/lookup controls; vector and tree node backing are separate.
    pub fn copy_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>() * 2,
            size_of::<Selection>(),
            size_of::<Map<usize, usize>>(),
            size_of::<Vec<usize>>(),
            size_of::<eredu_collections::ordered_map::Iter<'_, usize, usize>>(),
            size_of::<std::slice::Iter<'_, usize>>(),
            size_of::<[usize; 4]>(),
            size_of::<Option<&HostMetadataFunding>>(),
            size_of::<Result<Self, ComponentCoordinateConstructionError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Copies through the same source representation used by ordinary `Clone`.
    /// The enclosing caller retains `funding` with every escaping source/error.
    pub fn try_clone_with_funding(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Self, ComponentCoordinateConstructionError> {
        self.clone_worker(Some(funding))
    }

    fn clone_worker(
        &self,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<Self, ComponentCoordinateConstructionError> {
        controls(funding, Self::copy_control_bytes())?;
        let selection = match &self.selection {
            Selection::Range { start, end } => Selection::Range {
                start: *start,
                end: *end,
            },
            Selection::Indices { indices: source } => {
                let bytes = std::alloc::Layout::array::<usize>(source.len())
                    .map_err(|_| HostMetadataFundingError::Overflow)?
                    .size();
                controls(funding, Some(bytes))?;
                let mut indices = Vec::new();
                indices.try_reserve_exact(source.len())?;
                if indices.capacity() != source.len() {
                    return Err(ComponentCoordinateConstructionError::Capacity);
                }
                indices.extend_from_slice(source);
                Selection::Indices { indices }
            }
        };
        let mut inverse = Map::new();
        for (&global, &local) in self.inverse.iter() {
            insert(&mut inverse, global, local, funding)?;
        }
        Ok(Self {
            global_count: self.global_count,
            selection,
            inverse,
        })
    }

    /// Complete global component count.
    pub const fn global_count(&self) -> usize {
        self.global_count
    }

    /// Actual extent of the local component axis.
    pub fn local_count(&self) -> usize {
        match &self.selection {
            Selection::Range { start, end } => end - start,
            Selection::Indices { indices } => indices.len(),
        }
    }

    /// Contiguous representation, when one was declared.
    pub fn contiguous_range(&self) -> Option<Range<usize>> {
        match self.selection {
            Selection::Range { start, end } => Some(start..end),
            Selection::Indices { .. } => None,
        }
    }

    /// Resolves a valid global identity to its local storage index, if present.
    pub fn global_to_local(&self, global: usize) -> Option<usize> {
        match &self.selection {
            Selection::Range { start, end } => {
                (*start..*end).contains(&global).then(|| global - start)
            }
            Selection::Indices { .. } => self.inverse.get(&global).copied(),
        }
    }

    /// Resolves a local storage index to the original global identity.
    pub fn local_to_global(&self, local: usize) -> Option<usize> {
        match &self.selection {
            Selection::Range { start, end } => (local < end - start).then(|| start + local),
            Selection::Indices { indices } => indices.get(local).copied(),
        }
    }

    /// Validates a complete global mask index set before selecting local members.
    /// Keep/delete semantics remain the caller's original action: an empty local
    /// keep set zeros the local axis, while an empty delete set leaves it intact.
    pub fn localize_indices(&self, indices: &[u32]) -> Result<Vec<u32>, ComponentCoordinateError> {
        let mut scratch = vec![(0, 0); indices.len()];
        let plan = ComponentIndexProjectionPlan::prepare(self, indices, &mut scratch).map_err(
            |cause| match cause {
                ComponentIndexProjectionError::Coordinates(cause) => cause,
                ComponentIndexProjectionError::Storage { .. } => {
                    unreachable!("exact scratch extent")
                }
            },
        )?;
        let mut local = vec![0; plan.local_count()];
        plan.write(&mut local)
            .expect("validated exact local destination");
        Ok(local)
    }
}

impl Clone for ComponentCoordinateMap {
    fn clone(&self) -> Self {
        self.clone_worker(None)
            .expect("ordinary coordinate allocation")
    }
}
fn controls(
    funding: Option<&HostMetadataFunding>,
    bytes: Option<usize>,
) -> Result<(), HostMetadataFundingError> {
    let bytes = bytes.ok_or(HostMetadataFundingError::Overflow)?;
    if let Some(funding) = funding {
        funding.reserve_metadata(bytes)?;
    }
    Ok(())
}
fn insert(
    map: &mut Map<usize, usize>,
    global: usize,
    local: usize,
    funding: Option<&HostMetadataFunding>,
) -> Result<Option<usize>, HostMetadataFundingError> {
    controls(funding, map.insertion_control_bytes(&global))?;
    map.try_insert_with(global, local, |layout| {
        controls(funding, Some(layout.size()))
    })
    .map_err(|error| match error {
        TryInsertError::SizeOverflow => HostMetadataFundingError::Overflow,
        TryInsertError::Funding(error) => error,
    })
}

/// Invalid component coordinate declaration or mask selection.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ComponentCoordinateError {
    /// A declared global scalar axis must be nonempty.
    #[error("global component axis is empty")]
    EmptyGlobalAxis,
    /// The scalar count does not form integral semantic units.
    #[error("{count} components cannot be divided into {units} logical units")]
    InvalidUnits {
        /// Complete scalar extent.
        count: usize,
        /// Declared semantic unit count.
        units: usize,
    },
    /// A declared half-open range exceeds its axis or is reversed.
    #[error("component range {start}..{end} is invalid for extent {count}")]
    InvalidRange {
        /// Inclusive start.
        start: usize,
        /// Exclusive end.
        end: usize,
        /// Available axis extent.
        count: usize,
    },
    /// A global identity exceeds its axis.
    #[error("component index {index} exceeds extent {count}")]
    OutOfRange {
        /// Invalid global scalar identity.
        index: usize,
        /// Complete scalar extent.
        count: usize,
    },
    /// A selection repeats an identity.
    #[error("component index {0} occurs more than once")]
    Duplicate(usize),
    /// A local mask index cannot be represented by the intervention contract.
    #[error("local component index {0} exceeds the mask index type")]
    LocalIndexOverflow(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinate_clone_keeps_permuted_lookup_wire_identity_and_empty_owners() {
        for source in [
            ComponentCoordinateMap::range(12, 3..8).unwrap(),
            ComponentCoordinateMap::range(12, 8..8).unwrap(),
            ComponentCoordinateMap::indices(12, vec![9, 2, 7]).unwrap(),
            ComponentCoordinateMap::indices(12, vec![]).unwrap(),
        ] {
            let copied = source.clone();
            assert_eq!(copied, source);
            assert_eq!(
                serde_json::to_string(&copied).unwrap(),
                serde_json::to_string(&source).unwrap()
            );
            for global in 0..=source.global_count() {
                assert_eq!(
                    copied.global_to_local(global),
                    source.global_to_local(global)
                );
            }
            for local in 0..=source.local_count() {
                assert_eq!(copied.local_to_global(local), source.local_to_global(local));
            }
        }
    }

    #[test]
    fn partition_masks_keep_global_identity_and_empty_local_sets() {
        let left = ComponentCoordinateMap::partition_units(24, 3, 0..1).unwrap();
        let right = ComponentCoordinateMap::partition_units(24, 3, 1..3).unwrap();
        assert_eq!(left.local_count(), 8);
        assert_eq!(right.local_count(), 16);
        assert_eq!(left.localize_indices(&[19, 2, 8]).unwrap(), [2]);
        assert_eq!(right.localize_indices(&[19, 2, 8]).unwrap(), [11, 0]);
        assert!(left.localize_indices(&[19]).unwrap().is_empty());
        assert_eq!(right.local_to_global(11), Some(19));
        assert_eq!(right.local_to_global(16), None);
        assert!(matches!(
            left.localize_indices(&[19, 19]),
            Err(ComponentCoordinateError::Duplicate(19))
        ));
        assert!(matches!(
            left.localize_indices(&[24]),
            Err(ComponentCoordinateError::OutOfRange { .. })
        ));
    }

    #[test]
    fn noncontiguous_component_maps_preserve_storage_order_and_validate_wire() {
        let map = ComponentCoordinateMap::indices(12, vec![9, 2, 7]).unwrap();
        assert_eq!(map.localize_indices(&[2, 9, 6]).unwrap(), [1, 0]);
        assert_eq!(map.local_to_global(2), Some(7));
        let wire = serde_json::to_string(&map).unwrap();
        assert_eq!(
            serde_json::from_str::<ComponentCoordinateMap>(&wire).unwrap(),
            map
        );
        assert!(serde_json::from_str::<ComponentCoordinateMap>(
            r#"{"global_count":12,"selection":{"kind":"indices","indices":[2,2]}}"#
        )
        .is_err());
        assert!(ComponentCoordinateMap::range(12, 8..13).is_err());
        assert!(ComponentCoordinateMap::partition_units(13, 3, 0..1).is_err());
        assert!(ComponentCoordinateMap::partition_units(12, 3, 2..4).is_err());
        let huge = ComponentCoordinateMap::partition_units(usize::MAX, 1, 0..1).unwrap();
        assert_eq!(huge.local_to_global(usize::MAX - 1), Some(usize::MAX - 1));
        assert_eq!(huge.local_to_global(usize::MAX), None);
    }
}
#[cfg(test)]
mod funding_tests {
    use super::*;
    use crate::HostMetadataAccount;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    #[derive(Debug)]
    struct State {
        calls: AtomicUsize,
        stop: AtomicUsize,
    }
    #[derive(Debug)]
    struct Account(Arc<State>);
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
            let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
            assert!(
                call <= self.0.stop.load(Ordering::SeqCst),
                "producer reached after refusal"
            );
            if call == self.0.stop.load(Ordering::SeqCst) {
                Err(HostMetadataFundingError::Unavailable)
            } else {
                Ok(())
            }
        }
    }
    fn account(stop: usize) -> (HostMetadataFunding, Arc<State>) {
        let state = Arc::new(State {
            calls: AtomicUsize::new(0),
            stop: AtomicUsize::new(usize::MAX),
        });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        state.calls.store(0, Ordering::SeqCst);
        state.stop.store(stop, Ordering::SeqCst);
        (funding, state)
    }
    #[test]
    fn inverse_construction_preserves_original_validation_and_each_refusal() {
        for indices in [vec![], vec![9, 2, 7], vec![2, 2], vec![4, 12]] {
            let expected = ComponentCoordinateMap::indices(12, indices.clone());
            let (funding, state) = account(usize::MAX);
            let actual =
                ComponentCoordinateMap::indices_with_funding(12, indices.clone(), &funding);
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => {
                    assert_eq!(expected, actual);
                    for global in 0..13 {
                        assert_eq!(
                            expected.global_to_local(global),
                            actual.global_to_local(global)
                        );
                    }
                }
                (Err(expected), Err(ComponentCoordinateConstructionError::Coordinates(actual))) => {
                    assert_eq!(expected, actual)
                }
                value => panic!("coordinate semantics changed: {value:?}"),
            }
            for stop in 0..state.calls.load(Ordering::SeqCst) {
                let (funding, state) = account(stop);
                assert!(matches!(
                    ComponentCoordinateMap::indices_with_funding(12, indices.clone(), &funding),
                    Err(ComponentCoordinateConstructionError::Funding(
                        HostMetadataFundingError::Unavailable
                    ))
                ));
                assert_eq!(state.calls.load(Ordering::SeqCst), stop + 1);
            }
        }
    }
    #[test]
    fn source_clone_uses_one_inverse_representation_and_preserves_source_on_refusal() {
        for source in [
            ComponentCoordinateMap::range(12, 2..8).unwrap(),
            ComponentCoordinateMap::indices(12, vec![]).unwrap(),
            ComponentCoordinateMap::indices(12, vec![9, 2, 7]).unwrap(),
        ] {
            let original_wire = serde_json::to_string(&source).unwrap();
            let (funding, state) = account(usize::MAX);
            let copied = source.try_clone_with_funding(&funding).unwrap();
            assert_eq!(original_wire, serde_json::to_string(&copied).unwrap());
            assert_eq!(source, copied);
            for stop in 0..state.calls.load(Ordering::SeqCst) {
                let (funding, state) = account(stop);
                let error = source.try_clone_with_funding(&funding).unwrap_err();
                assert!(matches!(
                    error,
                    ComponentCoordinateConstructionError::Funding(
                        HostMetadataFundingError::Unavailable
                    )
                ));
                assert!(std::error::Error::source(&error)
                    .unwrap()
                    .is::<HostMetadataFundingError>());
                assert_eq!(state.calls.load(Ordering::SeqCst), stop + 1);
                assert_eq!(original_wire, serde_json::to_string(&source).unwrap());
            }
        }
    }
}
