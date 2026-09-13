//! Validated global identities for a local component axis.

use std::{collections::BTreeMap, ops::Range};

use serde::{Deserialize, Serialize};

/// A local scalar axis in global component coordinates. This map describes
/// coordinates only; it does not authorize observation, intervention or storage.
/// Empty local selections are permitted and do not imply missing global values.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "WireMap", into = "WireMap")]
pub struct ComponentCoordinateMap {
    global_count: usize,
    selection: Selection,
    inverse: BTreeMap<usize, usize>,
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
            inverse: BTreeMap::new(),
        })
    }

    /// Maps explicitly ordered global indices, preserving permutations and gaps.
    pub fn indices(
        global_count: usize,
        indices: Vec<usize>,
    ) -> Result<Self, ComponentCoordinateError> {
        if global_count == 0 {
            return Err(ComponentCoordinateError::EmptyGlobalAxis);
        }
        let mut inverse = BTreeMap::new();
        for (local, global) in indices.iter().copied().enumerate() {
            if global >= global_count {
                return Err(ComponentCoordinateError::OutOfRange {
                    index: global,
                    count: global_count,
                });
            }
            if inverse.insert(global, local).is_some() {
                return Err(ComponentCoordinateError::Duplicate(global));
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
        let mut seen = std::collections::BTreeSet::new();
        let mut local = Vec::new();
        for &global in indices {
            let global = global as usize;
            if global >= self.global_count {
                return Err(ComponentCoordinateError::OutOfRange {
                    index: global,
                    count: self.global_count,
                });
            }
            if !seen.insert(global) {
                return Err(ComponentCoordinateError::Duplicate(global));
            }
            if let Some(index) = self.global_to_local(global) {
                local.push(
                    u32::try_from(index)
                        .map_err(|_| ComponentCoordinateError::LocalIndexOverflow(index))?,
                );
            }
        }
        Ok(local)
    }
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
