//! Metadata about the already-established backing of existing tensors.
//!
//! A survey supplies local equality information, not persistent owner identity,
//! liveness, release authority, or an admission bound. It retains no tensor,
//! allocation, owner, or completion lease. Queries never evaluate lazy values or
//! allocate tensors; unavailable backing facts remain absent.

use crate::{Error, Tensor, TensorElementType};

/// Facts for one backing observed during a single storage survey.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorStorageBacking {
    /// The backing is owned by the backend's ordinary native allocator.
    /// External or custom storage is false. Distinct external backing records
    /// can refer to overlapping storage and do not prove disjoint ownership.
    pub allocator_owned: bool,
    /// Capacity accounted to this backing by that allocator, when known.
    /// This is neither process RSS nor a bound including all native, allocator,
    /// object, or graph overhead. Zero is a known capacity; absence is unknown.
    /// External/custom storage must leave this absent.
    pub allocator_capacity_bytes: Option<u64>,
}

/// Bounded metadata for the ordered values supplied to one storage query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStorageSurvey {
    /// One entry per supplied tensor. A known entry indexes `backings` within
    /// this survey only. Equal indices establish the same observed backing;
    /// different indices do not establish disjoint external storage. `None`
    /// means unavailable or not yet allocated, never a zero-byte allocation.
    pub values: Vec<Option<usize>>,
    /// Compact backing records, each referenced by at least one value. Indices
    /// cannot be compared across surveys, even for the same input tensors.
    pub backings: Vec<TensorStorageBacking>,
}

impl TensorStorageSurvey {
    /// Reports unavailable storage facts for exactly `values` input tensors.
    /// Only host metadata is allocated, proportional to the input count.
    pub fn unavailable(values: usize) -> Self {
        Self {
            values: vec![None; values],
            backings: Vec::new(),
        }
    }

    /// Validates count, local references, compactness, and capacity provenance.
    /// This does not verify native identities or establish resource lifetimes.
    pub fn validate(&self, expected_values: usize) -> Result<(), Error> {
        if self.values.len() != expected_values {
            return Err(Error::backend(
                "tensor storage survey value count differs from query",
            ));
        }
        if self.backings.len() > self.values.len() {
            return Err(Error::backend(
                "tensor storage survey contains unreferenced backings",
            ));
        }
        let mut referenced = vec![false; self.backings.len()];
        for index in self.values.iter().flatten() {
            let Some(used) = referenced.get_mut(*index) else {
                return Err(Error::backend(
                    "tensor storage survey backing index is out of range",
                ));
            };
            *used = true;
        }
        if referenced.iter().any(|used| !used) {
            return Err(Error::backend(
                "tensor storage survey contains unreferenced backings",
            ));
        }
        if self
            .backings
            .iter()
            .any(|backing| !backing.allocator_owned && backing.allocator_capacity_bytes.is_some())
        {
            return Err(Error::backend(
                "external tensor storage cannot claim native allocator capacity",
            ));
        }
        Ok(())
    }
}

/// Logical metadata for one semantically identified tensor root. Its logical
/// extent is not a backing allocation size: views and broadcasts may alias or
/// have larger logical extents than their observed backing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStorageValue<Role> {
    /// Meaning supplied by the architecture or other tensor owner. This value
    /// is retained unchanged; use tensor-free tags for independent metadata.
    pub role: Role,
    /// Actual logical axes, without evaluation or tensor-content inspection.
    pub shape: Vec<i32>,
    /// Actual scalar representation when the tensor exposes it.
    pub element: Option<TensorElementType>,
    /// Checked logical extent in bytes, absent when scalar representation is
    /// unavailable. A scalar has one element; a zero axis gives zero elements.
    pub logical_bytes: Option<u64>,
}

/// Semantic tensor metadata aligned with one joint storage survey. Querying all
/// roots together exposes backing equality across their semantic categories.
/// The inspection machinery retains no inspected tensor handles or allocation
/// leases. Caller-supplied roles are retained unchanged: choose tensor-free tags
/// when metadata must be independent of tensor lifetimes. This is not a
/// session-state checkpoint, complete allocation inventory, live peak, or
/// admission bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStorageSnapshot<Role> {
    /// Metadata in exactly the supplied root order; roles need not be unique.
    pub values: Vec<TensorStorageValue<Role>>,
    /// One survey whose value indices align with `values`.
    pub storage: TensorStorageSurvey,
}

impl<Role> TensorStorageSnapshot<Role> {
    /// Borrows ordered tensor roots and queries their backing metadata exactly
    /// once after validating all logical extents. No tensor is cloned, allocated,
    /// evaluated, or synchronized. Host metadata is proportional to the supplied
    /// roots and their shape ranks. Unknown backing facts remain unavailable.
    /// Caller-supplied role contents are moved into the returned metadata.
    pub fn inspect<'a, T: Tensor>(
        roots: impl IntoIterator<Item = (Role, &'a T)>,
    ) -> Result<Self, Error> {
        let mut values = Vec::new();
        let mut tensors = Vec::new();
        for (role, tensor) in roots {
            values.push(value_metadata(role, tensor.shape(), tensor.element_type())?);
            tensors.push(tensor);
        }
        let storage = T::inspect_storage(&tensors)?;
        storage.validate(values.len())?;
        Ok(Self { values, storage })
    }

    /// Validates logical byte extents and alignment with the joint survey.
    /// Backend backing equality and allocator facts remain backend assertions.
    pub fn validate(&self) -> Result<(), Error> {
        for value in &self.values {
            if value.logical_bytes != logical_bytes(&value.shape, value.element)? {
                return Err(Error::backend(
                    "tensor storage logical bytes differ from shape and representation",
                ));
            }
        }
        self.storage.validate(self.values.len())
    }
}

fn value_metadata<Role>(
    role: Role,
    shape: &[i32],
    element: Option<TensorElementType>,
) -> Result<TensorStorageValue<Role>, Error> {
    Ok(TensorStorageValue {
        role,
        shape: shape.to_vec(),
        element,
        logical_bytes: logical_bytes(shape, element)?,
    })
}

fn logical_bytes(shape: &[i32], element: Option<TensorElementType>) -> Result<Option<u64>, Error> {
    if shape.iter().any(|axis| *axis < 0) {
        return Err(Error::backend(
            "tensor storage logical shape contains a negative axis",
        ));
    }
    let elements = if shape.contains(&0) {
        0
    } else {
        shape.iter().try_fold(1u64, |product, axis| {
            product
                .checked_mul(*axis as u64)
                .ok_or_else(|| Error::backend("tensor storage logical element count overflowed"))
        })?
    };
    element
        .map(|element| {
            elements
                .checked_mul(crate::mechanism_memory::element_bytes(element))
                .ok_or_else(|| Error::backend("tensor storage logical byte extent overflowed"))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(capacity: Option<u64>) -> TensorStorageBacking {
        TensorStorageBacking {
            allocator_owned: true,
            allocator_capacity_bytes: capacity,
        }
    }

    #[test]
    fn aliases_share_one_record_only_within_their_own_survey() {
        let first = TensorStorageSurvey {
            values: vec![Some(0), Some(0), Some(1), None],
            backings: vec![owned(Some(4096)), owned(Some(2048))],
        };
        first.validate(4).unwrap();
        assert_eq!(first.values[0], first.values[1]);
        assert_eq!(first.backings.len(), 2);

        // A separately ordered query establishes its own index assignment.
        let second = TensorStorageSurvey {
            values: vec![Some(0), Some(1), Some(1)],
            backings: vec![owned(Some(2048)), owned(Some(4096))],
        };
        second.validate(3).unwrap();
        assert_ne!(first.backings[0], second.backings[0]);
        assert_eq!(first.backings[0], second.backings[1]);
    }

    #[test]
    fn unavailable_unknown_capacity_and_known_zero_are_distinct() {
        let unavailable = TensorStorageSurvey::unavailable(3);
        unavailable.validate(3).unwrap();
        assert_eq!(unavailable.values, vec![None, None, None]);
        assert!(unavailable.backings.is_empty());

        let survey = TensorStorageSurvey {
            values: vec![None, Some(0), Some(1)],
            backings: vec![owned(None), owned(Some(0))],
        };
        survey.validate(3).unwrap();
        assert_eq!(survey.backings[0].allocator_capacity_bytes, None);
        assert_eq!(survey.backings[1].allocator_capacity_bytes, Some(0));
        TensorStorageSurvey::unavailable(0).validate(0).unwrap();
    }

    #[test]
    fn malformed_counts_indices_and_unused_records_are_rejected() {
        assert!(TensorStorageSurvey::unavailable(2).validate(1).is_err());
        let mut survey = TensorStorageSurvey {
            values: vec![Some(1), None],
            backings: vec![owned(Some(16))],
        };
        assert!(survey.validate(2).is_err());
        survey.values[0] = Some(usize::MAX);
        assert!(survey.validate(2).is_err());
        survey.values[0] = None;
        assert!(survey.validate(2).is_err());
        survey.values[0] = Some(0);
        survey.backings.push(owned(None));
        assert!(survey.validate(2).is_err());
    }

    #[test]
    fn external_wrappers_have_unknown_capacity_and_no_disjointness_claim() {
        let external = TensorStorageBacking {
            allocator_owned: false,
            allocator_capacity_bytes: None,
        };
        let mut survey = TensorStorageSurvey {
            values: vec![Some(0), Some(1), Some(0)],
            backings: vec![external, external],
        };
        // Distinct native wrappers may refer to overlapping external memory.
        survey.validate(3).unwrap();
        for capacity in [0, 16] {
            survey.backings[0].allocator_capacity_bytes = Some(capacity);
            assert!(survey.validate(3).is_err());
        }
    }

    #[test]
    fn logical_extents_preserve_scalar_empty_and_unknown_representation() {
        let scalar = value_metadata("scalar", &[], Some(TensorElementType::F64)).unwrap();
        assert_eq!(scalar.logical_bytes, Some(8));
        let matrix = value_metadata("matrix", &[3, 5], Some(TensorElementType::Bf16)).unwrap();
        assert_eq!(matrix.logical_bytes, Some(30));
        let empty = value_metadata("empty", &[i32::MAX; 4], None);
        assert!(empty.is_err());
        let empty = value_metadata(
            "empty",
            &[i32::MAX, i32::MAX, i32::MAX, 0],
            Some(TensorElementType::F64),
        )
        .unwrap();
        assert_eq!(empty.logical_bytes, Some(0));
        let unknown = value_metadata("unknown", &[3, 5], None).unwrap();
        assert_eq!(unknown.logical_bytes, None);
        let unknown_empty = value_metadata("unknown_empty", &[0], None).unwrap();
        assert_eq!(unknown_empty.logical_bytes, None);
    }

    #[test]
    fn malformed_shapes_and_both_extent_overflows_are_rejected() {
        for shape in [&[-1][..], &[0, -1], &[-1, 0], &[i32::MAX; 3]] {
            for element in [None, Some(TensorElementType::F32)] {
                assert!(value_metadata((), shape, element).is_err());
            }
        }
        // This element count fits u64, but its F64 byte count does not.
        assert!(value_metadata((), &[i32::MAX; 2], None).is_ok());
        assert!(value_metadata((), &[i32::MAX; 2], Some(TensorElementType::F64)).is_err());
    }

    #[test]
    fn snapshot_alignment_and_logical_metadata_are_validated_independently() {
        let mut snapshot = TensorStorageSnapshot {
            values: vec![
                value_metadata("request", &[2, 3], Some(TensorElementType::F32)).unwrap(),
                value_metadata("output", &[6], Some(TensorElementType::F32)).unwrap(),
            ],
            storage: TensorStorageSurvey {
                values: vec![Some(0), Some(0)],
                backings: vec![owned(Some(32))],
            },
        };
        snapshot.validate().unwrap();
        // Logical view bytes are not an allocation sum or required capacity.
        assert_eq!(snapshot.values[0].logical_bytes, Some(24));
        assert_eq!(snapshot.storage.backings.len(), 1);
        snapshot.storage.values.pop();
        assert!(snapshot.validate().is_err());
        snapshot.storage.values.push(Some(0));
        snapshot.values[0].logical_bytes = None;
        assert!(snapshot.validate().is_err());
        snapshot.values[0].element = None;
        snapshot.validate().unwrap();
        snapshot.values[0].shape[0] = -1;
        assert!(snapshot.validate().is_err());

        TensorStorageSnapshot::<()> {
            values: vec![],
            storage: TensorStorageSurvey::unavailable(0),
        }
        .validate()
        .unwrap();
    }
}
