//! Admission for an explicit copy of a prepared input's complete wire values.
//!
//! Counts describe logical copied payloads, including repeated wire occurrences
//! of aliases. Host staging is separately bounded by backend-supplied allocation
//! capacity facts for a serial copy mechanism that retires each buffer at native
//! completion before allocating the next. Source and destination can coexist;
//! their backing allocations, kernel scratch and other native storage are outside
//! this staging ceiling. These limits do not replace completion-owned retention
//! or session authority and do not establish a whole-request physical peak.

use eredu_core::{checkpoint::TensorDtype, InputTensorIdentity, PreparedInputIdentity};

/// Independent per-copy ceilings. Zero denies that resource.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputTransferBudget {
    /// Number of payload and metadata tensors copied in wire order.
    pub tensor_count: u64,
    /// Sum of logical destination payload bytes.
    pub payload_bytes: u64,
    /// Peak physical capacity of serial, completion-retired host staging buffers.
    /// Excludes source/destination backing and other native allocations.
    pub staging_capacity_bytes: u64,
}

/// Logical copy quantities and selected host-staging capacity bound, derived
/// without inspecting native values or allocating copy buffers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputTransferResources {
    /// Wire tensor occurrences, without deduplicating aliased handles.
    pub tensor_count: u64,
    /// Logical bytes copied for all occurrences, not physical traffic or peak memory.
    pub payload_bytes: u64,
    /// Peak physical capacity of serial, completion-retired host staging buffers.
    /// Excludes source/destination backing and other native allocations.
    pub staging_capacity_bytes: u64,
}

/// Failure before allocating the explicit copy's native outputs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InputTransferError {
    /// Shape, count or byte arithmetic overflowed.
    #[error("prepared input transfer arithmetic overflow")]
    Overflow,
    /// Encoded storage requires an exact storage contract, not a guessed scalar width.
    #[error("prepared input transfer cannot size dtype {dtype:?}")]
    UnsupportedDtype {
        /// Dtype with no dense scalar width.
        dtype: TensorDtype,
    },
    /// The selected staging mechanism cannot establish an allocation capacity bound.
    #[error("prepared input transfer staging bound is unavailable: {detail}")]
    StagingUnavailable {
        /// Native mechanism fact that could not be established.
        detail: String,
    },
    /// A claimed staging capacity cannot contain the complete wire payload.
    #[error("prepared input transfer staging capacity {capacity_bytes} cannot contain {payload_bytes} payload bytes")]
    InvalidStagingBound {
        /// Complete tensor payload that must fit in the staging buffer.
        payload_bytes: u64,
        /// Claimed physical capacity upper bound.
        capacity_bytes: u64,
    },
    /// A request exceeds an explicit copy ceiling.
    #[error("prepared input transfer {resource} requires {required}, exceeding budget {limit}")]
    Exhausted {
        /// Resource and its unit.
        resource: &'static str,
        /// Derived requirement.
        required: u64,
        /// Caller ceiling.
        limit: u64,
    },
}

impl InputTransferResources {
    /// Checked logical wire payload extent for a dense tensor, before staging
    /// allocation rounding. Encoded values require a separate exact contract.
    pub fn tensor_payload_bytes(tensor: &InputTensorIdentity) -> Result<u64, InputTransferError> {
        let width: u64 = match tensor.dtype() {
            TensorDtype::Bool | TensorDtype::I8 | TensorDtype::U8 => 1,
            TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::U16 | TensorDtype::I16 => 2,
            TensorDtype::F32 | TensorDtype::I32 | TensorDtype::U32 => 4,
            TensorDtype::F64 | TensorDtype::I64 | TensorDtype::U64 | TensorDtype::Complex64 => 8,
            dtype @ TensorDtype::Encoded(_) => {
                return Err(InputTransferError::UnsupportedDtype {
                    dtype: dtype.clone(),
                })
            }
        };
        tensor.shape().iter().try_fold(width, |bytes, dim| {
            bytes
                .checked_mul(u64::try_from(*dim).map_err(|_| InputTransferError::Overflow)?)
                .ok_or(InputTransferError::Overflow)
        })
    }

    /// Sizes payloads and metadata in deterministic wire order, allocating no catalog.
    /// The callback supplies the selected host buffer's physical capacity upper
    /// bound and must query facts without allocating buffers or submitting work.
    /// It is called once for every wire occurrence, including repeated aliases.
    /// All logical extents are validated before the first callback invocation.
    ///
    /// Staging is the maximum capacity, not the sum: the executing mechanism must
    /// retain each buffer to terminal native completion and release it before
    /// allocating the next. Source and destination allocations remain separate.
    pub fn from_identity_with_staging(
        identity: &PreparedInputIdentity,
        mut staging_capacity: impl FnMut(&InputTensorIdentity) -> Result<u64, InputTransferError>,
    ) -> Result<Self, InputTransferError> {
        let tensors = || {
            identity
                .parts()
                .iter()
                .flat_map(|part| std::iter::once(part.payload()).chain(part.metadata().values()))
        };
        let mut report = Self::default();
        for tensor in tensors() {
            let bytes = Self::tensor_payload_bytes(tensor)?;
            report.tensor_count = report
                .tensor_count
                .checked_add(1)
                .ok_or(InputTransferError::Overflow)?;
            report.payload_bytes = report
                .payload_bytes
                .checked_add(bytes)
                .ok_or(InputTransferError::Overflow)?;
        }
        for tensor in tensors() {
            let capacity_bytes = staging_capacity(tensor)?;
            let payload_bytes = Self::tensor_payload_bytes(tensor)?;
            if capacity_bytes < payload_bytes {
                return Err(InputTransferError::InvalidStagingBound {
                    payload_bytes,
                    capacity_bytes,
                });
            }
            report.staging_capacity_bytes = report.staging_capacity_bytes.max(capacity_bytes);
        }
        Ok(report)
    }

    /// Returns an inclusive budget equal to this report.
    pub fn into_budget(self) -> InputTransferBudget {
        InputTransferBudget {
            tensor_count: self.tensor_count,
            payload_bytes: self.payload_bytes,
            staging_capacity_bytes: self.staging_capacity_bytes,
        }
    }
}

impl InputTransferBudget {
    /// Checks the whole request before cloning roots or submitting any copies.
    pub fn admit(self, required: InputTransferResources) -> Result<(), InputTransferError> {
        for (resource, required, limit) in [
            ("tensor count", required.tensor_count, self.tensor_count),
            ("payload bytes", required.payload_bytes, self.payload_bytes),
            (
                "staging capacity bytes",
                required.staging_capacity_bytes,
                self.staging_capacity_bytes,
            ),
        ] {
            if required > limit {
                return Err(InputTransferError::Exhausted {
                    resource,
                    required,
                    limit,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{
        InputMetadataKey, InputModality, InputPartDescriptor, InputPayloadKind, InputTensorIdentity,
    };

    fn identity(dtype: TensorDtype, shape: Vec<usize>) -> PreparedInputIdentity {
        PreparedInputIdentity::new(vec![InputPartDescriptor::new(
            InputModality::Image,
            InputPayloadKind::Tensor,
            InputTensorIdentity::new(dtype, shape).unwrap(),
            [(
                InputMetadataKey::PatchGrid,
                InputTensorIdentity::new(TensorDtype::I32, vec![1, 3]).unwrap(),
            )],
        )
        .unwrap()])
        .unwrap()
    }

    #[test]
    fn transfer_counts_metadata_and_independent_inclusive_limits() {
        let input = identity(TensorDtype::Bf16, vec![4, 6]);
        let report = InputTransferResources::from_identity_with_staging(&input, |tensor| {
            Ok(InputTransferResources::tensor_payload_bytes(tensor)?.next_multiple_of(32))
        })
        .unwrap();
        assert_eq!(
            report,
            InputTransferResources {
                tensor_count: 2,
                payload_bytes: 60,
                staging_capacity_bytes: 64,
            }
        );
        let exact = report.into_budget();
        exact.admit(report).unwrap();
        for budget in [
            InputTransferBudget {
                tensor_count: 1,
                ..exact
            },
            InputTransferBudget {
                payload_bytes: 59,
                ..exact
            },
            InputTransferBudget {
                staging_capacity_bytes: 63,
                ..exact
            },
        ] {
            assert!(matches!(
                budget.admit(report),
                Err(InputTransferError::Exhausted { .. })
            ));
        }
        let repeated = PreparedInputIdentity::new(
            input
                .parts()
                .iter()
                .cloned()
                .chain(input.parts().iter().cloned())
                .collect(),
        )
        .unwrap();
        let mut queries = 0;
        assert_eq!(
            InputTransferResources::from_identity_with_staging(&repeated, |tensor| {
                queries += 1;
                Ok(InputTransferResources::tensor_payload_bytes(tensor)?.next_multiple_of(32))
            })
            .unwrap(),
            InputTransferResources {
                tensor_count: 4,
                payload_bytes: 120,
                staging_capacity_bytes: 64,
            }
        );
        assert_eq!(
            queries, 4,
            "repeated wire values each require a staging bound"
        );
    }

    #[test]
    fn transfer_rejects_unknown_encodings_and_overflow_before_staging_queries() {
        let valid = identity(TensorDtype::U8, vec![1]);
        for (invalid, expected) in [
            (
                identity(TensorDtype::Encoded("opaque".into()), vec![4, 6]),
                InputTransferError::UnsupportedDtype {
                    dtype: TensorDtype::Encoded("opaque".into()),
                },
            ),
            (
                identity(TensorDtype::F64, vec![usize::MAX, usize::MAX, 2]),
                InputTransferError::Overflow,
            ),
        ] {
            let input = PreparedInputIdentity::new(
                valid
                    .parts()
                    .iter()
                    .chain(invalid.parts())
                    .cloned()
                    .collect(),
            )
            .unwrap();
            assert_eq!(
                InputTransferResources::from_identity_with_staging(&input, |_| {
                    panic!("invalid later wire value must reject before any staging query")
                }),
                Err(expected)
            );
        }
    }

    #[test]
    fn aggregate_payload_overflow_precedes_staging_queries() {
        // Every tensor fits u64 on both 32-bit and 64-bit hosts, but two do not.
        let one = identity(TensorDtype::U8, vec![1usize << 31, 1usize << 31, 3]);
        let repeated =
            PreparedInputIdentity::new(one.parts().iter().chain(one.parts()).cloned().collect())
                .unwrap();
        assert_eq!(
            InputTransferResources::from_identity_with_staging(&repeated, |_| {
                panic!("logical payload overflow must precede staging queries")
            }),
            Err(InputTransferError::Overflow)
        );
    }

    #[test]
    fn staging_capacity_facts_are_required_and_must_contain_each_payload() {
        let input = identity(TensorDtype::Bf16, vec![4, 6]);
        assert_eq!(
            InputTransferResources::from_identity_with_staging(&input, |_| Ok(47)),
            Err(InputTransferError::InvalidStagingBound {
                payload_bytes: 48,
                capacity_bytes: 47,
            })
        );
        let mut queried = Vec::new();
        let report = InputTransferResources::from_identity_with_staging(&input, |tensor| {
            queried.push(tensor.clone());
            // A smaller metadata payload can still have the larger allocation.
            Ok(if tensor.dtype() == &TensorDtype::I32 {
                4096
            } else {
                64
            })
        })
        .unwrap();
        assert_eq!(
            queried,
            vec![
                input.parts()[0].payload().clone(),
                input.parts()[0].metadata().values().next().unwrap().clone()
            ]
        );
        assert_eq!(report.staging_capacity_bytes, 4096);
        assert_eq!(
            InputTransferBudget {
                staging_capacity_bytes: 4095,
                ..report.into_budget()
            }
            .admit(report),
            Err(InputTransferError::Exhausted {
                resource: "staging capacity bytes",
                required: 4096,
                limit: 4095,
            })
        );
        let missing = InputTransferError::StagingUnavailable {
            detail: "selected transfer storage has no capacity bound".into(),
        };
        assert_eq!(
            InputTransferResources::from_identity_with_staging(&input, |_| Err(missing.clone())),
            Err(missing)
        );
        InputTransferBudget::default()
            .admit(InputTransferResources::default())
            .unwrap();
        assert!(InputTransferBudget::default().admit(report).is_err());
    }
}
