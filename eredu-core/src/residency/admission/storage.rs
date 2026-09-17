use super::*;
use std::collections::TryReserveError;

/// One backend-supplied physical capacity, indexed into independent batch IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyReservationRow {
    /// Index in the actual borrowed acquisition ID slice.
    pub input: usize,
    /// Required physical capacity; zero remains a ledger refusal.
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Candidate {
    pub plan: usize,
    pub priority: u8,
    pub frequency: u64,
    pub last_used: u64,
    pub bytes: u64,
}

/// A committed eviction, indexed into its retained actual plan source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyEvictedRow {
    /// Ordinal in the retained source, never an unbound ledger identifier.
    pub plan: usize,
    /// Backend storage tier to release after the ledger transition.
    pub tier: MemoryTier,
    /// Physical bytes removed from accounting.
    pub bytes: u64,
}

/// Exact capacity blocker flags, indexed into the retained actual plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyBlockerRow {
    /// Ordinal in the retained source.
    pub plan: usize,
    /// Lifetime policy forbids eviction.
    pub pinned: bool,
    /// Current ownership lease count.
    pub in_use: u64,
    /// A named window protects this copy.
    pub active_window: bool,
    /// The complete atomic acquisition protects this copy.
    pub request_protected: bool,
}

/// Neutral initialized destinations for one capacity-admission operation.
///
/// Construction is cold. Prepared calls never grow these buffers. Payload
/// funding and native authority belong to the enclosing backend owner.
#[derive(Debug)]
pub struct ResidencyAdmissionStorage {
    pub(super) order: Vec<usize>,
    pub(super) candidates: Vec<Candidate>,
    pub(super) evicted: Vec<ResidencyEvictedRow>,
    pub(super) blockers: Vec<ResidencyBlockerRow>,
    pub(super) primary: String,
    pub(super) source: ResidencyPlanSource,
    pub(super) prepared: bool,
}

/// Cold preparation retained every initialized destination on refusal.
#[derive(Debug)]
pub struct ResidencyAdmissionPreparationError {
    /// Actual allocation failure.
    pub cause: TryReserveError,
    /// Initialized source and buffer prefix; no payload is discarded early.
    pub prefix: ResidencyAdmissionStorage,
}

impl ResidencyAdmissionStorage {
    /// Prepares exact requested destination extents from an actual plan source.
    /// `maximum_id_bytes` bounds a primary non-catalog diagnostic supplied by
    /// the caller; it never expands the actual source or grants work authority.
    pub fn try_new(
        source: ResidencyPlanSource,
        maximum_batch: usize,
        maximum_id_bytes: usize,
    ) -> Result<Self, ResidencyAdmissionPreparationError> {
        let mut value = Self {
            order: Vec::new(),
            candidates: Vec::new(),
            evicted: Vec::new(),
            blockers: Vec::new(),
            primary: String::new(),
            source,
            prepared: true,
        };
        let result = (|| {
            value.order.try_reserve_exact(maximum_batch)?;
            value.order.resize(maximum_batch, 0);
            value.candidates.try_reserve_exact(value.source.len())?;
            value.evicted.try_reserve_exact(value.source.len())?;
            value.blockers.try_reserve_exact(value.source.len())?;
            value.primary.try_reserve_exact(maximum_id_bytes)?;
            Ok::<_, TryReserveError>(())
        })();
        match result {
            Ok(()) => Ok(value),
            Err(cause) => Err(ResidencyAdmissionPreparationError {
                cause,
                prefix: value,
            }),
        }
    }
    pub(super) fn ordinary(source: ResidencyPlanSource, batch: usize) -> Self {
        Self {
            order: vec![0; batch],
            candidates: Vec::new(),
            evicted: Vec::new(),
            blockers: Vec::new(),
            primary: String::new(),
            source,
            prepared: false,
        }
    }
    /// Source retained independently of the mutable ledger.
    pub fn source(&self) -> &ResidencyPlanSource {
        &self.source
    }
    /// Borrows final index storage without mutating validation state.
    pub fn order(&self) -> &[usize] {
        &self.order
    }
    /// Reusable index scratch for sequential ordered validations.
    pub fn order_mut(&mut self) -> &mut [usize] {
        &mut self.order
    }
    /// Exact committed rows; no ledger borrow is retained.
    pub fn evicted(&self) -> &[ResidencyEvictedRow] {
        &self.evicted
    }
    /// Requested buffer layouts, excluding shared plan payload and allocator
    /// rounding. The source derives actual N; caller derives actual batch/text.
    pub fn requested_bytes(
        units: usize,
        maximum_batch: usize,
        maximum_id_bytes: usize,
    ) -> Option<usize> {
        [
            Layout::array::<usize>(maximum_batch).ok()?.size(),
            Layout::array::<Candidate>(units).ok()?.size(),
            Layout::array::<ResidencyEvictedRow>(units).ok()?.size(),
            Layout::array::<ResidencyBlockerRow>(units).ok()?.size(),
            Layout::array::<u8>(maximum_id_bytes).ok()?.size(),
        ]
        .into_iter()
        .try_fold(std::mem::size_of::<Self>(), usize::checked_add)
    }
    /// Retains a protocol refusal in the same final source and buffer owner.
    pub fn destination_failure(
        self,
        field: &'static str,
        required: usize,
        available: usize,
    ) -> PreparedResidencyAdmissionFailure {
        PreparedResidencyAdmissionFailure {
            cause: Failure::Destination {
                field,
                required,
                available,
            },
            storage: self,
        }
    }
    /// Retains the actual initialization refusal without an ID allocation.
    pub fn not_initialized_failure(self) -> PreparedResidencyAdmissionFailure {
        PreparedResidencyAdmissionFailure {
            cause: Failure::NotInitialized,
            storage: self,
        }
    }
    /// Copies an unknown caller ID only into its preallocated final diagnostic
    /// destination. Insufficient diagnostic space refuses without truncation.
    pub fn batch_failure(
        mut self,
        error: ResidencyBatchFailure<'_>,
    ) -> PreparedResidencyAdmissionFailure {
        let cause = match error {
            ResidencyBatchFailure::Ledger(ResidencyBatchError::InvalidTargetTier { operation }) => {
                Failure::InvalidTarget { operation }
            }
            ResidencyBatchFailure::Ledger(ResidencyBatchError::DuplicateBatchUnit) => {
                Failure::Duplicate
            }
            ResidencyBatchFailure::Ledger(ResidencyBatchError::UnknownUnit { id }) => {
                match self.store_primary(id) {
                    Ok(()) => Failure::Unknown,
                    Err(cause) => cause,
                }
            }
            ResidencyBatchFailure::Destination {
                required,
                available,
            } => Failure::Destination {
                field: "batch index",
                required,
                available,
            },
        };
        PreparedResidencyAdmissionFailure {
            cause,
            storage: self,
        }
    }
    /// Checks exact source and ordered validation before returning the same
    /// storage. No initialization, mutation, or ordinary error conversion occurs.
    pub fn validate_batch(
        mut self,
        ledger: &ResidencyLedger,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Self, PreparedResidencyAdmissionFailure> {
        if !ledger.matches_plan_source(&self.source) {
            return Err(PreparedResidencyAdmissionFailure {
                cause: Failure::Source,
                storage: self,
            });
        }
        match ledger.validate_batch_in(ids, tier, &mut self.order) {
            Ok(()) => Ok(self),
            Err(cause) => Err(self.batch_failure(cause)),
        }
    }
    pub(super) fn reset(&mut self) {
        self.candidates.clear();
        self.evicted.clear();
        self.blockers.clear();
        self.primary.clear();
    }
    pub(super) fn store_primary(&mut self, id: &OffloadUnitId) -> Result<(), Failure> {
        if self.prepared && self.primary.capacity() < id.as_str().len() {
            return Err(Failure::Destination {
                field: "primary identifier",
                required: id.as_str().len(),
                available: self.primary.capacity(),
            });
        }
        self.primary.clear();
        self.primary.push_str(id.as_str());
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Failure {
    NotInitialized,
    InvalidTarget {
        operation: &'static str,
    },
    Duplicate,
    Unknown,
    Zero {
        plan: usize,
    },
    Exists {
        plan: usize,
        tier: MemoryTier,
    },
    Overflow {
        context: &'static str,
    },
    Budget {
        plan: usize,
        tier: MemoryTier,
        required: u64,
        budget: u64,
        resident: u64,
    },
    Inconsistent {
        plan: usize,
        tier: MemoryTier,
        operation: &'static str,
    },
    Source,
    Destination {
        field: &'static str,
        required: usize,
        available: usize,
    },
}

/// A complete prepared refusal retaining its actual source and finite outputs.
///
/// An enclosing backend owner must retain control custody after this value.
/// The type contains no native scope, observer, tensor or mutable controller.
#[derive(Debug)]
pub struct PreparedResidencyAdmissionFailure {
    pub(super) cause: Failure,
    pub(super) storage: ResidencyAdmissionStorage,
}
impl PreparedResidencyAdmissionFailure {
    /// Whether the actual cause was insufficient eligible capacity.
    pub fn is_budget_exhausted(&self) -> bool {
        matches!(self.cause, Failure::Budget { .. })
    }
    /// The exact target of a budget refusal; other refusals return None.
    pub fn budget_tier(&self) -> Option<MemoryTier> {
        match self.cause {
            Failure::Budget { tier, .. } => Some(tier),
            _ => None,
        }
    }
    /// Actual ordered blocker rows and their independent immutable source.
    pub fn blockers(&self) -> (&ResidencyPlanSource, &[ResidencyBlockerRow]) {
        (&self.storage.source, &self.storage.blockers)
    }
    /// Consume only when the caller deliberately reuses neutral storage after
    /// handling the refusal. This grants no retry or native submission authority.
    pub fn into_storage(self) -> ResidencyAdmissionStorage {
        self.storage
    }
}
impl fmt::Display for PreparedResidencyAdmissionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let unit = |plan| {
            self.storage
                .source
                .id(plan)
                .map(OffloadUnitId::as_str)
                .unwrap_or("<invalid source>")
        };
        match self.cause {
            Failure::NotInitialized => f.write_str("residency manager has not been initialized"),
            Failure::InvalidTarget { operation } => write!(f, "{operation} requires a host or device tier"),
            Failure::Duplicate => f.write_str("batched residency acquisition contains a duplicate unit"),
            Failure::Unknown => write!(f, "unknown residency unit: {}", self.storage.primary),
            Failure::Zero { plan } => write!(f, "residency unit {} cannot reserve zero bytes", unit(plan)),
            Failure::Exists { plan, tier } => write!(f, "residency unit {} already has a {tier:?} copy", unit(plan)),
            Failure::Overflow { context } => write!(f, "residency arithmetic overflow during {context}"),
            Failure::Budget { plan, tier, required, budget, resident } => write!(f, "cannot reserve {required} bytes for {} in {tier:?}: {resident}/{budget} bytes resident", unit(plan)),
            Failure::Inconsistent { plan, tier, operation } => write!(f, "residency ledger state is inconsistent for {} in {tier:?} during {operation}", unit(plan)),
            Failure::Source => f.write_str("prepared residency source does not match the actual ledger"),
            Failure::Destination { field, required, available } => write!(f, "prepared residency {field} needs {required} slots, has {available}"),
        }
    }
}
impl std::error::Error for PreparedResidencyAdmissionFailure {}
