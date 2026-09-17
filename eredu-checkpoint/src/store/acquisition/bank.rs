//! Finite final destinations matched by actual retained source and request.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};

/// A fixed-capacity collection of once-only final acquisition destinations.
/// The caller supplies actual request multiplicities and funds preparation.
/// This collection is storage, never a complete bound or execution authority.
#[derive(Debug)]
pub struct PreparedAcquisitionBank {
    slots: Vec<Option<PreparedCheckpointAcquisition>>,
    required: usize,
    sealed: bool,
}
/// Failure preserves any actual refused destination/lease inside `Acquisition`.
/// Existing successfully prepared bank slots remain owned by their caller.
#[derive(Debug)]
pub enum PreparedAcquisitionBankError {
    /// The requested slot layout is not representable.
    Overflow,
    /// Reserving the final slot buffer failed.
    Reserve(TryReserveError),
    /// The exact source has no prepared acquisition route.
    Unavailable,
    /// Preparation exceeded or did not fill the declared finite population.
    Population,
    /// No unconsumed destination matches this exact source and request.
    NoDestination,
    /// Actual authorized acquisition or final destination preparation failed.
    Acquisition(PreparedAcquisitionFailure),
}
/// Compact category of a refusal with no source, destination or allocation.
/// Owning acquisition/reservation failures never convert to this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedAcquisitionRefusal {
    /// Requested layout arithmetic is not representable.
    Overflow,
    /// No supported final destination accompanies the source.
    Unavailable,
    /// Declared preparation population or sealing state does not match.
    Population,
    /// No unconsumed ticket matches the exact borrowed request.
    NoDestination,
}
impl std::fmt::Display for PreparedAcquisitionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Overflow => "prepared acquisition slot layout overflow",
            Self::Unavailable => "source has no prepared acquisition destination",
            Self::Population => "prepared acquisition population mismatch",
            Self::NoDestination => "no matching prepared acquisition destination remains",
        })
    }
}
impl std::error::Error for PreparedAcquisitionRefusal {}

impl PreparedAcquisitionBankError {
    /// Copy only an actual nonowning refusal category. This never discards a
    /// reserve cause, partial destination, source lease or successful prefix.
    pub fn refusal(&self) -> Option<PreparedAcquisitionRefusal> {
        Some(match self {
            Self::Overflow => PreparedAcquisitionRefusal::Overflow,
            Self::Unavailable => PreparedAcquisitionRefusal::Unavailable,
            Self::Population => PreparedAcquisitionRefusal::Population,
            Self::NoDestination => PreparedAcquisitionRefusal::NoDestination,
            Self::Reserve(_) | Self::Acquisition(_) => return None,
        })
    }

    /// Preserve ordinary cache-capacity detection without discarding this owner.
    pub fn store_error(&self) -> Option<&StoreError> {
        match self {
            Self::Acquisition(error) => error.store_error(),
            _ => None,
        }
    }
}
impl std::fmt::Display for PreparedAcquisitionBankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(refusal) = self.refusal() {
            return refusal.fmt(f);
        }
        match self {
            Self::Reserve(error) => error.fmt(f),
            Self::Acquisition(error) => error.fmt(f),
            _ => unreachable!("fixed refusal handled above"),
        }
    }
}

impl std::error::Error for PreparedAcquisitionBankError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(error) => Some(error),
            Self::Acquisition(error) => Some(error),
            _ => None,
        }
    }
}
/// Actual bank buffer geometry; per-destination payloads and shared caches are
/// separate owners. No requested layout is reported as an allocator charge.
#[derive(Clone, Copy, Debug)]
pub struct PreparedAcquisitionStorage {
    /// Inline bank value.
    pub inline: Layout,
    /// Requested array layout before slot preparation.
    pub requested_slots: Layout,
    /// Actual retained slot capacity after reserve.
    pub retained_slots: Layout,
    /// Prepared, not yet consumed destination count.
    pub remaining: usize,
    /// Actual remaining owned buffer capacities and boxed payloads. This omits
    /// Arc headers, allocator charges, cold transient overlap and shared source/cache owners.
    pub owned_payload_capacity_bytes: usize,
    /// New final destination Arc allocations whose headers are not included above.
    pub destination_arc_allocations: usize,
}
impl PreparedAcquisitionBank {
    /// Reserve only the final slot array. No source acquisition occurs.
    pub fn new(required: usize) -> Result<Self, PreparedAcquisitionBankError> {
        Self::slot_layout(required).ok_or(PreparedAcquisitionBankError::Overflow)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(required)
            .map_err(PreparedAcquisitionBankError::Reserve)?;
        Ok(Self {
            slots,
            required,
            sealed: false,
        })
    }
    /// Actual slot type, including its inline destination/request/source owners.
    pub fn slot_layout(count: usize) -> Option<Layout> {
        Layout::array::<Option<PreparedCheckpointAcquisition>>(count).ok()
    }
    /// Prepare one final metadata/request ticket at the caller's cold boundary.
    /// SafeTensors/Memory payload allocation is deferred to actual acquisition;
    /// its peak and allocator charge require the caller's existing reservation.
    /// Authorization and source-specific preparation use the existing worker.
    pub fn prepare_next(
        &mut self,
        source: impl Into<RetainedCheckpointSource>,
        request: TensorReadRequest,
    ) -> Result<(), PreparedAcquisitionBankError> {
        if self.sealed || self.slots.len() == self.required {
            return Err(PreparedAcquisitionBankError::Population);
        }
        let destination = PreparedCheckpointAcquisition::prepare_deferred(source, request)
            .map_err(PreparedAcquisitionBankError::Acquisition)?
            .ok_or(PreparedAcquisitionBankError::Unavailable)?;
        self.slots.push(Some(destination));
        Ok(())
    }
    /// Insert a final destination from the retained selected GGUF plan. The
    /// caller funds the known source recipe before invoking this constructor.
    pub fn prepare_selected_gguf(
        &mut self,
        plan: &SelectedGgufConversionPlan,
    ) -> Result<(), PreparedAcquisitionBankError> {
        if self.sealed || self.slots.len() == self.required {
            return Err(PreparedAcquisitionBankError::Population);
        }
        let destination = plan
            .prepare_acquisition()
            .map_err(PreparedAcquisitionBankError::Acquisition)?
            .ok_or(PreparedAcquisitionBankError::Unavailable)?;
        self.slots.push(Some(destination));
        Ok(())
    }
    /// Finish the exact population before any hot consumption is possible.
    pub fn seal(&mut self) -> Result<(), PreparedAcquisitionBankError> {
        if self.sealed || self.slots.len() != self.required {
            return Err(PreparedAcquisitionBankError::Population);
        }
        self.sealed = true;
        Ok(())
    }
    /// Borrowed matching preserves wrapper identity and leaves all slots intact
    /// on a foreign/missing request. Equal repeated requests retain multiplicity.
    pub fn acquire(
        &mut self,
        source: &dyn CheckpointSource,
        key: &str,
        selection: &TensorSelection,
        policy: ReadPolicy,
    ) -> Result<CheckpointLease, PreparedAcquisitionBankError> {
        if !self.sealed {
            return Err(PreparedAcquisitionBankError::Population);
        }
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| {
                slot.as_ref().is_some_and(|value| {
                    value.matches_source_request(source, key, selection, policy)
                })
            })
            .ok_or(PreparedAcquisitionBankError::NoDestination)?;
        slot.take()
            .expect("matched actual destination")
            .acquire()
            .map_err(PreparedAcquisitionBankError::Acquisition)
    }
    /// Observed slot capacity, independent of allocator/cache/source bounds.
    pub fn storage(&self) -> Option<PreparedAcquisitionStorage> {
        let (payload, arcs) = self.slots.iter().filter_map(Option::as_ref).try_fold(
            (0usize, 0usize),
            |(bytes, arcs), slot| {
                let (payload, count) = slot.payload_storage()?;
                Some((bytes.checked_add(payload)?, arcs.checked_add(count)?))
            },
        )?;
        Some(PreparedAcquisitionStorage {
            inline: Layout::new::<Self>(),
            owned_payload_capacity_bytes: payload,
            destination_arc_allocations: arcs,
            requested_slots: Self::slot_layout(self.required)?,
            retained_slots: Self::slot_layout(self.slots.capacity())?,
            remaining: self.slots.iter().filter(|slot| slot.is_some()).count(),
        })
    }
}

#[cfg(test)]
mod tests;
