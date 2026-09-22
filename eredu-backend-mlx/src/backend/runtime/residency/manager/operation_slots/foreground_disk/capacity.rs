//! One request's bounded live source backing, refunded at native destruction.
use super::*;
use eredu_runtime::working_memory::{
    HostSourcePeakSelection, OriginalHostMetadataCustody, OriginalHostSourceBank,
    OriginalHostSourcePeakCapacity, OriginalTextControlGuard,
};
use safemlx::{RetirementCapacity, RetirementCapacityCause, RetirementCapacityPermit};
use std::sync::{Arc, Mutex};

/// A clone shares only the same finite request capacity. Its native control
/// retires before request custody, including for the final unused slot.
#[derive(Clone)]
pub(crate) struct ForegroundDiskSourceCapacity {
    native: RetirementCapacity,
    selection: HostSourcePeakSelection,
    sources: Option<Arc<Mutex<OriginalHostSourceBank>>>,
    custody: eredu_runtime::working_memory::OriginalHostSourceCustody,
}
impl ForegroundDiskSourceCapacity {
    pub(crate) fn new(
        bytes: usize,
        custody: OriginalTextControlGuard,
        reservation: &WorkingMemoryReservation,
    ) -> Result<Self, WorkingMemoryError> {
        custody.validate_reservation(reservation)?;
        Ok(Self {
            native: RetirementCapacity::new(bytes),
            selection: HostSourcePeakSelection::new(bytes as u64)?,
            sources: None,
            custody: custody.into(),
        })
    }
    /// Production capacity consumes the same selected, accepted source bank.
    /// The bank is bound to this exact native owner before any slot can split it.
    pub(crate) fn with_source_bank(
        selection: HostSourcePeakSelection,
        bank: OriginalHostSourceBank,
        custody: OriginalTextControlGuard,
        reservation: &WorkingMemoryReservation,
    ) -> Result<Self, WorkingMemoryError> {
        custody.validate_reservation(reservation)?;
        Self::with_source_account(selection, bank, custody.into(), Some(reservation))
    }
    pub(crate) fn with_source_account(
        selection: HostSourcePeakSelection,
        mut bank: OriginalHostSourceBank,
        custody: eredu_runtime::working_memory::OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
    ) -> Result<Self, WorkingMemoryError> {
        custody.validate_account(reservation)?;
        bank.validate_source_peak_selection(selection, &custody)?;
        let mut value = Self {
            native: RetirementCapacity::new(
                usize::try_from(selection.backing_bytes())
                    .map_err(|_| WorkingMemoryError::Overflow)?,
            ),
            selection,
            sources: None,
            custody,
        };
        bank.bind_peak_capacity(&value)?;
        value.sources = Some(Arc::new(Mutex::new(bank)));
        Ok(value)
    }
    pub(crate) fn matches_source(
        &self,
        source: &super::super::super::ForegroundDiskDescriptors,
    ) -> bool {
        self.selection.same_producer(source.peak_selection(0))
    }
    pub(crate) fn source_bank(
        &self,
        bytes: u64,
        outputs: usize,
    ) -> Result<OriginalHostSourceBank, eredu_runtime::working_memory::HostDestinationCause> {
        use eredu_runtime::working_memory::HostDestinationCause;
        let sources = self.sources.as_ref().ok_or(HostDestinationCause::Memory(
            WorkingMemoryError::IdentityMismatch,
        ))?;
        sources
            .lock()
            .map_err(|_| HostDestinationCause::Memory(WorkingMemoryError::Poisoned))?
            .split(bytes, outputs)
    }
    pub(crate) fn validate_permit(
        &self,
        selection: HostSourcePeakSelection,
        owner: usize,
        bytes: u64,
        permit: &RetirementCapacityPermit,
    ) -> Result<(), WorkingMemoryError> {
        if selection != self.selection
            || owner != self.native.owner_identity()
            || !permit.belongs_to(&self.native)
            || u64::try_from(permit.bytes()).ok() != Some(bytes)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    pub(crate) fn validate_reservation(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_account(Some(reservation))
    }
    pub(crate) fn validate_account(
        &self,
        reservation: Option<&WorkingMemoryReservation>,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_account(reservation)
    }
    pub(crate) fn custody(&self) -> &eredu_runtime::working_memory::OriginalHostSourceCustody {
        &self.custody
    }
    pub(crate) fn try_acquire(
        &self,
        bytes: usize,
    ) -> Result<RetirementCapacityPermit, RetirementCapacityCause> {
        self.native.try_acquire(bytes)
    }
    /// The single shared budget allocation is priced once per request. Each
    /// read constructor separately prices its wrapper/permit transports.
    pub(crate) fn control_bytes() -> Option<usize> {
        let fixed = [
            size_of::<Self>(),
            usize::try_from(
                OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<
                    Mutex<OriginalHostSourceBank>,
                >())
                .ok()?,
            )
            .ok()?,
            size_of::<std::sync::MutexGuard<'_, OriginalHostSourceBank>>(),
            size_of::<std::sync::PoisonError<std::sync::MutexGuard<'_, OriginalHostSourceBank>>>(),
            size_of::<
                Result<OriginalHostSourceBank, eredu_runtime::working_memory::HostDestinationCause>,
            >(),
            size_of::<HostSourcePeakSelection>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<eredu_runtime::working_memory::OriginalHostSourceCustody>(),
            size_of::<&WorkingMemoryReservation>(),
            size_of::<Option<&WorkingMemoryReservation>>(),
            size_of::<(&Self, Option<&WorkingMemoryReservation>)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<(&Self, &WorkingMemoryReservation)>(),
        ];
        fixed.into_iter().try_fold(
            RetirementCapacity::control_bytes()?.checked_add(size_of_val(&fixed))?,
            usize::checked_add,
        )
    }
}

impl OriginalHostSourcePeakCapacity for ForegroundDiskSourceCapacity {
    fn selection(&self) -> HostSourcePeakSelection {
        self.selection
    }
    fn owner_identity(&self) -> usize {
        self.native.owner_identity()
    }
    fn backing_bytes(&self) -> u64 {
        self.native.capacity() as u64
    }
    fn source_custody(&self) -> eredu_runtime::working_memory::OriginalHostSourceCustody {
        self.custody.clone().into()
    }
}
