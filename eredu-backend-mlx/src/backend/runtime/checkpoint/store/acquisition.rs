//! Final original-source tickets and their storage-first accounting custody.
use super::*;
use crate::backend::runtime::residency::manager::acquisition_destinations::SelectedSourceOccurrence;
use eredu_checkpoint::store::{
    CheckpointSource, PreparedAcquisitionBank, PreparedAcquisitionBankError,
    PreparedAcquisitionRefusal, ReadPolicy, RetainedCheckpointSource, TensorReadRequest,
};
use eredu_runtime::working_memory::{
    OriginalHostMetadataCustody, OriginalHostSourceBank, OriginalHostSourceError,
    OriginalTextControlGuard, OriginalOperationMetadataCustody, WorkingMemoryError,
};
use std::{alloc::Layout, mem::size_of};

pub(crate) struct PreparedSourceAcquisitions {
    bank: PreparedAcquisitionBank,
    preparation_temporary_bytes: usize,
    custody: OriginalOperationMetadataCustody,
    metadata: Option<OriginalHostMetadataCustody>,
}
#[derive(Debug)]
enum FailureBody {
    Acquisition {
        cause: PreparedAcquisitionBankError,
        prefix: Option<PreparedAcquisitionBank>,
    },
    Handoff {
        cause: CheckpointMaterializationError,
        lease: CheckpointLease,
    },
    WeightHandoff {
        cause: CheckpointMaterializationError,
        lease: super::leases::RetainedAcquisitionSource,
    },
}
#[derive(Debug)]
enum FailureStorage {
    Refused(PreparedAcquisitionRefusal),
    Debit(OriginalHostSourceError),
    Owned(Box<FailureBody>),
}
/// Refused work owns the actual prefix/lease. Missing and exhausted requests
/// cannot create another storage owner or extend the accepted hold.
#[derive(Debug)]
pub struct PreparedSourceAcquisitionFailure {
    storage: FailureStorage,
    // Both are outside the Box: its fields and physical shell retire first.
    custody: Option<OriginalOperationMetadataCustody>,
    metadata: Option<OriginalHostMetadataCustody>,
}
impl PreparedSourceAcquisitionFailure {
    pub(super) fn weight_handoff(
        cause: CheckpointMaterializationError,
        lease: WeightLease,
        metadata: Option<OriginalHostMetadataCustody>,
    ) -> Self {
        Self {
            storage: FailureStorage::Owned(Box::new(FailureBody::WeightHandoff {
                cause,
                lease: lease.into_acquisition_source(),
            })),
            custody: None,
            metadata,
        }
    }
    fn fixed(error: PreparedAcquisitionBankError) -> Self {
        Self {
            storage: FailureStorage::Refused(
                error.refusal().expect("only fixed nonowning category"),
            ),
            custody: None,
            metadata: None,
        }
    }
    fn bank(
        cause: PreparedAcquisitionBankError,
        prefix: Option<PreparedAcquisitionBank>,
        custody: OriginalOperationMetadataCustody,
        metadata: Option<OriginalHostMetadataCustody>,
    ) -> Self {
        // Only real constructor/consumed-ticket failures own a new shell.
        Self {
            storage: FailureStorage::Owned(Box::new(FailureBody::Acquisition { cause, prefix })),
            custody: Some(custody),
            metadata,
        }
    }
    #[cfg(test)]
    pub(crate) fn reserve(
        cause: std::collections::TryReserveError,
        custody: &OriginalTextControlGuard,
    ) -> Self {
        Self::bank(
            PreparedAcquisitionBankError::Reserve(cause),
            None,
            custody.metadata_custody().into(),
            None,
        )
    }
    fn cause(&self) -> &(dyn std::error::Error + 'static) {
        match &self.storage {
            FailureStorage::Refused(cause) => cause,
            FailureStorage::Debit(cause) => cause,
            FailureStorage::Owned(body) => match body.as_ref() {
                FailureBody::Acquisition { cause, .. } => cause,
                FailureBody::Handoff { cause, .. } | FailureBody::WeightHandoff { cause, .. } => {
                    cause
                }
            },
        }
    }
    pub(crate) fn store_error(&self) -> Option<&StoreError> {
        match &self.storage {
            FailureStorage::Owned(body) => match body.as_ref() {
                FailureBody::Acquisition { cause, .. } => cause.store_error(),
                FailureBody::Handoff {
                    cause: CheckpointMaterializationError::Store(cause),
                    ..
                }
                | FailureBody::WeightHandoff {
                    cause: CheckpointMaterializationError::Store(cause),
                    ..
                } => Some(cause),
                _ => None,
            },
            FailureStorage::Refused(_) => None,
            FailureStorage::Debit(_) => None,
        }
    }
}
impl std::fmt::Display for PreparedSourceAcquisitionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.cause(), f)
    }
}
impl std::error::Error for PreparedSourceAcquisitionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause())
    }
}
impl PreparedSourceAcquisitions {
    /// A retained host/direct route cannot acquire a fresh checkpoint lease.
    /// Its sealed empty bank rejects an unexpected fallback before any callback.
    pub(crate) fn unavailable(
        custody: OriginalOperationMetadataCustody,
    ) -> Result<Self, PreparedSourceAcquisitionFailure> {
        let mut bank = PreparedAcquisitionBank::new(0).map_err(|cause| {
            PreparedSourceAcquisitionFailure::bank(cause, None, custody.clone(), None)
        })?;
        bank.seal().map_err(|cause| {
            PreparedSourceAcquisitionFailure::bank(cause, None, custody.clone(), None)
        })?;
        Ok(Self {
            bank,
            preparation_temporary_bytes: 0,
            custody,
            metadata: None,
        })
    }
    #[cfg(test)]
    pub(crate) fn record_preparation_temporary_bytes(&mut self, bytes: usize) {
        self.preparation_temporary_bytes = bytes;
    }
    #[cfg(test)]
    pub(crate) fn new(
        count: usize,
        custody: OriginalTextControlGuard,
    ) -> Result<Self, PreparedSourceAcquisitionFailure> {
        let custody = custody.metadata_custody().into();
        match PreparedAcquisitionBank::new(count) {
            Ok(bank) => Ok(Self {
                bank,
                preparation_temporary_bytes: 0,
                custody,
                metadata: None,
            }),
            Err(cause) => Err(PreparedSourceAcquisitionFailure::bank(
                cause, None, custody, None,
            )),
        }
    }
    #[cfg(test)]
    pub(crate) fn prepare_next(
        &mut self,
        source: impl Into<RetainedCheckpointSource>,
        request: TensorReadRequest,
    ) -> Result<(), PreparedSourceAcquisitionFailure> {
        self.bank
            .prepare_next(source, request)
            .map_err(|cause| self.failure(cause))
    }
    pub(crate) fn seal(&mut self) -> Result<(), PreparedSourceAcquisitionFailure> {
        self.bank.seal().map_err(|cause| self.failure(cause))
    }
    fn failure(&self, cause: PreparedAcquisitionBankError) -> PreparedSourceAcquisitionFailure {
        if matches!(
            cause,
            PreparedAcquisitionBankError::Population
                | PreparedAcquisitionBankError::NoDestination
                | PreparedAcquisitionBankError::Unavailable
                | PreparedAcquisitionBankError::Overflow
        ) {
            PreparedSourceAcquisitionFailure::fixed(cause)
        } else {
            PreparedSourceAcquisitionFailure::bank(
                cause,
                None,
                self.custody.clone(),
                self.metadata.clone(),
            )
        }
    }

    pub(crate) fn acquire(
        &mut self,
        source: &dyn CheckpointSource,
        key: &str,
        selection: &TensorSelection,
    ) -> Result<AcquiredSourceLease, PreparedSourceAcquisitionFailure> {
        let lease = self
            .bank
            .acquire(source, key, selection, ReadPolicy::RequireBounded)
            .map_err(|cause| self.failure(cause))?;
        Ok(AcquiredSourceLease {
            lease,
            metadata: self.metadata.clone(),
        })
    }
    /// Exact cumulative final tickets plus a conservative reached failure per
    /// consumed ticket. Existing shared source/header owners are not duplicated.
    pub(crate) fn selected_storage_bytes(
        rows: &[SelectedSourceOccurrence],
        count: usize,
    ) -> Result<u64, WorkingMemoryError> {
        let mut bytes = Self::control_bytes(count).ok_or(WorkingMemoryError::Overflow)?;
        let mut actual = 0usize;
        for row in rows {
            let layout = row.acquisition;
            actual = actual
                .checked_add(row.multiplicity)
                .ok_or(WorkingMemoryError::Overflow)?;
            let handoff =
                super::leases::original_selection_error_bytes(layout.diagnostic_key_bytes())
                    .ok_or(WorkingMemoryError::Overflow)?;
            let ticket = layout
                .retained_bytes()
                .checked_add(layout.refusal_bytes().max(handoff))
                .and_then(|n| n.checked_add(layout.control_bytes()))
                .ok_or(WorkingMemoryError::Overflow)?;
            let ticket = OriginalHostMetadataCustody::fresh_clone_storage_bytes(
                Layout::array::<u8>(ticket).map_err(|_| WorkingMemoryError::Overflow)?,
            )?;
            bytes = bytes
                .checked_add(
                    ticket
                        .checked_mul(
                            u64::try_from(row.multiplicity)
                                .map_err(|_| WorkingMemoryError::Overflow)?,
                        )
                        .ok_or(WorkingMemoryError::Overflow)?,
                )
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        if actual != count {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Zero-row banks still require the same concrete allocator qualifier.
        OriginalHostMetadataCustody::boxed_storage_bytes(Layout::new::<FailureBody>())?;
        Ok(bytes)
    }
    /// Caller checked actual plan/window identity and partitioned this allowance
    /// from the same accepted paired bank. Debit precedes even slot reservation.
    pub(crate) fn prepare_selected(
        rows: &[SelectedSourceOccurrence],
        count: usize,
        bytes: u64,
        mut bank: OriginalHostSourceBank,
        custody: OriginalTextControlGuard,
    ) -> Result<Self, PreparedSourceAcquisitionFailure> {
        if !bank.belongs_to(&custody) {
            return Err(PreparedSourceAcquisitionFailure::fixed(
                PreparedAcquisitionBankError::NoDestination,
            ));
        }
        let custody: OriginalOperationMetadataCustody = custody.metadata_custody().into();
        // Stored source-owned layouts avoid a second provider walk here. A
        // caller cannot shorten the exact recipe while preserving its rows.
        match Self::selected_storage_bytes(rows, count) {
            Ok(required) if required == bytes => {}
            _ => {
                return Err(PreparedSourceAcquisitionFailure::fixed(
                    PreparedAcquisitionBankError::Population,
                ))
            }
        }
        let receipt = bank
            .try_debit(bytes)
            .map_err(|cause| PreparedSourceAcquisitionFailure {
                storage: FailureStorage::Debit(cause),
                custody: None, // Actual reached debit owns its own finite receipt.
                metadata: None,
            })?;
        // The raw owner exists before full-guard retirement, and is declared
        // before the slot allocation so every unwind releases storage first.
        let metadata = receipt.into_metadata();
        let mut slots = match PreparedAcquisitionBank::new(count) {
            Ok(slots) => slots,
            Err(cause) => {
                return Err(PreparedSourceAcquisitionFailure::bank(
                    cause,
                    None,
                    custody,
                    Some(metadata),
                ))
            }
        };
        for row in rows {
            for _ in 0..row.multiplicity {
                if let Err(cause) = slots.prepare_selected_gguf(&row.plan) {
                    return Err(PreparedSourceAcquisitionFailure::bank(
                        cause,
                        Some(slots),
                        custody,
                        Some(metadata),
                    ));
                }
            }
        }
        if let Err(cause) = slots.seal() {
            return Err(PreparedSourceAcquisitionFailure::bank(
                cause,
                Some(slots),
                custody,
                Some(metadata),
            ));
        }
        Ok(Self {
            bank: slots,
            preparation_temporary_bytes: 0,
            custody,
            metadata: Some(metadata),
        })
    }
    /// Final slot Vec and all named owning failure/carrier transports. Dynamic
    /// ticket fields are added by the genuine source recipe above.
    pub(crate) fn control_bytes(count: usize) -> Option<u64> {
        let failure = size_of::<FailureBody>()
            .checked_add(size_of::<OriginalOperationMetadataCustody>())?
            .checked_add(size_of::<std::fmt::Arguments<'static>>())?
            .checked_add(size_of::<Result<(), std::fmt::Error>>())?
            .checked_add(size_of::<String>())?
            .checked_add(size_of::<PreparedSourceAcquisitionFailure>())?
            .checked_add(size_of::<AcquiredSourceLease>())?
            .checked_add(size_of::<
                Result<AcquiredSourceLease, PreparedSourceAcquisitionFailure>,
            >())?;
        let bytes = PreparedAcquisitionBank::slot_layout(count)?
            .size()
            .checked_add(size_of::<Self>())?
            .checked_add(failure.checked_mul(count.checked_add(1)?)?)?;
        u64::try_from(bytes).ok()
    }
    #[cfg(test)]
    pub(crate) fn remaining(&self) -> usize {
        self.bank.storage().unwrap().remaining
    }
}

/// No Clone or raw-lease extraction. Only the original pending handoff consumes
/// this source. Ordinary public WeightLease remains unchanged.
#[derive(Debug)]
pub(crate) struct AcquiredSourceLease {
    lease: CheckpointLease,
    metadata: Option<OriginalHostMetadataCustody>,
}
impl AcquiredSourceLease {
    fn failure(self, cause: CheckpointMaterializationError) -> CheckpointMaterializationError {
        let Self { lease, metadata } = self;
        if metadata.is_none() {
            // Explicit legacy/partial test banks do not claim funded G4.
            // Preserve their original error shape and ordinary retirement.
            return cause;
        }
        PreparedSourceAcquisitionFailure {
            storage: FailureStorage::Owned(Box::new(FailureBody::Handoff { cause, lease })),
            custody: None,
            metadata,
        }
        .into()
    }
    pub(crate) fn prepare<'context>(
        self,
        context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
        source_stream: &Stream,
        execution_stream: &Stream,
        slots: &mut materialization::OriginalMaterializationSlots<'_>,
        observer: &safemlx::OriginalScopeObserver,
        borrowed: bool,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let context = context.into();
        let selected = match WeightLease::checkpoint_byte_len_original(&self.lease) {
            Ok(value) => value,
            Err(cause) => return Err(self.failure(cause)),
        };
        if let Err(cause) = materialization::validate_operation(observer) {
            return Err(self.failure(cause));
        }
        let ready = match slots.pending_weights.checkout() {
            Ok(ready) => ready,
            Err(cause) => {
                return Err(self.failure(
                    CheckpointMaterializationError::OriginalOperationCapacity {
                        family: "pending weight",
                        prepared: cause.prepared,
                    },
                ))
            }
        };
        self.finish(
            context,
            source_stream,
            execution_stream,
            ready,
            observer,
            borrowed,
            selected,
        )
    }
    fn finish<'context>(
        self,
        context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
        source_stream: &Stream,
        execution_stream: &Stream,
        mut ready: materialization::PreparedPendingWeight,
        observer: &safemlx::OriginalScopeObserver,
        borrowed: bool,
        selected: usize,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let context = context.into();
        let Self { lease, metadata } = self;
        ready.retain_acquisition_metadata(metadata);
        let lease = WeightLease::from_validated_checkpoint_lease(
            lease,
            context.converted_groups.clone(),
            selected,
        );
        let original = Some((ready, observer.clone()));
        if borrowed {
            lease.prepare_borrowed_materialization_impl(source_stream, original)
        } else {
            lease.prepare_materialization_impl(source_stream, execution_stream, original)
        }
    }
    #[cfg(test)]
    pub(in crate::backend::runtime::checkpoint::store) fn address(
        &self,
    ) -> *const NeutralGgufLease {
        let CheckpointLease::Gguf(lease) = &self.lease else {
            panic!("GGUF fixture")
        };
        std::ptr::from_ref(lease.as_ref())
    }
    #[cfg(test)]
    pub(in crate::backend::runtime::checkpoint::store) fn prepare_fixture(
        self,
        context: &MlxParameterMaterializationContext,
        stream: &Stream,
        ready: materialization::PreparedPendingWeight,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let selected = match WeightLease::checkpoint_byte_len_original(&self.lease) {
            Ok(value) => value,
            Err(cause) => return Err(self.failure(cause)),
        };
        if let Err(cause) = materialization::validate_operation(observer) {
            return Err(self.failure(cause));
        }
        self.finish(context, stream, stream, ready, observer, false, selected)
    }
}
