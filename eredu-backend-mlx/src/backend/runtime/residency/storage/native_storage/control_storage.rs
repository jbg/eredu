//! Concrete managed owners of the direct budget/publication mechanism.
//! Scope/Graph/Record/producer and full Work source inventories are independent.
use super::*;
use std::mem::size_of;

#[derive(Debug, Clone, Copy)]
pub(crate) struct NativePublicationOwnerLayout {
    budget: u64,
    bank: u64,
    attachment: u64,
    publication_frames: u64,
    failure: u64,
}
impl NativePublicationOwnerLayout {
    /// Finite direct-mechanism owners. This excludes native payload P, Scope and
    /// arenas, Data's inline sidecar list, full Work maps and source key payloads.
    /// The mutable constructor/selected producer must supply those independently.
    pub(crate) fn control_bytes(self, attempts: usize, rows: usize) -> Option<u64> {
        let attempts = u64::try_from(attempts).ok()?;
        let rows = u64::try_from(rows).ok()?;
        self.attachment
            .checked_mul(rows)?
            .checked_mul(attempts)?
            .checked_add(self.publication_frames.checked_mul(attempts)?)?
            .checked_add(self.failure.checked_mul(attempts.checked_add(1)?)?)?
            .checked_add(self.budget)?
            .checked_add(self.bank)
    }
}

impl MlxNativeStorage {
    /// Read the actual fixed node layouts from the linked runtime, and the
    /// qualified Rc bank layout from its neutral owner. No payload or native
    /// allowance is created; the separate source-key validator restricts keys
    /// to the representations covered by its finite clone-storage query.
    pub(crate) fn publication_owner_layout(
        &self,
        capacity: usize,
    ) -> Result<Option<NativePublicationOwnerLayout>, NativeStorageCause> {
        let runtime = match &self.runtime {
            Ok(runtime) => runtime,
            Err(cause) => return Err(NativeStorageCause::Cold(cause.retained())),
        };
        let Some(bank) = Bank::shared_borrowed_owner_bytes() else {
            return Ok(None);
        };
        let budget =
            PreparedOriginalBufferBudget::<OriginalNativeBudgetCustody>::layout(runtime, capacity)
                .map_err(NativeStorageCause::Fixed)?;
        let sidecar = PreparedAllocationOwner::<Registration>::layout();
        let sum = |parts: &[usize]| {
            parts
                .iter()
                .try_fold(0usize, |a, b| a.checked_add(*b))
                .and_then(|n| u64::try_from(n).ok())
        };
        let Some(budget) = budget
            .total_owner_bytes()
            .and_then(|n| u64::try_from(n).ok())
        else {
            return Err(NativeStorageCause::Fixed(
                OriginalBufferCause::InvalidLayout,
            ));
        };
        // All prepared sidecar heap nodes may survive the publication. The
        // observation, preparation and attachment loops call their workers
        // serially, however: none retains a worker's call frame per row. The
        // neutral plan separately prices the observation/attachment vectors.
        // Keep a complete frame suite per publication, including the failed
        // handoff; only the two actual heap nodes scale with the row count.
        let attachment = sidecar
            .allocation_bytes()
            .and_then(|heap| u64::try_from(heap).ok());
        let publication_frames = (|| {
            sum(&[
                sidecar.preparation_control_bytes(),
                sidecar.prepared_bytes(),
                sidecar.preparation_failure_bytes(),
                sidecar.attachment_failure_bytes(),
                sidecar.original_attachment_control_bytes(),
                OriginalBufferAliasWitness::inspection_control_bytes()?,
                OrdinaryBufferWitness::inspection_control_bytes()?,
                safemlx::ImmutableSourceWitness::inspection_control_bytes()?,
                safemlx::HostTransferArrayAliasWitness::inspection_control_bytes()?,
                size_of::<Observation<'static>>(),
                size_of::<Result<Observation<'static>, NativeStorageCause>>(),
                size_of::<NativeStorageObservation<StorageIdentity>>(),
                size_of::<MlxNativeStorage>(),
                size_of::<safemlx::OriginalBufferBudget>(),
            ])
        })();
        let failure = eredu_core::SharedBackendFailure::control_bytes::<PublicFailure>()
            .and_then(|shared| sum(&[
                shared,
                // Actual Box backing, independent of its inline pointer.
                size_of::<eredu_runtime::working_memory::NativeStorageError<NativeStorageCause>>(),
                size_of::<PublicFailure>(),
                size_of::<crate::backend::error::Error>(),
                size_of::<NativeStorageCause>(),
            ]));
        match (attachment, publication_frames, failure) {
            (Some(attachment), Some(publication_frames), Some(failure)) => {
                Ok(Some(NativePublicationOwnerLayout {
                    budget,
                    bank,
                    attachment,
                    publication_frames,
                    failure,
                }))
            }
            _ => Err(NativeStorageCause::Fixed(
                OriginalBufferCause::InvalidLayout,
            )),
        }
    }
}
