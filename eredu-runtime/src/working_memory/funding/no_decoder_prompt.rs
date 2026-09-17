//! One existing native scope for a prompt with no decoder table construction.

use super::*;

impl WorkingMemoryFundingRun {
    pub(in crate::working_memory) fn open_no_decoder_prompt_scope<
        K: Clone + Ord + Send + Sync + 'static,
    >(
        &self,
        reservation: &WorkingMemoryReservation,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<WorkingMemoryFundingScope, WorkingMemoryError> {
        if reservation.0.funding != Some(self.id) || !self.pool.same_domain(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Build/retire the pin bundle outside the usage lock. It belongs only
        // to native completion/quarantine; no fake zero-byte host owner exists.
        let pins = RegisteredStoragePin::aggregate(
            [
                Some(RegisteredStoragePin::new(complete_source.clone())),
                self.borrowed_storage.clone(),
            ]
            .into_iter()
            .flatten(),
        );
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &reservation.0.execution)?;
        complete_source.validate_copy_source(&self.pool, &usage)?;
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        state.validate_span_spend(None)?;
        let (scopes, native_scopes) = state.scope_counts_after(1, 1)?;
        // Only the scope count changes. The existing reservation, host holds,
        // physical charges, capacity and peak retain their original meaning.
        state.scopes = scopes;
        state.native_scopes = native_scopes;
        drop(usage);
        Ok(WorkingMemoryFundingScope {
            purpose: ScopePurpose::Native,
            pool: self.pool.clone(),
            id: self.id,
            active: true,
            borrowed_storage: Some(pins),
            capture_source: None,
            native_publication_identity: None,
        })
    }
}
