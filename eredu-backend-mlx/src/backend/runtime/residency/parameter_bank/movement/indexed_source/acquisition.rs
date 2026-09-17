//! Actual selected cache acquisition from completed IDs and explicit paid slots.
use super::*;
use crate::backend::runtime::residency::manager::OriginalResidencySlots;

impl OriginalIndexedChunkSource {
    /// One further parent completion is required for the grouped result.
    /// Its native traversal is supplied by the actual grouped child source.
    pub(crate) const fn acquisition_completions() -> usize { 1 }

    /// Consumes the already-prepared manager slots for these actual demands.
    /// This neither prepares a fallback source nor derives a grant from counts.
    pub(crate) fn acquire_from_slots(&self, demands: &IndexedDemandSource,
        entries: &[(ParameterBankKey, u64)], slots: &mut OriginalResidencySlots<'_>,
        stream: &Stream) -> Result<AcquiredParameterGroups, Error> {
        self.validate_parent(stream)?;
        let b = self.body();
        let census = b.identity.census;
        let completed = demands.source().and_then(|source| source.downcast_ref::<CompletedIds>())
            .ok_or_else(|| self.failure(Cause::Identity))?;
        if !completed.source.same_owner(&b.identity)
            || !demands.funding().is_some_and(|funding| funding.same_account(&b.funding))
            || !b.discovered.get() || !b.remapped.get() || b.acquire_attempted.get()
            || entries.is_empty() || entries.len() != demands.demands().len()
            || entries.len() > census.maximum_members() {
            return Err(self.failure(Cause::Identity));
        }
        let controls = [size_of::<AcquiredParameterGroups>(), size_of::<Result<AcquiredParameterGroups,Error>>(),
            size_of::<(Vec<ParameterBankKey>,Vec<u64>,Vec<(OffloadUnitId,u64)>)>(),
            size_of::<[u8;ParameterBankKey::unit_id_buffer_bytes()]>(),
            size_of::<(ResidencyManager,OriginalScopeObserver,[usize;8],[u64;4])>(),
            size_of::<OriginalResidencySlots<'_>>(), OriginalResidencySlots::control_bytes().ok_or_else(||self.failure(Cause::Overflow))?,
            size_of::<ResidentTransfer>(), size_of::<Result<ResidentTransfer,ResidencyError>>(),
            usize::try_from(ResidentTransfer::original_retirement_control_bytes()
                .ok_or_else(||self.failure(Cause::Overflow))?)
                .map_err(|_|self.failure(Cause::Overflow))?,
            OriginalScopeObserver::control_bytes().ok_or_else(||self.failure(Cause::Overflow))?,
            crate::backend::runtime::cache::value_completion_control_bytes(1).ok_or_else(||self.failure(Cause::Overflow))?,
            safemlx::OperationEvent::traversal_leaf_control_bytes().ok_or_else(||self.failure(Cause::Overflow))?,
            eredu_nn::Error::retained_source_control_bytes::<Failure>().ok_or_else(||self.failure(Cause::Overflow))?,
            Layout::array::<ParameterBankKey>(entries.len()).map_err(|_|self.failure(Cause::Overflow))?.size(),
            Layout::array::<u64>(entries.len()).map_err(|_|self.failure(Cause::Overflow))?.size(),
            Layout::array::<(OffloadUnitId,u64)>(entries.len()).map_err(|_|self.failure(Cause::Overflow))?.size(),
        ];
        let names = entries.iter().try_fold(0usize, |bytes,(key,_)| bytes.checked_add(key.unit_id_length()))
            .ok_or_else(||self.failure(Cause::Overflow))?;
        let bytes = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|bytes|bytes.checked_add(names)).ok_or_else(||self.failure(Cause::Overflow))?;
        b.funding.reserve_metadata(bytes).map_err(|error|self.failure(Cause::Funding(error)))?;
        // Validate global keys against the actual scoped bank's ordered local
        // domain in one traversal. The neutral provider retains the global/local
        // mapping; a different key order cannot use equal-shaped cache storage.
        let (manager, scratch_bytes) = b.identity.bank.with_workspace_source(&b.funding, |source| {
            let mut selected = 0usize;
            let mut members = 0usize;
            let mut maximum = 0u64;
            let mut scratch = 0u64;
            for (local, (key, bytes)) in source.unit_members(census.bank(), census.unit()).enumerate() {
                members = members.checked_add(1).ok_or_else(||self.failure(Cause::Overflow))?;
                maximum = maximum.max(bytes);
                if let Some(&(identity,count)) = demands.demands().get(selected) {
                    if identity == local {
                        if count == 0 || entries.get(selected) != Some(&(key,count)) {
                            return Err(self.failure(Cause::Identity));
                        }
                        scratch = scratch.checked_add(bytes).ok_or_else(||self.failure(Cause::Overflow))?;
                        selected += 1;
                    }
                }
            }
            let plan = AddressableChunkPlan::new(census.total_rows(), census.routes(), members, census.access(),
                Some(maximum), b.identity.bulk_target_bytes).map_err(|_|self.failure(Cause::Geometry))?;
            if !source.same_source(&b.identity.bank) || source.parameter_revision()!=b.identity.parameter_revision
                || plan != census.plan()
                || selected != entries.len() || scratch > source.scratch_bytes() {
                return Err(self.failure(Cause::Identity));
            }
            Ok::<_,Error>((source.manager().clone(), scratch))
        }).map_err(|error|self.failure(Cause::Bank(error)))??;
        let mut identities = Vec::new();
        let mut counts = Vec::new();
        let mut requests = Vec::new();
        identities.try_reserve_exact(entries.len()).map_err(|e|self.failure(Cause::Allocation(e)))?;
        counts.try_reserve_exact(entries.len()).map_err(|e|self.failure(Cause::Allocation(e)))?;
        requests.try_reserve_exact(entries.len()).map_err(|e|self.failure(Cause::Allocation(e)))?;
        for &(key,count) in entries {
            // Same finite canonical writer used by ordinary construction. Its
            // final String is covered by the exact name-length reservation.
            requests.push((key.unit_id(),count));
            identities.push(key);
            counts.push(count);
        }
        let pass = match census.access() { ParameterBankAccess::Bulk=>BankAccessClass::Bulk,
            ParameterBankAccess::Incremental=>BankAccessClass::Incremental,
            _=>return Err(self.failure(Cause::Geometry)) };
        // Every irreversible slot checkout follows identity/geometry/funding.
        // A failure leaves the attempt spent and its manager owns partial leases.
        if b.acquire_attempted.replace(true) { return Err(self.failure(Cause::Spent)); }
        let transfer = manager.acquire_many_with_original_transfer(&requests, MemoryTier::Device, slots, &b.observer)
            .map_err(|error|self.failure(Cause::Residency(error)))?;
        transfer.order_after_original(stream, slots.observations, &b.observer)
            .map_err(|error|self.failure(Cause::Residency(error)))?;
        self.validate_completion_source("acquired parameter context","acquired parameter bank")?;
        b.acquired.set(true);
        Ok(AcquiredParameterGroups { identities, demand:counts, scratch_bytes, pass, transfer, original:Some(self.clone()) })
    }

    pub(in crate::backend::runtime::residency::parameter_bank) fn validate_acquisition_bank(
        &self, bank: &SharedAddressableParameterBank) -> Result<(), Error> {
        let retained = &self.body().identity.bank;
        if !Arc::ptr_eq(&retained.inner, &bank.inner) || retained.scope != bank.scope {
            return Err(self.failure(Cause::Identity));
        }
        Ok(())
    }

    pub(in crate::backend::runtime::residency::parameter_bank) fn complete_acquisition(&self,
        mut acquisition: AcquiredParameterGroups, output: &MlxTensor, stream: &Stream) -> Result<(), Error> {
        self.validate_parent(stream)?;
        let b = self.body();
        if !b.acquired.get() || !b.bound.get() || b.completion_attempted.replace(true) { return Err(self.failure(Cause::Spent)); }
        self.validate_completion_source("computed output context","computed output bank")?;
        // Settle the input transfer's retained native owner before asking the
        // enclosing graph to start another nested traversal. Actual acquisition
        // pins remain alive through both transfer and output completion.
        acquisition.transfer.synchronize().map_err(|error|self.failure(Cause::Residency(error)))?;
        crate::backend::runtime::cache::complete_values([output.as_array()],stream)
            .map_err(|cause|self.failure(Cause::NativeCompletion { stage:"acquired output",cause }))?;
        safemlx::OperationEvent::validate_traversal_leaf(output.as_array(),&b.observer)
            .map_err(|error|self.failure(Cause::Native(error)))?;
        b.completed.set(true);
        // Both producer and dependent output have completed at this unlocked
        // boundary. Retire this exact application's pin collection before the
        // next compact chunk asks the same bounded cache for another member.
        // Failed or shared native owners retain their existing deferred path.
        acquisition.transfer.retire_completed_original();
        Ok(())
    }
}
