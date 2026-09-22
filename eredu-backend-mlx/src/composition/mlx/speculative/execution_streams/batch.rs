//! Closed finite assignment over already authenticated native request sources.
use super::*;
use eredu_core::SpeculativeRequestId;
use eredu_runtime::working_memory::WorkingMemoryError;

impl<'a> SpeculativeExecutionStreams<'a> {
    /// Bind once after all contexts and source owners have stable paid storage.
    /// The first request funds global table/coordination bookkeeping only; its
    /// occurrence/source authority cannot substitute for another lane.
    pub(crate) fn with_request_assignments(
        mut self,
        assignments: &'a [Self],
    ) -> Result<Self, Error> {
        let (first, _) = self
            .original_numerical()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let controls = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<(&Self, &[Self])>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<std::slice::Iter<'_, Self>>(),
            std::mem::size_of::<(usize, usize, SpeculativeRequestId)>(),
        ];
        first
            .metadata_funding()
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        eredu_nn::workspace::HostMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if assignments.is_empty()
            || self.batch_assignments.is_some()
            || self.batch_request.is_some()
            || self.embedded_invocation.is_some()
            || self.tensor_sources.is_some()
            || self.prefill_input.is_some()
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        for (index, value) in assignments.iter().enumerate() {
            let (source, environment) = value
                .original_numerical()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            if value.batch_assignments.is_some()
                || value.batch_request.is_some()
                || value.embedded_invocation.is_some()
                || value.tensor_sources.is_some()
                || value.prefill_input.is_some()
                || value.target != self.target
                || value.draft != self.draft
                || value.topology != self.topology
                || value.memory_owner.is_some()
                || value.capture.is_some()
                || !source.target_origin().same_origin(first.target_origin())
                || !environment.pool().same_ledger(first.pool())
                || (index == 0 && !std::ptr::eq(source, first))
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            if assignments[..index].iter().any(|prior| {
                std::ptr::eq(
                    prior
                        .original_numerical()
                        .expect("validated prefix")
                        .0
                        .request(),
                    source.request(),
                )
            }) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
        self.batch_assignments = Some(assignments);
        Ok(self)
    }
    /// Identity of this already selected assignment. The unselected outer
    /// context deliberately has no ID and cannot spend a lane occurrence.
    pub(crate) const fn selected_request(&self) -> Option<SpeculativeRequestId> {
        self.batch_request
    }
    /// Allocation-free projection by the shared table's existing insertion ID.
    /// All account/environment/cache-constructor checks ran before table binding.
    pub(crate) fn request_context(self, request: SpeculativeRequestId) -> Result<Self, Error> {
        match (self.batch_assignments, self.batch_request) {
            (Some(assignments), None) => {
                let mut selected = *assignments
                    .get(request.index())
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
                selected.batch_request = Some(request);
                Ok(selected)
            }
            (None, Some(selected)) if selected == request => Ok(self),
            (None, None) => Ok(self),
            _ => Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
        }
    }
}
