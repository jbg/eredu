//! Prepared forbidden-controller source and destination hooks. The ordinary
//! runtime remains the owner of tool policy and grammar activation semantics.
use super::*;
use eredu_core::speculative::ForbiddenControllerMutation;

impl ConstraintController {
    pub(in super::super) fn original_forbidden_source(
        &self,
    ) -> Option<&eredu_runtime::working_memory::OriginalForbiddenSource> {
        let ConstraintRuntime::PreparedForbidden { original, .. } = &self.runtime else {
            return None;
        };
        Some(original)
    }
    pub(in super::super) fn forbidden_source(&self) -> Option<ForbiddenControllerSource<'_>> {
        let ConstraintRuntime::PreparedForbidden {
            inputs,
            pending,
            original,
        } = &self.runtime
        else {
            return None;
        };
        let source = ForbiddenControllerSource::new(
            &self.committed_tokens,
            &self.validity,
            inputs,
            *pending,
        )
        .ok()?;
        Some(source.with_original_storage(eredu_core::OriginalSourceWitness::new(original)))
    }
    pub(in super::super) fn forbidden_copy_bytes(&self, capacity: usize) -> Option<usize> {
        let source = self.forbidden_source()?;
        if capacity < source.history().len() {
            return None;
        }
        let parts = [
            PlainControllerHistory::copy_metadata_bytes(capacity)?,
            ForbiddenControllerInputs::operation_control_bytes()?,
            size_of::<Self>(),
            size_of::<ConstraintRuntime>(),
            size_of::<SharedTokenFilter>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<ForbiddenControllerInputs>(),
            size_of::<eredu_runtime::working_memory::OriginalForbiddenSource>(),
            size_of::<Result<Self, ForbiddenControllerError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in super::super) fn copy_forbidden(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, ForbiddenControllerError> {
        let source = self
            .forbidden_source()
            .ok_or(ForbiddenControllerError::Unknown)?;
        let committed_tokens = self
            .committed_tokens
            .copy_prepared(capacity, host.clone())?;
        Ok(Self {
            runtime: ConstraintRuntime::PreparedForbidden {
                inputs: source.inputs().clone(),
                pending: source.prefix(),
                original: self
                    .original_forbidden_source()
                    .expect("forbidden source")
                    .clone(),
            },
            committed_tokens,
            validity: self.validity.clone(),
            authority: host,
            preparation: self.preparation.clone(),
        })
    }
    pub(in super::super) fn forbidden_mutation(
        &mut self,
    ) -> Result<ForbiddenControllerMutation<'_>, ForbiddenControllerError> {
        let ConstraintRuntime::PreparedForbidden {
            inputs, pending, ..
        } = &mut self.runtime
        else {
            return Err(ForbiddenControllerError::Unknown);
        };
        ForbiddenControllerMutation::new(
            &mut self.committed_tokens,
            &self.validity,
            inputs,
            pending,
        )
    }
}
