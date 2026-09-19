//! One exact built-in GGUF union, independent of request or native fit.
use super::{gguf_source::SourceAccount, qualified_storage, WorkingMemoryError, WorkingMemoryPool};
use eredu_checkpoint::store::{
    CompositeCheckpointSource, GgufCompositeBuildFailure, GgufCompositePlan,
    GgufCompositeStorageRequest,
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
struct CompositeCustody(SourceAccount);

/// The original pair or failed prefix, retained before its independent charge.
pub struct OriginalGgufCompositeError {
    accounting: Option<WorkingMemoryError>,
    construction: Option<GgufCompositeBuildFailure>,
    input: Option<GgufCompositePlan>,
    completed: Option<CompositeCheckpointSource>,
    account: Option<SourceAccount>,
}
impl OriginalGgufCompositeError {
    /// Qualification, origin or pre-admission refusal, without a construction.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// Actual refused union, including original stores and any completed rows.
    pub fn construction_failure(&self) -> Option<&GgufCompositeBuildFailure> {
        self.construction.as_ref()
    }
    /// The uncalled original pair on an admission or origin refusal.
    pub fn rejected_input(&self) -> Option<&GgufCompositePlan> {
        self.input.as_ref()
    }
}
impl fmt::Debug for OriginalGgufCompositeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalGgufCompositeError")
            .field("accounting", &self.accounting)
            .field("construction", &self.construction)
            .field("input", &self.input)
            .field("completed", &self.completed.is_some())
            .field("account", &self.account)
            .finish()
    }
}
impl fmt::Display for OriginalGgufCompositeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.construction {
            Some(cause) => fmt::Display::fmt(cause, f),
            None => fmt::Display::fmt(self.accounting.as_ref().expect("refusal"), f),
        }
    }
}
impl std::error::Error for OriginalGgufCompositeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.construction {
            Some(cause) => Some(cause),
            None => self.accounting.as_ref().map(|cause| cause as _),
        }
    }
}

impl WorkingMemoryPool {
    /// Exact fresh owner-directory/child-vector/initial recipe-PAL contribution.
    /// Existing input stores, public Arc erasure shells, future arbitrary recipe
    /// entries and manager/model construction remain separate contributions.
    pub fn gguf_composite_required_bytes(
        plan: &GgufCompositePlan,
    ) -> Result<u64, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let request = plan
            .requested_storage::<CompositeCustody>()
            .ok_or(WorkingMemoryError::Overflow)?;
        let mutex =
            super::fixed_baseline::pal_mutex_bytes().ok_or(WorkingMemoryError::UnknownBound)?;
        let controls = [
            size_of::<GgufCompositeStorageRequest>(),
            size_of::<Option<GgufCompositeStorageRequest>>(),
            size_of::<super::loaded_decode_source::Allowance>(),
            size_of::<SourceAccount>(),
            size_of::<CompositeCustody>(),
            size_of::<OriginalGgufCompositeError>(),
            size_of::<Result<CompositeCheckpointSource, OriginalGgufCompositeError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<WorkingMemoryError>>(),
            size_of::<std::array::IntoIter<&eredu_checkpoint::gguf_store::GgufWeightStore, 2>>(),
            size_of::<std::array::IntoIter<&eredu_checkpoint::store::RetainedCheckpointSource, 2>>(
            ),
            size_of::<u64>(),
            size_of::<Option<u64>>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        let account = SourceAccount::storage_bytes()?;
        u64::try_from(request.requested_bytes())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .checked_add(qualified_storage::shared_layout_bytes(
                request.custody_body(),
            )?)
            .and_then(|n| n.checked_add(account))
            .and_then(|n| n.checked_add(mutex.checked_mul(request.mutex_count() as u64)?))
            .and_then(|n| n.checked_add(controls as u64))
            .ok_or(WorkingMemoryError::Overflow)
    }

    /// Predebit before invoking the closed catalog-key worker. Both actual
    /// sources must already have original reader controls in this same pool.
    pub fn compile_gguf_composite(
        &self,
        plan: GgufCompositePlan,
    ) -> Result<CompositeCheckpointSource, OriginalGgufCompositeError> {
        let refused = |input, cause| OriginalGgufCompositeError {
            accounting: Some(cause),
            construction: None,
            input: Some(input),
            completed: None,
            account: None,
        };
        let validation = (|| {
            for source in plan.sources() {
                self.validate_gguf_source_controls(source)?;
            }
            if let Some(sources) = plan.retained_sources() {
                for source in sources {
                    self.validate_retained_source_controls(source)?;
                }
            }
            Ok(())
        })();
        if let Err(cause) = validation {
            return Err(refused(plan, cause));
        }
        let bytes = match Self::gguf_composite_required_bytes(&plan) {
            Ok(bytes) => bytes,
            Err(cause) => return Err(refused(plan, cause)),
        };
        let allowance = match self.admit_source_compiler(bytes) {
            Ok(allowance) => allowance,
            Err(cause) => return Err(refused(plan, cause)),
        };
        let account = allowance.into_source_account();
        match plan.build_with_custody(CompositeCustody(account.share())) {
            Err(construction) => Err(OriginalGgufCompositeError {
                accounting: None,
                construction: Some(construction),
                input: None,
                completed: None,
                account: Some(account),
            }),
            Ok(completed) => {
                if let Err(cause) = account.finish() {
                    return Err(OriginalGgufCompositeError {
                        accounting: Some(cause),
                        construction: None,
                        input: None,
                        completed: Some(completed),
                        account: Some(account),
                    });
                }
                Ok(completed)
            }
        }
    }

    /// Read-only origin comparison; an ordinary wrapper never becomes admitted.
    pub fn validate_gguf_composite_controls(
        &self,
        source: &CompositeCheckpointSource,
    ) -> Result<(), WorkingMemoryError> {
        let custody = source
            .constructor_control_owner::<CompositeCustody>()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if custody.0.matches_pool(self) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}

#[cfg(test)]
mod tests;
