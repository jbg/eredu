//! Same original preparation roles for an independently saved-source resume.
use super::*;
use crate::backend::submission_recovery::{PreparedRecovery, Recovery};
use crate::composition::mlx::session::model_session::ScopeRetention;
use safemlx::{OperationEvent, OriginalScopeObserver, PreparedResidentGraph};

type SavedReady = PreparedRecovery<ScopeRetention, OriginalPreparationScopeCustody>;

impl TextExecutionQuote {
    pub(in crate::composition::mlx::session::model_session) fn saved_copy_population(
        &self,
    ) -> Option<crate::backend::array_copy::OriginalResumeCopyPopulation> {
        self.native_recipe
            .as_ref()
            .and_then(|recipe| recipe.resume_copy())
    }

    pub(in crate::composition::mlx::session::model_session) fn claim_saved_prompt(
        &self,
        preparation: &InferenceTextPreparation,
    ) -> Result<
        (
            InferencePreparationStage,
            Option<OriginalPreparationScopeCustody>,
        ),
        Error,
    > {
        let Some(scopes) = &self.preparation_scopes else {
            if self.original_controls().is_some() {
                return Err(unknown());
            }
            return preparation
                .claim_prompt()
                .map(|stage| (stage, None))
                .map_err(memory);
        };
        let (checkout, pending) = Checkout::enter(&scopes.prompt)?;
        if let Some(pending) = pending {
            checkout.pending(pending);
            return Err(Error::PreparationScopeInputMismatch);
        }
        let claimed = scopes
            .bank
            .try_borrow_mut()
            .map_err(|_| Error::PreparationScopeReentrant)?
            .claim_prompt(preparation)
            .map_err(memory)?;
        checkout.spend();
        Ok((claimed.0, Some(claimed.1)))
    }

    pub(in crate::composition::mlx::session::model_session) fn claim_saved_sampling(
        &self,
        preparation: &InferenceTextPreparation,
    ) -> Result<
        (
            InferencePreparationStage,
            Option<OriginalPreparationScopeCustody>,
        ),
        Error,
    > {
        let Some(scopes) = &self.preparation_scopes else {
            if self.original_controls().is_some() {
                return Err(unknown());
            }
            return preparation
                .claim_sampling(self.config())
                .map(|stage| (stage, None))
                .map_err(memory);
        };
        let (checkout, pending) = Checkout::enter(&scopes.sampling)?;
        if let Some(pending) = pending {
            checkout.pending(pending);
            return Err(Error::PreparationScopeInputMismatch);
        }
        let claimed = scopes
            .bank
            .try_borrow_mut()
            .map_err(|_| Error::PreparationScopeReentrant)?
            .claim_sampling(preparation, self.config())
            .map_err(memory)?;
        checkout.spend();
        Ok((claimed.0, Some(claimed.1)))
    }

    pub(in crate::composition::mlx::session::model_session) fn begin_saved_prompt(
        &self,
        custody: OriginalPreparationScopeCustody,
        retention: ScopeRetention,
    ) -> Result<Recovery<ScopeRetention>, Error> {
        let prepared = SavedReady::new(retention, custody)
            .map_err(|failure| Error::PreparationScope(failure.cause))?
            .with_record_quota(Some(
                self.record_quota.as_ref().ok_or_else(unknown)?.clone(),
            ))
            .with_graph_quota(Some(self.graph_quota.as_ref().ok_or_else(unknown)?.clone()));
        let mut recovery = prepared
            .try_begin()
            .map_err(|failure| Error::PreparationScope(failure.cause))?;
        let controls = self.original_controls().ok_or_else(unknown)?;
        let bank = self.native_storage_bank().ok_or_else(unknown)?;
        recovery.configure_scope(|scope| bank.configure_preparation_scope(scope, &controls))?;
        Ok(recovery)
    }

    pub(in crate::composition::mlx::session::model_session) fn with_saved_prompt_construction<T>(
        &self,
        worker: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        let Some(copy) = self
            .native_recipe
            .as_ref()
            .and_then(|recipe| recipe.resume_copy())
        else {
            if self.original_controls().is_some() {
                return Err(unknown());
            }
            return worker();
        };
        let observer = OriginalScopeObserver::require_current()?;
        let layout = copy.layout();
        let mut bank = OperationEvent::prepare_resident_graph(layout.graph, &observer)?;
        bank.configure_nested_completions(&layout.traversal, layout.completion_attempts)?;
        // Owning local drops on both return and unwind before the outer
        // SessionOperation can seal or queue recovery. No bank is restored
        // while an accepted nested worker still owns its same construction slot.
        let result = worker();
        drop(bank);
        result
    }
}

pub(super) fn prompt_control_bytes() -> Result<u64, Error> {
    let base = SavedReady::control_bytes().ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let controls = [
        std::mem::size_of::<SavedReady>(),
        std::mem::size_of::<Recovery<ScopeRetention>>(),
        std::mem::size_of::<ScopeRetention>(),
        std::mem::size_of::<OriginalPreparationScopeCustody>(),
        std::mem::size_of::<Option<OriginalPreparationScopeCustody>>(),
        std::mem::size_of::<
            Result<
                (
                    InferencePreparationStage,
                    Option<OriginalPreparationScopeCustody>,
                ),
                Error,
            >,
        >(),
        std::mem::size_of::<Result<Recovery<ScopeRetention>, Error>>(),
        std::mem::size_of::<PreparedResidentGraph>(),
        std::mem::size_of::<OriginalScopeObserver>(),
        std::mem::size_of::<safemlx::SubmissionGraphQuota>(),
        std::mem::size_of::<safemlx::SubmissionRecordQuota>(),
        std::mem::size_of::<eredu_runtime::working_memory::OriginalTextControlGuard>(),
        crate::backend::runtime::residency::storage::native_storage::preparation_control_bytes()
            .ok_or_else(unknown)?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    base.checked_add(u64::try_from(controls).map_err(|_| memory(WorkingMemoryError::Overflow))?)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}
