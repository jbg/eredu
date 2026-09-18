//! One original five-role bank, issued only for its genuine active TextOperation.
use super::*;
use crate::backend::submission_recovery::prediction::{
    PredictionRole as OriginalPredictionScopeRole, PredictionSet as OriginalTextPredictionScopeSet,
};
use eredu_runtime::working_memory::{
    InferenceTextStep, OriginalTextPredictionScopes, TextPredictionScopeFacts,
};
use std::mem::size_of;

pub(super) struct PredictionScopes {
    bank: RefCell<OriginalTextPredictionScopes>,
    quota: Option<safemlx::SubmissionRecordQuota>,
    graph: Option<safemlx::SubmissionGraphQuota>,
    controls: eredu_runtime::working_memory::OriginalTextControlGuard,
    native_storage: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
}
impl std::fmt::Debug for PredictionScopes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictionScopes").finish_non_exhaustive()
    }
}
impl PredictionScopes {
    pub(super) fn with_native_storage(
        mut self,
        bank: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    ) -> Self {
        self.native_storage = bank;
        self
    }
    pub(super) fn new(
        bank: OriginalTextPredictionScopes,
        quota: Option<safemlx::SubmissionRecordQuota>,
        graph: Option<safemlx::SubmissionGraphQuota>,
        controls: eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> Self {
        Self {
            bank: RefCell::new(bank),
            quota,
            graph,
            controls,
            native_storage: None,
        }
    }
    pub(super) fn claim(
        &self,
        step: &InferenceTextStep,
    ) -> Result<OriginalTextPredictionScopeSet, Error> {
        // The neutral bank authenticates its actual reservation and either the
        // original run or its sealed sampling-extension origin. Comparing the
        // extension guard to the old model reservation would reject that origin.
        let result = {
            let mut bank = self
                .bank
                .try_borrow_mut()
                .map_err(|_| Error::PredictionScopeReentrant)?;
            bank.claim(step)
        };
        result
            .map(|original| {
                if self.quota.is_none() && self.graph.is_none() && self.native_storage.is_none() {
                    OriginalTextPredictionScopeSet::for_host_sequence(
                        original,
                        self.controls.clone(),
                    )
                } else {
                    OriginalTextPredictionScopeSet::new(
                        original,
                        self.quota.clone(),
                        self.graph.clone(),
                        self.controls.clone(),
                    )
                    .with_native_storage(self.native_storage.clone())
                }
            })
            .map_err(memory)
    }
}

// The enclosing quote Rc already prices its final bank field. These are the
// distinct extraction/mapping locals before constructing that enclosing value.
pub(super) fn bank_transfer_bytes() -> Option<u64> {
    let bytes = [
        size_of::<PredictionScopes>(),
        size_of::<Option<PredictionScopes>>(),
        size_of::<Result<Option<OriginalTextPredictionScopes>, Error>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    u64::try_from(bytes)
        .ok()?
        .checked_add(CaptureQuotation::prediction_bank_transfer_bytes()?)
}
fn sum(parts: &[usize]) -> Option<u64> {
    parts
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
}
fn added(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    a.zip(b).and_then(|(a, b)| a.checked_add(b))
}

fn role_facts() -> Result<[Option<u64>; 5], Error> {
    crate::backend::submission_recovery::prediction::validate_layouts()?;
    use super::super::{ObservationRetention, ResourceOperation, ScopeRetention, SessionOperation};
    use crate::backend::MlxCompletion;
    use crate::backend::submission_recovery::{Recovery, prediction::control_bytes};
    type EventSubmission = eredu_core::Submission<Array, MlxCompletion>;
    // Every role has an actual take/return. Scalar's loan is in TokenObservation;
    // the other four borrow the remaining set in SubmissionResources.
    let set_take = sum(&[
        size_of::<Option<OriginalPredictionScopeRole>>(),
        size_of::<Result<Option<OriginalPredictionScopeRole>, Error>>(),
        size_of::<std::cell::RefMut<'_, Option<OriginalTextPredictionScopeSet>>>(),
    ]);
    let scalar_take = sum(&[
        size_of::<Option<OriginalPredictionScopeRole>>(),
        size_of::<Result<Option<OriginalPredictionScopeRole>, Error>>(),
        size_of::<std::cell::RefMut<'_, Option<OriginalPredictionScopeRole>>>(),
    ]);
    // One whole-bank checkout per prediction, assigned only to ModelExecution.
    let claim = sum(&[
        size_of::<std::cell::RefMut<'_, OriginalTextPredictionScopes>>(),
        size_of::<Result<OriginalTextPredictionScopeSet, Error>>(),
        size_of::<Result<Option<OriginalTextPredictionScopeSet>, Error>>(),
        size_of::<Result<Option<OriginalTextPredictionScopeSet>, Error>>(), // TextOperation forwarding
    ]);
    let model = added(
        added(control_bytes::<ScopeRetention>(), set_take),
        added(
            added(claim, super::super::completion_roots::control_bytes()),
            sum(&[
                size_of::<SessionOperation<'static>>(),
                size_of::<Result<SessionOperation<'static>, Error>>(),
            ]),
        ),
    );
    let sampling = added(
        added(control_bytes::<ScopeRetention>(), set_take),
        added(
            crate::backend::runtime::generation::original_sampling_control_bytes(),
            sum(&[
                size_of::<ResourceOperation>(),
                size_of::<Result<ResourceOperation, Error>>(),
                size_of::<Result<EventSubmission, Error>>(),
                size_of::<Result<(EventSubmission, Recovery<ScopeRetention>), Error>>(),
            ]),
        ),
    );
    let event = added(MlxCompletion::prediction_scope_control_bytes(), set_take);
    let validation = added(
        added(control_bytes::<ObservationRetention>(), set_take),
        sum(&[size_of::<Result<(), Error>>()]),
    );
    let scalar = added(
        added(
            added(
                super::super::super::output_completion::token_scalar_control_bytes(),
                scalar_take,
            ),
            super::super::super::output_completion::token_observation_control_bytes(),
        ),
        sum(&[size_of::<Result<u32, Error>>()]),
    );
    let known = |n: Option<u64>| {
        n.map(Some)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))
    };
    Ok([known(model)?, known(sampling)?, known(event)?, known(validation)?, known(scalar)?])
}

pub(super) fn facts() -> Result<TextPredictionScopeFacts, Error> {
    let [model, sampling, event, validation, scalar] = role_facts()?;
    Ok(TextPredictionScopeFacts::new(model, sampling, event, validation, scalar))
}
pub(super) fn sampling_facts() -> Result<(Option<u64>, Option<u64>), Error> {
    let [_, sampling, event, _, _] = role_facts()?;
    Ok((sampling, event))
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod tests;
