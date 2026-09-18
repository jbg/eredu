//! Two named original Scope attempts. No later output/copy role is inferred.
mod saved_resume;
use super::super::text_funding::{PreparedFundedWork, PreparedWorkRetention};
use super::super::{prepare_text_prompt_input, start_text_generation_with_sampler};
use super::token_input::PromptIds;
use super::*;
use crate::backend::submission_recovery::{
    self, PreparedRecovery, PreparedRecoveryError, RecoveryPreparationError,
};
use crate::composition::mlx::session::generation::MlxOrdinarySampler;
use eredu_runtime::working_memory::{
    InferencePreparationStage, InferenceSamplerCompletion, OriginalPreparationScopeCustody,
    OriginalTextPreparationScopes, RunOwnedTextSampler, TextPreparationScopeFacts,
};
use safemlx::SubmissionScopeOwnerCause;
use std::mem::size_of;

struct PreparedTextPreparationRetention {
    _request: InferenceRequest,
    funding: PreparedWorkRetention,
}
impl submission_recovery::Retention for PreparedTextPreparationRetention {
    fn observe(&self, status: Status) {
        self.funding.observe(status);
    }
}

type Ready = PreparedRecovery<PreparedTextPreparationRetention, OriginalPreparationScopeCustody>;
// The never-started retention is kept once even if queue-node allocation fails.
enum NativePreparation {
    Unallocated(
        PreparedTextPreparationRetention,
        OriginalPreparationScopeCustody,
    ),
    Prepared(Ready),
}
impl NativePreparation {
    // Existing isolated Recovery tests exercise the ordinary no-quota path.
    #[cfg(test)]
    fn begin(
        self,
    ) -> Result<
        submission_recovery::Recovery<PreparedTextPreparationRetention>,
        (SubmissionScopeOwnerCause, Self),
    > {
        self.begin_with_arenas(None, None)
    }
    fn begin_with_arenas(
        self,
        quota: Option<&safemlx::SubmissionRecordQuota>,
        graph: Option<&safemlx::SubmissionGraphQuota>,
    ) -> Result<
        submission_recovery::Recovery<PreparedTextPreparationRetention>,
        (SubmissionScopeOwnerCause, Self),
    > {
        let prepared = match self {
            Self::Prepared(prepared) => prepared,
            Self::Unallocated(retention, custody) => match Ready::new(retention, custody) {
                Ok(prepared) => prepared
                    .with_record_quota(quota.cloned())
                    .with_graph_quota(graph.cloned()),
                Err(RecoveryPreparationError {
                    cause,
                    retention,
                    custody,
                }) => return Err((cause, Self::Unallocated(retention, custody))),
            },
        };
        prepared
            .try_begin()
            .map_err(|PreparedRecoveryError { cause, pending }| (cause, Self::Prepared(pending)))
    }
}
enum Slot<T> {
    Unclaimed,
    Pending(T),
    InFlight,
    Spent,
}
struct Checkout<'a, T> {
    slot: &'a RefCell<Slot<T>>,
    armed: bool,
}
impl<'a, T> Checkout<'a, T> {
    fn enter(slot: &'a RefCell<Slot<T>>) -> Result<(Self, Option<T>), Error> {
        let previous = {
            let mut slot = slot
                .try_borrow_mut()
                .map_err(|_| Error::PreparationScopeReentrant)?;
            match &*slot {
                Slot::InFlight => return Err(Error::PreparationScopeReentrant),
                Slot::Spent => return Err(Error::PreparationScopeUnavailable),
                _ => std::mem::replace(&mut *slot, Slot::InFlight),
            }
        };
        let pending = match previous {
            Slot::Pending(value) => Some(value),
            Slot::Unclaimed => None,
            _ => unreachable!(),
        };
        Ok((Self { slot, armed: true }, pending))
    }
    fn pending(mut self, pending: T) {
        let previous = self.slot.replace(Slot::Pending(pending));
        self.armed = false;
        drop(previous);
    }
    fn spend(mut self) {
        let previous = self.slot.replace(Slot::Spent);
        self.armed = false;
        drop(previous);
    }
}
impl<T> Drop for Checkout<'_, T> {
    fn drop(&mut self) {
        if self.armed {
            // A panic or terminal failure cannot make a claimed role available.
            // The extracted pending owner is a separate local, outside this loan.
            let previous = self.slot.replace(Slot::Spent);
            drop(previous);
        }
    }
}
struct Prompt {
    ids: PromptIds,
    stage: InferencePreparationStage,
    native: Option<NativePreparation>,
    work: PreparedFundedWork,
}
struct Sampling {
    sampler: Option<RunOwnedTextSampler>,
    stage: InferenceSamplerCompletion,
    native: Option<NativePreparation>,
    work: PreparedFundedWork,
}
pub(in crate::composition::mlx::session::model_session) struct PreparationScopes {
    prompt: RefCell<Slot<Prompt>>,
    sampling: RefCell<Slot<Sampling>>,
    // Last: pending controls and host payloads precede the original full guard.
    bank: RefCell<OriginalTextPreparationScopes>,
}
impl std::fmt::Debug for PreparationScopes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparationScopes").finish_non_exhaustive()
    }
}
impl PreparationScopes {
    // Scalar-only evidence for the existing actual pending-owner fixture.
    #[cfg(test)]
    pub(super) fn pending_prompt_identities(&self) -> Option<(usize, usize, usize)> {
        let loan = self.prompt.borrow();
        let Slot::Pending(pending) = &*loan else {
            return None;
        };
        let Some(NativePreparation::Prepared(native)) = &pending.native else {
            return None;
        };
        Some((
            pending.ids.tokens().as_ptr() as usize,
            pending.work.identity(),
            native.allocation_identity(),
        ))
    }
    pub(super) fn new(bank: OriginalTextPreparationScopes) -> Self {
        Self {
            prompt: RefCell::new(Slot::Unclaimed),
            sampling: RefCell::new(Slot::Unclaimed),
            bank: RefCell::new(bank),
        }
    }
    pub(in crate::composition::mlx::session::model_session) fn has_pending_prompt(
        &self,
    ) -> Result<bool, eredu_core::TokenInputRejection> {
        use eredu_core::TokenInputRejection as R;
        let slot = self.prompt.try_borrow().map_err(|_| R::Busy)?;
        match &*slot {
            Slot::Pending(_) => Ok(true),
            Slot::Unclaimed => Ok(false),
            Slot::InFlight => Err(R::Busy),
            Slot::Spent => Err(R::Unavailable),
        }
    }
    pub(in crate::composition::mlx::session::model_session) fn prompt(
        &self,
        quote: &TextExecutionQuote,
        backend: &MlxBackend<'_>,
        ids: Option<PromptIds>,
        preparation: &InferenceTextPreparation,
    ) -> Result<MlxModelInput, Error> {
        let validated = { self.bank.borrow().validate_preparation(preparation) };
        validated.map_err(memory)?;
        let (checkout, pending) = Checkout::enter(&self.prompt)?;
        let mut pending = match pending {
            Some(pending) => {
                if ids
                    .as_ref()
                    .is_some_and(|ids| ids.tokens() != pending.ids.tokens())
                {
                    checkout.pending(pending);
                    return Err(Error::PreparationScopeInputMismatch);
                }
                // Original backing stays in the pending owner; the new caller
                // input is destroyed outside all slot/bank loans.
                drop(ids);
                pending
            }
            None => {
                let ids = ids.ok_or(Error::PreparationScopeUnavailable)?;
                let claimed = { self.bank.borrow_mut().claim_prompt(preparation) };
                let (stage, custody) = claimed.map_err(memory)?;
                let work = quote.prepared_preparation_work()?;
                let native = NativePreparation::Unallocated(
                    PreparedTextPreparationRetention {
                        _request: preparation.request().clone(),
                        funding: work.retention(),
                    },
                    custody,
                );
                Prompt {
                    ids,
                    stage,
                    native: Some(native),
                    work,
                }
            }
        };
        pending.ids.validate(quote, preparation)?;
        let validated = {
            self.bank
                .borrow()
                .validate_prompt(preparation, &pending.stage)
        };
        validated.map_err(memory)?;
        let mut recovery = match pending
            .native
            .take()
            .expect("pending native preparation")
            .begin_with_arenas(quote.record_quota.as_ref(), quote.graph_quota.as_ref())
        {
            Ok(recovery) => recovery,
            Err((cause, native)) => {
                pending.native = Some(native);
                checkout.pending(pending);
                return Err(Error::PreparationScope(cause));
            }
        };
        if let Some(bank) = quote.native_storage_bank() {
            let controls = quote.original_controls().ok_or_else(unknown)?;
            recovery.configure_scope(|scope| bank.configure_preparation_scope(scope, &controls))?;
        }
        checkout.spend();
        let Prompt {
            ids, stage, work, ..
        } = pending;
        let work = work.activate();
        let prompt = submission_recovery::detached_with_recovery(
            recovery,
            || {
                prepare_text_prompt_input(
                    backend,
                    ids,
                    preparation.request(),
                    &work,
                    quote.native_storage.is_some(),
                )
            },
            |cause| cause,
        )?;
        stage.finish().map_err(memory)?;
        Ok(prompt)
    }
    pub(in crate::composition::mlx::session::model_session) fn sampling(
        &self,
        quote: &TextExecutionQuote,
        config: TextGenerationConfig,
        preparation: &InferenceTextPreparation,
    ) -> Result<MlxTextGenerationState, Error> {
        if quote.config() != config {
            return Err(Error::PreparationScopeInputMismatch);
        }
        let validated = { self.bank.borrow().validate_preparation(preparation) };
        validated.map_err(memory)?;
        let (checkout, pending) = Checkout::enter(&self.sampling)?;
        let mut pending = match pending {
            Some(pending) => pending,
            None => {
                let claimed = { self.bank.borrow_mut().claim_sampling(preparation, config) };
                let (stage, custody) = claimed.map_err(memory)?;
                let (sampler, stage) = stage
                    .construct_sampler(quote.sampler_scope()?)
                    .map_err(memory)?;
                let work = quote.prepared_preparation_work()?;
                let native = NativePreparation::Unallocated(
                    PreparedTextPreparationRetention {
                        _request: preparation.request().clone(),
                        funding: work.retention(),
                    },
                    custody,
                );
                Sampling {
                    sampler: Some(sampler),
                    stage,
                    native: Some(native),
                    work,
                }
            }
        };
        let validated = {
            self.bank
                .borrow()
                .validate_sampling(preparation, &pending.stage)
        };
        validated.map_err(memory)?;
        let mut recovery = match pending
            .native
            .take()
            .expect("pending native preparation")
            .begin_with_arenas(quote.record_quota.as_ref(), quote.graph_quota.as_ref())
        {
            Ok(recovery) => recovery,
            Err((cause, native)) => {
                pending.native = Some(native);
                checkout.pending(pending);
                return Err(Error::PreparationScope(cause));
            }
        };
        if let Some(bank) = quote.native_storage_bank() {
            let controls = quote.original_controls().ok_or_else(unknown)?;
            recovery.configure_scope(|scope| bank.configure_preparation_scope(scope, &controls))?;
        }
        checkout.spend();
        let Sampling {
            mut sampler,
            stage,
            work,
            ..
        } = pending;
        let work = work.activate();
        let mut state = submission_recovery::detached_with_recovery(
            recovery,
            || {
                let _graph = quote.prepare_sampling_graph()?;
                let state = start_text_generation_with_sampler(
                    config,
                    Some(MlxOrdinarySampler::Funded(
                        sampler.take().expect("original sampler"),
                    )),
                )?;
                if let Some(random) = &state.sampling.prng {
                    work.retain(random.as_array());
                }
                Ok(state)
            },
            |cause| cause,
        )?;
        state
            .sampling
            .inference_retention
            .retain(preparation.request());
        stage.finish().map_err(memory)?;
        Ok(state)
    }
}
fn common_control_bytes() -> Result<u64, Error> {
    let scope = Ready::control_bytes().ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let prepared =
        PreparedFundedWork::control_bytes().ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let native_storage =
        crate::backend::runtime::residency::storage::native_storage::preparation_control_bytes()
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let common = [
        native_storage,
        size_of::<NativePreparation>(),
        size_of::<Option<NativePreparation>>(),
        size_of::<
            Result<
                submission_recovery::Recovery<PreparedTextPreparationRetention>,
                (SubmissionScopeOwnerCause, NativePreparation),
            >,
        >(),
        size_of::<PreparedTextPreparationRetention>(),
        size_of::<OriginalPreparationScopeCustody>(),
        size_of::<Option<&safemlx::SubmissionGraphQuota>>(), // graph begin borrow
        size_of::<Option<safemlx::SubmissionGraphQuota>>(),  // graph clone handoff
        size_of::<Option<&safemlx::SubmissionRecordQuota>>(), // begin borrow
        size_of::<Option<safemlx::SubmissionRecordQuota>>(), // same-arena clone handoff
    ]
    .into_iter()
    .try_fold(prepared, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
    .and_then(|n| scope.checked_add(n))
    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    Ok(common)
}

/// The same native preparation/recovery worker, with only an actual RNG key
/// destination. No sampler/history constructor is invoked by a reseed.
pub(super) fn reseed(
    request: &InferenceRequest,
    seed: u64,
    program: &crate::backend::nn::workspace::ResidentSamplingProgram,
    extension: &mut eredu_runtime::working_memory::OriginalTextSamplingExtension,
    bank: &crate::backend::runtime::residency::storage::native_storage::BankOwner,
    quota: &safemlx::SubmissionRecordQuota,
    graph: &safemlx::SubmissionGraphQuota,
) -> Result<RandomState, Error> {
    let custody = extension.claim_reseed().map_err(memory)?;
    let controls = extension.control_guard();
    let work = PreparedFundedWork::new_with_native(
        extension.funding().prepare_scope().map_err(memory)?, controls.clone(), None, None, Some(bank.clone()))?;
    let native = NativePreparation::Unallocated(PreparedTextPreparationRetention {
        _request: request.clone(), funding: work.retention(),
    }, custody);
    let mut recovery = native.begin_with_arenas(Some(quota), Some(graph))
        .map_err(|(cause, pending)| { drop(pending); Error::PreparationScope(cause) })?;
    recovery.configure_scope(|scope| bank.configure_preparation_scope(scope, &controls))?;
    let work = work.activate();
    let random = submission_recovery::detached_with_recovery(recovery, || {
        let layout = program.preparation_graph().flatten().ok_or_else(unknown)?;
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        let _graph = safemlx::OperationEvent::prepare_resident_graph(layout, &observer)?;
        let random = RandomState::with_seed(seed)?;
        work.retain(random.as_array());
        Ok(random)
    }, |cause| cause)?;
    if let Some(cause) = work.take_collection_failure() { return Err(cause); }
    Ok(random)
}
pub(super) fn reseed_control_bytes() -> Result<u64, Error> {
    let controls = [size_of::<RandomState>(), size_of::<Result<RandomState, Error>>(),
        size_of::<Option<safemlx::PreparedResidentGraph>>(), size_of::<safemlx::ResidentGraphLayout>(),
        size_of::<safemlx::OriginalScopeObserver>(), size_of::<PreparedFundedWork>(),
        size_of::<OriginalPreparationScopeCustody>(), size_of::<u64>()];
    let local = controls.into_iter().try_fold(std::mem::size_of_val(&controls), usize::checked_add)
        .and_then(|n| u64::try_from(n).ok()).ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    common_control_bytes()?.checked_add(local).ok_or_else(|| memory(WorkingMemoryError::Overflow))
}

/// Exactly Prompt + Sampling, independent of output count. Final stored fields
/// are measured by A; these are the two additional concrete role peaks.
pub(super) fn facts() -> Result<TextPreparationScopeFacts, Error> {
    let common = common_control_bytes()?;
    fn peak<T, O>(common: u64) -> Result<u64, Error> {
        // All are actual named local/argument/result representations. The
        // retained Slot itself is already inside the original quote's A.
        [
            size_of::<T>(),         // pending construction/destructuring
            size_of::<Option<T>>(), // extracted pending input
            size_of::<Slot<T>>(),   // replace result outside the slot loan
            size_of::<Checkout<'_, T>>(),
            size_of::<(Checkout<'_, T>, Option<T>)>(),
            size_of::<Result<(Checkout<'_, T>, Option<T>), Error>>(),
            size_of::<std::cell::RefMut<'_, Slot<T>>>(),
            size_of::<std::cell::RefMut<'_, OriginalTextPreparationScopes>>(),
            size_of::<std::cell::Ref<'_, OriginalTextPreparationScopes>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<O, Error>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .and_then(|n| common.checked_add(n))
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
    }
    Ok(TextPreparationScopeFacts::new(
        Some(
            peak::<Prompt, MlxModelInput>(common)?
                .max(saved_resume::prompt_control_bytes()?)
                .checked_add(
                    super::super::text_prompt_input_control_bytes()
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
                )
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        ),
        Some(
            peak::<Sampling, MlxTextGenerationState>(common)?
                .checked_add(
                    u64::try_from(
                        std::mem::size_of::<Option<safemlx::PreparedResidentGraph>>()
                            + std::mem::size_of::<
                                Result<Option<safemlx::PreparedResidentGraph>, Error>,
                            >()
                            + std::mem::size_of::<Option<safemlx::ResidentGraphLayout>>()
                            + std::mem::size_of::<
                                Result<Option<safemlx::ResidentGraphLayout>, Error>,
                            >(),
                    )
                    .map_err(|_| memory(WorkingMemoryError::Overflow))?,
                )
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        ),
    ))
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
pub(super) mod tests;

// Exact read-only pending query introduced by the original input route. The
// returned bool carries no owner; the underlying slot stays in the quote A.
pub(super) fn original_input_borrow_control_bytes() -> Option<usize> {
    size_of::<std::cell::Ref<'_, Slot<Prompt>>>()
        .checked_add(size_of::<Result<bool, eredu_core::TokenInputRejection>>())
}
