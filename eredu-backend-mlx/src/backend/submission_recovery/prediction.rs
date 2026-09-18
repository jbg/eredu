//! One consumed original role, using the same prepared/active recovery engine.
use super::{
    PreparedRecovery, PreparedRecoveryError, Recovery, RecoveryPreparationError, Retention,
};
use crate::backend::error::Error;
use eredu_runtime::working_memory::{
    OriginalPredictionNativeCustody, OriginalPredictionRecoveryCustody,
    OriginalPredictionScopeRole, OriginalTextControlGuard,
};

mod setup;
pub(crate) use setup::validate_layouts;
pub(crate) use setup::{PredictionSetupCause, PredictionSetupFailure};

pub(crate) trait PredictionRetention: Retention {
    // Private concrete implementations only move into their previously empty,
    // last-field slot. No callback, allocation, source or native access.
    fn install_prediction_custody(&mut self, custody: OriginalPredictionRecoveryCustody);
    // Only the actual model-resource retention accepts a page source role.
    fn install_paged_scope(&mut self, _scope: crate::backend::nn::workspace::PagedScopeRetention) -> Result<(), Error> {
        Err(Error::PrefillScopeUnavailable)
    }
}

/// Native association of one genuine neutral role with its same original arena.
type OperationRegistration = Option<(
    crate::backend::runtime::execution::generic::OriginalOperationRegistration,
    eredu_runtime::working_memory::InferenceRequest,
)>;

#[derive(Debug)]
pub(crate) struct PredictionRole {
    host_sequence: bool,
    operations: OperationRegistration,
    paged: Option<(crate::backend::nn::workspace::ProjectedPagedSources, u64)>,
    native_storage: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    sampling: Option<crate::backend::nn::workspace::ResidentCompletionRecipe>,
    sampling_source: Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalSamplingSource>,
    original: OriginalPredictionScopeRole,
    quota: Option<safemlx::SubmissionRecordQuota>,
    graph: Option<safemlx::SubmissionGraphQuota>,
    controls: OriginalTextControlGuard,
}
impl PredictionRole {
    pub(crate) fn take_sampling_source(&mut self) -> Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalSamplingSource> {
        self.sampling_source.take()
    }
    fn with_sampling_source(mut self, source: Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalSamplingSource>) -> Self {
        self.sampling_source = source;
        self
    }
    pub(crate) fn is_host_sequence(&self) -> bool {
        self.host_sequence
    }
    fn with_host_sequence(mut self, host_sequence: bool) -> Self {
        self.host_sequence = host_sequence;
        self
    }
    #[cfg(test)]
    pub(crate) fn requiring_native_controls(mut self) -> Self {
        self.host_sequence = false;
        self
    }
    pub(crate) fn sampling_recipe(
        &self,
    ) -> Option<crate::backend::nn::workspace::ResidentCompletionRecipe> {
        self.sampling
    }
    fn with_sampling(
        mut self,
        recipe: Option<crate::backend::nn::workspace::ResidentCompletionRecipe>,
    ) -> Self {
        self.sampling = recipe;
        self
    }
    pub(crate) fn with_native_storage(
        mut self,
        bank: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    ) -> Self {
        self.native_storage = bank;
        self
    }
    pub(crate) fn control_guard(&self) -> &OriginalTextControlGuard {
        &self.controls
    }
    pub(crate) fn new(
        original: OriginalPredictionScopeRole,
        quota: Option<safemlx::SubmissionRecordQuota>,
        graph: Option<safemlx::SubmissionGraphQuota>,
        controls: OriginalTextControlGuard,
    ) -> Self {
        Self {
            host_sequence: false,
            operations: None,
            paged: None,
            native_storage: None,
            sampling: None,
            sampling_source: None,
            original,
            quota,
            graph,
            controls,
        }
    }
    fn with_paged(mut self, paged: Option<(crate::backend::nn::workspace::ProjectedPagedSources, u64)>) -> Self {
        self.paged = paged;
        self
    }
    fn with_operations(mut self, operations: OperationRegistration) -> Self {
        self.operations = operations;
        self
    }
    pub(crate) fn into_custody(
        self,
    ) -> (
        OriginalPredictionNativeCustody,
        OriginalPredictionRecoveryCustody,
        Option<safemlx::SubmissionRecordQuota>,
        Option<safemlx::SubmissionGraphQuota>,
    ) {
        let (native, recovery) = self.original.into_custody();
        (native, recovery, self.quota, self.graph)
    }
}
#[derive(Debug)]
struct SamplingRoles {
    sampling: Option<PredictionRole>,
    event: Option<PredictionRole>,
}
#[derive(Debug)]
pub(crate) struct PredictionSet {
    replacement: Option<SamplingRoles>,
    host_sequence: bool,
    operations: OperationRegistration,
    paged: Option<(crate::backend::nn::workspace::ProjectedPagedSources, u64)>,
    native_storage: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    sampling: Option<crate::backend::nn::workspace::ResidentCompletionRecipe>,
    sampling_source: Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalSamplingSource>,
    original: eredu_runtime::working_memory::OriginalTextPredictionScopeSet,
    quota: Option<safemlx::SubmissionRecordQuota>,
    graph: Option<safemlx::SubmissionGraphQuota>,
    controls: OriginalTextControlGuard,
}
impl PredictionSet {
    /// Both finite neutral banks have already authenticated the same active
    /// step. Keep only the two replacement roles; model roles remain original.
    pub(crate) fn with_sampling_replacement(mut self, mut replacement: Self)
        -> Result<Self, eredu_runtime::working_memory::WorkingMemoryError> {
        if self.replacement.is_some() { return Err(eredu_runtime::working_memory::WorkingMemoryError::AlreadyStarted); }
        replacement.operations = self.operations.clone();
        self.replacement = Some(SamplingRoles {
            sampling: Some(replacement.take_sampling()?),
            event: Some(replacement.take_sampling_event()?),
        });
        Ok(self)
    }
    pub(crate) fn with_sampling_source(mut self, source: Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalSamplingSource>) -> Self {
        self.sampling_source = source;
        self
    }
    pub(crate) fn with_paged_sources(mut self, paged: Option<crate::backend::nn::workspace::ProjectedPagedSources>, attempt: u64) -> Self {
        self.paged = if attempt == 0 { None } else { paged.map(|source| (source, attempt)) };
        self
    }
    // Only the validated quote factory selects this no-native-arena mode.
    // Direct role/set constructors continue to require original native controls.
    pub(crate) fn for_host_sequence(
        original: eredu_runtime::working_memory::OriginalTextPredictionScopeSet,
        controls: OriginalTextControlGuard,
    ) -> Self {
        let mut set = Self::new(original, None, None, controls);
        set.host_sequence = true;
        set
    }
    pub(crate) fn with_sampling(
        mut self,
        recipe: Option<crate::backend::nn::workspace::ResidentCompletionRecipe>,
    ) -> Self {
        self.sampling = recipe;
        self
    }
    pub(crate) fn with_native_storage(
        mut self,
        bank: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    ) -> Self {
        self.native_storage = bank;
        self
    }
    pub(crate) fn new(
        original: eredu_runtime::working_memory::OriginalTextPredictionScopeSet,
        quota: Option<safemlx::SubmissionRecordQuota>,
        graph: Option<safemlx::SubmissionGraphQuota>,
        controls: OriginalTextControlGuard,
    ) -> Self {
        Self {
            replacement: None,
            host_sequence: false,
            operations: None,
            paged: None,
            native_storage: None,
            sampling: None,
            sampling_source: None,
            original,
            quota,
            graph,
            controls,
        }
    }
    pub(crate) fn with_operations(
        mut self,
        operations: Option<
            crate::backend::runtime::execution::generic::OriginalOperationRegistration,
        >,
        request: eredu_runtime::working_memory::InferenceRequest,
    ) -> Self {
        self.operations = operations.map(|slot| (slot, request));
        self
    }
    pub(crate) fn take_model_execution(
        &mut self,
    ) -> Result<PredictionRole, eredu_runtime::working_memory::WorkingMemoryError> {
        self.original.take_model_execution().map(|original| {
            PredictionRole::new(
                original,
                self.quota.clone(),
                self.graph.clone(),
                self.controls.clone(),
            )
            .with_host_sequence(self.host_sequence)
            .with_operations(self.operations.clone())
            .with_paged(self.paged.clone())
            .with_native_storage(self.native_storage.clone())
        })
    }
    pub(crate) fn take_sampling(
        &mut self,
    ) -> Result<PredictionRole, eredu_runtime::working_memory::WorkingMemoryError> {
        if let Some(replacement) = &mut self.replacement {
            return replacement.sampling.take().ok_or(eredu_runtime::working_memory::WorkingMemoryError::AlreadyStarted);
        }
        self.original.take_sampling().map(|original| {
            PredictionRole::new(
                original,
                self.quota.clone(),
                self.graph.clone(),
                self.controls.clone(),
            )
            .with_host_sequence(self.host_sequence)
            .with_operations(self.operations.clone())
            .with_native_storage(self.native_storage.clone())
            .with_sampling(self.sampling)
        })
    }
    pub(crate) fn take_sampling_event(
        &mut self,
    ) -> Result<PredictionRole, eredu_runtime::working_memory::WorkingMemoryError> {
        if let Some(replacement) = &mut self.replacement {
            return replacement.event.take().ok_or(eredu_runtime::working_memory::WorkingMemoryError::AlreadyStarted);
        }
        self.original.take_sampling_event().map(|original| {
            PredictionRole::new(
                original,
                self.quota.clone(),
                self.graph.clone(),
                self.controls.clone(),
            )
            .with_host_sequence(self.host_sequence)
            .with_operations(self.operations.clone())
            .with_native_storage(self.native_storage.clone())
            .with_sampling(self.sampling)
            .with_sampling_source(self.sampling_source.take())
        })
    }
    pub(crate) fn take_model_validation(
        &mut self,
    ) -> Result<PredictionRole, eredu_runtime::working_memory::WorkingMemoryError> {
        self.original.take_model_validation().map(|original| {
            PredictionRole::new(
                original,
                self.quota.clone(),
                self.graph.clone(),
                self.controls.clone(),
            )
            .with_host_sequence(self.host_sequence)
            .with_operations(self.operations.clone())
            .with_native_storage(self.native_storage.clone())
        })
    }
    pub(crate) fn take_token_scalar(
        &mut self,
    ) -> Result<PredictionRole, eredu_runtime::working_memory::WorkingMemoryError> {
        self.original.take_token_scalar().map(|original| {
            PredictionRole::new(
                original,
                self.quota.clone(),
                self.graph.clone(),
                self.controls.clone(),
            )
            .with_host_sequence(self.host_sequence)
            .with_operations(self.operations.clone())
            .with_native_storage(self.native_storage.clone())
        })
    }
}

pub(crate) fn begin<T: PredictionRetention>(
    role: Option<PredictionRole>,
    retention: T,
) -> Result<Recovery<T>, Error> {
    let Some(role) = role else {
        return Recovery::begin(retention).map_err(Into::into);
    };
    let controls = role.controls.clone();
    setup::begin_original(role, retention, &controls)
        .map_err(|cause| setup::failure(cause, controls))
}

pub(crate) fn begin_model<T: PredictionRetention>(
    role: Option<PredictionRole>,
    retention: T,
    preparation: Option<super::prefill::ModelExecutionPreparation>,
) -> Result<(Recovery<T>, Option<super::prefill::ModelExecutionOwner>), Error> {
    let Some(role) = role else {
        if preparation.is_some() {
            return Err(Error::PrefillScopeUnavailable);
        }
        return Recovery::begin(retention)
            .map(|r| (r, None))
            .map_err(Into::into);
    };
    let controls = role.controls.clone();
    setup::begin_original_with_model(role, retention, &controls, preparation)
        .map_err(|cause| setup::failure(cause, controls))
}

/// Match a retained observer against the actual accepted, still-current role.
/// This neither constructs another Scope nor grants submission authority.
pub(crate) fn owns_observer<T: PredictionRetention>(
    active: &Recovery<T>,
    observer: &safemlx::OriginalScopeObserver,
) -> bool {
    observer.belongs_to(
        active
            .node
            .as_ref()
            .expect("accepted prediction node")
            .node()
            .probe
            .as_ref()
            .expect("accepted prediction scope"),
    )
}

pub(crate) fn control_bytes<T: Retention>() -> Option<u64> {
    use std::mem::size_of;
    let ready = PreparedRecovery::<T, OriginalPredictionNativeCustody>::control_bytes()?;
    let local = [
        size_of::<PredictionRole>(),
        size_of::<Option<PredictionRole>>(),
        size_of::<Option<safemlx::SubmissionRecordQuota>>(), // destructured role custody
        size_of::<Option<safemlx::SubmissionGraphQuota>>(),
        size_of::<safemlx::SubmissionGraphQuota>(),
        size_of::<safemlx::SubmissionRecordQuota>(), // same-arena clone/constructor handoff
        size_of::<(
            OriginalPredictionNativeCustody,
            OriginalPredictionRecoveryCustody,
            Option<safemlx::SubmissionRecordQuota>,
            Option<safemlx::SubmissionGraphQuota>,
        )>(),
        size_of::<Option<OriginalPredictionRecoveryCustody>>(),
        size_of::<Result<Recovery<T>, Error>>(),
        size_of::<T>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    ready
        .checked_add(u64::try_from(local).ok()?)?
        .checked_add(setup::control_bytes()?)
}

#[cfg(test)]
pub(crate) mod test_counts {
    use std::cell::Cell;
    // Lexical per-thread observation only; no source, custody, or native authority.
    thread_local! { static COUNTS: Cell<Option<([usize; 5], [usize; 5])>> = const { Cell::new(None) }; }
    pub(crate) struct Calls;
    impl Calls {
        pub(crate) fn new() -> Self {
            COUNTS.with(|counts| assert!(counts.replace(Some(([0; 5], [0; 5]))).is_none()));
            Self
        }
        pub(crate) fn original(&self) -> [usize; 5] {
            COUNTS.with(|counts| counts.get().unwrap().0)
        }
        pub(crate) fn legacy(&self) -> [usize; 5] {
            COUNTS.with(|counts| counts.get().unwrap().1)
        }
    }
    impl Drop for Calls {
        fn drop(&mut self) {
            COUNTS.with(|counts| {
                counts.set(None);
            });
        }
    }
    pub(crate) fn record(role: usize, original: bool) {
        COUNTS.with(|counts| {
            if let Some(mut n) = counts.get() {
                if original {
                    n.0[role] += 1;
                } else {
                    n.1[role] += 1;
                }
                counts.set(Some(n));
            }
        });
    }
}
