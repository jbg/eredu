//! Closed concrete Rc population; payload retirement follows allocation retirement.
use super::{
    Array, Error, ObservationRetention, ObservationRoots, Recovery, ScopeRetention,
    SubmissionResources,
};
use std::{
    alloc::Layout,
    cell::Cell,
    fmt,
    mem::{size_of, ManuallyDrop},
    ops::Deref,
    rc::{Rc, Weak},
};

/// No Rc/Weak export. Every strong exit uses the same consuming retirement.
/// No native completion, scope certification or new host hold is implied.
pub(in crate::composition::mlx::session) struct SubmissionResourcesOwner(
    Option<Rc<SubmissionResources>>,
);
impl SubmissionResourcesOwner {
    fn take_prediction_scope(
        &self,
        take: fn(
            &mut crate::backend::submission_recovery::prediction::PredictionSet,
        ) -> Result<
            crate::backend::submission_recovery::prediction::PredictionRole,
            eredu_runtime::working_memory::WorkingMemoryError,
        >,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionRole>, Error>
    {
        let result = {
            let mut slot = self
                .prediction_scopes
                .try_borrow_mut()
                .map_err(|_| Error::PredictionScopeReentrant)?;
            slot.as_mut().map(take).transpose()
        };
        result.map_err(|_| Error::PredictionScopeUnavailable)
    }
    pub(super) fn take_model_execution_scope(
        &self,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionRole>, Error>
    {
        self.take_prediction_scope(
            crate::backend::submission_recovery::prediction::PredictionSet::take_model_execution,
        )
    }
    pub(super) fn take_sampling_scope(
        &self,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionRole>, Error>
    {
        self.take_prediction_scope(
            crate::backend::submission_recovery::prediction::PredictionSet::take_sampling,
        )
    }
    pub(in crate::composition::mlx::session) fn take_sampling_event_scope(
        &self,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionRole>, Error>
    {
        self.take_prediction_scope(
            crate::backend::submission_recovery::prediction::PredictionSet::take_sampling_event,
        )
    }
    pub(in crate::composition::mlx::session) fn take_model_validation_scope(
        &self,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionRole>, Error>
    {
        self.take_prediction_scope(
            crate::backend::submission_recovery::prediction::PredictionSet::take_model_validation,
        )
    }
    pub(in crate::composition::mlx::session) fn take_token_scalar_scope(
        &self,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionRole>, Error>
    {
        self.take_prediction_scope(
            crate::backend::submission_recovery::prediction::PredictionSet::take_token_scalar,
        )
    }
    pub(in crate::composition::mlx::session) fn recovery_with_prediction(
        &self,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
    ) -> Result<Recovery<ScopeRetention>, Error> {
        self.scopes.set(self.scopes.get() + 1);
        crate::backend::submission_recovery::prediction::begin(
            role,
            ScopeRetention {
                owner: self.clone(),
                paged: None,
                _original: None,
            },
        )
    }
    pub(super) fn recovery_with_model_prediction(
        &self,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
        preparation: Option<
            crate::backend::submission_recovery::prefill::ModelExecutionPreparation,
        >,
    ) -> Result<
        (
            Recovery<ScopeRetention>,
            Option<crate::backend::submission_recovery::prefill::ModelExecutionOwner>,
        ),
        Error,
    > {
        self.scopes.set(self.scopes.get() + 1);
        crate::backend::submission_recovery::prediction::begin_model(
            role,
            ScopeRetention {
                owner: self.clone(),
                paged: None,
                _original: None,
            },
            preparation,
        )
    }
    pub(in crate::composition::mlx::session) fn observation_recovery_with_prediction(
        &self,
        roots: ObservationRoots,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
    ) -> Result<Recovery<ObservationRetention>, Error> {
        self.scopes.set(self.scopes.get() + 1);
        crate::backend::submission_recovery::prediction::begin(
            role,
            ObservationRetention {
                _roots: roots,
                ticket: ScopeRetention {
                    owner: self.clone(),
                paged: None,
                    _original: None,
                },
            },
        )
    }

    pub(super) fn new(value: SubmissionResources) -> Self {
        Self(Some(Rc::new(value)))
    }
    pub(in crate::composition::mlx::session) fn recovery(
        &self,
    ) -> Result<Recovery<ScopeRetention>, Error> {
        Recovery::begin(self.ticket()).map_err(Into::into)
    }

    pub(in crate::composition::mlx::session) fn observation_recovery(
        &self,
        roots: Vec<Array>,
    ) -> Result<Recovery<ObservationRetention>, Error> {
        Recovery::begin(ObservationRetention {
            _roots: ObservationRoots::Ordinary { _arrays: roots },
            ticket: self.ticket(),
        })
        .map_err(Into::into)
    }

    pub(in crate::composition::mlx::session) fn ticket(&self) -> ScopeRetention {
        self.scopes.set(self.scopes.get() + 1);
        ScopeRetention {
            owner: self.clone(),
            paged: None,
            _original: None,
        }
    }
}
impl Clone for SubmissionResourcesOwner {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live closed owner").clone()))
    }
}
impl Deref for SubmissionResourcesOwner {
    type Target = SubmissionResources;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live closed owner")
    }
}
impl AsRef<SubmissionResources> for SubmissionResourcesOwner {
    fn as_ref(&self) -> &SubmissionResources {
        self
    }
}
impl fmt::Debug for SubmissionResourcesOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionResourcesOwner")
            .finish_non_exhaustive()
    }
}
impl Drop for SubmissionResourcesOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // The closed population has no Weak. With the final strong owner,
            // the Rc block is gone before these source/native/custody fields.
            // Existing Recovery callbacks keep their runtime retirement guard through
            // this synchronous extraction and all native payload destruction.
            let payload = Rc::into_inner(owner);
            drop(payload);
        }
    }
}

/// Original diagnostics only. Supported Rust1.98 RcInner is repr(C, align(2))
/// with two Cell<usize> counters followed by T. The supported toolchain layout
/// includes trailing padding; this is requested allocation
/// storage, not allocator usable size/RSS. Reaudit on a toolchain layout change.
/// No separately owned payload buffer or arbitrary alias population is added.
pub(super) fn control_bytes() -> Option<u64> {
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    let block = header
        .extend(Layout::new::<SubmissionResources>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let bytes = block
        // Caller aggregate, owner constructor parameter, Rc::new parameter.
        .checked_add(size_of::<SubmissionResources>())?
        .checked_add(size_of::<SubmissionResources>())?
        .checked_add(size_of::<SubmissionResources>())?
        // Closed return/caller Option and the exact consuming-retirement locals.
        .checked_add(size_of::<SubmissionResourcesOwner>())?
        .checked_add(size_of::<Option<SubmissionResourcesOwner>>())?
        .checked_add(size_of::<Option<Rc<SubmissionResources>>>())?
        .checked_add(size_of::<Rc<SubmissionResources>>())?
        .checked_add(size_of::<ManuallyDrop<Rc<SubmissionResources>>>())?
        .checked_add(size_of::<Weak<SubmissionResources>>())?
        .checked_add(size_of::<
            Result<SubmissionResources, Rc<SubmissionResources>>,
        >())?
        .checked_add(size_of::<Option<SubmissionResources>>())?
        // Direct model/observation finish preserves a returned cleanup cause
        // while its same owner is fenced, without a new error allocation.
        .checked_add(size_of::<&SubmissionResourcesOwner>())?
        .checked_add(size_of::<Result<super::Status, crate::backend::runtime::execution::generic::RegisteredScopeRetirementCause>>())?
        .checked_add(size_of::<Result<super::Status, Error>>())?;
    u64::try_from(bytes).ok()
}
