use super::{Error, SubmissionResourcesOwner};
use crate::backend::error::OutputObservationFailure;
use std::{
    cell::{Cell, RefCell},
    ops::Deref,
    rc::Rc,
};

/// Fixed local state. No native operation, callback or owner Drop runs under
/// its RefCell loan. The separate gate also covers resource-only inspection.
pub(in crate::composition::mlx::session) struct Observation<T> {
    busy: Cell<bool>,
    interrupted: Cell<bool>,
    result: RefCell<Option<Result<T, OutputObservationFailure>>>,
    ordinary_host: Option<eredu_core::HostPreparationAuthority>,
}
impl<T: Copy> Observation<T> {
    pub(in crate::composition::mlx::session) const fn new() -> Self {
        Self {
            busy: Cell::new(false),
            interrupted: Cell::new(false),
            result: RefCell::new(None),
            ordinary_host: None,
        }
    }
    pub(in crate::composition::mlx::session) fn with_ordinary_capture(
        host: Option<eredu_core::HostPreparationAuthority>,
    ) -> Self {
        let mut result = Self::new();
        result.ordinary_host = host;
        result
    }
    pub(in crate::composition::mlx::session) fn ordinary_error_custody(
        &self,
    ) -> Option<&eredu_core::HostPreparationAuthority> {
        self.ordinary_host.as_ref()
    }
    pub(in crate::composition::mlx::session) fn retain_ordinary_capture(
        &mut self,
        host: Option<eredu_core::HostPreparationAuthority>,
    ) {
        if let Some(host) = host {
            // This exclusive constructor path has no RefCell loan or escaped token.
            // Keep an already cached failure's complete allocation under the new owner.
            let previous = self.result.get_mut().take();
            *self.result.get_mut() = previous.map(|result| {
                result.map_err(|error| {
                    OutputObservationFailure::new(
                        Error::OutputObservation(error).retain_ordinary_capture(Some(host.clone())),
                    )
                })
            });
            self.ordinary_host = Some(host);
        }
    }
    pub(super) fn enter(&self) -> Result<Loan<'_, T>, Error> {
        if self.busy.replace(true) {
            return Err(Error::OutputObservationReentrant);
        }
        Ok(Loan(self))
    }
}

pub(super) struct Loan<'a, T>(&'a Observation<T>);
impl<T: Copy> Loan<'_, T> {
    pub(super) fn cached(&self) -> Option<Result<T, Error>> {
        // Cloning the private strong reference invokes no provider code. Return
        // it only after the state loan ends; destruction is likewise outside.
        let result = self.0.result.borrow().as_ref().map(|result| match result {
            Ok(value) => Ok(*value),
            Err(error) => Err(Error::OutputObservation(error.retained())),
        });
        if matches!(result, Some(Err(_))) {
            result
        } else if self.0.interrupted.get() {
            Some(Err(Error::OutputObservationInterrupted))
        } else {
            result
        }
    }
    pub(super) fn complete(&self, result: Result<T, Error>) -> Result<T, Error> {
        let previous_failure = self.0.result.borrow().as_ref().and_then(|previous| {
            previous
                .as_ref()
                .err()
                .map(OutputObservationFailure::retained)
        });
        if let Some(error) = previous_failure {
            // Incoming errors can own arbitrary sources. Drop them only after
            // the state loan ends, and keep the first concrete cause unchanged.
            drop(result);
            return Err(Error::OutputObservation(error));
        }
        let result = result
            .map_err(|error| error.retain_ordinary_capture(self.0.ordinary_host.clone()))
            .map_err(OutputObservationFailure::new);
        let returned = match &result {
            Ok(value) => Ok(*value),
            Err(error) => Err(Error::OutputObservation(error.retained())),
        };
        let previous = self.0.result.replace(Some(result));
        // A successful cached value can become a failure when current owner
        // health changes. The previous value and any owned source drop here.
        drop(previous);
        returned
    }
}
impl<T> Drop for Loan<'_, T> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.interrupted.set(true);
        }
        self.0.busy.set(false);
    }
}

pub(super) struct TokenObservation {
    original: bool,
    scope: RefCell<Option<crate::backend::submission_recovery::prediction::PredictionRole>>,
    pub(super) sampling: Option<crate::backend::SamplingEventSource>,
    pub(super) value: Observation<u32>,
    pub(super) fixed: Option<super::scalar::FixedSnapshots>,
    pub(super) original_controls: Option<eredu_runtime::working_memory::OriginalTextControlGuard>,
    // Keeps the same producing owner through this new Rc allocation's removal.
    // This is retention, not a fresh hold or a Scope population allowance.
    _owner: SubmissionResourcesOwner,
}
impl TokenObservation {
    pub(super) fn take_scope(
        &self,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionRole>, Error>
    {
        let role = self
            .scope
            .try_borrow_mut()
            .map_err(|_| Error::PredictionScopeReentrant)?
            .take();
        if self.original && role.is_none() {
            return Err(Error::PredictionScopeUnavailable);
        }
        Ok(role)
    }
}
pub(super) struct TokenObservationOwner(Option<Rc<TokenObservation>>);
impl TokenObservationOwner {
    pub(super) fn new(owner: SubmissionResourcesOwner) -> Self {
        Self::new_with_scope(owner, None)
    }
    pub(super) fn new_with_scope(
        owner: SubmissionResourcesOwner,
        scope: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
    ) -> Self {
        Self::new_with_scope_and_capture(owner, scope, None)
    }
    pub(super) fn new_with_scope_and_capture(
        owner: SubmissionResourcesOwner,
        scope: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
        host: Option<eredu_core::HostPreparationAuthority>,
    ) -> Self {
        Self::new_with_sources(owner, scope, host, None)
    }
    pub(super) fn new_with_sources(
        owner: SubmissionResourcesOwner,
        scope: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
        host: Option<eredu_core::HostPreparationAuthority>,
        sampling: Option<crate::backend::SamplingEventSource>,
    ) -> Self {
        let original_controls = scope.as_ref().map(|role| role.control_guard().clone());
        let fixed = original_controls.as_ref().map(|scalar| {
            super::scalar::FixedSnapshots::new(
                scalar,
                sampling.as_ref().map(|source| source.controls()),
            )
        });
        Self(Some(Rc::new(TokenObservation {
            original: scope.is_some(),
            scope: RefCell::new(scope),
            sampling,
            value: Observation::with_ordinary_capture(host),
            fixed,
            original_controls,
            _owner: owner,
        })))
    }
}
impl Clone for TokenObservationOwner {
    fn clone(&self) -> Self {
        Self(Some(
            self.0.as_ref().expect("live token observation").clone(),
        ))
    }
}
impl Deref for TokenObservationOwner {
    type Target = TokenObservation;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live token observation")
    }
}
impl Drop for TokenObservationOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // No raw Rc/Weak leaves this module. The allocation is gone before
            // the cached original error and producing owner are destroyed.
            drop(Rc::into_inner(owner));
        }
    }
}

// Same pinned Rust 1.98 RcInner layout as the existing closed native owners.
// This covers this one allocation and its named moves, not cached Error payloads.
pub(in crate::composition::mlx::session) fn token_control_bytes() -> Option<u64> {
    use std::{
        alloc::Layout,
        mem::{size_of, ManuallyDrop},
        rc::Weak,
    };
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    let block = header
        .extend(Layout::new::<TokenObservation>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let moves = [
        size_of::<TokenObservation>(), // constructor aggregate
        size_of::<TokenObservation>(), // Rc::new argument
        size_of::<TokenObservation>(), // Rc::try_unwrap value (called by into_inner)
        size_of::<TokenObservationOwner>(),
        size_of::<Option<TokenObservationOwner>>(),
        size_of::<Option<Rc<TokenObservation>>>(),
        size_of::<Rc<TokenObservation>>(),
        size_of::<ManuallyDrop<Rc<TokenObservation>>>(),
        size_of::<Weak<TokenObservation>>(),
        size_of::<Result<TokenObservation, Rc<TokenObservation>>>(),
        size_of::<Option<TokenObservation>>(),
    ]
    .into_iter()
    .try_fold(block, usize::checked_add)?;
    u64::try_from(moves).ok()
}

#[cfg(test)]
mod tests;
