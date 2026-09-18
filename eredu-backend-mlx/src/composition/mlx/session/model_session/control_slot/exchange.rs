//! Finite old-state memory-owner handoff for one prepared original exchange.

use super::{MlxNativeTextState, NativeMemoryRetention, PreparedControlSlotError, error};
use crate::backend::{error::Error, managed_memory::NativeMemoryOwner};
use crate::composition::mlx::session::model_session::{MlxBackend, ModelRuntime};
use eredu_core::{HostPreparationAuthority, SessionAuthorityError};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::{
    cell::{Cell, Ref},
    rc::Rc,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum ExchangeSourceCause {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Authority(#[from] SessionAuthorityError),
    #[error("control exchange source is borrowed")]
    Borrowed(#[from] std::cell::BorrowError),
    #[error("control exchange source target differs")]
    Target,
    #[error("control exchange source-owner inventory changed")]
    Owners,
    #[error("control exchange payload remains shared")]
    Shared,
    #[error("control exchange owner rows could not be allocated")]
    Allocation(#[from] std::collections::TryReserveError),
}

impl ExchangeSourceCause {
    pub(crate) fn into_memory(self) -> WorkingMemoryError {
        match self {
            Self::Memory(error) => error,
            Self::Target | Self::Owners => WorkingMemoryError::IdentityMismatch,
            _ => WorkingMemoryError::UnknownBound,
        }
    }
}

/// A cold loan of actual source owners, not a count detached from those sources.
/// The operation-memory Ref prevents mutation throughout count and construction.
/// Source roots stay separately charged; only the new row vector is copied.
pub(crate) struct OriginalControlExchangePlan<'a> {
    model: Option<&'a NativeMemoryOwner>,
    state: &'a NativeMemoryRetention,
    operations: Ref<'a, NativeMemoryRetention>,
    target: &'a Rc<Cell<bool>>,
    rows: usize,
}

pub(crate) struct PreparedControlExchange {
    owners: Vec<NativeMemoryOwner>,
    target: Rc<Cell<bool>>,
    // Finite rows and all ordinary source aliases retire before constructor H.
    _host: HostPreparationAuthority,
}

fn check_target(runtime: &ModelRuntime<MlxBackend<'_>>) -> Result<(), ExchangeSourceCause> {
    let session = runtime.session();
    if session.poison.get() || session.failure.try_borrow()?.is_some() {
        return Err(WorkingMemoryError::ExecutionFenced.into());
    }
    if !runtime
        .backend()
        .matches_prepared_target(&session.payload.target)
    {
        return Err(ExchangeSourceCause::Target);
    }
    session.authority.try_borrow()?.require_idle()?;
    Ok(())
}

impl<'a> OriginalControlExchangePlan<'a> {
    pub(crate) fn inspect(
        runtime: &'a ModelRuntime<MlxBackend<'_>>,
    ) -> Result<Self, ExchangeSourceCause> {
        check_target(runtime)?;
        let payload = &runtime.session().payload;
        let mut out = Self {
            model: payload._memory_owner.as_ref(),
            state: &payload.state_memory,
            operations: payload.operation_memory.try_borrow()?,
            target: &runtime.session().poison,
            rows: 0,
        };
        // Bound the combined traversal ordinal as well as the unique result.
        out.state
            .owners()
            .len()
            .checked_add(usize::from(out.model.is_some()))
            .and_then(|count| count.checked_add(out.operations.owners().len()))
            .ok_or(WorkingMemoryError::Overflow)?;
        let mut rows = 0usize;
        for (ordinal, owner) in out.owners().enumerate() {
            if !out
                .owners()
                .take(ordinal)
                .any(|earlier| earlier.same_authority(owner))
            {
                rows = rows.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        out.rows = rows;
        Ok(out)
    }

    fn owners(&self) -> impl Iterator<Item = &NativeMemoryOwner> {
        self.state
            .owners()
            .iter()
            .chain(self.model)
            .chain(self.operations.owners())
    }

    /// Exact row capacity plus fixed constructor/error/return and install loans.
    /// Ordinary source accounts are retained, never relabeled as new admission.
    pub(crate) fn control_bytes(&self) -> Result<usize, WorkingMemoryError> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            self.rows
                .checked_mul(size_of::<NativeMemoryOwner>())
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<PreparedControlExchange>(),
            size_of::<Vec<NativeMemoryOwner>>(),
            size_of::<Option<usize>>(),
            size_of::<NativeMemoryOwner>(),
            size_of::<NativeMemoryRetention>(),
            size_of::<Option<PreparedControlExchange>>(),
            size_of::<Result<PreparedControlExchange, Error>>(),
            size_of::<Option<eredu_runtime::working_memory::CopiedMediaStateBinding>>(),
            size_of::<Result<Option<eredu_runtime::working_memory::CopiedMediaStateBinding>, Error>>(
            ),
            size_of::<Option<&eredu_runtime::working_memory::MediaSessionBinding>>(),
            size_of::<&eredu_nn::workspace::HostMetadataFunding>(),
            size_of::<Result<Self, ExchangeSourceCause>>(),
            size_of::<Result<(), ExchangeSourceCause>>(),
            size_of::<ExchangeSourceCause>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Ref<'_, NativeMemoryRetention>>(),
            size_of::<Result<Ref<'_, NativeMemoryRetention>, std::cell::BorrowError>>(),
            size_of::<Ref<'_, eredu_core::SessionAuthority>>(),
            size_of::<Result<Ref<'_, eredu_core::SessionAuthority>, std::cell::BorrowError>>(),
            size_of::<Ref<'_, Option<String>>>(),
            size_of::<Result<Ref<'_, Option<String>>, std::cell::BorrowError>>(),
            size_of::<Result<(), SessionAuthorityError>>(),
            size_of::<(&mut Self, usize, &NativeMemoryOwner)>(),
            size_of::<Rc<Cell<bool>>>(),
            size_of::<HostPreparationAuthority>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }

    /// Consumes the same retained source loan after its amount is admitted.
    pub(crate) fn construct(
        self,
        host: &HostPreparationAuthority,
    ) -> Result<PreparedControlExchange, Error> {
        let mut out = PreparedControlExchange {
            owners: Vec::new(),
            target: Rc::clone(self.target),
            _host: host.clone(),
        };
        out.owners
            .try_reserve_exact(self.rows)
            .map_err(|cause| error(ExchangeSourceCause::from(cause)))?;
        for owner in self.owners() {
            if !out
                .owners
                .iter()
                .any(|earlier| earlier.same_authority(owner))
            {
                // Count and fill borrow the same immutable source roots.
                if out.owners.len() == self.rows {
                    return Err(error(ExchangeSourceCause::Owners));
                }
                out.owners.push(owner.clone());
            }
        }
        Ok(out)
    }
}

impl PreparedControlExchange {
    /// Called only by the consumed prepared original resume. No global reap,
    /// ordinary support estimator, owner-list growth or second authority occurs.
    pub(crate) fn install(
        mut self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        slot: &mut MlxNativeTextState,
        metadata: &eredu_nn::workspace::HostMetadataFunding,
        media: Option<&eredu_runtime::working_memory::MediaSessionBinding>,
    ) -> Result<Option<eredu_runtime::working_memory::CopiedMediaStateBinding>, Error> {
        check_target(runtime).map_err(error)?;
        if !Rc::ptr_eq(&self.target, &runtime.session().poison)
            || slot.host_preparation.is_none()
            || !slot.memory_retention.owners().is_empty()
        {
            return Err(error(ExchangeSourceCause::Target));
        }
        {
            let current = OriginalControlExchangePlan::inspect(runtime).map_err(error)?;
            if current.rows != self.owners.len()
                || current
                    .owners()
                    .any(|owner| !self.owners.iter().any(|known| known.same_authority(owner)))
            {
                return Err(error(ExchangeSourceCause::Owners));
            }
        }
        let payload = runtime
            .session_mut()
            .payload
            .get_mut()
            .ok_or_else(|| error(ExchangeSourceCause::Shared))?;
        // The shared typed exchange validates owner/layout and votes before its
        // infallible state move. No source alias is released ahead of that move.
        let transition = payload.model.erased_mut().exchange_original_control_state(
            slot.state.as_mut(),
            metadata,
            media,
            None,
        )?;
        slot.displaced_placement = transition.displaced;
        let prior = std::mem::take(&mut payload.state_memory);
        slot.memory_retention =
            NativeMemoryRetention::from_prepared_owners(std::mem::take(&mut self.owners));
        // Slot's independent H remains last; prior metadata is replaced by the
        // exact separately constructed list, preserving every actual authority.
        drop(prior);
        Ok(transition.media)
    }
}

impl From<ExchangeSourceCause> for PreparedControlSlotError {
    fn from(cause: ExchangeSourceCause) -> Self {
        Self::Source(cause)
    }
}
