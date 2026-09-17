//! Lexical authority for a child accepted by the existing prefill role bank.
//! Slots, counts and custody stay in the whole-operation collector. This only
//! changes the authenticated observer while that accepted role is executing.
use super::*;
use safemlx::OriginalScopeObserver;
use std::mem::size_of;

pub(crate) struct TokenValidationParent(OriginalScopeObserver);
pub(crate) struct TokenValidationRole {
    parent: Option<OriginalScopeObserver>,
    child: OriginalScopeObserver,
}
impl TokenValidationParent {
    /// Capture before entering a child. An unrelated active scope cannot lend
    /// the batch; component calls without a token collector need no loan.
    pub(crate) fn capture() -> Result<Option<Self>, Exception> {
        let Some(current) = OriginalScopeObserver::try_current()? else {
            return Ok(None);
        };
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let slot = slot.try_borrow().map_err(|_| current.capacity_error())?;
            let Some(active) = slot.as_ref() else {
                return Ok(None);
            };
            if active.remaining.is_none()
                || !active
                    .observer
                    .as_ref()
                    .is_some_and(|owner| owner.same_scope(&current))
            {
                return Err(current.domain_error());
            }
            Ok(Some(Self(current)))
        })
    }
    /// Called only after the source-bound prefill bank accepts its next role.
    /// Failure leaves the parent collector and all remaining slots unchanged.
    pub(crate) fn bind(self) -> Result<TokenValidationRole, Exception> {
        let child = OriginalScopeObserver::require_current()?;
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot.try_borrow_mut().map_err(|_| child.capacity_error())?;
            let active = slot
                .as_mut()
                .filter(|active| {
                    active.remaining.is_some()
                        && active
                            .observer
                            .as_ref()
                            .is_some_and(|owner| owner.same_scope(&self.0))
                })
                .ok_or_else(|| child.domain_error())?;
            let parent = active.observer.replace(child.clone());
            Ok(TokenValidationRole { parent, child })
        })
    }
}
impl TokenValidationRole {
    pub(crate) fn control_bytes() -> Option<usize> {
        [
            size_of::<TokenValidationParent>(),
            size_of::<Option<TokenValidationParent>>(),
            size_of::<Result<Option<TokenValidationParent>, Exception>>(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Exception>>(),
            size_of::<std::cell::Ref<'static, Option<ActiveTokenValidations>>>(),
            size_of::<std::cell::RefMut<'static, Option<ActiveTokenValidations>>>(),
            OriginalScopeObserver::control_bytes()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}
impl Drop for TokenValidationRole {
    fn drop(&mut self) {
        // This guard precedes native recovery in ReservationGuard, so both
        // explicit finish and unwinding restore the parent before native seal.
        // No arrays, callbacks, allocations or completion claims occur here.
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some(active) = slot.as_mut().filter(|active| {
                active
                    .observer
                    .as_ref()
                    .is_some_and(|owner| owner.same_scope(&self.child))
            }) {
                active.observer = self.parent.take();
            }
        });
    }
}
