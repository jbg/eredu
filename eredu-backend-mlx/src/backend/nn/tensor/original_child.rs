//! Explicit collector loan for the already authenticated native child role.
//! Ordinary begin() remains non-nestable. Both suspended and failed collectors
//! keep their numerical aliases and metadata custody until their owner retires.
use super::*;
use crate::backend::{error::Error, nn::workspace::ResidentCompletionRecipe};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{OriginalTextControlGuard, WorkingMemoryError};
use safemlx::OriginalScopeObserver;
use std::mem::{size_of, size_of_val};

pub(crate) struct PreparedTokenChild {
    pending: RefCell<Option<PreparedTokenValidations>>,
    retired: RefCell<Option<ActiveTokenValidations>>,
    parent: OriginalScopeObserver,
    funding: HostMetadataFunding,
}
pub(crate) struct TokenChildScope<'a> {
    parent: Option<ActiveTokenValidations>,
    child: OriginalScopeObserver,
    storage: &'a PreparedTokenChild,
}
impl PreparedTokenChild {
    pub(crate) fn control_bytes(recipe: ResidentCompletionRecipe) -> Option<usize> {
        let controls = [size_of::<eredu_runtime::working_memory::OriginalOperationMetadataCustody>(),
            size_of::<Self>(), size_of::<Result<Self, Error>>(), size_of::<TokenChildScope<'_>>(),
            size_of::<Result<TokenChildScope<'_>, Error>>(), size_of::<PreparedTokenValidations>(),
            size_of::<ActiveTokenValidations>(), size_of::<Option<ActiveTokenValidations>>(),
            size_of::<std::cell::RefMut<'_, Option<ActiveTokenValidations>>>(),
            size_of::<std::cell::RefMut<'_, Option<PreparedTokenValidations>>>(),
            size_of::<Option<OriginalScopeObserver>>(), OriginalScopeObserver::control_bytes()?.checked_mul(2)?,
            usize::try_from(TokenValidationIngress::speculative_completion_control_bytes(recipe).ok()?).ok()?];
        controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
    }
    pub(crate) fn prepare(recipe: ResidentCompletionRecipe, controls: &OriginalTextControlGuard,
        parent: &OriginalScopeObserver, funding: &HostMetadataFunding) -> Result<Self, Error> {
        Self::prepare_metadata(recipe,controls.metadata_custody().into(),parent,funding)
    }
    /// Metadata custody keeps storage live; the actual current parent collector
    /// and scope are still authenticated before any child slots are prepared.
    pub(crate) fn prepare_metadata(recipe:ResidentCompletionRecipe,
        custody:eredu_runtime::working_memory::OriginalOperationMetadataCustody,
        parent:&OriginalScopeObserver,funding:&HostMetadataFunding)->Result<Self,Error> {
        funding.reserve_metadata(Self::control_bytes(recipe).ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let current = OriginalScopeObserver::require_current()?;
        if !parent.same_scope(&current) { return Err(parent.domain_error().into()); }
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let active = slot.try_borrow().map_err(|_| parent.capacity_error())?;
            if !active.as_ref().and_then(|active| active.observer.as_ref())
                .is_some_and(|actual| actual.same_scope(parent)) { return Err(parent.domain_error()); }
            Ok(())
        })?;
        let prepared = TokenValidationIngress::prepare_storage(recipe.validation_roots, recipe.grouped_outputs,
            TokenValidationCustody::Operation(custody))?;
        let TokenValidationIngress::Original(Some(prepared)) = prepared else { return Err(parent.domain_error().into()); };
        Ok(Self { pending: RefCell::new(Some(prepared)), retired: RefCell::new(None), parent: parent.clone(), funding: funding.clone() })
    }
    /// Caller holds the admitted child Scope; both explicit observer identities
    /// are checked before the private collector slot is exchanged.
    pub(crate) fn enter(&self, child: &OriginalScopeObserver) -> Result<TokenChildScope<'_>, Error> {
        let current = OriginalScopeObserver::require_current()?;
        if !child.same_scope(&current) || child.same_scope(&self.parent) { return Err(child.domain_error().into()); }
        let mut pending = self.pending.try_borrow_mut().map_err(|_| child.capacity_error())?;
        let retired = self.retired.try_borrow_mut().map_err(|_| child.capacity_error())?;
        if retired.is_some() { return Err(child.invalid_input_error().into()); }
        let result = TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot.try_borrow_mut().map_err(|_| child.capacity_error())?;
            if !slot.as_ref().and_then(|active| active.observer.as_ref())
                .is_some_and(|actual| actual.same_scope(&self.parent)) { return Err(child.domain_error()); }
            let prepared = pending.take().ok_or_else(|| child.invalid_input_error())?;
            let remaining = prepared.0.validations.capacity();
            let capture_parent=slot.as_ref().and_then(|active|active.capture_parent.as_ref())
                .unwrap_or(&self.parent).clone();
            let active = ActiveTokenValidations { observer: Some(child.clone()),capture_parent:Some(capture_parent),remaining: Some(remaining),
                batch: prepared.0, grouped_outputs: prepared.1 };
            Ok(slot.replace(active))
        });
        drop(retired);
        let parent = result?;
        Ok(TokenChildScope { parent, child: child.clone(), storage: self })
    }
}
impl TokenChildScope<'_> {
    pub(crate) fn finish(self) -> Result<(), Error> {
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let slot=slot.try_borrow().map_err(|_| self.child.capacity_error())?;
            if !slot.as_ref().and_then(|active|active.observer.as_ref())
                .is_some_and(|active|active.same_scope(&self.child)) { return Err(self.child.domain_error()); }
            Ok(())
        })?;
        Ok(())
    }
}
impl Drop for TokenChildScope<'_> {
    fn drop(&mut self) {
        // No callback runs under these private RefCell loans. Retired contains
        // every child assertion and unused grouped destination, including error.
        let child = TOKEN_VALIDATION_SCOPE.with(|slot| slot.borrow_mut().replace(
            self.parent.take().expect("authenticated parent collector")));
        *self.storage.retired.borrow_mut() = child;
    }
}


impl TokenValidationScope {
    /// Borrows only a currently entered descendant explicitly installed by
    /// PreparedTokenChild. The retained caller parent is mandatory; TLS merely
    /// checks that the accepted child's stored identity is still current.
    pub(crate) fn capture_observer_for(parent:&OriginalScopeObserver)->Result<OriginalScopeObserver,Exception>{
        if let Some(cause)=parent.retained_failure(){return Err(cause);}
        if parent.status().blocked(){return Err(parent.domain_error());}
        let current=OriginalScopeObserver::require_current()?;
        if parent.same_scope(&current){return Ok(parent.clone());}
        TOKEN_VALIDATION_SCOPE.with(|slot|{
            let active=slot.try_borrow().map_err(|_|parent.capacity_error())?;
            let active=active.as_ref().filter(|active|active.remaining.is_some()
                &&active.capture_parent.as_ref().is_some_and(|root|root.same_scope(parent)))
                .ok_or_else(||parent.domain_error())?;
            let child=active.observer.as_ref().filter(|child|child.same_scope(&current))
                .ok_or_else(||parent.domain_error())?;
            if let Some(cause)=child.retained_failure(){return Err(cause);}
            if child.status().blocked(){return Err(child.domain_error());}
            Ok(child.clone())
        })
    }
    pub(crate) fn capture_observer_control_bytes()->Option<usize>{
        let frames=[size_of::<&OriginalScopeObserver>(),size_of::<OriginalScopeObserver>(),
            size_of::<Result<OriginalScopeObserver,Exception>>(),
            size_of::<Option<Exception>>(),size_of::<std::cell::Ref<'static,Option<ActiveTokenValidations>>>(),
            OriginalScopeObserver::control_bytes()?.checked_mul(3)?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
