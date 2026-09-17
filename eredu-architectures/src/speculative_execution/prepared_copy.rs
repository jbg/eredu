//! Closed source-bound state copying for the shared embedded cache lifecycle.
use eredu_core::{BackendFailure, HostPreparationAuthority};
use std::{
    alloc::Layout,
    any::Any,
    cell::Cell,
    marker::PhantomData,
    mem::{size_of, size_of_val},
    rc::Rc,
};

/// Fixed shared-cache refusal. The actual source provider retains its error
/// transport before constructing an owned backend failure.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum PreparedEmbeddedCopyError {
    /// Checked constructor sizing exceeded the host address space.
    #[error("embedded copy metadata size overflow")]
    Overflow,
    /// A state/copy source was mixed with a different prepared lineage.
    #[error("embedded copy belongs to a different prepared source")]
    SourceMismatch,
    /// Exact prepared input identity was absent before initial binding.
    #[error("embedded speculative input is missing its prepared-input cache identity")]
    MissingPreparedInput,
}

/// Exact backend copy and host-destination mechanism for one native state type.
/// This provider must retain only source identity/copy prerequisites, never the
/// state which retains it. It grants no inference invocation or scope authority.
/// Native implementations authenticate their closed source and use the existing
/// separately admitted copy/completion/publication worker for each call.
pub trait PreparedEmbeddedCopyProvider<S>: 'static {
    /// Copies actual current state, preserving the source on refusal.
    fn copy_state(
        &self,
        source: &S,
        evidence: Option<&PreparedEmbeddedEvidence>,
    ) -> Result<PreparedEmbeddedPayload<S>, BackendFailure>;
    /// Copies through this exact source owner when the destination needs fresh
    /// source evidence. The default keeps existing provider behavior.
    fn copy_state_with_owner(
        &self, _owner: &PreparedEmbeddedCopy<S>, source: &S,
        evidence: Option<&PreparedEmbeddedEvidence>,
    ) -> Result<PreparedEmbeddedPayload<S>, BackendFailure> {
        self.copy_state(source, evidence)
    }
    /// Debits the actual cumulative source account before metadata construction.
    /// Returned lifetime custody alone is not a substitute for this debit.
    fn prepare_host(&self, bytes: usize) -> Result<HostPreparationAuthority, BackendFailure>;
    /// Retains a fixed rejection through the provider's actual paid error path.
    fn reject(&self, cause: PreparedEmbeddedCopyError) -> BackendFailure;
    /// Preserves a failed exact destination reserve through the same paying source.
    fn allocation_failure(&self, cause: std::collections::TryReserveError) -> BackendFailure;
}

/// Completed source-funded payload paired with its retirement authority.
/// Construction is a backend-mechanism boundary, not permission to adopt an
/// existing ordinary allocation. Only an actual prepaid producer may use it.
#[must_use]
pub struct PreparedEmbeddedPayload<S> {
    pub(super) value: S,
    pub(super) evidence: Option<PreparedEmbeddedEvidence>,
    pub(super) host: HostPreparationAuthority,
}
impl<S> PreparedEmbeddedPayload<S> {
    /// Pairs the already completed copy with the authority that paid its birth.
    pub fn new(value: S, host: HostPreparationAuthority) -> Self {
        Self {
            value,
            evidence: None,
            host,
        }
    }
    /// Moves both pieces into the enclosing prepared state owner.
    pub fn into_parts(
        self,
    ) -> (
        S,
        Option<PreparedEmbeddedEvidence>,
        HostPreparationAuthority,
    ) {
        (self.value, self.evidence, self.host)
    }
    /// Attaches the exact source evidence emitted for this actual state.
    pub fn with_evidence(mut self, evidence: PreparedEmbeddedEvidence) -> Self {
        self.evidence = Some(evidence);
        self
    }
}
impl<S> std::fmt::Debug for PreparedEmbeddedPayload<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEmbeddedPayload")
            .finish_non_exhaustive()
    }
}
struct Owner<P> {
    provider: P,
    _host: HostPreparationAuthority,
}
type Erased = Rc<dyn Any>;

/// Immutable, closed copy-source erasure. It owns no state and exports no Rc or
/// weak handle. Final shell retirement precedes provider/account destruction.
pub struct PreparedEmbeddedCopy<S> {
    owner: Option<Erased>,
    copy: fn(
        &dyn Any,
        &PreparedEmbeddedCopy<S>,
        &S,
        Option<&PreparedEmbeddedEvidence>,
    ) -> Result<PreparedEmbeddedPayload<S>, BackendFailure>,
    host: fn(&dyn Any, usize) -> Result<HostPreparationAuthority, BackendFailure>,
    reject: fn(&dyn Any, PreparedEmbeddedCopyError) -> BackendFailure,
    allocation: fn(&dyn Any, std::collections::TryReserveError) -> BackendFailure,
    retire: fn(Erased),
    marker: PhantomData<fn() -> S>,
}
impl<S> Clone for PreparedEmbeddedCopy<S> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            copy: self.copy,
            host: self.host,
            reject: self.reject,
            allocation: self.allocation,
            retire: self.retire,
            marker: PhantomData,
        }
    }
}
impl<S> Drop for PreparedEmbeddedCopy<S> {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            (self.retire)(owner);
        }
    }
}
impl<S> std::fmt::Debug for PreparedEmbeddedCopy<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEmbeddedCopy")
            .finish_non_exhaustive()
    }
}
fn owner<P: 'static>(erased: &dyn Any) -> &Owner<P> {
    erased
        .downcast_ref()
        .expect("closed embedded provider type")
}
fn retire<P: 'static>(erased: Erased) {
    let concrete = match erased.downcast::<Owner<P>>() {
        Ok(value) => value,
        Err(_) => unreachable!("closed embedded provider type"),
    };
    drop(Rc::into_inner(concrete));
}
impl<S> PreparedEmbeddedCopy<S> {
    /// Pays the exact provider shell and transport before erasure allocation.
    /// The provider itself must already have been prepared by its actual source.
    pub fn prepare<P: PreparedEmbeddedCopyProvider<S>>(
        provider: P,
    ) -> Result<Self, BackendFailure> {
        let bytes = Self::constructor_bytes::<P>()
            .ok_or_else(|| provider.reject(PreparedEmbeddedCopyError::Overflow))?;
        let host = provider.prepare_host(bytes)?;
        Ok(Self {
            owner: Some(Rc::new(Owner {
                provider,
                _host: host,
            })),
            copy: |erased, copy, value, evidence| owner::<P>(erased).provider.copy_state_with_owner(copy, value, evidence),
            host: |erased, bytes| owner::<P>(erased).provider.prepare_host(bytes),
            reject: |erased, cause| owner::<P>(erased).provider.reject(cause),
            allocation: |erased, cause| owner::<P>(erased).provider.allocation_failure(cause),
            retire: retire::<P>,
            marker: PhantomData,
        })
    }
    fn constructor_bytes<P: 'static>() -> Option<usize> {
        let shell = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Owner<P>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            shell,
            size_of::<Self>(),
            size_of::<Result<Self, BackendFailure>>(),
            size_of::<Owner<P>>(),
            size_of::<Option<Owner<P>>>(),
            size_of::<Rc<Owner<P>>>(),
            size_of::<Erased>(),
            size_of::<Option<Erased>>(),
            size_of::<Result<Rc<Owner<P>>, Erased>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<HostPreparationAuthority, BackendFailure>>(),
            size_of::<Layout>(),
            size_of::<Result<(Layout, usize), std::alloc::LayoutError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn erased(&self) -> &dyn Any {
        self.owner.as_deref().expect("live embedded copy provider")
    }
    /// Performs one separately admitted exact copy.
    pub fn copy_state(
        &self,
        source: &S,
        evidence: Option<&PreparedEmbeddedEvidence>,
    ) -> Result<PreparedEmbeddedPayload<S>, BackendFailure> {
        if evidence.is_some_and(|value| {
            !Rc::ptr_eq(
                self.owner.as_ref().expect("live provider"),
                value.source.as_ref().expect("live source"),
            )
        }) {
            return Err(self.reject(PreparedEmbeddedCopyError::SourceMismatch));
        }
        let parts = [
            size_of::<PreparedEmbeddedPayload<S>>(),
            size_of::<Result<PreparedEmbeddedPayload<S>, BackendFailure>>(),
            size_of::<Option<&PreparedEmbeddedEvidence>>(),
            size_of::<&S>(),
            size_of::<&Self>(),
            size_of::<Self>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| self.reject(PreparedEmbeddedCopyError::Overflow))?;
        let _controls = self.prepare_host(bytes)?;
        (self.copy)(self.erased(), self, source, evidence)
    }
    /// Retains one already constructed, exact completion/source witness.
    /// E owns source/account facts only, never the state or its native arrays.
    pub fn retain_evidence<E: 'static>(
        &self,
        evidence: E,
    ) -> Result<PreparedEmbeddedEvidence, BackendFailure> {
        let bytes = PreparedEmbeddedEvidence::retained_control_bytes::<E>()
            .ok_or_else(|| self.reject(PreparedEmbeddedCopyError::Overflow))?;
        let host = self.prepare_host(bytes)?;
        Ok(PreparedEmbeddedEvidence::from_parts(evidence, host, self.owner.clone(), self.retire))
    }

    /// Pays a shared-driver destination before its buffer/String/Box birth.
    pub fn prepare_host(&self, bytes: usize) -> Result<HostPreparationAuthority, BackendFailure> {
        (self.host)(self.erased(), bytes)
    }
    pub(super) fn reject(&self, cause: PreparedEmbeddedCopyError) -> BackendFailure {
        (self.reject)(self.erased(), cause)
    }
    pub(super) fn allocation_failure(
        &self,
        cause: std::collections::TryReserveError,
    ) -> BackendFailure {
        (self.allocation)(self.erased(), cause)
    }
    pub(super) fn same_source(&self, other: &Self) -> bool {
        Rc::ptr_eq(
            self.owner.as_ref().expect("live provider"),
            other.owner.as_ref().expect("live provider"),
        )
    }
}

/// One typed prepared native state and its immutable copy provider.
/// Payload retirement precedes both its numerical/host custody and source.
#[must_use]
pub struct PreparedEmbeddedState<S> {
    pub(super) value: S,
    pub(super) evidence: Option<PreparedEmbeddedEvidence>,
    pub(super) host: HostPreparationAuthority,
    pub(super) copy: PreparedEmbeddedCopy<S>,
}
impl<S> PreparedEmbeddedState<S> {
    /// Consumes an actual prepared result; never clones/adopts ordinary storage.
    pub fn new(payload: PreparedEmbeddedPayload<S>, copy: PreparedEmbeddedCopy<S>) -> Self {
        Self {
            value: payload.value,
            evidence: payload.evidence,
            host: payload.host,
            copy,
        }
    }
    /// Borrows actual current native state without changing custody.
    pub const fn value(&self) -> &S {
        &self.value
    }
    /// Mutates state only under the caller's existing invocation/completion rules.
    pub fn value_mut(&mut self) -> &mut S {
        &mut self.value
    }
    /// Copies current state and retains this same source provider.
    pub fn try_copy(&self) -> Result<Self, BackendFailure> {
        let _controls = self.copy.prepare_host(
            size_of::<Self>()
                .checked_add(size_of::<Result<Self, BackendFailure>>())
                .ok_or_else(|| self.copy.reject(PreparedEmbeddedCopyError::Overflow))?,
        )?;
        let copied = self.copy.copy_state(&self.value, self.evidence.as_ref())?;
        Ok(Self::new(copied, self.copy.clone()))
    }
}
impl<S> std::fmt::Debug for PreparedEmbeddedState<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEmbeddedState")
            .finish_non_exhaustive()
    }
}

pub(super) struct CacheOwnership<T, L> {
    pub target: PreparedEmbeddedCopy<T>,
    pub prediction: PreparedEmbeddedCopy<L>,
    pub target_evidence: Option<PreparedEmbeddedEvidence>,
    pub prediction_evidence: Option<PreparedEmbeddedEvidence>,
    pub target_host: HostPreparationAuthority,
    pub prediction_host: HostPreparationAuthority,
    pub metadata_host: HostPreparationAuthority,
}
pub(super) struct PredictionOwnership<L> {
    pub copy: PreparedEmbeddedCopy<L>,
    pub evidence: Option<PreparedEmbeddedEvidence>,
    pub payload_host: HostPreparationAuthority,
    pub metadata_host: HostPreparationAuthority,
}
/// Existing valid identity copied into a fresh exact destination, never adopted.
pub(super) fn identity<S, O>(
    source: &Option<eredu_runtime::SpeculativeIdentity>,
    copy: &PreparedEmbeddedCopy<S>,
) -> Result<
    (
        Option<eredu_runtime::SpeculativeIdentity>,
        HostPreparationAuthority,
    ),
    BackendFailure,
> {
    let length = source.as_ref().map_or(0, |id| id.as_str().len());
    let parts = [
        length,
        size_of::<O>(),
        size_of::<Option<eredu_runtime::SpeculativeIdentity>>(),
        size_of::<String>(),
        size_of::<std::collections::TryReserveError>(),
        size_of::<Result<(), std::collections::TryReserveError>>(),
        size_of::<
            Result<
                (
                    Option<eredu_runtime::SpeculativeIdentity>,
                    HostPreparationAuthority,
                ),
                BackendFailure,
            >,
        >(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or_else(|| copy.reject(PreparedEmbeddedCopyError::Overflow))?;
    let host = copy.prepare_host(bytes)?;
    let identity = source
        .as_ref()
        .map(|source| {
            let mut text = String::new();
            text.try_reserve_exact(length)
                .map_err(|cause| copy.allocation_failure(cause))?;
            text.push_str(source.as_str());
            // Same immutable validated bytes, so no policy error can be introduced.
            Ok::<_, BackendFailure>(
                eredu_runtime::SpeculativeIdentity::new(text).expect("copied validated identity"),
            )
        })
        .transpose()?;
    Ok((identity, host))
}

struct EvidenceOwner<E> {
    evidence: E,
    _host: HostPreparationAuthority,
}
/// Exact per-state completion/source evidence, independently retained by each
/// snapshot branch. A shared copy provider never stores a mutable current witness.
/// The native producer supplies source/account facts after actual completion;
/// holding this carrier does not imply that any new work has completed.
pub struct PreparedEmbeddedEvidence {
    owner: Option<Erased>,
    source: Option<Erased>,
    source_retire: fn(Erased),
    get: fn(&dyn Any) -> &dyn Any,
    retire: fn(Erased),
}
impl Clone for PreparedEmbeddedEvidence {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            source: self.source.clone(),
            source_retire: self.source_retire,
            get: self.get,
            retire: self.retire,
        }
    }
}
impl Drop for PreparedEmbeddedEvidence {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            (self.retire)(owner);
        }
        if let Some(source) = self.source.take() {
            (self.source_retire)(source);
        }
    }
}
fn retire_evidence<E: 'static>(owner: Erased) {
    let value = match owner.downcast::<EvidenceOwner<E>>() {
        Ok(value) => value,
        Err(_) => unreachable!("closed evidence type"),
    };
    drop(Rc::into_inner(value));
}
impl PreparedEmbeddedEvidence {
    /// Exact existing shared destination and constructor/retirement controls.
    /// The producer separately pays the supplied host-authority destination.
    pub fn retained_control_bytes<E: 'static>() -> Option<usize> {
        let shell = Layout::new::<[Cell<usize>; 2]>().extend(Layout::new::<EvidenceOwner<E>>())
            .ok()?.0.pad_to_align().size();
        let parts = [shell, size_of::<EvidenceOwner<E>>(), size_of::<Self>(),
            size_of::<Rc<EvidenceOwner<E>>>(), size_of::<Erased>(), size_of::<Option<Erased>>(),
            size_of::<Result<Rc<EvidenceOwner<E>>, Erased>>(), size_of::<Option<EvidenceOwner<E>>>(),
            size_of::<Result<Self, BackendFailure>>(),
            size_of::<(E, HostPreparationAuthority)>(),
            size_of::<(E, HostPreparationAuthority, Option<Erased>, fn(Erased))>()];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Retains a real completed source after its exact destination was prepaid.
    /// E must contain only source/account facts, never the state or native arrays
    /// which retain this witness. Custody cannot adopt an ordinary allocation.
    pub fn from_prepared<E: 'static>(evidence: E, host: HostPreparationAuthority) -> Self {
        Self::from_parts(evidence, host, None, drop)
    }
    fn from_parts<E: 'static>(evidence: E, host: HostPreparationAuthority,
        source: Option<Erased>, source_retire: fn(Erased)) -> Self {
        Self { owner: Some(Rc::new(EvidenceOwner { evidence, _host: host })),
            source, source_retire,
            get: |owner| &owner.downcast_ref::<EvidenceOwner<E>>().expect("closed evidence type").evidence,
            retire: retire_evidence::<E> }
    }
    /// Borrows the native producer's exact fixed evidence type.
    pub fn get<E: 'static>(&self) -> Option<&E> {
        (self.get)(self.owner.as_deref().expect("live evidence")).downcast_ref()
    }
}
impl std::fmt::Debug for PreparedEmbeddedEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEmbeddedEvidence")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative_execution::EmbeddedPredictionCache;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[derive(Default)]
    struct Account {
        live: AtomicUsize,
        leases: AtomicUsize,
        copies: AtomicUsize,
        refuse: AtomicUsize,
    }
    struct Lease(Arc<Account>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.leases.fetch_sub(1, Ordering::SeqCst);
        }
    }
    struct State {
        value: u32,
        account: Arc<Account>,
    }
    impl State {
        fn new(value: u32, account: &Arc<Account>) -> Self {
            account.live.fetch_add(1, Ordering::SeqCst);
            Self {
                value,
                account: account.clone(),
            }
        }
    }
    impl Clone for State {
        fn clone(&self) -> Self {
            panic!("prepared cache reached ordinary Clone")
        }
    }
    impl Drop for State {
        fn drop(&mut self) {
            self.account.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn completed_evidence_retains_host_until_last_branch_and_retires_payload_first() {
        struct Completed(Arc<Account>);
        impl Drop for Completed {
            fn drop(&mut self) {
                assert_eq!(self.0.leases.load(Ordering::SeqCst), 1);
                self.0.live.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let account = Arc::new(Account::default());
        let provider = Provider(account.clone());
        let bytes = PreparedEmbeddedEvidence::retained_control_bytes::<Completed>().unwrap();
        let host = provider.prepare_host(bytes).unwrap();
        account.live.fetch_add(1, Ordering::SeqCst);
        let evidence = PreparedEmbeddedEvidence::from_prepared(Completed(account.clone()), host);
        let branch = evidence.clone();
        assert!(branch.get::<Completed>().is_some());
        assert!(branch.get::<Receipt>().is_none());
        drop(evidence);
        drop(provider);
        assert_eq!(account.live.load(Ordering::SeqCst), 1);
        assert_eq!(account.leases.load(Ordering::SeqCst), 1);
        drop(branch);
        assert_eq!(account.live.load(Ordering::SeqCst), 0);
        assert_eq!(account.leases.load(Ordering::SeqCst), 0);
    }

    struct Receipt(u32);
    struct Provider(Arc<Account>);
    #[derive(Debug, thiserror::Error)]
    #[error("{cause}")]
    struct Refusal {
        #[source]
        cause: PreparedEmbeddedCopyError,
        _host: HostPreparationAuthority,
    }
    impl PreparedEmbeddedCopyProvider<State> for Provider {
        fn copy_state(
            &self,
            source: &State,
            evidence: Option<&PreparedEmbeddedEvidence>,
        ) -> Result<PreparedEmbeddedPayload<State>, BackendFailure> {
            if let Some(evidence) = evidence {
                if evidence.get::<Receipt>().map(|r| r.0) != Some(source.value) {
                    return Err(self.reject(PreparedEmbeddedCopyError::SourceMismatch));
                }
            }
            let call = self.0.copies.fetch_add(1, Ordering::SeqCst) + 1;
            if self.0.refuse.load(Ordering::SeqCst) == call {
                return Err(self.reject(PreparedEmbeddedCopyError::Overflow));
            }
            let host = self.prepare_host(size_of::<State>())?;
            Ok(PreparedEmbeddedPayload::new(
                State::new(source.value, &self.0),
                host,
            ))
        }
        fn prepare_host(&self, _: usize) -> Result<HostPreparationAuthority, BackendFailure> {
            self.0.leases.fetch_add(1, Ordering::SeqCst);
            Ok(HostPreparationAuthority::retain(Lease(self.0.clone())))
        }
        fn reject(&self, cause: PreparedEmbeddedCopyError) -> BackendFailure {
            BackendFailure::from_error(Refusal {
                cause,
                _host: self.prepare_host(size_of::<Refusal>()).unwrap(),
            })
        }
        fn allocation_failure(&self, cause: std::collections::TryReserveError) -> BackendFailure {
            BackendFailure::from_error(cause)
        }
    }
    fn prepared(value: u32, account: &Arc<Account>) -> PreparedEmbeddedState<State> {
        let provider = Provider(account.clone());
        let payload = PreparedEmbeddedPayload::new(
            State::new(value, account),
            provider.prepare_host(size_of::<State>()).unwrap(),
        );
        PreparedEmbeddedState::new(payload, PreparedEmbeddedCopy::prepare(provider).unwrap())
    }
    fn forbidden(_: &State) -> Result<State, std::convert::Infallible> {
        panic!("ordinary checkpoint used")
    }

    #[test]
    fn prepared_cache_copy_preserves_branch_evidence_refusal_and_final_custody() {
        let account = Arc::new(Account::default());
        let mut live =
            EmbeddedPredictionCache::from_prepared(prepared(17, &account), prepared(23, &account))
                .unwrap();
        assert!(live.retain_target_evidence(Receipt(17)).unwrap());
        assert!(live.prediction_phase_state().retain_evidence(Receipt(23)).unwrap());
        account.refuse.store(2, Ordering::SeqCst);
        let error = match live.checkpoint(forbidden) {
            Err(error) => error,
            Ok(_) => panic!("second-copy refusal was lost"),
        };
        assert_eq!(account.live.load(Ordering::SeqCst), 2);
        assert_eq!(live.target().unwrap().value, 17);
        assert_eq!(live.prediction().value, 23);
        drop(error);
        account.refuse.store(0, Ordering::SeqCst);
        let saved = live.checkpoint(forbidden).unwrap();
        {
            let mut phase = live.prediction_phase_state();
            assert_eq!(phase.evidence().and_then(|value| value.get::<Receipt>()).map(|v| v.0), Some(23));
            phase.state_mut().value = 31;
            assert!(phase.retain_evidence(Receipt(31)).unwrap());
        }
        let branch = live.prediction_fork().unwrap();
        assert_eq!(branch.prediction().value, 31);
        assert_eq!(saved.prediction().value, 23);
        live.restore(&saved, |_, _| -> Result<(), std::convert::Infallible> {
            panic!("ordinary restore used")
        })
        .unwrap();
        assert_eq!(live.prediction().value, 23);
        live.commit_prediction(&branch).unwrap();
        assert_eq!(live.prediction().value, 31);
        let rollback = live.prediction_fork().unwrap();
        live.prediction_mut().value = 47;
        account
            .refuse
            .store(account.copies.load(Ordering::SeqCst) + 1, Ordering::SeqCst);
        let before_rollback = account.copies.load(Ordering::SeqCst);
        live.rollback_prediction(rollback).unwrap();
        assert_eq!(live.prediction().value, 31);
        assert_eq!(account.copies.load(Ordering::SeqCst), before_rollback);
        account.refuse.store(0, Ordering::SeqCst);
        let escaped = branch.try_copy().unwrap();
        drop((live, saved, branch));
        assert_eq!(escaped.prediction().value, 31);
        assert_eq!(account.live.load(Ordering::SeqCst), 1);
        assert!(account.leases.load(Ordering::SeqCst) > 0);
        drop(escaped);
        assert_eq!(account.live.load(Ordering::SeqCst), 0);
        assert_eq!(account.leases.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn immutable_fused_output_keeps_its_source_after_branch_replacement_without_tensor_clone() {
        use crate::speculative_execution::EmbeddedPredictionLogitBlock;
        let account = Arc::new(Account::default());
        let copy = PreparedEmbeddedCopy::<State>::prepare(Provider(account.clone())).unwrap();
        let mut latest = copy.retain_evidence(Receipt(23)).unwrap();
        let host = copy.prepare_host(EmbeddedPredictionLogitBlock::<State>::retained_control_bytes().unwrap()).unwrap();
        let block = EmbeddedPredictionLogitBlock::from_prepared(State::new(23, &account), latest.clone(), host);
        let escaped = block.clone(); // State::clone deliberately panics.
        latest = copy.retain_evidence(Receipt(31)).unwrap();
        assert_eq!(latest.get::<Receipt>().unwrap().0, 31);
        drop((block, latest, copy));
        assert_eq!(escaped.value, 23);
        assert_eq!(escaped.evidence().unwrap().get::<Receipt>().unwrap().0, 23);
        assert_eq!(account.copies.load(Ordering::SeqCst), 0);
        assert_eq!(account.live.load(Ordering::SeqCst), 1);
        assert!(account.leases.load(Ordering::SeqCst) > 0);
        drop(escaped);
        assert_eq!(account.live.load(Ordering::SeqCst), 0);
        assert_eq!(account.leases.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn immutable_tensor_sharing_and_explicit_copy_preserve_separate_custody() {
        use crate::speculative_execution::EmbeddedPredictionTensor;
        let account = Arc::new(Account::default());
        let mut state = prepared(41, &account);
        state.evidence = Some(state.copy.retain_evidence(Receipt(41)).unwrap());
        let host = state.copy.prepare_host(
            EmbeddedPredictionTensor::<State>::retained_state_control_bytes().unwrap(),
        ).unwrap();
        let tensor = EmbeddedPredictionTensor::from_prepared_state(state, host);
        let escaped = tensor.clone(); // The native-state Clone above must never run.
        assert_eq!(account.copies.load(Ordering::SeqCst), 0);
        let copied = tensor.try_copy_prepared().unwrap().unwrap();
        assert_eq!(account.copies.load(Ordering::SeqCst), 1);
        assert_eq!(copied.value, 41);
        // This provider returns a fresh copy, with no stale completion receipt.
        assert!(copied.evidence().is_none());
        assert_eq!(escaped.evidence().unwrap().get::<Receipt>().unwrap().0, 41);
        account.refuse.store(2, Ordering::SeqCst);
        let failure = tensor.try_copy_prepared().unwrap_err();
        assert_eq!(account.copies.load(Ordering::SeqCst), 2);
        assert_eq!(account.live.load(Ordering::SeqCst), 2);
        assert_eq!(escaped.value, 41);
        drop((failure, tensor));
        assert_eq!(account.live.load(Ordering::SeqCst), 2);
        drop(copied);
        assert_eq!(account.live.load(Ordering::SeqCst), 1);
        assert!(account.leases.load(Ordering::SeqCst) > 0);
        drop(escaped);
        assert_eq!(account.live.load(Ordering::SeqCst), 0);
        assert_eq!(account.leases.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn immutable_view_keeps_predecessor_until_fresh_copy_emits_its_own_evidence() {
        use crate::speculative_execution::EmbeddedPredictionTensor;
        struct Fresh(Provider);
        struct CopyReceipt(usize);
        impl PreparedEmbeddedCopyProvider<State> for Fresh {
            fn copy_state(&self,value:&State,evidence:Option<&PreparedEmbeddedEvidence>)
                ->Result<PreparedEmbeddedPayload<State>,BackendFailure>{self.0.copy_state(value,evidence)}
            fn copy_state_with_owner(&self,owner:&PreparedEmbeddedCopy<State>,value:&State,
                evidence:Option<&PreparedEmbeddedEvidence>)->Result<PreparedEmbeddedPayload<State>,BackendFailure>{
                let output=self.0.copy_state(value,evidence)?;
                let serial=self.0.0.copies.load(Ordering::SeqCst);
                Ok(output.with_evidence(owner.retain_evidence(CopyReceipt(serial))?))
            }
            fn prepare_host(&self,bytes:usize)->Result<HostPreparationAuthority,BackendFailure>{self.0.prepare_host(bytes)}
            fn reject(&self,cause:PreparedEmbeddedCopyError)->BackendFailure{self.0.reject(cause)}
            fn allocation_failure(&self,cause:std::collections::TryReserveError)->BackendFailure{self.0.allocation_failure(cause)}
        }
        let account=Arc::new(Account::default());
        let parent=prepared(17,&account);
        let host=parent.copy.prepare_host(EmbeddedPredictionTensor::<State>::retained_state_control_bytes().unwrap()).unwrap();
        let parent=EmbeddedPredictionTensor::from_prepared_state(parent,host);
        let copy=PreparedEmbeddedCopy::prepare(Fresh(Provider(account.clone()))).unwrap();
        let host=copy.prepare_host(EmbeddedPredictionTensor::<State>::retained_state_control_bytes().unwrap()).unwrap();
        let evidence=copy.retain_evidence(Receipt(23)).unwrap();
        let state=PreparedEmbeddedState::new(PreparedEmbeddedPayload::new(State::new(23,&account),host.clone())
            .with_evidence(evidence),copy);
        let view=EmbeddedPredictionTensor::from_prepared_state_with_source(state,host,Some(parent.clone()));
        drop(parent);
        assert_eq!(account.live.load(Ordering::SeqCst),2);
        let copied=view.try_copy_prepared().unwrap().unwrap();
        assert_eq!(copied.evidence().unwrap().get::<CopyReceipt>().unwrap().0,1);
        assert!(view.evidence().unwrap().get::<CopyReceipt>().is_none());
        drop(view);
        // Independent copy has no dependency on the old view's source chain.
        assert_eq!(account.live.load(Ordering::SeqCst),1);
        assert_eq!(copied.value,23);
        drop(copied);
        assert_eq!(account.live.load(Ordering::SeqCst),0);
        assert_eq!(account.leases.load(Ordering::SeqCst),0);
    }

    #[test]
    fn immutable_concatenation_keeps_both_predecessors_through_escape() {
        use crate::speculative_execution::EmbeddedPredictionTensor;
        let account=Arc::new(Account::default());
        let packet=|value|{
            let mut state=prepared(value,&account);
            state.evidence=Some(state.copy.retain_evidence(Receipt(value)).unwrap());
            let host=state.copy.prepare_host(EmbeddedPredictionTensor::<State>::retained_state_control_bytes().unwrap()).unwrap();
            EmbeddedPredictionTensor::from_prepared_state(state,host)
        };
        let left=packet(17);
        let right=packet(23);
        let state=prepared(40,&account);
        let host=state.copy.prepare_host(EmbeddedPredictionTensor::<State>::retained_state_control_bytes().unwrap()).unwrap();
        let joined=EmbeddedPredictionTensor::from_prepared_state_with_sources(state,host,[Some(left.clone()),Some(right.clone())]);
        let escaped=joined.clone();
        drop((left,right,joined));
        assert_eq!(account.live.load(Ordering::SeqCst),3);
        assert_eq!(account.copies.load(Ordering::SeqCst),0);
        assert_eq!(escaped.value,40);
        let independent=escaped.try_copy_prepared().unwrap().unwrap();
        drop(escaped);
        assert_eq!(account.live.load(Ordering::SeqCst),1);
        assert_eq!(independent.value,40);
        drop(independent);
        assert_eq!(account.live.load(Ordering::SeqCst),0);
        assert_eq!(account.leases.load(Ordering::SeqCst),0);
    }

}
