//! Immutable prediction tensors retain their source through borrowed consumers.
use super::{PreparedEmbeddedEvidence, PreparedEmbeddedState, PreparedEmbeddedCopy, PreparedEmbeddedCopyError};
use eredu_core::HostPreparationAuthority;
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
    rc::Rc,
};
struct Payload<T> {
    value: T,
    evidence: Option<PreparedEmbeddedEvidence>,
    sources: [Option<EmbeddedPredictionTensor<T>>; 2],
    copy: Option<PreparedEmbeddedCopy<T>>,
    _payload_host: Option<HostPreparationAuthority>,
    _host: HostPreparationAuthority,
}
struct Owner<T>(Option<Rc<Payload<T>>>);
impl<T> Owner<T> {
    fn get(&self) -> &Payload<T> {
        self.0.as_deref().expect("live immutable prediction tensor")
    }
}
impl<T> Clone for Owner<T> {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(
            self.0.as_ref().expect("live immutable prediction tensor"),
        )))
    }
}
impl<T> Drop for Owner<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
enum Storage<T> {
    Ordinary(T),
    Prepared(Owner<T>),
}
/// Backend tensor and its particular completed source. Ordinary clones retain
/// ordinary behavior; prepared clones share the same immutable descriptor and
/// evidence without creating an unpriced native handle or a new copy witness.
pub struct EmbeddedPredictionTensor<T>(Storage<T>);
/// The fused logits spelling retains its existing fixed immutable owner.
pub type EmbeddedPredictionLogitBlock<T> = EmbeddedPredictionTensor<T>;
impl<T> EmbeddedPredictionTensor<T> {
    /// Existing ordinary ownership; no new allocation.
    pub const fn ordinary(value: T) -> Self {
        Self(Storage::Ordinary(value))
    }
    /// Actual shared destination and constructor/retirement controls, excluding
    /// the already owned tensor/evidence and caller's host-authority producer.
    pub fn retained_control_bytes() -> Option<usize> {
        let shared = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Payload<T>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            shared,
            size_of::<Self>(),
            size_of::<Storage<T>>(),
            size_of::<Payload<T>>(),
            size_of::<Owner<T>>(),
            size_of::<Option<Payload<T>>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<PreparedEmbeddedEvidence>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Final native binding after source authentication and destination debit.
    /// Custody alone does not adopt an earlier ordinary tensor allocation.
    pub fn from_prepared(
        value: T,
        evidence: PreparedEmbeddedEvidence,
        host: HostPreparationAuthority,
    ) -> Self {
        Self(Storage::Prepared(Owner(Some(Rc::new(Payload {
            value,
            evidence: Some(evidence),
            sources: [None, None],
            copy: None,
            _payload_host: None,
            _host: host,
        })))))
    }
    /// Fixed prepared-state handoff frames in addition to the shared owner.
    pub fn retained_state_control_bytes() -> Option<usize> {
        Self::retained_control_bytes()?
            .checked_add(size_of::<PreparedEmbeddedState<T>>())?
            .checked_add(size_of::<Option<PreparedEmbeddedCopy<T>>>())?
            .checked_add(size_of::<Option<HostPreparationAuthority>>())?
            .checked_add(size_of::<[Option<Self>; 2]>())
    }
    /// Moves a real prepared payload and its existing copy provider into the
    /// same shared descriptor. The caller has paid retained_state_control_bytes.
    pub fn from_prepared_state(state: PreparedEmbeddedState<T>, host: HostPreparationAuthority) -> Self {
        Self::from_prepared_state_with_source(state, host, None)
    }
    /// An actual view keeps its immutable source packet's numerical/host
    /// custody as well as the backing evidence, without cloning a native handle.
    pub fn from_prepared_state_with_source(
        state: PreparedEmbeddedState<T>, host: HostPreparationAuthority, source: Option<Self>,
    ) -> Self {
        Self::from_prepared_state_with_sources(state, host, [source, None])
    }
    /// Two-input numerical producers retain both exact immutable source packets
    /// in the same fixed owner, with no secondary collection or native clone.
    pub fn from_prepared_state_with_sources(
        state: PreparedEmbeddedState<T>, host: HostPreparationAuthority, sources: [Option<Self>; 2],
    ) -> Self {
        Self(Storage::Prepared(Owner(Some(Rc::new(Payload {
            value: state.value,
            evidence: state.evidence,
            sources,
            copy: Some(state.copy),
            _payload_host: Some(state.host),
            _host: host,
        })))))
    }
    /// Performs the provider's existing separately admitted copy. Ordinary
    /// values and fused descriptors without a mutable-state copier return None.
    /// Their caller retains its ordinary operation or immutable sharing policy.
    pub fn try_copy_prepared(&self) -> Result<Option<Self>, eredu_core::BackendFailure> {
        let Storage::Prepared(owner) = &self.0 else { return Ok(None); };
        let source = owner.get();
        let Some(copy) = &source.copy else { return Ok(None); };
        let bytes = Self::retained_state_control_bytes()
            .and_then(|bytes| bytes.checked_add(size_of::<Result<Option<Self>, eredu_core::BackendFailure>>()))
            .ok_or_else(|| copy.reject(PreparedEmbeddedCopyError::Overflow))?;
        let host = copy.prepare_host(bytes)?;
        let copied = copy.copy_state(&source.value, source.evidence.as_ref())?;
        Ok(Some(Self::from_prepared_state(PreparedEmbeddedState::new(copied, copy.clone()), host)))
    }
    /// Exact evidence retained for this tensor, independent of current state.
    pub fn evidence(&self) -> Option<&PreparedEmbeddedEvidence> {
        match &self.0 {
            Storage::Ordinary(_) => None,
            Storage::Prepared(owner) => owner.get().evidence.as_ref(),
        }
    }
}
impl<T: Clone> Clone for EmbeddedPredictionTensor<T> {
    fn clone(&self) -> Self {
        Self(match &self.0 {
            Storage::Ordinary(value) => Storage::Ordinary(value.clone()),
            Storage::Prepared(owner) => Storage::Prepared(owner.clone()),
        })
    }
}
impl<T> std::ops::Deref for EmbeddedPredictionTensor<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match &self.0 {
            Storage::Ordinary(value) => value,
            Storage::Prepared(owner) => &owner.get().value,
        }
    }
}
impl<T> std::fmt::Debug for EmbeddedPredictionTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EmbeddedPredictionTensor")
    }
}
