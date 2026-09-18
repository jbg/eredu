//! Charges that remain live with state, including rollback and branch owners.

use super::{
    InferenceExecutionIdentity, InferenceRequest, WorkingMemoryError, WorkingMemoryUnquotedLease,
};
use eredu_core::OutputDemand;
use std::sync::{Arc, OnceLock};

/// Identity of one retained branch revision, independent of its logical frontier.
/// Clones preserve identity for cold inspection. Restoration and state changes
/// issue another identity; a saved revision never authorizes a restored branch.
#[derive(Debug, Clone)]
pub struct InferenceStateRevision(RevisionIdentity);
#[derive(Debug)]
struct ResetRevision {
    _custody: super::resident_reset::ResetCustody,
}
struct PlannedRevision {
    _funding: eredu_nn::workspace::HostMetadataFunding,
}
impl std::fmt::Debug for PlannedRevision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PlannedRevision")
    }
}
#[derive(Debug)]
enum RevisionIdentity {
    Ordinary(Arc<()>),
    Planned(Option<Arc<PlannedRevision>>),
    Reset(Option<Arc<ResetRevision>>),
}
impl Clone for RevisionIdentity {
    fn clone(&self) -> Self {
        match self {
            Self::Ordinary(owner) => Self::Ordinary(Arc::clone(owner)),
            Self::Planned(owner) => Self::Planned(Some(Arc::clone(
                owner.as_ref().expect("live planned revision"),
            ))),
            Self::Reset(owner) => {
                Self::Reset(Some(Arc::clone(owner.as_ref().expect("live revision"))))
            }
        }
    }
}
impl Drop for RevisionIdentity {
    fn drop(&mut self) {
        match self {
            Self::Reset(owner) => {
                if let Some(owner) = owner.take() {
                    drop(Arc::into_inner(owner));
                }
            }
            Self::Planned(owner) => {
                if let Some(owner) = owner.take() {
                    drop(Arc::into_inner(owner));
                }
            }
            Self::Ordinary(_) => {}
        }
    }
}
impl PartialEq for InferenceStateRevision {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (RevisionIdentity::Ordinary(a), RevisionIdentity::Ordinary(b)) => Arc::ptr_eq(a, b),
            (RevisionIdentity::Planned(a), RevisionIdentity::Planned(b)) => Arc::ptr_eq(
                a.as_ref().expect("live planned revision"),
                b.as_ref().expect("live planned revision"),
            ),
            (RevisionIdentity::Reset(a), RevisionIdentity::Reset(b)) => Arc::ptr_eq(
                a.as_ref().expect("live revision"),
                b.as_ref().expect("live revision"),
            ),
            _ => false,
        }
    }
}
impl Eq for InferenceStateRevision {}
impl InferenceStateRevision {
    fn new() -> Self {
        Self(RevisionIdentity::Ordinary(Arc::new(())))
    }
    fn metadata_control_bytes()->Option<usize> {
        use std::mem::size_of;
        let allocation=std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize;2]>()
            .extend(std::alloc::Layout::new::<PlannedRevision>()).ok()?.0.pad_to_align().size();
        allocation.checked_add(size_of::<(Self,PlannedRevision,Option<Self>,
            Result<Self,eredu_nn::workspace::HostMetadataFundingError>)>())
    }
    pub(crate) fn prepare_metadata(
        funding:&eredu_nn::workspace::HostMetadataFunding,
    )->Result<Self,eredu_nn::workspace::HostMetadataFundingError> {
        let bytes=Self::metadata_control_bytes()
            .ok_or(eredu_nn::workspace::HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(Self(RevisionIdentity::Planned(Some(Arc::new(
            PlannedRevision {
                _funding: funding.clone(),
            },
        )))))
    }
    pub(super) fn reset_control_bytes() -> Option<usize> {
        let allocation = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<ResetRevision>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        allocation
            .checked_add(std::mem::size_of::<ResetRevision>())?
            .checked_add(std::mem::size_of::<Option<ResetRevision>>())?
            .checked_add(std::mem::size_of::<InferenceStateRevision>())
    }
}

/// Exact request and logical frontier carried with one installed state branch.
/// Restoring state restores this frontier, but never refunds backing charges.
#[derive(Debug, Clone)]
pub struct InferenceStateAdmission {
    request: InferenceRequest,
    position: u64,
}

impl InferenceStateAdmission {
    /// Request whose completed-span equations price the remaining work.
    pub fn request(&self) -> &InferenceRequest {
        &self.request
    }
    /// Next logical decoder position, including on explicitly stateless ranks.
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Rejects unquoted geometry before checkpointing or model execution.
    pub fn validate_span(
        &self,
        execution: &InferenceExecutionIdentity,
        shape: Option<[u64; 2]>,
        prefill: bool,
        output: OutputDemand,
        native_position: Option<u64>,
    ) -> Result<u64, WorkingMemoryError> {
        let geometry = self.request.geometry();
        self.request.validate(execution, geometry)?;
        if let Some(actual) = native_position {
            if actual != self.position {
                return Err(WorkingMemoryError::StateFrontierMismatch {
                    expected: self.position,
                    actual,
                });
            }
        }
        let prompt_end = geometry.cached_positions + geometry.input_positions;
        let limit = prompt_end + geometry.max_output_tokens;
        if self.position >= limit {
            return Err(WorkingMemoryError::OutputAllowanceExceeded {
                position: self.position,
                limit,
            });
        }
        if prefill != (self.position < prompt_end) {
            return Err(WorkingMemoryError::InvocationPhaseMismatch);
        }
        let positions = if prefill {
            geometry
                .prefill_chunk_positions
                .min(prompt_end - self.position)
        } else {
            1
        };
        let actual = shape.ok_or(WorkingMemoryError::UnknownBound)?;
        let expected = [geometry.batch_size, positions];
        if actual != expected {
            return Err(WorkingMemoryError::SpanShapeMismatch { expected, actual });
        }
        let admitted = if prefill {
            geometry
                .output
                .for_chunk(self.position + positions == prompt_end)
        } else {
            OutputDemand::LastPosition
        };
        let same_single_row = positions == 1
            && admitted != OutputDemand::StateOnly
            && output != OutputDemand::StateOnly;
        if admitted != output && !same_single_row {
            return Err(WorkingMemoryError::OutputDemandMismatch {
                admitted,
                required: output,
            });
        }
        Ok(self.position + positions)
    }
}

/// Request charges and unquoted ownership backing retained state. Clones retain
/// the same owners; they do not reserve additional storage or grant another
/// prefill authority. Independently allocated copies still need their own
/// admitted copy/growth bound or separately acquired unquoted ownership.
///
/// There is deliberately no clear operation. Restoration and partial reset can
/// leave backing allocations alive, so they cannot refund a request. A complete
/// owner replacement releases its charges when all descendants and outstanding
/// native completion owners have released theirs.
#[derive(Debug, Default)]
pub struct InferenceRetention {
    requests: Vec<InferenceRequest>,
    admission: Option<InferenceStateAdmission>,
    unquoted: Vec<WorkingMemoryUnquotedLease>,
    // Ordinary lazy identity keeps empty/stateless construction const. A
    // closed original reset installs its funded revision before publication;
    // escaped revision aliases retain that original account without payload.
    revision: OnceLock<InferenceStateRevision>,
}

impl Clone for InferenceRetention {
    fn clone(&self) -> Self {
        Self {
            requests: self.requests.clone(),
            admission: self.admission.clone(),
            unquoted: self.unquoted.clone(),
            revision: OnceLock::from(self.revision().clone()),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        // State restoration must not replace newer charges with an older set.
        self.restore_admission(source);
    }
}

impl InferenceRetention {
    /// Exact handle directories and lazy revision metadata used by a copied
    /// retention owner. Shared requests and leases are retained, never reissued.
    pub fn host_clone_bytes(&self)->Option<usize> {
        use std::mem::{size_of,size_of_val};
        let parts=[
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<InferenceRequest>(self.requests.len())?,
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<WorkingMemoryUnquotedLease>(self.unquoted.len())?,
            if self.revision.get().is_none(){InferenceStateRevision::metadata_control_bytes()?}else{0},
            size_of::<(Self,&Self,eredu_nn::workspace::HostMetadataFunding,Option<InferenceStateRevision>)>(),
            size_of::<Result<Self,eredu_core::BackendFailure>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Copies exact existing charge handles under one paid metadata source.
    /// Restoring or cloning this value grants no submission or budget refund.
    /// The caller retains funding after the returned directories.
    pub fn clone_with_host_source(&self,funding:&eredu_core::HostMetadataFunding)
        ->Result<Self,eredu_core::BackendFailure> {
        use eredu_nn::workspace::HostMetadataFunding;
        let funding=funding.clone();
        // Actual dynamic directories and revision producer reserve themselves.
        let fixed=std::mem::size_of::<(Self,&Self,HostMetadataFunding,Option<InferenceStateRevision>)>()
            .checked_add(std::mem::size_of::<Result<Self,eredu_core::BackendFailure>>())
            .and_then(|n|n.checked_add(std::mem::size_of::<[usize;5]>()))
            .ok_or(eredu_core::HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(fixed)?;
        self.initialize_metadata_revision(&funding).map_err(eredu_core::BackendFailure::from_error)?;
        let mut requests=funding.metadata_vec(self.requests.len()).map_err(eredu_core::BackendFailure::from_error)?;
        requests.extend(self.requests.iter().cloned());
        let mut unquoted=funding.metadata_vec(self.unquoted.len()).map_err(eredu_core::BackendFailure::from_error)?;
        unquoted.extend(self.unquoted.iter().cloned());
        Ok(Self{requests,unquoted,admission:self.admission.clone(),
            revision:OnceLock::from(self.revision.get().expect("funded initialized revision").clone())})
    }

    /// Initializes only the missing identity using its exact accounted producer.
    /// A competing publication retains its existing identity; the paid losing
    /// candidate retires normally and its spent metadata is never refunded.
    pub(crate) fn initialize_metadata_revision(
        &self,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<(), eredu_nn::workspace::HostMetadataFundingError> {
        if self.revision.get().is_none() {
            let revision = InferenceStateRevision::prepare_metadata(funding)?;
            let _ = self.revision.set(revision);
        }
        Ok(())
    }

    /// Existing revision without initializing the lazy identity allocation.
    pub(crate) fn initialized_revision(&self) -> Option<&InferenceStateRevision> {
        self.revision.get()
    }

    pub(super) fn install_original_reset_revision(
        &mut self,
        custody: super::resident_reset::ResetCustody,
    ) -> Result<(), WorkingMemoryError> {
        if !self.is_empty() || self.revision.get().is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let revision =
            InferenceStateRevision(RevisionIdentity::Reset(Some(Arc::new(ResetRevision {
                _custody: custody,
            }))));
        self.revision
            .set(revision)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)
    }

    /// Empty retention has no admitted charge or unquoted owner.
    pub const fn new() -> Self {
        Self {
            requests: Vec::new(),
            admission: None,
            unquoted: Vec::new(),
            revision: OnceLock::new(),
        }
    }

    /// Whether no request, unquoted owner or active span admission is retained.
    /// A revision marker alone is identity metadata and does not make this
    /// owner nonempty. This read-only check grants no state or work authority.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty() && self.unquoted.is_empty() && self.admission.is_none()
    }

    /// Borrows an already established revision without allocating an identity.
    pub(crate) fn established_revision(&self) -> Option<&InferenceStateRevision> {
        self.revision.get()
    }

    /// Exact current branch revision. Reading or cloning this metadata does not
    /// admit work, change a frontier or reserve/release managed storage.
    pub fn revision(&self) -> &InferenceStateRevision {
        self.revision.get_or_init(InferenceStateRevision::new)
    }

    /// Rejects evidence from another branch revision, including equal-position
    /// restoration. This grants no submission or native allocation authority.
    pub fn validate_revision(
        &self,
        expected: &InferenceStateRevision,
    ) -> Result<(), WorkingMemoryError> {
        if self.revision() == expected {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// Shared state lifecycle only: invalidate quotes without changing charge
    /// ownership. Native adapters use admission/restoration rather than minting
    /// or restoring revision identities themselves.
    pub(crate) fn invalidate_revision(&mut self) {
        self.revision = OnceLock::new();
    }

    /// Called only immediately after the shared exchange invalidates the old
    /// revision. The caller prepared this fresh owner before the state move.
    pub(crate) fn install_exchanged_revision(&mut self, revision: InferenceStateRevision) {
        debug_assert!(self.revision.get().is_none());
        self.revision = OnceLock::from(revision);
    }

    /// Retains an exact reserved request once, regardless of cloned handles.
    /// Explicitly unbudgeted execution adds no accounting metadata.
    pub fn retain(&mut self, request: &InferenceRequest) {
        if request.memory_reservation().is_some()
            && !self
                .requests
                .iter()
                .any(|retained| retained.validate_same_request(request).is_ok())
        {
            self.requests.push(request.clone());
        }
    }

    /// Retains one already acquired unquoted owner through this state and its
    /// descendants, including states with no tensor storage. Clones of the same
    /// lease are retained once; independent owners in one pool remain distinct.
    ///
    /// This grants no byte bound or request admission and cannot release or
    /// promote an owner. Acquire the lease before allocating unquoted work, and
    /// retain independent completion owners until that work safely settles.
    pub fn retain_unquoted(&mut self, lease: &WorkingMemoryUnquotedLease) {
        if !self
            .unquoted
            .iter()
            .any(|retained| std::sync::Arc::ptr_eq(&retained.0, &lease.0))
        {
            self.unquoted.push(lease.clone());
        }
    }

    /// Preserves both current and imported backing charges before restoration.
    pub fn extend_from(&mut self, source: &Self) {
        for request in &source.requests {
            self.retain(request);
        }
        for lease in &source.unquoted {
            self.retain_unquoted(lease);
        }
    }

    /// Current branch's admitted span sequence, separate from backing retention.
    pub fn admission(&self) -> Option<&InferenceStateAdmission> {
        self.admission.as_ref()
    }

    /// Begins a reserved request once, preserving advancement on later chunks.
    pub fn admit(&mut self, request: &InferenceRequest) {
        if request.memory_reservation().is_none() {
            return;
        }
        self.retain(request);
        if self
            .admission
            .as_ref()
            .is_none_or(|active| active.request.validate_same_request(request).is_err())
        {
            self.admission = Some(InferenceStateAdmission {
                request: request.clone(),
                position: request.geometry().cached_positions,
            });
            self.invalidate_revision();
        }
    }

    /// Publishes a frontier already checked by `validate_span` after execution.
    pub(crate) fn commit_span(&mut self, end: u64) {
        if let Some(admission) = &mut self.admission {
            admission.position = end;
        }
        self.invalidate_revision();
    }

    /// Restores logical request state while preserving both backing charge sets.
    pub fn restore_admission(&mut self, source: &Self) {
        self.invalidate_revision();
        self.extend_from(source);
        if let Some(admission) = &source.admission {
            self.admission = Some(admission.clone());
        }
    }

    /// Exact charge owners, without new submission or release authority.
    pub fn requests(&self) -> impl ExactSizeIterator<Item = &InferenceRequest> {
        self.requests.iter()
    }

    /// Logical out-of-line handle storage for an independently copied owner.
    /// Shared request, lease and pool allocations are not copied and are
    /// excluded here. A newly acquired copy owner requires its own allowance.
    pub fn logical_metadata_bytes(&self) -> Option<u64> {
        u64::try_from(self.requests.len())
            .ok()?
            .checked_mul(std::mem::size_of::<InferenceRequest>() as u64)
            .and_then(|bytes| {
                bytes.checked_add(
                    u64::try_from(self.unquoted.len())
                        .ok()?
                        .checked_mul(std::mem::size_of::<WorkingMemoryUnquotedLease>() as u64)?,
                )
            })
    }
}

/// State mechanism retaining accounting owners with all of its backing storage.
///
/// Implementations propagate retention through checkpoints, independent copies,
/// state exchange and imported segments. Restore merges charges before any
/// fallible native work; it never replaces newer accounting with an old snapshot.
/// Native completion and recovery retain their own request handles until safe
/// release. Merely retaining state does not establish completion or price a copy.
pub trait InferenceStateRetention {
    /// Charges retained by this state owner and inherited by descendants.
    fn inference_retention(&self) -> &InferenceRetention;

    /// Mutates the branch's admission only through the shared state lifecycle.
    fn inference_retention_mut(&mut self) -> &mut InferenceRetention;

    /// Attaches a reservation before input preparation or state mutation.
    fn retain_inference(&mut self, request: &InferenceRequest);

    /// Imports all backing charge owners before copying or restoring state.
    fn inherit_inference_retention(&mut self, source: &impl InferenceStateRetention) {
        for request in source.inference_retention().requests() {
            self.retain_inference(request);
        }
        for lease in &source.inference_retention().unquoted {
            self.inference_retention_mut().retain_unquoted(lease);
        }
        if self.inference_retention().admission.is_none() {
            let source = source.inference_retention();
            let target = self.inference_retention_mut();
            target.admission = source.admission.clone();
            target.revision = OnceLock::from(source.revision().clone());
        }
    }
}

/// Exchanges complete state without copying or releasing accounting owners.
/// Both placements receive new revisions, even when their frontiers are equal.
pub(crate) fn exchange_inference_state<S: InferenceStateRetention>(
    installed: &mut S,
    displaced: &mut S,
) {
    std::mem::swap(installed, displaced);
    installed.inference_retention_mut().invalidate_revision();
    displaced.inference_retention_mut().invalidate_revision();
}

#[cfg(test)]
mod tests;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
