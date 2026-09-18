//! Closed native truth for the selected neutral publication mechanism.
use super::StorageIdentity;
use eredu_runtime::working_memory::{
    NativeStorageObservation, NativeStorageRegistration, NativeStorageSelection,
    OriginalNativeBudgetCustody, OriginalNativeStorageMechanism,
};
use safemlx::{
    Array, OrdinaryBufferInspection, OrdinaryBufferWitness, OriginalBufferAliasWitness,
    OriginalBufferBudget, OriginalBufferCause, OriginalBufferWitness, PreparedAllocationOwner,
    PreparedAllocationOwnerCause, PreparedInputRuntime, PreparedOriginalBufferBudget,
};
use std::{fmt, rc::Rc};

mod control_storage;
mod prompt_input;
mod selected;
mod work_rows;
pub(crate) use selected::{NativeProgramStorage, NativeSamplingStorage};

pub(crate) type Registration = NativeStorageRegistration<StorageIdentity>;
pub(crate) type Bank = eredu_runtime::working_memory::OriginalNativeStorageBank<MlxNativeStorage>;
pub(crate) struct BankOwner(Option<Rc<std::cell::RefCell<Bank>>>);
impl BankOwner {
    pub(crate) fn new(bank: Bank) -> Self {
        Self(Some(Rc::new(std::cell::RefCell::new(bank))))
    }
    pub(crate) fn configure_preparation_scope(
        &self,
        scope: &mut safemlx::SubmissionScope,
        controls: &eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> Result<(), crate::backend::error::Error> {
        // The recovery already owns this empty scope. Every failure retires or
        // quarantines it before releasing the same preparation Work/custody.
        let wrap = |cause| {
            retained_failure(
                eredu_runtime::working_memory::NativeStorageError::Native(cause),
                controls.clone(),
                true,
            )
        };
        scope
            .enable_scoped_observation()
            .map_err(|cause| wrap(NativeStorageCause::Observation(cause)))?;
        scope
            .require_original_native_controls()
            .map_err(|cause| wrap(NativeStorageCause::Control(cause)))?;
        let prepared =
            safemlx::PreparedPrefillFailure::try_new(controls.clone()).map_err(|failure| {
                let (cause, _custody) = failure.into_parts();
                wrap(NativeStorageCause::Carrier(cause))
            })?;
        let failure = prepared.try_allocate().map_err(|failure| {
            let (cause, _preparation) = failure.into_parts();
            wrap(NativeStorageCause::Carrier(cause))
        })?;
        failure
            .bind_original_scope(scope)
            .map_err(|cause| wrap(NativeStorageCause::Control(cause)))?;
        scope
            .enable_original_native_controls()
            .map_err(|cause| wrap(NativeStorageCause::Control(cause)))?;
        self.bind_scope(scope, controls)
    }

    pub(crate) fn bind_scope(
        &self,
        scope: &mut safemlx::SubmissionScope,
        controls: &eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> Result<(), crate::backend::error::Error> {
        let budget = {
            let bank = self
                .try_borrow()
                .map_err(|_| crate::backend::error::Error::PrefillScopeReentrant)?;
            bank.budget_for_controls(controls)
                .map_err(crate::backend::error::Error::PrefillControl)?
                .clone()
        };
        // Scope configuration checks the actual accepted native role. No bank,
        // source or Usage loan remains across the native call.
        scope.bind_original_buffer_budget(&budget).map_err(|cause| {
            retained_failure(
                eredu_runtime::working_memory::NativeStorageError::Native(
                    NativeStorageCause::Fixed(cause),
                ),
                controls.clone(),
                true,
            )
        })
    }
}
impl Clone for BankOwner {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.0.as_ref().expect("live native bank"))))
    }
}
impl std::ops::Deref for BankOwner {
    type Target = std::cell::RefCell<Bank>;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("live native bank")
    }
}
impl Drop for BankOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // No Rc or Weak escapes. Free the shared bank allocation before
            // the extracted budget/control/partition payload can release Q/P.
            drop(Rc::into_inner(owner));
        }
    }
}
impl fmt::Debug for BankOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeStorageBankOwner")
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct PublicFailure {
    site: &'static str,
    cause: Box<eredu_runtime::working_memory::NativeStorageError<NativeStorageCause>>,
    // SharedBackendFailure retires its source allocation before this payload.
    // The inner cause box retires before the accepted original control guard.
    _controls: eredu_runtime::working_memory::OriginalTextControlGuard,
}
impl fmt::Display for PublicFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.site, self.cause)
    }
}
impl std::error::Error for PublicFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}
pub(crate) fn retained_failure(
    cause: eredu_runtime::working_memory::NativeStorageError<NativeStorageCause>,
    controls: eredu_runtime::working_memory::OriginalTextControlGuard,
    state_preserved: bool,
) -> crate::backend::error::Error {
    retained_failure_at(cause, controls, state_preserved, "native storage")
}

pub(crate) fn retained_failure_at(
    cause: eredu_runtime::working_memory::NativeStorageError<NativeStorageCause>,
    controls: eredu_runtime::working_memory::OriginalTextControlGuard,
    state_preserved: bool,
    site: &'static str,
) -> crate::backend::error::Error {
    crate::backend::error::Error::retained_original(
        eredu_core::SharedBackendFailure::new(
            eredu_core::BackendFailureKind::Other,
            PublicFailure {
                site,
                cause: Box::new(cause),
                _controls: controls,
            },
        ),
        state_preserved,
    )
}

pub(crate) struct MlxNativeStorage {
    runtime: Result<Rc<PreparedInputRuntime>, eredu_core::SharedBackendFailure>,
    selection: NativeStorageSelection,
    initial_publication: Option<super::RetainedStoragePublication>,
}
impl Clone for MlxNativeStorage {
    fn clone(&self) -> Self {
        Self {
            initial_publication: self.initial_publication.clone(),
            ..Self::new(&self.runtime, &self.selection)
        }
    }
}
impl fmt::Debug for MlxNativeStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxNativeStorage")
            .field("runtime_ready", &self.runtime.is_ok())
            .finish_non_exhaustive()
    }
}
impl MlxNativeStorage {
    // Only the already-selected native context supplies this ordinary cold
    // witness. Projection clones no payload and performs no runtime preparation.
    pub(crate) fn new(
        runtime: &Result<Rc<PreparedInputRuntime>, eredu_core::SharedBackendFailure>,
        selection: &NativeStorageSelection,
    ) -> Self {
        Self {
            runtime: match runtime {
                Ok(runtime) => Ok(Rc::clone(runtime)),
                Err(cause) => Err(cause.retained()),
            },
            selection: selection.clone(),
            initial_publication: None,
        }
    }
    pub(crate) fn with_initial_publication(mut self, publication: super::RetainedStoragePublication) -> Self {
        self.initial_publication = Some(publication);
        self
    }
}

/// Actual retained publication owners, borrowed without constructing a tensor.
#[derive(Clone, Copy)]
pub(crate) enum NativeStorageRoot<'a> {
    Array(&'a Array),
    CanonicalArray(&'a super::super::manager::CanonicalArrayOwner),
    Host(
        &'a safemlx::ImmutableHostTransferBuffer,
        Option<super::RetainedAllocationReceipt<'a>>,
    ),
}

#[derive(Clone, Copy)]
pub(crate) struct CanonicalObservation<'a> {
    cell: &'a super::super::manager::CanonicalArrayOwner,
    proof: Option<super::PublishedAllocation>,
}
impl<'a> CanonicalObservation<'a> {
    fn capture(cell: &'a super::super::manager::CanonicalArrayOwner) -> Self {
        Self {
            cell,
            proof: cell.proof(),
        }
    }
    fn receipt(self) -> Option<super::RetainedAllocationReceipt<'a>> {
        self.proof.map(|proof| proof.borrow(self.cell.custody()))
    }
}

pub(crate) enum Observation<'a> {
    Origin(OriginalBufferWitness<'a>, Option<CanonicalObservation<'a>>),
    Existing(
        OriginalBufferAliasWitness<'a>,
        Option<CanonicalObservation<'a>>,
    ),
    Ordinary(OrdinaryBufferWitness<'a>),
    Immutable(safemlx::ImmutableSourceWitness<'a>),
    Host(safemlx::HostTransferArrayAliasWitness<'a>),
    OrdinaryHostArray(safemlx::HostTransferArrayAliasWitness<'a>),
    HostBuffer(
        safemlx::ImmutableHostTransferWitness<'a>,
        Option<super::RetainedAllocationReceipt<'a>>,
    ),
    OrdinaryHost(safemlx::ImmutableHostTransferWitness<'a>),
    Empty,
}
impl<'a> Observation<'a> {
    fn cell_receipt(&self) -> Option<super::RetainedAllocationReceipt<'a>> {
        match self {
            Self::Origin(_, Some(cell)) | Self::Existing(_, Some(cell)) => (*cell).receipt(),
            _ => None,
        }
    }
    fn allocation(&self) -> Option<safemlx::AllocationInfo> {
        Some(match self {
            Self::Origin(witness, _) => witness.allocation(),
            Self::Existing(witness, _) => witness.allocation(),
            Self::Ordinary(witness) => witness.allocation(),
            Self::Immutable(witness) => witness.allocation(),
            Self::Host(witness) | Self::OrdinaryHostArray(witness) => witness.allocation(),
            Self::HostBuffer(witness, _) | Self::OrdinaryHost(witness) => witness.allocation(),
            Self::Empty => return None,
        })
    }
}

pub(crate) enum NativeStorageCause {
    Cold(eredu_core::SharedBackendFailure),
    Fixed(OriginalBufferCause),
    Control(safemlx::OriginalNativeControlError),
    Observation(safemlx::ScopedSubmissionProgress),
    Carrier(safemlx::PrefillFailureCause),
    // Failed budget creation publishes no native owner. Its unattached Rust
    // preparation node is cancelled before moving this same original custody.
    Budget {
        cause: OriginalBufferCause,
        _custody: OriginalNativeBudgetCustody,
    },
    // Keep the actual typed source and intact neutral payload. Native node
    // allocation failure precedes any attachment or canonical publication.
    Preparation {
        cause: PreparedAllocationOwnerCause,
        _owner: Registration,
    },
}
impl fmt::Debug for NativeStorageCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cold(cause) => f.debug_tuple("Cold").field(cause).finish(),
            Self::Control(cause) => f.debug_tuple("Control").field(cause).finish(),
            Self::Observation(cause) => f.debug_tuple("Observation").field(cause).finish(),
            Self::Carrier(cause) => f.debug_tuple("Carrier").field(cause).finish(),
            Self::Fixed(cause) | Self::Budget { cause, .. } => {
                f.debug_tuple("OriginalBuffer").field(cause).finish()
            }
            Self::Preparation { cause, .. } => f.debug_tuple("Preparation").field(cause).finish(),
        }
    }
}
impl fmt::Display for NativeStorageCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cold(cause) => fmt::Display::fmt(cause.source_error(), f),
            Self::Control(cause) => fmt::Display::fmt(cause, f),
            Self::Observation(cause) => {
                write!(f, "original preparation observation unavailable: {cause:?}")
            }
            Self::Carrier(cause) => fmt::Display::fmt(cause, f),
            Self::Fixed(cause) | Self::Budget { cause, .. } => fmt::Display::fmt(cause, f),
            Self::Preparation { cause, .. } => fmt::Display::fmt(cause, f),
        }
    }
}
impl std::error::Error for NativeStorageCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if matches!(self, Self::Observation(_)) {
            return None;
        }
        Some(match self {
            Self::Observation(_) => unreachable!(),
            Self::Cold(cause) => cause.source_error(),
            Self::Control(cause) => cause,
            Self::Carrier(cause) => cause,
            Self::Fixed(cause) | Self::Budget { cause, .. } => cause,
            Self::Preparation { cause, .. } => cause,
        })
    }
}

impl OriginalNativeStorageMechanism for MlxNativeStorage {
    type Key = StorageIdentity;
    type Budget = OriginalBufferBudget;
    type Root<'a> = NativeStorageRoot<'a>;
    type Attachment = PreparedAllocationOwner<Registration>;
    type Error = NativeStorageCause;
    type Observation<'a> = Observation<'a>;

    fn selection(&self) -> &NativeStorageSelection {
        &self.selection
    }

    fn key_clone_storage_bytes(&self) -> Option<u64> {
        // The source key validator excludes arbitrary inline byte allocations.
        // Native keys are scalars; authentic checkpoint keys retain their own
        // already paid source custody. The largest independent key is Arc<()>.
        eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes()
    }

    fn validate_source_key(
        &self,
        key: &Self::Key,
    ) -> Result<(), eredu_runtime::working_memory::WorkingMemoryError> {
        match key {
            StorageIdentity::Native(_)
            | StorageIdentity::GroupBuffer(_)
            | StorageIdentity::Source(_)
            | StorageIdentity::HostMetadata(_) => Ok(()),
            StorageIdentity::Bytes(_) | StorageIdentity::CapturePlan(_) => {
                Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
            }
        }
    }

    fn checkpoint_source_identity<'a>(
        &self,
        key: &'a Self::Key,
    ) -> Option<&'a eredu_checkpoint::store::SourceStorageIdentity> {
        match key {
            StorageIdentity::Source(identity) => Some(identity),
            _ => None,
        }
    }

    fn create_budget(
        &self,
        custody: OriginalNativeBudgetCustody,
    ) -> Result<Self::Budget, Self::Error> {
        let runtime = match &self.runtime {
            Ok(runtime) => runtime,
            Err(cause) => return Err(NativeStorageCause::Cold(cause.retained())),
        };
        let capacity = usize::try_from(custody.capacity_bytes())
            .map_err(|_| NativeStorageCause::Fixed(OriginalBufferCause::InvalidLayout))?;
        let prepared = PreparedOriginalBufferBudget::try_new(runtime, capacity, custody).map_err(
            |failure| {
                let (cause, custody) = failure.into_parts();
                NativeStorageCause::Budget {
                    cause,
                    _custody: custody,
                }
            },
        )?;
        prepared.try_allocate().map_err(|failure| {
            let (cause, prepared) = failure.into_parts();
            // C refusal guarantees null output and no retained native prefix.
            // Cancel only the unattached node, while its custody remains live.
            NativeStorageCause::Budget {
                cause,
                _custody: prepared.into_owner(),
            }
        })
    }

    fn observe<'a, 'root: 'a>(
        &'a self,
        budget: &'a Self::Budget,
        root: NativeStorageRoot<'root>,
    ) -> Result<Observation<'a>, Self::Error> {
        let (root, cell) = match root {
            NativeStorageRoot::Array(root) => (root, None),
            NativeStorageRoot::CanonicalArray(cell) => (cell.array(), Some(cell)),
            NativeStorageRoot::Host(host, receipt) => {
                let witness = host.inspect_original_source().map_err(NativeStorageCause::Fixed)?;
                // Prepared construction does not identify the payer: load-time
                // managers, saved copies and source banks share this worker.
                return if witness.is_prepared_source() {
                    Ok(Observation::HostBuffer(witness, receipt))
                } else if receipt.is_none() {
                    Ok(Observation::OrdinaryHost(witness))
                } else {
                    Err(NativeStorageCause::Fixed(OriginalBufferCause::UncertifiedBacking))
                };
            }
        };
        let observed = match budget.inspect_array(root) {
            Ok(Some(witness)) => Ok(Observation::Origin(witness, cell.map(CanonicalObservation::capture))),
            Err(OriginalBufferCause::ForeignDomain) => root
                .inspect_original_buffer_alias()
                .map_err(NativeStorageCause::Fixed)?
                .map(|witness| Observation::Existing(witness, cell.map(CanonicalObservation::capture)))
                .ok_or(NativeStorageCause::Fixed(
                    OriginalBufferCause::UncertifiedBacking,
                )),
            Err(cause) => Err(NativeStorageCause::Fixed(cause)),
            Ok(None) => match root
                .inspect_immutable_source()
                .map_err(NativeStorageCause::Fixed)?
            {
                safemlx::ImmutableSourceInspection::Allocation(witness) => {
                    Ok(Observation::Immutable(witness))
                }
                safemlx::ImmutableSourceInspection::Empty => Ok(Observation::Empty),
                safemlx::ImmutableSourceInspection::Unknown => match root
                    .inspect_host_transfer_alias()
                    .map_err(NativeStorageCause::Fixed)?
                {
                    Some(witness) if witness.is_prepared_source() => Ok(Observation::Host(witness)),
                    Some(witness) => Ok(Observation::OrdinaryHostArray(witness)),
                    None => ordinary_observation(root),
                },
            },
        }?;
        let proof = match &observed {
            Observation::Origin(_, Some(cell)) | Observation::Existing(_, Some(cell)) => cell.proof,
            _ => cell.and_then(|cell| cell.proof()),
        };
        if let Some(proof) = proof {
            if !matches!(&observed, Observation::Origin(..) | Observation::Existing(..))
                || observed.allocation() != Some(proof.allocation()) {
                return Err(NativeStorageCause::Fixed(OriginalBufferCause::UncertifiedBacking));
            }
        }
        Ok(observed)
    }

    fn describe(observation: &Observation<'_>) -> NativeStorageObservation<StorageIdentity> {
        let (facts, kind) = match observation {
            Observation::Origin(witness, _) => (witness.allocation(), 0),
            Observation::Existing(witness, _) => (witness.allocation(), 1),
            Observation::Ordinary(witness) => (witness.allocation(), 2),
            Observation::Immutable(witness) => (witness.allocation(), 3),
            Observation::Host(witness) => (witness.allocation(), 4),
            Observation::OrdinaryHostArray(witness) => (witness.allocation(), 2),
            Observation::HostBuffer(witness, _) => (witness.allocation(), 4),
            Observation::OrdinaryHost(witness) => (witness.allocation(), 2),
            Observation::Empty => return NativeStorageObservation::Empty,
        };
        let key = StorageIdentity::Native(facts.identity());
        // All supported Rust targets use usize <= u64; preserve the native
        // physical charge rather than logical shape bytes or usable CPU bytes.
        let bytes = facts.bytes() as u64;
        match kind {
            0 => NativeStorageObservation::Originating(key, bytes),
            1 => NativeStorageObservation::Existing(key, bytes),
            2 => NativeStorageObservation::Ordinary(key, bytes),
            3 => NativeStorageObservation::ExistingImmutable(key, bytes),
            _ => NativeStorageObservation::ExistingPhysical(key, bytes),
        }
    }

    fn has_retained_attachment(&self, previous: &Observation<'_>, observation: &Observation<'_>, pool: &eredu_runtime::working_memory::WorkingMemoryPool) -> bool {
        let Some(allocation) = observation.allocation() else { return false };
        if std::mem::discriminant(previous) != std::mem::discriminant(observation)
            || previous.allocation() != Some(allocation)
        { return false; }
        match (previous.cell_receipt(), observation.cell_receipt()) {
            (Some(prior), Some(current)) => return prior.matches(allocation, pool) && current.matches(allocation, pool),
            (Some(_), None) | (None, Some(_)) => return false,
            _ => {}
        }
        match (previous, observation) {
            (Observation::HostBuffer(_, Some(prior)), Observation::HostBuffer(_, Some(current))) =>
                return prior.matches(allocation, pool) && current.matches(allocation, pool),
            (Observation::HostBuffer(_, Some(_)), _) | (_, Observation::HostBuffer(_, Some(_))) =>
                return false,
            _ => {}
        }
        self.initial_publication.as_ref().is_some_and(|publication|
            publication.has_native_attachment(pool.shared_storage_domain(), allocation))
    }

    fn prepare_attachment(
        &self,
        registration: Registration,
    ) -> Result<Self::Attachment, Self::Error> {
        PreparedAllocationOwner::try_new(registration).map_err(|failure| {
            let (cause, owner) = failure.into_parts();
            NativeStorageCause::Preparation {
                cause,
                _owner: owner,
            }
        })
    }

    fn registration(attachment: &Self::Attachment) -> &Registration {
        attachment.owner()
    }

    fn attach(
        observation: Observation<'_>,
        attachment: Self::Attachment,
    ) -> Result<(), (Self::Error, Self::Attachment)> {
        let result = match observation {
            Observation::Origin(witness, cell) => {
                let allocation = witness.allocation();
                if cell.is_some_and(|cell| !attachment.owner().has_metadata_origin(cell.cell.custody())
                    || cell.proof.is_some_and(|proof| proof.allocation() != allocation)) {
                    return Err((NativeStorageCause::Fixed(OriginalBufferCause::UncertifiedBacking), attachment));
                }
                let result = witness.try_attach(attachment);
                if result.is_ok() {
                    if let Some(cell) = cell {
                        assert!(cell.cell.record_attachment(super::PublishedAllocation::attached(allocation)),
                            "immutable canonical allocation changed after checked attachment");
                    }
                }
                result
            }
            Observation::Existing(witness, _) => witness.try_attach(attachment),
            Observation::Ordinary(witness) => witness.try_attach(attachment),
            Observation::Immutable(witness) => witness.try_attach(attachment),
            Observation::Host(witness) | Observation::OrdinaryHostArray(witness) => witness.try_attach(attachment),
            Observation::HostBuffer(witness, _) | Observation::OrdinaryHost(witness) => witness.try_attach(attachment),
            Observation::Empty => {
                return Err((
                    NativeStorageCause::Fixed(OriginalBufferCause::UncertifiedBacking),
                    attachment,
                ));
            }
        };
        result.map_err(|failure| {
            let (cause, preparation) = failure.into_parts();
            (NativeStorageCause::Fixed(cause), preparation)
        })
    }
}

/// Requested preparation representations only. Managed Scope/arena owners,
/// source maps and cumulative retained generations remain separate inputs.
/// Allocator-private bookkeeping and unrelated process memory are excluded.
pub(crate) fn preparation_control_bytes() -> Option<usize> {
    use eredu_runtime::working_memory::OriginalTextControlGuard;
    let carrier = safemlx::PreparedPrefillFailure::<OriginalTextControlGuard>::layout()
        .ok()?
        .total_bytes()?;
    let native = safemlx::OriginalNativeControlLayout::inspect().ok()?;
    [
        carrier,
        native.fixed_control_bytes,
        eredu_core::SharedBackendFailure::control_bytes::<PublicFailure>()?,
        std::mem::size_of::<PublicFailure>(),
        std::mem::size_of::<NativeStorageCause>(),
        std::mem::size_of::<BankOwner>(),
        std::mem::size_of::<OriginalTextControlGuard>(),
        std::mem::size_of::<safemlx::RetainedPrefillFailure>(),
        std::mem::size_of::<Result<(), crate::backend::error::Error>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

fn ordinary_observation(root: &Array) -> Result<Observation<'_>, NativeStorageCause> {
    match root
        .inspect_ordinary_buffer()
        .map_err(NativeStorageCause::Fixed)?
    {
        OrdinaryBufferInspection::Allocation(witness) => Ok(Observation::Ordinary(witness)),
        OrdinaryBufferInspection::Empty => Ok(Observation::Empty),
        OrdinaryBufferInspection::Unknown => Err(NativeStorageCause::Fixed(
            OriginalBufferCause::UncertifiedBacking,
        )),
    }
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod tests;
