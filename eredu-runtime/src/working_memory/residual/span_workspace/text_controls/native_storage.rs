//! Selected request-wide native capacity and authenticated canonical publication.
use super::*;
use crate::working_memory::{
    funding::native_partition::{NativePartition, NativePublicationScopeIdentity},
    storage::native_publication::PreparedNativePublication,
    WorkingMemoryStorage,
};
use std::{any::TypeId, marker::PhantomData, sync::OnceLock};

pub(super) mod plan;
pub use plan::{NativeStorageCarryoverReport, PreparedNativeStoragePlan};
mod prefill_envelope;
pub use prefill_envelope::{NativeEquationStorage, NativePrefillEnvelope, NativePrefillEnvelopeBuilder};

/// Payload-free identity of the selected native mechanism. Equal capacities do
/// not identify a selection. Constructing this identity creates no authority.
#[derive(Clone, Debug, Default)]
pub struct NativeStorageSelection(Arc<()>);
impl NativeStorageSelection {
    /// Whether both diagnostics refer to the same selected mechanism instance.
    pub fn same_selection(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Closed classification produced by a backend's actual borrowed observation.
/// Implementations must authenticate the immutable native birth or positively
/// prove ordinary default-allocator backing. Unknown backing is an error, never
/// an ordinary input. Source-owned host/input storage uses its own source path.
pub enum NativeStorageObservation<K> {
    /// Birth authenticated against this request's actual native budget.
    Originating(K, u64),
    /// Authenticated foreign native birth; only an existing canonical row may
    /// supply its accounting origin. An absent row must refuse.
    Existing(K, u64),
    /// Actual immutable prepared source; only a prepaid-host canonical row can
    /// supply its origin. Never uses this request's mutable native partition.
    ExistingImmutable(K, u64),
    /// Positively authenticated live physical owner whose constructor does not
    /// identify its accounting origin. Requires an existing canonical row and
    /// preserves that row's full ordinary, copy-funded or prepaid coverage.
    /// It cannot create a row, supply an origin or claim a new native birth.
    ExistingPhysical(K, u64),
    /// Positively authenticated ordinary backing, retaining ordinary charging.
    Ordinary(K, u64),
    /// Positively authenticated absence of physical backing, not logical zero.
    Empty,
}

/// Backend ownership contract for the selected native budget and existing
/// canonical registry. Callbacks run outside the pool lock. This is a truthful
/// mechanism contract, not an independent byte-grant interface.
pub trait OriginalNativeStorageMechanism: Clone + 'static {
    /// Exact opaque physical storage identity in the existing registry.
    type Key: Clone + Ord + Send + Sync + 'static;
    /// Native budget alias. Its callback may retain only the supplied accounting
    /// custody; it must not retain the mechanism, this bank or an Array.
    type Budget: Clone + 'static;
    /// Borrowed actual native root; ownership stays with the publication caller.
    /// A mechanism may distinguish tensor and immutable source owners without
    /// constructing a tensor or cloning either owner. Copy duplicates only the loan.
    type Root<'a>: Copy where Self: 'a;
    /// Prepared native attachment, retaining its unchanged registration on error.
    type Attachment: 'static;
    /// Actual fixed/native failure. Failed native construction must retain every
    /// partially created owner and the supplied custody until safe retirement.
    type Error;
    /// Borrowed native truth, consumed by one compare-and-attach operation.
    type Observation<'a>
    where
        Self: 'a;

    /// Return the same cold identity used by the sealed selected plan.
    fn selection(&self) -> &NativeStorageSelection;
    /// Maximum additional managed storage retained by each cloned key from this
    /// selection, including a partially failed clone. This bound is immutable
    /// for the same selection and applies to every described/source key used by
    /// it. None keeps the qualified
    /// control plan unknown. Zero is valid only for a concrete payload-free key
    /// route; generic keys or unrelated source variants must not assume it.
    fn key_clone_storage_bytes(&self) -> Option<u64> {
        None
    }

    /// Check that a source key belongs to the selection's priced key domain.
    /// This runs before cloning the key and outside the pool lock. A refusal
    /// cannot retain a previously unpriced key allocation in the attempt.
    fn validate_source_key(&self, _key: &Self::Key) -> Result<(), WorkingMemoryError> {
        Ok(())
    }

    /// Borrow the actual checkpoint identity held by a source key, if present.
    /// This must be the identity represented by `key`, never a different source
    /// with equal capacity. It grants no native birth or funding: the publisher
    /// authenticates its original source account and requires an existing row.
    fn checkpoint_source_identity<'a>(
        &self,
        _key: &'a Self::Key,
    ) -> Option<&'a eredu_checkpoint::store::SourceStorageIdentity> {
        None
    }

    /// Construct one native counter with the exact accepted capacity. A native
    /// callback consumes this accounting-only custody without a budget backedge.
    fn create_budget(
        &self,
        custody: OriginalNativeBudgetCustody,
    ) -> Result<Self::Budget, Self::Error>;
    /// Observe this exact root against this budget without evaluation, retries,
    /// completion claims or fallback from unknown allocation provenance.
    fn observe<'a, 'root: 'a>(
        &'a self,
        budget: &'a Self::Budget,
        root: Self::Root<'root>,
    ) -> Result<Self::Observation<'a>, Self::Error>;
    /// Describe only facts authenticated by the supplied observation. Cloning a
    /// key must preserve its identity, capacity and actual payload custody.
    fn describe(observation: &Self::Observation<'_>) -> NativeStorageObservation<Self::Key>;
    /// Positive proof that this exact completed backing already owns an
    /// independent attachment in this pool. The mechanism must retain the
    /// complete successful attachment receipt, validate domain, non-reused
    /// allocation identity and capacity, and preserve custody for every alias.
    /// Registry presence or equal byte counts alone are insufficient. False
    /// keeps the ordinary checked publication worker; true produces no new
    /// registration or native sidecar for this root. The publisher repeats the
    /// actual native observation and this proof at its final attachment boundary;
    /// an earlier borrowed observation is not a durable descriptor witness.
    /// Both observations must name the same kind, generation and capacity.
    /// Initial eligibility passes the same observation in both positions.
    fn has_retained_attachment(
        &self,
        _previous: &Self::Observation<'_>,
        _observation: &Self::Observation<'_>,
        _pool: &WorkingMemoryPool,
    ) -> bool { false }
    /// Prepare the existing native sidecar using this exact neutral payload.
    /// Failure retains the payload as part of the actual typed failed owner.
    fn prepare_attachment(
        &self,
        registration: NativeStorageRegistration<Self::Key>,
    ) -> Result<Self::Attachment, Self::Error>;
    /// Borrow the unchanged neutral payload within the prepared sidecar.
    fn registration(attachment: &Self::Attachment) -> &NativeStorageRegistration<Self::Key>;
    /// Recheck kind/generation/capacity, and originating budget where applicable,
    /// in the same native operation before insertion. On refusal return the
    /// actual cause and unchanged preparation. No unchecked attach is allowed.
    fn attach(
        observation: Self::Observation<'_>,
        attachment: Self::Attachment,
    ) -> Result<(), (Self::Error, Self::Attachment)>;
}

/// Accounting-only native callback payload issued from one accepted selection.
/// It exposes no pool, source, Scope, Array, registry or grant constructor.
#[derive(Debug)]
pub struct OriginalNativeBudgetCustody {
    partition: NativePartition,
}
impl OriginalNativeBudgetCustody {
    /// Exact immutable request-wide physical capacity, shared by all roles.
    pub fn capacity_bytes(&self) -> u64 {
        self.partition.capacity()
    }
}

/// Sidecar payload whose registry entry can outlive the originating request.
/// The native preparation header retires before this payload; the canonical
/// registration retires before its already charged host metadata custody.
pub struct NativeStorageRegistration<K: Ord + Send + 'static> {
    pub(in crate::working_memory) registration: OnceLock<WorkingMemoryStorage<K>>,
    pub(in crate::working_memory) _raw: crate::working_memory::OriginalHostMetadataCustody,
}

impl<K: Ord + Send + 'static> NativeStorageRegistration<K> {
    /// Compare the actual sidecar metadata payer with an already retained
    /// operation account. This borrowed equality grants no publication or
    /// allocation authority; a receipt still requires successful native attach.
    /// Equality remains true after account closure or quarantine. Receipt reuse
    /// must separately validate the retained origin's pool and current health.
    pub fn has_metadata_origin(
        &self,
        origin: &crate::working_memory::OriginalOperationMetadataCustody,
    ) -> bool {
        origin.same_host_account(&self._raw)
    }

    /// Fixed transports of the borrowed account comparison, with no new owner.
    pub fn metadata_origin_control_bytes() -> Option<usize> {
        crate::working_memory::OriginalOperationMetadataCustody::host_account_control_bytes()?
            .checked_add(std::mem::size_of::<(&Self, &crate::working_memory::OriginalOperationMetadataCustody)>())?
            .checked_add(std::mem::size_of::<bool>())
    }
}

/// Fixed dispatch errors remain typed; native causes are never formatted away.
#[derive(Debug)]
pub enum NativeStorageError<E> {
    /// Reservation, identity, health or finite-population refusal.
    Memory(WorkingMemoryError),
    /// Actual backend constructor, inspection or attachment failure.
    Native(E),
}
impl<E: std::fmt::Display> std::fmt::Display for NativeStorageError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory(cause) => std::fmt::Display::fmt(cause, f),
            Self::Native(cause) => std::fmt::Display::fmt(cause, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativeStorageError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Memory(cause) => cause,
            Self::Native(cause) => cause,
        })
    }
}

/// One request-wide issuer and finite publication bank. Budget retirement
/// precedes partition custody. Canonical entries never retain this owner.
pub struct OriginalNativeStorageBank<M: OriginalNativeStorageMechanism> {
    budget: Option<M::Budget>,
    mechanism: Option<M>,
    layout: plan::NativeStorageLayoutOwner,
    attempts: usize,
    installation_started: bool,
    reservation: WorkingMemoryReservation,
    // Full source validators stay on the live request owner, not in the native
    // callback or canonical row. The partition carries only raw host metadata.
    controls: OriginalTextControlGuard,
    partition: NativePartition,
}
impl<M: OriginalNativeStorageMechanism> std::fmt::Debug for OriginalNativeStorageBank<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalNativeStorageBank")
            .field("capacity", &self.partition.capacity())
            .field("installed", &self.budget.is_some())
            .field("remaining_publications", &self.attempts)
            .finish_non_exhaustive()
    }
}

impl<M: OriginalNativeStorageMechanism> OriginalNativeStorageBank<M> {
    /// Immutable maximum collector rows from this bank's accepted population.
    /// Reading the bound neither claims an attempt nor grants allocation.
    pub fn collector_rows(&self) -> usize {
        self.layout
            .population
            .expect("accepted native population")
            .1
    }

    /// Qualified managed allocation for retaining this bank in one `Rc<RefCell<_>>`.
    /// This includes that allocation's concrete bank payload and shared header;
    /// nested backend budget/mechanism owners remain provider contributions.
    /// No owner is constructed and no reservation is issued by this layout query.
    pub fn shared_borrowed_owner_bytes() -> Option<u64> {
        crate::working_memory::qualified_storage::shared_bytes::<std::cell::RefCell<Self>>().ok()
    }

    /// Attempt construction exactly once. On failure the bank retains its
    /// original partition; any native failed owner is returned in the real cause.
    pub fn install(&mut self, mechanism: M) -> Result<(), NativeStorageError<M::Error>> {
        if self.installation_started {
            return Err(NativeStorageError::Memory(
                WorkingMemoryError::AlreadyStarted,
            ));
        }
        self.installation_started = true;
        self.controls
            .validate_reservation(&self.reservation)
            .map_err(NativeStorageError::Memory)?;
        if !self.layout.selection.same_selection(mechanism.selection()) {
            return Err(NativeStorageError::Memory(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.mechanism = Some(mechanism);
        let budget = self
            .mechanism
            .as_ref()
            .expect("installed mechanism")
            .create_budget(OriginalNativeBudgetCustody {
                partition: self.partition.clone(),
            })
            .map_err(NativeStorageError::Native)?;
        self.budget = Some(budget);
        Ok(())
    }

    /// Borrow the installed budget only for a matching live original role.
    /// This validates existing funding but does not create or certify a Scope.
    pub fn budget_for_scope(
        &self,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<&M::Budget, WorkingMemoryError> {
        self.controls.custody.validate_native_publication(scope)?;
        self.budget
            .as_ref()
            .ok_or(WorkingMemoryError::PreparationAlreadyStarted)
    }

    /// Borrow the same installed counter for a configured original role whose
    /// exact custody was extracted from this accepted span. The backend must
    /// also validate the concrete native role at its checked binding call.
    pub fn budget_for_controls(
        &self,
        controls: &OriginalTextControlGuard,
    ) -> Result<&M::Budget, WorkingMemoryError> {
        if !self.controls.custody.same(&controls.custody) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls.custody.validate_native_buffer_role()?;
        self.budget
            .as_ref()
            .ok_or(WorkingMemoryError::PreparationAlreadyStarted)
    }

    /// Remaining finite publication calls; exhaustion creates no owning result.
    pub fn remaining_publications(&self) -> usize {
        self.attempts
    }

    /// Claim one finite terminal attempt before cloning custody or creating any
    /// result storage. Caller-owned roots remain outside this accounting object;
    /// the native recovery owner must retain them on every publication refusal.
    pub fn claim_publication(
        &mut self,
        scope: &mut WorkingMemoryFundingScope,
    ) -> Result<OriginalNativePublication<M>, WorkingMemoryError> {
        if self.attempts == 0 {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        self.attempts -= 1;
        self.controls.custody.validate_native_publication(scope)?;
        let scope_identity = scope
            .native_publication_identity
            .get_or_insert_with(|| {
                NativePublicationScopeIdentity::new(self.controls.custody.raw().clone())
            })
            .clone();
        let mechanism = self
            .mechanism
            .as_ref()
            .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?;
        let budget = self
            .budget
            .as_ref()
            .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?;
        let slots = self
            .layout
            .population
            .ok_or(WorkingMemoryError::UnknownBound)?
            .1;
        let exact_storage = self.layout.exact_storage;
        let registry = if exact_storage {
            PreparedNativePublication::prepare_slots_exact(self.partition.clone(), slots)?
        } else {
            PreparedNativePublication::prepare_slots(self.partition.clone(), slots)
        };
        // These locals retire before this borrowed bank's custody on refusal;
        // attempts were already debited before the first owning construction.
        let attachments = crate::working_memory::qualified_storage::vector(slots, exact_storage)?;
        let root_inputs = crate::working_memory::qualified_storage::vector(slots, exact_storage)?;
        let source_inputs = crate::working_memory::qualified_storage::vector(slots, exact_storage)?;
        let source_outputs =
            crate::working_memory::qualified_storage::vector(slots, exact_storage)?;
        let attached_sources =
            crate::working_memory::qualified_storage::vector(slots, exact_storage)?;
        Ok(OriginalNativePublication {
            scope_identity,
            registry,
            attachments,
            root_inputs,
            source_inputs,
            source_outputs: Some(source_outputs),
            attached_sources,
            orphaned_registration: None,
            budget: budget.clone(),
            mechanism: mechanism.clone(),
            maximum_rows: slots,
            exact_storage,
            terminal: false,
            published: false,
            failure_site: "native publication preparation",
            controls: self.controls.clone(),
            partition: self.partition.clone(),
        })
    }
}

/// One terminal mixed publication. It retains every failed/unused preparation,
/// successful registry result and originating partition until actual recovery
/// retires it. It does not own or certify the caller's native root population.
pub struct OriginalNativePublication<M: OriginalNativeStorageMechanism> {
    scope_identity: NativePublicationScopeIdentity,
    registry: PreparedNativePublication<M::Key>,
    attachments: Vec<Option<M::Attachment>>,
    root_inputs: Vec<RootPublicationInput>,
    source_inputs: Vec<usize>,
    // Reserved before publication; after attachment this exact backing moves
    // to the caller's successful source owner. Failed attempts keep it intact.
    source_outputs: Option<Vec<WorkingMemoryStorage<M::Key>>>,
    attached_sources: Vec<bool>,
    orphaned_registration: Option<WorkingMemoryStorage<M::Key>>,
    budget: M::Budget,
    mechanism: M,
    maximum_rows: usize,
    exact_storage: bool,
    terminal: bool,
    published: bool,
    failure_site: &'static str,
    controls: OriginalTextControlGuard,
    partition: NativePartition,
}

enum RootPublicationInput {
    Empty,
    Retained,
    Registry(usize),
}
impl<M: OriginalNativeStorageMechanism> OriginalNativePublication<M> {
    /// Static validation step retained after a failed publication. It carries no
    /// source identity, authority or storage grant; its bytes are part of Self.
    pub fn failure_site(&self) -> &'static str {
        if self.failure_site == "native publication registry" {
            self.registry.failure_site()
        } else {
            self.failure_site
        }
    }

    /// Validate and publish one reached inventory. `source_inputs` must come
    /// from the caller's existing complete, validated source-owner inventory;
    /// they are ordinary charged entries and cannot claim native coverage.
    ///
    /// The caller retains ALL `roots` until success or its actual native recovery
    /// has established retirement. A returned error is not completion. The
    /// complete count is checked before observation, attachment or registry work.
    pub fn publish<'a>(
        &mut self,
        scope: &WorkingMemoryFundingScope,
        roots: impl IntoIterator<Item = M::Root<'a>>,
        source_inputs: &[(M::Key, u64)],
    ) -> Result<(), NativeStorageError<M::Error>> {
        self.failure_site = "native publication terminal attempt";
        if self.terminal {
            return Err(NativeStorageError::Memory(
                WorkingMemoryError::PreparationAlreadyStarted,
            ));
        }
        self.terminal = true;
        self.failure_site = "native publication scope identity";
        if !scope
            .native_publication_identity
            .as_ref()
            .is_some_and(|identity| identity.same(&self.scope_identity))
        {
            return Err(NativeStorageError::Memory(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.failure_site = "native publication inventory population";
        let available = self.maximum_rows.checked_sub(source_inputs.len()).ok_or(
            NativeStorageError::Memory(WorkingMemoryError::IdentityMismatch),
        )?;
        // Count actual references before any provider call. No ExactSizeIterator
        // or size_hint can substitute for the reached inventory population.
        let mut actual_roots =
            crate::working_memory::qualified_storage::vector(available, self.exact_storage)
                .map_err(NativeStorageError::Memory)?;
        for root in roots {
            if actual_roots.len() == available {
                return Err(NativeStorageError::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            actual_roots.push(root);
        }
        let roots = actual_roots;
        let count = roots
            .len()
            .checked_add(source_inputs.len())
            .ok_or(NativeStorageError::Memory(WorkingMemoryError::Overflow))?;
        if count > self.maximum_rows {
            return Err(NativeStorageError::Memory(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        // Retain these lexical borrows until attachment finishes. New sidecars
        // use the same checked observation; omitted sidecars need a fresh
        // receipt check at their final attachment boundary below.
        let mut observations =
            crate::working_memory::qualified_storage::vector(roots.len(), self.exact_storage)
                .map_err(NativeStorageError::Memory)?;
        self.failure_site = "native publication observations and attachments";
        for root in &roots {
            let observed = self
                .mechanism
                .observe(&self.budget, *root)
                .map_err(NativeStorageError::Native)?;
            let description = M::describe(&observed);
            if matches!(description, NativeStorageObservation::Empty) {
                self.root_inputs.push(RootPublicationInput::Empty);
                self.attachments.push(None);
            } else if self.mechanism.has_retained_attachment(&observed, &observed, scope.pool()) {
                self.root_inputs.push(RootPublicationInput::Retained);
                self.attachments.push(None);
            } else {
                let input = self
                    .registry
                    .push_observation(description)
                    .map_err(NativeStorageError::Memory)?;
                self.root_inputs.push(RootPublicationInput::Registry(input));
                let prepared = self
                    .mechanism
                    .prepare_attachment(NativeStorageRegistration {
                        registration: OnceLock::new(),
                        _raw: self.controls.custody.raw().clone().into(),
                    })
                    .map_err(NativeStorageError::Native)?;
                self.attachments.push(Some(prepared));
                if M::registration(
                    self.attachments
                        .last()
                        .and_then(Option::as_ref)
                        .expect("prepared sidecar"),
                )
                .registration
                .get()
                .is_some()
                {
                    return Err(NativeStorageError::Memory(
                        WorkingMemoryError::IdentityMismatch,
                    ));
                }
            }
            observations.push(Some(observed));
        }
        self.failure_site = "native publication source inventory";
        for (key, bytes) in source_inputs {
            self.mechanism
                .validate_source_key(key)
                .map_err(NativeStorageError::Memory)?;
            self.source_inputs.push(
                self.registry
                    .push_source(
                        key,
                        *bytes,
                        self.mechanism.checkpoint_source_identity(key),
                        scope.pool(),
                    )
                    .map_err(NativeStorageError::Memory)?,
            );
            self.attached_sources.push(false);
        }
        self.failure_site = "native publication registry";
        self.registry
            .publish_original(scope, &self.controls, &self.scope_identity)
            .map_err(NativeStorageError::Memory)?;
        self.failure_site = "native publication attachment commit";
        for (index, input) in self.root_inputs.iter().enumerate() {
            let RootPublicationInput::Registry(input) = input else {
                // Another alias may have changed the descriptor since initial
                // observation. Omission is valid only if the actual current
                // backing is still empty or already has its independent native
                // attachment. The native inspection performs its checked read
                // under the same runtime lock used by actual attachment.
                let current = self.mechanism
                    .observe(&self.budget, roots[index])
                    .map_err(NativeStorageError::Native)?;
                let unchanged = match input {
                    RootPublicationInput::Empty => matches!(M::describe(&current), NativeStorageObservation::Empty),
                    RootPublicationInput::Retained => self.mechanism.has_retained_attachment(
                        observations[index].as_ref().expect("original borrowed observation"),
                        &current, scope.pool()),
                    RootPublicationInput::Registry(_) => unreachable!("handled above"),
                };
                if !unchanged {
                    return Err(NativeStorageError::Memory(
                        WorkingMemoryError::IdentityMismatch,
                    ));
                }
                continue;
            };
            // One sidecar per distinct physical row. Duplicates keep their
            // unused preparations here; their registry ownership is not doubled.
            let Some(registration) = self.registry.take_input(*input) else {
                continue;
            };
            let preparation = self.attachments[index]
                .as_ref()
                .expect("prepared nonempty root");
            if let Err(registration) = M::registration(preparation).registration.set(registration) {
                self.orphaned_registration = Some(registration);
                return Err(NativeStorageError::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            let preparation = self.attachments[index]
                .take()
                .expect("prepared nonempty root");
            let observation = observations[index]
                .take()
                .expect("same borrowed observation");
            if let Err((cause, preparation)) = M::attach(observation, preparation) {
                self.attachments[index] = Some(preparation);
                return Err(NativeStorageError::Native(cause));
            }
        }
        self.published = true;
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::working_memory) fn control_capacities(&self) -> [usize; 6] {
        let registry = self.registry.control_capacities();
        [
            self.attachments.capacity(),
            self.root_inputs.capacity(),
            self.source_inputs.capacity(),
            registry[0],
            registry[1],
            registry[2],
        ]
    }

    /// Borrow one committed source registration without moving its failure
    /// custody. Duplicate input rows return None. The caller may clone this
    /// registration for attachment while the attempt retains the original.
    pub fn source(&self, index: usize) -> Option<&WorkingMemoryStorage<M::Key>> {
        self.published.then_some(())?;
        self.registry.input(*self.source_inputs.get(index)?)
    }

    /// Clone one source registration for its actual host attachment, keeping
    /// the original in this failure owner. The first call marks the row as
    /// attached for final transfer; later calls cannot take it again. A failed
    /// attachment must retain this attempt and must not finish source transfer.
    pub fn clone_source_for_attachment(
        &mut self,
        index: usize,
    ) -> Option<WorkingMemoryStorage<M::Key>> {
        if *self.attached_sources.get(index)? {
            return None;
        }
        let source = self.source(index)?.clone();
        self.attached_sources[index] = true;
        Some(source)
    }

    /// Move all remaining committed source registrations through the backing
    /// reserved by this accepted attempt. This allocates nothing, and succeeds
    /// only once after every native attachment has completed. It does not certify
    /// host attachment or device completion; the caller owns those obligations.
    pub fn take_remaining_sources(&mut self) -> Option<Vec<WorkingMemoryStorage<M::Key>>> {
        self.published.then_some(())?;
        let mut outputs = self.source_outputs.take()?;
        for (index, input) in self.source_inputs.iter().enumerate() {
            if self.attached_sources[index] {
                continue;
            }
            if let Some(registration) = self.registry.take_input(*input) {
                // source_inputs was bounded before registry publication, and
                // this exact vector was reserved for maximum_rows at claim.
                assert!(outputs.len() < self.maximum_rows);
                outputs.push(registration);
            }
        }
        Some(outputs)
    }

    /// Transfer one committed non-native source registration after all native
    /// attachments succeeded. Duplicates return None. No registration escapes
    /// from a refused or partially attached attempt.
    pub fn take_source(&mut self, index: usize) -> Option<WorkingMemoryStorage<M::Key>> {
        self.published.then_some(())?;
        if *self.attached_sources.get(index)? {
            return None;
        }
        self.registry.take_input(*self.source_inputs.get(index)?)
    }

    /// Whether this inventory completed registry publication and all required
    /// checked native attachments. This does not imply device completion.
    pub fn is_published(&self) -> bool {
        self.published
    }
}
