//! One-to-one physical/root bindings with finite metadata storage.
use super::*;
use crate::working_memory::{qualified_storage, storage::finite_pin};
use eredu_nn::workspace::WorkspaceBorrowedStorageError;
use std::{alloc::Layout, marker::PhantomData, mem::size_of};

/// Exact borrowed metadata roots paired with an existing-only storage pin.
/// Clones preserve the same metadata and accounting owner; no work is authorized.
#[derive(Debug, Clone)]
pub struct RegisteredWorkspaceStorage<K: Ord + Send + 'static> {
    pub(super) borrowed: WorkspaceBorrowedStorage,
    pub(super) pool: WorkingMemoryPool,
    pub(super) _registration: WorkingMemoryStorage<K>,
}

/// Closed combined source binding. The registered decoder mapping is not exposed
/// as a standalone source-copy certificate: its residual root selection also
/// includes the actual original prepared input's separately held B account.
#[derive(Debug)]
pub struct RegisteredPreparedWorkspaceStorage<K: Ord + Send + 'static> {
    registered: RegisteredWorkspaceStorage<K>,
    source: crate::input::OriginalPreparedWorkspaceSource,
}
impl<K: Clone + Ord + Send + Sync + 'static> RegisteredPreparedWorkspaceStorage<K> {
    /// Exact combined source-root selection, for the shared isolated-copy plan.
    pub fn borrowed_storage(&self) -> &WorkspaceBorrowedStorage {
        self.registered.borrowed_storage()
    }
    pub(in crate::working_memory) fn into_copy_parts(
        self,
    ) -> (
        RegisteredWorkspaceStorage<K>,
        crate::input::OriginalPreparedWorkspaceSource,
    ) {
        (self.registered, self.source)
    }
    /// Borrows the actual completed B account/root witness. The registered
    /// decoder mapping remains private and cannot become a standalone copy proof.
    pub fn prepared_source(&self) -> &crate::input::OriginalPreparedWorkspaceSource {
        &self.source
    }

    pub(super) fn compose_copied_source_metadata(
        &self,
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        full: ExecutionWorkspaceEstimate,
        incremental: ExecutionWorkspaceEstimate,
        metadata: super::super::WorkspaceReportMetadata<'_>,
    ) -> Result<IncrementalInferenceQuote, WorkspaceCopyCompositionError> {
        ResidualInferenceQuote::compose_parts_metadata(
            equations,
            state,
            full,
            incremental,
            &self.registered,
            None,
            None,
            Some(&self.source),
            metadata,
        )
        .map(ResidualInferenceQuote::into_incremental)
    }
    pub(super) fn pool(&self) -> &WorkingMemoryPool {
        &self.registered.pool
    }

    /// Runs the same residual/controller composition and returns only its closed
    /// incremental proof. Both accounting-only source pins survive reservation.
    pub fn compose_with_controller_metadata(
        &self,
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        contribution: ControllerWorkspaceEstimate,
        metadata: super::super::WorkspaceReportMetadata<'_>,
    ) -> Result<IncrementalInferenceQuote, WorkspaceCopyCompositionError> {
        ResidualInferenceQuote::compose_with_controller_source_metadata(
            equations,
            state,
            contribution,
            &self.registered,
            Some(&self.source),
            metadata,
        )
        .map(ResidualInferenceQuote::into_incremental)
    }
}

/// Finite construction layout from the actual maximum input-root population.
/// This owns no source, admission, native handle or allocation permission. Key
/// payloads, comparison callbacks and iterator construction belong to the
/// caller's source producer. This worker moves keys and only clones the existing
/// metadata Rc handles.
#[derive(Debug)]
pub struct RegisteredWorkspaceStorageLayout<K: Ord + Send + 'static> {
    slots: usize,
    prepared_roots: usize,
    bytes: usize,
    marker: PhantomData<fn() -> K>,
}
impl<K: Clone + Ord + Send + Sync + 'static> RegisteredWorkspaceStorageLayout<K> {
    /// Pure owning layout for the pinned fresh Vec/Rc/Arc storage producer.
    /// Unqualified allocator/compiler facts remain unknown before construction.
    pub fn new(slots: usize) -> Result<Self, WorkingMemoryError> {
        Self::new_with_source_roots(slots, 0)
    }
    /// Includes the exact completed B root selection without registering it or
    /// charging its already-owned physical storage a second time.
    pub fn new_with_prepared_source(
        slots: usize,
        source: &crate::input::OriginalPreparedWorkspaceSource,
    ) -> Result<Self, WorkingMemoryError> {
        let roots = source
            .borrowed_storage()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Self::new_with_source_roots(slots, roots.roots().len())
    }
    fn new_with_source_roots(
        slots: usize,
        prepared_roots: usize,
    ) -> Result<Self, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let input = Layout::array::<(K, WorkspaceExistingStorage)>(slots)
            .map_err(|_| WorkingMemoryError::Overflow)?
            .size();
        let pins = Layout::array::<(K, u64)>(slots)
            .map_err(|_| WorkingMemoryError::Overflow)?
            .size();
        let total_roots = slots
            .checked_add(prepared_roots)
            .ok_or(WorkingMemoryError::Overflow)?;
        let root = WorkspaceBorrowedStorage::construction_bytes(total_roots)
            .ok_or(WorkingMemoryError::Overflow)?;
        let pin = finite_pin::construction_bytes::<K>(slots)?;
        let frames = [
            size_of::<Self>(),
            size_of::<RegisteredWorkspaceStorage<K>>(),
            size_of::<RegisteredPreparedWorkspaceStorage<K>>(),
            size_of::<Result<RegisteredPreparedWorkspaceStorage<K>, WorkingMemoryError>>(),
            size_of::<
                Result<
                    (
                        RegisteredWorkspaceStorage<K>,
                        Option<crate::input::OriginalPreparedWorkspaceSource>,
                    ),
                    WorkingMemoryError,
                >,
            >(),
            size_of::<(Option<&WorkspaceBorrowedStorage>, usize)>(),
            size_of::<Result<RegisteredWorkspaceStorage<K>, WorkingMemoryError>>(),
            size_of::<Vec<(K, WorkspaceExistingStorage)>>(),
            size_of::<Vec<(K, u64)>>(),
            size_of::<std::vec::IntoIter<(K, WorkspaceExistingStorage)>>(),
            size_of::<std::slice::Iter<'static, (K, WorkspaceExistingStorage)>>(),
            size_of::<std::vec::IntoIter<(K, u64)>>(),
            size_of::<WorkspaceBorrowedStorageError>(),
            size_of::<Option<crate::input::OriginalPreparedWorkspaceSource>>(),
            size_of::<Result<Option<&WorkspaceBorrowedStorage>, WorkingMemoryError>>(),
            size_of::<WorkingMemoryError>(),
            size_of::<(&WorkingMemoryPool, &WorkspaceContext, usize, usize, bool, bool)>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        let vector = usize::try_from(qualified_storage::vector_control_bytes::<(
            K,
            WorkspaceExistingStorage,
        )>()?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
        let pin_vector = usize::try_from(qualified_storage::vector_control_bytes::<(K, u64)>()?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let bytes = [input, pins, root, pin, frames, vector, pin_vector]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            slots,
            prepared_roots,
            bytes,
            marker: PhantomData,
        })
    }
    pub(in crate::working_memory) fn with_completed_roots(
        slots: usize,
        roots: usize,
    ) -> Result<Self, WorkingMemoryError> {
        Self::new_with_source_roots(slots, roots)
    }
    pub(in crate::working_memory) fn construct_with_completed_roots(
        self,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
        storage: impl IntoIterator<Item = (K, WorkspaceExistingStorage)>,
        roots: &WorkspaceBorrowedStorage,
        select: bool,
    ) -> Result<RegisteredWorkspaceStorage<K>, WorkingMemoryError> {
        if roots.roots().len() != self.prepared_roots {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut rows = qualified_storage::vector(self.slots, true)?;
        for row in storage {
            if rows.len() == self.slots {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            rows.push(row);
        }
        bind_rows(pool, context, rows, true, Some(roots), select)
    }
    /// Binding constructor controls/backing; provider key/iterator work is separate.
    pub fn requested_bytes(&self) -> usize {
        self.bytes
    }
    /// Maximum input records, including duplicates, used by this finite worker.
    pub fn source_slots(&self) -> usize {
        self.slots
    }
    /// Consumes the planned slots after the caller's host admission. Failure
    /// drops input keys and partial metadata outside accounting locks; it neither
    /// registers new physical bytes nor changes the source context selection.
    pub fn construct(
        self,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
        storage: impl IntoIterator<Item = (K, WorkspaceExistingStorage)>,
    ) -> Result<RegisteredWorkspaceStorage<K>, WorkingMemoryError> {
        self.construct_with_source(pool, context, storage, None, true)
            .map(|(registered, _)| registered)
    }
    /// Validates and retains the same existing-only source mapping without
    /// installing a context selection. The caller retains this binding while
    /// installing one exact union before tracing; no native permission is added.
    pub fn construct_unselected(
        self,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
        storage: impl IntoIterator<Item = (K, WorkspaceExistingStorage)>,
    ) -> Result<RegisteredWorkspaceStorage<K>, WorkingMemoryError> {
        self.construct_with_source(pool, context, storage, None, false)
            .map(|(registered, _)| registered)
    }
    /// Consumes the same original B witness used by this layout, keeping its
    /// metadata separate from its closed account-only reservation pin.
    pub fn construct_with_prepared_source(
        self,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
        storage: impl IntoIterator<Item = (K, WorkspaceExistingStorage)>,
        source: crate::input::OriginalPreparedWorkspaceSource,
    ) -> Result<RegisteredPreparedWorkspaceStorage<K>, WorkingMemoryError> {
        self.construct_with_source(pool, context, storage, Some(source), true)
            .map(|(registered, source)| RegisteredPreparedWorkspaceStorage {
                registered,
                source: source.expect("supplied original source"),
            })
    }
    /// Retains the same genuine prepared-input account without changing the
    /// context selection. A caller can then select its complete source union.
    pub fn construct_unselected_with_prepared_source(
        self, pool: &WorkingMemoryPool, context: &WorkspaceContext,
        storage: impl IntoIterator<Item = (K, WorkspaceExistingStorage)>,
        source: crate::input::OriginalPreparedWorkspaceSource,
    ) -> Result<RegisteredPreparedWorkspaceStorage<K>, WorkingMemoryError> {
        self.construct_with_source(pool, context, storage, Some(source), false)
            .map(|(registered, source)| RegisteredPreparedWorkspaceStorage {
                registered, source: source.expect("supplied original source"),
            })
    }
    fn construct_with_source(
        self,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
        storage: impl IntoIterator<Item = (K, WorkspaceExistingStorage)>,
        source: Option<crate::input::OriginalPreparedWorkspaceSource>,
        select: bool,
    ) -> Result<
        (
            RegisteredWorkspaceStorage<K>,
            Option<crate::input::OriginalPreparedWorkspaceSource>,
        ),
        WorkingMemoryError,
    > {
        let roots = source
            .as_ref()
            .map(|source| {
                source
                    .borrowed_storage()
                    .ok_or(WorkingMemoryError::UnknownBound)
            })
            .transpose()?;
        if roots.map_or(0, |roots| roots.roots().len()) != self.prepared_roots {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut rows = qualified_storage::vector(self.slots, true)?;
        for row in storage {
            if rows.len() == self.slots {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            rows.push(row);
        }
        if source
            .as_ref()
            .is_some_and(|source| !source.pool().same_domain(pool))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let registered = bind_rows(pool, context, rows, true, roots, select)?;
        Ok((registered, source))
    }
}
impl<K: Clone + Ord + Send + Sync + 'static> RegisteredWorkspaceStorage<K> {
    /// Associates provider physical identities with exact known metadata roots.
    /// Duplicate identical pairs share a pin. Other many-to-one mappings reject
    /// before accounting mutation. Ordinary collection uses the same validator,
    /// sorted root order, finite root construction and existing-only pin commit.
    pub fn bind(
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
        storage: impl IntoIterator<Item = (K, WorkspaceExistingStorage)>,
    ) -> Result<Self, WorkingMemoryError> {
        bind_rows(pool, context, storage.into_iter().collect(), false, None, true)
    }
    /// Exact immutable roots for this binding. Unselected bindings require the
    /// caller to install their complete union before equation tracing.
    pub fn borrowed_storage(&self) -> &WorkspaceBorrowedStorage {
        &self.borrowed
    }
    /// Domain whose already-registered physical charges cover these roots.
    pub fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }
    pub(in crate::working_memory) fn registration(&self) -> &WorkingMemoryStorage<K> {
        &self._registration
    }
}
fn binding_error(error: WorkspaceBorrowedStorageError) -> WorkingMemoryError {
    match error {
        WorkspaceBorrowedStorageError::Reserve(cause) => {
            WorkingMemoryError::ControlStorageReserve(cause)
        }
        _ => WorkingMemoryError::IdentityMismatch,
    }
}
fn bind_rows<K: Ord + Send + 'static>(
    pool: &WorkingMemoryPool,
    context: &WorkspaceContext,
    mut rows: Vec<(K, WorkspaceExistingStorage)>,
    exact: bool,
    prepared_roots: Option<&WorkspaceBorrowedStorage>,
    select: bool,
) -> Result<RegisteredWorkspaceStorage<K>, WorkingMemoryError> {
    // Preserve input-order first failure, before sorting or dropping duplicates.
    for (i, (key, root)) in rows.iter().enumerate() {
        if root.capacity_bytes().is_none() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        for (prior_key, prior_root) in &rows[..i] {
            let same_key = prior_key.cmp(key).is_eq();
            if same_key != prior_root.same_storage(root) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
    }
    rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    rows.dedup_by(|a, b| a.0.cmp(&b.0).is_eq());
    let root_count = rows
        .len()
        .checked_add(prepared_roots.map_or(0, |roots| roots.roots().len()))
        .ok_or(WorkingMemoryError::Overflow)?;
    let borrowed = WorkspaceBorrowedStorage::new_finite(
        context,
        rows.iter()
            .map(|(_, root)| root)
            .chain(prepared_roots.into_iter().flat_map(|roots| roots.roots())),
        root_count,
    )
    .map_err(binding_error)?;
    let mut pins = qualified_storage::vector(rows.len(), exact)?;
    for (key, root) in rows {
        pins.push((key, root.capacity_bytes().expect("validated root")));
    }
    let registration = pool.pin_registered_storage_owned(pins, exact)?;
    if select {
        context
            .set_borrowed_storage_checked(borrowed.clone())
            .map_err(binding_error)?;
    }
    Ok(RegisteredWorkspaceStorage {
        borrowed,
        pool: pool.clone(),
        _registration: registration,
    })
}

impl<K: crate::working_memory::HostSlotStorageKey> RegisteredWorkspaceStorage<K> {
    /// Attaches the same existing-only source carrier without changing this
    /// workspace/root association. The carrier preserves constructor custody;
    /// it grants no additional numerical copy or native execution permission.
    pub fn with_retained_original_sources(
        mut self,
        sources: &mut crate::working_memory::RetainedOriginalStorageSources,
    ) -> Result<Self, WorkingMemoryError> {
        self._registration = self._registration.with_retained_original_sources(sources)?;
        Ok(self)
    }
}

impl<K: crate::working_memory::HostSlotStorageKey> RegisteredPreparedWorkspaceStorage<K> {
    /// Preserve the already accepted host constructor while retaining the B
    /// source separately; no native allocation is adopted into the registry.
    pub fn with_retained_original_sources(
        mut self,
        sources: &mut crate::working_memory::RetainedOriginalStorageSources,
    ) -> Result<Self, WorkingMemoryError> {
        self.registered = self.registered.with_retained_original_sources(sources)?;
        Ok(self)
    }
}
