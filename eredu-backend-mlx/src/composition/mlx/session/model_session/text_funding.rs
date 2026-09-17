//! Publication of surviving native storage before retiring a quoted work scope.

use super::*;
use crate::backend::runtime::residency::storage::{RetainedStorage, RetainedStoragePublication};
use eredu_runtime::working_memory::{
    CollectorCapacityKind, OriginalResidentResetSource, OriginalTextControlGuard,
    WorkingMemoryError, WorkingMemoryFundingScope,
};

// A read-only model obligation survives scope certification. Its actual pool
// permits validation of repeated publication without reopening or refilling it.
struct ModelTable {
    source: OriginalResidentResetSource,
    pool: eredu_runtime::working_memory::WorkingMemoryPool,
}

mod capture;
mod collectors;
mod native_publication;
mod owner;
mod prepared;
mod snapshot;
pub(super) use snapshot::SnapshotPublicationPlan;
mod publication_scope;
#[cfg(test)]
mod publication_tests;
pub(super) use owner::FundedWorkOwner;
pub(super) use prepared::{PreparedFundedWork, PreparedWorkRetention};

pub(in crate::composition::mlx::session) fn capture_replica_control_bytes() -> Option<usize> { capture::replica_control_bytes() }
pub(super) fn observer_error_control_bytes() -> Option<usize> {
    capture::observer_error_control_bytes()
}

/// Native recovery retains this independently of request identity and run state.
/// Failed or incomplete publication leaves the scope uncertified, preserving its
/// full remaining bound. Attached storage handles contain no native roots.
pub(super) struct FundedWork {
    scope: RefCell<Option<WorkingMemoryFundingScope>>,
    roots: RefCell<Vec<Array>>,
    root_clones: RefCell<Vec<safemlx::PreparedArrayClone>>,
    inventories: RefCell<Vec<RetainedStorage>>,
    collection_failure: RefCell<Option<Error>>,
    // Taking the first owned cause must never reopen the failed collector.
    collection_failed: Cell<bool>,
    collectors_prepared: Cell<bool>,
    metadata: RefCell<Vec<eredu_runtime::SharedHostMetadata>>,
    publications: RefCell<Vec<RetainedStoragePublication>>,
    native_publications: native_publication::NativePublications,
    native_storage: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    published: Cell<bool>,
    publishing: Cell<bool>,
    capture: RefCell<Option<capture::CaptureCarrierOwner>>,
    opening_rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
    text_interventions: Option<crate::composition::mlx::session::intervention::PreparedTextInterventionsOwner>,
    original_table: Option<ModelTable>,
    // Closed account/portable-root profile from the accepted candidate. It has
    // no native payload backedge and stays through publication/recovery.
    prepared_source: Option<eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    // Last: original P+Q outlives roots, carrier, publications and the native
    // scope, including quarantine. certify() never takes this guard.
    _controls: Option<OriginalTextControlGuard>,
    snapshot: Option<snapshot::SnapshotWork>,
}

impl FundedWork {
    pub(super) fn bind_snapshot_host_copy(
        &self,
        copy: &crate::backend::array_copy::PreparedSavedHostCopy,
    ) -> Result<(), Error> {
        self.snapshot
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
            .bind_host_copy(copy)
    }

    #[cfg(test)]
    pub(super) fn test_is_published(&self) -> bool {
        self.published.get()
    }

    #[cfg(test)]
    pub(super) fn test_scope_is_retired(&self) -> bool {
        self.scope.try_borrow_mut().unwrap().is_none()
    }

    pub(super) fn new(scope: WorkingMemoryFundingScope) -> FundedWorkOwner {
        Self::allocate(Some(scope), None, None, None, None, None, None, None)
    }

    pub(super) fn new_with_controls(
        scope: WorkingMemoryFundingScope,
        controls: Option<OriginalTextControlGuard>,
    ) -> Result<FundedWorkOwner, Error> {
        Self::new_with_opening_rows(scope, controls, None)
    }

    pub(super) fn new_with_opening_rows(
        scope: WorkingMemoryFundingScope,
        controls: Option<OriginalTextControlGuard>,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
    ) -> Result<FundedWorkOwner, Error> {
        Self::new_model(scope, controls, rows, None)
    }

    pub(super) fn new_model(
        scope: WorkingMemoryFundingScope,
        controls: Option<OriginalTextControlGuard>,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
        original_table: Option<OriginalResidentResetSource>,
    ) -> Result<FundedWorkOwner, Error> {
        Self::new_model_with_native(scope, controls, rows, None, original_table, None, None)
    }

    pub(super) fn new_model_with_native(
        scope: WorkingMemoryFundingScope,
        controls: Option<OriginalTextControlGuard>,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
        text_interventions: Option<crate::composition::mlx::session::intervention::PreparedTextInterventionsOwner>,
        original_table: Option<OriginalResidentResetSource>,
        prepared_source: Option<eredu_runtime::input::OriginalPreparedWorkspaceSource>,
        native_storage: Option<
            crate::backend::runtime::residency::storage::native_storage::BankOwner,
        >,
    ) -> Result<FundedWorkOwner, Error> {
        if prepared_source
            .as_ref()
            .is_some_and(|source| controls.is_none() || !source.pool().same_domain(scope.pool()))
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        if native_storage.is_some() && controls.is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        if let Some(edits) = &text_interventions {
            if controls.is_none() {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            edits.source().validate_pool(scope.pool()).map_err(Error::PrefillControl)?;
        }
        if let Some(source) = &original_table {
            if controls.is_none() {
                return Err(Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )));
            }
            scope
                .pool()
                .pin_original_reset_slots(source.metadata())
                .map_err(|error| Error::Other(Box::new(error)))?;
        }
        if let Some(rows) = &rows {
            rows.validate_scope(&scope)?;
        }
        if let Some(controls) = &controls {
            controls
                .validate_native_scope(&scope)
                .map_err(|error| Error::Other(Box::new(error)))?;
        }
        // Validation and original custody precede the Rc and later collectors.
        let original_table = original_table.map(|source| ModelTable {
            source,
            pool: scope.pool().clone(),
        });
        let work = Self::allocate(
            Some(scope),
            controls,
            rows,
            text_interventions,
            original_table,
            prepared_source,
            native_storage,
            None,
        );
        work.prepare_collectors()?;
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        if let Some(source) = &work.original_table {
            super::resident_reset::tests::consumer::record(&work, &source.source);
        }
        Ok(work)
    }

    fn allocate(
        scope: Option<WorkingMemoryFundingScope>,
        controls: Option<OriginalTextControlGuard>,
        opening_rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
        text_interventions: Option<crate::composition::mlx::session::intervention::PreparedTextInterventionsOwner>,
        original_table: Option<ModelTable>,
        prepared_source: Option<eredu_runtime::input::OriginalPreparedWorkspaceSource>,
        native_storage: Option<
            crate::backend::runtime::residency::storage::native_storage::BankOwner,
        >,
        snapshot: Option<snapshot::SnapshotWork>,
    ) -> FundedWorkOwner {
        FundedWorkOwner::new(Self {
            scope: RefCell::new(scope),
            roots: RefCell::new(Vec::new()),
            root_clones: RefCell::new(Vec::new()),
            inventories: RefCell::new(Vec::new()),
            collection_failure: RefCell::new(None),
            collection_failed: Cell::new(false),
            collectors_prepared: Cell::new(false),
            metadata: RefCell::new(Vec::new()),
            publications: RefCell::new(Vec::new()),
            native_publications: native_publication::NativePublications::default(),
            native_storage,
            published: Cell::new(false),
            publishing: Cell::new(false),
            capture: RefCell::new(None),
            opening_rows,
            text_interventions,
            original_table,
            prepared_source,
            _controls: controls,
            snapshot,
        })
    }

    fn record_collection_failure(&self, cause: Error) {
        if !self.collection_failed.replace(true) {
            *self.collection_failure.borrow_mut() = Some(cause);
        }
    }

    /// Infallible callbacks must return a consumed cause to this owner rather
    /// than discard it. The permanent fence remains set, and an earlier stored
    /// cause keeps its custody until a fallible caller takes it.
    pub(super) fn retain_callback_failure(&self, cause: Error) {
        self.collection_failed.set(true);
        let mut first = self.collection_failure.borrow_mut();
        if first.is_none() {
            *first = Some(cause);
        }
    }

    pub(super) fn take_collection_failure(&self) -> Option<Error> {
        self.collection_failed.get().then(|| {
            // Move the original native/source error once, preserving custody.
            // Subsequent calls stay fenced even after that owner has escaped.
            self.collection_failure
                .borrow_mut()
                .take()
                .unwrap_or(Error::PrefillControl(WorkingMemoryError::ExecutionFenced))
        })
    }

    fn fail_collection(&self, cause: Error) -> Error {
        self.record_collection_failure(cause);
        self.take_collection_failure().expect("failed collector")
    }

    pub(super) fn retain(&self, array: &Array) {
        if self.native_storage.is_none() && self.snapshot.is_none() {
            self.roots.borrow_mut().push(array.clone());
            self.published.set(false);
            return;
        }
        if self.collection_failed.get() {
            return;
        }
        let result =
            (|| {
                let mut roots = self.roots.borrow_mut();
                if roots.len() == roots.capacity() {
                    return Err(Error::PrefillControl(
                        WorkingMemoryError::CollectorCapacity {
                            kind: CollectorCapacityKind::WorkRoots,
                            used: roots.len(),
                            capacity: roots.capacity(),
                        },
                    ));
                }
                let mut slots = self.root_clones.borrow_mut();
                let capacity = slots.capacity();
                let used = capacity - slots.len();
                let slot = slots.last_mut().ok_or(Error::PrefillControl(
                    WorkingMemoryError::CollectorCapacity {
                        kind: CollectorCapacityKind::WorkCloneShells,
                        used,
                        capacity,
                    },
                ))?;
                let array =
                    slot.fill_for_inspection(array).map_err(|cause| {
                        Error::from(
                crate::backend::runtime::residency::manager::ResidencyError::OriginalClone(cause))
                    })?;
                slots.pop();
                roots.push(array);
                Ok(())
            })();
        if let Err(cause) = result {
            self.record_collection_failure(cause);
        }
        self.published.set(false);
    }

    /// Retains only metadata constructed by a worker included in this scope's
    /// original quote. A terminal overrun fences publication without growth.
    pub(super) fn retain_metadata(&self, owner: eredu_runtime::SharedHostMetadata) {
        if self.collection_failed.get() {
            return;
        }
        let mut metadata = self.metadata.borrow_mut();
        if (self.native_storage.is_some() || self.snapshot.is_some())
            && metadata.len() == metadata.capacity()
        {
            let cause = Error::PrefillControl(WorkingMemoryError::CollectorCapacity {
                kind: CollectorCapacityKind::WorkMetadata,
                used: metadata.len(),
                capacity: metadata.capacity(),
            });
            drop(metadata);
            self.record_collection_failure(cause);
        } else {
            metadata.push(owner);
        }
        self.published.set(false);
    }

    /// The caller establishes completion separately. Inventory collection never
    /// evaluates a missing backing or substitutes logical size for its capacity.
    pub(super) fn publish(&self, storage: RetainedStorage) -> Result<(), Error> {
        self.publish_parts(storage, None).map(|_| ())
    }

    /// Model sources belong to the session, not to an escaped token. Return
    /// their publication for installation on the payload, while this owner
    /// retains only decoder/output publications. Both use the same work scope.
    pub(super) fn publish_model(
        &self,
        nonstate: RetainedStorage,
        decoder: RetainedStorage,
    ) -> Result<Option<RetainedStoragePublication>, Error> {
        self.publish_parts(decoder, Some(nonstate))
    }

    fn publish_parts(
        &self,
        mut storage: RetainedStorage,
        nonstate: Option<RetainedStorage>,
    ) -> Result<Option<RetainedStoragePublication>, Error> {
        let _activity = publication_scope::Activity::begin(&self.publishing)?;
        if let Some(cause) = self.take_collection_failure() {
            return Err(cause);
        }
        if (self.native_storage.is_some() || self.snapshot.is_some())
            && (!storage.is_original_collector()
                || nonstate
                    .as_ref()
                    .is_some_and(|s| !s.is_original_collector()))
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        if self.native_storage.is_some() || self.snapshot.is_some() {
            let publications = self.publications.borrow();
            if publications.len() == publications.capacity() {
                let cause = Error::PrefillControl(WorkingMemoryError::CollectorCapacity {
                    kind: CollectorCapacityKind::WorkPublications,
                    used: publications.len(),
                    capacity: publications.capacity(),
                });
                drop(publications);
                return Err(self.fail_collection(cause));
            }
        }
        if self.snapshot.is_some() && nonstate.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        // Even an idempotent call must not accept a replaced/missing source.
        // The inline pool is the original scope's exact domain, not new authority.
        if let Some(expected) = &self.original_table {
            let nonstate = nonstate.as_ref().ok_or_else(|| {
                Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ))
            })?;
            nonstate.validate_original_sources(
                &expected.pool,
                Some(&expected.source),
                self.prepared_source.as_ref(),
            )?;
            storage.validate_original_sources(
                &expected.pool,
                None,
                self.prepared_source.as_ref(),
            )?;
        } else if let Some(source) = &self.prepared_source {
            if let Some(nonstate) = &nonstate {
                nonstate.validate_original_sources(source.pool(), None, Some(source))?;
            }
            storage.validate_original_sources(source.pool(), None, Some(source))?;
        }
        if self.published.get() && !self.capture_is_active()? {
            return Ok(None);
        }
        self.include_capture_roots(&mut storage)?;
        for root in self.roots.borrow().iter() {
            storage.include_array(root)?;
        }
        for owner in self.metadata.borrow().iter() {
            storage.include_metadata(owner.clone())?;
        }
        // Source/account callbacks and validation run outside the scope loan.
        // Validate BOTH inventories before the first adoption can mutate Usage.
        let pool = self
            .scope
            .borrow()
            .as_ref()
            .map(|scope| scope.pool().clone())
            .ok_or_else(|| {
                Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ))
            })?;
        match (&nonstate, &self.original_table) {
            (Some(nonstate), expected) => nonstate.validate_original_sources(
                &pool,
                expected.as_ref().map(|s| &s.source),
                self.prepared_source.as_ref(),
            )?,
            (None, Some(_)) => {
                return Err(Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )));
            }
            (None, None) => {}
        }
        if self.snapshot.is_none() {
            storage.validate_original_sources(&pool, None, self.prepared_source.as_ref())?;
        }
        // Native attachment can reclaim older owners and reenter accounting.
        // Own the actual scope during those calls without retaining a cell loan.
        // Both this ownership and the publication fence unwind before returning.
        let mut scope = publication_scope::OwnedScope::take(&self.scope)?;
        let (nonstate, publication) = if let Some(snapshot) = &self.snapshot {
            (None, snapshot.publish(storage, scope.get())?)
        } else if let Some(bank) = &self.native_storage {
            let controls = self
                ._controls
                .as_ref()
                .expect("selected native work controls");
            let nonstate = nonstate
                .map(|storage| {
                    self.native_publications.publish(
                        storage,
                        bank,
                        scope.get_mut(),
                        controls,
                        self.original_table.as_ref().map(|source| &source.source),
                        self.prepared_source.as_ref(),
                    )
                })
                .transpose()?;
            let publication = self.native_publications.publish(
                storage,
                bank,
                scope.get_mut(),
                controls,
                None,
                self.prepared_source.as_ref(),
            )?;
            (nonstate, publication)
        } else {
            let nonstate = nonstate
                .map(|storage| {
                    if self.original_table.is_some() || self.prepared_source.is_some() {
                        storage.publish_funded_with_original_sources(
                            scope.get(),
                            self.original_table.as_ref().map(|source| &source.source),
                            self.prepared_source.as_ref(),
                        )
                    } else {
                        storage.publish_funded(scope.get())
                    }
                })
                .transpose()?;
            let publication = if self.prepared_source.is_some() {
                storage.publish_funded_with_original_sources(
                    scope.get(),
                    None,
                    self.prepared_source.as_ref(),
                )?
            } else {
                storage.publish_funded(scope.get())?
            };
            (nonstate, publication)
        };
        // Restore the same scope before ordinary collector retirement. The
        // activity fence remains until all publication bookkeeping is complete.
        drop(scope);
        self.publications.borrow_mut().push(publication);
        // Attached charge destructors can reenter accounting. Drop payload
        // handles after releasing the collection borrows.
        let roots = std::mem::take(&mut *self.roots.borrow_mut());
        let metadata = std::mem::take(&mut *self.metadata.borrow_mut());
        drop((roots, metadata));
        // A live capture channel retains its own roots/publications until its
        // canonical ticket is consumed. Final publication cannot bypass it.
        self.published.set(!self.capture_is_active()?);
        Ok(nonstate)
    }

    pub(super) fn certify(&self) -> Result<(), Error> {
        publication_scope::ensure_idle(&self.publishing)?;
        if let Some(cause) = self.take_collection_failure() {
            return Err(cause);
        }
        self.require_no_capture_carrier()?;
        if self.published.get() {
            let scope = self.scope.borrow_mut().take();
            if let Some(scope) = scope {
                scope
                    .certify()
                    .map_err(|error| Error::Other(Box::new(error)))?;
            }
        }
        Ok(())
    }

    pub(super) fn observe_preparation(&self, status: Status) {
        if status.settled && !status.failed && !status.blocked {
            if let Err(cause) = self
                .prepare_inventory()
                .and_then(|storage| self.publish(storage))
                .and_then(|()| self.certify())
            {
                self.retain_callback_failure(cause);
            }
        }
    }
}

/// This field is placed after a completion's native roots. Its destruction does
/// not establish native completion: outstanding recovery tickets still prevent
/// certification and retain the native payload independently.
pub(in crate::composition::mlx::session) struct RetireFundedCompletion(
    pub(super) SubmissionResourcesOwner,
);

impl Drop for RetireFundedCompletion {
    fn drop(&mut self) {
        self.0.funding_retired.set(true);
        self.0.release_if_settled();
    }
}

/// Concrete work/submission allocation and retirement plus optional-carrier controls.
/// Original text or copy composition consumes this fact once; no late hold.
pub(super) fn work_control_bytes() -> Result<u64, Error> {
    capture::common_control_bytes()?
        .checked_add(
            super::text_quote::original_table_work_control_bytes().ok_or_else(|| {
                Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                ))
            })?,
        )
        .ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })?
        .checked_add(super::submission_owner::control_bytes().ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })?)
        .ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })?
        .checked_add(owner::control_bytes().ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })?)
        .ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })
}
/// Prompt and Sampling owners plus at most M inference work owners may overlap
/// through escaped input/state/completion aliases. Each quoted inference has one
/// submission owner; using the same M+2 ceiling also covers preparation controls.
/// This introduces no hold.
pub(super) fn text_work_control_bytes(predictions: u64) -> Result<u64, Error> {
    let common = work_control_bytes()?
        .checked_add(super::text_error::control_peak_bytes().ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })?)
        .ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })?;
    predictions
        .checked_add(2)
        .and_then(|n| n.checked_mul(common))
        .ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })
}
pub(super) fn capture_carrier_control_bytes(
    source: &eredu_core::capture::SharedCapturePlan,
    interventions: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
) -> Result<u64, Error> {
    capture::active_control_bytes(source, interventions)
}
/// A copy retains this measured ancillary control charge in its existing single
/// destination account. The caller's reserve remains additive and unchanged.
pub(super) fn copy_limits_with_work_controls(
    mut limits: eredu_runtime::working_memory::WorkspaceCopyLimits,
) -> Result<eredu_runtime::working_memory::WorkspaceCopyLimits, Error> {
    limits.safety_reserve_bytes = limits
        .safety_reserve_bytes
        .checked_add(work_control_bytes()?)
        .ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))
        })?;
    Ok(limits)
}

/// Actual original collector destinations and finite publication owner population.
/// Native P and neutral registry/attachment storage are separate contributions.
pub(super) fn native_collector_control_bytes(
    attempts: usize,
    rows: usize,
    works: usize,
) -> Result<Option<u64>, Error> {
    collectors::control_bytes(attempts, rows, works)
}
