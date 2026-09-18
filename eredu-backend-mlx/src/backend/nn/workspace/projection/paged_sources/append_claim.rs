//! Actual admitted role, retained source and one-use append program.
mod visible;
use super::programs::PagedAppendProgram;
use super::*;
use crate::backend::{
    error::Error as NativeError,
    runtime::cache::{
        kv::ProjectedPagedSource,
        residency::{
            CacheBlockArrays, CacheBlockMetadata, CacheResidencyError, CacheResidencyManager,
            CacheSourceError, InstalledManagerCatalog,
        },
    },
    submission_recovery::prefill::TransientRootsProjection,
};
use eredu_core::cache::{CacheBlockId, CacheRankIdentity};
use eredu_runtime::{cache::PagedAppendPlan, working_memory::WorkspacePagedGeometry};
use safemlx::{Array, Dtype, OriginalScopeObserver, error::Exception};
use std::{mem::size_of, rc::Weak};
pub(crate) use visible::OriginalPagedVisibleClaim;
pub(super) use visible::control_bytes as visible_control_bytes;

thread_local! { static CURRENT: RefCell<Weak<PagedSourceBank>> = const { RefCell::new(Weak::new()) }; }
pub(super) fn publish_current(source: &ProjectedPagedSources) {
    CURRENT.with(|slot| {
        *slot.borrow_mut() = Rc::downgrade(&source.inner);
    });
}
/// Describes actual current pager/input geometry; it is never itself authority.
pub(crate) struct PagedAppendInput<'a> {
    pub(crate) manager: &'a CacheResidencyManager,
    pub(crate) global_layer: usize,
    pub(crate) rank: Option<CacheRankIdentity>,
    pub(crate) geometry: WorkspacePagedGeometry,
    pub(crate) plan: PagedAppendPlan,
    pub(crate) dtypes: [Dtype; 2],
    pub(crate) tail_dtypes: Option<[Dtype; 2]>,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum PagedMutationCause {
    #[error(transparent)]
    Source(#[from] CacheSourceError),
    #[error(transparent)]
    Residency(#[from] CacheResidencyError),
    #[error(transparent)]
    Roots(#[from] NativeError),
    #[error(transparent)]
    Stream(#[from] safemlx::StreamCopyCause),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: PagedMutationCause,
    _custody: Exception,
    _funding: Option<HostMetadataFunding>,
}
pub(super) fn failure(
    cause: impl Into<PagedMutationCause>,
    observer: &OriginalScopeObserver,
    funding: Option<HostMetadataFunding>,
) -> Exception {
    Exception::from_retained_source(Failure {
        cause: cause.into(),
        _custody: observer.invalid_input_error(),
        _funding: funding,
    })
}

/// Constructor is private to this exact source/role bank. The manager may only
/// borrow its source proof during one ordered native append; no claim escapes.
pub(crate) struct OriginalPagedAppendClaim<'a> {
    source: &'a ProjectedPagedSource,
    host: Option<&'a mut super::host_program::PreparedPagedHostProgram>,
    bank: &'a PagedSourceBank,
    custody: eredu_runtime::working_memory::OriginalHostSourceCustody,
    installed: &'a InstalledManagerCatalog,
    program: &'a mut PagedAppendProgram,
    observer: OriginalScopeObserver,
    roots: TransientRootsProjection,
    retained: usize,
    published: usize,
    step: usize,
    tail_ready: bool,
    tail_cleared: bool,
    initial_tail_bytes: u64,
    prepared_tail_bytes: Option<u64>,
    context: &'a WorkspaceContext,
}
impl ProjectedPagedSources {
    /// Final completion must consume every quoted layer append of this exact
    /// active role. An unrelated root owner cannot acquire this source grant.
    pub(crate) fn validate_current_append_completion() -> Result<(), Exception> {
        let Some(observer) = OriginalScopeObserver::try_current()? else {
            return Ok(());
        };
        let source = CURRENT
            .with(|slot| slot.try_borrow().map(|slot| slot.upgrade()))
            .map_err(|_| observer.invalid_input_error())?;
        let Some(source) = source else {
            return Ok(());
        };
        let fail = |cause| failure(cause, &observer, source.context.metadata_funding());
        let roles = source
            .roles
            .try_borrow()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        let Some(active) = &roles.active else {
            return Ok(());
        };
        if !active.observer.same_scope(&observer) {
            return Ok(());
        }
        if roles.failed || active.append_active {
            return Err(fail(CacheSourceError::Identity));
        }
        let loan = source
            .catalogs
            .try_borrow()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        let catalogs = loan
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if !catalogs.programs.iter().all(|program| {
            program.as_ref().is_some_and(|program| {
                program.ordinal != active.ordinal
                    || (program.completed
                        && program
                            .scan
                            .as_ref()
                            .is_none_or(|scan| !scan.used || scan.completed))
            })
        }) {
            return Err(fail(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn with_current_append<R>(
        actual: PagedAppendInput<'_>,
        run: impl FnOnce(&mut OriginalPagedAppendClaim<'_>) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let CurrentRole {
            source,
            observer,
            ordinal,
            roots,
        } = current_role()?;
        let fail = |cause| failure(cause, &observer, source.context.metadata_funding());
        let mut catalogs_loan = source
            .catalogs
            .try_borrow_mut()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        let catalogs = catalogs_loan
            .as_mut()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let source_index = source
            .sources
            .iter()
            .position(|source| {
                source.manager().same_catalog(actual.manager)
                    && source.geometry().global_layer == actual.global_layer
            })
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let retained_source = &source.sources[source_index];
        let expected = retained_source.append_geometry();
        let current = actual.geometry;
        if actual
            .dtypes
            .into_iter()
            .any(|dtype| CacheBlockMetadata::floating_dtype_bytes(dtype).is_none())
            || actual.rank != retained_source.geometry().rank
            || current.block_size != expected.block_size
            || current.dimensions != expected.dimensions
            || current.window != expected.window
            || current.prefix_tokens != expected.prefix_tokens
            || current.key_only != expected.key_only
            || current.retain_discarded != expected.retain_discarded
        {
            return Err(fail(CacheSourceError::Geometry));
        }
        let installed = catalogs
            .installed
            .iter()
            .find(|entry| entry.manager().same_catalog(actual.manager))
            .ok_or_else(|| fail(CacheSourceError::Identity))?
            .clone();
        let program_index = catalogs
            .programs
            .iter()
            .position(|program| {
                program.as_ref().is_some_and(|program| {
                    program.source == source_index && program.ordinal == ordinal
                })
            })
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let program = catalogs.programs[program_index]
            .as_ref()
            .expect("found program");
        if program.used
            || program.plan != actual.plan
            || current.offset != actual.plan.initial_frontier().2
            || current.tail_start != actual.plan.initial_frontier().0
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let initial_length = actual.plan.initial_frontier().1;
        let initial_tail_bytes = match (initial_length, actual.tail_dtypes) {
            (0, None) => 0,
            (length, Some(dtypes)) if length > 0 => {
                pair_bytes(current, i64::from(length), dtypes).map_err(fail)?
            }
            _ => return Err(fail(CacheSourceError::Geometry)),
        };
        source
            .roles
            .borrow_mut()
            .active
            .as_mut()
            .expect("authenticated active role")
            .append_active = true;
        let mut program = catalogs.programs[program_index]
            .take()
            .expect("one program checkout");
        program.used = true;
        // Release the source RefCell loan before any native callback. Checkout
        // restores the same spent program on failure or unwind, never Ready.
        drop(catalogs_loan);
        let mut checkout = Checkout {
            source: Rc::clone(&source),
            index: program_index,
            program: Some(program),
            succeeded: false,
        };
        let custody = source
            .roles
            .try_borrow()
            .map_err(|_| fail(CacheSourceError::Busy))?
            .host_source_custody(&observer, ordinal)
            .map_err(fail)?;
        let proof = super::scan_claim::OriginalPagedScanSource {
            source: retained_source,
            installed: &installed,
            observer: observer.clone(),
            context: &source.context,
            custody: custody.clone(),
        };
        let mut host = super::host_program::HostCheckout::take(&source, &proof)?;
        let mut claim = OriginalPagedAppendClaim {
            host: host.value.as_mut(),
            bank: &source,
            custody,
            source: retained_source,
            installed: &installed,
            program: checkout.program.as_mut().expect("checked out program"),
            observer: observer.clone(),
            roots,
            retained: 0,
            published: 0,
            step: 0,
            tail_ready: false,
            tail_cleared: false,
            initial_tail_bytes,
            prepared_tail_bytes: None,
            context: &source.context,
        };
        let result = run(&mut claim).and_then(|output| {
            claim.finish()?;
            Ok(output)
        });
        drop(claim);
        checkout.succeeded = result.is_ok();
        result
    }
}
/// Common allocation-free exact role lookup for append and scan consumers.
pub(super) struct CurrentRole {
    pub(super) source: Rc<PagedSourceBank>,
    pub(super) observer: OriginalScopeObserver,
    pub(super) ordinal: usize,
    pub(super) roots: TransientRootsProjection,
}
pub(super) fn current_role() -> Result<CurrentRole, Exception> {
    let observer = OriginalScopeObserver::require_current()?;
    let source = CURRENT
        .with(|slot| slot.try_borrow().ok().and_then(|slot| slot.upgrade()))
        .ok_or_else(|| observer.invalid_input_error())?;
    let fail = |cause| failure(cause, &observer, source.context.metadata_funding());
    let (ordinal, roots) = {
        let roles = source
            .roles
            .try_borrow()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        let active = roles
            .active
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if roles.failed || active.append_active || !active.observer.same_scope(&observer) {
            return Err(fail(CacheSourceError::Identity));
        }
        (active.ordinal, active.roots.clone())
    };
    Ok(CurrentRole {
        source,
        observer,
        ordinal,
        roots,
    })
}
pub(super) struct Checkout {
    pub(super) source: Rc<PagedSourceBank>,
    pub(super) index: usize,
    pub(super) program: Option<PagedAppendProgram>,
    pub(super) succeeded: bool,
}
impl Drop for Checkout {
    fn drop(&mut self) {
        let program = self.program.take().expect("one program restoration");
        let previous = {
            let mut loan = self.source.catalogs.borrow_mut();
            loan.as_mut().expect("accepted catalogs").programs[self.index].replace(program)
        };
        debug_assert!(previous.is_none());
        drop(previous);
        let mut roles = self.source.roles.borrow_mut();
        if let Some(active) = &mut roles.active {
            active.append_active = false;
        }
        if !self.succeeded {
            roles.failed = true;
        }
    }
}
impl OriginalPagedAppendClaim<'_> {
    /// Error transport alone grants neither source nor mutation permission.
    pub(crate) fn input_error(
        observer: &OriginalScopeObserver,
        cause: CacheSourceError,
    ) -> Exception {
        failure(cause, observer, None)
    }
    pub(crate) fn plan(&self) -> PagedAppendPlan {
        self.program.plan
    }
    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        self.source.manager()
    }
    pub(crate) fn layer(&self) -> usize {
        self.source.geometry().global_layer
    }
    pub(crate) fn geometry(&self) -> WorkspacePagedGeometry {
        self.source.append_geometry()
    }
    pub(crate) fn observer(&self) -> &OriginalScopeObserver {
        &self.observer
    }
    pub(crate) fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        let current = OriginalScopeObserver::require_current()?;
        if !manager.same_catalog(self.installed.manager())
            || generation != self.installed.initial_generation()
            || !current.same_scope(&self.observer)
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn validate_retained_storage(
        &self,
        id: &CacheBlockId,
        phase: eredu_runtime::CacheStoragePhase,
        file: Option<&eredu_runtime::cache::LiveCacheBlockSource>,
    ) -> Result<(), Exception> {
        if phase == eredu_runtime::CacheStoragePhase::DiskReady {
            let same_file = file.is_some_and(|file| {
                self.source
                    .retained_file(id)
                    .is_some_and(|source| source.same_source(file))
                    || self
                        .host
                        .as_deref()
                        .is_some_and(|host| host.retains_file(self.source, id, file, self.bank))
            });
            return if same_file {
                Ok(())
            } else {
                Err(self.error(CacheSourceError::Identity))
            };
        }
        if self
            .host
            .as_deref()
            .is_some_and(|host| host.contains(self.source, id, self.bank))
            && matches!(
                phase,
                eredu_runtime::CacheStoragePhase::HostUnbacked
                    | eredu_runtime::CacheStoragePhase::HostBacked
            )
        {
            return Ok(());
        }
        if self
            .source
            .geometry()
            .blocks
            .iter()
            .any(|block| &block.id == id && block.phase == phase)
            && self.source.has_host_promotion(id)
        {
            Ok(())
        } else {
            Err(self.error(CacheSourceError::PromotionRequired))
        }
    }
    pub(crate) fn rebalance_host(
        &mut self,
        additional: u64,
        replacement_tail: Option<u64>,
        required: Option<&CacheBlockId>,
        stream: &safemlx::Stream,
    ) -> Result<(), Exception> {
        let proof = super::scan_claim::OriginalPagedScanSource {
            source: self.source,
            installed: self.installed,
            observer: self.observer.clone(),
            context: self.context,
            custody: self.custody.clone(),
        };
        if let Some(host) = self.host.as_deref_mut() {
            host.rebalance(
                self.bank,
                &proof,
                self.program.ordinal,
                required,
                additional,
                replacement_tail.map(|bytes| (self.source.geometry().global_layer, bytes)),
                false,
                &self.roots,
                stream,
            )?;
        }
        Ok(())
    }
    pub(crate) fn error(&self, cause: impl Into<PagedMutationCause>) -> Exception {
        failure(cause, &self.observer, self.context.metadata_funding())
    }
    /// Retain before the next fallible native or manager operation. On an
    /// unexpected append failure the actual root remains in this Q-owned bank.
    pub(crate) fn retain(&mut self, value: Array) -> Result<Array, Exception> {
        let maximum = self
            .program
            .plan
            .steps()
            .count()
            .checked_mul(4)
            .ok_or_else(|| self.error(CacheSourceError::Overflow))?;
        let result = if self.retained == maximum {
            Err(self.error(CacheSourceError::Identity))
        } else {
            self.roots.append(&value).map_err(|cause| self.error(cause))
        };
        if let Err(cause) = result {
            self.program.failed_root = Some(value);
            return Err(cause);
        }
        self.retained += 1;
        Ok(value)
    }
    /// Bind exact produced shape/dtypes before publishing its occupancy. The
    /// shared concatenation may promote the input and previous tail; only its
    /// actual outputs determine this step's native representation.
    pub(crate) fn prepare_tail(&mut self, arrays: [&Array; 2]) -> Result<(), Exception> {
        let step = self
            .program
            .plan
            .steps()
            .nth(self.step)
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        if self.prepared_tail_bytes.is_some() || self.tail_ready || self.tail_cleared {
            return Err(self.error(CacheSourceError::Identity));
        }
        let length = step.tail().end - step.tail().start;
        let geometry = self.geometry();
        let [batch, heads, width] = geometry.dimensions;
        let length_i32 =
            i32::try_from(length).map_err(|_| self.error(CacheSourceError::Geometry))?;
        if arrays[0].shape() != [batch, heads, length_i32, width]
            || arrays[1].shape()
                != [
                    batch,
                    heads,
                    length_i32,
                    if geometry.key_only { 1 } else { width },
                ]
        {
            return Err(self.error(CacheSourceError::Geometry));
        }
        let bytes = pair_bytes(geometry, length, [arrays[0].dtype(), arrays[1].dtype()])
            .map_err(|cause| self.error(cause))?;
        if (arrays[0].nbytes() as u64).checked_add(arrays[1].nbytes() as u64) != Some(bytes) {
            return Err(self.error(CacheSourceError::Geometry));
        }
        if !step.seals() {
            if self
                .program
                .scan
                .as_ref()
                .and_then(|scan| scan.tail.as_ref())
                .is_none()
            {
                return Err(self.error(CacheSourceError::Identity));
            }
            self.program
                .scan
                .as_mut()
                .expect("prepared scan")
                .tail
                .as_mut()
                .expect("partial tail")
                .fill(arrays, &self.observer)?;
        }
        self.prepared_tail_bytes = Some(bytes);
        Ok(())
    }
    pub(crate) fn validate_tail_update(
        &self,
        bytes: u64,
        end: i64,
        clearing: bool,
        restoring: bool,
    ) -> Result<(), Exception> {
        let step = self
            .program
            .plan
            .steps()
            .nth(self.step)
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        let range = step.tail();
        if (clearing && restoring)
            || end != range.end
            || (clearing && (!step.seals() || !self.tail_ready || self.tail_cleared))
            || (restoring && (!self.tail_ready || !self.tail_cleared))
            || (!clearing && !restoring && self.tail_ready)
            || bytes
                != if clearing {
                    0
                } else {
                    self.prepared_tail_bytes
                        .ok_or_else(|| self.error(CacheSourceError::Identity))?
                }
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn did_update_tail(&mut self, clearing: bool, restoring: bool) {
        if clearing {
            self.tail_cleared = true;
        } else if restoring {
            self.tail_cleared = false;
        } else if self
            .program
            .plan
            .steps()
            .nth(self.step)
            .expect("validated step")
            .seals()
        {
            self.tail_ready = true;
        } else {
            self.step += 1;
            self.prepared_tail_bytes = None;
        }
    }
    pub(crate) fn validate_rollback_tail(&self, bytes: u64, end: i64) -> Result<(), Exception> {
        let (_, _, offset) = self.program.plan.initial_frontier();
        if end != offset || bytes != self.initial_tail_bytes {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn take_publication(
        &mut self,
        id: &CacheBlockId,
        arrays: &CacheBlockArrays,
    ) -> Result<(CacheBlockMetadata, bool), Exception> {
        if !self.tail_ready || !self.tail_cleared {
            return Err(self.error(CacheSourceError::Identity));
        }
        let publication = self
            .program
            .publications
            .get(self.published)
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        if &publication.id != id {
            return Err(self.error(CacheSourceError::Identity));
        }
        publication
            .metadata
            .as_ref()
            .ok_or_else(|| self.error(CacheSourceError::Identity))?
            .validate_arrays(arrays)
            .map_err(|cause| self.error(cause))?;
        let publication = &mut self.program.publications[self.published];
        let protected = publication.protected_prefix;
        let metadata = publication
            .metadata
            .take()
            .expect("validated one-use metadata")
            .bind(arrays)
            .map_err(|cause| self.error(cause))?;
        Ok((metadata, protected))
    }
    /// Only the manager calls this after the canonical record is committed.
    pub(crate) fn published(&mut self, id: &CacheBlockId) -> Result<(), Exception> {
        if !self
            .program
            .publications
            .get(self.published)
            .is_some_and(|entry| &entry.id == id && entry.metadata.is_none())
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        self.published += 1;
        self.step += 1;
        self.tail_ready = false;
        self.tail_cleared = false;
        self.prepared_tail_bytes = None;
        Ok(())
    }
    pub(crate) fn publication_id(&self, index: usize) -> Option<&CacheBlockId> {
        self.program.publications.get(index).map(|value| &value.id)
    }
    pub(crate) fn published_count(&self) -> usize {
        self.published
    }
    fn validate_complete(&self) -> Result<(), Exception> {
        if self.published != self.program.publications.len()
            || self.step != self.program.plan.steps().count()
            || self.tail_ready
            || self.tail_cleared
            || self.prepared_tail_bytes.is_some()
            || Some(self.retained) != self.program.plan.steps().count().checked_mul(4)
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), Exception> {
        self.validate_complete()?;
        self.program.completed = true;
        Ok(())
    }
}
fn pair_bytes(
    geometry: WorkspacePagedGeometry,
    length: i64,
    dtypes: [Dtype; 2],
) -> Result<u64, CacheSourceError> {
    let length = i32::try_from(length).map_err(|_| CacheSourceError::Geometry)?;
    let [batch, heads, width] = geometry.dimensions;
    CacheBlockMetadata::floating_bytes(
        [
            &[batch, heads, length, width],
            &[
                batch,
                heads,
                length,
                if geometry.key_only { 1 } else { width },
            ],
        ],
        dtypes,
    )
}
pub(super) fn lookup_control_bytes() -> Option<usize> {
    Some(
        size_of::<RefCell<Weak<PagedSourceBank>>>()
            + size_of::<Weak<PagedSourceBank>>()
            + size_of::<Option<Rc<PagedSourceBank>>>(),
    )
}
pub(super) fn control_bytes(steps: usize) -> Option<usize> {
    let frames = [
        size_of::<OriginalPagedAppendClaim<'_>>(),
        size_of::<Checkout>(),
        size_of::<CurrentRole>(),
        size_of::<InstalledManagerCatalog>(),
        size_of::<PagedAppendInput<'_>>(),
        size_of::<TransientRootsProjection>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<std::cell::RefMut<'_, Option<super::catalogs::PreparedPagedCatalogs>>>(),
        size_of::<std::cell::Ref<'_, super::roles::RoleState>>(),
        size_of::<std::cell::Ref<'_, Option<super::catalogs::PreparedPagedCatalogs>>>(),
        size_of::<std::slice::Iter<'_, Option<PagedAppendProgram>>>(),
        size_of::<(&OriginalScopeObserver, &Rc<PagedSourceBank>)>(),
        size_of::<Option<eredu_runtime::cache::PagedAppendStep>>(),
        size_of::<eredu_runtime::cache::PagedAppendSteps>(),
        size_of::<std::ops::Range<i64>>(),
        size_of::<([i64; 4], u64, bool, bool)>(),
        size_of::<(usize, usize, bool)>(),
        size_of::<(WorkspacePagedGeometry, i64, [Dtype; 2])>(),
        size_of::<[[i32; 4]; 2]>(),
        size_of::<(Option<[Dtype; 2]>, u64, Option<u64>)>(),
        CacheBlockMetadata::fixed_controls()?.checked_mul(steps.checked_add(1)?)?,
        size_of::<Result<(), Exception>>(),
        size_of::<Option<Array>>(),
        size_of::<Result<(CacheBlockMetadata, bool), Exception>>(),
        size_of::<Failure>(),
        size_of::<PagedMutationCause>(),
        ProjectedPagedSource::promotion_access_control_bytes()?,
        size_of::<(
            &OriginalPagedAppendClaim<'_>,
            &CacheBlockId,
            eredu_runtime::CacheStoragePhase,
            Option<&eredu_runtime::cache::LiveCacheBlockSource>,
        )>(),
        super::host_program::PreparedPagedHostProgram::retained_file_control_bytes(),
        Exception::retained_source_control_bytes::<Failure>()?,
    ];
    let retained = steps
        .checked_mul(4)?
        .checked_mul(size_of::<Result<Array, Exception>>())?;
    frames.into_iter().try_fold(
        std::mem::size_of_val(&frames).checked_add(retained)?,
        usize::checked_add,
    )
}

pub(super) fn failure_control_bytes() -> Option<usize> {
    size_of::<Failure>()
        .checked_add(size_of::<PagedMutationCause>())?
        .checked_add(Exception::retained_source_control_bytes::<Failure>()?)?
        .checked_add(size_of::<(
            &OriginalScopeObserver,
            Option<HostMetadataFunding>,
        )>())
}
