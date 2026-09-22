//! Canonical paged custody moves out of temporary native projection handles.
use super::*;
mod catalogs;
mod ordinary;
mod ordinary_scan;
pub(crate) use ordinary_scan::PreparedOrdinaryPagedScan;
pub(crate) use ordinary::{OrdinaryPagedProgram,OrdinaryPagedWork,OrdinaryPagedAppend,OrdinaryPagedCause,OrdinaryPagedHostScan};
mod host_program;
pub(crate) use host_program::PagedHostStoreDeclaration;
mod programs;
mod resume;
mod scan_claim;
mod scan_program;
mod visible_program;
pub(crate) use scan_claim::{
    OriginalPagedAttentionBlock, OriginalPagedBlockSource, OriginalPagedDiscard,
    OriginalPagedDiskWriteSource, OriginalPagedHostEviction, OriginalPagedHostReturn,
    OriginalPagedScanClaim, OriginalPagedScanSource, PagedScanInput,
};
mod append_claim;
pub(crate) use append_claim::{
    OriginalPagedAppendClaim, OriginalPagedVisibleClaim, PagedAppendInput, PagedMutationCause,
};
use catalogs::PreparedPagedCatalogs;
mod roles;
pub(crate) use roles::PagedScopeRetention;
use std::{cell::RefCell, rc::Rc};

/// Canonical page pins and exact source geometry retained by an accepted
/// candidate. It contains no auxiliary Array clones or execution authority.
/// The vector backing retires before its independent planning account.
#[derive(Clone)]
pub(crate) struct ProjectedPagedSources {
    inner: Rc<PagedSourceBank>,
}
struct PagedSourceBank {
    host: RefCell<Option<host_program::PreparedPagedHostProgram>>,
    // A failed native append root retires before the canonical source pins.
    catalogs: RefCell<Option<PreparedPagedCatalogs>>,
    sources: Vec<crate::backend::runtime::cache::kv::ProjectedPagedSource>,
    roles: RefCell<roles::RoleState>,
    context: WorkspaceContext,
    _funding: Option<HostMetadataFunding>,
}
impl std::fmt::Debug for ProjectedPagedSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectedPagedSources")
            .field("sources", &self.inner.sources.len())
            .finish_non_exhaustive()
    }
}
impl ProjectedPagedSources {
    pub(crate) fn sources(&self) -> &[crate::backend::runtime::cache::kv::ProjectedPagedSource] {
        &self.inner.sources
    }
}
impl ProjectedNativeStorage {
    /// Transfers only canonical pins after native/portable root registration.
    /// The temporary Array inventory stays here and follows its existing
    /// retirement boundary. No source is rebuilt from shape or byte counts.
    pub(crate) fn take_paged_sources(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<Option<ProjectedPagedSources>, Error> {
        let frames = [
            std::mem::size_of::<ProjectedPagedSources>(),
            std::mem::size_of::<PagedSourceBank>(),
            std::mem::size_of::<
                Result<Rc<PagedSourceBank>, eredu_nn::workspace::WorkspaceMetadataError>,
            >(),
            std::mem::size_of::<Option<ProjectedPagedSources>>(),
            std::mem::size_of::<Result<Option<ProjectedPagedSources>, Error>>(),
            std::mem::size_of::<(&mut Self, &WorkspaceContext)>(),
            std::mem::size_of::<Vec<crate::backend::runtime::cache::kv::ProjectedPagedSource>>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(bytes)?;
        if self.paged_sources.is_empty() {
            return Ok(None);
        }
        Ok(Some(ProjectedPagedSources {
            inner: context.metadata_rc(PagedSourceBank {
                sources: std::mem::take(&mut self.paged_sources),
                catalogs: RefCell::new(None),
                host: RefCell::new(None),
                roles: RefCell::new(roles::RoleState::new()),
                context: context.clone(),
                _funding: self._funding.clone(),
            })?,
        }))
    }
}
