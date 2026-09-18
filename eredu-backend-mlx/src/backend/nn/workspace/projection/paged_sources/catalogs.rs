//! Canonical manager destinations derived from the exact retained request.
use super::*;
use crate::backend::runtime::cache::{
    kv::ProjectedPagedSource,
    residency::{
        CacheBlockSourceLoan, CacheSourceError, CacheSourceFailure, InstalledManagerCatalog,
        PreparedManagerCatalog,
    },
};
use eredu_runtime::{
    cache::PagedAppendPlan,
    working_memory::{InferenceSpanWorkspacePlan, InferenceWorkspaceSpan},
};

/// Immutable source preparation. Native mutation must still consume the exact
/// accepted role and operation schedule; table capacity is not that authority.
pub(super) struct PreparedPagedCatalogs {
    pub(super) plan: InferenceSpanWorkspacePlan,
    pub(super) entries: Vec<Option<PreparedManagerCatalog>>,
    pub(super) installed: Vec<InstalledManagerCatalog>,
    pub(super) programs: Vec<Option<super::programs::PagedAppendProgram>>,
    _funding: Option<HostMetadataFunding>,
}
impl ProjectedPagedSources {
    /// Prepares both authoritative catalogs once, before accepted execution.
    /// Managers are deduplicated by retained Arc identity; physical layer IDs
    /// are never replaced by their dense position in this source inventory.
    pub(crate) fn prepare_catalogs(
        &mut self,
        plan: &InferenceSpanWorkspacePlan,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        let failure = |cause| CacheSourceFailure::source(cause, context);
        let frames = [
            size_of::<PreparedPagedCatalogs>(),
            ProjectedPagedSource::original_program_validation_control_bytes(),
            size_of::<Result<PreparedPagedCatalogs, CacheSourceFailure>>(),
            size_of::<(&mut Self, &InferenceSpanWorkspacePlan, &WorkspaceContext)>(),
            size_of::<Vec<Option<PreparedManagerCatalog>>>(),
            size_of::<CatalogWork<'_>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, ProjectedPagedSource>>>(),
            size_of::<(&[ProjectedPagedSource], &ProjectedPagedSource)>(),
            size_of::<(usize, usize, usize)>(),
            size_of::<Result<(), CacheSourceFailure>>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| failure(CacheSourceError::Overflow))?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if self
            .inner
            .catalogs
            .try_borrow()
            .map_err(|_| failure(CacheSourceError::Busy))?
            .is_some()
            || plan.generation_records().is_none()
        {
            return Err(failure(CacheSourceError::Identity));
        }
        for source in &self.inner.sources {
            source.validate_original_program().map_err(failure)?;
        }
        let managers = self
            .inner
            .sources
            .iter()
            .enumerate()
            .filter(|(index, source)| {
                !self.inner.sources[..*index]
                    .iter()
                    .any(|previous| source.manager().same_catalog(previous.manager()))
            })
            .count();
        let mut entries = context
            .metadata_vec(managers)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for (index, source) in self.inner.sources.iter().enumerate() {
            if self.inner.sources[..index]
                .iter()
                .any(|previous| source.manager().same_catalog(previous.manager()))
            {
                continue;
            }
            let work = CatalogWork {
                sources: &self.inner.sources,
                first: source,
                plan,
                context,
            };
            let catalog =
                source
                    .manager()
                    .with_source_loan(source.selection(), context, move |loan| {
                        work.prepare(loan)
                    })?;
            entries.push(Some(catalog));
        }
        context
            .charge_metadata(
                roles::control_bytes(
                    plan.generation_forward_count()
                        .ok_or_else(|| failure(CacheSourceError::Identity))?,
                )
                .ok_or_else(|| failure(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let installed = context
            .metadata_vec(managers)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let programs = super::programs::prepare(&self.inner.sources, plan, &entries, context)?;
        *self
            .inner
            .catalogs
            .try_borrow_mut()
            .map_err(|_| failure(CacheSourceError::Busy))? = Some(PreparedPagedCatalogs {
            plan: plan.clone(),
            entries,
            installed,
            programs,
            _funding: context.metadata_funding(),
        });
        Ok(())
    }

    /// Exact retained traversal check before the role bank takes these
    /// destinations. This remains a source check, not execution permission.
    pub(crate) fn catalogs_match(&self, plan: &InferenceSpanWorkspacePlan) -> bool {
        self.inner
            .catalogs
            .try_borrow()
            .ok()
            .is_some_and(|catalogs| {
                catalogs
                    .as_ref()
                    .is_some_and(|catalogs| catalogs.plan.same_plan(plan))
            })
    }
}

struct CatalogWork<'a> {
    sources: &'a [ProjectedPagedSource],
    first: &'a ProjectedPagedSource,
    plan: &'a InferenceSpanWorkspacePlan,
    context: &'a WorkspaceContext,
}
impl CatalogWork<'_> {
    fn prepare(
        self,
        mut loan: CacheBlockSourceLoan<'_>,
    ) -> Result<PreparedManagerCatalog, CacheSourceFailure> {
        let failure = |cause| CacheSourceFailure::source(cause, self.context);
        let frames = [
            size_of::<Self>(),
            size_of::<CacheBlockSourceLoan<'_>>(),
            size_of::<Result<PreparedManagerCatalog, CacheSourceFailure>>(),
            size_of::<(usize, usize, usize, usize)>(),
            size_of::<(&ProjectedPagedSource, &InferenceSpanWorkspacePlan)>(),
            size_of::<PagedAppendPlan>(),
            size_of::<Result<PagedAppendPlan, eredu_runtime::cache::PagedAppendError>>(),
            size_of::<(i32, i32, i64, i64, u64, u64)>(),
            size_of::<Result<usize, CacheSourceError>>(),
            std::mem::size_of_val(&self.plan.generation_records()),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, ProjectedPagedSource>>>(),
            ProjectedPagedSource::catalog_validation_control_bytes()
                .ok_or_else(|| failure(CacheSourceError::Overflow))?,
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| failure(CacheSourceError::Overflow))?;
        self.context
            .charge_metadata(bytes)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), self.context))?;
        let (mut blocks, mut tails) = loan.catalog_population();
        for (index, source) in self.sources.iter().enumerate() {
            if !source.manager().same_catalog(self.first.manager()) {
                continue;
            }
            let selected = loan.selected(source.selection());
            source.validate_catalog(&selected).map_err(failure)?;
            // A repeated physical layer aliases the same canonical tail. Its
            // exact source is still validated above, but it needs one slot.
            if self.sources[..index].iter().any(|previous| {
                source.manager().same_catalog(previous.manager())
                    && source.geometry().global_layer == previous.geometry().global_layer
            }) {
                continue;
            }
            blocks = blocks
                .checked_add(source_growth(source, self.plan).map_err(failure)?)
                .ok_or_else(|| failure(CacheSourceError::Overflow))?;
            tails = tails
                .checked_add(usize::from(
                    selected.tail().is_none() && self.plan.generation_forward_count() != Some(0),
                ))
                .ok_or_else(|| failure(CacheSourceError::Overflow))?;
        }
        loan.prepare_catalog_for_layers(
            blocks,
            tails,
            self.sources
                .iter()
                .filter(|source| source.manager().same_catalog(self.first.manager()))
                .map(|source| source.geometry().global_layer),
            self.context,
        )
    }
}

fn source_growth(
    source: &ProjectedPagedSource,
    plan: &InferenceSpanWorkspacePlan,
) -> Result<usize, CacheSourceError> {
    let mut blocks = 0usize;
    visit_append_plans(source, plan, |_, append| {
        blocks = blocks
            .checked_add(append.sealed_blocks())
            .ok_or(CacheSourceError::Overflow)?;
        Ok(())
    })?;
    Ok(blocks)
}

/// One exact source/frontier traversal shared by slot sizing and publication
/// preparation; callback ordinal is the actual generation-record index.
pub(super) fn visit_append_plans(
    source: &ProjectedPagedSource,
    plan: &InferenceSpanWorkspacePlan,
    mut visit: impl FnMut(usize, PagedAppendPlan) -> Result<(), CacheSourceError>,
) -> Result<(), CacheSourceError> {
    let geometry = source.geometry();
    let request = plan.geometry();
    if u64::try_from(geometry.offset).ok() != Some(request.cached_positions) {
        return Err(CacheSourceError::Identity);
    }
    let mut offset = geometry.offset;
    let mut tail_start = geometry.tail_start;
    let mut tail_len = i32::try_from(
        offset
            .checked_sub(tail_start)
            .ok_or(CacheSourceError::Overflow)?,
    )
    .map_err(|_| CacheSourceError::Geometry)?;
    for (ordinal, record) in plan
        .generation_records()
        .ok_or(CacheSourceError::Identity)?
        .enumerate()
    {
        let (position, input) = match record.span() {
            InferenceWorkspaceSpan::Sampling(_) => return Err(CacheSourceError::Identity),
            InferenceWorkspaceSpan::Prefill(chunk) => (
                chunk.position,
                chunk
                    .input
                    .end
                    .checked_sub(chunk.input.start)
                    .ok_or(CacheSourceError::Geometry)?,
            ),
            InferenceWorkspaceSpan::Decode { position, .. } => (*position, 1),
        };
        if u64::try_from(offset).ok() != Some(position) {
            return Err(CacheSourceError::Identity);
        }
        let append = PagedAppendPlan::new(
            source.manager().options().block_size_tokens(),
            i32::try_from(input).map_err(|_| CacheSourceError::Overflow)?,
            tail_len,
            tail_start,
            offset,
        )
        .map_err(CacheSourceError::from)?;
        visit(ordinal, append)?;
        (tail_start, tail_len, offset) = append.resulting_tail();
    }
    Ok(())
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
