//! Finite publication metadata follows the same actual append partition.
use super::*;
use crate::backend::runtime::cache::{
    kv::ProjectedPagedSource,
    residency::{
        CacheResidencyManager, CacheSourceError, CacheSourceFailure, PreparedFloatingBlockMetadata,
        PreparedManagerCatalog,
    },
};
use eredu_core::cache::{CacheBlockId, CacheRepresentation};
use eredu_runtime::{
    cache::{PagedAppendPlan, PagedAppendSteps},
    working_memory::InferenceSpanWorkspacePlan,
};
use std::mem::size_of;

/// One actual source/layer occurrence, indexed by the retained generation role.
/// No native mutation permission is implied until that role claims the program.
pub(super) struct PagedAppendProgram {
    pub(super) source: usize,
    pub(super) ordinal: usize,
    pub(super) plan: PagedAppendPlan,
    pub(super) reporting_controls: usize,
    pub(super) publications: Vec<PagedBlockPublication>,
    pub(super) used: bool,
    pub(super) completed: bool,
    pub(super) scan: Option<super::scan_program::PreparedPagedScan>,
    pub(super) visible: Option<super::visible_program::PreparedPagedVisible>,
    pub(super) failed_root: Option<safemlx::Array>,
    _funding: Option<HostMetadataFunding>,
}
pub(super) struct PagedBlockPublication {
    pub(super) id: CacheBlockId,
    pub(super) protected_prefix: bool,
    pub(super) metadata: Option<PreparedFloatingBlockMetadata>,
}

pub(super) fn prepare(
    sources: &[ProjectedPagedSource],
    plan: &InferenceSpanWorkspacePlan,
    catalogs: &[Option<PreparedManagerCatalog>],
    context: &WorkspaceContext,
) -> Result<Vec<Option<PagedAppendProgram>>, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let unique = sources
        .iter()
        .enumerate()
        .filter(|(index, source)| !alias_before(sources, *index, source))
        .count();
    let count = unique
        .checked_mul(
            plan.generation_forward_count()
                .ok_or_else(|| fail(CacheSourceError::Identity))?,
        )
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    let frames = [
        size_of::<(
            &[ProjectedPagedSource],
            &InferenceSpanWorkspacePlan,
            &[Option<PreparedManagerCatalog>],
            &WorkspaceContext,
        )>(),
        size_of::<Vec<Option<PagedAppendProgram>>>(),
        size_of::<Result<Vec<Option<PagedAppendProgram>>, CacheSourceFailure>>(),
        size_of::<(
            &mut Vec<Option<PagedAppendProgram>>,
            &mut Option<CacheSourceFailure>,
            &ProjectedPagedSource,
            &WorkspaceContext,
            usize,
        )>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<Option<CacheSourceFailure>>(),
        size_of::<(usize, &[Option<PreparedManagerCatalog>])>(),
        std::mem::size_of_val(&plan.generation_records()),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, ProjectedPagedSource>>>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let mut programs = context
        .metadata_vec(count)
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    for (index, source) in sources.iter().enumerate() {
        if alias_before(sources, index, source) {
            let previous = sources[..index]
                .iter()
                .find(|previous| same_layer(previous, source))
                .expect("found alias");
            if previous.append_geometry() != source.append_geometry() {
                return Err(fail(CacheSourceError::Geometry));
            }
            continue;
        }
        // Keep the original metadata error and its account; the scalar visitor
        // stops immediately without formatting or replacing its source.
        let reporting_controls = catalogs
            .iter()
            .flatten()
            .find(|catalog| catalog.manager().same_catalog(source.manager()))
            .ok_or_else(|| fail(CacheSourceError::Identity))?
            .publication_control_bytes();
        let mut failure = None;
        let result =
            catalogs::visit_append_plans(source, plan, |ordinal, append| {
                match prepare_one(source, index, ordinal, append, reporting_controls, context) {
                    Ok(program) => {
                        programs.push(Some(program));
                        Ok(())
                    }
                    Err(cause) => {
                        failure = Some(cause);
                        Err(CacheSourceError::Geometry)
                    }
                }
            });
        if let Some(cause) = failure {
            return Err(cause);
        }
        result.map_err(fail)?;
    }
    debug_assert_eq!(programs.len(), count);
    super::scan_program::prepare_all(&mut programs, sources, context)?;
    super::visible_program::prepare_all(&mut programs, sources, context)?;
    Ok(programs)
}
fn same_layer(first: &ProjectedPagedSource, second: &ProjectedPagedSource) -> bool {
    first.manager().same_catalog(second.manager())
        && first.geometry().global_layer == second.geometry().global_layer
}
fn alias_before(
    sources: &[ProjectedPagedSource],
    index: usize,
    source: &ProjectedPagedSource,
) -> bool {
    sources[..index]
        .iter()
        .any(|previous| same_layer(previous, source))
}
fn prepare_one(
    source: &ProjectedPagedSource,
    index: usize,
    ordinal: usize,
    plan: PagedAppendPlan,
    reporting_controls: usize,
    context: &WorkspaceContext,
) -> Result<PagedAppendProgram, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let frames = [
        size_of::<PagedAppendProgram>(),
        size_of::<PagedBlockPublication>(),
        size_of::<PagedAppendSteps>(),
        size_of::<Vec<PagedBlockPublication>>(),
        size_of::<Result<PagedAppendProgram, CacheSourceFailure>>(),
        size_of::<(
            &ProjectedPagedSource,
            usize,
            usize,
            PagedAppendPlan,
            usize,
            &WorkspaceContext,
        )>(),
        size_of::<eredu_runtime::cache::PagedAppendStep>(),
        size_of::<std::ops::Range<i64>>(),
        size_of::<[i32; 3]>(),
        size_of::<eredu_runtime::working_memory::WorkspacePagedGeometry>(),
        size_of::<(i32, bool)>(),
        size_of::<[[i32; 4]; 2]>(),
        size_of::<CacheBlockId>(),
        size_of::<Result<PreparedFloatingBlockMetadata, CacheSourceFailure>>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    context
        .charge_metadata(
            super::append_claim::control_bytes(plan.steps().count())
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    context
        .charge_metadata(
            native_controls(plan, reporting_controls)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let geometry = source.append_geometry();
    let [batch, heads, width] = geometry.dimensions;
    let value_width = if geometry.key_only { 1 } else { width };
    let shapes = [
        [batch, heads, geometry.block_size, width],
        [batch, heads, geometry.block_size, value_width],
    ];
    let mut publications = context
        .metadata_vec(plan.sealed_blocks())
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    for step in plan.steps().filter(|step| step.seals()) {
        let tail = step.tail();
        let id = CacheBlockId {
            session_id: source.geometry().session_id,
            global_layer: source.geometry().global_layer,
            representation: CacheRepresentation::KeyValue,
            start: tail.start,
            end: tail.end,
            rank: source.geometry().rank,
        };
        let metadata = PreparedFloatingBlockMetadata::prepare(
            CacheRepresentation::KeyValue,
            [&shapes[0], &shapes[1]],
            context,
        )?;
        publications.push(PagedBlockPublication {
            id,
            protected_prefix: tail.end <= i64::from(geometry.prefix_tokens),
            metadata: Some(metadata),
        });
    }
    Ok(PagedAppendProgram {
        reporting_controls,
        source: index,
        ordinal,
        plan,
        publications,
        used: false,
        completed: false,
        scan: None,
        visible: None,
        failed_root: None,
        _funding: context.metadata_funding(),
    })
}

/// Actual successful transitions plus the one abort path of this one-use call.
/// Report storage itself belongs to the installed canonical manager catalog.
fn native_controls(plan: PagedAppendPlan, reporting: usize) -> Option<usize> {
    let partitions = plan.steps().count();
    let seals = plan.sealed_blocks();
    // Each partition publishes a tail. Each seal clears that tail and publishes
    // one block. The sole failed operation may restore its report, restore its
    // cleared tail, then roll back the append, with a possible failed restore.
    let publications = partitions.checked_add(seals.checked_mul(2)?)?;
    let abort_publications = 4usize;
    let reports = publications.checked_add(abort_publications)?;
    // Begin + each publication + one tail restoration + one outer rollback.
    // The rollback borrows again after dropping each committed native record.
    let manager_calls = publications.checked_add(3)?.checked_add(seals)?;
    let manager =
        CacheResidencyManager::original_append_control_bytes()?.checked_mul(manager_calls)?;
    let caller =
        crate::backend::runtime::cache::kv::PagedKeyValueCache::original_append_control_bytes()?;
    caller
        .checked_add(manager)?
        .checked_add(reporting.checked_mul(reports)?)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
