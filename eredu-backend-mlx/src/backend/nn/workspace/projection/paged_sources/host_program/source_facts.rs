//! The same Host-store source population, before a saved manager is copied.
use super::*;
use eredu_runtime::working_memory::{InferenceSpanWorkspacePlan, WorkingMemoryError};
use safemlx::PreparedHostTransferPlan;

impl ProjectedNativeStorage {
    /// Quote only the immutable traced constructor population. This neither
    /// reserves manager occupancy nor creates native sources. The actual copied
    /// manager must build the existing program and compare these same facts.
    pub(crate) fn paged_host_source_facts(
        &self,
        plan: &InferenceSpanWorkspacePlan,
        pool: &MemoryLedger,
        context: &WorkspaceContext,
    ) -> Result<Option<HostSourceConstructionFacts>, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let frames = [
            ProjectedPagedSource::retained_file_control_bytes(),
            size_of::<
                std::slice::Iter<'_, crate::backend::runtime::cache::kv::PagedCacheBlockGeometry>,
            >(),
            size_of::<(
                &Self,
                &InferenceSpanWorkspacePlan,
                &MemoryLedger,
                &WorkspaceContext,
            )>(),
            size_of::<Result<Option<HostSourceConstructionFacts>, CacheSourceFailure>>(),
            size_of::<Option<HostSourceConstructionFacts>>(),
            size_of::<std::slice::Iter<'_, ProjectedPagedSource>>(),
            size_of::<std::slice::Iter<'_, WorkspacePagedHostEntry>>(),
            size_of::<std::slice::Iter<'_, eredu_runtime::working_memory::WorkspacePagedHostLoad>>(
            ),
            size_of::<(&WorkspacePagedHostEntry, [[i32; 4]; 2], [Dtype; 2])>(),
            size_of::<(
                &ProjectedPagedSource,
                &InferenceSpanWorkspacePlan,
                &WorkspaceContext,
                &PreparedInputRuntime,
                &mut u64,
                &mut usize,
            )>(),
            size_of::<(
                &eredu_runtime::working_memory::WorkspacePagedHostLoad,
                &WorkspaceContext,
                &PreparedInputRuntime,
                &mut u64,
                &mut usize,
            )>(),
            size_of::<Result<HostSourceConstructionFacts, CacheSourceFailure>>(),
            size_of::<(u64, usize, [Dtype; 2], bool, bool)>(),
            size_of::<PreparedInputRuntime>(),
            size_of::<Result<PreparedInputRuntime, safemlx::InputAllocatorCause>>(),
            size_of::<Result<&safemlx::InitializedInputAllocator, WorkingMemoryError>>(),
            size_of::<Result<HostSourceConstructionFacts, WorkingMemoryError>>(),
            safemlx::InitializedInputAllocator::borrow_control_bytes()
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if !self.is_complete() {
            return Err(fail(CacheSourceError::Identity));
        }
        if !self
            .paged_sources
            .iter()
            .any(|source| source.host_trace().is_some())
        {
            return Ok(None);
        }
        let allocator = crate::backend::managed_memory::input_allocator::admitted_initializer(pool)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let runtime = allocator.try_borrow_runtime().map_err(|cause| {
            CacheSourceFailure::metadata(context.metadata_source(cause), context)
        })?;
        let mut bytes = 0u64;
        let mut attempts = 0usize;
        for source in &self.paged_sources {
            let Some(trace) = source.host_trace() else {
                continue;
            };
            trace
                .with_entries(|entries| -> Result<(), CacheSourceFailure> {
                    for entry in entries {
                        if !entry.requires_store()
                            || !selection::submitted(plan, entry.first_invocation(), context)?
                        {
                            continue;
                        }
                        for component in 0..2 {
                            let shape = entry
                                .shape(component)
                                .ok_or_else(|| fail(CacheSourceError::Geometry))?;
                            let dtype = dtype(
                                entry
                                    .floating_type(component)
                                    .ok_or_else(|| fail(CacheSourceError::Geometry))?,
                            );
                            if shape.len() != 4 || shape.iter().any(|n| *n <= 0) {
                                return Err(fail(CacheSourceError::Geometry));
                            }
                            let transfer = PreparedHostTransferPlan::new(&runtime, shape, dtype, 0)
                                .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
                            context
                                .charge_metadata(
                                    transfer
                                        .control_bytes()
                                        .and_then(|n| {
                                            n.checked_add(size_of::<(
                                                &WorkspacePagedHostEntry,
                                                usize,
                                                &[i32],
                                                Dtype,
                                                PreparedHostTransferPlan<'_>,
                                                Result<
                                                    PreparedHostTransferPlan<'_>,
                                                    safemlx::PreparedInputCause,
                                                >,
                                                Result<u64, WorkingMemoryError>,
                                            )>(
                                            ))
                                        })
                                        .ok_or_else(|| fail(CacheSourceError::Overflow))?,
                                )
                                .map_err(|cause| {
                                    CacheSourceFailure::metadata(cause.into(), context)
                                })?;
                            bytes = bytes
                                .checked_add(
                                    PreparedCacheHostDemotion::source_component_bytes(&transfer)
                                        .map_err(|cause| {
                                            CacheSourceFailure::metadata(
                                                context.metadata_source(cause),
                                                context,
                                            )
                                        })?,
                                )
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            attempts = attempts
                                .checked_add(1)
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                        }
                    }
                    Ok(())
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
            let writes_enabled = matches!(
                source.manager().options().live_disk_policy(),
                eredu_runtime::LiveCacheDiskPolicy::Enabled { .. }
            );
            {
                trace.with_loads(|loads| -> Result<(), CacheSourceFailure> {
                    for load in loads {
                        if !selection::submitted(plan, (load.query_start, load.context_end), context)? { continue; }
                        trace.with_entries(|entries| -> Result<(), CacheSourceFailure> {
                            let entry = entries.get(load.entry).ok_or_else(|| fail(CacheSourceError::Identity))?;
                            let retained_file = source.geometry().blocks.iter().any(|block| {
                                block.id.start == entry.range().start && block.id.end == entry.range().end
                                    && source.retained_file(&block.id).is_some()
                            });
                            if !writes_enabled && !retained_file { return Ok(()); }
                            let shapes = [
                                entry.shape(0).ok_or_else(|| fail(CacheSourceError::Geometry))?.try_into().map_err(|_| fail(CacheSourceError::Geometry))?,
                                entry.shape(1).ok_or_else(|| fail(CacheSourceError::Geometry))?.try_into().map_err(|_| fail(CacheSourceError::Geometry))?,
                            ];
                            let dtypes = [
                                dtype(entry.floating_type(0).ok_or_else(|| fail(CacheSourceError::Geometry))?),
                                dtype(entry.floating_type(1).ok_or_else(|| fail(CacheSourceError::Geometry))?),
                            ];
                            let facts = crate::backend::runtime::cache::residency::disk_read_source_facts(shapes, dtypes, &runtime, context)?;
                            bytes = bytes.checked_add(facts.capacity_bytes()).ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            attempts = attempts.checked_add(facts.maximum_attempts()).ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            Ok(())
                        }).map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
                    }
                    Ok(())
                }).map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
            }
        }
        if attempts == 0 {
            return Ok(None);
        }
        HostSourceConstructionFacts::new(bytes, attempts, 0)
            .map(Some)
            .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))
    }
}
