//! Exact source/future-publication binding for the finite Host transfer trace.
use super::*;
#[path = "host_program/consumer.rs"]
mod consumer;
#[path = "host_program/disk.rs"]
mod disk;
#[path = "host_program/selection.rs"]
mod selection;
#[path = "host_program/source_facts.rs"]
mod source_facts;
use crate::backend::runtime::cache::{
    kv::ProjectedPagedSource,
    residency::{
        CacheBlockSourceLoan, CacheSourceError, CacheSourceFailure, PreparedCacheHostDemotion,
        PreparedCacheHostPromotion, PreparedCacheHostPromotionSlots,
    },
};
pub(super) use consumer::HostCheckout;
use eredu_core::cache::CacheBlockId;
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, OriginalHostSourceBank, OriginalHostSourceCustody,
    WorkingMemoryPool, WorkspacePagedHostEntry,
};
use safemlx::{AllocationInfo, Dtype, PreparedInputRuntime};
use std::mem::size_of;

/// Construction is private to the exact source/program matcher below. This
/// describes a retained row or a future publication, never an arbitrary ID.
pub(crate) struct PagedHostStoreDeclaration<'a> {
    source: &'a ProjectedPagedSource,
    id: &'a CacheBlockId,
    retained: bool,
    requires_store: bool,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    context: &'a WorkspaceContext,
}
impl PagedHostStoreDeclaration<'_> {
    pub(crate) fn id(&self) -> &CacheBlockId {
        self.id
    }
    pub(crate) fn is_retained(&self) -> bool {
        self.retained
    }
    pub(crate) fn requires_store(&self) -> bool {
        self.requires_store
    }
    pub(crate) fn shapes(&self) -> [[i32; 4]; 2] {
        self.shapes
    }
    pub(crate) fn dtypes(&self) -> [Dtype; 2] {
        self.dtypes
    }
    pub(crate) fn validate_loan(
        &self,
        loan: &CacheBlockSourceLoan<'_>,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceError> {
        if !context.shares_trace(self.context)
            || !loan.manager().same_catalog(self.source.manager())
        {
            return Err(CacheSourceError::HostIdentity(
                "declaration context and manager comparison",
            ));
        }
        self.source
            .validate_catalog(loan)
            .map_err(|cause| match cause {
                CacheSourceError::Identity => {
                    CacheSourceError::HostIdentity("retained catalog comparison")
                }
                cause => cause,
            })?;
        if self.retained != loan.blocks().any(|row| row.id() == self.id) {
            return Err(CacheSourceError::HostIdentity(
                "retained declaration membership",
            ));
        }
        Ok(())
    }
}

struct StoreSlot {
    initial_disk: Option<crate::backend::runtime::cache::residency::PreparedInitialDiskReturn>,
    // Native destinations, events, replaced Device resources and their pool
    // reservation precede metadata/source custody in the enclosing owner.
    pair: super::scan_program::ScanSourcePair,
    mover: Option<PreparedCacheHostDemotion>,
    disk: Option<crate::backend::runtime::cache::residency::PreparedDiskWriteDestination>,
    write: Option<crate::backend::runtime::cache::residency::DiskWriteOperation>,
    pin: Option<crate::backend::runtime::cache::residency::PinnedCacheBlock>,
    current_load: Option<usize>,
    id: CacheBlockId,
    source: usize,
    entry: usize,
    first_ordinal: usize,
}
struct LoadSlot {
    read_operation: Option<crate::backend::runtime::cache::residency::DiskReadOperation>,
    disk_read: Option<crate::backend::runtime::cache::residency::PreparedDiskReadDestination>,
    promotion: Option<PreparedCacheHostPromotion>,
    slots: Option<PreparedCacheHostPromotionSlots>,
    consumed: bool,
    read: Option<consumer::HostRead>,
    store: usize,
    ordinal: usize,
    pass: usize,
}
pub(super) struct PreparedPagedHostProgram {
    stores: Vec<StoreSlot>,
    loads: Vec<LoadSlot>,
    disk_workers: Vec<disk::Worker>,
    runtime: PreparedInputRuntime,
    facts: Option<HostSourceConstructionFacts>,
    constructed: bool,
    context: WorkspaceContext,
    _funding: Option<HostMetadataFunding>,
}
fn dtype(value: eredu_nn::workspace::WorkspaceFloatingType) -> Dtype {
    match value {
        eredu_nn::workspace::WorkspaceFloatingType::Float32 => Dtype::Float32,
        eredu_nn::workspace::WorkspaceFloatingType::Float16 => Dtype::Float16,
        eredu_nn::workspace::WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
    }
}
fn declaration<'a>(
    source: &'a ProjectedPagedSource,
    source_index: usize,
    entry: &WorkspacePagedHostEntry,
    programs: &'a [Option<super::programs::PagedAppendProgram>],
    selected: &'a super::programs::PagedAppendProgram,
    context: &'a WorkspaceContext,
) -> Result<(PagedHostStoreDeclaration<'a>, usize), CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    context
        .charge_metadata(
            ProjectedPagedSource::catalog_validation_control_bytes()
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    context
        .charge_metadata(size_of::<(
            &ProjectedPagedSource,
            usize,
            &WorkspacePagedHostEntry,
            &[Option<super::programs::PagedAppendProgram>],
            &super::programs::PagedAppendProgram,
            &WorkspaceContext,
            PagedHostStoreDeclaration<'_>,
            [[i32; 4]; 2],
            [Dtype; 2],
            Result<(PagedHostStoreDeclaration<'_>, usize), CacheSourceFailure>,
            (i64, i64),
            (usize, bool),
            Option<&CacheBlockId>,
        )>())
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let program = selected;
    let range = entry.range();
    let (id, retained) = if let Some(block) = source
        .geometry()
        .blocks
        .iter()
        .find(|block| block.id.start == range.start && block.id.end == range.end)
    {
        if entry.requires_store()
            != (block.phase == eredu_runtime::CacheStoragePhase::Device
                && source.retained_file(&block.id).is_none())
        {
            return Err(fail(CacheSourceError::HostIdentity(
                "retained source tier selection",
            )));
        }
        (&block.id, true)
    } else {
        let id = programs
            .iter()
            .flatten()
            .filter(|prior| prior.source == source_index && prior.ordinal <= program.ordinal)
            .flat_map(|prior| &prior.publications)
            .find(|publication| {
                publication.id.start == range.start && publication.id.end == range.end
            })
            .map(|publication| &publication.id)
            .ok_or_else(|| {
                fail(CacheSourceError::HostIdentity(
                    "future publication selection",
                ))
            })?;
        if !entry.requires_store() {
            return Err(fail(CacheSourceError::HostIdentity(
                "future store qualification",
            )));
        }
        (id, false)
    };
    let shapes: [[i32; 4]; 2] = [
        entry
            .shape(0)
            .ok_or_else(|| fail(CacheSourceError::Geometry))?
            .try_into()
            .map_err(|_| fail(CacheSourceError::Geometry))?,
        entry
            .shape(1)
            .ok_or_else(|| fail(CacheSourceError::Geometry))?
            .try_into()
            .map_err(|_| fail(CacheSourceError::Geometry))?,
    ];
    let dtypes = [
        dtype(
            entry
                .floating_type(0)
                .ok_or_else(|| fail(CacheSourceError::Geometry))?,
        ),
        dtype(
            entry
                .floating_type(1)
                .ok_or_else(|| fail(CacheSourceError::Geometry))?,
        ),
    ];
    let [batch, heads, width] = source.append_geometry().dimensions;
    let tokens = i32::try_from(
        id.end
            .checked_sub(id.start)
            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
    )
    .map_err(|_| fail(CacheSourceError::Overflow))?;
    if tokens <= 0
        || shapes[0] != [batch, heads, tokens, width]
        || shapes[1]
            != [
                batch,
                heads,
                tokens,
                if source.append_geometry().key_only {
                    1
                } else {
                    width
                },
            ]
    {
        return Err(fail(CacheSourceError::Geometry));
    }
    if retained {
        let actual = source
            .geometry()
            .blocks
            .iter()
            .find(|block| &block.id == id)
            .expect("matched retained ID");
        if actual
            .arrays
            .iter()
            .enumerate()
            .any(|(index, array)| array.shape != shapes[index] || array.dtype != dtypes[index])
        {
            return Err(fail(CacheSourceError::Geometry));
        }
    }
    Ok((
        PagedHostStoreDeclaration {
            source,
            id,
            retained,
            requires_store: entry.requires_store(),
            shapes,
            dtypes,
            context,
        },
        program.ordinal,
    ))
}
impl PreparedPagedHostProgram {
    fn prepare(
        sources: &[ProjectedPagedSource],
        programs: &[Option<super::programs::PagedAppendProgram>],
        plan: &eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
    ) -> Result<Option<Self>, CacheSourceFailure> {
        if !sources.iter().any(|source| source.host_trace().is_some()) {
            return Ok(None);
        }
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let frames = [
            size_of::<Self>(),
            size_of::<StoreSlot>(),
            size_of::<LoadSlot>(),
            size_of::<Option<&super::programs::PagedAppendProgram>>(),
            size_of::<(
                &[ProjectedPagedSource],
                &[Option<super::programs::PagedAppendProgram>],
                &eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
                &WorkingMemoryPool,
                &WorkspaceContext,
            )>(),
            size_of::<Result<Option<Self>, CacheSourceFailure>>(),
            size_of::<PagedHostStoreDeclaration<'_>>(),
            size_of::<[[i32; 4]; 2]>(),
            size_of::<[Dtype; 2]>(),
            size_of::<Option<[AllocationInfo; 2]>>(),
            size_of::<(usize, usize, usize, u64)>(),
            size_of::<
                Result<
                    &safemlx::InitializedInputAllocator,
                    eredu_runtime::working_memory::WorkingMemoryError,
                >,
            >(),
            size_of::<Result<PreparedInputRuntime, safemlx::InputAllocatorCause>>(),
            size_of::<(
                &mut Self,
                Option<OriginalHostSourceBank>,
                &OriginalHostSourceCustody,
            )>(),
            size_of::<Option<PreparedCacheHostDemotion>>(),
            size_of::<std::slice::IterMut<'_, StoreSlot>>(),
            size_of::<std::slice::Iter<'_, LoadSlot>>(),
            size_of::<Result<(), CacheSourceFailure>>(),
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
        let allocator = crate::backend::managed_memory::input_allocator::admitted_initializer(pool)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let runtime = allocator.try_borrow_runtime().map_err(|cause| {
            CacheSourceFailure::metadata(context.metadata_source(cause), context)
        })?;
        let mut counts = (0usize, 0usize);
        for (source_index, source) in sources.iter().enumerate() {
            if let Some(trace) = source.host_trace() {
                trace
                    .with_entries(|entries| -> Result<(), CacheSourceFailure> {
                        for entry in entries {
                            if selection::program(
                                source_index,
                                entry.first_invocation(),
                                programs,
                                plan,
                                context,
                            )?
                            .is_some()
                            {
                                counts.0 = counts
                                    .0
                                    .checked_add(1)
                                    .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            }
                        }
                        Ok(())
                    })
                    .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
                trace
                    .with_loads(|loads| -> Result<(), CacheSourceFailure> {
                        for load in loads {
                            if selection::program(
                                source_index,
                                (load.query_start, load.context_end),
                                programs,
                                plan,
                                context,
                            )?
                            .is_some()
                            {
                                counts.1 = counts
                                    .1
                                    .checked_add(1)
                                    .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            }
                        }
                        Ok(())
                    })
                    .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
            }
        }
        let mut stores: Vec<StoreSlot> = context
            .metadata_vec(counts.0)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let mut loads: Vec<LoadSlot> = context
            .metadata_vec(counts.1)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let mut bytes = 0u64;
        let mut attempts = 0usize;
        for (source_index, source) in sources.iter().enumerate() {
            let Some(trace) = source.host_trace() else {
                continue;
            };
            trace
                .with_entries(|entries| -> Result<(), CacheSourceFailure> {
                    for (entry_index, entry) in entries.iter().enumerate() {
                        let Some(program) = selection::program(
                            source_index,
                            entry.first_invocation(),
                            programs,
                            plan,
                            context,
                        )?
                        else {
                            continue;
                        };
                        let (declaration, first_ordinal) =
                            declaration(source, source_index, entry, programs, program, context)?;
                        let mover = if entry.requires_store() {
                            let prepared = source
                                .manager()
                                .with_source_loan(source.selection(), context, |loan| {
                                    loan.prepare_declared_host_demotion(
                                        &declaration,
                                        &runtime,
                                        context,
                                    )
                                })
                                .map_err(|cause| {
                                    cause.at_host("declared Host demotion preparation")
                                })?;
                            bytes = bytes
                                .checked_add(prepared.source_storage_bytes())
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            attempts = attempts
                                .checked_add(2)
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            Some(prepared)
                        } else {
                            None
                        };
                        let disk = if source.retained_file(declaration.id()).is_none()
                            && matches!(
                                source.manager().options().live_disk_policy(),
                                eredu_runtime::LiveCacheDiskPolicy::Enabled { .. }
                            ) {
                            Some(source.manager().with_source_loan(
                                source.selection(),
                                context,
                                |loan| {
                                    loan.prepare_declared_disk_write(
                                        &declaration,
                                        &runtime,
                                        counts
                                            .0
                                            .checked_mul(3)
                                            .and_then(|n| n.checked_add(counts.1.checked_mul(3)?))
                                            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
                                        context,
                                    )
                                },
                            )?)
                        } else {
                            None
                        };
                        let initial_disk = if declaration.is_retained()
                            && source.retained_file(declaration.id()).is_some()
                            && source.geometry().blocks.iter().any(|block| {
                                &block.id == declaration.id()
                                    && block.phase == eredu_runtime::CacheStoragePhase::Device
                            }) {
                            Some(source.manager().with_source_loan(
                                source.selection(),
                                context,
                                |loan| {
                                    loan.prepare_initial_disk_return(
                                        &declaration,
                                        counts
                                            .0
                                            .checked_add(counts.1)
                                            .and_then(|n| n.checked_mul(3))
                                            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
                                        context,
                                    )
                                },
                            )?)
                        } else {
                            None
                        };
                        stores.push(StoreSlot {
                            initial_disk,
                            pair: super::scan_program::ScanSourcePair::prepare(context)?,
                            write: None,
                            pin: None,
                            current_load: None,
                            mover,
                            disk,
                            id: declaration.id().clone(),
                            source: source_index,
                            entry: entry_index,
                            first_ordinal,
                        });
                    }
                    Ok(())
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
            trace
                .with_loads(|entries| -> Result<(), CacheSourceFailure> {
                    for load in entries {
                        let Some(program) = selection::program(
                            source_index,
                            (load.query_start, load.context_end),
                            programs,
                            plan,
                            context,
                        )?
                        else {
                            continue;
                        };
                        let store = stores
                            .iter()
                            .position(|slot| {
                                slot.source == source_index && slot.entry == load.entry
                            })
                            .ok_or_else(|| {
                                fail(CacheSourceError::HostIdentity("selected load entry"))
                            })?;
                        if load.pass > 1 || program.ordinal < stores[store].first_ordinal {
                            return Err(fail(CacheSourceError::HostIdentity(
                                "load pass and order comparison",
                            )));
                        }
                        let (slots, disk_read) = trace
                            .with_entries(|entries| {
                                let entry = entries.get(load.entry).ok_or_else(|| {
                                    fail(CacheSourceError::HostIdentity(
                                        "load declaration selection",
                                    ))
                                })?;
                                let first = selection::program(
                                    source_index,
                                    entry.first_invocation(),
                                    programs,
                                    plan,
                                    context,
                                )?
                                .ok_or_else(|| {
                                    fail(CacheSourceError::HostIdentity(
                                        "selected load source invocation",
                                    ))
                                })?;
                                let (declaration, _) = declaration(
                                    source,
                                    source_index,
                                    entry,
                                    programs,
                                    first,
                                    context,
                                )?;
                                source.manager().with_source_loan(
                                    source.selection(),
                                    context,
                                    |loan| {
                                        let disk_layout = source
                                            .retained_file(declaration.id())
                                            .and_then(|file| file.writer_layout())
                                            .or_else(|| {
                                                stores[store]
                                                    .disk
                                                    .as_ref()
                                                    .map(|writer| writer.read_layout())
                                            });
                                        let reservations = if disk_layout.is_some() {
                                            counts
                                                .0
                                                .checked_add(counts.1)
                                                .and_then(|n| n.checked_mul(3))
                                        } else {
                                            counts.0.checked_add(counts.1)
                                        }
                                        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                                        let slots = loan.prepare_declared_host_promotion(
                                            &declaration,
                                            &runtime,
                                            reservations,
                                            context,
                                        )?;
                                        let disk_read = disk_layout
                                            .map(|layout| {
                                                loan.prepare_declared_disk_read(
                                                    &declaration,
                                                    layout,
                                                    &runtime,
                                                    reservations,
                                                    context,
                                                )
                                            })
                                            .transpose()?;
                                        Ok((slots, disk_read))
                                    },
                                )
                            })
                            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?
                            .map_err(|cause| {
                                cause.at_host("declared Host promotion preparation")
                            })?;
                        if let Some(read) = &disk_read {
                            let facts = read.source_facts().map_err(|cause| {
                                CacheSourceFailure::metadata(
                                    context.metadata_source(cause),
                                    context,
                                )
                            })?;
                            bytes = bytes
                                .checked_add(facts.capacity_bytes())
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                            attempts = attempts
                                .checked_add(facts.maximum_attempts())
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                        }
                        loads.push(LoadSlot {
                            read_operation: None,
                            disk_read,
                            promotion: None,
                            slots: Some(slots),
                            consumed: false,
                            read: None,
                            store,
                            ordinal: program.ordinal,
                            pass: load.pass,
                        });
                    }
                    Ok(())
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
        }
        let disk_workers = disk::prepare_workers(sources, &stores, &loads, context)?;
        let append_steps = programs
            .iter()
            .flatten()
            .try_fold(0usize, |n, program| {
                n.checked_add(program.plan.steps().count())
            })
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let actions = counts
            .0
            .checked_add(
                counts
                    .1
                    .checked_mul(2)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .and_then(|n| n.checked_add(append_steps.checked_mul(2)?))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        context
            .charge_metadata(
                consumer::control_bytes()
                    .and_then(|n| n.checked_mul(actions.checked_add(1)?))
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let facts = if attempts == 0 {
            None
        } else {
            Some(
                HostSourceConstructionFacts::new(bytes, attempts, 0).map_err(|cause| {
                    CacheSourceFailure::metadata(context.metadata_source(cause), context)
                })?,
            )
        };
        Ok(Some(Self {
            stores,
            loads,
            disk_workers,
            runtime,
            facts,
            constructed: false,
            context: context.clone(),
            _funding: context.metadata_funding(),
        }))
    }
    fn construct(
        &mut self,
        mut bank: Option<OriginalHostSourceBank>,
        custody: &OriginalHostSourceCustody,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, &self.context);
        if self.constructed {
            return Err(fail(CacheSourceError::HostIdentity(
                "once-only construction",
            )));
        }
        if let Some(facts) = self.facts {
            let bank = bank.as_ref().ok_or_else(|| {
                fail(CacheSourceError::HostIdentity(
                    "accepted source bank presence",
                ))
            })?;
            if !bank.belongs_to_source(custody)
                || bank.remaining_bytes() != facts.capacity_bytes()
                || bank.remaining_attempts() != facts.maximum_attempts()
            {
                return Err(fail(CacheSourceError::HostIdentity(
                    "accepted source account and population comparison",
                )));
            }
        } else if bank.is_some() {
            return Err(fail(CacheSourceError::HostIdentity(
                "empty source bank comparison",
            )));
        }
        self.constructed = true;
        disk::install_workers(&mut self.disk_workers, &self.context)?;
        for slot in &mut self.stores {
            if let Some(mover) = slot.mover.take() {
                slot.mover = Some(
                    mover
                        .construct(
                            &self.runtime,
                            bank.as_mut().expect("validated source component"),
                            custody,
                        )
                        .map_err(|cause| cause.at_host("Host destination construction"))?,
                );
            }
        }
        for load in &mut self.loads {
            if let Some(read) = &mut load.disk_read {
                read.construct(
                    &self.runtime,
                    bank.as_mut().expect("declared read source component"),
                    custody,
                )?;
            }
        }
        Ok(())
    }
}
impl ProjectedPagedSources {
    /// Bind exact retained/future IDs to the actual conditional workspace trace
    /// before comparison. This allocates no Host payload or Device tensor.
    pub(crate) fn prepare_host_program(
        &self,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        if !context.shares_trace(&self.inner.context)
            || self
                .inner
                .host
                .try_borrow()
                .map_err(|_| fail(CacheSourceError::Busy))?
                .is_some()
        {
            return Err(fail(CacheSourceError::HostIdentity(
                "preparation context and state comparison",
            )));
        }
        let catalogs = self
            .inner
            .catalogs
            .try_borrow()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        let catalogs = catalogs
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::HostIdentity("prepared catalog presence")))?;
        let prepared = PreparedPagedHostProgram::prepare(
            &self.inner.sources,
            &catalogs.programs,
            &catalogs.plan,
            pool,
            context,
        )?;
        *self
            .inner
            .host
            .try_borrow_mut()
            .map_err(|_| fail(CacheSourceError::Busy))? = prepared;
        Ok(())
    }
    pub(crate) fn host_source_facts(
        &self,
    ) -> Result<Option<HostSourceConstructionFacts>, CacheSourceFailure> {
        let host =
            self.inner.host.try_borrow().map_err(|_| {
                CacheSourceFailure::source(CacheSourceError::Busy, &self.inner.context)
            })?;
        Ok(host.as_ref().and_then(|host| host.facts))
    }
    /// Consume the actual accepted source component once. Destinations enter the
    /// retained source owner before any later constructor may fail.
    pub(crate) fn construct_host_program(
        &self,
        bank: Option<OriginalHostSourceBank>,
        custody: &OriginalHostSourceCustody,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, &self.inner.context);
        let mut host = self
            .inner
            .host
            .try_borrow_mut()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        match host.as_mut() {
            Some(host) => host.construct(bank, custody),
            None if bank.is_none() => Ok(()),
            None => Err(fail(CacheSourceError::HostIdentity(
                "missing prepared Host program",
            ))),
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
