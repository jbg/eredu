//! Independently admitted copies of the actual borrowed resident state.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    array_copy::{OriginalCopyLayoutBuilder, OriginalCopyPlan},
    nn::workspace::{MlxMetalWorkspaceMechanisms, ProjectionSourceLayout},
    runtime::residency::storage::{CopyPublicationLayout, RetainedStorage},
    submission_recovery::{Retention, Status},
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceContext, WorkspaceCopyPreparationLayoutBuilder,
    WorkspaceIsolatedCopyPlan,
};
use eredu_runtime::working_memory::{
    OriginalStorageSourcesLayout, RegisteredWorkspaceCopy, RegisteredWorkspaceStorageLayout,
    WorkingMemoryFundingScope, WorkspaceCopyAccountLayout, WorkspaceCopyLimits,
    WorkspaceCopyRetention,
};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
    rc::Rc,
};

mod source;
pub(crate) use source::{
    CompletedResidentSource, SourceBinding as OriginalResidentSourceBinding,
    bind_projected as bind_completed_resident_sources,
    bind_projected_unselected_with_priors as bind_unselected_completed_resident_source_priors,
    bind_projected_with_priors as bind_completed_resident_source_priors,
};

struct Work {
    roots: RefCell<Vec<Array>>,
    scope: RefCell<Option<WorkingMemoryFundingScope>>,
    discarded: Cell<bool>,
    healthy: Cell<bool>,
    _copy: WorkspaceCopyRetention,
    _host: HostPreparationAuthority,
}
/// No raw Rc escapes. The final shared shell retires before its H/Q owners.
#[derive(Clone)]
struct Owner(Option<Rc<Work>>);
impl Drop for Owner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
impl Owner {
    fn work(&self) -> &Work {
        self.0.as_deref().expect("live copy work")
    }
}
impl Retention for Owner {
    fn observe(&self, status: Status) {
        let work = self.work();
        work.healthy
            .set(status.settled && !status.failed && !status.blocked);
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if self.discarded.get() && self.healthy.get() {
            // Final closed Owner retirement follows native Recovery's probe
            // destruction (Node drops P before T). The lexical discard guard
            // also follows all provisional output/publication prefixes. No
            // native observer or destination still retains these new roots.
            let roots = std::mem::take(self.roots.get_mut());
            drop(roots);
            if let Some(scope) = self.scope.get_mut().take() {
                let _ = scope.certify();
            }
        }
        // Failed/blocked/unobservable work keeps the existing quarantine rule.
        // A healthy discarded copy has no surviving destination to publish.
    }
}
struct DiscardUnpublished(Option<Owner>);
impl Drop for DiscardUnpublished {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.work().discarded.set(true);
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{stage}: {cause}")]
struct Failure {
    stage: &'static str,
    source_rows: Option<source::BindingRows>,
    #[source]
    cause: Error,
    _host: HostPreparationAuthority,
}
fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}
fn reserve(funding: &HostMetadataFunding, bytes: usize) -> Result<(), Error> {
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)
}
fn paid(
    funding: &HostMetadataFunding,
    cause: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    Error::Neural(funding.metadata_source(cause))
}

/// H pays the finite source/host/control constructors. The existing registered
/// isolated-copy program alone supplies numerical demand, and the existing
/// native plan supplies its positive physical delta. Neither is a text grant.
pub(crate) fn copy(
    source: PreparedResidentDecoderCopy<'_>,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &safemlx::PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
    capacity: &eredu_core::MemoryLimits,
) -> Result<OriginalResidentState, Error> {
    copy_with_source(
        source,
        None,
        environment,
        initialized,
        mechanisms,
        funding,
        host,
        capacity,
    )
}

pub(crate) fn copy_with_source(
    source: PreparedResidentDecoderCopy<'_>,
    completed: Option<&CompletedResidentSource>,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &safemlx::PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
    capacity: &eredu_core::MemoryLimits,
) -> Result<OriginalResidentState, Error> {
    // Pay the enclosing typed source before any operation can fail with an
    // allocated diagnostic. A refusal here remains an inline funding failure.
    let frames = [
        size_of::<PreparedResidentDecoderCopy<'_>>(),
        size_of::<Option<&super::super::MlxKeyValueState>>(),
        size_of::<WorkspaceContext>(),
        size_of::<Result<WorkspaceContext, eredu_nn::workspace::WorkspaceMetadataError>>(),
        size_of::<Result<Option<super::super::MlxKeyValueState>, Error>>(),
        size_of::<Option<&CompletedResidentSource>>(),
        size_of::<Result<OriginalResidentState, Error>>(),
        size_of::<OriginalResidentState>(),
        size_of::<Result<OriginalResidentState, Error>>(),
        size_of::<Owner>(),
        size_of::<Work>(),
        size_of::<DiscardUnpublished>(),
        size_of::<Option<WorkingMemoryFundingScope>>(),
        size_of::<std::cell::RefMut<'_, Vec<Array>>>(),
        size_of::<std::cell::RefMut<'_, Option<WorkingMemoryFundingScope>>>(),
        size_of::<Failure>(),
        size_of::<Cell<&'static str>>(),
        size_of::<&Cell<&'static str>>(),
        size_of::<Cell<Option<source::BindingRows>>>(),
        size_of::<source::BindingDiagnostics<'_>>(),
        size_of::<Error>(),
        size_of::<Option<Error>>(),
        size_of::<OriginalCopyLayoutBuilder>(),
        size_of::<Option<OriginalCopyPlan<'_>>>(),
        size_of::<
            Result<Option<OriginalCopyPlan<'_>>, crate::backend::array_copy::OriginalCopyCause>,
        >(),
        size_of::<crate::backend::array_copy::OriginalCopyCause>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Box<dyn std::error::Error + Send + Sync>>(),
        BackendFailure::source_retention_peak_bytes::<Failure>()
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
    ];
    reserve(
        funding,
        frames.into_iter().try_fold(size_of_val(&frames), add)?,
    )?;
    let stage = Cell::new("source inspection");
    let source_rows = Cell::new(None);
    let result = copy_inner(
        source,
        completed,
        environment,
        initialized,
        mechanisms,
        funding,
        host,
        capacity,
        &stage,
        &source_rows,
    );
    result.map_err(|cause| {
        Error::StorageSource(
            BackendFailure::from_error(Failure {
                stage: stage.get(),
                source_rows: source_rows.get(),
                cause,
                _host: host.clone(),
            })
            .with_operation("copy original resident state"),
        )
    })
}
fn copy_inner(
    source: PreparedResidentDecoderCopy<'_>,
    completed: Option<&CompletedResidentSource>,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &safemlx::PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
    capacity: &eredu_core::MemoryLimits,
    stage: &Cell<&'static str>,
    source_rows: &Cell<Option<source::BindingRows>>,
) -> Result<OriginalResidentState, Error> {
    if let Some(live) = source.live_paged_key_value() {
        stage.set("live paged source copy");
        let context = WorkspaceContext::new_with_metadata_funding(mechanisms, funding.clone())
            .map_err(|cause| paid(funding, cause))?;
        return live
            .copy_original_paged(
                completed,
                environment,
                initialized,
                mechanisms,
                &context,
                host,
                capacity,
            )?
            .map(OriginalResidentState::KeyValue)
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch));
    }
    if let PreparedStorage::HybridGrouped(plan) = &source.storage {
        if plan.is_paged() {
            stage.set("grouped paged source copy");
            let context = WorkspaceContext::new_with_metadata_funding(mechanisms, funding.clone())
                .map_err(|cause| paid(funding, cause))?;
            return plan
                .copy_original_paged(
                    completed,
                    environment,
                    initialized,
                    mechanisms,
                    &context,
                    host,
                    capacity,
                )
                .map(OriginalResidentState::Hybrid);
        }
    }
    // Saved KV tables need their exact independent manager constructor. They
    // cannot enter the H-only resident table worker below.
    if source.is_paged() {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    let projection = SnapshotProjectionPlan::inspect(&source, None, None)
        .map_err(|cause| paid(funding, cause))?;
    let operands = projection.operand_count();
    let mut native = OriginalCopyLayoutBuilder::new();
    let mut retained = 0usize;
    let mut failure = None;
    source
        .visit_retained_arrays(&mut |array| {
            if failure.is_none() {
                failure = (|| {
                    retained = add(retained, 1)?;
                    native
                        .push_retained_source(array)
                        .map_err(|cause| paid(funding, cause))
                })()
                .err();
            }
        })
        .map_err(|cause| paid(funding, cause))?;
    source
        .visit_operands(&mut |array| {
            if failure.is_none() {
                failure = native
                    .push_operand(array)
                    .map_err(|cause| paid(funding, cause))
                    .err();
            }
        })
        .map_err(|cause| paid(funding, cause))?;
    if let Some(cause) = failure {
        return Err(cause);
    }
    let native = native
        .finish(environment)
        .map_err(|cause| paid(funding, cause))?;
    let mut layout = WorkspaceCopyPreparationLayoutBuilder::new();
    projection
        .visit_sources(&mut |array| {
            if failure.is_none() {
                failure = ProjectionSourceLayout::with_layout(array, |source| {
                    layout.push_source(source, &mechanisms)
                })
                .map_err(|cause| paid(funding, cause))
                .and_then(|result| result.map_err(|cause| paid(funding, cause)))
                .err();
            }
        })
        .map_err(|cause| paid(funding, cause))?;
    if let Some(cause) = failure {
        return Err(cause);
    }
    let layout = layout
        .finish(operands)
        .map_err(|cause| paid(funding, cause))?;
    let publication = CopyPublicationLayout::new(operands, 0).map_err(memory)?;
    let controls = [
        projection.requested_bytes(),
        layout.total_bytes(),
        publication.control_bytes(),
        WorkspaceCopyAccountLayout::workspace()
            .map_err(memory)?
            .requested_bytes(),
        RetainedStorage::original_collector_control_bytes(operands)
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
        native
            .as_ref()
            .map_or(Some(0), |plan| plan.control_bytes::<Owner>())
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
        Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Work>())
            .map_err(|_| memory(WorkingMemoryError::Overflow))?
            .0
            .pad_to_align()
            .size(),
    ];
    reserve(
        funding,
        controls.into_iter().try_fold(size_of_val(&controls), add)?,
    )?;
    stage.set("source projection");
    let projected = projection.construct(mechanisms, host)?;
    stage.set("completed source binding");
    let binding = source::bind(
        &projected,
        completed,
        environment,
        funding,
        host,
        source::BindingDiagnostics {
            stage,
            rows: source_rows,
        },
    )?;
    stage.set("isolated copy program");
    let program = WorkspaceIsolatedCopyPlan::prepare_finite_with_layout(
        &projected.context,
        binding.borrowed(),
        &projected.inputs,
        &mechanisms,
        layout,
    )
    .map_err(|cause| paid(funding, cause))?
    .construct()
    .map_err(|cause| paid(funding, cause))?;
    let topology = environment.pool().topology();
    let descriptor_bytes =
        eredu_core::DomainMemoryRequirements::construction_backing_bytes(topology, 0)
            .and_then(|bytes| {
                bytes
                    .checked_add(
                        std::mem::size_of::<eredu_core::DomainMemoryRequirements>() as u64
                            + 2 * std::mem::size_of::<usize>() as u64,
                    )
                    .ok_or(eredu_core::MemoryDomainError::Overflow)
            })
            .and_then(|bytes| {
                capacity
                    .named_initialization_bytes(topology)?
                    .checked_add(bytes)
                    .ok_or(eredu_core::MemoryDomainError::Overflow)
            })
            .map_err(|cause| paid(funding, cause))?;
    reserve(
        funding,
        usize::try_from(descriptor_bytes).map_err(|_| memory(WorkingMemoryError::Overflow))?,
    )?;
    let numerical = program
        .report()
        .physical_domains
        .as_ref()
        .map(|domains| &domains.native_allocations)
        .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
    let mut delta = eredu_core::DomainMemoryRequirements::zero(topology);
    if let Some(plan) = native.as_ref() {
        let domain = plan
            .physical_domain()
            .map_err(|cause| paid(funding, cause))?;
        let admitted = numerical
            .get(domain)
            .and_then(|charge| charge.total())
            .map_err(|cause| paid(funding, cause))?;
        let required = u64::try_from(plan.physical_bytes())
            .map_err(|_| memory(WorkingMemoryError::Overflow))?;
        let extra = if required > admitted {
            required - admitted
        } else {
            0
        };
        delta
            .add_allocation(
                extra,
                &eredu_core::MemoryPlacement::fixed(topology, domain)
                    .map_err(|cause| paid(funding, cause))?,
            )
            .map_err(|cause| paid(funding, cause))?;
    }
    let mut limits = WorkspaceCopyLimits::new(
        capacity
            .named(topology)
            .map_err(|cause| paid(funding, cause))?,
    );
    limits.additional_requirements = Some(std::sync::Arc::new(delta));
    stage.set("copy admission");
    let account = binding.admit(program, environment, limits, funding)?;
    let (custody, scope) = account.into_parts();
    let root_count = operands
        .checked_mul(2)
        .and_then(|n| n.checked_add(retained))
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let roots = match funding.metadata_vec(root_count) {
        Ok(roots) => roots,
        Err(cause) => {
            let _ = scope.certify();
            return Err(paid(funding, cause));
        }
    };
    stage.set("native copy preparation");
    let prepared = match native
        .map(|plan| plan.prepare(&custody, initialized))
        .transpose()
    {
        Ok(prepared) => prepared,
        Err(cause) => {
            let _ = scope.certify();
            return Err(cause.into());
        }
    };
    stage.set("destination publication preparation");
    let pending = publication.construct(
        &scope,
        &custody,
        host,
        prepared.as_ref().map(|p| p.budget()),
    );
    let pending = match pending {
        Ok(pending) => pending,
        Err(cause) => {
            let _ = scope.certify();
            return Err(cause);
        }
    };
    stage.set("destination inventory preparation");
    let inventory =
        match RetainedStorage::prepare_snapshot_publication(operands, environment.pool(), host) {
            Ok(inventory) => inventory,
            Err(cause) => {
                let _ = scope.certify();
                return Err(cause.into());
            }
        };
    let owner = Owner(Some(Rc::new(Work {
        roots: RefCell::new(roots),
        scope: RefCell::new(Some(scope)),
        discarded: Cell::new(false),
        // A zero-operand copy has no native scope or numerical work.
        healthy: Cell::new(operands == 0),
        _copy: custody.retention(),
        _host: host.clone(),
    })));
    // All destination/publication locals and the native execution/recovery
    // below retire before this guard can authorize discarded-root cleanup.
    let mut discarded = DiscardUnpublished(Some(owner.clone()));
    // Move these prepared containers after the guard so any later populated
    // publication prefix is destroyed before discarded Work can retire.
    let mut pending = pending;
    let mut inventory = inventory;
    stage.set("native copy scope");
    let (mut recovery, mut execution) = match prepared {
        Some(prepared) => {
            let (recovery, execution) = prepared.begin(owner.clone())?;
            (Some(recovery), Some(execution))
        }
        None => (None, None),
    };
    // The bank is born after recovery and is closed before any sealing/unwind.
    struct Construction<'a>(&'a mut Option<crate::backend::array_copy::OriginalCopyExecution>);
    impl Drop for Construction<'_> {
        fn drop(&mut self) {
            if let Some(execution) = self.0.as_mut() {
                execution.finish_construction();
            }
        }
    }
    stage.set("native copy execution");
    let result = (|| {
        let _construction = Construction(&mut execution);
        let roots = &owner.work().roots;
        let mut failure = None;
        source
            .visit_retained_arrays(&mut |array| {
                if failure.is_none() {
                    match array.try_clone_handle() {
                        Ok(array) => roots.borrow_mut().push(array),
                        Err(cause) => failure = Some(Error::from(cause)),
                    }
                }
            })
            .map_err(|cause| paid(funding, cause))?;
        if let Some(cause) = failure {
            Err(cause)
        } else {
            source
                .copy_dense_with_preparation(host, funding, environment.stream(), roots)
                .and_then(|destination| {
                    let prepared = destination
                        .prepare_copy_fixed()
                        .map_err(|cause| paid(funding, cause))?;
                    let mut failure = None;
                    let observer = if operands == 0 {
                        None
                    } else {
                        Some(safemlx::OriginalScopeObserver::require_current()?)
                    };
                    prepared
                        .visit_operands(&mut |array| {
                            if failure.is_none() {
                                failure = array
                                    .completed_in_original_scope(
                                        observer.as_ref().expect("nonempty copy"),
                                    )
                                    .map(|_| ())
                                    .map_err(Error::from)
                                    .err();
                            }
                        })
                        .map_err(|cause| paid(funding, cause))?;
                    if let Some(cause) = failure {
                        return Err(cause);
                    }
                    drop(prepared);
                    Ok(destination)
                })
        }
    })();
    if let Some(recovery) = recovery.as_mut() {
        recovery.seal();
    }
    let destination = result?;
    stage.set("native copy settlement");
    if let Some(recovery) = recovery.as_ref() {
        let status = recovery.progress();
        if !status.settled || status.failed || status.blocked {
            return Err(memory(WorkingMemoryError::UnknownBound));
        }
    }
    stage.set("completed destination inventory");
    let output = destination
        .prepare_copy_fixed()
        .map_err(|cause| paid(funding, cause))?;
    // The shared isolated leaves completed in this scope. The sealed recovery
    // remains live while the original budget authenticates final publication.
    let mut failure = None;
    output
        .visit_operands(&mut |array| {
            if failure.is_none() {
                failure = inventory.include_array(array).map_err(Error::from).err();
            }
        })
        .map_err(|cause| paid(funding, cause))?;
    if let Some(cause) = failure {
        return Err(cause);
    }
    drop(output);
    stage.set("completed destination publication");
    let published = {
        let scope = owner.work().scope.borrow();
        pending.publish(inventory, scope.as_ref().expect("copy scope"))?
    };
    drop(published);
    stage.set("completed copy certification");
    owner
        .work()
        .scope
        .borrow_mut()
        .take()
        .expect("single scope")
        .certify()
        .map_err(memory)?;
    drop(recovery);
    drop(execution);
    drop(discarded.0.take());
    Ok(destination)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
