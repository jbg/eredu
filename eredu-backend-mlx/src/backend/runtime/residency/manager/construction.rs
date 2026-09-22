//! Original finite manager construction before native model loading begins.
use super::owner::{FailureFlag, ManagerCustody, ManagerStream};
use super::*;
use crate::backend::runtime::checkpoint::recipe::{DirectRecipeRead, WeightRecipeError};
use crate::backend::runtime::checkpoint::store::{CacheHandle, PreparedMaterializationStreams};
use crate::backend::runtime::execution::generic::ParameterConstructors;
use crate::backend::runtime::execution::layerwise::{
    OriginalDenseControllerFacts, PreparedDenseController, PreparedDenseControllerError,
};
use eredu_runtime::ExecutionUnitLayout;
use eredu_runtime::working_memory::{
    MemoryLedger, OriginalHostMetadataCustody, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
};
use safemlx::PreparedHostTransferPlan;
use std::{alloc::Layout, fmt, mem::size_of, time::Duration};
mod foreground_disk;
mod read_source_plan;
mod source;
pub(crate) use foreground_disk::{
    ForegroundDiskDescriptors, ForegroundDiskReadError, ForegroundDiskReadLayout,
    ForegroundDiskReadPlan, ForegroundDiskSourceError, ForegroundMaterializationPopulation,
    PreparedForegroundDiskIo, PreparedForegroundDiskRead, ReadForegroundDiskBatch,
    prepare_foreground_disk_descriptors,
};
use read_source_plan::ReadSourcePlan;
pub(super) use source::OriginalHostSources;
mod addressable;
mod host_birth;
mod supplementary;
use supplementary::SupplementarySourcePlan;
struct PlannedRead {
    unit: usize,
    binding: usize,
    read: read_source_plan::ReadValue,
}

// These declarations are caller-owned cold planning inputs. Only the consumed
// constructor clones them into source-accounted manager storage.
pub(crate) struct OriginalManagerPlan {
    pub(crate) primary: eredu_checkpoint::store::RetainedCheckpointSource,
    pub(crate) sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
    pub(crate) plan: OffloadPlan,
    pub(crate) units: Vec<OffloadUnit>,
    pub(crate) groups: Vec<String>,
    target: Option<OriginalTargetPlan>,
    reads: Vec<PlannedRead>,
    catalogs: source::OriginalHostCatalogPlan,
    foreground: Option<ForegroundDiskDescriptors>,
    supplementary: Option<SupplementarySourcePlan>,
    host_units: Vec<usize>,
    host_reads: Vec<usize>,
    host_read_slots: Vec<Option<usize>>,
    initial_units: [Vec<OffloadUnitId>; 2],
    binding_reads: Vec<Vec<usize>>,
    array_handles: Vec<usize>,
    materialized: Option<host_birth::materialized::ConstructionPlan>,
    pub(crate) streams: PreparedMaterializationStreams,
    pub(crate) cache: CacheHandle,
    pub(crate) pool: MemoryLedger,
}
// Target execution topology is distinct from the actual bank source protocol.
// A source-only manager has no target layout, selection or dense controller.
struct OriginalTargetPlan {
    parameter_exclusions: Vec<String>,
    parameter_constructors: Option<Vec<ParameterConstructors>>,
    layout: ExecutionUnitLayout,
    depth: usize,
    selected_ids: Vec<OffloadUnitId>,
    selected_definitions: Vec<OffloadUnit>,
    dense_controller: Option<OriginalDenseControllerFacts>,
}
#[derive(Clone, Copy)]
struct OriginalTargetInputs<'a> {
    parameter_exclusions: &'a BTreeSet<String>,
    parameter_constructors: Option<&'a [ParameterConstructors]>,
    layout: &'a ExecutionUnitLayout,
    depth: usize,
    selected_ids: &'a [OffloadUnitId],
    dense_controller: Option<OriginalDenseControllerFacts>,
}
impl OriginalManagerPlan {
    fn read_source_plan(&self) -> ReadSourcePlan<'_> {
        ReadSourcePlan {
            units: &self.units,
            reads: &self.reads,
            catalogs: &self.catalogs,
        }
    }

    fn source(&self, id: &OffloadUnitId) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.sources.get(id).unwrap_or(&self.primary).as_ref()
    }
    fn read_range(&self, unit: usize) -> std::ops::Range<usize> {
        let start = self.reads.partition_point(|row| row.unit < unit);
        start..self.reads.partition_point(|row| row.unit <= unit)
    }
    fn read_index(&self, unit: &OffloadUnitId, name: &str) -> Option<usize> {
        self.reads.iter().position(|row| {
            self.units[row.unit].id() == unit
                && self.units[row.unit].bindings()[row.binding].name() == name
        })
    }
    fn canonical_read_index(&self, unit: &OffloadUnitId, binding: &WeightBinding) -> Option<usize> {
        let unit = self.units.iter().position(|row| row.id() == unit)?;
        let binding = self.units[unit]
            .bindings()
            .binary_search_by(|row| row.name().cmp(binding.name()))
            .ok()?;
        self.binding_reads.get(unit)?.get(binding).copied()
    }
    fn shape_for(&self, unit: &OffloadUnitId, binding: &WeightBinding) -> Option<&[i32]> {
        Some(
            self.reads[self.canonical_read_index(unit, binding)?]
                .read
                .shape(),
        )
    }
    fn array_handles(&self, index: usize) -> Option<usize> {
        self.array_handles.get(index).copied()
    }
    fn prepare_populations(&mut self) -> Result<(), PreparationFailure> {
        let prepared = ResidencyController::prepare_constructor(
            |id| self.source(id),
            &self.plan,
            &self.units,
            &self.groups,
        )
        .map_err(|_| WorkingMemoryError::UnknownBound)?;
        let mut scratch =
            vec![eredu_runtime::residency::ResidencyClosureSlot::default(); self.units.len()];
        let mut initial_units: [Vec<OffloadUnitId>; 2] = [Vec::new(), Vec::new()];
        for (index, tier) in [MemoryTier::Host, MemoryTier::Device]
            .into_iter()
            .enumerate()
        {
            let roots = self
                .plan
                .units()
                .iter()
                .filter(move |unit| unit.tier() == tier)
                .map(|unit| unit.id());
            let closure = prepared
                .operation_closure(roots, &mut scratch)
                .map_err(PreparationFailure::Closure)?;
            initial_units[index] = closure.units().map(|unit| unit.id().clone()).collect();
        }
        let reads: BTreeMap<_, _> = self
            .reads
            .iter()
            .enumerate()
            .map(|(index, row)| {
                (
                    (
                        self.units[row.unit].id().as_str(),
                        self.units[row.unit].bindings()[row.binding].name(),
                    ),
                    index,
                )
            })
            .collect();
        let mut binding_reads = Vec::with_capacity(self.units.len());
        let mut array_handles = vec![0usize; self.reads.len()];
        for unit in &self.units {
            let device = initial_units[1].contains(unit.id());
            let mut row = Vec::with_capacity(unit.bindings().len());
            for binding in unit.bindings() {
                let (owner, canonical) = if binding.is_alias() {
                    prepared
                        .binding_owner(unit.id(), binding)
                        .ok_or(WorkingMemoryError::IdentityMismatch)?
                } else {
                    (unit.id(), binding)
                };
                let index = *reads
                    .get(&(owner.as_str(), canonical.name()))
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                row.push(index);
                if device {
                    array_handles[index] = array_handles[index]
                        .checked_add(2)
                        .ok_or(WorkingMemoryError::Overflow)?;
                }
            }
            binding_reads.push(row);
        }
        drop(reads);
        drop(prepared);
        let host_units = self
            .units
            .iter()
            .enumerate()
            .filter(|(_, unit)| {
                self.foreground.is_none() || initial_units.iter().any(|ids| ids.contains(unit.id()))
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let host_reads = self
            .reads
            .iter()
            .enumerate()
            .filter(|(_, row)| host_units.binary_search(&row.unit).is_ok())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let mut host_read_slots = vec![None; self.reads.len()];
        for (slot, index) in host_reads.iter().copied().enumerate() {
            host_read_slots[index] = Some(slot);
        }
        self.host_units = host_units;
        self.host_reads = host_reads;
        self.host_read_slots = host_read_slots;
        self.initial_units = initial_units;
        self.binding_reads = binding_reads;
        self.array_handles = array_handles;
        Ok(())
    }
    pub(crate) fn prepare(self) -> Result<ResidencyManager, OriginalManagerError> {
        let pool = self.pool.clone();
        pool.initialize_shared_native(self)
            .map(|ready| ready.output().clone())
            .map_err(|e| OriginalManagerError(PreparationFailure::Construct(e)))
    }
}
impl SharedNativeInitializer for OriginalManagerPlan {
    type Output = ResidencyManager;
    type Error = ConstructionError;
    fn temporary_allocation_requirements(&self) -> Option<&eredu_core::DomainMemoryRequirements> {
        self.materialized.as_ref().map(|plan| &plan.requirements)
    }
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.streams.validate_pool(&self.pool)?;
        self.cache.validate_pool(&self.pool)?;
        let neutral = ResidencyController::prepare_constructor(
            |id| self.source(id),
            &self.plan,
            &self.units,
            &self.groups,
        )
        .map_err(|cause| match cause {
            eredu_runtime::residency::ResidencyConstructionError::MetadataUnavailable => {
                WorkingMemoryError::UnknownBound
            }
            eredu_runtime::residency::ResidencyConstructionError::Layout => {
                WorkingMemoryError::Overflow
            }
        })?;
        let mut bytes = neutral.required_bytes();
        let mut add = |n: usize| {
            bytes = bytes.checked_add(n).ok_or(WorkingMemoryError::Overflow)?;
            Ok::<_, WorkingMemoryError>(())
        };
        for n in [
            ManagerOwner::storage_bytes()?,
            FailureFlag::storage_bytes()?,
            OriginalHostMetadataCustody::initialized_mutex_bytes()?,
            OriginalHostMetadataCustody::initialized_condvar_bytes()?,
        ] {
            add(usize::try_from(n).map_err(|_| WorkingMemoryError::Overflow)?)?;
        }
        for layout in [
            rows::Rows::<OffloadUnitId, UnitStorage>::requested_layout(self.units.len()),
            rows::AliasPins::requested_layout(self.units.len()),
        ] {
            add(layout.ok_or(WorkingMemoryError::Overflow)?.size())?;
        }
        for unit in &self.units {
            // One storage key and the three canonical tier-pin keys.
            add(unit
                .id()
                .as_str()
                .len()
                .checked_mul(4)
                .ok_or(WorkingMemoryError::Overflow)?)?;
        }
        add(self.host_storage_bytes()?)?;
        if let Some(plan) = &self.supplementary {
            add(plan.storage_bytes(self)?)?;
        }
        if let Some(target) = &self.target {
            add(crate::backend::runtime::execution::generic::MlxParameterExclusions::construction_bytes(
                &target.parameter_exclusions)?)?;
            if let Some(constructors) = &target.parameter_constructors {
                add(Layout::array::<ParameterConstructors>(constructors.len())
                    .map_err(|_| WorkingMemoryError::Overflow)?
                    .size())?;
            }
            if let Some(facts) = target.dense_controller {
                add(PreparedDenseController::storage_bytes(
                    facts,
                    &target.layout,
                    &target.selected_ids,
                )?)?;
            }
        }
        for n in [
            size_of::<ManagerInner>(),
            size_of::<ManagerState>(),
            size_of::<ResidencySources>(),
            size_of::<ManagerCustody>(),
            size_of::<FailureFlag>(),
            size_of::<ResidencyManager>(),
            size_of::<std::sync::LockResult<MutexGuard<'static, ManagerState>>>(),
            size_of::<MutexGuard<'static, ManagerState>>(),
            size_of::<std::sync::WaitTimeoutResult>(),
            size_of::<
                std::sync::LockResult<(
                    MutexGuard<'static, ManagerState>,
                    std::sync::WaitTimeoutResult,
                )>,
            >(),
            size_of::<Duration>(),
            size_of::<ConstructionError>(),
            size_of::<Result<ResidencyManager, ConstructionError>>(),
            size_of::<(
                Option<&OriginalTargetPlan>,
                &OriginalTargetPlan,
                Option<&Vec<ParameterConstructors>>,
                &Vec<ParameterConstructors>,
                &[ParameterConstructors],
                Vec<ParameterConstructors>,
                Option<Vec<ParameterConstructors>>,
            )>(),
        ] {
            add(n)?;
        }
        Ok(bytes)
    }
    fn initialize(
        self,
        mut custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        let temporary = if self.materialized.is_some() {
            custody.take_allocation_scope().map(Some)
        } else {
            Ok(None)
        };
        let custody = ManagerCustody::new(custody);
        let result = (|| -> Result<ResidencyManager, ConstructionCause> {
            let temporary = temporary.map_err(ResidencyError::OriginalCache)?;
            let materialized = match (self.materialized.as_ref(), temporary) {
                (Some(plan), Some(scope)) => {
                    Some(host_birth::materialized::Session::new(plan, scope)?)
                }
                (None, None) => None,
                _ => return Err(ResidencyError::OriginalOperationDomain.into()),
            };
            let mut control = ResidencyController::prepare_constructor(
                |id| self.source(id),
                &self.plan,
                &self.units,
                &self.groups,
            )
            .map_err(|_| ResidencyError::OriginalCache(WorkingMemoryError::UnknownBound))?
            .construct()
            .map_err(ResidencyError::from)?;
            for id in self.sources.keys() {
                if control.unit(id).is_none() {
                    return Err(ResidencyError::from(
                        ResidencyControllerError::UnexpectedUnitDefinition { id: id.clone() },
                    )
                    .into());
                }
            }
            let materialization = crate::backend::runtime::checkpoint::store::ManagerMaterializationContext::prepared(
                self.streams.clone(),self.cache.clone(),&self.pool,
            ).map_err(ResidencyError::OriginalCache)?;
            let (sources, read_bytes, read_duration) = self.construct_hosts(
                &control,
                custody.clone(),
                materialization.view(),
                materialized.as_ref(),
            )?;
            if let Some(materialized) = materialized {
                materialized.finish()?;
            }
            control.ledger_mut().record_transfer(
                TransferDirection::DiskToHost,
                read_bytes,
                read_duration,
            );
            let sources = ResidencySources::Original(sources);
            let alias_owner_pins = rows::AliasPins::new(control.units().map(OffloadUnit::id));
            let storage = control
                .units()
                .map(|unit| (unit.id().clone(), UnitStorage::default()))
                .collect();
            let failed_transfer = FailureFlag::new(custody.clone());
            let mut manager = ResidencyManager {
                inner: ManagerOwner::new(
                    ManagerInner {
                        parameter_constructors: self.target.as_ref()
                            .and_then(|target| target.parameter_constructors.as_ref())
                            .map(|constructors| constructors.to_vec()),
                        parameter_exclusions: self.target.as_ref().map(|target|
                            crate::backend::runtime::execution::generic::MlxParameterExclusions::construct(
                                &target.parameter_exclusions, custody.clone())),
                        sources,
                        host_workspace: std::sync::OnceLock::new(),
                        dense_controller: std::sync::OnceLock::new(),
                        original_operation_source: std::sync::OnceLock::new(),
                        background_operation_source: std::sync::OnceLock::new(),
                        supplementary_source: std::sync::OnceLock::new(),
                        failed_transfer: failed_transfer.clone(),
                        state: Mutex::new(ManagerState {
                            failed_transfer,
                            control,
                            storage,
                            alias_owner_pins,
                            admitted_disk_route: std::sync::Weak::new(),
                            admitted_disk_window: BTreeSet::new(),
                            materialization,
                            source_stream: ManagerStream::Prepared {
                                streams: self.streams.clone(),
                                source: true,
                            },
                            device_stream: ManagerStream::Prepared {
                                streams: self.streams.clone(),
                                source: false,
                            },
                        }),
                        changed: Condvar::new(),
                    },
                    custody.clone(),
                ),
            };
            self.publish_initial_storage(&mut manager, &custody)?;
            if let Some(target) = &self.target {
                if let Some(facts) = target.dense_controller {
                    let controller = PreparedDenseController::construct(
                        facts,
                        &target.layout,
                        &target.selected_ids,
                        manager.inner.downgrade(),
                        custody.clone(),
                    )?;
                    manager
                        .inner
                        .dense_controller
                        .set(controller)
                        .map_err(|_| ResidencyError::OriginalOperationDomain)?;
                }
                if self.foreground.is_none() {
                    let allocation = original_allocation_facts()?;
                    manager
                        .initialize_host_workspace(
                            &target.selected_ids,
                            &target.layout,
                            target.depth,
                            allocation,
                        )
                        .map_err(ConstructionCause::Snapshot)?;
                    manager
                        .initialize_original_operation_source(
                            &target.selected_ids,
                            &target.layout,
                            target.depth,
                        )
                        .map_err(ConstructionCause::Operation)?;
                } else {
                    manager
                        .initialize_original_foreground_operation_source(
                            &self.pool,
                            &target.selected_ids,
                            &target.layout,
                            target.depth,
                        )
                        .map_err(ConstructionCause::Operation)?;
                    if let Some(facts) = target
                        .dense_controller
                        .filter(|facts| facts.options.host_budget_bytes() > 0)
                    {
                        manager
                            .initialize_original_background_operation_source(
                                &self.pool,
                                &target.selected_ids,
                                &target.layout,
                                facts.options.host_lookahead(),
                            )
                            .map_err(ConstructionCause::Operation)?;
                    }
                }
            }
            if let Some(plan) = &self.supplementary {
                plan.initialize(&mut manager, &self.pool)?;
            }
            // Source identity and window declarations are retained here;
            // request read slots and native operation grants remain separate.
            // Force the real PAL Mutex/Condvar producers while this unpublished
            // manager has exactly one constructor; no first-use race is priced.
            let guard = manager
                .inner
                .state
                .lock()
                .map_err(|_| ResidencyError::StatePoisoned)?;
            let (guard, _) = manager
                .inner
                .changed
                .wait_timeout(guard, Duration::ZERO)
                .map_err(|_| ResidencyError::StatePoisoned)?;
            drop(guard);
            Ok(manager)
        })();
        result.map_err(|cause| ConstructionError {
            cause,
            _custody: custody,
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum ConstructionCause {
    #[error(transparent)]
    Materialized(#[from] host_birth::materialized::Failure),
    #[error("original source selection: {0}")]
    SourceSelection(#[source] WorkingMemoryError),
    #[error(transparent)]
    DenseController(#[from] PreparedDenseControllerError),
    #[error(transparent)]
    Residency(#[from] ResidencyError),
    #[error(transparent)]
    Detached(#[from] eredu_checkpoint::store::DetachedReadBuildError<ManagerCustody>),
    #[error(transparent)]
    Read(#[from] eredu_checkpoint::store::DetachedReadFailure<ManagerCustody>),
    #[error(transparent)]
    Snapshot(#[from] super::host_workspace::HostCopyWorkspaceError),
    #[error(transparent)]
    Operation(#[from] super::operation_source::OperationSourceFailure),
    #[error(transparent)]
    Reserve(#[from] std::collections::TryReserveError),
    #[error(transparent)]
    Admission(#[from] eredu_core::residency::PreparedResidencyAdmissionFailure),
    #[error("residency admission storage allocation failed: {cause}")]
    AdmissionPreparation {
        #[source]
        cause: std::collections::TryReserveError,
        prefix: eredu_core::residency::ResidencyAdmissionStorage,
    },
    #[error(transparent)]
    Quota(#[from] safemlx::SubmissionGraphQuotaCause),
}
impl From<eredu_core::residency::ResidencyAdmissionPreparationError> for ConstructionCause {
    fn from(value: eredu_core::residency::ResidencyAdmissionPreparationError) -> Self {
        Self::AdmissionPreparation {
            cause: value.cause,
            prefix: value.prefix,
        }
    }
}
fn original_allocation_facts()
-> Result<crate::backend::nn::workspace::NativeAllocationFacts, ResidencyError> {
    #[cfg(target_vendor = "apple")]
    {
        crate::backend::nn::workspace::NativeAllocationFacts::current_host()
            .map_err(|_| ResidencyError::OriginalCache(WorkingMemoryError::UnknownBound))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        Err(ResidencyError::OriginalCache(
            WorkingMemoryError::UnknownBound,
        ))
    }
}
pub(crate) struct ConstructionError {
    cause: ConstructionCause,
    _custody: ManagerCustody,
}
impl fmt::Debug for ConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("OriginalManagerConstruction")
            .field(&self.cause)
            .finish()
    }
}
impl fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for ConstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
#[derive(Debug, thiserror::Error)]
enum PreparationFailure {
    #[error("initial residency closure: {0:?}")]
    Closure(eredu_runtime::residency::ResidencyClosureError),
    #[error(transparent)]
    Store(#[from] eredu_checkpoint::store::StoreError),
    #[error(transparent)]
    Policy(#[from] WorkingMemoryError),
    #[error(transparent)]
    Recipe(#[from] WeightRecipeError),
    #[error(transparent)]
    Streams(#[from] crate::backend::runtime::checkpoint::store::PreparedMaterializationStreamError),
    #[error(transparent)]
    Cache(#[from] crate::backend::runtime::checkpoint::store::CacheInitializationError),
    #[error(transparent)]
    Allocator(
        #[from]
        crate::backend::managed_memory::input_allocator::MlxInputAllocatorInitializationError,
    ),
    #[error(transparent)]
    Foreground(#[from] ForegroundDiskSourceError),
    #[error(transparent)]
    Construct(SharedNativeInitializationError<OriginalManagerPlan>),
}
/// Source preparation preserves its original failure and any admitted prefixes.
#[derive(Debug)]
pub struct OriginalManagerError(PreparationFailure);
impl fmt::Display for OriginalManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for OriginalManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
impl ResidencyManager {
    pub(crate) fn parameter_constructors(&self, ordinal: usize) -> Option<ParameterConstructors> {
        self.inner
            .parameter_constructors
            .as_ref()?
            .get(ordinal)
            .copied()
    }

    pub(crate) fn original_parameter_exclusions(
        &self,
    ) -> Option<&crate::backend::runtime::execution::generic::MlxParameterExclusions> {
        self.inner.parameter_exclusions.as_ref()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_original_host(
        primary: eredu_checkpoint::store::RetainedCheckpointSource,
        sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: &OffloadPlan,
        units: &[OffloadUnit],
        groups: &[String],
        selected_ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
        parameter_exclusions: &BTreeSet<String>,
        parameter_constructors: Option<&[ParameterConstructors]>,
        source_stream: &Stream,
        execution_stream: &Stream,
        pool: &MemoryLedger,
    ) -> Result<Option<Self>, OriginalManagerError> {
        Self::prepare_original_layerwise_impl(
            primary,
            sources,
            plan,
            units,
            groups,
            Some(OriginalTargetInputs {
                parameter_exclusions,
                parameter_constructors,
                selected_ids,
                layout,
                depth,
                dense_controller: None,
            }),
            source_stream,
            execution_stream,
            pool,
            false,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_original_foreground_disk(
        primary: eredu_checkpoint::store::RetainedCheckpointSource,
        sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: &OffloadPlan,
        units: &[OffloadUnit],
        groups: &[String],
        selected_ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
        parameter_exclusions: &BTreeSet<String>,
        parameter_constructors: Option<&[ParameterConstructors]>,
        source_stream: &Stream,
        execution_stream: &Stream,
        pool: &MemoryLedger,
    ) -> Result<Option<Self>, OriginalManagerError> {
        Self::prepare_original_layerwise_impl(
            primary,
            sources,
            plan,
            units,
            groups,
            Some(OriginalTargetInputs {
                parameter_exclusions,
                parameter_constructors,
                selected_ids,
                layout,
                depth,
                dense_controller: None,
            }),
            source_stream,
            execution_stream,
            pool,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_original_foreground_disk_with_controller(
        primary: eredu_checkpoint::store::RetainedCheckpointSource,
        sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: &OffloadPlan,
        units: &[OffloadUnit],
        groups: &[String],
        selected_ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
        parameter_exclusions: &BTreeSet<String>,
        parameter_constructors: Option<&[ParameterConstructors]>,
        source_stream: &Stream,
        execution_stream: &Stream,
        pool: &MemoryLedger,
        dense_controller: OriginalDenseControllerFacts,
    ) -> Result<Option<Self>, OriginalManagerError> {
        Self::prepare_original_layerwise_impl(
            primary,
            sources,
            plan,
            units,
            groups,
            Some(OriginalTargetInputs {
                parameter_exclusions,
                parameter_constructors,
                selected_ids,
                layout,
                depth,
                dense_controller: Some(dense_controller),
            }),
            source_stream,
            execution_stream,
            pool,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_original_layerwise_impl(
        primary: eredu_checkpoint::store::RetainedCheckpointSource,
        sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: &OffloadPlan,
        units: &[OffloadUnit],
        groups: &[String],
        target: Option<OriginalTargetInputs<'_>>,
        source_stream: &Stream,
        execution_stream: &Stream,
        pool: &MemoryLedger,
        foreground: bool,
    ) -> Result<Option<Self>, OriginalManagerError> {
        (|| -> Result<_, PreparationFailure> {
            if target.as_ref().is_some_and(|target| {
                target
                    .parameter_constructors
                    .is_some_and(|rows| rows.len() != target.selected_ids.len())
            }) {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            let has_disk = plan
                .units()
                .iter()
                .any(|unit| unit.tier() == MemoryTier::Disk);
            if foreground != has_disk {
                return Ok(None);
            }
            if safemlx::StreamCopyPlan::<ManagerCustody>::capture(source_stream)
                .map_err(|_| WorkingMemoryError::UnknownBound)?
                .device_type()
                != DeviceType::Cpu
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            // Both native destinations use the same prepared source manager.
            // The retained selected stream determines the actual copy worker;
            // no ordinary manager is substituted for CPU execution.
            safemlx::StreamCopyPlan::<ManagerCustody>::capture(execution_stream)
                .map_err(|_| WorkingMemoryError::UnknownBound)?;
            let Some(reads) = read_source_plan::prepare_host_reads(
                units,
                |id| sources.get(id).unwrap_or(&primary),
                pool,
                source_stream,
            )?
            else {
                return Ok(None);
            };
            if reads.iter().any(|row| {
                row.read.shape().is_empty()
                    || row.read.shape().len() > 10
                    || row.read.shape().iter().any(|d| *d <= 0)
            }) {
                return Ok(None);
            }
            if reads.is_empty() {
                return Ok(None);
            }
            // Actual shared native initialization precedes this source grant.
            // A later source failure never falls back to ordinary construction.
            let (runtime, _) =
                crate::backend::managed_memory::input_allocator::prepare_admitted(pool)?;
            for read in &reads {
                PreparedHostTransferPlan::new(&runtime, read.read.shape(), read.read.dtype(), 0)
                    .map_err(|_| WorkingMemoryError::UnknownBound)?;
            }
            drop(runtime);
            let streams =
                PreparedMaterializationStreams::prepare(pool, source_stream, execution_stream)?;
            let cache = CacheHandle::prepare(pool)?;
            let selected_ids = target
                .as_ref()
                .map_or(&[][..], |target| target.selected_ids);
            let target = target
                .map(|target| {
                    let selected_definitions = target
                        .selected_ids
                        .iter()
                        .map(|id| {
                            units
                                .iter()
                                .find(|unit| unit.id() == id)
                                .cloned()
                                .ok_or(WorkingMemoryError::IdentityMismatch)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok::<_, WorkingMemoryError>(OriginalTargetPlan {
                        parameter_exclusions: target.parameter_exclusions.iter().cloned().collect(),
                        parameter_constructors: target.parameter_constructors.map(<[_]>::to_vec),
                        selected_ids: target.selected_ids.to_vec(),
                        selected_definitions,
                        layout: target.layout.clone(),
                        depth: target.depth,
                        dense_controller: target.dense_controller,
                    })
                })
                .transpose()?;
            let catalogs = source::OriginalHostCatalogPlan::capture(&primary, &sources, units)?;
            let foreground = if foreground {
                Some(
                    foreground_disk::prepare_foreground_disk_descriptors_from_reads(
                        &primary, &sources, plan, units, groups, pool, &reads,
                    )?
                    .ok_or(WorkingMemoryError::UnknownBound)?,
                )
            } else {
                None
            };
            let supplementary = SupplementarySourcePlan::prepare(units, selected_ids)?;
            let mut value = OriginalManagerPlan {
                primary,
                sources,
                plan: plan.clone(),
                units: units.to_vec(),
                groups: groups.to_vec(),
                target,
                reads,
                catalogs,
                foreground,
                supplementary,
                host_units: Vec::new(),
                host_reads: Vec::new(),
                host_read_slots: Vec::new(),
                initial_units: [Vec::new(), Vec::new()],
                binding_reads: Vec::new(),
                array_handles: Vec::new(),
                materialized: None,
                streams,
                cache,
                pool: pool.clone(),
            };
            value.prepare_populations()?;
            value.materialized = host_birth::materialized::ConstructionPlan::prepare(&value)?;
            // Every cold descriptor and temporary requirement is complete.
            // Preserve the existing charge while closing growth and exclusion
            // before handing these immutable sources to the native constructor.
            for row in &mut value.reads {
                if let read_source_plan::ReadValue::Materialized(read) = &mut row.read {
                    read.seal_metadata().map_err(|cause| {
                        WorkingMemoryError::MetadataConstruction(
                            eredu_nn::workspace::WorkspaceMetadataError::Funding(cause),
                        )
                    })?;
                }
            }
            let pool = pool.clone();
            pool.initialize_shared_native(value)
                .map(|ready| Some(ready.output().clone()))
                .map_err(PreparationFailure::Construct)
        })()
        .map_err(OriginalManagerError)
    }
    pub(crate) fn validate_original_host_preparation(
        &self,
        primary: &eredu_checkpoint::store::RetainedCheckpointSource,
        sources: &BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: &OffloadPlan,
        units: &[OffloadUnit],
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<(), ResidencyError> {
        if self.original_foreground_disk_descriptors().is_some() {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        self.validate_original_layerwise_preparation(
            primary,
            sources,
            plan,
            units,
            source_stream,
            execution_stream,
        )
    }
    pub(crate) fn validate_original_layerwise_preparation(
        &self,
        primary: &eredu_checkpoint::store::RetainedCheckpointSource,
        sources: &BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: &OffloadPlan,
        units: &[OffloadUnit],
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<(), ResidencyError> {
        let ResidencySources::Original(original) = &self.inner.sources else {
            return Err(ResidencyError::OriginalOperationDomain);
        };
        let state = self
            .inner
            .state
            .try_lock()
            .map_err(|_| ResidencyError::OriginalManagerBusy)?;
        if state.control.ledger().plan() != plan
            || state.control.units().len() != units.len()
            || state.source_stream.as_ref() != source_stream
            || state.device_stream.as_ref() != execution_stream
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        if !original
            .matches_primary_catalog(primary.as_ref())
            .map_err(|_| ResidencyError::OriginalOperationDomain)?
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        for (index, unit) in units.iter().enumerate() {
            if state.control.unit(unit.id()) != Some(unit) {
                return Err(ResidencyError::OriginalOperationDomain);
            }
            let source = sources.get(unit.id()).unwrap_or(primary);
            // Prior successful units authenticate this same pair only. The
            // retained catalog check matters when formerly distinct overrides
            // are replaced by one incoming source.
            let authenticated = (original.same_primary_catalog(unit.id())
                && primary.same_source(source))
                || units[..index].iter().any(|previous| {
                    original.same_catalog(previous.id(), unit.id())
                        && sources
                            .get(previous.id())
                            .unwrap_or(primary)
                            .same_source(source)
                });
            if (!authenticated
                && !original
                    .matches_catalog(unit.id(), source.as_ref())
                    .map_err(|_| ResidencyError::OriginalOperationDomain)?)
                || !original
                    .matches_reads(unit, source.as_ref())
                    .map_err(|_| ResidencyError::OriginalOperationDomain)?
            {
                return Err(ResidencyError::OriginalOperationDomain);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod exclusions_tests;
