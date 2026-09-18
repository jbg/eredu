//! Request-bound storage for the actual selected local unit traversal.
//! Counts and controls are diagnostics until the same accepted Q and first
//! active step are supplied. Unclosed manager/source populations stay unknown.
use super::*;
use crate::backend::submission_recovery::observed::bank::{
    BankPreparationCause, BankPreparationError, PreparedOperationBank,
};
use eredu_runtime::{
    prefill::PrefillControlPlan,
    working_memory::{
        InferenceRequest, InferenceTextStep, OriginalTextControlGuard, OriginalTextPrefillScopeSet,
        TextPredictionScopeFacts, WorkingMemoryError,
    },
};
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    collections::TryReserveError,
    convert::Infallible,
    mem::size_of,
    rc::{Rc, Weak},
};

mod custody;
use custody::{OperationControls, OperationRequest};
pub(super) use custody::SpeculativeOperationRole;
mod prediction;
pub(crate) use prediction::{
    ActivePredictionModuleBank, PredictionModuleCall, PredictionModulePlan,
    PredictionModuleProjection, PreparedPredictionModuleBank,
};
mod speculative;
mod realtime;
pub(crate) use realtime::{RealtimeLayerwisePlan,QualifiedRealtimeLayerwisePlan};
mod registration;
mod activation;
pub(crate) use activation::OriginalOperationActivation;
mod selected_access;
pub(crate) use selected_access::{OriginalSelectedResidencyAccess,PreparedSelectedResidencyAccess};
use registration::Registry;
pub(crate) use registration::{OriginalOperationRegistration, RegisteredOriginalScope, RegisteredScopeRetirementFailure, RegisteredScopeRetirementCause};
mod storage;
use storage::{PreparedStorage, ResidencyPopulation};
pub(super) use storage::{ForegroundDiskRequestPlan, PreparedSpeculativeForegroundSource};
mod gguf_host;
pub(crate) use gguf_host::typed as gguf_host_typed;
mod neural;
mod resident;
pub(super) use neural::OrderedNeuralCompletion;
use neural::{NeuralPopulation, NeuralProducerFit};
/// Borrowed, once-per-claimed-bank partition of its already accepted source.
/// The callback returns the exact target component to the selected loader.
pub(crate) type SpeculativeSourcePartition<'a> = &'a mut dyn FnMut(
    Option<eredu_runtime::working_memory::OriginalHostSourceBank>,
) -> Result<Option<eredu_runtime::working_memory::OriginalHostSourceBank>, Error>;
pub(crate) use resident::{ResidentNeuralPlan, ResidentNeuralSlot, SpeculativeNeuralOwner,
    RealtimeNeuralPlan,QualifiedRealtimeNeuralPlan,RealtimeNeuralOwner};

fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn identity() -> Error {
    memory(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> Error {
    memory(WorkingMemoryError::Overflow)
}
fn unknown() -> Error {
    memory(WorkingMemoryError::UnknownBound)
}
fn rc_layout<T>() -> Option<usize> {
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    Some(
        header
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}

fn arc_layout<T>() -> Option<usize> {
    let header = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>();
    Some(
        header
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}

/// The slot has its own ordinary lifetime, not a policy/session/model backedge.
/// Only a request owner can install a typed weak view and clear that exact view.
pub(super) struct OriginalOperationSlot<U: 'static> {
    bounded: Cell<Option<OriginalOperationProjection<U>>>,
    resident: Cell<Option<resident::Projection>>,
}
impl<U: 'static> OriginalOperationSlot<U> {
    pub(super) fn new() -> Self {
        Self {
            bounded: Cell::new(None),
            resident: Cell::new(None),
        }
    }
    fn take(&self) -> Option<OriginalOperationProjection<U>> {
        self.bounded.take()
    }
    fn set(&self, value: Option<OriginalOperationProjection<U>>) {
        self.bounded.set(value);
    }
    fn replace(
        &self,
        value: Option<OriginalOperationProjection<U>>,
    ) -> Option<OriginalOperationProjection<U>> {
        self.bounded.replace(value)
    }
}
type InstallSlot<U> = Rc<OriginalOperationSlot<U>>;

pub(crate) struct OriginalOperationPlan<'source, U: 'static> {
    retained_sources: Option<&'source super::host_workspace::LayerwiseWorkspace>,
    owned_sources: Option<Rc<super::host_workspace::LayerwiseWorkspace>>,
    slot: InstallSlot<U>,
    identity: Arc<()>,
    sources: Arc<Vec<OffloadUnitId>>,
    window_sources: Option<crate::backend::runtime::residency::manager::OperationWindows>,
    manager: ResidencyManager,
    source_manager: Option<crate::backend::runtime::residency::manager::ManagerWeak>,
    geometry: Option<eredu_core::InferenceGeometry>,
    units: usize,
    pending: usize,
    scopes: usize,
    residency: Option<ResidencyPopulation>,
    neural: NeuralPopulation,
    neural_fit: NeuralProducerFit,
    selected_stream: Option<safemlx::StreamCopyPlan<()>>,
    gguf_host_runtime: Option<Rc<safemlx::PreparedInputRuntime>>,
    source_control_bytes: u64,
    source_arenas: Option<Result<gguf_host::source_arenas::SourceArenaPlan, Error>>,
    named_catalog: Option<crate::backend::runtime::residency::manager::NamedStorageLayout>,
    foreground_disk: Option<storage::ForegroundDiskRequestPlan>,
    foreground_disk_loan: Option<&'source storage::ForegroundDiskRequestPlan>,
    background: Option<storage::foreground_disk::BackgroundSelection>,
    preparation_funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OperationCounts {
    units: usize,
    scopes: usize,
}
fn operation_counts(
    unit_count: usize,
    geometry: eredu_core::InferenceGeometry,
) -> Result<OperationCounts, Error> {
    let plan = PrefillControlPlan::new(geometry, true).map_err(memory)?;
    let forwards = plan
        .span_count()
        .checked_add(geometry.max_output_tokens.saturating_sub(1))
        .ok_or_else(overflow)?;
    let units = u64::try_from(unit_count)
        .ok()
        .and_then(|n| n.checked_mul(forwards))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(overflow)?;
    let roles =
        TextPredictionScopeFacts::new(Some(0), Some(0), Some(0), Some(0), Some(0)).role_count();
    let scopes = u64::try_from(roles)
        .ok()
        .and_then(|n| n.checked_mul(geometry.max_output_tokens))
        .and_then(|n| n.checked_add(plan.scope_count()))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(overflow)?;
    Ok(OperationCounts { units, scopes })
}

impl<U: 'static, P> MlxLayerwisePolicy<U, P> {
    /// Actual new ordinary source/slot controls. Existing Vec/String payloads
    /// are moved unchanged and remain registered-source accounting obligations.
    pub(crate) fn original_operation_baseline_control_bytes() -> Option<u64> {
        u64::try_from(
            arc_layout::<Vec<OffloadUnitId>>()?
                .checked_add(rc_layout::<OriginalOperationSlot<U>>()?)?,
        )
        .ok()
    }

    /// No source/native work or allocation. The exact ordinary immutable IDs and
    /// independent install slot already exist; cloning these owners adds no storage.
    pub(crate) fn original_operation_plan<'source>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        retained_sources: Option<&'source super::host_workspace::LayerwiseWorkspace>,
        groups: eredu_runtime::GroupSubmissionMechanism,
    ) -> Result<OriginalOperationPlan<'source, U>, Error> {
        let counts=operation_counts(self.layout.len(),geometry)?;
        let neural=NeuralPopulation::from_execution(&self.layout,geometry,groups,true)?;
        let residency=ResidencyPopulation::from_policy(self,geometry)?;
        self.operation_plan_from_population(Some(geometry),counts,neural,residency,retained_sources)
    }
    fn operation_plan_from_population<'source>(&self,geometry:Option<eredu_core::InferenceGeometry>,
        counts:OperationCounts,neural:NeuralPopulation,residency:Option<ResidencyPopulation>,
        retained_sources:Option<&'source super::host_workspace::LayerwiseWorkspace>)
        ->Result<OriginalOperationPlan<'source,U>,Error> {
        let OperationCounts{units,scopes}=counts;
        if !self.pending.is_empty()
            || self.dense.as_ref().is_some_and(|dense| {
                dense.forward.is_some() || dense.windows.iter().any(Option::is_some)
            })
        {
            return Err(unknown());
        }

        let value = OriginalOperationPlan {
            retained_sources,
            owned_sources:None,
            slot: Rc::clone(&self.original_operations),
            identity: Arc::clone(&self.workspace_identity),
            sources: Arc::clone(&self.unit_ids),
            window_sources: self
                .operation_source
                .as_ref()
                .ok()
                .map(|source| source.window_owner()),
            manager: self.residency.clone(),
            source_manager: self
                .operation_source
                .as_ref()
                .ok()
                .and_then(|source| source.manager_owner()),
            geometry,
            units,
            pending: units,
            scopes,
            // The real declaration count is consumed below. Cross-unit alias
            // closure, retries and dynamic transfer payloads are not yet closed.
            residency,
            neural,
            neural_fit: NeuralProducerFit::pending(neural).ok_or_else(overflow)?,
            selected_stream: None,
            gguf_host_runtime: None,
            source_arenas: None,
            foreground_disk: None,
            foreground_disk_loan: None,
            preparation_funding: None,
            background: self.dense.as_ref().and_then(|dense| dense.controller.original_background_options())
                .map(|options| {
                    let source = self.residency.background_operation_source(self.unit_ids.as_slice(), &self.layout, options.host_lookahead())
                        .map_err(|_| identity())?;
                    Ok::<_, Error>(storage::foreground_disk::BackgroundSelection::new(source.clone(), options))
                }).transpose()?,
            named_catalog: self
                .operation_source
                .as_ref()
                .ok()
                .map(|source| source.named_layout()),
            source_control_bytes: match &self.operation_source {
                Ok(source) => source.retained_control_bytes().ok_or_else(overflow)?,
                Err(_) => u64::try_from(size_of::<
                    Result<
                        crate::backend::runtime::residency::manager::OriginalResidencySource,
                        crate::backend::runtime::residency::manager::OperationSourceFailure,
                    >,
                >())
                .map_err(|_| overflow())?,
            },
        };
        if value.window_sources.is_some()
            && !value
                .source_manager
                .as_ref()
                .is_some_and(|owner| value.manager.matches_operation_source(owner))
        {
            return Err(identity());
        }
        value.known_control_bytes().ok_or_else(overflow)?;
        Ok(value)
    }
    #[cfg(test)]
    pub(crate) fn inspect_operation_sources_for_test(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) {
        let source_owners = Arc::strong_count(&self.unit_ids);
        let slot_owners = Rc::strong_count(&self.original_operations);
        let plan = self.original_operation_plan(geometry, None,
            eredu_runtime::GroupSubmissionMechanism::LayeredGraph).unwrap();
        assert!(Arc::ptr_eq(&plan.sources, &self.unit_ids));
        assert!(Arc::ptr_eq(&plan.identity, &self.workspace_identity));
        assert_eq!(plan.sources.len(), self.layout.len());
        assert!(plan.known_control_bytes().unwrap() > 0);
        let spans = PrefillControlPlan::new(geometry, true)
            .unwrap()
            .span_count();
        let mut visits = 0;
        for _ in 0..spans {
            for _ in self.unit_ids.iter() {
                visits += 1;
            }
        }
        for _ in 1..geometry.max_output_tokens {
            for _ in self.unit_ids.iter() {
                visits += 1;
            }
        }
        assert_eq!(plan.units, visits);
        drop(plan);
        assert_eq!(Arc::strong_count(&self.unit_ids), source_owners);
        assert_eq!(Rc::strong_count(&self.original_operations), slot_owners);
    }
    pub(crate) fn original_operation_projection(&self) -> Option<OriginalOperationProjection<U>> {
        let current = self.original_operations.take();
        let view = current.as_ref().map(|view| OriginalOperationProjection {
            value: view.value.clone(),
            identity: Arc::clone(&self.workspace_identity),
            controls: view.controls.clone(),
        });
        self.original_operations.set(current);
        view
    }
}

impl<U: 'static> OriginalOperationPlan<'_, U> {
    fn retained_source(&self)->Option<&super::host_workspace::LayerwiseWorkspace> {
        self.retained_sources.or(self.owned_sources.as_deref())
    }

    fn with_selected_stream(mut self, stream: &Stream) -> Result<Self, Error> {
        self.selected_stream =
            Some(safemlx::StreamCopyPlan::capture(stream).map_err(|_| identity())?);
        Ok(self)
    }

    /// Borrowed projection of the already selected native model's cold result.
    /// A preparation failure remains its actual cause; this never initializes.
    pub(crate) fn with_gguf_host_runtime(
        mut self,
        runtime: &Result<Rc<safemlx::PreparedInputRuntime>, eredu_core::SharedBackendFailure>,
    ) -> Result<Self, Error> {
        if self.retained_source().is_none() && self.foreground_disk_plan().is_none() {
            self.gguf_host_runtime = Some(gguf_host::project_runtime(runtime)?);
        }
        Ok(self)
    }
    /// Select the already source-admitted foreground reader. This supplies
    /// concrete storage counts only; native/source policy fit stays independent.
    fn foreground_disk_plan(&self) -> Option<&storage::ForegroundDiskRequestPlan> {
        self.foreground_disk.as_ref().or(self.foreground_disk_loan)
    }
    /// Retain the actual cumulative host preparation account. Account-only Q
    /// custody cannot create this spending capability. It follows all cold
    /// declarations and the later background source/window producers.
    pub(crate) fn with_preparation_funding(mut self, funding: Option<&eredu_nn::workspace::HostMetadataFunding>) -> Self {
        self.preparation_funding = funding.cloned();
        self
    }
    pub(crate) fn with_foreground_disk_reads(
        mut self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<Self, Error> {
        if self
            .manager
            .original_foreground_disk_descriptors()
            .is_some()
        {
            if self.source_arenas.is_some() {
                return Err(identity());
            }
            self.foreground_disk = Some(storage::ForegroundDiskRequestPlan::new_with_metadata(
                &self.manager,
                pool,
                self.window_sources.as_ref().ok_or_else(unknown)?.as_slice(),
                self.sources.as_slice(),
                self.residency.ok_or_else(unknown)?,
                self.preparation_funding.as_ref(),
            )?.with_background(self.background.as_ref(), &self.manager, self.sources.as_slice(), self.window_sources.as_ref().ok_or_else(unknown)?.as_slice(), self.preparation_funding.as_ref())?);
        }
        Ok(self)
    }
    /// Same immutable source retained by the selected manager and every window plan.
    pub(crate) fn foreground_disk_descriptors(
        &self,
    ) -> Option<&crate::backend::runtime::residency::manager::ForegroundDiskDescriptors> {
        self.foreground_disk_plan()?;
        self.manager.original_foreground_disk_descriptors()
    }
    pub(crate) fn foreground_disk_population(
        &self,
    ) -> Option<crate::backend::runtime::residency::manager::ForegroundDiskPopulation> {
        self.foreground_disk_plan()?.forward_population()?.checked_mul(self.residency?.forwards)
    }
    pub(crate) fn foreground_disk_forward_population(
        &self,
    ) -> Option<crate::backend::runtime::residency::manager::ForegroundDiskPopulation> {
        self.foreground_disk_plan()?.forward_population()
    }
    pub(crate) fn foreground_disk_window_population(
        &self,
        index: usize,
    ) -> Option<crate::backend::runtime::residency::manager::ForegroundDiskPopulation> {
        self.foreground_disk_plan()?.window(index)?.population()
    }
    /// Invoke wrapper metadata queries only after session/policy loans end.
    /// These are real cold owners; this method does not qualify their producers.
    pub(crate) fn with_source_arena_plan(mut self) -> Result<Self, Error> {
        if self.foreground_disk_plan().is_some() {
            return Ok(self);
        }
        if let Some(source) = self.retained_source() {
            source.validate_original_policy(
                &self.manager,
                &self.identity,
                self.sources.as_slice(),
            )?;
            return Ok(self);
        }
        let plan = gguf_host::source_arenas::SourceArenaPlan::prepare(
            &self.manager,
            self.window_sources.as_ref().ok_or_else(unknown)?.as_slice(),
            self.sources.as_slice(),
            self.residency.ok_or_else(unknown)?,
            self.gguf_host_runtime.as_ref().ok_or_else(unknown)?,
        );
        // None/unavailable/custom routes remain an unknown contribution. Keep
        // the actual typed planning failure without changing ordinary behavior.
        self.source_arenas = Some(plan);
        Ok(self)
    }
    pub(crate) fn source_construction_facts(
        &self,
    ) -> Option<eredu_runtime::working_memory::HostSourceConstructionFacts> {
        if let Some(disk) = self.foreground_disk_plan() {
            return disk.source_facts();
        }
        self.source_arenas
            .as_ref()
            .and_then(|plan| plan.as_ref().ok())
            .map(|plan| plan.facts())
    }
    pub(crate) fn host_destination_facts(
        &self,
    ) -> Option<eredu_runtime::working_memory::HostDestinationFacts> {
        if let Some(disk) = self.foreground_disk_plan() {
            return disk.host_facts();
        }
        self.source_arenas
            .as_ref()
            .and_then(|plan| plan.as_ref().ok())
            .map(|plan| plan.host_facts())
    }
    pub(crate) fn pending_gguf_host_fit(&self) -> gguf_host::PendingGgufHostFit {
        gguf_host::PendingGgufHostFit::new(
            self.residency.as_ref().map(|p| p.pending),
            self.source_construction_facts(),
            self.host_destination_facts(),
        )
    }

    /// Named known controls remain checked even when another actual producer is unknown.
    pub(crate) fn known_control_bytes(&self) -> Option<u64> {
        let unit_factory = storage::unit_factory::<U>(None);
        let unit = storage::unit_layout(self.units, &unit_factory)?;
        let neural = self.neural.rust_control_bytes()?;
        let binding = match self.residency.as_ref() {
            Some(population) => crate::backend::runtime::checkpoint::binding::original_parameter_binding_control_bytes(
                population.unprepared.transfer.binding_rows)?.checked_mul(self.units)?,
            None => 0,
        };
        // Check every native entry-point population without treating the one
        // output root as a dependency-DAG bound or charging its arena twice.
        let _ = self.neural.native_requirements()?.fixed_control_bytes()?;
        let arrays = Layout::array::<safemlx::OriginalScopeObserver>(self.scopes)
            .ok()?
            .size()
            .checked_add(Layout::array::<MlxUnitLease<U>>(self.pending).ok()?.size())?
            .checked_add(
                Layout::array::<(OffloadUnitId, u64)>(self.sources.len())
                    .ok()?
                    .size(),
            )?;
        let source_names = self.sources.iter().try_fold(0usize, |sum, id| {
            sum.checked_add(Layout::array::<u8>(id.as_str().len()).ok()?.size())
        })?;
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<Option<Self>, Error>>(),
            size_of::<Bank<U>>(),
            size_of::<super::host_workspace::OriginalLayerwiseSourceCustody>(),
            size_of::<Result<super::host_workspace::OriginalLayerwiseSourceCustody, Error>>(),
            size_of::<OwnedBank<U>>(),
            size_of::<OriginalOperationBankOwner>(),
            size_of::<OriginalOperationProjection<U>>(),
            size_of::<Option<OriginalOperationProjection<U>>>(),
            size_of::<OriginalOperationAccess<U>>(),
            size_of::<PreparedStorage<U>>(),
            size_of::<OriginalResidencyAttempt>(),
            size_of::<Result<OriginalResidencyAttempt, Error>>(),
            size_of::<Option<OriginalResidencyAttempt>>(),
            // The canonical acquisition worker lends prepaid rows through the
            // same request representation used for ordinary owned rows. Funded
            // calls never construct the owned variant or retry after refusal.
            size_of::<std::borrow::Cow<'static, [(OffloadUnitId, u64)]>>(),
            size_of::<super::bounded::AcquisitionRecovery>(),
            // Added shared manager window frame; the caller's existing loop
            // controls and finite eviction/source population are unchanged.
            size_of::<(&ResidencyManager, &[OffloadUnitId], &[OffloadUnitId])>(),
            size_of::<Result<(), crate::backend::runtime::residency::manager::ResidencyError>>(),
            size_of::<Rc<Registry>>(),
            size_of::<Result<PreparedStorage<U>, Error>>(),
            size_of::<Registry>(),
            size_of::<PreparedContainers<U>>(),
            size_of::<Result<PreparedContainers<U>, Error>>(),
            size_of::<Option<ResidencyPopulation>>(),
            size_of::<Result<Option<OriginalOperationBankOwner>, Error>>(),
            size_of::<Result<OriginalOperationBankOwner, Error>>(),
            size_of::<std::cell::RefMut<'static, PreparedStorage<U>>>(),
            size_of::<std::cell::RefMut<'static, VecDeque<MlxUnitLease<U>>>>(),
            size_of::<Vec<(OffloadUnitId, u64)>>(),
            size_of::<Vec<safemlx::OriginalScopeObserver>>(),
            size_of::<VecDeque<MlxUnitLease<U>>>(),
            size_of::<TryReserveError>(),
            size_of::<PreparationFailure>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Box<PreparationFailure>>(),
            size_of::<OriginalTextControlGuard>(),
            size_of::<OperationControls>(),
            size_of::<OperationRequest>(),
            size_of::<Option<&OriginalTextControlGuard>>(),
            size_of::<Option<&eredu_runtime::working_memory::WorkingMemoryReservation>>(),
            size_of::<(&PreparedStorage<U>, &OriginalResidencyAttempt)>(),
            size_of::<eredu_runtime::working_memory::OriginalOperationMetadataCustody>(),
            size_of::<PreparationFailure<eredu_runtime::working_memory::OriginalOperationMetadataCustody>>(),
            size_of::<(PreparedUnit<U>, safemlx::OriginalScopeObserver)>(),
            size_of::<Result<(PreparedUnit<U>, safemlx::OriginalScopeObserver), Error>>(),
            size_of::<Box<dyn ErasedOwner>>(),
            size_of::<safemlx::OriginalScopeObserver>(),
        ]
        .into_iter()
        .try_fold(arrays.checked_add(source_names)?, usize::checked_add)?;
        let allocations = rc_layout::<Bank<U>>()?
            .checked_add(rc_layout::<Registry>()?)?
            .checked_add(rc_layout::<OriginalOperationSlot<U>>()?)?
            .checked_add(arc_layout::<Vec<OffloadUnitId>>()?)?
            .checked_add(Layout::new::<OwnedBank<U>>().size())?
            .checked_add(Layout::new::<PreparationFailure>().size().max(Layout::new::<PreparationFailure<eredu_runtime::working_memory::OriginalOperationMetadataCustody>>().size()))?;
        unit.checked_add(neural)?
            .checked_add(u64::try_from(activation::typed_control_bytes::<U>()?).ok()?)?
            .checked_add(u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<neural::NeuralBoundaryFailure<OperationControls>>()?).ok()?)?
            .checked_add(
                u64::try_from(
                    self.selected_stream
                        .map(|stream| stream.source_comparison_control_bytes())
                        .unwrap_or(Some(0))?,
                )
                .ok()?,
            )?
            .checked_add(selected_plan_control_bytes::<U>()?)?
            .checked_add(u64::try_from(binding).ok()?)?
            .checked_add(
                if self.retained_source().is_some() || self.foreground_disk_plan().is_some() {
                    0
                } else {
                    gguf_host::runtime_control_bytes()?
                },
            )?
            .checked_add(if self.foreground_disk_plan().is_some_and(|plan| plan.background().is_some()) {
                storage::background_acquisition_control_bytes(self.foreground_disk_plan()?.background()?.host_acquisitions(), self.residency?.forwards, self.residency?.controller_units)?
            } else { 0 })?
            .checked_add(self.source_control_bytes)?
            .checked_add(
                self.foreground_disk_plan()
                    .map(|plan| if self.foreground_disk_loan.is_some() { plan.operation_control_bytes() } else { plan.control_bytes() })
                    .unwrap_or(Some(0))?,
            )?
            .checked_add(
                self.retained_source()
                    .map(|source| source.original_pin_control_bytes())
                    .unwrap_or(Some(0))?,
            )?
            .checked_add(
                self.source_arenas
                    .as_ref()
                    .map(|plan| plan.as_ref().ok()?.retained_control_bytes())
                    .unwrap_or(Some(0))?,
            )?
            .checked_add(
                u64::try_from(
                    self.named_catalog
                        .map_or(0, |names| names.catalog_requested_bytes),
                )
                .ok()?,
            )?
            .checked_add(storage::named_error_control_bytes()?)?
            .checked_add(
                self.residency
                    .as_ref()
                    .map(|population| {
                        population.known_control_bytes(self.window_sources.as_ref()?.as_slice())
                    })
                    .unwrap_or(Some(0))?,
            )?
            .checked_add(u64::try_from(fixed.checked_add(allocations)?).ok()?)?
            .checked_add(
                u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
                    PreparationFailure,
                >()?)
                .ok()?,
            )
    }
    /// Native producer-fit composition consumes this exact row against the
    /// already charged registered role arenas and outside-arena owner facts.
    /// These counts alone do not certify successful Graph/Record fit.
    pub(crate) fn neural_native_requirements(&self) -> Option<neural::NeuralNativeRequirements> {
        self.neural.native_requirements()
    }
    pub(crate) fn pending_neural_fit(&self) -> neural::PendingNeuralFit {
        self.neural_fit.requirement()
    }
    pub(crate) fn control_bytes(&self) -> Option<u64> {
        self.selected_stream?;
        if self.retained_source().is_none() && self.foreground_disk_plan().is_none() {
            self.gguf_host_runtime.as_ref()?;
        }
        self.known_control_bytes()?
            .checked_add(
                if self.retained_source().is_some() || self.foreground_disk_plan().is_some() {
                    0
                } else {
                    self.pending_gguf_host_fit().additional_control_bytes()?
                },
            )?
            .checked_add(if self.foreground_disk_plan().is_some() {
                u64::try_from(self.manager.original_dense_controller_control_bytes()?).ok()?
            } else {
                0
            })?
            .checked_add(self.neural_fit.additional_control_bytes()?)?
            .checked_add(match self.retained_source() {
                Some(source) if self.foreground_disk_plan().is_some() => self
                    .residency
                    .as_ref()?
                    .prepared_foreground_payload_control_bytes(
                        self.window_sources.as_ref()?,
                        source,
                        &self.manager,
                    )?,
                Some(source) => self
                    .residency
                    .as_ref()?
                    .prepared_ready_host_payload_control_bytes(
                        self.window_sources.as_ref()?,
                        source,
                        &self.manager,
                    )?,
                None => self
                    .residency
                    .as_ref()?
                    .prepared_payload_control_bytes(self.window_sources.as_ref()?.as_slice())?,
            })
    }
    /// Called outside every session/policy inspection loan, after the genuine
    /// neutral once-only claim. No geometry-only caller can manufacture that set.
    pub(crate) fn prepare_install(
        self,
        original: &OriginalTextPrefillScopeSet,
        step: &InferenceTextStep,
        registration: OriginalOperationRegistration,
        controls: OriginalTextControlGuard,
        host_destinations: Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
    ) -> Result<OriginalOperationBankOwner, Error> {
        original.validate_request(step.request()).map_err(memory)?;
        // Missing fit is not equality evidence: two None facts cannot authorize
        // constructing current-miss source arenas without their reservation.
        let control_bytes = self.control_bytes().ok_or_else(unknown)?;
        if Some(original.facts().plan().geometry()) != self.geometry
            || original.facts().operation_control_bytes() != Some(control_bytes)
        {
            return Err(identity());
        }
        let reservation = step.request().memory_reservation().ok_or_else(identity)?;
        controls.validate_reservation(reservation).map_err(memory)?;
        let source_plan = self
            .source_arenas
            .as_ref()
            .and_then(|plan| plan.as_ref().ok());
        let mut source_bank = host_destinations;
        if let Some(disk) = self.foreground_disk_plan() {
            let bank = source_bank.as_ref().ok_or_else(identity)?;
            if source_plan.is_some()
                || original.facts().source_construction_facts() != disk.source_facts()
                || original.facts().host_destination_facts() != disk.host_facts()
                || !bank.belongs_to(&controls)
                || !bank.matches_facts(disk.host_facts().ok_or_else(unknown)?)
            {
                return Err(identity());
            }
        } else if self.retained_source().is_some() {
            if source_plan.is_some()
                || source_bank.is_some()
                || original.facts().source_construction_facts().is_some()
                || original.facts().host_destination_facts().is_some()
            {
                return Err(identity());
            }
        } else {
            let plan = source_plan.ok_or_else(unknown)?;
            let bank = source_bank.as_ref().ok_or_else(identity)?;
            if original.facts().source_construction_facts() != Some(plan.facts())
                || original.facts().host_destination_facts() != Some(plan.host_facts())
                || !bank.belongs_to(&controls)
                || !bank.matches_facts(plan.host_facts())
            {
                return Err(identity());
            }
        }
        let residency = self.residency.ok_or_else(unknown)?;
        if !self
            .manager
            .matches_operation_source(self.source_manager.as_ref().ok_or_else(unknown)?)
        {
            return Err(identity());
        }
        // Refuse replacement of live storage before any construction. A stale
        // weak installation can only be retired outside the slot's Cell access.
        self.slot.require_bounded_idle()?;
        // The selected plan retained this actual cache owner. Check its admitted
        // birth and original pool without charging another context per request.
        let retained_sources = if let Some(source) = self.retained_source() {
            source.validate_original_policy(
                &self.manager,
                &self.identity,
                self.sources.as_slice(),
            )?;
            Some(source.retain_original_sources(&controls)?)
        } else if self.foreground_disk_plan().is_some() {
            None
        } else {
            let plan = source_plan.ok_or_else(unknown)?;
            plan.validate_cache_origin(&self.manager, reservation)?;
            plan.validate_catalog_origins(reservation)?;
            None
        };
        let PreparedContainers { requests, scopes, pending } = self.prepare_containers(&controls)?;
        let mut prepared = PreparedStorage::new(
            self.units,
            residency,
            self.neural,
            self.gguf_host_runtime.as_ref(),
            self.window_sources.as_ref().ok_or_else(unknown)?.as_slice(),
            &self.manager,
            self.sources.as_slice(),
            &controls,
            self.foreground_disk_plan(),
            reservation,
            if self.foreground_disk_plan().is_some() {
                None
            } else {
                source_plan.zip(source_bank.take())
            },
            if self.foreground_disk_plan().is_some() {
                source_bank.take()
            } else {
                None
            },
        )?;
        let registry = Rc::new(Registry {
            request: step.request().clone().into(),
            scopes: RefCell::new(scopes),
            scope_limit: self.scopes,
            registered_scopes: Cell::new(0),
            active: Cell::new(true),
            entered: Cell::new(false),
            retirement_failure: Cell::new(None),
        });
        let background = prepared.background.take();
        let bank = Rc::new(Bank {
            background: RefCell::new(background),
            prepared: RefCell::new(prepared),
            pending: RefCell::new(pending),
            pending_limit: self.pending,
            binding_row_limit: residency.unprepared.transfer.binding_rows,
            requests,
            registry,
            _retained_sources: retained_sources,
            selected_stream: self.selected_stream.ok_or_else(identity)?,
            identity: self.identity,
        });
        registration.install(&bank.registry)?;
        let view = OriginalOperationProjection {
            value: Rc::downgrade(&bank),
            identity: Arc::clone(&bank.identity),
            controls: OperationControls::Text(controls.clone()),
        };
        let old = self.slot.replace(Some(view));
        drop(old);
        Ok(OriginalOperationBankOwner {
            value: Box::new(OwnedBank {
                bank,
                slot: self.slot,
                registration: Some(registration),
            }),
            controls: OperationControls::Text(controls),
        })
    }
}

struct PreparedContainers<U: 'static> {
    requests: Vec<(OffloadUnitId, u64)>,
    scopes: Vec<safemlx::OriginalScopeObserver>,
    pending: VecDeque<MlxUnitLease<U>>,
}
impl<U: 'static> OriginalOperationPlan<'_, U> {
    fn prepare_containers<C: Clone + std::fmt::Debug + Send + Sync + 'static>(
        &self, controls: &C,
    ) -> Result<PreparedContainers<U>, Error> {
        let mut requests = Vec::new();
        requests
            .try_reserve_exact(self.sources.len())
            .map_err(|e| reserve_error(e, controls))?;
        for id in self.sources.iter() {
            // Exact source bytes, prepared once. Future window consumers borrow
            // a slice; no per-acquire ID clone or Vec allocation remains.
            requests.push((id.clone(), 1));
        }
        let mut scopes = Vec::new();
        scopes
            .try_reserve_exact(self.scopes)
            .map_err(|e| reserve_error(e, controls))?;
        let mut pending = VecDeque::new();
        pending
            .try_reserve_exact(self.pending)
            .map_err(|e| reserve_error(e, controls))?;
        Ok(PreparedContainers { requests, scopes, pending })
    }
}

struct Bank<U: 'static> {
    selected_stream: safemlx::StreamCopyPlan<()>,
    prepared: RefCell<PreparedStorage<U>>,
    background: RefCell<Option<crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>>,
    pending: RefCell<VecDeque<MlxUnitLease<U>>>,
    pending_limit: usize,
    binding_row_limit: usize,
    requests: Vec<(OffloadUnitId, u64)>,
    registry: Rc<Registry>,
    identity: Arc<()>,
    _retained_sources: Option<super::host_workspace::OriginalLayerwiseSourceCustody>,
}
/// The actual Q owner is outside the policy. Escaped weak views cannot keep it
/// installed after request teardown; each view separately covers its weak header.
pub(crate) struct OriginalOperationBankOwner {
    value: Box<dyn ErasedOwner>,
    controls: OperationControls,
}
trait ErasedOwner {
    fn activate(&self, _controls: &OperationControls) -> Result<OriginalOperationActivation, Error> {
        Err(identity())
    }
    fn selected_residency_access(&self,_role:&SpeculativeOperationRole)->Result<OriginalSelectedResidencyAccess,Error> {
        Err(identity())
    }
}
struct OwnedBank<U: 'static> {
    bank: Rc<Bank<U>>,
    slot: InstallSlot<U>,
    registration: Option<OriginalOperationRegistration>,
}
impl<U: 'static> ErasedOwner for OwnedBank<U> {
    fn activate(&self, controls: &OperationControls) -> Result<OriginalOperationActivation, Error> {
        self.activate_text(controls)
    }
    fn selected_residency_access(&self,role:&SpeculativeOperationRole)->Result<OriginalSelectedResidencyAccess,Error> {
        OriginalSelectedResidencyAccess::registered(self.bank.registry.clone(),OperationControls::Speculative(role.clone()))
    }
}
impl<U: 'static> Drop for OwnedBank<U> {
    fn drop(&mut self) {
        let old = self.slot.take();
        if old
            .as_ref()
            .is_some_and(|view| Weak::ptr_eq(&view.value, &Rc::downgrade(&self.bank)))
        {
            drop(old);
        } else {
            self.slot.set(old);
        }
        self.bank.registry.active.set(false);
        if let Some(registration) = &self.registration { registration.clear(&self.bank.registry); }
        // Fields retire after this method: actual bank/unused nodes first,
        // independent slot and registration, then erased outer Q custody last.
    }
}
impl std::fmt::Debug for OriginalOperationBankOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalOperationBankOwner")
            .finish_non_exhaustive()
    }
}
pub(crate) struct OriginalOperationProjection<U: 'static> {
    value: Weak<Bank<U>>,
    identity: Arc<()>,
    controls: OperationControls,
}
impl<U: 'static> Clone for OriginalOperationProjection<U> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            identity: self.identity.clone(),
            controls: self.controls.clone(),
        }
    }
}
impl<U: 'static> OriginalOperationProjection<U> {
    pub(crate) fn access(&self) -> Result<OriginalOperationAccess<U>, Error> {
        let bank = self.value.upgrade().ok_or(Error::PrefillScopeUnavailable)?;
        if !Arc::ptr_eq(&self.identity, &bank.identity) {
            return Err(identity());
        }

        self.controls.validate(&bank.registry)?;
        Ok(OriginalOperationAccess {
            bank,
            controls: self.controls.clone(),
        })
    }
    pub(crate) fn checkout_unit(
        &self,
    ) -> Result<(PreparedUnit<U>, safemlx::OriginalScopeObserver), Error> {
        self.access()?.checkout_unit()
    }
}
/// One actual ordinal's move-only attempt. The registry alias keeps its exact
/// request identity and outlives every unused slot during cancellation/Drop.
/// Actual selected demand window, retaining its own paid closure scratch.
pub(crate) struct OriginalSelectedResidencyAttempt {
    value: storage::PreparedSelectedResidency,
    access: OriginalSelectedResidencyAccess,
}

pub(crate) struct OriginalResidencyAttempt {
    value: storage::PreparedResidencyAttempt,
    registry: Rc<Registry>,
}

/// A temporary strong storage loan. It does not borrow the policy, manager or
/// inspector. Every original producer validates its current registered role.
pub(crate) struct OriginalOperationAccess<U: 'static> {
    bank: Rc<Bank<U>>,
    controls: OperationControls,
}
impl<U: 'static> OriginalOperationAccess<U> {
    pub(crate) fn binding_row_limit(&self) -> Result<usize, Error> {
        self.bank.registry.authenticate()?;
        Ok(self.bank.binding_row_limit)
    }

    pub(crate) fn validate_observer(
        &self,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<(), Error> {
        let current = self.bank.registry.authenticate()?;
        if !current.same_scope(observer) {
            return Err(identity());
        }
        Ok(())
    }
    pub(crate) fn requests(
        &self,
        window: std::ops::Range<usize>,
    ) -> Result<&[(OffloadUnitId, u64)], Error> {
        self.bank.requests.get(window).ok_or_else(identity)
    }
    pub(crate) fn checkout_unit(
        &self,
    ) -> Result<(PreparedUnit<U>, safemlx::OriginalScopeObserver), Error> {
        let observer = self.bank.registry.authenticate()?;
        let slot = self
            .bank
            .prepared
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .units
            .checkout()
            .map_err(|_| Error::PrefillScopeUnavailable)?;
        Ok((slot, observer))
    }
    pub(crate) fn checkout_neural(
        &self,
    ) -> Result<
        (
            neural::PreparedNeuralSubmission,
            safemlx::OriginalScopeObserver,
        ),
        Error,
    > {
        // Authenticate before consuming any once-only slot. A foreign/current
        // mismatch or fenced registry leaves this bank unchanged.
        let observer = self.bank.registry.authenticate()?;
        let slot = self
            .bank
            .prepared
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .neural
            .checkout()
            .map_err(|_| Error::PrefillScopeUnavailable)?;
        Ok((slot, observer))
    }
    pub(crate) fn checkout_residency(
        &self,
        index: usize,
    ) -> Result<OriginalResidencyAttempt, Error> {
        self.bank.registry.authenticate()?;
        let value = self
            .bank
            .prepared
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .checkout_residency(index)?;
        Ok(OriginalResidencyAttempt {
            value,
            registry: Rc::clone(&self.bank.registry),
        })
    }
    pub(crate) fn with_residency<T>(
        &self,
        attempt: &mut OriginalResidencyAttempt,
        operation: impl FnOnce(
            &mut crate::backend::runtime::residency::manager::OriginalResidencySlots<'_>,
            &safemlx::OriginalScopeObserver,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if !Rc::ptr_eq(&attempt.registry, &self.bank.registry) {
            return Err(identity());
        }
        let observer = self.bank.registry.authenticate()?;
        let mut storage = self
            .bank
            .prepared
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        self.controls.validate(&self.bank.registry)?;
        operation(
            &mut storage.residency(&mut attempt.value, self.bank.registry.request.memory_reservation()),
            &observer,
        )
    }

    pub(crate) fn has_background(&self) -> Result<bool, Error> {
        Ok(self.bank.background.try_borrow().map_err(|_| Error::PrefillScopeReentrant)?.is_some())
    }
    pub(crate) fn with_background_window<T, F>(&self, index: usize, execute: F) -> Result<T, Error>
    where F: FnOnce(&crate::backend::runtime::residency::dense_stream::BackgroundHostReadService,
        crate::backend::runtime::residency::manager::PreparedBackgroundHostWindow) -> Result<T, Error> {
        self.bank.registry.authenticate()?;
        self.controls.validate(&self.bank.registry)?;
        // Successful earlier native leases must have passed the exact finish /
        // transfer synchronization / payload release worker before new reads.
        if !self.pending_is_empty()? { self.fence(); return Err(Error::PrefillScopeUnavailable); }
        let mut background = self.bank.background.try_borrow_mut().map_err(|_| Error::PrefillScopeReentrant)?;
        background.as_mut().ok_or_else(identity)?.with_window(index, self.bank.registry.request.memory_reservation(), execute)
    }
    pub(crate) fn with_host_and_device<T>(&self, index: usize, attempt: &mut OriginalResidencyAttempt,
        operation: impl FnOnce(&mut crate::backend::runtime::residency::manager::OriginalResidencySlots<'_>,
            &mut crate::backend::runtime::residency::manager::OriginalResidencySlots<'_>, &safemlx::OriginalScopeObserver) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if !Rc::ptr_eq(&attempt.registry, &self.bank.registry) { return Err(identity()); }
        let observer = self.bank.registry.authenticate()?;
        self.controls.validate(&self.bank.registry)?;
        let mut storage = self.bank.prepared.try_borrow_mut().map_err(|_| Error::PrefillScopeReentrant)?;
        storage.with_host_and_device(index, &mut attempt.value, self.bank.registry.request.memory_reservation(), |host, device| operation(host, device, &observer))
    }
    pub(crate) fn finish_background_forward(&self) -> Result<Option<eredu_core::residency::BackgroundPrefetchReport>, Error> {
        if !self.pending_is_empty()? {
            if let Ok(background) = self.bank.background.try_borrow() {
                if let Some(background) = background.as_ref() { background.close(); }
            }
            return Err(Error::PrefillScopeUnavailable);
        }
        self.controls.validate(&self.bank.registry)?;
        let mut background = self.bank.background.try_borrow_mut().map_err(|_| Error::PrefillScopeReentrant)?;
        match background.as_mut() { Some(value) => value.finish_forward().map(Some), None => Ok(None) }
    }
    pub(crate) fn push_pending(&self, lease: MlxUnitLease<U>) -> Result<(), Error> {
        let mut queue = self
            .bank
            .pending
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        if queue.len() == self.bank.pending_limit {
            return Err(Error::PrefillScopeUnavailable);
        }
        queue.push_back(lease);
        Ok(())
    }
    pub(crate) fn pop_pending(&self) -> Result<Option<MlxUnitLease<U>>, Error> {
        Ok(self
            .bank
            .pending
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .pop_front())
    }
    pub(crate) fn reap_completed(&self) -> Result<(), Error> {
        loop {
            let Some(lease) = self.pop_pending()? else {
                return Ok(());
            };
            let completed = lease.is_complete();
            if !matches!(completed, Ok(true)) {
                self.bank
                    .pending
                    .try_borrow_mut()
                    .map_err(|_| Error::PrefillScopeReentrant)?
                    .push_front(lease);
                return completed.map(|_| ());
            }
            lease.finish()?;
        }
    }
    pub(crate) fn fence(&self) {
        self.bank.registry.active.set(false);
        if let Ok(background) = self.bank.background.try_borrow() {
            if let Some(background) = background.as_ref() { background.close(); }
        }
    }
    pub(crate) fn discard_pending(&self) -> Result<(), Error> {
        while let Some(lease) = self.pop_pending()? {
            drop(lease);
        }
        Ok(())
    }
    pub(crate) fn pending_is_empty(&self) -> Result<bool, Error> {
        Ok(self
            .bank
            .pending
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .is_empty())
    }
}

#[derive(Debug)]
struct PreparationFailure<C = OriginalTextControlGuard> {
    cause: TryReserveError,
    controls: C,
}
impl<C> std::fmt::Display for PreparationFailure<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl<C: std::fmt::Debug> std::error::Error for PreparationFailure<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
fn reserve_error<C: Clone + std::fmt::Debug + Send + Sync + 'static>(cause: TryReserveError, controls: &C) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(PreparationFailure {
            cause,
            controls: controls.clone(),
        }),
        false,
    )
}
fn bank_error<S, F, C: Clone + std::fmt::Debug + Send + Sync + 'static>(
    error: BankPreparationError<S, F, Infallible>,
    controls: &C,
) -> Error {
    let (cause, prefix, factory) = error.into_parts();
    let error = match cause {
        BankPreparationCause::Overflow => overflow(),
        BankPreparationCause::Reserve(cause) => reserve_error(cause, controls),
        BankPreparationCause::Slot { cause, .. } => match cause {},
    };
    // These are never-started empty nodes. Preserve Q through all partial
    // storage retirement, and the actual reserve cause in the escaping error.
    drop(prefix);
    drop(factory);
    error
}

#[cfg(test)]
mod tests;

/// Selected physical policy plan; resident branches own only neural boundaries.
pub(crate) enum SelectedOriginalOperationPlan<'a, U: 'static> {
    Bounded(crate::backend::runtime::execution::generic::OriginalOperationPlan<'a, U>),
    Resident(crate::backend::runtime::execution::generic::ResidentNeuralPlan<U>),
}
impl<'a, U: 'static> SelectedOriginalOperationPlan<'a, U> {
    pub(crate) fn with_preparation_funding(self, funding: Option<&eredu_nn::workspace::HostMetadataFunding>) -> Self {
        match self {
            Self::Bounded(plan) => Self::Bounded(plan.with_preparation_funding(funding)),
            resident => resident,
        }
    }
    pub(crate) fn with_selected_stream(self, stream: &Stream) -> Result<Self, Error> {
        match self {
            Self::Bounded(plan) => plan.with_selected_stream(stream).map(Self::Bounded),
            Self::Resident(plan) => plan.with_selected_stream(stream).map(Self::Resident),
        }
    }

    pub(crate) fn with_foreground_disk_reads(
        self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<Self, Error> {
        match self {
            Self::Bounded(plan) => plan.with_foreground_disk_reads(pool).map(Self::Bounded),
            resident => Ok(resident),
        }
    }
    pub(crate) fn bind_neural_recipe(
        &self,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
    ) -> Result<(), Error> {
        match self {
            Self::Bounded(plan) => plan.bind_neural_recipe(recipe),
            Self::Resident(plan) => plan.bind_neural_recipe(recipe),
        }
    }
    pub(crate) fn with_neural_recipe(
        self,
        recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
    ) -> Result<Self, Error> {
        match self {
            Self::Bounded(plan) => plan.with_neural_recipe(recipe).map(Self::Bounded),
            Self::Resident(plan) => plan.with_neural_recipe(recipe).map(Self::Resident),
        }
    }
    pub(crate) fn with_gguf_host_runtime(
        self,
        runtime: &Result<
            std::rc::Rc<safemlx::PreparedInputRuntime>,
            eredu_core::SharedBackendFailure,
        >,
    ) -> Result<Self, Error> {
        match self {
            Self::Bounded(plan) => plan.with_gguf_host_runtime(runtime).map(Self::Bounded),
            resident => Ok(resident),
        }
    }
    pub(crate) fn with_source_arena_plan(self) -> Result<Self, Error> {
        match self {
            Self::Bounded(plan) => plan.with_source_arena_plan().map(Self::Bounded),
            resident => Ok(resident),
        }
    }
    pub(crate) fn control_bytes(&self) -> Option<u64> {
        match self {
            Self::Bounded(plan) => plan.control_bytes(),
            Self::Resident(plan) => plan.control_bytes(),
        }
    }
    pub(crate) fn source_construction_facts(
        &self,
    ) -> Option<eredu_runtime::working_memory::HostSourceConstructionFacts> {
        match self {
            Self::Bounded(plan) => plan.source_construction_facts(),
            Self::Resident(_) => None,
        }
    }
    pub(crate) fn host_destination_facts(
        &self,
    ) -> Option<eredu_runtime::working_memory::HostDestinationFacts> {
        match self {
            Self::Bounded(plan) => plan.host_destination_facts(),
            Self::Resident(_) => None,
        }
    }
    pub(crate) fn prepare_install(
        self,
        original: &eredu_runtime::working_memory::OriginalTextPrefillScopeSet,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        registration: crate::backend::runtime::execution::generic::OriginalOperationRegistration,
        controls: eredu_runtime::working_memory::OriginalTextControlGuard,
        host: Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
    ) -> Result<
        Option<crate::backend::runtime::execution::generic::OriginalOperationBankOwner>,
        Error,
    > {
        match self {
            Self::Bounded(plan) => plan
                .prepare_install(original, step, registration, controls, host)
                .map(Some),
            Self::Resident(plan) => {
                plan.prepare_install(original, step, registration, controls, host)
            }
        }
    }
}

fn selected_plan_control_bytes<U: 'static>() -> Option<u64> {
    // Descriptive source selection adds no retained owner. Price its actual
    // borrowed hook/argument transports, including the partitioned forwarding
    // branch and resident policy/slot branch maxima.
    type Mechanism = eredu_runtime::GroupSubmissionMechanism;
    let group_source = [
        // Both actual receivers are thin loans of sized runtime/executor
        // owners. The forwarding branch includes both hooks; direct uses one.
        size_of::<(&(), Mechanism)>().max(
            size_of::<(&(), Mechanism)>().checked_mul(2)?),
        size_of::<Mechanism>(), // Selected policy's plan argument.
        // Resident policy then slot, or the single bounded policy entry.
        size_of::<Mechanism>().max(size_of::<Mechanism>().checked_mul(2)?),
        size_of::<Mechanism>(), // NeuralPopulation::from_execution argument.
    ].into_iter().try_fold(0usize, usize::checked_add)?;
    u64::try_from(
        size_of::<SelectedOriginalOperationPlan<'static, U>>()
            .checked_add(group_source)?
            .checked_add(size_of::<usize>())? // This query's group_source local.
            .checked_add(size_of::<Option<SelectedOriginalOperationPlan<'static, U>>>())?
            .checked_add(size_of::<
                Result<SelectedOriginalOperationPlan<'static, U>, Error>,
            >())?,
    )
    .ok()
}

impl<U:'static> OriginalOperationAccess<U> {
    /// Narrow loan of this exact registered operation, independent of unit type.
    pub(crate) fn selected_residency_access(&self)->Result<OriginalSelectedResidencyAccess,Error> {
        OriginalSelectedResidencyAccess::registered(Rc::clone(&self.bank.registry),self.controls.clone())
    }
    pub(crate) fn validate_selected_residency_bank(&self,bank:&eredu_runtime::working_memory::OriginalHostSourceBank)->Result<(),Error> {
        self.selected_residency_access()?.validate_bank(bank)
    }
    pub(crate) fn prepare_selected_residency(&self,manager:&ResidencyManager,
        source:&crate::backend::runtime::residency::manager::SupplementaryResidencySource,
        roots:&[OffloadUnitId],bank:&mut eredu_runtime::working_memory::OriginalHostSourceBank,
    )->Result<OriginalSelectedResidencyAttempt,Error> {
        self.selected_residency_access()?.prepare(manager,source,roots,bank,None)
    }
    pub(crate) fn with_selected_residency<T>(&self,attempt:&mut OriginalSelectedResidencyAttempt,
        execute:impl FnOnce(&mut crate::backend::runtime::residency::manager::OriginalResidencySlots<'_>,
            &safemlx::OriginalScopeObserver)->Result<T,Error>,
    )->Result<T,Error> {self.selected_residency_access()?.with_residency(attempt,execute)}
}
impl OriginalSelectedResidencyAttempt {
    /// The same window factory supplies native transfers, waits, root slots and retries.
    pub(crate) fn native_transfer_population(
        population: crate::backend::runtime::residency::manager::WindowPopulation,
    ) -> Result<(usize, usize, usize, usize), Error> {
        storage::ResidencyPopulation::selected_native_transfers(population)
    }
    /// Complete descriptive constructor size for one exact source population.
    /// A caller's conservative population remains only a byte ceiling; runtime
    /// preparation recomputes the actual root closure before its second debit.
    pub(crate) fn construction_bytes(
        source:&crate::backend::runtime::residency::manager::SupplementaryResidencySource,
        population:crate::backend::runtime::residency::manager::WindowPopulation,
    )->Option<u64> {
        if population.controller_units!=source.source().controller_units {return None;}
        storage::PreparedSelectedResidency::scratch_bytes(source)?
            .checked_add(storage::PreparedSelectedResidency::window_bytes(population)?)
    }
    /// Scratch and exact final window each consume one finite attempt.
    pub(crate) const fn construction_attempts()->usize {2}
}

impl<U:'static> OriginalOperationAccess<U> {
    pub(crate) fn prepare_selected_disk_residency(&self,manager:&ResidencyManager,
        source:&crate::backend::runtime::residency::manager::SupplementaryResidencySource,
        roots:&[OffloadUnitId],bank:&mut eredu_runtime::working_memory::OriginalHostSourceBank,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool,
        capacity:&crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity,
        funding:&eredu_nn::workspace::HostMetadataFunding,
    )->Result<OriginalSelectedResidencyAttempt,Error> {
        self.selected_residency_access()?.prepare(manager,source,roots,bank,Some((pool,capacity,funding)))
    }
}
impl OriginalSelectedResidencyAttempt {
    /// Both quoted producer rows are derived from the same actual selected
    /// manager. Runtime still validates their descriptor source and root list.
    pub(crate) fn disk_construction_bytes(
        source:&crate::backend::runtime::residency::manager::SupplementaryResidencySource,
        population:crate::backend::runtime::residency::manager::WindowPopulation,
        disk:&crate::backend::runtime::residency::manager::ForegroundDiskWindowPlan,
    )->Option<u64> {
        Self::construction_bytes(source,population)?.checked_add(disk.attempt_control_bytes()?)
    }
}
