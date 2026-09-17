//! Exact supplementary-module occurrences using the existing residency worker.
use super::*;
use crate::backend::runtime::residency::manager::{SupplementaryResidencySource, WindowPopulation};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataFunding};
use eredu_runtime::working_memory::{
    OriginalEmbeddedSpeculativeRole, OriginalOperationMetadataCustody,
};
use safemlx::{OriginalScopeObserver, StreamCopyPlan};
mod roots;
use roots::{ModuleRoots, NestedRoots};

/// A recorded physical invocation, not an inventory row or an architecture depth.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PredictionModuleCall {
    pub(crate) source_ordinal: usize,
    pub(crate) retained_parameter_copies: usize,
    pub(crate) retained_roots: usize,
    pub(crate) completion: usize,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct PlanningFailure<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    funding: WorkspaceMetadataFunding,
}
fn planned_error<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    funding: &WorkspaceMetadataFunding,
) -> Error {
    Error::Neural(funding.metadata_source(PlanningFailure {
        cause,
        funding: funding.clone(),
    }))
}

/// Owned cold plan. Each ordered window is copied from the authenticated unique
/// source inventory; repeated calls remain repeated windows and slot populations.
pub(crate) struct PredictionModulePlan {
    manager: ResidencyManager,
    source: SupplementaryResidencySource,
    layerwise: super::super::host_workspace::LayerwiseWorkspace,
    calls: Vec<PredictionModuleCall>,
    windows: Vec<WindowPopulation>,
    population: ResidencyPopulation,
    disk: Option<PreparedSpeculativeForegroundSource>,
    validations: usize,
    selected_stream: StreamCopyPlan<()>,
    bound: Option<(
        eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
        eredu_runtime::speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    )>,
    funding: WorkspaceMetadataFunding,
}
impl PredictionModulePlan {
    pub(crate) fn inspect(
        manager: &ResidencyManager,
        source: &SupplementaryResidencySource,
        calls: Vec<PredictionModuleCall>,
        validations: usize,
        stream: &Stream,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let funding = context.metadata_funding().ok_or_else(unknown)?;
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<ResidencyPopulation>(),
            size_of::<Option<PreparedSpeculativeForegroundSource>>(),
            size_of::<PredictionModuleCall>(),
            size_of::<Vec<WindowPopulation>>(),
            size_of::<StreamCopyPlan<()>>(),
            size_of::<Result<StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<crate::backend::runtime::residency::manager::OperationWindows>(),
            source.projection_control_bytes().ok_or_else(overflow)?,
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        manager
            .validate_supplementary_source(source)
            .map_err(|cause| planned_error(cause, &funding))?;
        let layerwise =
            super::super::host_workspace::LayerwiseWorkspace::from_supplementary_source(
                manager, source, context,
            )?;
        let source_windows = source.source().window_owner();
        if !layerwise.has_original_ready_host_source(manager, &source_windows)
            && !layerwise.has_original_foreground_source(manager, &source_windows)
        {
            return Err(identity().at_speculative_stage("prediction source producer qualification"));
        }
        let selected_stream =
            StreamCopyPlan::capture(stream).map_err(|cause| planned_error(cause, &funding))?;
        let declared = source.source().windows();
        let mut windows = context.metadata_vec(calls.len()).map_err(Error::from)?;
        for (index, call) in calls.iter().enumerate() {
            if call.completion >= calls.len()
                || calls[..index]
                    .iter()
                    .any(|prior| prior.completion == call.completion)
            {
                return Err(identity().at_speculative_stage("prediction module completion permutation"));
            }
            let window = *declared.get(call.source_ordinal).ok_or_else(|| identity().at_speculative_stage("prediction module declared window"))?;
            if window.requested != 1
                || window.request_start != call.source_ordinal
                || window.request_end != call.source_ordinal.checked_add(1).ok_or_else(overflow)?
            {
                return Err(identity().at_speculative_stage("prediction singleton source window"));
            }
            call.retained_roots
                .checked_add(validations)
                .filter(|count| *count != 0)
                .ok_or_else(unknown)?;
            windows.push(window);
        }
        let population = ResidencyPopulation::from_window_forwards(
            source.source().controller_units,
            &windows,
            1,
        )?;
        let disk = if source.foreground().is_some() {
            let mut disk = PreparedSpeculativeForegroundSource::new(
                manager,
                pool,
                &windows,
                source.ids(),
                population,
                &funding,
            )?;
            // Nested shared invocations keep outer transfers alive. The same
            // request planner receives the actual completion permutation.
            disk.set_nested_completion_order(calls.iter().map(|call| call.completion))?;
            Some(disk)
        } else {
            None
        };
        Ok(Self {
            manager: manager.clone(),
            source: source.clone(),
            layerwise,
            calls,
            windows,
            population,
            disk,
            validations,
            selected_stream,
            bound: None,
            funding,
        })
    }
    pub(crate) fn source_facts(
        &self,
    ) -> Result<Option<eredu_runtime::working_memory::HostSourceConstructionFacts>, Error> {
        self.disk
            .as_ref()
            .map(|disk| disk.plan().source_facts().ok_or_else(unknown))
            .transpose()
    }
    pub(crate) fn boundary_roots(&self, completion: Option<usize>) -> Result<Vec<usize>, Error> {
        let count = self
            .calls
            .len()
            .checked_add(usize::from(completion.is_some()))
            .ok_or_else(overflow)?;
        let mut roots = self.funding.metadata_vec(count).map_err(Error::from)?;
        for call in &self.calls {
            roots.push(
                call.retained_roots
                    .checked_add(self.validations)
                    .ok_or_else(overflow)?,
            );
        }
        if let Some(count) = completion {
            if count == 0 {
                return Err(identity());
            }
            roots.push(count);
        }
        Ok(roots)
    }
    /// The equation's exact nested-root distribution is bound separately by the
    /// phase. These are actual canonical source copies and transfer waits only.
    pub(crate) fn bind_source_recipe(
        &mut self,
        recipe: &mut crate::backend::nn::workspace::EmbeddedEquationRecipe,
        completion_roots: Option<usize>,
    ) -> Result<(), Error> {
        if self.bound.is_some() {
            return Err(identity());
        }
        let roots = recipe.prediction_boundary_roots().ok_or_else(identity)?;
        if roots.len()
            != self
                .calls
                .len()
                .checked_add(usize::from(completion_roots.is_some()))
                .ok_or_else(overflow)?
        {
            return Err(identity());
        }
        for (index, call) in self.calls.iter().enumerate() {
            if roots[index]
                != call
                    .retained_roots
                    .checked_add(self.validations)
                    .ok_or_else(overflow)?
            {
                return Err(identity());
            }
        }
        if completion_roots.is_some_and(|count| roots.last() != Some(&count)) {
            return Err(identity());
        }
        if !recipe.matches_prediction_boundaries(roots, 0) {
            return Err(identity());
        }
        self.layerwise
            .validate_supplementary_policy(&self.manager, &self.source)
            .map_err(|cause| cause.at_speculative_stage("prediction supplementary source policy"))?;
        let p = self.population;
        // Actual ordered calls include repeated shared-module invocations.
        let retained_handle_copies = self.calls.iter().try_fold(0usize, |sum, call|
            sum.checked_add(call.retained_parameter_copies)).ok_or_else(overflow)?;
        recipe.with_native_recipe(|recipe| {
            recipe.bind_host_transfer_population(
                recipe.plan().geometry(),
                1,
                p.transfers,
                p.observations,
                p.unprepared.transfer.output_arrays,
                storage::TRANSFERS_PER_NONEMPTY_WINDOW,
                &self.windows,
            )?;
            match (&self.disk, self.source.host()) {
                (Some(disk), None) => {
                    let plan = disk.plan();
                    recipe.bind_foreground_disk_copies(
                        self.source.foreground().ok_or_else(identity)?.source(),
                        plan.forward_population().ok_or_else(unknown)?,
                        plan.population().ok_or_else(unknown)?,
                    )?;
                }
                (None, Some(host)) => recipe.bind_host_copies(host)
                    .map_err(|cause| cause.at_speculative_stage("prediction source-copy recipe"))?,
                _ => return Err(identity()),
            }
            recipe.bind_retained_parameter_slots(&self.layerwise, retained_handle_copies)
                .map_err(|cause| cause.at_speculative_stage("prediction retained source slots"))?;
            Ok(())
        })?;
        self.bound = Some((recipe.plan().clone(), recipe.workspace()));
        Ok(())
    }
    pub(crate) fn control_bytes(&self) -> Option<u64> {
        let p = self.population;
        let units = storage::unit_layout::<Vec<MlxTensor>, _>(
            self.calls.len(),
            &storage::unit_factory::<Vec<MlxTensor>>(None),
        )?;
        let empty_neural = empty_neural();
        let mut bytes = p
            .known_control_bytes(&self.windows)?
            .checked_add(u64::try_from(call::preparation_error_control_bytes()?).ok()?)?
            .checked_add(units)?
            .checked_add(empty_neural.rust_control_bytes()?)?
            .checked_add(self.layerwise.original_pin_control_bytes()?)?
            .checked_add(storage::named_error_control_bytes()?)?
            .checked_add(
                u64::try_from(self.source.source().named_layout().catalog_requested_bytes).ok()?,
            )?
            .checked_add(
                self.disk
                    .as_ref()
                    .map(|disk| disk.plan().operation_control_bytes())
                    .unwrap_or(Some(0))?,
            )?;
        let fixed = [
            size_of::<Self>(),
            size_of::<Inner>(),
            size_of::<PreparedPredictionModuleBank>(),
            size_of::<ActivePredictionModuleBank>(),
            size_of::<PredictionModuleProjection>(),
            size_of::<Option<PredictionModuleProjection>>(),
            size_of::<CallSlot>(),
            size_of::<Option<CallSlot>>(),
            size_of::<PreparedStorage<Vec<MlxTensor>>>(),
            size_of::<Result<PreparedStorage<Vec<MlxTensor>>, Error>>(),
            size_of::<std::cell::RefMut<'_, PreparedStorage<Vec<MlxTensor>>>>(),
            size_of::<std::cell::RefMut<'_, Vec<Option<CallSlot>>>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<OriginalEmbeddedSpeculativeRole>(),
            size_of::<StreamCopyPlan<()>>(),
            self.selected_stream.source_comparison_control_bytes()?,
            size_of::<storage::ForegroundDiskSourceTicket>(),
            size_of::<Option<storage::ForegroundDiskSourceTicket>>(),
            size_of::<super::super::host_workspace::OriginalLayerwiseSourceCustody>(),
            size_of::<Vec<(OffloadUnitId, u64)>>(),
            Layout::array::<Option<CallSlot>>(self.calls.len())
                .ok()?
                .size(),
            Layout::array::<(OffloadUnitId, u64)>(self.source.ids().len())
                .ok()?
                .size(),
            rc_layout::<Inner>()?,
        ];
        let fixed = fixed
            .into_iter()
            .try_fold(std::mem::size_of_val(&fixed), usize::checked_add)?;
        bytes = bytes.checked_add(u64::try_from(fixed).ok()?)?;
        for id in self.source.ids() {
            bytes = bytes.checked_add(u64::try_from(id.as_str().len()).ok()?)?;
        }
        for call in &self.calls {
            let capacity = call.retained_roots.checked_add(self.validations)?;
            bytes = bytes
                .checked_add(
                    u64::try_from(ModuleRoots::control_bytes(
                        call.retained_roots,
                        self.validations,
                    )?)
                    .ok()?,
                )?
                .checked_add(u64::try_from(NestedRoots::control_bytes(capacity)?).ok()?)?
                .checked_add(
                    u64::try_from(call_control_bytes(
                        self.population.unprepared.transfer.binding_rows,
                    )?)
                    .ok()?,
                )?
                .checked_add(u64::try_from(self.source.original_eviction_control_bytes()?).ok()?)?;
            bytes = bytes.checked_add(
                u64::try_from(self.selected_stream.source_comparison_control_bytes()?).ok()?,
            )?;
        }
        Some(bytes)
    }
    pub(crate) fn prepare(
        self,
        role: OriginalEmbeddedSpeculativeRole,
        stream: &Stream,
    ) -> Result<PreparedPredictionModuleBank, Error> {
        let (plan, workspace) = self.bound.as_ref().ok_or_else(identity)?;
        if !self.selected_stream.matches_source(stream) {
            return Err(identity());
        }
        role.validate_plan(plan).map_err(memory)?;
        role.validate_invocation(workspace.invocation())
            .map_err(memory)?;
        self.layerwise
            .validate_supplementary_policy(&self.manager, &self.source)
            .map_err(|cause| cause.at_speculative_stage("prediction supplementary source policy"))?;
        role.claim_neural_bank(self.control_bytes().ok_or_else(unknown)?)
            .map_err(memory)?;
        let custody: OriginalOperationMetadataCustody = role.budget_custody().into();
        let error_custody = custody.clone();
        let result = (|| {
            let retained = self.layerwise.retain_ready_host_sources(custody.clone())?;
            let disk_ticket = match (
                &self.disk,
                role.take_host_source_constructions().map_err(memory)?,
            ) {
                (Some(disk), Some(bank))
                    if bank.matches_facts(disk.plan().source_facts().ok_or_else(unknown)?) =>
                {
                    Some(storage::ForegroundDiskSourceTicket::Source {
                        bank,
                        custody: role.budget_custody().into(),
                    })
                }
                (None, None) => None,
                _ => return Err(identity()),
            };
            let prepared = PreparedStorage::new_with_custody(
                self.calls.len(),
                self.population,
                empty_neural(),
                None,
                &self.windows,
                &self.manager,
                self.source.ids(),
                &custody,
                None,
                self.disk.as_ref().map(|disk| disk.plan()),
                None,
                None,
                disk_ticket,
            )?;
            let mut requests = Vec::new();
            requests
                .try_reserve_exact(self.source.ids().len())
                .map_err(|cause| planned_error(cause, &self.funding))?;
            for id in self.source.ids() {
                requests.push((id.clone(), 1));
            }
            let mut slots = Vec::new();
            slots
                .try_reserve_exact(self.calls.len())
                .map_err(|cause| planned_error(cause, &self.funding))?;
            for call in &self.calls {
                let roots =
                    ModuleRoots::prepare(call.retained_roots, self.validations, &self.funding)?;
                let completion = NestedRoots::try_new(
                    call.retained_roots
                        .checked_add(self.validations)
                        .ok_or_else(overflow)?,
                    custody.clone(),
                )
                .map_err(|failure| {
                    let (cause, owner) = failure.into_parts();
                    let error = planned_error(cause, &self.funding);
                    drop(owner);
                    error
                })?;
                slots.push(Some(CallSlot { roots, completion }));
            }
            let selected_stream = self.selected_stream;
            Ok(PreparedPredictionModuleBank(Rc::new(Inner {
                prepared: RefCell::new(prepared),
                slots: RefCell::new(slots),
                requests,
                started: Cell::new(0),
                completed: Cell::new(0),
                observer: RefCell::new(None),
                active: Cell::new(false),
                selected_stream,
                role,
                plan: self,
                _retained: retained,
            })))
        })();
        result.map_err(|cause| call::retained_error(cause, error_custody))
    }
}
fn empty_neural() -> NeuralPopulation {
    NeuralPopulation {
        submissions: 0,
        per_forward: 0,
        shape: crate::backend::nn::shared::NeuralSubmissionShape::new(1, 0)
            .expect("one borrowed root shape"),
    }
}
struct CallSlot {
    roots: ModuleRoots,
    completion: NestedRoots,
}
struct Inner {
    prepared: RefCell<PreparedStorage<Vec<MlxTensor>>>,
    slots: RefCell<Vec<Option<CallSlot>>>,
    requests: Vec<(OffloadUnitId, u64)>,
    started: Cell<usize>,
    completed: Cell<usize>,
    observer: RefCell<Option<OriginalScopeObserver>>,
    active: Cell<bool>,
    selected_stream: StreamCopyPlan<()>,
    role: OriginalEmbeddedSpeculativeRole,
    plan: PredictionModulePlan,
    _retained: super::super::host_workspace::OriginalLayerwiseSourceCustody,
}
pub(crate) struct PreparedPredictionModuleBank(Rc<Inner>);
pub(crate) struct ActivePredictionModuleBank(Rc<Inner>);
#[derive(Clone)]
pub(crate) struct PredictionModuleProjection {
    value: Weak<Inner>,
    custody: OriginalOperationMetadataCustody,
}
impl std::fmt::Debug for PredictionModuleProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictionModuleProjection")
            .finish_non_exhaustive()
    }
}
impl PreparedPredictionModuleBank {
    pub(crate) fn activate(
        &self,
        scope: &safemlx::SubmissionScope,
    ) -> Result<ActivePredictionModuleBank, Error> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.belongs_to(scope)
            || self.0.active.get()
            || self.0.started.get() != 0
            || self.0.observer.borrow().is_some()
        {
            return Err(identity());
        }
        *self.0.observer.borrow_mut() = Some(current);
        self.0.active.set(true);
        Ok(ActivePredictionModuleBank(Rc::clone(&self.0)))
    }
    pub(crate) fn validate_complete(&self) -> Result<(), Error> {
        if self.0.started.get() != self.0.plan.calls.len()
            || self.0.completed.get() != self.0.plan.calls.len()
        {
            return Err(identity());
        }
        Ok(())
    }
}
impl ActivePredictionModuleBank {
    pub(crate) fn projection(&self) -> PredictionModuleProjection {
        PredictionModuleProjection {
            value: Rc::downgrade(&self.0),
            custody: self.0.role.budget_custody().into(),
        }
    }
}
impl PredictionModuleProjection {
    pub(crate) fn same_bank(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.value, &other.value)
    }
}
impl Drop for ActivePredictionModuleBank {
    fn drop(&mut self) {
        self.0.active.set(false);
    }
}

mod call;
use call::call_control_bytes;
