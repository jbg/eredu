//! MLX realization of the backend-neutral layerwise unit policy.

use eredu_checkpoint::store::RetainedCheckpointSource;
use eredu_runtime::{
    DenseDiskStreamReport, ExecutionUnitAddress, ExecutionUnitLayout, LayerWeightResidency,
    LayerwiseModelMetadata, ResidentLayerGroup, ResidentLayerGroupReport,
};

use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};

use eredu_nn::Parameterized;
use eredu_runtime::LayerwiseAcquireError;
use eredu_runtime::{
    LayeredArchitecture, LayerwisePolicy, OffloadUnit, RuntimeState, WeightBinding,
};
use safemlx::{transforms::async_eval_with_operation_event, OperationEvent, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    ordinary_retirement::OrdinaryRetirement,
    runtime::{
        checkpoint::binding::{
            binding_bytes, build_module_bindings, populate_module_from_lease,
            populate_module_from_lease_excluding, populate_module_from_original_lease_excluding,
        },
        execution::layerwise::{
            validate_device_budget, validate_host_budget, validate_unused, DenseControllerHandle,
            DensePreparedTransfer, DenseStreamController, DenseStreamForwardGuard,
            DenseStreamGroupGuard, DenseTransferWindow, OriginalDenseControllerFacts,
        },
        residency::manager::{
            host_capacity_upper_bound_for_bindings, ResidencyManager, ResidentTransfer,
            ResidentUnitLease,
        },
    },
};
use crate::MlxTensor;
use eredu_core::residency::{
    MemoryTier, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};

/// One populated MLX unit retained with its residency transfer.
enum MlxUnitTransfer {
    Ordinary { _transfer: ResidentTransfer },
    Dense { _transfer: DensePreparedTransfer },
}

mod submission;
pub use submission::MlxUnitLease;
pub(crate) use submission::PreparedUnit;
mod original_operations;
pub(crate) use original_operations::{
    RealtimeNeuralPlan,QualifiedRealtimeNeuralPlan,RealtimeNeuralOwner,
    RealtimeLayerwisePlan,QualifiedRealtimeLayerwisePlan,
    gguf_host_typed, ActivePredictionModuleBank, OriginalOperationBankOwner, OriginalOperationPlan,
    OriginalOperationAccess, OriginalResidencyAttempt, OriginalSelectedResidencyAttempt, OriginalSelectedResidencyAccess, PreparedSelectedResidencyAccess,
    OriginalOperationRegistration, PredictionModuleCall, PredictionModulePlan,
    PredictionModuleProjection, PreparedPredictionModuleBank, RegisteredOriginalScope,
    ResidentNeuralPlan, SelectedOriginalOperationPlan, SpeculativeNeuralOwner, SpeculativeSourcePartition,
};

/// Inspects one retained source transfer under the ordinary completion owner.
/// The callback evaluates exported copies; failures retain the original loan
/// through native terminal evidence just like execution-unit inspection.
pub(crate) fn with_parameter_transfer<T>(
    transfer: ResidentTransfer,
    stream: &Stream,
    operation: impl FnOnce(&ResidentUnitLease) -> Result<T, Error>,
) -> Result<T, Error> {
    transfer.order_after(stream)?;
    let mut loan = MlxUnitLease::new(
        MlxModule::new(()),
        MlxUnitTransfer::Ordinary {
            _transfer: transfer,
        },
    )?;
    let output = operation(loan.population_parts().1)?;
    let marker = safemlx::ops::zeros_dtype(&[1], safemlx::Dtype::Float32, stream)?;
    loan.submitted(async_eval_with_operation_event([&marker])?)?;
    loan.finish()?;
    crate::backend::ordinary_retirement::reclaim_all();
    Ok(output)
}

/// Executes one fallible module operation under its immutable-weight transfer.
/// Changed-state and output roots are returned even after an equation failure;
/// the transfer and those roots remain in the ordinary recovery owner until
/// completion or terminal teardown. A callback error never releases the loan.
pub(crate) fn with_module_transfer<T>(
    transfer: ResidentTransfer,
    stream: &Stream,
    operation: impl FnOnce(&ResidentUnitLease) -> (Result<T, Error>, Vec<MlxTensor>),
) -> Result<T, Error> {
    transfer.order_after(stream)?;
    let mut loan = MlxUnitLease::new(
        MlxModule::new(Vec::<MlxTensor>::new()),
        MlxUnitTransfer::Ordinary {
            _transfer: transfer,
        },
    )?;
    let (outcome, dependencies) = operation(loan.population_parts().1);
    loan.population_parts().0.inner = dependencies;
    let validations = crate::backend::nn::tensor::active_token_validation_arrays();
    let marker = safemlx::ops::zeros_dtype(&[1], safemlx::Dtype::Float32, stream)?;
    let event = async_eval_with_operation_event(
        loan.iter()
            .map(MlxTensor::as_array)
            .chain(validations.iter())
            .chain([&marker]),
    )?;
    loan.submitted(event)?;
    loan.finish()?;
    crate::backend::nn::tensor::validate_active_token_validations()?;
    crate::backend::ordinary_retirement::reclaim_all();
    outcome
}

/// Additional physical owner executed outside the ordinary unit window.
/// The source is exact for this owner, including retained load-time transforms.
pub struct SupplementaryResidencyUnit {
    /// Exact immutable bindings and the stable physical owner identity.
    pub definition: OffloadUnit,
    /// Already prepared source used only for this physical owner.
    pub source: RetainedCheckpointSource,
    /// Whether this owner overlaps each sequential auxiliary invocation.
    pub shared: bool,
}

/// Exact-completion MLX policy over generic parameterized execution units.
pub struct MlxLayerwisePolicy<U: 'static, P = ()> {
    workspace_identity: Arc<()>,
    residency: ResidencyManager,
    store: RetainedCheckpointSource,
    unit_ids: Arc<Vec<OffloadUnitId>>,
    original_operations: std::rc::Rc<original_operations::OriginalOperationSlot<U>>,
    // Optional scalar source description is prepared once alongside ordinary
    // policy sources. Failure preserves the ordinary execution path; original
    // admission still requires complete storage, not this descriptor alone.
    operation_source: Result<
        crate::backend::runtime::residency::manager::OriginalResidencySource,
        crate::backend::runtime::residency::manager::OperationSourceFailure,
    >,
    layout: ExecutionUnitLayout,
    window_depth: usize,
    populator: P,
    _static_leases: OrdinaryRetirement<Vec<ResidentUnitLease>>,
    pending: VecDeque<MlxUnitLease<U>>,
    dense: Option<OrdinaryRetirement<MlxDenseExecution>>,
    // Loading prepares admitted-file metadata once. An ordinary recipe keeps
    // its precise unavailable witness without changing unquoted execution.
    disk_reads: Option<
        Result<
            crate::backend::runtime::residency::manager::PreparedDiskReadPlans,
            Arc<crate::backend::runtime::residency::manager::DiskCopyWorkspaceError>,
        >,
    >,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

struct MlxDenseExecution {
    controller: DenseControllerHandle,
    windows: Vec<Option<DenseTransferWindow>>,
    forward: Option<DenseStreamForwardGuard>,
    groups: Vec<Option<DenseStreamGroupGuard>>,
    prefill: bool,
    aborted: bool,
}

mod resident;
pub use resident::{MlxResidentPolicy, MlxResidentUnit};

mod host_workspace;
pub(crate) use host_workspace::{
    DiskLayerwiseReceipt, LayerwiseWorkspace, LayerwiseWorkspaceIdentity,
};

/// Statically dispatched parameter population used by the MLX policy.
pub trait MlxUnitPopulator<U> {
    /// Whether the selected populator imports the prepared lease unchanged.
    /// This is a source declaration, not proof of residency or native readiness.
    /// Custom overrides remain unqualified until they describe their own source.
    fn preserves_prepared_parameter_source(&self) -> bool { false }

    /// Stable count for this actual immutable override topology. Unknown custom
    /// populators remain unknown even when their current visitor is complete.
    fn retained_value_slot_bound(&self) -> Option<usize> {
        None
    }

    /// Borrows all numerical values retained for future loads, including overrides.
    /// A custom populator must describe these owners before cold admission can
    /// treat its numerical-value inventory as complete.
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool;

    /// Replaces the complete immutable override set used after ordinary population.
    fn publish_parameter_replacements(
        &mut self,
        _values: &std::collections::BTreeMap<String, MlxTensor>,
        _active: bool,
    ) -> bool {
        false
    }

    /// Populates an authenticated prepared host unit without entering an
    /// unqualified custom populator. The row limit is supplied by its Q owner.
    fn populate_original(
        &mut self,
        _unit: &mut MlxModule<U>,
        _lease: &ResidentUnitLease,
        _row_limit: usize,
    ) -> Result<(), Error>
    where
        U: Parameterized<MlxTensor>,
    {
        Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }

    /// Populates the parameters owned by the execution-unit residency policy.
    ///
    /// Most units own every materialized parameter. Architectures with an
    /// independently managed addressable parameter class
    /// override this method with the same exclusion used while binding the
    /// unit. Keeping the ownership decision on the populator makes resident,
    /// bounded, and dense-stream execution follow one contract.
    fn populate(&mut self, unit: &mut MlxModule<U>, lease: &ResidentUnitLease) -> Result<(), Error>
    where
        U: Parameterized<MlxTensor>,
    {
        populate_module_from_lease(unit, lease)?;
        Ok(())
    }
}

impl<U> MlxUnitPopulator<U> for () {
    fn preserves_prepared_parameter_source(&self) -> bool { true }

    fn populate_original(
        &mut self,
        unit: &mut MlxModule<U>,
        lease: &ResidentUnitLease,
        row_limit: usize,
    ) -> Result<(), Error>
    where
        U: Parameterized<MlxTensor>,
    {
        populate_module_from_original_lease_excluding(unit, lease, row_limit, |_| false)?;
        Ok(())
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        Some(0)
    }

    fn visit_retained_values(&self, _: &mut dyn FnMut(&MlxTensor)) -> bool {
        true
    }
}

/// Exact parameter exclusions for units with independently stored members.
#[derive(Clone)]
pub struct MlxSelectiveUnitPopulator {
    excluded: Arc<BTreeSet<String>>,
    replacements: Arc<std::collections::BTreeMap<String, MlxTensor>>,
}

impl MlxSelectiveUnitPopulator {
    /// Creates one generic logical-parameter exclusion set.
    pub fn new(excluded: BTreeSet<String>) -> Self {
        Self {
            excluded: Arc::new(excluded),
            replacements: Arc::new(std::collections::BTreeMap::new()),
        }
    }
}

pub(super) struct ParameterPublisher<'a>(
    pub(super) &'a std::collections::BTreeMap<String, MlxTensor>,
);
impl<'a> eredu_nn::ParameterVisitorMut<'a, MlxTensor> for ParameterPublisher<'_> {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadata, value: &'a mut MlxTensor) {
        if let Some(replacement) = self.0.get(metadata.id.as_str()) {
            *value = replacement.clone();
        }
    }
}

impl<U> MlxUnitPopulator<U> for MlxSelectiveUnitPopulator {
    fn preserves_prepared_parameter_source(&self) -> bool {
        // Exclusions remove independently owned parameters from both the exact
        // prepared bindings and populate_original's lease visitor. Every value
        // this policy imports still comes unchanged from that prepared lease.
        // Replacements introduce another producer and need their own source.
        self.replacements.is_empty()
    }

    fn populate_original(
        &mut self,
        unit: &mut MlxModule<U>,
        lease: &ResidentUnitLease,
        row_limit: usize,
    ) -> Result<(), Error>
    where
        U: Parameterized<MlxTensor>,
    {
        // Replacement handles are a distinct producer from lease imports.
        // The selected prepared host source currently carries no overrides.
        if !self.replacements.is_empty() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ));
        }
        populate_module_from_original_lease_excluding(unit, lease, row_limit, |name| {
            self.excluded.contains(name)
        })?;
        Ok(())
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        Some(self.replacements.len())
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        for value in self.replacements.values() {
            visitor(value);
        }
        true
    }
    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, MlxTensor>,
        active: bool,
    ) -> bool {
        self.replacements = Arc::new(if active {
            values.clone()
        } else {
            Default::default()
        });
        true
    }

    fn populate(&mut self, unit: &mut MlxModule<U>, lease: &ResidentUnitLease) -> Result<(), Error>
    where
        U: Parameterized<MlxTensor>,
    {
        populate_module_from_lease_excluding(unit, lease, |name| self.excluded.contains(name))?;
        unit.inner
            .visit_parameters_mut(&mut ParameterPublisher(&self.replacements));
        Ok(())
    }
}

/// Builds one unloaded unit through the neutral architecture's declared flat layout.
pub fn construct_architecture_unit<A, S>(
    architecture: &A,
    layout: &ExecutionUnitLayout,
    ordinal: usize,
    stream: &Stream,
    _state: std::marker::PhantomData<S>,
) -> Result<A::Unit, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
{
    let address = layout.address(ordinal).ok_or_else(|| {
        Error::ArchitectureModel(format!(
            "architecture unit ordinal {ordinal} is outside 0..{}",
            layout.len()
        ))
    })?;
    architecture
        .build_unit(address.group(), address.index(), stream)
        .map_err(|error| Error::Other(Box::new(error)))
}

impl<U: 'static, P> MlxLayerwisePolicy<U, P> {
    /// Shares the existing ledger with separately invoked prepared modules.
    pub(crate) fn residency_manager(&self) -> &ResidencyManager {
        &self.residency
    }
    /// Creates a bounded policy over validated ordered residency units.
    pub fn new(
        residency: ResidencyManager,
        store: impl Into<RetainedCheckpointSource>,
        unit_ids: Vec<OffloadUnitId>,
        layout: ExecutionUnitLayout,
        window_depth: usize,
        populator: P,
        static_leases: Vec<ResidentUnitLease>,
        dense: Option<Arc<DenseStreamController>>,
        sample_mlx_memory: bool,
        sample_process_memory: bool,
    ) -> Result<Self, Error> {
        Self::new_with_controller(
            residency,
            store,
            unit_ids,
            layout,
            window_depth,
            populator,
            static_leases,
            dense.map(DenseControllerHandle::Ordinary),
            sample_mlx_memory,
            sample_process_memory,
        )
    }
    pub(crate) fn new_with_controller(
        residency: ResidencyManager,
        store: impl Into<RetainedCheckpointSource>,
        unit_ids: Vec<OffloadUnitId>,
        layout: ExecutionUnitLayout,
        window_depth: usize,
        populator: P,
        static_leases: Vec<ResidentUnitLease>,
        dense: Option<DenseControllerHandle>,
        sample_mlx_memory: bool,
        sample_process_memory: bool,
    ) -> Result<Self, Error> {
        let store = store.into();
        if unit_ids.is_empty() {
            return Err(Error::Parallel(
                "generic MLX layerwise policy requires at least one unit".into(),
            ));
        }
        if window_depth == 0 {
            return Err(Error::Parallel(
                "generic MLX layerwise policy window depth must be nonzero".into(),
            ));
        }
        if unit_ids.len() != layout.len() {
            return Err(Error::Parallel(
                "layerwise units differ from the execution layout".into(),
            ));
        }
        let disk_reads = if residency.original_foreground_disk_descriptors().is_some() {
            None
        } else if dense
            .as_ref()
            .is_some_and(|controller| controller.is_foreground())
        {
            use crate::backend::runtime::{
                checkpoint::recipe::PreparedDirectReadError,
                residency::manager::DiskCopyWorkspaceError,
            };
            use eredu_runtime::working_memory::WorkingMemoryError;
            Some(match residency.prepare_disk_reads(&unit_ids) {
                Ok(reads) => Ok(reads),
                Err(
                    error @ DiskCopyWorkspaceError::Unproved {
                        source: WorkingMemoryError::UnknownBound,
                        ..
                    },
                )
                | Err(
                    error @ DiskCopyWorkspaceError::Direct(PreparedDirectReadError::Unpriced {
                        source: WorkingMemoryError::UnknownBound,
                        ..
                    }),
                ) => Err(Arc::new(error)),
                Err(error) => return Err(Error::Other(Box::new(error))),
            })
        } else {
            None
        };
        let operation_source =
            residency.prepare_original_operation_source(&unit_ids, &layout, window_depth);
        Ok(Self {
            workspace_identity: Arc::new(()),
            residency,
            store,
            unit_ids: Arc::new(unit_ids),
            original_operations: std::rc::Rc::new(original_operations::OriginalOperationSlot::new()),
            operation_source,
            layout: layout.clone(),
            window_depth,
            populator,
            _static_leases: OrdinaryRetirement::new(static_leases),
            pending: VecDeque::new(),
            disk_reads,
            dense: dense.map(|controller| {
                let prepared = controller.is_source_prepared();
                OrdinaryRetirement::new(MlxDenseExecution {
                    controller,
                    windows: if prepared {
                        Vec::new()
                    } else {
                        (0..layout.group_count()).map(|_| None).collect()
                    },
                    forward: None,
                    groups: if prepared {
                        Vec::new()
                    } else {
                        (0..layout.group_count()).map(|_| None).collect()
                    },
                    prefill: false,
                    aborted: false,
                })
            }),
            sample_mlx_memory,
            sample_process_memory,
        })
    }

    fn reap_completed(&mut self) -> Result<(), Error> {
        if let Some(original) = self.original_operation_projection() {
            return original.access()?.reap_completed();
        }
        loop {
            let Some(lease) = self.pending.front() else {
                return Ok(());
            };
            if !lease.is_complete()? {
                return Ok(());
            }
            self.pending.pop_front().expect("completed unit").finish()?;
            crate::backend::ordinary_retirement::reclaim_all();
        }
    }

    fn drain_one(&mut self) -> Result<(), Error> {
        if let Some(original) = self.original_operation_projection() {
            if let Some(lease) = original.access()?.pop_pending()? {
                lease.finish()?;
            }
            return Ok(());
        }
        if let Some(lease) = self.pending.pop_front() {
            lease.finish()?;
            crate::backend::ordinary_retirement::reclaim_all();
        }
        Ok(())
    }

    fn pending_is_empty(&self) -> Result<bool, Error> {
        match self.original_operation_projection() {
            Some(original) => original.access()?.pending_is_empty(),
            None => Ok(self.pending.is_empty()),
        }
    }

    fn drain(&mut self) -> Result<(), Error> {
        while !self.pending_is_empty()? {
            self.drain_one()?;
        }
        Ok(())
    }

    fn trim_device_window(
        &self,
        current: usize,
        address: ExecutionUnitAddress,
    ) -> Result<(), Error> {
        if self.layout.address(current) != Some(address) {
            return Err(Error::Parallel(
                "execution window address differs from its layout".into(),
            ));
        }
        let range = self
            .layout
            .window_range(
                current,
                std::num::NonZeroUsize::new(self.window_depth).expect("validated window depth"),
            )
            .expect("validated window ordinal");
        if let Some(persistent) = self.residency.original_foreground_persistent_units() {
            for (index, id) in self.unit_ids.iter().enumerate() {
                if !range.contains(&index) && !persistent.clone().any(|owner| owner == id) {
                    self.residency.evict(id, MemoryTier::Device)?;
                }
            }
        } else {
            let persistent = self.residency.admitted_disk_persistent_units();
            for (index, id) in self.unit_ids.iter().enumerate() {
                if !range.contains(&index) && !persistent.contains(id) {
                    self.residency.evict(id, MemoryTier::Device)?;
                }
            }
        }
        Ok(())
    }

    /// Advances the admitted direct window only after its preceding consumer
    /// and transfer have completed. Ready entries inside the next range remain
    /// valid; retired cache copies outside it must not accumulate in roomy tiers.
    fn prepare_admitted_disk_window(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
    ) -> Result<bool, Error> {
        if !self.residency.admitted_disk_route_active() {
            return Ok(false);
        }
        if self.layout.address(ordinal) != Some(address)
            || !self.dense_window_matches_layout()
            || !self
                .dense
                .as_ref()
                .is_some_and(|dense| dense.controller.is_foreground())
        {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )));
        }
        let range = self
            .layout
            .window_range(
                ordinal,
                std::num::NonZeroUsize::new(self.window_depth).expect("validated depth"),
            )
            .expect("validated ordinal");
        // This validates the exact receipt's window before build(stream) emits
        // even a placeholder seed. Acquisition remains blocked outside it.
        if !self
            .residency
            .set_admitted_disk_window(&self.unit_ids[range])?
        {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )));
        }
        self.drain()?;
        crate::backend::ordinary_retirement::reclaim_all();
        self.trim_device_window(ordinal, address)?;
        crate::backend::ordinary_retirement::reclaim_all();
        Ok(true)
    }

    fn dense_window_matches_layout(&self) -> bool {
        let selected = std::num::NonZeroUsize::new(self.window_depth).expect("validated depth");
        let actual = std::num::NonZeroUsize::new(eredu_runtime::DENSE_TRANSFER_WINDOW)
            .expect("fixed dense transfer depth is nonzero");
        (0..self.layout.len()).all(|ordinal| {
            self.layout.window_range(ordinal, selected) == self.layout.window_range(ordinal, actual)
        })
    }

    /// Returns the checkpoint source backing this layerwise policy.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.store.as_ref()
    }

    /// Clones the shared checkpoint source backing this layerwise policy.
    pub fn retained_checkpoint_store(&self) -> RetainedCheckpointSource {
        self.store.clone()
    }

    /// Returns current weight-residency accounting.
    pub fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        self.residency.report().map_err(Into::into)
    }

    /// Returns dense disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.dense
            .as_ref()
            .map(|dense| dense.controller.report(&self.residency))
            .transpose()
    }

    /// Returns the number of pinned static-weight leases.
    pub fn static_lease_count(&self) -> usize {
        self._static_leases.len()
    }

    fn execution_group(&self, group: usize) -> Result<ResidentLayerGroup, Error> {
        let range = self
            .layout
            .group_range(group)
            .ok_or_else(|| Error::Parallel(format!("unknown execution group {group}")))?;
        let id = self
            .layout
            .group_id(group)
            .expect("validated layout names every execution group")
            .as_str();
        let depth = self.window_depth.min(range.len());
        ResidentLayerGroup::new(id, self.unit_ids[range].iter().cloned(), depth)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Returns residency reports for every semantic execution group.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        (0..self.layout.group_count())
            .map(|group| {
                self.execution_group(group)?
                    .report(&self.residency)
                    .map_err(Into::into)
            })
            .collect()
    }

    /// Populates every unit once and converts the bounded loader into a
    /// permanently resident policy without changing the architecture loop.
    pub fn into_resident<A, S>(
        mut self,
        architecture: &A,
        stream: &Stream,
        _state: std::marker::PhantomData<S>,
    ) -> Result<MlxResidentPolicy<U>, Error>
    where
        U: Parameterized<MlxTensor>,
        P: MlxUnitPopulator<U>,
        A: LayeredArchitecture<MlxNeuralBackend, S, Unit = U>,
        S: RuntimeState<MlxNeuralBackend>,
        A::Error: std::fmt::Display,
    {
        self.drain()?;
        let requests = self
            .unit_ids
            .iter()
            .cloned()
            .map(|id| (id, 1))
            .collect::<Vec<_>>();
        let mut transfer = self
            .residency
            .acquire_many_with_transfer(&requests, MemoryTier::Device)?;
        transfer.order_after(stream)?;
        let mut units = Vec::with_capacity(self.unit_ids.len());
        for (index, lease) in transfer.leases().iter().enumerate() {
            let address = self
                .layout
                .address(index)
                .expect("validated layout covers every resident unit");
            let mut unit = MlxModule::new(
                architecture
                    .build_unit(address.group(), address.index(), stream)
                    .map_err(|error| Error::Other(Box::new(error)))?,
            );
            self.populator.populate(&mut unit, lease)?;
            units.push(Some(unit));
        }
        transfer.synchronize()?;
        Ok(MlxResidentPolicy {
            units,
            residency: self.residency.clone(),
            store: self.store.clone(),
            unit_ids: self.unit_ids.as_ref().clone(),
            layout: self.layout.clone(),
            window_depth: self.window_depth,
            _transfer: transfer,
            original_neural: original_operations::ResidentNeuralSlot::from_slot(
                self.original_operations.clone(),
            ),
        })
    }

    /// Populates architecture units already constructed by neutral
    /// orchestration and converts this loader into a resident policy.
    pub fn into_resident_units(
        mut self,
        units: Vec<U>,
        stream: &Stream,
    ) -> Result<MlxResidentPolicy<U>, Error>
    where
        U: Parameterized<MlxTensor>,
        P: MlxUnitPopulator<U>,
    {
        self.drain()?;
        if units.len() != self.unit_ids.len() {
            return Err(Error::ArchitectureModel(format!(
                "neutral resident construction supplied {} units for layout length {}",
                units.len(),
                self.unit_ids.len()
            )));
        }
        let requests = self
            .unit_ids
            .iter()
            .cloned()
            .map(|id| (id, 1))
            .collect::<Vec<_>>();
        let mut transfer = self
            .residency
            .acquire_many_with_transfer(&requests, MemoryTier::Device)?;
        transfer.order_after(stream)?;
        let mut populated = Vec::with_capacity(units.len());
        for (unit, lease) in units.into_iter().zip(transfer.leases()) {
            let mut unit = MlxModule::new(unit);
            self.populator.populate(&mut unit, lease)?;
            populated.push(Some(unit));
        }
        // Loaded resident parameters become cold admission evidence. Establish
        // their exact completion here rather than forcing a later inventory to
        // poll/evaluate before it can price the starting residency.
        transfer.synchronize()?;
        Ok(MlxResidentPolicy {
            units: populated,
            residency: self.residency.clone(),
            store: self.store.clone(),
            unit_ids: self.unit_ids.as_ref().clone(),
            layout: self.layout.clone(),
            window_depth: self.window_depth,
            _transfer: transfer,
            original_neural: original_operations::ResidentNeuralSlot::from_slot(
                self.original_operations.clone(),
            ),
        })
    }
}

fn populate_selected_parameterized<M: Parameterized<MlxTensor>>(
    module: &mut M,
    lease: &ResidentUnitLease,
    selected: &BTreeSet<String>,
) -> Result<(), Error> {
    // A pipeline rank can retain unloaded handles for another rank's static
    // roles. Bind exactly the planned destinations, keeping every selected
    // destination mandatory even if its materialized value is missing.
    populate_module_from_lease_excluding(module, lease, |name| !selected.contains(name))
        .map_err(Error::from)
}

fn largest_window_bytes(layer_bytes: &[u64], depth: usize) -> Result<u64, Error> {
    let mut largest = 0u64;
    for start in 0..layer_bytes.len() {
        let mut current = 0u64;
        for bytes in layer_bytes.iter().skip(start).take(depth) {
            current = current
                .checked_add(*bytes)
                .ok_or_else(|| Error::Parallel("generic device window bytes overflowed".into()))?;
        }
        largest = largest.max(current);
    }
    Ok(largest)
}

/// Derives the canonical flat unit layout from a neutral architecture.
pub fn architecture_execution_layout<A, S>(architecture: &A) -> Result<ExecutionUnitLayout, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::Other(Box::new(error)))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            architecture
                .group_unit_count(group)
                .map_err(|error| Error::Other(Box::new(error)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts).map_err(|error| Error::Other(Box::new(error)))
}

/// Cold preparation for bounded and resident policies.
mod preparation;
pub(crate) use preparation::{
    prepare_foreground_layerwise_manager, prepare_layerwise_manager, prepare_manager_from_declarations,
    prepare_layerwise_declarations,
    prepare_layerwise_policy_with_prepared_manager, PreparedLayerwiseManager,
};
pub use preparation::{
    prepare_layerwise_policy, prepare_layerwise_policy_from_bindings,
    prepare_layerwise_policy_with_bindings, prepare_layerwise_policy_with_supplementary_bindings,
};

mod bounded;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod managed_disk_tests;
