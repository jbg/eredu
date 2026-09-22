//! Native binding metadata and slot access for architecture-enumerated modules.
use super::*;
use crate::backend::nn::shared::visit_parameter_map;
use crate::backend::runtime::{
    checkpoint::binding::populate_module_from_ordinary_lease,
    execution::generic::{with_module_transfer, SupplementaryResidencyUnit},
    residency::{manager::ResidencyManager, storage::RetainedStorage},
};
use eredu_architectures::prediction_extension::{
    MaterializedPredictionExecutor, PredictionModuleVisitor, PredictionResourceVisitor,
};
use eredu_core::residency::{MemoryTier, OffloadUnitId};
use eredu_nn::{ParameterMetadata, ParameterSlotVisitor, ParameterVisitorMut};
use eredu_runtime::parameter_operations::{
    ParameterSlotOperation, PreparedParameterLocation, PreparedParameterSlot,
};
use std::convert::Infallible;
use std::sync::OnceLock;

mod inspection;
pub(super) mod original;
mod workspace;
pub(crate) use original::{
    clear_prediction_module_bank, inspect_prediction_module_plan, install_prediction_module_bank,
};
pub(crate) use workspace::{
    prepare_prediction_parameters, MlxWorkspacePredictionParameterSource,
    NativePredictionParameters, PredictionParameterStorage,
};

type ManagerSlot = Arc<OnceLock<ResidencyManager>>;

/// Exact owners registered before target residency initialization.
#[derive(Default)]
pub(in crate::composition::mlx::replicated_text) struct PredictionResidency {
    pub units: Vec<SupplementaryResidencyUnit>,
    managers: Vec<ManagerSlot>,
}
impl PredictionResidency {
    pub fn install(&self, manager: &ResidencyManager) -> Result<(), Error> {
        for slot in &self.managers {
            slot.set(manager.clone()).map_err(|_| {
                Error::ArchitectureModel("prediction residency was already installed".into())
            })?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct MlxPredictionModule<M> {
    pub inner: M,
    pub parameters: Vec<PreparedParameterSlot>,
    pub tasks: Arc<Vec<ReplicatedTextMaterializationTask>>,
    pub(super) layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    pub materialization: eredu_runtime::WeightMaterializationReport,
    pub(super) source: RetainedCheckpointSource,
    pub(super) bindings: Vec<eredu_runtime::WeightBinding>,
    pub(super) residency: eredu_runtime::LayerWeightResidency,
    pub(super) shared: bool,
    pub(super) manager: ManagerSlot,
    pub(super) id: Option<OffloadUnitId>,
    pub(super) placeholders: BTreeMap<String, MlxTensor>,
    pub(super) replacements:
        eredu_runtime::parameter_operations::ParameterReplacementValues<MlxTensor>,
    pub(super) stream: Stream,
    pub(super) original:
        Option<crate::backend::runtime::execution::generic::PredictionModuleProjection>,
}
impl<M: std::fmt::Debug> std::fmt::Debug for MlxPredictionModule<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxPredictionModule")
            .field("inner", &self.inner)
            .field("parameters", &self.parameters)
            .field("residency", &self.residency)
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}
impl<M: Parameterized<MlxTensor>> MlxPredictionModule<M> {
    fn register_residency(
        &mut self,
        ordinal: usize,
        registry: &mut PredictionResidency,
    ) -> Result<(), Error> {
        if self.id.is_some() {
            return Err(Error::ArchitectureModel(
                "prediction module residency registered twice".into(),
            ));
        }
        let id = OffloadUnitId::new(format!("prediction.module.{ordinal:05}"))?;
        self.id = Some(id.clone());
        registry.units.push(SupplementaryResidencyUnit {
            definition: eredu_runtime::OffloadUnit::new(id, self.bindings.clone())?,
            source: self.source.clone(),
            shared: self.shared,
        });
        registry.managers.push(Arc::clone(&self.manager));
        Ok(())
    }

    fn count_parameter_owners(
        &self,
        ordinal: usize,
        counts: &mut ParameterOwnerCounts,
        guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        counts.prediction_module(self.id.is_some(), self.manager.get().is_some())?;
        counts.observe_source(
            ParameterOwnerRole::PredictionInner,
            Some(ordinal),
            &self.inner,
            guard,
        )?;
        counts.observe_map(
            ParameterOwnerRole::PredictionPlaceholder,
            Some(ordinal),
            &self.placeholders,
            guard,
        )?;
        counts.observe_replacements(
            ParameterOwnerRole::PredictionReplacement,
            Some(ordinal),
            &self.replacements,
            guard,
        )
    }

    /// Reads actual retained storage, including overrides while the module is
    /// unloaded. Native allocation identities deduplicate manager/module aliases.
    fn retained_storage(&self) -> Result<RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        storage.include_retained_values::<Error>(|visitor| {
            let complete = self.inner.visit_retained_values(visitor);
            for value in self.placeholders.values().chain(self.replacements.values()) {
                visitor(value);
            }
            Ok(complete)
        })?;
        storage.include_checkpoint_source(self.source.as_ref())?;
        match self.manager.get() {
            Some(manager) => manager.collect_retained_storage(storage)?,
            None => storage.mark_incomplete(),
        }
        Ok(())
    }

    pub(super) fn invoke<O>(
        &mut self,
        stream: &Stream,
        operation: impl FnOnce(&mut M) -> (Result<O, Error>, Vec<MlxTensor>),
    ) -> Result<O, Error> {
        if safemlx::OriginalScopeObserver::try_current()?.is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )
            .at_speculative_stage("ordinary prediction module entry"));
        }
        let manager = self
            .manager
            .get()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "prediction module has no target residency authority".into(),
                )
            })?
            .clone();
        let id = self
            .id
            .as_ref()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "prediction module has no physical residency identity".into(),
                )
            })?
            .clone();
        let transfer =
            manager.acquire_many_with_transfer(&[(id.clone(), 1)], MemoryTier::Device)?;
        let outcome = with_module_transfer(transfer, stream, |lease| {
            if let Err(error) = populate_module_from_ordinary_lease(&mut self.inner, lease) {
                return (Err(error.into()), Vec::new());
            }
            self.inner
                .visit_parameters_mut(&mut PublishReplacements(&self.replacements));
            operation(&mut self.inner)
        });
        // Clear every native module handle, including derived operator caches.
        // Unresolved native work retains its own roots and the real lease in
        // the completion recovery owner; unloading is not a completion signal.
        self.inner
            .visit_parameters_mut(&mut Publish(&self.placeholders));
        if !self.residency.is_fully_resident() {
            let eviction = manager.evict(&id, MemoryTier::Device);
            if outcome.is_ok() {
                eviction?;
            }
        }
        outcome
    }
}

/// Only module parameter components are observed. Prototype callbacks retain
/// explicit separate occurrence counts; their state/manager storage is unpriced.
pub(in crate::composition::mlx::replicated_text) fn count_parameter_owners<A, P>(
    extension: &P,
    counts: &mut ParameterOwnerCounts,
    guard: &mut safemlx::RuntimeCallGuard,
) -> Result<(), ParameterOwnerSourceError>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Count<'a, 'g> {
        counts: &'a mut ParameterOwnerCounts,
        guard: &'g mut safemlx::RuntimeCallGuard,
    }
    impl PredictionResourceVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        for Count<'_, '_>
    {
        type Error = ParameterOwnerSourceError;
        fn module<M: Parameterized<MlxTensor>>(
            &mut self,
            ordinal: usize,
            module: &MlxPredictionModule<M>,
        ) -> Result<(), Self::Error> {
            module.count_parameter_owners(ordinal, self.counts, self.guard)
        }
        fn pooling_state(
            &mut self,
            _state: &super::OwnedPredictionCache<
                crate::backend::runtime::cache::state::MlxPoolingAttentionCache,
            >,
        ) -> Result<(), Self::Error> {
            self.counts.pooling_prototype()
        }
        fn model_state(&mut self, _state: &MlxHybridState) -> Result<(), Self::Error> {
            self.counts.model_prototype()
        }
    }
    extension.visit_retained_resources(&mut Count { counts, guard })
}

/// Inventory the architecture-declared extension owners without executing a
/// prediction operation or creating a lane from a retained prototype.
pub(in crate::composition::mlx::replicated_text) fn retained_storage<A, P>(
    extension: &P,
) -> Result<RetainedStorage, Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    let mut storage = RetainedStorage::default();
    collect_retained_storage::<A, P>(extension, &mut storage)?;
    Ok(storage)
}

pub(in crate::composition::mlx::replicated_text) fn collect_retained_storage<A, P>(
    extension: &P,
    storage: &mut RetainedStorage,
) -> Result<(), Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Collect<'a>(&'a mut RetainedStorage);
    impl PredictionResourceVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        for Collect<'_>
    {
        type Error = Error;

        fn module<M: Parameterized<MlxTensor>>(
            &mut self,
            _ordinal: usize,
            module: &MlxPredictionModule<M>,
        ) -> Result<(), Error> {
            module.collect_retained_storage(self.0)?;
            Ok(())
        }

        fn pooling_state(
            &mut self,
            state: &super::OwnedPredictionCache<
                crate::backend::runtime::cache::state::MlxPoolingAttentionCache,
            >,
        ) -> Result<(), Error> {
            self.0.include_retained_values::<Error>(|visitor| {
                eredu_runtime::RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(
                    state.inner(),
                    visitor,
                );
                Ok(true)
            })?;
            if let Some(manager) = state.inner().residency_manager() {
                manager.collect_retained_storage(self.0)?;
            }
            Ok(())
        }

        fn model_state(&mut self, state: &MlxHybridState) -> Result<(), Error> {
            state.collect_retained_storage(self.0)?;
            // A dormant prototype owns its own layout and fixed layer/role
            // tables, independently of the target's live decoder state.
            MlxStateMechanisms::collect_retained_host_storage(state, self.0)?;
            Ok(())
        }
    }
    let mut collect = Collect(storage);
    extension.visit_retained_resources(&mut collect)?;
    Ok(())
}

pub(in crate::composition::mlx::replicated_text) fn residency<A, P>(
    extension: &mut P,
) -> Result<PredictionResidency, Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Collect(PredictionResidency);
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Collect {
        type Error = Error;
        fn visit<M: Parameterized<MlxTensor>>(
            &mut self,
            ordinal: usize,
            module: &mut MlxPredictionModule<M>,
        ) -> Result<(), Error> {
            module.register_residency(ordinal, &mut self.0)
        }
    }
    let mut collect = Collect(PredictionResidency::default());
    extension.visit_modules(&mut collect)?;
    Ok(collect.0)
}

impl<M> AsMut<M> for MlxPredictionModule<M> {
    fn as_mut(&mut self) -> &mut M {
        &mut self.inner
    }
}
impl<M> std::ops::Deref for MlxPredictionModule<M> {
    type Target = M;
    fn deref(&self) -> &M {
        &self.inner
    }
}
impl<M> std::ops::DerefMut for MlxPredictionModule<M> {
    fn deref_mut(&mut self) -> &mut M {
        &mut self.inner
    }
}
impl<M: Parameterized<MlxTensor>> Parameterized<MlxTensor> for MlxPredictionModule<M> {
    fn visit_parameter_sources<'a, V: eredu_nn::ParameterSourceVisitor<'a, MlxTensor>>(
        &'a self,
        visitor: &mut V,
    ) -> Result<(), eredu_nn::ParameterSourceError> {
        let mut __source_result = Ok(());

        __source_result = __source_result.and(self.inner.visit_parameter_sources(visitor));

        __source_result
    }
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, MlxTensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.inner.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.inner.set_trainable(trainable);
    }
}

pub(in crate::composition::mlx::replicated_text) fn collect<A, P>(
    extension: &mut P,
    slots: &mut Vec<PreparedParameterSlot>,
    tasks: &mut Vec<ReplicatedTextMaterializationTask>,
) -> Result<eredu_runtime::WeightMaterializationReport, Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Collect<'a> {
        slots: &'a mut Vec<PreparedParameterSlot>,
        tasks: &'a mut Vec<ReplicatedTextMaterializationTask>,
        report: &'a mut eredu_runtime::WeightMaterializationReport,
    }
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Collect<'_> {
        type Error = Error;
        fn visit<M: Parameterized<MlxTensor>>(
            &mut self,
            ordinal: usize,
            module: &mut MlxPredictionModule<M>,
        ) -> Result<(), Error> {
            self.report.merge(module.materialization.clone());
            for slot in &module.parameters {
                if self
                    .slots
                    .iter()
                    .any(|existing| existing.parameter.id == slot.parameter.id)
                {
                    return Err(Error::ArchitectureModel(format!(
                        "prediction parameter {} has multiple prepared owners",
                        slot.parameter.id.as_str()
                    )));
                }
                let mut slot = slot.clone();
                slot.location = PreparedParameterLocation::Prediction { module: ordinal };
                self.slots.push(slot);
            }
            for task in module.tasks.iter() {
                match self
                    .tasks
                    .iter()
                    .find(|existing| existing.name() == task.name())
                {
                    Some(existing) if existing != task => {
                        return Err(Error::ArchitectureModel(
                            "prediction materialization task disagrees with retained owner".into(),
                        ))
                    }
                    Some(_) => {}
                    None => self.tasks.push(task.clone()),
                }
            }
            Ok(())
        }
    }
    let mut report = Default::default();
    extension.visit_modules(&mut Collect {
        slots,
        tasks,
        report: &mut report,
    })?;
    slots.sort_unstable_by(|a, b| a.parameter.id.cmp(&b.parameter.id));
    Ok(report)
}

struct Slots<'a>(&'a mut dyn ParameterSlotVisitor<MlxTensor>);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Slots<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        self.0.visit_slot(metadata, value);
    }
}

pub(in crate::composition::mlx::replicated_text) fn visit<A, P>(
    extension: &mut P,
    visitor: &mut dyn ParameterSlotVisitor<MlxTensor>,
) where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Visit<'a>(&'a mut dyn ParameterSlotVisitor<MlxTensor>);
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Visit<'_> {
        type Error = Infallible;
        fn visit<M: Parameterized<MlxTensor>>(
            &mut self,
            _: usize,
            module: &mut MlxPredictionModule<M>,
        ) -> Result<(), Infallible> {
            module.inner.visit_parameters_mut(&mut Slots(self.0));
            Ok(())
        }
    }
    match extension.visit_modules(&mut Visit(visitor)) {
        Ok(()) => {}
        Err(never) => match never {},
    }
}

pub(in crate::composition::mlx::replicated_text) fn with_slots<A, P>(
    extension: &mut P,
    ordinal: usize,
    operation: &mut ParameterSlotOperation<'_, MlxTensor, Error>,
    preparation: Option<&crate::backend::runtime::execution::generic::MlxParameterPreparation<'_>>,
) -> Result<bool, Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Selected<'a, 'b, 'c> {
        ordinal: usize,
        found: bool,
        operation: &'a mut ParameterSlotOperation<'b, MlxTensor, Error>,
        preparation: &'a crate::backend::runtime::execution::generic::MlxParameterPreparation<'c>,
    }
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        for Selected<'_, '_, '_>
    {
        type Error = Error;
        fn visit<M: Parameterized<MlxTensor>>(
            &mut self,
            ordinal: usize,
            module: &mut MlxPredictionModule<M>,
        ) -> Result<(), Error> {
            if self.ordinal == ordinal {
                if self.found {
                    return Err(Error::ArchitectureModel(
                        "prediction module ordinal is duplicated".into(),
                    ));
                }
                self.found = true;
                inspection::with_slots(module, self.operation, self.preparation)?;
            }
            Ok(())
        }
    }
    let mut selected = Selected {
        ordinal,
        found: false,
        operation,
        preparation: preparation.ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?,
    };
    extension.visit_modules(&mut selected)?;
    Ok(selected.found)
}

pub(in crate::composition::mlx::replicated_text) struct Publish<'a>(
    pub &'a BTreeMap<String, MlxTensor>,
);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Publish<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        if let Some(replacement) = self.0.get(metadata.id().as_str()) {
            *value = replacement.clone();
        }
    }
}

struct PublishReplacements<'a>(
    &'a eredu_runtime::parameter_operations::ParameterReplacementValues<MlxTensor>,
);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for PublishReplacements<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        if let Some(replacement) = self.0.get(metadata.id().as_str()) {
            *value = replacement.clone();
        }
    }
}

/// Lends future-loader sources; prepared publication owns every displaced root.
pub(in crate::composition::mlx::replicated_text) fn visit_publication<A, P>(
    extension: &mut P,
    visitor: &mut dyn eredu_runtime::parameter_operations::ParameterPublication<MlxTensor>,
) where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Modules<'a>(
        &'a mut dyn eredu_runtime::parameter_operations::ParameterPublication<MlxTensor>,
    );
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Modules<'_> {
        type Error = Infallible;
        fn visit<M: Parameterized<MlxTensor>>(
            &mut self,
            _: usize,
            module: &mut MlxPredictionModule<M>,
        ) -> Result<(), Infallible> {
            self.0.replacement_source(&mut module.replacements);
            Ok(())
        }
    }
    match extension.visit_modules(&mut Modules(visitor)) {
        Ok(()) => {}
        Err(never) => match never {},
    }
}

/// Shape-preserving unloaded values backed by completed scalar storage. Keeping
/// the constructor's lazy full-shape tensors would retain unmeasured graphs;
/// evaluating those tensors would instead allocate another full parameter set.
fn placeholder(value: &MlxTensor, stream: &Stream) -> Result<MlxTensor, Error> {
    let scalar = safemlx::ops::zeros_dtype(&[], value.as_array().dtype(), stream)?;
    let view = safemlx::ops::broadcast_to(&scalar, value.as_array().shape(), stream)?;
    view.evaluated()?;
    Ok(MlxTensor::from_array(view))
}

pub(super) fn placeholders<M: Parameterized<MlxTensor>>(
    module: &mut M,
    stream: &Stream,
) -> Result<BTreeMap<String, MlxTensor>, Error> {
    struct Collect<'a> {
        values: BTreeMap<String, MlxTensor>,
        stream: &'a Stream,
        failure: Option<Error>,
    }
    impl<'a> eredu_nn::ParameterVisitor<'a, MlxTensor> for Collect<'_> {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a MlxTensor) {
            if self.failure.is_some() {
                return;
            }
            match placeholder(value, self.stream) {
                Ok(value) => {
                    self.values.insert(metadata.id().as_str().to_owned(), value);
                }
                Err(error) => self.failure = Some(error),
            }
        }
    }
    let mut collect = Collect {
        values: BTreeMap::new(),
        stream,
        failure: None,
    };
    module.visit_parameters(&mut collect)?;
    if let Some(error) = collect.failure {
        return Err(error);
    }
    module.visit_parameters_mut(&mut Publish(&collect.values));
    Ok(collect.values)
}

#[cfg(test)]
#[path = "parameters/storage_tests.rs"]
mod storage_tests;

#[cfg(test)]
mod owner_source_tests;
