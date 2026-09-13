//! Native binding metadata and slot access for architecture-enumerated modules.
use super::*;
use crate::backend::runtime::{
    checkpoint::binding::populate_module_from_lease,
    execution::generic::{with_module_transfer, SupplementaryResidencyUnit},
    residency::manager::ResidencyManager,
};
use eredu_architectures::prediction_extension::{
    MaterializedPredictionExecutor, PredictionModuleVisitor,
};
use eredu_core::residency::{MemoryTier, OffloadUnitId};
use eredu_nn::{ParameterMetadata, ParameterSlotVisitor, ParameterVisitorMut};
use eredu_runtime::parameter_operations::{
    ParameterSlotOperation, PreparedParameterLocation, PreparedParameterSlot,
};
use std::convert::Infallible;
use std::sync::OnceLock;

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
    pub tasks: Vec<ReplicatedTextMaterializationTask>,
    pub materialization: eredu_runtime::WeightMaterializationReport,
    pub(super) source: SharedCheckpointSource,
    pub(super) bindings: Vec<eredu_runtime::WeightBinding>,
    pub(super) residency: eredu_runtime::LayerWeightResidency,
    pub(super) shared: bool,
    pub(super) manager: ManagerSlot,
    pub(super) id: Option<OffloadUnitId>,
    pub(super) placeholders: BTreeMap<String, MlxTensor>,
    pub(super) replacements: BTreeMap<String, MlxTensor>,
    pub(super) stream: Stream,
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
    pub(super) fn invoke<O>(
        &mut self,
        stream: &Stream,
        operation: impl FnOnce(&mut M) -> (Result<O, Error>, Vec<MlxTensor>),
    ) -> Result<O, Error> {
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
            if let Err(error) = populate_module_from_lease(&mut self.inner, lease) {
                return (Err(error.into()), Vec::new());
            }
            self.inner
                .visit_parameters_mut(&mut Slots(&mut Publish(&self.replacements)));
            operation(&mut self.inner)
        });
        // Clear every native module handle, including derived operator caches.
        // Unresolved native work retains its own roots and the real lease in
        // the completion recovery owner; unloading is not a completion signal.
        self.inner
            .visit_parameters_mut(&mut Slots(&mut Publish(&self.placeholders)));
        if !self.residency.is_fully_resident() {
            let eviction = manager.evict(&id, MemoryTier::Device);
            if outcome.is_ok() {
                eviction?;
            }
        }
        outcome
    }
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
            if module.id.is_some() {
                return Err(Error::ArchitectureModel(
                    "prediction module residency registered twice".into(),
                ));
            }
            let id = OffloadUnitId::new(format!("prediction.module.{ordinal:05}"))?;
            module.id = Some(id.clone());
            self.0.units.push(SupplementaryResidencyUnit {
                definition: eredu_runtime::OffloadUnit::new(id, module.bindings.clone())?,
                source: Arc::clone(&module.source),
                shared: module.shared,
            });
            self.0.managers.push(Arc::clone(&module.manager));
            Ok(())
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
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, MlxTensor>>(
        &'a self,
        visitor: &mut V,
    ) {
        self.inner.visit_parameters(visitor);
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
            for task in &module.tasks {
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
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
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
) -> Result<bool, Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Selected<'a, 'b> {
        ordinal: usize,
        found: bool,
        operation: &'a mut ParameterSlotOperation<'b, MlxTensor, Error>,
    }
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        for Selected<'_, '_>
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
                let stream = module.stream.clone();
                module.invoke(&stream, |inner| {
                    let result = (self.operation)(&mut |visitor| {
                        inner.visit_parameters_mut(&mut Slots(visitor))
                    });
                    (result, Vec::new())
                })?;
            }
            Ok(())
        }
    }
    let mut selected = Selected {
        ordinal,
        found: false,
        operation,
    };
    extension.visit_modules(&mut selected)?;
    Ok(selected.found)
}

pub(in crate::composition::mlx::replicated_text) struct Publish<'a>(
    pub &'a BTreeMap<String, MlxTensor>,
);
impl ParameterSlotVisitor<MlxTensor> for Publish<'_> {
    fn visit_slot(&mut self, metadata: ParameterMetadata, value: &mut MlxTensor) {
        if let Some(replacement) = self.0.get(metadata.id.as_str()) {
            *value = replacement.clone();
        }
    }
}

/// Infallible publication after the enclosing atomic target/bank transaction.
/// The immutable source stays unchanged; each later loan reapplies this set.
pub(in crate::composition::mlx::replicated_text) fn publish<A, P>(
    extension: &mut P,
    values: &BTreeMap<String, MlxTensor>,
    active: bool,
) where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct PublishModules<'a> {
        values: &'a BTreeMap<String, MlxTensor>,
        active: bool,
    }
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        for PublishModules<'_>
    {
        type Error = Infallible;
        fn visit<M: Parameterized<MlxTensor>>(
            &mut self,
            _: usize,
            module: &mut MlxPredictionModule<M>,
        ) -> Result<(), Infallible> {
            module.replacements = if self.active {
                module
                    .parameters
                    .iter()
                    .filter_map(|slot| {
                        self.values
                            .get(slot.parameter.id.as_str())
                            .map(|value| (slot.parameter.id.as_str().to_owned(), value.clone()))
                    })
                    .collect()
            } else {
                BTreeMap::new()
            };
            Ok(())
        }
    }
    match extension.visit_modules(&mut PublishModules { values, active }) {
        Ok(()) => {}
        Err(never) => match never {},
    }
}

pub(super) fn placeholders<M: Parameterized<MlxTensor>>(module: &M) -> BTreeMap<String, MlxTensor> {
    struct Collect(BTreeMap<String, MlxTensor>);
    impl<'a> eredu_nn::ParameterVisitor<'a, MlxTensor> for Collect {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a MlxTensor) {
            self.0
                .insert(metadata.id.as_str().to_owned(), value.clone());
        }
    }
    let mut collect = Collect(BTreeMap::new());
    module.visit_parameters(&mut collect);
    collect.0
}
