//! Exact embedded prediction parameter and state binding contracts.

use crate::parameter_operations::{PreparedParameterLocation, PreparedParameterSlot};
use eredu_checkpoint::{quantization::WeightMaterializationReport, store::SharedCheckpointSource};
use eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization;
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_core::BackendFailure;
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Parameterized, Tensor,
};

use crate::{LocalModelLayout, PredictionModelState, ReplicatedTextMaterializationTask};

use eredu_core::residency::{parameter_binding::OffloadUnit, OffloadUnitId};
use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

/// Portable driver for the selected predictor's weight mechanisms.
pub type PredictionResidencyDriver<B, X> = crate::residency::ResidencyDriver<
    <X as NativePredictionMaterialization<B>>::ResidencyMechanisms,
>;

/// One prepared architecture module with portable parameter residency and replacement state.
#[derive(Clone)]
pub struct BoundPredictionModule<T, U, R, C> {
    pub(crate) inner: U,
    pub(crate) parameters: Vec<PreparedParameterSlot>,
    pub(crate) tasks: Vec<ReplicatedTextMaterializationTask>,
    pub(crate) materialization: WeightMaterializationReport,
    residency: PredictionModuleResidency<R>,
    placeholders: BTreeMap<String, T>,
    pub(crate) replacements: BTreeMap<String, T>,
    context: C,
}

/// Portable module paired with exact native tensor, storage, and queue descriptors.
pub type PredictionModule<B, M, U> = BoundPredictionModule<
    <B as eredu_nn::NeuralBackend>::Tensor,
    U,
    PredictionResidencyDriver<B, M>,
    <M as NativePredictionMaterialization<B>>::ModuleContext,
>;

impl<T, U, R, C> AsMut<U> for BoundPredictionModule<T, U, R, C> {
    fn as_mut(&mut self) -> &mut U {
        &mut self.inner
    }
}

impl<T, U, R, C> std::fmt::Debug for BoundPredictionModule<T, U, R, C>
where
    U: std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BoundPredictionModule")
            .field("module", &self.inner)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

fn placeholder_slots<T: eredu_nn::Tensor + Clone, U: Parameterized<T>>(
    module: &U,
) -> BTreeMap<String, T> {
    struct Collect<T>(BTreeMap<String, T>);
    impl<'a, T: Clone + 'a> eredu_nn::ParameterVisitor<'a, T> for Collect<T> {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadata, value: &'a T) {
            self.0
                .insert(metadata.id.as_str().to_owned(), value.clone());
        }
    }
    let mut slots = Collect(BTreeMap::new());
    module.visit_parameters(&mut slots);
    slots.0
}

/// Placement and installed residency authority for an architecture-owned prediction unit.
#[derive(Clone)]
pub struct PredictionModuleResidency<R> {
    source: SharedCheckpointSource,
    bindings: Vec<WeightBinding>,
    role: PredictionModuleRole,
    placement: crate::LayerWeightResidency,
    owner: Arc<OnceLock<R>>,
    id: Option<OffloadUnitId>,
}

impl<R> PredictionModuleResidency<R> {
    /// Retains the exact bindings and placement until portable registration.
    pub fn new(
        source: SharedCheckpointSource,
        bindings: Vec<WeightBinding>,
        role: PredictionModuleRole,
        placement: crate::LayerWeightResidency,
    ) -> Self {
        Self {
            source,
            bindings,
            role,
            placement,
            owner: Arc::new(OnceLock::new()),
            id: None,
        }
    }

    fn register(
        &mut self,
        ordinal: usize,
    ) -> Result<crate::SupplementaryResidencyUnit, BackendFailure> {
        if self.id.is_some() {
            return Err(BackendFailure::from_error(std::io::Error::other(
                "prediction module residency registered twice",
            )));
        }
        let id = OffloadUnitId::new(format!("prediction.module.{ordinal:05}"))
            .map_err(BackendFailure::from_error)?;
        let definition = OffloadUnit::new(id.clone(), self.bindings.clone())
            .map_err(BackendFailure::from_error)?;
        self.id = Some(id);
        Ok(crate::SupplementaryResidencyUnit {
            definition,
            source: self.source.clone(),
            shared: self.role == PredictionModuleRole::Shared,
        })
    }

    /// Borrows the installed owner and its exact module placement.
    pub fn installed(
        &self,
    ) -> Result<(&R, &OffloadUnitId, crate::LayerWeightResidency), BackendFailure> {
        let owner = self.owner.get().ok_or_else(|| {
            BackendFailure::from_error(std::io::Error::other(
                "prediction module has no target residency authority",
            ))
        })?;
        let id = self.id.as_ref().ok_or_else(|| {
            BackendFailure::from_error(std::io::Error::other(
                "prediction module has no physical residency identity",
            ))
        })?;
        Ok((owner, id, self.placement))
    }
}

/// Supplementary residency registration for the prediction modules of one target.
pub struct PredictionResidency<R> {
    units: Vec<crate::SupplementaryResidencyUnit>,
    owners: Vec<Arc<OnceLock<R>>>,
}

impl<R> Default for PredictionResidency<R> {
    fn default() -> Self {
        Self {
            units: Vec::new(),
            owners: Vec::new(),
        }
    }
}

impl<R: Clone> PredictionResidency<R> {
    /// Takes the unit definitions used to build the target's residency plan.
    pub fn take_units(&mut self) -> Vec<crate::SupplementaryResidencyUnit> {
        std::mem::take(&mut self.units)
    }

    /// Attaches the initialized target owner to each registered prediction unit.
    pub fn attach(&self, owner: &R) -> Result<(), BackendFailure> {
        for slot in &self.owners {
            slot.set(owner.clone()).map_err(|_| {
                BackendFailure::from_error(std::io::Error::other(
                    "prediction residency was already installed",
                ))
            })?;
        }
        Ok(())
    }
}

/// Registers exact supplementary residency units in the architecture's traversal order.
pub fn prepare_prediction_residency<A, B, X, P>(
    prediction: &mut P,
) -> Result<PredictionResidency<PredictionResidencyDriver<B, X>>, X::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    X: PredictionExtensionMaterializer<B>,
    P: super::MaterializedPredictionExecutor<A, B, X>,
{
    struct Collect<R>(PredictionResidency<R>);
    impl<B, X> PredictionModuleVisitor<B, X> for Collect<PredictionResidencyDriver<B, X>>
    where
        B: BlockwiseAttentionBackend
            + DistributedNeuralBackend
            + GroupedNeuralBackend
            + HyperNeuralBackend,
        X: PredictionExtensionMaterializer<B>,
    {
        type Error = X::Error;

        fn visit<M: Parameterized<B::Tensor>>(
            &mut self,
            ordinal: usize,
            module: &mut PredictionModule<B, X, M>,
        ) -> Result<(), Self::Error> {
            let residency = &mut module.residency;
            self.0.units.push(residency.register(ordinal)?);
            self.0.owners.push(residency.owner.clone());
            Ok(())
        }
    }
    let mut collect = Collect(PredictionResidency::default());
    prediction.visit_modules(&mut collect)?;
    Ok(collect.0)
}

/// Relationship between a physical prediction module and other parameter owners.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PredictionModuleRole {
    /// One physical unit; ordinary units execute sequentially.
    Unit,
    /// A shared module retained while ordinary units execute. Its bytes overlap
    /// the largest unit instead of requiring every unit to stay resident.
    Shared,
}

/// One architecture-constructed neutral module and its exact checkpoint recipes.
pub struct PreparedPredictionUnit<M> {
    source: M,
    local: M,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    source_layout: Option<std::sync::Arc<LocalModelLayout>>,
    residency: crate::LayerWeightResidency,
    role: PredictionModuleRole,
    local_parameters: Vec<eredu_nn::ParameterMetadata>,
}

impl<M> PreparedPredictionUnit<M> {
    /// Pairs source and selected modules with the parameter tasks already admitted for them.
    pub fn new<T: 'static>(
        source: M,
        local: M,
        tasks: Vec<ReplicatedTextMaterializationTask>,
        role: PredictionModuleRole,
    ) -> Result<Self, eredu_nn::ParameterTopologyError>
    where
        M: Parameterized<T>,
    {
        let local_parameters = eredu_nn::validate_parameter_topology(&local)?;
        Ok(Self {
            source,
            local,
            tasks,
            source_layout: None,
            residency: crate::LayerWeightResidency::FullyResident,
            role,
            local_parameters,
        })
    }

    /// Assigns the immutable parameter policy selected for this owner.
    pub fn set_residency(&mut self, residency: crate::LayerWeightResidency) {
        self.residency = residency;
    }

    /// Returns this owner with the selected immutable parameter policy.
    pub fn with_residency(mut self, residency: crate::LayerWeightResidency) -> Self {
        self.set_residency(residency);
        self
    }

    /// Records the source geometry used by the selected local transform.
    pub fn with_source_layout(mut self, layout: Option<std::sync::Arc<LocalModelLayout>>) -> Self {
        self.source_layout = layout;
        self
    }

    /// Source placement used by admitted local transforms, when present. The
    /// source module already has this rank's floating geometry; recipes remain
    /// global and are selected through this layout before bounded conversion.
    pub fn source_layout(&self) -> Option<&std::sync::Arc<LocalModelLayout>> {
        self.source_layout.as_ref()
    }

    /// Retained immutable-weight policy selected for the complete execution.
    /// A native module owner must account for this module under the same bounds
    /// as its target, including overlapping shared modules and target statics.
    pub const fn residency(&self) -> crate::LayerWeightResidency {
        self.residency
    }

    /// Architecture-declared overlap with other physical prediction owners.
    pub const fn role(&self) -> PredictionModuleRole {
        self.role
    }

    /// Consumes source/target modules and exact tasks. A source layout marks a
    /// rank-local conversion source; otherwise the source module is global.
    pub fn into_parts(self) -> (M, M, Vec<ReplicatedTextMaterializationTask>) {
        (self.source, self.local, self.tasks)
    }

    /// Projects the exact parameter owner used for portable resource accounting.
    pub fn resource_module(
        &self,
        ordinal: usize,
    ) -> crate::prediction_resources::PreparedPredictionModule {
        crate::prediction_resources::PreparedPredictionModule {
            ordinal,
            residency_owner: self
                .tasks
                .iter()
                .find_map(|task| task.auxiliary_residency())
                .cloned(),
            shared: self.role == PredictionModuleRole::Shared,
            parameters: self.local_parameters.clone(),
        }
    }
}

/// An equation outcome together with every output and changed-state tensor
/// needed to settle its module loan. An error does not imply native completion.
pub struct PredictionInvocation<T, O> {
    outcome: Result<O, eredu_nn::Error>,
    retained: Vec<T>,
}

impl<T, O> PredictionInvocation<T, O> {
    /// Retains the exact dependency set on both successful and failed work.
    pub fn new(outcome: Result<O, eredu_nn::Error>, retained: impl IntoIterator<Item = T>) -> Self {
        Self {
            outcome,
            retained: retained.into_iter().collect(),
        }
    }

    /// Borrows evaluation roots while the native module lease remains held.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.retained.iter()
    }

    /// Publishes the equation outcome after the caller has established safe
    /// completion, terminal failure, or teardown of the retained native work.
    pub fn into_outcome(self) -> Result<O, eredu_nn::Error> {
        self.outcome
    }
}

/// Visits the actual materialized prediction modules in architecture-declared
/// ownership order. A module ordinal identifies a physical parameter owner; it
/// is independent of proposal depth and mutable lane-state position.
pub trait PredictionModuleVisitor<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Traversal failure retained by the consumer.
    type Error;
    /// Borrows one complete typed module without rebuilding or loading parameters.
    fn visit<U: Parameterized<B::Tensor>>(
        &mut self,
        ordinal: usize,
        module: &mut PredictionModule<B, M, U>,
    ) -> Result<(), Self::Error>;

    /// Visits an ordered physical sequence from its declared first ordinal.
    /// Shared modules remain separate calls in the surrounding topology.
    fn visit_units<'a, U>(
        &mut self,
        first: usize,
        units: impl IntoIterator<Item = &'a mut PredictionModule<B, M, U>>,
    ) -> Result<(), Self::Error>
    where
        U: Parameterized<B::Tensor> + 'a,
        PredictionModule<B, M, U>: 'a,
    {
        for (index, module) in units.into_iter().enumerate() {
            self.visit::<U>(first + index, module)?;
        }
        Ok(())
    }
}

/// Portable compressed state of one sequential prediction layer.
pub type SequentialPredictionCache<B, M> = crate::cache::CompressedLatentCache<
    <M as NativePredictionMaterialization<B>>::SequentialStorage,
>;

/// Portable invocation and state over the selected native prediction mechanisms.
pub trait PredictionExtensionMaterializer<B>: NativePredictionMaterialization<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Portable state owner for the selected predictor geometry.
    type ModelState: PredictionModelState<B>
        + crate::state::PortableTextState<B, Self::ModelCacheStorage>
        + 'static;

    /// Invokes one typed module while its immutable parameters remain resident.
    fn invoke_module<U, O>(
        module: &mut PredictionModule<B, Self, U>,
        context: &<B::Tensor as Tensor>::Context,
        operation: impl FnOnce(&mut U) -> PredictionInvocation<B::Tensor, O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<B::Tensor>,
    {
        with_prediction_module::<B, Self, _, _, _>(module, Some(context), |inner| {
            let invocation = operation(inner);
            let roots = invocation.retained_values().cloned().collect();
            (invocation.into_outcome(), roots)
        })
        .map_err(eredu_nn::Error::backend_source)?
    }

    /// Invokes a unit with a distinct shared parameter owner. Both reservations
    /// remain live through their common output and state dependencies.
    fn invoke_module_with_shared<U, V, O>(
        module: &mut PredictionModule<B, Self, U>,
        shared: Option<&mut PredictionModule<B, Self, V>>,
        context: &<B::Tensor as Tensor>::Context,
        operation: impl FnOnce(&mut U, Option<&mut V>) -> PredictionInvocation<B::Tensor, O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<B::Tensor>,
        V: Parameterized<B::Tensor>,
    {
        Self::invoke_module(module, context, |module| match shared {
            None => operation(module, None),
            Some(shared) => {
                let mut retained = Vec::new();
                let outcome = Self::invoke_module(shared, context, |shared| {
                    let invocation = operation(module, Some(shared));
                    retained.extend(invocation.retained_values().cloned());
                    invocation
                });
                PredictionInvocation::new(outcome, retained)
            }
        })
    }

    /// Complete durable-copy bound for the declared model-state profile.
    fn model_snapshot_estimate(
        _state: &Self::ModelState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies the complete declared model-state profile after reservation.
    fn model_snapshot(
        _state: &Self::ModelState,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::ModelState>, eredu_core::BackendFailure> {
        Ok(None)
    }
}

impl<B, M> PredictionExtensionMaterializer<B> for M
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: NativePredictionMaterialization<
        B,
        ModelCacheStorage: crate::state::HybridCacheStorage + 'static,
    >,
{
    type ModelState = crate::state::HybridState<M::ModelCacheStorage>;
}

/// Portable pooling-attention state for one layer of the selected predictor.
pub type PredictionPoolingState<B, M> =
    crate::state::PoolingAttentionLayer<<M as NativePredictionMaterialization<B>>::PoolingStorage>;

/// Family-blind mechanism for invoking an architecture-owned operation against
/// the singular typed target session.
pub trait PredictionOperationInvoker<A, B, S>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
{
    /// Backend/session invocation failure.
    type Error;

    /// Invokes one typed architecture operation without erasing its target.
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Self::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>;

    /// Constructs a mechanism error for a rejected architecture contract.
    fn invalid(message: String) -> Self::Error;
}

/// Materializes one architecture module through the portable binding driver.
pub fn materialize_prediction_module<B, M, U>(
    context: &mut M::Context<'_>,
    prepared: PreparedPredictionUnit<U>,
    layout: Option<&LocalModelLayout>,
) -> Result<PredictionModule<B, M, U>, M::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    U: Parameterized<B::Tensor>,
{
    let (source, execution) = M::parameter_source(context);
    let mut store = source.clone();
    let PreparedPredictionUnit {
        source,
        local,
        tasks,
        source_layout,
        residency,
        role,
        ..
    } = prepared;
    let groups =
        crate::group_replicated_text_transform_tasks(&tasks).map_err(BackendFailure::from_error)?;
    let materialization = crate::exact_parameter_materialization::materialize_exact_transforms::<
        B,
        M::Parameters,
        U,
        U,
    >(
        &mut store,
        &source,
        &local,
        &[],
        &[],
        source_layout.as_deref(),
        &tasks,
        &groups,
        execution,
    )?;
    let references = tasks.iter().collect::<Vec<_>>();
    let bindings = crate::exact_parameter_materialization::exact_bindings::<B, M::Parameters, _>(
        &local,
        store.as_ref(),
        &references,
        &std::collections::BTreeSet::new(),
        layout,
    )?;
    let mut parameters = Vec::new();
    let mut declarations = Vec::new();
    crate::parameter_operations::collect_prepared_module_parameters::<B::Tensor, _>(
        &local,
        &bindings,
        PreparedParameterLocation::Prediction { module: 0 },
        store.as_ref(),
        &mut parameters,
        &mut declarations,
    )?;
    if parameters.len() != declarations.len() {
        return Err(BackendFailure::from_error(eredu_nn::Error::backend(
            "prediction parameters lack exact prepared binding owners",
        ))
        .into());
    }

    Ok(BoundPredictionModule {
        placeholders: placeholder_slots(&local),
        replacements: BTreeMap::new(),
        residency: PredictionModuleResidency::new(store, bindings, role, residency),
        context: M::retain_module_context(execution),
        inner: local,
        parameters,
        tasks,
        materialization,
    })
}

/// Materializes ordered units through their independent resource lifecycles.
/// Stops at the first error without acquiring later units.
pub fn materialize_prediction_units<B, M, U>(
    context: &mut M::Context<'_>,
    units: impl IntoIterator<Item = PreparedPredictionUnit<U>>,
    layout: Option<&LocalModelLayout>,
) -> Result<Vec<PredictionModule<B, M, U>>, M::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    U: Parameterized<B::Tensor>,
{
    units
        .into_iter()
        .map(|unit| materialize_prediction_module::<B, M, _>(context, unit, layout))
        .collect()
}

/// Installs one loaned prediction module, invokes its portable operation, and settles its roots.
pub(crate) fn with_prediction_module<B, P, U, O, E>(
    module: &mut PredictionModule<B, P, U>,
    context: Option<&<B::Tensor as Tensor>::Context>,
    operation: impl FnOnce(&mut U) -> (Result<O, E>, Vec<B::Tensor>),
) -> Result<Result<O, E>, P::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    P: PredictionExtensionMaterializer<B>,
    U: Parameterized<B::Tensor>,
{
    use crate::residency::{ResidencyAccess as _, ResidencyWindowManager as _};
    use eredu_core::residency::MemoryTier;

    let (residency, id, placement) = module.residency.installed()?;
    let (residency, id) = (residency.clone(), id.clone());
    let transfer = residency
        .acquire_many_with_transfer(&[(id.clone(), 1)], MemoryTier::Device)
        .map_err(BackendFailure::from_error)?;
    let context = context.unwrap_or_else(|| std::borrow::Borrow::borrow(&module.context));
    let mut scope = P::begin_module(transfer, context)?;
    let BoundPredictionModule {
        inner,
        replacements,
        placeholders,
        ..
    } = module;
    let installed = P::module_weights(&mut scope).and_then(|weights| {
        let weights =
            crate::MaterializedUnit::<P::ModuleParameters>::try_from_named_weights(weights)
                .map_err(BackendFailure::from_error)?;
        crate::bind_materialized_unit::<P::ModuleParameters, _>(inner, weights)
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    });
    let (outcome, roots) = match installed {
        Ok(()) => {
            eredu_nn::publish_parameter_replacements(inner, replacements);
            let (outcome, roots) = operation(inner);
            (Ok(outcome), roots)
        }
        Err(error) => (Err(error), Vec::new()),
    };
    let completed = P::finish_module(scope, roots);
    eredu_nn::publish_parameter_replacements(inner, placeholders);
    let released = if placement.is_fully_resident() {
        Ok(())
    } else {
        residency
            .evict(&id, MemoryTier::Device)
            .map(|_| ())
            .map_err(BackendFailure::from_error)
    };
    completed?;
    let outcome = outcome?;
    if outcome.is_ok() {
        released?;
    }
    Ok(outcome)
}
