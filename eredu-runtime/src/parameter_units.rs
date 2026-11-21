//! Portable binding of independently addressable parameter units.

use eredu_core::checkpoint_mechanisms::unit_binding::{
    ComponentPreparationScope, NativeParameterUnitBinding, ParameterStorageBinding,
};

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_core::residency::ResidencyStorageMechanisms;
use eredu_core::{
    residency::{
        MemoryTier, OffloadUnitId, ParameterConversionRetentionEligibility,
        ParameterConversionRetentionPolicy, ResidencyStorage,
    },
    BackendFailure,
};
use eredu_nn::{NeuralBackend, Parameterized};
use std::{collections::BTreeSet, sync::Arc};

use crate::{
    bounded_execution::units::BoundUnitStorage,
    dense::{DenseStreamController, DenseStreamMechanisms},
    residency::{PreparedResidency, ResidencyAccess, ResidencyDriver},
    ExecutionUnitAddressMap, ImmutableParameterPolicy, LayerWeightResidency,
    LayerwiseModelMetadata, LayerwisePolicy, PreparedParameterResources, UnitPolicyResidency,
};

/// Admitted residency for the parameter backend used by the native storage.
pub type PreparedUnitResidency<R> =
    PreparedResidency<<<R as ResidencyAccess>::Storage as ResidencyStorage>::Backend>;

/// Admitted layout for the portable owner of the bound parameter units.
pub struct ParameterUnitLayout {
    /// Ordered invocation and storage geometry.
    pub invocations: ExecutionUnitAddressMap,
    /// Physical unit identities in storage order.
    pub unit_ids: Vec<OffloadUnitId>,
    /// Selected storage and sampling configuration.
    pub residency: LayerWeightResidency,
    /// Capacity and residency metadata for the installed units.
    pub metadata: LayerwiseModelMetadata,
}

/// Exact leases and streaming resources prepared for the selected unit layout.
pub struct ParameterUnitStorageResources<R: DenseStreamMechanisms> {
    /// Device transfer and leases for each fully resident unit in storage order.
    pub resident_transfer: Option<R::Transfer>,
    /// Physical units pinned by the residency plan.
    pub pinned_units: BTreeSet<OffloadUnitId>,
    /// Bounded dense streaming controller when selected.
    pub dense: Option<Arc<DenseStreamController<R>>>,
}

type UnitResidency<B, M> = ResidencyDriver<<M as ParameterStorageBinding<B>>::ResidencyMechanisms>;

/// Portable storage constructed from the backend's native unit mechanisms.
pub type ParameterUnitStorage<B, M> = BoundUnitStorage<
    <M as ParameterStorageBinding<B>>::ResidencyMechanisms,
    <M as ParameterStorageBinding<B>>::UnitOperations,
>;

/// Portable layerwise execution over the exact storage and leases of a backend.
pub type ParameterUnitPolicy<B, U, M> = crate::bounded_execution::units::BoundParameterUnitPolicy<
    B,
    U,
    <M as ParameterStorageBinding<B>>::ResidencyMechanisms,
    <M as ParameterStorageBinding<B>>::UnitOperations,
    <M as ParameterStorageBinding<B>>::ImmutableParameters,
>;

/// Portable residency policy derived from the native unit mechanisms.
pub trait ParameterUnitMechanisms<B: NeuralBackend>: NativeParameterUnitBinding<B> {
    /// Selected residency policy for a sequence of parameterized units.
    type UnitPolicy<U>: LayerwisePolicy<B, U, Error = Self::UnitError>
        + crate::UnitResidencyTelemetry<Error = Self::UnitError>
        + crate::bounded_execution::units::BindResidentPolicy<B, U, ParameterUnitStorage<B, Self>>
        + crate::bounded_execution::units::BindBoundedPolicy<U, ParameterUnitStorage<B, Self>>
        + UnitPolicyResidency
        + crate::LocalParameterRetention
        + ImmutableParameterPolicy<B, Certificate = Self::ImmutableParameters>
        + Clone
        + 'static
    where
        U: Parameterized<B::Tensor> + 'static;
}

impl<B: NeuralBackend, M: NativeParameterUnitBinding<B>> ParameterUnitMechanisms<B> for M {
    type UnitPolicy<U>
        = ParameterUnitPolicy<B, U, M>
    where
        U: Parameterized<B::Tensor> + 'static;
}

/// Plans and materializes exact pinned and independently acquired parameter units.
#[allow(clippy::too_many_arguments)]
pub fn prepare_parameter_unit_storage<B, M, SM>(
    mechanisms: &M,
    source: SharedCheckpointSource,
    static_modules: &mut SM,
    invocations: &ExecutionUnitAddressMap,
    residency: LayerWeightResidency,
    pinned: Vec<WeightBinding>,
    units: Vec<Vec<WeightBinding>>,
    ignored: impl Fn(&str) -> bool,
    conversion_retention: Option<(
        Option<ParameterConversionRetentionPolicy>,
        ParameterConversionRetentionEligibility,
    )>,
) -> Result<(ParameterUnitStorage<B, M>, LayerwiseModelMetadata), BackendFailure>
where
    B: NeuralBackend,
    M: ParameterStorageBinding<B>,
    SM: Parameterized<B::Tensor>,
{
    crate::checkpoint_quantization::realize_prepared_quantization(&source, || {
        mechanisms.unit_quantizer()
    })?;
    let resources = crate::prepare_parameter_resources(
        source.as_ref(),
        invocations,
        residency,
        pinned,
        units,
        Vec::new(),
        ignored,
        |bindings| mechanisms.unit_host_capacity(bindings),
    )
    .map_err(BackendFailure::from_error)?;
    materialize_parameter_units(
        mechanisms,
        source,
        static_modules,
        resources,
        None,
        conversion_retention,
        None,
    )
}

/// Binds exact prepared parameters, constructing resident units in portable code.
pub fn bind_parameter_units<B, M, U>(
    mechanisms: &M,
    source: SharedCheckpointSource,
    invocations: ExecutionUnitAddressMap,
    units: Vec<Vec<WeightBinding>>,
    residency: LayerWeightResidency,
    build_resident: impl FnOnce() -> Result<Vec<U>, BackendFailure>,
) -> Result<(M::UnitPolicy<U>, LayerwiseModelMetadata), BackendFailure>
where
    B: NeuralBackend,
    M: ParameterUnitMechanisms<B>,
    U: Parameterized<B::Tensor> + 'static,
{
    let scope = mechanisms.begin_unit_binding()?;
    let result = (|| {
        let (storage, metadata) = prepare_parameter_unit_storage(
            mechanisms,
            source,
            &mut crate::EmptyParameterModule,
            &invocations,
            residency,
            Vec::new(),
            units,
            |_| false,
            None,
        )?;
        let policy = if residency.is_fully_resident() {
            let units = build_resident()?;
            bind_component_resident_units(mechanisms, storage, units)?
        } else {
            <M::UnitPolicy<U> as crate::bounded_execution::units::BindBoundedPolicy<_, _>>::bind_bounded(storage)
        };
        Ok((policy, metadata))
    })();
    scope.finish(result)
}

/// Validates and binds the admitted storage shared by local and pipeline components.
pub(crate) fn materialize_component_storage<B, S, A, M>(
    mechanisms: &M,
    architecture: &mut A,
    source: SharedCheckpointSource,
    invocations: &ExecutionUnitAddressMap,
    residency: LayerWeightResidency,
    pinned: Vec<WeightBinding>,
    units: Vec<Vec<WeightBinding>>,
    conversion_retention: Option<(
        Option<ParameterConversionRetentionPolicy>,
        ParameterConversionRetentionEligibility,
    )>,
) -> Result<(ParameterUnitStorage<B, M>, LayerwiseModelMetadata), BackendFailure>
where
    B: NeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    M: ParameterUnitMechanisms<B>,
{
    let declared = architecture
        .execution_unit_storage_map()
        .map_err(BackendFailure::from_error)?;
    let logical = architecture
        .execution_unit_layout()
        .map_err(BackendFailure::from_error)?;
    declared
        .validate_layout(&logical)
        .map_err(BackendFailure::from_error)?;
    if declared
        .select_invocations(invocations.logical_addresses())
        .map_err(BackendFailure::from_error)?
        != *invocations
    {
        return Err(BackendFailure::from_error(eredu_nn::Error::backend(
            "prepared invocation map differs from architecture storage ownership",
        )));
    }
    prepare_parameter_unit_storage(
        mechanisms,
        source,
        architecture.static_modules_mut(),
        invocations,
        residency,
        pinned,
        units,
        |_| false,
        conversion_retention,
    )
}

/// Opens admitted residency and pins static parameters before publishing unit storage.
pub(crate) fn materialize_parameter_units<B, M, SM>(
    mechanisms: &M,
    source: SharedCheckpointSource,
    static_modules: &mut SM,
    resources: PreparedParameterResources,
    population: Option<crate::bounded_execution::units::SelectiveUnitParameters<B::Tensor>>,
    conversion_retention: Option<(
        Option<ParameterConversionRetentionPolicy>,
        ParameterConversionRetentionEligibility,
    )>,
    supplementary: Option<crate::speculative::PredictionResidency<UnitResidency<B, M>>>,
) -> Result<(ParameterUnitStorage<B, M>, LayerwiseModelMetadata), BackendFailure>
where
    B: NeuralBackend,
    M: ParameterStorageBinding<B>,
    SM: Parameterized<B::Tensor>,
{
    let source = resources
        .restrict_primary_source(source)
        .map_err(BackendFailure::from_error)?;
    let PreparedParameterResources {
        residency: options,
        plan,
        definitions,
        sources,
        invocations,
        unit_ids,
        static_id,
        static_parameters,
        metadata,
        ..
    } = resources;
    let prepared = PreparedUnitResidency::<UnitResidency<B, M>>::new(
        source.clone(),
        sources,
        plan,
        definitions,
        Vec::new(),
    )
    .map_err(BackendFailure::from_error)?;
    let prepared = match conversion_retention {
        Some((requested, eligibility)) => {
            prepared.select_conversion_retention(requested, eligibility)
        }
        None => prepared,
    };
    let residency = prepared
        .install::<UnitResidency<B, M>>(
            mechanisms.unit_residency_context(options.dense().is_some())?,
        )
        .map_err(BackendFailure::from_error)?;
    residency.initialize().map_err(BackendFailure::from_error)?;
    let pinned = static_id
        .map(|id| {
            let lease = residency
                .acquire(&id, MemoryTier::Device)
                .map_err(BackendFailure::from_error)?;
            let unit = crate::parameter::resident_parameter_values::<M::Parameters, _>(
                &lease,
                M::retain_parameter,
            )?;
            crate::bind_materialized_unit_excluding::<M::Parameters, _, _>(
                static_modules,
                unit,
                |id| !static_parameters.contains(id.as_str()),
            )
            .map_err(BackendFailure::from_error)?;
            Ok::<_, BackendFailure>(lease)
        })
        .transpose()?;
    let layout = ParameterUnitLayout {
        invocations,
        unit_ids,
        residency: options,
        metadata: metadata.clone(),
    };
    let dense = DenseStreamController::for_parameter_units(&residency, &layout)
        .map_err(BackendFailure::from_error)?;
    let pinned_units = residency
        .pinned_unit_ids(&layout.unit_ids)
        .map_err(BackendFailure::from_error)?;
    let resident_transfer = if options.is_fully_resident() {
        let requests = layout
            .unit_ids
            .iter()
            .cloned()
            .map(|id| (id, 1))
            .collect::<Vec<_>>();
        Some(
            residency
                .acquire_many_with_transfer(&requests, MemoryTier::Device)
                .map_err(BackendFailure::from_error)?,
        )
    } else {
        None
    };
    let native = mechanisms.unit_operations();
    let supplementary_owner = supplementary.as_ref().map(|_| residency.clone());
    let storage = BoundUnitStorage::new(
        residency,
        source,
        layout,
        ParameterUnitStorageResources {
            resident_transfer,
            pinned_units,
            dense,
        },
        native,
        population,
        pinned,
    )
    .map_err(BackendFailure::from_error)?;
    if let (Some(supplementary), Some(owner)) = (supplementary, supplementary_owner) {
        supplementary.attach(&owner)?;
    }
    Ok((storage, metadata))
}

/// Native storage used to derive the portable owner of supplementary weights.
pub trait WeightResidencyStorage {
    /// Shared owner into which the runtime installs supplementary units.
    type Owner;
}

impl<N: ResidencyStorageMechanisms> WeightResidencyStorage for N {
    type Owner = ResidencyDriver<N>;
}

/// Portable weight inputs supplied by the selected local or partitioned route.
pub struct ReplicatedWeightOptions<N: WeightResidencyStorage> {
    /// Supplementary predictor units sharing the same residency authority.
    pub supplementary: crate::speculative::PredictionResidency<N::Owner>,
    /// Admitted checkpoint keys supplied by other owners.
    pub ignored_sources: BTreeSet<String>,
}

impl<N: WeightResidencyStorage> Default for ReplicatedWeightOptions<N> {
    fn default() -> Self {
        Self {
            supplementary: Default::default(),
            ignored_sources: BTreeSet::new(),
        }
    }
}

/// Installation of prepared parameter storage before the execution policy takes ownership.
pub trait BindReplicatedParameterStorage<B: NeuralBackend> {
    /// Owner that the selected supplementary units will share.
    type WeightStorage: crate::WeightResidencyStorage;

    /// Parameter storage supplied to the installed policy.
    type Storage;

    /// Binds and pins the admitted parameter storage.
    fn bind<SM: Parameterized<B::Tensor>>(
        self,
        prepared: Option<crate::PreparedReplicatedParameterBindings>,
        weights: ReplicatedWeightOptions<Self::WeightStorage>,
        static_modules: &mut SM,
        selected: &crate::SelectedReplicatedTextRealization,
    ) -> Result<Self::Storage, BackendFailure>;
}

impl<B, M> BindReplicatedParameterStorage<B> for &M
where
    B: NeuralBackend,
    M: ParameterStorageBinding<B>,
{
    type Storage = ParameterUnitStorage<B, M>;
    type WeightStorage = M::ResidencyMechanisms;

    fn bind<SM: Parameterized<B::Tensor>>(
        self,
        prepared: Option<crate::PreparedReplicatedParameterBindings>,
        weights: ReplicatedWeightOptions<Self::WeightStorage>,
        static_modules: &mut SM,
        selected: &crate::SelectedReplicatedTextRealization,
    ) -> Result<Self::Storage, BackendFailure> {
        let prepared = prepared.ok_or_else(|| {
            BackendFailure::from_error(std::io::Error::other(
                "parameter binding requires a retained checkpoint source",
            ))
        })?;
        let population = crate::bounded_execution::units::SelectiveUnitParameters::new(
            prepared.excluded_parameters.clone(),
        );
        let mut supplementary = weights.supplementary;
        let requirements = selected.requirements();
        let mut ignored_sources = weights.ignored_sources;
        ignored_sources.extend(requirements.nonparameter_sources().iter().cloned());
        for parameter in requirements.parameters() {
            ignored_sources.extend(parameter.admitted_redundant_sources().iter().cloned());
            if prepared.excluded_parameters.contains(parameter.name())
                || prepared
                    .local_parameters
                    .as_ref()
                    .is_some_and(|local| !local.contains(parameter.name()))
            {
                ignored_sources.extend(parameter.sources().iter().cloned());
                if let Some(recipe) = requirements.derived_recipes().get(parameter.name()) {
                    ignored_sources.extend(recipe.source_keys().into_iter().map(str::to_owned));
                }
            }
        }
        crate::checkpoint_quantization::realize_prepared_quantization(&prepared.source, || {
            self.unit_quantizer()
        })?;
        let resources = crate::prepare_parameter_resources(
            prepared.source.as_ref(),
            &prepared.invocations,
            selected.residency(),
            prepared.static_bindings,
            prepared.unit_bindings,
            supplementary.take_units(),
            |key| ignored_sources.contains(key),
            |bindings| self.unit_host_capacity(bindings),
        )
        .map_err(BackendFailure::from_error)?;
        let eligibility = if selected.topology().is_replicated() {
            ParameterConversionRetentionEligibility::Eligible
        } else {
            ParameterConversionRetentionEligibility::Unsupported {
                reason:
                    "partitioned execution has no cross-rank conversion retention budget authority"
                        .into(),
            }
        };
        let (storage, _) = materialize_parameter_units(
            self,
            prepared.source,
            static_modules,
            resources,
            Some(population),
            Some((selected.parameter_conversion_retention(), eligibility)),
            Some(supplementary),
        )?;
        Ok(storage)
    }
}

/// Constructs one unit per physical storage owner, in materialization order.
pub fn build_resident_units<B, S, A>(
    architecture: &A,
    invocations: &ExecutionUnitAddressMap,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<Vec<A::Unit>, BackendFailure>
where
    B: NeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
{
    (0..invocations.storage_layout().len())
        .map(|ordinal| {
            let storage = invocations
                .storage_location(ordinal)
                .expect("admitted physical storage ordinal");
            architecture
                .build_storage_unit(storage, context)
                .map_err(BackendFailure::from_error)
        })
        .collect()
}

pub(crate) fn bind_component_resident_units<B, M, U>(
    mechanisms: &M,
    storage: ParameterUnitStorage<B, M>,
    units: Vec<U>,
) -> Result<M::UnitPolicy<U>, BackendFailure>
where
    B: NeuralBackend,
    M: ParameterUnitMechanisms<B>,
    U: Parameterized<B::Tensor> + 'static,
{
    use crate::bounded_execution::units::BindResidentPolicy as _;
    let mut policy = M::UnitPolicy::<U>::bind_resident(storage, units, mechanisms.unit_executor())?;
    policy.set_immutable_certificate(mechanisms.resident_parameter_certificate()?);
    Ok(policy)
}
