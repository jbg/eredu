//! Binding discovery for reference runners that inspect constructed modules.

use eredu_backend_mlx::{
    backend::{error::Error, nn::shared::MlxNeuralBackend},
    native::MlxComponentBinder,
    MlxTensor,
};
use eredu_checkpoint::store::{CheckpointSource, SharedCheckpointSource};
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_nn::Parameterized;
use eredu_runtime::{
    LayerWeightResidency, LayeredArchitecture, LayerwiseModelMetadata, RuntimeState,
};
use safemlx::Stream;

pub fn prepare_reference_storage<A, S>(
    store: SharedCheckpointSource,
    architecture: &mut A,
    residency: LayerWeightResidency,
    execution: &Stream,
    weights: &Stream,
    ignored: impl Fn(&str) -> bool,
    static_bindings: impl FnOnce(
        &A::StaticModules,
        &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
    mut unit_bindings: impl FnMut(&A::Unit, &dyn CheckpointSource) -> Result<Vec<WeightBinding>, Error>,
) -> Result<
    (
        eredu_runtime::ParameterUnitStorage<MlxNeuralBackend, MlxComponentBinder>,
        LayerwiseModelMetadata,
    ),
    Error,
>
where
    S: RuntimeState<MlxNeuralBackend>,
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Unit: Parameterized<MlxTensor> + 'static,
{
    let invocations = architecture
        .execution_unit_storage_map()
        .map_err(|error| Error::Other(Box::new(error)))?;
    let logical = architecture
        .execution_unit_layout()
        .map_err(|error| Error::Other(Box::new(error)))?;
    invocations
        .validate_layout(&logical)
        .map_err(eredu_core::BackendFailure::from_error)?;
    let pinned = static_bindings(architecture.static_modules(), store.as_ref())?;
    let units = (0..invocations.storage_layout().len())
        .map(|ordinal| {
            let location = invocations
                .storage_location(ordinal)
                .expect("physical storage ordinal");
            let unit = architecture
                .build_storage_unit(location, execution)
                .map_err(|error| Error::Other(Box::new(error)))?;
            unit_bindings(&unit, store.as_ref())
        })
        .collect::<Result<Vec<_>, _>>()?;

    eredu_runtime::prepare_parameter_unit_storage(
        &MlxComponentBinder::new(weights, execution),
        store,
        architecture.static_modules_mut(),
        &invocations,
        residency,
        pinned,
        units,
        ignored,
        None,
    )
    .map_err(Error::from)
}
