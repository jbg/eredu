//! MLX parameter materialization for portable realtime layered construction.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::store::{CheckpointSource, SharedCheckpointSource};
use eredu_nn::Parameterized;
use eredu_runtime::{
    LayeredArchitecture, LayerwiseModelMetadata, ParameterGroupOwner, RealizedRealtimePolicy,
    RealtimeMaterializationTask, RealtimeModelConstructionMechanisms, SelectedRealtimeRealization,
    WeightBinding,
};
use safemlx::Stream;

use crate::backend::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    runtime::{
        cache::state::MlxKeyValueState,
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            },
            layerwise::{quantize_exact_realtime_tasks, shard_layer_bindings},
        },
    },
};

/// Generic MLX storage/materialization mechanisms used by the neutral constructor.
pub struct MlxRealtimeConstructionMechanisms<U: 'static> {
    store: SharedCheckpointSource,
    residency: eredu_runtime::LayerWeightResidency,
    weights_stream: Stream,
    transform: Option<eredu_checkpoint::WeightQuantization>,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    pending: Option<MlxLayerwisePolicy<U>>,
    metadata: Option<LayerwiseModelMetadata>,
    local_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
}

impl<U: 'static> MlxRealtimeConstructionMechanisms<U> {
    pub(crate) fn new(
        store: SharedCheckpointSource,
        residency: eredu_runtime::LayerWeightResidency,
        transform: Option<eredu_checkpoint::WeightQuantization>,
        weights_stream: &Stream,
        local_layout: Option<eredu_runtime::LocalModelLayout>,
    ) -> Self {
        Self {
            store,
            residency,
            weights_stream: weights_stream.clone(),
            transform,
            materialization: None,
            pending: None,
            metadata: None,
            local_layout: local_layout.map(Arc::new),
        }
    }

    /// Returns construction metadata after a policy has been realized.
    pub(crate) fn metadata(&self) -> Option<&LayerwiseModelMetadata> {
        self.metadata.as_ref()
    }
}
fn selected_task_bindings(
    tasks: &[RealtimeMaterializationTask],
    store: &dyn CheckpointSource,
    local_layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<
    (
        Vec<WeightBinding>,
        BTreeMap<ParameterGroupOwner, Vec<WeightBinding>>,
    ),
    Error,
> {
    let (mut pinned, mut units) = eredu_runtime::realtime_task_binding_plan(tasks, store)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .into_parts();
    if let Some(layout) = local_layout {
        pinned = shard_layer_bindings(pinned, store, layout)?;
        for bindings in units.values_mut() {
            *bindings = shard_layer_bindings(std::mem::take(bindings), store, layout)?;
        }
    }
    Ok((pinned, units))
}

impl<A, U> RealtimeModelConstructionMechanisms<A, MlxNeuralBackend>
    for MlxRealtimeConstructionMechanisms<U>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState, Unit = U>,
    A::Error: std::fmt::Display,
    U: Parameterized<crate::MlxTensor> + 'static,
{
    type State = MlxKeyValueState;
    type PolicyError = Error;
    type ResidentPolicy = MlxResidentPolicy<U>;
    type BoundedPolicy = MlxLayerwisePolicy<U>;
    type Error = Error;

    fn prepare_resident_materialization(
        &mut self,
        architecture: &mut A,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &Stream,
    ) -> Result<(), Self::Error> {
        eredu_runtime::preflight_realtime_materialization_tasks::<MlxNeuralBackend>(
            tasks,
            self.store.as_ref(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if source_architecture.is_some() || source_units.is_some() {
            let quantization = self.transform.ok_or_else(|| {
                Error::ArchitectureModel(
                    "source-format architecture was supplied without a selected MLX transform"
                        .into(),
                )
            })?;
            let (store, report) = quantize_exact_realtime_tasks(
                Arc::clone(&self.store),
                architecture.static_modules(),
                units,
                quantization,
                tasks,
                context,
            )?;
            self.store = store;
            self.materialization = Some(report);
        } else if self.transform.is_some() {
            return Err(Error::ArchitectureModel(
                "selected MLX transform has no source-format architecture".into(),
            ));
        }
        let (static_bindings, mut unit_bindings) =
            selected_task_bindings(tasks, self.store.as_ref(), self.local_layout.as_deref())?;
        let layout = selected.execution_units().clone();
        let (policy, metadata) = prepare_layerwise_policy_with_bindings(
            Arc::clone(&self.store),
            architecture,
            (),
            std::marker::PhantomData::<MlxKeyValueState>,
            self.residency,
            context,
            &self.weights_stream,
            |_| false,
            move |_modules, _store| Ok(static_bindings),
            move |_ordinal, address, _path, _unit, _store, _stream| {
                let group = layout
                    .group_id(address.group())
                    .ok_or_else(|| {
                        Error::ArchitectureModel("selected unit group is missing".into())
                    })?
                    .clone();
                unit_bindings
                    .remove(&ParameterGroupOwner::execution_unit(group, address.index()))
                    .ok_or_else(|| {
                        Error::ArchitectureModel(format!(
                            "selected unit bindings are missing for {address:?}"
                        ))
                    })
            },
        )?;
        self.pending = Some(policy);
        let mut metadata = metadata;
        metadata.set_materialization(self.materialization.clone());
        self.metadata = Some(metadata);
        Ok(())
    }

    fn resident_policy(
        &mut self,
        _architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedRealtimeRealization,
        context: &Stream,
    ) -> Result<RealizedRealtimePolicy<Self::ResidentPolicy>, Self::Error> {
        let policy = self
            .pending
            .take()
            .ok_or_else(|| Error::ArchitectureModel("resident MLX policy was not prepared".into()))?
            .into_resident_units(units, context)?;
        Ok(RealizedRealtimePolicy::new(policy, selected.residency()))
    }

    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        source_architecture: Option<&mut A>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &Stream,
    ) -> Result<RealizedRealtimePolicy<Self::BoundedPolicy>, Self::Error> {
        eredu_runtime::preflight_realtime_materialization_tasks::<MlxNeuralBackend>(
            tasks,
            self.store.as_ref(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if source_architecture.is_some() {
            let quantization = self.transform.ok_or_else(|| {
                Error::ArchitectureModel(
                    "source-format architecture was supplied without a selected MLX transform"
                        .into(),
                )
            })?;
            let units = (0..selected.execution_units().len())
                .map(|ordinal| {
                    let address = selected
                        .execution_units()
                        .address(ordinal)
                        .expect("selected execution ordinal has an address");
                    architecture
                        .build_unit(address.group(), address.index(), context)
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (store, report) = quantize_exact_realtime_tasks(
                Arc::clone(&self.store),
                architecture.static_modules(),
                &units,
                quantization,
                tasks,
                context,
            )?;
            self.store = store;
            self.materialization = Some(report);
        } else if self.transform.is_some() {
            return Err(Error::ArchitectureModel(
                "selected MLX transform has no source-format architecture".into(),
            ));
        }
        let (static_bindings, mut unit_bindings) =
            selected_task_bindings(tasks, self.store.as_ref(), self.local_layout.as_deref())?;
        let layout = selected.execution_units().clone();
        let (policy, metadata) = prepare_layerwise_policy_with_bindings(
            Arc::clone(&self.store),
            architecture,
            (),
            std::marker::PhantomData::<MlxKeyValueState>,
            self.residency,
            context,
            &self.weights_stream,
            |_| false,
            move |_modules, _store| Ok(static_bindings),
            move |_ordinal, address, _path, _unit, _store, _stream| {
                let group = layout
                    .group_id(address.group())
                    .ok_or_else(|| {
                        Error::ArchitectureModel("selected unit group is missing".into())
                    })?
                    .clone();
                unit_bindings
                    .remove(&ParameterGroupOwner::execution_unit(group, address.index()))
                    .ok_or_else(|| {
                        Error::ArchitectureModel(format!(
                            "selected unit bindings are missing for {address:?}"
                        ))
                    })
            },
        )?;
        let mut metadata = metadata;
        metadata.set_materialization(self.materialization.clone());
        self.metadata = Some(metadata);
        Ok(RealizedRealtimePolicy::new(policy, selected.residency()))
    }
}

impl<U> eredu_runtime::RealtimeResidencyTelemetry for MlxResidentPolicy<U> {
    type Error = Error;

    fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        self.residency_report()
    }

    fn dense_stream_report(&self) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        Ok(None)
    }

    fn execution_group_reports(
        &self,
    ) -> Result<Vec<eredu_runtime::ResidentLayerGroupReport>, Error> {
        self.execution_group_reports()
    }
}

impl<U: 'static> eredu_runtime::RealtimeResidencyTelemetry for MlxLayerwisePolicy<U> {
    type Error = Error;

    fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        self.residency_report()
    }

    fn dense_stream_report(&self) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.dense_stream_report()
    }

    fn execution_group_reports(
        &self,
    ) -> Result<Vec<eredu_runtime::ResidentLayerGroupReport>, Error> {
        self.execution_group_reports()
    }
}
