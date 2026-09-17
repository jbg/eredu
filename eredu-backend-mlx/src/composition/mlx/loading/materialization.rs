use super::super::replicated_text as binding;
use super::*;
use crate::backend::{
    distributed::MlxDistributedSession,
    runtime::cache::state::{MlxHybridState, MlxPoolingAttentionState},
};
use eredu_architectures::prepared_execution::{
    CompositeRoute, PartitionedCompositeRoute, PartitionedDenseRoute, PartitionedRoutedRoute,
    PredictionBinding, PreparedExecutableAssembler, PreparedExecutableParts,
    PreparedExecutionError, PreparedExecutionRoutes, PreparedPartitionPredictionResources,
    PreparedPartitionResources, ReplicatedRoute, RoutedRoute, construct_prepared_execution,
};

pub(crate) fn materialize_model_plan(
    sources: PreparedModelSources,
    distributed: Option<MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    materialize_model_plan_with_layerwise_manager(
        sources,
        distributed,
        stream,
        weights_stream,
        None,
            None,
        )
}

/// Receives source preparation completed before ordinary native load ownership.
/// The move-only manager reaches the same typed binder as ordinary loading.
pub(crate) fn materialize_model_plan_with_layerwise_manager(
    sources: PreparedModelSources,
    distributed: Option<MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
    layerwise_manager: Option<
        crate::backend::runtime::execution::generic::PreparedLayerwiseManager,
    >,
    addressable_manager: Option<crate::backend::runtime::residency::parameter_bank::PreparedAddressableSource>,
) -> Result<MlxModel, Error> {
    let capture_discovery = sources.prepare_discovery(
        eredu_core::ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            routed_unit_tensors: true,
            floating_to_f32: true,
        },
        super::super::session::bounded_capture::capabilities(),
    );
    let target = crate::backend::MlxPreparedTarget::new(stream, distributed.as_ref())?;
    let workspace = selected_workspace_mechanisms(stream)?;
    let materialize = |prepared, source: eredu_checkpoint::store::RetainedCheckpointSource| {
        binding::materialize_prediction_extension(prepared, source, stream, weights_stream)
    };
    let layerwise_manager = std::cell::Cell::new(layerwise_manager);
    let addressable_manager = std::cell::Cell::new(addressable_manager);
    let prediction_visitor = |facts: PredictionBinding| binding::PredictionBindingVisitor {
        addressable_manager: Some(&addressable_manager),
        stream,
        weights_stream,
        layerwise_manager: Some(&layerwise_manager),
        selected: facts.selected().clone(),
        capability: facts.capability().clone(),
    };
    let partition_prediction_visitor =
        |facts: PreparedPartitionPredictionResources<MlxDistributedSession>| {
            binding::PartitionedPredictionBindingVisitor {
        addressable_manager: Some(&addressable_manager),
                stream,
                weights_stream,
                distributed: facts.partition().communication().clone(),
                additional_claimed_sources: facts.partition().extension_sources().clone(),
                selected: facts.prediction().selected().clone(),
                capability: facts.prediction().capability().clone(),
            }
        };
    // Both visitors borrow the same stack slot; only the architecture-selected
    // route moves the already-admitted manager into its actual mechanisms.
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(
            ReplicatedRoute::<MlxNeuralBackend, _>::new(
                stream, weights_stream,
                eredu_architectures::replicated_text::SharedReplicatedTextVisitor::<
                    binding::MlxReplicatedStateProfiles, _
                >::new(binding::BindingVisitor { stream, weights_stream, layerwise_manager: Some(&layerwise_manager) }),
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, prediction_visitor,
            )
        )
        .with_routed(
            RoutedRoute::<MlxNeuralBackend, MlxHybridState, MlxPoolingAttentionState, _, _, _>::new(
                stream, weights_stream,
                binding::RoutedBindingVisitor {
        addressable_manager: Some(&addressable_manager), stream, weights_stream, layerwise_manager: Some(&layerwise_manager) },
                binding::Relu2RoutedBindingVisitor {
        addressable_manager: Some(&addressable_manager), stream, weights_stream, layerwise_manager: Some(&layerwise_manager) },
                binding::PoolingRoutedBindingVisitor {
        addressable_manager: Some(&addressable_manager), stream, weights_stream, layerwise_manager: Some(&layerwise_manager) },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, prediction_visitor,
            )
        )
        .with_composite(
            CompositeRoute::<MlxNeuralBackend, MlxHybridState, _>::new(
                stream, weights_stream, binding::CompositeBindingVisitor {
        addressable_manager: Some(&addressable_manager), stream, weights_stream, layerwise_manager: Some(&layerwise_manager) },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, prediction_visitor,
            )
        )
        .with_partitioned_dense(
            PartitionedDenseRoute::<MlxNeuralBackend, MlxHybridState, _>::new(
                stream, weights_stream, |resources: PreparedPartitionResources<MlxDistributedSession>| {
                    binding::PartitionedDenseDecoderBindingVisitor {
                        layerwise_manager: Some(&layerwise_manager),
                        distributed: resources.communication().clone(),
                        additional_claimed_sources: resources.extension_sources().clone(),
                        stream, weights_stream,
                    }
                },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, partition_prediction_visitor,
            )
        )
        .with_partitioned_routed(
            PartitionedRoutedRoute::<MlxNeuralBackend, MlxHybridState, MlxPoolingAttentionState, _, _>::new(
                stream, weights_stream,
                |resources: PreparedPartitionResources<MlxDistributedSession>| binding::PartitionedRoutedDecoderBindingVisitor {
        addressable_manager: Some(&addressable_manager),
                    distributed: resources.communication().clone(),
                    additional_claimed_sources: resources.extension_sources().clone(),
                    stream, weights_stream,
                },
                |resources: PreparedPartitionResources<MlxDistributedSession>| binding::PartitionedPoolingRoutedDecoderBindingVisitor {
        addressable_manager: Some(&addressable_manager),
                    distributed: resources.into_communication(), stream, weights_stream,
                },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, partition_prediction_visitor,
            )
        )
        .with_partitioned_composite(
            PartitionedCompositeRoute::<MlxNeuralBackend, MlxHybridState, _>::new(
                stream, weights_stream,
                |resources: PreparedPartitionResources<MlxDistributedSession>| binding::PartitionedCompositeBindingVisitor {
        addressable_manager: Some(&addressable_manager),
                    store: resources.target().clone(),
                    distributed: resources.into_communication(), stream, weights_stream,
                },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize,
                |facts: PreparedPartitionPredictionResources<MlxDistributedSession>| binding::PartitionedCompositePredictionBindingVisitor {
        addressable_manager: Some(&addressable_manager),
                    store: facts.partition().target().clone(),
                    distributed: facts.partition().communication().clone(),
                    selected: facts.prediction().selected().clone(),
                    capability: facts.prediction().capability().clone(),
                    stream, weights_stream,
                },
            )
        );
    construct_prepared_execution(
        sources,
        distributed,
        routes,
        MlxExecutableAssembler { target, workspace },
    )
    .map_err(|error| match error {
        PreparedExecutionError::Backend(error) => error,
        error => Error::ArchitectureModel(error.to_string()),
    })
    .and_then(|model| model.with_capture_discovery(capture_discovery))
}

struct MlxExecutableAssembler {
    workspace: Option<crate::backend::nn::workspace::ResidentExecutionMechanisms>,
    target: crate::backend::MlxPreparedTarget,
}

impl PreparedExecutableAssembler<MlxDistributedSession> for MlxExecutableAssembler {
    type Executable = Box<dyn binding::ErasedReplicatedTextExecutable>;
    type Output = MlxModel;
    type Error = Error;

    fn floating_state_dtype(
        &mut self,
        source: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Error> {
        super::super::replicated_text::floating_state_storage_dtype(source.dtype()).ok_or_else(
            || {
                Error::ArchitectureModel(format!(
                    "floating-state dtype source {:?} has unsupported MLX activation dtype {:?}",
                    source.checkpoint_tensor(),
                    source.dtype()
                ))
            },
        )
    }

    fn validate_communication(
        &mut self,
        manifest: &eredu_runtime::CommunicationManifest,
        communication: &MlxDistributedSession,
    ) -> Result<(), Error> {
        communication.validate_selected_manifest(manifest)
    }

    fn finish(
        self,
        mut parts: PreparedExecutableParts<Self::Executable, MlxDistributedSession>,
    ) -> Result<MlxModel, Error> {
        let inference = parts.inference_blueprint().clone();
        let floating_state_bytes = parts.floating_state_bytes();
        let state_residency = parts.state_residency().clone();
        let communication = parts.take_communication();
        let processor = parts.take_processor();
        #[cfg(not(any(feature = "image", feature = "audio")))]
        if processor.is_some() {
            return Err(Error::ArchitectureModel(
                "selected raw-media execution has no compiled MLX mechanisms".into(),
            ));
        }
        let mut model = MlxModel::new(
            Executable::new(parts.into_executable())
                .with_inference_blueprint(inference, self.workspace),
            floating_state_bytes,
            state_residency,
            self.target,
        );
        if let Some(communication) = communication {
            model = model.with_distributed(communication);
        }
        #[cfg(any(feature = "image", feature = "audio"))]
        {
            model = model.with_processor(processor.map(ModelProcessor::from_prepared));
        }
        Ok(model)
    }
}

#[cfg(test)]
pub(in crate::composition::mlx) fn bind_replicated_text(
    architecture_plan: &ArtifactArchitecturePlan,
    selected: eredu_runtime::SelectedReplicatedTextRealization,
    store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn super::super::replicated_text::ErasedReplicatedTextExecutable>, Error> {
    let store = store.into();
    let visitor = super::super::replicated_text::BindingVisitor {
        stream,
        weights_stream,
        layerwise_manager: None,
    };
    eredu_architectures::replicated_text::dispatch_replicated_text_architecture::<
        crate::backend::nn::shared::MlxNeuralBackend,
        _,
    >(
        architecture_plan,
        selected,
        store,
        stream,
        eredu_architectures::replicated_text::SharedReplicatedTextVisitor::<
            super::super::replicated_text::MlxReplicatedStateProfiles,
            _,
        >::new(visitor),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

#[cfg(test)]
pub(super) fn inspected_floating_state_dtype_bytes(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<std::num::NonZeroU8, Error> {
    let source = eredu_architectures::preparation::prepared_floating_state_dtype_source(inspection)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    mlx_floating_state_dtype_bytes(source.dtype()).map_err(|dtype| {
        Error::ArchitectureModel(format!(
            "floating-state dtype source {:?} has unsupported MLX activation dtype {dtype:?}",
            source.checkpoint_tensor()
        ))
    })
}

#[cfg(test)]
pub(super) fn mlx_floating_state_dtype_bytes(
    dtype: &eredu_core::checkpoint::TensorDtype,
) -> Result<std::num::NonZeroU8, eredu_core::checkpoint::TensorDtype> {
    super::super::replicated_text::floating_state_storage_dtype(dtype)
        .map(eredu_runtime::StateStorageDtype::bytes)
        .ok_or_else(|| dtype.clone())
}

#[cfg(test)]
pub(in crate::composition::mlx) fn prepared_safetensors_architecture(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::SafetensorsArchitecturePlan, Error> {
    plan.safetensors_architecture().ok_or_else(|| {
        Error::ArchitectureModel(
            "SafeTensors preparation omitted its validated architecture plan".into(),
        )
    })
}

fn selected_workspace_mechanisms(
    _stream: &Stream,
) -> Result<Option<crate::backend::nn::workspace::ResidentExecutionMechanisms>, Error> {
    #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
    {
        use crate::backend::nn::workspace::{MlxMetalWorkspaceMechanisms, ResidentExecutionMechanisms};
        let ordinary = MlxMetalWorkspaceMechanisms::current_host()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        return ResidentExecutionMechanisms::from_cold_stream(ordinary, _stream)
            .map(Some).map_err(|error| Error::ArchitectureModel(error.to_string()));
    }
    #[cfg(not(all(target_vendor = "apple", feature = "metal", not(feature = "cuda"))))]
    Ok(None)
}
