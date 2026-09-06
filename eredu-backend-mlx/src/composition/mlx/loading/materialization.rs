use super::super::replicated_text as binding;
use super::*;
use crate::backend::{
    distributed::MlxDistributedSession,
    runtime::cache::state::{MlxHybridState, MlxPoolingAttentionState},
};
use eredu_architectures::prepared_execution::{
    construct_prepared_execution, CompositeRoute, PartitionedCompositeRoute, PartitionedDenseRoute,
    PartitionedRoutedRoute, PredictionBinding, PreparedExecutableAssembler,
    PreparedExecutableParts, PreparedExecutionError, PreparedExecutionRoutes,
    PreparedPartitionPredictionResources, PreparedPartitionResources, ReplicatedRoute, RoutedRoute,
};

pub(crate) fn materialize_model_plan(
    sources: PreparedModelSources,
    distributed: Option<MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let target = crate::backend::MlxPreparedTarget::new(stream, distributed.as_ref())?;
    let materialize = |prepared, source: eredu_checkpoint::store::SharedCheckpointSource| {
        binding::materialize_prediction_extension(prepared, source.as_ref(), stream, weights_stream)
    };
    let prediction_visitor = |facts: PredictionBinding| binding::PredictionBindingVisitor {
        stream,
        weights_stream,
        selected: facts.selected().clone(),
        capability: facts.capability().clone(),
    };
    let partition_prediction_visitor =
        |facts: PreparedPartitionPredictionResources<MlxDistributedSession>| {
            binding::PartitionedPredictionBindingVisitor {
                stream,
                weights_stream,
                distributed: facts.partition().communication().clone(),
                additional_claimed_sources: facts.partition().extension_sources().clone(),
                selected: facts.prediction().selected().clone(),
                capability: facts.prediction().capability().clone(),
            }
        };
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(
            ReplicatedRoute::<MlxNeuralBackend, _>::new(
                stream, weights_stream,
                eredu_architectures::replicated_text::SharedReplicatedTextVisitor::<
                    binding::MlxReplicatedStateProfiles, _
                >::new(binding::BindingVisitor { stream, weights_stream }),
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, prediction_visitor,
            )
        )
        .with_routed(
            RoutedRoute::<MlxNeuralBackend, MlxHybridState, MlxPoolingAttentionState, _, _, _>::new(
                stream, weights_stream,
                binding::RoutedBindingVisitor { stream, weights_stream },
                binding::Relu2RoutedBindingVisitor { stream, weights_stream },
                binding::PoolingRoutedBindingVisitor { stream, weights_stream },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, prediction_visitor,
            )
        )
        .with_composite(
            CompositeRoute::<MlxNeuralBackend, MlxHybridState, _>::new(
                stream, weights_stream, binding::CompositeBindingVisitor { stream, weights_stream },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize, prediction_visitor,
            )
        )
        .with_partitioned_dense(
            PartitionedDenseRoute::<MlxNeuralBackend, MlxHybridState, _>::new(
                stream, weights_stream, |resources: PreparedPartitionResources<MlxDistributedSession>| {
                    binding::PartitionedDenseDecoderBindingVisitor {
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
                    distributed: resources.communication().clone(),
                    additional_claimed_sources: resources.extension_sources().clone(),
                    stream, weights_stream,
                },
                |resources: PreparedPartitionResources<MlxDistributedSession>| binding::PartitionedPoolingRoutedDecoderBindingVisitor {
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
                    store: Arc::clone(resources.target()),
                    distributed: resources.into_communication(), stream, weights_stream,
                },
            ).with_prediction::<binding::MlxEmbeddedPredictionMaterializer, _, _>(
                materialize,
                |facts: PreparedPartitionPredictionResources<MlxDistributedSession>| binding::PartitionedCompositePredictionBindingVisitor {
                    store: Arc::clone(facts.partition().target()),
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
        MlxExecutableAssembler { target },
    )
    .map_err(|error| match error {
        PreparedExecutionError::Backend(error) => error,
        error => Error::ArchitectureModel(error.to_string()),
    })
}

struct MlxExecutableAssembler {
    target: crate::backend::MlxPreparedTarget,
}

impl PreparedExecutableAssembler<MlxDistributedSession> for MlxExecutableAssembler {
    type Executable = Box<dyn binding::ErasedReplicatedTextExecutable>;
    type Output = MlxModel;
    type Error = Error;

    fn floating_state_bytes(
        &mut self,
        source: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<std::num::NonZeroU8, Error> {
        mlx_floating_state_dtype_bytes(source.dtype()).map_err(|dtype| {
            Error::ArchitectureModel(format!(
                "floating-state dtype source {:?} has unsupported MLX activation dtype {dtype:?}",
                source.checkpoint_tensor()
            ))
        })
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
            Executable::new(parts.into_executable()),
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
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn super::super::replicated_text::ErasedReplicatedTextExecutable>, Error> {
    let visitor = super::super::replicated_text::BindingVisitor {
        stream,
        weights_stream,
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

pub(super) fn mlx_floating_state_dtype_bytes(
    dtype: &eredu_core::checkpoint::TensorDtype,
) -> Result<std::num::NonZeroU8, eredu_core::checkpoint::TensorDtype> {
    use eredu_core::checkpoint::TensorDtype;

    let bytes = match dtype {
        TensorDtype::F16 | TensorDtype::Bf16 => 2,
        TensorDtype::F32 => 4,
        TensorDtype::F64 | TensorDtype::Complex64 => 8,
        // MLX materializes supported packed embeddings as Float32 activations.
        // These cases are reached only after the architecture schema resolved
        // the exact embedding parameter; they are not a fallback for an
        // unknown checkpoint name.
        TensorDtype::U32 | TensorDtype::Encoded(_) => 4,
        dtype => return Err(dtype.clone()),
    };
    Ok(std::num::NonZeroU8::new(bytes).expect("supported MLX activation widths are nonzero"))
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
