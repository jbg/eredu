use super::*;

pub(crate) fn materialize_model_plan(
    sources: PreparedModelSources,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let (selected, inspection, sources) = sources.into_parts();
    let (execution, _, _, prediction_realization) = selected.into_parts();
    let materializer = MlxSelectedExecutionMaterializer {
        inspection,
        sources,
        prediction_realization,
        distributed,
        stream,
        weights_stream,
    };
    execution.dispatch(materializer)
}

fn materialize_partitioned_composite(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::replicated_text::SelectedCompositeTextRealization,
        eredu_architectures::replicated_text::CompositeTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral composite binding has no realized communication".into())
    })?;
    let retained_distributed = distributed.clone();
    let state_residency = selected.base().execution().state().policy().clone();
    let selected_processor = selected.base().processor().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let target_store = Arc::clone(sources.target());
    let extension_store = sources.extension().cloned();
    let prediction_extension_execution = match (prediction_extension.as_ref(), extension_store) {
        (Some(extension), Some(extension_store)) => {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(
                extension,
                &selected,
                selected
                    .base()
                    .execution()
                    .auxiliary_materialization_tasks(),
                weights_stream,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::super::replicated_text::materialize_prediction_extension(
                prepared,
                extension_store.as_ref(),
                stream,
                weights_stream,
            )
            .map(Some)
            .map_err(|error| {
                Error::ArchitectureModel(format!(
                    "prediction extension materialization failed: {error}"
                ))
            })?
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let executable = match (
        prediction_extension,
        prediction_extension_execution,
        prediction_realization,
    ) {
        (Some(extension), Some(execution), Some(realization)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::composite_partitioned::visit_authoritative_composite_prediction_target_partition::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                super::super::replicated_text::MlxEmbeddedPredictionMaterializer,
                _,
            >(
                selected,
                execution,
                stream,
                super::super::replicated_text::PartitionedCompositePredictionBindingVisitor {
                    store: target_store,
                    distributed,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => super::super::replicated_text::bind_partitioned_composite(
            selected,
            target_store,
            distributed,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    )
    .with_distributed(retained_distributed);
    attach_selected_processor(model, &architecture_plan, &selected_processor)
}

fn materialize_partitioned_routed_decoder(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::RoutedTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral routed-decoder binding has no realized communication".into())
    })?;
    let retained_distributed = distributed.clone();
    let state_residency = selected.base().text().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let prediction_extension_sources = sources
        .extension()
        .map(|source| source.source_keys().into_iter().collect())
        .unwrap_or_default();
    let target_store = Arc::clone(sources.target());
    let extension_store = sources.extension().cloned();
    let prediction_extension_execution = match (prediction_extension.as_ref(), extension_store) {
        (Some(extension), Some(extension_store)) => {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(
                extension,
                &selected,
                selected
                    .base()
                    .text()
                    .auxiliary_materialization_tasks(),
                weights_stream,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::super::replicated_text::materialize_prediction_extension(
                prepared,
                extension_store.as_ref(),
                stream,
                weights_stream,
            )
            .map(Some)
            .map_err(|error| {
                Error::ArchitectureModel(format!(
                    "prediction extension materialization failed: {error}"
                ))
            })?
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let executable = match (
        prediction_extension,
        prediction_extension_execution,
        prediction_realization,
    ) {
        (Some(extension), Some(execution), Some(realization)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::super::replicated_text::bind_partitioned_routed_prediction_decoder(
                &inspection,
                selected,
                execution,
                realization,
                capability,
                target_store,
                distributed,
                prediction_extension_sources,
                stream,
                weights_stream,
            )?
        }
        (None, None, None) => super::super::replicated_text::bind_partitioned_routed_decoder(
            &inspection,
            selected,
            target_store,
            distributed,
            prediction_extension_sources,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    Ok(MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    )
    .with_distributed(retained_distributed))
}

fn materialize_partitioned_dense_decoder(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral dense-decoder binding has no realized communication".into())
    })?;
    let retained_distributed = distributed.clone();
    let state_residency = selected.base().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let prediction_extension_sources = sources
        .extension()
        .map(|source| source.source_keys().into_iter().collect())
        .unwrap_or_default();
    let target_store = Arc::clone(sources.target());
    let extension_store = sources.extension().cloned();
    let prediction_extension_execution = match (prediction_extension.as_ref(), extension_store) {
        (Some(extension), Some(extension_store)) => {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(
                extension,
                &selected,
                selected.base().auxiliary_materialization_tasks(),
                weights_stream,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::super::replicated_text::materialize_prediction_extension(
                prepared,
                extension_store.as_ref(),
                stream,
                weights_stream,
            )
            .map(Some)
            .map_err(|error| {
                Error::ArchitectureModel(format!(
                    "prediction extension materialization failed: {error}"
                ))
            })?
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let executable = match (
        prediction_extension,
        prediction_extension_execution,
        prediction_realization,
    ) {
        (Some(extension), Some(execution), Some(realization)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::partitioned_execution::visit_resident_partitioned_prediction_target_architecture::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                super::super::replicated_text::MlxEmbeddedPredictionMaterializer,
                _,
            >(
                &inspection,
                selected,
                execution,
                target_store,
                stream,
                super::super::replicated_text::PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: prediction_extension_sources,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => super::super::replicated_text::bind_partitioned_dense_decoder(
            &inspection,
            selected,
            target_store,
            distributed,
            prediction_extension_sources,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    Ok(MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    )
    .with_distributed(retained_distributed))
}

struct MlxSelectedExecutionMaterializer<'a> {
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl eredu_architectures::SelectedExecutionDispatcher for MlxSelectedExecutionMaterializer<'_> {
    type Output = MlxModel;
    type Error = Error;

    fn replicated(
        self,
        selected: eredu_runtime::SelectedReplicatedTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        debug_assert!(self.distributed.is_none());
        materialize_replicated_text_plan(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn routed(
        self,
        selected: eredu_architectures::SelectedRoutedTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        debug_assert!(self.distributed.is_none());
        materialize_routed_text_plan(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn composite(
        self,
        selected: eredu_architectures::replicated_text::SelectedCompositeTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        debug_assert!(self.distributed.is_none());
        materialize_composite_text_plan(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn partitioned_dense(
        self,
        selected: eredu_architectures::SelectedDensePartitionedExecution,
    ) -> Result<Self::Output, Self::Error> {
        materialize_partitioned_dense_decoder(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.distributed,
            self.stream,
            self.weights_stream,
        )
    }

    fn partitioned_routed(
        self,
        selected: eredu_architectures::SelectedRoutedPartitionedExecution,
    ) -> Result<Self::Output, Self::Error> {
        materialize_partitioned_routed_decoder(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.distributed,
            self.stream,
            self.weights_stream,
        )
    }

    fn partitioned_composite(
        self,
        selected: eredu_architectures::SelectedCompositePartitionedExecution,
    ) -> Result<Self::Output, Self::Error> {
        materialize_partitioned_composite(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.distributed,
            self.stream,
            self.weights_stream,
        )
    }
}

fn materialize_replicated_prediction_extension(
    extension: &eredu_architectures::configuration::PredictionExtensionPlan,
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<super::super::replicated_text::MaterializedEmbeddedPrediction, Error> {
    let prepared =
        eredu_architectures::prediction_extension::prepare_replicated_prediction_extension::<
            MlxNeuralBackend,
        >(extension, tasks, weights_stream, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    super::super::replicated_text::materialize_prediction_extension(
        prepared,
        store,
        stream,
        weights_stream,
    )
}

fn materialize_replicated_text_plan(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    realization: eredu_runtime::SelectedReplicatedTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let state_residency = realization.state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let materialized = match (prediction_extension.as_ref(), sources.extension()) {
        (Some(extension), Some(extension_store)) => {
            Some(materialize_replicated_prediction_extension(
                extension,
                realization.auxiliary_materialization_tasks(),
                extension_store.as_ref(),
                stream,
                weights_stream,
            )?)
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let target_store = Arc::clone(sources.target());
    let executable = match (prediction_extension, materialized, prediction_realization) {
        (Some(extension), Some(materialized), Some(selected)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::replicated_text::dispatch_replicated_prediction_target_architecture(
                &architecture_plan,
                realization,
                materialized,
                target_store,
                stream,
                super::super::replicated_text::PredictionBindingVisitor {
                    stream,
                    weights_stream,
                    selected,
                    capability,
                },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => bind_replicated_text(
            &architecture_plan,
            realization,
            target_store,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    );
    Ok(model)
}

fn materialize_routed_text_plan(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    realization: eredu_architectures::SelectedRoutedTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let state_residency = realization.text().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let materialized = match (prediction_extension.as_ref(), sources.extension()) {
        (Some(extension), Some(extension_store)) => {
            Some(materialize_replicated_prediction_extension(
                extension,
                realization.text().auxiliary_materialization_tasks(),
                extension_store.as_ref(),
                stream,
                weights_stream,
            )?)
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let target_store = Arc::clone(sources.target());
    let executable = match (prediction_extension, materialized, prediction_realization) {
        (Some(extension), Some(materialized), Some(selected)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::routed_text::dispatch_routed_prediction_target_architecture(
                &inspection,
                realization,
                materialized,
                target_store,
                stream,
                super::super::replicated_text::PredictionBindingVisitor {
                    stream,
                    weights_stream,
                    selected,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => super::super::replicated_text::bind_routed_text(
            &inspection,
            realization,
            target_store,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    );
    Ok(model)
}

fn materialize_composite_text_plan(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    realization: eredu_architectures::replicated_text::SelectedCompositeTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let state_residency = realization.execution().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let requirements =
        eredu_architectures::replicated_text::composite_text_requirements(&inspection)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let selected_processor = realization.processor().clone();
    let prediction_extension = sources.prediction_extension().cloned();
    let materialized = match (prediction_extension.as_ref(), sources.extension()) {
        (Some(extension), Some(extension_store)) => {
            Some(materialize_replicated_prediction_extension(
                extension,
                realization.execution().auxiliary_materialization_tasks(),
                extension_store.as_ref(),
                stream,
                weights_stream,
            )?)
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let target_store = Arc::clone(sources.target());
    let executable = match (prediction_extension, materialized, prediction_realization) {
        (Some(extension), Some(materialized), Some(selected)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::replicated_text::visit_composite_prediction_target_text_architecture::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                super::super::replicated_text::MlxEmbeddedPredictionMaterializer,
                _,
            >(
                requirements,
                realization,
                materialized,
                target_store,
                stream,
                super::super::replicated_text::PredictionBindingVisitor { stream, weights_stream, selected, capability },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => {
            eredu_architectures::replicated_text::visit_composite_text_architecture::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                _,
            >(
                requirements,
                realization,
                target_store,
                stream,
                super::super::replicated_text::CompositeBindingVisitor {
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    );
    attach_selected_processor(model, &architecture_plan, &selected_processor)
}

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
    >(architecture_plan, selected, store, stream, visitor)
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

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

fn attach_selected_processor(
    model: MlxModel,
    architecture_plan: &ArtifactArchitecturePlan,
    selected: &eredu_runtime::SelectedProcessorExecution,
) -> Result<MlxModel, Error> {
    #[cfg(any(feature = "image", feature = "audio"))]
    {
        Ok(model.with_processor(ModelProcessor::from_selected(architecture_plan, selected)?))
    }
    #[cfg(not(any(feature = "image", feature = "audio")))]
    {
        if selected.raw_media() {
            return Err(Error::ArchitectureModel(
                "selected raw-media execution has no compiled MLX mechanisms".into(),
            ));
        }
        let _ = architecture_plan;
        Ok(model)
    }
}
