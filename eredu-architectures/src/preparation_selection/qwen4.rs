//! Request-specific header authority; never retained in a request-independent cache.
use super::*;
use crate::selected_execution::SelectedQwen4Construction;
use eredu_runtime::StateStorageDtype;

pub(super) fn select(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    request: &NormalizedLoadRequest,
    mechanisms: &impl PreparationMechanismProvider,
    admission: eredu_core::PreparationAdmission,
) -> Result<SelectedPreparation, PreparationSelectionError> {
    let mut request = request.clone();
    let gguf_projector = inspection
        .architecture_plan()
        .gguf_media_projector()
        .is_some();
    let safetensors_vision = inspection
        .architecture_plan()
        .safetensors_architecture()
        .and_then(|plan| plan.qwen4_exp_target_plan())
        .is_some_and(|header| header.configuration().vision.is_some());
    if gguf_projector || safetensors_vision {
        match request.media_execution() {
            eredu_runtime::MediaLoadRequest::Disabled if gguf_projector => {
                return Err(PreparationSelectionError::MediaDisabledForComposite)
            }
            eredu_runtime::MediaLoadRequest::Disabled => {}
            eredu_runtime::MediaLoadRequest::ArchitectureDefault => {
                let processor = eredu_runtime::ProcessorSelectionRequest::new([
                    eredu_core::InputModality::Image,
                    eredu_core::InputModality::Video,
                ])
                .with_projected_embeddings(true)
                .with_available_raw_media(true);
                request = request.with_media_execution(eredu_runtime::MediaLoadRequest::Required(
                    eredu_runtime::MediaExecutionPolicy::new(processor)
                        .expect("default image/video intent includes media"),
                ));
            }
            eredu_runtime::MediaLoadRequest::Required(_) => {}
        }
    }
    let source = crate::preparation::prepared_floating_state_dtype_source(inspection)?;
    let element = match mechanisms.floating_state_dtype(source.dtype()) {
        Some(StateStorageDtype::F32) => eredu_nn::TensorElementType::F32,
        Some(StateStorageDtype::F16) => eredu_nn::TensorElementType::F16,
        Some(StateStorageDtype::Bf16) => eredu_nn::TensorElementType::Bf16,
        _ => {
            return Err(std::sync::Arc::new(
                crate::qwen4_exp::prepared::TargetLoadError::StateRepresentation,
            )
            .into())
        }
    };
    // The ordinary driver constructs every facility admitted by the backend,
    // even when the caller required only a subset of those facilities.
    let request = request
        .clone()
        .with_required_session_capabilities(admission.session_capabilities());
    let rows = PreparationRowLookupMechanisms::new(mechanisms);
    let load_error = |error| PreparationSelectionError::Qwen4TargetLoad(std::sync::Arc::new(error));
    let select_error =
        |error| PreparationSelectionError::Qwen4TargetSelection(std::sync::Arc::new(error));
    let mut prediction_realization = None;
    let construction = match inspection.format() {
        ArtifactFormat::SafeTensors => {
            let header = inspection
                .architecture_plan()
                .safetensors_architecture()
                .and_then(|plan| plan.qwen4_exp_target_plan())
                .ok_or_else(|| {
                    PreparationSelectionError::ExecutionClass(
                        "admitted Qwen4 SafeTensors header is missing".into(),
                    )
                })?;
            let request = crate::qwen4_exp::prepared::normalize_load_request(
                &request,
                header.configuration(),
            )
            .map_err(load_error)?;
            let physical = header
                .resolution()
                .source_keys()
                .iter()
                .map(|key| {
                    let tensor = inspection.tensors().get(key).ok_or_else(|| {
                        PreparationSelectionError::ExecutionClass(format!(
                            "missing admitted tensor {key}"
                        ))
                    })?;
                    let storage = tensor.storage.as_ref().ok_or_else(|| {
                        PreparationSelectionError::ExecutionClass(format!(
                            "missing physical storage for {key}"
                        ))
                    })?;
                    let dtype =
                        crate::replicated_text::stored_dtype(&tensor.dtype).map_err(|error| {
                            PreparationSelectionError::ExecutionClass(error.to_string())
                        })?;
                    let physical = eredu_runtime::ReplicatedTextPhysicalSource::new(
                        key,
                        key,
                        std::path::PathBuf::from(&storage.member),
                        key,
                        eredu_checkpoint::SourceTensorEncoding::Safetensors(dtype),
                        storage.length,
                    )
                    .map_err(|error| {
                        PreparationSelectionError::ExecutionClass(error.to_string())
                    })?;
                    Ok((key.clone(), physical))
                })
                .collect::<Result<std::collections::BTreeMap<_, _>, PreparationSelectionError>>()?;
            let vision = match request.media_execution() {
                eredu_runtime::MediaLoadRequest::Required(_) => Some(
                    retain_processor(
                        header
                            .vision_plan(physical.clone())
                            .map_err(|e| load_error(e.into()))?,
                        inspection,
                    )
                    .map_err(|e| load_error(e.into()))?,
                ),
                _ => None,
            };
            let embedded = request.drafting().embedded_capacity();
            let target_request = if embedded.is_some() {
                request.clone().with_drafting(DraftingLoadRequest::Disabled)
            } else {
                request.clone()
            }
            .with_media_execution(eredu_runtime::MediaLoadRequest::Disabled);
            let mut plan = header
                .execution_plan_for_load(&target_request, element, &rows, physical.clone())
                .map_err(load_error)?;
            let target_limits =
                crate::qwen4_exp::target::TargetLimits::from_load_request(&request, element)
                    .map_err(load_error)?;
            if embedded.is_some() {
                let limits = prediction_limits(target_limits, element);
                let spec = header
                    .prediction_spec(limits)
                    .map_err(|e| load_error(e.into()))?;
                let streams = prediction_streams(&spec, &request, limits);
                plan = plan
                    .with_prediction(limits, streams)
                    .map_err(|e| load_error(e.into()))?;
            }
            if let eredu_runtime::MediaLoadRequest::Required(media) = request.media_execution() {
                let vision = vision.expect("required media retains its vision header");
                let plan = plan
                    .with_vision(vision)
                    .map_err(|e| load_error(e.into()))?
                    .with_load_processor(media.processor().clone(), media.processor_budget());
                if request.has_parallel_execution() {
                    return finish_conditional_partition(
                        inspection,
                        plan,
                        &request,
                        target_limits,
                        None,
                        mechanisms,
                        admission,
                    );
                }
                return finish_conditional(
                    inspection,
                    plan,
                    &request,
                    target_limits,
                    media.processor(),
                    None,
                    mechanisms,
                    admission,
                );
            }
            if request.has_parallel_execution() {
                let partition = crate::qwen4_exp::prepared::load_partition_request(&request)
                    .map_err(load_error)?;
                let plan = plan
                    .partition(partition)
                    .map_err(|e| load_error(e.into()))?;
                return finish_partition(
                    inspection,
                    plan,
                    &request,
                    target_limits,
                    None,
                    mechanisms,
                    admission,
                );
            }
            let selection = plan
                .load_selection_request()
                .expect("normalized target retains policy")
                .clone();
            let facts = mechanisms
                .replicated_text_capabilities(plan.requirements().text(), selection.text());
            let prediction_facts = plan
                .prediction_state_requirements()
                .map(|state| mechanisms.state_capabilities(state, request.state_residency()));
            let selected = plan
                .select(&selection, &facts, prediction_facts.as_ref())
                .map_err(select_error)?;
            if let Some(capacity) = embedded {
                let spec = selected
                    .prediction_spec()
                    .map_err(|e| load_error(e.into()))?;
                prediction_realization = Some(select_prediction(
                    &spec,
                    target_limits,
                    selected.selected(),
                    capacity,
                    &request,
                    inspection.format(),
                    mechanisms,
                )?);
            }
            SelectedQwen4Construction::Safetensors(selected)
        }
        ArtifactFormat::Gguf => {
            let header = inspection
                .architecture_plan()
                .gguf_plan()
                .and_then(|plan| plan.qwen4_exp_target_plan())
                .ok_or_else(|| {
                    PreparationSelectionError::ExecutionClass(
                        "admitted Qwen4 GGUF header is missing".into(),
                    )
                })?;
            let request = crate::qwen4_exp::prepared::normalize_load_request(
                &request,
                header.text_plan().config(),
            )
            .map_err(load_error)?;
            let vision = match request.media_execution() {
                eredu_runtime::MediaLoadRequest::Required(_) => {
                    let vision = match inspection
                        .architecture_plan()
                        .gguf_media_projector()
                        .map(|p| p.model())
                    {
                        Some(crate::gguf_companion::GgufMediaProjectorConfig::Qwen4Exp(vision)) => {
                            vision.clone()
                        }
                        _ => {
                            return Err(PreparationSelectionError::ExecutionClass(
                                "required Flash-Next media has no bound projector declaration"
                                    .into(),
                            ))
                        }
                    };
                    Some(retain_processor(vision, inspection).map_err(|e| load_error(e.into()))?)
                }
                _ => None,
            };
            let embedded = request.drafting().embedded_capacity();
            let prediction = if embedded.is_some() {
                let path = request.prediction_source().ok_or_else(|| {
                    load_error(
                        crate::qwen4_exp::prepared::PreparationError::MissingPrediction.into(),
                    )
                })?;
                Some(std::sync::Arc::new(
                    crate::prepared_sources::prediction::InspectedPredictionSource::inspect(path)
                        .map_err(|error| {
                        PreparationSelectionError::PredictionSource(std::sync::Arc::new(error))
                    })?,
                ))
            } else {
                None
            };
            let target_request = request
                .clone()
                .without_prediction_source()
                .with_drafting(DraftingLoadRequest::Disabled)
                .with_media_execution(eredu_runtime::MediaLoadRequest::Disabled);
            let mut plan = header
                .execution_plan_for_load(&target_request, element, &rows)
                .map_err(load_error)?;
            let target_limits =
                crate::qwen4_exp::target::TargetLimits::from_load_request(&request, element)
                    .map_err(load_error)?;
            if let Some(prediction) = &prediction {
                let limits = prediction_limits(target_limits, element);
                let spec = prediction
                    .header_plan()
                    .prediction_spec(limits)
                    .map_err(|error| load_error(error.into()))?;
                let streams = prediction_streams(&spec, &request, limits);
                plan = plan
                    .with_prediction(prediction.header_plan().clone(), limits, streams)
                    .map_err(|error| load_error(error.into()))?;
            }
            if let eredu_runtime::MediaLoadRequest::Required(media) = request.media_execution() {
                let vision = vision.expect("required media retains its vision header");
                let plan = plan
                    .with_vision(vision)
                    .map_err(|e| load_error(e.into()))?
                    .with_load_processor(media.processor().clone(), media.processor_budget());
                if request.has_parallel_execution() {
                    return finish_conditional_partition(
                        inspection,
                        plan,
                        &request,
                        target_limits,
                        prediction.clone(),
                        mechanisms,
                        admission,
                    );
                }
                return finish_conditional(
                    inspection,
                    plan,
                    &request,
                    target_limits,
                    media.processor(),
                    prediction,
                    mechanisms,
                    admission,
                );
            }
            if request.has_parallel_execution() {
                let partition = crate::qwen4_exp::prepared::load_partition_request(&request)
                    .map_err(load_error)?;
                let plan = plan
                    .partition(partition)
                    .map_err(|e| load_error(e.into()))?;
                return finish_partition(
                    inspection,
                    plan,
                    &request,
                    target_limits,
                    prediction,
                    mechanisms,
                    admission,
                );
            }
            let selection = plan
                .load_selection_request()
                .expect("normalized target retains policy")
                .clone();
            let facts = mechanisms
                .replicated_text_capabilities(plan.requirements().text(), selection.text());
            let prediction_facts = plan
                .prediction_state_requirements()
                .map(|state| mechanisms.state_capabilities(state, request.state_residency()));
            let selected = plan
                .select(&selection, &facts, prediction_facts.as_ref())
                .map_err(select_error)?;
            if let Some(capacity) = embedded {
                prediction_realization = Some(select_prediction(
                    &selected
                        .prediction_spec()
                        .map_err(|error| load_error(error.into()))?,
                    target_limits,
                    selected.selected(),
                    capacity,
                    &request,
                    inspection.format(),
                    mechanisms,
                )?);
            }
            SelectedQwen4Construction::Gguf {
                selected,
                prediction,
            }
        }
        format => return Err(PreparationSelectionError::MissingArchitecturePlan { format }),
    };
    let selected = SelectedPreparation::new_qwen4(
        inspection.admission_token(),
        construction,
        admission,
        mechanisms.input_score_attention_workspace(),
    );
    Ok(match prediction_realization {
        Some(prediction) => selected.with_retained_prediction(prediction),
        None => selected,
    })
}

fn finish_conditional_partition(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    plan: crate::qwen4_exp::prepared::ConditionalHeaderExecutionPlan,
    load: &NormalizedLoadRequest,
    limits: crate::qwen4_exp::target::TargetLimits,
    prediction: Option<
        std::sync::Arc<crate::prepared_sources::prediction::InspectedPredictionSource>,
    >,
    mechanisms: &impl PreparationMechanismProvider,
    admission: eredu_core::PreparationAdmission,
) -> Result<SelectedPreparation, PreparationSelectionError> {
    let partition = crate::qwen4_exp::prepared::load_partition_request(load)
        .map_err(|error| PreparationSelectionError::Qwen4TargetLoad(std::sync::Arc::new(error)))?;
    let plan = plan.partition(inspection, partition).map_err(|error| {
        PreparationSelectionError::Qwen4TargetLoad(std::sync::Arc::new(error.into()))
    })?;
    let request = plan
        .load_selection_request()
        .expect("normalized conditional partition retains policy")
        .clone();
    let eredu_runtime::MediaLoadRequest::Required(media) = load.media_execution() else {
        return Err(PreparationSelectionError::MediaDisabledForComposite);
    };
    let facts = mechanisms
        .replicated_text_capabilities(plan.requirements().execution().execution(), request.text());
    let prediction_facts = plan
        .prediction_state_requirements()
        .map(|state| mechanisms.state_capabilities(state, load.state_residency()));
    let selected = plan
        .select(
            &request,
            &facts,
            prediction_facts.as_ref(),
            &mechanisms.communication_capabilities(),
            media.processor(),
            &mechanisms.processor_capabilities(),
        )
        .map_err(|error| {
            PreparationSelectionError::Qwen4TargetSelection(std::sync::Arc::new(error))
        })?;
    let realization = load
        .drafting()
        .embedded_capacity()
        .map(|capacity| {
            let spec = selected.prediction_spec().map_err(|e| {
                PreparationSelectionError::Qwen4TargetLoad(std::sync::Arc::new(e.into()))
            })?;
            let crate::replicated_text::SelectedCompositeTextRealization::Routed {
                execution, ..
            } = selected.selected().base()
            else {
                unreachable!("retained routed conditional selection")
            };
            select_prediction(
                &spec,
                limits,
                execution,
                capacity,
                load,
                inspection.format(),
                mechanisms,
            )
        })
        .transpose()?;
    let selected = SelectedPreparation::new_qwen4(
        inspection.admission_token(),
        SelectedQwen4Construction::ConditionalPartitioned {
            selected,
            prediction,
        },
        admission,
        mechanisms.input_score_attention_workspace(),
    );
    Ok(match realization {
        Some(realization) => selected.with_retained_prediction(realization),
        None => selected,
    })
}

fn finish_partition(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    plan: crate::qwen4_exp::prepared::TargetPartitionExecutionPlan,
    load: &NormalizedLoadRequest,
    limits: crate::qwen4_exp::target::TargetLimits,
    prediction: Option<
        std::sync::Arc<crate::prepared_sources::prediction::InspectedPredictionSource>,
    >,
    mechanisms: &impl PreparationMechanismProvider,
    admission: eredu_core::PreparationAdmission,
) -> Result<SelectedPreparation, PreparationSelectionError> {
    let request = plan
        .load_selection_request()
        .expect("normalized partition retains policy")
        .clone();
    let facts = mechanisms
        .replicated_text_capabilities(plan.requirements().execution().text(), request.text());
    let prediction_facts = plan
        .prediction_state_requirements()
        .map(|state| mechanisms.state_capabilities(state, load.state_residency()));
    let selected = plan
        .select(
            &request,
            &facts,
            prediction_facts.as_ref(),
            &mechanisms.communication_capabilities(),
        )
        .map_err(|error| {
            PreparationSelectionError::Qwen4TargetSelection(std::sync::Arc::new(error))
        })?;
    let realization = load
        .drafting()
        .embedded_capacity()
        .map(|capacity| {
            let spec = selected.prediction_spec().map_err(|e| {
                PreparationSelectionError::Qwen4TargetLoad(std::sync::Arc::new(e.into()))
            })?;
            select_prediction(
                &spec,
                limits,
                selected.selected().base(),
                capacity,
                load,
                inspection.format(),
                mechanisms,
            )
        })
        .transpose()?;
    let selected = SelectedPreparation::new_qwen4(
        inspection.admission_token(),
        SelectedQwen4Construction::Partitioned {
            selected,
            prediction,
        },
        admission,
        mechanisms.input_score_attention_workspace(),
    );
    Ok(match realization {
        Some(realization) => selected.with_retained_prediction(realization),
        None => selected,
    })
}

fn select_prediction(
    spec: &crate::qwen4_exp::mtp::PredictionSpec,
    limits: crate::qwen4_exp::target::TargetLimits,
    selected: &crate::SelectedRoutedTextRealization,
    capacity: NonZeroUsize,
    load: &NormalizedLoadRequest,
    format: ArtifactFormat,
    mechanisms: &impl PreparationMechanismProvider,
) -> Result<eredu_runtime::SelectedSpeculativeRealization, PreparationSelectionError> {
    let identity = selected.text().requirements().architecture_identity();
    let id = |value: &str| eredu_runtime::SpeculativeIdentity::new(value).map_err(prediction_error);
    let topology = match load.parallel_execution() {
        Some(parallel) => parallel.rank(),
        None => ParallelRankTopology::new(
            ParallelTopology::new(1, 1, 1, 1).map_err(prediction_error)?,
            0,
        )
        .map_err(prediction_error)?,
    };
    let contract = crate::prediction_extension::qwen4_speculative_contract(
        spec,
        identity,
        limits,
        EmbeddedSpeculativeContractRequest::new(
            id(identity)?,
            id("artifact/header/qwen4_exp")?,
            id(&format!("format/{format:?}"))?,
            topology,
            id("prepared-input/text-token-ids/v1")?,
            positive_bound("maximum batch size", limits.qsa.batch)?,
            positive_bound("maximum sequence length", limits.qsa.tokens)?,
            capacity,
        ),
    )
    .map_err(|error| PreparationSelectionError::Prediction(error.into()))?;
    eredu_runtime::select_speculative_realization(
        contract.requirements(),
        &contract.selection_request(eredu_runtime::SpeculativePlacementRequest::Single),
        &mechanisms.speculative_capabilities(),
    )
    .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn finish_conditional(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    plan: crate::qwen4_exp::prepared::ConditionalHeaderExecutionPlan,
    load: &NormalizedLoadRequest,
    limits: crate::qwen4_exp::target::TargetLimits,
    inputs: &eredu_runtime::ProcessorSelectionRequest,
    prediction: Option<
        std::sync::Arc<crate::prepared_sources::prediction::InspectedPredictionSource>,
    >,
    mechanisms: &impl PreparationMechanismProvider,
    admission: eredu_core::PreparationAdmission,
) -> Result<SelectedPreparation, PreparationSelectionError> {
    let request = plan
        .load_selection_request()
        .expect("normalized conditional plan retains policy")
        .clone();
    let facts = mechanisms.replicated_text_capabilities(plan.requirements().text(), request.text());
    let prediction_facts = plan
        .prediction_state_requirements()
        .map(|state| mechanisms.state_capabilities(state, load.state_residency()));
    let selected = plan
        .select(
            &request,
            &facts,
            prediction_facts.as_ref(),
            inputs,
            &mechanisms.processor_capabilities(),
        )
        .map_err(|e| PreparationSelectionError::Qwen4TargetSelection(std::sync::Arc::new(e)))?;
    let realization = load
        .drafting()
        .embedded_capacity()
        .map(|capacity| {
            let spec = selected.prediction_spec().map_err(|e| {
                PreparationSelectionError::Qwen4TargetLoad(std::sync::Arc::new(e.into()))
            })?;
            select_prediction(
                &spec,
                limits,
                selected.realization(),
                capacity,
                load,
                inspection.format(),
                mechanisms,
            )
        })
        .transpose()?;
    let selected = SelectedPreparation::new_qwen4(
        inspection.admission_token(),
        SelectedQwen4Construction::Conditional {
            selected,
            prediction,
        },
        admission,
        mechanisms.input_score_attention_workspace(),
    );
    Ok(match realization {
        Some(realization) => selected.with_retained_prediction(realization),
        None => selected,
    })
}

fn retain_processor(
    vision: crate::qwen4_exp::prepared::VisionPlan,
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<crate::qwen4_exp::prepared::VisionPlan, crate::qwen4_exp::prepared::PreparationError> {
    match inspection.architecture_plan().qwen() {
        Some(processor) => vision.with_processor(processor.clone()),
        None => Ok(vision),
    }
}

fn prediction_limits(
    limits: crate::qwen4_exp::target::TargetLimits,
    element: eredu_nn::TensorElementType,
) -> crate::qwen4_exp::mtp::PredictionLimits {
    crate::qwen4_exp::mtp::PredictionLimits {
        qsa: limits.qsa,
        tile_blocks: limits.tile_blocks,
        selection_workspace: limits.selection_workspace,
        element,
    }
}

fn prediction_streams(
    spec: &crate::qwen4_exp::mtp::PredictionSpec,
    request: &NormalizedLoadRequest,
    limits: crate::qwen4_exp::mtp::PredictionLimits,
) -> Vec<eredu_runtime::AppendStreamBinding> {
    let append = request
        .bounded_execution()
        .expect("normalized policy")
        .append();
    spec.units
        .iter()
        .enumerate()
        .flat_map(|(layer, unit)| match &unit.mixer {
            crate::qwen4_exp::target::MixerSpec::Indexed(attention) => attention
                .state
                .streams()
                .into_iter()
                .map(|spec| eredu_runtime::AppendStreamBinding {
                    layer,
                    lanes: limits.qsa.batch as u32,
                    spec,
                    limits: append.limits(),
                    payload_bytes: append.payload_bytes(),
                    scratch_bytes: append.scratch_bytes(),
                    catalog_bytes: append.catalog_bytes(),
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}
