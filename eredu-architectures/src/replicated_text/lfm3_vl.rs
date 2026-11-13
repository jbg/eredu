//! Cold requirements for the admitted LFM3 VL component.

use super::*;

use eredu_checkpoint::{
    recipe::{ArtifactCatalog, DerivedWeightRecipe, RecipeDtype},
    store::TensorSelection,
};
use eredu_runtime::{
    ModalityProcessorRequirements, ProcessorExecutionRequirements, ProcessorPrimitive,
};

fn invalid(error: impl std::fmt::Display) -> ReplicatedTextRequirementsError {
    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
}

/// Metadata-only requirements for the LFM3 VL component and its processor.
///
/// This does not publish an executable selection. Construction consumes
/// the selected component, processor and prepared sources separately.
pub struct Lfm3VlRequirements {
    /// Complete graph, state profile, and exact physical parameter sources.
    pub execution: ReplicatedTextRequirements,
    /// Raw and prepared input requirements for text, images, and video.
    pub processor: ProcessorExecutionRequirements,
}

pub(super) fn processor_requirements(
    plan: &ArtifactArchitecturePlan,
) -> Result<ProcessorExecutionRequirements, ReplicatedTextRequirementsError> {
    use ProcessorPrimitive as P;

    let modality = |modality, primitives| {
        ModalityProcessorRequirements::new(modality, primitives, true, false, i32::MAX as u64)
            .map_err(invalid)
    };
    let image = if plan.has_processor() {
        vec![
            P::RgbResizeBicubic,
            P::RgbNormalize,
            P::TensorF32,
            P::TensorI32,
        ]
    } else {
        Vec::new()
    };
    let mut video = image.clone();
    if plan.has_processor() {
        video.push(P::VideoSampling);
    }
    ProcessorExecutionRequirements::new([
        modality(InputModality::Text, vec![P::TensorU32])?,
        modality(InputModality::Image, image)?,
        modality(InputModality::Video, video)?,
    ])
    .map_err(invalid)
}

/// Derives the shared cold requirements of a generative VL checkpoint.
pub fn lfm3_vl_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<Lfm3VlRequirements, ReplicatedTextRequirementsError> {
    let plan = inspection.architecture_plan();
    let declaration = plan
        .lfm3_vl_declaration()
        .ok_or(ReplicatedTextIneligibility::Unrelated)?;
    let identity = plan
        .lfm3_vl_execution_identity()
        .expect("admitted VL declaration");
    let component = declaration.component();
    let description = component.parameters();
    let embedding = declaration
        .language
        .statics
        .embedding_spec()
        .weight
        .id
        .as_str();
    let logical =
        declared_parameters(description, &BTreeMap::new(), embedding, false).map_err(invalid)?;
    let catalog = artifact::target_catalog(inspection)?;
    let gguf = plan.gguf_plan().is_some();
    let mut recipes = if gguf {
        use crate::lfm3::vision::gguf;
        let mut recipes =
            gguf::backbone_recipes(catalog.as_ref(), &declaration.language).map_err(invalid)?;
        recipes.extend(
            gguf::projector_recipes(catalog.as_ref(), &declaration.vision).map_err(invalid)?,
        );
        recipes
    } else {
        BTreeMap::new()
    };
    let physical_shapes = description
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .map(|member| (member.target(), member.global_shape()))
        .collect::<BTreeMap<_, _>>();
    let mut derived = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    let parameters = logical
        .into_iter()
        .map(|(target, parameter)| {
            let recipe = if gguf {
                recipes
                    .remove(&target)
                    .ok_or_else(|| invalid(format!("missing VL recipe for {target}")))?
            } else {
                DerivedWeightRecipe::source(
                    declaration
                        .physical_parameter_key(&target)
                        .map_err(invalid)?,
                    TensorSelection::Full,
                )
            };
            let sources = recipe
                .source_keys()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let physical = sources
                .iter()
                .map(|key| exact_physical_source(catalog.as_ref(), key))
                .collect::<Result<Vec<_>, _>>()?;
            let logical_shape = parameter.shape.expect("declared component shape");
            let output = recipe.infer(catalog.as_ref()).map_err(invalid)?;
            if physical_shapes[target.as_str()] != output.shape {
                return Err(invalid(format!(
                    "VL parameter {target:?} differs from the admitted shape"
                )));
            }
            let first = &sources[0];
            let metadata = catalog.source_metadata(first).map_err(invalid)?;
            let physical_encoding = physical[0].source_encoding();
            let decoded = gguf
                && physical_encoding.scalar_dtype().is_none()
                && output.dtype == RecipeDtype::F32;
            let direct = !decoded
                && matches!(
                    recipe,
                    DerivedWeightRecipe::Source {
                        selection: TensorSelection::Full,
                        ..
                    }
                );
            let (encoding, physical_shape) = if direct && parameter.companion.is_none() {
                (physical_encoding.clone(), metadata.physical_shape)
            } else {
                (
                    SourceTensorEncoding::RecipeOutput(recipe_stored_dtype(&output.dtype)?),
                    output.shape.clone(),
                )
            };
            let output_dtype = output.dtype.clone();
            let aliases = if direct {
                (first != &target)
                    .then(|| first.clone())
                    .into_iter()
                    .collect()
            } else {
                derived.insert(target.clone(), recipe);
                outputs.insert(target.clone(), output);
                Vec::new()
            };
            let requirement = parameter_requirement(
                target,
                sources,
                physical,
                aliases,
                Some(encoding),
                Some(physical_shape),
                logical_shape,
                parameter.native.expect("declared component format"),
                None,
                parameter.role,
                parameter.owner.expect("declared component owner"),
                ReplicatedTextParameterPresence::Required,
            )?;
            if let Some((role, primary)) = parameter.companion {
                requirement
                    .with_linear_companion(role, primary.as_str())
                    .map(|requirement| {
                        requirement.with_permitted_native_source_dtypes(vec![output_dtype])
                    })
                    .map_err(invalid)
            } else {
                Ok(requirement)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transports = (0..description.graph().groups().len())
        .map(|group| component.group_transport(group).cloned().map_err(invalid))
        .collect::<Result<_, _>>()?;
    let state = component.state_layout();
    let floating = if gguf {
        let dtype = catalog
            .source_metadata(embedding)
            .map_err(invalid)?
            .stored_dtype;
        match dtype {
            eredu_checkpoint::StoredDtype::F16 => eredu_core::checkpoint::TensorDtype::F16,
            eredu_checkpoint::StoredDtype::BF16 => eredu_core::checkpoint::TensorDtype::Bf16,
            eredu_checkpoint::StoredDtype::F32
            | eredu_checkpoint::StoredDtype::U8
            | eredu_checkpoint::StoredDtype::U32 => eredu_core::checkpoint::TensorDtype::F32,
            _ => {
                return Err(invalid(
                    "VL GGUF embedding has no floating-point representation",
                ))
            }
        }
    } else {
        crate::preparation::prepared_floating_state_dtype_source(inspection)
            .map_err(|error| ReplicatedTextRequirementsError::InvalidArtifact(error.to_string()))?
            .dtype()
            .clone()
    };
    let topology = crate::lfm3::execution_topology(&declaration.language).map_err(invalid)?;
    let execution = ReplicatedTextRequirements::new(
        identity,
        NeuralOperatorCapabilities::CONV2D
            .union(NeuralOperatorCapabilities::FROM_I32_SLICE)
            .union(NeuralOperatorCapabilities::MULTI_AXIS_ROTARY_EMBEDDINGS)
            .union(NeuralOperatorCapabilities::ROPE_WITH_FREQUENCIES)
            .union(NeuralOperatorCapabilities::SIGMOID)
            .union(NeuralOperatorCapabilities::CAST_FLOAT),
        description.graph().clone(),
        description.unit_layout().clone(),
        transports,
        state.clone(),
        ReplicatedTextStateAccess::from_layout(state).map_err(invalid)?,
        parameters,
    )
    .and_then(|requirements| {
        requirements
            .with_chunked_prefill(true)
            .with_prefill_input_retention(
                eredu_runtime::memory_forecast::PrefillInputRetention::PreparedProducts,
            )
            .with_floating_state_source(floating)
            .with_execution_topology(Some(topology))
            .with_derived_recipes(derived, outputs)?
            .with_nonparameter_sources(
                plan.gguf_plan()
                    .into_iter()
                    .flat_map(|plan| &plan.checkpoint().common_tensors)
                    .filter(|tensor| {
                        tensor.role == eredu_checkpoint::schema::TensorRole::ArtifactMetadata
                    })
                    .map(|tensor| crate::lfm3::translate_gguf_weight_name(&tensor.key))
                    .collect(),
            )?
            .with_component_definition(component.as_ref().clone())
    })
    .map_err(invalid)?;
    Ok(Lfm3VlRequirements {
        execution,
        processor: processor_requirements(plan)?,
    })
}
