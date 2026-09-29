//! Source-free joint media rank admission and exact retained role binding.
use super::conditional_header::JointCatalog;
use super::partition_selection::SelectedPartitionPrediction;
use super::*;
use crate::composite_partitioned::{
    CompositePartitionBoundaries, CompositePartitionBoundary, CompositeSequenceExtent,
};
use crate::partitioned_execution::{
    composite_partitioned_admission_from_requirements, select_composite_partitioned_admission,
    CompositePartitionedAdmission, PartitionedSelectionRequest,
};
use crate::qwen4_exp::media::{processor::SelectedMediaProcessor, MediaIngress};
use crate::replicated_text::{CompositeTextRequirements, SelectedCompositeTextRealization};
use crate::routed_text::RoutedTextSelectionRequest;
use crate::selected_execution::SelectedCompositePartitionedExecution;
use eredu_core::artifact::ArtifactInspection;
use eredu_runtime::{
    ArchitectureBoundary, BackendMechanismCapabilities, BoundaryTensorDimension as Dim,
    BoundaryTensorDtype as Dtype, BoundaryTensorSpec, BoundaryWireSchema,
    CommunicationCapabilities, MediaPrimitiveCapabilities, ProcessorSelectionRequest,
    RowLookupDescriptor, StateRealizationRequirements,
};

fn invalid(error: impl std::fmt::Display) -> PreparationError {
    PreparationError::Contract(error.to_string())
}

/// Joint target/vision admission retaining original headers and exact rank state.
#[derive(Clone)]
pub struct ConditionalPartitionExecutionPlan {
    header: ConditionalHeaderExecutionPlan,
    target: TargetPartitionExecutionPlan,
    requirements: CompositePartitionedAdmission,
}

/// Selected mechanisms for one conditional TP/PP/EP owner, before source access.
#[derive(Clone)]
pub struct SelectedConditionalPartitionExecution {
    plan: ConditionalPartitionExecutionPlan,
    selected: SelectedCompositePartitionedExecution,
    processor: SelectedMediaProcessor,
    prediction: Option<SelectedPartitionPrediction>,
}

/// Original target/projector authority bound to the selected conditional rank.
#[derive(Clone)]
pub struct PreparedConditionalPartition {
    source: TargetExecutionPlan,
    tensor: PreparedTensorTarget,
    ingress: MediaIngress,
    selected: SelectedCompositePartitionedExecution,
    processor: SelectedMediaProcessor,
    prediction: Option<SelectedPartitionPrediction>,
}

impl ConditionalHeaderExecutionPlan {
    /// Authors rank-local state, semantic media edges and communication from headers.
    pub fn partition(
        self,
        inspection: &ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
        request: PartitionedSelectionRequest,
    ) -> Result<ConditionalPartitionExecutionPlan, PreparationError> {
        let target = TargetPartitionExecutionPlan::new(self.target_header().clone(), request)?;
        let spec = self.target_header().target_spec()?;
        let vision = self.vision_plan().config();
        // One request encodes media before target chunking. Its vision edges use
        // the retained complete-request ceiling rather than a target chunk size.
        let projected =
            i32::try_from(self.media_policy().maximum_request_tokens()).map_err(invalid)?;
        let patches = vision_partition_extent(vision, request, projected)?;
        let continuation = BoundaryWireSchema::new(
            "qwen4_exp.vision_continuation",
            BoundaryTensorSpec::new(
                "hidden",
                [Dim::Sequence, Dim::Fixed(vision.hidden_size)],
                Dtype::Activation,
            ),
            [],
        )
        .map_err(invalid)?;
        let output = BoundaryWireSchema::new(
            "qwen4_exp.vision_to_target",
            BoundaryTensorSpec::new(
                "hidden",
                [
                    Dim::Batch,
                    Dim::Sequence,
                    Dim::Fixed(vision.out_hidden_size),
                ],
                Dtype::Activation,
            ),
            [],
        )
        .map_err(invalid)?;
        let boundaries = CompositePartitionBoundaries::default()
            .with_boundary(
                1,
                1,
                CompositePartitionBoundary::fixed(
                    continuation,
                    CompositeSequenceExtent::Fixed(patches),
                ),
            )
            .map_err(invalid)?
            .with_boundary(
                1,
                0,
                CompositePartitionBoundary::fixed(
                    output,
                    CompositeSequenceExtent::Fixed(projected),
                ),
            )
            .map_err(invalid)?;
        let state = target.tensor_partition().local_spec().state_layout()?;
        let routed = self
            .requirements()
            .clone()
            .with_state_layout(state.clone())
            .map_err(invalid)?;
        let reduction = vision_reduction(
            super::partition_selection::row_reduction(routed.row_lookups())?,
            request,
            patches,
            projected,
            vision,
        )?;
        let requirements = CompositeTextRequirements::from_routed(
            inspection,
            routed.text().architecture_identity().to_owned(),
            routed,
            self.processor_requirements().map_err(invalid)?,
            eredu_core::InputModalities {
                text: true,
                image: true,
                video: true,
                audio: false,
            },
            boundaries,
        )
        .map_err(invalid)?;
        let requirements = composite_partitioned_admission_from_requirements(
            requirements,
            request,
            spec.boundary_schema()?.wire_schema().map_err(invalid)?,
            &state,
            super::partition_selection::prediction_capture_limits(self.prediction_spec(), request)?,
            reduction.as_ref(),
        )
        .map_err(invalid)?;
        if requirements.state() != target.requirements().state() {
            return Err(invalid(
                "conditional partition changed retained target state ownership",
            ));
        }
        Ok(ConditionalPartitionExecutionPlan {
            header: self,
            target,
            requirements,
        })
    }
}

fn vision_partition_extent(
    vision: &crate::qwen::vision::VisionConfig,
    request: PartitionedSelectionRequest,
    projected: i32,
) -> Result<i32, PreparationError> {
    if request.topology().pipeline() > vision.layer_count() {
        return Err(invalid(
            "conditional pipeline ranks exceed vision execution units",
        ));
    }
    let rank = eredu_core::ParallelRankTopology::new(request.topology(), request.global_rank())
        .map_err(invalid)?;
    let merger_width = vision
        .hidden_size
        .checked_mul(vision.spatial_merge_size)
        .and_then(|value| value.checked_mul(vision.spatial_merge_size))
        .ok_or_else(|| invalid("conditional vision merger width overflows i32"))?;
    for units in [vision.num_heads, vision.intermediate_size, merger_width] {
        eredu_core::balanced_contiguous_range(
            usize::try_from(units).map_err(invalid)?,
            request.topology().tensor(),
            rank.tensor_parallel_rank(),
            false,
        )
        .map_err(invalid)?;
    }
    projected
        .checked_mul(vision.spatial_merge_size)
        .and_then(|value| value.checked_mul(vision.spatial_merge_size))
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            invalid("conditional vision continuation extent is invalid or overflows i32")
        })
}

/// The encoder runs once for the complete request, before decoder chunking.
/// Its reductions must fit the same selected tensor collective as lexical rows.
fn vision_reduction(
    rows: Option<eredu_runtime::CommunicationOperationRequirement>,
    request: PartitionedSelectionRequest,
    patches: i32,
    projected: i32,
    vision: &crate::qwen::vision::VisionConfig,
) -> Result<Option<eredu_runtime::CommunicationOperationRequirement>, PreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{
        CommunicationOperation, CommunicationOperationRequirement, CommunicationTensorLimits,
        PipelineActivationDtype,
    };
    if request.topology().tensor() == 1 {
        return Ok(rows);
    }
    let dtype = match request.activation_dtype() {
        PipelineActivationDtype::Float32 => TensorDtype::F32,
        PipelineActivationDtype::Float16 => TensorDtype::F16,
        PipelineActivationDtype::Bfloat16 => TensorDtype::Bf16,
        _ => return Err(invalid("unsupported vision collective activation dtype")),
    };
    let mut dtypes = rows
        .as_ref()
        .map(|row| row.dtypes().to_vec())
        .unwrap_or_default();
    if !dtypes.contains(&dtype) {
        dtypes.push(dtype);
    }
    let old = rows.as_ref().and_then(|row| row.limits());
    let mut elements = old.map_or(1, |limits| {
        limits
            .max_tensor_elements()
            .max(limits.max_output_tensor_elements())
    });
    for (length, width) in [
        (patches, vision.hidden_size),
        (projected, vision.out_hidden_size),
    ] {
        let product = usize::try_from(length)
            .map_err(invalid)?
            .checked_mul(usize::try_from(width).map_err(invalid)?)
            .ok_or_else(|| invalid("vision collective extent overflows usize"))?;
        elements = elements.max(product);
    }
    let limits = CommunicationTensorLimits::new(
        old.map_or(1, |limits| limits.max_tensors().max(1)),
        old.map_or(2, |limits| limits.max_tensor_rank().max(2)),
        elements,
        None,
    )
    .map_err(invalid)?;
    Ok(Some(
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            dtypes,
            limits,
            true,
        )
        .map_err(invalid)?,
    ))
}

impl ConditionalPartitionExecutionPlan {
    /// Complete semantic ownership and communication for target group zero and vision one.
    pub fn requirements(&self) -> &CompositePartitionedAdmission {
        &self.requirements
    }
    /// Original source-free target and projector identities.
    pub fn header_plan(&self) -> &ConditionalHeaderExecutionPlan {
        &self.header
    }
    /// Source-format target TP selections and local state geometry.
    pub fn target_partition(&self) -> &TargetPartitionExecutionPlan {
        &self.target
    }
    /// Exact local state, including append streams, for this pipeline owner.
    pub fn state_requirements(&self) -> &StateRealizationRequirements {
        self.target.state_requirements()
    }
    /// Stage-owned lookup requirements with original global target unit addresses.
    pub fn row_requirements(&self) -> &[RowLookupDescriptor] {
        self.target.row_requirements()
    }
    /// Preserved ordinary load policy.
    pub fn load_selection_request(&self) -> Option<&RoutedTextSelectionRequest> {
        self.header.load_selection_request()
    }
    /// Selects weights, state, processor and communication without opening payload sources.
    pub fn select(
        self,
        request: &RoutedTextSelectionRequest,
        mechanisms: &BackendMechanismCapabilities,
        prediction_mechanisms: Option<&eredu_runtime::StateMechanismCapabilities>,
        communication: &CommunicationCapabilities,
        inputs: &ProcessorSelectionRequest,
        media: &MediaPrimitiveCapabilities,
    ) -> Result<SelectedConditionalPartitionExecution, TargetSelectionError> {
        if self
            .header
            .load_selection_request()
            .is_some_and(|retained| retained != request)
        {
            return Err(TargetSelectionError::LoadRequestMismatch);
        }
        if self
            .header
            .load_processor_request()
            .is_some_and(|retained| retained != inputs)
        {
            return Err(TargetSelectionError::ProcessorRequestMismatch);
        }
        let processor = self.header.media_policy().select_processor(
            self.header.vision_plan().processor(),
            inputs,
            media,
        )?;
        let execution = crate::routed_text::select_routed_text_realization(
            self.requirements
                .execution()
                .routed_execution()
                .expect("authored routed conditional"),
            request,
            mechanisms,
        )?;
        let prediction =
            self.target
                .select_prediction(&execution, request, prediction_mechanisms)?;
        super::execution::selected_stream_allowances(
            &execution,
            prediction.as_ref().map(SelectedPartitionPrediction::state),
        )?;
        let selected = select_composite_partitioned_admission(
            self.requirements.clone(),
            SelectedCompositeTextRealization::Routed {
                execution,
                processor: processor.realization().clone(),
            },
            communication,
        )
        .map_err(|error| TargetSelectionError::Contract(error.to_string()))?;
        Ok(SelectedConditionalPartitionExecution {
            plan: self,
            selected,
            processor,
            prediction,
        })
    }
}

impl SelectedConditionalPartitionExecution {
    pub(crate) fn selected_target_spec(&self) -> Result<TargetSpec, PreparationError> {
        self.plan.target.selected_target_spec(self.realization())
    }
    /// Joint routed realization used by the shared prediction selection.
    pub fn realization(&self) -> &crate::routed_text::SelectedRoutedTextRealization {
        match self.selected.base() {
            SelectedCompositeTextRealization::Routed { execution, .. } => execution,
            SelectedCompositeTextRealization::Direct(_) => {
                unreachable!("authored routed conditional")
            }
        }
    }
    /// Checked shared composite execution proof.
    pub fn selected(&self) -> &SelectedCompositePartitionedExecution {
        &self.selected
    }
    /// Retained joint headers for original source-role acquisition.
    pub fn header_plan(&self) -> &ConditionalHeaderExecutionPlan {
        &self.plan.header
    }
    /// Exact local requirements selected for this owner.
    pub fn plan(&self) -> &ConditionalPartitionExecutionPlan {
        &self.plan
    }
    /// Selected media processor and original-ID policy.
    pub fn processor(&self) -> &SelectedMediaProcessor {
        &self.processor
    }
    /// Recipe workspace over retained target and projector catalogs.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        crate::SelectedExecution::partitioned_composite(self.selected.clone())
            .parameter_materialization_workspace(
                &JointCatalog {
                    target: self.plan.header.target_header(),
                    vision: self.plan.header.vision_plan(),
                },
                None,
                mechanisms,
            )
    }
    /// Binds original roles once; only target integer controls are acquired eagerly.
    pub fn bind(
        self,
        target_source: SharedCheckpointSource,
        vision_source: SharedCheckpointSource,
        prediction_source: Option<SharedCheckpointSource>,
    ) -> Result<PreparedConditionalPartition, PreparationError> {
        let vision = self.plan.header.vision_plan().clone().bind(vision_source)?;
        let source = self
            .plan
            .header
            .target_header()
            .clone()
            .bind(target_source, prediction_source)?;
        let (source, ingress) = source.with_vision(vision)?.into_parts();
        let tensor = source
            .target
            .bind_tensor_partition(self.plan.target.tensor_partition().clone())?;
        Ok(PreparedConditionalPartition {
            source,
            tensor,
            ingress,
            selected: self.selected,
            processor: self.processor,
            prediction: self.prediction,
        })
    }
}

impl PreparedConditionalPartition {
    /// Canonical target recipes retaining the complete joint artifact source.
    pub fn prepared_target(&self) -> &PreparedTarget {
        &self.source.target
    }
    /// Source-format tensor selections, before selected transforms.
    pub fn tensor_target(&self) -> &PreparedTensorTarget {
        &self.tensor
    }
    /// Prepared vision and architecture-owned input admission.
    pub fn ingress(&self) -> &MediaIngress {
        &self.ingress
    }
    /// Exact selected composite rank proof.
    pub fn selected(&self) -> &SelectedCompositePartitionedExecution {
        &self.selected
    }
    /// Exact selected processor used by ordinary input preparation.
    pub fn processor(&self) -> &SelectedMediaProcessor {
        &self.processor
    }
    /// Original lazy table sources, with stage ownership applied by the visitor.
    pub fn row_sources(&self) -> &eredu_runtime::PreparedRowLookups {
        &self.source.row_sources
    }
    /// Retained joint architecture capability context.
    pub fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.source.capability
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{checkpoint::TensorDtype, ParallelTopology};
    use eredu_runtime::{
        CommunicationOperation, CommunicationOperationRequirement, CommunicationTensorLimits,
        PipelineActivationDtype,
    };

    fn vision() -> crate::qwen::vision::VisionConfig {
        serde_json::from_value::<crate::qwen::vision::VisionConfigSource>(serde_json::json!({
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "num_position_embeddings": 16, "in_channels": 3, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 32,
            "deepstack_visual_indexes": []
        }))
        .unwrap()
        .normalize_qwen3_vl()
        .unwrap()
    }

    fn request(tp: usize, pp: usize) -> PartitionedSelectionRequest {
        PartitionedSelectionRequest::new(
            ParallelTopology::new(tp, pp, 1, 1).unwrap(),
            0,
            1,
            4,
            PipelineActivationDtype::Float32,
        )
        .unwrap()
    }

    #[test]
    fn complete_media_request_reductions_exceed_decoder_chunk_without_losing_row_dtypes() {
        let request = request(2, 2);
        let vision = vision();
        let patches = vision_partition_extent(&vision, request, 256).unwrap();
        let rows = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::I32, TensorDtype::Bf16],
            CommunicationTensorLimits::new(1, 2, 128, None).unwrap(),
            true,
        )
        .unwrap();
        let reduction = vision_reduction(Some(rows), request, patches, 256, &vision)
            .unwrap()
            .unwrap();
        assert_eq!(reduction.limits().unwrap().max_tensor_elements(), 32768);
        assert_eq!(
            reduction.limits().unwrap().max_output_tensor_elements(),
            32768
        );
        assert!(reduction.limits().unwrap().max_tensor_elements() > 4 * 32 * 4);
        assert_eq!(
            reduction.dtypes(),
            &[TensorDtype::I32, TensorDtype::Bf16, TensorDtype::F32]
        );
        assert!(
            vision_reduction(None, self::request(1, 2), patches, 256, &vision)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn cold_vision_geometry_rejects_empty_owners_and_invalid_extents() {
        let mut vision = vision();
        assert!(vision_partition_extent(&vision, request(2, 2), 8).is_ok());
        assert!(vision_partition_extent(&vision, request(5, 2), 8).is_err());
        assert!(vision_partition_extent(&vision, request(2, 3), 8).is_err());
        assert!(vision_partition_extent(&vision, request(2, 2), 0).is_err());
        assert!(vision_partition_extent(&vision, request(2, 2), i32::MAX).is_err());
        vision.intermediate_size = 1;
        assert!(vision_partition_extent(&vision, request(2, 2), 8).is_err());
        vision.intermediate_size = 64;
        vision.hidden_size = i32::MAX;
        assert!(vision_partition_extent(&vision, request(2, 2), 8).is_err());
    }
}

impl ConditionalPartitionExecutionPlan {
    /// Independent TP-local predictor state, replicated across target PP/EP owners.
    pub fn prediction_state_requirements(&self) -> Option<&StateRealizationRequirements> {
        self.target.prediction_state_requirements()
    }
}
impl SelectedConditionalPartitionExecution {
    /// Selected tensor-local prediction geometry.
    pub fn prediction_spec(
        &self,
    ) -> Result<crate::qwen4_exp::mtp::PredictionSpec, PreparationError> {
        self.prediction
            .as_ref()
            .map(|prediction| prediction.spec().clone())
            .ok_or(PreparationError::MissingPrediction)
    }
    /// Independent prediction state selected with the conditional rank.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction
            .as_ref()
            .map(SelectedPartitionPrediction::state)
    }
    pub(crate) fn prediction_descriptor(
        &self,
    ) -> Result<eredu_core::ArchitectureDescriptor, PreparationError> {
        let target = self.plan.header.target_header().target_spec()?;
        let prediction = self
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?
            .global_spec()
            .clone();
        super::graph::Graph::new(&target, &prediction)
            .with_prediction_observations(&target, &prediction)
    }
}
impl PreparedConditionalPartition {
    /// Selected tensor-local prediction geometry.
    pub fn prediction_spec(
        &self,
    ) -> Result<crate::qwen4_exp::mtp::PredictionSpec, PreparationError> {
        self.prediction
            .as_ref()
            .map(|prediction| prediction.spec().clone())
            .ok_or(PreparationError::MissingPrediction)
    }
    /// Exact executable prediction tensor selections retained through binding.
    pub fn prediction_partition(
        &self,
    ) -> Option<&crate::qwen4_exp::mtp::PredictionTensorPartition> {
        self.prediction
            .as_ref()
            .map(SelectedPartitionPrediction::partition)
    }
    /// Original prepared prediction source and canonical recipes.
    pub fn prediction_source(&self) -> Option<&PreparedPrediction> {
        self.source
            .prediction
            .as_ref()
            .map(|(prediction, _)| prediction)
    }
    /// Independent prediction state selected before source acquisition.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction
            .as_ref()
            .map(SelectedPartitionPrediction::state)
    }
    /// Builds local prediction modules from retained target/prediction source authority.
    pub fn prediction_weights<B>(
        &self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::prediction_extension::PreparedQwen4PredictionWeights<B>, PreparationError>
    where
        B: eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend
            + eredu_nn::HyperNeuralBackend,
    {
        let SelectedCompositeTextRealization::Routed { execution, .. } = self.selected.base()
        else {
            return Err(invalid(
                "conditional prediction requires routed target execution",
            ));
        };
        self.prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?
            .weights(&self.source, execution, context)
    }
}
