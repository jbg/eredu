//! Retained cold tensor/pipeline/expert admission over exact target header authority.
use super::*;
use crate::partitioned_execution::{
    routed_partitioned_admission_from_requirements, select_routed_partitioned_admission,
    PartitionedSelectionRequest, RoutedPartitionedAdmission,
};
use crate::qwen4_exp::target::TargetTensorPartition;
use crate::routed_text::{select_routed_text_realization, RoutedTextSelectionRequest};
use crate::selected_execution::SelectedRoutedPartitionedExecution;
use eredu_core::{checkpoint::TensorDtype, ParallelRankTopology};
use eredu_runtime::{
    ArchitectureBoundary, BackendMechanismCapabilities, CommunicationCapabilities,
    CommunicationOperation, CommunicationOperationRequirement, CommunicationTensorLimits,
    RowLookupDescriptor, StateRealizationRequirements,
};

pub(crate) fn load_partition_request(
    request: &eredu_runtime::NormalizedLoadRequest,
) -> Result<PartitionedSelectionRequest, TargetLoadError> {
    let parallel = request
        .parallel_execution()
        .ok_or(TargetLoadError::PartitionPreparationRequired)?;
    let (batch, tokens) = parallel.invocation_limits();
    Ok(PartitionedSelectionRequest::new(
        parallel.rank().topology(),
        parallel.rank().global_rank(),
        batch,
        tokens,
        parallel.wire().activation_dtype(),
    )
    .map_err(PreparationError::Contract)?
    .with_completion_policy(parallel.completion()))
}

/// Source-free tensor/pipeline/expert authority. This plan does not publish an executable.
#[derive(Clone)]
pub struct TargetPartitionExecutionPlan {
    header: TargetPreparationPlan,
    tensor: TargetTensorPartition,
    requirements: RoutedPartitionedAdmission,
    local_state: StateRealizationRequirements,
    rows: Vec<RowLookupDescriptor>,
    prediction: Option<(
        crate::qwen4_exp::mtp::PredictionTensorPartition,
        StateRealizationRequirements,
    )>,
}

/// Exact mechanisms selected before readable sources or native groups are opened.
#[derive(Clone)]
pub struct SelectedTargetPartitionExecution {
    plan: TargetPartitionExecutionPlan,
    selected: SelectedRoutedPartitionedExecution,
    prediction: Option<SelectedPartitionPrediction>,
}

/// Selected rank authority bound to the original source and exact canonical recipes.
#[derive(Clone)]
pub struct PreparedTargetPartition {
    source: TargetExecutionPlan,
    tensor: PreparedTensorTarget,
    selected: SelectedRoutedPartitionedExecution,
    prediction: Option<SelectedPartitionPrediction>,
}

impl TargetPreparationPlan {
    /// Retains the same canonical target through TP/PP/EP selection.
    pub fn partition(
        self,
        request: PartitionedSelectionRequest,
    ) -> Result<TargetPartitionExecutionPlan, PreparationError> {
        TargetPartitionExecutionPlan::new(self, request)
    }
}

impl TargetPartitionExecutionPlan {
    pub(super) fn new(
        header: TargetPreparationPlan,
        request: PartitionedSelectionRequest,
    ) -> Result<Self, PreparationError> {
        let invalid = |e: String| PreparationError::Contract(e);
        let topology = request.topology();
        if topology.data() != 1 || topology.is_replicated() {
            return Err(invalid(
                "target partition construction requires TP, PP and/or EP with DP1".into(),
            ));
        }
        let rank = ParallelRankTopology::new(topology, request.global_rank())
            .map_err(|e| invalid(e.to_string()))?;
        let spec = header.target_spec()?;
        if request.maximum_batch_size() != spec.limits.qsa.batch
            || request.maximum_sequence_length() != spec.limits.qsa.tokens
        {
            return Err(invalid(
                "partition invocation differs from retained target limits".into(),
            ));
        }
        if topology.pipeline() > spec.units.len() {
            return Err(invalid(
                "target pipeline ranks exceed independently addressable units".into(),
            ));
        }
        let tensor =
            spec.tensor_partition(rank.tensor_parallel_rank(), rank.tensor_parallel_size())?;
        tensor.local_spec().expert_realization(rank)?;
        let prediction = match (
            header.prediction_spec(),
            header.prediction_state_requirements(),
        ) {
            (Some(spec), Some(state)) => {
                let tensor = spec
                    .tensor_partition(rank.tensor_parallel_rank(), rank.tensor_parallel_size())?;
                let state = StateRealizationRequirements::new(
                    tensor.local_spec().state_layout()?,
                    state.access(),
                    state.floating_source().cloned(),
                    state.append_streams().to_vec(),
                )
                .map_err(|error| invalid(error.to_string()))?;
                Some((tensor, state))
            }
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "prediction weights and state declarations differ".into(),
                ))
            }
        };
        let layout = tensor.local_spec().state_layout()?;
        let execution = header
            .requirements()
            .clone()
            .with_state_layout(layout.clone())
            .map_err(|e| invalid(e.to_string()))?;
        let reduction = row_reduction(execution.row_lookups())?;
        let requirements = routed_partitioned_admission_from_requirements(
            execution,
            request,
            spec.boundary_schema()?
                .wire_schema()
                .map_err(|e| invalid(e.to_string()))?,
            &layout,
            prediction_capture_limits(header.prediction_spec(), request)?,
            reduction.as_ref(),
        )
        .map_err(invalid)?;
        let state = requirements
            .state()
            .ok_or_else(|| invalid("target partition has no local state".into()))?;
        let range = state.global_layers();
        let state_requirements = requirements.execution().text().state_requirements();
        let streams: Vec<_> = state_requirements
            .append_streams()
            .iter()
            .filter(|stream| range.contains(&stream.layer))
            .cloned()
            .map(|mut stream| {
                stream.layer -= range.start;
                stream
            })
            .collect();
        let local_state = StateRealizationRequirements::new(
            state.layout().clone(),
            if streams.is_empty() {
                eredu_runtime::ReplicatedTextStateAccess::Fixed
            } else {
                state_requirements.access()
            },
            state_requirements.floating_source().cloned(),
            streams,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let rows = requirements
            .execution()
            .row_lookups()
            .into_iter()
            .flat_map(|rows| rows.descriptors().entries().values())
            .filter(|row| range.contains(&row.spec().unit))
            .cloned()
            .collect();
        Ok(Self {
            header,
            tensor,
            requirements,
            local_state,
            rows,
            prediction,
        })
    }
    pub(crate) fn selected_target_spec(
        &self,
        selected: &crate::SelectedRoutedTextRealization,
    ) -> Result<TargetSpec, PreparationError> {
        self.header.selected_target_spec(selected)
    }
    /// Rank-local ownership, source tasks, state and checked communication requirements.
    pub fn requirements(&self) -> &RoutedPartitionedAdmission {
        &self.requirements
    }
    /// Original-source TP geometry; selected-format geometry is bound separately.
    pub fn tensor_partition(&self) -> &TargetTensorPartition {
        &self.tensor
    }
    /// Local state ordinals and append streams for this pipeline owner.
    pub fn state_requirements(&self) -> &StateRealizationRequirements {
        &self.local_state
    }
    /// Tables invoked by this stage; unit identities remain architecture-global.
    pub fn row_requirements(&self) -> &[RowLookupDescriptor] {
        &self.rows
    }
    /// Only tensor-group rank zero acquires table payloads for this stage.
    pub const fn table_owner(&self) -> usize {
        0
    }
    /// Exact normalized weight/state policy, if this plan was prepared from a load request.
    pub fn load_selection_request(&self) -> Option<&RoutedTextSelectionRequest> {
        self.header.load_selection_request()
    }
    /// Selects weight, state and communication mechanisms without source or native access.
    pub fn select(
        self,
        request: &RoutedTextSelectionRequest,
        mechanisms: &BackendMechanismCapabilities,
        prediction_mechanisms: Option<&eredu_runtime::StateMechanismCapabilities>,
        communication: &CommunicationCapabilities,
    ) -> Result<SelectedTargetPartitionExecution, TargetSelectionError> {
        if self
            .header
            .load_selection_request()
            .is_some_and(|retained| retained != request)
        {
            return Err(TargetSelectionError::LoadRequestMismatch);
        }
        let base =
            select_routed_text_realization(self.requirements.execution(), request, mechanisms)?;
        let prediction = self.select_prediction(&base, request, prediction_mechanisms)?;
        super::execution::selected_stream_allowances(&base, prediction.as_ref().map(|p| &p.state))?;
        let selected =
            select_routed_partitioned_admission(self.requirements.clone(), base, communication)
                .map_err(|e| TargetSelectionError::Contract(e.to_string()))?;
        Ok(SelectedTargetPartitionExecution {
            plan: self,
            selected,
            prediction,
        })
    }
}

impl SelectedTargetPartitionExecution {
    /// The exact selected partition proof consumed by the shared construction driver.
    pub fn selected(&self) -> &SelectedRoutedPartitionedExecution {
        &self.selected
    }
    /// Common artifact authority retained through partition selection.
    pub fn target_artifact(&self) -> &TargetArtifactDeclaration {
        self.plan.header.artifact()
    }

    /// Conservative recipe workspace over exact retained header recipes.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        crate::SelectedExecution::partitioned_routed(self.selected.clone())
            .parameter_materialization_workspace(&self.plan.header, None, mechanisms)
    }
    /// Exact stage state, rows and source partition without native realization.
    pub fn plan(&self) -> &TargetPartitionExecutionPlan {
        &self.plan
    }
    /// Checks source provenance once and retains the cold-selected source partition.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        prediction_source: Option<SharedCheckpointSource>,
    ) -> Result<PreparedTargetPartition, PreparationError> {
        let source = self.plan.header.bind(source, prediction_source)?;
        let tensor = source.target.bind_tensor_partition(self.plan.tensor)?;
        Ok(PreparedTargetPartition {
            source,
            tensor,
            selected: self.selected,
            prediction: self.prediction,
        })
    }
}

impl PreparedTargetPartition {
    /// Original canonical sources; layout must be applied only once during materialization.
    pub fn prepared_target(&self) -> &PreparedTarget {
        &self.source.target
    }
    /// Independently addressable source-format tensor-rank parameters.
    pub fn tensor_target(&self) -> &PreparedTensorTarget {
        &self.tensor
    }
    /// Exact selected topology, state and communication proof.
    pub fn selected(&self) -> &SelectedRoutedPartitionedExecution {
        &self.selected
    }
    /// Original row source authority; provider binding applies selected stage/table ownership.
    pub fn row_sources(&self) -> &eredu_runtime::PreparedRowLookups {
        &self.source.row_sources
    }
    /// Family capability context retained by source preparation.
    pub fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.source.capability
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        TargetExecutionPlan,
        PreparedTensorTarget,
        SelectedRoutedPartitionedExecution,
    ) {
        (self.source, self.tensor, self.selected)
    }
}

pub(super) fn row_reduction(
    rows: Option<&eredu_runtime::SelectedRowLookupPlans>,
) -> Result<Option<CommunicationOperationRequirement>, PreparationError> {
    let Some(rows) = rows.filter(|rows| !rows.descriptors().entries().is_empty()) else {
        return Ok(None);
    };
    let mut elements = 1usize;
    let mut dtypes = vec![TensorDtype::I32];
    for row in rows.descriptors().entries().values() {
        let width = usize::try_from(row.spec().dimensions)
            .map_err(|e| PreparationError::Contract(e.to_string()))?;
        elements =
            elements.max(row.limits().requests.checked_mul(width).ok_or_else(|| {
                PreparationError::Contract("row collective extent overflow".into())
            })?);
        let dtype = match row.spec().output_type {
            eredu_nn::TensorElementType::F32 => TensorDtype::F32,
            eredu_nn::TensorElementType::F16 => TensorDtype::F16,
            eredu_nn::TensorElementType::Bf16 => TensorDtype::Bf16,
            _ => {
                return Err(PreparationError::Contract(
                    "row collective requires floating outputs".into(),
                ))
            }
        };
        if !dtypes.contains(&dtype) {
            dtypes.push(dtype);
        }
    }
    let limits = CommunicationTensorLimits::new(1, 2, elements, None)
        .map_err(|e| PreparationError::Contract(e.to_string()))?;
    Ok(Some(
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            dtypes,
            limits,
            true,
        )
        .map_err(|e| PreparationError::Contract(e.to_string()))?,
    ))
}

/// Source and executable prediction coordinates retained independently from PP/EP target ownership.
#[derive(Clone)]
pub(super) struct SelectedPartitionPrediction {
    source: crate::qwen4_exp::mtp::PredictionTensorPartition,
    global: crate::qwen4_exp::mtp::PredictionSpec,
    tensor: crate::qwen4_exp::mtp::PredictionTensorPartition,
    state: eredu_runtime::SelectedStateRealization,
}
impl SelectedPartitionPrediction {
    pub(super) fn global_spec(&self) -> &crate::qwen4_exp::mtp::PredictionSpec {
        &self.global
    }

    pub(super) fn spec(&self) -> &crate::qwen4_exp::mtp::PredictionSpec {
        self.tensor.local_spec()
    }
    pub(super) fn state(&self) -> &eredu_runtime::SelectedStateRealization {
        &self.state
    }
    pub(super) fn partition(&self) -> &crate::qwen4_exp::mtp::PredictionTensorPartition {
        &self.tensor
    }
    pub(super) fn weights<B>(
        &self,
        source: &TargetExecutionPlan,
        selected: &crate::routed_text::SelectedRoutedTextRealization,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::prediction_extension::PreparedQwen4PredictionWeights<B>, PreparationError>
    where
        B: eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend
            + eredu_nn::HyperNeuralBackend,
    {
        let (prediction, _) = source
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        crate::prediction_extension::PreparedQwen4PredictionWeights::new(
            prediction.artifact.clone(),
            &prediction.spec,
            self.global.clone(),
            Some((&self.source, &self.tensor)),
            self.state.clone(),
            selected.text().auxiliary_materialization_tasks(),
            selected.text().residency(),
            context,
        )
        .map_err(|error| PreparationError::Contract(error.to_string()))
    }
}
impl TargetPartitionExecutionPlan {
    /// Complete independent prediction state replicated across PP/EP and localized by TP.
    pub fn prediction_state_requirements(&self) -> Option<&StateRealizationRequirements> {
        self.prediction.as_ref().map(|(_, state)| state)
    }
    /// Separate SafeTensors prediction headers retained for a GGUF target.
    pub fn prediction_header(&self) -> Option<&SafetensorsPredictionPlan> {
        self.header.prediction_header()
    }
    pub(super) fn select_prediction(
        &self,
        selected: &crate::routed_text::SelectedRoutedTextRealization,
        request: &RoutedTextSelectionRequest,
        mechanisms: Option<&eredu_runtime::StateMechanismCapabilities>,
    ) -> Result<Option<SelectedPartitionPrediction>, TargetSelectionError> {
        match (&self.prediction, mechanisms) {
            (Some((source, requirements)), Some(mechanisms)) => {
                let global = self
                    .header
                    .selected_prediction_spec(selected)
                    .map_err(|error| TargetSelectionError::Contract(error.to_string()))?;
                let tensor = global
                    .tensor_partition(source.rank(), source.ranks())
                    .map_err(|error| TargetSelectionError::Contract(error.to_string()))?;
                let state = eredu_runtime::select_state_realization(
                    requirements,
                    request.text(),
                    mechanisms,
                )?;
                if tensor
                    .local_spec()
                    .state_layout()
                    .map_err(|error| TargetSelectionError::Contract(error.to_string()))?
                    != *state.layout()
                {
                    return Err(TargetSelectionError::Contract(
                        "selected prediction tensor state changed geometry".into(),
                    ));
                }
                Ok(Some(SelectedPartitionPrediction {
                    source: source.clone(),
                    global,
                    tensor,
                    state,
                }))
            }
            (None, None) => Ok(None),
            _ => Err(TargetSelectionError::PredictionStatePresence),
        }
    }
}
impl SelectedTargetPartitionExecution {
    /// Selected tensor-local predictor geometry.
    pub fn prediction_spec(
        &self,
    ) -> Result<crate::qwen4_exp::mtp::PredictionSpec, PreparationError> {
        self.prediction
            .as_ref()
            .map(|p| p.spec().clone())
            .ok_or(PreparationError::MissingPrediction)
    }
    /// Independent predictor state selected before source acquisition.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction
            .as_ref()
            .map(SelectedPartitionPrediction::state)
    }
    pub(crate) fn prediction_descriptor(
        &self,
    ) -> Result<eredu_core::ArchitectureDescriptor, PreparationError> {
        let target = self.plan.header.target_spec()?;
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
impl PreparedTargetPartition {
    /// Exact executable prediction geometry retained through source binding.
    pub fn prediction_spec(
        &self,
    ) -> Result<crate::qwen4_exp::mtp::PredictionSpec, PreparationError> {
        self.prediction
            .as_ref()
            .map(|p| p.spec().clone())
            .ok_or(PreparationError::MissingPrediction)
    }
    /// Exact selected tensor coordinates, with all prediction experts retained per EP/PP replica.
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
    /// Independent predictor state selected before source acquisition.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction
            .as_ref()
            .map(SelectedPartitionPrediction::state)
    }
    /// Constructs exact source/executable prediction modules without reopening artifacts.
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
        self.prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?
            .weights(&self.source, self.selected.base(), context)
    }
}

/// Publication transports the retained pre-collapse residual, including its stream axis.
pub(super) fn prediction_capture_limits(
    spec: Option<&crate::qwen4_exp::mtp::PredictionSpec>,
    request: PartitionedSelectionRequest,
) -> Result<Option<(usize, usize)>, PreparationError> {
    spec.map(|spec| {
        let geometry = spec.fusion.geometry;
        let elements = [
            request.maximum_batch_size(),
            request.maximum_sequence_length(),
            geometry.streams(),
            geometry.hidden_size(),
        ]
        .into_iter()
        .try_fold(1usize, |count, axis| {
            usize::try_from(axis)
                .ok()
                .and_then(|axis| count.checked_mul(axis))
        })
        .ok_or_else(|| {
            PreparationError::Contract("prediction capture extent overflows usize".into())
        })?;
        Ok((4, elements))
    })
    .transpose()
}
