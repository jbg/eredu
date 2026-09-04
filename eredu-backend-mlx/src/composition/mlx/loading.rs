//! MLX checkpoint materialization after backend-neutral planning.

use std::sync::Arc;

use eredu_checkpoint::WeightQuantization;

use eredu_architectures::processor_plan::ArtifactArchitecturePlan;
use eredu_architectures::ModelKind;
use eredu_core::{ModelArtifact, ModelPreparationPlan};
use eredu_gguf::MetadataValue as GgufMetadataValue;
use eredu_runtime::{
    select_replicated_text_realization, CacheResidencyPolicy, ReplicatedTextSelectionRequest,
};
use safemlx::Stream;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::composition::mlx::ModelProcessor;

use crate::{
    backend::error::Error,
    backend::MlxModel,
    composition::{
        mlx::{structural, Executable},
        MlxNeuralBackend,
    },
    MlxLoadRequest,
};

use super::realization::{
    AddressableParameterBankBinding, FamilyBinding, FixedGgufBinding, GgufBinding,
    QuantizedGgufBinding,
};

struct PredictionTargetCheckpointSource {
    source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    extension_sources: std::collections::BTreeSet<String>,
}

impl PredictionTargetCheckpointSource {
    fn new(
        source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        extension_sources: std::collections::BTreeSet<String>,
    ) -> Self {
        Self {
            source,
            extension_sources,
        }
    }

    fn authorize(&self, key: &str) -> Result<(), eredu_checkpoint::store::StoreError> {
        if self.extension_sources.contains(key) {
            Err(eredu_checkpoint::store::StoreError::UnauthorizedTensor {
                contract: "prediction-target".into(),
                key: key.into(),
            })
        } else {
            Ok(())
        }
    }
}

impl eredu_checkpoint::store::CheckpointSource for PredictionTargetCheckpointSource {
    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .filter(|key| !self.extension_sources.contains(key))
            .collect()
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.authorize(key)?;
        self.source.source_metadata(key)
    }

    fn acquire_lease(
        &self,
        request: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError> {
        self.authorize(&request.key)?;
        self.source.acquire_lease(request)
    }

    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, eredu_checkpoint::store::StoreError>
    {
        self.source.source_diagnostics()
    }

    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        self.authorize(key)?;
        self.source.source_provenance(key)
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.source
            .materialized_source_keys()
            .into_iter()
            .filter(|key| !self.extension_sources.contains(key))
            .collect()
    }

    fn materialized_source_shards(&self) -> Vec<std::path::PathBuf> {
        self.source.materialized_source_shards()
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.source
            .unclaimed_checkpoint_keys()
            .into_iter()
            .filter(|key| !self.extension_sources.contains(key))
            .collect()
    }

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        !self.extension_sources.contains(key) && self.source.is_authoritative_materialized_key(key)
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.source.is_checkpoint_contract_resolved()
    }
}

/// Opaque MLX model configuration selected before payloads are opened.
#[derive(Debug, Clone)]
pub struct MlxModelConfig {
    pub(crate) plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    pub(crate) selected: MlxSelectedPreparation,
}

impl MlxModelConfig {
    pub(crate) const fn new(
        plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
        selected: MlxSelectedPreparation,
    ) -> Self {
        Self { plan, selected }
    }
}

/// Opaque, authoritative MLX construction policy selected before payloads are opened.
#[derive(Debug, Clone)]
pub struct MlxSelectedPreparation {
    execution: MlxSelectedExecution,
    session: eredu_core::SessionCapabilities,
    rank_context: Option<crate::backend::MlxRankContext>,
    prediction_extension: Option<eredu_architectures::configuration::PredictionExtensionPlan>,
}

type OrdinaryMlxSelectedExecution =
    eredu_architectures::replicated_text::SelectedReplicatedTextExecution<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::replicated_text::SelectedCompositeTextRealization,
        SelectedMlxConstruction,
    >;

#[derive(Debug, Clone)]
enum MlxSelectedExecution {
    Ordinary(OrdinaryMlxSelectedExecution),
    PartitionedDense {
        selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
            eredu_runtime::SelectedReplicatedTextRealization,
            eredu_runtime::ReplicatedTextRequirements,
        >,
    },
    PartitionedRouted {
        selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
            eredu_architectures::SelectedRoutedTextRealization,
            eredu_architectures::RoutedTextRequirements,
        >,
    },
    PartitionedComposite {
        selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
            eredu_architectures::replicated_text::SelectedCompositeTextRealization,
            eredu_architectures::replicated_text::CompositeTextRequirements,
        >,
    },
}

impl MlxSelectedPreparation {
    const fn new(
        execution: MlxSelectedExecution,
        session: eredu_core::SessionCapabilities,
        rank_context: Option<crate::backend::MlxRankContext>,
        prediction_extension: Option<eredu_architectures::configuration::PredictionExtensionPlan>,
    ) -> Self {
        Self {
            execution,
            session,
            rank_context,
            prediction_extension,
        }
    }

    pub(crate) const fn session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.session
    }

    #[cfg(test)]
    pub(crate) const fn prediction_extension_kind(
        &self,
    ) -> Option<eredu_architectures::configuration::PredictionExtensionKind> {
        match &self.prediction_extension {
            Some(extension) => Some(extension.kind()),
            None => None,
        }
    }

    pub(crate) const fn rank_context(&self) -> Option<crate::backend::MlxRankContext> {
        self.rank_context
    }

    /// Returns the architecture-owned opaque manifest selected for an included path.
    #[cfg(test)]
    pub(crate) const fn communication_manifest(
        &self,
    ) -> Option<&eredu_runtime::CommunicationManifest> {
        match &self.execution {
            MlxSelectedExecution::PartitionedDense { selected } => {
                Some(selected.requirements().communication())
            }
            MlxSelectedExecution::PartitionedRouted { selected } => {
                Some(selected.requirements().communication())
            }
            MlxSelectedExecution::PartitionedComposite { selected } => {
                Some(selected.requirements().communication())
            }
            MlxSelectedExecution::Ordinary(_) => None,
        }
    }

    /// Returns the architecture-admitted activation dtype without entering a
    /// family-specific materializer.
    #[cfg(test)]
    pub(crate) const fn partitioned_activation_dtype(
        &self,
    ) -> Option<eredu_runtime::PipelineActivationDtype> {
        match &self.execution {
            MlxSelectedExecution::PartitionedDense { selected } => {
                Some(selected.requirements().activation_dtype())
            }
            MlxSelectedExecution::PartitionedRouted { selected } => {
                Some(selected.requirements().activation_dtype())
            }
            MlxSelectedExecution::PartitionedComposite { selected } => {
                Some(selected.requirements().activation_dtype())
            }
            MlxSelectedExecution::Ordinary(_) => None,
        }
    }

    /// Returns the architecture-selected manifest consumed by partitioned materialization.
    pub(crate) const fn realized_communication_manifest(
        &self,
    ) -> Option<&eredu_runtime::CommunicationManifest> {
        match &self.execution {
            MlxSelectedExecution::PartitionedDense { selected } => {
                Some(selected.requirements().communication())
            }
            MlxSelectedExecution::PartitionedRouted { selected } => {
                Some(selected.requirements().communication())
            }
            MlxSelectedExecution::Ordinary(_) => None,
            MlxSelectedExecution::PartitionedComposite { selected } => {
                Some(selected.requirements().communication())
            }
        }
    }
}

/// Fully resolved construction inputs for non-replicated-text MLX composition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct SelectedMlxConstruction {
    quantization: Option<WeightQuantization>,
    pub(crate) parallel: Option<eredu_core::ParallelRankTopology>,
    pub(crate) weight_residency: eredu_runtime::WeightResidency,
    pub(crate) state_residency: CacheResidencyPolicy,
    session: eredu_core::SessionCapabilities,
    processor: Option<eredu_runtime::SelectedProcessorExecution>,
}

impl SelectedMlxConstruction {
    pub(crate) fn from_request(
        request: MlxLoadRequest,
        session: eredu_core::SessionCapabilities,
    ) -> Result<Self, Error> {
        Ok(Self {
            quantization: request.weight_quantization()?,
            parallel: request.parallel_topology(),
            weight_residency: request.weight_residency(),
            state_residency: request.state_residency().clone(),
            session,
            processor: None,
        })
    }

    fn with_processor(
        mut self,
        processor: Option<eredu_runtime::SelectedProcessorExecution>,
    ) -> Self {
        self.processor = processor;
        self
    }

    pub(crate) const fn parallel_topology(&self) -> Option<eredu_core::ParallelRankTopology> {
        self.parallel
    }

    pub(crate) const fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization
    }
}

/// MLX arrays, modules, and media preprocessing from one GGUF artifact.
struct MaterializedGgufModel {
    model: Executable,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

/// Proof that a family-specific complete-model loader serves an execution
/// responsibility outside replicated text composition.
pub(crate) struct ExcludedFamilyRoute(());

pub(crate) fn validate_gguf_projector_requirement(
    architecture: eredu_architectures::GgufArchitecture,
    has_projector: bool,
) -> Result<(), Error> {
    let requirement = eredu_architectures::preparation::gguf_composite_artifact_plan(architecture)
        .media_projector_requirement();
    if requirement == eredu_architectures::preparation::GgufMediaProjectorRequirement::Required
        && !has_projector
    {
        return Err(Error::ArchitectureModel(format!(
            "{} preparation omitted its architecture-required media projector",
            architecture.model_kind().canonical_name()
        )));
    }
    Ok(())
}

fn materialize_gguf_model(
    source: &structural::AdmittedGguf,
    projector: Option<&structural::AdmittedGgufProjector>,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Executable, Error> {
    let kind = source.architecture().model_kind();
    let quantization = options.weight_quantization();
    let binding = GgufBinding::for_kind(kind).ok_or_else(|| {
        Error::ArchitectureModel(format!(
            "MLX has no GGUF realization for {}",
            kind.canonical_name()
        ))
    })?;
    validate_gguf_projector_requirement(source.architecture(), projector.is_some())?;
    structural::validate_complete_gguf_quantization(kind, quantization.is_some())?;
    let model = match binding {
        GgufBinding::Quantized(QuantizedGgufBinding::KimiLinear) => {
            let loaded = crate::composition::kimi_linear::load_kimi_linear_gguf_model(
                source,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::kimi_linear(kind, loaded)?
        }
        GgufBinding::Fixed(FixedGgufBinding::DeepSeekV3) => {
            let loaded = crate::composition::deepseek::load_gguf(
                source,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::deepseek(kind, Box::new(loaded))?
        }
        GgufBinding::Fixed(FixedGgufBinding::DeepSeekV4) => {
            let loaded = crate::composition::deepseek::load_gguf(
                source,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::deepseek(kind, Box::new(loaded))?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::GptOss) => {
            let loaded = crate::composition::gpt_oss::load_gpt_oss_gguf_model(
                source,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::gpt_oss(kind, loaded)?
        }
        GgufBinding::Fixed(FixedGgufBinding::Inkling) => {
            let loaded = crate::composition::inkling::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::inkling(kind, loaded)?
        }
        GgufBinding::Fixed(FixedGgufBinding::Gemma4) => {
            let loaded = crate::composition::gemma4::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::gemma4(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::Llama) => unreachable!(
            "architecture-owned eligibility rejects replicated GGUF before family binding"
        ),
        GgufBinding::Fixed(FixedGgufBinding::MuseGlimmer) => {
            let loaded = crate::composition::muse_glimmer::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Executable::muse_glimmer(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::Lfm2) => {
            let loaded = crate::composition::lfm2::load_lfm2_gguf_model(
                source,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::lfm2(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::NemotronH) => {
            let loaded = crate::composition::nemotron_h::load_nemotron_h_gguf_model(
                source,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::nemotron_h(kind, loaded)?
        }
        GgufBinding::Quantized(QuantizedGgufBinding::Qwen) => {
            let loaded = crate::composition::qwen::load_qwen_gguf_model(
                source,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            Executable::qwen(kind, loaded)?
        }
        GgufBinding::Quantized(
            QuantizedGgufBinding::Qwen3Vl | QuantizedGgufBinding::Qwen3VlMoe,
        ) => return Err(Error::ArchitectureModel(
            "replicated Qwen3-VL GGUF requires the architecture-selected composite materializer"
                .into(),
        )),
        GgufBinding::Quantized(
            binding @ (QuantizedGgufBinding::Qwen35 | QuantizedGgufBinding::Qwen3Next),
        ) => {
            let loaded = crate::composition::qwen::hybrid::load_gguf(
                source,
                projector,
                family_route,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?;
            match binding {
                QuantizedGgufBinding::Qwen35 => Executable::qwen35(kind, loaded)?,
                QuantizedGgufBinding::Qwen3Next => Executable::qwen3_next(kind, loaded)?,
                _ => unreachable!(),
            }
        }
    };
    Ok(model)
}

pub(crate) fn select_preparation(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    policy: eredu_core::PreparationPolicy,
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_grouped_capabilities(
        inspection,
        options,
        policy,
        &super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
    )
}

#[derive(Clone)]
struct MlxExecutionClassSelection {
    options: MlxLoadRequest,
    policy: eredu_core::PreparationPolicy,
    admitted_session: eredu_core::SessionCapabilities,
    processor: Option<eredu_runtime::SelectedProcessorExecution>,
    partitioned_base: bool,
}

struct MlxPartitionedAdmissionSelection<'a> {
    inspection: &'a eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    base: MlxExecutionClassSelection,
    communication: &'a eredu_runtime::CommunicationCapabilities,
}

impl eredu_architectures::partitioned_execution::PartitionedAdmissionDispatcher
    for MlxPartitionedAdmissionSelection<'_>
{
    type Output = MlxSelectedExecution;
    type Error = Error;

    fn direct(
        self,
        requirements: eredu_architectures::partitioned_execution::DirectPartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        use eredu_architectures::replicated_text::ReplicatedTextExecutionClassDispatcher as _;

        let selected = self
            .base
            .clone()
            .replicated(requirements.execution().clone())?;
        let selected =
            eredu_architectures::partitioned_execution::select_direct_partitioned_admission(
                requirements,
                selected,
                self.communication,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        match eredu_architectures::partitioned_execution::dense_decoder_partitioned_production_route(
            self.inspection,
            &selected,
        ) {
            eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::NeutralPartitioned => {
                Ok(MlxSelectedExecution::PartitionedDense { selected })
            }
            eredu_architectures::partitioned_execution::DenseDecoderPartitionedProductionRoute::Unsupported(reason) => {
                Err(Error::ArchitectureModel(format!(
                    "selected direct partition has no neutral production implementation: {reason:?}"
                )))
            }
        }
    }

    fn routed(
        self,
        requirements: eredu_architectures::partitioned_execution::RoutedPartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        use eredu_architectures::replicated_text::ReplicatedTextExecutionClassDispatcher as _;

        let selected = self.base.clone().routed(requirements.execution().clone())?;
        let selected =
            eredu_architectures::partitioned_execution::select_routed_partitioned_admission(
                requirements,
                selected,
                self.communication,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let supported =
            eredu_architectures::partitioned_execution::routed_partitioned_production_supported(
                self.inspection,
                &selected,
            );
        if !supported {
            return Err(Error::ArchitectureModel(
                "selected routed execution has no neutral production implementation".into(),
            ));
        }
        Ok(MlxSelectedExecution::PartitionedRouted { selected })
    }

    fn composite(
        self,
        requirements: eredu_architectures::partitioned_execution::CompositePartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        use eredu_architectures::replicated_text::ReplicatedTextExecutionClassDispatcher as _;

        let selected = self
            .base
            .clone()
            .composite(requirements.execution().clone())?;
        let selected =
            eredu_architectures::partitioned_execution::select_composite_partitioned_admission(
                requirements,
                selected,
                self.communication,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        match eredu_architectures::composite_partitioned::composite_partitioned_production_decision(&selected) {
            eredu_architectures::composite_partitioned::CompositePartitionedProductionDecision::Resident => {
                Ok(MlxSelectedExecution::PartitionedComposite { selected })
            }
            eredu_architectures::composite_partitioned::CompositePartitionedProductionDecision::Unsupported(reason) => {
                Err(Error::ArchitectureModel(reason.into()))
            }
        }
    }
}

impl eredu_architectures::replicated_text::ReplicatedTextExecutionClassDispatcher
    for MlxExecutionClassSelection
{
    type Replicated = eredu_runtime::SelectedReplicatedTextRealization;
    type Routed = eredu_architectures::SelectedRoutedTextRealization;
    type Composite = eredu_architectures::replicated_text::SelectedCompositeTextRealization;
    type Other = SelectedMlxConstruction;
    type Error = Error;

    fn replicated(
        self,
        requirements: eredu_runtime::ReplicatedTextRequirements,
    ) -> Result<Self::Replicated, Self::Error> {
        let mut request = ReplicatedTextSelectionRequest::new(
            self.options.weight_residency.layers(),
            self.options.state_residency().clone(),
        )
        .with_session(self.admitted_session)
        .with_prompt_cache(matches!(
            self.options.state_residency(),
            CacheResidencyPolicy::Paged(_)
        ))
        .with_exact_completion(true);
        if !self.partitioned_base {
            if let Some(topology) = self.policy.topology() {
                request = request.with_topology(topology);
            }
        }
        if let Some(quantization) = self.policy.quantization() {
            request = request.with_quantization(quantization);
        }
        select_replicated_text_realization(
            &requirements,
            &request,
            &super::replicated_text::capabilities(&requirements, &request),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn routed(
        self,
        requirements: eredu_architectures::RoutedTextRequirements,
    ) -> Result<Self::Routed, Self::Error> {
        let mut text = ReplicatedTextSelectionRequest::new(
            self.options.weight_residency.layers(),
            self.options.state_residency().clone(),
        )
        .with_session(self.admitted_session)
        .with_prompt_cache(matches!(
            self.options.state_residency(),
            CacheResidencyPolicy::Paged(_)
        ))
        .with_exact_completion(true);
        if !self.partitioned_base {
            if let Some(topology) = self.policy.topology() {
                text = text.with_topology(topology);
            }
        }
        if let Some(quantization) = self.policy.quantization() {
            text = text.with_quantization(quantization);
        }
        let request = eredu_architectures::RoutedTextSelectionRequest::new(
            text,
            self.options.weight_residency,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        eredu_architectures::select_routed_text_realization(
            &requirements,
            &request,
            &super::replicated_text::capabilities(requirements.text(), request.text()),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn composite(
        self,
        requirements: eredu_architectures::replicated_text::CompositeTextRequirements,
    ) -> Result<Self::Composite, Self::Error> {
        let mut execution_request = ReplicatedTextSelectionRequest::new(
            self.options.weight_residency.layers(),
            self.options.state_residency().clone(),
        )
        .with_session(self.admitted_session)
        .with_prompt_cache(matches!(
            self.options.state_residency(),
            CacheResidencyPolicy::Paged(_)
        ))
        .with_exact_completion(true);
        if !self.partitioned_base {
            if let Some(topology) = self.policy.topology() {
                execution_request = execution_request.with_topology(topology);
            }
        }
        if let Some(quantization) = self.policy.quantization() {
            execution_request = execution_request.with_quantization(quantization);
        }
        let processor = self.processor.ok_or_else(|| {
            Error::ArchitectureModel(
                "composite execution has no selected processor realization".into(),
            )
        })?;
        eredu_architectures::replicated_text::select_composite_text_realization_with_processor(
            &requirements,
            &execution_request,
            self.options.weight_residency,
            &super::replicated_text::capabilities(requirements.execution(), &execution_request),
            processor,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn other(self) -> Result<Self::Other, Self::Error> {
        SelectedMlxConstruction::from_request(self.options, self.admitted_session)
            .map(|selected| selected.with_processor(self.processor))
    }
}

pub(crate) fn select_preparation_with_grouped_capabilities(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    policy: eredu_core::PreparationPolicy,
    grouped_capabilities: &[eredu_runtime::GroupedOperationRequirement],
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_capabilities(
        inspection,
        options,
        policy,
        grouped_capabilities,
        &crate::backend::runtime::distributed::topology::mlx_communication_capabilities(),
    )
}

fn select_preparation_with_capabilities(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    policy: eredu_core::PreparationPolicy,
    grouped_capabilities: &[eredu_runtime::GroupedOperationRequirement],
    communication_capabilities: &eredu_runtime::CommunicationCapabilities,
) -> Result<MlxSelectedPreparation, Error> {
    if let Some(kind) = inspection
        .architecture_plan()
        .required_gguf_special_tokens()
    {
        return Err(Error::ArchitectureModel(format!(
            "GGUF {kind:?} media token IDs must be resolved by the facade before MLX preparation"
        )));
    }
    if policy != options.preparation_policy()? {
        return Err(Error::ArchitectureModel(
            "MLX preparation policy does not match the caller request".into(),
        ));
    }
    structural::validate_inspected_preparation(inspection, policy)?;
    let admitted_session = structural::inspected_session_capabilities(inspection, policy)?;
    let grouped_requirements = inspection
        .architecture_plan()
        .grouped_operation_requirements(policy.topology());
    let missing_grouped = grouped_requirements
        .iter()
        .filter(|required| !grouped_capabilities.contains(required))
        .collect::<Vec<_>>();
    if !missing_grouped.is_empty() {
        return Err(Error::ArchitectureModel(format!(
            "backend is missing architecture-required grouped mechanisms: {missing_grouped:?}"
        )));
    }
    let projection = inspection
        .architecture_plan()
        .prediction_target_projection()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let prediction_extension = projection.as_ref().map(|(_, extension)| extension.clone());
    let projected_inspection =
        projection.map(|(target, _)| inspection.clone().map_architecture_plan(|_complete| target));
    let inspection = projected_inspection.as_ref().unwrap_or(inspection);
    let processor =
        eredu_architectures::replicated_text::composite_processor_execution_requirements(
            inspection.architecture_plan(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .map(|requirements| {
            let request = eredu_runtime::ProcessorSelectionRequest::new(
                requirements
                    .modalities()
                    .iter()
                    .map(eredu_runtime::ModalityProcessorRequirements::modality),
            )
            .with_prepared_tensors(true)
            .with_projected_modalities(
                requirements
                    .modalities()
                    .iter()
                    .filter(|requirement| requirement.projected_embeddings())
                    .map(eredu_runtime::ModalityProcessorRequirements::modality),
            )
            .with_available_raw_media(
                policy
                    .topology()
                    .is_none_or(eredu_core::ParallelTopology::is_replicated)
                    && inspection.architecture_plan().has_processor(),
            );
            eredu_runtime::select_processor_execution(
                &requirements,
                &request,
                &super::processor::capabilities(),
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .transpose()?;
    let parallel_topology = options
        .parallel_topology()
        .filter(|topology| !topology.is_replicated());
    let rank_context = options.parallel_rank_context()?;
    if let (Some(extension), Some(topology)) = (prediction_extension.as_ref(), parallel_topology) {
        eredu_architectures::prediction_extension::validate_partitioned_prediction_extension(
            extension, topology,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    }
    let base = MlxExecutionClassSelection {
        options,
        policy,
        admitted_session,
        processor,
        partitioned_base: parallel_topology.is_some(),
    };
    let execution = if let Some(topology) = parallel_topology {
        let (maximum_batch_size, maximum_sequence_length) = base
            .options
            .partitioned_invocation_limits()?
            .ok_or_else(|| {
                Error::Parallel("partitioned selection requires explicit invocation limits".into())
            })?;
        let activation_dtype = base
            .options
            .pipeline_wire_contract()
            .ok_or_else(|| {
                Error::Parallel("partitioned selection requires an activation wire contract".into())
            })?
            .activation_dtype();
        let completion_policy = base
            .options
            .communication_completion_policy()?
            .ok_or_else(|| {
                Error::Parallel(
                    "partitioned selection requires an explicit bounded communication completion policy"
                        .into(),
                )
            })?;
        let request = eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
            topology.topology(),
            topology.global_rank(),
            maximum_batch_size,
            maximum_sequence_length,
            activation_dtype,
        )
        .map_err(Error::ArchitectureModel)?
        .with_completion_policy(completion_policy);
        match eredu_architectures::partitioned_execution::dispatch_partitioned_admission(
            inspection,
            request,
            MlxPartitionedAdmissionSelection {
                inspection,
                base: base.clone(),
                communication: communication_capabilities,
            },
        ) {
            Ok(selected) => selected,
            Err(
                error @ eredu_architectures::partitioned_execution::PartitionedAdmissionError::Unsupported(_),
            ) => {
                if prediction_extension.is_some() {
                    return Err(Error::ArchitectureModel(format!(
                        "neutral prediction target admission failed: {error}"
                    )));
                }
                return Err(Error::ArchitectureModel(error.to_string()));
            }
            Err(eredu_architectures::partitioned_execution::PartitionedAdmissionError::Dispatch(
                error,
            )) => return Err(error),
        }
    } else {
        let selected =
            eredu_architectures::replicated_text::dispatch_replicated_text_execution_class(
                inspection,
                policy.topology(),
                base,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        MlxSelectedExecution::Ordinary(selected)
    };
    Ok(MlxSelectedPreparation::new(
        execution,
        admitted_session,
        rank_context,
        prediction_extension,
    ))
}

pub fn materialize_model_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected: MlxSelectedPreparation,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let MlxSelectedPreparation {
        execution,
        session: _,
        rank_context: _,
        prediction_extension,
    } = selected;
    let materializer = MlxSelectedExecutionMaterializer {
        plan,
        prediction_extension,
        stream,
        weights_stream,
    };
    match execution {
        MlxSelectedExecution::Ordinary(selected) => {
            debug_assert!(distributed.is_none());
            selected.dispatch(materializer)
        }
        MlxSelectedExecution::PartitionedDense { selected } => {
            materialize_partitioned_dense_decoder(
                materializer.plan,
                materializer.prediction_extension,
                selected,
                distributed,
                stream,
                weights_stream,
            )
        }
        MlxSelectedExecution::PartitionedRouted { selected } => {
            materialize_partitioned_routed_decoder(
                materializer.plan,
                materializer.prediction_extension,
                selected,
                distributed,
                stream,
                weights_stream,
            )
        }
        MlxSelectedExecution::PartitionedComposite { selected } => {
            materialize_partitioned_composite(
                materializer.plan,
                materializer.prediction_extension,
                selected,
                distributed,
                stream,
                weights_stream,
            )
        }
    }
}

fn materialize_partitioned_composite(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected_prediction_extension: Option<
        eredu_architectures::configuration::PredictionExtensionPlan,
    >,
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
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let state_residency = selected.base().execution().state().policy().clone();
    let max_cached_shards = selected.base().execution().residency().max_cached_shards();
    let selected_processor = selected.base().processor().clone();
    let inspection = plan.inspection().clone();
    let projection = inspection
        .architecture_plan()
        .prediction_target_projection()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let prediction_extension = projection
        .as_ref()
        .map(|(_, extension)| extension.clone())
        .or(selected_prediction_extension)
        .or_else(|| {
            inspection
                .architecture_plan()
                .prediction_extension()
                .cloned()
        });
    let architecture_plan = projection.map_or_else(
        || inspection.architecture_plan().clone(),
        |(target, _)| target,
    );
    let prediction_extension_sources = prediction_extension
        .as_ref()
        .map(|extension| {
            prepared_safetensors_architecture(&architecture_plan).and_then(|target| {
                extension
                    .source_keys(target)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
        })
        .transpose()?
        .unwrap_or_default();
    let kind = prepared_model_kind(&architecture_plan);
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = match plan.into_artifact() {
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let store_architecture = prediction_extension.as_ref().map_or_else(
                || prepared_safetensors_architecture(&architecture_plan).cloned(),
                |extension| Ok(extension.complete_architecture().clone()),
            )?;
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                store_architecture,
                tensors,
                shards,
                max_cached_shards,
            )?;
            prepared.store()
        }
        ModelArtifact::Gguf { validated, .. } => {
            let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
            let (source, projector) = structural::AdmittedGguf::from_admission(
                architecture,
                architecture_plan.gguf_media_projector().cloned(),
                validated,
            )?;
            #[cfg(test)]
            super::path_instrumentation::payload_open();
            let primary_mapping = projector
                .as_ref()
                .map_or(source.plan().tensor_mapping(), |projector| {
                    projector.plan().primary_tensor_mapping()
                });
            let primary: Arc<dyn eredu_checkpoint::store::CheckpointSource> = Arc::new(
                crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                    source.checkpoint().clone(),
                    source.plan().checkpoint(),
                    primary_mapping,
                    max_cached_shards,
                )?,
            );
            match projector {
                None => primary,
                Some(projector) => {
                    let companion: Arc<dyn eredu_checkpoint::store::CheckpointSource> = Arc::new(
                        crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                            projector.checkpoint().clone(),
                            projector.plan().checkpoint(),
                            projector.plan().tensor_mapping(),
                            max_cached_shards,
                        )?,
                    );
                    Arc::new(eredu_checkpoint::store::CompositeCheckpointSource::new([
                        primary, companion,
                    ])?)
                }
            }
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact route for partitioned composite composition".into(),
            ));
        }
    };
    let prediction_extension_execution = prediction_extension
        .as_ref()
        .map(|extension| {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(extension, &selected, store.as_ref(), weights_stream, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::replicated_text::materialize_prediction_extension(
                prepared,
                store.as_ref(),
                stream,
                weights_stream,
            )
        })
        .transpose()
        .map_err(|error| {
            Error::ArchitectureModel(format!(
                "prediction extension materialization failed: {error}"
            ))
        })?;
    let target_store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        if prediction_extension_sources.is_empty() {
            store
        } else {
            Arc::new(PredictionTargetCheckpointSource::new(
                store,
                prediction_extension_sources,
            ))
        };
    let mut executable = super::replicated_text::bind_partitioned_composite(
        selected,
        target_store,
        distributed,
        stream,
        weights_stream,
    )?;
    if let Some((extension, execution)) = prediction_extension.zip(prediction_extension_execution) {
        let capability =
            eredu_architectures::prediction_extension::prediction_extension_capability(&extension)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        executable.install_prediction_extension_contract(execution, capability)?;
    }
    let model = MlxModel::complete(
        Executable::replicated_text(kind, executable)?,
        floating_state_dtype_bytes,
        state_residency,
    );
    attach_selected_processor(model, &architecture_plan, &selected_processor)
}

fn materialize_partitioned_routed_decoder(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected_prediction_extension: Option<
        eredu_architectures::configuration::PredictionExtensionPlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::RoutedTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let kind = prepared_model_kind(plan.inspection().architecture_plan());
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral routed-decoder binding has no realized communication".into())
    })?;
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let state_residency = selected.base().text().state().policy().clone();
    let max_cached_shards = selected.base().text().residency().max_cached_shards();
    let inspection = plan.inspection().clone();
    let projection = inspection
        .architecture_plan()
        .prediction_target_projection()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let prediction_extension = projection
        .as_ref()
        .map(|(_, extension)| extension.clone())
        .or(selected_prediction_extension)
        .or_else(|| {
            inspection
                .architecture_plan()
                .prediction_extension()
                .cloned()
        });
    let architecture_plan = projection.map_or_else(
        || inspection.architecture_plan().clone(),
        |(target, _)| target,
    );
    let prediction_extension_sources = prediction_extension
        .as_ref()
        .map(|extension| {
            prepared_safetensors_architecture(&architecture_plan).and_then(|target| {
                extension
                    .source_keys(target)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
        })
        .transpose()?
        .unwrap_or_default();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = match plan.into_artifact() {
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let store_architecture = prediction_extension.as_ref().map_or_else(
                || prepared_safetensors_architecture(&architecture_plan).cloned(),
                |extension| Ok(extension.complete_architecture().clone()),
            )?;
            super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                store_architecture,
                tensors,
                shards,
                max_cached_shards,
            )?
            .store()
        }
        ModelArtifact::Gguf { validated, .. } => {
            let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
            let (source, projector) = structural::AdmittedGguf::from_admission(
                architecture,
                architecture_plan.gguf_media_projector().cloned(),
                validated,
            )?;
            if projector.is_some() {
                return Err(Error::ArchitectureModel(
                    "partitioned routed text composition cannot bind a media projector".into(),
                ));
            }
            #[cfg(test)]
            super::path_instrumentation::payload_open();
            Arc::new(
                crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                    source.checkpoint().clone(),
                    source.plan().checkpoint(),
                    source.plan().tensor_mapping(),
                    max_cached_shards,
                )?,
            )
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact route for partitioned routed text composition".into(),
            ));
        }
    };
    let prediction_extension_execution = prediction_extension
        .as_ref()
        .map(|extension| {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(extension, &selected, store.as_ref(), weights_stream, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::replicated_text::materialize_prediction_extension(
                prepared,
                store.as_ref(),
                stream,
                weights_stream,
            )
        })
        .transpose()
        .map_err(|error| {
            Error::ArchitectureModel(format!(
                "prediction extension materialization failed: {error}"
            ))
        })?;
    let target_store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        if prediction_extension_sources.is_empty() {
            store
        } else {
            Arc::new(PredictionTargetCheckpointSource::new(
                store,
                prediction_extension_sources.clone(),
            ))
        };
    let mut executable = super::replicated_text::bind_partitioned_routed_decoder(
        &inspection,
        selected,
        target_store,
        distributed,
        prediction_extension_sources,
        stream,
        weights_stream,
    )?;
    if let Some((extension, execution)) = prediction_extension.zip(prediction_extension_execution) {
        let capability =
            eredu_architectures::prediction_extension::prediction_extension_capability(&extension)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        executable.install_prediction_extension_contract(execution, capability)?;
    }
    Ok(MlxModel::complete(
        Executable::replicated_text(kind, executable)?,
        floating_state_dtype_bytes,
        state_residency,
    ))
}

fn materialize_partitioned_dense_decoder(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected_prediction_extension: Option<
        eredu_architectures::configuration::PredictionExtensionPlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let kind = prepared_model_kind(plan.inspection().architecture_plan());
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral dense-decoder binding has no realized communication".into())
    })?;
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let state_residency = selected.base().state().policy().clone();
    let max_cached_shards = selected.base().residency().max_cached_shards();
    let inspection = plan.inspection().clone();
    let projection = inspection
        .architecture_plan()
        .prediction_target_projection()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let prediction_extension = projection
        .as_ref()
        .map(|(_, extension)| extension.clone())
        .or(selected_prediction_extension)
        .or_else(|| {
            inspection
                .architecture_plan()
                .prediction_extension()
                .cloned()
        });
    let architecture_plan = projection.map_or_else(
        || inspection.architecture_plan().clone(),
        |(target, _)| target,
    );
    let prediction_extension_sources = prediction_extension
        .as_ref()
        .map(|extension| {
            prepared_safetensors_architecture(&architecture_plan).and_then(|target| {
                extension
                    .source_keys(target)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
        })
        .transpose()?
        .unwrap_or_default();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let artifact = plan.into_artifact();
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = match artifact {
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let store_architecture = prediction_extension.as_ref().map_or_else(
                || prepared_safetensors_architecture(&architecture_plan).cloned(),
                |extension| Ok(extension.complete_architecture().clone()),
            )?;
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                store_architecture,
                tensors,
                shards,
                max_cached_shards,
            )?;
            prepared.store()
        }
        ModelArtifact::Gguf { validated, .. } => {
            let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
            let (source, projector) = structural::AdmittedGguf::from_admission(
                architecture,
                architecture_plan.gguf_media_projector().cloned(),
                validated,
            )?;
            if projector.is_some() {
                return Err(Error::ArchitectureModel(
                    "partitioned dense text composition cannot bind a media projector".into(),
                ));
            }
            #[cfg(test)]
            super::path_instrumentation::payload_open();
            Arc::new(
                crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                    source.checkpoint().clone(),
                    source.plan().checkpoint(),
                    source.plan().tensor_mapping(),
                    max_cached_shards,
                )?,
            )
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact route for partitioned dense text composition".into(),
            ));
        }
    };
    let missing_prediction_sources = prediction_extension_sources
        .iter()
        .filter(|source| {
            !store
                .source_keys()
                .iter()
                .any(|available| available == *source)
        })
        .cloned()
        .collect::<Vec<_>>();
    if !missing_prediction_sources.is_empty() {
        return Err(Error::ArchitectureModel(format!(
            "prediction extension sources are absent from the admitted checkpoint: {missing_prediction_sources:?}"
        )));
    }
    let prediction_extension_execution = prediction_extension
        .as_ref()
        .map(|extension| {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(extension, &selected, store.as_ref(), weights_stream, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::replicated_text::materialize_prediction_extension(
                prepared,
                store.as_ref(),
                stream,
                weights_stream,
            )
        })
        .transpose()
        .map_err(|error| {
            Error::ArchitectureModel(format!(
                "prediction extension materialization failed: {error}"
            ))
        })?;
    let target_store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        if prediction_extension_sources.is_empty() {
            store
        } else {
            Arc::new(PredictionTargetCheckpointSource::new(
                store,
                prediction_extension_sources.clone(),
            ))
        };
    let mut executable = super::replicated_text::bind_partitioned_dense_decoder(
        &inspection,
        selected,
        target_store,
        distributed,
        prediction_extension_sources,
        stream,
        weights_stream,
    )?;
    if let Some((extension, execution)) = prediction_extension.zip(prediction_extension_execution) {
        let capability =
            eredu_architectures::prediction_extension::prediction_extension_capability(&extension)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        executable.install_prediction_extension_contract(execution, capability)?;
    }
    Ok(MlxModel::complete(
        Executable::replicated_text(kind, executable)?,
        floating_state_dtype_bytes,
        state_residency,
    ))
}

struct MlxSelectedExecutionMaterializer<'a> {
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    prediction_extension: Option<eredu_architectures::configuration::PredictionExtensionPlan>,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl
    eredu_architectures::replicated_text::SelectedReplicatedTextExecutionDispatcher<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::replicated_text::SelectedCompositeTextRealization,
        SelectedMlxConstruction,
    > for MlxSelectedExecutionMaterializer<'_>
{
    type Output = MlxModel;
    type Error = Error;

    fn replicated(
        self,
        selected: eredu_runtime::SelectedReplicatedTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        materialize_replicated_text_plan(
            self.plan,
            self.prediction_extension,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn routed(
        self,
        selected: eredu_architectures::SelectedRoutedTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        materialize_routed_text_plan(
            self.plan,
            self.prediction_extension,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn composite(
        self,
        selected: eredu_architectures::replicated_text::SelectedCompositeTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        materialize_composite_text_plan(
            self.plan,
            self.prediction_extension,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn other(self, selected: SelectedMlxConstruction) -> Result<Self::Output, Self::Error> {
        let family_route = ExcludedFamilyRoute(());
        materialize_excluded_model_plan(
            self.plan,
            &family_route,
            selected,
            self.stream,
            self.weights_stream,
        )
    }
}

fn replicated_prediction_projection(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    selected: Option<eredu_architectures::configuration::PredictionExtensionPlan>,
) -> Result<
    (
        ArtifactArchitecturePlan,
        Option<eredu_architectures::configuration::PredictionExtensionPlan>,
        std::collections::BTreeSet<String>,
    ),
    Error,
> {
    let projection = inspection
        .architecture_plan()
        .prediction_target_projection()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let extension = projection
        .as_ref()
        .map(|(_, extension)| extension.clone())
        .or(selected);
    let target = projection.map_or_else(
        || inspection.architecture_plan().clone(),
        |(target, _)| target,
    );
    let sources = extension
        .as_ref()
        .map(|extension| {
            prepared_safetensors_architecture(&target).and_then(|target| {
                extension
                    .source_keys(target)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
        })
        .transpose()?
        .unwrap_or_default();
    Ok((target, extension, sources))
}

fn materialize_replicated_prediction_extension(
    extension: &eredu_architectures::configuration::PredictionExtensionPlan,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<super::replicated_text::MlxPredictionExtension, Error> {
    let prepared =
        eredu_architectures::prediction_extension::prepare_replicated_prediction_extension::<
            MlxNeuralBackend,
        >(extension, store, weights_stream, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    super::replicated_text::materialize_prediction_extension(
        prepared,
        store,
        stream,
        weights_stream,
    )
}

fn install_replicated_prediction_extension(
    executable: &mut Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>,
    extension: eredu_architectures::configuration::PredictionExtensionPlan,
    materialized: super::replicated_text::MlxPredictionExtension,
) -> Result<(), Error> {
    let capability =
        eredu_architectures::prediction_extension::prediction_extension_capability(&extension)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    executable.install_prediction_extension_contract(materialized, capability)
}

fn materialize_replicated_text_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected_prediction_extension: Option<
        eredu_architectures::configuration::PredictionExtensionPlan,
    >,
    realization: eredu_runtime::SelectedReplicatedTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let max_cached_shards = realization.residency().max_cached_shards();
    let state_residency = realization.state().policy().clone();
    let (architecture_plan, prediction_extension, prediction_extension_sources) =
        replicated_prediction_projection(plan.inspection(), selected_prediction_extension)?;
    let artifact = plan.into_artifact();
    let kind = prepared_model_kind(&architecture_plan);
    let executable = match artifact {
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let store_architecture = prediction_extension.as_ref().map_or_else(
                || prepared_safetensors_architecture(&architecture_plan).cloned(),
                |extension| Ok(extension.complete_architecture().clone()),
            )?;
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                store_architecture,
                tensors,
                shards,
                max_cached_shards,
            )?;
            let store = prepared.store();
            let materialized = prediction_extension
                .as_ref()
                .map(|extension| {
                    materialize_replicated_prediction_extension(
                        extension,
                        store.as_ref(),
                        stream,
                        weights_stream,
                    )
                })
                .transpose()?;
            let target_store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                if prediction_extension_sources.is_empty() {
                    store
                } else {
                    Arc::new(PredictionTargetCheckpointSource::new(
                        store,
                        prediction_extension_sources,
                    ))
                };
            let mut executable = bind_replicated_text(
                &architecture_plan,
                realization,
                target_store,
                stream,
                weights_stream,
            )?;
            if let Some((extension, materialized)) = prediction_extension.zip(materialized) {
                install_replicated_prediction_extension(&mut executable, extension, materialized)?;
            }
            executable
        }
        ModelArtifact::Gguf { validated, .. } => {
            let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
            let (source, projector) = structural::AdmittedGguf::from_admission(
                architecture,
                architecture_plan.gguf_media_projector().cloned(),
                validated,
            )?;
            if projector.is_some() {
                return Err(Error::ArchitectureModel(
                    "replicated text composition cannot bind a media projector".into(),
                ));
            }
            #[cfg(test)]
            super::path_instrumentation::payload_open();
            let store = Arc::new(
                crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                    source.checkpoint().clone(),
                    source.plan().checkpoint(),
                    source.plan().tensor_mapping(),
                    max_cached_shards,
                )?,
            );
            bind_replicated_text(
                &architecture_plan,
                realization,
                store,
                stream,
                weights_stream,
            )?
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact route for replicated text composition".into(),
            ));
        }
    };
    let model = MlxModel::complete(
        Executable::replicated_text(kind, executable)?,
        floating_state_dtype_bytes,
        state_residency,
    );
    Ok(model)
}

fn materialize_routed_text_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected_prediction_extension: Option<
        eredu_architectures::configuration::PredictionExtensionPlan,
    >,
    realization: eredu_architectures::SelectedRoutedTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let max_cached_shards = realization.text().residency().max_cached_shards();
    let state_residency = realization.text().state().policy().clone();
    let complete_inspection = plan.inspection().clone();
    let (architecture_plan, prediction_extension, prediction_extension_sources) =
        replicated_prediction_projection(&complete_inspection, selected_prediction_extension)?;
    let inspection =
        complete_inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let artifact = plan.into_artifact();
    let kind = prepared_model_kind(&architecture_plan);
    let executable = match artifact {
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let store_architecture = prediction_extension.as_ref().map_or_else(
                || prepared_safetensors_architecture(&architecture_plan).cloned(),
                |extension| Ok(extension.complete_architecture().clone()),
            )?;
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                store_architecture,
                tensors,
                shards,
                max_cached_shards,
            )?;
            let store = prepared.store();
            let materialized = prediction_extension
                .as_ref()
                .map(|extension| {
                    materialize_replicated_prediction_extension(
                        extension,
                        store.as_ref(),
                        stream,
                        weights_stream,
                    )
                })
                .transpose()?;
            let target_store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                if prediction_extension_sources.is_empty() {
                    store
                } else {
                    Arc::new(PredictionTargetCheckpointSource::new(
                        store,
                        prediction_extension_sources,
                    ))
                };
            let mut executable = super::replicated_text::bind_routed_text(
                &inspection,
                realization,
                target_store,
                stream,
                weights_stream,
            )?;
            if let Some((extension, materialized)) = prediction_extension.zip(materialized) {
                install_replicated_prediction_extension(&mut executable, extension, materialized)?;
            }
            executable
        }
        ModelArtifact::Gguf { validated, .. } => {
            let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
            let (source, projector) = structural::AdmittedGguf::from_admission(
                architecture,
                architecture_plan.gguf_media_projector().cloned(),
                validated,
            )?;
            if projector.is_some() {
                return Err(Error::ArchitectureModel(
                    "replicated routed text composition cannot bind a media projector".into(),
                ));
            }
            #[cfg(test)]
            super::path_instrumentation::payload_open();
            let store = Arc::new(
                crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                    source.checkpoint().clone(),
                    source.plan().checkpoint(),
                    source.plan().tensor_mapping(),
                    max_cached_shards,
                )?,
            );
            super::replicated_text::bind_routed_text(
                &inspection,
                realization,
                store,
                stream,
                weights_stream,
            )?
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact route for replicated routed text composition".into(),
            ));
        }
    };
    let model = MlxModel::complete(
        Executable::replicated_text(kind, executable)?,
        floating_state_dtype_bytes,
        state_residency,
    );
    Ok(model)
}

fn materialize_composite_text_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected_prediction_extension: Option<
        eredu_architectures::configuration::PredictionExtensionPlan,
    >,
    realization: eredu_architectures::replicated_text::SelectedCompositeTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let max_cached_shards = realization.execution().residency().max_cached_shards();
    let state_residency = realization.execution().state().policy().clone();
    let complete_inspection = plan.inspection().clone();
    let (architecture_plan, prediction_extension, prediction_extension_sources) =
        replicated_prediction_projection(&complete_inspection, selected_prediction_extension)?;
    let inspection =
        complete_inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let requirements =
        eredu_architectures::replicated_text::composite_text_requirements(&inspection)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let selected_processor = realization.processor().clone();
    let artifact = plan.into_artifact();
    let kind = prepared_model_kind(&architecture_plan);
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = match artifact {
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let store_architecture = prediction_extension.as_ref().map_or_else(
                || prepared_safetensors_architecture(&architecture_plan).cloned(),
                |extension| Ok(extension.complete_architecture().clone()),
            )?;
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                store_architecture,
                tensors,
                shards,
                max_cached_shards,
            )?;
            Arc::clone(&prepared.store())
        }
        ModelArtifact::Gguf { validated, .. } => {
            let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
            let (source, projector) = structural::AdmittedGguf::from_admission(
                architecture,
                architecture_plan.gguf_media_projector().cloned(),
                validated,
            )?;
            #[cfg(test)]
            super::path_instrumentation::payload_open();
            let primary_mapping = projector
                .as_ref()
                .map_or(source.plan().tensor_mapping(), |projector| {
                    projector.plan().primary_tensor_mapping()
                });
            let primary: Arc<dyn eredu_checkpoint::store::CheckpointSource> = Arc::new(
                crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                    source.checkpoint().clone(),
                    source.plan().checkpoint(),
                    primary_mapping,
                    max_cached_shards,
                )?,
            );
            match projector {
                None => primary,
                Some(projector) => {
                    let companion: Arc<dyn eredu_checkpoint::store::CheckpointSource> = Arc::new(
                        crate::backend::runtime::checkpoint::store::open_gguf_checkpoint_source(
                            projector.checkpoint().clone(),
                            projector.plan().checkpoint(),
                            projector.plan().tensor_mapping(),
                            max_cached_shards,
                        )?,
                    );
                    Arc::new(eredu_checkpoint::store::CompositeCheckpointSource::new([
                        primary, companion,
                    ])?)
                }
            }
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact route for replicated composite composition".into(),
            ));
        }
    };
    let materialized = prediction_extension
        .as_ref()
        .map(|extension| {
            materialize_replicated_prediction_extension(
                extension,
                store.as_ref(),
                stream,
                weights_stream,
            )
        })
        .transpose()?;
    let target_store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        if prediction_extension_sources.is_empty() {
            store
        } else {
            Arc::new(PredictionTargetCheckpointSource::new(
                store,
                prediction_extension_sources,
            ))
        };
    let mut executable = eredu_architectures::replicated_text::visit_composite_text_architecture::<
        crate::backend::nn::shared::MlxNeuralBackend,
        crate::backend::runtime::cache::state::MlxHybridState,
        _,
    >(
        requirements,
        realization,
        target_store,
        stream,
        super::replicated_text::CompositeBindingVisitor {
            stream,
            weights_stream,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if let Some((extension, materialized)) = prediction_extension.zip(materialized) {
        install_replicated_prediction_extension(&mut executable, extension, materialized)?;
    }
    let model = MlxModel::complete(
        Executable::replicated_text(kind, executable)?,
        floating_state_dtype_bytes,
        state_residency,
    );
    attach_selected_processor(model, &architecture_plan, &selected_processor)
}

fn materialize_excluded_model_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let selected_processor = options.processor.clone();
    if options
        .parallel_topology()
        .is_none_or(|topology| topology.is_replicated())
        && matches!(
            eredu_architectures::replicated_text::replicated_text_execution_class(
                plan.inspection(),
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            eredu_architectures::replicated_text::ReplicatedTextExecutionClass::Composite(_)
        )
    {
        return Err(Error::ArchitectureModel(
            "replicated composite execution requires the architecture-selected composite materializer"
                .into(),
        ));
    }
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(plan.inspection())?;
    let state_residency = options.state_residency.clone();
    if options
        .parallel_topology()
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::ArchitectureModel(
            "selected distributed execution has no neutral production implementation".into(),
        ));
    }
    let architecture_plan = plan.inspection().architecture_plan().clone();
    let artifact = plan.into_artifact();
    match artifact {
        artifact @ ModelArtifact::Gguf { .. } => materialize_gguf_artifact(
            artifact,
            architecture_plan,
            family_route,
            options,
            stream,
            weights_stream,
        )
        .map(|model| {
            complete_gguf_model(model, floating_state_dtype_bytes, state_residency.clone())
        }),
        ModelArtifact::SafeTensors {
            path: _,
            configuration,
            tensors,
            shards,
        } => {
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                prepared_safetensors_architecture(&architecture_plan)?.clone(),
                tensors,
                shards,
                options.weight_residency.max_cached_shards(),
            )?;
            let model =
                materialize_safetensors(&prepared, family_route, options, stream, weights_stream)
                    .map(|model| {
                    MlxModel::complete(model, floating_state_dtype_bytes, state_residency.clone())
                })?;
            attach_optional_selected_processor(
                model,
                &architecture_plan,
                selected_processor.as_ref(),
            )
        }
        _ => Err(Error::ArchitectureModel(
            "unsupported artifact route for model composition".into(),
        )),
    }
}

pub(super) fn bind_replicated_text(
    architecture_plan: &ArtifactArchitecturePlan,
    selected: eredu_runtime::SelectedReplicatedTextRealization,
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>, Error> {
    let visitor = super::replicated_text::BindingVisitor {
        stream,
        weights_stream,
    };
    eredu_architectures::replicated_text::dispatch_replicated_text_architecture::<
        crate::backend::nn::shared::MlxNeuralBackend,
        _,
    >(architecture_plan, selected, store, stream, visitor)
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

fn inspected_floating_state_dtype_bytes(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<std::num::NonZeroU8, Error> {
    let source = match inspection.format() {
        eredu_core::ArtifactFormat::Gguf => {
            eredu_architectures::preparation::prepared_gguf_floating_state_dtype_source(
                prepared_gguf_plan(inspection.architecture_plan())?,
                inspection.tensors(),
            )
        }
        eredu_core::ArtifactFormat::SafeTensors => {
            eredu_architectures::preparation::prepared_safetensors_floating_state_dtype_source(
                prepared_safetensors_architecture(inspection.architecture_plan())?,
                inspection.tensors(),
            )
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact format for floating-state dtype inspection".into(),
            ));
        }
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    mlx_floating_state_dtype_bytes(source.dtype()).map_err(|dtype| {
        Error::ArchitectureModel(format!(
            "floating-state dtype source {:?} has unsupported MLX activation dtype {dtype:?}",
            source.checkpoint_tensor()
        ))
    })
}

fn mlx_floating_state_dtype_bytes(
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
#[allow(
    clippy::items_after_test_module,
    reason = "floating-state dtype tests stay adjacent to dtype resolution"
)]
mod floating_state_dtype_tests {
    use super::{
        inspected_floating_state_dtype_bytes, mlx_floating_state_dtype_bytes,
        select_preparation_with_capabilities, MlxSelectedExecution,
    };
    use crate::backend::{ExecutionContext, MlxBackend};
    use eredu_core::{
        checkpoint::TensorDtype, residency::OffloadConfig, ModelLoadingBackend as _,
        ParallelTopology,
    };
    use eredu_gguf::{GgmlType, MetadataValue, TensorInput, Writer};

    #[test]
    fn deepseek_v4_prediction_selection_retains_neutral_extension_contract() {
        let root = tempfile::tempdir().unwrap();
        crate::tests::distributed_pipeline_ring::write_deepseek_v4_fixture(root.path(), 1);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            4,
            4096,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        );
        let policy = options.preparation_policy().unwrap();

        let selected = super::select_preparation(&inspection, options, policy).unwrap();

        let (target, extension) = inspection
            .architecture_plan()
            .prediction_target_projection()
            .unwrap()
            .unwrap();
        let extension_sources = extension
            .source_keys(target.safetensors_architecture().unwrap())
            .unwrap();
        assert_eq!(extension_sources.len(), 41);
        let target_sources = target
            .safetensors_architecture()
            .unwrap()
            .checkpoint_resolution()
            .unwrap()
            .source_keys();
        let complete_sources = extension
            .complete_architecture()
            .checkpoint_resolution()
            .unwrap()
            .source_keys();
        assert!(target_sources.is_disjoint(&extension_sources));
        assert_eq!(
            target_sources
                .union(&extension_sources)
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            *complete_sources
        );
        assert!(selected.prediction_extension.is_some());
        assert!(matches!(
            selected.execution,
            MlxSelectedExecution::PartitionedDense { .. }
                | MlxSelectedExecution::PartitionedRouted { .. }
        ));
        assert!(selected.realized_communication_manifest().is_some());
        assert!(selected.rank_context().is_some());
    }
    use safemlx::{Device, DeviceType};
    use std::collections::BTreeMap;

    fn write_minimal_llama_gguf(path: &std::path::Path, dtype: GgmlType) {
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ("llama.embedding_length".into(), MetadataValue::Uint32(16)),
            (
                "llama.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "llama.feed_forward_length".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            ("llama.vocab_size".into(), MetadataValue::Uint32(16)),
            ("llama.context_length".into(), MetadataValue::Uint32(1)),
        ]);
        let vector_data = [0_u8; 32];
        let matrix_data = [0_u8; 512];
        let tensor = |name, dimensions| TensorInput {
            name,
            dimensions,
            ggml_type: dtype,
            data: if dimensions.len() == 1 {
                &vector_data
            } else {
                &matrix_data
            },
        };
        let tensors = [
            tensor("token_embd.weight", &[16, 16]),
            tensor("output_norm.weight", &[16]),
            tensor("blk.0.attn_norm.weight", &[16]),
            tensor("blk.0.ffn_norm.weight", &[16]),
            tensor("blk.0.attn_q.weight", &[16, 16]),
            tensor("blk.0.attn_k.weight", &[16, 16]),
            tensor("blk.0.attn_v.weight", &[16, 16]),
            tensor("blk.0.attn_output.weight", &[16, 16]),
            tensor("blk.0.ffn_gate.weight", &[16, 16]),
            tensor("blk.0.ffn_up.weight", &[16, 16]),
            tensor("blk.0.ffn_down.weight", &[16, 16]),
            tensor("blk.1.attn_norm.weight", &[16]),
            tensor("blk.1.ffn_norm.weight", &[16]),
            tensor("blk.1.attn_q.weight", &[16, 16]),
            tensor("blk.1.attn_k.weight", &[16, 16]),
            tensor("blk.1.attn_v.weight", &[16, 16]),
            tensor("blk.1.attn_output.weight", &[16, 16]),
            tensor("blk.1.ffn_gate.weight", &[16, 16]),
            tensor("blk.1.ffn_up.weight", &[16, 16]),
            tensor("blk.1.ffn_down.weight", &[16, 16]),
        ];
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
            .unwrap();
    }

    #[test]
    fn partitioned_capability_failure_precedes_native_or_payload_work() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        );
        let policy = options.preparation_policy().unwrap();
        super::super::path_instrumentation::reset();

        let error = select_preparation_with_capabilities(
            &inspection,
            options,
            policy,
            &super::super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
            &eredu_runtime::CommunicationCapabilities::new([]).unwrap(),
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("communication"),
            "unexpected selection failure: {error}"
        );
        assert_eq!(
            super::super::path_instrumentation::communication_realization_attempts(),
            0
        );
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            Default::default()
        );
    }

    #[test]
    fn data_parallel_request_reaches_neutral_selection_and_fails_before_native_or_payload_work() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology =
            eredu_core::ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 2).unwrap(), 0)
                .unwrap();
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        );
        let policy = options.preparation_policy().unwrap();
        super::super::path_instrumentation::reset();

        let error = super::select_preparation(&inspection, options, policy).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("data-parallel execution is not supported"),
            "unexpected data-parallel rejection: {error}"
        );
        assert_eq!(
            super::super::path_instrumentation::communication_realization_attempts(),
            0
        );
        assert_eq!(
            super::super::path_instrumentation::manifest_communication_realization_attempts(),
            0
        );
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            Default::default()
        );
    }

    #[test]
    fn public_preparation_rejects_every_oversubscribed_pipeline_rank_before_native_work() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let backend = MlxBackend::new(execution.stream(), execution.stream());

        for rank in 0..3 {
            let topology = crate::test_parallel_rank(rank, 1, 3, 1);
            let options = crate::MlxLoadRequest::with_parallel(
                topology,
                crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                1,
                crate::MlxLoadRequest::test_communication_completion_policy(),
            );
            let policy = backend.preparation_policy(&options).unwrap();
            super::super::path_instrumentation::reset();

            let error = backend
                .select_preparation(&inspection, &options, policy)
                .unwrap_err();

            assert!(
                error
                    .to_string()
                    .contains("decoder execution group 0 has 2 units for 3 pipeline stages"),
                "rank {rank} returned an unexpected selection failure: {error}"
            );
            assert_eq!(
                super::super::path_instrumentation::communication_realization_attempts(),
                0,
                "rank {rank} reached native communication realization"
            );
            assert_eq!(
                super::super::path_instrumentation::manifest_communication_realization_attempts(),
                0,
                "rank {rank} reached opaque manifest realization"
            );
            assert_eq!(
                super::super::path_instrumentation::snapshot(),
                Default::default(),
                "rank {rank} performed payload, construction, state, or execution work"
            );
        }
    }

    #[test]
    fn gguf_llama_tp_selects_neutral_partitioned_execution() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        );
        let policy = options.preparation_policy().unwrap();

        let selected = super::select_preparation(&inspection, options, policy).unwrap();
        assert!(matches!(
            &selected.execution,
            MlxSelectedExecution::PartitionedDense { .. }
        ));
        let manifest = selected
            .communication_manifest()
            .expect("prediction-free Llama TP must retain neutral communication");
        assert_eq!(manifest.world_size(), 2);
        assert_eq!(manifest.rank(), 0);
        assert!(!manifest.groups().is_empty());
        assert!(selected.rank_context().is_some());
    }

    #[test]
    fn gguf_llama_pp_selects_neutral_partitioned_execution() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 1, 2, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        );
        let policy = options.preparation_policy().unwrap();

        let selected = super::select_preparation(&inspection, options, policy).unwrap();
        assert!(matches!(
            &selected.execution,
            MlxSelectedExecution::PartitionedDense { .. }
        ));
        let manifest = selected
            .communication_manifest()
            .expect("prediction-free Llama PP must retain neutral communication");
        assert_eq!(manifest.world_size(), 2);
        assert!(!manifest.routes().is_empty());
        let session_group = match &selected.execution {
            MlxSelectedExecution::PartitionedDense { selected } => {
                selected.requirements().session_group()
            }
            _ => None,
        }
        .expect("PP admission must select one session-wide publication group");
        let descriptor = manifest
            .groups()
            .iter()
            .find(|group| group.id() == session_group)
            .expect("selected session group must be present in the manifest");
        assert_eq!(descriptor.creation_order(), 0);
        assert_eq!(descriptor.members(), [0, 1]);
        assert_eq!(descriptor.local_index(), Some(0));
        assert_eq!(
            descriptor
                .requirements()
                .operations()
                .iter()
                .map(eredu_runtime::CommunicationOperationRequirement::operation)
                .collect::<Vec<_>>(),
            [
                eredu_runtime::CommunicationOperation::Broadcast,
                eredu_runtime::CommunicationOperation::FailureAgreement,
            ]
        );
        let broadcast = &descriptor.requirements().operations()[0];
        assert_eq!(
            broadcast.dtypes(),
            [eredu_core::checkpoint::TensorDtype::F32]
        );
        assert!(broadcast.exact_completion());
        assert_eq!(broadcast.limits().unwrap().max_tensors(), 1);
        assert_eq!(broadcast.limits().unwrap().max_tensor_rank(), 3);
        assert!(descriptor.requirements().operations()[1].limits().is_none());
        assert!(selected.rank_context().is_some());
        assert!(selected.realized_communication_manifest().is_some());
    }

    #[test]
    fn gguf_llama_bounded_residency_selects_neutral_partitioned_execution() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let layerwise = eredu_runtime::WeightResidency::layerwise_host(
            eredu_runtime::LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
        );
        let dense = eredu_runtime::WeightResidency::dense_disk_stream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
        );

        for residency in [layerwise, dense] {
            let topology = crate::test_parallel_rank(0, 2, 1, 1);
            let options = crate::MlxLoadRequest::with_parallel(
                topology,
                crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                1,
                crate::MlxLoadRequest::test_communication_completion_policy(),
            )
            .with_weight_residency(residency);
            let policy = options.preparation_policy().unwrap();

            let selected = super::select_preparation(&inspection, options, policy).unwrap();
            assert!(matches!(
                &selected.execution,
                MlxSelectedExecution::PartitionedDense { .. }
            ));
            assert!(selected.communication_manifest().is_some());
            assert!(selected.rank_context().is_some());
        }
    }

    #[test]
    fn gguf_llama_tp_transform_selects_the_immutable_neutral_route() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options =
            crate::MlxLoadRequest::with_quantization(eredu_core::QuantizationRequest::Affine {
                group_size: 16,
                bits: 4,
            })
            .with_parallel_topology(
                topology,
                crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                1,
                crate::MlxLoadRequest::test_communication_completion_policy(),
            );
        let policy = options.preparation_policy().unwrap();
        super::super::path_instrumentation::reset();

        let selected = super::select_preparation(&inspection, options, policy).unwrap();
        assert!(matches!(
            &selected.execution,
            MlxSelectedExecution::PartitionedDense { .. }
        ));
        assert!(selected.communication_manifest().is_some());
        assert!(selected.rank_context().is_some());
    }

    #[cfg(feature = "image")]
    #[test]
    fn mlx_processor_consumes_retained_qwen_plan_after_sidecar_removal() {
        use eredu_core::{ArtifactFormat, ModelConfigurationResolver};

        let root = tempfile::tempdir().unwrap();
        let sidecar = root
            .path()
            .join(eredu_architectures::processor_plan::PROCESSOR_CONFIG_FILENAME);
        std::fs::write(
            &sidecar,
            br#"{
                "size":{"shortest_edge":16,"longest_edge":64},
                "patch_size":2,"temporal_patch_size":2,"merge_size":2,
                "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]
            }"#,
        )
        .unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_vl", "image_token_id": 61, "video_token_id": 62,
            "vision_start_token_id": 44, "vision_end_token_id": 45,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_vl_text", "hidden_size": 32,
                "num_hidden_layers": 3, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 0.000001, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [2, 1, 1], "mrope_interleaved": true}
            },
            "vision_config": {
                "depth": 4, "hidden_size": 16, "intermediate_size": 24,
                "num_heads": 4, "num_position_embeddings": 16, "in_channels": 3,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "deepstack_visual_indexes": [1, 3]
            }
        });
        let (configuration, resolved_plan) =
            eredu_architectures::configuration::MODEL_CONFIGURATIONS
                .resolve_safetensors(&config)
                .unwrap()
                .into_parts();
        let checkpoint = resolved_plan
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let catalog = eredu_core::checkpoint::TensorCatalog::new(
            checkpoint
                .common_tensors
                .iter()
                .chain(
                    checkpoint
                        .layout_groups
                        .iter()
                        .filter(|group| group.required)
                        .filter_map(|group| group.variants.first())
                        .flat_map(|variant| variant.tensors.iter()),
                )
                .filter(|tensor| {
                    tensor.requirement == eredu_checkpoint::schema::TensorRequirement::Required
                })
                .map(|tensor| eredu_core::checkpoint::TensorDescriptor {
                    name: tensor.key.clone(),
                    shape: tensor.shape.clone(),
                    dtype: eredu_core::checkpoint::TensorDtype::F32,
                    storage: None,
                }),
        )
        .unwrap();
        let architecture_plan = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &configuration,
                &catalog,
                None,
                resolved_plan,
            )
            .unwrap();
        std::fs::remove_file(sidecar).unwrap();

        assert_eq!(architecture_plan.model_kind(), super::ModelKind::Qwen3Vl);
        assert!(crate::composition::mlx::ModelProcessor::from_plan(&architecture_plan).is_some());
    }

    #[test]
    fn resolved_floating_dtype_selects_runtime_state_width() {
        for (dtype, bytes) in [
            (TensorDtype::F16, 2),
            (TensorDtype::Bf16, 2),
            (TensorDtype::F32, 4),
            (TensorDtype::F64, 8),
        ] {
            assert_eq!(mlx_floating_state_dtype_bytes(&dtype).unwrap().get(), bytes);
        }
    }

    #[test]
    fn dense_half_gguf_embeddings_select_two_byte_runtime_state() {
        for dtype in [GgmlType::F16, GgmlType::Bf16] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("model.gguf");
            write_minimal_llama_gguf(&path, dtype);
            let inspection = eredu_core::inspect_artifact(
                &path,
                &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
            )
            .unwrap();

            assert_eq!(
                inspected_floating_state_dtype_bytes(&inspection)
                    .unwrap()
                    .get(),
                2
            );
        }
    }

    #[test]
    fn packed_embedding_dtype_uses_known_mlx_materialization_width() {
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::Encoded("F8_E4M3".into()))
                .unwrap()
                .get(),
            4
        );
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::U32)
                .unwrap()
                .get(),
            4
        );
    }

    #[test]
    fn invalid_activation_dtype_does_not_silently_default() {
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::U8),
            Err(TensorDtype::U8)
        );
    }
}

fn prepared_model_kind(plan: &ArtifactArchitecturePlan) -> ModelKind {
    plan.model_kind()
}

pub(super) fn prepared_safetensors_architecture(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::SafetensorsArchitecturePlan, Error> {
    plan.safetensors_architecture().ok_or_else(|| {
        Error::ArchitectureModel(
            "SafeTensors preparation omitted its validated architecture plan".into(),
        )
    })
}

fn prepared_gguf_plan(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::GgufArchitecturePlan, Error> {
    plan.gguf_plan().ok_or_else(|| {
        Error::ArchitectureModel("GGUF preparation omitted its validated architecture plan".into())
    })
}

fn attach_optional_selected_processor(
    model: MlxModel,
    architecture_plan: &ArtifactArchitecturePlan,
    selected: Option<&eredu_runtime::SelectedProcessorExecution>,
) -> Result<MlxModel, Error> {
    match selected {
        Some(selected) => attach_selected_processor(model, architecture_plan, selected),
        None => Ok(model),
    }
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

fn complete_gguf_model(
    materialized: MaterializedGgufModel,
    floating_state_dtype_bytes: std::num::NonZeroU8,
    state_residency: CacheResidencyPolicy,
) -> MlxModel {
    let model = MlxModel::complete(
        materialized.model,
        floating_state_dtype_bytes,
        state_residency,
    );
    #[cfg(any(feature = "image", feature = "audio"))]
    let model = model.with_processor(materialized.processor);
    model
}

fn materialize_gguf_artifact(
    artifact: ModelArtifact,
    architecture_plan: ArtifactArchitecturePlan,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedGgufModel, Error> {
    let ModelArtifact::Gguf {
        path: _,
        configuration: _,
        validated,
        ..
    } = artifact
    else {
        return Err(Error::ArchitectureModel(
            "MLX GGUF materializer received a SafeTensors plan".into(),
        ));
    };
    let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
    let projector_plan = architecture_plan.gguf_media_projector().cloned();
    let (source, projector) =
        structural::AdmittedGguf::from_admission(architecture, projector_plan, validated)?;
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    validate_gguf_quantization_source(checkpoint, metadata, options.weight_quantization())?;
    #[cfg(any(feature = "image", feature = "audio"))]
    let processor = options
        .processor
        .as_ref()
        .map(|selected| ModelProcessor::from_selected(&architecture_plan, selected))
        .transpose()?
        .flatten();
    let model = materialize_gguf_model(
        &source,
        projector.as_ref(),
        family_route,
        options,
        stream,
        weights_stream,
    )?;
    Ok(MaterializedGgufModel {
        model,
        #[cfg(any(feature = "image", feature = "audio"))]
        processor,
    })
}

pub fn validate_gguf_quantization_source(
    source: &crate::backend::runtime::checkpoint::gguf::GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
) -> Result<(), Error> {
    let Some(quantization) = quantization else {
        return Ok(());
    };
    quantization.validate()?;

    let has_packed_companions = source
        .catalog()
        .tensors()
        .any(|tensor| tensor.affine().is_some());
    if has_packed_companions {
        return Err(Error::Quantization(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; packed GGUF tensors cannot be implicitly transcoded"
                .into(),
        ));
    }

    let file_type = metadata
        .get("general.file_type")
        .ok_or_else(|| {
            Error::Quantization(
                "GGUF general.file_type metadata is required to verify that load-time quantization is not transcoding packed weights"
                    .into(),
            )
        })?
        .as_i64()
        .ok_or_else(|| {
            Error::Quantization("GGUF general.file_type metadata must be an integer".into())
        })?;
    // llama.cpp's unquantized file types: ALL_F32, MOSTLY_F16, and MOSTLY_BF16.
    if !matches!(file_type, 0 | 1 | 32) {
        return Err(Error::Quantization(format!(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; general.file_type={file_type} is already quantized"
        )));
    }
    Ok(())
}

pub(super) fn materialize_safetensors(
    artifact: &super::artifact::PreparedSafetensorsArtifact,
    family_route: &ExcludedFamilyRoute,
    options: SelectedMlxConstruction,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Executable, Error> {
    let kind = artifact.architecture().model_kind();
    let quantization = options.weight_quantization();
    if let (Some(parameter_bank), Some(non_expert)) = (
        options.weight_residency.parameter_bank_cache(),
        options.weight_residency.ordinary_residency(),
    ) {
        let binding = AddressableParameterBankBinding::for_kind(kind)
            .ok_or_else(|| Error::ArchitectureModel(format!(
                "independent expert caching is unavailable for the normalized {} architecture on MLX",
                kind.canonical_name()
            )))?;
        return match binding {
            AddressableParameterBankBinding::KimiLinear => Ok(Executable::kimi_linear(
                kind,
                crate::composition::kimi_linear::load_kimi_linear_model(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::DeepSeek => Ok(Executable::deepseek(
                kind,
                Box::new(crate::composition::deepseek::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?),
            )?),
            AddressableParameterBankBinding::GptOss => Ok(Executable::gpt_oss(
                kind,
                crate::composition::gpt_oss::load_safetensors(
                    artifact,
                    options.weight_residency,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Gemma4 => Ok(Executable::gemma4(
                kind,
                crate::composition::gemma4::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Inkling => Ok(Executable::inkling(
                kind,
                crate::composition::inkling::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Lfm2 => Ok(Executable::lfm2(
                kind,
                crate::composition::lfm2::load_lfm2_model(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::MuseGlimmer => Ok(Executable::muse_glimmer(
                kind,
                crate::composition::muse_glimmer::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::NemotronH => Ok(Executable::nemotron_h(
                kind,
                crate::composition::nemotron_h::load_nemotron_h_model(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Qwen => Ok(Executable::qwen(
                kind,
                crate::composition::qwen::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Qwen3Next => Ok(Executable::qwen3_next(
                kind,
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
            AddressableParameterBankBinding::Qwen3VlMoe => Err(Error::ArchitectureModel(
                "replicated Qwen3-VL-MoE SafeTensors requires the architecture-selected composite materializer"
                    .into(),
            )),
            AddressableParameterBankBinding::Qwen35 => Ok(Executable::qwen35(
                kind,
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    artifact,
                    family_route,
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        non_expert,
                        parameter_bank,
                    ),
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )?),
        };
    }
    let execution = options.weight_residency.layers();
    if let Some(quantization) = quantization {
        quantization.validate()?;
    }
    let binding = FamilyBinding::for_kind(kind).ok_or_else(|| {
        Error::ArchitectureModel(format!(
            "{} bounded layer residency is selected through another loading protocol",
            kind.canonical_name()
        ))
    })?;
    match binding {
        FamilyBinding::DeepSeekV3 | FamilyBinding::DeepSeekV4 => Ok(Executable::deepseek(
            kind,
            Box::new(crate::composition::deepseek::load_safetensors(
                artifact,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?),
        )?),
        FamilyBinding::Gemma4 => Ok(Executable::gemma4(
            kind,
            crate::composition::gemma4::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Inkling => Ok(Executable::inkling(
            kind,
            crate::composition::inkling::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::KimiLinear => Ok(Executable::kimi_linear(kind, {
            crate::composition::kimi_linear::load_kimi_linear_model(
                artifact,
                family_route,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::Llama => unreachable!(
            "architecture-owned eligibility rejects replicated SafeTensors before family binding"
        ),
        FamilyBinding::MuseGlimmer => Ok(Executable::muse_glimmer(
            kind,
            crate::composition::muse_glimmer::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Qwen => Ok(Executable::qwen(
            kind,
            crate::composition::qwen::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::GptOss => Ok(Executable::gpt_oss(
            kind,
            crate::composition::gpt_oss::load_safetensors(
                artifact,
                options.weight_residency,
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
        FamilyBinding::Lfm2 => Ok(Executable::lfm2(kind, {
            crate::composition::lfm2::load_lfm2_model(
                artifact,
                family_route,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::NemotronH => Ok(Executable::nemotron_h(kind, {
            crate::composition::nemotron_h::load_nemotron_h_model(
                artifact,
                family_route,
                eredu_runtime::WeightResidency::with_layers(execution),
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::Qwen3Next => Ok(Executable::qwen3_next(kind, {
            crate::composition::qwen::hybrid::load_safetensors(
                artifact,
                family_route,
                execution,
                quantization,
                stream,
                weights_stream,
            )?
        })?),
        FamilyBinding::Qwen3Vl | FamilyBinding::Qwen3VlMoe => {
            Err(Error::ArchitectureModel(
                "replicated Qwen3-VL SafeTensors requires the architecture-selected composite materializer"
                    .into(),
            ))
        }
        FamilyBinding::Qwen35 => Ok(Executable::qwen35(
            kind,
            crate::composition::qwen::hybrid::load_safetensors(
                artifact,
                family_route,
                execution,
                quantization,
                stream,
                weights_stream,
            )?,
        )?),
    }
}
