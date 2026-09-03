//! Mechanism-only selection contracts for composite model input.

use std::collections::BTreeSet;

use eredu_core::InputModality;

use crate::{
    select_replicated_text_realization, BackendMechanismCapabilities, ReplicatedTextRequirements,
    ReplicatedTextSelectionError, ReplicatedTextSelectionRequest,
    SelectedReplicatedTextRealization,
};

/// One generic host-media or tensor operation required by neutral preparation.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ProcessorPrimitive {
    /// Bicubic RGB image resizing.
    RgbResizeBicubic,
    /// Three-lobe Lanczos RGB image resizing.
    RgbResizeLanczos3,
    /// Channel-wise rescaling and normalization.
    RgbNormalize,
    /// Deterministic decoded-video frame selection and timestamps.
    VideoSampling,
    /// Windowed mono PCM framing.
    AudioWindow,
    /// Real-valued spectrum calculation.
    AudioSpectrum,
    /// Mel filter-bank projection.
    AudioMelFilter,
    /// Numeric logarithm over audio features.
    AudioLogarithm,
    /// Unsigned token tensor construction.
    TensorU32,
    /// Floating-point tensor construction.
    TensorF32,
    /// Signed metadata tensor construction.
    TensorI32,
    /// Boolean metadata tensor construction.
    TensorBool,
    /// Tensor padding.
    Padding,
    /// Tensor concatenation.
    Concatenation,
    /// Tensor slicing.
    Slicing,
    /// Indexed tensor movement.
    Indexing,
    /// Boolean or causal mask construction.
    MaskConstruction,
    /// Scatter or ordered merge into decoder positions.
    Merge,
    /// Native encoder module execution.
    Encoder,
    /// Native projector module execution.
    Projector,
    /// Exact small integer and Boolean metadata evaluation.
    MetadataInspection,
}

/// Architecture requirements for one admitted input modality.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModalityProcessorRequirements {
    modality: InputModality,
    raw_primitives: BTreeSet<ProcessorPrimitive>,
    prepared_tensor: bool,
    projected_embeddings: bool,
    maximum_dimension: u64,
}

impl ModalityProcessorRequirements {
    /// Creates an exact modality requirement.
    pub fn new(
        modality: InputModality,
        raw_primitives: impl IntoIterator<Item = ProcessorPrimitive>,
        prepared_tensor: bool,
        projected_embeddings: bool,
        maximum_dimension: u64,
    ) -> Result<Self, ProcessorSelectionError> {
        if maximum_dimension == 0 {
            return Err(ProcessorSelectionError::new([
                "architecture declared a zero native dimension bound".into(),
            ]));
        }
        Ok(Self {
            modality,
            raw_primitives: raw_primitives.into_iter().collect(),
            prepared_tensor,
            projected_embeddings,
            maximum_dimension,
        })
    }

    /// Semantic modality.
    pub const fn modality(&self) -> InputModality {
        self.modality
    }

    /// Generic operations needed when raw decoded media is requested.
    pub const fn raw_primitives(&self) -> &BTreeSet<ProcessorPrimitive> {
        &self.raw_primitives
    }

    /// Whether architecture admission accepts a native prepared tensor.
    pub const fn prepared_tensor(&self) -> bool {
        self.prepared_tensor
    }

    /// Whether architecture admission accepts decoder-width embeddings.
    pub const fn projected_embeddings(&self) -> bool {
        self.projected_embeddings
    }

    /// Largest architecture-declared dimension that mechanisms must represent.
    pub const fn maximum_dimension(&self) -> u64 {
        self.maximum_dimension
    }
}

/// Complete processor and prepared-input requirements of one architecture.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ProcessorExecutionRequirements {
    modalities: Vec<ModalityProcessorRequirements>,
}

impl ProcessorExecutionRequirements {
    /// Creates requirements with unique modality entries.
    pub fn new(
        modalities: impl IntoIterator<Item = ModalityProcessorRequirements>,
    ) -> Result<Self, ProcessorSelectionError> {
        let mut modalities = modalities.into_iter().collect::<Vec<_>>();
        modalities.sort_by_key(|requirement| modality_order(requirement.modality));
        if modalities
            .windows(2)
            .any(|pair| pair[0].modality == pair[1].modality)
        {
            return Err(ProcessorSelectionError::new([
                "architecture repeats an input modality requirement".into(),
            ]));
        }
        if modalities.is_empty() {
            return Err(ProcessorSelectionError::new([
                "architecture declares no input modalities".into(),
            ]));
        }
        Ok(Self { modalities })
    }

    /// Requirements in stable modality order.
    pub fn modalities(&self) -> &[ModalityProcessorRequirements] {
        &self.modalities
    }

    /// Looks up one admitted modality.
    pub fn modality(&self, modality: InputModality) -> Option<&ModalityProcessorRequirements> {
        self.modalities
            .iter()
            .find(|requirement| requirement.modality == modality)
    }
}

/// Caller policy for guaranteed input readiness.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ProcessorSelectionRequest {
    modalities: BTreeSet<InputModality>,
    raw_media: bool,
    available_raw_media: bool,
    prepared_tensors: bool,
    projected_modalities: BTreeSet<InputModality>,
}

impl ProcessorSelectionRequest {
    /// Requests readiness for the supplied semantic modalities.
    pub fn new(modalities: impl IntoIterator<Item = InputModality>) -> Self {
        Self {
            modalities: modalities.into_iter().collect(),
            raw_media: false,
            available_raw_media: false,
            prepared_tensors: true,
            projected_modalities: BTreeSet::new(),
        }
    }

    /// Requires raw decoded-media preparation for requested non-text modalities.
    pub const fn with_raw_media(mut self, required: bool) -> Self {
        self.raw_media = required;
        self
    }

    /// Selects raw preparation when every required mechanism is available.
    pub const fn with_available_raw_media(mut self, enabled: bool) -> Self {
        self.available_raw_media = enabled;
        self
    }

    /// Requires native prepared-tensor acceptance.
    pub const fn with_prepared_tensors(mut self, required: bool) -> Self {
        self.prepared_tensors = required;
        self
    }

    /// Requires decoder-width projected-embedding acceptance.
    pub fn with_projected_embeddings(mut self, required: bool) -> Self {
        if required {
            self.projected_modalities = self.modalities.clone();
        } else {
            self.projected_modalities.clear();
        }
        self
    }

    /// Requires projected-embedding readiness for only the supplied modalities.
    pub fn with_projected_modalities(
        mut self,
        modalities: impl IntoIterator<Item = InputModality>,
    ) -> Self {
        self.projected_modalities = modalities.into_iter().collect();
        self
    }

    /// Requested semantic modalities.
    pub const fn modalities(&self) -> &BTreeSet<InputModality> {
        &self.modalities
    }
}

/// Generic mechanisms and native bounds implemented by one backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MediaPrimitiveCapabilities {
    raw_modalities: BTreeSet<InputModality>,
    prepared_modalities: BTreeSet<InputModality>,
    projected_modalities: BTreeSet<InputModality>,
    primitives: BTreeSet<ProcessorPrimitive>,
    maximum_dimension: u64,
}

impl MediaPrimitiveCapabilities {
    /// Creates a fail-closed mechanism report.
    pub fn new(
        raw_modalities: impl IntoIterator<Item = InputModality>,
        prepared_modalities: impl IntoIterator<Item = InputModality>,
        projected_modalities: impl IntoIterator<Item = InputModality>,
        primitives: impl IntoIterator<Item = ProcessorPrimitive>,
        maximum_dimension: u64,
    ) -> Self {
        Self {
            raw_modalities: raw_modalities.into_iter().collect(),
            prepared_modalities: prepared_modalities.into_iter().collect(),
            projected_modalities: projected_modalities.into_iter().collect(),
            primitives: primitives.into_iter().collect(),
            maximum_dimension,
        }
    }

    /// Operations implemented with their exact neutral semantics.
    pub const fn primitives(&self) -> &BTreeSet<ProcessorPrimitive> {
        &self.primitives
    }
}

/// Authoritative processor realization selected before construction or payload work.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedProcessorExecution {
    requirements: ProcessorExecutionRequirements,
    modalities: BTreeSet<InputModality>,
    raw_media: bool,
    prepared_tensors: bool,
    projected_modalities: BTreeSet<InputModality>,
}

/// Authoritative combination of text/session and composite-input selection.
#[derive(Debug, Clone)]
pub struct SelectedCompositeRealization {
    execution: SelectedReplicatedTextRealization,
    processor: SelectedProcessorExecution,
}

impl SelectedCompositeRealization {
    /// Combines independently selected execution and processor proofs.
    pub fn from_parts(
        execution: SelectedReplicatedTextRealization,
        processor: SelectedProcessorExecution,
    ) -> Self {
        Self {
            execution,
            processor,
        }
    }

    /// Exact selected graph, parameters, state, residency, and session mechanisms.
    pub const fn execution(&self) -> &SelectedReplicatedTextRealization {
        &self.execution
    }

    /// Exact selected input readiness and processor mechanisms.
    pub const fn processor(&self) -> &SelectedProcessorExecution {
        &self.processor
    }

    /// Consumes the combined proof into its two independently selected values.
    pub fn into_parts(
        self,
    ) -> (
        SelectedReplicatedTextRealization,
        SelectedProcessorExecution,
    ) {
        (self.execution, self.processor)
    }
}

/// Fail-closed error from combined replicated composite selection.
#[derive(Debug, thiserror::Error)]
pub enum CompositeSelectionError {
    /// Graph, parameter, storage, state, or session selection failed.
    #[error(transparent)]
    Execution(#[from] ReplicatedTextSelectionError),
    /// Input representation or processor mechanism selection failed.
    #[error(transparent)]
    Processor(#[from] ProcessorSelectionError),
}

/// Selects the whole composite realization before architecture construction or payload access.
pub fn select_composite_realization(
    execution_requirements: &ReplicatedTextRequirements,
    processor_requirements: &ProcessorExecutionRequirements,
    execution_request: &ReplicatedTextSelectionRequest,
    processor_request: &ProcessorSelectionRequest,
    execution_capabilities: &BackendMechanismCapabilities,
    processor_capabilities: &MediaPrimitiveCapabilities,
) -> Result<SelectedCompositeRealization, CompositeSelectionError> {
    let execution = select_replicated_text_realization(
        execution_requirements,
        execution_request,
        execution_capabilities,
    )?;
    let processor = select_processor_execution(
        processor_requirements,
        processor_request,
        processor_capabilities,
    )?;
    Ok(SelectedCompositeRealization::from_parts(
        execution, processor,
    ))
}

impl SelectedProcessorExecution {
    /// Exact architecture requirements retained by selection.
    pub const fn requirements(&self) -> &ProcessorExecutionRequirements {
        &self.requirements
    }

    /// Caller-required modalities admitted by both architecture and mechanisms.
    pub const fn modalities(&self) -> &BTreeSet<InputModality> {
        &self.modalities
    }

    /// Whether raw decoded-media preparation was selected.
    pub const fn raw_media(&self) -> bool {
        self.raw_media
    }

    /// Whether native prepared tensors were selected.
    pub const fn prepared_tensors(&self) -> bool {
        self.prepared_tensors
    }

    /// Modalities for which projected embeddings were selected.
    pub const fn projected_modalities(&self) -> &BTreeSet<InputModality> {
        &self.projected_modalities
    }
}

/// Complete fail-closed processor-selection diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("composite input realization is unsupported: {issues}", issues = .issues.join("; "))]
pub struct ProcessorSelectionError {
    issues: Vec<String>,
}

impl ProcessorSelectionError {
    fn new(issues: impl IntoIterator<Item = String>) -> Self {
        Self {
            issues: issues.into_iter().collect(),
        }
    }

    /// Every missing semantic or mechanism requirement in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

/// Selects input preparation using architecture facts, caller policy, and mechanisms only.
pub fn select_processor_execution(
    requirements: &ProcessorExecutionRequirements,
    request: &ProcessorSelectionRequest,
    capabilities: &MediaPrimitiveCapabilities,
) -> Result<SelectedProcessorExecution, ProcessorSelectionError> {
    let mut issues = Vec::new();
    let mut available_raw_media = request.available_raw_media
        && request
            .modalities
            .iter()
            .any(|modality| *modality != InputModality::Text);
    for modality in &request.modalities {
        let Some(requirement) = requirements.modality(*modality) else {
            issues.push(format!("architecture input modality {}", modality.as_str()));
            continue;
        };
        if requirement.maximum_dimension > capabilities.maximum_dimension {
            issues.push(format!(
                "{} native dimension {}",
                modality.as_str(),
                requirement.maximum_dimension
            ));
        }
        if request.prepared_tensors
            && (!requirement.prepared_tensor
                || !capabilities.prepared_modalities.contains(modality))
        {
            issues.push(format!("{} prepared tensors", modality.as_str()));
        }
        if request.projected_modalities.contains(modality)
            && (!requirement.projected_embeddings
                || !capabilities.projected_modalities.contains(modality))
        {
            issues.push(format!("{} projected embeddings", modality.as_str()));
        }
        if (request.raw_media || request.available_raw_media) && *modality != InputModality::Text {
            let mut raw_issues = Vec::new();
            if requirement.raw_primitives.is_empty()
                || !capabilities.raw_modalities.contains(modality)
            {
                raw_issues.push(format!("{} raw media", modality.as_str()));
            }
            for primitive in requirement
                .raw_primitives
                .difference(&capabilities.primitives)
            {
                raw_issues.push(format!("{} primitive {primitive:?}", modality.as_str()));
            }
            if raw_issues.is_empty() {
                continue;
            }
            available_raw_media = false;
            if request.raw_media {
                issues.extend(raw_issues);
            }
        }
    }
    if issues.is_empty() {
        Ok(SelectedProcessorExecution {
            requirements: requirements.clone(),
            modalities: request.modalities.clone(),
            raw_media: request.raw_media || available_raw_media,
            prepared_tensors: request.prepared_tensors,
            projected_modalities: request.projected_modalities.clone(),
        })
    } else {
        Err(ProcessorSelectionError::new(issues))
    }
}

fn modality_order(modality: InputModality) -> u8 {
    match modality {
        InputModality::Text => 0,
        InputModality::Image => 1,
        InputModality::Video => 2,
        InputModality::Audio => 3,
        _ => u8::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirements() -> ProcessorExecutionRequirements {
        ProcessorExecutionRequirements::new([
            ModalityProcessorRequirements::new(
                InputModality::Text,
                [ProcessorPrimitive::TensorU32],
                true,
                true,
                64,
            )
            .unwrap(),
            ModalityProcessorRequirements::new(
                InputModality::Image,
                [
                    ProcessorPrimitive::RgbResizeBicubic,
                    ProcessorPrimitive::RgbNormalize,
                    ProcessorPrimitive::TensorF32,
                    ProcessorPrimitive::TensorI32,
                ],
                true,
                true,
                1024,
            )
            .unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn selection_denies_each_missing_mechanism_without_callbacks() {
        let request = ProcessorSelectionRequest::new([InputModality::Image])
            .with_raw_media(true)
            .with_projected_embeddings(true);
        let capabilities = MediaPrimitiveCapabilities::new(
            [InputModality::Image],
            [InputModality::Image],
            [InputModality::Image],
            [
                ProcessorPrimitive::RgbResizeBicubic,
                ProcessorPrimitive::RgbNormalize,
                ProcessorPrimitive::TensorF32,
            ],
            512,
        );

        let error = select_processor_execution(&requirements(), &request, &capabilities)
            .expect_err("missing metadata construction and native extent must fail");
        assert_eq!(
            error.issues(),
            ["image native dimension 1024", "image primitive TensorI32",]
        );
    }

    #[test]
    fn text_only_selection_does_not_require_unrequested_image_primitives() {
        let request = ProcessorSelectionRequest::new([InputModality::Text]);
        let capabilities = MediaPrimitiveCapabilities::new([], [InputModality::Text], [], [], 64);
        let selected =
            select_processor_execution(&requirements(), &request, &capabilities).unwrap();
        assert_eq!(
            selected.modalities(),
            &BTreeSet::from([InputModality::Text])
        );
        assert!(!selected.raw_media());
    }

    #[test]
    fn projected_readiness_requires_architecture_admission_and_backend_support() {
        let requirements =
            ProcessorExecutionRequirements::new([ModalityProcessorRequirements::new(
                InputModality::Audio,
                [],
                true,
                false,
                64,
            )
            .unwrap()])
            .unwrap();
        let request =
            ProcessorSelectionRequest::new([InputModality::Audio]).with_projected_embeddings(true);
        let capabilities = MediaPrimitiveCapabilities::new(
            [],
            [InputModality::Audio],
            [InputModality::Audio],
            [],
            64,
        );
        let error = select_processor_execution(&requirements, &request, &capabilities)
            .expect_err("architecture-ineligible projected audio was admitted");
        assert_eq!(error.issues(), ["audio projected embeddings"]);
    }

    #[test]
    fn optional_raw_readiness_never_overstates_selected_mechanisms() {
        let request = ProcessorSelectionRequest::new([InputModality::Image])
            .with_available_raw_media(true)
            .with_projected_embeddings(true);
        let incomplete = MediaPrimitiveCapabilities::new(
            [InputModality::Image],
            [InputModality::Image],
            [InputModality::Image],
            [
                ProcessorPrimitive::RgbResizeBicubic,
                ProcessorPrimitive::RgbNormalize,
                ProcessorPrimitive::TensorF32,
            ],
            1024,
        );
        let selected = select_processor_execution(&requirements(), &request, &incomplete).unwrap();
        assert!(!selected.raw_media());
        assert!(selected.prepared_tensors());
        assert_eq!(
            selected.projected_modalities(),
            &BTreeSet::from([InputModality::Image])
        );

        let complete = MediaPrimitiveCapabilities::new(
            [InputModality::Image],
            [InputModality::Image],
            [InputModality::Image],
            [
                ProcessorPrimitive::RgbResizeBicubic,
                ProcessorPrimitive::RgbNormalize,
                ProcessorPrimitive::TensorF32,
                ProcessorPrimitive::TensorI32,
            ],
            1024,
        );
        let selected = select_processor_execution(&requirements(), &request, &complete).unwrap();
        assert!(selected.raw_media());
        assert!(selected.prepared_tensors());

        let required = ProcessorSelectionRequest::new([InputModality::Image])
            .with_raw_media(true)
            .with_projected_embeddings(true);
        let error = select_processor_execution(&requirements(), &required, &incomplete)
            .expect_err("required raw readiness was admitted without every primitive");
        assert_eq!(error.issues(), ["image primitive TensorI32"]);
    }
}
