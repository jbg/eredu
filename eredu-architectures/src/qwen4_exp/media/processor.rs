//! Retained processor semantics and authoritative input representation selection.
use super::{AdmittedMediaInput, MediaAdmissionConfig, MediaIngress, MediaInputError};
use crate::processor_execution::{
    PreparedProcessor, ProcessorExecutionError, ProcessorInputAdmission, ProcessorMechanisms,
};
use eredu_core::{InputModality as Modality, InputPayloadKind, TokenizedMultimodalRequest};
use eredu_runtime::{
    ModalityProcessorRequirements, PreparedInputInspector, PreparedModelInput,
    ProcessorExecutionRequirements, ProcessorPrimitive as Primitive, SelectedProcessorExecution,
};

/// Source-independent selected preparation; owns no checkpoint or native context.
#[derive(Clone)]
pub struct SelectedMediaProcessor {
    selected: SelectedProcessorExecution,
    processor: PreparedProcessor,
    admission: MediaAdmissionConfig,
}
impl ProcessorInputAdmission for MediaAdmissionConfig {
    fn validate<T>(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<(), crate::processor_execution::ProcessorInputAdmissionError> {
        self.admit(input, inspector).map(|_| ()).map_err(|error| {
            eredu_core::CapabilityError::UnsupportedInput {
                architecture: "qwen4_exp".into(),
                reason: error.to_string(),
            }
            .into()
        })
    }
}
impl MediaAdmissionConfig {
    /// Exact raw/prepared/projected requirements derived from retained processor policy.
    pub fn processor_requirements(
        &self,
        processor: Option<&crate::processor_plan::QwenProcessorPlan>,
    ) -> Result<ProcessorExecutionRequirements, eredu_runtime::ProcessorSelectionError> {
        let raw: Vec<_> = processor
            .into_iter()
            .flat_map(|p| p.raw_modalities())
            .collect();
        ProcessorExecutionRequirements::new(
            [Modality::Text, Modality::Image, Modality::Video]
                .into_iter()
                .map(|modality| {
                    let primitives = match modality {
                        Modality::Text => vec![Primitive::TensorU32],
                        Modality::Image | Modality::Video if raw.contains(&modality) => {
                            let mut primitives = vec![
                                Primitive::RgbResizeBicubic,
                                Primitive::RgbNormalize,
                                Primitive::TensorF32,
                                Primitive::TensorI32,
                                Primitive::TensorU32,
                            ];
                            if modality == Modality::Video {
                                primitives.push(Primitive::VideoSampling);
                            }
                            primitives
                        }
                        _ => vec![],
                    };
                    ModalityProcessorRequirements::new(
                        modality,
                        primitives,
                        true,
                        true,
                        i32::MAX as u64,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
    /// Selects input representations using only retained policy and mechanism facts.
    pub fn select_processor(
        &self,
        processor: Option<&crate::processor_plan::QwenProcessorPlan>,
        request: &eredu_runtime::ProcessorSelectionRequest,
        mechanisms: &eredu_runtime::MediaPrimitiveCapabilities,
    ) -> Result<SelectedMediaProcessor, eredu_runtime::ProcessorSelectionError> {
        // Every raw media product contains decoder framing text, even for an image-only request.
        let request = request.clone().with_additional_modalities([Modality::Text]);
        let selected = eredu_runtime::select_processor_execution(
            &self.processor_requirements(processor)?,
            &request,
            mechanisms,
        )?;
        Ok(SelectedMediaProcessor {
            selected,
            processor: PreparedProcessor::from_qwen(
                processor
                    .cloned()
                    .unwrap_or_else(crate::processor_plan::QwenProcessorPlan::tokens_only),
            ),
            admission: self.clone(),
        })
    }
}
impl MediaIngress {
    /// Exact raw/prepared/projected requirements from the retained source-free policy.
    pub fn processor_requirements(
        &self,
    ) -> Result<ProcessorExecutionRequirements, eredu_runtime::ProcessorSelectionError> {
        self.admission_config()
            .processor_requirements(self.vision().processor())
    }

    pub(crate) fn select_processor(
        &self,
        request: &eredu_runtime::ProcessorSelectionRequest,
        mechanisms: &eredu_runtime::MediaPrimitiveCapabilities,
    ) -> Result<SelectedMediaProcessor, eredu_runtime::ProcessorSelectionError> {
        self.admission_config()
            .select_processor(self.vision().processor(), request, mechanisms)
    }
}
impl SelectedMediaProcessor {
    /// Exact cold-selected input readiness.
    pub fn realization(&self) -> &SelectedProcessorExecution {
        &self.selected
    }

    /// Explicitly applies coarse caller limits to host media processing, then
    /// validates prepared geometry before native lowering.
    pub fn prepare<M: ProcessorMechanisms, E: std::fmt::Display>(
        &self,
        request: &TokenizedMultimodalRequest,
        mechanisms: &mut M,
        budget: eredu_runtime::processor_resources::ProcessorRequestBudget,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<
        crate::processor_execution::BudgetedProcessorInput<M::Tensor>,
        ProcessorExecutionError<E, M::Error>,
    > {
        self.prepared_processor(Some(budget))
            .prepare_budgeted(request, mechanisms, encode_text)
    }

    /// Retains selected readiness, architecture admission and optional coarse limits
    /// in the processor used by ordinary prepared-execution construction.
    pub fn prepared_processor(
        &self,
        budget: Option<eredu_runtime::processor_resources::ProcessorRequestBudget>,
    ) -> PreparedProcessor {
        let budget = budget.map(|mut budget| {
            budget.decoder_positions = budget
                .decoder_positions
                .min(self.admission.max_tokens as u64);
            budget
        });
        self.processor.clone().with_media_authority(
            self.selected.clone(),
            self.admission.clone(),
            budget,
        )
    }

    /// Admits only selected representations, preserving mandatory original token IDs.
    pub fn admit<'a, T>(
        &self,
        input: &'a PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<AdmittedMediaInput<'a, T>, MediaInputError> {
        for part in input.identity().parts() {
            if !self.selected.modalities().contains(&part.modality()) {
                return Err(MediaInputError::Geometry("input modality was not selected"));
            }
            let allowed = match part.payload_kind() {
                InputPayloadKind::Embeddings => self
                    .selected
                    .projected_modalities()
                    .contains(&part.modality()),
                _ => self.selected.prepared_tensors() || self.selected.raw_media(),
            };
            if !allowed {
                return Err(MediaInputError::Geometry(
                    "input representation was not selected",
                ));
            }
        }
        self.admission.admit(input, inspector)
    }
}
