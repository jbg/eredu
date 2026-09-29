//! Portable input representation intent, independent of weight residency.
use crate::ProcessorSelectionRequest;
use eredu_core::InputModality;

/// Caller intent for preparing conditional input execution.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum MediaLoadRequest {
    /// Use the architecture's ordinary input preparation for retained artifacts.
    #[default]
    ArchitectureDefault,
    /// Prepare text-only execution, without an encoder or media processor.
    Disabled,
    /// Require the exact input representations supplied here.
    Required(MediaExecutionPolicy),
}

/// Input-readiness requirements with optional coarse host-processing limits.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MediaExecutionPolicy {
    processor: ProcessorSelectionRequest,
    processor_budget: Option<crate::processor_resources::ProcessorRequestBudget>,
}

/// Invalid portable media intent before architecture or backend selection.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MediaExecutionPolicyError {
    /// Required media execution must request at least one non-text modality.
    #[error("required media execution must request at least one non-text modality")]
    NoMediaModalities,
}
impl MediaExecutionPolicy {
    /// Retains exact modality and input representation intent.
    pub fn new(processor: ProcessorSelectionRequest) -> Result<Self, MediaExecutionPolicyError> {
        if !processor
            .modalities()
            .iter()
            .any(|modality| *modality != InputModality::Text)
        {
            return Err(MediaExecutionPolicyError::NoMediaModalities);
        }
        Ok(Self {
            processor,
            processor_budget: None,
        })
    }
    /// Adds optional caller limits for host media processing. Ordinary media
    /// preparation does not require an accounting policy.
    pub fn with_processor_budget(
        mut self,
        budget: crate::processor_resources::ProcessorRequestBudget,
    ) -> Self {
        self.processor_budget = Some(budget);
        self
    }
    /// Optional caller limits for raw input processing.
    pub const fn processor_budget(
        &self,
    ) -> Option<crate::processor_resources::ProcessorRequestBudget> {
        self.processor_budget
    }
    /// Exact caller modality and representation requirements.
    pub const fn processor(&self) -> &ProcessorSelectionRequest {
        &self.processor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NormalizedLoadRequest;
    fn policy() -> MediaExecutionPolicy {
        MediaExecutionPolicy::new(
            ProcessorSelectionRequest::new([InputModality::Image])
                .with_raw_media(true)
                .with_projected_embeddings(true),
        )
        .unwrap()
    }
    #[test]
    fn normalized_media_intent_preserves_validation_and_representation_identity() {
        let policy = policy();
        let default = NormalizedLoadRequest::default();
        assert_eq!(
            default.media_execution(),
            &MediaLoadRequest::ArchitectureDefault
        );
        let disabled = default
            .clone()
            .with_media_execution(MediaLoadRequest::Disabled);
        let request = default
            .clone()
            .with_media_execution(MediaLoadRequest::Required(policy.clone()));
        assert_ne!(default, disabled);
        assert_ne!(disabled, request);
        assert_ne!(default, request);
        assert_eq!(
            request.validate_model_preparation().unwrap().request(),
            &request
        );
        for processor in [
            policy.processor.clone().with_raw_media(false),
            policy.processor.clone().with_prepared_tensors(false),
            policy.processor.clone().with_projected_embeddings(false),
            policy.processor.clone().with_available_raw_media(true),
            policy
                .processor
                .clone()
                .with_additional_modalities([InputModality::Video]),
            policy
                .processor
                .clone()
                .with_projected_modalities([InputModality::Video]),
        ] {
            let changed = MediaExecutionPolicy::new(processor).unwrap();
            assert_ne!(
                request,
                default
                    .clone()
                    .with_media_execution(MediaLoadRequest::Required(changed))
            );
        }
        let reordered =
            ProcessorSelectionRequest::new([InputModality::Image, InputModality::Image])
                .with_raw_media(true)
                .with_projected_embeddings(true);
        assert_eq!(policy.processor(), &reordered);
    }
    #[test]
    fn media_policy_requires_nontext_intent_without_requiring_resource_budgets() {
        for modalities in [vec![], vec![InputModality::Text]] {
            assert_eq!(
                MediaExecutionPolicy::new(ProcessorSelectionRequest::new(modalities)),
                Err(MediaExecutionPolicyError::NoMediaModalities)
            );
        }
        for modality in [
            InputModality::Image,
            InputModality::Video,
            InputModality::Audio,
        ] {
            assert_eq!(
                MediaExecutionPolicy::new(ProcessorSelectionRequest::new([modality]))
                    .unwrap()
                    .processor_budget(),
                None
            );
        }
    }
    #[test]
    fn optional_processor_limits_remain_part_of_request_identity() {
        let budget = crate::processor_resources::ProcessorRequestBudget {
            decoded_input_bytes: 1024,
            host_buffer_bytes: 4096,
            output_tensor_bytes: 512,
            planning_items: 100,
            decoder_positions: 64,
        };
        let retained = policy().with_processor_budget(budget);
        assert_eq!(retained.processor_budget(), Some(budget));
        assert_ne!(policy(), retained);
        for field in 0..5 {
            let mut changed = budget;
            match field {
                0 => changed.decoded_input_bytes += 1,
                1 => changed.host_buffer_bytes += 1,
                2 => changed.output_tensor_bytes += 1,
                3 => changed.planning_items += 1,
                _ => changed.decoder_positions += 1,
            }
            assert_ne!(retained, policy().with_processor_budget(changed));
        }
    }
}
