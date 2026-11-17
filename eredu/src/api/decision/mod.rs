//! Finite, non-autoregressive decisions.

pub use eredu_core::decision::{
    BoxDetection, ChoiceAnswer, ChoiceEvaluation, ChoiceOption, ChoiceProbability, ChoiceQuestion,
    DecisionAnswer, DecisionEvaluation, DecisionQuestion, ImageBox, SpanExtraction, TextSpan,
};

mod refinement;
pub use refinement::evaluate_refinement;

/// One question and its caller-assigned identifier.
#[derive(Debug, Clone)]
pub struct DecisionItem {
    /// Identifier echoed in the response.
    pub id: String,
    /// Typed readout and written instructions.
    pub question: DecisionQuestion,
    /// Spoken instructions that replace the written instructions.
    pub spoken: Option<eredu_core::Audio>,
}

/// Decoded input for a finite decision operation.
#[derive(Debug, Clone, Default)]
pub struct DecisionRequest {
    /// Shared textual evidence.
    pub state: Option<String>,
    /// Shared media in input order.
    pub media: Vec<eredu_core::Media>,
    /// Questions in answer order.
    pub questions: Vec<DecisionItem>,
    /// Optional denoising steps for refinement models.
    pub refinement_steps: Option<u32>,
    /// Optional refinement seed.
    pub seed: Option<u64>,
}

impl DecisionRequest {
    fn validate(&self) -> anyhow::Result<()> {
        let mut ids = std::collections::BTreeSet::new();
        anyhow::ensure!(
            !self.questions.is_empty(),
            "at least one question is required"
        );
        anyhow::ensure!(
            self.questions
                .iter()
                .all(|q| !q.id.trim().is_empty() && ids.insert(&q.id)),
            "question IDs must be nonempty and unique"
        );
        Ok(())
    }
}

/// One typed result in request order.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecisionResult {
    /// Identifier from the request.
    pub id: String,
    /// Typed answer.
    pub answer: DecisionAnswer,
    /// Full-vocabulary mass of the permitted labels, when observed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_mass: Option<f64>,
}

/// Answers, token usage and optional calibration provenance.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecisionResponse {
    /// Answers in caller order.
    pub answers: Vec<DecisionResult>,
    /// Number of input tokens.
    pub input_tokens: usize,
    /// Number of generated tokens.
    pub generated_tokens: usize,
    /// SHA-256 of the calibration asset, encoded as hexadecimal.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_sha256: Option<String>,
}

mod task;
pub use eredu_architectures::decision_task::DecisionBackendFactory;
pub use task::evaluate_decisions;
