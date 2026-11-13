//! Finite, non-autoregressive decisions.

pub use eredu_architectures::lfm3::decision::{DecisionError, PreparedTextChoiceModel};
pub use eredu_core::decision::{
    BoxDetection, ChoiceAnswer, ChoiceEvaluation, ChoiceOption, ChoiceProbability, ChoiceQuestion,
    DecisionAnswer, DecisionEvaluation, DecisionQuestion, ImageBox, SpanExtraction, TextSpan,
};

/// Placement and per-component residency for a distributed decision model.
#[derive(Debug, Clone, Copy)]
pub struct DecisionDistribution {
    /// Tensor and pipeline topology of the Ring world.
    pub topology: eredu_core::ParallelTopology,
    /// Maximum positions in one complete packed question row.
    pub maximum_positions: std::num::NonZeroUsize,
    /// Residency of each rank's decoder parameters.
    pub decoder_residency: eredu_runtime::LayerWeightResidency,
    /// Residency of the vision replica, when retained.
    pub vision_residency: eredu_runtime::LayerWeightResidency,
    /// Residency of the speech input replica, when retained.
    pub speech_residency: eredu_runtime::LayerWeightResidency,
}

#[cfg(feature = "mlx")]
mod mlx;
#[cfg(feature = "mlx")]
pub use mlx::{ChoiceSession, DecisionExecutionError, DecisionSession, TextChoiceModel};
