//! Architecture-owned requirements for optional backend forward operators.

use eredu_nn::{Error, NeuralBackend, NeuralOperatorCapabilities as C};

/// Kimi Linear KDA execution requirements.
pub const KIMI_LINEAR: C = C::SIGMOID
    .union(C::SOFTPLUS)
    .union(C::EXP)
    .union(C::RMS_NORM_WITHOUT_WEIGHT)
    .union(C::GATED_DELTA_SCAN);

/// Qwen3-Next and Qwen3.5 hybrid execution requirements.
pub const QWEN_HYBRID: C = C::SIGMOID
    .union(C::SOFTPLUS)
    .union(C::EXP)
    .union(C::L2_NORMALIZE)
    .union(C::SILU_GATED_GROUP_RMS_NORM)
    .union(C::GATED_DELTA_SCAN);

/// Nemotron-H Mamba execution requirements.
pub const NEMOTRON_H: C = C::GATED_GROUP_RMS_NORM.union(C::SELECTIVE_STATE_SPACE_SCAN);

/// Qwen vision encoder execution requirements.
pub const QWEN_VISION: C = C::GELU_APPROXIMATE.union(C::SEGMENTED_ATTENTION);

/// DeepSeek-V4 sparse/compressed attention execution requirements.
pub const DEEPSEEK_V4: C = C::INDEXED_ATTENTION
    .union(C::POOLED_ATTENTION)
    .union(C::POOLED_POSITION_SELECTION)
    .union(C::POOLED_MASK_GATHER)
    .union(C::ATTENTION_SINKS)
    .union(C::RMS_NORM_WITHOUT_WEIGHT)
    .union(C::GROUPED_LINEAR);

/// Inkling learned-relative attention and routed/shared expert requirements.
pub const INKLING: C = C::RELATIVE_ATTENTION.union(C::JOINT_EXPERT_ROUTING);

/// Gemma 4 text and media execution requirements.
pub const GEMMA4: C = C::SIGMOID
    .union(C::SOFTPLUS)
    .union(C::EXP)
    .union(C::RMS_NORM_WITHOUT_WEIGHT);

/// Muse-Glimmer text and vision execution requirements.
pub const MUSE_GLIMMER: C = C::SIGMOID.union(C::RMS_NORM_WITHOUT_WEIGHT);

/// Validates one architecture requirement against a backend's declaration.
pub fn require<B: NeuralBackend>(architecture: &'static str, requirements: C) -> Result<(), Error> {
    B::require_operator_capabilities(architecture, requirements)
}
