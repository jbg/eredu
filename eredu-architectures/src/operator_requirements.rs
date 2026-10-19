//! Architecture-owned requirements for optional backend forward operators.

use eredu_nn::{Error, NeuralBackend, NeuralOperatorCapabilities as C};

/// Kimi Linear KDA execution requirements.
pub const KIMI_LINEAR: C = C::SIGMOID
    .union(C::SOFTPLUS)
    .union(C::EXP)
    .union(C::RMS_NORM_WITHOUT_WEIGHT)
    .union(C::GATED_DELTA_SCAN);

/// Qwen3-Next and Qwen3.5 hybrid execution requirements.
pub const QWEN_HYBRID: C = crate::gated_delta::REQUIRED_OPERATORS;

/// Nemotron-H Mamba execution requirements.
pub const NEMOTRON_H: C = C::GATED_GROUP_RMS_NORM
    .union(C::SELECTIVE_STATE_SPACE_SCAN)
    .union(C::BROADCAST_TO);

/// Qwen vision encoder execution requirements.
pub const QWEN_VISION: C = C::GELU_APPROXIMATE
    .union(C::SEGMENTED_ATTENTION)
    .union(C::FROM_I32_SLICE)
    .union(C::MULTI_AXIS_ROTARY_EMBEDDINGS);

/// Qwen vision-language assembly and execution requirements.
pub const QWEN_VL: C = QWEN_VISION
    .union(C::FULL_I32)
    .union(C::ZEROS_LIKE)
    .union(C::EQUAL_I32)
    .union(C::LOGICAL_OR)
    .union(C::MASKED_SCATTER);

/// DeepSeek-V3 target and MTP execution requirements.
pub const DEEPSEEK_V3: C = C::BROADCAST_TO;

/// DeepSeek-V4 sparse/compressed attention execution requirements.
pub const DEEPSEEK_V4: C = C::INDEXED_ATTENTION
    .union(C::POOLED_ATTENTION)
    .union(C::POOLED_POSITION_SELECTION)
    .union(C::POOLED_MASK_GATHER)
    .union(C::ATTENTION_SINKS)
    .union(C::RMS_NORM_WITHOUT_WEIGHT)
    .union(C::GROUPED_LINEAR)
    .union(C::UNLOADED_I32)
    .union(C::FULL_F32)
    .union(C::FULL_I32)
    .union(C::SOFTMAX_AXIS)
    .union(C::BROADCAST_TO)
    .union(C::ROPE_WITH_FREQUENCIES);

/// Inkling learned-relative attention and routed/shared expert requirements.
pub const INKLING: C = C::RELATIVE_ATTENTION
    .union(C::JOINT_GROUP_SELECTION)
    .union(C::FROM_I32_SLICE)
    .union(C::BROADCAST_TO);

/// Gemma 4 text and media execution requirements.
pub const GEMMA4: C = C::SIGMOID
    .union(C::SOFTPLUS)
    .union(C::EXP)
    .union(C::RMS_NORM_WITHOUT_WEIGHT)
    .union(C::UNLOADED_I32)
    .union(C::FULL_F32)
    .union(C::TANH)
    .union(C::CLIP)
    .union(C::SOFTMAX_AXIS)
    .union(C::CONV2D)
    .union(C::MULTI_AXIS_ROTARY_EMBEDDINGS)
    .union(C::MASKED_OUTPUT_PROJECTION);

/// Shared Muse text and media execution requirements.
pub const MUSE: C = C::SIGMOID
    .union(C::RMS_NORM_WITHOUT_WEIGHT)
    .union(C::FROM_I32_SLICE)
    .union(C::FULL_F32)
    .union(C::TANH)
    .union(C::MULTI_AXIS_ROTARY_EMBEDDINGS);

/// Additional mechanism used by the audio convolution's independent FP32 sums.
pub const MUSE_AUDIO: C = C::ZEROS_LIKE
    .union(C::CHANNEL_MAJOR_RMS)
    .union(C::CAST_FLOAT);

/// Exact media-dependent requirements of a selected Muse configuration.
pub fn muse(config: &crate::muse::DecoderConfig) -> C {
    let mut requirements = MUSE;
    if matches!(
        config.attention_equation,
        crate::muse::AttentionEquation::NormalizedHeads
            | crate::muse::AttentionEquation::NormalizedHeadsScoreScaled
    ) || config.block_normalization == crate::muse::BlockNormalization::InputScaledFeedForward
        || config.vocabulary_arithmetic() == crate::muse::VocabularyArithmetic::StageRounded
        || config.vision_config.as_ref().is_some_and(|vision| {
            matches!(
                vision.perception_normalization,
                crate::muse::PerceptionNormalization::Float32 { .. }
            )
        })
    {
        requirements = requirements.union(C::CASCADED_RMS).union(C::CAST_FLOAT);
    }
    if matches!(
        config.residual_equation,
        crate::muse::ResidualEquation::Learned { .. }
    ) {
        requirements = requirements.union(C::CAST_FLOAT).union(C::SQRT);
    } else if config.residual_equation == crate::muse::ResidualEquation::StoredCoefficients
        || config.vocabulary_arithmetic() == crate::muse::VocabularyArithmetic::Float32
    {
        requirements = requirements.union(C::CAST_FLOAT);
    }
    if config.audio_config.is_some() {
        requirements = requirements.union(MUSE_AUDIO);
    }
    if config
        .vision_config
        .as_ref()
        .is_some_and(|vision| vision.arithmetic == crate::muse::VisionArithmetic::Rounded)
    {
        requirements = requirements
            .union(C::CASCADED_LAYER_NORM)
            .union(C::CAST_FLOAT);
    }
    requirements
}

/// Validates one architecture requirement against a backend's declaration.
pub fn require<B: NeuralBackend>(architecture: &'static str, requirements: C) -> Result<(), Error> {
    B::require_operator_capabilities(architecture, requirements)
}
