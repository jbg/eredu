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
    .union(C::GATED_DELTA_SCAN)
    .union(C::BROADCAST_TO);

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
    .union(C::JOINT_EXPERT_ROUTING)
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

/// Muse-Glimmer text and vision execution requirements.
pub const MUSE_GLIMMER: C = C::SIGMOID
    .union(C::RMS_NORM_WITHOUT_WEIGHT)
    .union(C::FROM_I32_SLICE)
    .union(C::FULL_F32)
    .union(C::TANH)
    .union(C::MULTI_AXIS_ROTARY_EMBEDDINGS);

/// Validates one architecture requirement against a backend's declaration.
pub fn require<B: NeuralBackend>(architecture: &'static str, requirements: C) -> Result<(), Error> {
    B::require_operator_capabilities(architecture, requirements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_admits_every_optional_tensor_operation_it_uses() {
        for required in [
            C::UNLOADED_I32,
            C::FULL_F32,
            C::TANH,
            C::CLIP,
            C::SOFTMAX_AXIS,
            C::CONV2D,
            C::MULTI_AXIS_ROTARY_EMBEDDINGS,
            C::MASKED_OUTPUT_PROJECTION,
        ] {
            assert!(GEMMA4.contains(required));
        }
    }

    #[test]
    fn qwen_vl_admits_multimodal_tensor_assembly() {
        assert!(QWEN_VL.contains(QWEN_VISION));
        for required in [
            C::FULL_I32,
            C::ZEROS_LIKE,
            C::EQUAL_I32,
            C::LOGICAL_OR,
            C::MASKED_SCATTER,
        ] {
            assert!(QWEN_VL.contains(required));
        }
    }
}
