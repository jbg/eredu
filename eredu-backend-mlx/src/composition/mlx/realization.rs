//! Typed MLX materializer bindings for normalized architecture families.

use eredu_architectures::ModelKind;
use eredu_core::{ParallelAxis, ParallelTopology};

/// Family binding consumed by ordinary SafeTensors and distributed-stage loaders.
///
/// Absence of a binding means that the family uses another loading protocol.
/// The variants are exhaustively dispatched by the materializers, so this is
/// implementation selection rather than a parallel capability declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FamilyBinding {
    DeepSeekV3,
    DeepSeekV4,
    Gemma4,
    GptOss,
    Inkling,
    KimiLinear,
    Llama,
    MuseGlimmer,
    Lfm2,
    NemotronH,
    Qwen,
    Qwen3Next,
    Qwen3Vl,
    Qwen3VlMoe,
    Qwen35,
}

impl FamilyBinding {
    /// Selects the ordinary MLX materializer for `kind`.
    pub(crate) const fn for_kind(kind: ModelKind) -> Option<Self> {
        Some(match kind {
            ModelKind::DeepSeekV3 => Self::DeepSeekV3,
            ModelKind::DeepSeekV4 => Self::DeepSeekV4,
            ModelKind::Gemma4 => Self::Gemma4,
            ModelKind::GptOss => Self::GptOss,
            ModelKind::Inkling => Self::Inkling,
            ModelKind::KimiLinear => Self::KimiLinear,
            ModelKind::Llama => Self::Llama,
            ModelKind::MuseGlimmer => Self::MuseGlimmer,
            ModelKind::Lfm2 => Self::Lfm2,
            ModelKind::NemotronH => Self::NemotronH,
            ModelKind::Moshi => return None,
            ModelKind::Qwen2 | ModelKind::Qwen3 => Self::Qwen,
            ModelKind::Qwen3Next => Self::Qwen3Next,
            ModelKind::Qwen3Vl => Self::Qwen3Vl,
            ModelKind::Qwen3VlMoe => Self::Qwen3VlMoe,
            ModelKind::Qwen35 => Self::Qwen35,
        })
    }
}

/// GGUF loader whose call accepts load-time quantization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QuantizedGgufBinding {
    GptOss,
    KimiLinear,
    Llama,
    Lfm2,
    NemotronH,
    Qwen,
    Qwen3Next,
    Qwen3Vl,
    Qwen3VlMoe,
    Qwen35,
}

/// GGUF loader whose call has no load-time quantization input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FixedGgufBinding {
    DeepSeekV3,
    DeepSeekV4,
    Gemma4,
    Inkling,
    MuseGlimmer,
}

/// Concrete GGUF dispatch route.
///
/// The route category determines whether the actual loader receives a
/// quantization policy. Preflight derives support from this same route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GgufBinding {
    Quantized(QuantizedGgufBinding),
    Fixed(FixedGgufBinding),
}

impl GgufBinding {
    /// Selects the GGUF materializer for `kind`.
    pub(crate) const fn for_kind(kind: ModelKind) -> Option<Self> {
        use FixedGgufBinding as Fixed;
        use QuantizedGgufBinding as Quantized;

        Some(match kind {
            ModelKind::DeepSeekV3 => Self::Fixed(Fixed::DeepSeekV3),
            ModelKind::DeepSeekV4 => Self::Fixed(Fixed::DeepSeekV4),
            ModelKind::Gemma4 => Self::Fixed(Fixed::Gemma4),
            ModelKind::GptOss => Self::Quantized(Quantized::GptOss),
            ModelKind::Inkling => Self::Fixed(Fixed::Inkling),
            ModelKind::KimiLinear => Self::Quantized(Quantized::KimiLinear),
            ModelKind::Llama => Self::Quantized(Quantized::Llama),
            ModelKind::MuseGlimmer => Self::Fixed(Fixed::MuseGlimmer),
            ModelKind::Lfm2 => Self::Quantized(Quantized::Lfm2),
            ModelKind::NemotronH => Self::Quantized(Quantized::NemotronH),
            ModelKind::Moshi => return None,
            ModelKind::Qwen2 | ModelKind::Qwen3 => Self::Quantized(Quantized::Qwen),
            ModelKind::Qwen3Next => Self::Quantized(Quantized::Qwen3Next),
            ModelKind::Qwen3Vl => Self::Quantized(Quantized::Qwen3Vl),
            ModelKind::Qwen3VlMoe => Self::Quantized(Quantized::Qwen3VlMoe),
            ModelKind::Qwen35 => Self::Quantized(Quantized::Qwen35),
        })
    }

    pub(crate) const fn accepts_quantization(self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

/// Family binding consumed by the complete tensor-parallel materializer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompleteTensorParallelBinding {
    Gemma4,
    GptOss,
    Inkling,
    KimiLinear,
    Llama,
    MuseGlimmer,
    Lfm2,
    NemotronH,
    Qwen,
}

impl CompleteTensorParallelBinding {
    pub(crate) const fn for_kind(kind: ModelKind) -> Option<Self> {
        Some(match kind {
            ModelKind::Gemma4 => Self::Gemma4,
            ModelKind::GptOss => Self::GptOss,
            ModelKind::Inkling => Self::Inkling,
            ModelKind::KimiLinear => Self::KimiLinear,
            ModelKind::Llama => Self::Llama,
            ModelKind::MuseGlimmer => Self::MuseGlimmer,
            ModelKind::Lfm2 => Self::Lfm2,
            ModelKind::NemotronH => Self::NemotronH,
            ModelKind::Qwen2 | ModelKind::Qwen3 => Self::Qwen,
            ModelKind::DeepSeekV3
            | ModelKind::DeepSeekV4
            | ModelKind::Moshi
            | ModelKind::Qwen3Next
            | ModelKind::Qwen3Vl
            | ModelKind::Qwen3VlMoe
            | ModelKind::Qwen35 => return None,
        })
    }
}

/// Family binding consumed by independent expert-cache materialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExpertCacheBinding {
    DeepSeek,
    Gemma4,
    GptOss,
    Inkling,
    KimiLinear,
    MuseGlimmer,
    Lfm2,
    NemotronH,
    Qwen,
    Qwen3Next,
    Qwen3VlMoe,
    Qwen35,
}

impl ExpertCacheBinding {
    pub(crate) const fn for_kind(kind: ModelKind) -> Option<Self> {
        Some(match kind {
            ModelKind::DeepSeekV3 | ModelKind::DeepSeekV4 => Self::DeepSeek,
            ModelKind::Gemma4 => Self::Gemma4,
            ModelKind::GptOss => Self::GptOss,
            ModelKind::Inkling => Self::Inkling,
            ModelKind::KimiLinear => Self::KimiLinear,
            ModelKind::MuseGlimmer => Self::MuseGlimmer,
            ModelKind::Lfm2 => Self::Lfm2,
            ModelKind::NemotronH => Self::NemotronH,
            ModelKind::Qwen3 => Self::Qwen,
            ModelKind::Qwen3Next => Self::Qwen3Next,
            ModelKind::Qwen3VlMoe => Self::Qwen3VlMoe,
            ModelKind::Qwen35 => Self::Qwen35,
            ModelKind::Llama | ModelKind::Moshi | ModelKind::Qwen2 | ModelKind::Qwen3Vl => {
                return None;
            }
        })
    }
}

/// Whether non-replicated materialization must use a distributed stage.
pub(crate) const fn requires_distributed_stage(
    kind: ModelKind,
    topology: ParallelTopology,
) -> bool {
    FamilyBinding::for_kind(kind).is_some()
        && (topology.is_axis_active(ParallelAxis::Pipeline)
            || topology.is_axis_active(ParallelAxis::Expert)
            || CompleteTensorParallelBinding::for_kind(kind).is_none())
}
