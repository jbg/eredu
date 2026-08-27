//! Authoritative MLX realization capabilities for each normalized family.

use eredu_architectures::ModelKind;
use eredu_core::{ParallelAxis, ParallelTopology};

/// MLX materializer used for a non-replicated family realization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ParallelRealization {
    /// The complete executable owns a pure tensor-parallel realization.
    Complete(CompleteTensorParallelBinding),
    /// The placed distributed-stage materializer owns tensor parallelism.
    DistributedStage,
    /// The family is realized through a different loading protocol.
    Unavailable,
}

/// Family binding implemented by the complete tensor-parallel materializer.
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

/// Family binding implemented by independent expert-cache materialization.
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

/// MLX composition implementation selected for a normalized family.
///
/// Multiple normalized families may intentionally share one implementation.
/// Loader dispatch is exhaustive over this MLX composition type, so adding a
/// family to the architecture registry changes availability only in
/// [`FamilyRealization::for_kind`]. A genuinely new implementation adds a
/// variant here and the compiler identifies every materializer that must bind
/// it.
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
    MoshiRealtime,
    Qwen,
    Qwen3Next,
    Qwen3Vl,
    Qwen3VlMoe,
    Qwen35,
}

/// Complete MLX realization surface for one normalized architecture family.
///
/// This is intentionally an exhaustive `ModelKind` match. Adding a family to
/// the architecture registry therefore requires one explicit decision for all
/// MLX artifact, parallel, residency, and quantization paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FamilyRealization {
    binding: FamilyBinding,
    gguf: bool,
    tensor_parallel: ParallelRealization,
    expert_cache: Option<ExpertCacheBinding>,
    safetensors_quantization: bool,
    complete_gguf_quantization: bool,
}

impl FamilyRealization {
    const fn new(
        binding: FamilyBinding,
        gguf: bool,
        tensor_parallel: ParallelRealization,
        expert_cache: Option<ExpertCacheBinding>,
        safetensors_quantization: bool,
        complete_gguf_quantization: bool,
    ) -> Self {
        Self {
            binding,
            gguf,
            tensor_parallel,
            expert_cache,
            safetensors_quantization,
            complete_gguf_quantization,
        }
    }

    /// Resolves the one authoritative MLX descriptor for `kind`.
    pub(crate) const fn for_kind(kind: ModelKind) -> Self {
        use CompleteTensorParallelBinding as Tp;
        use ExpertCacheBinding as Cache;
        use FamilyBinding as Family;
        use ParallelRealization::{DistributedStage, Unavailable};

        match kind {
            ModelKind::DeepSeekV3 => Self::new(
                Family::DeepSeekV3,
                true,
                DistributedStage,
                Some(Cache::DeepSeek),
                true,
                false,
            ),
            ModelKind::DeepSeekV4 => Self::new(
                Family::DeepSeekV4,
                true,
                DistributedStage,
                Some(Cache::DeepSeek),
                true,
                false,
            ),
            ModelKind::Gemma4 => Self::new(
                Family::Gemma4,
                true,
                ParallelRealization::Complete(Tp::Gemma4),
                Some(Cache::Gemma4),
                true,
                false,
            ),
            ModelKind::GptOss => Self::new(
                Family::GptOss,
                true,
                ParallelRealization::Complete(Tp::GptOss),
                Some(Cache::GptOss),
                true,
                true,
            ),
            ModelKind::Inkling => Self::new(
                Family::Inkling,
                true,
                ParallelRealization::Complete(Tp::Inkling),
                Some(Cache::Inkling),
                true,
                false,
            ),
            ModelKind::KimiLinear => Self::new(
                Family::KimiLinear,
                true,
                ParallelRealization::Complete(Tp::KimiLinear),
                Some(Cache::KimiLinear),
                true,
                true,
            ),
            ModelKind::Llama => Self::new(
                Family::Llama,
                true,
                ParallelRealization::Complete(Tp::Llama),
                None,
                true,
                true,
            ),
            ModelKind::MuseGlimmer => Self::new(
                Family::MuseGlimmer,
                true,
                ParallelRealization::Complete(Tp::MuseGlimmer),
                Some(Cache::MuseGlimmer),
                true,
                false,
            ),
            ModelKind::Lfm2 => Self::new(
                Family::Lfm2,
                true,
                ParallelRealization::Complete(Tp::Lfm2),
                Some(Cache::Lfm2),
                true,
                true,
            ),
            ModelKind::NemotronH => Self::new(
                Family::NemotronH,
                true,
                ParallelRealization::Complete(Tp::NemotronH),
                Some(Cache::NemotronH),
                true,
                true,
            ),
            ModelKind::Moshi => Self::new(
                Family::MoshiRealtime,
                false,
                Unavailable,
                None,
                false,
                false,
            ),
            ModelKind::Qwen2 => Self::new(
                Family::Qwen,
                true,
                ParallelRealization::Complete(Tp::Qwen),
                None,
                true,
                true,
            ),
            ModelKind::Qwen3 => Self::new(
                Family::Qwen,
                true,
                ParallelRealization::Complete(Tp::Qwen),
                Some(Cache::Qwen),
                true,
                true,
            ),
            ModelKind::Qwen3Next => Self::new(
                Family::Qwen3Next,
                true,
                DistributedStage,
                Some(Cache::Qwen3Next),
                true,
                true,
            ),
            ModelKind::Qwen3Vl => {
                Self::new(Family::Qwen3Vl, true, DistributedStage, None, true, true)
            }
            ModelKind::Qwen3VlMoe => Self::new(
                Family::Qwen3VlMoe,
                true,
                DistributedStage,
                Some(Cache::Qwen3VlMoe),
                true,
                true,
            ),
            ModelKind::Qwen35 => Self::new(
                Family::Qwen35,
                true,
                DistributedStage,
                Some(Cache::Qwen35),
                true,
                true,
            ),
        }
    }

    pub(crate) const fn binding(self) -> FamilyBinding {
        self.binding
    }

    pub(crate) const fn supports_gguf(self) -> bool {
        self.gguf
    }

    pub(crate) const fn tensor_parallel(self) -> ParallelRealization {
        self.tensor_parallel
    }

    pub(crate) const fn supports_expert_cache(self) -> bool {
        self.expert_cache.is_some()
    }

    pub(crate) const fn expert_cache(self) -> Option<ExpertCacheBinding> {
        self.expert_cache
    }

    pub(crate) const fn supports_safetensors_quantization(self) -> bool {
        self.safetensors_quantization
    }

    pub(crate) const fn supports_complete_gguf_quantization(self) -> bool {
        self.complete_gguf_quantization
    }

    pub(crate) const fn requires_distributed_stage(self, topology: ParallelTopology) -> bool {
        !matches!(self.tensor_parallel, ParallelRealization::Unavailable)
            && (topology.is_axis_active(ParallelAxis::Pipeline)
                || topology.is_axis_active(ParallelAxis::Expert)
                || matches!(self.tensor_parallel, ParallelRealization::DistributedStage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_family_has_a_coherent_mlx_realization() {
        for kind in ModelKind::ALL {
            let realization = FamilyRealization::for_kind(kind);
            if !realization.supports_gguf() {
                assert_eq!(
                    realization.tensor_parallel(),
                    ParallelRealization::Unavailable
                );
                assert!(!realization.supports_expert_cache());
                assert!(!realization.supports_safetensors_quantization());
                assert!(!realization.supports_complete_gguf_quantization());
            }
            if realization.supports_complete_gguf_quantization() {
                assert!(realization.supports_gguf());
            }
            if realization.supports_expert_cache() {
                assert!(realization.supports_safetensors_quantization());
            }
        }
    }
}
