#[test]
fn pipeline_activation_dtype_comes_from_wire_contract_not_weights() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let topology = crate::test_parallel_rank(0, 1, 2, 1);
    let wire_contract =
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Bfloat16);
    let request = MlxLoadRequest::with_parallel(
        topology,
        DeviceAssignment::new(DeviceType::Cpu, 0),
        wire_contract,
        4,
        4096,
        ring_completion_policy(),
    )
    .unwrap();
    let inspection = eredu_architectures::configuration::inspect_artifact(checkpoint.path())
        .expect("fixture inspection");
    let selected = crate::composition::mlx::loading::select_preparation(&inspection, request)
        .expect("public neutral selection");

    assert_eq!(
        selected.neutral().partitioned_activation_dtype(),
        Some(wire_contract.activation_dtype())
    );
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FixtureFamily {
    K2Dense,
    K2Mova,
    Llama,
    Mistral,
    DeepSeek,
    DeepSeekV4,
    DeepSeekGguf,
    Gemma,
    MuseGlimmer,
    Qwen2,
    Qwen2Gguf,
    Qwen3,
    Qwen3Gguf,
    Qwen3Moe,
    Qwen3MoeTied,
    Qwen3MoeGguf,
    GptOss,
    GptOssGguf,
    Lfm2,
    Lfm2Moe,
    Lfm2MoeGguf,
    KimiLinear,
    KimiLinearGguf,
    NemotronH,
    NemotronHGguf,
    Qwen3Next,
    Qwen3NextMoe,
    Qwen35,
    Qwen35Moe,
    Qwen35Multimodal,
    Qwen35ZeroPrediction,
    Qwen35MoeMultimodal,
    Qwen3Vl,
    Qwen3VlMoe,
    Inkling,
    InklingDense,
    InklingDenseMultimodal,
    InklingMultimodal,
    InklingGguf,
}

impl FixtureFamily {
    const fn name(self) -> &'static str {
        match self {
            Self::K2Dense => "k2-dense",
            Self::K2Mova => "k2-mova",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::DeepSeek => "deepseek",
            Self::DeepSeekV4 => "deepseek-v4",
            Self::DeepSeekGguf => "deepseek-gguf",
            Self::Gemma => "gemma",
            Self::MuseGlimmer => "muse-glimmer",
            Self::Qwen2 => "qwen2",
            Self::Qwen2Gguf => "qwen2-gguf",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Gguf => "qwen3-gguf",
            Self::Qwen3Moe => "qwen3-moe",
            Self::Qwen3MoeTied => "qwen3-moe-tied",
            Self::Qwen3MoeGguf => "qwen3-moe-gguf",
            Self::GptOss => "gpt-oss",
            Self::GptOssGguf => "gpt-oss-gguf",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2-moe",
            Self::Lfm2MoeGguf => "lfm2-moe-gguf",
            Self::KimiLinear => "kimi-linear",
            Self::KimiLinearGguf => "kimi-linear-gguf",
            Self::NemotronH => "nemotron-h",
            Self::NemotronHGguf => "nemotron-h-gguf",
            Self::Qwen3Next => "qwen3-next",
            Self::Qwen3NextMoe => "qwen3-next-moe",
            Self::Qwen35 => "qwen3.5",
            Self::Qwen35Moe => "qwen3.5-moe",
            Self::Qwen35Multimodal => "qwen3.5-multimodal",
            Self::Qwen35ZeroPrediction => "qwen3.5-zero-prediction",
            Self::Qwen35MoeMultimodal => "qwen3.5-moe-multimodal",
            Self::Qwen3Vl => "qwen3-vl",
            Self::Qwen3VlMoe => "qwen3-vl-moe",
            Self::Inkling => "inkling",
            Self::InklingDense => "inkling-dense",
            Self::InklingDenseMultimodal => "inkling-dense-multimodal",
            Self::InklingMultimodal => "inkling-multimodal",
            Self::InklingGguf => "inkling-gguf",
        }
    }

    fn parse(value: &str) -> Self {
        for family in [
            Self::K2Dense,
            Self::K2Mova,
            Self::Llama,
            Self::Mistral,
            Self::DeepSeek,
            Self::DeepSeekV4,
            Self::DeepSeekGguf,
            Self::Gemma,
            Self::MuseGlimmer,
            Self::Qwen2,
            Self::Qwen2Gguf,
            Self::Qwen3,
            Self::Qwen3Gguf,
            Self::Qwen3Moe,
            Self::Qwen3MoeTied,
            Self::Qwen3MoeGguf,
            Self::GptOss,
            Self::GptOssGguf,
            Self::Lfm2,
            Self::Lfm2Moe,
            Self::Lfm2MoeGguf,
            Self::KimiLinear,
            Self::KimiLinearGguf,
            Self::NemotronH,
            Self::NemotronHGguf,
            Self::Qwen3Next,
            Self::Qwen3NextMoe,
            Self::Qwen35,
            Self::Qwen35Moe,
            Self::Qwen35Multimodal,
            Self::Qwen35ZeroPrediction,
            Self::Qwen35MoeMultimodal,
            Self::Qwen3Vl,
            Self::Qwen3VlMoe,
            Self::Inkling,
            Self::InklingDense,
            Self::InklingDenseMultimodal,
            Self::InklingMultimodal,
            Self::InklingGguf,
        ] {
            if family.name() == value {
                return family;
            }
        }
        panic!("unexpected pipeline fixture family {value:?}")
    }

    fn layer_count(self) -> usize {
        match self {
            Self::K2Dense | Self::K2Mova => 3,
            Self::Llama
            | Self::Mistral
            | Self::DeepSeek
            | Self::DeepSeekV4
            | Self::DeepSeekGguf
            | Self::Qwen2
            | Self::Qwen2Gguf
            | Self::Qwen3
            | Self::Qwen3Gguf
            | Self::Qwen3Moe
            | Self::Qwen3MoeTied
            | Self::Qwen3MoeGguf
            | Self::GptOss
            | Self::GptOssGguf
            | Self::Lfm2
            | Self::Lfm2Moe
            | Self::Lfm2MoeGguf
            | Self::KimiLinear
            | Self::KimiLinearGguf
            | Self::Qwen3Next
            | Self::Qwen3NextMoe
            | Self::Qwen35
            | Self::Qwen35Moe
            | Self::Qwen35Multimodal => 2,
            Self::Qwen35ZeroPrediction
            | Self::Qwen35MoeMultimodal
            | Self::Qwen3Vl
            | Self::Qwen3VlMoe
            | Self::MuseGlimmer => 2,
            Self::Gemma | Self::NemotronH | Self::NemotronHGguf => 4,
            Self::Inkling
            | Self::InklingDense
            | Self::InklingDenseMultimodal
            | Self::InklingMultimodal
            | Self::InklingGguf => 3,
        }
    }

    fn stage_range(self, rank: usize) -> std::ops::Range<usize> {
        match (self, rank) {
            (Self::K2Dense | Self::K2Mova, 0) => 0..2,
            (Self::K2Dense | Self::K2Mova, 1) => 2..3,
            (Self::Gemma, 0) => 0..1,
            (Self::Gemma, 1) => 1..4,
            (Self::NemotronH | Self::NemotronHGguf, 0) => 0..2,
            (Self::NemotronH | Self::NemotronHGguf, 1) => 2..4,
            (
                Self::Inkling
                | Self::InklingDense
                | Self::InklingDenseMultimodal
                | Self::InklingMultimodal
                | Self::InklingGguf,
                0,
            ) => 0..2,
            (
                Self::Inkling
                | Self::InklingDense
                | Self::InklingDenseMultimodal
                | Self::InklingMultimodal
                | Self::InklingGguf,
                1,
            ) => 2..3,
            (_, rank) => rank..rank + 1,
        }
    }

    fn expert_layer_count(self, range: std::ops::Range<usize>) -> usize {
        match self {
            Self::K2Dense => 0,
            Self::K2Mova => range.filter(|index| *index >= 1).count(),
            Self::DeepSeek
            | Self::DeepSeekGguf
            | Self::Lfm2Moe
            | Self::Lfm2MoeGguf
            | Self::KimiLinear
            | Self::KimiLinearGguf => range.filter(|index| *index == 1).count(),
            Self::DeepSeekV4 => range.len(),
            Self::Inkling | Self::InklingMultimodal | Self::InklingGguf => {
                range.filter(|index| matches!(*index, 1 | 2)).count()
            }
            Self::InklingDense | Self::InklingDenseMultimodal => 0,
            Self::NemotronH => range.filter(|index| *index == 2).count(),
            Self::NemotronHGguf => range.filter(|index| matches!(*index, 1 | 2)).count(),
            _ => range.len(),
        }
    }

    fn effective_model_type(self) -> &'static str {
        match self {
            Self::K2Dense | Self::K2Mova => "k2_horizon",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::DeepSeek | Self::DeepSeekGguf => "deepseek_v3",
            Self::DeepSeekV4 => "deepseek_v4",
            Self::Gemma => "gemma4_text",
            Self::MuseGlimmer => "muse_glimmer_text",
            Self::Qwen2 | Self::Qwen2Gguf => "qwen2",
            Self::Qwen3 | Self::Qwen3Gguf => "qwen3",
            Self::Qwen3Moe | Self::Qwen3MoeTied | Self::Qwen3MoeGguf => "qwen3_moe",
            Self::GptOss | Self::GptOssGguf => "gpt_oss",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe | Self::Lfm2MoeGguf => "lfm2_moe",
            Self::KimiLinear | Self::KimiLinearGguf => "kimi_linear",
            Self::NemotronH | Self::NemotronHGguf => "nemotron_h",
            Self::Qwen3Next | Self::Qwen3NextMoe => "qwen3_next",
            Self::Qwen35 | Self::Qwen35Multimodal | Self::Qwen35ZeroPrediction => "qwen3_5_text",
            Self::Qwen35Moe | Self::Qwen35MoeMultimodal => "qwen3_5_moe_text",
            Self::Qwen3Vl => "qwen3_vl_text",
            Self::Qwen3VlMoe => "qwen3_vl_moe_text",
            Self::Inkling
            | Self::InklingDense
            | Self::InklingDenseMultimodal
            | Self::InklingMultimodal
            | Self::InklingGguf => "inkling_mm_model",
        }
    }

    const fn needs_opaque_reference(self) -> bool {
        matches!(
            self,
            Self::K2Dense
                | Self::K2Mova
                | Self::Llama
                | Self::Mistral
                | Self::Gemma
                | Self::MuseGlimmer
                | Self::Qwen2
                | Self::Qwen2Gguf
                | Self::Qwen3
                | Self::Qwen3Gguf
                | Self::Qwen3Moe
                | Self::GptOss
                | Self::KimiLinear
                | Self::NemotronH
                | Self::Qwen3Next
                | Self::Qwen35
                | Self::Qwen35Multimodal
                | Self::Qwen35ZeroPrediction
                | Self::Qwen3Vl
                | Self::DeepSeek
                | Self::DeepSeekV4
                | Self::InklingDense
                | Self::InklingDenseMultimodal
        )
    }

    const fn is_multimodal(self) -> bool {
        matches!(
            self,
            Self::InklingMultimodal
                | Self::InklingDenseMultimodal
                | Self::Qwen35Multimodal
                | Self::Qwen35ZeroPrediction
                | Self::Qwen35MoeMultimodal
                | Self::Qwen3Vl
                | Self::Qwen3VlMoe
        )
    }

    const fn comparison_tolerance(self) -> f32 {
        match self {
            Self::DeepSeekGguf => 3e-3,
            Self::DeepSeekV4 | Self::KimiLinear | Self::KimiLinearGguf => 1e-3,
            Self::Qwen3Next
            | Self::Qwen3NextMoe
            | Self::Qwen35
            | Self::Qwen35Moe
            | Self::Qwen35Multimodal
            | Self::Qwen35ZeroPrediction
            | Self::Qwen35MoeMultimodal => 2e-3,
            Self::NemotronH | Self::NemotronHGguf => 1e-3,
            _ if self.is_multimodal() => 5e-4,
            _ => 1e-4,
        }
    }
}
