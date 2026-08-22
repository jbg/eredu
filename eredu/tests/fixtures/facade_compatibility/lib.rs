//! Compile-only fixture for public paths retained during backend extraction.

pub use eredu::backend::mlx::{
    runtime::{
        generation::sampler::MirostatV2Sampler,
        media::input::{InputPart, ModelInput},
    },
    MlxBackend, MlxModelConfig, MlxParallelContext, ModelLoadOptions,
};
pub use eredu::composition::{
    deepseek::DeepSeekModel,
    llama::LlamaModel,
    mlx::{
        automatic::MlxBackendFactory,
        speculative::{MlxDrafter, MtpComponentTimingGuard, MtpExecutionStreams},
        MlxInspectionOptions, MlxModelInput, MlxModelSession,
    },
    qwen::QwenModel,
};
