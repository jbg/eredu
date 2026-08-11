//! Model-family implementations and architecture-specific adapters.

/// DeepSeek-V3 and DeepSeek-R1 implementations.
pub mod deepseek_v3;
/// Architecture-dispatched distributed model adapters.
pub mod distributed;
/// Gemma 4 text and multimodal implementations.
pub mod gemma4;
/// GPT-OSS implementations and output format.
pub mod gpt_oss;
/// Inkling multimodal implementations.
pub mod inkling;
/// Moonshot Kimi Linear hybrid KDA/MLA implementations.
pub mod kimi_linear;
/// LFM2 and LFM2.5 implementations.
pub mod lfm2;
/// Llama and Mistral-compatible implementations.
pub mod llama;
/// Moshi and PersonaPlex realtime-token implementations.
pub mod moshi;
/// Meta Muse-Glimmer dense multimodal implementations.
pub mod muse_glimmer;
/// Nemotron-H implementations.
pub mod nemotron_h;
/// Qwen-family implementations.
pub mod qwen;
