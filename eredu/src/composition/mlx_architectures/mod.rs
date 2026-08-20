//! Model-family implementations and architecture-specific adapters.

/// MLX-internal rank-local distributed model adapters.
pub(crate) mod distributed;
/// Gemma 4 text and multimodal implementations.
pub mod gemma4;
/// GPT-OSS implementations and output format.
pub mod gpt_oss;
/// Inkling multimodal implementations.
pub mod inkling;
/// Moonshot Kimi Linear hybrid KDA/MLA implementations.
/// Moshi and PersonaPlex realtime-token implementations.
pub mod moshi;
/// Meta Muse-Glimmer dense multimodal implementations.
pub mod muse_glimmer;
/// Qwen-family implementations.
pub mod qwen;
