//! Model-family implementations and architecture-specific adapters.

/// MLX-internal rank-local distributed model adapters.
pub(crate) mod distributed;
/// GPT-OSS implementations and output format.
pub mod gpt_oss;
/// Inkling multimodal implementations.
/// Moonshot Kimi Linear hybrid KDA/MLA implementations.
/// Moshi and PersonaPlex realtime-token implementations.
pub mod moshi;
