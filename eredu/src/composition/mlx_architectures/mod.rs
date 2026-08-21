//! Model-family implementations and architecture-specific adapters.

/// MLX-internal rank-local distributed model adapters.
pub(crate) mod distributed;
/// Inkling multimodal implementations.
/// Moonshot Kimi Linear hybrid KDA/MLA implementations.
/// Moshi and PersonaPlex realtime-token implementations.
pub mod moshi;
