//! Qwen model families and their shared subfamily implementations.

/// Shared dense Qwen2/Qwen2.5/Qwen3 text decoder family.
pub mod dense;
/// Qwen3-Next and Qwen3.5 hybrid decoder family.
pub mod hybrid;
/// Qwen vision-language family.
pub mod vl;
