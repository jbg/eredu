//! Backend-neutral language-model facade.
//!
//! This module is available without an execution backend. Enabling the
//! default `mlx` feature adds the concrete MLX loader, model implementations,
//! prepared-chat execution, and native runtime diagnostics.
//! MLX executable, cache, load-policy, and generation types live under
//! `backend::mlx`, not in this namespace.

mod portable;

pub use portable::{
    LoadedModel, LoadedTextModelConfig, TextDecoder, TextDecoderError, TextModelError,
};
pub use safemlx_lm_utils::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};

#[cfg(feature = "mlx")]
#[path = "mlx.rs"]
mod mlx;
#[cfg(feature = "mlx")]
pub use mlx::*;
