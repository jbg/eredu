//! Backend-neutral tokenizer construction and chat-template rendering.
//!
//! This crate adapts Hugging Face tokenizers and Jinja chat templates for
//! language-model runtimes. It can reconstruct tokenizers from GGUF metadata,
//! import supported tiktoken vocabularies, and render structured conversations
//! without depending on a model execution backend.

#![warn(missing_docs)]

/// Error types returned by tokenizer and template operations.
pub mod error;
/// Tokenizer reconstruction from GGUF metadata.
pub mod gguf;
/// Importers for rank-ordered tiktoken vocabularies.
pub mod tiktoken;
/// Tokenizer wrappers and chat-template rendering utilities.
pub mod tokenizer;
