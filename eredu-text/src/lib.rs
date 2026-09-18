//! Backend-neutral tokenizer construction and chat-template rendering.
//!
//! This crate adapts Hugging Face tokenizers and Jinja chat templates for
//! language-model runtimes. It can reconstruct tokenizers from GGUF metadata,
//! import supported tiktoken vocabularies, and render structured conversations
//! without depending on a model execution backend.

#![warn(missing_docs)]

/// Closed fresh chat-template construction and shared VM rendering.
pub mod chat_storage;

/// Prepared plain-join and ByteLevel decoding with fixed caller-owned destinations.
pub mod decoder_storage;
/// Error types returned by tokenizer and template operations.
pub mod error;
/// Tokenizer reconstruction from GGUF metadata.
pub mod gguf;
/// Literal UTF-8 stop matching with fixed destinations and borrowed output.
pub mod stop_storage;
/// Importers for rank-ordered tiktoken vocabularies.
pub mod tiktoken;
/// Tokenizer wrappers and chat-template rendering utilities.
pub mod tokenizer;

/// Fresh aggregate tokenizer construction; no accounting or public managed encoding.
pub mod tokenizer_storage;

mod generation_blocks;

/// Exact lexical token-byte conversion and borrowed packed vocabulary plans.
pub mod token_bytes;

/// Borrowed delimiter/channel transitions shared by semantic storage consumers.
pub mod semantic_channels;

/// Exact tokenizer-derived grammar trie construction and retained failure prefixes.
pub mod token_trie_storage;

/// Shared incremental JSON field and container boundary scanners.
pub mod json_fragments;
