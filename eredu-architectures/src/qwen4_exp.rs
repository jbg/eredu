//! Qwen4 experimental family equations and exact checkpoint contracts.
//!
//! Family admission is enabled only after complete prepared execution is wired.
/// Distinct released-family configuration and schedule normalization.
pub mod config;
pub mod ngram;

/// Exact released n-gram aliases, integer buffers and compact table recipes.
pub mod checkpoint;
/// Released prediction fusion, decoder owners and readout, separate from target state.
pub mod mtp;
/// Dilated, gated lexical injection over complete residual streams.
pub mod ple;
/// Cached micro-block index summaries and bounded selected-position equations.
pub mod qsa;

/// Recurrent sublayer construction over the shared gated-delta implementation.
pub mod recurrent;
/// Released residual mixer identities and shared decoder ingress/readout.
pub mod residual;

pub mod conditional;
/// Shared/routed feed-forward sublayer with gated residual transport.
pub mod feed_forward;
/// Indexed attention composed with cached QSA and complete residual streams.
pub mod indexed;
/// Exact original request provenance and residual-stream transport.
pub mod input;
/// Bounded original-ID-preserving media ingress and shared rotary construction.
pub mod media;
/// Exact persisted media rotary offset through the shared fixed-state lifecycle.
pub mod position;
/// Retained SafeTensors target owners and generic row-bank binding contracts.
pub mod prepared;
/// Layer-sized target units and shared layered execution.
pub mod target;
