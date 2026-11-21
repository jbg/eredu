//! Released text/adapter and image-context acceptance through ordinary loading.
use super::*;
#[path = "nemotron/adapter.rs"]
mod adapter;
#[cfg(feature = "image")]
#[path = "nemotron/vlm_reference.rs"]
mod vlm_reference;
