//! Shared-backbone Nemotron-Labs-Diffusion equations and artifact contracts.

pub mod adapter;
mod checkpoint;
pub mod composite;
mod config;
pub mod definition;
pub mod generation;
pub mod gguf;
pub mod input;
pub mod model;
pub mod parallel;
pub mod partitioned;
pub mod processing;
pub mod tokenizer;
pub mod vision;
pub mod vision_gguf;

pub use checkpoint::{canonical_parameter_name, safetensors_plan, source_recipes};
pub use config::{ConfigError, FamilyConfig, PixtralConfig};
