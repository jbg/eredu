//! Backend-neutral text, multimodal, and realtime model architectures.
//!
//! Architecture code is monomorphized over an [`eredu_nn::NeuralBackend`]. Concrete
//! backends retain their native tensors, packed weights, lazy graphs, fused
//! kernels, caches, and collective implementations.

#![warn(missing_docs)]
// Architecture entry points intentionally expose complete execution context,
// and neutral operator enums stay inline to avoid backend-visible indirection.
#![allow(
    clippy::large_enum_variant,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

mod cache_identity;
/// Portable model capabilities and scalar runtime-state estimates.
pub mod capability;
/// Exact architecture-owned plans for dense checkpoint conversion.
pub mod checkpoint_conversion;
/// Authoritative model-family identity and Hugging Face/GGUF configuration parsing.
pub mod configuration;
pub use configuration::{GgufArchitecture, ModelKind};
/// Architecture-owned external assistant inspection and preparation.
pub mod external_assistant;
pub use external_assistant::{
    prepare_external_assistant, ExternalAssistantCheckpoint, ExternalAssistantPreparationPlan,
    Gemma4AssistantPreparationPlan, MuseGlimmerAssistantPreparationPlan,
};
/// Backend-neutral schedules and recipes for independent expert residency.
pub mod expert_residency;
mod gguf_admission;
mod gguf_catalog;
pub use gguf_catalog::GgufTensorCatalog;
/// Backend-neutral family and checkpoint admission for sibling GGUF projectors.
pub mod gguf_companion;
mod linear_format;
/// Backend-neutral prepared-media admission and workspace plans.
pub mod media_plan;
/// Optional backend operators required by each architecture family.
pub mod operator_requirements;
/// Architecture-derived capabilities used before backend materialization.
pub mod preparation;
/// Backend-neutral family preprocessing and framing plans.
pub mod processor_plan;
/// Architecture-owned external rotary configuration values.
pub mod rotary;
mod static_parameters;
mod transport;
pub use expert_residency::{
    ExpertParameterRecipe, ExpertParameterRole, ExpertRealizationPlan, ExpertRealizationPlanError,
    ExpertResidencyCatalog, ExpertResidencyCatalogError, ExpertResidencyDistribution,
    ExpertResidencyUnit,
};

/// Shared decoder mechanics used by backend-neutral text architectures.
pub mod decoder;
/// Shared assembly for heterogeneous stateful text decoders.
pub mod hybrid_decoder;

/// Inkling multimodal routed decoder family.
pub mod inkling;
pub mod muse_glimmer;

/// DeepSeek V3/R1 and V4 compressed-attention decoder family.
pub mod deepseek;
/// Neutral Gemma 4 family implementation.
pub mod gemma4;

/// OpenAI GPT-OSS sparse causal decoder architecture.
pub mod gpt_oss;

/// Llama and Mistral-compatible decoder architecture.
pub mod llama;

/// Moshi-family realtime temporal/depth architecture policy.
pub mod moshi;

/// Kimi Linear hybrid KDA/MLA decoder family.
pub mod kimi_linear;
/// LFM2 and LFM2-MoE hybrid decoder architecture.
pub mod lfm2;
pub mod nemotron_h;

/// Qwen2, Qwen3, and Qwen3-MoE text decoder architecture.
pub mod qwen;
