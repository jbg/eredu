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
/// Shared typed prepared-input ingress for replicated composite graphs.
pub mod composite_execution;
/// Architecture-owned typed preparation for dense partitioned composite models.
pub mod composite_partitioned;
/// Authoritative model-family identity and Hugging Face/GGUF configuration parsing.
pub mod configuration;
/// Logical architecture and observation catalog generation from admitted plans.
pub mod discovery;
pub use configuration::{GgufArchitecture, ModelKind};
/// Architecture-owned external assistant inspection and preparation.
pub mod external_assistant;
pub use external_assistant::{
    prepare_execution_plan_assistant, prepare_external_assistant,
    CompatibleExternalAssistantPreparation, ExternalAssistantArchitecture,
    ExternalAssistantCheckpoint, ExternalAssistantExecutionMechanisms,
    ExternalAssistantExecutorVisitor, ExternalAssistantPreparation,
    ExternalAssistantPreparationVisitor, ExternalAssistantTargetProfile,
    ExternalAssistantTensorPlacement, ExternalAssistantTransfer, ExternalSpeculativeContract,
    ExternalSpeculativeContractRequest, MaterializedExternalAssistant,
    MaterializedExternalAssistantExecution, MaterializedExternalAssistantVisitor,
    PreparedCompatibleExternalAssistant, PreparedExternalAssistantExecution,
    PreparedExternalAssistantSource, SelectedExternalAssistantPreparation,
};
/// Backend-neutral schedules and recipes for independent expert residency.
pub mod expert_residency;
mod gguf_admission;
mod gguf_catalog;
pub use gguf_catalog::GgufTensorCatalog;
/// Backend-neutral family and checkpoint admission for sibling GGUF projectors.
pub mod gguf_companion;
mod inspection_validation;
mod linear_format;
/// Backend-neutral prepared-media admission and workspace plans.
pub mod media_plan;
/// Architecture-aware total artifact inspection and preparation retention.
pub mod model_inspection;
pub use model_inspection::{
    inspect_model, inspect_selected_model, prepare_inspected_model_sources, ModelInspectionOutcome,
    SelectedModelInspection,
};
/// Optional backend operators required by each architecture family.
pub mod operator_requirements;
/// Architecture-owned admission and typed handoff for partitioned execution.
pub mod partitioned_execution;
/// Architecture-owned construction and local geometry for embedded prediction extensions.
pub mod prediction_extension;
/// Architecture-derived capabilities used before backend materialization.
pub mod preparation;
/// Architecture-owned total cold-preparation selection.
pub mod preparation_selection;
pub use preparation_selection::{
    select_preparation, PreparationMechanismProvider, PreparationSelectionError,
};
/// Total prepared-model construction through typed native mechanism visitors.
pub mod prepared_execution;
/// Exact backend-neutral checkpoint source graphs prepared after selection.
pub mod prepared_sources;
/// Total backend-neutral execution and preparation selections.
pub mod selected_execution;
pub use selected_execution::{
    SelectedCompositePartitionedExecution, SelectedDensePartitionedExecution, SelectedExecution,
    SelectedExecutionDispatcher, SelectedPreparation, SelectedRoutedPartitionedExecution,
};
/// Architecture-owned execution of retained media processor plans.
pub mod processor_execution;
/// Backend-neutral family preprocessing and framing plans.
pub mod processor_plan;
mod replicated_model;
/// Architecture-owned admission for replicated text execution.
pub mod replicated_text;
/// Architecture-owned external rotary configuration values.
pub mod rotary;
/// Architecture-owned routed execution over generic backend mechanisms.
pub mod routed_text;
/// Backend-generic speculative execution over architecture-owned strategies.
pub mod speculative_execution;
mod static_parameters;
mod transport;
pub use expert_residency::{
    agree_expert_route_counts, exchange_expert_rows, execute_expert_route_exchange,
    execute_expert_route_exchange_tensor_parallel, execute_routed_gated_product,
    ExpertParameterRecipe, ExpertParameterRole, ExpertRealizationPlan, ExpertRealizationPlanError,
    ExpertResidencyCatalog, ExpertResidencyCatalogError, ExpertResidencyDistribution,
    ExpertResidencyUnit, ExpertRouteCountPlan, ExpertRouteExchangeDirection,
    ExpertRoutePackingPlan, PartitionExpertRouteExchange, RoutedMechanismExecutionError,
};
pub use routed_text::{
    routed_text_requirements, select_routed_text_realization, visit_gated_routed_text_architecture,
    visit_pooling_routed_text_architecture, visit_relu2_routed_text_architecture,
    EmptyPartitionRoutedExpertProvider, GatedProductOperation, GatedRoutedTextArchitectureVisitor,
    PlannedAddressableGatedProduct, PlannedAddressableRelu2, PlannedResidentGatedProduct,
    PlannedResidentRelu2, PreparedRelu2RoutedTextArchitecture, PreparedRoutedTextArchitecture,
    Relu2Operation, Relu2RoutedTextArchitectureVisitor, RoutedGroupedOperation,
    RoutedGroupedOperationValidation, RoutedGroupedPlan, RoutedTextDispatchError,
    RoutedTextExecutionError, RoutedTextPreparationError, RoutedTextRequirements,
    RoutedTextRequirementsError, RoutedTextSelectionError, RoutedTextSelectionRequest,
    SelectedRoutedTextRealization,
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

/// Nanbeige shared-weight repeated decoder family.
pub mod nanbeige;

/// Moshi-family realtime temporal/depth architecture policy.
pub mod moshi;

/// Kimi Linear hybrid KDA/MLA decoder family.
pub mod kimi_linear;
/// LFM2 and LFM2-MoE hybrid decoder architecture.
pub mod lfm2;
pub mod nemotron_h;

/// Qwen2, Qwen3, and Qwen3-MoE text decoder architecture.
pub mod qwen;
