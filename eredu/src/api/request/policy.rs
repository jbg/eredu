//! Request policy over a recognized profile, independent of rendering and execution.
mod compiled;
mod original;
use super::{ChatTemplateRequest, text_chat_eligibility};
use crate::runtime::chat::{
    CapabilitySupport, ChatCapabilities, NativeToolSupport, PreparedFormatProfile, ProfileStrings,
    SemanticSupport, ToolChoice,
    dialect::{DialectParameters, FormatDialect},
};
pub(crate) use compiled::{CompiledChatPolicy, Failure as CompilationFailure};
pub(crate) use original::{Failure as OriginalPolicyFailure, compile_original};

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub(crate) enum Rejection {
    #[error(
        "format profile {profile:?} does not preserve reasoning semantics while native tools are active"
    )]
    ToolReasoning { profile: &'static str },
    #[error(
        "thinking was explicitly enabled, but no semantic reasoning protocol was recognized: {reason}; set allow_unparsed_reasoning to opt into raw output"
    )]
    UnparsedReasoning { reason: &'static str },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Runtime {
    pub(crate) dialect: &'static dyn FormatDialect,
    pub(crate) parameters: DialectParameters,
    pub(crate) structural_tokens: ProfileStrings,
    pub(crate) has_tool_surface: bool,
}

/// Fixed declarations shared by inspection and executable chat preparation.
/// Constructing this value neither compiles a grammar nor grants execution.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Selection {
    pub(crate) runtime: Option<Runtime>,
    pub(crate) native_tool_support: NativeToolSupport,
    pub(crate) semantic_support: SemanticSupport,
    pub(crate) text_generation_support: CapabilitySupport,
    pub(crate) capabilities: ChatCapabilities,
}
impl Selection {
    pub(crate) fn prepare(
        profile: &PreparedFormatProfile,
        request: &ChatTemplateRequest,
        compiler_available: bool,
    ) -> Result<Self, Rejection> {
        if request.tool_choice != ToolChoice::None
            && request.enable_thinking == Some(true)
            && !request.tools.is_empty()
            && !profile.supports_tool_reasoning
        {
            return Err(Rejection::ToolReasoning {
                profile: profile.identity.unwrap_or("unregistered"),
            });
        }
        let semantic_failure = profile
            .native_tool_unavailable_reason
            .unwrap_or("no semantic protocol was recognized");
        let tool_surface_requested =
            !request.tools.is_empty() || request.tool_choice == ToolChoice::Required;
        let tool_protocol_available = profile.tool_dialect.is_some()
            && profile.tool_dialect_parameters.is_some()
            && compiler_available;
        let runtime = if tool_surface_requested && tool_protocol_available {
            profile
                .tool_dialect
                .zip(profile.tool_dialect_parameters)
                .map(|(dialect, parameters)| Runtime {
                    dialect,
                    parameters,
                    structural_tokens: profile.tool_required_structural_tokens,
                    has_tool_surface: true,
                })
        } else {
            profile
                .dialect
                .zip(profile.dialect_parameters)
                .map(|(dialect, parameters)| Runtime {
                    dialect,
                    parameters,
                    structural_tokens: profile.required_structural_tokens,
                    has_tool_surface: false,
                })
        };
        if request.enable_thinking == Some(true)
            && (runtime.is_none() || !profile.supports_reasoning_parsing)
            && !request.allow_unparsed_reasoning
        {
            return Err(Rejection::UnparsedReasoning {
                reason: semantic_failure,
            });
        }
        Ok(Self {
            native_tool_support: if tool_protocol_available {
                NativeToolSupport::Supported
            } else {
                NativeToolSupport::Unsupported {
                    reason: semantic_failure,
                }
            },
            semantic_support: if runtime.is_some() {
                SemanticSupport::Supported
            } else {
                SemanticSupport::Unsupported {
                    reason: semantic_failure,
                }
            },
            text_generation_support: match text_chat_eligibility(request) {
                Ok(()) => CapabilitySupport::Supported,
                Err(reason) => CapabilitySupport::Unsupported {
                    reason: reason.reason(),
                },
            },
            capabilities: ChatCapabilities {
                reasoning_parser: capability(
                    runtime.is_some() && profile.supports_reasoning_parsing,
                    "the selected protocol does not provide a recognized reasoning channel",
                ),
                visible_text_parser: capability(
                    runtime.is_some(),
                    "no semantic visible-text parser was recognized",
                ),
                tool_output_parser: capability(
                    profile.tool_dialect.is_some(),
                    "generated tool-call envelopes were not recognized",
                ),
                tool_input_rendering: capability(
                    profile.supports_tool_input_rendering,
                    "tool-call and tool-response rendering probes did not establish support",
                ),
                mapping_tool_arguments: capability(
                    profile.supports_mapping_tool_arguments,
                    "tool-call history with mapping arguments was not established",
                ),
                string_tool_arguments: capability(
                    profile.supports_string_tool_arguments,
                    "tool-call history with serialized string arguments was not established",
                ),
                constrained_tool_generation: capability(
                    profile.tool_dialect.is_some() && compiler_available,
                    "no compatible tokenizer constraint compiler is available",
                ),
            },
            runtime,
        })
    }

    /// Fixed producer and return controls, paid before selecting request policy.
    pub(crate) fn control_bytes() -> usize {
        std::mem::size_of::<(
            Self,
            Rejection,
            Result<Self, Rejection>,
            &PreparedFormatProfile,
            &ChatTemplateRequest,
            bool,
            bool,
            bool,
            &'static str,
            Option<Runtime>,
        )>()
    }
}

fn capability(condition: bool, reason: &'static str) -> CapabilitySupport {
    if condition {
        CapabilitySupport::Supported
    } else {
        CapabilitySupport::Unsupported { reason }
    }
}
