//! Backend-independent tokenizer, chat-template, and semantic inspection enrichment.

pub use eredu_architectures::ModelInspectionOutcome;
pub use eredu_core::artifact::SafetensorsHeader;
pub use eredu_core::artifact::{
    ArtifactMetadata, CheckpointMetadata, GgufCompanionMetadata, MetadataProvenance,
};
pub use eredu_gguf::CheckpointHeader as GgufHeader;

use std::path::Path;

/// Inspects artifact headers and returns its logical architecture and capture catalog.
/// Available with no backend feature; no native device or tensor is created.
pub fn inspect_architecture(
    path: impl AsRef<Path>,
) -> Result<eredu_core::ArchitectureDescriptor, eredu_core::artifact::ArtifactError> {
    let artifact = eredu_architectures::configuration::inspect_artifact(path)?;
    Ok(artifact.architecture_plan().architecture_descriptor())
}

use eredu_core::{
    InspectionIssueCode, InspectionReadiness, InspectionSeverity, ModelInspectionReport,
};
use serde_json::json;

use super::request::prepare_chat_from_parts;
use super::{LoadedTextModelConfig, PreparedTextModel};
use crate::runtime::chat::{
    constraints::ConstraintCompiler, ChatTemplateRequest, NativeToolSupport, PreparedChat,
    SemanticSupport, ToolChoice,
};
use eredu_text::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};

/// Optional behavioral probes over the exact text state retained for loading.
#[derive(Debug, Clone, Default)]
pub struct TextInspectionOptions {
    /// Optional concrete chat request to render and behaviorally probe.
    pub chat_request: Option<ChatTemplateRequest>,
}

impl PreparedTextModel {
    /// Reports text readiness without selecting an execution backend.
    /// Requested-load readiness remains unverified until execution selection.
    pub fn inspect(&mut self, options: TextInspectionOptions) -> ModelInspectionReport {
        let mut report =
            ModelInspectionReport::unverified(self.artifact.path(), self.artifact.format());
        report.record_artifact_inspection(&self.artifact);
        enrich_text_inspection(report, &mut self.tokenizer, &self.config, options)
    }
}

pub(super) fn enrich_text_inspection(
    mut report: ModelInspectionReport,
    tokenizer: &mut ChatTokenizer,
    config: &LoadedTextModelConfig,
    options: TextInspectionOptions,
) -> ModelInspectionReport {
    report.tokenizer = InspectionReadiness::Ready;
    if let Some(template) = &config.chat_template {
        report.chat_template = InspectionReadiness::Ready;
        inspect_chat_behavior(
            &mut report,
            tokenizer,
            template.clone(),
            &config.eos_token_ids,
            &config.model_id,
            options.chat_request,
        );
    } else {
        report.chat_template = InspectionReadiness::Missing;
        report.semantic_streaming = InspectionReadiness::Missing;
        report.native_tools = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingChatTemplate,
            InspectionSeverity::Warning,
            "no checkpoint chat template or explicit override is available",
            Some(report.path.clone()),
        );
    }
    finalize_text_readiness(&mut report);
    report
}

fn inspect_chat_behavior(
    report: &mut ModelInspectionReport,
    tokenizer: &mut ChatTokenizer,
    template: ModelChatTemplate,
    eos_token_ids: &[u32],
    model_id: &str,
    request: Option<ChatTemplateRequest>,
) {
    let compiler = ConstraintCompiler::from_tokenizer(tokenizer, eos_token_ids);
    if let Some(request) = request {
        match prepare_chat_from_parts(
            tokenizer,
            template,
            model_id,
            eos_token_ids,
            Some(&compiler),
            request,
        ) {
            Ok(prepared) => apply_prepared_chat(report, &prepared),
            Err(error) => {
                report.semantic_streaming = InspectionReadiness::Unsupported;
                report.native_tools = InspectionReadiness::Unsupported;
                report.issue(
                    InspectionIssueCode::UnsupportedSemanticProtocol,
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(report.path.clone()),
                );
            }
        }
        return;
    }

    let semantic_request = ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "__eredu_inspection_probe__"})],
        tool_choice: ToolChoice::None,
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    };
    match prepare_chat_from_parts(
        tokenizer,
        template.clone(),
        model_id,
        eos_token_ids,
        Some(&compiler),
        semantic_request,
    ) {
        Ok(prepared) => {
            report.semantic_streaming = match prepared.semantic_support() {
                SemanticSupport::Supported => InspectionReadiness::Ready,
                SemanticSupport::Unsupported { reason } => {
                    report.issue(
                        InspectionIssueCode::UnsupportedSemanticProtocol,
                        InspectionSeverity::Warning,
                        reason.clone(),
                        Some(report.path.clone()),
                    );
                    InspectionReadiness::Unsupported
                }
            };
        }
        Err(error) => {
            report.semantic_streaming = InspectionReadiness::Unsupported;
            report.issue(
                InspectionIssueCode::UnsupportedSemanticProtocol,
                InspectionSeverity::Warning,
                error.to_string(),
                Some(report.path.clone()),
            );
        }
    }

    let tool_request = ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "__eredu_tool_probe__"})],
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "eredu_probe",
                "description": "inspection probe",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"]
                }
            }
        })],
        tool_choice: ToolChoice::Required,
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    };
    match prepare_chat_from_parts(
        tokenizer,
        template,
        model_id,
        eos_token_ids,
        Some(&compiler),
        tool_request,
    ) {
        Ok(prepared) => {
            report.native_tools = match prepared.native_tool_support() {
                NativeToolSupport::Supported => InspectionReadiness::Ready,
                NativeToolSupport::Unsupported { rejection } => {
                    report.issue(
                        InspectionIssueCode::UnsupportedToolProtocol,
                        InspectionSeverity::Warning,
                        rejection.to_string(),
                        Some(report.path.clone()),
                    );
                    InspectionReadiness::Unsupported
                }
            };
        }
        Err(error) => {
            report.native_tools = InspectionReadiness::Unsupported;
            report.issue(
                InspectionIssueCode::UnsupportedToolProtocol,
                InspectionSeverity::Warning,
                error.to_string(),
                Some(report.path.clone()),
            );
        }
    }
    report.issue(
        InspectionIssueCode::RequestSpecificValidation,
        InspectionSeverity::Info,
        "native-tool readiness used a bounded behavioral probe; validate real messages, tool schemas, choices, parallel-call policy, and template kwargs with chat_request",
        Some(report.path.clone()),
    );
}

fn apply_prepared_chat(report: &mut ModelInspectionReport, prepared: &PreparedChat) {
    report.semantic_streaming = match prepared.semantic_support() {
        SemanticSupport::Supported => InspectionReadiness::Ready,
        SemanticSupport::Unsupported { reason } => {
            report.issue(
                InspectionIssueCode::UnsupportedSemanticProtocol,
                InspectionSeverity::Warning,
                reason.clone(),
                Some(report.path.clone()),
            );
            InspectionReadiness::Unsupported
        }
    };
    report.native_tools = match prepared.native_tool_support() {
        NativeToolSupport::Supported => InspectionReadiness::Ready,
        NativeToolSupport::Unsupported { rejection } => {
            report.issue(
                InspectionIssueCode::UnsupportedToolProtocol,
                InspectionSeverity::Warning,
                rejection.to_string(),
                Some(report.path.clone()),
            );
            InspectionReadiness::Unsupported
        }
    };
}

fn finalize_text_readiness(report: &mut ModelInspectionReport) {
    report.text_generation = if report.model_loadability == InspectionReadiness::Ready
        && report.requested_load == InspectionReadiness::Ready
        && report.tokenizer == InspectionReadiness::Ready
    {
        InspectionReadiness::Ready
    } else if report.model_loadability == InspectionReadiness::Invalid
        || report.container == InspectionReadiness::Invalid
    {
        InspectionReadiness::Invalid
    } else if report.model_loadability == InspectionReadiness::Unsupported
        || report.requested_load == InspectionReadiness::Unsupported
    {
        InspectionReadiness::Unsupported
    } else if report.tokenizer == InspectionReadiness::Missing
        || report.model_loadability == InspectionReadiness::Missing
    {
        InspectionReadiness::Missing
    } else {
        InspectionReadiness::Unverified
    };
}
