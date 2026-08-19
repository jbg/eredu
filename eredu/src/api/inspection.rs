//! Backend-independent tokenizer, chat-template, and semantic inspection enrichment.

use std::path::Path;

use eredu_core::{
    ArtifactFormat, InspectionIssue, InspectionIssueCode, InspectionReadiness, InspectionSeverity,
    ModelInspectionReport,
};
use eredu_gguf::MetadataValue as GgufMetadataValue;
use serde_json::{json, Map, Value};

use super::load_tokenizer;
use super::metadata::{
    eos_token_ids_from_sidecar_dir, gguf_eos_token_ids, merge_eos_token_id_sources,
};
use super::request::prepare_chat_from_parts;
use super::tokenizer::{
    gguf_sidecar_dir, load_chat_template, load_gguf_tokenizer_from_metadata,
    load_tokenizer_template_kwargs,
};
use crate::runtime::chat::{
    constraints::ConstraintCompiler, ChatTemplateRequest, NativeToolSupport, PreparedChat,
    SemanticSupport, ToolChoice,
};
use eredu_text::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};

/// Portable text-sidecar checks applied after a selected backend inspects an artifact.
#[derive(Debug, Clone, Default)]
pub struct TextInspectionOptions {
    /// Optional concrete chat request to render and behaviorally probe.
    pub chat_request: Option<ChatTemplateRequest>,
}

/// Enriches a backend inspection report with tokenizer, chat-template, EOS,
/// semantic-streaming, and native-tool readiness.
///
/// This function does not inspect tensor structure or materialize weights. A
/// selected backend must produce the input report first.
pub fn inspect_text_model(
    mut report: ModelInspectionReport,
    options: TextInspectionOptions,
) -> ModelInspectionReport {
    let path = report.path.clone();
    match report.artifact_format {
        ArtifactFormat::SafeTensors => {
            inspect_safetensors_sidecars(&mut report, &path, options.chat_request);
        }
        ArtifactFormat::Gguf => match eredu_gguf::Reader::open(&path) {
            Ok(reader) => {
                let metadata = reader
                    .metadata()
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                inspect_gguf_sidecars(&mut report, &path, &metadata, options.chat_request);
            }
            Err(error) => {
                report.tokenizer = InspectionReadiness::Invalid;
                report.chat_template = InspectionReadiness::Unverified;
                report.semantic_streaming = InspectionReadiness::Unverified;
                report.native_tools = InspectionReadiness::Unverified;
                report.issue(
                    InspectionIssueCode::Io,
                    InspectionSeverity::Error,
                    format!("could not reopen GGUF metadata for text inspection: {error}"),
                    Some(path),
                );
            }
        },
    }
    finalize_text_readiness(&mut report);
    report
}

fn inspect_safetensors_sidecars(
    report: &mut ModelInspectionReport,
    path: &Path,
    request: Option<ChatTemplateRequest>,
) {
    let tokenizer = match load_tokenizer(path) {
        Ok(tokenizer) => {
            report.tokenizer = InspectionReadiness::Ready;
            Some(tokenizer)
        }
        Err(error) => {
            report.tokenizer = InspectionReadiness::Missing;
            report.issue(
                InspectionIssueCode::MissingTokenizer,
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.join("tokenizer.json")),
            );
            None
        }
    };
    let template = match load_chat_template(path) {
        Ok(Some(template)) => {
            report.chat_template = InspectionReadiness::Ready;
            Some(template)
        }
        Ok(None) => {
            report.chat_template = InspectionReadiness::Missing;
            report.issue(
                InspectionIssueCode::MissingChatTemplate,
                InspectionSeverity::Warning,
                "no tokenizer_config.json chat_template or chat_template.jinja is available",
                Some(path.to_path_buf()),
            );
            None
        }
        Err(error) => {
            report.chat_template = InspectionReadiness::Invalid;
            report.issue(
                InspectionIssueCode::MissingChatTemplate,
                InspectionSeverity::Warning,
                error.to_string(),
                Some(path.to_path_buf()),
            );
            None
        }
    };
    if let (Some(tokenizer), Some(template)) = (tokenizer, template) {
        let kwargs = load_tokenizer_template_kwargs(path).unwrap_or_default();
        let eos = eos_token_ids_from_sidecar_dir(path).unwrap_or_default();
        inspect_chat_behavior(report, tokenizer, template, kwargs, eos, request);
    } else {
        report.semantic_streaming = InspectionReadiness::Missing;
        report.native_tools = InspectionReadiness::Missing;
    }
}

fn inspect_gguf_sidecars(
    report: &mut ModelInspectionReport,
    path: &Path,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    request: Option<ChatTemplateRequest>,
) {
    let tokenizer = match load_gguf_tokenizer_from_metadata(path, metadata) {
        Ok(tokenizer) => {
            report.tokenizer = InspectionReadiness::Ready;
            Some(tokenizer)
        }
        Err(error) => {
            report.tokenizer = InspectionReadiness::Missing;
            report.issue(
                InspectionIssueCode::MissingTokenizer,
                InspectionSeverity::Error,
                format!(
                    "GGUF tokenizer metadata is unusable and no acceptable sibling tokenizer.json was loaded: {error}"
                ),
                Some(gguf_sidecar_dir(path).join("tokenizer.json")),
            );
            None
        }
    };
    let embedded = match metadata.get("tokenizer.chat_template") {
        Some(GgufMetadataValue::String(template)) => {
            Some(ModelChatTemplate::Single(template.clone()))
        }
        Some(_) => {
            report.chat_template = InspectionReadiness::Invalid;
            report.issues.push(InspectionIssue {
                code: InspectionIssueCode::MissingChatTemplate,
                severity: InspectionSeverity::Warning,
                detail: "GGUF tokenizer.chat_template must be a string".into(),
                path: Some(path.to_path_buf()),
                metadata_key: Some("tokenizer.chat_template".into()),
                tensor_name: None,
                tensor_type_code: None,
            });
            None
        }
        None => None,
    };
    let template = embedded.or_else(|| load_chat_template(gguf_sidecar_dir(path)).ok().flatten());
    if template.is_some() {
        report.chat_template = InspectionReadiness::Ready;
    } else if report.chat_template != InspectionReadiness::Invalid {
        report.chat_template = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingChatTemplate,
            InspectionSeverity::Warning,
            "GGUF has no embedded chat template and no acceptable sidecar template",
            Some(path.to_path_buf()),
        );
    }
    if let (Some(tokenizer), Some(template)) = (tokenizer, template) {
        let eos = merge_eos_token_id_sources([
            eos_token_ids_from_sidecar_dir(gguf_sidecar_dir(path)).unwrap_or_default(),
            gguf_eos_token_ids(metadata).unwrap_or_default(),
        ]);
        inspect_chat_behavior(
            report,
            tokenizer.tokenizer,
            template,
            tokenizer.template_kwargs,
            eos,
            request,
        );
    } else {
        report.semantic_streaming = InspectionReadiness::Missing;
        report.native_tools = InspectionReadiness::Missing;
    }
}

fn inspect_chat_behavior(
    report: &mut ModelInspectionReport,
    tokenizer: tokenizers::Tokenizer,
    template: ModelChatTemplate,
    kwargs: Map<String, Value>,
    eos_token_ids: Vec<u32>,
    request: Option<ChatTemplateRequest>,
) {
    let mut tokenizer = ChatTokenizer::from_tokenizer(tokenizer);
    tokenizer.set_template_kwargs(kwargs);
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos_token_ids);
    let model_id = report.path.display().to_string();
    if let Some(request) = request {
        match prepare_chat_from_parts(
            &mut tokenizer,
            template,
            &model_id,
            &eos_token_ids,
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
        messages: vec![json!({"role": "user", "content": "__safemlx_inspection_probe__"})],
        tool_choice: ToolChoice::None,
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    };
    match prepare_chat_from_parts(
        &mut tokenizer,
        template.clone(),
        &model_id,
        &eos_token_ids,
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
        messages: vec![json!({"role": "user", "content": "__safemlx_tool_probe__"})],
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "safemlx_probe",
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
        &mut tokenizer,
        template,
        &model_id,
        &eos_token_ids,
        Some(&compiler),
        tool_request,
    ) {
        Ok(prepared) => {
            report.native_tools = match prepared.native_tool_support() {
                NativeToolSupport::Supported => InspectionReadiness::Ready,
                NativeToolSupport::Unsupported { reason } => {
                    report.issue(
                        InspectionIssueCode::UnsupportedToolProtocol,
                        InspectionSeverity::Warning,
                        reason.clone(),
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
        NativeToolSupport::Unsupported { reason } => {
            report.issue(
                InspectionIssueCode::UnsupportedToolProtocol,
                InspectionSeverity::Warning,
                reason.clone(),
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
