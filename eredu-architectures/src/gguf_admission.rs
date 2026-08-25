//! Complete backend-neutral GGUF family admission.

use std::collections::{BTreeSet, HashMap};

use eredu_checkpoint::validation::{
    validate_gguf_plan, CheckpointIssue, CheckpointIssueKind, CheckpointValidation,
};
use eredu_gguf::{Checkpoint, MetadataValue};

use crate::GgufArchitecture;

struct ExactCatalog<'a>(&'a Checkpoint);

impl ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0
            .tensors()
            .any(|tensor| tensor.descriptor().name == name)
    }

    fn any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
        self.0
            .tensors()
            .any(|tensor| predicate(&tensor.descriptor().name))
    }
}

impl crate::deepseek::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }
}

impl crate::gemma4::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }
}

impl crate::kimi_linear::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        ExactCatalog::any(self, predicate)
    }
}

impl crate::lfm2::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        ExactCatalog::any(self, predicate)
    }
}

impl crate::llama::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }

    fn any(&self, predicate: &mut dyn FnMut(&str) -> bool) -> bool {
        ExactCatalog::any(self, predicate)
    }
}

impl crate::muse_glimmer::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }
}

impl crate::nemotron_h::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        ExactCatalog::any(self, predicate)
    }
}

impl crate::qwen::GgufTensorCatalog for ExactCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        ExactCatalog::contains(self, name)
    }
}

pub(crate) fn validate(
    architecture: GgufArchitecture,
    checkpoint: &Checkpoint,
) -> CheckpointValidation {
    match validate_family(architecture, checkpoint) {
        Ok(validation) => validation,
        Err(detail) => invalid_geometry(detail),
    }
}

fn validate_family(
    architecture: GgufArchitecture,
    checkpoint: &Checkpoint,
) -> Result<CheckpointValidation, String> {
    let metadata = checkpoint
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<HashMap<String, MetadataValue>>();
    let catalog = ExactCatalog(checkpoint);
    match architecture {
        GgufArchitecture::DeepSeek2 => {
            let args = crate::deepseek::parse_v3_gguf(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            translated(checkpoint, crate::deepseek::translate_v3_gguf_weight_name)?;
            plan(checkpoint, crate::deepseek::v3_gguf_plan(&args))
        }
        GgufArchitecture::DeepSeek4 => {
            let args =
                crate::deepseek::parse_v4_gguf(&metadata).map_err(|error| error.to_string())?;
            translated(checkpoint, crate::deepseek::translate_v4_gguf_weight_name)?;
            plan(checkpoint, crate::deepseek::v4_gguf_plan(&args))
        }
        GgufArchitecture::Gemma4 => {
            let names = checkpoint
                .logical_outputs()
                .map(|output| output.name.clone())
                .collect::<BTreeSet<_>>();
            let text = crate::gemma4::ModelArgs::from_gguf_metadata(&names, &metadata)
                .map_err(|error| error.to_string())?;
            let family = crate::gemma4::family_from_gguf_metadata(text, &metadata, None)
                .map_err(|error| error.to_string())?;
            plan(checkpoint, crate::gemma4::gguf_plan(&family.text))
        }
        GgufArchitecture::GptOss => {
            translated(checkpoint, crate::gpt_oss::translate_gguf_weight_name)?;
            let args = crate::gpt_oss::model_args_from_gguf_catalog(&metadata)
                .map_err(|error| error.to_string())?;
            Ok(crate::gpt_oss::validate_gguf(checkpoint, &args))
        }
        GgufArchitecture::Inkling => {
            let args = crate::inkling::ModelArgs::from_gguf_metadata(&metadata)
                .map_err(|error| error.to_string())?;
            plan(checkpoint, crate::inkling::gguf_plan(&args))
        }
        GgufArchitecture::KimiLinear => {
            let args = crate::kimi_linear::model_args_from_gguf_catalog(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            translated(checkpoint, crate::kimi_linear::translate_gguf_weight_name)?;
            plan(checkpoint, crate::kimi_linear::gguf_plan(&args))
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let args = crate::lfm2::model_args_from_gguf_catalog(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            let is_moe = args.has_sparse_moe_layers();
            translated(checkpoint, |name| {
                crate::lfm2::translate_gguf_weight_name(name, is_moe)
            })?;
            plan(checkpoint, crate::lfm2::gguf_plan(&args))
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let args = crate::llama::model_args_from_gguf_catalog(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            if args.num_hidden_layers as usize > checkpoint.physical_tensor_count() {
                return Err(format!(
                    "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
                    args.num_hidden_layers,
                    checkpoint.physical_tensor_count()
                ));
            }
            translated(checkpoint, crate::llama::translate_gguf_weight_name)?;
            plan(
                checkpoint,
                crate::llama::gguf_plan(&args).map_err(|error| error.to_string()),
            )
        }
        GgufArchitecture::MuseGlimmer => {
            let args = crate::muse_glimmer::DecoderConfig::from_gguf_catalog(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            plan(checkpoint, crate::muse_glimmer::gguf_plan(&args))
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let args = crate::nemotron_h::model_args_from_gguf_catalog(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            translated(checkpoint, crate::nemotron_h::translate_gguf_weight_name)?;
            plan(checkpoint, crate::nemotron_h::gguf_plan(&args))
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let args = crate::qwen::model_args_from_gguf_catalog(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            translated(checkpoint, |name| {
                crate::qwen::translate_gguf_weight_name(name, args.is_moe())
            })?;
            plan(checkpoint, crate::qwen::gguf_plan(&args))
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            let is_moe = architecture == GgufArchitecture::Qwen3VlMoe;
            let context = if is_moe {
                crate::qwen::TextConfigContext::Qwen3VlMoe
            } else {
                crate::qwen::TextConfigContext::Qwen3Vl
            };
            let args = crate::qwen::model_args_from_gguf_catalog_with_context(
                &catalog, &metadata, context,
            )
            .map_err(|error| error.to_string())?;
            if args.is_moe() != is_moe {
                return Err("Qwen3-VL GGUF architecture and expert geometry disagree".into());
            }
            translated(checkpoint, |name| {
                crate::qwen::translate_gguf_weight_name(name, is_moe)
            })?;
            plan(checkpoint, crate::qwen::gguf_plan(&args))
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            let parsed = crate::qwen::hybrid::model_args_from_gguf_catalog(&catalog, &metadata)
                .map_err(|error| error.to_string())?;
            translated(checkpoint, crate::qwen::hybrid::translate_gguf_weight_name)?;
            plan(checkpoint, crate::qwen::hybrid::gguf_plan(&parsed.text))
        }
    }
}

fn translated(
    checkpoint: &Checkpoint,
    translate: impl FnMut(&str) -> String,
) -> Result<(), String> {
    checkpoint
        .translated_outputs(translate)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn plan(
    checkpoint: &Checkpoint,
    plan: Result<eredu_checkpoint::schema::GgufCheckpointPlan, String>,
) -> Result<CheckpointValidation, String> {
    Ok(validate_gguf_plan(checkpoint, &plan?))
}

fn invalid_geometry(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}
