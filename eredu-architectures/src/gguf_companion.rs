//! Backend-neutral admission of architecture-declared GGUF companions.

use std::collections::HashMap;

use eredu_checkpoint::{
    schema::GgufCheckpointPlan, validation::validate_gguf_plan, AffineQuantization,
    WeightQuantization,
};
use eredu_gguf::{Checkpoint, MetadataValue};

use crate::configuration::{GgufArchitecturePlan, GgufModelConfig};

/// Typed family geometry retained for one admitted media-projector companion.
#[derive(Debug, Clone)]
pub enum GgufMediaProjectorConfig {
    /// Content-backed Pixtral/projector companion over the same admitted text source.
    NemotronLabsDiffusion {
        /// Exact image and shared text equations.
        config: crate::nemotron_labs_diffusion::FamilyConfig,
        /// Retained released source receipt and physical geometry.
        vision: crate::nemotron_labs_diffusion::vision_gguf::VisionGgufPlan,
    },
    /// Flash-Next projector with exact tokenizer protocol identities bound.
    Qwen4Exp(crate::qwen4_exp::prepared::VisionPlan),
    /// Header-only Flash-Next projector awaiting tokenizer protocol identities.
    Qwen4ExpPending {
        /// Exact recipes, geometry and prepared source authority.
        vision: crate::qwen4_exp::prepared::GgufVisionPlan,
        /// Exact n-gram reset identity declared by the primary artifact.
        reset_token_id: Option<u32>,
    },
    /// Gemma 4 text plus its admitted vision and/or audio projector.
    Gemma4(crate::gemma4::FamilyConfig),
    /// Inkling text plus its admitted vision/audio projector.
    Inkling(crate::inkling::ModelArgs),
    /// Shared Muse decoder plus its admitted vision and/or audio projector.
    Muse(crate::muse::DecoderConfig),
    /// Qwen3-VL text plus its admitted DeepStack vision projector.
    Qwen3Vl(crate::qwen::vl::ModelArgs),
    /// Structurally admitted Qwen3-VL awaiting facade-resolved media token IDs.
    Qwen3VlPending(crate::qwen::vl::GgufModelArgs),
    /// Qwen3.5 text plus its admitted window-scheduled vision projector.
    Qwen35(crate::qwen::hybrid::ParsedHybridConfig),
    /// Structurally admitted Qwen3.5 awaiting facade-resolved media token IDs.
    Qwen35Pending(crate::qwen::hybrid::ParsedHybridConfig),
}

/// Architecture-owned proof that a resolved media-projector GGUF matches the model.
#[derive(Debug, Clone)]
pub struct GgufMediaProjectorPlan {
    model: GgufMediaProjectorConfig,
    checkpoint: GgufCheckpointPlan,
    primary_tensor_mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    tensor_mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
}

impl GgufMediaProjectorPlan {
    pub(crate) fn replace_muse_execution_config(&mut self, config: crate::muse::DecoderConfig) {
        match &mut self.model {
            GgufMediaProjectorConfig::Muse(current) => *current = config,
            _ => unreachable!("Muse media enrichment retains its admitted family"),
        }
    }

    pub(crate) fn attach_draft_adapter(&mut self, adapter: crate::nemotron_labs_diffusion::adapter::DraftAdapterConfig) -> Result<(), String> {
        match &mut self.model {
            GgufMediaProjectorConfig::NemotronLabsDiffusion { config, .. } => *config = config.clone().with_draft_adapter(adapter),
            _ => return Err("draft adapter cannot attach to this media companion".into()),
        }
        Ok(())
    }

    /// Typed composite family geometry derived from the primary and companion headers.
    pub const fn model(&self) -> &GgufMediaProjectorConfig {
        &self.model
    }

    /// Exact companion checkpoint schema proven by portable inspection.
    pub const fn checkpoint(&self) -> &GgufCheckpointPlan {
        &self.checkpoint
    }

    /// Canonical primary-model mapping in the complete composite namespace.
    pub fn primary_tensor_mapping(&self) -> &[eredu_gguf::TranslatedTensorLayout] {
        &self.primary_tensor_mapping
    }

    /// Canonical physical-to-logical tensor mapping resolved during admission.
    pub fn tensor_mapping(&self) -> &[eredu_gguf::TranslatedTensorLayout] {
        &self.tensor_mapping
    }

    pub(crate) fn bind_qwen_media_tokens(
        &mut self,
        tokens: crate::processor_plan::QwenMediaTokenIds,
    ) -> Result<(), String> {
        let crate::processor_plan::QwenMediaTokenIds {
            image_token_id,
            video_token_id,
            vision_start_token_id,
            vision_end_token_id,
        } = tokens;
        let vocabulary = match &self.model {
            GgufMediaProjectorConfig::Qwen3VlPending(args) => Some(args.text.vocab_size),
            GgufMediaProjectorConfig::Qwen35Pending(args) => Some(args.text.vocab_size),
            _ => None,
        };
        if let Some(vocabulary) = vocabulary {
            let ids = [
                image_token_id,
                video_token_id,
                vision_start_token_id,
                vision_end_token_id,
            ];
            if ids.iter().any(|&id| i64::from(id) >= i64::from(vocabulary))
                || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id))
            {
                return Err(
                    "Qwen media protocol IDs must be distinct and fit the structural vocabulary"
                        .into(),
                );
            }
        }
        let bound = match &self.model {
            GgufMediaProjectorConfig::Qwen4ExpPending {
                vision,
                reset_token_id,
            } => GgufMediaProjectorConfig::Qwen4Exp(
                vision
                    .clone()
                    .bind_media_tokens(
                        crate::qwen4_exp::config::MediaTokens {
                            image: image_token_id,
                            video: video_token_id,
                            start: vision_start_token_id,
                            end: vision_end_token_id,
                        },
                        *reset_token_id,
                    )
                    .map_err(|error| error.to_string())?,
            ),
            GgufMediaProjectorConfig::Qwen3VlPending(args) => GgufMediaProjectorConfig::Qwen3Vl(
                args.clone()
                    .with_media_token_ids(image_token_id, video_token_id)
                    .map_err(|error| error.to_string())?,
            ),
            GgufMediaProjectorConfig::Qwen35Pending(args) => GgufMediaProjectorConfig::Qwen35(
                crate::qwen::hybrid::with_media_token_ids(
                    args.clone(),
                    image_token_id,
                    video_token_id,
                )
                .map_err(|error| error.to_string())?,
            ),
            GgufMediaProjectorConfig::Qwen3Vl(_)
            | GgufMediaProjectorConfig::Qwen35(_)
            | GgufMediaProjectorConfig::Qwen4Exp(_) => {
                return Err("Qwen GGUF media token IDs were already bound".into())
            }
            _ => return Err("non-Qwen GGUF projector received Qwen token IDs".into()),
        };
        self.model = bound;
        Ok(())
    }
}

struct ExactCatalog<'a>(&'a Checkpoint);

impl crate::qwen::vision::VisionGgufCatalog for ExactCatalog<'_> {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.0
            .tensors()
            .find(|tensor| tensor.descriptor().name == name)
            .map(|tensor| tensor.descriptor().row_major_shape())
            .and_then(|shape| {
                shape
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
            })
    }
}

/// Resolves typed companion geometry and validates its exact physical schema.
pub(crate) fn resolve_media_projector(
    primary_plan: &GgufArchitecturePlan,
    primary: &Checkpoint,
    projector: &Checkpoint,
) -> Result<GgufMediaProjectorPlan, String> {
    if let GgufModelConfig::NemotronLabsDiffusion(text) = primary_plan.model() {
        let vision = crate::nemotron_labs_diffusion::vision_gguf::VisionGgufPlan::from_checkpoint(projector, text.config())
            .map_err(|error| error.to_string())?;
        return Ok(GgufMediaProjectorPlan {
            checkpoint: vision.checkpoint().clone(),
            primary_tensor_mapping: text.mapping().to_vec(),
            tensor_mapping: vision.mapping().to_vec(),
            model: GgufMediaProjectorConfig::NemotronLabsDiffusion { config: text.config().clone(), vision },
        });
    }
    let model_metadata = metadata(primary);
    let projector_metadata = metadata(projector);
    let (model, checkpoint) = match primary_plan.model() {
        GgufModelConfig::Qwen4Exp(target) => {
            let vision = crate::qwen4_exp::prepared::GgufVisionPlan::prepare(
                target.text_plan().config(),
                projector,
            )
            .map_err(|error| error.to_string())?;
            let reset_token_id = model_metadata
                .get("qwen4exp.ple.image_token_id")
                .map(|value| {
                    value
                        .as_i64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or_else(|| "invalid qwen4exp.ple.image_token_id".to_owned())
                })
                .transpose()?;
            let checkpoint = vision.gguf_source().checkpoint_plan().clone();
            (
                GgufMediaProjectorConfig::Qwen4ExpPending {
                    vision,
                    reset_token_id,
                },
                checkpoint,
            )
        }
        GgufModelConfig::Gemma4(family) => {
            let family = crate::gemma4::family_from_gguf_metadata(
                family.text.clone(),
                &model_metadata,
                Some(&projector_metadata),
            )
            .map_err(|error| error.to_string())?;
            let checkpoint = crate::gemma4::mmproj_gguf_plan(&family)?;
            (GgufMediaProjectorConfig::Gemma4(family), checkpoint)
        }
        GgufModelConfig::Inkling(args) => {
            let formats = weight_formats(projector, crate::inkling::translate_mmproj_weight_name)?;
            let args = args
                .clone()
                .with_gguf_projector_metadata(&model_metadata, &projector_metadata, formats)
                .map_err(|error| error.to_string())?;
            let checkpoint = crate::inkling::mmproj_gguf_plan(&args)?;
            (GgufMediaProjectorConfig::Inkling(args), checkpoint)
        }
        GgufModelConfig::Muse(args) => {
            let audio = if projector_metadata.get("clip.has_audio_encoder")
                .and_then(MetadataValue::as_bool) == Some(true) {
                Some(crate::muse::AudioConfig::from_gguf_metadata(&projector_metadata, args.hidden_size)
                    .map_err(|error| error.to_string())?)
            } else { None };
            let formats = weight_formats(
                projector,
                |name| crate::muse::translate_projector_gguf_name(name, audio.as_ref()),
            )?;
            let args = args
                .clone()
                .with_gguf_projector_metadata(&projector_metadata, formats)
                .map_err(|error| error.to_string())?;
            let checkpoint = crate::muse::projector_gguf_plan(&args)?;
            (GgufMediaProjectorConfig::Muse(args), checkpoint)
        }
        GgufModelConfig::Qwen(text) => {
            let mut vision = crate::qwen::vl::vision_config_from_gguf_catalog(
                &ExactCatalog(projector),
                &projector_metadata,
            )
            .map_err(|error| error.to_string())?;
            qwen_vision_checkpoint_formats(&mut vision, projector)?;
            let model =
                crate::qwen::vl::model_args_from_gguf_parts(text.clone(), &model_metadata, vision)
                    .map_err(|error| error.to_string())?;
            let checkpoint = crate::qwen::vision::gguf_plan(&model.vision, model.text.hidden_size)?;
            (GgufMediaProjectorConfig::Qwen3VlPending(model), checkpoint)
        }
        GgufModelConfig::QwenHybrid(parsed) => {
            let mut vision = crate::qwen::hybrid::vision_config_from_gguf_catalog(
                &ExactCatalog(projector),
                &projector_metadata,
            )
            .map_err(|error| error.to_string())?;
            qwen_vision_checkpoint_formats(&mut vision, projector)?;
            let model = crate::qwen::hybrid::with_gguf_vision_projector(parsed.clone(), vision)
                .map_err(|error| error.to_string())?;
            let checkpoint = crate::qwen::hybrid::conditional_projector_gguf_plan(&model)?;
            (GgufMediaProjectorConfig::Qwen35Pending(model), checkpoint)
        }
        _ => {
            return Err(format!(
                "GGUF architecture {:?} does not admit a media-projector companion",
                primary_plan.architecture()
            ))
        }
    };
    validate_gguf_plan(projector, &checkpoint)
        .into_loader_result()
        .map_err(|failure| strict_failure("media-projector GGUF", failure))?;
    let primary_tensor_mapping = match &model {
        GgufMediaProjectorConfig::Qwen3Vl(model) => primary
            .translated_outputs(|name| {
                crate::qwen::vl::translate_text_gguf_weight_name(name, model.text.is_moe())
            })
            .map_err(|error| error.to_string())?,
        GgufMediaProjectorConfig::Qwen3VlPending(model) => primary
            .translated_outputs(|name| {
                crate::qwen::vl::translate_text_gguf_weight_name(name, model.text.is_moe())
            })
            .map_err(|error| error.to_string())?,
        _ => primary_plan.tensor_mapping().to_vec(),
    };
    let tensor_mapping = canonical_projector_mapping(projector, &model)?;
    Ok(GgufMediaProjectorPlan {
        model,
        checkpoint,
        primary_tensor_mapping,
        tensor_mapping,
    })
}

fn canonical_projector_mapping(
    projector: &Checkpoint,
    model: &GgufMediaProjectorConfig,
) -> Result<Vec<eredu_gguf::TranslatedTensorLayout>, String> {
    let mapping = match model {
        GgufMediaProjectorConfig::NemotronLabsDiffusion { vision, .. } => return Ok(vision.mapping().to_vec()),
        GgufMediaProjectorConfig::Qwen4ExpPending { vision, .. } => {
            return Ok(vision.gguf_source().mapping().to_vec());
        }
        GgufMediaProjectorConfig::Qwen4Exp(vision) => {
            return vision
                .gguf_source()
                .map(|source| source.mapping().to_vec())
                .ok_or_else(|| "GGUF projector lost its exact source declaration".into());
        }
        GgufMediaProjectorConfig::Gemma4(_) => {
            projector.translated_outputs(crate::gemma4::translate_mmproj_weight_name)
        }
        GgufMediaProjectorConfig::Inkling(_) => {
            projector.translated_outputs(crate::inkling::translate_mmproj_weight_name)
        }
        GgufMediaProjectorConfig::Muse(args) => {
            projector.translated_outputs(|name| crate::muse::translate_projector_gguf_name(name, args.audio_config.as_ref()))
        }
        GgufMediaProjectorConfig::Qwen3Vl(model) => {
            let deepstack = model.vision.deepstack_layers();
            projector.translated_outputs(|name| {
                crate::qwen::vl::translate_vision_gguf_weight_name(name, &deepstack)
            })
        }
        GgufMediaProjectorConfig::Qwen3VlPending(model) => {
            let deepstack = model.vision.deepstack_layers();
            projector.translated_outputs(|name| {
                crate::qwen::vl::translate_vision_gguf_weight_name(name, &deepstack)
            })
        }
        GgufMediaProjectorConfig::Qwen35(model)
        | GgufMediaProjectorConfig::Qwen35Pending(model) => {
            let deepstack = model
                .vision
                .as_ref()
                .ok_or("admitted Qwen3.5 projector omitted its vision geometry")?
                .deepstack_layers();
            projector.translated_outputs(|name| {
                crate::qwen::hybrid::translate_vision_gguf_weight_name(name, &deepstack)
            })
        }
    };
    mapping.map_err(|error| error.to_string())
}

fn metadata(checkpoint: &Checkpoint) -> HashMap<String, MetadataValue> {
    checkpoint
        .metadata()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn qwen_vision_checkpoint_formats(
    vision: &mut crate::qwen::vision::VisionConfig,
    checkpoint: &Checkpoint,
) -> Result<(), String> {
    let deepstack = vision.deepstack_layers();
    vision.linear_formats = weight_formats(checkpoint, |name| {
        crate::qwen::vision::translate_gguf_weight_name(name, &deepstack)
    })?
    .into_iter()
    .map(|(name, format)| (name, format.into()))
    .collect();
    Ok(())
}

fn weight_formats(
    checkpoint: &Checkpoint,
    mut translate: impl FnMut(&str) -> String,
) -> Result<HashMap<String, WeightQuantization>, String> {
    let mut formats = HashMap::new();
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            let descriptor = tensor.descriptor();
            let format = if let Some((bits, group_size)) = tensor.affine() {
                let group_size = i32::try_from(group_size)
                    .map_err(|_| format!("GGUF group size {group_size} exceeds i32"))?;
                Some(WeightQuantization::Affine(
                    AffineQuantization::new(group_size, i32::from(bits))
                        .map_err(|error| error.to_string())?,
                ))
            } else if tensor.is_mxfp4() {
                Some(WeightQuantization::MxFp4)
            } else if descriptor.ggml_type.has_native_execution() {
                Some(WeightQuantization::GgufIQuant {
                    ggml_type: descriptor.ggml_type,
                    endian: shard.endian(),
                })
            } else {
                None
            };
            let Some(format) = format else {
                continue;
            };
            let source = tensor
                .outputs()
                .first()
                .map(|output| output.name.as_str())
                .unwrap_or(descriptor.name.as_str());
            let name = translate(source);
            if formats.insert(name.clone(), format).is_some() {
                return Err(format!("GGUF tensors collide after translating {name:?}"));
            }
        }
    }
    Ok(formats)
}

fn strict_failure(
    identity: &str,
    failure: eredu_checkpoint::validation::StrictLoadFailure,
) -> String {
    let mut details = failure
        .missing
        .into_iter()
        .map(|name| format!("missing {name:?}"))
        .collect::<Vec<_>>();
    details.extend(failure.unused);
    format!("invalid {identity}: {}", details.join("; "))
}
