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
    /// Gemma 4 text plus its admitted vision and/or audio projector.
    Gemma4(crate::gemma4::FamilyConfig),
    /// Inkling text plus its admitted vision/audio projector.
    Inkling(crate::inkling::ModelArgs),
    /// Muse-Glimmer decoder plus its admitted vision projector.
    MuseGlimmer(crate::muse_glimmer::DecoderConfig),
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

    pub(crate) fn bind_qwen_token_ids(
        &mut self,
        image_token_id: u32,
        video_token_id: u32,
    ) -> Result<(), String> {
        let bound = match &self.model {
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
            GgufMediaProjectorConfig::Qwen3Vl(_) | GgufMediaProjectorConfig::Qwen35(_) => {
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
    let model_metadata = metadata(primary);
    let projector_metadata = metadata(projector);
    let (model, checkpoint) = match primary_plan.model() {
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
        GgufModelConfig::MuseGlimmer(args) => {
            let formats = weight_formats(
                projector,
                crate::muse_glimmer::translate_projector_gguf_name,
            )?;
            let args = args
                .clone()
                .with_gguf_projector_metadata(&projector_metadata, formats)
                .map_err(|error| error.to_string())?;
            let checkpoint = crate::muse_glimmer::projector_gguf_plan(&args)?;
            (GgufMediaProjectorConfig::MuseGlimmer(args), checkpoint)
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
        GgufMediaProjectorConfig::Gemma4(_) => {
            projector.translated_outputs(crate::gemma4::translate_mmproj_weight_name)
        }
        GgufMediaProjectorConfig::Inkling(_) => {
            projector.translated_outputs(crate::inkling::translate_mmproj_weight_name)
        }
        GgufMediaProjectorConfig::MuseGlimmer(_) => {
            projector.translated_outputs(crate::muse_glimmer::translate_projector_gguf_name)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufArchitecture;
    use eredu_gguf::{GgmlType, MetadataArray, TensorInput, Writer};
    use std::{collections::BTreeMap, fs::File};

    fn muse_plan() -> GgufArchitecturePlan {
        struct Catalog;

        impl crate::GgufTensorCatalog for Catalog {
            fn contains(&self, name: &str) -> bool {
                name == "output.weight"
            }

            fn any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
                predicate("output.weight")
            }
        }

        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("muse-glimmer".into()),
            ),
            ("muse-glimmer.block_count".into(), MetadataValue::Uint32(2)),
            (
                "muse-glimmer.embedding_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "muse-glimmer.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "muse-glimmer.attention.head_count_kv".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "muse-glimmer.attention.key_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "muse-glimmer.feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "muse-glimmer.vocab_size".into(),
                MetadataValue::Uint32(200_100),
            ),
            (
                "muse-glimmer.context_length".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "muse-glimmer.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            (
                "muse-glimmer.attention.sliding_window".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "muse-glimmer.logit_scale".into(),
                MetadataValue::Float32(0.25),
            ),
            (
                "muse-glimmer.final_logit_softcapping".into(),
                MetadataValue::Float32(20.0),
            ),
            (
                "muse-glimmer.attention.sliding_window_pattern".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![true, false])),
            ),
        ]);
        let args =
            crate::muse_glimmer::DecoderConfig::from_gguf_catalog(&Catalog, &metadata).unwrap();
        let checkpoint = crate::muse_glimmer::gguf_plan(&args).unwrap();
        GgufArchitecturePlan::new(
            GgufArchitecture::MuseGlimmer,
            GgufModelConfig::MuseGlimmer(args),
            checkpoint,
            Vec::new(),
        )
    }

    fn write_checkpoint(
        metadata: &BTreeMap<String, MetadataValue>,
    ) -> (tempfile::TempDir, Checkpoint) {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("projector.gguf");
        let scalar = 1.0_f32.to_le_bytes();
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                metadata,
                &[TensorInput {
                    name: "placeholder.weight",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &scalar,
                }],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(path).unwrap();
        (root, checkpoint)
    }

    fn muse_projector_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            (
                "general.type".into(),
                MetadataValue::String("mmproj".into()),
            ),
            (
                "clip.projector_type".into(),
                MetadataValue::String("muse-glimmer".into()),
            ),
            ("clip.has_vision_encoder".into(), MetadataValue::Bool(true)),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(1_536),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                MetadataValue::Uint32(8_960),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(50)),
            (
                "clip.vision.attention.head_count".into(),
                MetadataValue::Uint32(16),
            ),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(14)),
            (
                "clip.vision.spatial_merge_size".into(),
                MetadataValue::Uint32(2),
            ),
            ("clip.vision.image_size".into(), MetadataValue::Uint32(896)),
            (
                "clip.vision.projection_dim".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
        ])
    }

    #[test]
    fn portable_admission_rejects_wrong_companion_family() {
        let mut metadata = muse_projector_metadata();
        metadata.insert(
            "clip.projector_type".into(),
            MetadataValue::String("other-family".into()),
        );
        let (_root, checkpoint) = write_checkpoint(&metadata);
        let error = resolve_media_projector(&muse_plan(), &checkpoint, &checkpoint).unwrap_err();
        assert!(error.contains("clip.projector_type"));
    }

    #[test]
    fn portable_admission_rejects_companion_schema_mismatch() {
        let (_root, checkpoint) = write_checkpoint(&muse_projector_metadata());
        let error = resolve_media_projector(&muse_plan(), &checkpoint, &checkpoint).unwrap_err();
        assert!(error.contains("invalid media-projector GGUF"), "{error}");
        assert!(error.contains("missing"), "{error}");
    }

    #[test]
    fn qwen_vl_projector_mapping_preserves_both_temporal_recipe_sources() {
        use eredu_checkpoint::{recipe::RecipeCatalog, store::TensorMetadata, StoredDtype};
        let args = crate::qwen::vl::model_args_from_config_value(&serde_json::json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":24,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap();
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("temporal.gguf");
        let first = vec![1_f32.to_le_bytes(); 16 * 3 * 2 * 2].concat();
        let second = vec![2_f32.to_le_bytes(); 16 * 3 * 2 * 2].concat();
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "v.patch_embd.weight",
                        dimensions: &[2, 2, 3, 16],
                        ggml_type: GgmlType::F32,
                        data: &first,
                    },
                    TensorInput {
                        name: "v.patch_embd.weight.1",
                        dimensions: &[2, 2, 3, 16],
                        ggml_type: GgmlType::F32,
                        data: &second,
                    },
                ],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(path).unwrap();
        struct Catalog(BTreeMap<String, TensorMetadata>);
        impl RecipeCatalog for Catalog {
            fn tensor_metadata(
                &self,
                key: &str,
            ) -> Result<TensorMetadata, eredu_checkpoint::store::StoreError> {
                self.0.get(key).cloned().ok_or_else(|| {
                    eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() }
                })
            }
        }
        for model in [
            GgufMediaProjectorConfig::Qwen3VlPending(crate::qwen::vl::GgufModelArgs {
                text: args.text.clone(),
                vision: args.vision.clone(),
                mrope_section: args.mrope_section,
                model_type: args.model_type.clone(),
            }),
            GgufMediaProjectorConfig::Qwen3Vl(args),
        ] {
            let mapping = canonical_projector_mapping(&checkpoint, &model).unwrap();
            let catalog = Catalog(
                mapping
                    .iter()
                    .map(|mapped| {
                        let shape = mapped
                            .layout
                            .shape
                            .iter()
                            .map(|&d| d as usize)
                            .collect::<Vec<_>>();
                        (
                            mapped.layout.name.clone(),
                            TensorMetadata {
                                name: mapped.layout.name.clone(),
                                logical_shape: shape.clone(),
                                physical_shape: shape,
                                stored_dtype: StoredDtype::F32,
                                encoded_byte_len: first.len() as u64,
                                backing_shard: None,
                            },
                        )
                    })
                    .collect(),
            );
            let recipes = crate::qwen::vl::static_recipes(&catalog);
            let recipe = &recipes["model.visual.patch_embed.proj.weight"];
            assert_eq!(
                recipe.source_keys(),
                [
                    "model.visual.patch_embed.proj.weight.0",
                    "model.visual.patch_embed.proj.weight.1",
                ]
            );
            assert_eq!(recipe.infer(&catalog).unwrap().shape, [16, 3, 2, 2, 2]);
        }
    }

    #[test]
    fn portable_weight_formats_cover_affine_mxfp4_and_native_ggml() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("formats.gguf");
        let affine = vec![0_u8; 18];
        let mxfp4 = vec![0_u8; 17];
        let native = vec![0_u8; 144];
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "affine.weight",
                        dimensions: &[32],
                        ggml_type: GgmlType::Q4_0,
                        data: &affine,
                    },
                    TensorInput {
                        name: "mxfp4.weight",
                        dimensions: &[32],
                        ggml_type: GgmlType::MxFp4,
                        data: &mxfp4,
                    },
                    TensorInput {
                        name: "native.weight",
                        dimensions: &[256],
                        ggml_type: GgmlType::Q4K,
                        data: &native,
                    },
                ],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(path).unwrap();
        let formats = weight_formats(&checkpoint, str::to_owned).unwrap();
        assert!(matches!(
            formats["affine.weight"],
            WeightQuantization::Affine(_)
        ));
        assert_eq!(formats["mxfp4.weight"], WeightQuantization::MxFp4);
        assert!(matches!(
            formats["native.weight"],
            WeightQuantization::GgufIQuant {
                ggml_type: GgmlType::Q4K,
                ..
            }
        ));
    }

    #[test]
    fn qwen_projector_packed_formats_reach_full_and_deepstack_linear_bindings() {
        let mut args = crate::qwen::vl::model_args_from_config_value(&serde_json::json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":32,"intermediate_size":64,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap();
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("packed-projector.gguf");
        let qkv = vec![1u8; 96 * 34];
        let merger = vec![1u8; 128 * 4 * 18];
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "v.blk.0.attn_qkv.weight",
                        dimensions: &[32, 96],
                        ggml_type: GgmlType::Q8_0,
                        data: &qkv,
                    },
                    TensorInput {
                        name: "v.deepstack.0.fc1.weight",
                        dimensions: &[128, 128],
                        ggml_type: GgmlType::IQ4NL,
                        data: &merger,
                    },
                ],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(path).unwrap();
        qwen_vision_checkpoint_formats(&mut args.vision, &checkpoint).unwrap();
        for (path, ggml_type) in [
            ("blocks.0.attn.qkv.weight", GgmlType::Q8_0),
            ("deepstack_merger_list.0.linear_fc1.weight", GgmlType::IQ4NL),
        ] {
            let expected = eredu_checkpoint::LinearFormat::GgufIQuant {
                ggml_type,
                endian: eredu_gguf::Endian::Little,
            };
            assert_eq!(args.vision.linear_format(path), expected);
            assert_eq!(
                args.vision.linear_format(&format!("model.visual.{path}")),
                expected
            );
        }
        assert_eq!(
            args.vision.linear_format("patch_embed.proj.weight"),
            eredu_checkpoint::LinearFormat::Dense
        );
    }
}
