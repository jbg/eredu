//! Physical validation and canonical bindings for released VL GGUF media projectors.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_checkpoint::{
    gguf_store::GgufWeightStore,
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint},
    store::{SharedCheckpointSource, TensorSelection},
    validation::{resolve_gguf_plan, ResolvedCheckpointPlan},
};
use eredu_gguf::{Checkpoint, GgmlType, MetadataValue};
use eredu_nn::Error;

use super::{Declaration, VisionUnit};

const TEMPORAL_KERNELS: [&str; 2] = ["v.patch_embd.weight", "v.patch_embd.weight.1"];

/// Resolves the released BF16, F16 or Q8_0 media projector against its vision declaration.
pub fn resolve_gguf_projector(
    checkpoint: &Checkpoint,
    declaration: &Declaration,
) -> Result<ResolvedCheckpointPlan, Error> {
    resolve_gguf_plan(checkpoint, &projector_plan(checkpoint, declaration)?).map_err(|validation| {
        Error::backend(format!(
            "LFM3 VL GGUF projector schema differs: {validation:?}"
        ))
    })
}

pub(crate) fn projector_plan(
    checkpoint: &Checkpoint,
    declaration: &Declaration,
) -> Result<GgufCheckpointPlan, Error> {
    media_projector_plan(checkpoint, declaration, "LFM3-VL")
}

pub(in crate::lfm3) fn omni_projector_plan(
    checkpoint: &Checkpoint,
    declaration: &Declaration,
) -> Result<GgufCheckpointPlan, Error> {
    media_projector_plan(checkpoint, declaration, "LFM3-Omni")
}

fn media_projector_plan(
    checkpoint: &Checkpoint,
    declaration: &Declaration,
    basename: &str,
) -> Result<GgufCheckpointPlan, Error> {
    let metadata = checkpoint.metadata();
    let text = |key| metadata.get(key).and_then(MetadataValue::as_str);
    if text("general.architecture") != Some("clip")
        || text("general.type") != Some("mmproj")
        || text("general.basename") != Some(basename)
        || text("clip.vision.projector_type") != Some("lfm3vl")
        || metadata
            .get("clip.has_vision_encoder")
            .and_then(MetadataValue::as_bool)
            != Some(true)
    {
        return Err(Error::backend(
            "expected the released LFM3 GGUF vision projector",
        ));
    }
    let integer = |key| metadata.get(key).and_then(MetadataValue::as_i64);
    let vision = declaration.config();
    for (key, expected) in [
        (
            "clip.vision.projection_dim",
            i64::from(declaration.projector_config().text_hidden_size),
        ),
        ("clip.vision.patch_size", i64::from(vision.patch_size)),
        (
            "clip.vision.embedding_length",
            i64::from(vision.hidden_size),
        ),
        (
            "clip.vision.feed_forward_length",
            i64::from(vision.intermediate_size),
        ),
        (
            "clip.vision.block_count",
            i64::try_from(vision.num_hidden_layers).map_err(Error::backend_source)?,
        ),
        (
            "clip.vision.attention.head_count",
            i64::from(vision.num_attention_heads),
        ),
        (
            "clip.vision.attention.head_count_kv",
            i64::from(vision.num_key_value_heads),
        ),
        (
            "clip.vision.spatial_merge_size",
            i64::from(vision.spatial_merge_size),
        ),
    ] {
        if integer(key) != Some(expected) {
            return Err(Error::backend(format!(
                "LFM3 GGUF vision projector geometry differs at {key}"
            )));
        }
    }
    if vision.temporal_patch_size != 2 {
        return Err(Error::backend(
            "LFM3 GGUF requires two temporal patch kernels",
        ));
    }
    if metadata
        .get("clip.vision.attention.layer_norm_epsilon")
        .and_then(MetadataValue::as_f32)
        != Some(vision.norm_eps)
    {
        return Err(Error::backend(
            "LFM3 GGUF vision normalization epsilon differs",
        ));
    }
    use GgufTypeConstraint::{Exact, OneOf};
    let matrix = match integer("general.file_type") {
        Some(32) => Exact(GgmlType::Bf16),
        Some(1) => Exact(GgmlType::F16),
        Some(7) => Exact(GgmlType::Q8_0),
        Some(2) if basename == "LFM3-Omni" => Exact(GgmlType::Q4_0),
        Some(15) if basename == "LFM3-Omni" => OneOf(vec![GgmlType::Q4K, GgmlType::Q6K]),
        _ => {
            return Err(Error::backend(
                "unsupported LFM3 VL GGUF projector encoding",
            ))
        }
    };
    let mut tensors = Vec::new();
    for unit in declaration.units() {
        for parameter in declaration.unit_parameters(unit)? {
            let id = parameter.parameter.id.as_str();
            let shape = parameter.shape;
            if is_patch_weight(unit, id) {
                let side = vision.patch_size as usize;
                for key in TEMPORAL_KERNELS {
                    tensors.push(GgufTensorConstraint::required(
                        key,
                        [
                            vision.hidden_size as usize,
                            vision.in_channels as usize,
                            side,
                            side,
                        ],
                        GgufTypeConstraint::Exact(GgmlType::F16),
                    ));
                }
                continue;
            }
            let key = physical_name(unit, id)?;
            let encoding = if shape.len() == 2 {
                matrix.clone()
            } else {
                Exact(GgmlType::F32)
            };
            tensors.push(GgufTensorConstraint::required(key, shape, encoding));
        }
    }
    GgufCheckpointPlan::new(
        "LFM3 VL GGUF vision projector",
        tensors,
        vec![],
        CatalogPolicy::strict(),
    )
    .map_err(Error::backend_source)
}

pub(super) fn retain_projector(
    path: &Path,
    declaration: &Declaration,
    max_cached_readers: usize,
) -> Result<
    (
        Declaration,
        SharedCheckpointSource,
        BTreeMap<String, DerivedWeightRecipe>,
    ),
    Error,
> {
    let checkpoint = Checkpoint::open(path).map_err(Error::backend_source)?;
    let plan = projector_plan(&checkpoint, declaration)?;
    retain_vision(checkpoint, declaration, plan, max_cached_readers, None)
}

pub(in crate::lfm3) fn decision_projector_plan(
    checkpoint: &Checkpoint,
    declaration: &Declaration,
    omni: bool,
) -> Result<GgufCheckpointPlan, Error> {
    let mut plan =
        media_projector_plan(checkpoint, declaration, if omni { "d1-omni" } else { "d1" })?;
    if omni {
        plan.common_tensors.extend(
            crate::lfm3::audio::gguf::input_projector_tensors(checkpoint)
                .map_err(Error::backend_source)?,
        );
    }
    Ok(plan)
}

pub(super) fn retain_decision_projector(
    path: &Path,
    declaration: &Declaration,
    audio: Option<&crate::lfm3::audio::Declaration>,
    omni: bool,
    max_cached_readers: usize,
    prepared_cache: Option<&eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy>,
) -> Result<
    (
        Declaration,
        Option<crate::lfm3::audio::Declaration>,
        SharedCheckpointSource,
        BTreeMap<String, DerivedWeightRecipe>,
    ),
    Error,
> {
    let checkpoint = crate::lfm3::checkpoint::packed_gguf_catalog(
        Checkpoint::open(path).map_err(Error::backend_source)?,
    )
    .map_err(Error::backend_source)?;
    let audio = audio
        .map(|declaration| {
            crate::lfm3::audio::gguf::select_input_formats(&checkpoint, declaration)
                .map(|(declaration, _)| declaration)
                .map_err(Error::backend_source)
        })
        .transpose()?;
    let checkpoint = if let Some(declaration) = &audio {
        crate::lfm3::audio::gguf::input_projector_catalog(checkpoint, declaration)
            .map_err(Error::backend_source)?
    } else {
        checkpoint
    };
    let plan = decision_projector_plan(&checkpoint, declaration, omni)?;
    let (vision, source, recipes) = retain_vision(
        checkpoint,
        declaration,
        plan,
        max_cached_readers,
        prepared_cache,
    )?;
    Ok((vision, audio, source, recipes))
}

fn retain_vision(
    checkpoint: Checkpoint,
    declaration: &Declaration,
    plan: GgufCheckpointPlan,
    max_cached_readers: usize,
    prepared_cache: Option<&eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy>,
) -> Result<
    (
        Declaration,
        SharedCheckpointSource,
        BTreeMap<String, DerivedWeightRecipe>,
    ),
    Error,
> {
    let mut declaration = declaration.clone();
    select_projector_formats(&checkpoint, &mut declaration)?;
    let source = retain(
        checkpoint,
        &plan,
        str::to_owned,
        max_cached_readers,
        prepared_cache,
    )?;
    let recipes = projector_recipes(source.as_ref(), &declaration)?;
    Ok((declaration, source, recipes))
}

pub(crate) fn select_projector_formats(
    checkpoint: &Checkpoint,
    declaration: &mut Declaration,
) -> Result<(), Error> {
    let physical = checkpoint
        .shards()
        .iter()
        .flat_map(|shard| {
            shard
                .tensors()
                .iter()
                .map(move |tensor| (tensor.descriptor().name.as_str(), (tensor, shard.endian())))
        })
        .collect::<BTreeMap<_, _>>();
    let mut formats = BTreeMap::new();
    for unit in declaration.units() {
        for parameter in declaration.unit_parameters(unit)? {
            let target = parameter.parameter.id.as_str();
            if parameter.linear_format.is_none() || is_patch_weight(unit, target) {
                continue;
            }
            let key = physical_name(unit, target)?;
            let (tensor, endian) = physical.get(key.as_str()).copied().ok_or_else(|| {
                Error::backend(format!("missing LFM3 GGUF projector tensor {key}"))
            })?;
            let format =
                crate::linear_format::gguf_tensor_format(tensor, endian).map_err(Error::backend)?;
            formats.insert(target.to_owned(), format);
        }
    }
    declaration.select_formats(&formats)
}

pub(crate) fn projector_recipes(
    source: &(impl RecipeCatalog + ?Sized),
    declaration: &Declaration,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let tensor = |key: &str| DerivedWeightRecipe::source(key, TensorSelection::Full);
    let mut recipes = BTreeMap::new();
    for unit in declaration.units() {
        for member in declaration.unit_members(unit)? {
            let id = member.target();
            let recipe = if is_patch_weight(unit, id) {
                // The patch vector is channel-major with time inside each channel.
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::Stack {
                        axis: 2,
                        inputs: TEMPORAL_KERNELS.into_iter().map(tensor).collect(),
                    }),
                    shape: member.global_shape().to_vec(),
                }
            } else {
                tensor(&physical_name(unit, id)?)
            };
            check_shape(&recipe, source, id, member.global_shape())?;
            recipes.insert(id.to_owned(), recipe);
        }
    }
    Ok(recipes)
}

pub(crate) fn vl_backbone_plan(
    checkpoint: &Checkpoint,
    declaration: &crate::lfm3::Declaration,
) -> Result<GgufCheckpointPlan, String> {
    let mut plan = crate::lfm3::gguf_plan(declaration)?;
    let mut nvfp4 = checkpoint
        .tensors()
        .map(|tensor| tensor.descriptor())
        .filter(|descriptor| descriptor.ggml_type == GgmlType::NvFp4)
        .map(|descriptor| descriptor.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    if nvfp4.is_empty() {
        return Ok(plan);
    }
    let mut companions = Vec::new();
    for layer in 0..declaration.args().num_hidden_layers {
        for projection in ["ffn_gate", "ffn_down", "ffn_up"] {
            let root = format!("blk.{layer}.{projection}");
            let weight = format!("{root}.weight");
            if !nvfp4.remove(weight.as_str()) {
                return Err(format!("released LFM3 VL NVFP4 requires {weight}"));
            }
            companions.push(GgufTensorConstraint::required(
                format!("{root}.scale"),
                [1],
                GgufTypeConstraint::Exact(GgmlType::F32),
            ));
            // The FFN graph applies the weight multiplier. The input scale
            // remains checkpoint metadata.
            let mut input = GgufTensorConstraint::required(
                format!("{root}.input_scale"),
                [1],
                GgufTypeConstraint::Exact(GgmlType::F32),
            );
            input.role = eredu_checkpoint::schema::TensorRole::ArtifactMetadata;
            companions.push(input);
        }
    }
    if !nvfp4.is_empty() {
        return Err(format!("unexpected LFM3 VL NVFP4 tensors: {nvfp4:?}"));
    }
    plan.common_tensors.extend(companions);
    plan.catalog_policy = CatalogPolicy::strict();
    Ok(plan)
}

pub(super) fn retain_backbone(
    path: &Path,
    declaration: &crate::lfm3::Declaration,
    max_cached_readers: usize,
) -> Result<
    (
        Arc<crate::lfm3::Declaration>,
        SharedCheckpointSource,
        BTreeMap<String, DerivedWeightRecipe>,
    ),
    Error,
> {
    let checkpoint = Checkpoint::open(path).map_err(Error::backend_source)?;
    if !is_released_backbone(&checkpoint) {
        return Err(Error::backend("expected an LFM3 VL GGUF backbone"));
    }
    validate_backbone_geometry(&checkpoint, declaration)?;
    crate::lfm3::checkpoint::validate_gguf_rope(&checkpoint, declaration.args())
        .map_err(Error::backend)?;
    let plan = vl_backbone_plan(&checkpoint, declaration).map_err(Error::backend)?;
    crate::lfm3::checkpoint::gguf_language_bindings(
        checkpoint,
        &plan,
        declaration,
        max_cached_readers,
    )
}

fn validate_backbone_geometry(
    checkpoint: &Checkpoint,
    declaration: &crate::lfm3::Declaration,
) -> Result<(), Error> {
    let metadata = checkpoint.metadata();
    let metadata = metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let args = crate::lfm3::model_args_from_gguf_catalog(checkpoint, &metadata)
        .map_err(Error::backend_source)?;
    if &args != declaration.args() {
        return Err(Error::backend(
            "LFM3 VL GGUF backbone differs from the supplied decoder configuration",
        ));
    }
    Ok(())
}

pub(in crate::lfm3) fn decision_backbone_plan(
    checkpoint: &Checkpoint,
    declaration: &crate::lfm3::Declaration,
    omni: bool,
) -> Result<GgufCheckpointPlan, Error> {
    use crate::lfm3::audio::gguf::{backbone_plan, AudioGgufEncoding};
    validate_backbone_geometry(checkpoint, declaration)?;
    let encoding = AudioGgufEncoding::from_checkpoint(checkpoint).map_err(Error::backend_source)?;
    if omni && matches!(encoding, AudioGgufEncoding::Q4_0 | AudioGgufEncoding::Q4KM) {
        return Err(Error::backend("unpublished d1 Omni GGUF backbone encoding"));
    }
    backbone_plan(declaration, encoding).map_err(Error::backend_source)
}

pub(super) fn retain_decision_backbone(
    checkpoint: Checkpoint,
    declaration: &crate::lfm3::Declaration,
    omni: bool,
    max_cached_readers: usize,
) -> Result<
    (
        Arc<crate::lfm3::Declaration>,
        SharedCheckpointSource,
        BTreeMap<String, DerivedWeightRecipe>,
    ),
    Error,
> {
    let plan = decision_backbone_plan(&checkpoint, declaration, omni)?;
    crate::lfm3::checkpoint::validate_gguf_rope(&checkpoint, declaration.args())
        .map_err(Error::backend)?;
    crate::lfm3::checkpoint::gguf_language_bindings(
        checkpoint,
        &plan,
        declaration,
        max_cached_readers,
    )
}

pub(crate) use crate::lfm3::checkpoint::gguf_language_recipes as backbone_recipes;

pub(crate) fn is_released_backbone(checkpoint: &Checkpoint) -> bool {
    let text = |key| {
        checkpoint
            .metadata()
            .get(key)
            .and_then(MetadataValue::as_str)
    };
    text("general.architecture") == Some("lfm3")
        && text("general.type") == Some("model")
        && text("general.basename") == Some("LFM3-VL")
}

fn retain(
    checkpoint: Checkpoint,
    plan: &GgufCheckpointPlan,
    translate: impl FnMut(&str) -> String,
    max_cached_readers: usize,
    prepared_cache: Option<&eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy>,
) -> Result<SharedCheckpointSource, Error> {
    let mapping = checkpoint
        .translated_outputs(translate)
        .map_err(Error::backend_source)?;
    Ok(Arc::new(
        GgufWeightStore::builder()
            .require_prepared_decoding(prepared_cache)
            .max_cached_readers(max_cached_readers)
            .and_then(|builder| builder.add_checkpoint(checkpoint, plan, &mapping))
            .and_then(|builder| builder.build())
            .map_err(Error::backend_source)?,
    ))
}

fn check_shape(
    recipe: &DerivedWeightRecipe,
    source: &(impl RecipeCatalog + ?Sized),
    id: &str,
    expected: &[usize],
) -> Result<(), Error> {
    let actual = recipe.infer(source).map_err(Error::backend_source)?.shape;
    if actual != expected {
        return Err(Error::backend(format!(
            "LFM3 VL GGUF binding {id} has shape {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn is_patch_weight(unit: VisionUnit, id: &str) -> bool {
    unit == VisionUnit::Patchifier && id.ends_with(".patchifier.weight")
}

fn physical_name(unit: VisionUnit, id: &str) -> Result<String, Error> {
    let unknown = || Error::backend(format!("no LFM3 VL GGUF projector slot for {id}"));
    let (parent, field) = id.rsplit_once('.').ok_or_else(unknown)?;
    let (parent, module) = parent.rsplit_once('.').ok_or_else(unknown)?;
    let key = match unit {
        VisionUnit::Patchifier if module == "embedding_norm" => "v.embedding_norm".into(),
        VisionUnit::Projector => match module {
            "layer_norm" => "mm.input_norm".into(),
            "linear_1" => "mm.0".into(),
            "linear_2" => "mm.2".into(),
            _ => return Err(unknown()),
        },
        VisionUnit::Block(index) => {
            let slot = match module {
                "operator_norm" => "ln1",
                "ffn_norm" => "ln2",
                "q_proj" => "attn_q",
                "k_proj" => "attn_k",
                "v_proj" => "attn_v",
                "out_proj" if parent.ends_with(".self_attn") => "attn_out",
                "q_layernorm" => "attn_q_norm",
                "k_layernorm" => "attn_k_norm",
                "in_proj" => "conv2d_in_proj",
                "out_proj" => "conv2d_out_proj",
                "spatial_conv" => "conv2d_spatial",
                "w1" => "ffn_gate",
                "w3" => "ffn_up",
                "w2" => "ffn_down",
                _ => return Err(unknown()),
            };
            format!("v.blk.{index}.{slot}")
        }
        _ => return Err(unknown()),
    };
    Ok(format!("{key}.{field}"))
}
