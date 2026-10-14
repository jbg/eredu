//! Content-backed admission of the published Pixtral/projector companion.
use super::{ConfigError, FamilyConfig};
use crate::decoder::Config;
use eredu_checkpoint::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint, TensorRole,
};
use eredu_gguf::{Checkpoint, GgmlType, MetadataValue};

/// Exact separately addressable vision source; no generic `clip` alias authority.
#[derive(Debug, Clone)]
pub struct VisionGgufPlan {
    fingerprints: std::sync::Arc<eredu_checkpoint::artifact::ArtifactFingerprintSource>,
    checkpoint: GgufCheckpointPlan,
    resolution: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
}
impl VisionGgufPlan {
    /// Opens the pinned companion and checks its complete fingerprint, geometry,
    /// processor metadata, and tensor encodings before native materialization.
    pub fn open(
        path: impl AsRef<std::path::Path>,
        config: &FamilyConfig,
    ) -> Result<(Checkpoint, Self), ConfigError> {
        let invalid = |message: String| ConfigError::Invalid(message);
        let primary_guard = eredu_checkpoint::artifact::ArtifactFingerprintSource::new([
            eredu_checkpoint::artifact::ArtifactFile::new("gguf.vision", path.as_ref()),
        ])
        .map_err(|e| invalid(e.to_string()))?;
        let checkpoint = Checkpoint::open(path.as_ref()).map_err(|e| invalid(e.to_string()))?;
        primary_guard
            .validate_unchanged()
            .map_err(|e| invalid(e.to_string()))?;
        let plan = Self::from_checkpoint(&checkpoint, config)?;
        Ok((checkpoint, plan))
    }
    /// Checks the already-open source against the published companion receipt.
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        config: &FamilyConfig,
    ) -> Result<Self, ConfigError> {
        let invalid = |message: String| ConfigError::Invalid(message);
        let fingerprints = eredu_checkpoint::artifact::ArtifactFingerprintSource::new(
            checkpoint
                .shards()
                .iter()
                .enumerate()
                .map(|(index, shard)| {
                    eredu_checkpoint::artifact::ArtifactFile::new(
                        format!("gguf.vision.{index}"),
                        shard.path(),
                    )
                }),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let members = fingerprints
            .fingerprint()
            .map_err(|e| invalid(e.to_string()))?;
        let member = &members[0];
        let digest = member
            .digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if members.len() != 1
            || member.length() != 857_077_888
            || digest != "0705006bdaac2a3e330a9e0b80767653fce5a07fedd99995c869082b0278ff75"
        {
            let digest = eredu_checkpoint::gguf_fingerprint::fingerprint_gguf_catalog(checkpoint)
                .map_err(|e| invalid(e.to_string()))?;
            let hash = digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            if hash != "195f488cd774120cd1812734984764936fa25d7bfee25f5bc675176683d4495c" {
                return Err(invalid(
                    "vision GGUF has no retained Nemotron companion provenance".into(),
                ));
            }
        }
        Self::admit(checkpoint, config, std::sync::Arc::new(fingerprints))
    }

    fn admit(
        checkpoint: &Checkpoint,
        config: &FamilyConfig,
        fingerprints: std::sync::Arc<eredu_checkpoint::artifact::ArtifactFingerprintSource>,
    ) -> Result<Self, ConfigError> {
        let invalid = |message: String| ConfigError::Invalid(message);
        config
            .validate_config()
            .map_err(|e| invalid(e.to_string()))?;
        let vision = config
            .vision
            .as_ref()
            .ok_or_else(|| invalid("vision companion requires a Pixtral configuration".into()))?;
        if config.hidden_size() != 4096
            || config.spatial_merge_size != 2
            || vision.hidden_size != 1024
            || vision.intermediate_size != 4096
            || vision.num_hidden_layers != 24
            || vision.num_attention_heads != 16
            || vision.head_dim != 64
            || vision.num_channels != 3
            || vision.patch_size != 14
            || vision.image_size != 1540
            || vision.rope_parameters["rope_theta"].as_f64() != Some(10000.0)
        {
            return Err(invalid(
                "vision companion differs from pinned Pixtral geometry".into(),
            ));
        }
        let metadata = checkpoint.metadata();
        for (key, expected) in [
            ("general.architecture", "clip"),
            ("general.type", "mmproj"),
            ("clip.projector_type", "pixtral"),
        ] {
            if metadata.get(key).and_then(MetadataValue::as_str) != Some(expected) {
                return Err(invalid(format!(
                    "vision GGUF {key} differs from the pinned companion"
                )));
            }
        }
        for key in ["clip.has_vision_encoder", "clip.use_silu"] {
            if metadata.get(key).and_then(MetadataValue::as_bool) != Some(true) {
                return Err(invalid(format!(
                    "vision GGUF {key} differs from Pixtral equations"
                )));
            }
        }
        for (key, expected) in [
            ("clip.vision.projection_dim", 4096),
            ("clip.vision.image_size", 1540),
            ("clip.vision.patch_size", 14),
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.feed_forward_length", 4096),
            ("clip.vision.block_count", 24),
            ("clip.vision.attention.head_count", 16),
            ("clip.vision.spatial_merge_size", 2),
        ] {
            if metadata.get(key).and_then(MetadataValue::as_i64) != Some(expected) {
                return Err(invalid(format!(
                    "vision GGUF {key} differs from admitted geometry"
                )));
            }
        }
        let epsilon = metadata
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(MetadataValue::as_f32);
        if epsilon.is_none_or(|x| !x.is_finite() || (x - 1e-5).abs() > 1e-11) {
            return Err(invalid(
                "vision GGUF RMS epsilon differs from Pixtral equations".into(),
            ));
        }
        let dense = super::safetensors_plan(config).map_err(invalid)?;
        let mut tensors = Vec::new();
        for tensor in dense.common_tensors.into_iter().filter(|tensor| {
            tensor.key.starts_with("model.vision_tower.")
                || tensor.key.starts_with("model.multi_modal_projector.")
        }) {
            let physical = physical_name(&tensor.key)
                .ok_or_else(|| invalid(format!("no Pixtral GGUF source for {}", tensor.key)))?;
            tensors.push(GgufTensorConstraint::required(
                physical,
                tensor.shape.clone(),
                GgufTypeConstraint::Exact(if tensor.shape.len() == 1 {
                    GgmlType::F32
                } else {
                    GgmlType::F16
                }),
            ));
        }
        let mut marker = GgufTensorConstraint::required(
            "v.token_embd.img_break",
            vec![4096],
            GgufTypeConstraint::Exact(GgmlType::F32),
        );
        marker.role = TensorRole::ArtifactMetadata;
        tensors.push(marker);
        let plan = GgufCheckpointPlan::new(
            "Nemotron-Labs-Diffusion Pixtral companion",
            tensors,
            Vec::new(),
            CatalogPolicy::strict(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let resolution = eredu_checkpoint::validation::resolve_gguf_plan(checkpoint, &plan)
            .map_err(|issues| invalid(format!("Pixtral GGUF tensor admission: {issues:?}")))?;
        let mapping = checkpoint
            .translated_outputs(|key| {
                let canonical = canonical_name(key).unwrap_or_else(|| key.to_owned());
                if let Some((index, field)) = vision_pair(&canonical) {
                    format!("nemotron.gguf.pixtral.layers.{index}.{field}.weight")
                } else {
                    canonical
                }
            })
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            fingerprints,
            checkpoint: plan,
            resolution,
            mapping,
        })
    }
    /// Constructs exact lazy sources and checks continuity with the hashed
    /// member after the store captures guards for all future bounded leases.
    pub fn into_source(
        self,
        checkpoint: Checkpoint,
        max_cached_readers: usize,
    ) -> Result<eredu_checkpoint::gguf_store::GgufWeightStore, ConfigError> {
        let source = eredu_checkpoint::gguf_store::GgufWeightStoreBuilder::default()
            .max_cached_readers(max_cached_readers)
            .and_then(|builder| {
                builder.add_resolved_checkpoint(checkpoint, &self.resolution, &self.mapping)
            })
            .and_then(|builder| builder.build())
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        self.fingerprints
            .validate_unchanged()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        Ok(source)
    }
    /// Strict physical companion catalog, including validated marker metadata.
    pub fn checkpoint(&self) -> &GgufCheckpointPlan {
        &self.checkpoint
    }
    /// Exact resolved companion roles.
    pub fn resolution(&self) -> &eredu_checkpoint::validation::ResolvedCheckpointPlan {
        &self.resolution
    }
    /// Canonical source identities; Q/K retain explicit conversion coordinates.
    pub fn mapping(&self) -> &[eredu_gguf::TranslatedTensorLayout] {
        &self.mapping
    }
}

fn physical_name(canonical: &str) -> Option<String> {
    for (source, target) in direct_names() {
        if canonical == target {
            return Some(source.into());
        }
    }
    let rest = canonical.strip_prefix("model.vision_tower.transformer.layers.")?;
    let (index, field) = rest.split_once('.')?;
    index.parse::<usize>().ok()?;
    for (source, target) in layer_names() {
        if field == format!("{target}.weight") {
            return Some(format!("v.blk.{index}.{source}.weight"));
        }
    }
    None
}
fn canonical_name(source: &str) -> Option<String> {
    for (key, target) in direct_names() {
        if source == key {
            return Some(target.into());
        }
    }
    let rest = source.strip_prefix("v.blk.")?;
    let (index, field) = rest.split_once('.')?;
    index.parse::<usize>().ok()?;
    for (key, target) in layer_names() {
        if field == format!("{key}.weight") {
            return Some(format!(
                "model.vision_tower.transformer.layers.{index}.{target}.weight"
            ));
        }
    }
    None
}
pub(crate) fn vision_pair(canonical: &str) -> Option<(&str, &str)> {
    let rest = canonical.strip_prefix("model.vision_tower.transformer.layers.")?;
    let (index, suffix) = rest.split_once('.')?;
    let field = match suffix {
        "attention.q_proj.weight" => "q_proj",
        "attention.k_proj.weight" => "k_proj",
        _ => return None,
    };
    Some((index, field))
}
fn direct_names() -> [(&'static str, &'static str); 7] {
    [
        (
            "v.patch_embd.weight",
            "model.vision_tower.patch_conv.weight",
        ),
        ("v.pre_ln.weight", "model.vision_tower.ln_pre.weight"),
        ("mm.1.weight", "model.multi_modal_projector.linear_1.weight"),
        ("mm.2.weight", "model.multi_modal_projector.linear_2.weight"),
        (
            "mm.input_norm.weight",
            "model.multi_modal_projector.norm.weight",
        ),
        (
            "mm.patch_merger.weight",
            "model.multi_modal_projector.patch_merger.merging_layer.weight",
        ),
        (
            "v.token_embd.img_break",
            "nemotron.gguf.image_break_metadata",
        ),
    ]
}
fn layer_names() -> [(&'static str, &'static str); 9] {
    [
        ("attn_q", "attention.q_proj"),
        ("attn_k", "attention.k_proj"),
        ("attn_v", "attention.v_proj"),
        ("attn_out", "attention.o_proj"),
        ("ln1", "attention_norm"),
        ("ln2", "ffn_norm"),
        ("ffn_gate", "feed_forward.gate_proj"),
        ("ffn_up", "feed_forward.up_proj"),
        ("ffn_down", "feed_forward.down_proj"),
    ]
}
