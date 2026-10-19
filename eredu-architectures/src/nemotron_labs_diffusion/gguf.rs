//! Exact pinned text conversions. Generic GGUF architecture aliases carry no authority.
use super::{ConfigError, FamilyConfig};
use crate::decoder::Config;
use eredu_checkpoint::{
    artifact::ArtifactMemberFingerprint, schema::GgufCheckpointPlan,
    validation::ResolvedCheckpointPlan,
};
use eredu_gguf::{Checkpoint, MetadataValue};

/// Content-backed semantic authority for a published conversion with incomplete metadata.
/// Exact payload or complete encoded-catalog fingerprints construct this receipt.
#[derive(Debug, Clone)]
pub struct TextConversionProvenance {
    repository: &'static str,
    revision: &'static str,
    size: u64,
    digest: [u8; 32],
    hidden: i32,
    layers: i32,
    intermediate: i32,
    theta: f64,
    adjacent_pairs: bool,
    vlm: bool,
    members: Vec<ArtifactMemberFingerprint>,
    catalog_digest: Option<[u8; 32]>,
}
impl TextConversionProvenance {
    /// Matches exact admitted bytes, not publisher names or generic architecture tags.
    pub fn from_fingerprint(member: &ArtifactMemberFingerprint) -> Result<Self, ConfigError> {
        let hash = member
            .digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let mut provenance = Self::from_published_digest(&hash, member.digest())?;
        if member.length() != provenance.size {
            return Err(ConfigError::Invalid(
                "GGUF payload length differs from the pinned conversion".into(),
            ));
        }
        provenance.members.push(member.clone());
        Ok(provenance)
    }
    fn from_published_digest(hash: &str, digest: [u8; 32]) -> Result<Self, ConfigError> {
        let vlm = hash == "4b5f15a545e5272b587a5150ad37cc6b5e338dc8352693a11973d1fa8f4862c7";
        let (repository, revision, size, hidden, layers, intermediate, theta, adjacent_pairs) =
            match hash {
                "d25119a965e4781b5f1d4b5b2cf446e4102d949d9752d86144b94820368fa4d1" => (
                    "spiritbuun/Nemotron-Labs-Diffusion-14B-Q8_0-GGUF",
                    "7ec2bb277055ffbbcc8cb7e56e179216d3f4952d",
                    14_359_313_600,
                    5120,
                    40,
                    16384,
                    1e9,
                    false,
                ),
                "2473a97abba7effed8bd0baf49beed09d988c45c3834885eca12565cce77132e" => (
                    "suzukimain/Nemotron-Labs-Diffusion-14B-GGUF",
                    "f9fd26730ff276b02361c0f776085f60a122ec70",
                    8_239_070_432,
                    5120,
                    40,
                    16384,
                    1e9,
                    false,
                ),
                "f0a4db3dc37dc80de075eb36554f25ec80147d6f0e4bfc65ecfc42147b321224" => (
                    "IntellAgents/Nemotron-Labs-Diffusion-3B-GGUF",
                    "b1367a57265c62f63e18f0faacaac202731ee0c4",
                    2_372_993_280,
                    3072,
                    26,
                    9216,
                    1e6,
                    true,
                ),
                "4b5f15a545e5272b587a5150ad37cc6b5e338dc8352693a11973d1fa8f4862c7" => (
                    "IntellAgents/Nemotron-Labs-Diffusion-VLM-8B-GGUF",
                    "a436a745feda80389c99c7a15ba3cc46fbfee8e0",
                    16_987_579_808,
                    4096,
                    34,
                    14336,
                    1e6,
                    true,
                ),
                _ => {
                    return Err(ConfigError::Invalid(
                        "GGUF payload has no retained Nemotron conversion provenance".into(),
                    ));
                }
            };
        Ok(Self {
            repository,
            revision,
            size,
            digest,
            hidden,
            layers,
            intermediate,
            theta,
            adjacent_pairs,
            vlm,
            members: Vec::new(),
            catalog_digest: None,
        })
    }
    fn from_catalog_digest(
        digest: [u8; 32],
        members: Vec<ArtifactMemberFingerprint>,
    ) -> Result<Self, ConfigError> {
        let hash = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let published = match hash.as_str() {
            "8150e7be3f0ccf2fa5fd698c7c14253b5b43dc073faf0e95c0960c9375542b40" => {
                "d25119a965e4781b5f1d4b5b2cf446e4102d949d9752d86144b94820368fa4d1"
            }
            "d12b236198fc2bdb8b1927a966e7b36e95b6431d16313bf332b637fb6969c4ea" => {
                "f0a4db3dc37dc80de075eb36554f25ec80147d6f0e4bfc65ecfc42147b321224"
            }
            "3c80acd729e5d43cec1e5bc5372f73f725365623e830883edb42043f9d76389f" => {
                "2473a97abba7effed8bd0baf49beed09d988c45c3834885eca12565cce77132e"
            }
            "7136058da3f2648c3b5331b2334d471709d7efc735ec2b78fc9efd6f630d54cb" => {
                "4b5f15a545e5272b587a5150ad37cc6b5e338dc8352693a11973d1fa8f4862c7"
            }
            _ => {
                return Err(ConfigError::Invalid(
                    "GGUF catalog has no retained Nemotron conversion provenance".into(),
                ));
            }
        };
        let mut bytes = [0; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&published[index * 2..index * 2 + 2], 16)
                .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        }
        let mut provenance = Self::from_published_digest(published, bytes)?;
        provenance.members = members;
        provenance.catalog_digest = Some(digest);
        Ok(provenance)
    }
    /// Pinned published repository.
    pub fn repository(&self) -> &'static str {
        self.repository
    }
    /// Pinned published revision.
    pub fn revision(&self) -> &'static str {
        self.revision
    }
    /// Original published payload size and SHA256; split membership is separate.
    pub fn published_artifact(&self) -> (u64, [u8; 32]) {
        (self.size, self.digest)
    }
    /// Complete actual single-file or split source fingerprints.
    pub fn source_members(&self) -> &[ArtifactMemberFingerprint] {
        &self.members
    }
    /// Logical identity proving an exact repack of the published tensors/tokenizer.
    pub fn catalog_digest(&self) -> Option<[u8; 32]> {
        self.catalog_digest
    }
}

/// Retained configuration, exact tensor contract and resolved physical naming.
#[derive(Debug, Clone)]
pub struct TextGgufPlan {
    fingerprints: std::sync::Arc<eredu_checkpoint::artifact::ArtifactFingerprintSource>,
    source: Checkpoint,
    config: FamilyConfig,
    text_config: FamilyConfig,
    checkpoint: GgufCheckpointPlan,
    resolution: ResolvedCheckpointPlan,
    mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    provenance: TextConversionProvenance,
}
impl TextGgufPlan {
    /// Opens and fingerprints the same cold source before deriving legacy
    /// semantics. Quantized matrices use bounded F32 decoding; stored vectors
    /// retain their precision until the selected binding recipe casts them.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<(Checkpoint, Self), ConfigError> {
        let primary_guard = eredu_checkpoint::artifact::ArtifactFingerprintSource::new([
            eredu_checkpoint::artifact::ArtifactFile::new("gguf.primary", path.as_ref()),
        ])
        .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        // The admitted 14B embedding/readout each decode to 2,684,354,560
        // bytes. Conversion stays bounded by a matrix, not the complete stack.
        let limits = eredu_gguf::Limits {
            max_allocation_bytes: 3 << 30,
            ..Default::default()
        };
        let checkpoint = Checkpoint::open_with_representation(
            path.as_ref(),
            limits,
            eredu_gguf::QuantizedTensorRepresentation::DecodedF32,
        )
        .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        primary_guard
            .validate_unchanged()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let plan = Self::from_checkpoint(&checkpoint)?;
        Ok((checkpoint, plan))
    }
    /// Admits the exact already-open catalog and projects conversion headers only.
    /// Complete released artifact hashes or shard-independent catalog hashes prove
    /// semantics; a generic architecture tag alone never grants admission.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self, ConfigError> {
        let names = checkpoint
            .shards()
            .iter()
            .flat_map(|shard| shard.tensors())
            .filter_map(|tensor| {
                tensor
                    .descriptor()
                    .ggml_type
                    .block_and_bytes()
                    .ok()
                    .filter(|(block, _)| *block > 1)
                    .map(|_| tensor.descriptor().name.clone())
            })
            .collect::<Vec<_>>();
        let checkpoint = checkpoint
            .clone()
            .into_tensor_representation(
                names,
                eredu_gguf::QuantizedTensorRepresentation::DecodedF32,
            )
            .and_then(|source| source.into_allocation_limit(3 << 30))
            .map_err(|error| ConfigError::Invalid(error.to_string()))?;
        let fingerprints = eredu_checkpoint::artifact::ArtifactFingerprintSource::new(
            checkpoint
                .shards()
                .iter()
                .enumerate()
                .map(|(index, shard)| {
                    eredu_checkpoint::artifact::ArtifactFile::new(
                        format!("gguf.shard.{index}"),
                        shard.path(),
                    )
                }),
        )
        .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let members = fingerprints
            .fingerprint()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let direct = (members.len() == 1)
            .then(|| TextConversionProvenance::from_fingerprint(&members[0]))
            .transpose();
        let provenance = match direct {
            Ok(Some(provenance)) => provenance,
            _ => {
                let digest =
                    eredu_checkpoint::gguf_fingerprint::fingerprint_gguf_catalog(&checkpoint)
                        .map_err(|e| ConfigError::Invalid(e.to_string()))?;
                TextConversionProvenance::from_catalog_digest(digest, members)?
            }
        };
        Self::admit(&checkpoint, provenance, std::sync::Arc::new(fingerprints))
    }
    /// Header-only converted source retained by total prepared construction.
    pub fn prepared_checkpoint(&self) -> &Checkpoint {
        &self.source
    }
    /// Strictly validates published geometry and encoded tensors before materialization.
    fn admit(
        checkpoint: &Checkpoint,
        provenance: TextConversionProvenance,
        fingerprints: std::sync::Arc<eredu_checkpoint::artifact::ArtifactFingerprintSource>,
    ) -> Result<Self, ConfigError> {
        let invalid = |message: String| ConfigError::Invalid(message);
        let metadata = checkpoint.metadata();
        let namespace = metadata
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            .ok_or_else(|| invalid("missing GGUF architecture namespace".into()))?;
        if namespace
            != if provenance.adjacent_pairs {
                "mistral3"
            } else {
                "dream"
            }
        {
            return Err(invalid(
                "GGUF namespace differs from its content-backed provenance".into(),
            ));
        }
        let integer = |suffix: &str, expected: i64| -> Result<(), ConfigError> {
            let key = format!("{namespace}.{suffix}");
            if metadata.get(&key).and_then(MetadataValue::as_i64) != Some(expected) {
                return Err(invalid(format!(
                    "GGUF {key} differs from pinned source geometry"
                )));
            }
            Ok(())
        };
        for (suffix, expected) in [
            ("embedding_length", i64::from(provenance.hidden)),
            ("block_count", i64::from(provenance.layers)),
            ("feed_forward_length", i64::from(provenance.intermediate)),
            ("context_length", 262144),
            ("attention.head_count", 32),
            ("attention.head_count_kv", 8),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("rope.scaling.original_context_length", 16384),
        ] {
            integer(suffix, expected)?;
        }
        for (suffix, expected) in [
            ("rope.freq_base", provenance.theta),
            ("rope.scaling.factor", 16.0),
            ("rope.scaling.yarn_beta_fast", 32.0),
            ("rope.scaling.yarn_beta_slow", 1.0),
            ("attention.layer_norm_rms_epsilon", 1e-5),
        ] {
            let key = format!("{namespace}.{suffix}");
            let value = metadata
                .get(&key)
                .and_then(MetadataValue::as_f32)
                .ok_or_else(|| invalid(format!("missing floating GGUF {key}")))?;
            if !value.is_finite() || (f64::from(value) - expected).abs() > expected.abs() * 1e-6 {
                return Err(invalid(format!(
                    "GGUF {key} differs from pinned source equations"
                )));
            }
        }
        if metadata
            .get(&format!("{namespace}.rope.scaling.type"))
            .and_then(MetadataValue::as_str)
            != Some("yarn")
        {
            return Err(invalid("Nemotron conversion requires YaRN".into()));
        }
        for (key, expected) in [
            ("tokenizer.ggml.mask_token_id", 100),
            ("tokenizer.ggml.bos_token_id", 1),
            ("tokenizer.ggml.eos_token_id", 11),
        ] {
            if metadata.get(key).and_then(MetadataValue::as_i64) != Some(expected) {
                return Err(invalid(format!(
                    "GGUF {key} violates the released tokenizer policy"
                )));
            }
        }
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(MetadataValue::as_strings)
            .ok_or_else(|| invalid("missing complete GGUF vocabulary".into()))?;
        if tokens.len() != if provenance.vlm { 131073 } else { 131072 }
            || tokens[1] != "<s>"
            || tokens[11] != "<|im_end|>"
            || tokens[100] != "<SPECIAL_100>"
            || (provenance.vlm && tokens[131072] != "|<MASK>|")
        {
            return Err(invalid(
                "GGUF vocabulary extent or reserved identities disagree".into(),
            ));
        }
        if provenance.adjacent_pairs {
            for (key, expected) in [
                ("mistral3.attention.temperature_scale", 0.1),
                ("mistral3.rope.scaling.yarn_log_multiplier", 1.0),
            ] {
                if metadata
                    .get(key)
                    .and_then(MetadataValue::as_f32)
                    .is_none_or(|value| !value.is_finite() || (value - expected).abs() > 1e-6)
                {
                    return Err(invalid(format!(
                        "GGUF {key} disagrees with published query/rotary scaling"
                    )));
                }
            }
            if metadata
                .get("nemotron_labs.source_architecture")
                .and_then(MetadataValue::as_str)
                != Some(if provenance.vlm {
                    "NemotronLabsDiffusionVLMModel"
                } else {
                    "NemotronLabsDiffusionModel"
                })
                || metadata
                    .get("nemotron_labs.diffusion.block_length")
                    .and_then(MetadataValue::as_i64)
                    != Some(32)
                || metadata
                    .get("nemotron_labs.diffusion.shift_logits")
                    .and_then(MetadataValue::as_bool)
                    != Some(false)
                || metadata
                    .get("nemotron_labs.diffusion.seed_first_token")
                    .and_then(MetadataValue::as_bool)
                    != Some(!provenance.vlm)
            {
                return Err(invalid(
                    "GGUF does not declare exact Nemotron masked-token semantics".into(),
                ));
            }
        } else if metadata
            .get("dream.attention.causal")
            .and_then(MetadataValue::as_bool)
            != Some(false)
            || metadata
                .get("diffusion.shift_logits")
                .and_then(MetadataValue::as_bool)
                != Some(false)
        {
            return Err(invalid(
                "GGUF diffusion semantics differ from the pinned legacy conversion".into(),
            ));
        }
        let mut source = serde_json::json!({"model_type":"nemotron_labs_diffusion","hidden_size":provenance.hidden,"num_hidden_layers":provenance.layers,"intermediate_size":provenance.intermediate,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"vocab_size":if provenance.vlm {131073} else {131072},"rms_norm_eps":1e-5,"max_position_embeddings":262144,"block_size":32,"mask_token_id":100,"hidden_act":"silu","dlm_paradigm":"bidirectional","tie_word_embeddings":false,"attention_bias":false,"mlp_bias":false,"rope_traditional":provenance.adjacent_pairs,"rope_parameters":{"rope_type":"yarn","factor":16.0,"rope_theta":provenance.theta,"original_max_position_embeddings":16384,"beta_fast":32.0,"beta_slow":1.0,"mscale":1.0,"mscale_all_dim":1.0,"llama_4_scaling_beta":0.1}});
        if provenance.vlm {
            source["model_type"] = serde_json::json!("nemotron_labs_diffusion_vlm");
            source["spatial_merge_size"] = serde_json::json!(2);
            source["projector_hidden_act"] = serde_json::json!("gelu");
            source["multimodal_projector_bias"] = serde_json::json!(false);
            source["vision_feature_layer"] = serde_json::json!(-1);
            source["vision_config"] = serde_json::json!({"model_type":"pixtral","hidden_size":1024,"intermediate_size":4096,"num_hidden_layers":24,"num_attention_heads":16,"head_dim":64,"num_channels":3,"patch_size":14,"image_size":1540,"hidden_act":"silu","rope_parameters":{"rope_type":"default","rope_theta":10000.0}});
        }
        let config = FamilyConfig::from_hf_value(&source)?;
        config
            .validate_config()
            .map_err(|e| invalid(e.to_string()))?;
        let plan = crate::llama::gguf_plan(config.dense()).map_err(invalid)?;
        let resolution = eredu_checkpoint::validation::resolve_gguf_plan(checkpoint, &plan)
            .map_err(|issues| invalid(format!("Nemotron GGUF tensor admission: {issues:?}")))?;
        let mapping = checkpoint
            .translated_outputs(crate::llama::translate_gguf_weight_name)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            fingerprints,
            source: checkpoint.clone(),
            text_config: {
                let mut text = config.clone();
                text.vision = None;
                text
            },
            config,
            checkpoint: plan,
            resolution,
            mapping,
            provenance,
        })
    }
    /// Binds the retained exact resolution and validates fingerprint continuity
    /// after the store captures the guards used by future bounded leases.
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
    pub(crate) fn with_draft_adapter(
        mut self,
        adapter: super::adapter::DraftAdapterConfig,
    ) -> Self {
        self.config = self.config.with_draft_adapter(adapter.clone());
        self.text_config = self.text_config.with_draft_adapter(adapter);
        self
    }

    /// One normalized geometry for every shared-backbone invocation.
    pub fn config(&self) -> &FamilyConfig {
        &self.config
    }
    /// Text-only execution retains vocabulary and equations without advertising image ingress.
    pub fn text_config(&self) -> &FamilyConfig {
        &self.text_config
    }
    /// Strict physical catalog contract.
    pub fn checkpoint(&self) -> &GgufCheckpointPlan {
        &self.checkpoint
    }
    /// Exact complete tensor admission.
    pub fn resolution(&self) -> &ResolvedCheckpointPlan {
        &self.resolution
    }
    /// Canonical names over the retained physical source.
    pub fn mapping(&self) -> &[eredu_gguf::TranslatedTensorLayout] {
        &self.mapping
    }
    /// Source evidence retained with the normalized equations.
    pub fn provenance(&self) -> &TextConversionProvenance {
        &self.provenance
    }
}
