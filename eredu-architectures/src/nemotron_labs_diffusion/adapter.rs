//! Admission and topology of the optional published draft-only LoRA.
use super::FamilyConfig;
use crate::decoder::Config;
use eredu_checkpoint::{
    StoredDtype,
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
};
use eredu_nn::{Error, NeuralBackend, ParameterSpec, Tensor, low_rank::LowRankProjection};

/// Validated bias-free, ordinary low-rank output projection adapter.
#[derive(Debug, Clone)]
pub struct DraftAdapterConfig {
    rank: i32,
    scale: f32,
    base: String,
}
impl DraftAdapterConfig {
    /// Validates the PEFT equations and exact base ancestry before source admission.
    pub fn from_json(bytes: &[u8], expected_base: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if value["peft_type"] != "LORA"
            || value["base_model_name_or_path"] != expected_base
            || !expected_base.starts_with("nvidia/Nemotron-Labs-Diffusion-")
            || value["bias"] != "none"
            || value["target_modules"] != serde_json::json!(["o_proj"])
            || ["use_dora", "use_rslora", "fan_in_fan_out", "lora_bias"]
                .iter()
                .any(|key| value[*key].as_bool() != Some(false))
            || value["lora_dropout"].as_f64() != Some(0.0)
            || ["alpha_pattern", "rank_pattern"]
                .iter()
                .any(|key| value[*key].as_object().is_none_or(|v| !v.is_empty()))
            || [
                "modules_to_save",
                "layer_replication",
                "layers_to_transform",
                "layers_pattern",
                "exclude_modules",
                "eva_config",
                "megatron_config",
            ]
            .iter()
            .any(|key| value.get(*key).is_some_and(|v| !v.is_null()))
            || value
                .get("loftq_config")
                .is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|v| !v.is_empty()))
        {
            return Err("draft adapter ancestry or PEFT equations differ from the admitted low-rank contract".into());
        }
        let rank = value["r"]
            .as_i64()
            .and_then(|r| i32::try_from(r).ok())
            .filter(|r| *r > 0)
            .ok_or("invalid draft adapter rank")?;
        let alpha = value["lora_alpha"]
            .as_f64()
            .filter(|a| a.is_finite() && *a > 0.0)
            .ok_or("invalid draft adapter alpha")?;
        let scale = (alpha / f64::from(rank)) as f32;
        if !scale.is_finite() || scale <= 0.0 {
            return Err("invalid draft adapter scale".into());
        }
        Ok(Self {
            rank,
            scale,
            base: expected_base.into(),
        })
    }
    /// Declared base artifact identity, distinct from adapter storage identity.
    pub fn base_artifact(&self) -> &str {
        &self.base
    }
    /// Complete replicated rank width.
    pub fn rank(&self) -> i32 {
        self.rank
    }
    /// Ordinary alpha/rank multiplier.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub(crate) fn layer<B: NeuralBackend>(
        &self,
        index: usize,
        input: i32,
        output: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LowRankProjection<B>, Error> {
        let name = format!("draft_adapter.layers.{index}.o_proj");
        LowRankProjection::new(
            input,
            output,
            self.rank,
            self.scale,
            ParameterSpec::trainable(format!("{name}.lora_A.weight")).map_err(Error::backend)?,
            ParameterSpec::trainable(format!("{name}.lora_B.weight")).map_err(Error::backend)?,
            context,
        )
    }
}

/// Strict separate-parameter admission; no merged or duplicate base weight is accepted.
pub fn safetensors_plan(
    config: &FamilyConfig,
    adapter: &DraftAdapterConfig,
) -> Result<SafetensorsCheckpointPlan, String> {
    config.validate_config().map_err(|e| e.to_string())?;
    let mut tensors = Vec::new();
    for layer in 0..config.num_hidden_layers() {
        for (suffix, shape) in [
            (
                "lora_A",
                vec![
                    adapter.rank as usize,
                    (config.num_attention_heads() * config.head_dim()) as usize,
                ],
            ),
            (
                "lora_B",
                vec![config.hidden_size() as usize, adapter.rank as usize],
            ),
        ] {
            let key = format!("draft_adapter.layers.{layer}.o_proj.{suffix}.weight");
            let mut tensor = SafetensorsTensorConstraint::required(
                key,
                shape,
                StoredDtypeConstraint::OneOf(vec![
                    StoredDtype::F32,
                    StoredDtype::F16,
                    StoredDtype::BF16,
                ]),
            );
            tensor.aliases.push(format!(
                "base_model.model.encoder.layers.{layer}.self_attn.o_proj.{suffix}.weight"
            ));
            tensors.push(tensor);
        }
    }
    SafetensorsCheckpointPlan::new(
        "Nemotron-Labs-Diffusion draft LoRA",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

/// Strictly admitted source authority retained with the execution selection.
#[derive(Debug, Clone)]
pub(crate) struct AdmittedDraftAdapter {
    pub config: DraftAdapterConfig,
    pub schema: SafetensorsCheckpointPlan,
    pub resolution: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    pub catalog: eredu_checkpoint::safetensors::SafetensorsMetadataCatalog,
    pub tensors: eredu_core::checkpoint::TensorCatalog,
    config_path: std::path::PathBuf,
    config_bytes: std::sync::Arc<[u8]>,
    fingerprints: std::sync::Arc<eredu_checkpoint::artifact::ArtifactFingerprintSource>,
}

impl AdmittedDraftAdapter {
    pub fn admit(
        config: &FamilyConfig,
        source: &eredu_core::DraftAdapterSource,
    ) -> Result<Self, String> {
        use eredu_core::checkpoint::{TensorCatalog, TensorDescriptor, TensorDtype, TensorStorage};
        if source.path().as_os_str().is_empty() || source.expected_base().trim().is_empty() {
            return Err(
                "draft adapter requires a nonempty directory and expected base identity".into(),
            );
        }
        let root = std::fs::canonicalize(source.path()).map_err(|e| e.to_string())?;
        let config_path =
            std::fs::canonicalize(root.join("adapter_config.json")).map_err(|e| e.to_string())?;
        let weight_path = std::fs::canonicalize(root.join("adapter_model.safetensors"))
            .map_err(|e| e.to_string())?;
        if !config_path.starts_with(&root) || !weight_path.starts_with(&root) {
            return Err("draft adapter member escapes its declared directory".into());
        }
        let fingerprints = std::sync::Arc::new(
            eredu_checkpoint::artifact::ArtifactFingerprintSource::new([
                eredu_core::artifact::ArtifactFile::new("draft-adapter/config", &config_path),
                eredu_core::artifact::ArtifactFile::new("draft-adapter/parameters", &weight_path),
            ])
            .map_err(|e| e.to_string())?,
        );
        let config_bytes = std::fs::read(&config_path).map_err(|e| e.to_string())?;
        let adapter = DraftAdapterConfig::from_json(&config_bytes, source.expected_base())?;
        let schema = safetensors_plan(config, &adapter)?;
        let catalog =
            eredu_checkpoint::safetensors::SafetensorsMetadataCatalog::discover(weight_path)
                .map_err(|e| e.to_string())?;
        let resolution = eredu_checkpoint::validation::resolve_safetensors_plan(&catalog, &schema)
            .map_err(|e| format!("{e:?}"))?;
        let tensors = TensorCatalog::new(catalog.tensors().iter().map(|(name, metadata)| {
            let dtype = match metadata.stored_dtype {
                StoredDtype::F32 => TensorDtype::F32,
                StoredDtype::F16 => TensorDtype::F16,
                StoredDtype::BF16 => TensorDtype::Bf16,
                _ => unreachable!("strict floating adapter admission"),
            };
            TensorDescriptor {
                name: name.clone(),
                shape: metadata.logical_shape.clone(),
                dtype,
                storage: metadata.backing_shard.as_ref().map(|path| TensorStorage {
                    member: path.to_string_lossy().into_owned(),
                    offset: catalog
                        .tensor_offset(name)
                        .expect("admitted payload offset"),
                    length: metadata.encoded_byte_len,
                }),
            }
        }))
        .map_err(|e| e.to_string())?;
        fingerprints
            .validate_unchanged()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            config: adapter,
            schema,
            resolution,
            catalog,
            tensors,
            config_path,
            config_bytes: config_bytes.into(),
            fingerprints,
        })
    }

    pub fn verify_configuration(&self) -> Result<(), String> {
        self.fingerprints
            .validate_unchanged()
            .map_err(|e| e.to_string())?;
        let current = std::fs::read(&self.config_path).map_err(|e| e.to_string())?;
        if current.as_slice() != self.config_bytes.as_ref() {
            return Err("draft adapter configuration changed after admission".into());
        }
        self.fingerprints
            .validate_unchanged()
            .map_err(|e| e.to_string())
    }

    pub fn files(&self) -> Vec<eredu_core::artifact::ArtifactFile> {
        std::iter::once(eredu_core::artifact::ArtifactFile::new(
            "draft-adapter/config",
            &self.config_path,
        ))
        .chain(
            self.catalog
                .shards()
                .logical_payload_paths()
                .iter()
                .map(|(role, path)| {
                    eredu_core::artifact::ArtifactFile::new(format!("draft-adapter/{role}"), path)
                }),
        )
        .collect()
    }
}
