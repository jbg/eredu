//! Admission and topology of the optional published draft-only LoRA.
use super::FamilyConfig;
use crate::decoder::Config;
use eredu_checkpoint::{
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    StoredDtype,
};
use eredu_nn::{
    low_rank::LowRankProjectionSpec, DeclareLogicalParameters, Error, LinearSpec,
    LogicalParameterDeclaration, ParameterSpec,
};

/// Validated bias-free, ordinary low-rank output projection adapter.
#[derive(Debug, Clone, PartialEq)]
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
}

/// One draft-only invocation's separately owned low-rank matrices and head partition domain.
#[derive(Debug, Clone)]
pub struct DraftAdapterLayerSpec {
    pub(crate) root: String,
    pub(crate) projection: LowRankProjectionSpec,
    heads: i32,
    head_dim: i32,
}
impl DraftAdapterLayerSpec {
    /// Declares global adapter geometry without a backend or a second base projection.
    fn new(
        config: &FamilyConfig,
        adapter: &DraftAdapterConfig,
        index: usize,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        if index >= config.num_hidden_layers() as usize {
            return Err(Error::backend(
                "draft adapter layer outside shared backbone",
            ));
        }
        let root = format!("draft_adapter.layers.{index}.o_proj");
        let linear = |field: &str, input, output| {
            let mut weight = ParameterSpec::trainable(format!("{root}.{field}.weight"))
                .map_err(Error::backend)?;
            weight.group = Some(root.clone());
            Ok::<_, Error>(LinearSpec {
                input,
                output,
                weight,
                bias: None,
                format: eredu_nn::LinearFormatSpec::unscaled(
                    eredu_checkpoint::LinearFormat::Dense,
                )?,
            })
        };
        let input = config
            .num_attention_heads()
            .checked_mul(config.head_dim())
            .ok_or_else(|| Error::backend("draft adapter input width overflow"))?;
        let projection = LowRankProjectionSpec {
            down: linear("lora_A", input, adapter.rank)?,
            up: linear("lora_B", adapter.rank, config.hidden_size())?,
            scale: adapter.scale,
        };
        projection.validate()?;
        Ok(Self {
            root,
            projection,
            heads: config.num_attention_heads(),
            head_dim: config.head_dim(),
        })
    }

    /// Exact shared-rank affine construction policy.
    pub fn projection(&self) -> &LowRankProjectionSpec {
        &self.projection
    }

    /// Selects the same complete heads already admitted for the base attention invocation.
    pub(crate) fn with_local_heads(mut self, heads: i32) -> Result<Self, Error> {
        if heads <= 0 || heads > self.heads {
            return Err(Error::backend(
                "draft adapter head selection outside base attention",
            ));
        }
        self.projection.down.input = heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::backend("draft adapter local input width overflow"))?;
        self.heads = heads;
        self.projection.validate()?;
        Ok(self)
    }

    /// Input channels follow the base heads; rank activations and the up matrix remain replicated.
    pub fn parameter_group(
        &self,
    ) -> Result<eredu_runtime::ParameterGroupSpec, eredu_runtime::ParallelPlanError> {
        use eredu_runtime::{
            logical_parameter_members, MemberSharding, ParameterGroupSpec, ParameterRole,
        };
        let mut members = logical_parameter_members(&self.projection.down, |_| {
            MemberSharding::Partitioned { axis: 1 }
        })?;
        members.extend(logical_parameter_members(&self.projection.up, |_| {
            MemberSharding::Replicated
        })?);
        ParameterGroupSpec::partitioned(
            &self.root,
            ParameterRole::AttentionHeads,
            self.heads as usize,
            members,
        )
    }
}
impl DeclareLogicalParameters for DraftAdapterLayerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.projection.logical_parameters()
    }
}

/// Complete source-owned draft matrices, distinct from shared-backbone invocations.
#[derive(Debug, Clone)]
pub struct DraftAdapterDeclaration {
    policy: DraftAdapterConfig,
    layers: Vec<DraftAdapterLayerSpec>,
}
impl DraftAdapterDeclaration {
    pub(in crate::nemotron_labs_diffusion) fn new(
        config: &FamilyConfig,
        policy: DraftAdapterConfig,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        let layers = (0..config.num_hidden_layers() as usize)
            .map(|layer| DraftAdapterLayerSpec::new(config, &policy, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let declaration = Self { policy, layers };
        declaration.validate_for(config)?;
        Ok(declaration)
    }
    /// Released ancestry and alpha/rank policy retained independently of placement.
    pub fn policy(&self) -> &DraftAdapterConfig {
        &self.policy
    }
    /// Source scopes in backbone execution order.
    pub fn layers(&self) -> &[DraftAdapterLayerSpec] {
        &self.layers
    }
    /// Resolves one independently acquired depth before native allocation.
    pub fn layer(&self, index: usize) -> Result<&DraftAdapterLayerSpec, Error> {
        self.layers
            .get(index)
            .ok_or_else(|| Error::backend("draft adapter layer outside retained backbone"))
    }
    /// Projects input heads while retaining rank, output ownership and operator arithmetic.
    pub fn project_heads(&self, heads: i32) -> Result<Self, Error> {
        let mut selected = self.clone();
        selected.layers = self
            .layers
            .iter()
            .cloned()
            .map(|layer| layer.with_local_heads(heads))
            .collect::<Result<_, _>>()?;
        eredu_nn::validate_logical_parameter_topology(&selected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(selected)
    }
    pub(crate) fn validate_for(&self, config: &impl Config) -> Result<(), Error> {
        let input = config
            .num_attention_heads()
            .checked_mul(config.head_dim())
            .ok_or_else(|| Error::backend("draft adapter input width overflow"))?;
        if self.layers.len() != config.num_hidden_layers() as usize
            || self.layers.iter().any(|layer| {
                layer.heads != config.num_attention_heads()
                    || layer.head_dim != config.head_dim()
                    || layer.projection.down.input != input
                    || layer.projection.down.output != self.policy.rank
                    || layer.projection.up.input != self.policy.rank
                    || layer.projection.up.output != config.hidden_size()
                    || layer.projection.scale != self.policy.scale
            })
        {
            return Err(Error::backend(
                "draft adapter scopes differ from retained backbone geometry or policy",
            ));
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(())
    }
    /// Strict separate-source admission with explicit released aliases and floating dtype rules.
    pub fn safetensors_plan(&self) -> Result<SafetensorsCheckpointPlan, String> {
        let mut tensors = Vec::new();
        for (layer, declaration) in self.layers.iter().enumerate() {
            for parameter in declaration
                .logical_parameters()
                .map_err(|e| e.to_string())?
            {
                let suffix = parameter
                    .parameter
                    .id
                    .as_str()
                    .strip_prefix(&format!("{}.", declaration.root))
                    .expect("declared adapter identity under its owner");
                let mut tensor = SafetensorsTensorConstraint::required(
                    parameter.parameter.id.as_str(),
                    parameter.shape,
                    StoredDtypeConstraint::OneOf(vec![
                        StoredDtype::F32,
                        StoredDtype::F16,
                        StoredDtype::BF16,
                    ]),
                );
                tensor.aliases.push(format!(
                    "base_model.model.encoder.layers.{layer}.self_attn.o_proj.{suffix}"
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
}
impl DeclareLogicalParameters for DraftAdapterDeclaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.layers.logical_parameters()
    }
}

/// Strictly admitted source authority retained with the execution selection.
#[derive(Debug, Clone)]
pub(crate) struct AdmittedDraftAdapter {
    pub declaration: std::sync::Arc<DraftAdapterDeclaration>,
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
        let declaration = std::sync::Arc::new(
            DraftAdapterDeclaration::new(config, adapter).map_err(|e| e.to_string())?,
        );
        let schema = declaration.safetensors_plan()?;
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
            declaration,
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
