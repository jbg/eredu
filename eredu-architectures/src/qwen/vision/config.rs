//! Validated schedules and geometry for the shared Qwen vision encoder.

use std::collections::{BTreeSet, HashMap};

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::attention::LayerSchedule;
use eredu_gguf::{MetadataArray, MetadataValue};
use serde::Deserialize;

/// Stable cache identity for the complete shared vision policy and parameter formats.
pub fn prompt_cache_architecture_fingerprint(config: &VisionConfig) -> String {
    eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
        "qwen_vision",
        [
            ("mode", format!("{:?}", config.mode)),
            ("schedule", config.layer_schedule_fingerprint()),
            ("hidden", config.hidden_size.to_string()),
            ("activation", config.hidden_act.clone()),
            ("intermediate", config.intermediate_size.to_string()),
            ("heads", config.num_heads.to_string()),
            ("positions", config.num_position_embeddings.to_string()),
            ("channels", config.in_channels.to_string()),
            ("patch", config.patch_size.to_string()),
            ("spatial_merge", config.spatial_merge_size.to_string()),
            ("temporal_patch", config.temporal_patch_size.to_string()),
            ("window", config.window_size.to_string()),
            ("output", config.out_hidden_size.to_string()),
            (
                "linear_formats",
                crate::cache_identity::debug_map(Some(&config.linear_formats)),
            ),
        ],
    )
}

/// Attention topology for one vision transformer block.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum VisionAttentionPolicy {
    /// Attend across each complete image or video sequence.
    Full,
    /// Attend within validated spatial windows.
    Windowed,
}

/// Canonical execution policy for one vision transformer block.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct VisionLayerPolicy {
    /// Attention topology used by the block.
    pub attention: VisionAttentionPolicy,
    /// DeepStack merger bank captured after this block, when present.
    pub deepstack_merger: Option<u32>,
}

impl VisionLayerPolicy {
    const fn new(attention: VisionAttentionPolicy) -> Self {
        Self {
            attention,
            deepstack_merger: None,
        }
    }
}

/// Architecture policy selecting the genuinely different vision schedules.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionMode {
    /// Full-attention Qwen3-VL schedule with DeepStack capture.
    DeepStack,
    /// Qwen3.5 full/window schedule with optional DeepStack capture.
    WindowScheduled,
}

/// Shared vision encoder configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionConfig {
    /// Architecture-owned position, scheduling, and merger semantics.
    pub mode: VisionMode,
    /// Authoritative ordered execution policy for every block.
    pub layer_schedule: LayerSchedule<VisionLayerPolicy>,
    /// Vision transformer hidden size.
    pub hidden_size: i32,
    /// Vision MLP activation function.
    pub hidden_act: String,
    /// Vision MLP intermediate size.
    pub intermediate_size: i32,
    /// Number of attention heads.
    pub num_heads: i32,
    /// Number of learned spatial positions.
    pub num_position_embeddings: i32,
    /// Input channel count.
    pub in_channels: i32,
    /// Spatial patch size.
    pub patch_size: i32,
    /// Spatial patch merge factor.
    pub spatial_merge_size: i32,
    /// Temporal patch size.
    pub temporal_patch_size: i32,
    /// Spatial window width for windowed blocks.
    pub window_size: i32,
    /// Projection width in text hidden space.
    pub out_hidden_size: i32,
    /// Per-parameter checkpoint-native formats.
    pub linear_formats: HashMap<String, LinearFormat>,
}

impl VisionConfig {
    /// Number of vision blocks.
    pub fn layer_count(&self) -> usize {
        self.layer_schedule.len()
    }

    /// Returns one block policy without an out-of-range fallback.
    pub fn layer_policy(&self, layer: usize) -> Option<&VisionLayerPolicy> {
        self.layer_schedule.get(layer)
    }

    /// Number of configured DeepStack merger banks.
    pub fn deepstack_layer_count(&self) -> usize {
        self.layer_schedule
            .iter()
            .filter(|policy| policy.deepstack_merger.is_some())
            .count()
    }

    /// DeepStack source layers in merger-bank order.
    pub fn deepstack_layers(&self) -> Vec<i32> {
        let mut layers = self
            .layer_schedule
            .iter()
            .enumerate()
            .filter_map(|(layer, policy)| {
                policy.deepstack_merger.map(|merger| (merger, layer as i32))
            })
            .collect::<Vec<_>>();
        layers.sort_unstable_by_key(|(merger, _)| *merger);
        layers.into_iter().map(|(_, layer)| layer).collect()
    }

    /// Stable execution identity used by diagnostics and prompt fingerprints.
    pub fn layer_schedule_fingerprint(&self) -> String {
        let mode = match self.mode {
            VisionMode::DeepStack => "deepstack",
            VisionMode::WindowScheduled => "window_scheduled",
        };
        let schedule = self
            .layer_schedule
            .iter()
            .map(|policy| {
                let attention = match policy.attention {
                    VisionAttentionPolicy::Full => "f",
                    VisionAttentionPolicy::Windowed => "w",
                };
                policy.deepstack_merger.map_or_else(
                    || attention.to_owned(),
                    |merger| format!("{attention}d{merger}"),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("{mode}:{schedule}")
    }

    /// Resolves one canonical parameter's complete linear format.
    pub fn linear_format(&self, name: &str) -> LinearFormat {
        self.linear_formats
            .get(name)
            .copied()
            .unwrap_or(LinearFormat::Dense)
    }

    /// Applies an explicit in-memory quantization policy to aligned vision
    /// projections. This changes storage intent only; it performs no I/O.
    pub(crate) fn apply_load_time_quantization(&mut self, quantization: WeightQuantization) {
        let aligned =
            |input: i32| input > 0 && input % quantization.group_size() == 0 && input % 32 == 0;
        self.linear_formats.clear();
        for index in 0..self.layer_count() {
            for (name, input) in [
                (format!("blocks.{index}.attn.qkv.weight"), self.hidden_size),
                (format!("blocks.{index}.attn.proj.weight"), self.hidden_size),
                (
                    format!("blocks.{index}.mlp.linear_fc1.weight"),
                    self.hidden_size,
                ),
                (
                    format!("blocks.{index}.mlp.linear_fc2.weight"),
                    self.intermediate_size,
                ),
            ] {
                if aligned(input) {
                    self.linear_formats.insert(name, quantization.into());
                }
            }
        }
        let merger_input = self.hidden_size * self.spatial_merge_size * self.spatial_merge_size;
        for prefix in std::iter::once("merger".to_owned()).chain(
            (0..self.deepstack_layer_count()).map(|index| format!("deepstack_merger_list.{index}")),
        ) {
            for name in ["linear_fc1", "linear_fc2"] {
                if aligned(merger_input) {
                    self.linear_formats
                        .insert(format!("{prefix}.{name}.weight"), quantization.into());
                }
            }
        }
    }

    /// Validates exact shared geometry and mode-specific scheduling.
    pub fn validate(&self) -> Result<(), VisionConfigError> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_heads", self.num_heads),
            ("num_position_embeddings", self.num_position_embeddings),
            ("in_channels", self.in_channels),
            ("patch_size", self.patch_size),
            ("spatial_merge_size", self.spatial_merge_size),
            ("temporal_patch_size", self.temporal_patch_size),
            ("out_hidden_size", self.out_hidden_size),
        ] {
            if value <= 0 {
                return Err(VisionConfigError::Invalid(format!(
                    "vision {name} must be positive, got {value}"
                )));
            }
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(VisionConfigError::Invalid(format!(
                "vision hidden_size {} is not divisible by num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }
        if !matches!(
            self.hidden_act.as_str(),
            "silu" | "gelu" | "gelu_pytorch_tanh"
        ) {
            return Err(VisionConfigError::Invalid(format!(
                "unsupported vision activation {:?}",
                self.hidden_act
            )));
        }
        let mut mergers = self
            .layer_schedule
            .iter()
            .filter_map(|policy| policy.deepstack_merger)
            .collect::<Vec<_>>();
        mergers.sort_unstable();
        let expected = (0..mergers.len())
            .map(|index| index as u32)
            .collect::<Vec<_>>();
        if mergers != expected {
            return Err(VisionConfigError::Invalid(format!(
                "DeepStack merger banks must be unique and contiguous from zero, got {mergers:?}"
            )));
        }
        if self.mode == VisionMode::DeepStack
            && self
                .layer_schedule
                .iter()
                .any(|policy| policy.attention != VisionAttentionPolicy::Full)
        {
            return Err(VisionConfigError::Invalid(
                "DeepStack vision schedules require full attention in every block".into(),
            ));
        }
        if self
            .layer_schedule
            .iter()
            .any(|policy| policy.attention == VisionAttentionPolicy::Windowed)
            && self.window_size <= 0
        {
            return Err(VisionConfigError::Invalid(
                "windowed vision schedules require a positive window_size".into(),
            ));
        }
        for (name, format) in &self.linear_formats {
            if name.trim().is_empty() {
                return Err(VisionConfigError::Invalid(
                    "vision linear-format identity must not be empty".into(),
                ));
            }
            format
                .validate()
                .map_err(|error| VisionConfigError::Invalid(error.to_string()))?;
        }
        Ok(())
    }

    /// Validates this configuration for the consuming Qwen family without
    /// allowing the consumer to reinterpret its execution mode.
    pub fn validate_for(&self, expected: VisionMode) -> Result<(), VisionConfigError> {
        if self.mode != expected {
            return Err(VisionConfigError::Invalid(format!(
                "vision mode {:?} does not match required {:?} semantics",
                self.mode, expected
            )));
        }
        self.validate()
    }
}

/// Deserializable source accepted from nested family configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfigSource {
    #[serde(default = "default_vision_depth")]
    depth: i32,
    #[serde(default = "default_vision_hidden_size")]
    hidden_size: i32,
    #[serde(default = "default_vision_hidden_act")]
    hidden_act: String,
    #[serde(default = "default_vision_intermediate_size")]
    intermediate_size: i32,
    #[serde(default = "default_vision_num_heads")]
    num_heads: i32,
    #[serde(default = "default_vision_num_position_embeddings")]
    num_position_embeddings: i32,
    #[serde(default = "default_vision_in_channels")]
    in_channels: i32,
    #[serde(default = "default_vision_patch_size")]
    patch_size: i32,
    #[serde(default = "default_vision_spatial_merge_size")]
    spatial_merge_size: i32,
    #[serde(default = "default_vision_temporal_patch_size")]
    temporal_patch_size: i32,
    #[serde(default)]
    window_size: Option<i32>,
    #[serde(default = "default_vision_out_hidden_size")]
    out_hidden_size: i32,
    #[serde(default)]
    fullatt_block_indexes: Option<Vec<i32>>,
    #[serde(default)]
    deepstack_visual_indexes: Option<Vec<i32>>,
}

impl VisionConfigSource {
    /// Normalizes the Qwen3-VL full-attention DeepStack policy.
    pub fn normalize_qwen3_vl(self) -> Result<VisionConfig, VisionConfigError> {
        if self.window_size.is_some() || self.fullatt_block_indexes.is_some() {
            return Err(VisionConfigError::Invalid(
                "qwen3_vl vision_config must not define window_size or fullatt_block_indexes"
                    .into(),
            ));
        }
        let deepstack = self
            .deepstack_visual_indexes
            .clone()
            .unwrap_or_else(|| vec![8, 16, 24]);
        let depth = positive_depth(self.depth, "qwen3_vl")?;
        self.normalize(
            vec![VisionAttentionPolicy::Full; depth],
            deepstack,
            default_vision_window_size(),
            VisionMode::DeepStack,
            "qwen3_vl",
        )
    }

    /// Normalizes the Qwen3.5 window/full schedule and optional DeepStack policy.
    pub fn normalize_qwen3_5(self) -> Result<VisionConfig, VisionConfigError> {
        let depth = positive_depth(self.depth, "qwen3_5")?;
        let explicit_full = self.fullatt_block_indexes.clone();
        let mut attention = if explicit_full.is_some() {
            vec![VisionAttentionPolicy::Windowed; depth]
        } else {
            vec![VisionAttentionPolicy::Full; depth]
        };
        let mut seen = BTreeSet::new();
        for layer in explicit_full.unwrap_or_default() {
            let index = checked_layer_index(layer, depth, "qwen3_5 full-attention")?;
            if !seen.insert(index) {
                return Err(VisionConfigError::Invalid(format!(
                    "qwen3_5 full-attention vision layer {layer} is duplicated"
                )));
            }
            attention[index] = VisionAttentionPolicy::Full;
        }
        let window_size = self.window_size.unwrap_or_else(default_vision_window_size);
        if window_size <= 0 {
            return Err(VisionConfigError::Invalid(
                "qwen3_5 vision window_size must be positive".into(),
            ));
        }
        let deepstack = self.deepstack_visual_indexes.clone().unwrap_or_default();
        self.normalize(
            attention,
            deepstack,
            window_size,
            VisionMode::WindowScheduled,
            "qwen3_5",
        )
    }

    fn normalize(
        self,
        attention: Vec<VisionAttentionPolicy>,
        deepstack: Vec<i32>,
        window_size: i32,
        mode: VisionMode,
        architecture: &str,
    ) -> Result<VisionConfig, VisionConfigError> {
        let depth = positive_depth(self.depth, architecture)?;
        let mut policies = attention
            .into_iter()
            .map(VisionLayerPolicy::new)
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        for (merger, layer) in deepstack.into_iter().enumerate() {
            let index = checked_layer_index(layer, depth, &format!("{architecture} DeepStack"))?;
            if !seen.insert(index) {
                return Err(VisionConfigError::Invalid(format!(
                    "{architecture} DeepStack vision layer {layer} is duplicated"
                )));
            }
            policies[index].deepstack_merger = Some(u32::try_from(merger).map_err(|_| {
                VisionConfigError::Invalid(format!(
                    "{architecture} has too many DeepStack vision layers"
                ))
            })?);
        }
        let layer_schedule = LayerSchedule::new(depth, policies).map_err(|error| {
            VisionConfigError::Invalid(format!("{architecture} vision {error}"))
        })?;
        let config = VisionConfig {
            mode,
            layer_schedule,
            hidden_size: self.hidden_size,
            hidden_act: self.hidden_act,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            num_position_embeddings: self.num_position_embeddings,
            in_channels: self.in_channels,
            patch_size: self.patch_size,
            spatial_merge_size: self.spatial_merge_size,
            temporal_patch_size: self.temporal_patch_size,
            window_size,
            out_hidden_size: self.out_hidden_size,
            linear_formats: HashMap::new(),
        };
        config.validate()?;
        Ok(config)
    }
}

fn positive_depth(depth: i32, architecture: &str) -> Result<usize, VisionConfigError> {
    usize::try_from(depth)
        .ok()
        .filter(|depth| *depth > 0)
        .ok_or_else(|| {
            VisionConfigError::Invalid(format!("{architecture} vision depth must be positive"))
        })
}

fn checked_layer_index(layer: i32, depth: usize, label: &str) -> Result<usize, VisionConfigError> {
    usize::try_from(layer)
        .ok()
        .filter(|layer| *layer < depth)
        .ok_or_else(|| {
            VisionConfigError::Invalid(format!(
                "{label} layer {layer} is outside vision depth {depth}"
            ))
        })
}

/// Header-only tensor geometry needed to normalize a Qwen projector GGUF.
pub trait VisionGgufCatalog {
    /// Returns the logical row-major shape of a physical tensor.
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
}

/// Normalizes the shared llama.cpp Qwen vision projector contract under the
/// consuming family's explicit execution mode.
pub fn config_from_gguf_catalog(
    catalog: &impl VisionGgufCatalog,
    metadata: &HashMap<String, MetadataValue>,
    mode: VisionMode,
) -> Result<VisionConfig, VisionConfigError> {
    let string = |key: &str| match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value.as_str()),
        Some(_) => Err(VisionConfigError::Invalid(format!(
            "GGUF metadata key {key:?} must be a string"
        ))),
        None => Err(VisionConfigError::Invalid(format!(
            "GGUF metadata is missing {key:?}"
        ))),
    };
    if string("general.architecture")? != "clip"
        || string("clip.projector_type")? != "qwen3vl_merger"
    {
        return Err(VisionConfigError::Invalid(
            "expected a clip/qwen3vl_merger projector GGUF".into(),
        ));
    }
    let integer = |key: &str| {
        metadata
            .get(key)
            .and_then(MetadataValue::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| {
                VisionConfigError::Invalid(format!(
                    "GGUF metadata key {key:?} must be an i32 integer"
                ))
            })
    };
    let hidden_size = integer("clip.vision.embedding_length")?;
    let position_shape = catalog.shape("v.position_embd.weight").ok_or_else(|| {
        VisionConfigError::Invalid("projector is missing v.position_embd.weight".into())
    })?;
    if position_shape.len() != 2 || position_shape[1] != hidden_size as usize {
        return Err(VisionConfigError::Invalid(format!(
            "unexpected projector position embedding shape {position_shape:?}"
        )));
    }
    let depth = positive_depth(integer("clip.vision.block_count")?, "Qwen GGUF")?;
    let enabled = match metadata.get("clip.vision.is_deepstack_layers") {
        Some(MetadataValue::Array(MetadataArray::Bool(values))) => values,
        Some(_) => {
            return Err(VisionConfigError::Invalid(
                "clip.vision.is_deepstack_layers must be a bool array".into(),
            ))
        }
        None => {
            return Err(VisionConfigError::Invalid(
                "Qwen projector is missing DeepStack layer metadata".into(),
            ))
        }
    };
    if enabled.len() != depth {
        return Err(VisionConfigError::Invalid(format!(
            "DeepStack mask has {} entries for vision depth {depth}",
            enabled.len()
        )));
    }
    let mut merger = 0u32;
    let policies = enabled
        .iter()
        .map(|enabled| {
            let deepstack_merger = enabled.then(|| {
                let current = merger;
                merger += 1;
                current
            });
            VisionLayerPolicy {
                attention: VisionAttentionPolicy::Full,
                deepstack_merger,
            }
        })
        .collect::<Vec<_>>();
    let config = VisionConfig {
        mode,
        layer_schedule: LayerSchedule::new(depth, policies)
            .map_err(|error| VisionConfigError::Invalid(error.to_string()))?,
        hidden_size,
        hidden_act: "gelu_pytorch_tanh".into(),
        intermediate_size: integer("clip.vision.feed_forward_length")?,
        num_heads: integer("clip.vision.attention.head_count")?,
        num_position_embeddings: i32::try_from(position_shape[0]).map_err(|_| {
            VisionConfigError::Invalid("projector position count exceeds i32".into())
        })?,
        in_channels: 3,
        patch_size: integer("clip.vision.patch_size")?,
        spatial_merge_size: integer("clip.vision.spatial_merge_size")?,
        temporal_patch_size: 2,
        window_size: 112,
        out_hidden_size: integer("clip.vision.projection_dim")?,
        linear_formats: HashMap::new(),
    };
    config.validate()?;
    Ok(config)
}

const fn default_vision_depth() -> i32 {
    32
}
const fn default_vision_hidden_size() -> i32 {
    3584
}
fn default_vision_hidden_act() -> String {
    "silu".into()
}
const fn default_vision_intermediate_size() -> i32 {
    3420
}
const fn default_vision_num_heads() -> i32 {
    16
}
const fn default_vision_num_position_embeddings() -> i32 {
    2304
}
const fn default_vision_in_channels() -> i32 {
    3
}
const fn default_vision_patch_size() -> i32 {
    14
}
const fn default_vision_spatial_merge_size() -> i32 {
    2
}
const fn default_vision_temporal_patch_size() -> i32 {
    2
}
const fn default_vision_window_size() -> i32 {
    112
}
const fn default_vision_out_hidden_size() -> i32 {
    3584
}

/// Invalid family-owned vision geometry or schedule.
#[derive(Debug, Clone, thiserror::Error, Eq, PartialEq)]
pub enum VisionConfigError {
    /// Configuration violates a validated architecture invariant.
    #[error("{0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    struct Catalog;

    impl VisionGgufCatalog for Catalog {
        fn shape(&self, name: &str) -> Option<Vec<usize>> {
            (name == "v.position_embd.weight").then_some(vec![16, 16])
        }
    }

    fn projector_metadata() -> HashMap<String, MetadataValue> {
        HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            (
                "clip.projector_type".into(),
                MetadataValue::String("qwen3vl_merger".into()),
            ),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                MetadataValue::Uint32(24),
            ),
            (
                "clip.vision.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(2)),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(2)),
            (
                "clip.vision.spatial_merge_size".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "clip.vision.projection_dim".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "clip.vision.is_deepstack_layers".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![true, false])),
            ),
        ])
    }

    #[test]
    fn freezes_full_and_windowed_schedules_with_deepstack_identity() {
        let deepstack: VisionConfigSource = serde_json::from_value(json!({
            "depth": 4,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_heads": 4,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [1, 3]
        }))
        .unwrap();
        let deepstack = deepstack.normalize_qwen3_vl().unwrap();
        assert_eq!(
            deepstack.layer_schedule_fingerprint(),
            "deepstack:f,fd0,f,fd1"
        );
        assert_eq!(deepstack.mode, VisionMode::DeepStack);
        assert_eq!(deepstack.deepstack_layers(), vec![1, 3]);

        let windowed: VisionConfigSource = serde_json::from_value(json!({
            "depth": 4,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_heads": 4,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "window_size": 8,
            "fullatt_block_indexes": [1, 3],
            "deepstack_visual_indexes": [2]
        }))
        .unwrap();
        let windowed = windowed.normalize_qwen3_5().unwrap();
        assert_eq!(
            windowed.layer_schedule_fingerprint(),
            "window_scheduled:w,f,wd0,f"
        );
        assert_eq!(windowed.mode, VisionMode::WindowScheduled);
    }

    #[test]
    fn rejects_mode_schedule_and_geometry_drift() {
        let duplicate: VisionConfigSource = serde_json::from_value(json!({
            "depth": 2,
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_heads": 2,
            "num_position_embeddings": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [1, 1]
        }))
        .unwrap();
        assert!(duplicate.normalize_qwen3_5().is_err());
    }

    #[test]
    fn gguf_projector_retains_the_consuming_family_mode() {
        let metadata = projector_metadata();
        let qwen3_vl =
            crate::qwen::vl::vision_config_from_gguf_catalog(&Catalog, &metadata).unwrap();
        let qwen35 =
            crate::qwen::hybrid::vision_config_from_gguf_catalog(&Catalog, &metadata).unwrap();

        assert_eq!(qwen3_vl.mode, VisionMode::DeepStack);
        assert_eq!(qwen35.mode, VisionMode::WindowScheduled);
        assert!(qwen3_vl.validate_for(VisionMode::WindowScheduled).is_err());
        assert!(qwen35.validate_for(VisionMode::DeepStack).is_err());
        assert_eq!(qwen3_vl.deepstack_layers(), vec![0]);
        assert_eq!(qwen35.deepstack_layers(), vec![0]);
        assert_eq!(qwen3_vl.layer_schedule_fingerprint(), "deepstack:fd0,f");
        assert_eq!(
            qwen35.layer_schedule_fingerprint(),
            "window_scheduled:fd0,f"
        );
    }
}
