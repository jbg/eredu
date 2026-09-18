//! Semantic tensor-parallel placement for Muse-Glimmer text and vision components.

use eredu_core::{cache::LayerCachePolicy, LayerSchedule};
use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{
    expand_linear_format_parameter_groups, ArchitecturePartition, LocalModelLayout, MemberSharding,
    ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole, PartitionOwnership,
    StateLayout, TensorPlacement,
};
use std::ops::Range;

use crate::linear_format::standard_parallel_linear_format;

use super::DecoderConfig;

/// Complete planner-derived construction and mutable-state geometry for one rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    text_blocks: Vec<DecoderConfig>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    vision_layers: usize,
    architecture_fingerprint: String,
}

/// Exact TP-local and PP-local geometry for one Muse-Glimmer composite partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    vision_units: Option<Range<usize>>,
    text_units: Range<usize>,
    text_blocks: Vec<DecoderConfig>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    complete_state_layout: StateLayout,
    static_roles: Vec<String>,
    architecture_fingerprint: String,
}

/// Validated architecture-owned handoff for one Muse-Glimmer composite partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalFoundation {
    geometry: PartitionLocalGeometry,
    parameter_targets: Vec<String>,
}

impl PartitionLocalFoundation {
    /// Validates group ownership, state offset, boundary schema, and selected tasks together.
    pub fn from_partition(
        args: &DecoderConfig,
        partition: &ArchitecturePartition<
            PartitionLocalGeometry,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
    ) -> Result<Self, ParallelPlanError> {
        let geometry = partition.local_geometry();
        geometry.validate_for(args)?;
        let expected_groups = [
            geometry
                .vision_units()
                .map(|range| (super::VISION_EXECUTION_GROUP, range)),
            Some((super::TEXT_EXECUTION_GROUP, geometry.text_units())),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let actual_groups = partition
            .groups()
            .iter()
            .map(|group| (group.group().as_str(), group.global_units()))
            .collect::<Vec<_>>();
        if actual_groups != expected_groups {
            return Err(invalid(
                "Muse-Glimmer partition groups differ from family-local geometry",
            ));
        }
        let state = partition
            .state()
            .ok_or_else(|| invalid("Muse-Glimmer text partition has no selected state"))?;
        if state.global_layer_offset() != geometry.text_units.start
            || state.layout() != &geometry.local_state_layout()?
        {
            return Err(invalid(
                "Muse-Glimmer partition state differs from its text unit range",
            ));
        }
        if partition.boundary_schema()
            != &eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size)
        {
            return Err(invalid(
                "Muse-Glimmer partition boundary differs from its decoder width",
            ));
        }
        let parameter_targets = partition
            .parameter_bindings()
            .iter()
            .flat_map(|binding| binding.members())
            .map(|member| member.target().to_owned())
            .collect::<Vec<_>>();
        Ok(Self {
            geometry: geometry.clone(),
            parameter_targets,
        })
    }

    /// Exact family-local construction geometry.
    pub const fn geometry(&self) -> &PartitionLocalGeometry {
        &self.geometry
    }

    /// Canonical materialization targets selected for this rank.
    pub fn parameter_targets(&self) -> &[String] {
        &self.parameter_targets
    }
}

impl PartitionLocalGeometry {
    /// Architecture-global units owned from the optional vision root.
    pub fn vision_units(&self) -> Option<Range<usize>> {
        self.vision_units.clone()
    }

    /// Architecture-global text units owned by this pipeline coordinate.
    pub fn text_units(&self) -> Range<usize> {
        self.text_units.clone()
    }

    /// TP-local configuration for one owned architecture-global text unit.
    pub fn text_block(&self, global_unit: usize) -> Option<&DecoderConfig> {
        self.text_units
            .contains(&global_unit)
            .then(|| &self.text_blocks[global_unit - self.text_units.start])
    }

    /// Complete TP-local state geometry before PP slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }

    /// Exact local state slice and its architecture-global offset.
    pub fn local_state_layout(&self) -> Result<StateLayout, ParallelPlanError> {
        self.complete_state_layout
            .slice(self.text_units.clone())
            .map_err(|error| invalid(error.to_string()))
    }

    /// Static roles selected for this pipeline coordinate.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    /// Input-embedding vocabulary ownership for this tensor coordinate.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Untied output-head vocabulary ownership for this tensor coordinate.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    pub(super) fn validate_for(&self, args: &DecoderConfig) -> Result<(), ParallelPlanError> {
        let text_count = args.num_hidden_layers as usize;
        let vision_count = args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.layer_count());
        if self.text_units.is_empty()
            || self.text_units.end > text_count
            || self.text_blocks.len() != self.text_units.len()
            || self.architecture_fingerprint != args.architecture_fingerprint()
        {
            return Err(invalid(
                "partition-local Muse-Glimmer geometry belongs to a different model or text range",
            ));
        }
        if self
            .vision_units
            .as_ref()
            .is_some_and(|range| range.is_empty() || range.end > vision_count)
            || (vision_count == 0 && self.vision_units.is_some())
        {
            return Err(invalid(
                "partition-local Muse-Glimmer vision range is invalid",
            ));
        }
        let mut expected_roles: Vec<String> = Vec::new();
        if self
            .vision_units
            .as_ref()
            .is_some_and(|range| range.start == 0)
        {
            expected_roles.push("vision".into());
        }
        if self.text_units.start == 0 {
            expected_roles.push("embedding".into());
        }
        if self.text_units.end == text_count {
            // The output consumer exists even when its parameter is the
            // shared embedding group retained by either endpoint.
            expected_roles.extend(["norm".into(), "output".into()]);
        }
        if self.static_roles != expected_roles {
            return Err(invalid(format!(
                "partition-local Muse-Glimmer static roles {:?} differ from {:?}",
                self.static_roles, expected_roles
            )));
        }
        if self.complete_state_layout.len() != text_count {
            return Err(invalid(
                "partition-local Muse-Glimmer complete state length is invalid",
            ));
        }
        self.local_state_layout()?;
        Ok(())
    }
}

impl LocalGeometry {
    /// Returns one rank-local text-block configuration.
    pub fn text_block(&self, layer: usize) -> Option<&DecoderConfig> {
        self.text_blocks.get(layer)
    }

    /// Returns all rank-local text-block configurations in execution order.
    pub fn text_blocks(&self) -> &[DecoderConfig] {
        &self.text_blocks
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the authoritative rank-local text state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Returns the replicated media units owned by this rank.
    pub const fn vision_layers(&self) -> usize {
        self.vision_layers
    }

    pub(super) fn validate_for(&self, args: &DecoderConfig) -> Result<(), ParallelPlanError> {
        if self.architecture_fingerprint != args.architecture_fingerprint()
            || self.text_blocks.len() != args.num_hidden_layers as usize
            || self.vision_layers
                != args
                    .vision_config
                    .as_ref()
                    .map_or(0, |vision| vision.layer_count())
        {
            return Err(invalid(
                "rank-local Muse-Glimmer geometry belongs to a different configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (args.tie_word_embeddings, &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(args.vocab_size)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => {
                return Err(invalid(
                    "tied Muse-Glimmer output has a separate vocabulary range",
                ))
            }
            (false, None) => {
                return Err(invalid(
                    "untied Muse-Glimmer output has no vocabulary range",
                ))
            }
        }
        let expected = local_state_layout(&self.text_blocks)?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local Muse-Glimmer state layout drifted from text geometry",
            ));
        }
        Ok(())
    }
}

/// Derives all rank-local Muse-Glimmer construction geometry from one typed layout.
pub fn local_geometry(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let text_blocks = (0..args.num_hidden_layers as usize)
        .map(|layer| local_decoder_config(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = local_state_layout(&text_blocks)?;
    let vocabulary = dim(args.vocab_size)?;
    let embedding_range = vocabulary_range(layout, "model.embed_tokens", vocabulary)?;
    let output_range = if args.tie_word_embeddings {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", vocabulary)?)
    };
    let geometry = LocalGeometry {
        text_blocks,
        embedding_range,
        output_range,
        state_layout,
        // Media parameters are deliberately outside the text TP plan and are
        // replicated on every rank. Keeping their unit ownership here makes
        // the canonical multimodal graph authoritative for both lifecycles.
        vision_layers: args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.layer_count()),
        architecture_fingerprint: args.architecture_fingerprint(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives exact TP-local and PP-local Muse-Glimmer composite geometry.
pub fn partition_local_geometry(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if args.is_moe() {
        return Err(invalid(
            "partition-local Muse-Glimmer foundation does not admit routed text blocks",
        ));
    }
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

pub(crate) fn routed_partition_local_geometry(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if !args.is_moe() {
        return Err(invalid(
            "routed Muse-Glimmer partition has no sparse text blocks",
        ));
    }
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

fn partition_local_geometry_impl(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let mut vision_units = None;
    let mut text_units = None;
    for (group, range) in group_ranges {
        if range.is_empty() {
            return Err(invalid(
                "Muse-Glimmer partition group range cannot be empty",
            ));
        }
        let slot = match group.as_ref() {
            super::VISION_EXECUTION_GROUP => &mut vision_units,
            super::TEXT_EXECUTION_GROUP => &mut text_units,
            other => {
                return Err(invalid(format!(
                    "unknown Muse-Glimmer partition group {other:?}"
                )))
            }
        };
        if slot.replace(range).is_some() {
            return Err(invalid("duplicate Muse-Glimmer partition group"));
        }
    }
    let text_units = text_units
        .ok_or_else(|| invalid("Muse-Glimmer partition must own a non-empty text decoder range"))?;
    let complete = local_geometry(args, layout)?;
    let text_blocks = text_units
        .clone()
        .map(|global| {
            complete
                .text_block(global)
                .cloned()
                .ok_or_else(|| invalid(format!("Muse-Glimmer has no local text block {global}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = PartitionLocalGeometry {
        vision_units,
        text_units,
        text_blocks,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        static_roles: ownership.static_roles().to_vec(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

fn local_state_layout(blocks: &[DecoderConfig]) -> Result<StateLayout, ParallelPlanError> {
    let layers = blocks
        .iter()
        .enumerate()
        .map(|(layer, block)| {
            let policy = block.attention_schedule.get(layer).ok_or_else(|| {
                invalid(format!(
                    "rank-local Muse-Glimmer block {layer} has no attention policy"
                ))
            })?;
            LayerCachePolicy::key_value(*policy, block.num_key_value_heads, block.head_dim)
                .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers).map_err(|error| invalid(error.to_string()))?,
    )
    .map_err(|error| invalid(error.to_string()))
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    vocabulary: usize,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected = None;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        if tensor.global_shape().first().copied() != Some(vocabulary) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Muse-Glimmer vocabulary member {target} has global shape {:?}, expected {vocabulary} rows",
                tensor.global_shape()
            )));
        }
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated | TensorPlacement::Local => 0..vocabulary,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "Muse-Glimmer vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Muse-Glimmer vocabulary group {logical_name} has inconsistent selections"
            )));
        }
        selected = Some(range);
    }
    let range = VocabularyParallelRange {
        global_vocabulary: vocabulary,
        local: selected.ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "missing local Muse-Glimmer vocabulary layout for {logical_name}"
            ))
        })?,
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Derives rank-local text construction geometry from the semantic placement.
pub fn local_decoder_config(
    args: &DecoderConfig,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<DecoderConfig, ParallelPlanError> {
    if layer >= args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "Muse-Glimmer layer {layer} is out of range"
        )));
    }
    let root = format!("model.layers.{layer}");
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{root}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}")))
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {root}.{suffix}"
                ))
            })
    };
    let query_width = local_axis(tensor("self_attn.q_proj")?, 0, "query width")?;
    let key_width = local_axis(tensor("self_attn.k_proj")?, 0, "key width")?;
    if query_width % args.head_dim != 0 || key_width % args.head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Muse-Glimmer attention widths q={query_width}, k={key_width} split head dimension {}",
            args.head_dim
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = query_width / args.head_dim;
    local.num_key_value_heads = key_width / args.head_dim;
    if args.is_moe() {
        let fused = local_axis(
            tensor("mlp.experts.gate_up_proj")?,
            1,
            "expert gate/up width",
        )?;
        if fused % 2 != 0 {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "local Muse-Glimmer expert gate/up width {fused} is not even"
            )));
        }
        local.moe_intermediate_size = fused / 2;
    } else {
        local.intermediate_size =
            local_axis(tensor("mlp.gate_proj")?, 0, "dense intermediate width")?;
    }
    Ok(local)
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<i32, ParallelPlanError> {
    let value = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "local Muse-Glimmer {label} tensor has no axis {axis}"
        ))
    })?;
    i32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "local Muse-Glimmer {label} must be positive and fit i32"
            ))
        })
}

mod declaration;
pub(crate) use declaration::{static_parameter_groups as static_parameter_groups_in,layer_parameter_groups as layer_parameter_groups_in,vision_static_parameter_groups as vision_static_parameter_groups_in,vision_layer_parameter_groups as vision_layer_parameter_groups_in};
pub fn static_parameter_groups(
    args: &DecoderConfig,
)->Result<Vec<ParameterGroupSpec>,ParallelPlanError>{
    declaration::static_parameter_groups(args,crate::decoder::parameter_metadata::DeclarationDestination(None))
        .map_err(crate::decoder::parameter_metadata::ParameterGroupError::ordinary)
}
pub fn layer_parameter_groups(
    args: &DecoderConfig,
    layer: usize,
)->Result<Vec<ParameterGroupSpec>,ParallelPlanError>{
    declaration::layer_parameter_groups(args, layer,crate::decoder::parameter_metadata::DeclarationDestination(None))
        .map_err(crate::decoder::parameter_metadata::ParameterGroupError::ordinary)
}
pub fn vision_parameter_groups(
    args: &DecoderConfig,
)->Result<Vec<ParameterGroupSpec>,ParallelPlanError>{
    declaration::vision_parameter_groups(args,crate::decoder::parameter_metadata::DeclarationDestination(None))
        .map_err(crate::decoder::parameter_metadata::ParameterGroupError::ordinary)
}
pub fn vision_static_parameter_groups(
    args: &DecoderConfig,
)->Result<Vec<ParameterGroupSpec>,ParallelPlanError>{
    declaration::vision_static_parameter_groups(args,crate::decoder::parameter_metadata::DeclarationDestination(None))
        .map_err(crate::decoder::parameter_metadata::ParameterGroupError::ordinary)
}
pub fn vision_layer_parameter_groups(
    args: &DecoderConfig,
    layer: usize,
)->Result<Vec<ParameterGroupSpec>,ParallelPlanError>{
    declaration::vision_layer_parameter_groups(args, layer,crate::decoder::parameter_metadata::DeclarationDestination(None))
        .map_err(crate::decoder::parameter_metadata::ParameterGroupError::ordinary)
}
/// Vision units whose owners need each pinned static module. Every vision
/// partition constructs the parameter-free request context through `begin`;
/// patch/position and static norm groups therefore remain available on those
/// owners. The large merge adapter and projection are needed only at the end.
pub(crate) fn vision_static_consumer_units(
    args: &DecoderConfig,
    target: &str,
) -> Option<std::ops::Range<usize>> {
    let count = args.vision_config.as_ref()?.layer_count();
    if count == 0
        || !target.starts_with("model.vision_")
        || target.starts_with("model.vision_tower.layers.")
    {
        return None;
    }
    Some(
        if target.starts_with("model.vision_adapter.")
            || target.starts_with("model.vision_projection.")
        {
            count - 1..count
        } else {
            0..count
        },
    )
}


fn group(
    name: impl Into<String>,
    role: ParameterRole,
    members: impl IntoIterator<Item = (impl Into<String>, Vec<usize>, MemberSharding)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    ParameterGroupSpec::new(
        name,
        role,
        members
            .into_iter()
            .map(|(name, shape, sharding)| member(name, shape, sharding)),
    )
}

fn replicated(
    name: impl Into<String>,
    members: impl IntoIterator<Item = (impl Into<String>, Vec<usize>)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    group(
        name,
        ParameterRole::Replicated,
        members
            .into_iter()
            .map(|(name, shape)| (name, shape, MemberSharding::Replicated)),
    )
}

fn member(
    name: impl Into<String>,
    shape: Vec<usize>,
    sharding: MemberSharding,
) -> ParameterMemberSpec {
    ParameterMemberSpec::new(name, shape, sharding)
}

const fn partitioned(axis: usize) -> MemberSharding {
    MemberSharding::Partitioned { axis }
}

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("invalid Muse-Glimmer dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}
#[cfg(test)]
mod tests {
    use eredu_checkpoint::AffineQuantization;
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, TensorPlacement};

    use super::*;

    #[test]
    fn canonical_muse_glimmer_parameters_state_and_identity_use_the_actual_destination() {
        crate::architecture_parameter_metadata_tests::exercise(|context| crate::muse_glimmer::LayeredModel::<eredu_nn::workspace::WorkspaceBackend>::new(args(),context));
    }

    #[test]
    fn canonical_muse_vision_boundary_preserves_geometry_and_every_reached_refusal() {
        let args = args();
        for continuation in [false, true] {
            crate::architecture_parameter_metadata_tests::boundary(|metadata| {
                let schema = crate::muse_glimmer::model::vision_partition_boundary_schema(
                    &args, continuation, metadata)?;
                crate::composite_execution::graph::Destination(metadata)
                    .resolve_boundary(&schema, 2, &[7])
            });
        }
    }

    fn args() -> DecoderConfig {
        DecoderConfig::from_hf_value(&serde_json::json!({
          "architectures":["MuseGlimmerForConditionalGeneration"],
          "model_type":"muse_glimmer",
          "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":16,
          "text_config":{"model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":1,
            "intermediate_size":0,"moe_intermediate_size":12,"num_experts":4,"num_experts_per_tok":2,
            "norm_topk_prob":true,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,"max_position_embeddings":64,
            "rope_theta":10000.0,"layer_types":["sliding_attention"],"layer_rope_theta":[10000.0],
            "sliding_window":8,"tie_word_embeddings":false,"hidden_act":"silu","attention_dropout":0.0,
            "qk_scale_factor":1.0,"output_multiplier":1.0,"final_logit_softcapping":30.0},
          "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,"intermediate_size":12,
            "num_attention_heads":2,"num_hidden_layers":1,"patch_size":2,"patch_temporal":1,"merge_size":2,
            "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
            "hidden_act":"gelu","layer_types":["full_attention"],
            "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        }))
        .unwrap()
    }

    #[test]
    fn distinguishes_text_experts_router_and_replicated_vision() {
        let args = args();
        let text = layer_parameter_groups(&args, 0).unwrap();
        assert!(text
            .iter()
            .any(|group| group.role() == ParameterRole::ExpertIntermediate));
        assert!(text
            .iter()
            .any(|group| group.logical_name().ends_with("mlp.router")));
        let vision = vision_parameter_groups(&args).unwrap();
        assert!(vision
            .iter()
            .any(|group| group.logical_name().ends_with("attention_heads")));
        assert!(vision.iter().all(|group| group.partition_units().is_none()));
        assert!(vision.iter().all(|group| group
            .members()
            .iter()
            .all(|member| member.sharding() == &MemberSharding::Replicated)));
    }

    #[test]
    fn gguf_head_norm_gains_are_replicated_and_hf_norms_remain_weightless() {
        let mut args = args();
        for convention in [
            super::super::WeightConvention::HuggingFace,
            super::super::WeightConvention::Gguf,
        ] {
            args.weight_convention = convention;
            let groups = layer_parameter_groups(&args, 0).unwrap();
            for suffix in ["q_norm.weight", "k_norm.weight"] {
                let name = format!("model.layers.0.self_attn.{suffix}");
                let members = groups
                    .iter()
                    .flat_map(|g| g.members())
                    .filter(|m| m.target() == name)
                    .collect::<Vec<_>>();
                if convention == super::super::WeightConvention::Gguf {
                    assert_eq!(members.len(), 1);
                    assert_eq!(members[0].global_shape(), &[args.head_dim as usize]);
                    assert_eq!(members[0].sharding(), &MemberSharding::Replicated);
                } else {
                    assert!(members.is_empty());
                }
            }
        }
    }

    #[test]
    fn affine_text_plan_publishes_weight_companions() {
        let mut args = args();
        args.quantization = Some(AffineQuantization::new(16, 4).unwrap().into());
        args.quantized_weights = Some(std::collections::HashSet::from([
            "model.layers.0.self_attn.q_proj.weight".to_owned(),
            "model.layers.0.mlp.experts.gate_up_proj".to_owned(),
        ]));
        let targets = layer_parameter_groups(&args, 0)
            .unwrap()
            .into_iter()
            .flat_map(|group| group.members().to_vec())
            .map(|member| member.target().to_owned())
            .collect::<Vec<_>>();
        assert!(targets
            .iter()
            .any(|name| name == "model.layers.0.self_attn.q_proj.scales"));
        assert!(targets
            .iter()
            .any(|name| name == "model.layers.0.mlp.experts.gate_up_proj_scales"));
    }

    fn local_tensor(shape: Vec<usize>) -> LocalTensorLayout {
        LocalTensorLayout::new(
            "test",
            ParameterRole::AttentionHeads,
            shape.clone(),
            shape,
            TensorPlacement::Local,
            None,
            None,
            false,
        )
    }

    fn insert(
        layout: &mut LocalModelLayout,
        target: &str,
        logical_name: &str,
        global_shape: Vec<usize>,
        local_shape: Vec<usize>,
        placement: TensorPlacement,
    ) {
        layout.insert(
            target.into(),
            LocalTensorLayout::new(
                logical_name,
                ParameterRole::AttentionHeads,
                global_shape,
                local_shape,
                placement,
                None,
                None,
                false,
            ),
        );
    }

    fn row_range(end: usize) -> TensorPlacement {
        TensorPlacement::Range {
            axis: 0,
            start: 0,
            end,
        }
    }

    fn valid_layout(include_output: bool) -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        insert(
            &mut layout,
            "model.embed_tokens.weight",
            "model.embed_tokens",
            vec![24, 16],
            vec![12, 16],
            row_range(12),
        );
        if include_output {
            insert(
                &mut layout,
                "lm_head.weight",
                "lm_head",
                vec![24, 16],
                vec![12, 16],
                row_range(12),
            );
        }
        insert(
            &mut layout,
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.query_heads",
            vec![16, 16],
            vec![8, 16],
            row_range(8),
        );
        insert(
            &mut layout,
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.key_value_heads",
            vec![8, 16],
            vec![4, 16],
            row_range(4),
        );
        insert(
            &mut layout,
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.intermediate",
            vec![4, 24, 16],
            vec![4, 12, 16],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: 12,
            },
        );
        layout
    }

    #[test]
    fn local_decoder_geometry_tracks_heads_and_expert_intermediate() {
        let args = args();
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            local_tensor(vec![8, 16]),
        );
        layout.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            local_tensor(vec![4, 16]),
        );
        layout.insert(
            "model.layers.0.mlp.experts.gate_up_proj".into(),
            local_tensor(vec![4, 12, 16]),
        );
        let local = local_decoder_config(&args, 0, &layout).unwrap();
        assert_eq!(local.num_attention_heads, 2);
        assert_eq!(local.num_key_value_heads, 1);
        assert_eq!(local.moe_intermediate_size, 6);
    }

    #[test]
    fn local_geometry_owns_text_vocabulary_state_and_media_together() {
        let args = args();
        let geometry = local_geometry(&args, &valid_layout(true)).unwrap();
        assert_eq!(geometry.text_blocks().len(), 1);
        assert_eq!(geometry.text_block(0).unwrap().num_attention_heads, 2);
        assert_eq!(geometry.text_block(0).unwrap().num_key_value_heads, 1);
        assert_eq!(geometry.embedding_range().local, 0..12);
        assert_eq!(geometry.output_range().unwrap().local, 0..12);
        assert_eq!(geometry.vision_layers(), 1);
        assert_ne!(
            geometry.state_layout(),
            &crate::muse_glimmer::state_layout(&args).unwrap()
        );
        geometry.validate_for(&args).unwrap();
    }

    fn partition_args_and_layout() -> (DecoderConfig, LocalModelLayout) {
        let args = DecoderConfig::from_hf_value(&serde_json::json!({
          "architectures":["MuseGlimmerForConditionalGeneration"],
          "model_type":"muse_glimmer",
          "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":16,
          "text_config":{"model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":2,
            "intermediate_size":24,"moe_intermediate_size":0,"num_experts":0,"num_experts_per_tok":0,
            "norm_topk_prob":false,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,"max_position_embeddings":64,
            "rope_theta":10000.0,"layer_types":["sliding_attention","full_attention"],
            "layer_rope_theta":[10000.0,0.0],"sliding_window":8,"tie_word_embeddings":false,
            "hidden_act":"silu","attention_dropout":0.0,"qk_scale_factor":1.0,
            "output_multiplier":1.0,"final_logit_softcapping":30.0},
          "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,"intermediate_size":12,
            "num_attention_heads":2,"num_hidden_layers":1,"patch_size":2,"patch_temporal":1,"merge_size":2,
            "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
            "hidden_act":"gelu","layer_types":["full_attention"],
            "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        }))
        .unwrap();
        let mut layout = valid_layout(true);
        layout.insert(
            "model.layers.0.mlp.gate_proj.weight".into(),
            LocalTensorLayout::new(
                "model.layers.0.mlp.intermediate",
                ParameterRole::FeedForwardIntermediate,
                vec![24, 16],
                vec![12, 16],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 12,
                },
                None,
                None,
                false,
            ),
        );
        for (suffix, logical, global, local, placement) in [
            (
                "self_attn.q_proj.weight",
                "query_heads",
                vec![16, 16],
                vec![8, 16],
                row_range(8),
            ),
            (
                "self_attn.k_proj.weight",
                "key_value_heads",
                vec![8, 16],
                vec![4, 16],
                row_range(4),
            ),
            (
                "mlp.gate_proj.weight",
                "mlp.intermediate",
                vec![24, 16],
                vec![12, 16],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 12,
                },
            ),
        ] {
            insert(
                &mut layout,
                &format!("model.layers.1.{suffix}"),
                &format!("model.layers.1.{logical}"),
                global,
                local,
                placement,
            );
        }
        (args, layout)
    }

    #[test]
    fn partition_geometry_preserves_dense_text_and_optional_root_ownership() {
        let (partition_args, layout) = partition_args_and_layout();
        let first = PartitionOwnership::new(true, false, ["vision", "embedding"]).unwrap();
        let first = partition_local_geometry(
            &partition_args,
            &layout,
            [
                (super::super::VISION_EXECUTION_GROUP, 0..1),
                (super::super::TEXT_EXECUTION_GROUP, 0..1),
            ],
            &first,
        )
        .unwrap();
        assert_eq!(first.vision_units(), Some(0..1));
        assert_eq!(first.text_units(), 0..1);
        assert_eq!(first.text_block(0).unwrap().num_experts, 0);
        assert_eq!(first.text_block(0).unwrap().intermediate_size, 12);
        assert_eq!(first.local_state_layout().unwrap().len(), 1);

        let last = PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
        let last = partition_local_geometry(
            &partition_args,
            &layout,
            [(super::super::TEXT_EXECUTION_GROUP, 1..2)],
            &last,
        )
        .unwrap();
        assert_eq!(last.text_units(), 1..2);
        assert_eq!(last.static_roles(), ["norm", "output"]);
        assert_eq!(last.complete_state_layout().len(), 2);
        assert_eq!(last.local_state_layout().unwrap().len(), 1);
    }

    #[test]
    fn partition_geometry_rejects_wrong_optional_root_and_static_placement() {
        let (partition_args, layout) = partition_args_and_layout();
        let wrong = PartitionOwnership::new(false, true, ["vision"]).unwrap();
        assert!(partition_local_geometry(
            &partition_args,
            &layout,
            [(super::super::TEXT_EXECUTION_GROUP, 1..2)],
            &wrong,
        )
        .is_err());
        let roles = PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
        assert!(partition_local_geometry(
            &partition_args,
            &layout,
            [
                (super::super::VISION_EXECUTION_GROUP, 1..2),
                (super::super::TEXT_EXECUTION_GROUP, 1..2),
            ],
            &roles,
        )
        .is_err());
        let routed = args();
        let routed_roles =
            PartitionOwnership::new(true, true, ["vision", "embedding", "norm", "output"]).unwrap();
        let error = partition_local_geometry(
            &routed,
            &valid_layout(true),
            [
                (super::super::VISION_EXECUTION_GROUP, 0..1),
                (super::super::TEXT_EXECUTION_GROUP, 0..1),
            ],
            &routed_roles,
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not admit routed"));
    }

    #[test]
    fn local_geometry_distinguishes_tied_and_untied_vocabulary_ownership() {
        let mut tied = args();
        tied.tie_word_embeddings = true;
        let geometry = local_geometry(&tied, &valid_layout(false)).unwrap();
        assert!(geometry.output_range().is_none());

        let error = local_geometry(&args(), &valid_layout(false)).unwrap_err();
        assert!(error.to_string().contains("lm_head"));
    }

    #[test]
    fn local_geometry_rejects_zero_widths_and_vocabulary_companion_drift() {
        let args = args();
        let mut zero = valid_layout(true);
        insert(
            &mut zero,
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.key_value_heads",
            vec![8, 16],
            vec![0, 16],
            row_range(0),
        );
        assert!(local_geometry(&args, &zero)
            .unwrap_err()
            .to_string()
            .contains("must be positive"));

        let mut drift = valid_layout(true);
        insert(
            &mut drift,
            "model.embed_tokens.scales",
            "model.embed_tokens",
            vec![24, 1],
            vec![12, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 12,
                end: 24,
            },
        );
        assert!(local_geometry(&args, &drift)
            .unwrap_err()
            .to_string()
            .contains("inconsistent selections"));
    }
}
