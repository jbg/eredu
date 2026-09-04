//! Planner-derived tensor-parallel geometry for the composite Qwen3-VL graph.

use std::ops::Range;

use eredu_runtime::{
    ArchitecturePartition, LocalModelLayout, ParallelPlanError, PartitionOwnership, StateLayout,
};

use crate::qwen::{self, vision};

use super::{prompt_cache_architecture_fingerprint, state_layout_with_key_value_heads, ModelArgs};

/// Complete local text, vision, vocabulary, media, and state geometry.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    text: qwen::LocalGeometry,
    vision_blocks: Vec<(i32, i32)>,
    merger_widths: Vec<i32>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

/// Exact TP/PP-local Qwen3-VL geometry, optionally bound to one selected EP realization.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    vision_units: Option<Range<usize>>,
    vision_blocks: Vec<(i32, i32)>,
    text_units: Range<usize>,
    text: qwen::PartitionLocalGeometry<qwen::ModelArgs>,
    merger_widths: Vec<i32>,
    deepstack_layers: Vec<i32>,
    complete_state_layout: StateLayout,
    static_roles: Vec<String>,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
    architecture_fingerprint: String,
}

/// Validated architecture-owned handoff for one Qwen3-VL composite partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalFoundation {
    geometry: PartitionLocalGeometry,
    parameter_targets: Vec<String>,
}

impl PartitionLocalGeometry {
    /// Architecture-global optional vision units owned here.
    pub fn vision_units(&self) -> Option<Range<usize>> {
        self.vision_units.clone()
    }

    /// Architecture-global text units owned here.
    pub fn text_units(&self) -> Range<usize> {
        self.text_units.clone()
    }

    /// TP-local geometry for one owned vision unit.
    pub fn vision_block(&self, global_unit: usize) -> Option<(i32, i32)> {
        let range = self.vision_units.as_ref()?;
        range
            .contains(&global_unit)
            .then(|| self.vision_blocks[global_unit - range.start])
    }

    /// TP/EP-local text configuration for one owned text unit.
    pub fn text_block(&self, global_unit: usize) -> Option<&qwen::ModelArgs> {
        self.text.block(global_unit)
    }

    /// Main and DeepStack merger widths owned by the optional vision static root.
    pub fn merger_widths(&self) -> &[i32] {
        &self.merger_widths
    }

    /// Vision-layer indices whose projected values are transported to decoder layers.
    pub fn deepstack_layers(&self) -> &[i32] {
        &self.deepstack_layers
    }

    /// Complete TP-local text state before PP slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }

    /// Exact local text state slice.
    pub fn local_state_layout(&self) -> Result<StateLayout, ParallelPlanError> {
        self.complete_state_layout
            .slice(self.text_units.clone())
            .map_err(|error| invalid(error.to_string()))
    }

    /// Selected static roles in canonical graph order.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    /// Immutable selected routed-expert realization.
    pub const fn expert_realization(
        &self,
    ) -> Option<&crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>> {
        self.expert_realization.as_ref()
    }

    fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        let text_count = usize::try_from(args.text.num_hidden_layers)
            .map_err(|_| invalid("Qwen3-VL text layer count is negative"))?;
        let vision_count = args.vision.layer_count();
        if self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.text_units.is_empty()
            || self.text_units.end > text_count
            || self.text.owned_units() != self.text_units
            || self.complete_state_layout.len() != text_count
        {
            return Err(invalid(
                "partition-local Qwen3-VL geometry belongs to another model or text range",
            ));
        }
        match &self.vision_units {
            Some(range)
                if !range.is_empty()
                    && range.end <= vision_count
                    && self.vision_blocks.len() == range.len() => {}
            None if self.vision_blocks.is_empty() => {}
            _ => return Err(invalid("partition-local Qwen3-VL vision range is invalid")),
        }
        if self.merger_widths.len() != args.vision.deepstack_layer_count() + 1
            || self.deepstack_layers != args.vision.deepstack_layers()
        {
            return Err(invalid(
                "partition-local Qwen3-VL merger or DeepStack geometry drifted",
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
            expected_roles.push("norm".into());
            let output = if args.text.tie_word_embeddings {
                "embedding"
            } else {
                "output"
            };
            if !expected_roles.iter().any(|role| role == output) {
                expected_roles.push(output.into());
            }
        }
        if self.static_roles != expected_roles {
            return Err(invalid(format!(
                "partition-local Qwen3-VL static roles {:?} differ from {:?}",
                self.static_roles, expected_roles
            )));
        }
        match (args.text.is_moe(), &self.expert_realization) {
            (false, None) | (true, Some(_)) => {}
            (false, Some(_)) => {
                return Err(invalid("dense Qwen3-VL retained an expert realization"))
            }
            (true, None) => {
                return Err(invalid(
                    "routed Qwen3-VL has no selected expert realization",
                ))
            }
        }
        self.local_state_layout()?;
        Ok(())
    }
}

impl PartitionLocalFoundation {
    /// Validates selected groups, state, DeepStack boundary, and parameter targets together.
    pub fn from_partition(
        args: &ModelArgs,
        partition: &ArchitecturePartition<PartitionLocalGeometry, super::PipelineBoundarySchema>,
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
                "Qwen3-VL partition groups differ from family-local geometry",
            ));
        }
        let state = partition
            .state()
            .ok_or_else(|| invalid("Qwen3-VL text partition has no selected state"))?;
        if state.global_layer_offset() != geometry.text_units.start
            || state.layout() != &geometry.local_state_layout()?
        {
            return Err(invalid(
                "Qwen3-VL partition state differs from its text unit range",
            ));
        }
        if partition.boundary_schema() != &super::PipelineBoundarySchema::from_args(args) {
            return Err(invalid(
                "Qwen3-VL partition boundary differs from mRoPE/DeepStack geometry",
            ));
        }
        let parameter_targets = partition
            .parameter_bindings()
            .iter()
            .flat_map(|binding| binding.members())
            .map(|member| member.target().to_owned())
            .collect();
        Ok(Self {
            geometry: geometry.clone(),
            parameter_targets,
        })
    }

    /// Exact family-local construction geometry.
    pub const fn geometry(&self) -> &PartitionLocalGeometry {
        &self.geometry
    }

    /// Canonical selected materialization targets.
    pub fn parameter_targets(&self) -> &[String] {
        &self.parameter_targets
    }
}

impl LocalGeometry {
    /// Returns the local ordinary-Qwen decoder geometry.
    pub const fn text(&self) -> &qwen::LocalGeometry {
        &self.text
    }

    /// Returns one vision block's local `(heads, intermediate)` geometry.
    pub fn vision_block(&self, layer: usize) -> Option<(i32, i32)> {
        self.vision_blocks.get(layer).copied()
    }

    /// Returns local main/deepstack merger intermediate widths.
    pub fn merger_widths(&self) -> &[i32] {
        &self.merger_widths
    }

    /// Returns authoritative decoder state plus persisted position delta.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        let text_layers = usize::try_from(args.text.num_hidden_layers)
            .map_err(|_| invalid("Qwen3-VL text layer count is negative"))?;
        if self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.text.blocks().len() != text_layers
            || self.vision_blocks.len() != args.vision.layer_count()
            || self.merger_widths.len() != args.vision.deepstack_layer_count() + 1
        {
            return Err(invalid(
                "rank-local Qwen3-VL geometry belongs to another configuration",
            ));
        }
        if self
            .vision_blocks
            .iter()
            .any(|(heads, intermediate)| *heads <= 0 || *intermediate <= 0)
            || self.merger_widths.iter().any(|width| *width <= 0)
        {
            return Err(invalid("Qwen3-VL local vision geometry is zero"));
        }
        let expected = state_layout_with_key_value_heads(
            args,
            &self
                .text
                .blocks()
                .iter()
                .map(|block| block.num_key_value_heads)
                .collect::<Vec<_>>(),
        )
        .map_err(|error| invalid(error.to_string()))?;
        if expected != self.state_layout {
            return Err(invalid("Qwen3-VL local state geometry drifted"));
        }
        Ok(())
    }
}

/// Derives the complete local composite geometry exclusively from one plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let text = qwen::local_geometry(&args.text, layout)?;
    let vision_blocks = (0..args.vision.layer_count())
        .map(|layer| vision::local_block_geometry(&args.vision, "model.visual", layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let merger_widths = vision::local_merger_widths(&args.vision, "model.visual", layout)?;
    let state_layout = state_layout_with_key_value_heads(
        args,
        &text
            .blocks()
            .iter()
            .map(|block| block.num_key_value_heads)
            .collect::<Vec<_>>(),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let geometry = LocalGeometry {
        text,
        vision_blocks,
        merger_widths,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

fn partition_ranges(
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
) -> Result<(Option<Range<usize>>, Range<usize>), ParallelPlanError> {
    let mut vision = None;
    let mut text = None;
    for (group, range) in group_ranges {
        if range.is_empty() {
            return Err(invalid("Qwen3-VL partition group range cannot be empty"));
        }
        let slot = match group.as_ref() {
            super::VISION_EXECUTION_GROUP => &mut vision,
            super::TEXT_EXECUTION_GROUP => &mut text,
            other => {
                return Err(invalid(format!(
                    "unknown Qwen3-VL partition group {other:?}"
                )))
            }
        };
        if slot.replace(range).is_some() {
            return Err(invalid("duplicate Qwen3-VL partition group"));
        }
    }
    Ok((
        vision,
        text.ok_or_else(|| invalid("Qwen3-VL partition has no text decoder range"))?,
    ))
}

/// Derives exact dense TP/PP-local Qwen3-VL composite geometry.
pub fn partition_local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if args.text.is_moe() {
        return Err(invalid(
            "dense partition-local Qwen3-VL requires an explicit selected expert realization",
        ));
    }
    let (vision_units, text_units) = partition_ranges(group_ranges)?;
    let complete = local_geometry(args, layout)?;
    let text = qwen::partition_local_geometry(&args.text, layout, text_units.clone())?;
    finish_partition_geometry(
        args,
        complete,
        vision_units,
        text_units,
        text,
        ownership,
        None,
    )
}

/// Derives exact routed TP/PP/EP-local Qwen3-VL geometry from routed-expert authority.
pub fn partition_local_routed_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if !args.text.is_moe() {
        return Err(invalid(
            "routed partition-local Qwen3-VL requires Qwen3-VL-MoE",
        ));
    }
    let (vision_units, text_units) = partition_ranges(group_ranges)?;
    let complete = local_geometry(args, layout)?;
    let text = qwen::partition_local_routed_geometry(
        &args.text,
        layout,
        text_units.clone(),
        topology,
        realization,
    )?;
    finish_partition_geometry(
        args,
        complete,
        vision_units,
        text_units,
        text,
        ownership,
        Some(realization.clone()),
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_partition_geometry(
    args: &ModelArgs,
    complete: LocalGeometry,
    vision_units: Option<Range<usize>>,
    text_units: Range<usize>,
    text: qwen::PartitionLocalGeometry<qwen::ModelArgs>,
    ownership: &PartitionOwnership,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let vision_blocks = vision_units
        .clone()
        .into_iter()
        .flatten()
        .map(|global| {
            complete
                .vision_block(global)
                .ok_or_else(|| invalid(format!("Qwen3-VL has no local vision block {global}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = PartitionLocalGeometry {
        vision_units,
        vision_blocks,
        text_units,
        text,
        merger_widths: complete.merger_widths.clone(),
        deepstack_layers: args.vision.deepstack_layers(),
        complete_state_layout: complete.state_layout.clone(),
        static_roles: ownership.static_roles().to_vec(),
        expert_realization,
        architecture_fingerprint: complete.architecture_fingerprint,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidTensor(message.into())
}
