//! Planner-derived tensor-parallel geometry for the composite Qwen3-VL graph.

use eredu_runtime::{LocalModelLayout, ParallelPlanError, StateLayout};

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

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidTensor(message.into())
}
