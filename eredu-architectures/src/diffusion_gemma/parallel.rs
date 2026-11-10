//! Shared-matrix placement with independent invocation and feedback parameters.

use super::FamilyConfig;
use crate::gemma4::ModelArgs;
use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{
    LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
    StateLayout,
};

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidTensor(message.into())
}

/// Declares shared vocabulary/readout and independently sharded feedback units.
pub fn static_parameter_groups(
    config: &FamilyConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = config
        .pinned()
        .ordinary
        .pinned_parameter_groups(
            crate::decoder::ParameterEncoding::Stored,
            "model.decoder.embed_tokens",
            "model.decoder.norm",
            "lm_head",
        )?
        .into_iter()
        .map(|(_, group)| group)
        .collect::<Vec<_>>();
    let declaration = config.conditioning();
    groups.extend(
        declaration
            .parameter_groups()
            .map_err(|e| invalid(e.to_string()))?,
    );
    Ok(groups)
}

/// Declares shared layer matrices exactly once and both invocation scales.
pub fn layer_parameter_groups(
    config: &FamilyConfig,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = crate::gemma4::parallel::layer_parameter_groups_at(
        &config.text,
        index,
        "model.decoder.layers",
        config.block(index).map_err(|e| invalid(e.to_string()))?,
        config.dense(index).map_err(|e| invalid(e.to_string()))?,
        config
            .attention(index)
            .map_err(|e| invalid(e.to_string()))?,
        config.routed(index).map_err(|e| invalid(e.to_string()))?,
    )?;
    let spec = config
        .encoder_scalar(index)
        .map_err(|e| invalid(e.to_string()))?;
    groups.push(ParameterGroupSpec::new(
        spec.weight.id.as_str(),
        ParameterRole::Replicated,
        eredu_runtime::logical_parameter_members(spec, |_| MemberSharding::Replicated)?,
    )?);
    Ok(groups)
}

/// Exact rank-local construction geometry derived from retained parameter placement.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    blocks: Vec<ModelArgs>,
    units: std::ops::Range<usize>,
    embedding: Option<VocabularyParallelRange>,
    output: Option<VocabularyParallelRange>,
    conditioning_width: Option<i32>,
    norm: bool,
    state: Option<StateLayout>,
    fingerprint: String,
}

impl LocalGeometry {
    /// Shared stack units owned by this pipeline stage.
    pub fn units(&self) -> std::ops::Range<usize> {
        self.units.clone()
    }
    /// Local state geometry in the same order as owned stack units.
    pub fn state_layout(&self) -> Option<&StateLayout> {
        self.state.as_ref()
    }
    /// Complete global slot schedule with this coordinate's owned heads localized.
    /// Unowned slots retain global geometry and are excluded by partition selection.
    pub fn complete_state_layout(
        &self,
        config: &FamilyConfig,
    ) -> Result<StateLayout, ParallelPlanError> {
        self.validate_for(config)?;
        let mut schedule = super::state_layout(config)
            .map_err(|error| invalid(error.to_string()))?
            .layers()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for (global, policy) in self
            .units()
            .zip(self.state.iter().flat_map(|state| state.layers().iter()))
        {
            schedule[global] = policy.clone();
        }
        StateLayout::new(
            eredu_core::LayerSchedule::new(schedule.len(), schedule)
                .map_err(|error| invalid(error.to_string()))?,
        )
        .map_err(|error| invalid(error.to_string()))
    }
    /// Exact per-unit head, dense and routed widths.
    pub fn block_args(&self, global_index: usize) -> Option<&ModelArgs> {
        global_index
            .checked_sub(self.units.start)
            .and_then(|index| self.blocks.get(index))
    }
    /// Vocabulary rows owned by the shared lookup and tied readout.
    pub fn embedding_range(&self) -> Option<&VocabularyParallelRange> {
        self.embedding.as_ref()
    }
    /// Vocabulary rows owned by an independent output projection.
    pub fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output.as_ref()
    }
    /// Whether this stage owns final normalization.
    pub fn owns_norm(&self) -> bool {
        self.norm
    }
    /// Feedback gate/up units owned by this rank.
    pub fn conditioning_width(&self) -> Option<i32> {
        self.conditioning_width
    }
    /// Projects pinned ownership by canonical identity without changing retained placement.
    pub(super) fn select_pinned_resources(
        &self,
        config: &FamilyConfig,
        definition: &eredu_runtime::ComponentDefinition,
        bindings: &[eredu_runtime::OwnedParameterGroupSpec],
    ) -> Result<Self, eredu_nn::Error> {
        self.validate_for(config)
            .map_err(eredu_nn::Error::backend_source)?;
        let roles = definition.select_complete_pinned_roles(
            bindings,
            &["embedding", "norm", "output", "conditioning"],
        )?;
        let mut selected = self.clone();
        let embedding = roles.contains("embedding")
            || (config.text.tie_word_embeddings && roles.contains("output"));
        let output = !config.text.tie_word_embeddings && roles.contains("output");
        if (embedding && self.embedding.is_none())
            || (output && self.output.is_none())
            || (roles.contains("norm") && !self.norm)
            || (roles.contains("conditioning") && self.conditioning_width.is_none())
        {
            return Err(eredu_nn::Error::backend(
                "diffusion selected pinned resource has no retained placement",
            ));
        }
        if !embedding {
            selected.embedding = None;
        }
        if !output {
            selected.output = None;
        }
        if !roles.contains("conditioning") {
            selected.conditioning_width = None;
        }
        selected.norm = roles.contains("norm");
        Ok(selected)
    }
    /// Rejects reuse of local geometry with a different family configuration.
    pub fn validate_for(&self, config: &FamilyConfig) -> Result<(), ParallelPlanError> {
        if self.fingerprint != config.architecture_fingerprint() {
            return Err(invalid(
                "diffusion local geometry belongs to a different architecture",
            ));
        }
        Ok(())
    }
}

/// Derives local invocation geometry without duplicating shared parameters.
pub fn local_geometry(
    config: &FamilyConfig,
    layout: &LocalModelLayout,
    units: std::ops::Range<usize>,
) -> Result<LocalGeometry, ParallelPlanError> {
    if units.start > units.end || units.end > config.text.num_hidden_layers() {
        return Err(invalid(
            "diffusion local stack range is reversed or out of bounds",
        ));
    }
    let blocks = units
        .clone()
        .map(|index| {
            crate::gemma4::parallel::local_block_args_at(
                &config.text,
                index,
                layout,
                "model.decoder.layers",
                config
                    .attention(index)
                    .map_err(|e| invalid(e.to_string()))?,
                config
                    .attention(
                        config
                            .text
                            .attention_state_owner(index)
                            .map_err(|e| invalid(e.to_string()))?,
                    )
                    .map_err(|e| invalid(e.to_string()))?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (local, index) in blocks.iter().zip(units.clone()) {
        if let Some(scope) = config.routed(index).map_err(|e| invalid(e.to_string()))? {
            scope
                .for_invocation(local, index, None)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let width = local
            .layer_policy(index)
            .ok_or_else(|| invalid("dense layer outside local schedule"))?
            .intermediate_size
            .get() as i32;
        config
            .dense(index)
            .and_then(|spec| spec.clone().with_intermediate(width))
            .map_err(|e| invalid(e.to_string()))?;
        config
            .block(index)
            .and_then(|spec| {
                spec.clone()
                    .with_input_width(local.hidden_size_per_layer_input)
            })
            .map_err(|e| invalid(e.to_string()))?;
    }
    let policies = blocks
        .iter()
        .zip(units.clone())
        .map(|(args, index)| {
            config
                .attention(index)
                .and_then(|spec| spec.for_invocation(args, index))
                .and_then(|spec| spec.state_policy(None, Vec::new()))
                .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state = if policies.is_empty() {
        None
    } else {
        Some(
            StateLayout::new(
                eredu_core::LayerSchedule::new(policies.len(), policies)
                    .map_err(|error| invalid(error.to_string()))?,
            )
            .map_err(|error| invalid(error.to_string()))?,
        )
    };
    let conditioning = config.conditioning();
    let conditioning_width = layout
        .tensor(conditioning.product.gate.weight.id.as_str())
        .map(|tensor| {
            let width = *tensor
                .local_shape()
                .first()
                .ok_or_else(|| invalid("feedback projection has no output axis"))?;
            i32::try_from(width)
                .ok()
                .filter(|width| *width > 0)
                .ok_or_else(|| invalid("feedback local width is empty or exceeds i32"))
        })
        .transpose()?;
    let (embedding, output) = config.pinned().ordinary.selected_vocabulary_ranges(
        layout,
        "model.decoder.embed_tokens",
        "model.decoder.norm",
        "lm_head",
    )?;
    Ok(LocalGeometry {
        blocks,
        units,
        state,
        conditioning_width,
        norm: layout.tensor("model.decoder.norm.weight").is_some(),
        embedding,
        output,
        fingerprint: config.architecture_fingerprint(),
    })
}

/// Constructs the retained pipeline partition with exact rank-local state.
/// The complete ordinal schedule preserves global positions; only its selected
/// interval is materialized or used for admission on this rank.
pub fn partition(
    config: &FamilyConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    layout: &LocalModelLayout,
    units: std::ops::Range<usize>,
    ownership: eredu_runtime::PartitionOwnership,
) -> Result<
    eredu_runtime::ArchitecturePartition<LocalGeometry, super::boundary::TextBoundarySchema>,
    ParallelPlanError,
> {
    let geometry = local_geometry(config, layout, units.clone())?;
    let definition = super::model::select_definition(config, definition, Some(&geometry))
        .map_err(|error| invalid(error.to_string()))?;
    eredu_runtime::ArchitecturePartition::from_description(
        definition.parameters(),
        [(super::model::TEXT_EXECUTION_GROUP, units)],
        ownership,
        definition.state_layout(),
        definition.state_partition_plan(),
        geometry,
        super::boundary::TextBoundarySchema::new(config.text.hidden_size),
    )
    .map_err(|error| invalid(error.to_string()))
}

/// Preserves the family's authoritative K/V representation while localizing heads.
pub fn tensor_state_layout(
    config: &FamilyConfig,
    rank: eredu_core::ParallelRankTopology,
) -> Result<StateLayout, ParallelPlanError> {
    let policies = (0..config.text.num_hidden_layers())
        .map(|layer| {
            let spec = config
                .attention(layer)
                .map_err(|e| invalid(e.to_string()))?;
            let range = eredu_core::balanced_contiguous_range(
                spec.geometry.key_value_heads() as usize,
                rank.tensor_parallel_size(),
                rank.tensor_parallel_rank(),
                false,
            )
            .map_err(|error| invalid(error.to_string()))?;
            spec.state_policy(Some(range), Vec::new())
                .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(
        eredu_core::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| invalid(error.to_string()))?,
    )
    .map_err(|error| invalid(error.to_string()))
}

/// Declares global shared parameters and their semantic owners without a backend.
pub fn parameter_description(
    config: &FamilyConfig,
) -> Result<eredu_runtime::ArchitectureParameterDescription, ParallelPlanError> {
    let graph = eredu_runtime::ExecutionGraph::chain([super::model::TEXT_EXECUTION_GROUP])
        .map_err(|error| invalid(error.to_string()))?;
    let layout = eredu_runtime::ExecutionUnitLayout::new(&graph, [config.text.num_hidden_layers()])
        .map_err(|error| invalid(error.to_string()))?;
    let mut owned = Vec::new();
    for group in static_parameter_groups(config).map_err(|error| invalid(error.to_string()))? {
        let owner = if group.logical_name().ends_with("embed_tokens") {
            if config.text.tie_word_embeddings {
                eredu_runtime::ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                eredu_runtime::ParameterGroupOwner::static_role("embedding")
            }
        } else if group.logical_name() == "model.decoder.norm" {
            eredu_runtime::ParameterGroupOwner::static_role("norm")
        } else if group.logical_name() == "lm_head" {
            eredu_runtime::ParameterGroupOwner::static_role("output")
        } else {
            eredu_runtime::ParameterGroupOwner::static_role("conditioning")
        };
        owned.push(eredu_runtime::OwnedParameterGroupSpec::new(owner, group));
    }
    for index in 0..config.text.num_hidden_layers() {
        for group in
            layer_parameter_groups(config, index).map_err(|error| invalid(error.to_string()))?
        {
            owned.push(eredu_runtime::OwnedParameterGroupSpec::new(
                eredu_runtime::ParameterGroupOwner::execution_unit(
                    layout.group_id(0).unwrap().clone(),
                    index,
                ),
                group,
            ));
        }
    }
    eredu_runtime::ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(|error| invalid(error.to_string()))
}

/// Exact image-stack cuts and rank-local shared text state.
#[derive(Debug, Clone)]
pub struct CompositeGeometry {
    /// Local text parameters and mutable context ownership.
    pub text: LocalGeometry,
    /// Independently traversed image layers on this stage.
    pub vision_units: std::ops::Range<usize>,
    /// Complete tensor-local schedule used to derive the selected state view.
    pub complete_state: StateLayout,
}
