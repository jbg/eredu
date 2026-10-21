//! Shared-matrix placement with independent invocation and feedback parameters.

use super::FamilyConfig;
use crate::gemma4::ModelArgs;
use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{
    expand_linear_format_parameter_groups, LocalModelLayout, MemberSharding, ParallelPlanError,
    ParameterGroupSpec, ParameterMemberSpec, ParameterRole, StateLayout, TensorPlacement,
};

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidTensor(message.into())
}

/// Declares shared vocabulary/readout and independently sharded feedback units.
pub fn static_parameter_groups(
    config: &FamilyConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let args = &config.text;
    let mut groups = crate::gemma4::parallel::static_parameter_groups_at(args, "model.decoder")?;
    let hidden =
        usize::try_from(args.hidden_size).map_err(|_| invalid("invalid feedback hidden width"))?;
    let width = args
        .layer_policy(0)
        .ok_or_else(|| invalid("empty diffusion stack"))?
        .intermediate_size
        .get() as usize;
    let root = "model.decoder.self_conditioning";
    let member = |field: &str, shape, sharding| {
        ParameterMemberSpec::new(format!("{root}.{field}.weight"), shape, sharding)
    };
    let feedback = vec![
        ParameterGroupSpec::new(
            format!("{root}.norm"),
            ParameterRole::Replicated,
            [member("pre_norm", vec![hidden], MemberSharding::Replicated)],
        )?,
        ParameterGroupSpec::partitioned(
            format!("{root}.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            width,
            [
                member(
                    "gate_proj",
                    vec![width, hidden],
                    MemberSharding::Partitioned { axis: 0 },
                ),
                member(
                    "up_proj",
                    vec![width, hidden],
                    MemberSharding::Partitioned { axis: 0 },
                ),
                member(
                    "down_proj",
                    vec![hidden, width],
                    MemberSharding::Partitioned { axis: 1 },
                ),
            ],
        )?,
    ];
    groups.extend(expand_linear_format_parameter_groups(feedback, |member| {
        crate::linear_format::standard_parallel_linear_format(
            member,
            args.linear_format_for(member.target()),
        )
    })?);
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
    )?;
    let target = format!("model.encoder.language_model.layers.{index}.layer_scalar");
    groups.push(ParameterGroupSpec::new(
        target.clone(),
        ParameterRole::Replicated,
        [ParameterMemberSpec::new(
            target,
            vec![1],
            MemberSharding::Replicated,
        )],
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

fn vocabulary_range(
    layout: &LocalModelLayout,
    target: &str,
    vocabulary: i32,
) -> Result<Option<VocabularyParallelRange>, ParallelPlanError> {
    let Some(tensor) = layout.tensor(target) else {
        return Ok(None);
    };
    let local = match tensor.placement() {
        TensorPlacement::Replicated => 0..vocabulary as usize,
        TensorPlacement::Range {
            axis: 0,
            start,
            end,
        } => *start..*end,
        _ => {
            return Err(invalid(
                "diffusion vocabulary placement must select complete rows",
            ))
        }
    };
    let range = VocabularyParallelRange {
        global_vocabulary: vocabulary as usize,
        local,
    };
    range
        .validate_global_rows(vocabulary)
        .map_err(|error| invalid(error.to_string()))?;
    Ok(Some(range))
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
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let policies = blocks
        .iter()
        .zip(units.clone())
        .map(|(args, index)| {
            let policy = args
                .layer_policy(index)
                .ok_or_else(|| invalid("missing local diffusion attention policy"))?;
            eredu_core::cache::LayerCachePolicy::key_value(
                policy.attention,
                policy.num_key_value_heads.get() as i32,
                policy.head_dim.get() as i32,
            )
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
    let conditioning_width = layout
        .tensor("model.decoder.self_conditioning.gate_proj.weight")
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
    Ok(LocalGeometry {
        blocks,
        units,
        state,
        conditioning_width,
        norm: layout.tensor("model.decoder.norm.weight").is_some(),
        embedding: vocabulary_range(
            layout,
            "model.decoder.embed_tokens.weight",
            config.text.vocab_size,
        )?,
        output: vocabulary_range(layout, "lm_head.weight", config.text.vocab_size)?,
        fingerprint: config.architecture_fingerprint(),
    })
}

/// Constructs the retained pipeline partition with exact rank-local state.
/// The complete ordinal schedule preserves global positions; only its selected
/// interval is materialized or used for admission on this rank.
pub fn partition(
    config: &FamilyConfig,
    parameters: &eredu_runtime::ArchitectureParameterDescription,
    layout: &LocalModelLayout,
    units: std::ops::Range<usize>,
    ownership: eredu_runtime::PartitionOwnership,
) -> Result<
    eredu_runtime::ArchitecturePartition<LocalGeometry, super::boundary::TextBoundarySchema>,
    ParallelPlanError,
> {
    let geometry = local_geometry(config, layout, units.clone())?;
    let definition = super::model::component_definition(config, Some(&geometry))
        .map_err(|error| invalid(error.to_string()))?;
    if parameters != definition.parameters() {
        return Err(invalid(
            "diffusion text partition changes canonical parameter ownership",
        ));
    }
    eredu_runtime::ArchitecturePartition::from_description(
        parameters,
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
    let mut local = config.clone();
    let layers = config
        .text
        .layer_schedule
        .iter()
        .copied()
        .map(|mut policy| {
            let range = eredu_core::balanced_contiguous_range(
                policy.num_key_value_heads.get() as usize,
                rank.tensor_parallel_size(),
                rank.tensor_parallel_rank(),
                false,
            )
            .map_err(|error| invalid(error.to_string()))?;
            policy.num_key_value_heads = std::num::NonZeroU32::new(range.len() as u32)
                .ok_or_else(|| invalid("local authoritative head count is zero"))?;
            Ok(policy)
        })
        .collect::<Result<Vec<_>, ParallelPlanError>>()?;
    local.text.layer_schedule = eredu_core::LayerSchedule::new(layers.len(), layers)
        .map_err(|error| invalid(error.to_string()))?;
    super::state_layout(&local).map_err(|error| invalid(error.to_string()))
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
    eredu_runtime::ArchitectureParameterDescription::new(
        &graph,
        &layout,
        owned.iter().map(|owned| owned.group().clone()),
        owned.clone(),
    )
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
