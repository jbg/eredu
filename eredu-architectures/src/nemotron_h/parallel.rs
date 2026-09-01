//! Semantic placement and rank-local geometry for Nemotron-H physical units.

use eredu_nn::{GroupedNeuralBackend, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    partitioned_projection_group, LocalModelLayout, MemberSharding, ParallelPlanError,
    ParameterGroupSpec, ParameterRole, ProjectionSharding, StateLayout, TensorPlacement,
};

use super::{
    prompt_cache_architecture_fingerprint, state_layout_with_geometry, Block, DenseMlp,
    LayerGeometry, LayerPolicy, ModelArgs, Operator, PredictionUnit, Unit,
};
use crate::decoder::StaticModules;

/// Complete planner-derived construction and state geometry for one Nemotron-H rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    target: Vec<LayerGeometry>,
    prediction: Vec<LayerGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
    global_target: Vec<LayerGeometry>,
    global_prediction: Vec<LayerGeometry>,
    prediction_steps: usize,
    prediction_pattern: usize,
    tied_head: bool,
}

impl LocalGeometry {
    /// Returns target-unit geometry in physical execution order.
    pub fn target_units(&self) -> &[LayerGeometry] {
        &self.target
    }

    /// Returns one target unit's rank-local geometry.
    pub fn target_unit(&self, index: usize) -> Option<&LayerGeometry> {
        self.target.get(index)
    }

    /// Returns appended MTP geometry in global physical order.
    pub fn prediction_units(&self) -> &[LayerGeometry] {
        &self.prediction
    }

    /// Returns one MTP unit's rank-local geometry by physical index.
    pub fn prediction_unit(&self, physical: usize) -> Option<&LayerGeometry> {
        self.prediction.get(physical)
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the authoritative heterogeneous target-plus-MTP state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        args.validate()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let expected_target = global_target_geometry(args)?;
        let expected_prediction = global_prediction_geometry(args)?;
        let prediction_steps = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("Nemotron-H prediction-step count exceeds usize".into())
        })?;
        let prediction_pattern = if prediction_steps == 0 {
            0
        } else {
            if expected_prediction.len() % prediction_steps != 0 {
                return Err(ParallelPlanError::InvalidGroup(
                    "Nemotron-H MTP geometry does not divide into prediction steps".into(),
                ));
            }
            expected_prediction
                .len()
                .checked_div(prediction_steps)
                .filter(|pattern| *pattern > 0)
                .ok_or_else(|| {
                    ParallelPlanError::InvalidGroup(
                        "Nemotron-H MTP geometry has an empty prediction pattern".into(),
                    )
                })?
        };
        if self.target.len() != expected_target.len()
            || self.prediction.len() != expected_prediction.len()
            || self.global_target != expected_target
            || self.global_prediction != expected_prediction
            || self.prediction_steps != prediction_steps
            || self.prediction_pattern != prediction_pattern
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.tied_head != args.tie_word_embeddings
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Nemotron-H geometry belongs to a different model configuration".into(),
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
                return Err(ParallelPlanError::InvalidGroup(
                    "tied Nemotron-H output unexpectedly owns a separate vocabulary range".into(),
                ))
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied Nemotron-H output is missing vocabulary ownership".into(),
                ))
            }
        }
        let geometry = self
            .target
            .iter()
            .chain(&self.prediction)
            .copied()
            .collect::<Vec<_>>();
        let expected = state_layout_with_geometry(args, &geometry)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Nemotron-H state layout drifted from unit geometry".into(),
            ));
        }
        Ok(())
    }
}

fn local_width(
    layout: &LocalModelLayout,
    name: &str,
    axis: usize,
) -> Result<i32, ParallelPlanError> {
    let tensor = layout.tensor(name).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing local Nemotron-H layout for {name}"))
    })?;
    let local = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Nemotron-H tensor {name} has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "Nemotron-H tensor {name} has no global axis {axis}"
        ))
    })?;
    if local == 0 || local > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Nemotron-H tensor {name} has invalid local width {local} of global width {global}"
        )));
    }
    i32::try_from(local).map_err(|_| {
        ParallelPlanError::InvalidTensor(format!("Nemotron-H width for {name} exceeds i32"))
    })
}

/// Derives one target block's local geometry from the resolved parameter layout.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<LayerGeometry, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    match args.layer_schedule.get(layer).copied().ok_or_else(|| {
        ParallelPlanError::InvalidGroup(format!("Nemotron-H has no layer {layer}"))
    })? {
        LayerPolicy::Mamba => {
            let heads = local_width(layout, &format!("{root}.mamba.dt_bias"), 0)?;
            if args.n_groups <= 0 || args.mamba_num_heads % args.n_groups != 0 {
                return Err(ParallelPlanError::InvalidGroup(
                    "Nemotron-H Mamba heads do not divide into state groups".into(),
                ));
            }
            let heads_per_group = args.mamba_num_heads / args.n_groups;
            if heads_per_group <= 0 || heads % heads_per_group != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local Mamba heads do not contain complete state groups".into(),
                ));
            }
            Ok(LayerGeometry::Mamba {
                heads,
                groups: heads / heads_per_group,
            })
        }
        LayerPolicy::SelfAttention(_) => {
            let query = local_width(layout, &format!("{root}.attention.q_proj.weight"), 0)?;
            let key = local_width(layout, &format!("{root}.attention.k_proj.weight"), 0)?;
            if args.head_dim <= 0 || query % args.head_dim != 0 || key % args.head_dim != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local attention widths do not contain complete heads".into(),
                ));
            }
            Ok(LayerGeometry::Attention {
                query_heads: query / args.head_dim,
                kv_heads: key / args.head_dim,
            })
        }
        LayerPolicy::DenseMlp => Ok(LayerGeometry::DenseMlp {
            intermediate: local_width(layout, &format!("{root}.mlp.up_proj.weight"), 0)?,
        }),
        LayerPolicy::SparseMoe => Ok(LayerGeometry::SparseMoe {
            routed: local_width(layout, &format!("{root}.moe.experts.up_proj"), 1)?,
            shared: local_width(
                layout,
                &format!("{root}.moe.shared_experts.up_proj.weight"),
                0,
            )?,
        }),
    }
}

/// Returns resolved state geometry for target and appended MTP units.
pub fn local_state_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    let mut geometry = (0..args.num_hidden_layers as usize)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let target = args.num_hidden_layers as usize;
    for (physical, policy) in args
        .mtp_policies()
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?
        .into_iter()
        .enumerate()
    {
        let root = format!("model.mtp.layers.{physical}.mixer");
        geometry.push(match policy {
            LayerPolicy::SelfAttention(_) => {
                let query = local_width(layout, &format!("{root}.q_proj.weight"), 0)?;
                let key = local_width(layout, &format!("{root}.k_proj.weight"), 0)?;
                if args.head_dim <= 0
                    || query % args.head_dim != 0
                    || key % args.head_dim != 0
                {
                    return Err(ParallelPlanError::InvalidTensor(format!(
                        "local Nemotron-H MTP attention widths at physical layer {physical} do not contain complete heads"
                    )));
                }
                LayerGeometry::Attention {
                    query_heads: query / args.head_dim,
                    kv_heads: key / args.head_dim,
                }
            }
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: local_width(layout, &format!("{root}.experts.up_proj"), 1)?,
                shared: local_width(layout, &format!("{root}.shared_experts.up_proj.weight"), 0)?,
            },
            _ => {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "unsupported MTP policy at global state layer {}",
                    target + physical
                )))
            }
        });
    }
    Ok(geometry)
}

fn global_target_geometry(args: &ModelArgs) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    Ok(args
        .layer_schedule
        .iter()
        .map(|policy| match policy {
            LayerPolicy::Mamba => LayerGeometry::Mamba {
                heads: args.mamba_num_heads,
                groups: args.n_groups,
            },
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::DenseMlp => LayerGeometry::DenseMlp {
                intermediate: args.intermediate_size,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            },
        })
        .collect())
}

fn global_prediction_geometry(args: &ModelArgs) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    Ok(args
        .mtp_policies()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
        .into_iter()
        .map(|policy| match policy {
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            },
            _ => unreachable!("validated Nemotron-H MTP schedules contain attention and MoE"),
        })
        .collect())
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    global_vocabulary: usize,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected = None;
    let mut found = false;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        found = true;
        if tensor.global_shape().first().copied() != Some(global_vocabulary) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Nemotron-H vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
                tensor.global_shape()
            )));
        }
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated => 0..global_vocabulary,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "Nemotron-H vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Nemotron-H vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local Nemotron-H vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Derives complete rank-local target, MTP, vocabulary, and state geometry.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let geometry = local_state_geometry(args, layout)?;
    let target_count = usize::try_from(args.num_hidden_layers).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H target layer count exceeds usize".into())
    })?;
    let (target, prediction) = geometry.split_at(target_count);
    let state_layout = state_layout_with_geometry(args, &geometry)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let vocabulary = usize::try_from(args.vocab_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H vocabulary exceeds usize".into())
    })?;
    let embedding_range = vocabulary_range(layout, "model.embeddings", vocabulary)?;
    let output_range = if args.tie_word_embeddings {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", vocabulary)?)
    };
    let prediction_steps = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H prediction-step count exceeds usize".into())
    })?;
    let prediction_pattern = if prediction_steps == 0 {
        0
    } else {
        if prediction.len() % prediction_steps != 0 {
            return Err(ParallelPlanError::InvalidGroup(
                "Nemotron-H MTP geometry does not divide into prediction steps".into(),
            ));
        }
        prediction
            .len()
            .checked_div(prediction_steps)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                ParallelPlanError::InvalidGroup(
                    "Nemotron-H MTP geometry has an empty prediction pattern".into(),
                )
            })?
    };
    let local = LocalGeometry {
        target: target.to_vec(),
        prediction: prediction.to_vec(),
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        global_target: global_target_geometry(args)?,
        global_prediction: global_prediction_geometry(args)?,
        prediction_steps,
        prediction_pattern,
        tied_head: args.tie_word_embeddings,
    };
    local.validate_for(args)?;
    Ok(local)
}

/// Declares vocabulary and replicated final-normalization groups.
pub fn static_parallel_parameter_groups<B: GroupedNeuralBackend>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            "model.embeddings",
            ParameterRole::Vocabulary,
            &modules.embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "Nemotron-H embedding is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.norm_f",
            ParameterRole::Replicated,
            &modules.norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ];
    if let Some(head) = &modules.lm_head {
        groups.push(module_parameter_group::<B::Tensor, _>(
            "lm_head",
            ParameterRole::Vocabulary,
            head,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "Nemotron-H output is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

fn dense_groups<B: GroupedNeuralBackend>(
    root: &str,
    mlp: &DenseMlp<B>,
    width: i32,
    role: ParameterRole,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![partitioned_projection_group::<B::Tensor, B::Linear>(
        format!("{root}.intermediate"),
        role,
        &[
            (&mlp.up_proj, ProjectionSharding::Column),
            (&mlp.down_proj, ProjectionSharding::Row),
        ],
        aligned_partition_units(
            root,
            usize::try_from(width).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "Nemotron-H intermediate width exceeds usize".into(),
                )
            })?,
            1,
            1,
        )?,
    )?])
}

/// Declares semantic groups for one target physical block.
pub fn layer_parallel_parameter_groups<B: GroupedNeuralBackend>(
    block: &Block<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    block_parallel_parameter_groups(block, args, &root)
}

fn block_parallel_parameter_groups<B: GroupedNeuralBackend>(
    block: &Block<B>,
    args: &ModelArgs,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![module_parameter_group::<B::Tensor, _>(
        format!("{root}.norm"),
        ParameterRole::Replicated,
        &block.norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?];
    match &block.operator {
        Operator::Mamba(mamba) => {
            let heads = usize::try_from(args.mamba_num_heads)
                .map_err(|_| ParallelPlanError::InvalidGroup("Mamba heads exceed usize".into()))?;
            let intermediate = usize::try_from(args.mamba_num_heads * args.mamba_head_dim)
                .map_err(|_| ParallelPlanError::InvalidGroup("Mamba width exceeds usize".into()))?;
            let grouped = usize::try_from(args.n_groups * args.ssm_state_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Mamba state width exceeds usize".into())
            })?;
            let segments = vec![
                0..intermediate,
                intermediate..2 * intermediate,
                2 * intermediate..2 * intermediate + grouped,
                2 * intermediate + grouped..2 * intermediate + 2 * grouped,
                2 * intermediate + 2 * grouped..2 * intermediate + 2 * grouped + heads,
            ];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.mamba.heads"),
                ParameterRole::Channels,
                usize::try_from(args.n_groups).map_err(|_| {
                    ParallelPlanError::InvalidGroup("Mamba group count exceeds usize".into())
                })?,
                mamba,
                |metadata, shape| {
                    let name = metadata
                        .linear_companion_of
                        .as_ref()
                        .unwrap_or(&metadata.id)
                        .as_str();
                    if name.ends_with("in_proj.weight") || name.ends_with("in_proj.bias") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: segments.clone(),
                        })
                    } else if name.ends_with("conv1d.weight")
                        || name.ends_with("conv1d.bias")
                        || name.ends_with("dt_bias")
                        || name.ends_with("A_log")
                        || name.ends_with("D")
                        || name.ends_with("norm.weight")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("out_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
        }
        Operator::Attention(attention) => {
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.attention.heads"),
                ParameterRole::AttentionHeads,
                usize::try_from(args.num_key_value_heads).map_err(|_| {
                    ParallelPlanError::InvalidGroup("attention head count exceeds usize".into())
                })?,
                attention,
                |metadata, shape| {
                    let name = metadata
                        .linear_companion_of
                        .as_ref()
                        .unwrap_or(&metadata.id)
                        .as_str();
                    if name.ends_with("q_proj.weight")
                        || name.ends_with("k_proj.weight")
                        || name.ends_with("v_proj.weight")
                        || name.ends_with("q_proj.bias")
                        || name.ends_with("k_proj.bias")
                        || name.ends_with("v_proj.bias")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("o_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?)
        }
        Operator::Dense(mlp) => groups.extend(dense_groups(
            &format!("{root}.mlp"),
            mlp,
            args.intermediate_size,
            ParameterRole::FeedForwardIntermediate,
        )?),
        Operator::Sparse(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.moe.gate"),
                ParameterRole::Replicated,
                &moe.gate,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.moe.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                usize::try_from(args.moe_intermediate_size).map_err(|_| {
                    ParallelPlanError::InvalidGroup("expert width exceeds usize".into())
                })?,
                &moe.experts,
                |metadata, _| {
                    let name = metadata
                        .linear_companion_of
                        .as_ref()
                        .unwrap_or(&metadata.id)
                        .as_str();
                    if name.contains("up_proj") {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    }
                },
            )?);
            groups.extend(dense_groups(
                &format!("{root}.moe.shared_experts"),
                &moe.shared_experts,
                args.moe_shared_expert_intermediate_size,
                ParameterRole::ExpertIntermediate,
            )?);
        }
    }
    Ok(groups)
}

/// Declares semantic placement for one target or appended prediction unit.
pub fn unit_parallel_parameter_groups<B: GroupedNeuralBackend>(
    unit: &Unit<B>,
    args: &ModelArgs,
    flat: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("layer count exceeds usize".into()))?;
    match unit {
        Unit::Target(block) if flat < target => layer_parallel_parameter_groups(block, args, flat),
        Unit::Prediction(prediction) if flat >= target => {
            prediction_parallel_parameter_groups(prediction, args, flat - target)
        }
        _ => Err(ParallelPlanError::InvalidGroup(format!(
            "Nemotron-H unit kind does not match flat position {flat}"
        ))),
    }
}

fn prediction_parallel_parameter_groups<B: GroupedNeuralBackend>(
    unit: &PredictionUnit<B>,
    args: &ModelArgs,
    physical: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.mtp.layers.{physical}");
    let mut groups = Vec::new();
    if let Some(norm) = &unit.embedding_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.enorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(norm) = &unit.hidden_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.hnorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(fusion) = &unit.fusion {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.eh_proj"),
            ParameterRole::Replicated,
            fusion,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    groups.extend(block_parallel_parameter_groups(
        &unit.block,
        args,
        &format!("{root}.mixer"),
    )?);
    if let Some(norm) = &unit.final_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.final_layernorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::{LocalTensorLayout, ParameterRole};

    fn args() -> ModelArgs {
        crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-E", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
            "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
            "num_experts_per_tok":2, "n_group":2, "topk_group":1,
            "num_nextn_predict_layers":1, "mtp_hybrid_override_pattern":"*E",
            "tie_word_embeddings":false
        }))
        .unwrap()
    }

    fn insert(
        layout: &mut LocalModelLayout,
        target: &str,
        logical: &str,
        global: Vec<usize>,
        local: Vec<usize>,
        placement: TensorPlacement,
    ) {
        layout.insert(
            target.into(),
            LocalTensorLayout::new(
                logical,
                ParameterRole::Replicated,
                global,
                local,
                placement,
                None,
                None,
                false,
            ),
        );
    }

    fn range(axis: usize, end: usize) -> TensorPlacement {
        TensorPlacement::Range {
            axis,
            start: 0,
            end,
        }
    }

    fn valid_layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        insert(
            &mut layout,
            "model.embeddings.weight",
            "model.embeddings",
            vec![32, 16],
            vec![16, 16],
            range(0, 16),
        );
        insert(
            &mut layout,
            "lm_head.weight",
            "lm_head",
            vec![32, 16],
            vec![16, 16],
            range(0, 16),
        );
        insert(
            &mut layout,
            "model.layers.0.mamba.dt_bias",
            "model.layers.0.mamba.heads",
            vec![4],
            vec![2],
            range(0, 2),
        );
        insert(
            &mut layout,
            "model.layers.1.attention.q_proj.weight",
            "model.layers.1.attention.heads",
            vec![16, 16],
            vec![8, 16],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.1.attention.k_proj.weight",
            "model.layers.1.attention.heads",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        insert(
            &mut layout,
            "model.layers.2.mlp.up_proj.weight",
            "model.layers.2.mlp.intermediate",
            vec![24, 16],
            vec![12, 16],
            range(0, 12),
        );
        insert(
            &mut layout,
            "model.layers.3.moe.experts.up_proj",
            "model.layers.3.moe.experts.intermediate",
            vec![4, 8, 16],
            vec![4, 4, 16],
            range(1, 4),
        );
        insert(
            &mut layout,
            "model.layers.3.moe.shared_experts.up_proj.weight",
            "model.layers.3.moe.shared_experts.intermediate",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.q_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![16, 16],
            vec![8, 16],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.k_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        insert(
            &mut layout,
            "model.mtp.layers.1.mixer.experts.up_proj",
            "model.mtp.layers.1.mixer.experts.intermediate",
            vec![4, 8, 16],
            vec![4, 4, 16],
            range(1, 4),
        );
        insert(
            &mut layout,
            "model.mtp.layers.1.mixer.shared_experts.up_proj.weight",
            "model.mtp.layers.1.mixer.shared_experts.intermediate",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        layout
    }

    #[test]
    fn local_geometry_owns_target_mtp_vocabulary_and_state_together() {
        let args = args();
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert_eq!(geometry.target_units().len(), 4);
        assert_eq!(geometry.prediction_units().len(), 2);
        assert_eq!(
            geometry.target_unit(0),
            Some(&LayerGeometry::Mamba {
                heads: 2,
                groups: 1
            })
        );
        assert_eq!(
            geometry.prediction_unit(0),
            Some(&LayerGeometry::Attention {
                query_heads: 2,
                kv_heads: 1
            })
        );
        assert_eq!(geometry.embedding_range().local, 0..16);
        assert_eq!(geometry.output_range().unwrap().local, 0..16);
        assert_ne!(
            geometry.state_layout(),
            &crate::nemotron_h::state_layout(&args).unwrap()
        );
        geometry.validate_for(&args).unwrap();
    }

    #[test]
    fn local_geometry_rejects_incomplete_or_zero_mtp_heads() {
        let args = args();
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.q_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![16, 16],
            vec![7, 16],
            range(0, 7),
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("complete heads"));

        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.k_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![8, 16],
            vec![0, 16],
            range(0, 0),
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("invalid local width 0"));
    }

    #[test]
    fn local_geometry_rejects_vocabulary_companion_drift() {
        let args = args();
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.embeddings.scales",
            "model.embeddings",
            vec![32, 1],
            vec![16, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 16,
                end: 32,
            },
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("inconsistent companion selections"));
    }

    #[test]
    fn local_geometry_validation_preserves_prediction_grouping() {
        let args = args();
        let mut geometry = local_geometry(&args, &valid_layout()).unwrap();
        geometry.prediction_steps = 2;
        geometry.prediction_pattern = 1;
        assert!(geometry
            .validate_for(&args)
            .unwrap_err()
            .to_string()
            .contains("different model configuration"));
    }
}
