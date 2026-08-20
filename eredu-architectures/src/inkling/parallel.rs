//! Semantic tensor-parallel placement for Inkling text and media components.

use std::collections::BTreeMap;

use eredu_runtime::{
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
};

use super::{FeedForwardPolicy, ModelArgs};

/// Derives the construction geometry for one rank-local Inkling decoder layer.
///
/// Hidden-width tensors and replicated state remain global. Head and
/// intermediate dimensions are taken from the semantic placement result so
/// the ordinary layer constructor creates exactly the local projections.
pub fn local_text_args(
    args: &super::TextArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<super::TextArgs, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Inkling layer {layer} is out of range")))?;
    let root = format!("model.layers.{layer}");
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{root}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}.inner.weight")))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}")))
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {root}.{suffix}"
                ))
            })
    };
    let query_width = local_axis(tensor("self_attn.q_proj")?, 0, "query width")?;
    let key_width = local_axis(tensor("self_attn.k_proj")?, 0, "key width")?;
    let head_dim = args.attention_head_dim(policy.attention.window().is_some());
    if query_width % head_dim != 0 || key_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Inkling attention widths q={query_width}, k={key_width} split head dimension {head_dim}"
        )));
    }
    let mut local = args.clone();
    if policy.attention.window().is_some() {
        local.swa_num_attention_heads = Some(query_width / head_dim);
        local.swa_num_key_value_heads = Some(key_width / head_dim);
    } else {
        local.num_attention_heads = query_width / head_dim;
        local.num_key_value_heads = key_width / head_dim;
    }
    match policy.feed_forward {
        FeedForwardPolicy::Dense => {
            local.dense_intermediate_size = Some(local_axis(
                tensor("dense.gate_proj")?,
                0,
                "dense intermediate width",
            )?);
        }
        FeedForwardPolicy::SparseMoe => {
            let fused = local_axis(
                tensor("moe.experts.gate_up_proj")?,
                1,
                "expert gate/up width",
            )?;
            if fused % 2 != 0 {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "local Inkling expert gate/up width {fused} is not even"
                )));
            }
            local.moe_intermediate_size = Some(fused / 2);
        }
    }
    Ok(local)
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<i32, ParallelPlanError> {
    let value = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("local Inkling {label} tensor has no axis {axis}"))
    })?;
    i32::try_from(value)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("local Inkling {label} exceeds i32")))
}

/// Declares vocabulary, final normalization, output, and small media roots.
pub fn static_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let hidden = dim(args.text_config.hidden_size)?;
    let vocabulary = dim(args.text_config.vocab_size)?;
    let mut groups = vec![
        group(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            [(
                "model.embed_tokens.weight",
                vec![vocabulary, hidden],
                balanced(0),
            )],
        )?,
        replicated(
            "model.text_norms",
            [
                ("model.embed_norm.weight", vec![hidden]),
                ("model.norm.weight", vec![hidden]),
            ],
        )?,
        group(
            "lm_head",
            ParameterRole::Vocabulary,
            [("lm_head.weight", vec![vocabulary, hidden], balanced(0))],
        )?,
    ];
    if let Some(audio) = &args.audio_config {
        groups.push(replicated(
            "audio.codebook_embedding",
            [(
                "audio.encoder.weight",
                vec![
                    dim(audio.num_codebooks)?
                        .checked_mul(dim(audio.codebook_size)?)
                        .ok_or_else(|| invalid("Inkling audio vocabulary overflow"))?,
                    hidden,
                ],
            )],
        )?);
        groups.push(replicated(
            "audio.final_norm",
            [("audio.final_norm.weight", vec![hidden])],
        )?);
    }
    if let Some(vision) = &args.vision_config {
        for (layer, (input, output, _, _)) in vision.layer_specs().into_iter().enumerate() {
            let output = dim(output)?;
            groups.push(ParameterGroupSpec::partitioned(
                format!("visual.layers.{layer}.channels"),
                ParameterRole::Channels,
                output,
                [member(
                    format!("visual.layers.{layer}.projection.weight"),
                    vec![output, dim(input)?],
                    partitioned(0),
                )],
            )?);
            if layer + 1 != vision.layer_specs().len() {
                groups.push(replicated(
                    format!("visual.layers.{layer}.norm"),
                    [(
                        format!("visual.layers.{layer}.layer_norm.weight"),
                        vec![output],
                    )],
                )?);
            }
        }
        groups.push(replicated(
            "visual.final_norm",
            [("visual.final_norm.weight", vec![hidden])],
        )?);
    }
    Ok(groups)
}

/// Declares checkpoint-embedded prediction modules as replicated TP state.
///
/// Predictor depths consume the complete target activation produced after
/// each decoder all-reduce. Keeping their comparatively small blocks
/// replicated preserves the released equations while the shared vocabulary
/// ingress and egress remain sharded by [`static_parameter_groups`].
pub fn mtp_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    if args
        .mtp_config
        .as_ref()
        .is_none_or(|mtp| mtp.num_nextn_predict_layers <= 0)
    {
        return Ok(Vec::new());
    }
    let plan = super::safetensors_plan(args).map_err(invalid)?;
    let mut members = BTreeMap::<String, Vec<usize>>::new();
    let mut add = |constraint: &eredu_checkpoint::schema::SafetensorsTensorConstraint| {
        let target = constraint.aliases.last().unwrap_or(&constraint.key);
        if target.starts_with("model.mtp.") {
            members.insert(target.clone(), constraint.shape.clone());
        }
    };
    for constraint in &plan.common_tensors {
        add(constraint);
    }
    for group in &plan.layout_groups {
        if !group.variants.iter().any(|variant| {
            variant.tensors.iter().any(|constraint| {
                constraint.key.starts_with("model.mtp.")
                    || constraint
                        .aliases
                        .iter()
                        .any(|alias| alias.starts_with("model.mtp."))
            })
        }) {
            continue;
        }
        let canonical = group
            .variants
            .iter()
            .find(|variant| variant.id == "canonical split")
            .ok_or_else(|| {
                invalid(format!(
                    "Inkling MTP layout group {:?} has no canonical split",
                    group.id
                ))
            })?;
        for constraint in &canonical.tensors {
            add(constraint);
        }
    }
    if members.is_empty() {
        return Err(invalid(
            "Inkling checkpoint declares MTP depths but no MTP parameters",
        ));
    }
    Ok(vec![replicated("model.mtp", members)?])
}

/// Declares one decoder layer's head, convolution, dense/expert, and replicated
/// parameter groups.
pub fn layer_parameter_groups(
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let text = &args.text_config;
    let policy = text
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Inkling layer {layer} is out of range")))?;
    let local = policy.attention.window().is_some();
    let hidden = dim(text.hidden_size)?;
    let query_heads = dim(text.query_heads(local))?;
    let key_value_heads = dim(text.key_value_heads(local))?;
    let head = dim(text.attention_head_dim(local))?;
    let query_width = query_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Inkling query width overflow"))?;
    let key_value_width = key_value_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Inkling key/value width overflow"))?;
    let relative_width = query_heads
        .checked_mul(dim(text.d_rel)?)
        .ok_or_else(|| invalid("Inkling relative-query width overflow"))?;
    let root = format!("model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let mut query_members = vec![
        member(
            format!("{attention}.q_proj.weight"),
            vec![query_width, hidden],
            partitioned(0),
        ),
        member(
            format!("{attention}.r_proj.weight"),
            vec![relative_width, hidden],
            partitioned(0),
        ),
        member(
            format!("{attention}.o_proj.weight"),
            vec![hidden, query_width],
            partitioned(1),
        ),
    ];
    if text.q_bias {
        query_members.push(member(
            format!("{attention}.q_proj.bias"),
            vec![query_width],
            partitioned(0),
        ));
    }
    if text.o_bias {
        query_members.push(member(
            format!("{attention}.o_proj.bias"),
            vec![hidden],
            MemberSharding::Replicated,
        ));
    }
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{attention}.query_heads"),
        ParameterRole::AttentionHeads,
        query_heads,
        query_members,
    )?];
    groups.push(ParameterGroupSpec::partitioned(
        format!("{attention}.key_value_heads"),
        ParameterRole::AttentionHeads,
        key_value_heads,
        [
            member(
                format!("{attention}.k_proj.weight"),
                vec![key_value_width, hidden],
                partitioned(0),
            ),
            member(
                format!("{attention}.v_proj.weight"),
                vec![key_value_width, hidden],
                partitioned(0),
            ),
            member(
                format!("{attention}.k_sconv.weight"),
                vec![key_value_width, 1, dim(text.sconv_kernel_size)?],
                partitioned(0),
            ),
            member(
                format!("{attention}.v_sconv.weight"),
                vec![key_value_width, 1, dim(text.sconv_kernel_size)?],
                partitioned(0),
            ),
        ],
    )?);
    match policy.feed_forward {
        FeedForwardPolicy::Dense => {
            let intermediate = dim(text.dense_intermediate_size())?;
            groups.push(ParameterGroupSpec::partitioned(
                format!("{root}.dense.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                intermediate,
                [
                    member(
                        format!("{root}.dense.gate_proj.weight"),
                        vec![intermediate, hidden],
                        partitioned(0),
                    ),
                    member(
                        format!("{root}.dense.up_proj.weight"),
                        vec![intermediate, hidden],
                        partitioned(0),
                    ),
                    member(
                        format!("{root}.dense.down_proj.weight"),
                        vec![hidden, intermediate],
                        partitioned(1),
                    ),
                ],
            )?);
        }
        FeedForwardPolicy::SparseMoe => {
            let intermediate = dim(text.moe_intermediate_size())?;
            let routed = dim(text.n_routed_experts)?;
            let shared = dim(text.n_shared_experts)?;
            for (name, count) in [("experts", routed), ("shared_experts", shared)] {
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{root}.moe.{name}.intermediate"),
                    ParameterRole::ExpertIntermediate,
                    intermediate,
                    [
                        member(
                            format!("{root}.moe.{name}.gate_up_proj"),
                            vec![count, 2 * intermediate, hidden],
                            MemberSharding::PartitionedSegments {
                                axis: 1,
                                segments: vec![0..intermediate, intermediate..2 * intermediate],
                            },
                        ),
                        member(
                            format!("{root}.moe.{name}.down_proj"),
                            vec![count, hidden, intermediate],
                            partitioned(2),
                        ),
                    ],
                )?);
            }
            groups.push(replicated(
                format!("{root}.moe.router"),
                [
                    (
                        format!("{root}.moe.router.weight"),
                        vec![routed + shared, hidden],
                    ),
                    (format!("{root}.moe.router.bias"), vec![routed]),
                    (format!("{root}.moe.router.global_scale"), vec![1]),
                ],
            )?);
        }
    }
    let relative_extent = policy
        .attention
        .window()
        .map(|window| window.get() as usize)
        .unwrap_or(dim(text.rel_extent)?);
    let mut replicated_members = vec![
        (format!("{attention}.q_norm.weight"), vec![head]),
        (format!("{attention}.k_norm.weight"), vec![head]),
        (
            format!("{attention}.rel_proj"),
            vec![dim(text.d_rel)?, relative_extent],
        ),
        (format!("{root}.input_layernorm.weight"), vec![hidden]),
        (
            format!("{root}.post_attention_layernorm.weight"),
            vec![hidden],
        ),
        (
            format!("{root}.attn_sconv.weight"),
            vec![hidden, 1, dim(text.sconv_kernel_size)?],
        ),
        (
            format!("{root}.mlp_sconv.weight"),
            vec![hidden, 1, dim(text.sconv_kernel_size)?],
        ),
    ];
    if policy.feed_forward == FeedForwardPolicy::Dense {
        replicated_members.push((format!("{root}.dense_global_scale"), vec![1]));
    }
    groups.push(replicated(
        format!("{root}.replicated"),
        replicated_members,
    )?);
    Ok(groups)
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

const fn balanced(axis: usize) -> MemberSharding {
    MemberSharding::Balanced { axis }
}

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("invalid Inkling dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

#[cfg(test)]
mod tests {
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, ParameterRole, TensorPlacement};

    use super::*;

    fn args() -> ModelArgs {
        ModelArgs::from_hf_json(
            br#"{
              "model_type":"inkling_mm_model",
              "text_config":{
                "hidden_size":16,"num_hidden_layers":2,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,"local_layer_ids":[1],
                "mlp_layer_types":["dense","moe"],"sconv_kernel_size":4,
                "d_rel":2,"intermediate_size":12,"n_routed_experts":4,
                "num_experts_per_tok":2,"n_shared_experts":1
              }
            }"#,
        )
        .unwrap()
    }

    fn mtp_args() -> ModelArgs {
        ModelArgs::from_hf_json(
            br#"{
              "model_type":"inkling_mm_model",
              "text_config":{
                "hidden_size":16,"num_hidden_layers":1,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,"local_layer_ids":[0],
                "mlp_layer_types":["dense"],"sconv_kernel_size":4,
                "d_rel":2,"intermediate_size":12,"n_routed_experts":4,
                "num_experts_per_tok":2,"n_shared_experts":1
              },
              "mtp_config":{
                "num_nextn_predict_layers":2,"local_layer_ids":[1],
                "chain_hidden_post_norm":true,"dense_intermediate_size":12
              }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn sparse_groups_distinguish_routed_shared_and_replicated_router() {
        let groups = layer_parameter_groups(&args(), 1).unwrap();
        assert_eq!(
            groups
                .iter()
                .filter(|group| group.role() == ParameterRole::ExpertIntermediate)
                .count(),
            2
        );
        assert!(groups
            .iter()
            .any(|group| group.logical_name().ends_with("moe.router")));
        assert!(groups
            .iter()
            .any(|group| group.logical_name().ends_with("key_value_heads")));
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

    #[test]
    fn local_text_geometry_uses_planned_heads_and_intermediate_widths() {
        let args = args();
        let mut dense = LocalModelLayout::default();
        dense.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            local_tensor(vec![8, 16]),
        );
        dense.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            local_tensor(vec![4, 16]),
        );
        dense.insert(
            "model.layers.0.dense.gate_proj.weight".into(),
            local_tensor(vec![6, 16]),
        );
        let local = local_text_args(&args.text_config, 0, &dense).unwrap();
        assert_eq!(local.num_attention_heads, 2);
        assert_eq!(local.num_key_value_heads, 1);
        assert_eq!(local.dense_intermediate_size(), 6);

        let mut sparse = LocalModelLayout::default();
        sparse.insert(
            "model.layers.1.self_attn.q_proj.weight".into(),
            local_tensor(vec![8, 16]),
        );
        sparse.insert(
            "model.layers.1.self_attn.k_proj.weight".into(),
            local_tensor(vec![4, 16]),
        );
        sparse.insert(
            "model.layers.1.moe.experts.gate_up_proj".into(),
            local_tensor(vec![4, 6, 16]),
        );
        let local = local_text_args(&args.text_config, 1, &sparse).unwrap();
        assert_eq!(local.swa_num_attention_heads, Some(2));
        assert_eq!(local.swa_num_key_value_heads, Some(1));
        assert_eq!(local.moe_intermediate_size(), 3);
    }

    #[test]
    fn embedded_predictor_parameters_are_complete_and_replicated() {
        let groups = mtp_parameter_groups(&mtp_args()).unwrap();
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.logical_name(), "model.mtp");
        assert_eq!(group.role(), ParameterRole::Replicated);
        assert!(group.members().iter().all(|member| {
            member.target().starts_with("model.mtp.")
                && member.sharding() == &MemberSharding::Replicated
        }));
        assert!(group.members().iter().any(|member| {
            member.target() == "model.mtp.layers.0.transformer_block.dense.gate_proj.weight"
        }));
        assert!(group
            .members()
            .iter()
            .any(|member| member.target() == "model.mtp.chain_norm.weight"));
        assert!(!group
            .members()
            .iter()
            .any(|member| member.target().contains("w13_dn")));
    }
}
