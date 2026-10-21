//! Source-free physical topology for the shared Gemma vision equations.
use super::VisionConfig;
use eredu_runtime::{
    expand_linear_format_parameter_groups, MemberSharding, ParallelPlanError, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole,
};

fn member(
    name: impl Into<String>,
    shape: &[i32],
) -> Result<ParameterMemberSpec, ParallelPlanError> {
    let shape = shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                ParallelPlanError::InvalidTensor("negative vision parameter dimension".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ParameterMemberSpec::new(
        name,
        shape,
        MemberSharding::Replicated,
    ))
}

fn group(
    name: impl Into<String>,
    members: Vec<ParameterMemberSpec>,
    format: impl Fn(&ParameterMemberSpec) -> eredu_checkpoint::LinearFormat,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    expand_linear_format_parameter_groups(
        vec![ParameterGroupSpec::new(
            name,
            ParameterRole::Replicated,
            members,
        )?],
        |member| crate::linear_format::standard_parallel_linear_format(member, format(member)),
    )
}

fn validate(config: &VisionConfig) -> Result<(), ParallelPlanError> {
    config
        .validate()
        .map_err(|e| ParallelPlanError::InvalidTensor(e.to_string()))
}

/// Patch projection, position table and optional pooled standardization.
pub(crate) fn static_groups(
    config: &VisionConfig,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    validate(config)?;
    let input = 3 * config.patch_size * config.patch_size;
    let mut members = vec![
        member(
            format!("{root}.patch_embedder.input_proj.weight"),
            &[config.hidden_size, input],
        )?,
        member(
            format!("{root}.patch_embedder.position_embedding_table"),
            &[2, config.position_embedding_size, config.hidden_size],
        )?,
    ];
    if config.standardize {
        members.push(member(format!("{root}.std_bias"), &[config.hidden_size])?);
        members.push(member(format!("{root}.std_scale"), &[config.hidden_size])?);
    }
    group(format!("{root}.static"), members, |member| {
        config.linear_format_for(member.target(), input)
    })
}

/// One independently leased attention and gated feed-forward block.
pub(crate) fn layer_groups(
    config: &VisionConfig,
    root: &str,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    validate(config)?;
    if index >= config.num_hidden_layers as usize {
        return Err(ParallelPlanError::InvalidTensor(
            "vision layer outside declared stack".into(),
        ));
    }
    let prefix = format!("{root}.encoder.layers.{index}");
    let mut members = Vec::new();
    let query = config.num_attention_heads * config.head_dim;
    let kv = config.num_key_value_heads * config.head_dim;
    for (field, input, output) in [
        ("self_attn.q_proj", config.hidden_size, query),
        ("self_attn.k_proj", config.hidden_size, kv),
        ("self_attn.v_proj", config.hidden_size, kv),
        ("self_attn.o_proj", query, config.hidden_size),
    ] {
        linear_members(
            &mut members,
            config,
            &format!("{prefix}.{field}"),
            input,
            output,
        )?;
    }
    for field in ["q_norm", "k_norm"] {
        members.push(member(
            format!("{prefix}.self_attn.{field}.weight"),
            &[config.head_dim],
        )?);
    }
    for (field, input, output) in [
        ("gate_proj", config.hidden_size, config.intermediate_size),
        ("up_proj", config.hidden_size, config.intermediate_size),
        ("down_proj", config.intermediate_size, config.hidden_size),
    ] {
        linear_members(
            &mut members,
            config,
            &format!("{prefix}.mlp.{field}"),
            input,
            output,
        )?;
    }
    for field in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ] {
        members.push(member(
            format!("{prefix}.{field}.weight"),
            &[config.hidden_size],
        )?);
    }
    group(prefix, members, |member| {
        let input = member.global_shape().last().copied().unwrap_or(0) as i32;
        config.linear_format_for(member.target(), input)
    })
}

fn linear_members(
    members: &mut Vec<ParameterMemberSpec>,
    config: &VisionConfig,
    prefix: &str,
    input: i32,
    output: i32,
) -> Result<(), ParallelPlanError> {
    members.push(member(format!("{prefix}.linear.weight"), &[output, input])?);
    if config.use_clipped_linears {
        for field in ["input_min", "input_max", "output_min", "output_max"] {
            members.push(member(format!("{prefix}.{field}"), &[])?);
        }
    }
    Ok(())
}
