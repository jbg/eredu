//! Published speech-backbone storage extensions over ordinary Nemotron-H specs.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::{
    nemotron_h::{declarations::TargetOperatorParameters, Declaration},
    voicechat::backbone::checkpoint_parameter_name,
};
use eredu_gguf::GgmlType;
use eredu_nn::{DeclareLogicalParameters, ParameterSpec};
use std::collections::BTreeMap;
enum Layout {
    Direct,
    VectorColumn,
    DepthwiseMatrix,
    GroupedGain(usize),
}
pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declaration: &Declaration,
) -> Result<(), String> {
    let args = declaration.args();
    let mut add = |spec: &dyn DeclareLogicalParameters,
                   encoded: Vec<&eredu_nn::ParameterId>|
     -> Result<(), String> {
        for p in spec.logical_parameters().map_err(|e| e.to_string())? {
            out.push(ReleasedTensor {
                component: Component::Backbone,
                key: checkpoint_parameter_name(&args, p.parameter.id.as_str())?,
                shape: p.shape,
                affine_source_format: p.linear_format.cloned(),
                integer: false,
                quantized: encoded.contains(&&p.parameter.id),
            });
        }
        Ok(())
    };
    let statics = &declaration.statics;
    let mut encoded = vec![&statics.embedding_spec().weight.id];
    if let Some(head) = statics.head_spec() {
        encoded.push(&head.weight.id);
    }
    add(statics, encoded)?;
    for block in &declaration.blocks {
        add(&block.norm, vec![])?;
        match &block.operator {
            TargetOperatorParameters::Mamba(s) => {
                add(s, vec![&s.in_proj.weight.id, &s.out_proj.weight.id])?;
            }
            TargetOperatorParameters::Attention(spec) => {
                add(
                    spec,
                    spec.projection
                        .projections()
                        .iter()
                        .map(|s| &s.weight.id)
                        .collect(),
                )?;
            }
            TargetOperatorParameters::Dense(spec) => {
                add(
                    spec,
                    spec.network
                        .projections()
                        .iter()
                        .map(|s| &s.weight.id)
                        .collect(),
                )?;
            }
            TargetOperatorParameters::Sparse(_) => {
                return Err("published speech backbone does not declare routed slots".into());
            }
        }
    }
    Ok(())
}
pub(super) fn split(out: &mut Vec<SplitTensor>, declaration: &Declaration) -> Result<(), String> {
    let args = declaration.args();
    let statics = &declaration.statics;
    let mut add = |source: String,
                   spec: &dyn DeclareLogicalParameters,
                   slot: &ParameterSpec,
                   encoding,
                   transform,
                   layout: Layout|
     -> Result<(), String> {
        let mut shape = spec
            .logical_parameters()
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| p.parameter.id == slot.id)
            .ok_or_else(|| format!("unresolved backbone slot {}", slot.id))?
            .shape;
        match layout {
            Layout::Direct => {}
            Layout::VectorColumn => {
                if shape.len() != 1 {
                    return Err("split SSM control requires a vector".into());
                }
                shape.push(1);
            }
            Layout::DepthwiseMatrix => {
                if shape.len() != 3 || shape[1] != 1 {
                    return Err("split SSM convolution requires a depthwise kernel".into());
                }
                shape.remove(1);
            }
            Layout::GroupedGain(groups) => {
                if shape.len() != 1 || groups == 0 || shape[0] % groups != 0 {
                    return Err("split SSM gain does not form complete groups".into());
                }
                shape = vec![groups, shape[0] / groups];
            }
        }
        out.push(SplitTensor {
            source,
            target: checkpoint_parameter_name(&args, slot.id.as_str())?,
            shape,
            encoding,
            transform,
        });
        Ok(())
    };
    add(
        "token_embd.weight".into(),
        statics.embedding_spec(),
        &statics.embedding_spec().weight,
        GgmlType::Q4_0,
        Transform::Identity,
        Layout::Direct,
    )?;
    let norm = statics.normalization_spec();
    let norm_slot = norm.logical_parameters().map_err(|e| e.to_string())?[0].parameter;
    add(
        "output_norm.weight".into(),
        norm,
        norm_slot,
        GgmlType::F32,
        Transform::Identity,
        Layout::Direct,
    )?;
    let head = statics
        .head_spec()
        .ok_or("published speech backbone requires its independent readout")?;
    add(
        "output.weight".into(),
        head,
        &head.weight,
        GgmlType::Q4_0,
        Transform::Identity,
        Layout::Direct,
    )?;
    for (layer, block) in declaration.blocks.iter().enumerate() {
        let norm = &block.norm;
        let norm_slot = norm.logical_parameters().map_err(|e| e.to_string())?[0].parameter;
        add(
            format!("blk.{layer}.attn_norm.weight"),
            norm,
            norm_slot,
            GgmlType::F32,
            Transform::Identity,
            Layout::Direct,
        )?;
        match &block.operator {
            TargetOperatorParameters::Mamba(s) => {
                for (alias, slot, encoding, transform, layout) in [
                    (
                        "ssm_in.weight",
                        &s.in_proj.weight,
                        GgmlType::Q4_0,
                        Transform::Identity,
                        Layout::Direct,
                    ),
                    (
                        "ssm_out.weight",
                        &s.out_proj.weight,
                        GgmlType::Q4_0,
                        Transform::Identity,
                        Layout::Direct,
                    ),
                    (
                        "ssm_conv1d.weight",
                        &s.convolution.weight,
                        GgmlType::F32,
                        Transform::Identity,
                        Layout::DepthwiseMatrix,
                    ),
                    (
                        "ssm_a",
                        &s.a_log,
                        GgmlType::F32,
                        Transform::NegLog,
                        Layout::VectorColumn,
                    ),
                    (
                        "ssm_d",
                        &s.d,
                        GgmlType::F32,
                        Transform::Identity,
                        Layout::VectorColumn,
                    ),
                    (
                        "ssm_dt.bias",
                        &s.dt_bias,
                        GgmlType::F32,
                        Transform::Identity,
                        Layout::Direct,
                    ),
                    (
                        "ssm_norm.weight",
                        &s.norm_weight,
                        GgmlType::F32,
                        Transform::Identity,
                        Layout::GroupedGain(s.geometry.groups() as usize),
                    ),
                ] {
                    add(
                        format!("blk.{layer}.{alias}"),
                        s,
                        slot,
                        encoding,
                        transform,
                        layout,
                    )?;
                }
                if let Some(bias) = &s.convolution.bias {
                    add(
                        format!("blk.{layer}.ssm_conv1d.bias"),
                        s,
                        bias,
                        GgmlType::F32,
                        Transform::Identity,
                        Layout::Direct,
                    )?;
                }
                if s.in_proj.bias.is_some() || s.out_proj.bias.is_some() {
                    return Err("published split SSM projections have no bias slots".into());
                }
            }
            TargetOperatorParameters::Attention(spec) => {
                for (s, alias) in spec.projection.projections().into_iter().zip([
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_output",
                ]) {
                    if s.bias.is_some() {
                        return Err("published split attention has no bias slots".into());
                    }
                    add(
                        format!("blk.{layer}.{alias}.weight"),
                        s,
                        &s.weight,
                        GgmlType::Q4_0,
                        Transform::Identity,
                        Layout::Direct,
                    )?;
                }
            }
            TargetOperatorParameters::Dense(spec) => {
                for (s, alias) in spec
                    .network
                    .projections()
                    .into_iter()
                    .zip(["ffn_up", "ffn_down"])
                {
                    if s.bias.is_some() {
                        return Err("published split dense MLP has no bias slots".into());
                    }
                    add(
                        format!("blk.{layer}.{alias}.weight"),
                        s,
                        &s.weight,
                        GgmlType::Q4_0,
                        Transform::Identity,
                        Layout::Direct,
                    )?;
                }
            }
            TargetOperatorParameters::Sparse(_) => {
                return Err("published split speech backbone does not declare routed slots".into());
            }
        }
    }
    Ok(())
}
/// Architecture dimensions follow Nemotron declarations; tokenizer and container flags
/// remain explicit published format requirements.
pub(super) fn metadata_expectations(
    declaration: &Declaration,
) -> Result<BTreeMap<String, serde_json::Value>, String> {
    let args = declaration.args();
    let s = declaration
        .blocks
        .iter()
        .find_map(|block| match &block.operator {
            TargetOperatorParameters::Mamba(spec) => Some(spec),
            _ => None,
        })
        .ok_or("published backbone metadata requires its declared SSM geometry")?;
    let value = serde_json::json!({
        "general.architecture":"nemotron_h",
        "general.type":"model",
        "nemotron_h.context_length":args.max_position_embeddings,
        "nemotron_h.embedding_length":args.hidden_size,
        "nemotron_h.block_count":args.num_hidden_layers,
        "nemotron_h.vocab_size":args.vocab_size,
        "nemotron_h.feed_forward_length":declaration.blocks.iter().map(|b|match &b.operator{TargetOperatorParameters::Dense(s)=>s.network.up.output,_=>0}).collect::<Vec<_>>(),
        "nemotron_h.attention.head_count":args.num_attention_heads,
        "nemotron_h.attention.head_count_kv":declaration.blocks.iter().map(|b|match &b.operator{TargetOperatorParameters::Attention(s)=>s.projection.geometry.key_value_heads(),_=>0}).collect::<Vec<_>>(),
        "nemotron_h.attention.key_length":args.head_dim,
        "nemotron_h.attention.value_length":args.head_dim,
        "nemotron_h.attention.layer_norm_rms_epsilon":args.norm_eps,
        "nemotron_h.attention.layer_norm_epsilon":args.layer_norm_epsilon,
        "nemotron_h.ssm.conv_kernel":s.convolution.kernel_size,
        "nemotron_h.ssm.inner_size":s.out_proj.input,
        "nemotron_h.ssm.state_size":s.geometry.state_dim(),
        "nemotron_h.ssm.group_count":s.geometry.groups(),
        "nemotron_h.ssm.time_step_rank":s.geometry.heads(),
        "general.file_type":2,
        "tokenizer.ggml.model":"gpt2",
        "tokenizer.ggml.pre":"tekken",
        "tokenizer.ggml.bos_token_id":crate::voicechat::channels::TEXT_BOS_ID,
        "tokenizer.ggml.eos_token_id":crate::voicechat::channels::TEXT_EOS_ID,
        "tokenizer.ggml.padding_token_id":crate::voicechat::channels::TEXT_PAD_ID,
        "tokenizer.ggml.add_bos_token":false,
        "tokenizer.ggml.add_eos_token":false
    });
    serde_json::from_value(value).map_err(|e| e.to_string())
}
