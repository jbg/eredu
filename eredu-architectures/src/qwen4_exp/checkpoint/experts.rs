//! Physical expert layouts projected from the canonical neural bank declaration.
use eredu_nn::{GroupedGatedProductSpec, PackedGatedProductParameters, ParameterId};

/// Released source organization, independent of runtime residency and invocation ownership.
#[derive(Clone, Copy)]
pub(super) enum SourceLayout {
    Packed,
    Separate,
    SplitBanks,
}

/// A physical source references one authoritative logical bank slot.
pub(super) struct ExpertSource {
    pub name: String,
    pub logical: ParameterId,
    pub shape: Vec<usize>,
}

pub(super) fn sources(
    experts: &GroupedGatedProductSpec,
    layout: SourceLayout,
) -> Result<Vec<ExpertSource>, String> {
    let source = experts
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    if source.gate_up_bias().is_some() || source.down_bias().is_some() {
        return Err("Qwen4 expert sources do not declare affine biases".into());
    }
    let gate = source.gate_up();
    let down = source.down();
    if matches!(layout, SourceLayout::Packed) {
        return Ok([gate, down]
            .map(|d| ExpertSource {
                name: d.parameter.id.to_string(),
                logical: d.parameter.id.clone(),
                shape: d.shape.clone(),
            })
            .into());
    }
    let root = source_root(&source)?;
    let half = source.unfused_shape();
    let split = [
        ("gate_proj", gate, half.clone()),
        ("up_proj", gate, half),
        ("down_proj", down, down.shape.clone()),
    ];
    let mut result = Vec::new();
    match layout {
        SourceLayout::SplitBanks => {
            for (suffix, logical, shape) in split {
                result.push(ExpertSource {
                    name: format!("{root}.{suffix}"),
                    logical: logical.parameter.id.clone(),
                    shape,
                });
            }
        }
        SourceLayout::Separate => {
            for expert in 0..source.group_count() {
                for (suffix, logical, shape) in &split {
                    result.push(ExpertSource {
                        name: format!("{root}.{expert}.{suffix}.weight"),
                        logical: logical.parameter.id.clone(),
                        shape: shape[1..].to_vec(),
                    });
                }
            }
        }
        SourceLayout::Packed => unreachable!("packed sources return above"),
    }
    Ok(result)
}

/// Released bank namespace resolved from the authoritative fused primary.
pub(in crate::qwen4_exp) fn source_root<'a>(
    source: &'a PackedGatedProductParameters<'_>,
) -> Result<&'a str, String> {
    source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or_else(|| "expert source does not reference the released fused bank identity".into())
}

/// Canonical source extension outputs reference declared primaries and supported companions.
pub(in crate::qwen4_exp) fn binding_target(
    source: &PackedGatedProductParameters<'_>,
    binding: &str,
) -> Result<String, String> {
    let (parameter, suffix) = binding
        .strip_prefix("gate_up_proj")
        .map(|suffix| (source.gate_up(), suffix))
        .or_else(|| {
            binding
                .strip_prefix("down_proj")
                .map(|suffix| (source.down(), suffix))
        })
        .filter(|(_, suffix)| matches!(*suffix, "" | "_scales" | "_biases"))
        .ok_or_else(|| format!("unresolved Qwen4 expert binding {binding}"))?;
    Ok(format!("{}{suffix}", parameter.parameter.id))
}
