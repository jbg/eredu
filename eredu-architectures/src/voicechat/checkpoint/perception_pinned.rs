//! Released boundary-unit source layouts and encoding rules over typed perception slots.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::perception_specs::PerceptionDeclaration;
use eredu_gguf::GgmlType;
use eredu_nn::{DeclareLogicalParameters, ParameterSpec};
use std::collections::BTreeMap;
pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declaration: &PerceptionDeclaration,
) -> Result<(), String> {
    let spec = &declaration.spec().pinned;
    // Published complete GGUF variants encode the two ordinary projections;
    // OIHW kernels and all biases retain their separate floating constraints.
    let encoded = [
        &spec.subsampling.output.weight.id,
        &spec.projection.weight.id,
    ];
    for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
        out.push(ReleasedTensor {
            component: Component::Perception,
            key: parameter.parameter.id.as_str().to_owned(),
            shape: parameter.shape,
            affine_source_format: parameter.linear_format.cloned(),
            integer: false,
            quantized: encoded.contains(&&parameter.parameter.id),
        });
    }
    Ok(())
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declaration: &PerceptionDeclaration,
) -> Result<(), String> {
    let spec = &declaration.spec().pinned;
    let shapes = spec
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| (&p.parameter.id, p.shape))
        .collect::<BTreeMap<_, _>>();
    let mut add = |source: String,
                   slot: &ParameterSpec,
                   encoding: GgmlType,
                   suffix: &[usize]|
     -> Result<(), String> {
        let mut shape = shapes
            .get(&slot.id)
            .ok_or_else(|| format!("unresolved perception boundary parameter {}", slot.id))?
            .clone();
        shape.extend_from_slice(suffix);
        out.push(SplitTensor {
            source,
            target: slot.id.as_str().to_owned(),
            shape,
            encoding,
            transform: Transform::Identity,
        });
        Ok(())
    };
    for convolution in &spec.subsampling.convolutions {
        let index = convolution.index;
        add(
            format!("a.conv1d.{index}.weight"),
            &convolution.weight,
            GgmlType::F16,
            &[],
        )?;
        // The split container stores bias vectors with two physical singleton axes.
        add(
            format!("a.conv1d.{index}.bias"),
            &convolution.bias,
            GgmlType::F32,
            &[1, 1],
        )?;
    }
    for (alias, projection) in [
        ("a.pre_encode.out", &spec.subsampling.output),
        ("mm.a.proj", &spec.projection),
    ] {
        add(
            format!("{alias}.weight"),
            &projection.weight,
            GgmlType::Q4_0,
            &[],
        )?;
        add(
            format!("{alias}.bias"),
            projection
                .bias
                .as_ref()
                .ok_or("released perception projection requires its bias")?,
            GgmlType::F32,
            &[],
        )?;
    }
    Ok(())
}
