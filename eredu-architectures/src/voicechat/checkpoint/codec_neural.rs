//! Published EAR neural encodings and decoder aliases over codec-owned declarations.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::codec::CodecDeclarations;
use eredu_codec::ear_specs::{ConvolutionSpec, DecoderParameters};
use eredu_gguf::GgmlType;
use eredu_nn::{DeclareLogicalParameters, ParameterSpec};
use std::collections::BTreeMap;

pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declarations: &CodecDeclarations,
) -> Result<(), String> {
    let e = declarations.encoder.spec();
    let d = declarations.decoder.spec();
    for (component, spec) in [
        (Component::CodecEncoder, e as &dyn DeclareLogicalParameters),
        (Component::CodecDecoder, d),
    ] {
        for p in spec.logical_parameters().map_err(|e| e.to_string())? {
            // All published complete-container neural codec slots retain floating constraints.
            out.push(ReleasedTensor {
                component,
                key: p.parameter.id.as_str().into(),
                shape: p.shape,
                affine_source_format: p.linear_format.cloned(),
                integer: false,
                quantized: false,
            });
        }
    }
    Ok(())
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declarations: &CodecDeclarations,
) -> Result<(), String> {
    let d = declarations.decoder.spec();
    let shapes = d
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| (&p.parameter.id, p.shape))
        .collect::<BTreeMap<_, _>>();
    let mut add = |alias: String,
                   slot: &ParameterSpec,
                   encoding: GgmlType,
                   pointwise: bool|
     -> Result<(), String> {
        let mut shape = shapes
            .get(&slot.id)
            .ok_or_else(|| format!("unresolved EAR storage slot {}", slot.id))?
            .clone();
        if pointwise {
            if shape.len() != 3 || shape[2] != 1 {
                return Err("split EAR pointwise weight requires one kernel tap".into());
            }
            shape.pop();
        }
        out.push(SplitTensor {
            source: alias,
            target: slot.id.as_str().into(),
            shape,
            encoding,
            transform: Transform::Identity,
        });
        Ok(())
    };
    let mut block = 0;
    for unit in d.units() {
        match unit.parameters() {
            DecoderParameters::Upsample(c) => add(
                format!("codec.dec.up.{}.weight", unit.unit().layer()),
                c.weight(),
                GgmlType::F16,
                false,
            )?,
            DecoderParameters::Output(c) => add(
                "codec.dec.head.weight".into(),
                c.weight(),
                GgmlType::F16,
                false,
            )?,
            DecoderParameters::Block(b) => {
                let root = format!("codec.dec.blk.{block}");
                for (name, c, pointwise) in [
                    ("dwconv", b.depthwise(), false),
                    ("pw1", b.expand(), true),
                    ("pw2", b.project(), true),
                ] {
                    add(
                        format!("{root}.{name}.weight"),
                        c.weight(),
                        GgmlType::F16,
                        pointwise,
                    )?;
                    add(
                        format!("{root}.{name}.bias"),
                        required_bias(c)?,
                        GgmlType::F32,
                        false,
                    )?;
                }
                let norm = b.normalization();
                add(
                    format!("{root}.norm.weight"),
                    norm.weight
                        .as_ref()
                        .ok_or("split EAR normalization requires a gain")?,
                    GgmlType::F32,
                    false,
                )?;
                add(
                    format!("{root}.norm.bias"),
                    norm.bias
                        .as_ref()
                        .ok_or("split EAR normalization requires a bias")?,
                    GgmlType::F32,
                    false,
                )?;
                block += 1;
            }
        }
    }
    Ok(())
}
fn required_bias(spec: &ConvolutionSpec) -> Result<&ParameterSpec, String> {
    spec.bias()
        .ok_or_else(|| "split EAR block requires its declared convolution bias".into())
}
