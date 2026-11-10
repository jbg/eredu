//! Published codec codebook aliases and scalar storage over codec-owned declarations.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::codec::CodecDeclarations;
use eredu_gguf::GgmlType;
use eredu_nn::DeclareLogicalParameters;
use std::collections::BTreeMap;
pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declarations: &CodecDeclarations,
) -> Result<(), String> {
    let declaration = declarations.quantizer.spec();
    let encoded = declaration
        .codebooks()
        .iter()
        .map(|b| &b.means().id)
        .collect::<Vec<_>>();
    for p in declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        out.push(ReleasedTensor {
            component: Component::CodecQuantizer,
            key: p.parameter.id.as_str().to_owned(),
            shape: p.shape,
            affine_source_format: p.linear_format.cloned(),
            integer: false,
            quantized: encoded.contains(&&p.parameter.id),
        });
    }
    Ok(())
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declarations: &CodecDeclarations,
) -> Result<(), String> {
    let declaration = declarations.quantizer.spec();
    let shapes = declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| (&p.parameter.id, p.shape))
        .collect::<BTreeMap<_, _>>();
    for book in declaration.codebooks() {
        for (alias, slot, encoding, scalar) in [
            ("mus", book.means(), GgmlType::Q4_0, false),
            ("variance", book.variance(), GgmlType::F32, true),
        ] {
            let mut shape = shapes
                .get(&slot.id)
                .ok_or_else(|| format!("unresolved codec codebook slot {}", slot.id))?
                .clone();
            if scalar {
                if !shape.is_empty() {
                    return Err("split codec variance requires a logical scalar".into());
                }
                shape.push(1);
            }
            out.push(SplitTensor {
                source: format!("codec.rvq.{}.{alias}", book.depth()),
                target: slot.id.as_str().into(),
                shape,
                encoding,
                transform: Transform::Identity,
            });
        }
    }
    Ok(())
}
