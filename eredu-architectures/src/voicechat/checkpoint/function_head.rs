//! Published function-head storage aliases and encoding eligibility over declared identities.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::function_head_layered::FunctionHeadDeclaration;
use eredu_nn::DeclareLogicalParameters;

pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declaration: &FunctionHeadDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
    for parameter in declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        out.push(ReleasedTensor {
            component: Component::FunctionHead,
            key: parameter.parameter.id.as_str().to_owned(),
            shape: parameter.shape,
            affine_source_format: parameter.linear_format.cloned(),
            integer: false,
            // The complete published GGUF variants explicitly encode this matrix.
            quantized: parameter.parameter.id == declaration.projection.weight.id,
        });
    }
    Ok(())
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declaration: &FunctionHeadDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
    for parameter in declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        if parameter.parameter.id != declaration.projection.weight.id {
            return Err("split function-head source does not encode this declared slot".into());
        }
        out.push(SplitTensor {
            source: "function_head.weight".into(),
            target: parameter.parameter.id.as_str().to_owned(),
            shape: parameter.shape,
            encoding: eredu_gguf::GgmlType::Q4_0,
            transform: Transform::Identity,
        });
    }
    Ok(())
}
