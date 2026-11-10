//! Exact family parameter encodings derived from neutral checkpoint metadata.
use super::{ConfigError, ModelArgs};
use eredu_checkpoint::LinearFormat;

pub(crate) fn normalize_source_formats(args: &mut ModelArgs) -> Result<(), ConfigError> {
    let source =
        super::Declaration::new(args.clone()).map_err(|e| ConfigError::Invalid(e.to_string()))?;
    let Some(metadata) = args.fp8.clone() else {
        *args = source.args().clone();
        return Ok(());
    };
    for super::SourceParameter {
        name, shape, role, ..
    } in super::source_parameters(&source, super::SourceLayout::Safetensors)
        .map_err(ConfigError::Invalid)?
    {
        // The source format applies FP8 to matrix sources while retaining the
        // vocabulary table in dense storage. Exclusions remain format metadata.
        if shape.len() == 2
            && role != eredu_nn::LogicalParameterRole::Embedding
            && metadata.includes(&name)
        {
            args.formats
                .insert(name, LinearFormat::E4M3BlockFp8(metadata.format()));
        }
    }
    normalize_projected_formats(args, &source, super::SourceLayout::Safetensors)
        .map_err(ConfigError::Invalid)?;
    args.validate()?;
    let selected = source
        .with_formats(args.clone())
        .map_err(|e| ConfigError::Invalid(e.to_string()))?;
    *args = selected.args().clone();
    Ok(())
}

/// Retains each GGUF encoding, then proves the two independently packed banks.
pub(crate) fn normalize_gguf_formats(
    args: &mut ModelArgs,
    checkpoint: &eredu_gguf::Checkpoint,
) -> Result<(), String> {
    let source = super::Declaration::new(args.clone()).map_err(|e| e.to_string())?;
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            let name = super::translate_gguf_weight_name(&tensor.descriptor().name);
            if name.ends_with(".weight") && tensor.descriptor().dimensions.len() >= 2 {
                args.formats.insert(
                    name,
                    crate::linear_format::gguf_tensor_format(tensor, shard.endian())?,
                );
            }
        }
    }
    normalize_projected_formats(args, &source, super::SourceLayout::Gguf)?;
    args.validate().map_err(|e| e.to_string())?;
    let selected = source
        .with_formats(args.clone())
        .map_err(|e| e.to_string())?;
    *args = selected.args().clone();
    Ok(())
}

/// Each physical source contributes its encoding to its declared logical owner.
/// Packed owners require one encoding across all their split or expert sources.
fn normalize_projected_formats(
    args: &mut ModelArgs,
    declaration: &super::Declaration,
    layout: super::SourceLayout,
) -> Result<(), String> {
    let mut owners = std::collections::BTreeMap::new();
    for parameter in super::source_parameters(declaration, layout)? {
        if parameter.name == parameter.logical.as_str() {
            continue;
        }
        let format = args.linear_format_for(&parameter.name);
        if let Some(previous) = owners.insert(parameter.logical.clone(), format) {
            if previous != format {
                return Err(format!(
                    "packed parameter {} requires homogeneous source encodings",
                    parameter.logical
                ));
            }
        }
    }
    args.formats.extend(
        owners
            .into_iter()
            .map(|(owner, format)| (owner.to_string(), format)),
    );
    Ok(())
}
