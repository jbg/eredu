//! Complete encoded expert units, including matching scale and bias selections.
use eredu_checkpoint::{recipe::RecipeCatalog, store::TensorSelection, LinearFormat};

pub(super) fn unit_parameters<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    bank: &eredu_checkpoint::expert::GatedProductExpertRecipes,
    outputs: &std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    expert: usize,
    intermediate: Option<std::ops::Range<usize>>,
    formats: [LinearFormat; 2],
) -> Result<Vec<crate::ExpertParameterRecipe>, String> {
    use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype};
    let metadata = bank
        .gate_up
        .infer(catalog)
        .map_err(|error| error.to_string())?;
    let width = *metadata
        .shape()
        .get(1)
        .ok_or("expert gate/up has no row axis")?;
    if width % 2 != 0 {
        return Err("expert gate/up has an odd row count".into());
    }
    let width = width / 2;
    if let Some(range) = &intermediate {
        if range.start >= range.end || range.end > width {
            return Err("expert intermediate selection is outside its logical rows".into());
        }
    }
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert.checked_add(1).ok_or("expert selection overflowed")?,
    };
    let mut parameters = Vec::new();
    for ((binding, target), format) in [
        ("gate_up_proj", &bank.target_gate_up),
        ("down_proj", &bank.target_down),
    ]
    .into_iter()
    .zip(formats)
    {
        let metadata = outputs[target]
            .infer(catalog)
            .map_err(|error| error.to_string())?;
        let encoded = matches!(metadata.dtype(), RecipeDtype::F8E4M3 | RecipeDtype::U32);
        let has_scales = outputs.contains_key(&format!("{target}_scales"));
        let has_biases = outputs.contains_key(&format!("{target}_biases"));
        if encoded && (!has_scales || (matches!(format, LinearFormat::Affine(_)) && !has_biases)) {
            return Err(format!(
                "encoded expert {target} is missing required companions"
            ));
        }
        // A load-time conversion may choose an encoded destination for an
        // ordinary source. Recipe coordinates still describe the source.
        let source_format = if matches!(
            metadata.dtype(),
            RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32 | RecipeDtype::F64
        ) {
            LinearFormat::Dense
        } else {
            format
        };
        for suffix in ["", "_scales", "_biases"] {
            let target = format!("{target}{suffix}");
            let Some(recipe) = outputs.get(&target) else {
                continue;
            };
            let mut recipe = recipe
                .select_bounded(catalog, selection.clone())
                .map_err(|error| error.to_string())?;
            if let Some(range) = &intermediate {
                let companion = !suffix.is_empty();
                let block = match source_format {
                    LinearFormat::E4M3BlockFp8(fp8) => fp8.block_columns as usize,
                    LinearFormat::Affine(quant) => quant.group_size as usize,
                    LinearFormat::MxFp4 => 32,
                    LinearFormat::GgufIQuant { ggml_type, .. } => {
                        ggml_type
                            .block_and_bytes()
                            .map_err(|error| error.to_string())?
                            .0 as usize
                    }
                    LinearFormat::Dense => 1,
                };
                if block == 0
                    || !range.start.is_multiple_of(block)
                    || !(range.end.is_multiple_of(block) || range.end == width)
                {
                    return Err(
                        "expert intermediate selection splits an encoded contraction block".into(),
                    );
                }
                let slice = |recipe: &DerivedWeightRecipe, axis, range: std::ops::Range<usize>| {
                    recipe
                        .select_bounded(
                            catalog,
                            TensorSelection::Range {
                                axis,
                                start: range.start,
                                end: range.end,
                            },
                        )
                        .map_err(|error| error.to_string())
                };
                if binding == "gate_up_proj" {
                    let row_block = match source_format {
                        LinearFormat::E4M3BlockFp8(fp8) if companion => fp8.block_rows as usize,
                        _ => 1,
                    };
                    if row_block == 0
                        || !range.start.is_multiple_of(row_block)
                        || !(range.end.is_multiple_of(row_block) || range.end == width)
                    {
                        return Err(
                            "expert intermediate selection splits an encoded row block".into()
                        );
                    }
                    let half = width.div_ceil(row_block);
                    let selected = range.start / row_block..range.end.div_ceil(row_block);
                    recipe = DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            slice(&recipe, 1, selected.clone())?,
                            slice(&recipe, 1, half + selected.start..half + selected.end)?,
                        ],
                    };
                } else {
                    let selected = if companion {
                        range.start / block..range.end.div_ceil(block)
                    } else {
                        match source_format {
                            LinearFormat::Affine(quant) => {
                                let bits = quant.bits as usize;
                                range
                                    .start
                                    .checked_mul(bits)
                                    .ok_or("expert packed selection overflowed")?
                                    / 32
                                    ..range
                                        .end
                                        .checked_mul(bits)
                                        .ok_or("expert packed selection overflowed")?
                                        / 32
                            }
                            LinearFormat::MxFp4 => range.start / 8..range.end / 8,
                            // GGUF recipes expose logical coordinates; the
                            // source store owns their physical block mapping.
                            _ => range.clone(),
                        }
                    };
                    recipe = slice(&recipe, 2, selected)?;
                }
            }
            recipe.infer(catalog).map_err(|error| error.to_string())?;
            let role = if suffix.is_empty() && !has_scales && !has_biases {
                crate::ExpertParameterRole::quantizable_projection(
                    format!("{binding}_scales"),
                    format!("{binding}_biases"),
                )
            } else {
                crate::ExpertParameterRole::Preserved
            };
            parameters.push(
                crate::ExpertParameterRecipe::new(
                    format!("{binding}{suffix}"),
                    target,
                    recipe,
                    role,
                )
                .map_err(|error| error.to_string())?,
            );
        }
    }
    Ok(parameters)
}
