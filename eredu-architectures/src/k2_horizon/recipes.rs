//! Bounded official expert recipes, retaining separate value and SwiGLU banks.
use super::{ExpertBank, ModelArgs};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    store::TensorSelection,
};
use eredu_runtime::{ExecutionGroupId, ParameterBankKey};
use std::collections::BTreeMap;

fn source<C: RecipeCatalog + ?Sized>(catalog: &C, names: &[String]) -> Result<String, String> {
    let found = names
        .iter()
        .filter(|name| catalog.tensor_metadata(name).is_ok())
        .collect::<Vec<_>>();
    match found.as_slice() {
        [name] => Ok((*name).clone()),
        [] => Err(format!("missing expert source, expected one of {names:?}")),
        _ => Err(format!("ambiguous expert sources {found:?}")),
    }
}

fn individual_or_packed<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    individual: String,
    packed: String,
    gguf: String,
    expert: usize,
) -> Result<DerivedWeightRecipe, String> {
    if catalog.tensor_metadata(&individual).is_ok() {
        if [&packed, &gguf]
            .iter()
            .any(|name| catalog.tensor_metadata(name).is_ok())
        {
            return Err("mixed individual and packed expert layouts".into());
        }
        Ok(DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::source(
                individual,
                TensorSelection::Full,
            )],
        })
    } else {
        let name = source(catalog, &[packed, gguf])?;
        Ok(DerivedWeightRecipe::source(
            name,
            TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            },
        ))
    }
}

/// Recipes for one selected expert. Every source is one individual tensor or
/// a bounded slice of the published packed GGUF expert axis.
pub fn expert_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    layer: usize,
    bank: ExpertBank,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let count = match bank {
        ExpertBank::AttentionValue if args.is_mova_layer(layer) => args.mova_num_experts,
        ExpertBank::FeedForward if args.is_sparse_layer(layer) => args.num_experts,
        _ => return Err("layer does not declare this expert bank".into()),
    };
    if expert >= count as usize {
        return Err("expert identity exceeds the selected bank cardinality".into());
    }
    let root = format!("model.layers.{layer}");
    let mut recipes = BTreeMap::new();
    match bank {
        ExpertBank::AttentionValue => {
            recipes.insert(
                "weight".into(),
                individual_or_packed(
                    catalog,
                    format!("{root}.self_attn.v_experts.{expert}.weight"),
                    format!("{root}.self_attn.v_experts.weight"),
                    format!("blk.{layer}.attn_v_exps.weight"),
                    expert,
                )?,
            );
        }
        ExpertBank::FeedForward => {
            let projection = |field: &str, gguf: &str| {
                individual_or_packed(
                    catalog,
                    format!("{root}.mlp.experts.{expert}.{field}.weight"),
                    format!("{root}.mlp.experts.{field}.weight"),
                    format!("blk.{layer}.{gguf}.weight"),
                    expert,
                )
            };
            recipes.insert(
                "gate_up_proj".into(),
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        projection("gate_proj", "ffn_gate_exps")?,
                        projection("up_proj", "ffn_up_exps")?,
                    ],
                },
            );
            recipes.insert(
                "down_proj".into(),
                projection("down_proj", "ffn_down_exps")?,
            );
        }
    }
    let fp8 = |name: &str| {
        matches!(
            args.linear_format_for(name),
            eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
        )
    };
    match bank {
        ExpertBank::AttentionValue if fp8(&format!("{root}.self_attn.v_experts.weight")) => {
            recipes.insert(
                "weight_scale_inv".into(),
                stack_scale(
                    catalog,
                    args,
                    &format!("{root}.self_attn.v_experts.{expert}.weight"),
                )?,
            );
        }
        ExpertBank::FeedForward => {
            for (binding, fields) in [
                ("gate_up_proj", &["gate_proj", "up_proj"][..]),
                ("down_proj", &["down_proj"][..]),
            ] {
                if fp8(&format!("{root}.mlp.experts.{binding}")) {
                    let inputs = fields
                        .iter()
                        .map(|field| {
                            stack_scale(
                                catalog,
                                args,
                                &format!("{root}.mlp.experts.{expert}.{field}.weight"),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    recipes.insert(
                        format!("{binding}_scales"),
                        if inputs.len() == 1 {
                            inputs.into_iter().next().expect("one scale")
                        } else {
                            DerivedWeightRecipe::Concatenate { axis: 1, inputs }
                        },
                    );
                }
            }
        }
        _ => {}
    }
    // Canonical GGUF conversion exposes compact companions alongside each
    // physical expert axis. Slice and concatenate them with the same rows.
    for companion in ["scales", "biases"] {
        let primary = match bank {
            ExpertBank::AttentionValue => format!("{root}.self_attn.v_experts.weight"),
            ExpertBank::FeedForward => format!("{root}.mlp.experts.gate_up_proj"),
        };
        let needs = |format: eredu_checkpoint::LinearFormat| {
            matches!(format, eredu_checkpoint::LinearFormat::Affine(_))
                || (companion == "scales" && format == eredu_checkpoint::LinearFormat::MxFp4)
        };
        match bank {
            ExpertBank::AttentionValue if needs(args.linear_format_for(&primary)) => {
                recipes.insert(
                    companion.into(),
                    individual_or_packed(
                        catalog,
                        format!("{root}.self_attn.v_experts.{expert}.{companion}"),
                        format!("{root}.self_attn.v_experts.{companion}"),
                        format!("blk.{layer}.attn_v_exps.{companion}"),
                        expert,
                    )?,
                );
            }
            ExpertBank::FeedForward => {
                for (binding, fields) in [
                    (
                        "gate_up_proj",
                        &[("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps")][..],
                    ),
                    ("down_proj", &[("down_proj", "ffn_down_exps")][..]),
                ] {
                    if !needs(args.linear_format_for(&format!("{root}.mlp.experts.{binding}"))) {
                        continue;
                    }
                    let inputs = fields
                        .iter()
                        .map(|(field, gguf)| {
                            individual_or_packed(
                                catalog,
                                format!("{root}.mlp.experts.{expert}.{field}.{companion}"),
                                format!("{root}.mlp.experts.{field}.{companion}"),
                                format!("blk.{layer}.{gguf}.{companion}"),
                                expert,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    recipes.insert(
                        format!("{binding}_{companion}"),
                        DerivedWeightRecipe::Concatenate { axis: 1, inputs },
                    );
                }
            }
            _ => {}
        }
    }
    for recipe in recipes.values() {
        recipe.infer(catalog).map_err(|e| e.to_string())?;
    }
    Ok(recipes)
}

fn target(layer: usize, bank: ExpertBank, binding: &str) -> String {
    match bank {
        ExpertBank::AttentionValue => format!("model.layers.{layer}.self_attn.v_experts.{binding}"),
        ExpertBank::FeedForward => format!("model.layers.{layer}.mlp.experts.{binding}"),
    }
}

/// Complete resident or rank-local bank recipes assembled from bounded members.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    layer: usize,
    bank: ExpertBank,
    experts: &[usize],
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let unique = experts
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if experts.is_empty() || unique.len() != experts.len() {
        return Err("expert recipe identities must be nonempty and distinct".into());
    }
    let mut grouped = BTreeMap::<String, Vec<DerivedWeightRecipe>>::new();
    for &expert in experts {
        for (binding, recipe) in expert_unit_recipes(catalog, args, layer, bank, expert)? {
            grouped.entry(binding).or_default().push(recipe);
        }
    }
    grouped
        .into_iter()
        .map(|(binding, inputs)| {
            let recipe = DerivedWeightRecipe::Concatenate { axis: 0, inputs };
            recipe.infer(catalog).map_err(|e| e.to_string())?;
            Ok((target(layer, bank, &binding), recipe))
        })
        .collect()
}

/// One residency catalog per independent bank. Logical layer ordinals remain
/// unchanged and each catalog owns its own cardinality and parameter targets.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    bank: ExpertBank,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let group = ExecutionGroupId::new("text_decoder").map_err(|e| e.to_string())?;
    let mut units = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let count = match bank {
            ExpertBank::AttentionValue if args.is_mova_layer(layer) => args.mova_num_experts,
            ExpertBank::FeedForward if args.is_sparse_layer(layer) => args.num_experts,
            _ => continue,
        };
        for expert in 0..count as usize {
            let parameters = expert_unit_recipes(catalog, args, layer, bank, expert)?
                .into_iter()
                .map(|(binding, recipe)| {
                    let (scales, biases) = if bank == ExpertBank::AttentionValue {
                        ("scales".into(), "biases".into())
                    } else {
                        (format!("{binding}_scales"), format!("{binding}_biases"))
                    };
                    crate::ExpertParameterRecipe::new(
                        &binding,
                        target(layer, bank, &binding),
                        recipe,
                        if binding.ends_with("_scales")
                            || binding.ends_with("_biases")
                            || matches!(binding.as_str(), "scales" | "biases" | "weight_scale_inv")
                            || args.linear_format_for(&target(layer, bank, &binding))
                                != eredu_checkpoint::LinearFormat::Dense
                        {
                            crate::ExpertParameterRole::Preserved
                        } else {
                            crate::ExpertParameterRole::quantizable_projection(scales, biases)
                        },
                    )
                    .map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            units.push(
                crate::ExpertResidencyUnit::new(
                    ParameterBankKey::new(bank.id().value() as usize, layer, expert),
                    group.clone(),
                    layer,
                    format!(
                        "model.layers.{layer}.{}",
                        match bank {
                            ExpertBank::AttentionValue => "self_attn.values",
                            ExpertBank::FeedForward => "mlp",
                        }
                    ),
                    crate::ExpertResidencyDistribution::ExpertParallel,
                    parameters,
                )
                .map_err(|e| e.to_string())?,
            );
        }
    }
    crate::ExpertResidencyCatalog::new(units)
        .and_then(|c| c.with_inferred_byte_geometry(catalog))
        .map_err(|e| e.to_string())
}

fn scale_recipe<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    weight: &str,
) -> Result<DerivedWeightRecipe, String> {
    let canonical = format!("{}.weight_scale_inv", weight.trim_end_matches(".weight"));
    let source_name = args
        .fp8_metadata()
        .ok_or("missing normalized FP8 metadata")?
        .source_scale_name(weight)?;
    let names = if canonical == source_name {
        vec![canonical]
    } else {
        vec![canonical, source_name]
    };
    Ok(DerivedWeightRecipe::source(
        source(catalog, &names)?,
        TensorSelection::Full,
    ))
}
fn stack_scale<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    weight: &str,
) -> Result<DerivedWeightRecipe, String> {
    Ok(DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![scale_recipe(catalog, args, weight)?],
    })
}

/// Resolves source scale spelling for ordinary FP8 projections before binding.
pub fn linear_companion_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = BTreeMap::new();
    for (name, shape) in super::parameter_shapes(args, false)? {
        if shape.len() == 2
            && !name.contains(".mlp.experts.")
            && !name.contains(".self_attn.v_experts.")
            && matches!(
                args.linear_format_for(&name),
                eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
            )
        {
            recipes.insert(
                format!("{}.weight_scale_inv", name.trim_end_matches(".weight")),
                scale_recipe(catalog, args, &name)?,
            );
        }
    }
    Ok(recipes)
}
