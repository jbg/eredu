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

/// Borrowed format-extension view of construction-owned bank parameters.
enum BankParameters<'a> {
    FeedForward(eredu_nn::PackedGatedProductParameters<'a>),
    AttentionValue(&'a eredu_nn::GroupedLinearSpec),
}
impl<'a> BankParameters<'a> {
    fn from_block(
        block: &'a super::specs::BlockSpec,
        bank: ExpertBank,
    ) -> Result<Option<Self>, String> {
        match bank {
            ExpertBank::FeedForward => block
                .routed
                .as_ref()
                .map(|routed| {
                    routed
                        .parameters
                        .experts
                        .packed_parameters()
                        .map(Self::FeedForward)
                        .map_err(|error| error.to_string())
                })
                .transpose(),
            ExpertBank::AttentionValue => Ok(block
                .values
                .as_ref()
                .map(|values| Self::AttentionValue(&values.experts))),
        }
    }
    fn group_count(&self) -> usize {
        match self {
            Self::FeedForward(parameters) => parameters.group_count(),
            Self::AttentionValue(spec) => spec.group_count() as usize,
        }
    }
    fn primary(
        &self,
        binding: &str,
    ) -> Result<(&eredu_nn::ParameterSpec, &eredu_nn::LinearFormatSpec), String> {
        match (self, binding) {
            (Self::FeedForward(parameters), "gate_up_proj" | "down_proj") => {
                let parameter = if binding == "gate_up_proj" {
                    parameters.gate_up()
                } else {
                    parameters.down()
                };
                Ok((
                    parameter.parameter,
                    parameter
                        .linear_format
                        .ok_or("bank projection has no declared encoding")?,
                ))
            }
            (Self::AttentionValue(spec), "weight") => {
                Ok((spec.projection().weight(), spec.projection().format()))
            }
            _ => Err(format!("undeclared expert projection {binding}")),
        }
    }
    fn source_root(&self) -> Result<&str, String> {
        let (binding, suffix) = match self {
            Self::FeedForward(_) => ("gate_up_proj", ".gate_up_proj"),
            Self::AttentionValue(_) => ("weight", ".weight"),
        };
        self.primary(binding)?
            .0
            .id
            .as_str()
            .strip_suffix(suffix)
            .ok_or_else(|| "expert projection has no declared source root".into())
    }
    fn target(&self, binding: &str) -> Result<&eredu_nn::ParameterSpec, String> {
        if let Ok((parameter, _)) = self.primary(binding) {
            return Ok(parameter);
        }
        let (primary, scale) = match self {
            Self::AttentionValue(_) => (
                "weight",
                match binding {
                    "scales" | "weight_scale_inv" => true,
                    "biases" => false,
                    _ => return Err(format!("undeclared value-bank binding {binding}")),
                },
            ),
            Self::FeedForward(_) => {
                if let Some(primary) = binding.strip_suffix("_scales") {
                    (primary, true)
                } else if let Some(primary) = binding.strip_suffix("_biases") {
                    (primary, false)
                } else {
                    return Err(format!("undeclared feed-forward binding {binding}"));
                }
            }
        };
        let format = self.primary(primary)?.1;
        (if scale {
            format.scale()
        } else {
            format.affine_bias()
        })
        .ok_or_else(|| format!("expert binding {binding} has no declared companion"))
    }
}

/// Recipes for one selected expert. Every source is one individual tensor or
/// a bounded slice of the published packed GGUF expert axis.
pub fn expert_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    layer: usize,
    bank: ExpertBank,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    let block = declaration
        .block(layer)
        .map_err(|error| error.to_string())?;
    let parameters = BankParameters::from_block(&block, bank)?
        .ok_or("layer does not declare this expert bank")?;
    declared_expert_unit_recipes(catalog, args, layer, bank, &parameters, expert)
}

fn declared_expert_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    layer: usize,
    bank: ExpertBank,
    parameters: &BankParameters<'_>,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if expert >= parameters.group_count() {
        return Err("expert identity exceeds the selected bank cardinality".into());
    }
    let root = parameters.source_root()?;
    let mut recipes = BTreeMap::new();
    match bank {
        ExpertBank::AttentionValue => {
            recipes.insert(
                "weight".into(),
                individual_or_packed(
                    catalog,
                    format!("{root}.{expert}.weight"),
                    format!("{root}.weight"),
                    format!("blk.{layer}.attn_v_exps.weight"),
                    expert,
                )?,
            );
        }
        ExpertBank::FeedForward => {
            let projection = |field: &str, gguf: &str| {
                individual_or_packed(
                    catalog,
                    format!("{root}.{expert}.{field}.weight"),
                    format!("{root}.{field}.weight"),
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
    let encoding = |binding: &str| {
        parameters
            .primary(binding)
            .map(|(_, format)| format.encoding())
    };
    let fp8 = |binding: &str| -> Result<bool, String> {
        Ok(matches!(
            encoding(binding)?,
            eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
        ))
    };
    match bank {
        ExpertBank::AttentionValue if fp8("weight")? => {
            recipes.insert(
                "weight_scale_inv".into(),
                stack_scale(catalog, args, &format!("{root}.{expert}.weight"))?,
            );
        }
        ExpertBank::FeedForward => {
            for (binding, fields) in [
                ("gate_up_proj", &["gate_proj", "up_proj"][..]),
                ("down_proj", &["down_proj"][..]),
            ] {
                if fp8(binding)? {
                    let inputs = fields
                        .iter()
                        .map(|field| {
                            stack_scale(catalog, args, &format!("{root}.{expert}.{field}.weight"))
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
        let needs = |format: eredu_checkpoint::LinearFormat| {
            matches!(format, eredu_checkpoint::LinearFormat::Affine(_))
                || (companion == "scales" && format == eredu_checkpoint::LinearFormat::MxFp4)
        };
        match bank {
            ExpertBank::AttentionValue if needs(encoding("weight")?) => {
                recipes.insert(
                    companion.into(),
                    individual_or_packed(
                        catalog,
                        format!("{root}.{expert}.{companion}"),
                        format!("{root}.{companion}"),
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
                    if !needs(encoding(binding)?) {
                        continue;
                    }
                    let inputs = fields
                        .iter()
                        .map(|(field, gguf)| {
                            individual_or_packed(
                                catalog,
                                format!("{root}.{expert}.{field}.{companion}"),
                                format!("{root}.{field}.{companion}"),
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

/// Complete resident or rank-local bank recipes assembled from bounded members.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    layer: usize,
    bank: ExpertBank,
    experts: &[usize],
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    let block = declaration
        .block(layer)
        .map_err(|error| error.to_string())?;
    let parameters = BankParameters::from_block(&block, bank)?
        .ok_or("layer does not declare this expert bank")?;
    let unique = experts
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if experts.is_empty()
        || unique.len() != experts.len()
        || experts
            .iter()
            .any(|&expert| expert >= parameters.group_count())
    {
        return Err("expert recipe identities must be nonempty and distinct".into());
    }
    let mut grouped = BTreeMap::<String, Vec<DerivedWeightRecipe>>::new();
    for &expert in experts {
        for (binding, recipe) in
            declared_expert_unit_recipes(catalog, args, layer, bank, &parameters, expert)?
        {
            grouped.entry(binding).or_default().push(recipe);
        }
    }
    grouped
        .into_iter()
        .map(|(binding, inputs)| {
            let recipe = DerivedWeightRecipe::Concatenate { axis: 0, inputs };
            recipe.infer(catalog).map_err(|e| e.to_string())?;
            Ok((parameters.target(&binding)?.id.to_string(), recipe))
        })
        .collect()
}

/// One residency catalog per independent bank. Logical layer ordinals remain
/// unchanged and each catalog owns its own cardinality and parameter targets.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    bank: ExpertBank,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let args = declaration.args();
    let blocks = declaration.blocks().map_err(|e| e.to_string())?;
    let group = ExecutionGroupId::new(crate::decoder::TEXT_DECODER_EXECUTION_GROUP)
        .map_err(|e| e.to_string())?;
    let mut units = Vec::new();
    for (layer, block) in blocks.iter().enumerate() {
        let Some(parameters) = BankParameters::from_block(block, bank)? else {
            continue;
        };
        for expert in 0..parameters.group_count() {
            let recipes =
                declared_expert_unit_recipes(catalog, args, layer, bank, &parameters, expert)?
                    .into_iter()
                    .map(|(binding, recipe)| {
                        let (scales, biases) = if bank == ExpertBank::AttentionValue {
                            ("scales".into(), "biases".into())
                        } else {
                            (format!("{binding}_scales"), format!("{binding}_biases"))
                        };
                        crate::ExpertParameterRecipe::new(
                            &binding,
                            parameters.target(&binding)?.id.to_string(),
                            recipe,
                            if parameters.primary(&binding).is_err()
                                || parameters.primary(&binding)?.1.encoding()
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
                    recipes,
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
    declaration: &super::Declaration,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    let mut recipes = BTreeMap::new();
    for super::SourceParameter { name, shape, .. } in
        super::source_parameters(declaration, super::SourceLayout::Safetensors)?
    {
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
