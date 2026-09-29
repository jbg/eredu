//! Shared physical expert recipes; families supply exact namespace aliases.
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    store::TensorSelection,
};
use std::collections::BTreeMap;

/// Selects packed, split-bank or independent expert parameters, preserving
/// physical source identities and bounded selections for every companion.
pub fn gated_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    root: &str,
    experts: std::ops::Range<usize>,
    selection: TensorSelection,
    aliases: impl Fn(&str) -> Vec<String>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let source = |name: &str| {
        std::iter::once(name.to_owned())
            .chain(aliases(name))
            .find(|name| catalog.tensor_metadata(name).is_ok())
    };
    let packed = source(&format!("{root}.gate_up_proj")).is_some();
    let split_banks = ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .all(|name| source(&format!("{root}.{name}")).is_some());
    let mut recipes = BTreeMap::new();
    if packed {
        for (target_name, required) in [
            ("gate_up_proj", true),
            ("gate_up_proj_scale_inv", false),
            ("gate_up_proj_scales", false),
            ("gate_up_proj_biases", false),
            ("down_proj", true),
            ("down_proj_scale_inv", false),
            ("down_proj_scales", false),
            ("down_proj_biases", false),
        ] {
            let name = format!("{root}.{target_name}");
            let Some(source) = source(&name) else {
                if required {
                    return Err(format!("missing packed gated expert tensor {name}"));
                }
                continue;
            };
            recipes.insert(
                target_name.replace("_scale_inv", "_scales"),
                DerivedWeightRecipe::source(source, selection.clone()),
            );
        }
    } else if split_banks {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: ["gate_proj", "up_proj"]
                    .into_iter()
                    .map(|name| {
                        DerivedWeightRecipe::source(
                            source(&format!("{root}.{name}")).expect("split bank exists"),
                            selection.clone(),
                        )
                    })
                    .collect(),
            },
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(
                source(&format!("{root}.down_proj")).expect("split bank exists"),
                selection.clone(),
            ),
        );
        for suffix in ["scales", "biases", "scale_inv"] {
            let gate = source(&format!("{root}.gate_proj_{suffix}"));
            let up = source(&format!("{root}.up_proj_{suffix}"));
            let target_suffix = if suffix == "scale_inv" {
                "scales"
            } else {
                suffix
            };
            if let (Some(gate), Some(up)) = (gate, up) {
                recipes.insert(
                    format!("gate_up_proj_{target_suffix}"),
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate, selection.clone()),
                            DerivedWeightRecipe::source(up, selection.clone()),
                        ],
                    },
                );
            }
            if let Some(down) = source(&format!("{root}.down_proj_{suffix}")) {
                recipes.insert(
                    format!("down_proj_{target_suffix}"),
                    DerivedWeightRecipe::source(down, selection.clone()),
                );
            }
        }
    } else {
        for (suffix, target_suffix) in [("weight", ""), ("weight_scale_inv", "_scales")] {
            let mut gate_up_inputs = Vec::new();
            let mut down_inputs = Vec::new();
            for expert in experts.clone() {
                let projection = |names: &[&str]| {
                    names.iter().find_map(|name| {
                        source(&format!("{root}.{expert}.{name}.{suffix}"))
                            .map(|name| DerivedWeightRecipe::source(name, TensorSelection::Full))
                    })
                };
                match (
                    projection(&["gate_proj", "w1"]),
                    projection(&["up_proj", "w3"]),
                    projection(&["down_proj", "w2"]),
                ) {
                    (Some(gate), Some(up), Some(down)) => {
                        gate_up_inputs.push(DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![gate, up],
                        });
                        down_inputs.push(down);
                    }
                    (None, None, None) if suffix != "weight" => {}
                    _ => {
                        return Err(format!(
                            "missing split gated expert {expert} {suffix} tensor under {root}"
                        ));
                    }
                }
            }
            if gate_up_inputs.is_empty() {
                continue;
            }
            if gate_up_inputs.len() != experts.len() {
                return Err(format!(
                    "incomplete gated expert {suffix} bank under {root}"
                ));
            }
            recipes.insert(
                format!("gate_up_proj{target_suffix}"),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: gate_up_inputs,
                },
            );
            recipes.insert(
                format!("down_proj{target_suffix}"),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: down_inputs,
                },
            );
        }
    }
    Ok(recipes)
}
