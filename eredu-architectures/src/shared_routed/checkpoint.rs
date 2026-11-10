//! Shared physical expert recipes; families supply exact namespace aliases.
use eredu_checkpoint::{
    expert::{
        gated_product_materialization_recipes, GatedProductExpertRecipes,
        GatedProductExpertStorageLayout,
    },
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    store::TensorSelection,
};
use std::collections::BTreeMap;

/// Selects packed, split-bank or independent expert primaries. Checkpoint
/// mechanisms project all encoding companions through the same recipe graph.
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
    let (layout, gate_up, down) = if let Some(gate_up) = source(&format!("{root}.gate_up_proj")) {
        let down = source(&format!("{root}.down_proj"))
            .ok_or_else(|| format!("missing packed down projection under {root}"))?;
        (
            GatedProductExpertStorageLayout::Packed,
            DerivedWeightRecipe::source(gate_up, selection.clone()),
            DerivedWeightRecipe::source(down, selection),
        )
    } else if let (Some(gate), Some(up), Some(down)) = (
        source(&format!("{root}.gate_proj")),
        source(&format!("{root}.up_proj")),
        source(&format!("{root}.down_proj")),
    ) {
        (
            GatedProductExpertStorageLayout::SeparatePacked,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, selection.clone()),
                    DerivedWeightRecipe::source(up, selection.clone()),
                ],
            },
            DerivedWeightRecipe::source(down, selection),
        )
    } else {
        let mut gate_up = Vec::new();
        let mut down = Vec::new();
        for expert in experts {
            let projection = |names: &[&str]| {
                names
                    .iter()
                    .find_map(|name| source(&format!("{root}.{expert}.{name}.weight")))
                    .map(|key| DerivedWeightRecipe::source(key, TensorSelection::Full))
            };
            let gate = projection(&["gate_proj", "w1"]).ok_or_else(|| {
                format!("missing gate projection for expert {expert} under {root}")
            })?;
            let up = projection(&["up_proj", "w3"])
                .ok_or_else(|| format!("missing up projection for expert {expert} under {root}"))?;
            let output = projection(&["down_proj", "w2"]).ok_or_else(|| {
                format!("missing down projection for expert {expert} under {root}")
            })?;
            gate_up.push(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![gate, up],
            });
            down.push(output);
        }
        (
            GatedProductExpertStorageLayout::Independent,
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: gate_up,
            },
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: down,
            },
        )
    };
    let recipes = gated_product_materialization_recipes(
        catalog,
        &GatedProductExpertRecipes {
            target_gate_up: "gate_up_proj".into(),
            target_down: "down_proj".into(),
            layout,
            gate_up,
            down,
        },
    )?;
    if layout == GatedProductExpertStorageLayout::Independent {
        for suffix in ["scales", "biases", "global_scale"] {
            if recipes.contains_key(&format!("gate_up_proj_{suffix}"))
                != recipes.contains_key(&format!("down_proj_{suffix}"))
            {
                return Err(format!(
                    "incomplete gated expert {suffix} bank under {root}"
                ));
            }
        }
    }
    Ok(recipes)
}
