//! Retained physical names and bounded ownership for target, prediction and vision.
use super::schema::{aliases, canonical_name};
use crate::qwen4_exp::config::Config;
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    store::TensorSelection,
};
use std::collections::BTreeMap;

/// Static parameters and layer-sized execution units have separate source views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterScope {
    /// Embedding/readout, prediction fusion and vision ingress/output parameters.
    Static,
    /// Target mixer, residual modules and shared branch, excluding lexical injection.
    Target(usize),
    /// Independently owned lexical projections, norms and convolution.
    Lexical(usize),
    /// One embedded prediction decoder block; its state is independently owned.
    Prediction(usize),
    /// One shared vision transformer block.
    Vision(usize),
}
fn root(config: &Config, scope: ParameterScope) -> Result<Option<String>, String> {
    match scope {
        ParameterScope::Static => Ok(None),
        ParameterScope::Target(layer) if layer < config.layers.len() => {
            Ok(Some(format!("model.layers.{layer}")))
        }
        ParameterScope::Lexical(layer) if config.ngram.layers.contains(&layer) => {
            Ok(Some(format!("model.layers.{layer}.ple")))
        }
        ParameterScope::Prediction(layer)
            if config
                .prediction
                .as_ref()
                .is_some_and(|p| layer < p.layers.len()) =>
        {
            Ok(Some(format!("mtp.layers.{layer}")))
        }
        ParameterScope::Vision(layer)
            if config
                .vision
                .as_ref()
                .is_some_and(|v| layer < v.layer_count()) =>
        {
            Ok(Some(format!("model.visual.blocks.{layer}")))
        }
        _ => Err("parameter scope is outside the declared family schedule".into()),
    }
}
/// Exact ordinary parameter recipes for one declared owner. Table payloads and
/// integer controls are retained by the row source; expert payloads by their bank.
/// No table concatenation or full expert-bank read enters a layer residency unit.
pub fn parameter_recipes(
    source_keys: impl IntoIterator<Item = String>,
    config: &Config,
    scope: ParameterScope,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let root = root(config, scope)?;
    let mut recipes = BTreeMap::new();
    for source_name in source_keys {
        let name = canonical_name(&source_name);
        let owned = match &root {
            Some(root) => name
                .strip_prefix(root)
                .is_some_and(|suffix| suffix.starts_with('.')),
            None => {
                !name.starts_with("model.layers.")
                    && !name.starts_with("mtp.layers.")
                    && !name.starts_with("model.visual.blocks.")
            }
        };
        if !owned
            || (matches!(scope, ParameterScope::Target(_)) && name.contains(".ple."))
            || name.contains(".mlp.experts.")
            || name.contains(".ple.ple_embedding.")
            || name.ends_with("rotary_emb.inv_freq")
        {
            continue;
        }
        if recipes
            .insert(
                name.clone(),
                DerivedWeightRecipe::source(source_name, TensorSelection::Full),
            )
            .is_some()
        {
            return Err(format!("ambiguous parameter namespace for {name}"));
        }
    }
    Ok(recipes)
}
/// Shared packed/split-bank recipes restricted to one target or prediction expert.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &Config,
    scope: ParameterScope,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if !matches!(
        scope,
        ParameterScope::Target(_) | ParameterScope::Prediction(_)
    ) || expert >= config.experts.count as usize
    {
        return Err("expert is outside the declared routed unit".into());
    }
    let root = format!(
        "{}.mlp.experts",
        root(config, scope)?.ok_or("expert requires a layer owner")?
    );
    crate::shared_routed::checkpoint::gated_expert_recipes(
        catalog,
        &root,
        expert..expert + 1,
        TensorSelection::Range {
            axis: 0,
            start: expert,
            end: expert + 1,
        },
        aliases,
    )
}
