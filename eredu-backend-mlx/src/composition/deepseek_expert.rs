//! MLX expert-residency binding for the neutral DeepSeek architectures.

use std::{collections::BTreeSet, ops::Range};

use eredu_architectures::deepseek::{self, LayerPolicy, V3Args, V4Args};
use eredu_checkpoint::{expert::GatedProductExpertRecipes, recipe::DerivedWeightRecipe};
use eredu_nn::GatedProductExpertBankSpec;
use eredu_runtime::{ExpertIdentity, OffloadUnit, WeightBinding};

use crate::backend::mlx::{
    error::Error,
    runtime::{
        checkpoint::binding_plan::{BindingPlan, PlannedBinding},
        residency::{
            expert_cache::{ExpertCache, ExpertCatalogEntry},
            expert_provider::CachedGatedProductExpertProvider,
        },
    },
};

pub fn v3_catalog(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 layer count".into()))?;
    let total = target
        + usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| Error::UnsupportedArchitecture("invalid V3 prediction count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 expert count".into()))?;
    for layer in 0..total {
        if layer < target && args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        let recipes = deepseek::v3_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(&mut entries, layer, experts, recipes, None, store)?;
    }
    Ok(entries)
}

/// Physical checkpoint keys owned by the independent V3 expert cache.
///
/// These come from the architecture-declared expert recipes rather than from
/// runtime parameter names, because packed checkpoints may realize one logical
/// projection from multiple source tensors.
pub fn v3_checkpoint_keys(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeSet<String>, Error> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 layer count".into()))?;
    let total = target
        + usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| Error::UnsupportedArchitecture("invalid V3 prediction count".into()))?;
    let mut keys = BTreeSet::new();
    for layer in 0..total {
        if layer < target && args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        let recipes = deepseek::v3_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        keys.extend(
            recipes
                .gate_up
                .source_keys()
                .into_iter()
                .chain(recipes.down.source_keys())
                .map(str::to_owned),
        );
    }
    Ok(keys)
}

pub fn v3_parallel_catalog(
    args: &V3Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    v3_catalog_with_intermediate(args, Some(intermediate), store)
}

fn v3_catalog_with_intermediate(
    args: &V3Args,
    intermediate: Option<Range<usize>>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 layer count".into()))?;
    let total = target
        + usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| Error::UnsupportedArchitecture("invalid V3 prediction count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 expert count".into()))?;
    for layer in 0..total {
        if layer < target && args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        let recipes = deepseek::v3_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(
            &mut entries,
            layer,
            experts,
            recipes,
            intermediate.clone(),
            store,
        )?;
    }
    Ok(entries)
}

pub fn v4_catalog(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 layer count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 expert count".into()))?;
    let mut entries = Vec::new();
    for layer in 0..total {
        let recipes = deepseek::v4_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(&mut entries, layer, experts, recipes, None, store)?;
    }
    Ok(entries)
}

/// Physical checkpoint keys owned by the independent V4 expert cache.
pub fn v4_checkpoint_keys(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeSet<String>, Error> {
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 layer count".into()))?;
    let mut keys = BTreeSet::new();
    for layer in 0..total {
        let recipes = deepseek::v4_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        keys.extend(
            recipes
                .gate_up
                .source_keys()
                .into_iter()
                .chain(recipes.down.source_keys())
                .map(str::to_owned),
        );
    }
    Ok(keys)
}

pub fn v4_parallel_catalog(
    args: &V4Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 layer count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 expert count".into()))?;
    let mut entries = Vec::new();
    for layer in 0..total {
        let recipes = deepseek::v4_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(
            &mut entries,
            layer,
            experts,
            recipes,
            Some(intermediate.clone()),
            store,
        )?;
    }
    Ok(entries)
}

fn append_entries(
    entries: &mut Vec<ExpertCatalogEntry>,
    layer: usize,
    experts: usize,
    bank: GatedProductExpertRecipes,
    intermediate: Option<Range<usize>>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<(), Error> {
    for expert in 0..experts {
        let recipes = deepseek::expert_unit_recipes(store, &bank, expert, intermediate.clone())
            .map_err(Error::UnsupportedArchitecture)?;
        let bindings = vec![
            recipe_binding("gate_up_proj", recipes.gate_up, store)?,
            recipe_binding("down_proj", recipes.down, store)?,
        ];
        let bytes = bindings.iter().try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                Error::UnsupportedArchitecture("DeepSeek expert byte total overflowed".into())
            })
        })?;
        let identity = ExpertIdentity::new(layer, expert);
        entries.push(ExpertCatalogEntry::new(
            identity,
            OffloadUnit::new(identity.unit_id(), bindings)?,
            bytes,
        )?);
    }
    Ok(())
}

fn recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<WeightBinding, Error> {
    let metadata = recipe.infer(store)?;
    let mut bindings = BindingPlan::new(vec![PlannedBinding {
        target_name: name.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    }])
    .and_then(|plan| plan.build_bindings(store))
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(bindings.pop().expect("one expert binding"))
}

pub const fn v3_provider<'a>(
    cache: &'a ExpertCache,
    _args: &V3Args,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

pub const fn v4_provider<'a>(
    cache: &'a ExpertCache,
    _args: &V4Args,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

pub fn v3_spec(args: &V3Args, layer: usize) -> Result<GatedProductExpertBankSpec, Error> {
    let policy = deepseek::v3::moe_policy(args, layer)?;
    Ok(deepseek::moe::expert_bank_spec(&policy)?)
}

pub fn v4_spec(args: &V4Args, layer: usize) -> Result<GatedProductExpertBankSpec, Error> {
    Ok(deepseek::v4::expert_bank_spec(args, layer)?)
}
