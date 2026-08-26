//! MLX expert-residency binding for the neutral DeepSeek architectures.

use std::{collections::BTreeSet, ops::Range};

use eredu_architectures::deepseek::{self, V3Args, V4Args};

use crate::backend::{
    error::Error,
    runtime::residency::{
        expert_cache::{ExpertCache, ExpertCatalogEntry},
        expert_provider::CachedGatedProductExpertProvider,
    },
};

pub fn v3_catalog(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    v3_catalog_selected(args, store, |_, _| true)
}

/// Canonical V3 expert catalog for selected architecture-owned units.
pub fn v3_catalog_selected(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v3_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    lower_selected_catalog(catalog, store, owns_unit)
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
    let catalog = deepseek::v3_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    Ok(checkpoint_keys(&catalog))
}

/// Canonical tensor-parallel V3 expert catalog for selected owned units.
pub fn v3_parallel_catalog_selected(
    args: &V3Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v3_expert_residency_catalog(store, args, Some(intermediate))
        .map_err(Error::ArchitectureModel)?;
    lower_selected_catalog(catalog, store, owns_unit)
}

pub fn v4_catalog(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    v4_catalog_selected(args, store, |_, _| true)
}

/// Canonical V4 expert catalog for selected architecture-owned units.
pub fn v4_catalog_selected(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v4_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    lower_selected_catalog(catalog, store, owns_unit)
}

/// Physical checkpoint keys owned by the independent V4 expert cache.
pub fn v4_checkpoint_keys(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeSet<String>, Error> {
    let catalog = deepseek::v4_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    Ok(checkpoint_keys(&catalog))
}

/// Canonical tensor-parallel V4 expert catalog for selected owned units.
pub fn v4_parallel_catalog_selected(
    args: &V4Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v4_expert_residency_catalog(store, args, Some(intermediate))
        .map_err(Error::ArchitectureModel)?;
    lower_selected_catalog(catalog, store, owns_unit)
}

fn lower_selected_catalog(
    catalog: eredu_architectures::ExpertResidencyCatalog,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let units = catalog.into_units_selected_by_owner(owns_unit);
    crate::composition::architecture_expert_units(units, store, None)
}

fn checkpoint_keys(catalog: &eredu_architectures::ExpertResidencyCatalog) -> BTreeSet<String> {
    catalog
        .units()
        .iter()
        .flat_map(|unit| unit.parameters())
        .flat_map(|parameter| parameter.recipe().source_keys())
        .map(str::to_owned)
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::store::MemoryWeightStore;
    use safetensors::Dtype;

    #[test]
    fn selection_uses_canonical_owner_instead_of_cache_layer_identity() {
        let args = deepseek::parse_v3_config(&serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "q_lora_rank": 128,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "v_head_dim": 64,
            "first_k_dense_replace": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let tensors = [
            ("model.layers.1.mlp.experts.gate_up_proj", vec![4, 128, 128]),
            ("model.layers.1.mlp.experts.down_proj", vec![4, 128, 64]),
            ("model.layers.2.mlp.experts.gate_up_proj", vec![4, 128, 128]),
            ("model.layers.2.mlp.experts.down_proj", vec![4, 128, 64]),
        ]
        .into_iter()
        .map(|(name, shape)| {
            let bytes = vec![0; shape.iter().product::<usize>() * size_of::<f32>()];
            (name.to_owned(), Dtype::F32, shape, bytes)
        });
        let store = MemoryWeightStore::from_safetensors(tensors).unwrap();

        let selected = v3_parallel_catalog_selected(&args, 16..48, &store, |group, unit| {
            group.as_str() == "mtp.0" && unit == 0
        })
        .unwrap();

        assert_eq!(selected.len(), 4);
        assert!(selected.iter().all(|entry| entry.identity().layer == 2));
    }
}
