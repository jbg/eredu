//! MLX residency adapter for neutral Gemma 4 routed experts.

use std::collections::BTreeSet;

use eredu_architectures::gemma4::ModelArgs;

use crate::backend::{
    error::Error,
    runtime::residency::{
        expert_cache::{ExpertCache, ExpertCatalogEntry},
        expert_provider::CachedGatedProductExpertProvider,
    },
};

/// Returns one independently leasable cache unit for every routed expert.
pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = eredu_architectures::gemma4::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

/// Physical checkpoint keys owned by the independent Gemma 4 expert cache.
///
/// The complete architecture catalog owns sparse-layer selection and physical
/// source discovery. Composition only projects its declared recipes into the
/// source-key set excluded from ordinary layer residency.
pub fn checkpoint_keys(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeSet<String>, Error> {
    let catalog = eredu_architectures::gemma4::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    Ok(catalog
        .units()
        .iter()
        .flat_map(|unit| unit.parameters())
        .flat_map(|parameter| parameter.recipe().source_keys())
        .map(str::to_owned)
        .collect())
}

pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::store::MemoryWeightStore;
    use safetensors::Dtype;

    fn sparse_args() -> ModelArgs {
        ModelArgs::from_hf_json(
            br#"{
                "model_type":"gemma4","hidden_size":16,"num_hidden_layers":1,
                "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":64,
                "max_position_embeddings":128,"layer_types":["full_attention"],
                "enable_moe_block":true,"num_experts":4,"top_k_experts":2,
                "moe_intermediate_size":8
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn checkpoint_keys_come_from_complete_architecture_catalog() {
        let root = "model.language_model.layers.0.experts.switch_glu";
        let tensors = [
            (format!("{root}.gate_proj.weight"), vec![4, 8, 16]),
            (format!("{root}.up_proj.weight"), vec![4, 8, 16]),
            (format!("{root}.down_proj.weight"), vec![4, 16, 8]),
        ];
        let store = MemoryWeightStore::from_safetensors(tensors.iter().map(|(name, shape)| {
            (
                name.clone(),
                Dtype::F32,
                shape.clone(),
                vec![0; shape.iter().product::<usize>() * size_of::<f32>()],
            )
        }))
        .unwrap();

        assert_eq!(
            checkpoint_keys(&sparse_args(), &store).unwrap(),
            tensors
                .into_iter()
                .map(|(name, _)| name)
                .collect::<BTreeSet<_>>()
        );
    }
}
