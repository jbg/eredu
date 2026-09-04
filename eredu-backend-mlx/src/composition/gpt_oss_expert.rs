// MLX storage binding for neutral GPT-OSS routed expert banks.

use eredu_architectures::gpt_oss::ModelArgs;

use crate::backend::{
    error::Error,
    runtime::residency::parameter_bank::{AddressableParameterBank, ParameterBankEntry},
};
use crate::composition::grouped_provider::CachedGatedProductGroupProvider;

/// Lowers the architecture-owned expert schedule into MLX cache entries.
pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog = eredu_architectures::gpt_oss::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, layout)
}

/// Adapts an expert cache to the neutral routed-provider contract.
pub const fn cached_provider<'a>(
    cache: &'a AddressableParameterBank,
    _args: &ModelArgs,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> ModelArgs {
        eredu_architectures::gpt_oss::model_args_from_config_value(&serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "rms_norm_eps": 1e-5,
            "sliding_window": 128,
            "max_position_embeddings": 4096,
            "rope_theta": 150000.0,
            "quantization_config": { "quant_method": "mxfp4" },
            "swiglu_limit": 7.0
        }))
        .unwrap()
    }

    #[test]
    fn cached_bank_freezes_native_format_biases_and_exact_policy() {
        let args = args();
        let spec = eredu_architectures::gpt_oss::moe::expert_bank_spec(&args, 0).unwrap();
        let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
            panic!("GPT-OSS experts must use packed architecture geometry");
        };
        assert_eq!(
            gate_up.format().encoding().weight_quantization(),
            Some(eredu_checkpoint::WeightQuantization::MxFp4)
        );
        assert_eq!(
            down.format().encoding().weight_quantization(),
            Some(eredu_checkpoint::WeightQuantization::MxFp4)
        );
        assert!(gate_up.bias().is_some());
        assert!(down.bias().is_some());
        assert_eq!(spec.policy(), args.gated_product_policy);
        assert_eq!(spec.policy().sigmoid_multiplier(), 1.702);
        assert_eq!(spec.policy().up_offset(), 1.0);
        assert_eq!(spec.policy().gate_upper_bound(), Some(7.0));
        assert_eq!(spec.policy().up_absolute_bound(), Some(7.0));
    }
}
