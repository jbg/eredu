//! Backend-neutral layered GPT-OSS model assembly.

use eredu_nn::{Error, GroupedNeuralBackend, Tensor};

use super::{block::GptOssBlockFactory, config::ModelArgs};

/// Shared layered lifecycle specialized to GPT-OSS blocks.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs, GptOssBlockFactory>;

/// Pipeline-local GPT-OSS model with global routing and local expert banks.
pub type PartitionedLayeredModel<B> =
    crate::decoder::PartitionedLayeredModel<B, ModelArgs, GptOssBlockFactory>;

impl crate::decoder::PartitionedConfig for ModelArgs {
    fn set_local_geometry(
        &mut self,
        query_heads: i32,
        key_value_heads: i32,
        intermediate: i32,
    ) -> Result<(), Error> {
        if query_heads <= 0 || key_value_heads <= 0 || intermediate <= 0 {
            return Err(Error::backend(
                "local GPT-OSS geometry must remain positive",
            ));
        }
        self.num_attention_heads = query_heads;
        self.num_key_value_heads = key_value_heads;
        self.intermediate_size = intermediate;
        Ok(())
    }

    fn local_block_config(
        &self,
        layer: usize,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Self, Error> {
        super::parallel::local_block_args(self, layer, layout).map_err(Error::backend)
    }

    fn validate_partition_parameters(
        &self,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
    ) -> Result<(), Error> {
        crate::decoder::validate_partitioned_decoder_description(self, parameters)?;
        let experts = usize::try_from(self.num_local_experts).map_err(Error::backend)?;
        let layers = usize::try_from(self.num_hidden_layers).map_err(Error::backend)?;
        for layer in 0..layers {
            let prefix = format!("{}.layers.{layer}", self.parameter_root);
            for required in [
                format!("{prefix}.self_attn.sinks"),
                format!("{prefix}.mlp.router"),
                format!("{prefix}.mlp.experts.intermediate"),
            ] {
                if !parameters
                    .groups()
                    .iter()
                    .any(|owned| owned.group().logical_name() == required)
                {
                    return Err(Error::backend(format!(
                        "GPT-OSS parameter description omits {required}"
                    )));
                }
            }
            let expert_group = parameters
                .groups()
                .iter()
                .find(|owned| {
                    owned.group().logical_name() == format!("{prefix}.mlp.experts.intermediate")
                })
                .expect("required expert group was checked above");
            if expert_group
                .group()
                .members()
                .iter()
                .any(|member| member.global_shape().first().copied() != Some(experts))
            {
                return Err(Error::backend(format!(
                    "GPT-OSS unit {layer} expert axis differs from {experts}"
                )));
            }
        }
        Ok(())
    }
}

/// Builds one layered GPT-OSS model with pinned static modules.
pub fn new_layered_model<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    args: ModelArgs,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<LayeredModel<B>, Error> {
    LayeredModel::new(args, context)
}
