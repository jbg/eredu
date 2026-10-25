//! Backend-neutral layered GPT-OSS model assembly.

use eredu_nn::Error;

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

    fn partition_parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_runtime::ParallelPlanError>
    {
        super::parallel::parameter_description(self)
    }
}
