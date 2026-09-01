// MLX residency adapter for the neutral GPT-OSS routed-expert graph.

use eredu_architectures::gpt_oss::ModelArgs;
use eredu_nn::{GroupedGatedProductOperator, GroupedGatedProductSpec};
use eredu_runtime::{
    ExpertPass, RoutedExpertProvider, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
};
use safemlx::{Array, Stream};
use crate::composition::grouped_provider::*;
use crate::composition::expert_dispatch::{
    dispatch_local_tensor_parallel, dispatch_local_with, dispatch_replicated_tensor_parallel,
    dispatch_replicated_with, DispatchedRoutes, ExpertAssignment, LocalExpertBank,
    RoutingStatistics,
};

use crate::backend::runtime::distributed::Group;

use crate::backend::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    runtime::{
        residency::{
            parameter_bank::{AddressableParameterBank, ParameterBankEntry},
        },
    },
};

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

/// Executes rows already compacted by an EP dispatcher.
pub fn execute_cached_dispatched(
    cache: &AddressableParameterBank,
    spec: &GroupedGatedProductSpec,
    layer: usize,
    hidden: &Array,
    global_group_indices: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_gated_product_dispatched(
        cache,
        spec,
        layer,
        hidden,
        global_group_indices,
        pass,
        stream,
    )
}

/// Executes compact EP rows as a typed TP partial so callers can all-sum the
/// reducible projection and add routed down bias exactly once afterwards.
pub fn execute_cached_dispatched_tensor_parallel(
    cache: &AddressableParameterBank,
    spec: &GroupedGatedProductSpec,
    layer: usize,
    hidden: &Array,
    global_group_indices: &Array,
    pass: ExpertPass,
    partitions: usize,
    stream: &Stream,
) -> Result<eredu_nn::TensorParallelGroupedOutput<Array>, Error> {
    if partitions == 0 {
        return Err(Error::Parallel(
            "GPT-OSS cached expert execution requires a positive TP size".into(),
        ));
    }
    let group_indices = global_group_indices.reshape(&[-1, 1], stream)?;
    let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    execute_cached_gated_product_tensor_parallel(
        cache,
        spec,
        layer,
        hidden,
        &group_indices,
        &weights,
        pass,
        partitions,
        stream,
    )
}

/// Builds the neutral cache-backed provider used by EP and Cartesian TP+EP.
///
/// The provider delegates compaction, route weighting, EP recombination, and
/// statistics to the shared dispatcher. Its tensor-parallel result keeps the
/// reducible projection and routed replicated down bias distinct so the model
/// can all-sum the former and add the latter exactly once.
pub fn distributed_provider<'a>(
    _args: &'a ModelArgs,
    assignment: &'a ExpertAssignment,
    expert_group: Option<&'a Group>,
    cache: &'a AddressableParameterBank,
    statistics: &'a mut RoutingStatistics,
) -> impl RoutedExpertProvider<MlxNeuralBackend, Error = Error> + 'a {
    DistributedCachedProvider {
        assignment,
        expert_group,
        cache,
        statistics,
    }
}

struct DistributedCachedProvider<'a> {
    assignment: &'a ExpertAssignment,
    expert_group: Option<&'a Group>,
    cache: &'a AddressableParameterBank,
    statistics: &'a mut RoutingStatistics,
}

struct CachedLocalBank<'a> {
    spec: &'a GroupedGatedProductSpec,
    layer: usize,
    pass: ExpertPass,
    cache: &'a AddressableParameterBank,
    local_global_group_indices: &'a [usize],
}

impl CachedLocalBank<'_> {
    fn global_ids(&self, local_ids: &Array, stream: &Stream) -> Result<Array, Error> {
        let ids = self
            .local_global_group_indices
            .iter()
            .map(|id| i32::try_from(*id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Error::Parallel("GPT-OSS expert id exceeds i32".into()))?;
        let lookup = Array::from_slice(&ids, &[ids.len() as i32]);
        Ok(lookup.take_axis(local_ids, 0, stream)?)
    }
}

impl LocalExpertBank for CachedLocalBank<'_> {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        execute_cached_dispatched(
            self.cache,
            self.spec,
            self.layer,
            hidden,
            &self.global_ids(local_group_indices, stream)?,
            self.pass,
            stream,
        )
    }

    fn execute_local_routes_tensor_parallel(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<Array>, Error> {
        execute_cached_dispatched_tensor_parallel(
            self.cache,
            self.spec,
            self.layer,
            hidden,
            &self.global_ids(local_group_indices, stream)?,
            self.pass,
            partitions,
            stream,
        )
    }
}

impl RoutedExpertProvider<MlxNeuralBackend> for DistributedCachedProvider<'_> {
    type Error = Error;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, crate::MlxTensor>,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Self::Error> {
        let input = request.input.as_array();
        let group_indices_input = request.routes.group_indices().as_array();
        let coefficients = request.routes.coefficients().as_array();
        let original_shape = input.shape().to_vec();
        let hidden = input.reshape(&[-1, input.dim(-1)], stream)?;
        let group_indices = group_indices_input.reshape(&[-1, group_indices_input.dim(-1)], stream)?;
        let weights = coefficients.reshape(&[-1, coefficients.dim(-1)], stream)?;
        let execute = |routes: &DispatchedRoutes, stream: &Stream| {
            execute_cached_dispatched(
                self.cache,
                resident_bank.spec(),
                request.layer,
                &routes.hidden,
                &routes.global_group_indices,
                request.pass,
                stream,
            )
        };
        let returned = match self.expert_group {
            Some(group) => dispatch_replicated_with(
                &hidden,
                &group_indices,
                &weights,
                self.assignment,
                group,
                stream,
                execute,
            )?,
            None => dispatch_local_with(
                &hidden,
                &group_indices,
                &weights,
                self.assignment,
                stream,
                execute,
            )?,
        };
        self.statistics.accumulate(&returned.statistics);
        Ok(crate::MlxTensor::from_array(
            returned.reduced_output.reshape(&original_shape, stream)?,
        ))
    }

    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, crate::MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<crate::MlxTensor>, Self::Error> {
        let input = request.input.as_array();
        let group_indices_input = request.routes.group_indices().as_array();
        let coefficients = request.routes.coefficients().as_array();
        let original_shape = input.shape().to_vec();
        let hidden = input.reshape(&[-1, input.dim(-1)], stream)?;
        let group_indices = group_indices_input.reshape(&[-1, group_indices_input.dim(-1)], stream)?;
        let weights = coefficients.reshape(&[-1, coefficients.dim(-1)], stream)?;
        let mut bank = CachedLocalBank {
            spec: resident_bank.spec(),
            layer: request.layer,
            pass: request.pass,
            cache: self.cache,
            local_global_group_indices: self.assignment.local_global_group_indices(),
        };
        let returned = match self.expert_group {
            Some(group) => dispatch_replicated_tensor_parallel(
                &hidden,
                &group_indices,
                &weights,
                self.assignment,
                &mut bank,
                group,
                partitions,
                stream,
            )?,
            None => dispatch_local_tensor_parallel(
                &hidden,
                &group_indices,
                &weights,
                self.assignment,
                &mut bank,
                partitions,
                stream,
            )?,
        };
        self.statistics.accumulate(&returned.statistics);
        let (reducible, post_reduce) = returned.output.into_parts();
        let reducible = reducible.reshape(&original_shape, stream)?;
        let post_reduce = post_reduce
            .map(|bias| bias.reshape(&original_shape, stream))
            .transpose()?;
        Ok(RoutedExpertTensorParallelOutput::Partial(
            eredu_nn::TensorParallelGroupedOutput::new(
                crate::MlxTensor::from_array(reducible),
                post_reduce.map(crate::MlxTensor::from_array),
            ),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        _request: RoutedExpertRequest<'_, crate::MlxTensor>,
        _stream: &Stream,
    ) -> Result<crate::MlxTensor, Self::Error> {
        Err(Error::ArchitectureModel(
            "GPT-OSS cannot execute a ReLU2 expert bank".into(),
        ))
    }
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
