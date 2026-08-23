// MLX residency adapter for the neutral GPT-OSS routed-expert graph.

use eredu_architectures::gpt_oss::{expert_recipes, ModelArgs};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_nn::{GatedProductExpertBankOperator, GatedProductExpertBankSpec};
use eredu_runtime::{
    ExpertIdentity, ExpertPass, LayeredArchitecture, OffloadUnit, ParameterGroupOwner,
    ParameterRole, RoutedExpertProvider, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
    WeightBinding,
};
use safemlx::{distributed::Group, Array, Stream};

use crate::backend::mlx::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    runtime::{
        checkpoint::{
            binding_plan::{BindingPlan, PlannedBinding},
            recipe::lower_mxfp4_recipe,
        },
        distributed::expert::{
            dispatch_local_tensor_parallel, dispatch_local_with,
            dispatch_replicated_tensor_parallel, dispatch_replicated_with, DispatchedRoutes,
            ExpertAssignment, LocalExpertBank, RoutingStatistics,
        },
        execution::layerwise::shard_layer_bindings,
        residency::{
            expert_cache::{ExpertCache, ExpertCatalogEntry},
            expert_provider::{
                execute_cached_gated_product_dispatched,
                execute_cached_gated_product_tensor_parallel, CachedGatedProductExpertProvider,
            },
        },
    },
};

/// Builds expert-granular canonical bindings with optional semantic TP selection.
///
/// The architecture recipe first normalizes alternating SafeTensors rows or
/// translated separate GGUF projections to one component-major family. Expert
/// selection is then pushed through that recipe before TP placement is applied,
/// keeping native blocks, E8M0 scales, and ordinary biases in one atomic unit.
pub fn expert_catalog_cartesian(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    stream: &Stream,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let architecture = eredu_architectures::gpt_oss::LayeredModel::<MlxNeuralBackend>::new(
        args.clone(),
        stream,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let description = eredu_architectures::gpt_oss::parameter_description(&architecture, stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let graph = <eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
        MlxNeuralBackend,
        super::Cache,
    >>::execution_graph(&architecture)
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let layers = positive_dimension(args.num_hidden_layers, "layer count")?;
    let experts = positive_dimension(args.num_local_experts, "expert count")?;
    let capacity = layers.checked_mul(experts).ok_or_else(|| {
        Error::UnsupportedArchitecture("GPT-OSS expert catalog size overflowed".into())
    })?;
    let mut entries = Vec::with_capacity(capacity);

    for layer in 0..layers {
        let mut owner_group = None;
        let expert_targets = description
            .groups()
            .iter()
            .filter(|owned| owned.role() == ParameterRole::ExpertIntermediate)
            .filter_map(|owned| match owned.owner() {
                ParameterGroupOwner::ExecutionUnit { group, global_unit }
                    if *global_unit == layer =>
                {
                    owner_group.get_or_insert(group.clone());
                    Some(owned.members())
                }
                _ => None,
            })
            .flatten()
            .map(|member| member.target().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        let owner_group = owner_group.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GPT-OSS layer {layer} declares no expert-intermediate owner"
            ))
        })?;
        let group = graph
            .groups()
            .iter()
            .position(|candidate| candidate.id() == owner_group.as_str())
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "GPT-OSS expert owner group {owner_group:?} is absent from the execution graph"
                ))
            })?;
        let unit_path = <eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            super::Cache,
        >>::unit_path(&architecture, group, layer)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let resolved =
            expert_recipes(store, args, layer).map_err(Error::UnsupportedArchitecture)?;
        let canonical = resolved.into_outputs().into_outputs();

        for expert in 0..experts {
            let expert_selection = TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            };
            let mut bindings = Vec::with_capacity(canonical.len());
            for (target, recipe) in &canonical {
                if !expert_targets.contains(target) {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "GPT-OSS expert recipe target {target:?} is not owned by the architecture expert-intermediate role"
                    )));
                }
                let local_name = target.rsplit('.').next().ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "GPT-OSS expert recipe target {target:?} has no compact binding name"
                    ))
                })?;
                let selected = recipe.select_bounded(store, expert_selection.clone())?;
                let selected = if selected.infer(store)?.dtype()
                    == &eredu_checkpoint::recipe::RecipeDtype::F4
                {
                    lower_mxfp4_recipe(selected, store)?
                } else {
                    selected
                };
                bindings.push(recipe_binding(local_name, target, selected, store)?);
            }

            if let Some(layout) = layout {
                bindings = shard_layer_bindings(bindings, &unit_path, store, layout)?;
            }

            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("GPT-OSS expert byte total overflowed".into())
                })
            })?;
            let identity = ExpertIdentity::new(layer, expert);
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

/// Adapts an expert cache to the neutral routed-provider contract.
pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

/// Executes rows already compacted by an EP dispatcher.
pub fn execute_cached_dispatched(
    cache: &ExpertCache,
    spec: &GatedProductExpertBankSpec,
    layer: usize,
    hidden: &Array,
    global_expert_ids: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_gated_product_dispatched(
        cache,
        spec,
        layer,
        hidden,
        global_expert_ids,
        pass,
        stream,
    )
}

/// Executes compact EP rows as a typed TP partial so callers can all-sum the
/// reducible projection and add routed down bias exactly once afterwards.
pub fn execute_cached_dispatched_tensor_parallel(
    cache: &ExpertCache,
    spec: &GatedProductExpertBankSpec,
    layer: usize,
    hidden: &Array,
    global_expert_ids: &Array,
    pass: ExpertPass,
    partitions: usize,
    stream: &Stream,
) -> Result<eredu_nn::TensorParallelExpertOutput<Array>, Error> {
    if partitions == 0 {
        return Err(Error::Parallel(
            "GPT-OSS cached expert execution requires a positive TP size".into(),
        ));
    }
    let expert_ids = global_expert_ids.reshape(&[-1, 1], stream)?;
    let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    execute_cached_gated_product_tensor_parallel(
        cache,
        spec,
        layer,
        hidden,
        &expert_ids,
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
    cache: &'a ExpertCache,
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
    cache: &'a ExpertCache,
    statistics: &'a mut RoutingStatistics,
}

struct CachedLocalBank<'a> {
    spec: &'a GatedProductExpertBankSpec,
    layer: usize,
    pass: ExpertPass,
    cache: &'a ExpertCache,
    local_global_expert_ids: &'a [usize],
}

impl CachedLocalBank<'_> {
    fn global_ids(&self, local_ids: &Array, stream: &Stream) -> Result<Array, Error> {
        let ids = self
            .local_global_expert_ids
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
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        execute_cached_dispatched(
            self.cache,
            self.spec,
            self.layer,
            hidden,
            &self.global_ids(local_expert_ids, stream)?,
            self.pass,
            stream,
        )
    }

    fn execute_local_routes_tensor_parallel(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<eredu_nn::TensorParallelExpertOutput<Array>, Error> {
        execute_cached_dispatched_tensor_parallel(
            self.cache,
            self.spec,
            self.layer,
            hidden,
            &self.global_ids(local_expert_ids, stream)?,
            self.pass,
            partitions,
            stream,
        )
    }
}

impl RoutedExpertProvider<MlxNeuralBackend> for DistributedCachedProvider<'_> {
    type Error = Error;

    fn forward_routed(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, crate::MlxTensor>,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Self::Error> {
        let input = request.input.as_array();
        let expert_ids_input = request.routes.expert_ids.as_array();
        let route_weights = request.routes.route_weights.as_array();
        let original_shape = input.shape().to_vec();
        let hidden = input.reshape(&[-1, input.dim(-1)], stream)?;
        let expert_ids = expert_ids_input.reshape(&[-1, expert_ids_input.dim(-1)], stream)?;
        let weights = route_weights.reshape(&[-1, route_weights.dim(-1)], stream)?;
        let execute = |routes: &DispatchedRoutes, stream: &Stream| {
            execute_cached_dispatched(
                self.cache,
                resident_bank.spec(),
                request.layer,
                &routes.hidden,
                &routes.global_expert_ids,
                request.pass,
                stream,
            )
        };
        let returned = match self.expert_group {
            Some(group) => dispatch_replicated_with(
                &hidden,
                &expert_ids,
                &weights,
                self.assignment,
                group,
                stream,
                execute,
            )?,
            None => dispatch_local_with(
                &hidden,
                &expert_ids,
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

    fn forward_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, crate::MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<crate::MlxTensor>, Self::Error> {
        let input = request.input.as_array();
        let expert_ids_input = request.routes.expert_ids.as_array();
        let route_weights = request.routes.route_weights.as_array();
        let original_shape = input.shape().to_vec();
        let hidden = input.reshape(&[-1, input.dim(-1)], stream)?;
        let expert_ids = expert_ids_input.reshape(&[-1, expert_ids_input.dim(-1)], stream)?;
        let weights = route_weights.reshape(&[-1, route_weights.dim(-1)], stream)?;
        let mut bank = CachedLocalBank {
            spec: resident_bank.spec(),
            layer: request.layer,
            pass: request.pass,
            cache: self.cache,
            local_global_expert_ids: self.assignment.local_global_expert_ids(),
        };
        let returned = match self.expert_group {
            Some(group) => dispatch_replicated_tensor_parallel(
                &hidden,
                &expert_ids,
                &weights,
                self.assignment,
                &mut bank,
                group,
                partitions,
                stream,
            )?,
            None => dispatch_local_tensor_parallel(
                &hidden,
                &expert_ids,
                &weights,
                self.assignment,
                &mut bank,
                partitions,
                stream,
            )?,
        };
        self.statistics.accumulate(&returned.statistics);
        let reducible = returned.output.reducible.reshape(&original_shape, stream)?;
        let post_reduce = returned
            .output
            .post_reduce
            .map(|bias| bias.reshape(&original_shape, stream))
            .transpose()?;
        Ok(RoutedExpertTensorParallelOutput::Partial(
            eredu_nn::TensorParallelExpertOutput {
                reducible: crate::MlxTensor::from_array(reducible),
                post_reduce: post_reduce.map(crate::MlxTensor::from_array),
            },
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        _request: RoutedExpertRequest<'_, crate::MlxTensor>,
        _stream: &Stream,
    ) -> Result<crate::MlxTensor, Self::Error> {
        Err(Error::UnsupportedArchitecture(
            "GPT-OSS cannot execute a ReLU2 expert bank".into(),
        ))
    }
}

fn recipe_binding(
    local_name: &str,
    logical_target: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<WeightBinding, Error> {
    let metadata = recipe.infer(store)?;
    let mut bindings = BindingPlan::new(vec![PlannedBinding {
        target_name: logical_target.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    }])
    .and_then(|plan| plan.build_bindings(store))
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    bindings
        .pop()
        .expect("single planned GPT-OSS expert binding")
        .with_name(local_name)
        .map_err(Error::from)
}

fn positive_dimension(value: i32, name: &str) -> Result<usize, Error> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GPT-OSS {name} must be positive, got {value}"))
        })
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
        let eredu_nn::GatedProductExpertLayout::Packed { gate_up, down } = &spec.layout else {
            panic!("GPT-OSS experts must use packed architecture geometry");
        };
        assert_eq!(
            gate_up.format.weight_quantization(),
            Some(eredu_checkpoint::WeightQuantization::MxFp4)
        );
        assert_eq!(
            down.format.weight_quantization(),
            Some(eredu_checkpoint::WeightQuantization::MxFp4)
        );
        assert!(gate_up.bias.is_some());
        assert!(down.bias.is_some());
        assert_eq!(spec.policy, args.gated_product_policy);
        assert_eq!(spec.policy.sigmoid_multiplier(), 1.702);
        assert_eq!(spec.policy.up_offset(), 1.0);
        assert_eq!(spec.policy.gate_upper_bound(), Some(7.0));
        assert_eq!(spec.policy.up_absolute_bound(), Some(7.0));
    }

    #[test]
    fn invalid_catalog_geometry_is_rejected_before_store_access() {
        assert!(positive_dimension(0, "expert count").is_err());
        assert!(positive_dimension(-1, "layer count").is_err());
    }
}
