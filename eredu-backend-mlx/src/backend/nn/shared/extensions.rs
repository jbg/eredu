use super::*;
use super::{operators::*, parameters::*};

impl HyperNeuralBackend for MlxNeuralBackend {
    type HyperConnection = MlxHyperConnection;
    type HyperHead = MlxHyperHead;

    fn hyper_connection(
        spec: HyperConnectionSpec,
        context: &Stream,
    ) -> Result<Self::HyperConnection, ComputeError> {
        spec.validate()?;
        let module = compute(common::hyper_connections::HyperConnection::unloaded(
            spec.streams,
            spec.hidden_size,
            spec.sinkhorn_iterations,
            spec.epsilon,
            context,
        ))?;
        let topology = exact_parameter_topology(
            &module,
            [
                ("function", spec.function),
                ("base", spec.base),
                ("scale", spec.scale),
            ],
        )?;
        Ok(MlxHyperConnection { module, topology })
    }

    fn hyper_head(spec: HyperHeadSpec, context: &Stream) -> Result<Self::HyperHead, ComputeError> {
        spec.validate()?;
        let module = compute(common::hyper_connections::HyperHead::unloaded(
            spec.streams,
            spec.hidden_size,
            spec.norm_epsilon,
            spec.epsilon,
            context,
        ))?;
        let topology = exact_parameter_topology(
            &module,
            [
                ("function", spec.function),
                ("base", spec.base),
                ("scale", spec.scale),
            ],
        )?;
        Ok(MlxHyperHead { module, topology })
    }
}

impl GroupedNeuralBackend for MlxNeuralBackend {
    type LinearGroups = super::selected_linear::MlxGroupedLinear;

    fn grouped_linear_bank(
        spec: eredu_nn::GroupedLinearSpec,
        context: &Stream,
    ) -> Result<Self::LinearGroups, ComputeError> {
        Self::LinearGroups::new(spec, context)
    }

    type Selector = MlxTopKGroupSelector;
    type GatedProductGroups = MlxGroupedGatedProduct;
    type Relu2Groups = MlxGroupedRelu2;

    fn grouped_linear(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        groups: i32,
        output_per_group: i32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        if input.ndim() != 4 || input.dim(1) != groups || groups <= 0 || output_per_group <= 0 {
            return Err(ComputeError::backend(format!(
                "grouped linear expects [batch, {groups}, tokens, input] and positive output width, got {:?}",
                input.shape()
            )));
        }
        let projected = compute(linear.module.forward(input, context))?;
        if projected.dim(-1) != groups * output_per_group {
            return Err(ComputeError::backend(format!(
                "grouped linear produced width {}, expected {}",
                projected.dim(-1),
                groups * output_per_group
            )));
        }
        let mut pieces = Vec::with_capacity(groups as usize);
        for group in 0..groups {
            let selected = compute(projected.try_index_device(
                (
                    ..,
                    group,
                    ..,
                    group * output_per_group..(group + 1) * output_per_group,
                ),
                context,
            ))?;
            pieces.push(compute(selected.expand_dims(1, context))?);
        }
        compute_tensor(concatenate_axis(&pieces, 1, context))
    }

    fn joint_group_selection(
        input: JointGroupSelectionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<JointGroupSelection<MlxTensor>, ComputeError> {
        input.validate()?;
        let hidden_width = input.hidden().as_array().dim(-1);
        let flat = compute(
            input
                .hidden()
                .as_array()
                .reshape(&[-1, hidden_width], context),
        )?;
        let logits = compute(matmul(
            &flat,
            &compute(input.weight().as_array().transpose(context))?,
            context,
        ))?;
        let primary = compute(logits.try_index_device((.., ..input.selectable_groups()), context))?;
        let always_on =
            compute(logits.try_index_device((.., input.selectable_groups()..), context))?;
        let choice = compute(sigmoid(&primary, context))?;
        let choice = compute(choice.add(input.correction_bias().as_array(), context))?;
        let primary_indices = compute(argpartition_axis(choice, -input.top_k(), -1, context))?;
        let primary_indices =
            compute(primary_indices.try_index_device((.., -input.top_k()..), context))?;
        let selected_logits = compute(take_along_axis(&primary, &primary_indices, -1, context))?;
        let all_logits = compute(concatenate_axis(&[selected_logits, always_on], -1, context))?;
        let coefficients = compute(nn::log_sigmoid(all_logits, context))?;
        let coefficients = compute(softmax_axis(coefficients, -1, true, context))?;
        let coefficients =
            compute(coefficients.multiply(Array::from_f32(input.coefficient_scale()), context))?;
        let coefficients =
            compute(coefficients.multiply(input.global_scale().as_array(), context))?;
        let primary_coefficients =
            compute(coefficients.try_index_device((.., ..input.top_k()), context))?;
        let always_on_coefficients =
            compute(coefficients.try_index_device((.., input.top_k()..), context))?;
        Ok(JointGroupSelection::new(
            MlxTensor::from_array(primary_indices),
            MlxTensor::from_array(primary_coefficients),
            MlxTensor::from_array(always_on_coefficients),
        ))
    }

    fn top_k_group_selector(
        spec: TopKGroupSelectorSpec,
        context: &Stream,
    ) -> Result<Self::Selector, ComputeError> {
        spec.validate()?;
        let selection = spec.selection();
        let score_function = match selection.scoring() {
            GroupScoring::Softmax => common::grouped::TopKGroupScoring::Softmax,
            GroupScoring::SelectedSoftmax => common::grouped::TopKGroupScoring::SelectedSoftmax,
            GroupScoring::Sigmoid => common::grouped::TopKGroupScoring::Sigmoid,
            GroupScoring::SqrtSoftplus => common::grouped::TopKGroupScoring::SqrtSoftplus,
            _ => return Err(ComputeError::backend("unsupported group scoring policy")),
        };
        let module = compute(common::grouped::TopKGroupSelector::new_with_quantization(
            compute(common::grouped::TopKGroupSelectorConfig::new(
                selection.top_k(),
                selection.group_count(),
                spec.input_dimensions(),
                score_function,
                selection.normalize_selected(),
                selection.normalization_epsilon(),
                selection.coefficient_scale(),
                selection.selection_partitions(),
                selection.selected_groups(),
                spec.bias().is_some(),
                spec.correction_bias().is_some(),
                spec.input_transform().map(|transform| transform.epsilon()),
                spec.input_transform()
                    .is_some_and(|transform| transform.inverse_sqrt_dimensions()),
                spec.coefficient_scale().is_some(),
            ))?
            .with_arithmetic(spec.arithmetic()),
            spec.format().encoding().weight_quantization(),
            context,
        ))?;
        let weight = spec.weight().clone();
        let mut topology = vec![("weight", weight.clone())];
        if let Some(bias) = spec.bias() {
            topology.push(("bias", bias.clone()));
        }
        if let Some(scale) = spec.format().scale() {
            topology.push(("scales", bind_linear_companion(&weight, scale.clone())));
        }
        if let Some(bias) = spec.format().affine_bias() {
            topology.push(("biases", bind_linear_companion(&weight, bias.clone())));
        }
        if let Some(correction_bias) = spec.correction_bias() {
            topology.push(("e_score_correction_bias", correction_bias.clone()));
        }
        if let Some(transform) = spec.input_transform() {
            topology.push(("input_scale", transform.scale().clone()));
        }
        if let Some(learned_coefficient_scale) = spec.coefficient_scale() {
            topology.push((
                "learned_coefficient_scale",
                learned_coefficient_scale.clone(),
            ));
        }
        Ok(MlxTopKGroupSelector {
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }

    fn grouped_gated_product(
        spec: GroupedGatedProductSpec,
        context: &Stream,
    ) -> Result<Self::GatedProductGroups, ComputeError> {
        spec.validate()?;
        if spec.input_dimensions() != spec.output_dimensions() {
            return Err(ComputeError::backend(
                "MLX packed gated-product groups require equal input and output dimensions",
            ));
        }
        let policy = spec.policy();
        let GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
            return Err(ComputeError::backend(
                "independent group units must be acquired through a runtime group provider",
            ));
        };
        let native_fp8 = match (gate_up.format().encoding(), down.format().encoding()) {
            (LinearFormat::E4M3BlockFp8(gate), LinearFormat::E4M3BlockFp8(down))
                if gate == down =>
            {
                Some(gate)
            }
            (LinearFormat::E4M3BlockFp8(_), LinearFormat::E4M3BlockFp8(_)) => {
                return Err(ComputeError::backend(
                    "MLX packed block-FP8 groups require matching formats",
                ));
            }
            (LinearFormat::E4M3BlockFp8(_), _) | (_, LinearFormat::E4M3BlockFp8(_)) => {
                return Err(ComputeError::backend(
                    "packed group projections must use one physical format",
                ));
            }
            _ => None,
        };
        let mut module = compute(common::grouped::PackedGatedProductGroups::new(
            spec.group_count(),
            spec.input_dimensions(),
            spec.intermediate_dimensions(),
            gate_up.format().encoding().weight_quantization(),
            down.format().encoding().weight_quantization(),
            [gate_up.bias().is_some(), down.bias().is_some()],
            context,
        ))?;
        module = compute(module.with_policy(policy))?;
        module.reduction = spec.reduction();
        if let Some(format) = native_fp8 {
            module = compute(module.with_native_fp8(format, context))?;
        }
        let mut topology = vec![
            ("gate_up_proj", gate_up.weight().clone()),
            ("down_proj", down.weight().clone()),
        ];
        if let Some(bias) = gate_up.bias() {
            topology.push(("gate_up_proj_bias", bias.clone()));
        }
        if let Some(bias) = down.bias() {
            topology.push(("down_proj_bias", bias.clone()));
        }
        if let Some(scale) = gate_up.format().scale() {
            topology.push((
                "gate_up_proj_scales",
                bind_linear_companion(gate_up.weight(), scale.clone()),
            ));
        }
        if let Some(bias) = gate_up.format().affine_bias() {
            topology.push((
                "gate_up_proj_biases",
                bind_linear_companion(gate_up.weight(), bias.clone()),
            ));
        }
        if let Some(scale) = down.format().scale() {
            topology.push((
                "down_proj_scales",
                bind_linear_companion(down.weight(), scale.clone()),
            ));
        }
        if let Some(bias) = down.format().affine_bias() {
            topology.push((
                "down_proj_biases",
                bind_linear_companion(down.weight(), bias.clone()),
            ));
        }
        Ok(MlxGroupedGatedProduct {
            spec,
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }

    fn grouped_relu2(
        spec: GroupedRelu2Spec,
        context: &Stream,
    ) -> Result<Self::Relu2Groups, ComputeError> {
        spec.validate()?;
        if spec.up().bias().is_some() || spec.down().bias().is_some() {
            return Err(ComputeError::backend(
                "MLX packed ReLU2 groups do not support ordinary projection biases",
            ));
        }
        let module = compute(common::grouped::PackedRelu2Groups::new(
            spec.group_count(),
            spec.hidden_dimensions(),
            spec.intermediate_dimensions(),
            [
                spec.up().format().encoding().weight_quantization(),
                spec.down().format().encoding().weight_quantization(),
            ],
            context,
        ))?;
        let mut topology = vec![
            ("up_proj", spec.up().weight().clone()),
            ("down_proj", spec.down().weight().clone()),
        ];
        if let Some(scale) = spec.up().format().scale() {
            topology.push((
                "up_proj_scales",
                bind_linear_companion(spec.up().weight(), scale.clone()),
            ));
        }
        if let Some(bias) = spec.up().format().affine_bias() {
            topology.push((
                "up_proj_biases",
                bind_linear_companion(spec.up().weight(), bias.clone()),
            ));
        }
        if let Some(scale) = spec.down().format().scale() {
            topology.push((
                "down_proj_scales",
                bind_linear_companion(spec.down().weight(), scale.clone()),
            ));
        }
        if let Some(bias) = spec.down().format().affine_bias() {
            topology.push((
                "down_proj_biases",
                bind_linear_companion(spec.down().weight(), bias.clone()),
            ));
        }
        Ok(MlxGroupedRelu2 {
            spec,
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }
}

macro_rules! impl_attention_cache {
    ($type:ty) => {
        impl AttentionCache<MlxTensor> for $type {
            fn uses_blockwise_attention(&self) -> bool {
                KeyValueCache::is_paged(self)
            }
            fn offset(&self) -> i32 {
                KeyValueCache::offset(self)
            }
            fn max_size(&self) -> Option<i32> {
                KeyValueCache::max_size(self)
            }
            fn update_for_attention(
                &mut self,
                keys: MlxTensor,
                values: MlxTensor,
                context: &Stream,
            ) -> Result<(MlxTensor, MlxTensor), ComputeError> {
                compute(KeyValueCache::update_for_attention(
                    self,
                    keys.into_array(),
                    values.into_array(),
                    context,
                ))
                .map(|(keys, values)| (MlxTensor::from_array(keys), MlxTensor::from_array(values)))
            }
            fn attention(
                &mut self,
                request: AttentionRequest<'_, MlxTensor>,
                context: &Stream,
            ) -> Result<MlxTensor, ComputeError> {
                request.validate()?;
                if let Some(output) = compute(KeyValueCache::paged_attention(
                    self,
                    request.queries.as_array(),
                    request.scale,
                    request.mask.map(MlxTensor::as_array),
                    request.sinks.map(MlxTensor::as_array),
                    request.softcap,
                    request.arithmetic,
                    context,
                ))? {
                    return Ok(MlxTensor::from_array(output));
                }
                compute_tensor(crate::backend::nn::attention::attention_with_softcap(
                    request.queries.as_array(),
                    request.keys.as_array(),
                    request.values.as_array(),
                    request.scale,
                    request.mask.map(MlxTensor::as_array),
                    request.sinks.map(MlxTensor::as_array),
                    request.softcap,
                    request.arithmetic,
                    context,
                ))
            }
        }
    };
}

impl_attention_cache!(ConcatKeyValueCache);
impl_attention_cache!(PagedKeyValueCache);
impl_attention_cache!(crate::backend::runtime::cache::state::MlxKeyValueLayerState);

impl eredu_nn::AuxiliaryConvolutionState<MlxTensor>
    for crate::backend::runtime::cache::state::MlxHybridLayerState
{
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<MlxTensor>, ComputeError> {
        eredu_runtime::RuntimeStateComponents::fixed_component(
            self,
            eredu_core::cache::StateTensorRole::Convolution { slot },
        )
        .map_err(ComputeError::backend)
    }
}
