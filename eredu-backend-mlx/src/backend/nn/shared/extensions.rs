use super::*;
use eredu_nn::GroupedLinearSpec;
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
        Self::grouped_linear_with_input_observer(
            linear,
            input,
            groups,
            output_per_group,
            context,
            None,
        )
    }

    fn grouped_linear_with_input_observer(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        groups: i32,
        output_per_group: i32,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        let shape = input.as_array();
        if shape.ndim() != 4 || shape.dim(1) != groups || groups <= 0 || output_per_group <= 0 {
            return Err(ComputeError::backend(format!(
                "grouped linear expects [batch, {groups}, tokens, input] and positive output width, got {:?}",
                shape.shape()
            )));
        }
        let width = groups
            .checked_mul(output_per_group)
            .ok_or_else(|| ComputeError::backend("grouped projection width overflow"))?;
        let projected = linear.forward_with_input_observer(input, context, observer)?;
        let projected = projected.as_array();
        if projected.dim(-1) != width {
            return Err(ComputeError::backend(format!(
                "grouped linear produced width {}, expected {}",
                projected.dim(-1),
                width
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
        common::grouped::joint_selection(input, context)
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

    fn grouped_gated_product(spec: GroupedGatedProductSpec, context: &Stream)
        -> Result<Self::GatedProductGroups, ComputeError> {
        construct_gated(spec, grouped_construction::Constructor::ordinary(context))
    }
    fn grouped_relu2(spec: GroupedRelu2Spec, context: &Stream)
        -> Result<Self::Relu2Groups, ComputeError> {
        construct_relu2(spec, grouped_construction::Constructor::ordinary(context))
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
            fn relative_attention<N: NeuralBackend<Tensor = MlxTensor>>(
                &mut self,
                request: eredu_nn::RelativeAttentionInput<'_, MlxTensor>,
                context: &Stream,
            ) -> Result<MlxTensor, ComputeError> {
                request.validate()?;
                if let Some(output) = compute(KeyValueCache::paged_relative_attention(
                    self, &request, context,
                ))? {
                    return Ok(MlxTensor::from_array(output));
                }
                N::relative_attention(request, context)
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
        .map_err(ComputeError::backend_retained_source)
    }
}

#[cfg(test)]
mod convolution_error_tests {
    use crate::backend::runtime::cache::state::MlxHybridState;
    use eredu_core::{
        cache::{LayerCachePolicy, StateTensorRole},
        LayerSchedule,
    };
    use eredu_nn::AuxiliaryConvolutionState;
    use eredu_runtime::{StateError, StateLayout};
    use std::error::Error as _;

    #[test]
    fn missing_convolution_errors_keep_typed_roles_after_native_state_drop() {
        // The actual NoState realization requires no device, stream or tensor.
        let layout =
            StateLayout::new(LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap())
                .unwrap();
        let mut state = MlxHybridState::device(layout).unwrap();
        let mut errors = Vec::new();
        for slot in 0..4 {
            errors.push((
                slot,
                state.layers_mut()[0].convolution_state(slot).unwrap_err(),
            ));
        }
        assert_eq!(state.offset(), 0);
        drop(state);
        for (slot, error) in errors {
            let cloned = error.clone();
            drop(error);
            let role = StateTensorRole::Convolution { slot };
            assert!(matches!(
                cloned.source().and_then(|cause| cause.downcast_ref::<StateError>()),
                Some(StateError::UnknownComponent { role: actual }) if *actual == role
            ));
            assert_eq!(
                cloned.to_string(),
                StateError::UnknownComponent { role }.to_string()
            );
        }
    }
}


impl MlxNeuralBackend {
    pub(crate) fn grouped_gated_product_from_bindings(spec:GroupedGatedProductSpec,
        bindings:PreparedCompactBindings<'_>)->Result<MlxGroupedGatedProduct,ComputeError> {
        construct_gated(spec,grouped_construction::Constructor::prepared::<MlxGroupedGatedProduct>(bindings)?)
    }
    pub(crate) fn grouped_relu2_from_bindings(spec:GroupedRelu2Spec,
        bindings:PreparedCompactBindings<'_>)->Result<MlxGroupedRelu2,ComputeError> {
        construct_relu2(spec,grouped_construction::Constructor::prepared::<MlxGroupedRelu2>(bindings)?)
    }
}
fn construct_gated(spec:GroupedGatedProductSpec, mut constructor:grouped_construction::Constructor<'_,'_>)
    ->Result<MlxGroupedGatedProduct,ComputeError> {
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

    let mut module=common::grouped::PackedGatedProductGroups::new_with_parameter_factory(
        spec.group_count(),spec.input_dimensions(),spec.intermediate_dimensions(),
        gate_up.format().encoding().weight_quantization(),down.format().encoding().weight_quantization(),
        [gate_up.bias().is_some(),down.bias().is_some()],Dtype::Float32,
        native_fp8.map(|format|(format,gate_up.format().row_layout())),&mut constructor)?;
    module=compute(module.with_policy(policy))?;
    module.reduction=spec.reduction();
    let rows=gated_declarations(&spec)?;
    let module=constructor.named(module,&rows)?;
    Ok(MlxGroupedGatedProduct{spec,module})
}
fn construct_relu2(spec:GroupedRelu2Spec, mut constructor:grouped_construction::Constructor<'_,'_>)
    ->Result<MlxGroupedRelu2,ComputeError> {
        spec.validate()?;
        if spec.up().bias().is_some() || spec.down().bias().is_some() {
            return Err(ComputeError::backend(
                "MLX packed ReLU2 groups do not support ordinary projection biases",
            ));
        }

    let module=common::grouped::PackedRelu2Groups::new_with_parameter_factory(
        spec.group_count(),spec.hidden_dimensions(),spec.intermediate_dimensions(),
        [spec.up().format().encoding().weight_quantization(),spec.down().format().encoding().weight_quantization()],
        Dtype::Float32,&mut constructor)?;
    let rows=relu2_declarations(&spec);
    let module=constructor.named(module,&rows)?;
    Ok(MlxGroupedRelu2{spec,module})
}

fn gated_declarations(spec:&GroupedGatedProductSpec)
    ->Result<[Option<grouped_construction::Declaration<'_>>;8],ComputeError> {
    use grouped_construction::Declaration as D;
    let GatedProductGroupLayout::Packed{gate_up,down}=spec.layout() else {
        return Err(ComputeError::backend("compact gated bank requires the retained packed declaration"));
    };
    Ok([Some(D::parameter("down_proj",down.weight())),
        down.bias().map(|p|D::parameter("down_proj_bias",p)),
        down.format().affine_bias().map(|p|D::companion("down_proj_biases",down.weight(),p)),
        down.format().scale().map(|p|D::companion("down_proj_scales",down.weight(),p)),
        Some(D::parameter("gate_up_proj",gate_up.weight())),
        gate_up.bias().map(|p|D::parameter("gate_up_proj_bias",p)),
        gate_up.format().affine_bias().map(|p|D::companion("gate_up_proj_biases",gate_up.weight(),p)),
        gate_up.format().scale().map(|p|D::companion("gate_up_proj_scales",gate_up.weight(),p))])
}
fn relu2_declarations(spec:&GroupedRelu2Spec)
    ->[Option<grouped_construction::Declaration<'_>>;6] {
    use grouped_construction::Declaration as D;
    [Some(D::parameter("down_proj",spec.down().weight())),
        spec.down().format().affine_bias().map(|p|D::companion("down_proj_biases",spec.down().weight(),p)),
        spec.down().format().scale().map(|p|D::companion("down_proj_scales",spec.down().weight(),p)),
        Some(D::parameter("up_proj",spec.up().weight())),
        spec.up().format().affine_bias().map(|p|D::companion("up_proj_biases",spec.up().weight(),p)),
        spec.up().format().scale().map(|p|D::companion("up_proj_scales",spec.up().weight(),p))]
}
impl MlxNeuralBackend {
    pub(crate) fn gated_compact_parameter_names(spec:&GroupedGatedProductSpec)
        ->Result<[Option<&'static str>;8],ComputeError> {
        Ok(gated_declarations(spec)?.map(|row|row.map(|row|row.name)))
    }
    pub(crate) fn relu2_compact_parameter_names(spec:&GroupedRelu2Spec)->[Option<&'static str>;6] {
        relu2_declarations(spec).map(|row|row.map(|row|row.name))
    }
    pub(crate) fn linear_compact_parameter_names(spec:&GroupedLinearSpec)->[Option<&str>;4] {
        super::selected_linear::MlxGroupedLinear::local_bindings(spec)
    }
    pub(crate) fn grouped_linear_from_bindings(spec:GroupedLinearSpec,bindings:PreparedCompactBindings<'_>)
        ->Result<super::selected_linear::MlxGroupedLinear,ComputeError> {
        super::selected_linear::MlxGroupedLinear::from_prepared_bindings(spec,bindings)
    }
}
