use super::*;
use eredu_nn::{GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec, Parameter};

#[cfg(test)]
mod retained_tests;
pub(super) mod original;

/// MLX indexed affine projections with activation before weighted reduction.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "MlxTensor")]
pub struct MlxGroupedLinear {
    #[parameter(skip, metadata)]
    spec: GroupedLinearSpec,
    weight: Parameter<MlxTensor>,
    scale: Option<Parameter<MlxTensor>>,
    affine_bias: Option<Parameter<MlxTensor>>,
    bias: Option<Parameter<MlxTensor>>,
    #[parameter(skip, metadata)]
    construction_funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}

impl MlxGroupedLinear {
    pub(super) fn construction_metadata_bytes(spec: &GroupedLinearSpec) -> Option<usize> {
        use super::grouped_construction::Constructor;
        let projection = spec.projection();
        let rows = Self::local_bindings(spec).into_iter().flatten().count();
        let mut bytes = Constructor::control_bytes::<Self>(rows)?
            .checked_add(Constructor::parameter_metadata_bytes(projection.weight(), None)?)?;
        for source in [projection.format().scale(), projection.format().affine_bias()].into_iter().flatten() {
            bytes = bytes.checked_add(Constructor::parameter_metadata_bytes(source, Some(projection.weight()))?)?;
        }
        if let Some(source) = projection.bias() {
            bytes = bytes.checked_add(Constructor::parameter_metadata_bytes(source, None)?)?;
        }
        Some(bytes)
    }
    pub(crate) fn original_fp8_control_bytes() -> Option<usize> {
        original::control_bytes()
    }
    pub(super) fn local_bindings(spec: &GroupedLinearSpec) -> [Option<&str>; 4] {
        let projection = spec.projection();
        [Some(projection.weight()), projection.format().scale(),
            projection.format().affine_bias(), projection.bias()]
            .map(|spec| spec.map(|spec| spec.id.as_str().rsplit('.').next()
                .expect("validated parameter identity")))
    }

    pub(crate) fn local_parameter_names(&self) -> Vec<String> {
        Self::local_bindings(&self.spec).into_iter().flatten().map(str::to_owned).collect()
    }

    pub(crate) fn bind_local_parameters(&mut self, mut bindings: BTreeMap<String, Array>)
        -> Result<(), ComputeError> {
        self.bind_compact_values(&mut bindings)
    }

    pub(crate) fn bind_prepared_local_parameters(&mut self,
        bindings: &mut super::parameters::PreparedCompactBindings<'_>) -> Result<(), ComputeError> {
        self.bind_compact_values(bindings)
    }

    fn bind_compact_values<V: super::parameters::CompactBindingValues + ?Sized>(
        &mut self, bindings: &mut V) -> Result<(), ComputeError> {
        use super::parameters::compact::CompactBindingCause;
        bindings.begin(super::parameters::compact::linear_control_bytes())?;
        let names = Self::local_bindings(&self.spec);
        if bindings.len() != names.iter().flatten().count() || !bindings.ready() {
            return Err(bindings.failure(CompactBindingCause::Identity));
        }
        // The immutable spec names are borrowed from their retained owner.
        // Every shape/encoding is checked before any Array moves into a slot.
        let floating_shape = [self.spec.group_count(), self.spec.output_dimensions(), self.spec.input_dimensions()];
        for (slot, (name, parameter)) in names.iter().zip([
            Some(&self.weight), self.scale.as_ref(), self.affine_bias.as_ref(), self.bias.as_ref(),
        ]).enumerate() {
            match (name, parameter) {
                (Some(name), Some(parameter)) => {
                    let value = bindings.value(name).ok_or_else(|| bindings.failure(CompactBindingCause::Identity))?;
                    super::parameters::compact::validate(parameter.as_ref().as_array(), value,
                        (slot == 0).then_some(floating_shape.as_slice())).map_err(|cause|bindings.failure(cause))?;
                }
                (None, None) => {}
                _ => return Err(bindings.failure(CompactBindingCause::Identity)),
            }
        }
        for (name, parameter) in names.iter().zip([
            Some(&mut self.weight), self.scale.as_mut(), self.affine_bias.as_mut(), self.bias.as_mut(),
        ]) {
            if let (Some(name), Some(parameter)) = (name, parameter) {
                parameter.replace(MlxTensor::from_array(bindings.take(name)
                    .expect("validated unique physical binding")));
            }
        }
        Ok(())
    }

    pub(super) fn new(spec: GroupedLinearSpec, context: &Stream) -> Result<Self, ComputeError> {
        Self::construct(spec,super::grouped_construction::Constructor::ordinary(context))
    }
    pub(crate) fn from_prepared_bindings(spec:GroupedLinearSpec,bindings:super::parameters::PreparedCompactBindings<'_>)
        ->Result<Self,ComputeError> {
        Self::construct(spec,super::grouped_construction::Constructor::prepared::<Self>(bindings)?)
    }
    fn construct(spec:GroupedLinearSpec,mut constructor:super::grouped_construction::Constructor<'_,'_>)
        ->Result<Self,ComputeError> {
        use crate::backend::nn::grouped::ParameterFactory;
        spec.validate()?;
        let (groups, rows, columns) = (
            spec.group_count(),
            spec.output_dimensions(),
            spec.input_dimensions(),
        );
        let projection = spec.projection();
        let encoding = projection.format().encoding();
        let mut shape = [groups, rows, columns];
        let mut scale_shape = shape;
        match encoding {
            LinearFormat::Dense => {}
            LinearFormat::E4M3BlockFp8(fp8) => {
                if fp8.block_rows != 128 || fp8.block_columns != 128 {
                    return Err(ComputeError::backend(
                        "MLX indexed FP8 projection requires 128 by 128 scale blocks",
                    ));
                }
                scale_shape[1] = (rows as u32).div_ceil(fp8.block_rows as u32) as i32;
                scale_shape[2] = (columns as u32).div_ceil(fp8.block_columns as u32) as i32;
            }
            LinearFormat::GgufIQuant { .. } => {
                let quantization = encoding.weight_quantization().expect("GGUF packed format");
                let (ty, _) = quantization.gguf_iquant().expect("GGUF packed type");
                let (values, bytes) = ty.block_and_bytes().map_err(ComputeError::backend)?;
                if columns % values as i32 != 0 {
                    return Err(ComputeError::backend(
                        "grouped GGUF row is not block aligned",
                    ));
                }
                shape[2] = columns / values as i32 * bytes as i32;
            }
            LinearFormat::Affine(_) | LinearFormat::MxFp4 => {
                let quantization = encoding
                    .weight_quantization()
                    .expect("packed linear format");
                if columns % quantization.group_size() != 0 || columns % 32 != 0 {
                    return Err(ComputeError::backend("grouped packed row is not aligned"));
                }
                shape[2] = columns / 32 * quantization.bits();
                scale_shape[2] = columns / quantization.group_size();
            }
        }
        let weight_dtype = match encoding {
            LinearFormat::Dense => safemlx::Dtype::Float32,
            LinearFormat::Affine(_) | LinearFormat::MxFp4 => safemlx::Dtype::Uint32,
            LinearFormat::E4M3BlockFp8(_) | LinearFormat::GgufIQuant { .. } => {
                safemlx::Dtype::Uint8
            }
        };
        let scale_dtype = match encoding {
            LinearFormat::MxFp4 => safemlx::Dtype::Uint8,
            LinearFormat::E4M3BlockFp8(fp8)
                if fp8.scale_encoding == eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 =>
            {
                safemlx::Dtype::Uint8
            }
            _ => safemlx::Dtype::Float32,
        };
        let names=Self::local_bindings(&spec);
        let weight_spec=constructor.clone_parameter(projection.weight(),None)?;
        let weight_value=constructor.array(names[0].expect("declared grouped weight"),&shape,weight_dtype,
            Some(&[groups,rows,columns]))?;
        let weight=Parameter::new(weight_spec,MlxTensor::from_array(weight_value));
        let scale=projection.format().scale().map(|source|{
            let metadata=constructor.clone_parameter(source,Some(projection.weight()))?;
            let value=constructor.array(names[1].expect("declared grouped scale"),&scale_shape,scale_dtype,None)?;
            Ok::<_,ComputeError>(Parameter::new(metadata,MlxTensor::from_array(value)))
        }).transpose()?;
        let affine_bias=projection.format().affine_bias().map(|source|{
            let metadata=constructor.clone_parameter(source,Some(projection.weight()))?;
            let value=constructor.array(names[2].expect("declared grouped affine bias"),&scale_shape,Dtype::Float32,None)?;
            Ok::<_,ComputeError>(Parameter::new(metadata,MlxTensor::from_array(value)))
        }).transpose()?;
        let bias=projection.bias().map(|source|{
            let metadata=constructor.clone_parameter(source,None)?;
            let value=constructor.array(names[3].expect("declared grouped bias"),&[groups,rows],Dtype::Float32,None)?;
            Ok::<_,ComputeError>(Parameter::new(metadata,MlxTensor::from_array(value)))
        }).transpose()?;
        let construction_funding=constructor.finish()?;
        Ok(MlxGroupedLinear {
            spec,
            weight,
            scale,
            affine_bias,
            bias,
            construction_funding,
        })
    }
}

impl GroupedLinearOperator<MlxTensor> for MlxGroupedLinear {
    fn spec(&self) -> &GroupedLinearSpec {
        &self.spec
    }

    fn forward_grouped(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Linear,
            context,
        )?;
        let transport = original::Transport::new(matches!(
            self.spec.projection().format().encoding(),
            LinearFormat::E4M3BlockFp8(_)
        ));
        let input = input.as_array();
        let ids = selections.group_indices().as_array();
        if input.ndim() != 2
            || input.dim(1) != self.spec.input_dimensions()
            || ids.ndim() != 2
            || ids.dim(0) != input.dim(0)
            || selections.coefficients().shape() != ids.shape()
        {
            return Err(
                transport.invalid("selected grouped-linear input or routing geometry mismatch")
            );
        }
        let ids = match transport.compute(common::tensor::original_group_indices(
            ids,
            self.spec.group_count(),
            context,
        ))? {
            Some(safe) => safe,
            None => transport.compute(common::tensor::validate_token_domain(
                ids,
                self.spec.group_count(),
                None,
                context,
            ))?,
        };
        let plan = transport.compute(common::grouping::group_by_id(&ids, context))?;
        let rows =
            transport.compute(common::grouping::gather_grouped_rows(input, &plan, context))?;
        let weight = self.weight.as_ref().as_array();
        let scale = self.scale.as_ref().map(|p| p.as_ref().as_array());
        let encoding = self.spec.projection().format().encoding();
        let mut projected = match encoding {
            LinearFormat::Dense => transport.compute(common::grouping::grouped_matmul(
                &rows,
                transport.compute(weight.swap_axes(-1, -2, context))?,
                &plan.sorted_group_ids,
                true,
                context,
            ))?,
            LinearFormat::E4M3BlockFp8(_) => transport.compute(common::fp8::grouped_linear(
                &rows,
                weight,
                scale.expect("validated FP8 scale"),
                &plan.sorted_group_ids,
                context,
            ))?,
            LinearFormat::GgufIQuant { .. } => {
                let quantization = encoding.weight_quantization().expect("GGUF format");
                let (ty, endian) = quantization.gguf_iquant().expect("GGUF type");
                transport.compute(
                    common::native_quantization::native_grouped_linear_from_array(
                        &rows,
                        weight,
                        &[
                            self.spec.group_count(),
                            self.spec.output_dimensions(),
                            self.spec.input_dimensions(),
                        ],
                        ty,
                        endian,
                        &plan.sorted_group_ids,
                        context,
                    ),
                )?
            }
            LinearFormat::Affine(_) | LinearFormat::MxFp4 => {
                transport.compute(common::grouped::packed_grouped_linear(
                    &rows,
                    weight,
                    scale.expect("validated packed scale"),
                    self.affine_bias.as_ref().map(|p| p.as_ref().as_array()),
                    &plan.sorted_group_ids,
                    encoding.weight_quantization().expect("packed encoding"),
                    context,
                ))?
            }
        };
        if let Some(bias) = &self.bias {
            projected =
                transport.compute(projected.add(
                    transport.compute(bias.as_ref().as_array().take_axis(
                        &plan.sorted_group_ids,
                        0,
                        context,
                    ))?,
                    context,
                ))?;
        }
        if self.spec.activation() == GroupedLinearActivation::Silu {
            projected = transport.compute(common::layers::silu(projected, context))?;
        }
        transport.tensor(common::grouped::weighted_group_sum(
            projected,
            selections.coefficients().as_array(),
            &plan,
            input.dim(0),
            self.spec.reduction(),
            context,
        ))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "explicit native promoted grouped-linear conformance"]
    fn compact_packed_linear_accepts_logical_float_weights_and_rejects_partial_updates() {
        let stream = crate::test_stream();
        let spec = eredu_nn::GroupedProjectionSpec::new(
            ParameterSpec::trainable("bank.weight").unwrap(),
            None,
            LinearFormatSpec::scaled(
                LinearFormat::MxFp4,
                ParameterSpec::trainable("bank.scales").unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let mut bank = MlxNeuralBackend::grouped_linear_bank(
            GroupedLinearSpec::new(2, 64, 32, GroupedLinearActivation::Identity, spec).unwrap(),
            stream,
        )
        .unwrap();
        let values = (0..2 * 32 * 64)
            .map(|i| ((i * 7 % 31) as f32 - 15.) / 8.)
            .collect::<Vec<_>>();
        let input_values = (0..64)
            .map(|i| ((i % 9) as f32 - 4.) / 4.)
            .collect::<Vec<_>>();
        let input = MlxTensor::from_array(Array::from_slice(&input_values, &[1, 64]));
        let coefficients = MlxTensor::from_array(Array::from_slice(&[0.25f32, 0.75], &[1, 2]));
        let routes = GroupSelection::new(
            MlxTensor::from_array(Array::from_slice(&[1i32, 0], &[1, 2])),
            coefficients.clone(),
            coefficients,
        );
        let expected = (0..32)
            .map(|row| {
                [1usize, 0]
                    .into_iter()
                    .zip([0.25, 0.75])
                    .map(|(expert, coefficient)| {
                        (0..64)
                            .map(|col| values[(expert * 32 + row) * 64 + col] * input_values[col])
                            .sum::<f32>()
                            * coefficient
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let bindings = BTreeMap::from([
                (
                    "weight".into(),
                    Array::from_slice(&values, &[2, 32, 64])
                        .as_dtype(dtype, stream)
                        .unwrap(),
                ),
                (
                    "scales".into(),
                    bank.scale.as_ref().unwrap().as_ref().as_array().clone(),
                ),
            ]);
            bank.bind_local_parameters(bindings.clone()).unwrap();
            let mut invalid = bindings.clone();
            invalid.insert(
                "weight".into(),
                safemlx::ops::zeros_dtype(&[2, 32, 8], dtype, stream).unwrap(),
            );
            assert!(bank.bind_local_parameters(invalid).is_err());
            // A valid changed weight must not publish before a later malformed
            // companion is rejected. The scalar reference below stays unchanged.
            let mut invalid = bindings;
            invalid.insert(
                "weight".into(),
                safemlx::ops::zeros_dtype(&[2, 32, 64], dtype, stream).unwrap(),
            );
            invalid.insert(
                "scales".into(),
                safemlx::ops::zeros_dtype(&[1], Dtype::Uint8, stream).unwrap(),
            );
            assert!(bank.bind_local_parameters(invalid).is_err());
            let output = bank.forward_grouped(&input, &routes, stream).unwrap();
            let actual = output.as_array().evaluated().unwrap();
            assert_eq!(actual.as_slice::<f32>(), expected);
        }
    }

    #[test]
    #[ignore = "explicit native grouped-linear conformance"]
    fn selected_linear_applies_activation_before_mixing_and_partitions_output() {
        let stream = crate::test_stream();
        let spec = eredu_nn::GroupedProjectionSpec::new(
            ParameterSpec::trainable("bank.weight").unwrap(),
            None,
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        )
        .unwrap();
        let spec = GroupedLinearSpec::new(3, 2, 4, GroupedLinearActivation::Silu, spec).unwrap();
        let values: Vec<f32> = (0..24).map(|i| ((i * 7 % 19) as f32 - 9.0) / 5.0).collect();
        let input_values = [1.5f32, -0.7, 0.25, 2.0];
        let ids = [2i32, 0, 1, 2];
        let weights = [0.3f32, 0.7, 0.6, 0.4];
        let input = MlxTensor::from_array(Array::from_slice(&input_values, &[2, 2]));
        let weights_tensor = MlxTensor::from_array(Array::from_slice(&weights, &[2, 2]));
        let routes = GroupSelection::new(
            MlxTensor::from_array(Array::from_slice(&ids, &[2, 2])),
            weights_tensor.clone(),
            weights_tensor,
        );
        let mut expected = vec![0.0f32; 8];
        for token in 0..2 {
            for slot in 0..2 {
                for out in 0..4 {
                    let base = ids[token * 2 + slot] as usize * 8 + out * 2;
                    let x = input_values[token * 2] * values[base]
                        + input_values[token * 2 + 1] * values[base + 1];
                    expected[token * 4 + out] += weights[token * 2 + slot] * x / (1.0 + (-x).exp());
                }
            }
        }
        for range in [0..4, 0..2, 2..4] {
            let mut bank = MlxNeuralBackend::grouped_linear_bank(
                spec.clone().partition_output(range.clone()).unwrap(),
                stream,
            )
            .unwrap();
            let local = (0..3)
                .flat_map(|group| {
                    range.clone().flat_map({
                        let values = &values;
                        move |out| {
                            values[group * 8 + out as usize * 2..group * 8 + out as usize * 2 + 2]
                                .iter()
                                .copied()
                        }
                    })
                })
                .collect::<Vec<_>>();
            bank.weight.replace(MlxTensor::from_array(Array::from_slice(
                &local,
                &[3, range.end - range.start, 2],
            )));
            let output = bank.forward_grouped(&input, &routes, stream).unwrap();
            let actual = output.as_array().evaluated().unwrap();
            for (i, x) in actual.as_slice::<f32>().iter().enumerate() {
                let width = (range.end - range.start) as usize;
                let expected = expected[(i / width) * 4 + range.start as usize + i % width];
                assert!((x - expected).abs() < 1e-5, "{range:?} {x} != {expected}");
            }
        }
    }
}
