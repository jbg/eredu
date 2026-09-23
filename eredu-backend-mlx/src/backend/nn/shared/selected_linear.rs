use super::*;
use eredu_nn::{GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec, Parameter};

/// MLX indexed affine projections with activation before weighted reduction.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "MlxTensor")]
pub struct MlxGroupedLinear {
    #[parameter(skip)]
    spec: GroupedLinearSpec,
    weight: Parameter<MlxTensor>,
    scale: Option<Parameter<MlxTensor>>,
    affine_bias: Option<Parameter<MlxTensor>>,
    bias: Option<Parameter<MlxTensor>>,
}

impl MlxGroupedLinear {
    fn local_bindings(&self) -> [Option<String>; 4] {
        let projection = self.spec.projection();
        [
            Some(projection.weight()),
            projection.format().scale(),
            projection.format().affine_bias(),
            projection.bias(),
        ]
        .map(|spec| {
            spec.map(|spec| {
                spec.id
                    .as_str()
                    .rsplit('.')
                    .next()
                    .expect("validated parameter identity")
                    .to_owned()
            })
        })
    }

    pub(crate) fn local_parameter_names(&self) -> Vec<String> {
        self.local_bindings().into_iter().flatten().collect()
    }

    pub(crate) fn bind_local_parameters(
        &mut self,
        mut bindings: BTreeMap<String, Array>,
    ) -> Result<(), ComputeError> {
        let names = self.local_bindings();
        let expected = names.iter().flatten().cloned().collect::<BTreeSet<_>>();
        if bindings.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(ComputeError::backend(
                "compact linear bank bindings differ from its specification",
            ));
        }
        // Validate every shape before changing any parameter slot.
        let floating_shape = [
            self.spec.group_count(),
            self.spec.output_dimensions(),
            self.spec.input_dimensions(),
        ];
        for (name, parameter) in names.iter().zip([
            Some(&self.weight),
            self.scale.as_ref(),
            self.affine_bias.as_ref(),
            self.bias.as_ref(),
        ]) {
            if let (Some(name), Some(parameter)) = (name, parameter) {
                super::parameters::validate_compact_binding(
                    name,
                    parameter.as_ref().as_array(),
                    &bindings[name],
                    (Some(name) == names[0].as_ref()).then_some(floating_shape.as_slice()),
                )?;
            }
        }
        for (name, parameter) in names.iter().zip([
            Some(&mut self.weight),
            self.scale.as_mut(),
            self.affine_bias.as_mut(),
            self.bias.as_mut(),
        ]) {
            if let (Some(name), Some(parameter)) = (name, parameter) {
                parameter.replace(MlxTensor::from_array(
                    bindings.remove(name).expect("validated binding"),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn new(spec: GroupedLinearSpec, context: &Stream) -> Result<Self, ComputeError> {
        spec.validate()?;
        let (groups, rows, columns) = (
            spec.group_count(),
            spec.output_dimensions(),
            spec.input_dimensions(),
        );
        let projection = spec.projection();
        let encoding = projection.format().encoding();
        let mut shape = vec![groups, rows, columns];
        let mut scale_shape = shape.clone();
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
        let unloaded = |spec, shape: &[i32], dtype| -> Result<Parameter<MlxTensor>, ComputeError> {
            Ok(Parameter::new(
                spec,
                compute_tensor(safemlx::ops::zeros_dtype(shape, dtype, context))?,
            ))
        };
        let weight = unloaded(projection.weight().clone(), &shape, weight_dtype)?;
        let scale = projection
            .format()
            .scale()
            .map(|p| {
                unloaded(
                    bind_linear_companion(projection.weight(), p.clone()),
                    &scale_shape,
                    scale_dtype,
                )
            })
            .transpose()?;
        let affine_bias = projection
            .format()
            .affine_bias()
            .map(|p| {
                Parameter::unloaded(
                    bind_linear_companion(projection.weight(), p.clone()),
                    &scale_shape,
                    context,
                )
            })
            .transpose()?;
        let bias = projection
            .bias()
            .map(|p| Parameter::unloaded(p.clone(), &[groups, rows], context))
            .transpose()?;
        Ok(MlxGroupedLinear {
            spec,
            weight,
            scale,
            affine_bias,
            bias,
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
        let input = input.as_array();
        let ids = selections.group_indices().as_array();
        if input.ndim() != 2
            || input.dim(1) != self.spec.input_dimensions()
            || ids.ndim() != 2
            || ids.dim(0) != input.dim(0)
            || selections.coefficients().shape() != ids.shape()
        {
            return Err(ComputeError::backend(
                "selected grouped-linear input or routing geometry mismatch",
            ));
        }
        let ids = compute(common::tensor::validate_token_domain(
            ids,
            self.spec.group_count(),
            None,
            context,
        ))?;
        let plan = compute(common::grouping::group_by_id(&ids, context))?;
        let rows = compute(common::grouping::gather_grouped_rows(input, &plan, context))?;
        let weight = self.weight.as_ref().as_array();
        let scale = self.scale.as_ref().map(|p| p.as_ref().as_array());
        let encoding = self.spec.projection().format().encoding();
        let mut projected = match encoding {
            LinearFormat::Dense => compute(common::grouping::grouped_matmul(
                &rows,
                compute(weight.swap_axes(-1, -2, context))?,
                &plan.sorted_group_ids,
                true,
                context,
            ))?,
            LinearFormat::E4M3BlockFp8(_) => compute(common::fp8::grouped_linear(
                &rows,
                weight,
                scale.expect("validated FP8 scale"),
                &plan.sorted_group_ids,
                context,
            ))?,
            LinearFormat::GgufIQuant { .. } => {
                let quantization = encoding.weight_quantization().expect("GGUF format");
                let (ty, endian) = quantization.gguf_iquant().expect("GGUF type");
                compute(
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
                compute(common::grouped::packed_grouped_linear(
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
                compute(projected.add(
                    compute(bias.as_ref().as_array().take_axis(
                        &plan.sorted_group_ids,
                        0,
                        context,
                    ))?,
                    context,
                ))?;
        }
        if self.spec.activation() == GroupedLinearActivation::Silu {
            projected = compute(common::layers::silu(projected, context))?;
        }
        compute_tensor(common::grouped::weighted_group_sum(
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
        let mut expected = [0.0f32; 8];
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
