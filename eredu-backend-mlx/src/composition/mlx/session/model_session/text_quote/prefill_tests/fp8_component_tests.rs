use super::operation_component_tests::exercise_component;
use super::*;
use crate::{
    MlxTensor,
    backend::nn::{
        shared::MlxNeuralBackend,
        workspace::{MlxMetalWorkspaceMechanisms, OriginalComponentTestPlan},
    },
};
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
use eredu_nn::{
    LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend, ParameterMetadata, ParameterSpec,
    ParameterVisitorMut, Parameterized, Tensor,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceTensor},
};
use safemlx::Array;

fn spec(encoding: BlockFp8ScaleEncoding) -> LinearSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    LinearSpec {
        input: 259,
        output: 137,
        weight: parameter("matrix.weight"),
        bias: Some(parameter("matrix.bias")),
        format: LinearFormatSpec::scaled(
            LinearFormat::E4M3BlockFp8(BlockFp8Format::new(128, 128, encoding).unwrap()),
            parameter("matrix.scales"),
        )
        .unwrap(),
    }
}
struct Bind<'a> {
    weight: &'a MlxTensor,
    scales: &'a MlxTensor,
    bias: &'a MlxTensor,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
        *value = match metadata.id.as_str() {
            "matrix.weight" => self.weight,
            "matrix.scales" => self.scales,
            "matrix.bias" => self.bias,
            name => panic!("unexpected FP8 binding {name}"),
        }
        .clone();
    }
}

fn exercise_projection(observed: bool) {
    let mut fixture = OriginalOperationFixture::prepare(true);
    let stream = fixture.stream.clone();
    for rows in [3, 9] {
        for encoding in [
            BlockFp8ScaleEncoding::FloatingPoint,
            BlockFp8ScaleEncoding::Ue8m0,
        ] {
            // Nonzero input/weight values, tail blocks on both matrix axes,
            // and nonuniform scales catch source and runtime-width mistakes.
            let values: Vec<f32> = (0..137 * 259)
                .map(|i| ((i * 7 % 31) as f32 - 15.0) / 16.0)
                .collect();
            let weight = MlxTensor::from_array(
                Array::from_slice(&values, &[137, 259])
                    .to_fp8(&stream)
                    .unwrap(),
            );
            let scales = MlxTensor::from_array(match encoding {
                BlockFp8ScaleEncoding::FloatingPoint => {
                    Array::from_slice(&[0.5f32, 1.0, 2.0, 1.0, 2.0, 0.5], &[2, 3])
                }
                BlockFp8ScaleEncoding::Ue8m0 => {
                    Array::from_slice(&[126u8, 127, 128, 127, 128, 126], &[2, 3])
                }
            });
            let values: Vec<f32> = (0..137).map(|i| (i % 7) as f32 / 32.0 - 0.1).collect();
            let bias = MlxTensor::from_array(Array::from_slice(&values, &[137]));
            let values: Vec<f32> = (0..rows * 259)
                .map(|i| ((i * 3 % 23) as f32 - 11.0) / 16.0)
                .collect();
            let input = MlxTensor::from_array(Array::from_slice(&values, &[1, rows, 259]));
            let mut projection = MlxNeuralBackend::linear(spec(encoding), &stream).unwrap();
            projection.visit_parameters_mut(&mut Bind {
                weight: &weight,
                scales: &scales,
                bias: &bias,
            });
            let expected = if observed {
                observed_projection(&mut projection, &input, &stream)
            } else {
                projection.forward(&input, &stream)
            }
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
            let context =
                WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
            let metadata = WorkspaceTensor::existing(
                context
                    .layout(&[1, rows, 259], WorkspaceDtype::Float32)
                    .unwrap(),
                &context,
            )
            .unwrap();
            let mut planned = WorkspaceBackend::linear(spec(encoding), &context).unwrap();
            context.begin_span();
            let output = if observed {
                observed_projection(&mut planned, &metadata, &context)
            } else {
                planned.forward(&metadata, &context)
            }
            .unwrap();
            let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
            exercise_component(
                &mut fixture,
                &plan,
                &[
                    input.as_array(),
                    weight.as_array(),
                    scales.as_array(),
                    bias.as_array(),
                ],
                &expected,
                || {
                    if observed {
                        observed_projection(&mut projection, &input, &stream)
                            .unwrap()
                            .into()
                    } else {
                        projection.forward(&input, &stream).unwrap().into()
                    }
                },
            );
        }
    }
    drop(stream);
    fixture.finish();
}

struct Capture<T>(Option<T>);
impl<T: Tensor> eredu_nn::ProjectionInputObserver<T> for Capture<T> {
    fn observe(&mut self, input: &T) -> Result<(), eredu_nn::Error> {
        self.0 = Some(input.clone());
        Ok(())
    }
    fn observe_generated(
        &mut self,
        _: &T,
        source: &eredu_nn::GeneratedTensorSource,
        generate: &mut dyn FnMut() -> Result<T, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        assert_eq!(source.element_type, Some(eredu_nn::TensorElementType::F32));
        self.0 = Some(generate()?);
        Ok(())
    }
}
fn observed_projection<T: Tensor, L: LinearOperator<T>>(
    module: &mut L,
    input: &T,
    context: &T::Context,
) -> Result<T, eredu_nn::Error> {
    let mut capture = Capture(None);
    let output = module.forward_with_input_observer(input, context, Some(&mut capture))?;
    let captured = capture
        .0
        .take()
        .expect("selected effective-input reconstruction");
    assert_eq!(captured.shape(), input.shape());
    // One output owns both the real projection and the reconstructed effective
    // input. The same concatenation appears in its ordinary and workspace trace.
    T::concatenate(
        &[
            output.reshape(&[-1], context)?,
            captured.reshape(&[-1], context)?,
        ],
        0,
        context,
    )
}

#[test]
fn original_block_fp8_projection_matches_both_shared_kernels_and_scale_encodings() {
    exercise_projection(false);
}
#[test]
fn original_observed_block_fp8_projection_and_effective_input_match_ordinary() {
    exercise_projection(true);
}

struct RefuseGenerated;
impl<T: Tensor> eredu_nn::ProjectionInputObserver<T> for RefuseGenerated {
    fn observe(&mut self, _: &T) -> Result<(), eredu_nn::Error> {
        panic!("the actual packed FP8 producer must offer generated input")
    }
    fn observe_generated(
        &mut self,
        _: &T,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<T, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Text.into())
    }
}
#[test]
fn original_observed_fp8_refusal_keeps_typed_cause_and_retires_partial_prepare() {
    use std::error::Error as _;
    let mut fixture = OriginalOperationFixture::prepare(true);
    let stream = fixture.stream.clone();
    let encoding = BlockFp8ScaleEncoding::Ue8m0;
    let weight = MlxTensor::from_array(Array::from_slice(&vec![0x38u8; 137 * 259], &[137, 259]));
    let scales = MlxTensor::from_array(Array::from_slice(&[127u8; 6], &[2, 3]));
    let bias = MlxTensor::from_array(Array::from_slice(&[0.125f32; 137], &[137]));
    let values: Vec<f32> = (0..259)
        .map(|i| ((i * 3 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 1, 259]));
    let mut module = MlxNeuralBackend::linear(spec(encoding), &stream).unwrap();
    module.visit_parameters_mut(&mut Bind {
        weight: &weight,
        scales: &scales,
        bias: &bias,
    });
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let metadata = WorkspaceTensor::existing(
        context
            .layout(&[1, 1, 259], WorkspaceDtype::Float32)
            .unwrap(),
        &context,
    )
    .unwrap();
    let mut planned = WorkspaceBackend::linear(spec(encoding), &context).unwrap();
    context.begin_span();
    assert!(
        planned
            .forward_with_input_observer(&metadata, &context, Some(&mut RefuseGenerated))
            .is_err()
    );
    // The rejected projection contains prepare only. A separate, explicitly
    // traced identity consumer gives the fixture an owned completion root;
    // it is never published as a successful projection or production fallback.
    let output = metadata.multiply_scalar(1.0, &context).unwrap();
    let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
    exercise_component(
        &mut fixture,
        &plan,
        &[
            input.as_array(),
            weight.as_array(),
            scales.as_array(),
            bias.as_array(),
        ],
        &values,
        || {
            let failure = module
                .forward_with_input_observer(&input, &stream, Some(&mut RefuseGenerated))
                .unwrap_err();
            let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&failure);
            let mut preserved = false;
            while let Some(current) = cause {
                preserved |= current.downcast_ref::<eredu_nn::workspace::WorkspaceMetadataError>()
                    == Some(&eredu_nn::workspace::WorkspaceMetadataError::Text);
                cause = current.source();
            }
            assert!(
                preserved,
                "the original callback error must survive both erased boundaries"
            );
            assert!(failure.source().is_some());
            drop(failure);
            input.multiply_scalar(1.0, &stream).unwrap().into()
        },
    );
    drop(stream);
    fixture.finish();
}

#[test]
fn original_grouped_block_fp8_projection_matches_ordinary_with_derived_receipt() {
    use eredu_nn::{
        GroupSelection, GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec,
        GroupedNeuralBackend, GroupedProjectionSpec,
    };
    use std::collections::BTreeMap;
    let mut fixture = OriginalOperationFixture::prepare(true);
    let stream = fixture.stream.clone();
    // Four and ten actual routed rows exercise tiled and scalar kernels.
    // Both dimensions have tail blocks; group/row-dependent scales catch
    // incorrect SCALE_OUT, ROW_WIDTH and grouped source addressing.
    for tokens in [2, 5] {
        for encoding in [
            BlockFp8ScaleEncoding::FloatingPoint,
            BlockFp8ScaleEncoding::Ue8m0,
        ] {
            let parameter = |name| ParameterSpec::trainable(name).unwrap();
            let spec = GroupedLinearSpec::new(
                3,
                259,
                137,
                if tokens == 2 {
                    GroupedLinearActivation::Identity
                } else {
                    GroupedLinearActivation::Silu
                },
                GroupedProjectionSpec::new(
                    parameter("bank.weight"),
                    Some(parameter("bank.bias")),
                    LinearFormatSpec::scaled(
                        LinearFormat::E4M3BlockFp8(
                            BlockFp8Format::new(128, 128, encoding).unwrap(),
                        ),
                        parameter("bank.scales"),
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
            let weight: Vec<f32> = (0..3 * 137 * 259)
                .map(|i| ((i * 7 + i / 259 * 3) % 31) as f32 / 64.0 - 0.2)
                .collect();
            let weight = Array::from_slice(&weight, &[3, 137, 259])
                .to_fp8(&stream)
                .unwrap();
            let codes: Vec<u8> = (0..18).map(|i| 125 + (i % 3) as u8).collect();
            let scales = match encoding {
                BlockFp8ScaleEncoding::FloatingPoint => Array::from_slice(
                    &codes
                        .iter()
                        .map(|&code| 2.0f32.powi(i32::from(code) - 127))
                        .collect::<Vec<_>>(),
                    &[3, 2, 3],
                ),
                BlockFp8ScaleEncoding::Ue8m0 => Array::from_slice(&codes, &[3, 2, 3]),
            };
            let bias: Vec<f32> = (0..3 * 137)
                .map(|i| (i % 13) as f32 / 64.0 - 0.05)
                .collect();
            let bias = Array::from_slice(&bias, &[3, 137]);
            let mut bank = MlxNeuralBackend::grouped_linear_bank(spec.clone(), &stream).unwrap();
            bank.bind_local_parameters(BTreeMap::from([
                ("weight".to_owned(), weight.clone()),
                ("scales".to_owned(), scales.clone()),
                ("bias".to_owned(), bias.clone()),
            ]))
            .unwrap();
            let values: Vec<f32> = (0..tokens * 259)
                .map(|i| ((i * 3 % 23) as f32 - 11.0) / 16.0)
                .collect();
            let input = MlxTensor::from_array(Array::from_slice(&values, &[tokens, 259]));
            let ids: Vec<i32> = (0..tokens * 2).map(|i| (i * 2 + 2) % 3).collect();
            let ids = MlxTensor::from_array(Array::from_slice(&ids, &[tokens, 2]));
            let coefficients: Vec<f32> = (0..tokens * 2)
                .map(|i| if i % 2 == 0 { 0.3 } else { 0.7 })
                .collect();
            let coefficients =
                MlxTensor::from_array(Array::from_slice(&coefficients, &[tokens, 2]));
            let selection =
                GroupSelection::new(ids.clone(), coefficients.clone(), coefficients.clone());
            let expected = bank
                .forward_grouped(&input, &selection, &stream)
                .unwrap()
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            let context =
                WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
            let source = |shape: &[i32], dtype| {
                WorkspaceTensor::existing(context.layout(shape, dtype).unwrap(), &context).unwrap()
            };
            let metadata = source(&[tokens, 259], WorkspaceDtype::Float32);
            let metadata_ids = source(&[tokens, 2], WorkspaceDtype::Int32);
            let metadata_coefficients = source(&[tokens, 2], WorkspaceDtype::Float32);
            let metadata_selection = GroupSelection::new(
                metadata_ids,
                metadata_coefficients.clone(),
                metadata_coefficients,
            );
            let mut planned = WorkspaceBackend::grouped_linear_bank(spec, &context).unwrap();
            context.begin_span();
            let output = planned
                .forward_grouped(&metadata, &metadata_selection, &context)
                .unwrap();
            let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
            assert_eq!(plan.completion.validation_roots, 1);
            exercise_component(
                &mut fixture,
                &plan,
                &[
                    input.as_array(),
                    ids.as_array(),
                    coefficients.as_array(),
                    &weight,
                    &scales,
                    &bias,
                ],
                &expected,
                || {
                    bank.forward_grouped(&input, &selection, &stream)
                        .unwrap()
                        .into()
                },
            );
        }
    }
    drop(stream);
    fixture.finish();
}


#[test]
fn original_packed_fp8_independent_rows_and_chunk_tail_match_ordinary() {
    use eredu_nn::{
        GatedProductGroupLayout, GatedProductPolicy, GroupSelection, GroupedGatedProductOperator,
        GroupedGatedProductSpec, GroupedNeuralBackend, GroupedProjectionSpec, LinearRowLayout,
    };
    use std::collections::BTreeMap;
    let mut fixture = OriginalOperationFixture::prepare(true);
    let stream = fixture.stream.clone();
    // The 137-row components each own two scale rows. Their packed bank needs
    // four rows, not ceil(274 / 128)=3. Sixty-five tokens take two full chunks
    // and a one-token tail; the smaller case takes the tiled worker directly.
    for (tokens, encoding) in [
        (2, BlockFp8ScaleEncoding::FloatingPoint),
        (65, BlockFp8ScaleEncoding::Ue8m0),
    ] {
        let projection = |name: &str, layout| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(format!("{name}.weight")).unwrap(),
                Some(ParameterSpec::trainable(format!("{name}.bias")).unwrap()),
                LinearFormatSpec::scaled(
                    LinearFormat::E4M3BlockFp8(BlockFp8Format::new(128, 128, encoding).unwrap()),
                    ParameterSpec::trainable(format!("{name}.scales")).unwrap(),
                )
                .unwrap()
                .with_row_layout(layout)
                .unwrap(),
            )
            .unwrap()
        };
        let spec = GroupedGatedProductSpec::new(
            2,
            130,
            137,
            130,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("read", LinearRowLayout::equal_partitions(2).unwrap()),
                down: projection("write", LinearRowLayout::Contiguous),
            },
        )
        .unwrap();
        let weight = |rows: i32, columns: i32| {
            let values: Vec<f32> = (0..2 * rows * columns)
                .map(|i| ((i * 7 + i / columns * 3) % 31) as f32 / 128.0 - 0.1)
                .collect();
            Array::from_slice(&values, &[2, rows, columns])
                .to_fp8(&stream)
                .unwrap()
        };
        let scales = |rows: i32| {
            let codes: Vec<u8> = (0..2 * rows * 2).map(|i| 125 + (i % 3) as u8).collect();
            match encoding {
                BlockFp8ScaleEncoding::FloatingPoint => Array::from_slice(
                    &codes
                        .iter()
                        .map(|&code| 2.0f32.powi(i32::from(code) - 127))
                        .collect::<Vec<_>>(),
                    &[2, rows, 2],
                ),
                BlockFp8ScaleEncoding::Ue8m0 => Array::from_slice(&codes, &[2, rows, 2]),
            }
        };
        let bias = |rows: i32| {
            let values: Vec<f32> = (0..2 * rows)
                .map(|i| (i % 13) as f32 / 64.0 - 0.05)
                .collect();
            Array::from_slice(&values, &[2, rows])
        };
        let gate_up = weight(274, 130);
        let down = weight(130, 137);
        let gate_up_scales = scales(4);
        let down_scales = scales(2);
        let gate_up_bias = bias(274);
        let down_bias = bias(130);
        let mut bank = MlxNeuralBackend::grouped_gated_product(spec.clone(), &stream).unwrap();
        bank.bind_local_parameters(BTreeMap::from([
            ("gate_up_proj".to_owned(), gate_up.clone()),
            ("gate_up_proj_scales".to_owned(), gate_up_scales.clone()),
            ("gate_up_proj_bias".to_owned(), gate_up_bias.clone()),
            ("down_proj".to_owned(), down.clone()),
            ("down_proj_scales".to_owned(), down_scales.clone()),
            ("down_proj_bias".to_owned(), down_bias.clone()),
        ]))
        .unwrap();
        let values: Vec<f32> = (0..tokens * 130)
            .map(|i| ((i * 3 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let input = MlxTensor::from_array(Array::from_slice(&values, &[tokens, 130]));
        let ids: Vec<i32> = (0..tokens * 2).map(|i| (i / 2 + i) % 2).collect();
        let ids = MlxTensor::from_array(Array::from_slice(&ids, &[tokens, 2]));
        let coefficients: Vec<f32> = (0..tokens * 2)
            .map(|i| if i % 2 == 0 { 0.3 } else { 0.7 })
            .collect();
        let coefficients = MlxTensor::from_array(Array::from_slice(&coefficients, &[tokens, 2]));
        let selection =
            GroupSelection::new(ids.clone(), coefficients.clone(), coefficients.clone());
        let expected = bank
            .forward_grouped(&input, &selection, &stream)
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let source = |shape: &[i32], dtype| {
            WorkspaceTensor::existing(context.layout(shape, dtype).unwrap(), &context).unwrap()
        };
        let metadata = source(&[tokens, 130], WorkspaceDtype::Float32);
        let metadata_ids = source(&[tokens, 2], WorkspaceDtype::Int32);
        let metadata_coefficients = source(&[tokens, 2], WorkspaceDtype::Float32);
        let metadata_selection = GroupSelection::new(
            metadata_ids,
            metadata_coefficients.clone(),
            metadata_coefficients,
        );
        let mut planned = WorkspaceBackend::grouped_gated_product(spec, &context).unwrap();
        context.begin_span();
        let output = planned
            .forward_grouped(&metadata, &metadata_selection, &context)
            .unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        assert_eq!(plan.completion.validation_roots, 1);
        exercise_component(
            &mut fixture,
            &plan,
            &[
                input.as_array(),
                ids.as_array(),
                coefficients.as_array(),
                &gate_up,
                &gate_up_scales,
                &gate_up_bias,
                &down,
                &down_scales,
                &down_bias,
            ],
            &expected,
            || {
                bank.forward_grouped(&input, &selection, &stream)
                    .unwrap()
                    .into()
            },
        );
    }
    drop(stream);
    fixture.finish();
}


struct ProbeUnits<'a, T: Tensor> {
    context: &'a T::Context,
    refuse_shape: bool,
    captured: Option<T>,
}
impl<T: Tensor> eredu_nn::GroupedUnitObserver<T> for ProbeUnits<'_, T> {
    fn observe(
        &mut self,
        batch: &eredu_nn::GroupedUnitBatch<'_, T>,
    ) -> Result<(), eredu_nn::Error> {
        assert_eq!(batch.values.shape().len(), 2);
        assert_eq!(batch.group_indices.shape(), &[batch.values.shape()[0]]);
        assert_eq!(batch.selection_indices.shape(), batch.group_indices.shape());
        assert_eq!(batch.token_indices.shape(), batch.group_indices.shape());
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &eredu_nn::GroupedUnitBatch<'_, T>,
    ) -> Result<Option<T>, eredu_nn::Error> {
        if self.refuse_shape {
            batch.values.reshape(&[-1], self.context).map(Some)
        } else {
            batch.values.multiply_scalar(0.75, self.context).map(Some)
        }
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_nn::GroupedUnitBatch<'_, T>,
    ) -> Result<(), eredu_nn::Error> {
        self.captured = Some(batch.values.clone());
        Ok(())
    }
}
fn observed_grouped<T: Tensor, B: eredu_nn::GroupedGatedProductOperator<T>>(
    bank: &mut B,
    input: &T,
    selection: &eredu_nn::GroupSelection<T>,
    context: &T::Context,
    refuse_shape: bool,
) -> Result<T, eredu_nn::Error> {
    let mut observer = ProbeUnits {
        context,
        refuse_shape,
        captured: None,
    };
    let result =
        bank.forward_grouped_with_unit_observer(input, selection, context, Some(&mut observer));
    if refuse_shape {
        let failure = match result {
            Ok(_) => panic!("invalid grouped-unit shape must refuse"),
            Err(failure) => failure,
        };
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&failure);
        let mut shape = false;
        while let Some(cause) = source {
            if let Some(eredu_nn::GroupedUnitError::ReplacementShape { expected, actual }) =
                cause.downcast_ref()
            {
                assert_eq!(expected.len(), 2);
                assert_eq!(actual, &[expected[0] * expected[1]]);
                shape = true;
            }
            source = cause.source();
        }
        assert!(
            shape,
            "the exact typed shape error must survive the native callback"
        );
        drop(failure);
        // A separately traced identity root exercises safe retirement of the
        // failed prefix. It is not a successful grouped-operation fallback.
        return input.multiply_scalar(1.0, context);
    }
    let output = result?;
    let captured = observer.captured.expect("last actual effective chunk");
    T::concatenate(
        &[
            output.reshape(&[-1], context)?,
            captured.reshape(&[-1], context)?,
        ],
        0,
        context,
    )
}

#[test]
fn original_grouped_units_dense_fp8_and_shape_refusal_keep_custody() {
    use eredu_nn::{
        GatedProductGroupLayout, GatedProductPolicy, GroupSelection, GroupedGatedProductSpec,
        GroupedNeuralBackend, GroupedProjectionSpec, LinearRowLayout,
    };
    use std::collections::BTreeMap;
    let mut fixture = OriginalOperationFixture::prepare(true);
    let stream = fixture.stream.clone();
    for (fp8, refuse_shape, tokens) in [(false, false, 2), (true, false, 65), (true, true, 2)] {
        eprintln!("grouped Units case: fp8={fp8}, refuse_shape={refuse_shape}, tokens={tokens}");
        let (width, units) = if fp8 { (130, 137) } else { (128, 128) };
        let projection = |name: &str, layout| {
            let format = if fp8 {
                LinearFormatSpec::scaled(
                    LinearFormat::E4M3BlockFp8(
                        BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0).unwrap(),
                    ),
                    ParameterSpec::trainable(format!("{name}.scales")).unwrap(),
                )
                .unwrap()
                .with_row_layout(layout)
                .unwrap()
            } else {
                LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()
            };
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(format!("{name}.weight")).unwrap(),
                Some(ParameterSpec::trainable(format!("{name}.bias")).unwrap()),
                format,
            )
            .unwrap()
        };
        let spec = GroupedGatedProductSpec::new(
            2,
            width,
            units,
            width,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("read", LinearRowLayout::equal_partitions(2).unwrap()),
                down: projection("write", LinearRowLayout::Contiguous),
            },
        )
        .unwrap();
        let weight = |rows: i32, columns: i32| {
            let values: Vec<f32> = (0..2 * rows * columns)
                .map(|i| ((i * 7 + i / columns * 3) % 31) as f32 / 128.0 - 0.1)
                .collect();
            let array = Array::from_slice(&values, &[2, rows, columns]);
            if fp8 {
                array.to_fp8(&stream).unwrap()
            } else {
                array
            }
        };
        let scale = |rows: i32| {
            let codes: Vec<u8> = (0..2 * rows * 2).map(|i| 125 + (i % 3) as u8).collect();
            Array::from_slice(&codes, &[2, rows, 2])
        };
        let bias = |rows: i32| {
            let values: Vec<f32> = (0..2 * rows)
                .map(|i| (i % 13) as f32 / 64.0 - 0.05)
                .collect();
            Array::from_slice(&values, &[2, rows])
        };
        let gate_up = weight(2 * units, width);
        let down = weight(width, units);
        // The deliberate callback refusal below never evaluates its grouped
        // prefix. Materialize both source conversion DAGs independently first;
        // the component helper then observes/detaches their completed ordinary
        // events before entering the original role, exactly as for a successful
        // reference run. No lazy ConvertFP8 becomes an original source leaf.
        for weight in [&gate_up, &down] {
            weight.evaluated().unwrap();
        }
        let gate_up_scales = fp8.then(|| scale(4));
        let down_scales = fp8.then(|| scale(2));
        let gate_up_bias = bias(2 * units);
        let down_bias = bias(width);
        let mut bindings = BTreeMap::from([
            ("gate_up_proj".into(), gate_up.clone()),
            ("down_proj".into(), down.clone()),
            ("gate_up_proj_bias".into(), gate_up_bias.clone()),
            ("down_proj_bias".into(), down_bias.clone()),
        ]);
        if let (Some(first), Some(second)) = (&gate_up_scales, &down_scales) {
            bindings.insert("gate_up_proj_scales".into(), first.clone());
            bindings.insert("down_proj_scales".into(), second.clone());
        }
        let mut bank = MlxNeuralBackend::grouped_gated_product(spec.clone(), &stream).unwrap();
        bank.bind_local_parameters(bindings).unwrap();
        let values: Vec<f32> = (0..tokens * width)
            .map(|i| ((i * 3 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let input = MlxTensor::from_array(Array::from_slice(&values, &[tokens, width]));
        let indices: Vec<i32> = (0..tokens * 2).map(|i| (i / 2 + i) % 2).collect();
        let ids = MlxTensor::from_array(Array::from_slice(&indices, &[tokens, 2]));
        let values: Vec<f32> = (0..tokens * 2)
            .map(|i| if i % 2 == 0 { 0.3 } else { 0.7 })
            .collect();
        let coefficients = MlxTensor::from_array(Array::from_slice(&values, &[tokens, 2]));
        let selection =
            GroupSelection::new(ids.clone(), coefficients.clone(), coefficients.clone());
        let expected = observed_grouped(&mut bank, &input, &selection, &stream, refuse_shape)
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let source = |shape: &[i32], dtype| {
            // The fixture lends the actual immutable F32 source representation;
            // integer layouts discard this optional floating descriptor.
            let layout = context
                .layout(shape, dtype)
                .unwrap()
                .with_representation(Some(eredu_nn::workspace::WorkspaceRepresentation::new(
                    eredu_nn::workspace::WorkspaceFloatingType::Float32,
                    true,
                )));
            WorkspaceTensor::existing(layout, &context).unwrap()
        };
        let metadata = source(&[tokens, width], WorkspaceDtype::Float32);
        let metadata_ids = source(&[tokens, 2], WorkspaceDtype::Int32);
        let metadata_coefficients = source(&[tokens, 2], WorkspaceDtype::Float32);
        let metadata_selection = GroupSelection::new(
            metadata_ids,
            metadata_coefficients.clone(),
            metadata_coefficients,
        );
        let mut planned = WorkspaceBackend::grouped_gated_product(spec, &context).unwrap();
        context.begin_span();
        let output = observed_grouped(
            &mut planned,
            &metadata,
            &metadata_selection,
            &context,
            refuse_shape,
        )
        .unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        assert_eq!(plan.completion.validation_roots, 1);
        assert_eq!(plan.completion.grouped_outputs.unit_observers, 1);
        let leaves: Vec<&Array> = [
            Some(input.as_array()),
            Some(ids.as_array()),
            Some(coefficients.as_array()),
            Some(&gate_up),
            gate_up_scales.as_ref(),
            Some(&gate_up_bias),
            Some(&down),
            down_scales.as_ref(),
            Some(&down_bias),
        ]
        .into_iter()
        .flatten()
        .collect();
        exercise_component(&mut fixture, &plan, &leaves, &expected, || {
            observed_grouped(&mut bank, &input, &selection, &stream, refuse_shape)
                .unwrap()
                .into()
        });
    }
    drop(stream);
    fixture.finish();
}
