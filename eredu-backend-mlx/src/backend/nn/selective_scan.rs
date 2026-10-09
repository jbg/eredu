//! Single-step selective recurrence; family equations remain in portable architectures.
use crate::MlxTensor;
use eredu_nn::{Error, SelectiveStateSpaceScanInput, SelectiveStateSpaceScanOutput};
use safemlx::Stream;

pub(crate) fn single_step(
    input: &SelectiveStateSpaceScanInput<'_, MlxTensor>,
    stream: &Stream,
) -> Result<Option<SelectiveStateSpaceScanOutput<MlxTensor>>, Error> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use eredu_nn::Tensor;
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            ops::{sum_axis, zeros},
            DeviceType, Dtype,
        };
        use std::cell::RefCell;
        let eligible = input.values.dim(1) == 1
            && input.time_step_floor.is_finite()
            && input.time_step_floor >= 0.0
            && [
                input.values,
                input.input_state,
                input.output_state,
                input.time_step,
                input.time_step_bias,
                input.transition_log,
                input.skip,
            ]
            .into_iter()
            .chain(input.initial_state)
            .all(|t| t.as_array().dtype() == Dtype::Float32);
        if eligible
            && stream
                .get_device()
                .and_then(|d| d.get_type())
                .map_err(Error::backend_source)?
                == DeviceType::Gpu
        {
            let b = input.values.dim(0);
            let h = input.values.dim(2);
            let d = input.values.dim(3);
            let n = input.input_state.dim(3);
            let state_shape = [b, h, d, n];
            let output_shape = [b, 1, h, d];
            let elements = [b, h, d, n]
                .into_iter()
                .try_fold(1_i32, |a, b| a.checked_mul(b))
                .ok_or_else(|| Error::backend("selective recurrence grid overflow"))?;
            let initial = match input.initial_state {
                Some(t) => t.as_array().clone(),
                None => zeros::<f32>(&state_shape, stream).map_err(Error::backend_source)?,
            };
            let floor = safemlx::Array::from_f32(input.time_step_floor);
            let columns = d
                .checked_mul(n)
                .and_then(|v| v.checked_add(255))
                .map(|v| (v / 256) * 256)
                .ok_or_else(|| Error::backend("selective recurrence tile overflow"))?;
            let groups = b
                .checked_mul(h)
                .ok_or_else(|| Error::backend("selective recurrence group overflow"))?;
            let config = CustomKernelConfig::new()
                .with_template_arg_int("HEADS", h)
                .with_template_arg_int("WIDTH", d)
                .with_template_arg_int("STATE_WIDTH", n)
                .with_template_arg_int("ELEMENTS", elements)
                .with_grid([columns, groups, 1])
                .with_thread_group([256, 1, 1])
                .with_output_arg(state_shape, Dtype::Float32)
                .with_output_arg(state_shape, Dtype::Float32)
                .with_output_arg(output_shape, Dtype::Float32);
            thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
            let mut outputs = KERNEL
                .with(|cell| {
                    if cell.borrow().is_none() {
                        *cell.borrow_mut() = Some(MetalKernel::new(
                            "selective_state_update_f32",
                            [
                                "values",
                                "input_vector",
                                "output_vector",
                                "time_step",
                                "bias",
                                "transition_log",
                                "skip",
                                "initial",
                                "floor",
                            ],
                            ["state", "weighted", "direct"],
                            include_str!("selective_scan.metal"),
                            "#pragma clang fp contract(off)\n",
                            true,
                            false,
                        )?);
                    }
                    cell.borrow()
                        .as_ref()
                        .expect("initialized selective kernel")
                        .apply_device(
                            [
                                input.values.as_array(),
                                input.input_state.as_array(),
                                input.output_state.as_array(),
                                input.time_step.as_array(),
                                input.time_step_bias.as_array(),
                                input.transition_log.as_array(),
                                input.skip.as_array(),
                                &initial,
                                &floor,
                            ],
                            &config,
                            stream,
                        )
                })
                .map_err(Error::backend_source)?;
            let direct = outputs.pop().expect("direct output");
            let weighted = outputs.pop().expect("weighted output");
            let state = outputs.pop().expect("state output");
            // Preserve MLX's projection reduction order, including cancellation.
            let output = sum_axis(&weighted, -1, false, stream)
                .and_then(|t| t.reshape(&output_shape, stream))
                .and_then(|t| t.add(&direct, stream))
                .map_err(Error::backend_source)?;
            return Ok(Some(SelectiveStateSpaceScanOutput {
                state: MlxTensor::from_array(state),
                output: MlxTensor::from_array(output),
            }));
        }
    }
    let _ = (input, stream);
    Ok(None)
}
