use super::*;
use crate::MlxTensor;
use eredu_nn::{Error, SelectiveStateSpaceScanOutput};
use safemlx::{
    Array, Dtype, OriginalScopeObserver, Stream,
    error::Exception,
    ops::{
        self,
        indexing::{NewAxis, TryIndexOp},
    },
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Refusal<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    _custody: Exception,
}
fn failure<E: std::error::Error + Send + Sync + 'static>(cause: E) -> Exception {
    match OriginalScopeObserver::try_current() {
        Ok(Some(scope)) => Exception::from_retained_source(Refusal {
            cause,
            _custody: scope.invalid_input_error(),
        }),
        Err(custody) => Exception::from_retained_source(Refusal {
            cause,
            _custody: custody,
        }),
        Ok(None) => Exception::from_source(cause),
    }
}
struct Native<'a>(&'a Stream);
impl Worker for Native<'_> {
    type Value = Array;
    type Outputs = Vec<Array>;
    type Error = Exception;
    fn f32(&mut self, input: &Array) -> Result<Array, Exception> {
        input.as_dtype(Dtype::Float32, self.0)
    }
    fn scalar(&mut self, value: f32) -> Result<Array, Exception> {
        Array::try_from_f32(value)
    }
    fn zeros(&mut self, shape: &[i32]) -> Result<Array, Exception> {
        ops::zeros::<f32>(shape, self.0)
    }
    fn exp(&mut self, input: &Array) -> Result<Array, Exception> {
        ops::exp(input, self.0)
    }
    fn binary(&mut self, kind: Binary, a: &Array, b: &Array) -> Result<Array, Exception> {
        match kind {
            Binary::Add => a.add(b, self.0),
            Binary::Multiply => a.multiply(b, self.0),
            Binary::Maximum => ops::maximum(a, b, self.0),
        }
    }
    fn reshape(&mut self, input: &Array, shape: &[i32]) -> Result<Array, Exception> {
        input.reshape(shape, self.0)
    }
    fn token(&mut self, input: &Array, token: i32, retain_axis: bool) -> Result<Array, Exception> {
        if retain_axis {
            input.try_index_device((.., token..token + 1, ..), self.0)
        } else {
            input.try_index_device((.., token, .., ..), self.0)
        }
    }
    fn expand(&mut self, input: &Array, axis: usize) -> Result<Array, Exception> {
        match axis {
            1 => input.try_index_device((.., NewAxis, .., ..), self.0),
            2 => input.try_index_device((.., .., NewAxis, ..), self.0),
            3 => input.try_index_device((.., .., .., NewAxis), self.0),
            _ => unreachable!("closed scan equation axes"),
        }
    }
    fn softplus(&mut self, input: &Array) -> Result<Array, Exception> {
        super::super::layers::softplus(input, self.0)
    }
    fn sum_last(&mut self, input: &Array) -> Result<Array, Exception> {
        ops::sum_axis(input, -1, false, self.0)
    }
    fn outputs(&mut self, count: usize) -> Result<Vec<Array>, Exception> {
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(failure)?;
        Ok(values)
    }
    fn push(&mut self, outputs: &mut Vec<Array>, value: Array) -> Result<(), Exception> {
        assert!(
            outputs.len() < outputs.capacity(),
            "exact scan output destinations"
        );
        outputs.push(value);
        Ok(())
    }
    fn concatenate(&mut self, outputs: &Vec<Array>) -> Result<Array, Exception> {
        ops::concatenate_axis(outputs, 1, self.0)
    }
    fn output_dtype(&mut self, output: &Array, original: &Array) -> Result<Array, Exception> {
        output.as_dtype(original.dtype(), self.0)
    }
}

pub(crate) fn execute(
    input: SelectiveStateSpaceScanInput<'_, MlxTensor>,
    stream: &Stream,
) -> Result<SelectiveStateSpaceScanOutput<MlxTensor>, Error> {
    let input = SelectiveStateSpaceScanInput {
        values: input.values.as_array(),
        input_state: input.input_state.as_array(),
        output_state: input.output_state.as_array(),
        time_step: input.time_step.as_array(),
        time_step_bias: input.time_step_bias.as_array(),
        transition_log: input.transition_log.as_array(),
        skip: input.skip.as_array(),
        initial_state: input.initial_state.map(MlxTensor::as_array),
        chunk_size: input.chunk_size,
        time_step_floor: input.time_step_floor,
    };
    let result: Result<(Array, Array), Exception> = (|| {
        let geometry = SelectiveScanGeometry::new(
            [
                input.values.shape(),
                input.input_state.shape(),
                input.output_state.shape(),
                input.time_step.shape(),
                input.time_step_bias.shape(),
                input.transition_log.shape(),
                input.skip.shape(),
            ],
            input.initial_state.map(Array::shape),
            input.chunk_size,
            input.time_step_floor,
        )
        .map_err(failure)?;
        if let Some(observer) = OriginalScopeObserver::try_current()? {
            if stream.device_type()? != safemlx::DeviceType::Gpu
                || [
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
                .any(|a| !matches!(a.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16))
            {
                return Err(observer.invalid_input_error());
            }
        }
        run(input, geometry, &mut Native(stream))
    })();
    let (state, output) = result.map_err(|cause| match OriginalScopeObserver::try_current() {
        Ok(None) => Error::backend_retained_source(cause),
        _ => Error::backend_retained_source(cause),
    })?;
    Ok(SelectiveStateSpaceScanOutput {
        state: MlxTensor::from_array(state),
        output: MlxTensor::from_array(output),
    })
}

pub(crate) fn control_bytes(sequence: usize, handles: usize, indices: usize) -> Option<usize> {
    let controls = [
        std::alloc::Layout::array::<Array>(sequence).ok()?.size(),
        size_of::<Native<'_>>(),
        size_of::<SelectiveScanGeometry>(),
        size_of::<SelectiveStateSpaceScanInput<'_, Array>>(),
        size_of::<SelectiveStateSpaceScanInput<'_, MlxTensor>>(),
        size_of::<SelectiveStateSpaceScanOutput<MlxTensor>>(),
        size_of::<Result<SelectiveStateSpaceScanOutput<MlxTensor>, Error>>(),
        size_of::<Result<(Array, Array), Exception>>(),
        // One returned native handle and fallible transport per actual worker call.
        // This includes shadowed locals until their enclosing token scope retires.
        std::alloc::Layout::array::<(Array, Result<Array, Exception>)>(handles)
            .ok()?
            .size(),
        size_of::<Vec<Array>>(),
        size_of::<[i32; 4]>(),
        size_of::<[i32; 3]>(),
        size_of::<[i32; 2]>(),
        size_of::<[i32; 8]>(),
        size_of::<[&Array; 7]>(),
        size_of::<[&[i32]; 7]>(),
        size_of::<Option<&Array>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Option<OriginalScopeObserver>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        size_of::<
            Result<SelectiveScanGeometry, eredu_nn::operation_geometry::SelectiveScanGeometryError>,
        >(),
        Error::retained_source_construction_bytes::<Exception>()?,
        Exception::retained_source_control_bytes::<
            Refusal<eredu_nn::operation_geometry::SelectiveScanGeometryError>,
        >()?,
        Exception::retained_source_control_bytes::<Refusal<std::collections::TryReserveError>>()?,
        OriginalScopeObserver::control_bytes()?,
        Stream::device_type_control_bytes()?,
        ops::concatenate_axis_control_bytes()?,
        size_of::<std::ops::Range<i32>>(),
        size_of::<Binary>(),
        size_of::<[i32; 5]>(),
        size_of::<Result<(), Exception>>(),
        ops::indexing::inline_basic_index_control_bytes()?.checked_mul(indices)?,
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
