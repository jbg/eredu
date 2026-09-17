//! One ordinary recurrence, consumed by native execution and its cold census.
use eredu_nn::{SelectiveStateSpaceScanInput, operation_geometry::SelectiveScanGeometry};
mod native;
pub(crate) use native::{control_bytes, execute};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Binary {
    Add,
    Multiply,
    Maximum,
}

/// Primitive adapters preserve the actual native sequence. The cold adapter is
/// stack-only: it never constructs a tensor, source, stream or allocation grant.
pub(crate) trait Worker {
    type Value;
    type Outputs;
    type Error;
    fn f32(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn scalar(&mut self, value: f32) -> Result<Self::Value, Self::Error>;
    fn zeros(&mut self, shape: &[i32]) -> Result<Self::Value, Self::Error>;
    fn exp(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn binary(
        &mut self,
        kind: Binary,
        a: &Self::Value,
        b: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    fn reshape(&mut self, input: &Self::Value, shape: &[i32]) -> Result<Self::Value, Self::Error>;
    fn token(
        &mut self,
        input: &Self::Value,
        token: i32,
        retain_axis: bool,
    ) -> Result<Self::Value, Self::Error>;
    fn expand(&mut self, input: &Self::Value, axis: usize) -> Result<Self::Value, Self::Error>;
    fn softplus(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn sum_last(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn outputs(&mut self, count: usize) -> Result<Self::Outputs, Self::Error>;
    fn push(&mut self, outputs: &mut Self::Outputs, value: Self::Value) -> Result<(), Self::Error>;
    fn concatenate(&mut self, outputs: &Self::Outputs) -> Result<Self::Value, Self::Error>;
    fn output_dtype(
        &mut self,
        output: &Self::Value,
        original: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}

pub(crate) fn run<W: Worker>(
    input: SelectiveStateSpaceScanInput<'_, W::Value>,
    geometry: SelectiveScanGeometry,
    worker: &mut W,
) -> Result<(W::Value, W::Value), W::Error> {
    let [batch, sequence, heads, _] = geometry.values();
    let mut state = match input.initial_state {
        Some(state) => worker.f32(state)?,
        None => worker.zeros(&geometry.state())?,
    };
    let values = worker.f32(input.values)?;
    let input_state = worker.f32(input.input_state)?;
    let output_state = worker.f32(input.output_state)?;
    let transition = worker.f32(input.transition_log)?;
    let transition = worker.exp(&transition)?;
    let minus_one = worker.scalar(-1.0)?;
    let transition = worker.binary(Binary::Multiply, &transition, &minus_one)?;
    let transition = worker.reshape(&transition, &[1, heads, 1, 1])?;
    let skip = worker.f32(input.skip)?;
    let skip = worker.reshape(&skip, &[1, heads, 1])?;
    let bias = worker.f32(input.time_step_bias)?;
    let bias = worker.reshape(&bias, &[1, 1, heads])?;
    let mut outputs = worker.outputs(sequence as usize)?;
    let mut start = 0;
    while start < sequence {
        let end = geometry.chunk_end(start).expect("validated scan cursor");
        for token in start..end {
            let value = worker.token(&values, token, false)?;
            let b = worker.token(&input_state, token, false)?;
            let c = worker.token(&output_state, token, false)?;
            let dt = worker.token(input.time_step, token, true)?;
            let dt = worker.binary(Binary::Add, &dt, &bias)?;
            let dt = worker.softplus(&dt)?;
            let floor = worker.scalar(input.time_step_floor)?;
            let dt = worker.binary(Binary::Maximum, &dt, &floor)?;
            let dt = worker.f32(&dt)?;
            let dt = worker.reshape(&dt, &[batch, heads])?;
            let dt_transition = worker.reshape(&dt, &[batch, heads, 1, 1])?;
            let decay = worker.binary(Binary::Multiply, &dt_transition, &transition)?;
            let decay = worker.exp(&decay)?;
            let dt_input = worker.reshape(&dt, &[batch, heads, 1])?;
            let discretized_b = worker.binary(Binary::Multiply, &dt_input, &b)?;
            let value_column = worker.expand(&value, 3)?;
            let input_row = worker.expand(&discretized_b, 2)?;
            let update = worker.binary(Binary::Multiply, &value_column, &input_row)?;
            let decayed = worker.binary(Binary::Multiply, &state, &decay)?;
            state = worker.binary(Binary::Add, &decayed, &update)?;
            let output_row = worker.expand(&c, 2)?;
            let projected = worker.binary(Binary::Multiply, &state, &output_row)?;
            let projected = worker.sum_last(&projected)?;
            let skipped = worker.binary(Binary::Multiply, &value, &skip)?;
            let output = worker.binary(Binary::Add, &projected, &skipped)?;
            let output = worker.expand(&output, 1)?;
            worker.push(&mut outputs, output)?;
        }
        start = end;
    }
    let output = worker.concatenate(&outputs)?;
    let output = worker.output_dtype(&output, input.values)?;
    Ok((state, output))
}
