//! Count the actual shared selective recurrence without a tensor or heap owner.
use super::*;
use crate::backend::nn::selective_scan::{self as scan, Binary, Worker};
use eredu_nn::{SelectiveStateSpaceScanInput, operation_geometry::SelectiveScanGeometry};
use facts::{Emitter, FactResult, Output, add, buffer_capacity, mul};

#[derive(Clone, Copy, Debug)]
struct Value {
    shape: [i32; 4],
    rank: usize,
}
impl Value {
    fn new(shape: &[i32]) -> FactResult<Self> {
        if shape.len() > 4 || shape.iter().any(|d| *d <= 0) {
            return invalid();
        }
        let mut value = Self {
            shape: [1; 4],
            rank: shape.len(),
        };
        value.shape[..shape.len()].copy_from_slice(shape);
        Ok(value)
    }
    fn shape(&self) -> &[i32] {
        &self.shape[..self.rank]
    }
    fn elements(self) -> FactResult<u64> {
        self.shape().iter().try_fold(1, |n, d| mul(n, *d as u64))
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Structure {
    pub primitives: usize,
    pub edges: usize,
    pub seeds: usize,
    pub births: usize,
    pub handles: usize,
    pub indices: usize,
    pub reshapes: usize,
    pub operands: usize,
    pub concat_inputs: usize,
    pub controls: usize,
}
pub(super) struct Population {
    pub structure: Structure,
    pub bytes: u64,
    pub state: u64,
    pub output: u64,
}
// None performs the same descriptor census only. It yields no physical fact.
struct Counter {
    allocation: Option<NativeAllocationFacts>,
    p: Structure,
    bytes: u64,
}
struct Outputs {
    first: Option<Value>,
    count: usize,
    capacity: usize,
}
fn invalid<T>() -> FactResult<T> {
    Err(MlxWorkspaceFactError::descriptor(
        "invalid Metal selective scan descriptor",
    ))
}
fn checked(a: usize, b: usize) -> FactResult<usize> {
    a.checked_add(b)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
}
impl Counter {
    fn capacity(&self, value: Value) -> FactResult<u64> {
        self.allocation.map_or(Ok(0), |allocation| {
            buffer_capacity(allocation, mul(value.elements()?, 4)?)
        })
    }
    fn record(
        &mut self,
        value: Value,
        primitives: usize,
        edges: usize,
        seeds: usize,
        extra_births: usize,
        bytes: u64,
    ) -> FactResult<Value> {
        self.p.primitives = checked(self.p.primitives, primitives)?;
        self.p.edges = checked(self.p.edges, edges)?;
        self.p.seeds = checked(self.p.seeds, seeds)?;
        self.p.births = checked(
            self.p.births,
            checked(checked(primitives, seeds)?, extra_births)?,
        )?;
        self.p.handles = checked(self.p.handles, 1)?;
        self.bytes = add(self.bytes, bytes)?;
        Ok(value)
    }
    fn binary_value(&self, a: Value, b: Value) -> FactResult<Value> {
        let rank = a.rank.max(b.rank);
        let mut shape = [1; 4];
        for i in 0..rank {
            let ai = if i >= rank - a.rank {
                a.shape[i - (rank - a.rank)]
            } else {
                1
            };
            let bi = if i >= rank - b.rank {
                b.shape[i - (rank - b.rank)]
            } else {
                1
            };
            if ai != bi && ai != 1 && bi != 1 {
                return invalid();
            }
            shape[i] = ai.max(bi);
        }
        Value::new(&shape[..rank])
    }
}
impl Worker for Counter {
    type Value = Value;
    type Outputs = Outputs;
    type Error = MlxWorkspaceFactError;
    fn f32(&mut self, input: &Value) -> FactResult<Value> {
        self.record(*input, 1, 1, 0, 0, self.capacity(*input)?)
    }
    fn scalar(&mut self, _: f32) -> FactResult<Value> {
        let value = Value::new(&[])?;
        self.record(value, 0, 0, 1, 0, self.capacity(value)?)
    }
    fn zeros(&mut self, shape: &[i32]) -> FactResult<Value> {
        let value = Value::new(shape)?;
        // Existing scalar initialization envelope: both fill/copy candidates
        // plus the eager scalar; no alias/donation credit.
        let bytes = add(
            mul(2, self.capacity(value)?)?,
            self.capacity(Value::new(&[])?)?,
        )?;
        self.record(value, 4, 4, 1, 0, bytes)
    }
    fn exp(&mut self, input: &Value) -> FactResult<Value> {
        self.record(*input, 2, 2, 0, 0, mul(2, self.capacity(*input)?)?)
    }
    fn binary(&mut self, _: Binary, a: &Value, b: &Value) -> FactResult<Value> {
        let output = self.binary_value(*a, *b)?;
        // Two dtype-cast candidates, two metadata broadcasts, one result.
        let bytes = add(
            add(self.capacity(*a)?, self.capacity(*b)?)?,
            self.capacity(output)?,
        )?;
        self.record(output, 5, 6, 0, 0, bytes)
    }
    fn reshape(&mut self, input: &Value, shape: &[i32]) -> FactResult<Value> {
        let output = Value::new(shape)?;
        if input.elements()? != output.elements()? {
            return invalid();
        }
        self.p.reshapes = checked(self.p.reshapes, 1)?;
        self.record(output, 1, 1, 0, 0, self.capacity(output)?)
    }
    fn token(&mut self, input: &Value, _: i32, retain_axis: bool) -> FactResult<Value> {
        let mut shape = input.shape;
        let rank = if retain_axis {
            shape[1] = 1;
            input.rank
        } else {
            for i in 1..input.rank - 1 {
                shape[i] = shape[i + 1];
            }
            input.rank - 1
        };
        let output = Value::new(&shape[..rank])?;
        self.p.indices = checked(self.p.indices, 1)?;
        self.p.reshapes = checked(self.p.reshapes, usize::from(!retain_axis))?;
        self.record(
            output,
            2,
            2,
            0,
            0,
            if retain_axis {
                0
            } else {
                self.capacity(output)?
            },
        )
    }
    fn expand(&mut self, input: &Value, axis: usize) -> FactResult<Value> {
        if input.rank >= 4 || axis > input.rank {
            return invalid();
        }
        let mut shape = [1; 4];
        shape[..axis].copy_from_slice(&input.shape()[..axis]);
        shape[axis + 1..input.rank + 1].copy_from_slice(&input.shape()[axis..]);
        let output = Value::new(&shape[..input.rank + 1])?;
        self.p.indices = checked(self.p.indices, 1)?;
        self.p.reshapes = checked(self.p.reshapes, 1)?;
        self.record(output, 2, 2, 0, 0, self.capacity(output)?)
    }
    fn softplus(&mut self, input: &Value) -> FactResult<Value> {
        // layers::softplus is logaddexp(x, eager I32 zero), not beta-softplus.
        // The same floating binary worker casts/broadcasts both arguments.
        let scalar = self.capacity(Value::new(&[])?)?;
        self.record(
            *input,
            5,
            6,
            1,
            0,
            add(mul(2, self.capacity(*input)?)?, mul(2, scalar)?)?,
        )
    }
    fn sum_last(&mut self, input: &Value) -> FactResult<Value> {
        let output = Value::new(&input.shape()[..input.rank - 1])?;
        let bytes = self.allocation.map_or(Ok(0), |allocation| {
            reduction::sum_cost_fixed(
                allocation,
                input.elements()?,
                output.elements()?,
                input.shape[input.rank - 1] as u64,
            )
        })?;
        // Reduce + Squeeze, with GeneralReduce compaction and its optional
        // two-pass partial array supplied by the shared native reduction facts.
        self.p.reshapes = checked(self.p.reshapes, 1)?;
        self.record(output, 2, 2, 0, 2, bytes)
    }
    fn outputs(&mut self, count: usize) -> FactResult<Outputs> {
        Ok(Outputs {
            first: None,
            count: 0,
            capacity: count,
        })
    }
    fn push(&mut self, outputs: &mut Outputs, value: Value) -> FactResult<()> {
        if outputs.count == outputs.capacity
            || outputs
                .first
                .is_some_and(|first| first.shape() != value.shape())
        {
            return invalid();
        }
        outputs.first = Some(value);
        outputs.count += 1;
        Ok(())
    }
    fn concatenate(&mut self, outputs: &Outputs) -> FactResult<Value> {
        if outputs.count != outputs.capacity {
            return invalid();
        }
        let first = outputs
            .first
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        let mut output = first;
        output.shape[1] = i32::try_from(outputs.count).map_err(MlxWorkspaceFactError::integer)?;
        let count = outputs.count;
        self.p.concat_inputs = count;
        self.p.operands = self.p.operands.max(count);
        // Cast candidate for each input plus Concatenate; all casts/output
        // backings remain counted, including the singleton alias alternative.
        self.record(
            output,
            checked(count, 1)?,
            count
                .checked_mul(2)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            0,
            0,
            add(
                mul(count as u64, self.capacity(first)?)?,
                self.capacity(output)?,
            )?,
        )
    }
    fn output_dtype(&mut self, output: &Value, _: &Value) -> FactResult<Value> {
        self.f32(output)
    }
}

fn inspect_inner(
    operation: WorkspaceOperationView<'_>,
    allocation: Option<NativeAllocationFacts>,
) -> FactResult<Option<Population>> {
    let WorkspaceOperationKindView::SelectiveStateSpaceScan(chunk, floor) = operation.kind else {
        return Ok(None);
    };
    if !(7..=8).contains(&operation.inputs.len()) || operation.outputs.len() != 2 {
        return invalid();
    }
    if operation
        .inputs
        .iter()
        .chain(operation.outputs.iter())
        .any(|v| v.dtype() != WorkspaceDtype::Float32)
    {
        return Ok(None);
    }
    let shapes = std::array::from_fn(|i| operation.inputs.get(i).expect("seven inputs").shape());
    let geometry = SelectiveScanGeometry::new(
        shapes,
        operation.inputs.get(7).map(|v| v.shape()),
        chunk,
        floor,
    )
    .map_err(|_| MlxWorkspaceFactError::descriptor("invalid Metal selective scan geometry"))?;
    if operation.outputs.get(0).unwrap().shape() != geometry.state()
        || operation.outputs.get(1).unwrap().shape() != geometry.values()
    {
        return invalid();
    }
    let mut values = [Value {
        shape: [1; 4],
        rank: 0,
    }; 8];
    for (slot, input) in values.iter_mut().zip(operation.inputs.iter()) {
        *slot = Value::new(input.shape())?;
    }
    let input = SelectiveStateSpaceScanInput {
        values: &values[0],
        input_state: &values[1],
        output_state: &values[2],
        time_step: &values[3],
        time_step_bias: &values[4],
        transition_log: &values[5],
        skip: &values[6],
        initial_state: (operation.inputs.len() == 8).then_some(&values[7]),
        chunk_size: chunk,
        time_step_floor: floor,
    };
    let mut counter = Counter {
        allocation,
        p: Structure {
            operands: 4,
            ..Structure::default()
        },
        bytes: 0,
    };
    let (state, output) = scan::run(input, geometry, &mut counter)?;
    let state = counter.capacity(state)?;
    let output = counter.capacity(output)?;
    counter.p.controls = scan::control_bytes(
        geometry.sequence() as usize,
        counter.p.handles,
        counter.p.indices,
    )
    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(Population {
        structure: counter.p,
        bytes: counter.bytes,
        state,
        output,
    }))
}

/// Only descriptor/constructor populations escape this cold no-allocation path.
/// Physical facts require the selected allocator in emit; these counts confer
/// neither a physical capacity nor source ownership.
pub(super) fn structure(operation: WorkspaceOperationView<'_>) -> FactResult<Option<Structure>> {
    Ok(inspect_inner(operation, None)?.map(|p| p.structure))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(p) = inspect_inner(operation, Some(allocation))? else {
        return Ok(None);
    };
    sink.output(Output::Allocate(p.state))?;
    sink.output(Output::Allocate(p.output))?;
    let scratch = p
        .bytes
        .checked_sub(add(p.state, p.output)?)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    sink.finish(scratch,format_args!("shared selective recurrence: every cast, slice reshape, scalar, pointwise result, reduction partial and concatenation retained; FP32 accumulation, explicit final dtype cast; no donation or completion credit")).map(Some)
}
