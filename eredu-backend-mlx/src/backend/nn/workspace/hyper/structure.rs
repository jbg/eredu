//! Native descriptor census from the same hyper worker; no tensor or heap owner.
use super::*;
use crate::backend::nn::hyper_connections::{
    self as native,
    worker::{self, Binary, Unary, Worker},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Value {
    axes: [i32; 4],
    rank: usize,
}
impl Value {
    fn new(shape: &[i32]) -> FactResult<Self> {
        if shape.len() > 4 || shape.iter().any(|d| *d < 0) {
            return Err(invalid());
        }
        let mut axes = [1; 4];
        axes[..shape.len()].copy_from_slice(shape);
        Ok(Self {
            axes,
            rank: shape.len(),
        })
    }
    fn shape(&self) -> &[i32] {
        &self.axes[..self.rank]
    }
    fn elements(self) -> FactResult<u64> {
        product(self.shape())
    }
}
#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct Structure {
    pub primitives: usize,
    pub edges: usize,
    pub seeds: usize,
    pub births: usize,
    pub handles: usize,
    pub aliases: usize,
    pub reshapes: usize,
    pub controls: usize,
}
fn add_count(a: usize, b: usize) -> FactResult<usize> {
    a.checked_add(b)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
}
fn mul_count(a: usize, b: usize) -> FactResult<usize> {
    a.checked_mul(b)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
}
#[derive(Default)]
struct Counter(Structure);
impl Counter {
    fn record(
        &mut self,
        value: Value,
        nodes: usize,
        edges: usize,
        seeds: usize,
        extra: usize,
    ) -> FactResult<Value> {
        self.0.primitives = add_count(self.0.primitives, nodes)?;
        self.0.edges = add_count(self.0.edges, edges)?;
        self.0.seeds = add_count(self.0.seeds, seeds)?;
        self.0.births = add_count(self.0.births, add_count(add_count(nodes, seeds)?, extra)?)?;
        self.0.handles = add_count(self.0.handles, 1)?;
        Ok(value)
    }
    fn binary_value(&self, a: Value, b: Value) -> FactResult<Value> {
        let rank = a.rank.max(b.rank);
        let mut shape = [1; 4];
        for axis in 0..rank {
            let x = if axis >= rank - a.rank {
                a.axes[axis - (rank - a.rank)]
            } else {
                1
            };
            let y = if axis >= rank - b.rank {
                b.axes[axis - (rank - b.rank)]
            } else {
                1
            };
            if x != y && x != 1 && y != 1 {
                return Err(invalid());
            }
            shape[axis] = if x == 1 { y } else { x };
        }
        Value::new(&shape[..rank])
    }
}
impl Worker for Counter {
    type Value = Value;
    type Error = MlxWorkspaceFactError;
    fn shape<'a>(&self, value: &'a Value) -> &'a [i32] {
        value.shape()
    }
    fn alias(&mut self, value: &Value) -> FactResult<Value> {
        self.0.aliases = add_count(self.0.aliases, 1)?;
        self.record(*value, 0, 0, 0, 0)
    }
    fn f32(&mut self, value: &Value) -> FactResult<Value> {
        self.record(*value, 1, 1, 0, 0)
    }
    fn cast_like(&mut self, value: &Value, _: &Value) -> FactResult<Value> {
        self.f32(value)
    }
    fn scalar(&mut self, _: f32) -> FactResult<Value> {
        self.record(Value::new(&[])?, 0, 0, 1, 0)
    }
    fn zeros_like(&mut self, shape: &[i32], _: &Value) -> FactResult<Value> {
        self.record(Value::new(shape)?, 4, 4, 1, 0)
    }
    fn reshape(&mut self, input: &Value, requested: &[i32]) -> FactResult<Value> {
        if requested.len() > 4 {
            return Err(invalid());
        }
        let mut shape = [1; 4];
        let mut infer = None;
        let mut known = 1u64;
        for (i, &n) in requested.iter().enumerate() {
            if n == -1 {
                if infer.replace(i).is_some() {
                    return Err(invalid());
                }
            } else {
                known = mul(known, u64::try_from(n).map_err(|_| invalid())?)?;
                shape[i] = n;
            }
        }
        let elements = input.elements()?;
        if let Some(axis) = infer {
            if known == 0 || elements % known != 0 {
                return Err(invalid());
            }
            shape[axis] = i32::try_from(elements / known).map_err(|_| invalid())?;
        } else if known != elements {
            return Err(invalid());
        }
        self.0.reshapes = add_count(self.0.reshapes, 1)?;
        self.record(Value::new(&shape[..requested.len()])?, 1, 1, 0, 0)
    }
    fn transpose(&mut self, input: &Value, axes: &[i32]) -> FactResult<Value> {
        if axes.len() != input.rank {
            return Err(invalid());
        }
        let mut shape = [1; 4];
        let mut seen = [false; 4];
        for (i, &axis) in axes.iter().enumerate() {
            let axis = usize::try_from(axis).map_err(|_| invalid())?;
            if axis >= input.rank || seen[axis] {
                return Err(invalid());
            }
            seen[axis] = true;
            shape[i] = input.axes[axis];
        }
        self.record(Value::new(&shape[..input.rank])?, 1, 1, 0, 0)
    }
    fn slice(&mut self, input: &Value, axis: usize, start: i32, end: i32) -> FactResult<Value> {
        if axis >= input.rank || start < 0 || end < start || end > input.axes[axis] {
            return Err(invalid());
        }
        let mut value = *input;
        value.axes[axis] = end - start;
        self.record(value, 1, 1, 0, 0)
    }
    fn unary(&mut self, kind: Unary, input: &Value) -> FactResult<Value> {
        // Direct MLX unary worker, not the separately optimized shared NN
        // sigmoid wrapper: only its actual dtype cast and unary descriptor.
        let count = match kind {
            Unary::Square => 1,
            Unary::Rsqrt | Unary::Sigmoid => 2,
        };
        self.record(*input, count, count, 0, 0)
    }
    fn binary(&mut self, _: Binary, a: &Value, b: &Value) -> FactResult<Value> {
        self.record(self.binary_value(*a, *b)?, 5, 6, 0, 0)
    }
    fn matmul(&mut self, a: &Value, b: &Value) -> FactResult<Value> {
        if a.rank < 2 || b.rank < 2 || a.axes[a.rank - 1] != b.axes[b.rank - 2] {
            return Err(invalid());
        }
        let prefix = self.binary_value(
            Value::new(&a.shape()[..a.rank - 2])?,
            Value::new(&b.shape()[..b.rank - 2])?,
        )?;
        let mut shape = prefix.axes;
        shape[prefix.rank] = a.axes[a.rank - 2];
        shape[prefix.rank + 1] = b.axes[b.rank - 1];
        let output = Value::new(&shape[..prefix.rank + 2])?;
        // The existing general Matmul envelope covers rank expansions, casts,
        // broadcasts/flatten/restoration. Native empty inputs create one eager
        // zero for fill_gpu; nonempty execution permits four operand copies and
        // two partial buffers. Both alternatives are source-derived here.
        let empty = a.elements()? == 0 || b.elements()? == 0;
        self.record(output, 8, 9, usize::from(empty), if empty { 0 } else { 6 })
    }
    fn mean_last(&mut self, input: &Value) -> FactResult<Value> {
        let mut output = *input;
        output.axes[output.rank - 1] = 1;
        // mean_owned: keepdims Reduce, one exact count scalar, Divide's two
        // casts/two broadcasts/result. GeneralReduce may compact and use a partial.
        self.record(output, 6, 7, 1, 2)
    }
    fn sum(&mut self, input: &Value, axis: usize, keep: bool) -> FactResult<Value> {
        if axis >= input.rank {
            return Err(invalid());
        }
        let mut output = *input;
        if keep {
            output.axes[axis] = 1;
        } else {
            for i in axis..input.rank - 1 {
                output.axes[i] = input.axes[i + 1];
            }
            output.rank -= 1;
            self.0.reshapes = add_count(self.0.reshapes, 1)?;
        }
        let count = 1 + usize::from(!keep);
        self.record(output, count, count, 0, 2)
    }
    fn softmax_last(&mut self, input: &Value) -> FactResult<Value> {
        self.record(*input, 2, 2, 0, 1)
    }
    fn repeat<F>(&mut self, count: usize, value: Value, mut step: F) -> FactResult<Value>
    where
        F: FnMut(&mut Self, &Value) -> FactResult<Value>,
    {
        if count == 0 {
            return Ok(value);
        }
        let before = self.0;
        let output = step(self, &value)?;
        if output != value {
            return Err(invalid());
        }
        macro_rules! extend { ($($field:ident),* $(,)?) => { $(
            self.0.$field = add_count(before.$field,
                mul_count(self.0.$field.checked_sub(before.$field).ok_or_else(invalid)?, count)?)?;
        )* }; }
        extend!(primitives, edges, seeds, births, handles, aliases, reshapes);
        Ok(output)
    }
}

pub(crate) fn inspect(op: WorkspaceOperationView<'_>) -> FactResult<Option<Structure>> {
    let Some(geometry) = super::geometry(op)? else {
        return Ok(None);
    };
    let mut counter = Counter::default();
    let mut values = [Value::new(&[])?; 4];
    for (value, input) in values.iter_mut().zip(op.inputs.iter()) {
        *value = Value::new(input.shape())?;
    }
    match geometry {
        Geometry::Collapse {
            shape,
            spec,
            epsilon,
            ..
        } => {
            let (collapsed, split) = worker::collapse(
                &mut counter,
                &values[0],
                &values[1],
                &values[2],
                &values[3],
                shape,
                spec.sinkhorn_iterations,
                spec.epsilon,
                epsilon,
            )?;
            for (actual, expected) in [collapsed, split.pre, split.post, split.combination]
                .into_iter()
                .zip(op.outputs.iter())
            {
                if actual.shape() != expected.shape() {
                    return Err(invalid());
                }
            }
        }
        Geometry::Expand(shape) => {
            let output = worker::expand(
                &mut counter,
                &values[0],
                &values[1],
                &values[2],
                &values[3],
                shape,
            )?;
            if output.shape() != op.outputs.get(0).unwrap().shape() {
                return Err(invalid());
            }
        }
        Geometry::HeadCoefficients { shape, spec } => {
            let (_, pre) = worker::head_coefficients(
                &mut counter,
                &values[0],
                &values[1],
                &values[2],
                &values[3],
                shape,
                spec.norm_epsilon,
                spec.epsilon,
            )?;
            if pre.shape() != op.outputs.get(0).unwrap().shape() {
                return Err(invalid());
            }
        }
        Geometry::HeadSum(shape) => {
            // The native head retains its actual F32 source from coefficient
            // preparation across the observer. This call performs no second cast.
            let output = worker::head_sum(&mut counter, &values[0], &values[1], &values[0], shape)?;
            if output.shape() != op.outputs.get(0).unwrap().shape() {
                return Err(invalid());
            }
        }
    }
    counter.0.controls = native::control_bytes(counter.0.handles, counter.0.aliases)
        .and_then(|n| n.checked_add(std::mem::size_of::<Counter>()))
        .and_then(|n| n.checked_add(std::mem::size_of::<[Value; 6]>()))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(counter.0))
}
