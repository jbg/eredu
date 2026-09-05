use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
};

use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointLease, CheckpointSource, EncodedTensorLease, TensorReadRequest},
};
use eredu_nn::{
    AttentionMask, EmbeddingOperator, EmbeddingSpec, Error, GatedProductPolicy, Index,
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, PadMode, ParameterVisitor, ParameterVisitorMut, Parameterized,
    RotaryOperator, RotaryPosition, RotarySpec, Tensor,
};
use eredu_runtime::ParameterBackend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailAt {
    None,
    Capability,
    Read,
    Recipe,
    Completion,
    Validate,
    DelayedCompletion,
    Execution,
}

#[derive(Clone, Debug, Default)]
pub struct Counters {
    pub preflight: usize,
    pub payload_reads: usize,
    pub direct_recipes: usize,
    pub transpose_recipes: usize,
    pub direct_materializations: usize,
    pub materialized_cache_hits: usize,
    pub materializations: usize,
    pub completions: usize,
    pub completions_with_live_leases: usize,
    pub delayed_completion_waits: usize,
    pub guard_drops: usize,
    pub live_encoded_leases: usize,
    pub max_live_encoded_leases: usize,
    pub encoded_lease_drops: usize,
    pub validations: usize,
    pub binds: usize,
    pub unloaded_allocations: usize,
    pub execution_failures: usize,
}

#[derive(Debug, Default)]
struct CompletionGate {
    waiting: bool,
    released: bool,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub trace: Arc<Mutex<Counters>>,
    fail_at: Arc<Mutex<FailAt>>,
    completion_gate: Arc<(Mutex<CompletionGate>, Condvar)>,
    weight_cache: Arc<Mutex<BTreeMap<String, NumericTensor>>>,
}

impl Context {
    pub fn new(fail_at: FailAt) -> Self {
        Self {
            trace: Arc::new(Mutex::new(Counters::default())),
            fail_at: Arc::new(Mutex::new(fail_at)),
            completion_gate: Arc::new((Mutex::new(CompletionGate::default()), Condvar::new())),
            weight_cache: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn counters(&self) -> Counters {
        self.trace.lock().unwrap().clone()
    }

    fn bump(&self, update: impl FnOnce(&mut Counters)) {
        update(&mut self.trace.lock().unwrap());
    }

    pub fn set_failure(&self, failure: FailAt) {
        *self.fail_at.lock().unwrap() = failure;
    }

    fn failure(&self) -> FailAt {
        *self.fail_at.lock().unwrap()
    }

    pub fn wait_for_delayed_completion(&self) {
        let (lock, ready) = self.completion_gate.as_ref();
        let mut state = lock.lock().unwrap();
        while !state.waiting {
            state = ready.wait(state).unwrap();
        }
    }

    pub fn release_delayed_completion(&self) {
        let (lock, ready) = self.completion_gate.as_ref();
        let mut state = lock.lock().unwrap();
        state.released = true;
        ready.notify_all();
    }

    fn delay_first_completion(&self) {
        let (lock, ready) = self.completion_gate.as_ref();
        let mut state = lock.lock().unwrap();
        if state.waiting {
            return;
        }
        state.waiting = true;
        self.bump(|c| c.delayed_completion_waits += 1);
        ready.notify_all();
        while !state.released {
            state = ready.wait(state).unwrap();
        }
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Context>> = const { RefCell::new(None) };
}

pub fn activate(context: &Context) {
    ACTIVE.with(|active| *active.borrow_mut() = Some(context.clone()));
}

fn active() -> Context {
    ACTIVE.with(|active| {
        active
            .borrow()
            .clone()
            .expect("numeric backend context is active")
    })
}

#[derive(Clone, Debug)]
enum Data {
    F32Dense(Arc<Vec<f32>>),
    F32Sparse(Arc<BTreeMap<usize, f32>>),
    I32(Arc<Vec<i32>>),
}

#[derive(Clone, Debug)]
pub struct NumericTensor {
    shape: Vec<i32>,
    data: Data,
}

fn elements(shape: &[i32]) -> usize {
    shape.iter().map(|&value| value as usize).product()
}

fn strides(shape: &[i32]) -> Vec<usize> {
    let mut result = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        result[axis] = result[axis + 1] * shape[axis + 1] as usize;
    }
    result
}

fn coordinates(mut flat: usize, shape: &[i32]) -> Vec<usize> {
    let stride = strides(shape);
    stride
        .iter()
        .map(|&step| {
            let value = flat / step;
            flat %= step;
            value
        })
        .collect()
}

fn flat_index(coords: &[usize], shape: &[i32]) -> usize {
    coords
        .iter()
        .zip(strides(shape))
        .map(|(coord, stride)| coord * stride)
        .sum()
}

fn normalized_axis(axis: i32, rank: usize) -> usize {
    if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    }
}

impl NumericTensor {
    pub fn f32(values: Vec<f32>, shape: &[i32]) -> Self {
        assert_eq!(values.len(), elements(shape));
        Self {
            shape: shape.to_vec(),
            data: Data::F32Dense(Arc::new(values)),
        }
    }

    fn zero(shape: &[i32]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: Data::F32Sparse(Arc::new(BTreeMap::new())),
        }
    }

    fn sparse(shape: &[i32], values: BTreeMap<usize, f32>) -> Self {
        Self {
            shape: shape.to_vec(),
            data: Data::F32Sparse(Arc::new(values)),
        }
    }

    fn is_zero(&self) -> bool {
        match &self.data {
            Data::F32Sparse(values) => values.is_empty(),
            Data::F32Dense(values) => values.iter().all(|&value| value == 0.0),
            Data::I32(values) => values.iter().all(|&value| value == 0),
        }
    }

    fn is_everywhere_nonzero(&self) -> bool {
        (0..elements(&self.shape)).all(|index| self.get_f32(index) != 0.0)
    }

    fn get_f32(&self, index: usize) -> f32 {
        match &self.data {
            Data::F32Dense(values) => values[index],
            Data::F32Sparse(values) => values.get(&index).copied().unwrap_or(0.0),
            Data::I32(values) => values[index] as f32,
        }
    }

    fn get_i32(&self, index: usize) -> i32 {
        match &self.data {
            Data::I32(values) => values[index],
            _ => self.get_f32(index) as i32,
        }
    }

    fn dense_f32(&self) -> Vec<f32> {
        (0..elements(&self.shape))
            .map(|i| self.get_f32(i))
            .collect()
    }

    fn broadcast_value(&self, output_coords: &[usize], output_shape: &[i32]) -> f32 {
        let offset = output_shape.len() - self.shape.len();
        let coords = self
            .shape
            .iter()
            .enumerate()
            .map(|(axis, &dim)| {
                if dim == 1 {
                    0
                } else {
                    output_coords[offset + axis]
                }
            })
            .collect::<Vec<_>>();
        self.get_f32(flat_index(&coords, &self.shape))
    }

    fn binary(&self, rhs: &Self, operation: impl Fn(f32, f32) -> f32) -> Result<Self, Error> {
        let rank = self.shape.len().max(rhs.shape.len());
        let mut shape = vec![1; rank];
        for (axis, output_dimension) in shape.iter_mut().enumerate() {
            let left = *self
                .shape
                .get(self.shape.len().wrapping_sub(rank - axis))
                .unwrap_or(&1);
            let right = *rhs
                .shape
                .get(rhs.shape.len().wrapping_sub(rank - axis))
                .unwrap_or(&1);
            if left != right && left != 1 && right != 1 {
                return Err(Error::backend("incompatible reference broadcast"));
            }
            *output_dimension = left.max(right);
        }
        let values = (0..elements(&shape))
            .map(|flat| {
                let coords = coordinates(flat, &shape);
                operation(
                    self.broadcast_value(&coords, &shape),
                    rhs.broadcast_value(&coords, &shape),
                )
            })
            .collect();
        Ok(Self::f32(values, &shape))
    }

    fn map(&self, operation: impl Fn(f32) -> f32) -> Self {
        Self::f32(
            (0..elements(&self.shape))
                .map(|index| operation(self.get_f32(index)))
                .collect(),
            &self.shape,
        )
    }
}

impl Tensor for NumericTensor {
    type Context = Context;

    fn shape(&self) -> &[i32] {
        &self.shape
    }

    fn unloaded_f32(shape: &[i32], context: &Context) -> Result<Self, Error> {
        context.bump(|c| c.unloaded_allocations += 1);
        Ok(Self::zero(shape))
    }

    fn unloaded_i32(shape: &[i32], context: &Context) -> Result<Self, Error> {
        context.bump(|c| c.unloaded_allocations += 1);
        Ok(Self {
            shape: shape.to_vec(),
            data: Data::I32(Arc::new(vec![0; elements(shape)])),
        })
    }

    fn from_f32_slice(values: &[f32], shape: &[i32], _: &Context) -> Result<Self, Error> {
        Ok(Self::f32(values.to_vec(), shape))
    }

    fn from_i32_slice(values: &[i32], shape: &[i32], _: &Context) -> Result<Self, Error> {
        Ok(Self {
            shape: shape.to_vec(),
            data: Data::I32(Arc::new(values.to_vec())),
        })
    }

    fn to_f32_vec(&self, _: &Context) -> Result<Vec<f32>, Error> {
        Ok(self.dense_f32())
    }
    fn to_i32_vec(&self, _: &Context) -> Result<Vec<i32>, Error> {
        Ok((0..elements(&self.shape))
            .map(|i| self.get_i32(i))
            .collect())
    }
    fn full_f32(value: f32, shape: &[i32], _: &Context) -> Result<Self, Error> {
        Ok(Self::f32(vec![value; elements(shape)], shape))
    }
    fn full_i32(value: i32, shape: &[i32], _: &Context) -> Result<Self, Error> {
        Ok(Self {
            shape: shape.to_vec(),
            data: Data::I32(Arc::new(vec![value; elements(shape)])),
        })
    }

    fn add(&self, rhs: &Self, _: &Context) -> Result<Self, Error> {
        self.binary(rhs, |a, b| a + b)
    }
    fn subtract(&self, rhs: &Self, _: &Context) -> Result<Self, Error> {
        self.binary(rhs, |a, b| a - b)
    }
    fn multiply(&self, rhs: &Self, _: &Context) -> Result<Self, Error> {
        self.binary(rhs, |a, b| a * b)
    }
    fn multiply_scalar(&self, rhs: f32, _: &Context) -> Result<Self, Error> {
        if rhs == 0.0 || self.is_zero() {
            return Ok(Self::zero(&self.shape));
        }
        Ok(self.map(|a| a * rhs))
    }
    fn divide(&self, rhs: &Self, _: &Context) -> Result<Self, Error> {
        if self.is_zero() && rhs.is_everywhere_nonzero() {
            return Ok(Self::zero(&self.shape));
        }
        self.binary(rhs, |a, b| a / b)
    }
    fn square(&self, _: &Context) -> Result<Self, Error> {
        if self.is_zero() {
            return Ok(Self::zero(&self.shape));
        }
        Ok(self.map(|a| a * a))
    }
    fn tanh(&self, _: &Context) -> Result<Self, Error> {
        Ok(self.map(f32::tanh))
    }
    fn maximum_scalar(&self, rhs: f32, _: &Context) -> Result<Self, Error> {
        Ok(self.map(|a| a.max(rhs)))
    }

    fn reshape(&self, shape: &[i32], _: &Context) -> Result<Self, Error> {
        let mut shape = shape.to_vec();
        if let Some(position) = shape.iter().position(|&dim| dim == -1) {
            let known: i32 = shape.iter().filter(|&&dim| dim != -1).product();
            shape[position] = elements(&self.shape) as i32 / known;
        }
        if elements(&shape) != elements(&self.shape) {
            return Err(Error::backend("invalid reshape"));
        }
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }

    fn transpose_axes(&self, axes: &[i32], _: &Context) -> Result<Self, Error> {
        let shape = axes
            .iter()
            .map(|&axis| self.shape[axis as usize])
            .collect::<Vec<_>>();
        let mut sparse = BTreeMap::new();
        if let Data::F32Sparse(values) = &self.data {
            for (&flat, &value) in values.iter() {
                let old = coordinates(flat, &self.shape);
                let new = axes
                    .iter()
                    .map(|&axis| old[axis as usize])
                    .collect::<Vec<_>>();
                sparse.insert(flat_index(&new, &shape), value);
            }
            return Ok(Self::sparse(&shape, sparse));
        }
        let values = (0..elements(&shape))
            .map(|flat| {
                let new = coordinates(flat, &shape);
                let mut old = vec![0; axes.len()];
                for (new_axis, &old_axis) in axes.iter().enumerate() {
                    old[old_axis as usize] = new[new_axis];
                }
                self.get_f32(flat_index(&old, &self.shape))
            })
            .collect();
        Ok(Self::f32(values, &shape))
    }

    fn swap_axes(&self, left: i32, right: i32, context: &Context) -> Result<Self, Error> {
        let mut axes = (0..self.shape.len() as i32).collect::<Vec<_>>();
        let rank = axes.len();
        axes.swap(normalized_axis(left, rank), normalized_axis(right, rank));
        self.transpose_axes(&axes, context)
    }
    fn transpose(&self, context: &Context) -> Result<Self, Error> {
        self.transpose_axes(&[1, 0], context)
    }
    fn expand_dims(&self, axis: i32, _: &Context) -> Result<Self, Error> {
        let mut shape = self.shape.clone();
        let axis = if axis < 0 {
            (shape.len() as i32 + axis + 1) as usize
        } else {
            axis as usize
        };
        shape.insert(axis, 1);
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }
    fn squeeze_axes(&self, axes: &[i32], _: &Context) -> Result<Self, Error> {
        let remove = axes
            .iter()
            .map(|&a| normalized_axis(a, self.shape.len()))
            .collect::<Vec<_>>();
        let shape = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (!remove.contains(&i)).then_some(d))
            .collect();
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }

    fn index(&self, indexes: &[Index], _: &Context) -> Result<Self, Error> {
        let mut shape = Vec::new();
        for (axis, index) in indexes.iter().enumerate() {
            match *index {
                Index::Full => shape.push(self.shape[axis]),
                Index::At(_) => {}
                Index::Range(a, b) => shape.push(b - a),
            }
        }
        let values = (0..elements(&shape))
            .map(|flat| {
                let output = coordinates(flat, &shape);
                let mut cursor = 0;
                let input = indexes
                    .iter()
                    .map(|index| match *index {
                        Index::Full => {
                            let v = output[cursor];
                            cursor += 1;
                            v
                        }
                        Index::At(value) => value as usize,
                        Index::Range(start, _) => {
                            let v = start as usize + output[cursor];
                            cursor += 1;
                            v
                        }
                    })
                    .collect::<Vec<_>>();
                self.get_f32(flat_index(&input, &self.shape))
            })
            .collect();
        Ok(Self::f32(values, &shape))
    }

    fn take_axis(&self, indexes: &Self, axis: i32, _: &Context) -> Result<Self, Error> {
        let axis = normalized_axis(axis, self.shape.len());
        let mut shape = self.shape.clone();
        shape[axis] = elements(&indexes.shape) as i32;
        let values = (0..elements(&shape))
            .map(|flat| {
                let mut coords = coordinates(flat, &shape);
                coords[axis] = indexes.get_i32(coords[axis]) as usize;
                self.get_f32(flat_index(&coords, &self.shape))
            })
            .collect();
        Ok(Self::f32(values, &shape))
    }

    fn concatenate(values: &[Self], axis: i32, _: &Context) -> Result<Self, Error> {
        let axis = normalized_axis(axis, values[0].shape.len());
        let mut shape = values[0].shape.clone();
        shape[axis] = values.iter().map(|v| v.shape[axis]).sum();
        let output = (0..elements(&shape))
            .map(|flat| {
                let mut coords = coordinates(flat, &shape);
                let mut position = coords[axis];
                for value in values {
                    if position < value.shape[axis] as usize {
                        coords[axis] = position;
                        return value.get_f32(flat_index(&coords, &value.shape));
                    }
                    position -= value.shape[axis] as usize;
                }
                unreachable!()
            })
            .collect();
        Ok(Self::f32(output, &shape))
    }

    fn stack(values: &[Self], axis: i32, context: &Context) -> Result<Self, Error> {
        let axis = if axis < 0 {
            values[0].shape.len() as i32 + axis + 1
        } else {
            axis
        };
        let expanded = values
            .iter()
            .map(|v| v.expand_dims(axis, context))
            .collect::<Result<Vec<_>, _>>()?;
        Self::concatenate(&expanded, axis, context)
    }

    fn matmul(lhs: &Self, rhs: &Self, _: &Context) -> Result<Self, Error> {
        let m = lhs.shape[lhs.shape.len() - 2] as usize;
        let k = lhs.shape[lhs.shape.len() - 1] as usize;
        let n = rhs.shape[rhs.shape.len() - 1] as usize;
        let mut shape = lhs.shape[..lhs.shape.len() - 2].to_vec();
        shape.push(m as i32);
        shape.push(n as i32);
        if lhs.is_zero() || rhs.is_zero() {
            return Ok(Self::zero(&shape));
        }
        let batches = elements(&shape[..shape.len() - 2]);
        let mut output = vec![0.0; batches * m * n];
        for batch in 0..batches {
            for i in 0..m {
                for j in 0..n {
                    output[(batch * m + i) * n + j] = (0..k)
                        .map(|p| {
                            lhs.get_f32((batch * m + i) * k + p)
                                * rhs.get_f32((batch * k + p) * n + j)
                        })
                        .sum();
                }
            }
        }
        Ok(Self::f32(output, &shape))
    }

    fn sum_axis(value: &Self, axis: i32, keep_dims: bool, _: &Context) -> Result<Self, Error> {
        let axis = normalized_axis(axis, value.shape.len());
        let mut shape = value.shape.clone();
        let width = shape[axis] as usize;
        if keep_dims {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        if value.is_zero() {
            return Ok(Self::zero(&shape));
        }
        let output = (0..elements(&shape))
            .map(|flat| {
                let coords = coordinates(flat, &shape);
                (0..width)
                    .map(|position| {
                        let mut input = coords.clone();
                        if keep_dims {
                            input[axis] = position;
                        } else {
                            input.insert(axis, position);
                        }
                        value.get_f32(flat_index(&input, &value.shape))
                    })
                    .sum()
            })
            .collect();
        Ok(Self::f32(output, &shape))
    }

    fn argmin_axis(value: &Self, axis: i32, keep_dims: bool, _: &Context) -> Result<Self, Error> {
        let axis = normalized_axis(axis, value.shape.len());
        let mut shape = value.shape.clone();
        let width = shape[axis] as usize;
        if keep_dims {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        let output = (0..elements(&shape))
            .map(|flat| {
                let coords = coordinates(flat, &shape);
                let mut best = (0, f32::INFINITY);
                for position in 0..width {
                    let mut input = coords.clone();
                    if keep_dims {
                        input[axis] = position
                    } else {
                        input.insert(axis, position);
                    }
                    let candidate = value.get_f32(flat_index(&input, &value.shape));
                    if candidate < best.1 {
                        best = (position, candidate);
                    }
                }
                best.0 as i32
            })
            .collect();
        Ok(Self {
            shape,
            data: Data::I32(Arc::new(output)),
        })
    }

    fn pad(value: &Self, widths: &[(i32, i32)], mode: PadMode, _: &Context) -> Result<Self, Error> {
        let shape = value
            .shape
            .iter()
            .zip(widths)
            .map(|(&d, &(l, r))| d + l + r)
            .collect::<Vec<_>>();
        let output = (0..elements(&shape))
            .map(|flat| {
                let out = coordinates(flat, &shape);
                let mut input = Vec::new();
                for (axis, &coord) in out.iter().enumerate() {
                    let left = widths[axis].0 as usize;
                    if coord < left || coord >= left + value.shape[axis] as usize {
                        if mode == PadMode::Constant {
                            return 0.0;
                        }
                        input.push(if coord < left {
                            0
                        } else {
                            value.shape[axis] as usize - 1
                        });
                    } else {
                        input.push(coord - left);
                    }
                }
                value.get_f32(flat_index(&input, &value.shape))
            })
            .collect();
        Ok(Self::f32(output, &shape))
    }

    fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        context: &Context,
    ) -> Result<Self, Error> {
        if context.failure() == FailAt::Execution {
            context.bump(|c| c.execution_failures += 1);
            return Err(Error::backend("injected execution failure"));
        }
        let (batch, time, channels) = (
            input.shape[0] as usize,
            input.shape[1] as usize,
            input.shape[2] as usize,
        );
        let (out_channels, kernel, in_per_group) = (
            weight.shape[0] as usize,
            weight.shape[1] as usize,
            weight.shape[2] as usize,
        );
        let out_time = ((time as i32 + 2 * padding - dilation * (kernel as i32 - 1) - 1) / stride
            + 1) as usize;
        let mut output = vec![0.0; batch * out_time * out_channels];
        let out_per_group = out_channels / groups as usize;
        let nonzero = match &weight.data {
            Data::F32Sparse(values) => values.iter().map(|(&i, &v)| (i, v)).collect::<Vec<_>>(),
            _ => (0..elements(&weight.shape))
                .filter_map(|index| {
                    let value = weight.get_f32(index);
                    (value != 0.0).then_some((index, value))
                })
                .collect(),
        };
        for (flat, weight_value) in nonzero {
            let i = flat % in_per_group;
            let flat = flat / in_per_group;
            let k = flat % kernel;
            let o = flat / kernel;
            let group = o / out_per_group;
            for b in 0..batch {
                for t in 0..out_time {
                    let source = t as i32 * stride - padding + k as i32 * dilation;
                    if source >= 0 && source < time as i32 {
                        output[(b * out_time + t) * out_channels + o] += input.get_f32(
                            (b * time + source as usize) * channels + group * in_per_group + i,
                        ) * weight_value;
                    }
                }
            }
        }
        Ok(Self::f32(
            output,
            &[batch as i32, out_time as i32, out_channels as i32],
        ))
    }

    fn conv_transpose1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
        _: &Context,
    ) -> Result<Self, Error> {
        let (batch, time, in_channels) = (
            input.shape[0] as usize,
            input.shape[1] as usize,
            input.shape[2] as usize,
        );
        let (out_channels, kernel, in_per_group) = (
            weight.shape[0] as usize,
            weight.shape[1] as usize,
            weight.shape[2] as usize,
        );
        let out_time = ((time as i32 - 1) * stride - 2 * padding
            + dilation * (kernel as i32 - 1)
            + output_padding
            + 1) as usize;
        let mut output = vec![0.0; batch * out_time * out_channels];
        let in_group = in_channels / groups as usize;
        let out_group = out_channels / groups as usize;
        let nonzero = match &weight.data {
            Data::F32Sparse(values) => values.iter().map(|(&i, &v)| (i, v)).collect::<Vec<_>>(),
            _ => (0..elements(&weight.shape))
                .filter_map(|index| {
                    let value = weight.get_f32(index);
                    (value != 0.0).then_some((index, value))
                })
                .collect(),
        };
        for (flat, weight_value) in nonzero {
            let local_i = flat % in_per_group;
            let flat = flat / in_per_group;
            let k = flat % kernel;
            let o = flat / kernel;
            let group = o / out_group;
            let i = group * in_group + local_i;
            for b in 0..batch {
                for t in 0..time {
                    let target = t as i32 * stride - padding + k as i32 * dilation;
                    if target >= 0 && target < out_time as i32 {
                        output[(b * out_time + target as usize) * out_channels + o] +=
                            input.get_f32((b * time + t) * in_channels + i) * weight_value;
                    }
                }
            }
        }
        Ok(Self::f32(
            output,
            &[batch as i32, out_time as i32, out_channels as i32],
        ))
    }

    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        _: &Context,
    ) -> Result<Self, Error> {
        let input_width = *input.shape.last().unwrap() as usize;
        let output_width = weight.shape[0] as usize;
        let rows = elements(&input.shape) / input_width;
        let mut shape = input.shape.clone();
        *shape.last_mut().unwrap() = output_width as i32;
        if weight.is_zero() && bias.is_none_or(Self::is_zero) {
            return Ok(Self::zero(&shape));
        }
        let mut output = vec![0.0; rows * output_width];
        for row in 0..rows {
            for out in 0..output_width {
                let mut value = bias.map_or(0.0, |b| b.get_f32(out));
                for i in 0..input_width {
                    value += input.get_f32(row * input_width + i)
                        * weight.get_f32(out * input_width + i);
                }
                output[row * output_width + out] = value;
            }
        }
        Ok(Self::f32(output, &shape))
    }

    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        _: &Context,
    ) -> Result<Self, Error> {
        let width = *input.shape.last().unwrap() as usize;
        let rows = elements(&input.shape) / width;
        let mut output = vec![0.0; elements(&input.shape)];
        for row in 0..rows {
            let mean = (0..width)
                .map(|i| input.get_f32(row * width + i))
                .sum::<f32>()
                / width as f32;
            let variance = (0..width)
                .map(|i| {
                    let d = input.get_f32(row * width + i) - mean;
                    d * d
                })
                .sum::<f32>()
                / width as f32;
            for i in 0..width {
                let normalized =
                    (input.get_f32(row * width + i) - mean) / (variance + epsilon).sqrt();
                output[row * width + i] = normalized * weight.map_or(1.0, |w| w.get_f32(i))
                    + bias.map_or(0.0, |b| b.get_f32(i));
            }
        }
        Ok(Self::f32(output, &input.shape))
    }
    fn gelu(input: &Self, _: &Context) -> Result<Self, Error> {
        Ok(input.map(|x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())))
    }
    fn elu(input: &Self, alpha: f32, _: &Context) -> Result<Self, Error> {
        Ok(input.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }))
    }
    fn rope(
        input: &Self,
        dimensions: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
        _: &Context,
    ) -> Result<Self, Error> {
        if input.is_zero() {
            return Ok(input.clone());
        }
        let mut output = input.dense_f32();
        let width = *input.shape.last().unwrap() as usize;
        let rows = elements(&input.shape) / width;
        let dims = dimensions as usize;
        for row in 0..rows {
            let position = (offset
                + (row / (input.shape.last().copied().unwrap_or(1) as usize)) as i32)
                as f32
                * scale;
            for pair in 0..dims / 2 {
                let (a, b) = if traditional {
                    (2 * pair, 2 * pair + 1)
                } else {
                    (pair, pair + dims / 2)
                };
                let theta = position / base.powf(2.0 * pair as f32 / dims as f32);
                let (sin, cos) = theta.sin_cos();
                let x = output[row * width + a];
                let y = output[row * width + b];
                output[row * width + a] = x * cos - y * sin;
                output[row * width + b] = x * sin + y * cos;
            }
        }
        Ok(Self::f32(output, &input.shape))
    }
    fn scaled_dot_product_attention(
        queries: &Self,
        _keys: &Self,
        values: &Self,
        _scale: f32,
        _mask: AttentionMask<'_, Self>,
        _: &Context,
    ) -> Result<Self, Error> {
        let mut shape = queries.shape.clone();
        *shape.last_mut().unwrap() = *values.shape.last().unwrap();
        if values.is_zero() {
            Ok(Self::zero(&shape))
        } else {
            Err(Error::backend(
                "nonzero reference attention is outside this fixture",
            ))
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NullOperator;
impl Parameterized<NumericTensor> for NullOperator {
    fn visit_parameters<'a, V>(&'a self, _: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, _: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
    }
    fn set_trainable(&mut self, _: bool) {}
}
impl LinearOperator<NumericTensor> for NullOperator {
    fn forward(&mut self, _: &NumericTensor, _: &Context) -> Result<NumericTensor, Error> {
        Err(Error::backend("unused operator"))
    }
}
impl EmbeddingOperator<NumericTensor> for NullOperator {
    fn forward(&mut self, _: &NumericTensor, _: &Context) -> Result<NumericTensor, Error> {
        Err(Error::backend("unused operator"))
    }
    fn as_linear(&mut self, _: &NumericTensor, _: &Context) -> Result<NumericTensor, Error> {
        Err(Error::backend("unused operator"))
    }
}
impl NormalizationOperator<NumericTensor> for NullOperator {
    fn forward(&mut self, input: &NumericTensor, _: &Context) -> Result<NumericTensor, Error> {
        Ok(input.clone())
    }
}
impl RotaryOperator<NumericTensor> for NullOperator {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: RotaryPosition<'_, NumericTensor>,
        _: &Context,
    ) -> Result<NumericTensor, Error> {
        Ok(input.clone())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceBackend;
impl NeuralBackend for ReferenceBackend {
    type Tensor = NumericTensor;
    type Linear = NullOperator;
    type Embedding = NullOperator;
    type Normalization = NullOperator;
    type Rotary = NullOperator;
    type ParallelContext = ();
    fn linear(_: LinearSpec, _: &Context) -> Result<Self::Linear, Error> {
        Ok(NullOperator)
    }
    fn embedding(_: EmbeddingSpec, _: &Context) -> Result<Self::Embedding, Error> {
        Ok(NullOperator)
    }
    fn normalization(
        _: NormalizationConstructionSpec,
        _: &Context,
    ) -> Result<Self::Normalization, Error> {
        Ok(NullOperator)
    }
    fn rotary(_: RotarySpec, _: &Context) -> Result<Self::Rotary, Error> {
        Ok(NullOperator)
    }
    fn silu(input: NumericTensor, _: &Context) -> Result<NumericTensor, Error> {
        Ok(input.map(|x| x / (1.0 + (-x).exp())))
    }
    fn gated_product(
        gate: NumericTensor,
        up: NumericTensor,
        _: GatedProductPolicy,
        context: &Context,
    ) -> Result<NumericTensor, Error> {
        gate.multiply(&up, context)
    }
    fn attention(
        queries: NumericTensor,
        keys: NumericTensor,
        values: NumericTensor,
        scale: f32,
        mask: Option<&NumericTensor>,
        context: &Context,
    ) -> Result<NumericTensor, Error> {
        NumericTensor::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            mask.map_or(AttentionMask::None, AttentionMask::Tensor),
            context,
        )
    }
    fn sliding_window_attention(
        queries: NumericTensor,
        keys: NumericTensor,
        values: NumericTensor,
        scale: f32,
        _: i32,
        _: i32,
        context: &Context,
    ) -> Result<NumericTensor, Error> {
        NumericTensor::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            AttentionMask::Causal,
            context,
        )
    }
    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        _: &Context,
    ) -> Result<NumericTensor, Error> {
        let mut values = Vec::with_capacity((sequence * sequence) as usize);
        for q in 0..sequence {
            for k in 0..sequence {
                values.push(
                    if k <= q + offset && window.is_none_or(|w| q + offset - k < w) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    },
                );
            }
        }
        Ok(NumericTensor::f32(values, &[sequence, sequence]))
    }
    fn row_parallel_linear(
        linear: &mut NullOperator,
        input: &NumericTensor,
        _: &(),
        context: &Context,
    ) -> Result<NumericTensor, Error> {
        LinearOperator::forward(linear, input, context)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("injected {0} failure")]
pub struct BackendError(&'static str);

pub struct Materialization {
    weight: Option<NumericTensor>,
    retained_leases: Vec<RetainedLease>,
    context: Context,
}
impl Drop for Materialization {
    fn drop(&mut self) {
        self.context.bump(|c| c.guard_drops += 1);
    }
}

struct RetainedLease {
    _lease: CheckpointLease,
    context: Context,
}

impl RetainedLease {
    fn new(lease: CheckpointLease, context: &Context) -> Self {
        context.bump(|c| {
            c.live_encoded_leases += 1;
            c.max_live_encoded_leases = c.max_live_encoded_leases.max(c.live_encoded_leases);
        });
        Self {
            _lease: lease,
            context: context.clone(),
        }
    }
}

impl Drop for RetainedLease {
    fn drop(&mut self) {
        self.context.bump(|c| {
            c.live_encoded_leases -= 1;
            c.encoded_lease_drops += 1;
        });
    }
}

fn tensor_from_lease(lease: &CheckpointLease) -> Result<NumericTensor, BackendError> {
    let shape = lease
        .output_shape()
        .iter()
        .map(|&d| d as i32)
        .collect::<Vec<_>>();
    let bytes = lease.encoded_bytes().ok_or(BackendError("read"))?;
    let mut sparse = BTreeMap::new();
    for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes(*chunk);
        if value != 0.0 {
            sparse.insert(index, value);
        }
    }
    Ok(NumericTensor::sparse(&shape, sparse))
}

fn tensor_from_cached_lease(
    lease: &CheckpointLease,
    context: &Context,
) -> Result<NumericTensor, BackendError> {
    let key = format!("{:?}|{:?}", lease.metadata(), lease.selection());
    let cached = context.weight_cache.lock().unwrap().get(&key).cloned();
    if let Some(tensor) = cached {
        context.bump(|c| c.materialized_cache_hits += 1);
        return Ok(tensor);
    }
    let tensor = tensor_from_lease(lease)?;
    context
        .weight_cache
        .lock()
        .unwrap()
        .insert(key, tensor.clone());
    Ok(tensor)
}

fn lower(
    recipe: &DerivedWeightRecipe,
    source: &dyn CheckpointSource,
    context: &Context,
) -> Result<(NumericTensor, Vec<RetainedLease>), BackendError> {
    match recipe {
        DerivedWeightRecipe::Source { key, selection } => {
            context.bump(|c| c.direct_recipes += 1);
            if context.failure() == FailAt::Read {
                return Err(BackendError("read"));
            }
            let metadata = source
                .source_metadata(key)
                .map_err(|_| BackendError("read"))?;
            let cache_key = format!("{metadata:?}|{selection:?}");
            let cached = context
                .weight_cache
                .lock()
                .unwrap()
                .get(&cache_key)
                .cloned();
            if let Some(tensor) = cached {
                context.bump(|c| c.materialized_cache_hits += 1);
                return Ok((tensor, Vec::new()));
            }
            context.bump(|c| c.payload_reads += 1);
            let lease = source
                .acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: selection.clone(),
                    policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
                })
                .map_err(|_| BackendError("read"))?;
            let tensor = tensor_from_cached_lease(&lease, context)?;
            Ok((tensor, vec![RetainedLease::new(lease, context)]))
        }
        DerivedWeightRecipe::Transpose { input, axes } => {
            context.bump(|c| c.transpose_recipes += 1);
            let (tensor, leases) = lower(input, source, context)?;
            if context.failure() == FailAt::Recipe {
                return Err(BackendError("recipe"));
            }
            let tensor = tensor
                .transpose_axes(&axes.iter().map(|&a| a as i32).collect::<Vec<_>>(), context)
                .map_err(|_| BackendError("recipe"))?;
            Ok((tensor, leases))
        }
        _ => Err(BackendError("unsupported recipe")),
    }
}

impl ParameterBackend for ReferenceBackend {
    type Parameter = NumericTensor;
    type MaterializedWeight = NumericTensor;
    type MaterializationContext = Context;
    type Materialization = Materialization;
    type ParameterError = BackendError;
    fn preflight_recipe(
        recipe: &DerivedWeightRecipe,
        _: &dyn CheckpointSource,
    ) -> Result<(), BackendError> {
        let context = active();
        context.bump(|c| c.preflight += 1);
        if context.failure() == FailAt::Capability {
            return Err(BackendError("capability"));
        }
        fn supported(recipe: &DerivedWeightRecipe) -> bool {
            match recipe {
                DerivedWeightRecipe::Source { .. } => true,
                DerivedWeightRecipe::Transpose { input, .. } => supported(input),
                _ => false,
            }
        }
        if supported(recipe) {
            Ok(())
        } else {
            Err(BackendError("capability"))
        }
    }
    fn materialize(
        lease: CheckpointLease,
        context: &Context,
    ) -> Result<Materialization, BackendError> {
        context.bump(|c| c.payload_reads += 1);
        let retained = RetainedLease::new(lease, context);
        if context.failure() == FailAt::Read {
            return Err(BackendError("read"));
        }
        let weight = tensor_from_cached_lease(&retained._lease, context)?;
        context.bump(|c| {
            c.direct_materializations += 1;
            c.materializations += 1;
        });
        Ok(Materialization {
            weight: Some(weight),
            retained_leases: vec![retained],
            context: context.clone(),
        })
    }
    fn materialize_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        context: &Context,
    ) -> Result<Materialization, BackendError> {
        let (weight, retained_leases) = lower(recipe, source, context)?;
        context.bump(|c| c.materializations += 1);
        Ok(Materialization {
            weight: Some(weight),
            retained_leases,
            context: context.clone(),
        })
    }
    fn materialized_weight(materialization: &Materialization) -> &NumericTensor {
        materialization.weight.as_ref().unwrap()
    }
    fn finish_materialization(
        mut materialization: Materialization,
    ) -> Result<NumericTensor, BackendError> {
        materialization.context.bump(|c| {
            c.completions += 1;
            if !materialization.retained_leases.is_empty() && c.live_encoded_leases > 0 {
                c.completions_with_live_leases += 1;
            }
        });
        if materialization.context.failure() == FailAt::Completion {
            return Err(BackendError("completion"));
        }
        if materialization.context.failure() == FailAt::DelayedCompletion {
            materialization.context.delay_first_completion();
        }
        Ok(materialization.weight.take().unwrap())
    }
    fn share_materialized_weight(weight: &NumericTensor) -> Result<NumericTensor, BackendError> {
        Ok(weight.clone())
    }
    fn validate_bind(
        parameter: &NumericTensor,
        weight: &NumericTensor,
    ) -> Result<(), BackendError> {
        let context = active();
        context.bump(|c| c.validations += 1);
        if context.failure() == FailAt::Validate {
            return Err(BackendError("validate"));
        }
        if parameter.shape != weight.shape {
            return Err(BackendError("shape"));
        }
        Ok(())
    }
    fn bind(parameter: &mut NumericTensor, weight: NumericTensor) {
        active().bump(|c| c.binds += 1);
        *parameter = weight;
    }
}
