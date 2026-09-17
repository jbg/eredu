//! One deterministic native equation, also executed by the cold fixed counter.
use smallvec::SmallVec;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Unary {
    Square,
    Rsqrt,
    Sigmoid,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Binary {
    Add,
    Multiply,
    Divide,
}

pub(crate) trait Worker {
    type Value;
    type Error;
    fn shape<'a>(&self, input: &'a Self::Value) -> &'a [i32];
    fn alias(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn f32(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn cast_like(
        &mut self,
        input: &Self::Value,
        source: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    fn scalar(&mut self, value: f32) -> Result<Self::Value, Self::Error>;
    fn zeros_like(
        &mut self,
        shape: &[i32],
        source: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    fn reshape(&mut self, input: &Self::Value, shape: &[i32]) -> Result<Self::Value, Self::Error>;
    fn transpose(&mut self, input: &Self::Value, axes: &[i32]) -> Result<Self::Value, Self::Error>;
    fn slice(
        &mut self,
        input: &Self::Value,
        axis: usize,
        start: i32,
        end: i32,
    ) -> Result<Self::Value, Self::Error>;
    fn unary(&mut self, kind: Unary, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn binary(
        &mut self,
        kind: Binary,
        a: &Self::Value,
        b: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    fn matmul(&mut self, a: &Self::Value, b: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn mean_last(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn sum(
        &mut self,
        input: &Self::Value,
        axis: usize,
        keep: bool,
    ) -> Result<Self::Value, Self::Error>;
    fn softmax_last(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn repeat<F>(
        &mut self,
        count: usize,
        mut value: Self::Value,
        mut step: F,
    ) -> Result<Self::Value, Self::Error>
    where
        Self: Sized,
        F: FnMut(&mut Self, &Self::Value) -> Result<Self::Value, Self::Error>,
    {
        for _ in 0..count {
            value = step(self, &value)?;
        }
        Ok(value)
    }
}

pub(crate) struct Split<V> {
    pub pre: V,
    pub post: V,
    pub combination: V,
}
fn empty<W: Worker>(worker: &W, value: &W::Value) -> bool {
    worker.shape(value).contains(&0)
}
fn rms<W: Worker>(worker: &mut W, value: &W::Value, epsilon: f32) -> Result<W::Value, W::Error> {
    if empty(worker, value) {
        return worker.alias(value);
    }
    let square = worker.unary(Unary::Square, value)?;
    let variance = worker.mean_last(&square)?;
    let epsilon = worker.scalar(epsilon)?;
    let shifted = worker.binary(Binary::Add, &variance, &epsilon)?;
    let inverse = worker.unary(Unary::Rsqrt, &shifted)?;
    worker.binary(Binary::Multiply, value, &inverse)
}
fn normalize<W: Worker>(
    worker: &mut W,
    value: &W::Value,
    axis: usize,
    epsilon: f32,
) -> Result<W::Value, W::Error> {
    if empty(worker, value) {
        return worker.alias(value);
    }
    let sum = worker.sum(value, axis, true)?;
    let epsilon = worker.scalar(epsilon)?;
    let shifted = worker.binary(Binary::Add, &sum, &epsilon)?;
    worker.binary(Binary::Divide, value, &shifted)
}
fn scalar_at<W: Worker>(
    worker: &mut W,
    input: &W::Value,
    index: i32,
) -> Result<W::Value, W::Error> {
    let selected = worker.slice(input, 0, index, index + 1)?;
    worker.reshape(&selected, &[])
}

pub(crate) fn split<W: Worker>(
    worker: &mut W,
    mixes: &W::Value,
    scale: &W::Value,
    base: &W::Value,
    streams: i32,
    iterations: usize,
    epsilon: f32,
) -> Result<Split<W::Value>, W::Error> {
    // The real original collapse has two prefix axes. Ordinary standalone
    // split keeps arbitrary leading ranks through the same spill-capable type.
    let shape = worker.shape(mixes);
    let width = shape[shape.len() - 1];
    let mut vector_shape: SmallVec<[i32; 4]> = SmallVec::from_slice(&shape[..shape.len() - 1]);
    vector_shape.push(streams);
    let mut matrix_shape = vector_shape.clone();
    matrix_shape.push(streams);
    let mixes = worker.f32(mixes)?;
    let mixes = worker.reshape(&mixes, &[-1, width])?;
    let scale = worker.f32(scale)?;
    let base = worker.f32(base)?;
    let pre_logits = worker.slice(&mixes, 1, 0, streams)?;
    let pre_base = worker.slice(&base, 0, 0, streams)?;
    let pre_scale = scalar_at(worker, &scale, 0)?;
    let pre = worker.binary(Binary::Multiply, &pre_logits, &pre_scale)?;
    let pre = worker.binary(Binary::Add, &pre, &pre_base)?;
    let pre = worker.unary(Unary::Sigmoid, &pre)?;
    let constant = worker.scalar(epsilon)?;
    let pre = worker.binary(Binary::Add, &pre, &constant)?;
    let post_logits = worker.slice(&mixes, 1, streams, 2 * streams)?;
    let post_base = worker.slice(&base, 0, streams, 2 * streams)?;
    let post_scale = scalar_at(worker, &scale, 1)?;
    let post = worker.binary(Binary::Multiply, &post_logits, &post_scale)?;
    let post = worker.binary(Binary::Add, &post, &post_base)?;
    let post = worker.unary(Unary::Sigmoid, &post)?;
    let constant = worker.scalar(2.0)?;
    let post = worker.binary(Binary::Multiply, &post, &constant)?;
    let combination = worker.slice(&mixes, 1, 2 * streams, width)?;
    let scale = scalar_at(worker, &scale, 2)?;
    let combination = worker.binary(Binary::Multiply, &combination, &scale)?;
    let base = worker.slice(&base, 0, 2 * streams, width)?;
    let combination = worker.binary(Binary::Add, &combination, &base)?;
    let combination = worker.reshape(&combination, &[-1, streams, streams])?;
    let combination = worker.softmax_last(&combination)?;
    let constant = worker.scalar(epsilon)?;
    let combination = worker.binary(Binary::Add, &combination, &constant)?;
    let combination = normalize(worker, &combination, 1, epsilon)?;
    // A complete iteration preserves shape. The counter executes this closure
    // once and scales checked deltas; native execution uses the ordinary loop.
    let combination = worker.repeat(iterations - 1, combination, |worker, value| {
        let row = normalize(worker, value, 2, epsilon)?;
        normalize(worker, &row, 1, epsilon)
    })?;
    Ok(Split {
        pre: worker.reshape(&pre, &vector_shape)?,
        post: worker.reshape(&post, &vector_shape)?,
        combination: worker.reshape(&combination, &matrix_shape)?,
    })
}

pub(crate) fn collapse<W: Worker>(
    worker: &mut W,
    residual: &W::Value,
    function: &W::Value,
    base: &W::Value,
    scale: &W::Value,
    dimensions: [i32; 4],
    iterations: usize,
    epsilon: f32,
    norm_epsilon: f32,
) -> Result<(W::Value, Split<W::Value>), W::Error> {
    let [b, t, s, h] = dimensions;
    let fp32 = worker.f32(residual)?;
    let flat = worker.reshape(&fp32, &[b, t, s * h])?;
    let normalized = rms(worker, &flat, norm_epsilon)?;
    let weights = worker.transpose(function, &[1, 0])?;
    let mixes = worker.matmul(&normalized, &weights)?;
    let split = split(worker, &mixes, scale, base, s, iterations, epsilon)?;
    // Exact blh,blhd->bld: identical stream contraction order; no parser,
    // path selection, label maps or temporary operand vectors.
    let pre = worker.reshape(&split.pre, &[b, t, 1, s])?;
    let reduced = worker.matmul(&pre, &fp32)?;
    let reduced = worker.reshape(&reduced, &[b, t, h])?;
    let reduced = worker.cast_like(&reduced, residual)?;
    Ok((reduced, split))
}

pub(crate) fn expand<W: Worker>(
    worker: &mut W,
    sublayer: &W::Value,
    residual: &W::Value,
    post: &W::Value,
    combination: &W::Value,
    dimensions: [i32; 4],
) -> Result<W::Value, W::Error> {
    let [b, t, s, h] = dimensions;
    let post = worker.reshape(post, &[b, t, s, 1])?;
    let sublayer_fp32 = worker.f32(sublayer)?;
    let column = worker.reshape(&sublayer_fp32, &[b, t, 1, h])?;
    let injected = worker.binary(Binary::Multiply, &post, &column)?;
    // Exact blji,bljd->blid: transpose the two stream axes, then contract j.
    let combination = worker.transpose(combination, &[0, 1, 3, 2])?;
    let fp32 = worker.f32(residual)?;
    let mixed = worker.matmul(&combination, &fp32)?;
    let output = worker.binary(Binary::Add, &injected, &mixed)?;
    worker.cast_like(&output, sublayer)
}

pub(crate) fn head_coefficients<W: Worker>(
    worker: &mut W,
    residual: &W::Value,
    function: &W::Value,
    base: &W::Value,
    scale: &W::Value,
    dimensions: [i32; 4],
    norm_epsilon: f32,
    epsilon: f32,
) -> Result<(W::Value, W::Value), W::Error> {
    let [b, t, s, h] = dimensions;
    let fp32 = worker.f32(residual)?;
    let flat = worker.reshape(&fp32, &[b, t, s * h])?;
    let normalized = rms(worker, &flat, norm_epsilon)?;
    let weights = worker.transpose(function, &[1, 0])?;
    let logits = worker.matmul(&normalized, &weights)?;
    let scaled = worker.binary(Binary::Multiply, &logits, scale)?;
    let shifted = worker.binary(Binary::Add, &scaled, base)?;
    let sigmoid = worker.unary(Unary::Sigmoid, &shifted)?;
    let epsilon = worker.scalar(epsilon)?;
    let pre = worker.binary(Binary::Add, &sigmoid, &epsilon)?;
    Ok((fp32, pre))
}

pub(crate) fn head_sum<W: Worker>(
    worker: &mut W,
    fp32: &W::Value,
    pre: &W::Value,
    original: &W::Value,
    dimensions: [i32; 4],
) -> Result<W::Value, W::Error> {
    let [b, t, s, h] = dimensions;
    if empty(worker, fp32) {
        return worker.zeros_like(&[b, t, h], original);
    }
    let pre = worker.reshape(pre, &[b, t, s, 1])?;
    let weighted = worker.binary(Binary::Multiply, &pre, fp32)?;
    let sum = worker.sum(&weighted, 2, false)?;
    worker.cast_like(&sum, original)
}
