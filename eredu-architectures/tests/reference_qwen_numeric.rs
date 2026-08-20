use std::cell::Cell;

use eredu_architectures::{deepseek, qwen};
use eredu_nn::{
    AttentionCache, AttentionMask, BlockwiseAttentionBackend, BlockwiseAttentionSpec,
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, EmbeddingOperator, EmbeddingSpec, Error,
    HyperConnection, HyperConnectionOperator, HyperConnectionSpec, HyperConnectionState, HyperHead,
    HyperHeadOperator, HyperHeadSpec, HyperNeuralBackend, Index, IndexedAttentionInput,
    LinearOperator, LinearSpec, LowRankProjection, LowRankProjectionSpec, NeuralBackend,
    NormalizationOperator, NormalizationSpec, PadMode, ParameterMetadata, ParameterSpec,
    ParameterVisitor, ParameterVisitorMut, Parameterized, PooledAttentionInput,
    PooledPositionInput, PoolingAttentionCache, PoolingOverlap, PoolingWindows, RotaryOperator,
    RotaryPosition, RotarySpec, RotarySubspace, RoutedNeuralBackend, RoutingOperator,
    RoutingResult, SwiGluExpertBankOperator, SwiGluExpertBankSpec, SwiGluExpertLayout, Tensor,
    TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{DeviceState, LayerRuntimeState, ResidentRuntime, RuntimeLayerState};

#[derive(Debug, Clone)]
struct NumericTensor {
    shape: Vec<i32>,
    data: Vec<f32>,
}

impl NumericTensor {
    fn new(shape: impl Into<Vec<i32>>, data: Vec<f32>) -> Self {
        let shape = shape.into();
        assert_eq!(elements(&shape), data.len());
        Self { shape, data }
    }

    fn zeros(shape: impl Into<Vec<i32>>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![0.0; elements(&shape)],
            shape,
        }
    }

    fn token_ids(ids: &[usize]) -> Self {
        Self::new(
            vec![1, i32::try_from(ids.len()).unwrap()],
            ids.iter().map(|id| *id as f32).collect(),
        )
    }

    fn axis_slice(&self, axis: usize, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.shape[axis] as usize);
        let mut shape = self.shape.clone();
        shape[axis] = (end - start) as i32;
        let mut output = Self::zeros(shape);
        for output_index in 0..output.data.len() {
            let mut coordinate = unravel(output_index, &output.shape);
            coordinate[axis] += start;
            output.data[output_index] = self.data[offset(&coordinate, &self.shape)];
        }
        output
    }

    fn map(&self, operation: impl Fn(f32) -> f32) -> Self {
        Self::new(
            self.shape.clone(),
            self.data.iter().copied().map(operation).collect(),
        )
    }

    fn zip(&self, rhs: &Self, operation: impl Fn(f32, f32) -> f32) -> Result<Self, Error> {
        if self.shape != rhs.shape {
            return Err(Error::backend(format!(
                "numeric tensor shape mismatch: {:?} versus {:?}",
                self.shape, rhs.shape
            )));
        }
        Ok(Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .copied()
                .zip(rhs.data.iter().copied())
                .map(|(left, right)| operation(left, right))
                .collect(),
        ))
    }
}

#[derive(Debug, Clone)]
struct NumericCompressedCache {
    state: Option<CompressedAttentionState<NumericTensor>>,
    offset: i32,
    block_size: Option<i32>,
}

impl NumericCompressedCache {
    fn resident() -> Self {
        Self {
            state: None,
            offset: 0,
            block_size: None,
        }
    }

    fn paged(block_size: i32) -> Self {
        assert!(block_size > 0);
        Self {
            state: None,
            offset: 0,
            block_size: Some(block_size),
        }
    }
}

impl CompressedAttentionCache<NumericTensor> for NumericCompressedCache {
    type Checkpoint = Self;

    fn offset(&self) -> i32 {
        self.offset
    }

    fn is_paged(&self) -> bool {
        self.block_size.is_some()
    }

    fn append(
        &mut self,
        state: CompressedAttentionState<NumericTensor>,
        context: &NumericContext,
    ) -> Result<CompressedAttentionView<NumericTensor>, Error> {
        if state.latent.shape.len() != 3
            || state.rotary.shape.len() != 3
            || state.latent.shape[..2] != state.rotary.shape[..2]
            || state.latent.shape[1] <= 0
        {
            return Err(Error::backend("numeric compressed-cache geometry mismatch"));
        }
        let appended = state.clone();
        self.state = Some(match self.state.take() {
            Some(previous) => CompressedAttentionState {
                latent: NumericTensor::concatenate(&[previous.latent, state.latent], 1, context)?,
                rotary: NumericTensor::concatenate(&[previous.rotary, state.rotary], 1, context)?,
            },
            None => state,
        });
        self.offset += appended.latent.shape[1];
        if self.is_paged() {
            Ok(CompressedAttentionView::Paged { appended })
        } else {
            Ok(CompressedAttentionView::Resident(
                self.state.as_ref().unwrap().clone(),
            ))
        }
    }

    fn visit_blocks<F>(
        &mut self,
        _: i32,
        _: &NumericContext,
        mut visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<NumericTensor>) -> Result<u64, Error>,
    {
        let block_size = self
            .block_size
            .ok_or_else(|| Error::backend("numeric compressed scan requires paging"))?;
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| Error::backend("numeric compressed scan requires state"))?;
        let mut scan = CompressedAttentionScan::default();
        let mut start = 0;
        while start < self.offset {
            let end = (start + block_size).min(self.offset);
            let block = CompressedAttentionBlock {
                start: start as i64,
                end: end as i64,
                state: CompressedAttentionState {
                    latent: state.latent.axis_slice(1, start as usize, end as usize),
                    rotary: state.rotary.axis_slice(1, start as usize, end as usize),
                },
            };
            scan.bytes +=
                (block.state.latent.data.len() + block.state.rotary.data.len()) as u64 * 4;
            scan.reconstruction_scratch_bytes =
                scan.reconstruction_scratch_bytes.max(visitor(block)?);
            scan.blocks += 1;
            start = end;
        }
        Ok(scan)
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint, _: &NumericContext) -> Result<(), Error> {
        self.clone_from(checkpoint);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn clear(&mut self) -> Result<(), Error> {
        self.state = None;
        self.offset = 0;
        Ok(())
    }
}

impl RuntimeLayerState<NumericBackend> for NumericCompressedCache {
    type RetainedValues<'a> = std::iter::Chain<
        std::option::IntoIter<&'a NumericTensor>,
        std::option::IntoIter<&'a NumericTensor>,
    >;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.state
            .as_ref()
            .map(|state| &state.latent)
            .into_iter()
            .chain(self.state.as_ref().map(|state| &state.rotary))
    }
}

#[derive(Debug, Clone)]
struct NumericPoolStream {
    ratio: i32,
    pending_values: Option<NumericTensor>,
    pending_gates: Option<NumericTensor>,
    pooled: Option<NumericTensor>,
    overlap_values: Option<NumericTensor>,
    overlap_gates: Option<NumericTensor>,
    processed: i32,
}

impl NumericPoolStream {
    fn new(ratio: i32) -> Self {
        Self {
            ratio,
            pending_values: None,
            pending_gates: None,
            pooled: None,
            overlap_values: None,
            overlap_gates: None,
            processed: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct NumericPoolingCache {
    local: Option<NumericTensor>,
    offset: i32,
    window: i32,
    attention_local_tokens: i32,
    streams: Vec<NumericPoolStream>,
}

impl NumericPoolingCache {
    fn new(window: i32, ratios: &[i32]) -> Self {
        Self {
            local: None,
            offset: 0,
            window,
            attention_local_tokens: 0,
            streams: ratios.iter().copied().map(NumericPoolStream::new).collect(),
        }
    }

    fn stream(&self, stream: u32) -> Result<&NumericPoolStream, Error> {
        self.streams
            .get(stream as usize)
            .ok_or_else(|| Error::backend("numeric pooling stream is absent"))
    }

    fn stream_mut(&mut self, stream: u32) -> Result<&mut NumericPoolStream, Error> {
        self.streams
            .get_mut(stream as usize)
            .ok_or_else(|| Error::backend("numeric pooling stream is absent"))
    }
}

impl PoolingAttentionCache<NumericTensor> for NumericPoolingCache {
    type Checkpoint = Self;

    fn offset(&self) -> i32 {
        self.offset
    }

    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.streams.get(stream as usize).map(|stream| stream.ratio)
    }

    fn append_local(
        &mut self,
        keys: NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        if keys.shape.len() != 3 || keys.shape[1] <= 0 {
            return Err(Error::backend("numeric local cache geometry mismatch"));
        }
        self.offset += keys.shape[1];
        let full = match self.local.take() {
            Some(previous) => NumericTensor::concatenate(&[previous, keys], 1, context)?,
            None => keys,
        };
        self.attention_local_tokens = full.shape[1];
        let attention = full.clone();
        let retained = full.shape[1].min(self.window);
        self.local = Some(full.axis_slice(
            1,
            (full.shape[1] - retained) as usize,
            full.shape[1] as usize,
        ));
        Ok(attention)
    }

    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let keys = self.attention_local_tokens;
        let key_offset = offset + query_tokens - keys;
        let mut mask = NumericTensor::zeros(vec![query_tokens, keys]);
        for query in 0..query_tokens {
            for key in 0..keys {
                let query_position = offset + query;
                let key_position = key_offset + key;
                if key_position > query_position || key_position <= query_position - self.window {
                    mask.data[(query * keys + key) as usize] = -1.0e9;
                }
            }
        }
        Ok(mask)
    }

    fn accumulate_pooling_windows(
        &mut self,
        stream: u32,
        values: NumericTensor,
        gates: NumericTensor,
        absolute_offset: i32,
        context: &NumericContext,
    ) -> Result<PoolingWindows<NumericTensor>, Error> {
        let stream = self.stream_mut(stream)?;
        if absolute_offset != stream.processed
            || values.shape.len() != 3
            || values.shape[..2] != gates.shape[..2]
        {
            return Err(Error::backend("numeric pooling accumulation mismatch"));
        }
        let newly_added = values.shape[1];
        let values = match stream.pending_values.take() {
            Some(previous) => NumericTensor::concatenate(&[previous, values], 1, context)?,
            None => values,
        };
        let gates = match stream.pending_gates.take() {
            Some(previous) => NumericTensor::concatenate(&[previous, gates], 1, context)?,
            None => gates,
        };
        let usable = values.shape[1] / stream.ratio * stream.ratio;
        let ready_values = values.axis_slice(1, 0, usable as usize);
        let ready_gates = gates.axis_slice(1, 0, usable as usize);
        if usable < values.shape[1] {
            stream.pending_values =
                Some(values.axis_slice(1, usable as usize, values.shape[1] as usize));
            stream.pending_gates =
                Some(gates.axis_slice(1, usable as usize, gates.shape[1] as usize));
        }
        let base_position = absolute_offset - (values.shape[1] - newly_added);
        stream.processed += newly_added;
        Ok(PoolingWindows {
            values: ready_values,
            gates: ready_gates,
            base_position,
        })
    }

    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: NumericTensor,
        gates: NumericTensor,
    ) -> Result<PoolingOverlap<NumericTensor>, Error> {
        let stream = self.stream_mut(stream)?;
        Ok(PoolingOverlap {
            values: stream.overlap_values.replace(values),
            gates: stream.overlap_gates.replace(gates),
        })
    }

    fn append_pooled(
        &mut self,
        stream: u32,
        values: NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let stream = self.stream_mut(stream)?;
        let empty_shape = vec![values.shape[0], 0, values.shape[2]];
        if values.shape[1] > 0 {
            stream.pooled = Some(match stream.pooled.take() {
                Some(previous) => NumericTensor::concatenate(&[previous, values], 1, context)?,
                None => values,
            });
        }
        Ok(stream
            .pooled
            .clone()
            .unwrap_or_else(|| NumericTensor::zeros(empty_shape)))
    }

    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        _: &NumericContext,
    ) -> Result<Option<NumericTensor>, Error> {
        let stream = self.stream(stream)?;
        let pooled = stream.pooled.as_ref().map_or(0, |pooled| pooled.shape[1]);
        if pooled == 0 || query_tokens == 1 {
            return Ok(None);
        }
        let mut mask = NumericTensor::zeros(vec![query_tokens, pooled]);
        for query in 0..query_tokens {
            let visible = (offset + query + 1) / stream.ratio;
            for position in visible..pooled {
                mask.data[(query * pooled + position) as usize] = -1.0e9;
            }
        }
        Ok(Some(mask))
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint, _: &NumericContext) -> Result<(), Error> {
        self.clone_from(checkpoint);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn clear(&mut self) -> Result<(), Error> {
        self.local = None;
        self.offset = 0;
        self.attention_local_tokens = 0;
        for stream in &mut self.streams {
            *stream = NumericPoolStream::new(stream.ratio);
        }
        Ok(())
    }
}

impl RuntimeLayerState<NumericBackend> for NumericPoolingCache {
    type RetainedValues<'a> = std::iter::Empty<&'a NumericTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
}

fn elements(shape: &[i32]) -> usize {
    shape
        .iter()
        .map(|dimension| usize::try_from(*dimension).unwrap())
        .product()
}

fn strides(shape: &[i32]) -> Vec<usize> {
    let mut stride = 1;
    let mut result = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        result[axis] = stride;
        stride *= usize::try_from(*dimension).unwrap();
    }
    result
}

fn unravel(mut index: usize, shape: &[i32]) -> Vec<usize> {
    let tensor_strides = strides(shape);
    tensor_strides
        .iter()
        .map(|stride| {
            let coordinate = index / stride;
            index %= stride;
            coordinate
        })
        .collect()
}

fn offset(coordinate: &[usize], shape: &[i32]) -> usize {
    coordinate
        .iter()
        .zip(strides(shape))
        .map(|(coordinate, stride)| coordinate * stride)
        .sum()
}

fn axis(axis: i32, rank: usize, insertion: bool) -> Result<usize, Error> {
    let limit = if insertion { rank + 1 } else { rank };
    let normalized = if axis < 0 {
        i32::try_from(limit).unwrap() + axis
    } else {
        axis
    };
    let normalized = usize::try_from(normalized).map_err(Error::backend)?;
    if normalized >= limit {
        return Err(Error::backend(format!(
            "axis {axis} is outside rank {rank}"
        )));
    }
    Ok(normalized)
}

fn unsupported<T>(operation: &str) -> Result<T, Error> {
    Err(Error::backend(format!(
        "numeric Qwen reference does not use {operation}"
    )))
}

impl Tensor for NumericTensor {
    type Context = NumericContext;

    fn shape(&self) -> &[i32] {
        &self.shape
    }

    fn unloaded_f32(shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::zeros(shape.to_vec()))
    }

    fn unloaded_i32(shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::zeros(shape.to_vec()))
    }

    fn from_f32_slice(values: &[f32], shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(shape.to_vec(), values.to_vec()))
    }

    fn full_f32(value: f32, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(shape.to_vec(), vec![value; elements(shape)]))
    }

    fn full_i32(value: i32, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(
            shape.to_vec(),
            vec![value as f32; elements(shape)],
        ))
    }

    fn add(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left + right)
    }

    fn subtract(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left - right)
    }

    fn multiply(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left * right)
    }

    fn multiply_scalar(&self, rhs: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value * rhs))
    }

    fn divide(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        self.zip(rhs, |left, right| left / right)
    }

    fn square(&self, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value * value))
    }

    fn maximum_scalar(&self, rhs: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value.max(rhs)))
    }

    fn softmax_axis(&self, selected: i32, _: bool, _: &NumericContext) -> Result<Self, Error> {
        let selected = axis(selected, self.shape.len(), false)?;
        let width = self.shape[selected] as usize;
        let mut output = self.clone();
        let mut base_shape = self.shape.clone();
        base_shape.remove(selected);
        for base_index in 0..elements(&base_shape) {
            let base = unravel(base_index, &base_shape);
            let values = (0..width)
                .map(|position| {
                    let mut coordinate = base.clone();
                    coordinate.insert(selected, position);
                    self.data[offset(&coordinate, &self.shape)]
                })
                .collect::<Vec<_>>();
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = values
                .iter()
                .map(|value| (*value - maximum).exp())
                .sum::<f32>();
            for (position, value) in values.into_iter().enumerate() {
                let mut coordinate = base.clone();
                coordinate.insert(selected, position);
                output.data[offset(&coordinate, &self.shape)] =
                    (value - maximum).exp() / denominator;
            }
        }
        Ok(output)
    }

    fn reshape(&self, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        let mut shape = shape.to_vec();
        let inferred = shape.iter().position(|dimension| *dimension == -1);
        if let Some(inferred) = inferred {
            if shape.iter().filter(|dimension| **dimension == -1).count() != 1 {
                return Err(Error::backend("numeric reshape has multiple inferred axes"));
            }
            let known = shape
                .iter()
                .filter(|dimension| **dimension != -1)
                .map(|dimension| usize::try_from(*dimension).unwrap())
                .product::<usize>();
            shape[inferred] = i32::try_from(self.data.len() / known).map_err(Error::backend)?;
        }
        if elements(&shape) != self.data.len() {
            return Err(Error::backend("numeric reshape changes element count"));
        }
        Ok(Self::new(shape, self.data.clone()))
    }

    fn broadcast_to(&self, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        if shape.len() < self.shape.len() {
            return Err(Error::backend("numeric broadcast cannot reduce rank"));
        }
        let leading = shape.len() - self.shape.len();
        for (source, target) in self.shape.iter().zip(&shape[leading..]) {
            if *source != 1 && source != target {
                return Err(Error::backend("numeric broadcast geometry mismatch"));
            }
        }
        let mut output = Self::zeros(shape.to_vec());
        for output_index in 0..output.data.len() {
            let coordinate = unravel(output_index, shape);
            let source = self
                .shape
                .iter()
                .enumerate()
                .map(|(selected, dimension)| {
                    if *dimension == 1 {
                        0
                    } else {
                        coordinate[leading + selected]
                    }
                })
                .collect::<Vec<_>>();
            output.data[output_index] = self.data[offset(&source, &self.shape)];
        }
        Ok(output)
    }

    fn transpose_axes(&self, axes: &[i32], _: &NumericContext) -> Result<Self, Error> {
        if axes.len() != self.shape.len() {
            return Err(Error::backend("numeric transpose rank mismatch"));
        }
        let axes = axes
            .iter()
            .map(|selected| axis(*selected, self.shape.len(), false))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = vec![false; axes.len()];
        for selected in &axes {
            if std::mem::replace(&mut seen[*selected], true) {
                return Err(Error::backend("numeric transpose repeats an axis"));
            }
        }
        let shape = axes
            .iter()
            .map(|selected| self.shape[*selected])
            .collect::<Vec<_>>();
        let mut output = Self::zeros(shape);
        for output_index in 0..output.data.len() {
            let output_coordinate = unravel(output_index, &output.shape);
            let mut input_coordinate = vec![0; axes.len()];
            for (output_axis, input_axis) in axes.iter().enumerate() {
                input_coordinate[*input_axis] = output_coordinate[output_axis];
            }
            output.data[output_index] = self.data[offset(&input_coordinate, &self.shape)];
        }
        Ok(output)
    }

    fn swap_axes(&self, left: i32, right: i32, context: &NumericContext) -> Result<Self, Error> {
        let left = axis(left, self.shape.len(), false)?;
        let right = axis(right, self.shape.len(), false)?;
        let mut axes = (0..self.shape.len())
            .map(|axis| axis as i32)
            .collect::<Vec<_>>();
        axes.swap(left, right);
        self.transpose_axes(&axes, context)
    }

    fn transpose(&self, context: &NumericContext) -> Result<Self, Error> {
        if self.shape.len() != 2 {
            return Err(Error::backend("numeric transpose requires rank two"));
        }
        self.transpose_axes(&[1, 0], context)
    }

    fn expand_dims(&self, selected: i32, _: &NumericContext) -> Result<Self, Error> {
        let selected = axis(selected, self.shape.len(), true)?;
        let mut shape = self.shape.clone();
        shape.insert(selected, 1);
        Ok(Self::new(shape, self.data.clone()))
    }

    fn squeeze_axes(&self, axes: &[i32], _: &NumericContext) -> Result<Self, Error> {
        let mut axes = axes
            .iter()
            .map(|selected| axis(*selected, self.shape.len(), false))
            .collect::<Result<Vec<_>, _>>()?;
        axes.sort_unstable();
        axes.dedup();
        let mut shape = self.shape.clone();
        for selected in axes.into_iter().rev() {
            if shape[selected] != 1 {
                return Err(Error::backend("numeric squeeze selected a non-unit axis"));
            }
            shape.remove(selected);
        }
        Ok(Self::new(shape, self.data.clone()))
    }

    fn index(&self, indexes: &[Index], _: &NumericContext) -> Result<Self, Error> {
        if indexes.len() > self.shape.len() {
            return Err(Error::backend("numeric index exceeds tensor rank"));
        }
        let normalized = (0..self.shape.len())
            .map(|selected| {
                let dimension = self.shape[selected];
                match indexes.get(selected).copied().unwrap_or(Index::Full) {
                    Index::Full => Ok((false, 0, dimension)),
                    Index::At(position) => {
                        let position = if position < 0 {
                            dimension + position
                        } else {
                            position
                        };
                        if !(0..dimension).contains(&position) {
                            return Err(Error::backend("numeric index position is out of bounds"));
                        }
                        Ok((true, position, position + 1))
                    }
                    Index::Range(start, end) => {
                        let start = if start < 0 { dimension + start } else { start };
                        let end = if end < 0 { dimension + end } else { end };
                        if start < 0 || end < start || end > dimension {
                            return Err(Error::backend("numeric index range is out of bounds"));
                        }
                        Ok((false, start, end))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_shape = normalized
            .iter()
            .filter_map(|(removed, start, end)| (!removed).then_some(end - start))
            .collect::<Vec<_>>();
        let mut output = Self::zeros(output_shape);
        for output_index in 0..output.data.len() {
            let output_coordinate = unravel(output_index, &output.shape);
            let mut input_coordinate = Vec::with_capacity(self.shape.len());
            let mut retained = 0;
            for (removed, start, _) in &normalized {
                input_coordinate.push(if *removed {
                    *start as usize
                } else {
                    let coordinate = output_coordinate[retained] + *start as usize;
                    retained += 1;
                    coordinate
                });
            }
            output.data[output_index] = self.data[offset(&input_coordinate, &self.shape)];
        }
        Ok(output)
    }

    fn take_axis(&self, indexes: &Self, selected: i32, _: &NumericContext) -> Result<Self, Error> {
        let selected = axis(selected, self.shape.len(), false)?;
        if selected != 0 || indexes.shape.len() != 1 {
            return unsupported("take_axis geometry");
        }
        let row_width = self.data.len() / self.shape[0] as usize;
        let mut shape = vec![indexes.shape[0]];
        shape.extend_from_slice(&self.shape[1..]);
        let mut output = Self::zeros(shape);
        for (output_row, raw) in indexes.data.iter().copied().enumerate() {
            let row = raw as usize;
            if row >= self.shape[0] as usize || row as f32 != raw {
                return Err(Error::backend("numeric take index is invalid"));
            }
            output.data[output_row * row_width..(output_row + 1) * row_width]
                .copy_from_slice(&self.data[row * row_width..(row + 1) * row_width]);
        }
        Ok(output)
    }

    fn rope_with_frequencies(
        &self,
        dimensions: i32,
        traditional: bool,
        position_offset: i32,
        frequencies: &Self,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        let sequence = self.shape[self.shape.len() - 2];
        let half = dimensions as usize / 2;
        if frequencies.shape != [half as i32] {
            return Err(Error::backend("numeric explicit rotary frequency mismatch"));
        }
        let mut cosine = NumericTensor::zeros(vec![sequence, half as i32]);
        let mut sine = cosine.clone();
        for position in 0..sequence as usize {
            for frequency in 0..half {
                let theta =
                    (position_offset as f32 + position as f32) / frequencies.data[frequency];
                cosine.data[position * half + frequency] = theta.cos();
                sine.data[position * half + frequency] = theta.sin();
            }
        }
        rotary_embeddings(self, dimensions, traditional, &cosine, &sine)
    }

    fn concatenate(values: &[Self], selected: i32, _: &NumericContext) -> Result<Self, Error> {
        let first = values
            .first()
            .ok_or_else(|| Error::backend("numeric concatenate requires inputs"))?;
        let selected = axis(selected, first.shape.len(), false)?;
        let mut shape = first.shape.clone();
        shape[selected] = 0;
        for value in values {
            if value.shape.len() != shape.len()
                || value
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(current, dimension)| current != selected && *dimension != shape[current])
            {
                return Err(Error::backend("numeric concatenate shape mismatch"));
            }
            shape[selected] += value.shape[selected];
        }
        let mut output = Self::zeros(shape);
        let mut axis_base = 0;
        for value in values {
            for input_index in 0..value.data.len() {
                let input_coordinate = unravel(input_index, &value.shape);
                let mut output_coordinate = input_coordinate.clone();
                output_coordinate[selected] += axis_base;
                let output_index = offset(&output_coordinate, &output.shape);
                output.data[output_index] = value.data[input_index];
            }
            axis_base += value.shape[selected] as usize;
        }
        Ok(output)
    }

    fn stack(values: &[Self], selected: i32, context: &NumericContext) -> Result<Self, Error> {
        let first = values
            .first()
            .ok_or_else(|| Error::backend("numeric stack requires inputs"))?;
        let selected = axis(selected, first.shape.len(), true)?;
        let expanded = values
            .iter()
            .map(|value| value.expand_dims(selected as i32, context))
            .collect::<Result<Vec<_>, _>>()?;
        Self::concatenate(&expanded, selected as i32, context)
    }

    fn matmul(lhs: &Self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
            return Err(Error::backend(
                "numeric matmul requires compatible matrices",
            ));
        }
        let rows = lhs.shape[0] as usize;
        let inner_width = lhs.shape[1] as usize;
        let columns = rhs.shape[1] as usize;
        let mut output = Self::zeros(vec![rows as i32, columns as i32]);
        for row in 0..rows {
            for column in 0..columns {
                output.data[row * columns + column] = (0..inner_width)
                    .map(|inner_index| {
                        lhs.data[row * inner_width + inner_index]
                            * rhs.data[inner_index * columns + column]
                    })
                    .sum();
            }
        }
        Ok(output)
    }

    fn sum_axis(
        value: &Self,
        selected: i32,
        keep_dims: bool,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        let selected = axis(selected, value.shape.len(), false)?;
        let mut shape = value.shape.clone();
        if keep_dims {
            shape[selected] = 1;
        } else {
            shape.remove(selected);
        }
        let mut output = Self::zeros(shape);
        for input_index in 0..value.data.len() {
            let mut coordinate = unravel(input_index, &value.shape);
            if keep_dims {
                coordinate[selected] = 0;
            } else {
                coordinate.remove(selected);
            }
            let output_index = offset(&coordinate, &output.shape);
            output.data[output_index] += value.data[input_index];
        }
        Ok(output)
    }

    fn argmin_axis(_: &Self, _: i32, _: bool, _: &NumericContext) -> Result<Self, Error> {
        unsupported("argmin_axis")
    }

    fn pad(_: &Self, _: &[(i32, i32)], _: PadMode, _: &NumericContext) -> Result<Self, Error> {
        unsupported("pad")
    }

    fn conv1d(
        _: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        unsupported("conv1d")
    }

    fn conv_transpose1d(
        _: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        unsupported("conv_transpose1d")
    }

    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        linear(input, weight, bias)
    }

    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        let width = *input
            .shape
            .last()
            .ok_or_else(|| Error::backend("numeric layer norm requires rank"))?
            as usize;
        let mut output = input.clone();
        for (input_row, output_row) in input
            .data
            .chunks_exact(width)
            .zip(output.data.chunks_exact_mut(width))
        {
            let mean = input_row.iter().sum::<f32>() / width as f32;
            let variance = input_row
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f32>()
                / width as f32;
            for index in 0..width {
                let scale = weight.map_or(1.0, |weight| weight.data[index]);
                let shift = bias.map_or(0.0, |bias| bias.data[index]);
                output_row[index] =
                    (input_row[index] - mean) / (variance + epsilon).sqrt() * scale + shift;
            }
        }
        Ok(output)
    }

    fn gelu(input: &Self, _: &NumericContext) -> Result<Self, Error> {
        Ok(input.map(|value| {
            0.5 * value
                * (1.0
                    + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value.powi(3)))
                        .tanh())
        }))
    }

    fn elu(input: &Self, alpha: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(input.map(|value| {
            if value >= 0.0 {
                value
            } else {
                alpha * value.exp_m1()
            }
        }))
    }

    fn rope(
        input: &Self,
        dimensions: i32,
        traditional: bool,
        base: f32,
        _: f32,
        position_offset: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        rotary_offset(input, dimensions, traditional, base, position_offset)
    }

    fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask<'_, Self>,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        let mask = match mask {
            AttentionMask::Tensor(mask) => Some(mask),
            AttentionMask::None | AttentionMask::Causal => None,
        };
        attention(queries, keys, values, scale, mask, None, 0)
    }
}

fn deterministic_values(spec: &ParameterSpec, length: usize, norm: bool) -> Vec<f32> {
    let mut hash = 2_166_136_261_u32;
    for byte in spec.id.as_str().bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    (0..length)
        .map(|index| {
            let mixed = hash
                .wrapping_add((index as u32).wrapping_mul(747_796_405))
                .rotate_left((index % 23) as u32);
            let centered = (mixed % 2001) as f32 - 1000.0;
            if norm {
                1.0 + centered / 50_000.0
            } else {
                centered / 25_000.0
            }
        })
        .collect()
}

fn parameter(spec: &ParameterSpec, shape: Vec<i32>, norm: bool) -> NumericTensor {
    NumericTensor::new(
        shape.clone(),
        deterministic_values(spec, elements(&shape), norm),
    )
}

fn visit<'a, V>(metadata: &ParameterMetadata, value: &'a NumericTensor, visitor: &mut V)
where
    V: ParameterVisitor<'a, NumericTensor>,
{
    visitor.visit(metadata.clone(), value);
}

fn visit_mut<'a, V>(metadata: &ParameterMetadata, value: &'a mut NumericTensor, visitor: &mut V)
where
    V: ParameterVisitorMut<'a, NumericTensor>,
{
    visitor.visit_mut(metadata.clone(), value);
}

fn linear(
    input: &NumericTensor,
    weight: &NumericTensor,
    bias: Option<&NumericTensor>,
) -> Result<NumericTensor, Error> {
    if weight.shape.len() != 2
        || input.shape.last() != weight.shape.get(1)
        || bias.is_some_and(|bias| bias.shape != [weight.shape[0]])
    {
        return Err(Error::backend(format!(
            "numeric linear geometry mismatch: input={:?}, weight={:?}, bias={:?}",
            input.shape,
            weight.shape,
            bias.map(|bias| bias.shape.as_slice())
        )));
    }
    let input_width = weight.shape[1] as usize;
    let output_width = weight.shape[0] as usize;
    let rows = input.data.len() / input_width;
    let mut shape = input.shape.clone();
    *shape.last_mut().unwrap() = weight.shape[0];
    let mut output = NumericTensor::zeros(shape);
    for row in 0..rows {
        for output_column in 0..output_width {
            let mut value = bias.map_or(0.0, |bias| bias.data[output_column]);
            for input_column in 0..input_width {
                value += input.data[row * input_width + input_column]
                    * weight.data[output_column * input_width + input_column];
            }
            output.data[row * output_width + output_column] = value;
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct NumericLinear {
    weight: NumericTensor,
    weight_metadata: ParameterMetadata,
    bias: Option<(NumericTensor, ParameterMetadata)>,
}

impl Parameterized<NumericTensor> for NumericLinear {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.weight_metadata, &self.weight, visitor);
        if let Some((bias, metadata)) = &self.bias {
            visit(metadata, bias, visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.weight_metadata, &mut self.weight, visitor);
        if let Some((bias, metadata)) = &mut self.bias {
            visit_mut(metadata, bias, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.weight_metadata.trainable = trainable;
        if let Some((_, metadata)) = &mut self.bias {
            metadata.trainable = trainable;
        }
    }
}

impl LinearOperator<NumericTensor> for NumericLinear {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        linear(
            input,
            &self.weight,
            self.bias.as_ref().map(|(bias, _)| bias),
        )
    }
}

#[derive(Debug, Clone)]
struct NumericEmbedding {
    weight: NumericTensor,
    metadata: ParameterMetadata,
}

impl Parameterized<NumericTensor> for NumericEmbedding {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.metadata, &self.weight, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.metadata, &mut self.weight, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl EmbeddingOperator<NumericTensor> for NumericEmbedding {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let vocabulary = self.weight.shape[0] as usize;
        let dimensions = self.weight.shape[1] as usize;
        let mut shape = input.shape.clone();
        shape.push(dimensions as i32);
        let mut output = NumericTensor::zeros(shape);
        for (token_index, token) in input.data.iter().enumerate() {
            let token = *token as usize;
            if token >= vocabulary || token as f32 != input.data[token_index] {
                return Err(Error::backend("numeric embedding token is invalid"));
            }
            output.data[token_index * dimensions..(token_index + 1) * dimensions]
                .copy_from_slice(&self.weight.data[token * dimensions..(token + 1) * dimensions]);
        }
        Ok(output)
    }

    fn as_linear(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        linear(input, &self.weight, None)
    }
}

#[derive(Debug, Clone)]
struct NumericNorm {
    weight: NumericTensor,
    metadata: ParameterMetadata,
    epsilon: f32,
}

impl Parameterized<NumericTensor> for NumericNorm {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.metadata, &self.weight, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.metadata, &mut self.weight, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl NormalizationOperator<NumericTensor> for NumericNorm {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let dimensions = self.weight.data.len();
        if input.shape.last().copied() != Some(dimensions as i32) {
            return Err(Error::backend("numeric RMSNorm geometry mismatch"));
        }
        let mut output = input.clone();
        for (input_row, output_row) in input
            .data
            .chunks_exact(dimensions)
            .zip(output.data.chunks_exact_mut(dimensions))
        {
            let rms = (input_row.iter().map(|value| value * value).sum::<f32>()
                / dimensions as f32
                + self.epsilon)
                .sqrt();
            for index in 0..dimensions {
                output_row[index] = input_row[index] / rms * self.weight.data[index];
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct NumericRotary {
    dimensions: i32,
    traditional: bool,
    base: f32,
}

impl Parameterized<NumericTensor> for NumericRotary {
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

impl RotaryOperator<NumericTensor> for NumericRotary {
    fn forward(
        &mut self,
        input: &NumericTensor,
        position: RotaryPosition<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        match position {
            RotaryPosition::Offset(position_offset) => rotary_offset(
                input,
                self.dimensions,
                self.traditional,
                self.base,
                position_offset,
            ),
            RotaryPosition::Embeddings { cosine, sine } => {
                rotary_embeddings(input, self.dimensions, self.traditional, cosine, sine)
            }
        }
    }
}

fn rotary_offset(
    input: &NumericTensor,
    dimensions: i32,
    traditional: bool,
    base: f32,
    position_offset: i32,
) -> Result<NumericTensor, Error> {
    let sequence = input.shape.get(input.shape.len().wrapping_sub(2)).copied();
    let Some(sequence) = sequence else {
        return Err(Error::backend(
            "numeric rotary requires sequence and feature axes",
        ));
    };
    let half = dimensions as usize / 2;
    let mut cosine = NumericTensor::zeros(vec![sequence, half as i32]);
    let mut sine = cosine.clone();
    for position in 0..sequence as usize {
        for frequency in 0..half {
            let theta = (position_offset as f32 + position as f32)
                / base.powf(2.0 * frequency as f32 / dimensions as f32);
            cosine.data[position * half + frequency] = theta.cos();
            sine.data[position * half + frequency] = theta.sin();
        }
    }
    rotary_embeddings(input, dimensions, traditional, &cosine, &sine)
}

fn rotary_embeddings(
    input: &NumericTensor,
    dimensions: i32,
    traditional: bool,
    cosine: &NumericTensor,
    sine: &NumericTensor,
) -> Result<NumericTensor, Error> {
    if dimensions <= 0 || dimensions % 2 != 0 || input.shape.last().copied() != Some(dimensions) {
        return Err(Error::backend("numeric rotary geometry mismatch"));
    }
    let sequence = input.shape[input.shape.len() - 2] as usize;
    let half = dimensions as usize / 2;
    if cosine.shape != [sequence as i32, half as i32] || sine.shape != cosine.shape {
        return Err(Error::backend(
            "numeric explicit rotary embedding shape mismatch",
        ));
    }
    let dimensions = dimensions as usize;
    let rows = input.data.len() / dimensions;
    let mut output = input.clone();
    for row in 0..rows {
        let position = row % sequence;
        for frequency in 0..half {
            let (left, right) = if traditional {
                (2 * frequency, 2 * frequency + 1)
            } else {
                (frequency, frequency + half)
            };
            let cosine = cosine.data[position * half + frequency];
            let sine = sine.data[position * half + frequency];
            let left_value = input.data[row * dimensions + left];
            let right_value = input.data[row * dimensions + right];
            output.data[row * dimensions + left] = left_value * cosine - right_value * sine;
            output.data[row * dimensions + right] = right_value * cosine + left_value * sine;
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct NumericHyperConnection {
    streams: usize,
    hidden_size: usize,
    iterations: usize,
    epsilon: f32,
    function: (NumericTensor, ParameterMetadata),
    base: (NumericTensor, ParameterMetadata),
    scale: (NumericTensor, ParameterMetadata),
}

impl Parameterized<NumericTensor> for NumericHyperConnection {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.function.1, &self.function.0, visitor);
        visit(&self.base.1, &self.base.0, visitor);
        visit(&self.scale.1, &self.scale.0, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.function.1, &mut self.function.0, visitor);
        visit_mut(&self.base.1, &mut self.base.0, visitor);
        visit_mut(&self.scale.1, &mut self.scale.0, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.function.1.trainable = trainable;
        self.base.1.trainable = trainable;
        self.scale.1.trainable = trainable;
    }
}

impl HyperConnectionOperator<NumericTensor> for NumericHyperConnection {
    fn collapse(
        &mut self,
        residual: &NumericTensor,
        norm_epsilon: f32,
        _: &NumericContext,
    ) -> Result<HyperConnectionState<NumericTensor>, Error> {
        if residual.shape.len() != 4
            || residual.shape[2] as usize != self.streams
            || residual.shape[3] as usize != self.hidden_size
        {
            return Err(Error::backend("numeric hyper-connection geometry mismatch"));
        }
        let rows = residual.data.len() / (self.streams * self.hidden_size);
        let mixed_width = (2 + self.streams) * self.streams;
        let mut pre = NumericTensor::zeros(vec![
            residual.shape[0],
            residual.shape[1],
            self.streams as i32,
        ]);
        let mut post = pre.clone();
        let mut combination = NumericTensor::zeros(vec![
            residual.shape[0],
            residual.shape[1],
            self.streams as i32,
            self.streams as i32,
        ]);
        for row in 0..rows {
            let flat = &residual.data[row * self.streams * self.hidden_size
                ..(row + 1) * self.streams * self.hidden_size];
            let rms = (flat.iter().map(|value| value * value).sum::<f32>() / flat.len() as f32
                + norm_epsilon)
                .sqrt();
            let mixes = (0..mixed_width)
                .map(|output| {
                    flat.iter()
                        .enumerate()
                        .map(|(input, value)| {
                            value / rms
                                * self.function.0.data
                                    [output * self.streams * self.hidden_size + input]
                        })
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            for stream in 0..self.streams {
                pre.data[row * self.streams + stream] =
                    sigmoid_scalar(mixes[stream] * self.scale.0.data[0] + self.base.0.data[stream])
                        + self.epsilon;
                post.data[row * self.streams + stream] = 2.0
                    * sigmoid_scalar(
                        mixes[self.streams + stream] * self.scale.0.data[1]
                            + self.base.0.data[self.streams + stream],
                    );
            }
            let start = 2 * self.streams;
            let epsilon = self.epsilon;
            let mut matrix = (0..self.streams)
                .flat_map(|left| {
                    let logits = (0..self.streams)
                        .map(|right| {
                            let index = start + left * self.streams + right;
                            mixes[index] * self.scale.0.data[2] + self.base.0.data[index]
                        })
                        .collect::<Vec<_>>();
                    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denominator = logits
                        .iter()
                        .map(|value| (*value - maximum).exp())
                        .sum::<f32>();
                    logits
                        .into_iter()
                        .map(move |value| (value - maximum).exp() / denominator + epsilon)
                })
                .collect::<Vec<_>>();
            normalize_hyper_axis(&mut matrix, self.streams, false, self.epsilon);
            for _ in 1..self.iterations {
                normalize_hyper_axis(&mut matrix, self.streams, true, self.epsilon);
                normalize_hyper_axis(&mut matrix, self.streams, false, self.epsilon);
            }
            combination.data
                [row * self.streams * self.streams..(row + 1) * self.streams * self.streams]
                .copy_from_slice(&matrix);
        }
        let mut collapsed = NumericTensor::zeros(vec![
            residual.shape[0],
            residual.shape[1],
            self.hidden_size as i32,
        ]);
        for row in 0..rows {
            for dimension in 0..self.hidden_size {
                collapsed.data[row * self.hidden_size + dimension] = (0..self.streams)
                    .map(|stream| {
                        pre.data[row * self.streams + stream]
                            * residual.data
                                [(row * self.streams + stream) * self.hidden_size + dimension]
                    })
                    .sum();
            }
        }
        Ok(HyperConnectionState {
            collapsed,
            pre,
            post,
            combination,
        })
    }

    fn expand(
        &mut self,
        sublayer: &NumericTensor,
        residual: &NumericTensor,
        state: &HyperConnectionState<NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let rows = residual.data.len() / (self.streams * self.hidden_size);
        if sublayer.shape
            != [
                residual.shape[0],
                residual.shape[1],
                self.hidden_size as i32,
            ]
        {
            return Err(Error::backend("numeric hyper expansion geometry mismatch"));
        }
        let mut output = residual.clone();
        for row in 0..rows {
            for output_stream in 0..self.streams {
                for dimension in 0..self.hidden_size {
                    let injected = state.post.data[row * self.streams + output_stream]
                        * sublayer.data[row * self.hidden_size + dimension];
                    let mixed = (0..self.streams)
                        .map(|input_stream| {
                            state.combination.data[row * self.streams * self.streams
                                + input_stream * self.streams
                                + output_stream]
                                * residual.data[(row * self.streams + input_stream)
                                    * self.hidden_size
                                    + dimension]
                        })
                        .sum::<f32>();
                    output.data
                        [(row * self.streams + output_stream) * self.hidden_size + dimension] =
                        injected + mixed;
                }
            }
        }
        Ok(output)
    }
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn normalize_hyper_axis(matrix: &mut [f32], streams: usize, rows: bool, epsilon: f32) {
    for outer in 0..streams {
        let sum = (0..streams)
            .map(|inner| {
                let index = if rows {
                    outer * streams + inner
                } else {
                    inner * streams + outer
                };
                matrix[index]
            })
            .sum::<f32>()
            + epsilon;
        for inner in 0..streams {
            let index = if rows {
                outer * streams + inner
            } else {
                inner * streams + outer
            };
            matrix[index] /= sum;
        }
    }
}

#[derive(Debug, Clone)]
struct NumericHyperHead {
    streams: usize,
    hidden_size: usize,
    norm_epsilon: f32,
    epsilon: f32,
    function: (NumericTensor, ParameterMetadata),
    base: (NumericTensor, ParameterMetadata),
    scale: (NumericTensor, ParameterMetadata),
}

impl Parameterized<NumericTensor> for NumericHyperHead {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.function.1, &self.function.0, visitor);
        visit(&self.base.1, &self.base.0, visitor);
        visit(&self.scale.1, &self.scale.0, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.function.1, &mut self.function.0, visitor);
        visit_mut(&self.base.1, &mut self.base.0, visitor);
        visit_mut(&self.scale.1, &mut self.scale.0, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.function.1.trainable = trainable;
        self.base.1.trainable = trainable;
        self.scale.1.trainable = trainable;
    }
}

impl HyperHeadOperator<NumericTensor> for NumericHyperHead {
    fn forward(
        &mut self,
        residual: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        if residual.shape.len() != 4
            || residual.shape[2] as usize != self.streams
            || residual.shape[3] as usize != self.hidden_size
        {
            return Err(Error::backend("numeric hyper-head geometry mismatch"));
        }
        let rows = residual.data.len() / (self.streams * self.hidden_size);
        let mut output = NumericTensor::zeros(vec![
            residual.shape[0],
            residual.shape[1],
            self.hidden_size as i32,
        ]);
        for row in 0..rows {
            let flat = &residual.data[row * self.streams * self.hidden_size
                ..(row + 1) * self.streams * self.hidden_size];
            let rms = (flat.iter().map(|value| value * value).sum::<f32>() / flat.len() as f32
                + self.norm_epsilon)
                .sqrt();
            for stream in 0..self.streams {
                let logit = flat
                    .iter()
                    .enumerate()
                    .map(|(input, value)| {
                        value / rms
                            * self.function.0.data[stream * self.streams * self.hidden_size + input]
                    })
                    .sum::<f32>();
                let coefficient =
                    sigmoid_scalar(logit * self.scale.0.data[0] + self.base.0.data[stream])
                        + self.epsilon;
                for dimension in 0..self.hidden_size {
                    output.data[row * self.hidden_size + dimension] += coefficient
                        * residual.data
                            [(row * self.streams + stream) * self.hidden_size + dimension];
                }
            }
        }
        Ok(output)
    }
}

#[derive(Default)]
struct NumericContext {
    sliding_attention_calls: Cell<usize>,
}

struct NumericBackend;

struct NumericBlockwiseAttention {
    queries: NumericTensor,
    scale: f32,
    mask: Option<NumericTensor>,
    query_start: i64,
    mask_origin: i64,
    running_max: Vec<f32>,
    running_sum: Vec<f32>,
    values: Vec<f32>,
    value_dimensions: usize,
}

impl BlockwiseAttentionBackend for NumericBackend {
    type BlockwiseAccumulator = NumericBlockwiseAttention;

    fn begin_blockwise_attention(
        spec: BlockwiseAttentionSpec<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<Self::BlockwiseAccumulator, Error> {
        if spec.queries.shape.len() != 4
            || spec.sliding_window.is_some()
            || spec.prefix_tokens != 0
            || spec.sinks.is_some()
        {
            return Err(Error::backend(
                "unsupported numeric blockwise-attention request",
            ));
        }
        let rows = usize::try_from(spec.queries.dim(0) * spec.queries.dim(1) * spec.queries.dim(2))
            .map_err(Error::backend)?;
        Ok(NumericBlockwiseAttention {
            queries: spec.queries.clone(),
            scale: spec.scale,
            mask: spec.mask.cloned(),
            query_start: spec.query_start,
            mask_origin: spec
                .mask
                .map_or(0, |mask| spec.context_end - i64::from(mask.dim(1))),
            running_max: vec![f32::NEG_INFINITY; rows],
            running_sum: vec![0.0; rows],
            values: Vec::new(),
            value_dimensions: 0,
        })
    }

    fn accumulate_blockwise_attention(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: NumericTensor,
        values: NumericTensor,
        _: &NumericContext,
    ) -> Result<u64, Error> {
        if keys.shape.len() != 4
            || values.shape.len() != 4
            || keys.shape[..3] != values.shape[..3]
            || i64::from(keys.dim(2)) != end - start
        {
            return Err(Error::backend(
                "numeric blockwise key/value geometry mismatch",
            ));
        }
        let batch = accumulator.queries.dim(0) as usize;
        let heads = accumulator.queries.dim(1) as usize;
        let query_tokens = accumulator.queries.dim(2) as usize;
        let dimensions = accumulator.queries.dim(3) as usize;
        let key_heads = keys.dim(1) as usize;
        let key_tokens = keys.dim(2) as usize;
        let value_dimensions = values.dim(3) as usize;
        if dimensions != keys.dim(3) as usize || !heads.is_multiple_of(key_heads) {
            return Err(Error::backend("numeric blockwise attention head mismatch"));
        }
        if accumulator.value_dimensions == 0 {
            accumulator.value_dimensions = value_dimensions;
            accumulator.values = vec![0.0; batch * heads * query_tokens * value_dimensions];
        } else if accumulator.value_dimensions != value_dimensions {
            return Err(Error::backend("numeric blockwise value width changed"));
        }
        for batch_index in 0..batch {
            for head in 0..heads {
                let key_head = head % key_heads;
                for query in 0..query_tokens {
                    let row = (batch_index * heads + head) * query_tokens + query;
                    let absolute_query = accumulator.query_start + query as i64;
                    for key in 0..key_tokens {
                        let absolute_key = start + key as i64;
                        if absolute_key > absolute_query {
                            continue;
                        }
                        let mut score = 0.0;
                        for dimension in 0..dimensions {
                            let query_offset = (((batch_index * heads + head) * query_tokens
                                + query)
                                * dimensions)
                                + dimension;
                            let key_offset = (((batch_index * key_heads + key_head) * key_tokens
                                + key)
                                * dimensions)
                                + dimension;
                            score += accumulator.queries.data[query_offset] * keys.data[key_offset];
                        }
                        score *= accumulator.scale;
                        if let Some(mask) = &accumulator.mask {
                            let mask_key = absolute_key - accumulator.mask_origin;
                            if mask_key < 0 || mask_key >= i64::from(mask.dim(1)) {
                                return Err(Error::backend(
                                    "numeric blockwise mask misses a block",
                                ));
                            }
                            let bias = mask.data[query * mask.dim(1) as usize + mask_key as usize];
                            if !bias.is_finite() || bias <= -1.0e20 {
                                continue;
                            }
                            score += bias;
                        }
                        let old_max = accumulator.running_max[row];
                        let new_max = old_max.max(score);
                        let old_scale = if old_max.is_finite() {
                            (old_max - new_max).exp()
                        } else {
                            0.0
                        };
                        let weight = (score - new_max).exp();
                        accumulator.running_sum[row] =
                            accumulator.running_sum[row] * old_scale + weight;
                        for dimension in 0..value_dimensions {
                            let output = row * value_dimensions + dimension;
                            let value = (((batch_index * key_heads + key_head) * key_tokens + key)
                                * value_dimensions)
                                + dimension;
                            accumulator.values[output] = accumulator.values[output] * old_scale
                                + weight * values.data[value];
                        }
                        accumulator.running_max[row] = new_max;
                    }
                }
            }
        }
        Ok(u64::try_from((keys.data.len() + values.data.len()) * 4).unwrap())
    }

    fn finish_blockwise_attention(
        mut accumulator: Self::BlockwiseAccumulator,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        if accumulator.value_dimensions == 0 {
            return Err(Error::backend(
                "numeric blockwise attention received no blocks",
            ));
        }
        for (row, denominator) in accumulator.running_sum.iter().copied().enumerate() {
            if denominator > 0.0 {
                for dimension in 0..accumulator.value_dimensions {
                    accumulator.values[row * accumulator.value_dimensions + dimension] /=
                        denominator;
                }
            }
        }
        Ok(NumericTensor::new(
            vec![
                accumulator.queries.dim(0),
                accumulator.queries.dim(1),
                accumulator.queries.dim(2),
                i32::try_from(accumulator.value_dimensions).map_err(Error::backend)?,
            ],
            accumulator.values,
        ))
    }
}

impl NeuralBackend for NumericBackend {
    type Tensor = NumericTensor;
    type Linear = NumericLinear;
    type Embedding = NumericEmbedding;
    type Normalization = NumericNorm;
    type Rotary = NumericRotary;
    type ParallelContext = ();

    fn linear(spec: LinearSpec, _: &NumericContext) -> Result<Self::Linear, Error> {
        let weight = parameter(&spec.weight, vec![spec.output, spec.input], false);
        let weight_metadata = ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable);
        let bias = spec.bias.map(|bias| {
            let value = parameter(&bias, vec![spec.output], false);
            let metadata = ParameterMetadata::from_spec(&bias, bias.trainable);
            (value, metadata)
        });
        Ok(NumericLinear {
            weight,
            weight_metadata,
            bias,
        })
    }

    fn embedding(spec: EmbeddingSpec, _: &NumericContext) -> Result<Self::Embedding, Error> {
        Ok(NumericEmbedding {
            weight: parameter(&spec.weight, vec![spec.vocabulary, spec.dimensions], false),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
        })
    }

    fn rms_norm(spec: NormalizationSpec, _: &NumericContext) -> Result<Self::Normalization, Error> {
        Ok(NumericNorm {
            weight: parameter(&spec.weight, vec![spec.dimensions], true),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            epsilon: spec.epsilon,
        })
    }

    fn rotary(spec: RotarySpec<'_>, _: &NumericContext) -> Result<Self::Rotary, Error> {
        Ok(NumericRotary {
            dimensions: spec.dimensions,
            traditional: spec.traditional,
            base: spec.base,
        })
    }

    fn silu(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(|value| value / (1.0 + (-value).exp())))
    }

    fn swiglu(
        gate: Self::Tensor,
        up: Self::Tensor,
        limit: Option<eredu_nn::SwiGluLimit>,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let gate = gate.map(|value| limit.map_or(value, |bound| value.min(bound.get())));
        let up =
            up.map(|value| limit.map_or(value, |bound| value.clamp(-bound.get(), bound.get())));
        gate.map(|value| value / (1.0 + (-value).exp()))
            .zip(&up, |left, right| left * right)
    }

    fn attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        mask: Option<&Self::Tensor>,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        attention(&queries, &keys, &values, scale, mask, None, 0)
    }

    fn indexed_attention(
        input: IndexedAttentionInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        indexed_attention(input)
    }

    fn pooled_attention(
        input: PooledAttentionInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        pooled_attention(input)
    }

    fn select_pooled_positions(
        input: PooledPositionInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        select_pooled_positions(input)
    }

    fn gather_pooled_mask(
        mask: &NumericTensor,
        selected_positions: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        if mask.shape.len() != 2
            || selected_positions.shape.len() != 3
            || mask.shape[0] != selected_positions.shape[1]
        {
            return Err(Error::backend("numeric pooled-mask gathering mismatch"));
        }
        let batch = selected_positions.shape[0] as usize;
        let queries = selected_positions.shape[1] as usize;
        let selected = selected_positions.shape[2] as usize;
        let pooled = mask.shape[1] as usize;
        let mut output =
            NumericTensor::zeros(vec![batch as i32, 1, queries as i32, selected as i32]);
        for b in 0..batch {
            for query in 0..queries {
                for route in 0..selected {
                    let raw = selected_positions.data[(b * queries + query) * selected + route];
                    let position = raw as usize;
                    if position >= pooled || position as f32 != raw {
                        return Err(Error::backend("numeric pooled position is invalid"));
                    }
                    output.data[(b * queries + query) * selected + route] =
                        mask.data[query * pooled + position];
                }
            }
        }
        Ok(output)
    }

    fn attention_with_sinks(
        queries: NumericTensor,
        keys: NumericTensor,
        values: NumericTensor,
        scale: f32,
        mask: Option<&NumericTensor>,
        sinks: Option<&NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        attention_with_sinks(&queries, &keys, &values, scale, mask, sinks)
    }

    fn rms_norm_without_weight(
        input: &NumericTensor,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let dimensions = input.shape.last().copied().unwrap() as usize;
        let mut output = input.clone();
        for (source, target) in input
            .data
            .chunks_exact(dimensions)
            .zip(output.data.chunks_exact_mut(dimensions))
        {
            let rms = (source.iter().map(|value| value * value).sum::<f32>() / dimensions as f32
                + epsilon)
                .sqrt();
            for (target, source) in target.iter_mut().zip(source) {
                *target = *source / rms;
            }
        }
        Ok(output)
    }

    fn grouped_linear(
        linear: &mut NumericLinear,
        input: &NumericTensor,
        groups: i32,
        output_per_group: i32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        if input.shape.len() != 4
            || input.shape[1] != groups
            || linear.weight.shape != [groups * output_per_group, input.shape[3]]
        {
            return Err(Error::backend("numeric grouped-linear geometry mismatch"));
        }
        let batch = input.shape[0] as usize;
        let groups = groups as usize;
        let tokens = input.shape[2] as usize;
        let width = input.shape[3] as usize;
        let output_width = output_per_group as usize;
        let mut output = NumericTensor::zeros(vec![
            batch as i32,
            groups as i32,
            tokens as i32,
            output_per_group,
        ]);
        for b in 0..batch {
            for group in 0..groups {
                for token in 0..tokens {
                    let input_base = ((b * groups + group) * tokens + token) * width;
                    for out in 0..output_width {
                        let weight_base = (group * output_width + out) * width;
                        output.data[((b * groups + group) * tokens + token) * output_width + out] =
                            (0..width)
                                .map(|inner| {
                                    input.data[input_base + inner]
                                        * linear.weight.data[weight_base + inner]
                                })
                                .sum();
                    }
                }
            }
        }
        Ok(output)
    }

    fn sliding_window_attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        context
            .sliding_attention_calls
            .set(context.sliding_attention_calls.get() + 1);
        let batch = queries.shape[0];
        let sequence = queries.shape[2];
        let attended = attention(
            &queries,
            &keys,
            &values,
            scale,
            None,
            Some(window),
            position_offset,
        )?;
        attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)
    }

    fn causal_mask(
        sequence: i32,
        position_offset: i32,
        window: Option<i32>,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let keys = position_offset + sequence;
        let mut mask = NumericTensor::zeros(vec![sequence, keys]);
        for query in 0..sequence {
            let query_position = position_offset + query;
            for key in 0..keys {
                let too_new = key > query_position;
                let too_old = window.is_some_and(|window| key <= query_position - window);
                if too_new || too_old {
                    mask.data[(query * keys + key) as usize] = -1.0e9;
                }
            }
        }
        Ok(mask)
    }

    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        _: &(),
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        linear.forward(input, context)
    }
}

fn attention(
    queries: &NumericTensor,
    keys: &NumericTensor,
    values: &NumericTensor,
    scale: f32,
    mask: Option<&NumericTensor>,
    window: Option<i32>,
    query_position_offset: i32,
) -> Result<NumericTensor, Error> {
    if queries.shape.len() != 4
        || keys.shape.len() != 4
        || values.shape.len() != 4
        || values.shape[..3] != keys.shape[..3]
        || queries.shape[0] != keys.shape[0]
        || queries.shape[3] != keys.shape[3]
        || queries.shape[1] % keys.shape[1] != 0
    {
        return Err(Error::backend("numeric attention geometry mismatch"));
    }
    let batch = queries.shape[0] as usize;
    let query_heads = queries.shape[1] as usize;
    let key_heads = keys.shape[1] as usize;
    let query_sequence = queries.shape[2] as usize;
    let key_sequence = keys.shape[2] as usize;
    let dimensions = queries.shape[3] as usize;
    let value_dimensions = values.shape[3] as usize;
    if let Some(mask) = mask {
        if mask.shape != [query_sequence as i32, key_sequence as i32] {
            return Err(Error::backend("numeric attention mask shape mismatch"));
        }
    }
    let key_position_start = query_position_offset + query_sequence as i32 - key_sequence as i32;
    let groups = query_heads / key_heads;
    let mut output = NumericTensor::zeros(vec![
        queries.shape[0],
        queries.shape[1],
        queries.shape[2],
        values.shape[3],
    ]);
    for batch_index in 0..batch {
        for query_head in 0..query_heads {
            let key_head = query_head / groups;
            for query_position in 0..query_sequence {
                let global_query = query_position_offset + query_position as i32;
                let mut scores = vec![f32::NEG_INFINITY; key_sequence];
                for (key_position, score) in scores.iter_mut().enumerate() {
                    let global_key = key_position_start + key_position as i32;
                    let causal = global_key <= global_query;
                    let local = window.is_none_or(|window| global_key > global_query - window);
                    if causal && local {
                        let query_base = ((batch_index * query_heads + query_head)
                            * query_sequence
                            + query_position)
                            * dimensions;
                        let key_base = ((batch_index * key_heads + key_head) * key_sequence
                            + key_position)
                            * dimensions;
                        *score = (0..dimensions)
                            .map(|dimension| {
                                queries.data[query_base + dimension]
                                    * keys.data[key_base + dimension]
                            })
                            .sum::<f32>()
                            * scale
                            + mask.map_or(0.0, |mask| {
                                mask.data[query_position * key_sequence + key_position]
                            });
                    }
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probabilities = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = probabilities.iter().sum::<f32>();
                for probability in &mut probabilities {
                    *probability /= denominator;
                }
                for dimension in 0..value_dimensions {
                    let output_index = (((batch_index * query_heads + query_head)
                        * query_sequence
                        + query_position)
                        * value_dimensions)
                        + dimension;
                    output.data[output_index] = (0..key_sequence)
                        .map(|key_position| {
                            let value_index = (((batch_index * key_heads + key_head)
                                * key_sequence
                                + key_position)
                                * value_dimensions)
                                + dimension;
                            probabilities[key_position] * values.data[value_index]
                        })
                        .sum();
                }
            }
        }
    }
    Ok(output)
}

fn indexed_attention(
    input: IndexedAttentionInput<'_, NumericTensor>,
) -> Result<NumericTensor, Error> {
    input.validate()?;
    let batch = input.queries.shape[0] as usize;
    let heads = input.queries.shape[1] as usize;
    let query_tokens = input.queries.shape[2] as usize;
    let key_dimensions = input.queries.shape[3] as usize;
    let value_dimensions = input.local_values.shape[2] as usize;
    let local_tokens = input.local_keys.shape[1] as usize;
    let pooled_tokens = input.pooled_keys.shape[1] as usize;
    let selected = input.selected_positions.shape[2] as usize;
    let mask_value = |mask: Option<&NumericTensor>, b: usize, h: usize, q: usize, k: usize| {
        let Some(mask) = mask else { return Ok(0.0) };
        match mask.shape.as_slice() {
            [queries, keys] if *queries as usize == query_tokens => {
                Ok(mask.data[q * *keys as usize + k])
            }
            [batches, queries, keys]
                if *batches as usize == batch && *queries as usize == query_tokens =>
            {
                Ok(mask.data[(b * *queries as usize + q) * *keys as usize + k])
            }
            [batches, mask_heads, queries, keys]
                if *batches as usize == batch
                    && (*mask_heads == 1 || *mask_heads as usize == heads)
                    && *queries as usize == query_tokens =>
            {
                let selected_head = if *mask_heads == 1 { 0 } else { h };
                Ok(
                    mask.data[((b * *mask_heads as usize + selected_head) * *queries as usize + q)
                        * *keys as usize
                        + k],
                )
            }
            _ => Err(Error::backend(
                "numeric indexed-attention mask geometry mismatch",
            )),
        }
    };
    let mut output = NumericTensor::zeros(vec![
        batch as i32,
        heads as i32,
        query_tokens as i32,
        value_dimensions as i32,
    ]);
    for b in 0..batch {
        for h in 0..heads {
            for q in 0..query_tokens {
                let query_base = ((b * heads + h) * query_tokens + q) * key_dimensions;
                let mut scores = Vec::with_capacity(local_tokens + selected + 1);
                for local in 0..local_tokens {
                    let key_base = (b * local_tokens + local) * key_dimensions;
                    let score = (0..key_dimensions)
                        .map(|dimension| {
                            input.queries.data[query_base + dimension]
                                * input.local_keys.data[key_base + dimension]
                        })
                        .sum::<f32>()
                        * input.scale
                        + mask_value(input.local_mask, b, h, q, local)?;
                    scores.push(score);
                }
                let mut selected_ids = Vec::with_capacity(selected);
                for route in 0..selected {
                    let raw =
                        input.selected_positions.data[(b * query_tokens + q) * selected + route];
                    let position = raw as usize;
                    if position >= pooled_tokens || position as f32 != raw {
                        return Err(Error::backend(
                            "numeric indexed-attention position is invalid",
                        ));
                    }
                    selected_ids.push(position);
                    let key_base = (b * pooled_tokens + position) * key_dimensions;
                    let score = (0..key_dimensions)
                        .map(|dimension| {
                            input.queries.data[query_base + dimension]
                                * input.pooled_keys.data[key_base + dimension]
                        })
                        .sum::<f32>()
                        * input.scale
                        + mask_value(input.pooled_mask, b, h, q, route)?;
                    scores.push(score);
                }
                if let Some(sinks) = input.sinks {
                    scores.push(sinks.data[h]);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                for dimension in 0..value_dimensions {
                    let local = (0..local_tokens)
                        .map(|token| {
                            weights[token]
                                * input.local_values.data
                                    [(b * local_tokens + token) * value_dimensions + dimension]
                        })
                        .sum::<f32>();
                    let pooled = selected_ids
                        .iter()
                        .enumerate()
                        .map(|(route, token)| {
                            weights[local_tokens + route]
                                * input.pooled_values.data
                                    [(b * pooled_tokens + *token) * value_dimensions + dimension]
                        })
                        .sum::<f32>();
                    output.data
                        [((b * heads + h) * query_tokens + q) * value_dimensions + dimension] =
                        (local + pooled) / denominator;
                }
            }
        }
    }
    Ok(output)
}

fn attention_with_sinks(
    queries: &NumericTensor,
    keys: &NumericTensor,
    values: &NumericTensor,
    scale: f32,
    mask: Option<&NumericTensor>,
    sinks: Option<&NumericTensor>,
) -> Result<NumericTensor, Error> {
    if sinks.is_none() {
        return attention(queries, keys, values, scale, mask, None, 0);
    }
    if queries.shape.len() != 4
        || keys.shape.len() != 4
        || values.shape != keys.shape
        || keys.shape[1] != 1
        || queries.shape[0] != keys.shape[0]
        || queries.shape[3] != keys.shape[3]
    {
        return Err(Error::backend("numeric sink-attention geometry mismatch"));
    }
    let batch = queries.shape[0] as usize;
    let heads = queries.shape[1] as usize;
    let query_tokens = queries.shape[2] as usize;
    let key_tokens = keys.shape[2] as usize;
    let dimensions = queries.shape[3] as usize;
    let sinks = sinks.unwrap();
    if sinks.shape != [heads as i32]
        || mask.is_some_and(|mask| mask.shape != [query_tokens as i32, key_tokens as i32])
    {
        return Err(Error::backend(
            "numeric sink-attention mask or sink mismatch",
        ));
    }
    let mut output = NumericTensor::zeros(queries.shape.clone());
    for b in 0..batch {
        for head in 0..heads {
            for query in 0..query_tokens {
                let query_base = ((b * heads + head) * query_tokens + query) * dimensions;
                let mut scores = (0..key_tokens)
                    .map(|key| {
                        let key_base = (b * key_tokens + key) * dimensions;
                        (0..dimensions)
                            .map(|dimension| {
                                queries.data[query_base + dimension]
                                    * keys.data[key_base + dimension]
                            })
                            .sum::<f32>()
                            * scale
                            + mask.map_or(0.0, |mask| mask.data[query * key_tokens + key])
                    })
                    .collect::<Vec<_>>();
                scores.push(sinks.data[head]);
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                for dimension in 0..dimensions {
                    output.data[query_base + dimension] = (0..key_tokens)
                        .map(|key| {
                            weights[key]
                                * values.data[(b * key_tokens + key) * dimensions + dimension]
                        })
                        .sum::<f32>()
                        / denominator;
                }
            }
        }
    }
    Ok(output)
}

fn pooled_attention(
    input: PooledAttentionInput<'_, NumericTensor>,
) -> Result<NumericTensor, Error> {
    let batch = input.queries.shape[0] as usize;
    let queries = input.queries.shape[2] as usize;
    let pooled = input.pooled.shape[1] as usize;
    if pooled == 0 {
        let mut shape = input.local.shape.clone();
        shape.insert(1, 1);
        let local = NumericTensor::new(shape, input.local.data.clone());
        return attention_with_sinks(
            input.queries,
            &local,
            &local,
            input.scale,
            input.local_mask,
            input.sinks,
        );
    }
    let mut positions = NumericTensor::zeros(vec![batch as i32, queries as i32, pooled as i32]);
    for b in 0..batch {
        for query in 0..queries {
            for position in 0..pooled {
                positions.data[(b * queries + query) * pooled + position] = position as f32;
            }
        }
    }
    indexed_attention(IndexedAttentionInput {
        queries: input.queries,
        local_keys: input.local,
        local_values: input.local,
        pooled_keys: input.pooled,
        pooled_values: input.pooled,
        selected_positions: &positions,
        scale: input.scale,
        local_mask: input.local_mask,
        pooled_mask: input.pooled_mask,
        sinks: input.sinks,
    })
}

fn select_pooled_positions(
    input: PooledPositionInput<'_, NumericTensor>,
) -> Result<NumericTensor, Error> {
    let queries = input.queries.shape.as_slice();
    let pooled = input.pooled_keys.shape.as_slice();
    let weights = input.head_weights.shape.as_slice();
    if queries.len() != 4
        || pooled.len() != 3
        || weights != [queries[0], queries[2], queries[1]]
        || pooled[0] != queries[0]
        || pooled[2] != queries[3]
        || input.top_k <= 0
    {
        return Err(Error::backend("numeric pooled-position geometry mismatch"));
    }
    let batch = queries[0] as usize;
    let heads = queries[1] as usize;
    let query_tokens = queries[2] as usize;
    let dimensions = queries[3] as usize;
    let pooled_tokens = pooled[1] as usize;
    let top_k = (input.top_k as usize).min(pooled_tokens);
    let mut output = NumericTensor::zeros(vec![batch as i32, query_tokens as i32, top_k as i32]);
    for b in 0..batch {
        for query in 0..query_tokens {
            let mut scores = (0..pooled_tokens)
                .map(|position| {
                    let mut score = 0.0;
                    for head in 0..heads {
                        let query_base = ((b * heads + head) * query_tokens + query) * dimensions;
                        let pooled_base = (b * pooled_tokens + position) * dimensions;
                        let dot = (0..dimensions)
                            .map(|dimension| {
                                input.queries.data[query_base + dimension]
                                    * input.pooled_keys.data[pooled_base + dimension]
                            })
                            .sum::<f32>()
                            .max(0.0)
                            * input.scale;
                        score += dot
                            * input.head_weights.data[(b * query_tokens + query) * heads + head]
                            * input.head_scale;
                    }
                    if let Some(mask) = input.mask {
                        score += mask.data[query * pooled_tokens + position];
                    }
                    (position, score)
                })
                .collect::<Vec<_>>();
            scores.sort_by(|left, right| right.1.total_cmp(&left.1));
            for (route, (position, _)) in scores.iter().take(top_k).enumerate() {
                output.data[(b * query_tokens + query) * top_k + route] = *position as f32;
            }
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct NumericRouter {
    linear: NumericLinear,
    routing: TopKRoutingSpec,
    correction_bias: Option<(NumericTensor, ParameterMetadata)>,
}

impl Parameterized<NumericTensor> for NumericRouter {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        self.linear.visit_parameters(visitor);
        if let Some((value, metadata)) = &self.correction_bias {
            visit(metadata, value, visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        self.linear.visit_parameters_mut(visitor);
        if let Some((value, metadata)) = &mut self.correction_bias {
            visit_mut(metadata, value, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.linear.set_trainable(trainable);
        if let Some((_, metadata)) = &mut self.correction_bias {
            metadata.trainable = trainable;
        }
    }
}

impl RoutingOperator<NumericTensor> for NumericRouter {
    fn route(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<RoutingResult<NumericTensor>, Error> {
        let logits = self.linear.forward(input, context)?;
        let experts = self.routing.expert_count() as usize;
        let top_k = self.routing.top_k() as usize;
        let tokens = logits.data.len() / experts;
        let route_shape = vec![tokens as i32, top_k as i32];
        let mut expert_ids = NumericTensor::zeros(route_shape.clone());
        let mut selected_scores = NumericTensor::zeros(route_shape.clone());
        let mut route_weights = NumericTensor::zeros(route_shape);
        for token in 0..tokens {
            let row = &logits.data[token * experts..(token + 1) * experts];
            let scores: Vec<f32> = match self.routing.scoring() {
                eredu_nn::RoutingScoring::Softmax => {
                    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exponentials = row
                        .iter()
                        .map(|value| (*value - maximum).exp())
                        .collect::<Vec<_>>();
                    let sum = exponentials.iter().sum::<f32>();
                    exponentials.iter().map(|value| *value / sum).collect()
                }
                eredu_nn::RoutingScoring::Sigmoid => row
                    .iter()
                    .map(|value| 1.0 / (1.0 + (-value).exp()))
                    .collect(),
                eredu_nn::RoutingScoring::SqrtSoftplus => row
                    .iter()
                    .map(|value| (1.0 + value.exp()).ln().sqrt())
                    .collect(),
            };
            let selection = scores
                .iter()
                .enumerate()
                .map(|(expert, score)| {
                    *score
                        + self
                            .correction_bias
                            .as_ref()
                            .map_or(0.0, |(bias, _)| bias.data[expert])
                })
                .collect::<Vec<_>>();
            let groups = self.routing.expert_groups() as usize;
            let selected_groups = self.routing.selected_groups() as usize;
            let experts_per_group = experts / groups;
            let mut group_order = (0..groups).collect::<Vec<_>>();
            group_order.sort_by(|left, right| {
                let score = |group: usize| {
                    let mut values = selection
                        [group * experts_per_group..(group + 1) * experts_per_group]
                        .to_vec();
                    values.sort_by(|left, right| right.total_cmp(left));
                    values.into_iter().take(2).sum::<f32>()
                };
                score(*right).total_cmp(&score(*left))
            });
            let eligible = &group_order[..selected_groups];
            let mut order = (0..experts)
                .filter(|expert| eligible.contains(&(expert / experts_per_group)))
                .collect::<Vec<_>>();
            order.sort_by(|left, right| selection[*right].total_cmp(&selection[*left]));
            let selected_sum = order[..top_k]
                .iter()
                .map(|expert| scores[*expert])
                .sum::<f32>();
            for (route, expert) in order.iter().copied().take(top_k).enumerate() {
                let selected = scores[expert];
                let index = token * top_k + route;
                expert_ids.data[index] = expert as f32;
                selected_scores.data[index] = selected;
                route_weights.data[index] = if self.routing.normalize_selected() {
                    selected / (selected_sum + self.routing.normalization_epsilon())
                } else {
                    selected
                } * self.routing.routed_scaling();
            }
        }
        Ok(RoutingResult {
            expert_ids,
            selected_scores,
            route_weights,
        })
    }

    fn route_selected(
        &mut self,
        input: &NumericTensor,
        expert_ids: &NumericTensor,
        context: &NumericContext,
    ) -> Result<RoutingResult<NumericTensor>, Error> {
        let logits = self.linear.forward(input, context)?;
        let experts = self.routing.expert_count() as usize;
        let top_k = self.routing.top_k() as usize;
        let tokens = logits.data.len() / experts;
        if expert_ids.shape != [tokens as i32, top_k as i32] {
            return Err(Error::backend("caller-selected route geometry mismatch"));
        }
        let mut selected_scores = NumericTensor::zeros(expert_ids.shape.clone());
        let mut route_weights = NumericTensor::zeros(expert_ids.shape.clone());
        for token in 0..tokens {
            let row = &logits.data[token * experts..(token + 1) * experts];
            let scores = match self.routing.scoring() {
                eredu_nn::RoutingScoring::Softmax => {
                    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let values = row
                        .iter()
                        .map(|value| (*value - maximum).exp())
                        .collect::<Vec<_>>();
                    let sum = values.iter().sum::<f32>();
                    values
                        .into_iter()
                        .map(|value| value / sum)
                        .collect::<Vec<_>>()
                }
                eredu_nn::RoutingScoring::Sigmoid => row
                    .iter()
                    .map(|value| 1.0 / (1.0 + (-value).exp()))
                    .collect(),
                eredu_nn::RoutingScoring::SqrtSoftplus => row
                    .iter()
                    .map(|value| (1.0 + value.exp()).ln().sqrt())
                    .collect(),
            };
            let mut sum = 0.0;
            for route in 0..top_k {
                let index = token * top_k + route;
                let expert = expert_ids.data[index] as usize;
                if expert >= experts || expert as f32 != expert_ids.data[index] {
                    return Err(Error::backend("caller-selected expert id is invalid"));
                }
                selected_scores.data[index] = scores[expert];
                sum += scores[expert];
            }
            for route in 0..top_k {
                let index = token * top_k + route;
                route_weights.data[index] = if self.routing.normalize_selected() {
                    selected_scores.data[index] / (sum + self.routing.normalization_epsilon())
                } else {
                    selected_scores.data[index]
                } * self.routing.routed_scaling();
            }
        }
        Ok(RoutingResult {
            expert_ids: expert_ids.clone(),
            selected_scores,
            route_weights,
        })
    }
}

#[derive(Debug, Clone)]
struct NumericExpert {
    gate: NumericTensor,
    up: NumericTensor,
    down: NumericTensor,
}

#[derive(Debug, Clone)]
struct NumericExpertBank {
    experts: Vec<NumericExpert>,
    parameters: Vec<(NumericTensor, ParameterMetadata)>,
    limit: Option<eredu_nn::SwiGluLimit>,
}

impl Parameterized<NumericTensor> for NumericExpertBank {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        for (parameter, metadata) in &self.parameters {
            visit(metadata, parameter, visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        for (parameter, metadata) in &mut self.parameters {
            visit_mut(metadata, parameter, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        for (_, metadata) in &mut self.parameters {
            metadata.trainable = trainable;
        }
    }
}

impl SwiGluExpertBankOperator<NumericTensor> for NumericExpertBank {
    fn forward_routed(
        &mut self,
        input: &NumericTensor,
        routes: &RoutingResult<NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let hidden = input.shape.last().copied().unwrap() as usize;
        let tokens = input.data.len() / hidden;
        let top_k = routes.expert_ids.shape[1] as usize;
        if routes.expert_ids.shape != [tokens as i32, top_k as i32]
            || routes.route_weights.shape != routes.expert_ids.shape
        {
            return Err(Error::backend("numeric expert route geometry mismatch"));
        }
        let mut output = NumericTensor::zeros(input.shape.clone());
        for token in 0..tokens {
            let token_input = NumericTensor::new(
                vec![1, hidden as i32],
                input.data[token * hidden..(token + 1) * hidden].to_vec(),
            );
            for route in 0..top_k {
                let route_index = token * top_k + route;
                let expert_id = routes.expert_ids.data[route_index] as usize;
                let expert = self
                    .experts
                    .get(expert_id)
                    .ok_or_else(|| Error::backend("numeric expert id is out of range"))?;
                let gate = linear(&token_input, &expert.gate, None)?
                    .map(|value| self.limit.map_or(value, |bound| value.min(bound.get())));
                let gate = gate.map(|value| value / (1.0 + (-value).exp()));
                let up = linear(&token_input, &expert.up, None)?.map(|value| {
                    self.limit
                        .map_or(value, |bound| value.clamp(-bound.get(), bound.get()))
                });
                let activated = gate.zip(&up, |left, right| left * right)?;
                let expert_output = linear(&activated, &expert.down, None)?;
                let weight = routes.route_weights.data[route_index];
                for dimension in 0..hidden {
                    output.data[token * hidden + dimension] +=
                        weight * expert_output.data[dimension];
                }
            }
        }
        Ok(output)
    }
}

impl HyperNeuralBackend for NumericBackend {
    type HyperConnection = NumericHyperConnection;
    type HyperHead = NumericHyperHead;

    fn hyper_connection(
        spec: HyperConnectionSpec,
        _: &NumericContext,
    ) -> Result<Self::HyperConnection, Error> {
        spec.validate()?;
        let streams = spec.streams as usize;
        let hidden_size = spec.hidden_size as usize;
        let width = (2 + streams) * streams;
        Ok(NumericHyperConnection {
            streams,
            hidden_size,
            iterations: spec.sinkhorn_iterations,
            epsilon: spec.epsilon,
            function: (
                parameter(
                    &spec.function,
                    vec![width as i32, (streams * hidden_size) as i32],
                    false,
                ),
                ParameterMetadata::from_spec(&spec.function, spec.function.trainable),
            ),
            base: (
                parameter(&spec.base, vec![width as i32], false),
                ParameterMetadata::from_spec(&spec.base, spec.base.trainable),
            ),
            scale: (
                parameter(&spec.scale, vec![3], false),
                ParameterMetadata::from_spec(&spec.scale, spec.scale.trainable),
            ),
        })
    }

    fn hyper_head(spec: HyperHeadSpec, _: &NumericContext) -> Result<Self::HyperHead, Error> {
        spec.validate()?;
        let streams = spec.streams as usize;
        let hidden_size = spec.hidden_size as usize;
        Ok(NumericHyperHead {
            streams,
            hidden_size,
            norm_epsilon: spec.norm_epsilon,
            epsilon: spec.epsilon,
            function: (
                parameter(
                    &spec.function,
                    vec![streams as i32, (streams * hidden_size) as i32],
                    false,
                ),
                ParameterMetadata::from_spec(&spec.function, spec.function.trainable),
            ),
            base: (
                parameter(&spec.base, vec![streams as i32], false),
                ParameterMetadata::from_spec(&spec.base, spec.base.trainable),
            ),
            scale: (
                parameter(&spec.scale, vec![1], false),
                ParameterMetadata::from_spec(&spec.scale, spec.scale.trainable),
            ),
        })
    }
}

impl RoutedNeuralBackend for NumericBackend {
    type Router = NumericRouter;
    type SwiGluExpertBank = NumericExpertBank;

    fn top_k_router(spec: TopKRouterSpec, context: &NumericContext) -> Result<Self::Router, Error> {
        spec.validate()?;
        let routing = spec.routing;
        let linear = Self::linear(
            LinearSpec {
                input: spec.input_dimensions,
                output: routing.expert_count(),
                weight: spec.weight,
                bias: None,
                format: spec.quantization.into(),
            },
            context,
        )?;
        let correction_bias = spec.correction_bias.map(|parameter_spec| {
            let value = parameter(&parameter_spec, vec![routing.expert_count()], true);
            let metadata = ParameterMetadata::from_spec(&parameter_spec, parameter_spec.trainable);
            (value, metadata)
        });
        Ok(NumericRouter {
            linear,
            routing,
            correction_bias,
        })
    }

    fn swiglu_expert_bank(
        spec: SwiGluExpertBankSpec,
        _: &NumericContext,
    ) -> Result<Self::SwiGluExpertBank, Error> {
        spec.validate()?;
        let expert_count = spec.expert_count as usize;
        let hidden = spec.input_dimensions;
        let intermediate = spec.intermediate_dimensions;
        let mut experts = Vec::with_capacity(expert_count);
        let mut parameters = Vec::new();
        match spec.layout {
            SwiGluExpertLayout::Packed { gate_up, down } => {
                let packed_gate_up = parameter(
                    &gate_up.weight,
                    vec![spec.expert_count, 2 * intermediate, hidden],
                    false,
                );
                let packed_down = parameter(
                    &down.weight,
                    vec![spec.expert_count, spec.output_dimensions, intermediate],
                    false,
                );
                let gate_up_per_expert = (2 * intermediate * hidden) as usize;
                let projection_per_expert = (intermediate * hidden) as usize;
                let down_per_expert = (spec.output_dimensions * intermediate) as usize;
                for expert in 0..expert_count {
                    let gate_up_start = expert * gate_up_per_expert;
                    let down_start = expert * down_per_expert;
                    experts.push(NumericExpert {
                        gate: NumericTensor::new(
                            vec![intermediate, hidden],
                            packed_gate_up.data
                                [gate_up_start..gate_up_start + projection_per_expert]
                                .to_vec(),
                        ),
                        up: NumericTensor::new(
                            vec![intermediate, hidden],
                            packed_gate_up.data[gate_up_start + projection_per_expert
                                ..gate_up_start + 2 * projection_per_expert]
                                .to_vec(),
                        ),
                        down: NumericTensor::new(
                            vec![spec.output_dimensions, intermediate],
                            packed_down.data[down_start..down_start + down_per_expert].to_vec(),
                        ),
                    });
                }
                parameters.push((
                    packed_gate_up,
                    ParameterMetadata::from_spec(&gate_up.weight, gate_up.weight.trainable),
                ));
                parameters.push((
                    packed_down,
                    ParameterMetadata::from_spec(&down.weight, down.weight.trainable),
                ));
            }
            SwiGluExpertLayout::Independent(specs) => {
                for expert_spec in specs {
                    let gate =
                        parameter(&expert_spec.gate.weight, vec![intermediate, hidden], false);
                    let up = parameter(&expert_spec.up.weight, vec![intermediate, hidden], false);
                    let down = parameter(
                        &expert_spec.down.weight,
                        vec![spec.output_dimensions, intermediate],
                        false,
                    );
                    experts.push(NumericExpert {
                        gate: gate.clone(),
                        up: up.clone(),
                        down: down.clone(),
                    });
                    parameters.extend([
                        (
                            gate,
                            ParameterMetadata::from_spec(
                                &expert_spec.gate.weight,
                                expert_spec.gate.weight.trainable,
                            ),
                        ),
                        (
                            up,
                            ParameterMetadata::from_spec(
                                &expert_spec.up.weight,
                                expert_spec.up.weight.trainable,
                            ),
                        ),
                        (
                            down,
                            ParameterMetadata::from_spec(
                                &expert_spec.down.weight,
                                expert_spec.down.weight.trainable,
                            ),
                        ),
                    ]);
                }
            }
        }
        Ok(NumericExpertBank {
            experts,
            parameters,
            limit: spec.limit,
        })
    }
}

#[derive(Debug, Clone)]
struct NumericCache {
    offset: i32,
    window: Option<i32>,
    keys: Option<NumericTensor>,
    values: Option<NumericTensor>,
}

impl NumericCache {
    fn new(window: Option<i32>) -> Self {
        Self {
            offset: 0,
            window,
            keys: None,
            values: None,
        }
    }

    fn retained(&self) -> i32 {
        self.keys.as_ref().map_or(0, |keys| keys.shape[2])
    }
}

impl AttentionCache<NumericTensor> for NumericCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        self.window
    }

    fn update_for_attention(
        &mut self,
        keys: NumericTensor,
        values: NumericTensor,
        context: &NumericContext,
    ) -> Result<(NumericTensor, NumericTensor), Error> {
        let added = keys.shape[2];
        let attention_keys = if let Some(previous) = &self.keys {
            NumericTensor::concatenate(&[previous.clone(), keys], 2, context)?
        } else {
            keys
        };
        let attention_values = if let Some(previous) = &self.values {
            NumericTensor::concatenate(&[previous.clone(), values], 2, context)?
        } else {
            values
        };
        self.offset += added;
        let retained_start =
            self.window
                .map_or(0, |window| (attention_keys.shape[2] - window).max(0)) as usize;
        self.keys =
            Some(attention_keys.axis_slice(2, retained_start, attention_keys.shape[2] as usize));
        self.values = Some(attention_values.axis_slice(
            2,
            retained_start,
            attention_values.shape[2] as usize,
        ));
        Ok((attention_keys, attention_values))
    }

    fn attention(
        &mut self,
        queries: NumericTensor,
        keys: NumericTensor,
        values: NumericTensor,
        scale: f32,
        mask: Option<&NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let query_offset = self.offset - queries.shape[2];
        attention(
            &queries,
            &keys,
            &values,
            scale,
            mask,
            self.window,
            query_offset,
        )
    }
}

fn config(model_type: &str, tied: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": model_type,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 17,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tied
    });
    if model_type == "qwen2" {
        config["attention_bias"] = true.into();
        config["use_sliding_window"] = true.into();
        config["sliding_window"] = 2.into();
        config["max_window_layers"] = 0.into();
    }
    if model_type == "qwen3_moe" {
        config["intermediate_size"] = 0.into();
        config["moe_intermediate_size"] = 6.into();
        config["num_experts"] = 4.into();
        config["num_experts_per_tok"] = 2.into();
        config["norm_topk_prob"] = true.into();
    }
    config
}

struct ForwardResult {
    prefill: Vec<f32>,
    decode: Vec<f32>,
    retained: Vec<i32>,
    sliding_calls: usize,
}

fn forward(model_type: &str, tied: bool) -> Result<ForwardResult, Error> {
    let args =
        qwen::model_args_from_config_value(&config(model_type, tied)).map_err(Error::backend)?;
    let context = NumericContext::default();
    let mut decoder = qwen::new_decoder::<NumericBackend>(&args, &context)?;
    let mut head = (!tied)
        .then(|| {
            NumericBackend::linear(
                LinearSpec {
                    input: args.hidden_size,
                    output: args.vocab_size,
                    weight: ParameterSpec::trainable("lm_head.weight").map_err(Error::backend)?,
                    bias: None,
                    format: eredu_checkpoint::LinearFormat::Dense,
                },
                &context,
            )
        })
        .transpose()?;
    let mut caches = qwen::create_caches(&args, |_, window| NumericCache::new(window))?;

    let prefill_tokens = NumericTensor::token_ids(&[1, 4, 2]);
    let mask = NumericBackend::causal_mask(3, 0, None, &context)?;
    let hidden = decoder.embed(&prefill_tokens, &context)?;
    let hidden = decoder.forward_embedded(hidden, Some(&mask), true, &mut caches, &context)?;
    let prefill_logits = match &mut head {
        Some(head) => head.forward(&hidden, &context)?,
        None => decoder.embeddings.as_linear(&hidden, &context)?,
    };

    let decode_tokens = NumericTensor::token_ids(&[3]);
    let hidden = decoder.forward(&decode_tokens, None, &mut caches, &context)?;
    let decode_logits = match &mut head {
        Some(head) => head.forward(&hidden, &context)?,
        None => decoder.embeddings.as_linear(&hidden, &context)?,
    };
    let cache_state = caches
        .iter()
        .map(|cache| cache.as_ref().unwrap().retained())
        .collect();
    Ok(ForwardResult {
        prefill: prefill_logits.data,
        decode: decode_logits.data,
        retained: cache_state,
        sliding_calls: context.sliding_attention_calls.get(),
    })
}

fn checksum(values: &[f32]) -> [f32; 3] {
    [
        values.iter().sum(),
        values
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f32 * value)
            .sum(),
        values.iter().map(|value| value * value).sum(),
    ]
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2.0e-5,
        "expected {expected:.9}, got {actual:.9}"
    );
}

#[test]
fn numerical_qwen2_qwen3_and_moe_prefill_decode_goldens() {
    let cases = [
        (
            "qwen2",
            false,
            [0.362_308_77, 8.959_988, 0.310_638_3],
            [-0.036_465_22, 0.943_931_04, 0.095_674_14],
        ),
        (
            "qwen3",
            true,
            [0.331_483_6, 9.507_395, 0.279_646_04],
            [0.250_913_5, 2.956_005, 0.086_268_68],
        ),
        (
            "qwen3_moe",
            false,
            [0.327_792_88, 7.773_554, 0.309_266_42],
            [-0.042_136_565, 0.978_026_4, 0.092_903_41],
        ),
    ];
    for (model_type, tied, expected_prefill, expected_decode) in cases {
        let result = forward(model_type, tied).unwrap();
        let actual_prefill = checksum(&result.prefill);
        let actual_decode = checksum(&result.decode);
        for (actual, expected) in actual_prefill.into_iter().zip(expected_prefill) {
            assert_close(actual, expected);
        }
        for (actual, expected) in actual_decode.into_iter().zip(expected_decode) {
            assert_close(actual, expected);
        }
        assert_eq!(
            result.retained,
            if model_type == "qwen2" {
                vec![2]
            } else {
                vec![4]
            }
        );
        assert_eq!(result.sliding_calls, usize::from(model_type == "qwen2"));
    }
}

#[test]
fn explicit_rotary_embeddings_match_offset_positions() {
    let context = NumericContext::default();
    let input = NumericTensor::new(
        vec![1, 1, 2, 4],
        vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
    );
    let mut rotary = NumericRotary {
        dimensions: 4,
        traditional: false,
        base: 10_000.0,
    };
    let offset_output = rotary
        .forward(&input, RotaryPosition::Offset(3), &context)
        .unwrap();
    let mut cosine = NumericTensor::zeros(vec![2, 2]);
    let mut sine = cosine.clone();
    for position in 0..2 {
        for frequency in 0..2 {
            let theta = (3 + position) as f32 / 10_000_f32.powf(frequency as f32 / 2.0);
            cosine.data[position * 2 + frequency] = theta.cos();
            sine.data[position * 2 + frequency] = theta.sin();
        }
    }
    let embedded_output = rotary
        .forward(
            &input,
            RotaryPosition::Embeddings {
                cosine: &cosine,
                sine: &sine,
            },
            &context,
        )
        .unwrap();
    for (offset, embedded) in offset_output.data.iter().zip(embedded_output.data) {
        assert_close(*offset, embedded);
    }
}

#[test]
fn partial_rotary_preserves_non_rotary_head_dimensions() {
    let context = NumericContext::default();
    let input = NumericTensor::new(vec![1, 1, 6], vec![9.0, 1.0, 2.0, 3.0, 4.0, -7.0]);
    let mut rotary = NumericRotary {
        dimensions: 4,
        traditional: false,
        base: 10_000.0,
    };
    let output = rotary
        .forward_subspace(
            &input,
            RotarySubspace::Range {
                start: 1,
                dimensions: 4,
            },
            RotaryPosition::Offset(2),
            &context,
        )
        .unwrap();
    let expected_rotary = rotary
        .forward(
            &input.axis_slice(2, 1, 5),
            RotaryPosition::Offset(2),
            &context,
        )
        .unwrap();
    assert_eq!(output.data[0], 9.0);
    assert_eq!(output.data[5], -7.0);
    for (actual, expected) in output.data[1..5].iter().zip(expected_rotary.data) {
        assert_close(*actual, expected);
    }
}

#[test]
fn hyper_connection_sinkhorn_and_head_match_reference_semantics() {
    struct ZeroParameters;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for ZeroParameters {
        fn visit_mut(&mut self, _: ParameterMetadata, value: &'a mut NumericTensor) {
            value.data.fill(0.0);
        }
    }

    let context = NumericContext::default();
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    let mut connection = HyperConnection::<NumericBackend>::new(
        HyperConnectionSpec {
            streams: 2,
            hidden_size: 1,
            sinkhorn_iterations: 4,
            epsilon: 1e-3,
            function: parameter("hc.function"),
            base: parameter("hc.base"),
            scale: parameter("hc.scale"),
        },
        &context,
    )
    .unwrap();
    connection.visit_parameters_mut(&mut ZeroParameters);
    let residual = NumericTensor::new(vec![1, 1, 2, 1], vec![2.0, 6.0]);
    let state = connection.collapse(&residual, 1e-6, &context).unwrap();
    assert_close(state.pre.data[0], 0.501);
    assert_close(state.pre.data[1], 0.501);
    assert_close(state.post.data[0], 1.0);
    assert_close(state.post.data[1], 1.0);
    assert_close(state.collapsed.data[0], 4.008);
    for value in state.combination.data.iter().skip(1) {
        assert_close(*value, state.combination.data[0]);
    }
    let expanded = connection
        .expand(
            &NumericTensor::new(vec![1, 1, 1], vec![10.0]),
            &residual,
            &state,
            &context,
        )
        .unwrap();
    let expected = 10.0 + 8.0 * state.combination.data[0];
    assert_close(expanded.data[0], expected);
    assert_close(expanded.data[1], expected);

    let mut head = HyperHead::<NumericBackend>::new(
        HyperHeadSpec {
            streams: 2,
            hidden_size: 1,
            norm_epsilon: 1e-6,
            epsilon: 1e-3,
            function: parameter("head.function"),
            base: parameter("head.base"),
            scale: parameter("head.scale"),
        },
        &context,
    )
    .unwrap();
    head.visit_parameters_mut(&mut ZeroParameters);
    let collapsed = head.forward(&residual, &context).unwrap();
    assert_close(collapsed.data[0], 4.008);
}

#[test]
fn compressed_cache_growth_boundaries_and_rollback_are_backend_neutral() {
    let context = NumericContext::default();
    let state = |start: i32, tokens: i32| CompressedAttentionState {
        latent: NumericTensor::new(
            vec![1, tokens, 1],
            (start..start + tokens).map(|value| value as f32).collect(),
        ),
        rotary: NumericTensor::new(
            vec![1, tokens, 1],
            (start..start + tokens)
                .map(|value| 100.0 + value as f32)
                .collect(),
        ),
    };

    let mut resident = NumericCompressedCache::resident();
    resident.append(state(0, 2), &context).unwrap();
    let checkpoint = resident.checkpoint();
    let view = resident.append(state(2, 1), &context).unwrap();
    assert_eq!(resident.offset(), 3);
    assert_eq!(view.resident().unwrap().latent.data, vec![0.0, 1.0, 2.0]);
    resident.restore(&checkpoint, &context).unwrap();
    assert_eq!(resident.offset(), 2);
    assert_eq!(
        resident.state.as_ref().unwrap().rotary.data,
        vec![100.0, 101.0]
    );

    let mut paged = NumericCompressedCache::paged(2);
    let view = paged.append(state(0, 3), &context).unwrap();
    assert!(view.is_paged());
    paged.append(state(3, 2), &context).unwrap();
    let mut ranges = Vec::new();
    let scan = paged
        .visit_blocks(1, &context, |block| {
            ranges.push((block.start, block.end));
            Ok(block.state.latent.data.len() as u64 * 8)
        })
        .unwrap();
    assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 5)]);
    assert_eq!(scan.blocks, 3);
    assert_eq!(scan.reconstruction_scratch_bytes, 16);
    paged.finalize().unwrap();
    paged.clear().unwrap();
    assert_eq!(paged.offset(), 0);
}

#[test]
fn reference_attention_obeys_causal_and_sliding_semantics() {
    let queries = NumericTensor::new(vec![1, 1, 2, 1], vec![1.0, 1.0]);
    let keys = NumericTensor::new(vec![1, 1, 2, 1], vec![0.0, 3.0_f32.ln()]);
    let values = NumericTensor::new(vec![1, 1, 2, 1], vec![2.0, 10.0]);

    let causal = attention(&queries, &keys, &values, 1.0, None, None, 0).unwrap();
    assert_close(causal.data[0], 2.0);
    assert_close(causal.data[1], 8.0);

    let sliding = attention(&queries, &keys, &values, 1.0, None, Some(1), 0).unwrap();
    assert_close(sliding.data[0], 2.0);
    assert_close(sliding.data[1], 10.0);
}

#[test]
fn reference_indexed_attention_shares_softmax_with_sink() {
    let queries = NumericTensor::new(vec![1, 1, 1, 1], vec![1.0]);
    let local_keys = NumericTensor::new(vec![1, 1, 1], vec![1.0]);
    let local_values = NumericTensor::new(vec![1, 1, 1], vec![10.0]);
    let pooled_keys = NumericTensor::new(vec![1, 2, 1], vec![2.0, 3.0]);
    let pooled_values = NumericTensor::new(vec![1, 2, 1], vec![20.0, 30.0]);
    let selected_positions = NumericTensor::new(vec![1, 1, 1], vec![1.0]);
    let sinks = NumericTensor::new(vec![1], vec![0.0]);
    let output = indexed_attention(IndexedAttentionInput {
        queries: &queries,
        local_keys: &local_keys,
        local_values: &local_values,
        pooled_keys: &pooled_keys,
        pooled_values: &pooled_values,
        selected_positions: &selected_positions,
        scale: 1.0,
        local_mask: None,
        pooled_mask: None,
        sinks: Some(&sinks),
    })
    .unwrap();
    let expected =
        (10.0 * 1.0_f32.exp() + 30.0 * 3.0_f32.exp()) / (1.0_f32.exp() + 3.0_f32.exp() + 1.0);
    assert_close(output.data[0], expected);
}

#[test]
fn reference_router_and_packed_experts_match_analytical_values() {
    let context = NumericContext::default();
    let router_spec = ParameterSpec::trainable("router.weight").unwrap();
    let mut router = NumericRouter {
        linear: NumericLinear {
            weight: NumericTensor::new(
                vec![3, 2],
                vec![0.0, 0.0, 2.0_f32.ln(), 0.0, 4.0_f32.ln(), 0.0],
            ),
            weight_metadata: ParameterMetadata::from_spec(&router_spec, true),
            bias: None,
        },
        routing: TopKRoutingSpec::new(3, 2, eredu_nn::RoutingScoring::Softmax, true).unwrap(),
        correction_bias: None,
    };
    let input = NumericTensor::new(vec![1, 2], vec![1.0, 0.0]);
    let routes = router.route(&input, &context).unwrap();
    assert_eq!(routes.expert_ids.data, [2.0, 1.0]);
    assert_close(routes.selected_scores.data[0], 4.0 / 7.0);
    assert_close(routes.selected_scores.data[1], 2.0 / 7.0);
    assert_close(routes.route_weights.data[0], 2.0 / 3.0);
    assert_close(routes.route_weights.data[1], 1.0 / 3.0);

    let mut experts = NumericExpertBank {
        experts: vec![
            NumericExpert {
                gate: NumericTensor::new(vec![1, 2], vec![1.0, 0.0]),
                up: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                down: NumericTensor::new(vec![2, 1], vec![1.0, 2.0]),
            },
            NumericExpert {
                gate: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                up: NumericTensor::new(vec![1, 2], vec![1.0, 0.0]),
                down: NumericTensor::new(vec![2, 1], vec![2.0, -1.0]),
            },
        ],
        parameters: Vec::new(),
        limit: None,
    };
    let routes = RoutingResult {
        expert_ids: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
        selected_scores: NumericTensor::new(vec![1, 2], vec![0.25, 0.75]),
        route_weights: NumericTensor::new(vec![1, 2], vec![0.25, 0.75]),
    };
    let output = experts
        .forward_routed(
            &NumericTensor::new(vec![1, 2], vec![2.0, 1.0]),
            &routes,
            &context,
        )
        .unwrap();
    assert_close(output.data[0], 2.633_574_2);
    assert_close(output.data[1], -0.215_790_8);
}

#[test]
fn normalized_low_rank_projection_matches_analytical_reference() {
    let parameter = |name: &str| ParameterSpec::trainable(name).unwrap();
    let mut projection = LowRankProjection::<NumericBackend>::new(
        LowRankProjectionSpec {
            first: Some(LinearSpec {
                input: 2,
                output: 2,
                weight: parameter("low_rank.first.weight"),
                bias: None,
                format: eredu_nn::LinearFormat::Dense,
            }),
            normalization: NormalizationSpec {
                dimensions: 2,
                epsilon: 0.0,
                weight: parameter("low_rank.norm.weight"),
            },
            second: LinearSpec {
                input: 2,
                output: 1,
                weight: parameter("low_rank.second.weight"),
                bias: None,
                format: eredu_nn::LinearFormat::Dense,
            },
        },
        &NumericContext::default(),
    )
    .unwrap();

    struct Loader;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Loader {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            value.data = match metadata.id.as_str() {
                "low_rank.first.weight" => vec![1.0, 0.0, 0.0, 1.0],
                "low_rank.norm.weight" => vec![1.0, 1.0],
                "low_rank.second.weight" => vec![2.0, -1.0],
                unexpected => panic!("unexpected low-rank parameter {unexpected}"),
            };
        }
    }
    projection.visit_parameters_mut(&mut Loader);
    let output = projection
        .forward(
            &NumericTensor::new(vec![1, 2], vec![3.0, 4.0]),
            &NumericContext::default(),
        )
        .unwrap();
    assert_eq!(output.shape, [1, 1]);
    assert_close(output.data[0], 2.0 / 12.5_f32.sqrt());
}

#[test]
fn grouped_sigmoid_and_caller_selected_routes_preserve_unbiased_scores() {
    let router_weight = ParameterSpec::trainable("router.weight").unwrap();
    let correction = ParameterSpec::trainable("router.correction_bias").unwrap();
    let routing = TopKRoutingSpec::new(4, 1, eredu_nn::RoutingScoring::Sigmoid, true)
        .unwrap()
        .with_groups(2, 1)
        .unwrap()
        .with_weight_policy(1e-20, 2.0)
        .unwrap();
    let mut router = NumericRouter {
        linear: NumericLinear {
            weight: NumericTensor::new(vec![4, 1], vec![2.0, 1.0, 3.0, 0.0]),
            weight_metadata: ParameterMetadata::from_spec(&router_weight, true),
            bias: None,
        },
        routing,
        correction_bias: Some((
            NumericTensor::new(vec![4], vec![0.0, 0.0, 2.0, 0.0]),
            ParameterMetadata::from_spec(&correction, true),
        )),
    };
    let input = NumericTensor::new(vec![1, 1], vec![1.0]);
    let learned = router.route(&input, &NumericContext::default()).unwrap();
    assert_eq!(learned.expert_ids.data, [2.0]);
    assert_close(
        learned.selected_scores.data[0],
        1.0 / (1.0 + (-3.0_f32).exp()),
    );
    assert_close(learned.route_weights.data[0], 2.0);

    router.routing = TopKRoutingSpec::new(4, 1, eredu_nn::RoutingScoring::SqrtSoftplus, true)
        .unwrap()
        .with_weight_policy(1e-20, 1.5)
        .unwrap();
    let selected = router
        .route_selected(
            &input,
            &NumericTensor::new(vec![1, 1], vec![3.0]),
            &NumericContext::default(),
        )
        .unwrap();
    assert_eq!(selected.expert_ids.data, [3.0]);
    assert_close(selected.selected_scores.data[0], 2.0_f32.ln().sqrt());
    assert_close(selected.route_weights.data[0], 1.5);
}

#[test]
fn limited_swiglu_caps_gate_and_up_before_activation() {
    let output = NumericBackend::swiglu(
        NumericTensor::new(vec![1, 2], vec![10.0, -10.0]),
        NumericTensor::new(vec![1, 2], vec![10.0, -10.0]),
        Some(eredu_nn::SwiGluLimit::new(2.0).unwrap()),
        &NumericContext::default(),
    )
    .unwrap();
    assert_close(output.data[0], 4.0 / (1.0 + (-2.0_f32).exp()));
    assert_close(output.data[1], 20.0 / (1.0 + 10.0_f32.exp()));
}

#[test]
fn v3_mla_resident_and_paged_state_share_one_neutral_forward() {
    for query_rank in [None, Some(2)] {
        let mut config = serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 4,
            "intermediate_size": 8,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 8,
            "max_position_embeddings": 64,
            "kv_lora_rank": 2,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "first_k_dense_replace": 1,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "n_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false
        });
        if let Some(rank) = query_rank {
            config["q_lora_rank"] = serde_json::json!(rank);
        }
        let args = deepseek::parse_v3_config(&config).unwrap();
        let context = NumericContext::default();
        let mut resident_attention =
            deepseek::attention::v3::Attention::<NumericBackend>::new(&args, 0, &context).unwrap();
        let mut paged_attention =
            deepseek::attention::v3::Attention::<NumericBackend>::new(&args, 0, &context).unwrap();
        let mut resident = NumericCompressedCache::resident();
        let mut paged = NumericCompressedCache::paged(1);
        let prefill = NumericTensor::new(
            vec![1, 2, 4],
            vec![0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3],
        );
        let resident_prefill = resident_attention
            .forward(&prefill, None, Some(&mut resident), &context)
            .unwrap();
        let paged_prefill = paged_attention
            .forward(&prefill, None, Some(&mut paged), &context)
            .unwrap();
        assert_eq!(resident_prefill.shape, paged_prefill.shape);
        for (resident, paged) in resident_prefill.data.iter().zip(&paged_prefill.data) {
            assert_close(*resident, *paged);
        }

        let decode = NumericTensor::new(vec![1, 1, 4], vec![0.1, 0.3, -0.4, 0.2]);
        let resident_decode = resident_attention
            .forward(&decode, None, Some(&mut resident), &context)
            .unwrap();
        let paged_decode = paged_attention
            .forward(&decode, None, Some(&mut paged), &context)
            .unwrap();
        assert_eq!(resident_decode.shape, paged_decode.shape);
        for (resident, paged) in resident_decode.data.iter().zip(&paged_decode.data) {
            assert_close(*resident, *paged);
        }
    }
}

#[test]
fn v3_target_model_runs_end_to_end_through_resident_runtime() {
    let args = deepseek::parse_v3_config(&serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 8,
        "max_position_embeddings": 64,
        "q_lora_rank": 2,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let context = NumericContext::default();
    let layout = deepseek::v3::state_layout(&args).unwrap();
    let mut resident_state = DeviceState::<NumericBackend, _>::create(layout.clone(), |_, _| {
        Ok::<_, Error>(NumericCompressedCache::resident())
    })
    .unwrap();
    let mut paged_state = DeviceState::<NumericBackend, _>::create(layout, |_, _| {
        Ok::<_, Error>(NumericCompressedCache::paged(1))
    })
    .unwrap();
    let resident_model =
        deepseek::v3::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
    let paged_model = deepseek::v3::Model::<NumericBackend>::new(args, &context).unwrap();
    let mut resident = ResidentRuntime::new(resident_model, &context).unwrap();
    let mut paged = ResidentRuntime::new(paged_model, &context).unwrap();

    for tokens in [
        NumericTensor::token_ids(&[1, 3]),
        NumericTensor::token_ids(&[2]),
    ] {
        let resident_output = resident
            .forward(
                deepseek::mtp::EmbeddedInput::target(&tokens, None),
                &mut resident_state,
                &context,
            )
            .unwrap();
        let paged_output = paged
            .forward(
                deepseek::mtp::EmbeddedInput::target(&tokens, None),
                &mut paged_state,
                &context,
            )
            .unwrap();
        assert_eq!(resident_output.shape, [1, tokens.dim(1), 8]);
        assert_eq!(resident_output.shape, paged_output.shape);
        for (resident, paged) in resident_output.data.iter().zip(&paged_output.data) {
            assert_close(*resident, *paged);
        }
    }
    assert_eq!(resident_state.layer(0).unwrap().offset(), 3);
    assert_eq!(paged_state.layer(0).unwrap().offset(), 3);
}

fn tiny_v4_args() -> deepseek::V4Args {
    deepseek::parse_v4_config(&serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "qk_rope_head_dim": 2,
        "q_lora_rank": 2,
        "o_lora_rank": 2,
        "o_groups": 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "sliding_window": 4,
        "compress_ratios": [0, 4, 128],
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 1,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 1,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 4.0
    }))
    .unwrap()
}

#[test]
fn v4_attention_policies_are_invariant_to_prefill_chunking() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    let input = NumericTensor::new(
        vec![1, 8, 4],
        (0..32)
            .map(|index| ((index * 17 % 31) as f32 - 15.0) / 20.0)
            .collect(),
    );
    for (layer, ratios) in [(0, vec![]), (1, vec![4, 4]), (2, vec![128])] {
        let mut whole =
            deepseek::attention::v4::Attention::<NumericBackend>::new(&args, layer, &context)
                .unwrap();
        let mut chunked =
            deepseek::attention::v4::Attention::<NumericBackend>::new(&args, layer, &context)
                .unwrap();
        let mut whole_cache = NumericPoolingCache::new(args.sliding_window, &ratios);
        let mut chunked_cache = NumericPoolingCache::new(args.sliding_window, &ratios);
        let expected = whole
            .forward(&input, None, Some(&mut whole_cache), &context)
            .unwrap();
        let mut outputs = Vec::new();
        for (start, end) in [(0, 3), (3, 5), (5, 8)] {
            outputs.push(
                chunked
                    .forward(
                        &input.axis_slice(1, start, end),
                        None,
                        Some(&mut chunked_cache),
                        &context,
                    )
                    .unwrap(),
            );
        }
        let actual = NumericTensor::concatenate(&outputs, 1, &context).unwrap();
        assert_eq!(expected.shape, [1, 8, 4]);
        assert_eq!(expected.shape, actual.shape);
        for (index, (expected, actual)) in expected.data.iter().zip(&actual.data).enumerate() {
            assert!(
                (*expected - *actual).abs() <= 2.0e-5,
                "V4 layer {layer} output {index}: expected {expected:.9}, got {actual:.9}"
            );
        }
        assert_eq!(whole_cache.offset(), 8);
        assert_eq!(chunked_cache.offset(), 8);
    }
}

#[test]
fn v4_hyper_block_shares_hash_and_learned_moe_execution() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    let hidden = NumericTensor::new(
        vec![1, 8, 2, 4],
        (0..64)
            .map(|index| ((index * 11 % 37) as f32 - 18.0) / 30.0)
            .collect(),
    );
    let tokens = NumericTensor::token_ids(&[1, 2, 3, 4, 5, 6, 7, 8]);

    let mut hash_block =
        deepseek::block::V4Block::<NumericBackend>::new(&args, 0, &context).unwrap();
    let mut local_cache = NumericPoolingCache::new(args.sliding_window, &[]);
    let hash_output = hash_block
        .forward(&hidden, &tokens, None, Some(&mut local_cache), &context)
        .unwrap();
    assert_eq!(hash_output.shape, [1, 8, 2, 4]);

    let mut whole = deepseek::block::V4Block::<NumericBackend>::new(&args, 1, &context).unwrap();
    let mut chunked = deepseek::block::V4Block::<NumericBackend>::new(&args, 1, &context).unwrap();
    let mut whole_cache = NumericPoolingCache::new(args.sliding_window, &[4, 4]);
    let mut chunked_cache = NumericPoolingCache::new(args.sliding_window, &[4, 4]);
    let expected = whole
        .forward(&hidden, &tokens, None, Some(&mut whole_cache), &context)
        .unwrap();
    let mut outputs = Vec::new();
    for (start, end) in [(0, 3), (3, 5), (5, 8)] {
        outputs.push(
            chunked
                .forward(
                    &hidden.axis_slice(1, start, end),
                    &tokens.axis_slice(1, start, end),
                    None,
                    Some(&mut chunked_cache),
                    &context,
                )
                .unwrap(),
        );
    }
    let actual = NumericTensor::concatenate(&outputs, 1, &context).unwrap();
    assert_eq!(expected.shape, [1, 8, 2, 4]);
    for (index, (expected, actual)) in expected.data.iter().zip(&actual.data).enumerate() {
        assert!(
            (*expected - *actual).abs() <= 2.0e-5,
            "V4 hyper block output {index}: expected {expected:.9}, got {actual:.9}"
        );
    }
}

#[test]
fn v4_target_model_runs_end_to_end_through_resident_runtime() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    let layout = deepseek::v4::state_layout(&args).unwrap();
    let state_args = args.clone();
    let mut state = DeviceState::<NumericBackend, _>::create(layout, move |layer, _| {
        let ratios = match state_args.attention_policy(layer).unwrap() {
            deepseek::V4AttentionPolicy::Local => Vec::new(),
            deepseek::V4AttentionPolicy::Compressed { ratio: 4 } => vec![4, 4],
            deepseek::V4AttentionPolicy::Compressed { ratio } => vec![ratio],
        };
        Ok::<_, Error>(NumericPoolingCache::new(state_args.sliding_window, &ratios))
    })
    .unwrap();
    let model = deepseek::v4::Model::<NumericBackend>::new(args, &context).unwrap();
    let mut runtime = ResidentRuntime::new(model, &context).unwrap();

    for tokens in [
        NumericTensor::token_ids(&[1, 2, 3, 4, 5, 6, 7, 8]),
        NumericTensor::token_ids(&[9]),
    ] {
        let output = runtime
            .forward(
                deepseek::mtp::EmbeddedInput::target(&tokens, None),
                &mut state,
                &context,
            )
            .unwrap();
        assert_eq!(output.shape, [1, tokens.dim(1), 16]);
        assert!(output.data.iter().all(|value| value.is_finite()));
    }
    for layer in 0..3 {
        assert_eq!(state.layer(layer).unwrap().offset(), 9);
    }
}

#[test]
fn embedded_v3_and_v4_prediction_layers_reuse_target_blocks() {
    let v3 = deepseek::parse_v3_config(&serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 8,
        "max_position_embeddings": 64,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let context = NumericContext::default();
    let tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let embedded = NumericTensor::new(
        vec![1, 3, 4],
        vec![
            0.1, 0.2, -0.1, 0.3, -0.2, 0.4, 0.1, 0.0, 0.3, -0.3, 0.2, 0.1,
        ],
    );
    let hidden = NumericTensor::new(
        vec![1, 3, 4],
        vec![
            0.2, -0.1, 0.4, 0.1, 0.3, 0.2, -0.2, 0.1, -0.1, 0.5, 0.2, -0.3,
        ],
    );
    let mut v3_layer =
        deepseek::mtp::V3PredictionLayer::<NumericBackend>::new(&v3, 0, &context).unwrap();
    let mut v3_cache = NumericCompressedCache::resident();
    let v3_output = v3_layer
        .forward(&hidden, &embedded, &tokens, &mut v3_cache, &context)
        .unwrap();
    assert_eq!(v3_output.logits.shape, [1, 3, 8]);
    assert_eq!(v3_output.hidden.shape, [1, 3, 4]);
    assert_eq!(v3_output.tokens.data, tokens.data);
    assert_eq!(v3_cache.offset(), 3);

    let v4 = deepseek::parse_v4_config(&serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "qk_rope_head_dim": 2,
        "q_lora_rank": 2,
        "o_lora_rank": 2,
        "o_groups": 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "sliding_window": 4,
        "compress_ratios": [0, 4, 128, 0],
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 1,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 1,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 4.0,
        "num_nextn_predict_layers": 1
    }))
    .unwrap();
    let mut v4_layer =
        deepseek::mtp::V4PredictionLayer::<NumericBackend>::new(&v4, 0, &context).unwrap();
    let mut v4_cache = NumericPoolingCache::new(v4.sliding_window, &[]);
    let v4_hidden = hidden
        .expand_dims(2, &context)
        .unwrap()
        .broadcast_to(&[1, 3, 2, 4], &context)
        .unwrap();
    let mut v4_head = NumericBackend::linear(
        LinearSpec {
            input: v4.hidden_size,
            output: v4.vocab_size,
            weight: ParameterSpec::trainable("head.weight").unwrap(),
            bias: None,
            format: v4.linear_format,
        },
        &context,
    )
    .unwrap();
    let v4_output = v4_layer
        .forward(
            &v4_hidden,
            &embedded,
            &tokens,
            &mut v4_cache,
            &mut v4_head,
            &context,
        )
        .unwrap();
    assert_eq!(v4_output.logits.shape, [1, 3, 16]);
    assert_eq!(v4_output.hidden.shape, [1, 3, 2, 4]);
    assert_eq!(v4_output.tokens.data, tokens.data);
    assert_eq!(v4_cache.offset(), 3);
}

#[test]
fn deepseek_execution_graphs_run_target_mtp_and_dspark_transactions() {
    let context = NumericContext::default();
    let v3 = deepseek::parse_v3_config(&serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 8,
        "max_position_embeddings": 64,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let mut v3_state = DeviceState::<NumericBackend, _>::create(
        deepseek::v3::state_layout(&v3).unwrap(),
        |_, _| Ok::<_, Error>(NumericCompressedCache::resident()),
    )
    .unwrap();
    let mut v3_runtime = ResidentRuntime::new(
        deepseek::v3::Model::<NumericBackend>::new(v3, &context).unwrap(),
        &context,
    )
    .unwrap();
    let target_tokens = NumericTensor::token_ids(&[1, 2]);
    let (target_logits, target_context) = v3_runtime
        .forward_with_context(
            deepseek::mtp::EmbeddedInput::target(&target_tokens, None),
            &mut v3_state,
            &context,
        )
        .unwrap();
    let capture = target_context.target_capture().unwrap().axis_slice(1, 1, 2);
    assert_eq!(target_logits.shape, [1, 2, 8]);
    assert_eq!(v3_state.layer(0).unwrap().offset(), 2);
    assert_eq!(v3_state.layer(1).unwrap().offset(), 0);
    let draft_token = NumericTensor::token_ids(&[3]);
    let draft_logits = v3_runtime
        .forward(
            deepseek::mtp::EmbeddedInput::draft(&draft_token, &capture, 0),
            &mut v3_state,
            &context,
        )
        .unwrap();
    assert_eq!(draft_logits.shape, [1, 1, 8]);
    assert_eq!(v3_state.layer(0).unwrap().offset(), 2);
    assert_eq!(v3_state.layer(1).unwrap().offset(), 1);

    let v4 = deepseek::parse_v4_config(&serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "qk_rope_head_dim": 2,
        "q_lora_rank": 2,
        "o_lora_rank": 2,
        "o_groups": 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "sliding_window": 8,
        "compress_ratios": [0, 0, 0, 0],
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 1,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 0,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 4.0,
        "num_nextn_predict_layers": 2,
        "dspark_block_size": 2,
        "dspark_noise_token_id": 0,
        "dspark_target_layer_ids": [0, 1],
        "dspark_markov_rank": 2
    }))
    .unwrap();
    let mut v4_state = DeviceState::<NumericBackend, _>::create(
        deepseek::v4::state_layout(&v4).unwrap(),
        |_, _| Ok::<_, Error>(NumericPoolingCache::new(8, &[])),
    )
    .unwrap();
    let mut v4_runtime = ResidentRuntime::new(
        deepseek::v4::Model::<NumericBackend>::new(v4, &context).unwrap(),
        &context,
    )
    .unwrap();
    let (target_logits, target_context) = v4_runtime
        .forward_with_context(
            deepseek::mtp::EmbeddedInput::target(&target_tokens, None),
            &mut v4_state,
            &context,
        )
        .unwrap();
    let captures = target_context.target_capture().unwrap().clone();
    assert_eq!(target_logits.shape, [1, 2, 16]);
    assert_eq!(captures.shape, [1, 2, 8]);
    v4_runtime
        .forward(
            deepseek::mtp::EmbeddedInput::dspark_context(&captures),
            &mut v4_state,
            &context,
        )
        .unwrap();
    assert_eq!(v4_state.layer(2).unwrap().offset(), 2);
    assert_eq!(v4_state.layer(3).unwrap().offset(), 2);

    let mut transaction = eredu_runtime::DraftStateTransaction::fork(&v4_state);
    let anchor = NumericTensor::token_ids(&[3]);
    let proposal = v4_runtime
        .forward(
            deepseek::mtp::EmbeddedInput::dspark_proposal(&anchor, 2),
            transaction.draft_mut(),
            &context,
        )
        .unwrap();
    assert_eq!(proposal.shape, [1, 2, 16]);
    assert_eq!(transaction.draft_mut().layer(2).unwrap().offset(), 4);
    assert_eq!(v4_state.layer(2).unwrap().offset(), 2);
    transaction.rollback(&mut v4_state);
    assert_eq!(v4_state.layer(2).unwrap().offset(), 2);
}

#[test]
fn v4_observer_reports_sparse_indexes_hyper_streams_and_routes() {
    struct Observer {
        paths: Vec<String>,
    }

    impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observer {
        fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
            self.paths.push(path.into());
            Ok(())
        }

        fn intervene(
            &mut self,
            path: &str,
            value: &NumericTensor,
        ) -> Result<Option<NumericTensor>, Error> {
            Ok((path == "layers.1.output").then(|| NumericTensor::zeros(value.shape.clone())))
        }

        fn observe_routing(
            &mut self,
            routing: eredu_runtime::RoutingObservation<'_, NumericTensor>,
        ) -> Result<(), Error> {
            self.paths.push(format!("{}.routing", routing.path));
            Ok(())
        }
    }

    let args = tiny_v4_args();
    let context = NumericContext::default();
    let mut block = deepseek::block::V4Block::<NumericBackend>::new(&args, 1, &context).unwrap();
    let tokens = NumericTensor::token_ids(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let hidden = NumericTensor::new(
        vec![1, 8, 2, 4],
        (0..64).map(|index| index as f32 / 64.0 - 0.5).collect(),
    );
    let mut cache = NumericPoolingCache::new(args.sliding_window, &[4, 4]);
    let mut observer = Observer { paths: Vec::new() };
    let output = block
        .forward_observed(
            "layers.1",
            &hidden,
            &tokens,
            None,
            Some(&mut cache),
            &context,
            &mut observer,
        )
        .unwrap();
    assert!(output.data.iter().all(|value| *value == 0.0));
    for expected in [
        "layers.1.compressed_attention.selected_indexes",
        "layers.1.hyper.attention.streams",
        "layers.1.feed_forward.routing",
        "layers.1.output",
    ] {
        assert!(
            observer.paths.iter().any(|path| path == expected),
            "missing observation {expected:?} from {:?}",
            observer.paths
        );
    }
}
