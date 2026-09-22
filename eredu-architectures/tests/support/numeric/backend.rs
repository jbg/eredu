#[derive(Debug, Clone)]
struct NumericTensor {
    shape: Vec<i32>,
    data: Vec<f32>,
    dtype: eredu_core::checkpoint::TensorDtype,
    // Test-only lifetime evidence, never used by scalar arithmetic.
    retirement_probe: Option<Arc<()>>,
    // A prepared publication copy keeps its real metadata payer until it retires.
    publication_funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}

impl NumericTensor {
    fn new(shape: impl Into<Vec<i32>>, data: Vec<f32>) -> Self {
        let shape = shape.into();
        assert_eq!(elements(&shape), data.len());
        Self {
            shape,
            data,
            dtype: eredu_core::checkpoint::TensorDtype::F32,
            retirement_probe: None,
            publication_funding: None,
        }
    }

    fn with_dtype(mut self, dtype: eredu_core::checkpoint::TensorDtype) -> Self {
        self.dtype = dtype;
        self
    }

    fn zeros(shape: impl Into<Vec<i32>>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![0.0; elements(&shape)],
            shape,
            dtype: eredu_core::checkpoint::TensorDtype::F32,
            retirement_probe: None,
            publication_funding: None,
        }
    }

    fn token_ids(ids: &[usize]) -> Self {
        Self::new(
            vec![1, i32::try_from(ids.len()).unwrap()],
            ids.iter().map(|id| *id as f32).collect(),
        )
        .with_dtype(eredu_core::checkpoint::TensorDtype::U32)
    }

    fn axis_slice(&self, axis: usize, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.shape[axis] as usize);
        let mut shape = self.shape.clone();
        shape[axis] = (end - start) as i32;
        let mut output = Self::zeros(shape);
        output.dtype = self.dtype.clone();
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
        let rank = self.shape.len().max(rhs.shape.len());
        let mut shape = Vec::with_capacity(rank);
        for axis in 0..rank {
            let left = axis
                .checked_sub(rank - self.shape.len())
                .map_or(1, |axis| self.shape[axis]);
            let right = axis
                .checked_sub(rank - rhs.shape.len())
                .map_or(1, |axis| rhs.shape[axis]);
            if left != right && left != 1 && right != 1 {
                return Err(Error::backend(format!(
                    "numeric tensor shape mismatch: {:?} versus {:?}",
                    self.shape, rhs.shape
                )));
            }
            shape.push(left.max(right));
        }
        let mut data = Vec::with_capacity(elements(&shape));
        for index in 0..elements(&shape) {
            let coordinate = unravel(index, &shape);
            let project = |source: &NumericTensor| {
                let skip = rank - source.shape.len();
                coordinate[skip..]
                    .iter()
                    .zip(&source.shape)
                    .map(|(index, dimension)| if *dimension == 1 { 0 } else { *index })
                    .collect::<Vec<_>>()
            };
            data.push(operation(
                self.data[offset(&project(self), &self.shape)],
                rhs.data[offset(&project(rhs), &rhs.shape)],
            ));
        }
        Ok(Self::new(shape, data))
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
                let outside_window = self.window != i32::MAX
                    && key_position <= query_position.saturating_sub(self.window);
                if key_position > query_position || outside_window {
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

    fn checkpoint(&self) -> Result<Self::Checkpoint, Error> {
        Ok(self.clone())
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
        Ok(Self::zeros(shape.to_vec()).with_dtype(eredu_core::checkpoint::TensorDtype::I32))
    }

    fn from_f32_slice(values: &[f32], shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(shape.to_vec(), values.to_vec()))
    }

    fn from_i32_slice(values: &[i32], shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(
            shape.to_vec(),
            values.iter().map(|value| *value as f32).collect(),
        )
        .with_dtype(eredu_core::checkpoint::TensorDtype::I32))
    }

    fn to_i32_vec(&self, _: &NumericContext) -> Result<eredu_core::HostTensorBuffer<i32>, Error> {
        self.data
            .iter()
            .map(|value| {
                if !value.is_finite() || value.fract() != 0.0 {
                    return Err(Error::backend("numeric tensor is not integral"));
                }
                i32::try_from(*value as i64)
                    .map_err(|_| Error::backend("numeric integer is outside i32"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|values| eredu_core::HostTensorBuffer::new(values, ()))
    }

    fn full_f32(value: f32, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(shape.to_vec(), vec![value; elements(shape)]))
    }

    fn full_i32(value: i32, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(
            Self::new(shape.to_vec(), vec![value as f32; elements(shape)])
                .with_dtype(eredu_core::checkpoint::TensorDtype::I32),
        )
    }
    fn full_u32(value: u32, shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(shape.to_vec(), vec![value as f32; elements(shape)])
            .with_dtype(eredu_core::checkpoint::TensorDtype::U32))
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

    fn tanh(&self, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(f32::tanh))
    }

    fn maximum_scalar(&self, rhs: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value.max(rhs)))
    }

    fn zeros_like(&self, _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::zeros(self.shape.clone()).with_dtype(self.dtype.clone()))
    }

    fn equal_i32(&self, value: i32, _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .map(|element| f32::from(*element == value as f32))
                .collect(),
        )
        .with_dtype(eredu_core::checkpoint::TensorDtype::Bool))
    }

    fn logical_or(&self, rhs: &Self, _: &NumericContext) -> Result<Self, Error> {
        Ok(self
            .zip(rhs, |left, right| f32::from(left != 0.0 || right != 0.0))?
            .with_dtype(eredu_core::checkpoint::TensorDtype::Bool))
    }

    fn masked_scatter(
        &self,
        mask: &Self,
        source: &Self,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        if mask.shape.len() >= self.shape.len() || self.shape[..mask.shape.len()] != mask.shape {
            return Err(Error::backend(
                "numeric masked scatter mask geometry mismatch",
            ));
        }
        let row_width = elements(&self.shape[mask.shape.len()..]);
        let selected = mask.data.iter().filter(|value| **value != 0.0).count();
        if source.data.len() != selected * row_width {
            return Err(Error::backend(
                "numeric masked scatter source geometry mismatch",
            ));
        }
        let mut output = self.clone();
        let mut source_row = 0;
        for (row, selected) in mask.data.iter().enumerate() {
            if *selected == 0.0 {
                continue;
            }
            let destination = row * row_width;
            let source_offset = source_row * row_width;
            output.data[destination..destination + row_width]
                .copy_from_slice(&source.data[source_offset..source_offset + row_width]);
            source_row += 1;
        }
        Ok(output)
    }

    fn clip(&self, minimum: &Self, maximum: &Self, _: &NumericContext) -> Result<Self, Error> {
        let minimum = *minimum
            .data
            .first()
            .ok_or_else(|| Error::backend("numeric clip minimum is empty"))?;
        let maximum = *maximum
            .data
            .first()
            .ok_or_else(|| Error::backend("numeric clip maximum is empty"))?;
        Ok(self.map(|value| value.clamp(minimum, maximum)))
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
        if selected != 0 || indexes.shape.is_empty() {
            return unsupported("take_axis geometry");
        }
        let row_width = self.data.len() / self.shape[0] as usize;
        let mut shape = indexes.shape.clone();
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
        let integer_mix = values.iter().all(|value| {
            matches!(
                value.dtype,
                eredu_core::checkpoint::TensorDtype::I32 | eredu_core::checkpoint::TensorDtype::U32
            )
        });
        for value in values {
            if (!integer_mix && value.dtype != first.dtype)
                || value.shape.len() != shape.len()
                || value
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(current, dimension)| current != selected && *dimension != shape[current])
            {
                return Err(Error::backend(format!(
                    "numeric concatenate shape mismatch on axis {selected}: first {:?} {:?}, current {:?} {:?}",
                    first.dtype, first.shape, value.dtype, value.shape
                )));
            }
            shape[selected] += value.shape[selected];
        }
        let mut output = Self::zeros(shape);
        output.dtype = if integer_mix {
            eredu_core::checkpoint::TensorDtype::I32
        } else {
            first.dtype.clone()
        };
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

    fn matmul(lhs: &Self, rhs: &Self, context: &NumericContext) -> Result<Self, Error> {
        if lhs.shape.len() < 2
            || rhs.shape.len() < 2
            || lhs.shape[lhs.shape.len() - 1] != rhs.shape[rhs.shape.len() - 2]
        {
            return Err(Error::backend(format!(
                "numeric matmul requires compatible matrices, got {:?} and {:?}",
                lhs.shape, rhs.shape
            )));
        }
        let rank = lhs.shape.len().max(rhs.shape.len());
        let prefix_rank = rank - 2;
        let mut prefix = Vec::with_capacity(prefix_rank);
        for selected in 0..prefix_rank {
            let left = selected
                .checked_sub(rank - lhs.shape.len())
                .map_or(1, |axis| lhs.shape[axis]);
            let right = selected
                .checked_sub(rank - rhs.shape.len())
                .map_or(1, |axis| rhs.shape[axis]);
            if left != right && left != 1 && right != 1 {
                return Err(Error::backend(format!(
                    "numeric matmul batch mismatch: {:?} versus {:?}",
                    lhs.shape, rhs.shape
                )));
            }
            prefix.push(left.max(right));
        }
        let rows = lhs.shape[lhs.shape.len() - 2];
        let inner_width = lhs.shape[lhs.shape.len() - 1];
        let columns = rhs.shape[rhs.shape.len() - 1];
        let mut left_shape = prefix.clone();
        left_shape.extend([rows, inner_width]);
        let mut right_shape = prefix.clone();
        right_shape.extend([inner_width, columns]);
        let lhs = lhs.broadcast_to(&left_shape, context)?;
        let rhs = rhs.broadcast_to(&right_shape, context)?;
        let mut output_shape = prefix.clone();
        output_shape.extend([rows, columns]);
        let mut output = Self::zeros(output_shape);
        for batch_index in 0..elements(&prefix) {
            let batch = unravel(batch_index, &prefix);
            for row in 0..rows as usize {
                for column in 0..columns as usize {
                    let value = (0..inner_width as usize)
                        .map(|inner| {
                            let mut left = batch.clone();
                            left.extend([row, inner]);
                            let mut right = batch.clone();
                            right.extend([inner, column]);
                            lhs.data[offset(&left, &left_shape)]
                                * rhs.data[offset(&right, &right_shape)]
                        })
                        .sum();
                    let mut coordinate = batch.clone();
                    coordinate.extend([row, column]);
                    let output_shape = output.shape.clone();
                    output.data[offset(&coordinate, &output_shape)] = value;
                }
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

    fn pad(
        value: &Self,
        widths: &[(i32, i32)],
        mode: PadMode,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        if mode != PadMode::Constant || widths.len() != value.shape.len() {
            return unsupported("non-constant or rank-changing pad");
        }
        if widths
            .iter()
            .any(|(before, after)| *before < 0 || *after < 0)
        {
            return Err(Error::backend("numeric pad widths must be non-negative"));
        }
        let shape = value
            .shape
            .iter()
            .zip(widths)
            .map(|(dimension, (before, after))| dimension + before + after)
            .collect::<Vec<_>>();
        let mut output = Self::zeros(shape);
        for input_index in 0..value.data.len() {
            let input_coordinate = unravel(input_index, &value.shape);
            let output_coordinate = input_coordinate
                .iter()
                .enumerate()
                .map(|(axis, coordinate)| coordinate + widths[axis].0 as usize)
                .collect::<Vec<_>>();
            let output_index = offset(&output_coordinate, &output.shape);
            output.data[output_index] = value.data[input_index];
        }
        Ok(output)
    }

    fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        if input.shape.len() != 3
            || weight.shape.len() != 3
            || stride != 1
            || padding != 0
            || dilation != 1
            || groups != input.shape[2]
            || weight.shape[0] != groups
            || weight.shape[2] != 1
        {
            return unsupported("general conv1d geometry");
        }
        let batch = input.shape[0] as usize;
        let input_tokens = input.shape[1] as usize;
        let channels = input.shape[2] as usize;
        let kernel = weight.shape[1] as usize;
        if kernel > input_tokens {
            return Err(Error::backend("numeric conv1d kernel exceeds input"));
        }
        let output_tokens = input_tokens - kernel + 1;
        let mut output = Self::zeros(vec![batch as i32, output_tokens as i32, channels as i32]);
        for batch_index in 0..batch {
            for token in 0..output_tokens {
                for channel in 0..channels {
                    output.data[(batch_index * output_tokens + token) * channels + channel] = (0
                        ..kernel)
                        .map(|kernel_index| {
                            input.data[((batch_index * input_tokens + token + kernel_index)
                                * channels)
                                + channel]
                                * weight.data[(channel * kernel) + kernel_index]
                        })
                        .sum();
                }
            }
        }
        Ok(output)
    }

    fn conv2d(
        input: &Self,
        weight: &Self,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
        _: &NumericContext,
    ) -> Result<Self, Error> {
        if input.shape.len() != 4 || weight.shape.len() != 4 {
            return unsupported("non-NHWC conv2d geometry");
        }
        let input_shape = input
            .shape
            .iter()
            .copied()
            .map(|dimension| usize::try_from(dimension).map_err(Error::backend))
            .collect::<Result<Vec<_>, _>>()?;
        let weight_shape = weight
            .shape
            .iter()
            .copied()
            .map(|dimension| usize::try_from(dimension).map_err(Error::backend))
            .collect::<Result<Vec<_>, _>>()?;
        let (values, shape) = eredu_nn::multimodal::reference_patch_convolution_2d(
            &input.data,
            input_shape.try_into().expect("validated conv2d input rank"),
            &weight.data,
            weight_shape
                .try_into()
                .expect("validated conv2d weight rank"),
            eredu_nn::multimodal::PatchConvolution2dSpec {
                stride,
                padding,
                dilation,
                groups,
            },
        )?;
        Ok(Self::new(
            shape
                .into_iter()
                .map(|dimension| i32::try_from(dimension).map_err(Error::backend))
                .collect::<Result<Vec<_>, _>>()?,
            values,
        ))
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
            // Evaluate the erf definition in F64. This positive-term series
            // avoids cancellation inside erf; the discarded Gaussian tail
            // outside |x| < 10 is below the fixture's F32 precision.
            let x = f64::from(value);
            if x.abs() >= 10.0 {
                return value.max(0.0);
            }
            let z = x.abs() * std::f64::consts::FRAC_1_SQRT_2;
            let mut term = z;
            let mut sum = term;
            for n in 1..256 {
                term *= 2.0 * z * z / f64::from(2 * n + 1);
                sum += term;
                if term <= f64::EPSILON * sum {
                    break;
                }
            }
            let erf = (std::f64::consts::FRAC_2_SQRT_PI * (-z * z).exp() * sum).copysign(x);
            (0.5 * x * (1.0 + erf)) as f32
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

    fn multi_axis_rotary_embeddings(
        position_ids: &Self,
        spec: &eredu_nn::multimodal::MultiAxisRotarySpec,
        _: &NumericContext,
    ) -> Result<(Self, Self), Error> {
        if position_ids.shape.len() < 2
            || position_ids.shape.last().copied()
                != Some(i32::try_from(spec.axes.len()).map_err(Error::backend)?)
        {
            return Err(Error::backend(
                "numeric multi-axis position geometry mismatch",
            ));
        }
        let rows = position_ids.shape[..position_ids.shape.len() - 1]
            .iter()
            .try_fold(1usize, |rows, dimension| {
                rows.checked_mul(usize::try_from(*dimension).map_err(Error::backend)?)
                    .ok_or_else(|| Error::backend("numeric multi-axis position count overflow"))
            })?;
        let positions = position_ids
            .data
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>();
        let dimensions = spec.dimensions()?;
        let (cosine, sine) =
            eredu_nn::multimodal::reference_multi_axis_rotary_embeddings(&positions, rows, spec)?;
        let mut shape = position_ids.shape[..position_ids.shape.len() - 1].to_vec();
        shape.push(dimensions);
        Ok((
            NumericTensor::new(shape.clone(), cosine),
            NumericTensor::new(shape, sine),
        ))
    }

    fn multi_axis_rotary_embeddings_prepared(
        position_ids: &Self,
        prepared: eredu_nn::multimodal::PreparedMultiAxisRotary<'_>,
        _: &NumericContext,
    ) -> Result<(Self, Self), Error> {
        let spec = prepared.spec();
        if position_ids.shape.len() < 2
            || position_ids.shape.last().copied() != Some(spec.axes.len() as i32)
        {
            return Err(Error::backend(
                "numeric multi-axis position geometry mismatch",
            ));
        }
        let rows = position_ids.shape[..position_ids.shape.len() - 1]
            .iter()
            .try_fold(1usize, |rows, dimension| {
                rows.checked_mul(usize::try_from(*dimension).map_err(Error::backend)?)
                    .ok_or_else(|| Error::backend("numeric multi-axis position count overflow"))
            })?;
        let positions = position_ids
            .data
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>();
        let (cosine, sine) = eredu_nn::multimodal::reference_multi_axis_rotary_embeddings_prepared(
            &positions, rows, prepared,
        )?;
        let mut shape = position_ids.shape[..position_ids.shape.len() - 1].to_vec();
        shape.push(spec.dimensions().map_err(Error::backend_retained_source)?);
        Ok((
            NumericTensor::new(shape.clone(), cosine),
            NumericTensor::new(shape, sine),
        ))
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

fn select_parameter(
    value: &NumericTensor,
    placement: &TensorPlacement,
) -> Result<NumericTensor, Error> {
    match placement {
        TensorPlacement::Replicated | TensorPlacement::Local => Ok(value.clone()),
        TensorPlacement::Shard { axis, index, parts } => {
            let width = usize::try_from(value.shape[*axis]).map_err(Error::backend)?;
            if *parts == 0 || !width.is_multiple_of(*parts) || *index >= *parts {
                return Err(Error::backend("invalid numeric equal shard"));
            }
            let shard = width / parts;
            Ok(value.axis_slice(*axis, index * shard, (index + 1) * shard))
        }
        TensorPlacement::Range { axis, start, end } => Ok(value.axis_slice(*axis, *start, *end)),
        TensorPlacement::Indices { axis, indices } => {
            let width = usize::try_from(value.shape[*axis]).map_err(Error::backend)?;
            if indices.iter().any(|index| *index >= width) {
                return Err(Error::backend("numeric parameter index is out of range"));
            }
            let mut shape = value.shape.clone();
            shape[*axis] = i32::try_from(indices.len()).map_err(Error::backend)?;
            let mut output = NumericTensor::zeros(shape);
            for output_index in 0..output.data.len() {
                let mut coordinate = unravel(output_index, &output.shape);
                coordinate[*axis] = indices[coordinate[*axis]];
                output.data[output_index] = value.data[offset(&coordinate, &value.shape)];
            }
            Ok(output)
        }
        TensorPlacement::Omit | TensorPlacement::Rank { .. } => Err(Error::backend(
            "numeric parameter layout does not materialize this tensor",
        )),
    }
}

fn local_parameter(
    spec: &ParameterSpec,
    shape: Vec<i32>,
    norm: bool,
    context: &NumericContext,
) -> Result<NumericTensor, Error> {
    let Some(layout) = context.tensor_layout(spec.id.as_str()) else {
        return Ok(parameter(spec, shape, norm));
    };
    let global_shape = layout
        .global_shape()
        .iter()
        .copied()
        .map(|dimension| i32::try_from(dimension).map_err(Error::backend))
        .collect::<Result<Vec<_>, _>>()?;
    if shape == global_shape {
        return Ok(parameter(spec, global_shape, norm));
    }
    let expected_local = layout
        .local_shape()
        .iter()
        .copied()
        .map(|dimension| i32::try_from(dimension).map_err(Error::backend))
        .collect::<Result<Vec<_>, _>>()?;
    if expected_local != shape {
        return Err(Error::backend(format!(
            "numeric local parameter {} requested shape {shape:?}, planned {expected_local:?}",
            spec.id.as_str()
        )));
    }
    let mut selected = parameter(spec, global_shape, norm);
    for placement in layout.additional_placements() {
        selected = select_parameter(&selected, placement)?;
    }
    let selected = select_parameter(&selected, layout.placement())?;
    if selected.shape != expected_local {
        return Err(Error::backend(format!(
            "numeric local parameter {} selected shape {:?}, planned {expected_local:?}",
            spec.id.as_str(),
            selected.shape
        )));
    }
    Ok(selected)
}

fn balanced_rank_range(units: usize, size: usize, rank: usize) -> std::ops::Range<usize> {
    assert!(size > 0 && rank < size);
    let base = units / size;
    let remainder = units % size;
    let start = rank * base + rank.min(remainder);
    let length = base + usize::from(rank < remainder);
    start..start + length
}

fn numeric_local_layout(
    groups: &[ParameterGroupSpec],
    size: usize,
    rank: usize,
) -> Result<LocalModelLayout, Error> {
    let mut layout = LocalModelLayout::default();
    for group in groups {
        let logical_range = group
            .partition_units()
            .map(|units| balanced_rank_range(units, size, rank));
        for member in group.members() {
            let mut local_shape = member.global_shape().to_vec();
            let chunk_range =
                |extent: usize, chunk: usize| -> Result<std::ops::Range<usize>, Error> {
                    let units = group.partition_units().ok_or_else(|| {
                        Error::backend("test chunked member has no logical units")
                    })?;
                    let logical = logical_range.as_ref().unwrap();
                    if chunk == 0 || extent.div_ceil(chunk) != units || logical.end > units {
                        return Err(Error::backend(
                            "test chunked member has incompatible geometry",
                        ));
                    }
                    let boundary = |unit: usize| {
                        if unit == units {
                            Ok(extent)
                        } else {
                            unit.checked_mul(chunk)
                                .ok_or_else(|| Error::backend("test chunk boundary overflows"))
                        }
                    };
                    Ok(boundary(logical.start)?..boundary(logical.end)?)
                };
            let (placement, member_logical_range) = match member.sharding() {
                MemberSharding::Replicated => (TensorPlacement::Replicated, None),
                MemberSharding::Equal { axis } => {
                    let width = member.global_shape()[*axis];
                    if !width.is_multiple_of(size) {
                        return Err(Error::backend(format!(
                            "numeric equal shard {} does not divide {width} by {size}",
                            member.target()
                        )));
                    }
                    local_shape[*axis] = width / size;
                    (
                        TensorPlacement::Shard {
                            axis: *axis,
                            index: rank,
                            parts: size,
                        },
                        Some(rank * (width / size)..(rank + 1) * (width / size)),
                    )
                }
                MemberSharding::Balanced { axis } => {
                    let range = balanced_rank_range(member.global_shape()[*axis], size, rank);
                    local_shape[*axis] = range.len();
                    (
                        TensorPlacement::Range {
                            axis: *axis,
                            start: range.start,
                            end: range.end,
                        },
                        Some(range),
                    )
                }
                MemberSharding::Partitioned { axis } => {
                    let units = group.partition_units().ok_or_else(|| {
                        Error::backend("numeric partitioned member has no logical units")
                    })?;
                    let range = logical_range.as_ref().unwrap();
                    let width = member.global_shape()[*axis];
                    if !width.is_multiple_of(units) {
                        return Err(Error::backend(format!(
                            "numeric partitioned member {} width {width} does not divide by {units}",
                            member.target()
                        )));
                    }
                    let width_per_unit = width / units;
                    let start = range.start * width_per_unit;
                    let end = range.end * width_per_unit;
                    local_shape[*axis] = end - start;
                    (
                        TensorPlacement::Range {
                            axis: *axis,
                            start,
                            end,
                        },
                        Some(range.clone()),
                    )
                }
                MemberSharding::PartitionedChunks { axis, chunk_size } => {
                    let range = chunk_range(member.global_shape()[*axis], *chunk_size)?;
                    local_shape[*axis] = range.len();
                    (
                        TensorPlacement::Range {
                            axis: *axis,
                            start: range.start,
                            end: range.end,
                        },
                        logical_range.clone(),
                    )
                }
                MemberSharding::PartitionedChunkSegments {
                    axis,
                    segments,
                    chunk_size,
                } => {
                    let mut indices = Vec::new();
                    for segment in segments {
                        let range = chunk_range(segment.len(), *chunk_size)?;
                        indices.extend(segment.start + range.start..segment.start + range.end);
                    }
                    local_shape[*axis] = indices.len();
                    (
                        TensorPlacement::Indices {
                            axis: *axis,
                            indices,
                        },
                        logical_range.clone(),
                    )
                }
                MemberSharding::PartitionedSegments { axis, segments } => {
                    let units = group.partition_units().ok_or_else(|| {
                        Error::backend("numeric segmented member has no logical units")
                    })?;
                    let range = logical_range.as_ref().unwrap();
                    let mut indices = Vec::new();
                    for segment in segments {
                        if !segment.len().is_multiple_of(units) {
                            return Err(Error::backend(format!(
                                "numeric segment in {} does not divide by {units}",
                                member.target()
                            )));
                        }
                        let width = segment.len() / units;
                        indices.extend(
                            segment.start + range.start * width..segment.start + range.end * width,
                        );
                    }
                    local_shape[*axis] = indices.len();
                    (
                        TensorPlacement::Indices {
                            axis: *axis,
                            indices,
                        },
                        Some(range.clone()),
                    )
                }
                MemberSharding::Segmented { axis, segments } => {
                    let mut indices = Vec::new();
                    for segment in segments {
                        let range = balanced_rank_range(segment.len(), size, rank);
                        indices.extend(segment.start + range.start..segment.start + range.end);
                    }
                    local_shape[*axis] = indices.len();
                    (
                        TensorPlacement::Indices {
                            axis: *axis,
                            indices,
                        },
                        None,
                    )
                }
            };
            if layout.contains(member.target()) {
                return Err(Error::backend(format!(
                    "numeric layout repeats {}",
                    member.target()
                )));
            }
            layout.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    local_shape,
                    placement,
                    group.partition_units(),
                    member_logical_range,
                    false,
                )
                .with_partition_chunk_size(match member.sharding() {
                    MemberSharding::PartitionedChunks { chunk_size, .. }
                    | MemberSharding::PartitionedChunkSegments { chunk_size, .. } => {
                        Some(*chunk_size)
                    }
                    _ => None,
                }),
            );
        }
    }
    Ok(layout)
}

fn visit<'a, V>(metadata: &'a ParameterMetadata, value: &'a NumericTensor, visitor: &mut V)
where
    V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
{
    visitor.parameter(metadata.as_view(), value);
}

fn visit_mut<'a, V>(metadata: &ParameterMetadata, value: &'a mut NumericTensor, visitor: &mut V)
where
    V: ParameterVisitorMut<'a, NumericTensor>,
{
    visitor.visit_mut(metadata.as_view(), value);
}

fn numeric_companion_metadata(
    companion: &ParameterSpec,
    primary: &ParameterSpec,
) -> ParameterMetadata {
    let mut metadata = ParameterMetadata::from_spec(companion, companion.trainable);
    metadata.linear_companion_of = Some(primary.id.clone());
    metadata
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
    execution_weight: Option<NumericTensor>,
    format_companions: Vec<(NumericTensor, ParameterMetadata)>,
}

impl Parameterized<NumericTensor> for NumericLinear {


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

        visit(&self.weight_metadata, &self.weight, visitor);
        if let Some((bias, metadata)) = &self.bias {
            visit(metadata, bias, visitor);
        }
        for (companion, metadata) in &self.format_companions {
            visit(metadata, companion, visitor);
        }



        if let Some(value) = &self.execution_weight {
            visitor.retained(value);
        }

__source_result
}

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.weight_metadata, &mut self.weight, visitor);
        if let Some((bias, metadata)) = &mut self.bias {
            visit_mut(metadata, bias, visitor);
        }
        for (companion, metadata) in &mut self.format_companions {
            visit_mut(metadata, companion, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.weight_metadata.trainable = trainable;
        if let Some((_, metadata)) = &mut self.bias {
            metadata.trainable = trainable;
        }
        for (_, metadata) in &mut self.format_companions {
            metadata.trainable = trainable;
        }
    }
}

impl LinearOperator<NumericTensor> for NumericLinear {
    fn forward(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        context.record_projection(self.weight_metadata.id.as_str().to_string(), input);
        linear(
            input,
            self.execution_weight.as_ref().unwrap_or(&self.weight),
            self.bias.as_ref().map(|(bias, _)| bias),
        )
    }
}

#[derive(Debug, Clone)]
struct NumericEmbedding {
    weight: NumericTensor,
    metadata: ParameterMetadata,
    vocabulary_range: Option<VocabularyParallelRange>,
    execution_weight: Option<NumericTensor>,
    format_companions: Vec<(NumericTensor, ParameterMetadata)>,
}

impl NumericEmbedding {
    fn execution_weight(&self) -> &NumericTensor {
        self.execution_weight.as_ref().unwrap_or(&self.weight)
    }
}

impl Parameterized<NumericTensor> for NumericEmbedding {


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

        visit(&self.metadata, &self.weight, visitor);
        for (companion, metadata) in &self.format_companions {
            visit(metadata, companion, visitor);
        }



        if let Some(value) = &self.execution_weight {
            visitor.retained(value);
        }

__source_result
}

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.metadata, &mut self.weight, visitor);
        for (companion, metadata) in &mut self.format_companions {
            visit_mut(metadata, companion, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
        for (_, metadata) in &mut self.format_companions {
            metadata.trainable = trainable;
        }
    }
}

impl EmbeddingOperator<NumericTensor> for NumericEmbedding {
    fn forward(
        &mut self,
        input: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let weight = self.execution_weight();
        let vocabulary = self
            .vocabulary_range
            .as_ref()
            .map_or(weight.shape[0] as usize, |range| range.global_vocabulary);
        let dimensions = weight.shape[1] as usize;
        let mut shape = input.shape.clone();
        shape.push(dimensions as i32);
        let mut output = NumericTensor::zeros(shape);
        for (token_index, token) in input.data.iter().enumerate() {
            let token = *token as usize;
            if token >= vocabulary || token as f32 != input.data[token_index] {
                return Err(Error::backend("numeric embedding token is invalid"));
            }
            let local = self.vocabulary_range.as_ref().map_or(Some(token), |range| {
                range
                    .local
                    .contains(&token)
                    .then(|| token - range.local.start)
            });
            if let Some(local) = local {
                output.data[token_index * dimensions..(token_index + 1) * dimensions]
                    .copy_from_slice(&weight.data[local * dimensions..(local + 1) * dimensions]);
            }
        }
        Ok(output)
    }

    fn as_linear(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        context.record_projection(self.metadata.id.as_str().to_string(), input);
        linear(input, self.execution_weight(), None)
    }

    fn lookup(
        &mut self,
        input: &NumericTensor,
        policy: EmbeddingLookupPolicy,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        policy.validate()?;
        let EmbeddingLookupPolicy::ZeroSentinel(sentinel) = policy else {
            return self.forward(input, context);
        };
        let weight = self.execution_weight();
        let vocabulary = self
            .vocabulary_range
            .as_ref()
            .map_or(weight.shape[0] as usize, |range| range.global_vocabulary);
        let dimensions = weight.shape[1] as usize;
        let mut shape = input.shape.clone();
        shape.push(dimensions as i32);
        let mut output = NumericTensor::zeros(shape);
        for (token_index, token) in input.data.iter().copied().enumerate() {
            if token == sentinel as f32 {
                continue;
            }
            let row = token as usize;
            if token < 0.0 || row >= vocabulary || row as f32 != token {
                return Err(Error::backend("numeric embedding token is invalid"));
            }
            let local = self.vocabulary_range.as_ref().map_or(Some(row), |range| {
                range.local.contains(&row).then(|| row - range.local.start)
            });
            if let Some(local) = local {
                output.data[token_index * dimensions..(token_index + 1) * dimensions]
                    .copy_from_slice(&weight.data[local * dimensions..(local + 1) * dimensions]);
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct NumericNorm {
    groups: usize,
    weight: NumericTensor,
    offset: f32,
    metadata: ParameterMetadata,
    epsilon: f32,
}

impl Parameterized<NumericTensor> for NumericNorm {


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

        visit(&self.metadata, &self.weight, visitor);

 __source_result
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
            let width = dimensions / self.groups;
            for group in 0..self.groups {
                let start = group * width;
                let rms = (input_row[start..start + width]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    / width as f32
                    + self.epsilon)
                    .sqrt();
                for index in start..start + width {
                    output_row[index] =
                        input_row[index] / rms * (self.weight.data[index] + self.offset);
                }
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct NumericRotary {
    algorithm: eredu_nn::RotaryAlgorithm,
    dimensions: i32,
    traditional: bool,
    base: f32,
}

impl Parameterized<NumericTensor> for NumericRotary {


    fn visit_parameter_sources<'a, V>(&'a self, _: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());


 __source_result
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
            RotaryPosition::Offset(position_offset) => {
                rotary::scaled_rotary(input, self, position_offset)
            }
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
    if (cosine.shape != [sequence as i32, half as i32]
        && cosine.shape != [sequence as i32, dimensions])
        || sine.shape != cosine.shape
    {
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
            let embedding_width = cosine.shape[1] as usize;
            let embedding_index = if embedding_width == half {
                frequency
            } else if traditional {
                2 * frequency
            } else {
                frequency
            };
            let cosine = cosine.data[position * embedding_width + embedding_index];
            let sine = sine.data[position * embedding_width + embedding_index];
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


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

        visit(&self.function.1, &self.function.0, visitor);
        visit(&self.base.1, &self.base.0, visitor);
        visit(&self.scale.1, &self.scale.0, visitor);

 __source_result
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


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

        visit(&self.function.1, &self.function.0, visitor);
        visit(&self.base.1, &self.base.0, visitor);
        visit(&self.scale.1, &self.scale.0, visitor);

 __source_result
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
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.forward_with_coefficients_observer(residual, context, None)
    }

    fn forward_with_coefficients_observer(
        &mut self,
        residual: &NumericTensor,
        _: &NumericContext,
        observer: Option<&mut dyn eredu_nn::TensorValueObserver<NumericTensor>>,
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
        let mut coefficients = NumericTensor::zeros(vec![
            residual.shape[0],
            residual.shape[1],
            self.streams as i32,
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
                coefficients.data[row * self.streams + stream] = coefficient;
            }
        }
        if let Some(observer) = observer {
            observer.observe(&coefficients)?;
        }
        for row in 0..rows {
            for stream in 0..self.streams {
                for dimension in 0..self.hidden_size {
                    output.data[row * self.hidden_size + dimension] += coefficients.data
                        [row * self.streams + stream]
                        * residual.data
                            [(row * self.streams + stream) * self.hidden_size + dimension];
                }
            }
        }
        Ok(output)
    }
}

#[derive(Default, Clone)]
struct NumericContext {
    bind_checkpoint_values: bool,
    cache_owned_attention: bool,
    sliding_attention_calls: Cell<usize>,
    local_layout: Option<Arc<LocalModelLayout>>,
    // Affine numerical execution stores expanded weights while admission and
    // transport retain the selected packed layout.
    expanded_weight_layout: Option<Arc<LocalModelLayout>>,
    mechanisms: Arc<Mutex<Vec<NumericMechanismTrace>>>,
    projections: Arc<Mutex<Vec<(String, Vec<i32>)>>>,
    // Concrete root/value observations only; not native completion or allocation proof.
    media_completions: Arc<Mutex<Vec<Vec<(Vec<i32>, Vec<f32>)>>>>,
    fail_media_completion: Arc<std::sync::atomic::AtomicBool>,
    partition: Option<NumericPartitionContext>,
}

impl NumericContext {
    fn with_local_layout(layout: LocalModelLayout) -> Self {
        Self {
            bind_checkpoint_values: false,
            cache_owned_attention: false,
            sliding_attention_calls: Cell::new(0),
            local_layout: Some(Arc::new(layout)),
            expanded_weight_layout: None,
            mechanisms: Arc::default(),
            projections: Arc::default(),
            media_completions: Arc::default(),
            fail_media_completion: Arc::default(),
            partition: None,
        }
    }

    fn with_partition(
        layout: LocalModelLayout,
        rank: usize,
        world: Arc<NumericPartitionWorld>,
    ) -> Self {
        Self {
            bind_checkpoint_values: false,
            cache_owned_attention: false,
            sliding_attention_calls: Cell::new(0),
            local_layout: Some(Arc::new(layout)),
            expanded_weight_layout: None,
            mechanisms: Arc::default(),
            projections: Arc::default(),
            media_completions: Arc::default(),
            fail_media_completion: Arc::default(),
            partition: Some(NumericPartitionContext::new(rank, world)),
        }
    }

    fn tensor_layout(&self, target: &str) -> Option<&LocalTensorLayout> {
        self.expanded_weight_layout
            .as_deref()
            .and_then(|layout| layout.tensor(target))
            .or_else(|| {
                self.local_layout
                    .as_deref()
                    .and_then(|layout| layout.tensor(target))
            })
    }

    fn record_bank_lookup(&self, key: impl Into<String>) {
        self.mechanisms
            .lock()
            .expect("numeric mechanism trace lock")
            .push(NumericMechanismTrace::BankLookup(key.into()));
    }

    fn mechanism_trace(&self) -> Vec<NumericMechanismTrace> {
        self.mechanisms
            .lock()
            .expect("numeric mechanism trace lock")
            .clone()
    }

    fn record_projection(&self, name: String, input: &NumericTensor) {
        self.projections
            .lock()
            .unwrap()
            .push((name, input.shape.clone()));
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum NumericMechanismTrace {
    BankLookup(String),
    AllToAll(u64),
    VariableAllToAll {
        group: u64,
        send: Vec<usize>,
        receive: Vec<usize>,
    },
    AllGatherEven {
        group: u64,
        axis: usize,
    },
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum NumericCollectiveKind {
    Sum,
    GatherVocabulary,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct NumericCollectiveTrace {
    sequence: usize,
    kind: NumericCollectiveKind,
    input_shape: Vec<i32>,
    output_shape: Vec<i32>,
}

#[derive(Default)]
struct NumericCollectiveSlot {
    kind: Option<NumericCollectiveKind>,
    values: Vec<Option<NumericTensor>>,
    output: Option<NumericTensor>,
    readers: usize,
}

struct NumericParallelGroup {
    size: usize,
    slots: Mutex<BTreeMap<usize, NumericCollectiveSlot>>,
    ready: Condvar,
}

impl NumericParallelGroup {
    fn new(size: usize) -> Arc<Self> {
        assert!(size > 0);
        Arc::new(Self {
            size,
            slots: Mutex::new(BTreeMap::new()),
            ready: Condvar::new(),
        })
    }
}

struct NumericParallelContext {
    rank: usize,
    global_rank: usize,
    group: Arc<NumericParallelGroup>,
    world: Option<Arc<NumericPartitionWorld>>,
    next_sequence: Cell<usize>,
    trace: RefCell<Vec<NumericCollectiveTrace>>,
}

impl NumericParallelContext {
    fn new(rank: usize, group: Arc<NumericParallelGroup>) -> Self {
        Self::new_inner(rank, rank, group, None)
    }

    fn new_partitioned(
        rank: usize,
        global_rank: usize,
        group: Arc<NumericParallelGroup>,
        world: Arc<NumericPartitionWorld>,
    ) -> Self {
        Self::new_inner(rank, global_rank, group, Some(world))
    }

    fn new_inner(
        rank: usize,
        global_rank: usize,
        group: Arc<NumericParallelGroup>,
        world: Option<Arc<NumericPartitionWorld>>,
    ) -> Self {
        assert!(rank < group.size);
        Self {
            rank,
            global_rank,
            group,
            world,
            next_sequence: Cell::new(0),
            trace: RefCell::new(Vec::new()),
        }
    }

    fn collective(
        &self,
        kind: NumericCollectiveKind,
        value: NumericTensor,
    ) -> Result<NumericTensor, Error> {
        if let Some(world) = &self.world {
            world
                .tensor_collective_calls
                .fetch_add(1, Ordering::Relaxed);
        }
        if self.world.as_ref().is_some_and(|world| {
            world.take_fault(self.global_rank, NumericPartitionFault::Collective)
        }) {
            return Err(Error::backend("injected numeric tensor collective failure"));
        }
        let sequence = self.next_sequence.get();
        self.next_sequence.set(sequence + 1);
        let input_shape = value.shape.clone();
        let mut slots = self
            .group
            .slots
            .lock()
            .map_err(|_| Error::backend("numeric collective lock poisoned"))?;
        {
            let slot = slots.entry(sequence).or_default();
            if slot.values.is_empty() {
                slot.values.resize(self.group.size, None);
                slot.kind = Some(kind);
            }
            if slot.kind != Some(kind) {
                return Err(Error::backend(format!(
                    "numeric collective {sequence} kind mismatch"
                )));
            }
            if slot.values[self.rank].replace(value).is_some() {
                return Err(Error::backend(format!(
                    "numeric collective {sequence} rank {} submitted twice",
                    self.rank
                )));
            }
            if slot.values.iter().all(Option::is_some) {
                let values = slot
                    .values
                    .iter()
                    .map(|value| value.as_ref().unwrap())
                    .collect::<Vec<_>>();
                let output = match kind {
                    NumericCollectiveKind::Sum => {
                        let shape = values[0].shape.clone();
                        if values.iter().any(|value| value.shape != shape) {
                            return Err(Error::backend(format!(
                                "numeric sum collective {sequence} shape mismatch"
                            )));
                        }
                        let mut output = NumericTensor::zeros(shape);
                        for value in values {
                            for (output, value) in output.data.iter_mut().zip(&value.data) {
                                *output += value;
                            }
                        }
                        output
                    }
                    NumericCollectiveKind::GatherVocabulary => {
                        let rank = values[0].shape.len();
                        if rank == 0 {
                            return Err(Error::backend(
                                "numeric vocabulary gather received a scalar",
                            ));
                        }
                        NumericTensor::concatenate(
                            &values.into_iter().cloned().collect::<Vec<_>>(),
                            i32::try_from(rank - 1).map_err(Error::backend)?,
                            &NumericContext::default(),
                        )?
                    }
                };
                slot.output = Some(output);
                self.group.ready.notify_all();
            }
        }
        while slots
            .get(&sequence)
            .and_then(|slot| slot.output.as_ref())
            .is_none()
        {
            slots = self
                .group
                .ready
                .wait(slots)
                .map_err(|_| Error::backend("numeric collective lock poisoned"))?;
        }
        let (output, remove) = {
            let slot = slots.get_mut(&sequence).unwrap();
            let output = slot.output.as_ref().unwrap().clone();
            slot.readers += 1;
            (output, slot.readers == self.group.size)
        };
        if remove {
            slots.remove(&sequence);
        }
        drop(slots);
        self.trace.borrow_mut().push(NumericCollectiveTrace {
            sequence,
            kind,
            input_shape,
            output_shape: output.shape.clone(),
        });
        Ok(output)
    }

    fn trace(&self) -> Vec<NumericCollectiveTrace> {
        self.trace.borrow().clone()
    }
}

#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum NumericOpaqueOperation {
    AllGatherEven,
    AllGatherUneven,
    AllReduceSum,
    Broadcast,
    FailureAgreement,
    PointToPoint,
    VariableAllToAll,
}

#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum NumericPartitionFault {
    LocalExecution,
    Transfer,
    Collective,
    PeerSubmission,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct NumericOpaqueTrace {
    rank: usize,
    operation: NumericOpaqueOperation,
    id: u64,
    sequence: usize,
}

#[derive(Default)]
struct NumericOpaqueSlot {
    members: Vec<usize>,
    tensors: BTreeMap<usize, Vec<NumericTensor>>,
    booleans: BTreeMap<usize, bool>,
    output: Option<Vec<NumericTensor>>,
    peer_outputs: BTreeMap<usize, Vec<NumericTensor>>,
    agreement: Option<bool>,
    readers: usize,
}

#[derive(Default)]
struct NumericPartitionWorldState {
    groups: BTreeMap<u64, Vec<usize>>,
    routes: BTreeMap<u64, (usize, usize)>,
    parallel_groups: BTreeMap<u64, Arc<NumericParallelGroup>>,
    slots: BTreeMap<(NumericOpaqueOperation, u64, usize), NumericOpaqueSlot>,
    trace: Vec<NumericOpaqueTrace>,
    materializations: BTreeMap<usize, usize>,
    fail_completion_ranks: BTreeSet<usize>,
    fault_ranks: BTreeSet<(usize, NumericPartitionFault)>,
    deadline_ranks: BTreeSet<(usize, NumericOpaqueOperation)>,
    completion_delay: std::time::Duration,
    completion_timeout: Option<std::time::Duration>,
    submission_calls: BTreeMap<NumericOpaqueOperation, usize>,
    completion_waits: BTreeMap<NumericOpaqueOperation, usize>,
    completion_successes: BTreeMap<NumericOpaqueOperation, usize>,
    commits: BTreeMap<usize, usize>,
    prompt_cache_identities: BTreeMap<usize, eredu_core::cache::PromptCacheModelIdentity>,
    cache_load_attempts: BTreeMap<usize, usize>,
    prompt_caches: BTreeMap<
        (std::path::PathBuf, usize),
        (
            DeviceState<NumericBackend, NumericHybridLayerState>,
            eredu_core::cache::PromptCacheManifest,
        ),
    >,
}

#[derive(Default)]
struct NumericPartitionWorld {
    state: Mutex<NumericPartitionWorldState>,
    ready: Condvar,
    tensor_collective_calls: AtomicUsize,
}

type NumericLifecycleCounts = (
    BTreeMap<NumericOpaqueOperation, usize>,
    BTreeMap<NumericOpaqueOperation, usize>,
    BTreeMap<NumericOpaqueOperation, usize>,
    BTreeMap<usize, usize>,
);

impl NumericPartitionWorld {
    fn tensor_collective_calls(&self) -> usize {
        self.tensor_collective_calls.load(Ordering::Relaxed)
    }

    fn realize_manifest(&self, manifest: &CommunicationManifest) -> Result<(), Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::backend("numeric partition world lock poisoned"))?;
        let timeout = manifest
            .completion_policy()
            .ok_or_else(|| Error::backend("numeric manifest has no bounded completion policy"))?
            .timeout();
        if state
            .completion_timeout
            .replace(timeout)
            .is_some_and(|selected| selected != timeout)
        {
            return Err(Error::backend(
                "numeric ranks selected different completion deadlines",
            ));
        }
        for group in manifest.groups() {
            let members = group.members().to_vec();
            if state
                .groups
                .insert(u64::from(group.id().value()), members.clone())
                .is_some_and(|existing| existing != members)
            {
                return Err(Error::backend("numeric opaque group membership drifted"));
            }
        }
        for route in manifest.routes() {
            let endpoints = (route.source(), route.destination());
            if state
                .routes
                .insert(route.id().value(), endpoints)
                .is_some_and(|existing| existing != endpoints)
            {
                return Err(Error::backend("numeric opaque route endpoints drifted"));
            }
        }
        Ok(())
    }

    fn fail_one_completion_per_agreement_group(&self) {
        let mut state = self.state.lock().expect("numeric partition world lock");
        let group_ids = state
            .trace
            .iter()
            .filter(|event| event.operation == NumericOpaqueOperation::FailureAgreement)
            .map(|event| event.id)
            .collect::<BTreeSet<_>>();
        let representatives = group_ids
            .into_iter()
            .map(|id| {
                state
                    .groups
                    .get(&id)
                    .and_then(|members| members.first())
                    .copied()
                    .expect("numeric agreement group has members")
            })
            .collect::<Vec<_>>();
        state.fail_completion_ranks.extend(representatives);
    }

    fn arm_fault_all(&self, world_size: usize, fault: NumericPartitionFault) {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .fault_ranks
            .extend((0..world_size).map(|rank| (rank, fault)));
    }

    fn arm_fault_one_per_agreement_group(&self, fault: NumericPartitionFault) {
        let mut state = self.state.lock().expect("numeric partition world lock");
        let group_ids = state
            .trace
            .iter()
            .filter(|event| event.operation == NumericOpaqueOperation::FailureAgreement)
            .map(|event| event.id)
            .collect::<BTreeSet<_>>();
        let representatives = group_ids
            .into_iter()
            .filter_map(|id| state.groups.get(&id).and_then(|members| members.first()))
            .copied()
            .collect::<Vec<_>>();
        state
            .fault_ranks
            .extend(representatives.into_iter().map(|rank| (rank, fault)));
    }

    fn take_fault(&self, rank: usize, fault: NumericPartitionFault) -> bool {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .fault_ranks
            .remove(&(rank, fault))
    }

    fn set_completion_delay(&self, delay: std::time::Duration) {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .completion_delay = delay;
    }

    fn arm_deadline_all(&self, world_size: usize, operation: NumericOpaqueOperation) {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .deadline_ranks
            .extend((0..world_size).map(|rank| (rank, operation)));
    }

    fn record_submission(&self, operation: NumericOpaqueOperation) {
        *self
            .state
            .lock()
            .expect("numeric partition world lock")
            .submission_calls
            .entry(operation)
            .or_default() += 1;
    }

    fn wait_completion(
        &self,
        rank: usize,
        operation: NumericOpaqueOperation,
        policy: eredu_core::BoundedCompletionWait,
    ) -> eredu_core::BoundedCompletionOutcome {
        let (delay, deadline) = {
            let mut state = self.state.lock().expect("numeric partition world lock");
            *state.completion_waits.entry(operation).or_default() += 1;
            (
                state.completion_delay,
                state.deadline_ranks.remove(&(rank, operation)),
            )
        };
        if deadline || delay > policy.timeout() {
            return eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
                cancellation: policy.cancellation(),
            };
        }
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        *self
            .state
            .lock()
            .expect("numeric partition world lock")
            .completion_successes
            .entry(operation)
            .or_default() += 1;
        eredu_core::BoundedCompletionOutcome::Completed
    }

    fn record_commit(&self, rank: usize) {
        *self
            .state
            .lock()
            .expect("numeric partition world lock")
            .commits
            .entry(rank)
            .or_default() += 1;
    }

    fn record_prompt_cache_identity(
        &self,
        rank: usize,
        identity: eredu_core::cache::PromptCacheModelIdentity,
    ) {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .prompt_cache_identities
            .insert(rank, identity);
    }

    fn prompt_cache_identities(
        &self,
    ) -> BTreeMap<usize, eredu_core::cache::PromptCacheModelIdentity> {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .prompt_cache_identities
            .clone()
    }

    fn lifecycle_counts(&self) -> NumericLifecycleCounts {
        let state = self.state.lock().expect("numeric partition world lock");
        (
            state.submission_calls.clone(),
            state.completion_waits.clone(),
            state.completion_successes.clone(),
            state.commits.clone(),
        )
    }

    fn parallel_group(&self, id: CollectiveGroupId) -> Result<Arc<NumericParallelGroup>, Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::backend("numeric partition world lock poisoned"))?;
        let size = state
            .groups
            .get(&u64::from(id.value()))
            .ok_or_else(|| Error::backend("numeric tensor group is not realized"))?
            .len();
        Ok(state
            .parallel_groups
            .entry(u64::from(id.value()))
            .or_insert_with(|| NumericParallelGroup::new(size))
            .clone())
    }

    fn take_completion_failure(&self, rank: usize) -> bool {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .fail_completion_ranks
            .remove(&rank)
    }

    fn trace(&self) -> Vec<NumericOpaqueTrace> {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .trace
            .clone()
    }

    fn groups(&self) -> BTreeMap<u64, Vec<usize>> {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .groups
            .clone()
    }

    fn routes(&self) -> BTreeMap<u64, (usize, usize)> {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .routes
            .clone()
    }

    fn record_materialization(&self, rank: usize, tasks: usize) {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .materializations
            .insert(rank, tasks);
    }

    fn materializations(&self) -> BTreeMap<usize, usize> {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .materializations
            .clone()
    }

    fn save_prompt_cache(
        &self,
        path: &std::path::Path,
        rank: usize,
        state: DeviceState<NumericBackend, NumericHybridLayerState>,
        manifest: eredu_core::cache::PromptCacheManifest,
    ) {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .prompt_caches
            .insert((path.to_path_buf(), rank), (state, manifest));
    }

    fn load_prompt_cache(
        &self,
        path: &std::path::Path,
        rank: usize,
    ) -> Option<(
        DeviceState<NumericBackend, NumericHybridLayerState>,
        eredu_core::cache::PromptCacheManifest,
    )> {
        let mut state = self.state.lock().expect("numeric partition world lock");
        *state.cache_load_attempts.entry(rank).or_default() += 1;
        state
            .prompt_caches
            .get(&(path.to_path_buf(), rank))
            .cloned()
    }

    fn cache_load_attempts(&self, rank: usize) -> usize {
        self.state
            .lock()
            .expect("numeric partition world lock")
            .cache_load_attempts
            .get(&rank)
            .copied()
            .unwrap_or_default()
    }
}

#[derive(Clone)]
struct NumericPartitionContext {
    rank: usize,
    world: Arc<NumericPartitionWorld>,
    sequences: Arc<Mutex<BTreeMap<(NumericOpaqueOperation, u64), usize>>>,
}

impl NumericPartitionContext {
    fn new(rank: usize, world: Arc<NumericPartitionWorld>) -> Self {
        Self {
            rank,
            world,
            sequences: Arc::default(),
        }
    }

    fn next_sequence(&self, operation: NumericOpaqueOperation, id: u64) -> usize {
        let mut sequences = self.sequences.lock().expect("numeric sequence lock");
        let sequence = sequences.entry((operation, id)).or_default();
        let current = *sequence;
        *sequence += 1;
        current
    }

    fn group_tensors(
        &self,
        operation: NumericOpaqueOperation,
        id: u64,
        values: Vec<NumericTensor>,
        root: Option<usize>,
    ) -> Result<Vec<NumericTensor>, Error> {
        let sequence = self.next_sequence(operation, id);
        let key = (operation, id, sequence);
        let mut state = self
            .world
            .state
            .lock()
            .map_err(|_| Error::backend("numeric partition world lock poisoned"))?;
        let members =
            state.groups.get(&id).cloned().ok_or_else(|| {
                Error::backend(format!("numeric opaque group {id} is not realized"))
            })?;
        let local_index = members
            .iter()
            .position(|member| *member == self.rank)
            .ok_or_else(|| Error::backend("numeric rank is outside its opaque group"))?;
        let slot = state.slots.entry(key).or_default();
        if slot.members.is_empty() {
            slot.members = members.clone();
        }
        if slot.members != members || slot.tensors.insert(self.rank, values).is_some() {
            return Err(Error::backend("numeric opaque tensor submission drifted"));
        }
        if slot.tensors.len() == members.len() {
            let owner = root.unwrap_or(0);
            let owner_rank = *members
                .get(owner)
                .ok_or_else(|| Error::backend("numeric broadcast root is outside its group"))?;
            slot.output = Some(
                slot.tensors
                    .get(&owner_rank)
                    .ok_or_else(|| Error::backend("numeric broadcast owner did not submit"))?
                    .clone(),
            );
            self.world.ready.notify_all();
        }
        while state
            .slots
            .get(&key)
            .and_then(|slot| slot.output.as_ref())
            .is_none()
        {
            state = self
                .world
                .ready
                .wait(state)
                .map_err(|_| Error::backend("numeric opaque tensor wait poisoned"))?;
        }
        let (output, remove) = {
            let slot = state.slots.get_mut(&key).unwrap();
            let output = slot.output.as_ref().unwrap().clone();
            slot.readers += 1;
            (output, slot.readers == members.len())
        };
        state.trace.push(NumericOpaqueTrace {
            rank: self.rank,
            operation,
            id,
            sequence,
        });
        if remove {
            state.slots.remove(&key);
        }
        let _ = local_index;
        Ok(output)
    }

    fn gather(
        &self,
        operation: NumericOpaqueOperation,
        id: u64,
        value: NumericTensor,
        axis: usize,
    ) -> Result<NumericTensor, Error> {
        let sequence = self.next_sequence(operation, id);
        let key = (operation, id, sequence);
        let mut state = self
            .world
            .state
            .lock()
            .map_err(|_| Error::backend("numeric partition world lock poisoned"))?;
        let members =
            state.groups.get(&id).cloned().ok_or_else(|| {
                Error::backend(format!("numeric opaque group {id} is not realized"))
            })?;
        let slot = state.slots.entry(key).or_default();
        if slot.members.is_empty() {
            slot.members = members.clone();
        }
        if slot.members != members || slot.tensors.insert(self.rank, vec![value]).is_some() {
            return Err(Error::backend("numeric even-gather submission drifted"));
        }
        if slot.tensors.len() == members.len() {
            let values = members
                .iter()
                .map(|member| slot.tensors[member][0].clone())
                .collect::<Vec<_>>();
            slot.output = Some(vec![NumericTensor::concatenate(
                &values,
                i32::try_from(axis).map_err(Error::backend)?,
                &NumericContext::default(),
            )?]);
            self.world.ready.notify_all();
        }
        while state
            .slots
            .get(&key)
            .and_then(|slot| slot.output.as_ref())
            .is_none()
        {
            let timeout = state
                .completion_timeout
                .ok_or_else(|| Error::backend("numeric collective has no selected deadline"))?;
            let (next, waited) = self
                .world
                .ready
                .wait_timeout(state, timeout)
                .map_err(|_| Error::backend("numeric even-gather wait poisoned"))?;
            state = next;
            if waited.timed_out()
                && state
                    .slots
                    .get(&key)
                    .and_then(|slot| slot.output.as_ref())
                    .is_none()
            {
                return Err(Error::backend("numeric even-gather deadline exceeded"));
            }
        }
        let (output, remove) = {
            let slot = state.slots.get_mut(&key).unwrap();
            let output = slot.output.as_ref().unwrap()[0].clone();
            slot.readers += 1;
            (output, slot.readers == members.len())
        };
        state.trace.push(NumericOpaqueTrace {
            rank: self.rank,
            operation,
            id,
            sequence,
        });
        if remove {
            state.slots.remove(&key);
        }
        Ok(output)
    }

    fn all_gather_even(&self, id: u64, value: NumericTensor) -> Result<NumericTensor, Error> {
        self.gather(NumericOpaqueOperation::AllGatherEven, id, value, 0)
    }

    fn all_reduce_sum(&self, id: u64, value: NumericTensor) -> Result<NumericTensor, Error> {
        let input_shape = value.shape.clone();
        let gathered = self.gather(NumericOpaqueOperation::AllReduceSum, id, value, 0)?;
        if gathered.shape.len() != input_shape.len()
            || gathered.shape[1..] != input_shape[1..]
            || gathered.shape[0] % input_shape[0] != 0
        {
            return Err(Error::backend("numeric sum-reduction geometry drifted"));
        }
        let rows = usize::try_from(input_shape[0]).unwrap();
        let row_elements = elements(&input_shape[1..]);
        let partitions = usize::try_from(gathered.shape[0] / input_shape[0]).unwrap();
        let mut output = NumericTensor::zeros(input_shape).with_dtype(gathered.dtype.clone());
        for partition in 0..partitions {
            for index in 0..rows * row_elements {
                output.data[index] += gathered.data[partition * rows * row_elements + index];
            }
        }
        Ok(output)
    }

    fn variable_all_to_all(
        &self,
        id: u64,
        value: NumericTensor,
        counts: &eredu_runtime::CommunicationPeerCounts,
    ) -> Result<NumericTensor, Error> {
        let operation = NumericOpaqueOperation::VariableAllToAll;
        let sequence = self.next_sequence(operation, id);
        let key = (operation, id, sequence);
        let mut state = self
            .world
            .state
            .lock()
            .map_err(|_| Error::backend("numeric partition world lock poisoned"))?;
        let members =
            state.groups.get(&id).cloned().ok_or_else(|| {
                Error::backend(format!("numeric opaque group {id} is not realized"))
            })?;
        let encode_counts = |values: &[usize]| {
            values
                .iter()
                .map(|value| i32::try_from(*value).map_err(Error::backend))
                .collect::<Result<Vec<_>, _>>()
                .map(|values| {
                    NumericTensor::new(
                        vec![i32::try_from(values.len()).unwrap()],
                        values.into_iter().map(|value| value as f32).collect(),
                    )
                    .with_dtype(eredu_core::checkpoint::TensorDtype::I32)
                })
        };
        let submitted = vec![
            value,
            encode_counts(counts.send())?,
            encode_counts(counts.receive())?,
        ];
        let slot = state.slots.entry(key).or_default();
        if slot.members.is_empty() {
            slot.members = members.clone();
        }
        if slot.members != members || slot.tensors.insert(self.rank, submitted).is_some() {
            return Err(Error::backend(
                "numeric variable-all-to-all submission drifted",
            ));
        }
        if slot.tensors.len() == members.len() {
            let submissions = slot.tensors.clone();
            for (destination_index, destination) in members.iter().copied().enumerate() {
                let mut pieces = Vec::with_capacity(members.len());
                for (source_index, source) in members.iter().copied().enumerate() {
                    let submitted = &submissions[&source];
                    let send = submitted[1]
                        .to_i32_vec(&NumericContext::default())?
                        .into_iter()
                        .map(|value| usize::try_from(value).map_err(Error::backend))
                        .collect::<Result<Vec<_>, _>>()?;
                    let receive = submissions[&destination][2]
                        .to_i32_vec(&NumericContext::default())?
                        .into_iter()
                        .map(|value| usize::try_from(value).map_err(Error::backend))
                        .collect::<Result<Vec<_>, _>>()?;
                    if send.len() != members.len()
                        || receive.len() != members.len()
                        || receive[source_index] != send[destination_index]
                    {
                        return Err(Error::backend(
                            "numeric variable-all-to-all peer counts disagree",
                        ));
                    }
                    let start = send[..destination_index].iter().sum::<usize>();
                    let end = start + send[destination_index];
                    pieces.push(submitted[0].axis_slice(0, start, end));
                }
                let output = NumericTensor::concatenate(&pieces, 0, &NumericContext::default())?;
                state
                    .slots
                    .get_mut(&key)
                    .unwrap()
                    .peer_outputs
                    .insert(destination, vec![output]);
            }
            self.world.ready.notify_all();
        }
        while state
            .slots
            .get(&key)
            .and_then(|slot| slot.peer_outputs.get(&self.rank))
            .is_none()
        {
            let timeout = state
                .completion_timeout
                .ok_or_else(|| Error::backend("numeric collective has no selected deadline"))?;
            let (next, waited) = self
                .world
                .ready
                .wait_timeout(state, timeout)
                .map_err(|_| Error::backend("numeric variable-all-to-all wait poisoned"))?;
            state = next;
            if waited.timed_out()
                && state
                    .slots
                    .get(&key)
                    .and_then(|slot| slot.peer_outputs.get(&self.rank))
                    .is_none()
            {
                return Err(Error::backend(
                    "numeric variable-all-to-all deadline exceeded",
                ));
            }
        }
        let (output, remove) = {
            let slot = state.slots.get_mut(&key).unwrap();
            let output = slot.peer_outputs[&self.rank][0].clone();
            slot.readers += 1;
            (output, slot.readers == members.len())
        };
        state.trace.push(NumericOpaqueTrace {
            rank: self.rank,
            operation,
            id,
            sequence,
        });
        if remove {
            state.slots.remove(&key);
        }
        Ok(output)
    }

    fn agree(&self, id: u64, success: bool) -> Result<bool, Error> {
        let operation = NumericOpaqueOperation::FailureAgreement;
        let sequence = self.next_sequence(operation, id);
        let key = (operation, id, sequence);
        let mut state = self
            .world
            .state
            .lock()
            .map_err(|_| Error::backend("numeric partition world lock poisoned"))?;
        let members =
            state.groups.get(&id).cloned().ok_or_else(|| {
                Error::backend(format!("numeric opaque group {id} is not realized"))
            })?;
        let slot = state.slots.entry(key).or_default();
        if slot.members.is_empty() {
            slot.members = members.clone();
        }
        if slot.members != members || slot.booleans.insert(self.rank, success).is_some() {
            return Err(Error::backend(
                "numeric failure-agreement submission drifted",
            ));
        }
        if slot.booleans.len() == members.len() {
            slot.agreement = Some(slot.booleans.values().all(|success| *success));
            self.world.ready.notify_all();
        }
        while state
            .slots
            .get(&key)
            .and_then(|slot| slot.agreement)
            .is_none()
        {
            let timeout = state
                .completion_timeout
                .ok_or_else(|| Error::backend("numeric agreement has no selected deadline"))?;
            let (next, waited) = self
                .world
                .ready
                .wait_timeout(state, timeout)
                .map_err(|_| Error::backend("numeric failure-agreement wait poisoned"))?;
            state = next;
            if waited.timed_out()
                && state
                    .slots
                    .get(&key)
                    .and_then(|slot| slot.agreement)
                    .is_none()
            {
                return Err(Error::backend(
                    "numeric failure-agreement deadline exceeded",
                ));
            }
        }
        let (agreement, remove) = {
            let slot = state.slots.get_mut(&key).unwrap();
            let agreement = slot.agreement.unwrap();
            slot.readers += 1;
            (agreement, slot.readers == members.len())
        };
        state.trace.push(NumericOpaqueTrace {
            rank: self.rank,
            operation,
            id,
            sequence,
        });
        if remove {
            state.slots.remove(&key);
        }
        Ok(agreement)
    }

    fn route(&self, id: u64, values: Vec<NumericTensor>) -> Result<Vec<NumericTensor>, Error> {
        let operation = NumericOpaqueOperation::PointToPoint;
        let sequence = self.next_sequence(operation, id);
        let key = (operation, id, sequence);
        let mut state = self
            .world
            .state
            .lock()
            .map_err(|_| Error::backend("numeric partition world lock poisoned"))?;
        let (source, destination) = *state
            .routes
            .get(&id)
            .ok_or_else(|| Error::backend(format!("numeric opaque route {id} is not realized")))?;
        if self.rank != source && self.rank != destination {
            return Err(Error::backend("numeric rank is outside its opaque route"));
        }
        let slot = state.slots.entry(key).or_default();
        if slot.members.is_empty() {
            slot.members = vec![source, destination];
        }
        if slot.tensors.insert(self.rank, values).is_some() {
            return Err(Error::backend("numeric route endpoint submitted twice"));
        }
        if slot.tensors.len() == 2 {
            slot.output = Some(slot.tensors.get(&source).unwrap().clone());
            self.world.ready.notify_all();
        }
        while state
            .slots
            .get(&key)
            .and_then(|slot| slot.output.as_ref())
            .is_none()
        {
            state = self
                .world
                .ready
                .wait(state)
                .map_err(|_| Error::backend("numeric route wait poisoned"))?;
        }
        let (output, remove) = {
            let slot = state.slots.get_mut(&key).unwrap();
            let output = slot.output.as_ref().unwrap().clone();
            slot.readers += 1;
            (output, slot.readers == 2)
        };
        state.trace.push(NumericOpaqueTrace {
            rank: self.rank,
            operation,
            id,
            sequence,
        });
        if remove {
            state.slots.remove(&key);
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, Copy)]
struct NumericBackend;

#[derive(Clone)]
struct NumericCompletion {
    partition: Option<NumericPartitionContext>,
    operation: NumericOpaqueOperation,
}

impl std::fmt::Debug for NumericCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NumericCompletion")
            .field(
                "rank",
                &self.partition.as_ref().map(|partition| partition.rank),
            )
            .field("operation", &self.operation)
            .finish()
    }
}

impl NumericCompletion {
    fn immediate() -> Self {
        Self {
            partition: None,
            operation: NumericOpaqueOperation::FailureAgreement,
        }
    }

    fn partition(partition: &NumericPartitionContext, operation: NumericOpaqueOperation) -> Self {
        Self {
            partition: Some(partition.clone()),
            operation,
        }
    }
}

impl Completion for NumericCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl eredu_core::BoundedCompletion for NumericCompletion {
    fn wait_bounded(
        self,
        policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        Ok(self.partition.map_or(
            eredu_core::BoundedCompletionOutcome::Completed,
            |partition| {
                partition
                    .world
                    .wait_completion(partition.rank, self.operation, policy)
            },
        ))
    }
}

impl SubmissionBackend for NumericBackend {
    type Executor = NumericContext;
    type OwnedExecutor = NumericContext;
    type Completion = NumericCompletion;

    fn fork_executors(
        executor: &Self::Executor,
        count: usize,
    ) -> Result<Vec<Self::OwnedExecutor>, Error> {
        Ok(vec![executor.clone(); count])
    }

    fn submit<'a, I>(_: &Self::Executor, _: I) -> Result<Self::Completion, Error>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        Ok(NumericCompletion::immediate())
    }

    fn order_after(_: &Self::Completion, _: &Self::Executor) -> Result<(), Error> {
        Ok(())
    }

    fn retain_until_complete<T: Send + 'static>(
        _: &Self::Executor,
        _: &Self::Completion,
        _: T,
    ) -> Result<(), Error> {
        Ok(())
    }
}

impl CollectiveBackend for NumericBackend {
    type Group = u64;
    type CollectiveError = std::convert::Infallible;

    fn all_reduce(
        value: Self::Tensor,
        _: &Self::Group,
        _: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError> {
        Ok(value)
    }

    fn all_gather(
        value: Self::Tensor,
        _: &Self::Group,
        _: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError> {
        Ok(value)
    }

    fn all_to_all(
        value: Self::Tensor,
        group: &Self::Group,
        context: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError> {
        context
            .mechanisms
            .lock()
            .expect("numeric mechanism trace lock")
            .push(NumericMechanismTrace::AllToAll(*group));
        Ok(value)
    }
}

impl CommunicationBackend for NumericBackend {
    type CommunicationGroup = u64;
    type CommunicationRoute = u64;
    type CommunicationCompletion = NumericCompletion;
    type CommunicationError = Error;

    fn submit_local_dependencies<'a, I>(
        values: I,
        context: &Self::Executor,
    ) -> Result<eredu_core::Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        let _ = values.into_iter().count();
        let completion =
            context
                .partition
                .as_ref()
                .map_or_else(NumericCompletion::immediate, |partition| {
                    partition
                        .world
                        .record_submission(NumericOpaqueOperation::PointToPoint);
                    NumericCompletion::partition(partition, NumericOpaqueOperation::PointToPoint)
                });
        Ok(eredu_core::Submission {
            output: (),
            completion,
        })
    }
}

impl eredu_runtime::BroadcastBackend for NumericBackend {
    fn broadcast(
        value: Self::Tensor,
        root: usize,
        group: &Self::CommunicationGroup,
        context: &Self::Executor,
    ) -> Result<eredu_core::Submission<Self::Tensor, NumericCompletion>, Error> {
        let partition = context
            .partition
            .as_ref()
            .ok_or_else(|| Error::backend("numeric broadcast has no partition context"))?;
        partition
            .world
            .record_submission(NumericOpaqueOperation::Broadcast);
        let mut output = partition.group_tensors(
            NumericOpaqueOperation::Broadcast,
            *group,
            vec![value],
            Some(root),
        )?;
        Ok(eredu_core::Submission {
            output: output.remove(0),
            completion: NumericCompletion::partition(partition, NumericOpaqueOperation::Broadcast),
        })
    }
}

impl eredu_runtime::FailureAgreementBackend for NumericBackend {
    type FailureAgreementOutput = bool;

    fn agree_success(
        local_success: bool,
        group: &Self::CommunicationGroup,
        context: &Self::Executor,
    ) -> Result<eredu_core::Submission<bool, NumericCompletion>, Error> {
        let partition = context
            .partition
            .as_ref()
            .ok_or_else(|| Error::backend("numeric agreement has no partition context"))?;
        partition
            .world
            .record_submission(NumericOpaqueOperation::FailureAgreement);
        Ok(eredu_core::Submission {
            output: partition.agree(*group, local_success)?,
            completion: NumericCompletion::partition(
                partition,
                NumericOpaqueOperation::FailureAgreement,
            ),
        })
    }

    fn resolve_failure_agreement(output: bool) -> Result<bool, Error> {
        Ok(output)
    }
}

impl eredu_runtime::PointToPointBackend for NumericBackend {
    fn send_receive(
        values: Vec<eredu_runtime::RoleExactBoundaryValue<Self::Tensor>>,
        route: &Self::CommunicationRoute,
        context: &Self::Executor,
    ) -> Result<eredu_core::Submission<Vec<Self::Tensor>, NumericCompletion>, Error> {
        let partition = context
            .partition
            .as_ref()
            .ok_or_else(|| Error::backend("numeric route has no partition context"))?;
        partition
            .world
            .record_submission(NumericOpaqueOperation::PointToPoint);
        if partition
            .world
            .take_fault(partition.rank, NumericPartitionFault::PeerSubmission)
        {
            return Err(Error::backend("injected numeric peer submission failure"));
        }
        let values = values
            .into_iter()
            .map(eredu_runtime::RoleExactBoundaryValue::into_parts)
            .map(|(_, tensor)| tensor)
            .collect();
        Ok(eredu_core::Submission {
            output: partition.route(*route, values)?,
            completion: NumericCompletion::partition(
                partition,
                NumericOpaqueOperation::PointToPoint,
            ),
        })
    }
}

impl VariableAllToAllBackend for NumericBackend {
    fn variable_all_to_all(
        value: Self::Tensor,
        counts: &eredu_runtime::CommunicationPeerCounts,
        axis: usize,
        group: &Self::CommunicationGroup,
        context: &Self::Executor,
    ) -> Result<
        eredu_core::Submission<Self::Tensor, Self::CommunicationCompletion>,
        Self::CommunicationError,
    > {
        assert_eq!(axis, 0);
        context
            .mechanisms
            .lock()
            .expect("numeric mechanism trace lock")
            .push(NumericMechanismTrace::VariableAllToAll {
                group: *group,
                send: counts.send().to_vec(),
                receive: counts.receive().to_vec(),
            });
        let Some(partition) = context.partition.as_ref() else {
            let receive_rows = counts.receive().iter().sum::<usize>();
            let source_rows = usize::try_from(value.shape[0]).unwrap();
            let row_elements = value.data.len().checked_div(source_rows.max(1)).unwrap();
            let mut shape = value.shape.clone();
            shape[0] = i32::try_from(receive_rows).unwrap();
            let dtype = value.dtype.clone();
            let mut data = value.data;
            data.resize(receive_rows * row_elements, 0.0);
            return Ok(eredu_core::Submission {
                output: NumericTensor::new(shape, data).with_dtype(dtype),
                completion: NumericCompletion::immediate(),
            });
        };
        partition
            .world
            .record_submission(NumericOpaqueOperation::VariableAllToAll);
        let output = partition.variable_all_to_all(*group, value, counts)?;
        Ok(eredu_core::Submission {
            output,
            completion: NumericCompletion::partition(
                partition,
                NumericOpaqueOperation::VariableAllToAll,
            ),
        })
    }
}

impl EvenGatherBackend for NumericBackend {
    fn all_gather_even(
        value: Self::Tensor,
        axis: usize,
        group: &Self::CommunicationGroup,
        context: &Self::Executor,
    ) -> Result<
        eredu_core::Submission<Self::Tensor, Self::CommunicationCompletion>,
        Self::CommunicationError,
    > {
        assert_eq!(axis, 0);
        context
            .mechanisms
            .lock()
            .expect("numeric mechanism trace lock")
            .push(NumericMechanismTrace::AllGatherEven {
                group: *group,
                axis,
            });
        let Some(partition) = context.partition.as_ref() else {
            return Ok(eredu_core::Submission {
                output: NumericTensor::concatenate(&[value.clone(), value], 0, context).unwrap(),
                completion: NumericCompletion::immediate(),
            });
        };
        partition
            .world
            .record_submission(NumericOpaqueOperation::AllGatherEven);
        let output = partition.all_gather_even(*group, value)?;
        Ok(eredu_core::Submission {
            output,
            completion: NumericCompletion::partition(
                partition,
                NumericOpaqueOperation::AllGatherEven,
            ),
        })
    }
}

impl eredu_runtime::UnevenGatherBackend for NumericBackend {
    fn all_gather_uneven(
        value: Self::Tensor,
        counts: &[usize],
        axis: usize,
        group: &Self::CommunicationGroup,
        context: &Self::Executor,
    ) -> Result<
        eredu_core::Submission<Self::Tensor, Self::CommunicationCompletion>,
        Self::CommunicationError,
    > {
        let Some(partition) = context.partition.as_ref() else {
            return Ok(eredu_core::Submission {
                output: value,
                completion: NumericCompletion::immediate(),
            });
        };
        let output =
            partition.gather(NumericOpaqueOperation::AllGatherUneven, *group, value, axis)?;
        if output
            .shape
            .get(axis)
            .and_then(|dimension| usize::try_from(*dimension).ok())
            != Some(counts.iter().sum())
        {
            return Err(Error::backend(
                "numeric uneven gather count geometry drifted",
            ));
        }
        Ok(eredu_core::Submission {
            output,
            completion: NumericCompletion::partition(
                partition,
                NumericOpaqueOperation::AllGatherUneven,
            ),
        })
    }
}

impl eredu_runtime::SumReductionBackend for NumericBackend {
    fn all_reduce_sum(
        value: Self::Tensor,
        group: &Self::CommunicationGroup,
        context: &Self::Executor,
    ) -> Result<
        eredu_core::Submission<Self::Tensor, Self::CommunicationCompletion>,
        Self::CommunicationError,
    > {
        let Some(partition) = context.partition.as_ref() else {
            return Ok(eredu_core::Submission {
                output: value,
                completion: NumericCompletion::immediate(),
            });
        };
        let output = partition.all_reduce_sum(*group, value)?;
        Ok(eredu_core::Submission {
            output,
            completion: NumericCompletion::partition(
                partition,
                NumericOpaqueOperation::AllReduceSum,
            ),
        })
    }
}

#[derive(Debug, Clone)]
struct NumericCommunicationMetadata(Option<eredu_core::checkpoint::TensorDtype>);

impl CommunicationTensorMetadata<NumericBackend> for NumericCommunicationMetadata {
    fn dtype(&self, tensor: &NumericTensor) -> eredu_core::checkpoint::TensorDtype {
        self.0.clone().unwrap_or_else(|| tensor.dtype.clone())
    }

    fn shape(&self, tensor: &NumericTensor) -> Vec<usize> {
        tensor
            .shape()
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect()
    }
}

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
    type ParameterPreparation<'a> = ();
    const OPERATOR_CAPABILITIES: eredu_nn::NeuralOperatorCapabilities =
        eredu_nn::NeuralOperatorCapabilities::ALL;

    type Tensor = NumericTensor;
    type Linear = NumericLinear;
    type Embedding = NumericEmbedding;
    type Normalization = NumericNorm;
    type Rotary = NumericRotary;
    type ParallelContext = NumericParallelContext;

    fn linear(spec: LinearSpec, context: &NumericContext) -> Result<Self::Linear, Error> {
        let logical_weight =
            local_parameter(&spec.weight, vec![spec.output, spec.input], false, context)?;
        let (weight, execution_weight, format_companions) = match spec.format.encoding() {
            eredu_checkpoint::LinearFormat::MxFp4 => {
                let mut companions = Vec::new();
                if let Some(parameter) = spec.format.scale() {
                    companions.push((
                        local_parameter(
                            parameter,
                            vec![spec.output, spec.input / 32],
                            false,
                            context,
                        )?,
                        numeric_companion_metadata(parameter, &spec.weight),
                    ));
                }
                (logical_weight, None, companions)
            }
            eredu_checkpoint::LinearFormat::Affine(format) => {
                let groups = spec
                    .input
                    .checked_div(format.group_size)
                    .ok_or_else(|| Error::backend("numeric affine companion shape overflowed"))?;
                let mut companions = Vec::new();
                for parameter in [spec.format.scale(), spec.format.affine_bias()]
                    .into_iter()
                    .flatten()
                {
                    companions.push((
                        local_parameter(parameter, vec![spec.output, groups], false, context)?,
                        numeric_companion_metadata(parameter, &spec.weight),
                    ));
                }
                (logical_weight, None, companions)
            }
            _ => (logical_weight, None, Vec::new()),
        };
        let weight_metadata = ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable);
        let bias = spec
            .bias
            .map(|bias| -> Result<_, Error> {
                let value = local_parameter(&bias, vec![spec.output], false, context)?;
                let metadata = ParameterMetadata::from_spec(&bias, bias.trainable);
                Ok((value, metadata))
            })
            .transpose()?;
        Ok(NumericLinear {
            weight,
            weight_metadata,
            bias,
            execution_weight,
            format_companions,
        })
    }

    fn embedding(spec: EmbeddingSpec, context: &NumericContext) -> Result<Self::Embedding, Error> {
        let logical_weight = local_parameter(
            &spec.weight,
            vec![spec.vocabulary, spec.dimensions],
            false,
            context,
        )?;
        let (weight, execution_weight, format_companions) = match spec.format.encoding() {
            eredu_checkpoint::LinearFormat::Affine(format) => {
                let groups = spec
                    .dimensions
                    .checked_div(format.group_size)
                    .ok_or_else(|| {
                        Error::backend("numeric affine embedding companion shape overflowed")
                    })?;
                let mut companions = Vec::new();
                for parameter in [spec.format.scale(), spec.format.affine_bias()]
                    .into_iter()
                    .flatten()
                {
                    companions.push((
                        local_parameter(parameter, vec![spec.vocabulary, groups], false, context)?,
                        numeric_companion_metadata(parameter, &spec.weight),
                    ));
                }
                (logical_weight, None, companions)
            }
            _ => (logical_weight, None, Vec::new()),
        };
        Ok(NumericEmbedding {
            weight,
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            vocabulary_range: None,
            execution_weight,
            format_companions,
        })
    }

    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &NumericContext,
    ) -> Result<Self::Normalization, Error> {
        spec.validate()?;
        let dimensions = spec.dimensions;
        let epsilon = spec.epsilon;
        let (weight, metadata, offset) = match spec.scale {
            NormalizationScale::Learned(weight) => {
                let value = local_parameter(&weight, vec![dimensions], true, context)?;
                let metadata = ParameterMetadata::from_spec(&weight, weight.trainable);
                (value, metadata, 0.0)
            }
            NormalizationScale::LearnedOffset { weight, offset } => {
                let value = local_parameter(&weight, vec![dimensions], false, context)?;
                let metadata = ParameterMetadata::from_spec(&weight, weight.trainable);
                (value, metadata, offset)
            }
            NormalizationScale::Unit => {
                let parameter =
                    ParameterSpec::trainable("numeric.unit_norm.weight").map_err(Error::backend)?;
                (
                    NumericTensor::new(vec![dimensions], vec![1.0; dimensions as usize]),
                    ParameterMetadata::from_spec(&parameter, false),
                    0.0,
                )
            }
        };
        Ok(NumericNorm {
            groups: spec.groups.unwrap_or(1) as usize,
            weight,
            offset,
            metadata,
            epsilon,
        })
    }

    fn rotary(spec: RotarySpec, _: &NumericContext) -> Result<Self::Rotary, Error> {
        Ok(NumericRotary {
            algorithm: spec.algorithm,
            dimensions: spec.dimensions,
            traditional: spec.traditional,
            base: spec.base,
        })
    }

    fn silu(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(|value| value / (1.0 + (-value).exp())))
    }

    fn gelu_approximate(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(|value| {
            0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
        }))
    }

    fn sigmoid(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(|value| 1.0 / (1.0 + (-value).exp())))
    }

    fn softplus(input: Self::Tensor, beta: f32, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input
            .map(|value| ((beta * value).max(0.0) + (-(beta * value).abs()).exp().ln_1p()) / beta))
    }

    fn exp(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(f32::exp))
    }

    fn l2_normalize(
        input: &Self::Tensor,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let geometry =
            eredu_nn::operation_geometry::NormalizationGeometry::new(&input.shape, epsilon)?;
        let width = usize::try_from(geometry.width()).map_err(Error::backend)?;
        let mut output = input.clone();
        for (source, target) in input
            .data
            .chunks_exact(width)
            .zip(output.data.chunks_exact_mut(width))
        {
            let norm = (source.iter().map(|value| value * value).sum::<f32>() + epsilon).sqrt();
            for (target, source) in target.iter_mut().zip(source) {
                *target = *source / norm;
            }
        }
        Ok(output)
    }

    fn gated_group_rms_norm(
        input: &NumericTensor,
        gate: &NumericTensor,
        weight: &NumericTensor,
        groups: i32,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            &input.shape,
            &gate.shape,
            groups,
            epsilon,
        )?;
        if weight.shape != [geometry.width()] {
            return Err(Error::backend("numeric gated RMS geometry mismatch"));
        }
        let width = usize::try_from(geometry.width()).map_err(Error::backend)?;
        let group_width = usize::try_from(geometry.group_width()).map_err(Error::backend)?;
        let gated = input
            .data
            .iter()
            .zip(&gate.data)
            .map(|(input, gate)| input * gate / (1.0 + (-gate).exp()))
            .collect::<Vec<_>>();
        let mut output = vec![0.0; gated.len()];
        for (group_index, (source, target)) in gated
            .chunks_exact(group_width)
            .zip(output.chunks_exact_mut(group_width))
            .enumerate()
        {
            let rms = (source.iter().map(|value| value * value).sum::<f32>() / group_width as f32
                + epsilon)
                .sqrt();
            let group_offset = (group_index * group_width) % width;
            for (index, (target, source)) in target.iter_mut().zip(source).enumerate() {
                *target = *source / rms * weight.data[(group_offset + index) % width];
            }
        }
        Ok(NumericTensor::new(input.shape.clone(), output))
    }

    fn silu_gated_group_rms_norm(
        input: &Self::Tensor,
        gate: &Self::Tensor,
        weight: &Self::Tensor,
        groups: i32,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            &input.shape,
            &gate.shape,
            groups,
            epsilon,
        )?;
        let width = usize::try_from(geometry.width()).map_err(Error::backend)?;
        let groups = usize::try_from(groups).map_err(Error::backend)?;
        let group_width = usize::try_from(geometry.group_width()).map_err(Error::backend)?;
        if weight.shape != [group_width as i32] && weight.shape != [width as i32] {
            return Err(Error::backend(
                "numeric SiLU-gated RMS weight geometry mismatch",
            ));
        }
        let mut output = NumericTensor::zeros(input.shape.clone());
        for row in 0..input.data.len() / width {
            for group in 0..groups {
                let start = row * width + group * group_width;
                let source = &input.data[start..start + group_width];
                let rms = (source.iter().map(|value| value * value).sum::<f32>()
                    / group_width as f32
                    + epsilon)
                    .sqrt();
                for (dimension, source_value) in source.iter().copied().enumerate() {
                    let gate = gate.data[start + dimension];
                    let scale = if weight.data.len() == width {
                        weight.data[group * group_width + dimension]
                    } else {
                        weight.data[dimension]
                    };
                    output.data[start + dimension] =
                        source_value / rms * scale * (gate / (1.0 + (-gate).exp()));
                }
            }
        }
        Ok(output)
    }

    fn gated_delta_scan(
        input: GatedDeltaScanInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<GatedDeltaScanOutput<NumericTensor>, Error> {
        let [batch, sequence, heads, key_dimensions] = input.query.shape.as_slice() else {
            return Err(Error::backend(
                "numeric gated-delta query must be rank four",
            ));
        };
        let value_dimensions = *input
            .value
            .shape
            .last()
            .ok_or_else(|| Error::backend("numeric gated-delta value has no width"))?;
        let vector_decay = input.log_decay.shape.len() == 4;
        let (state, output) = reference_gated_delta_scan(
            *batch as usize,
            *sequence as usize,
            *heads as usize,
            *key_dimensions as usize,
            value_dimensions as usize,
            &input.query.data,
            &input.key.data,
            &input.value.data,
            &input.log_decay.data,
            vector_decay,
            &input.beta.data,
            input.initial_state.map(|state| state.data.as_slice()),
        )?;
        Ok(GatedDeltaScanOutput {
            state: NumericTensor::new(
                vec![*batch, *heads, *key_dimensions, value_dimensions],
                state,
            ),
            output: NumericTensor::new(vec![*batch, *sequence, *heads, value_dimensions], output),
        })
    }

    fn selective_state_space_scan(
        input: SelectiveStateSpaceScanInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<SelectiveStateSpaceScanOutput<NumericTensor>, Error> {
        let [batch, sequence, heads, head_dimensions] = input.values.shape.as_slice() else {
            return Err(Error::backend(
                "numeric selective-scan values must be rank four",
            ));
        };
        let state_dimensions = *input
            .input_state
            .shape
            .last()
            .ok_or_else(|| Error::backend("numeric selective scan has no state width"))?;
        let (state, output) = reference_selective_state_space_scan(
            *batch as usize,
            *sequence as usize,
            *heads as usize,
            *head_dimensions as usize,
            state_dimensions as usize,
            &input.values.data,
            &input.input_state.data,
            &input.output_state.data,
            &input.time_step.data,
            &input.time_step_bias.data,
            &input.transition_log.data,
            &input.skip.data,
            input.time_step_floor,
            input.initial_state.map(|state| state.data.as_slice()),
        )?;
        Ok(SelectiveStateSpaceScanOutput {
            state: NumericTensor::new(
                vec![*batch, *heads, *head_dimensions, state_dimensions],
                state,
            ),
            output: NumericTensor::new(vec![*batch, *sequence, *heads, *head_dimensions], output),
        })
    }

    fn gated_product(
        gate: Self::Tensor,
        up: Self::Tensor,
        policy: eredu_nn::GatedProductPolicy,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        policy.validate()?;
        let gate = gate.map(|value| {
            policy
                .gate_upper_bound()
                .map_or(value, |bound| value.min(bound))
        });
        let up = up.map(|value| {
            policy
                .up_absolute_bound()
                .map_or(value, |bound| value.clamp(-bound, bound))
                + policy.up_offset()
        });
        let gate = match policy.activation() {
            eredu_nn::GatedProductActivation::Silu => {
                gate.map(|value| value / (1.0 + (-policy.sigmoid_multiplier() * value).exp()))
            }
            eredu_nn::GatedProductActivation::GeluApproximate => gate.map(|value| {
                0.5 * value
                    * (1.0 + (0.797_884_6 * (value + 0.044_715 * value * value * value)).tanh())
            }),
            _ => return Err(Error::backend("unsupported gated-product activation")),
        };
        gate.zip(&up, |left, right| left * right)
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

    fn relative_attention(
        input: RelativeAttentionInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        input.validate()?;
        let [batch, heads, queries, dimensions] = input.queries.shape.as_slice() else {
            unreachable!()
        };
        let key_heads = input.keys.shape[1];
        let keys = input.keys.shape[2];
        let extent = input.profiles.shape[3];
        let mut output = NumericTensor::zeros(input.queries.shape.clone());
        for b in 0..*batch as usize {
            for head in 0..*heads as usize {
                // GQA repeats each KV head for a contiguous group of query heads.
                let key_head = head / (*heads / key_heads) as usize;
                for query in 0..*queries as usize {
                    let query_position = input.query_offset + query as i32;
                    let tau = if input.window.is_none() {
                        input.log_scaling_floor.map_or(1.0, |floor| {
                            1.0 + input.log_scaling_alpha
                                * (((query_position + 1) as f32 / floor as f32).max(1.0).ln())
                        })
                    } else {
                        1.0
                    };
                    let mut scores = vec![f32::NEG_INFINITY; keys as usize];
                    for (key, score) in scores.iter_mut().enumerate() {
                        let distance = query_position - (input.key_offset + key as i32);
                        if distance < 0 || input.window.is_some_and(|window| distance >= window) {
                            continue;
                        }
                        let query_base = ((b * *heads as usize + head) * *queries as usize + query)
                            * *dimensions as usize;
                        let key_base = ((b * key_heads as usize + key_head) * keys as usize + key)
                            * *dimensions as usize;
                        let dot = (0..*dimensions as usize)
                            .map(|dimension| {
                                input.queries.data[query_base + dimension]
                                    * input.keys.data[key_base + dimension]
                            })
                            .sum::<f32>()
                            / *dimensions as f32;
                        let bias = if distance < extent {
                            let base = ((b * *heads as usize + head) * *queries as usize + query)
                                * extent as usize;
                            input.profiles.data[base + distance as usize]
                        } else {
                            0.0
                        };
                        *score = (dot + bias) * tau;
                    }
                    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denominator = scores
                        .iter()
                        .map(|score| (*score - maximum).exp())
                        .sum::<f32>();
                    let output_base = ((b * *heads as usize + head) * *queries as usize + query)
                        * *dimensions as usize;
                    for (key, score) in scores.iter().enumerate() {
                        let probability = (*score - maximum).exp() / denominator;
                        let value_base = ((b * key_heads as usize + key_head) * keys as usize
                            + key)
                            * *dimensions as usize;
                        for dimension in 0..*dimensions as usize {
                            output.data[output_base + dimension] +=
                                probability * input.values.data[value_base + dimension];
                        }
                    }
                }
            }
        }
        Ok(output)
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
        request: AttentionRequest<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        request.validate()?;
        attention_with_softcap(
            &request.queries,
            &request.keys,
            &request.values,
            request.scale,
            request.mask,
            request.sinks,
            request.softcap,
        )
    }

    fn sliding_window_attention_with_sinks(
        request: AttentionRequest<'_, NumericTensor>,
        window: i32,
        position_offset: i32,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        request.validate()?;
        context
            .sliding_attention_calls
            .set(context.sliding_attention_calls.get() + 1);
        let batch = request.queries.shape[0];
        let sequence = request.queries.shape[2];
        let attended = attention_with_softcap_windowed(
            &request.queries,
            &request.keys,
            &request.values,
            request.scale,
            request.sinks,
            window,
            position_offset,
            request.softcap,
        )?;
        attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)
    }

    fn rms_norm_without_weight(
        input: &NumericTensor,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let geometry =
            eredu_nn::operation_geometry::NormalizationGeometry::new(&input.shape, epsilon)?;
        let dimensions = usize::try_from(geometry.width()).map_err(Error::backend)?;
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
        let geometry = eredu_nn::operation_geometry::CausalMaskGeometry::new(
            sequence,
            position_offset,
            window,
        )?;
        let keys = geometry.keys();
        let mut mask = NumericTensor::zeros(vec![sequence, keys]);
        for query in 0..sequence {
            for key in 0..keys {
                if !geometry.allows(query, key) {
                    mask.data[(query * keys + key) as usize] = -1.0e9;
                }
            }
        }
        Ok(mask)
    }

    fn segmented_attention(
        input: eredu_nn::SegmentedAttentionInput<'_, Self::Tensor>,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        input.validate()?;
        let tokens = usize::try_from(input.queries.shape[0]).map_err(Error::backend)?;
        let heads = usize::try_from(input.queries.shape[1]).map_err(Error::backend)?;
        let dimensions = usize::try_from(input.queries.shape[2]).map_err(Error::backend)?;
        let value_dimensions = usize::try_from(input.values.shape[2]).map_err(Error::backend)?;
        let output = eredu_nn::reference_segmented_attention(
            tokens,
            heads,
            dimensions,
            value_dimensions,
            &input.queries.data,
            &input.keys.data,
            &input.values.data,
            input.segment_lengths,
            input.scale,
        )?;
        Ok(NumericTensor::new(
            [
                input.queries.shape[0],
                input.queries.shape[1],
                input.values.shape[2],
            ],
            output,
        ))
    }

    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        parallel: &Self::ParallelContext,
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let bias = linear.bias.take();
        let local = linear.forward(input, context)?;
        let reduced = parallel.collective(NumericCollectiveKind::Sum, local)?;
        let output = match &bias {
            Some((bias, _)) => reduced.add(bias, context)?,
            None => reduced,
        };
        linear.bias = bias;
        Ok(output)
    }

    fn parallel_size(parallel: &Self::ParallelContext) -> usize {
        parallel.group.size
    }
}

impl eredu_nn::DistributedNeuralBackend for NumericBackend {
    fn vocabulary_parallel_embedding(
        spec: EmbeddingSpec,
        range: VocabularyParallelRange,
        _context: &NumericContext,
    ) -> Result<Self::Embedding, Error> {
        range.validate_global_rows(spec.vocabulary)?;
        let global = parameter(&spec.weight, vec![spec.vocabulary, spec.dimensions], false);
        Ok(NumericEmbedding {
            weight: global.axis_slice(0, range.local.start, range.local.end),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            vocabulary_range: Some(range),
            execution_weight: None,
            format_companions: Vec::new(),
        })
    }

    fn vocabulary_parallel_linear(
        spec: LinearSpec,
        range: VocabularyParallelRange,
        _context: &NumericContext,
    ) -> Result<Self::Linear, Error> {
        range.validate_global_rows(spec.output)?;
        let global = parameter(&spec.weight, vec![spec.output, spec.input], false);
        let weight = global.axis_slice(0, range.local.start, range.local.end);
        let bias = spec.bias.map(|bias| {
            let value = parameter(&bias, vec![spec.output], false).axis_slice(
                0,
                range.local.start,
                range.local.end,
            );
            let metadata = ParameterMetadata::from_spec(&bias, bias.trainable);
            (value, metadata)
        });
        Ok(NumericLinear {
            weight,
            weight_metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            bias,
            execution_weight: None,
            format_companions: Vec::new(),
        })
    }

    fn vocabulary_parallel_lookup(
        embedding: &mut Self::Embedding,
        input: &Self::Tensor,
        _policy: EmbeddingLookupPolicy,
        parallel: &Self::ParallelContext,
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let local = embedding.lookup(input, _policy, context)?;
        parallel.collective(NumericCollectiveKind::Sum, local)
    }

    fn vocabulary_parallel_project(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        parallel: &Self::ParallelContext,
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let local = linear.forward(input, context)?;
        parallel.collective(NumericCollectiveKind::GatherVocabulary, local)
    }

    fn vocabulary_parallel_embedding_project(
        embedding: &mut Self::Embedding,
        input: &Self::Tensor,
        parallel: &Self::ParallelContext,
        context: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let mut linear = NumericLinear {
            weight: embedding.weight.clone(),
            weight_metadata: embedding.metadata.clone(),
            bias: None,
            execution_weight: embedding.execution_weight.clone(),
            format_companions: Vec::new(),
        };
        let local = linear.forward(input, context)?;
        parallel.collective(NumericCollectiveKind::GatherVocabulary, local)
    }

    fn sum_parallel(
        value: Self::Tensor,
        parallel: &Self::ParallelContext,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        parallel.collective(NumericCollectiveKind::Sum, value)
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
    let mask_value = |mask: &NumericTensor,
                      batch_index: usize,
                      query_head: usize,
                      query_position: usize,
                      key_position: usize|
     -> Result<f32, Error> {
        if mask.shape.len() > 4 {
            return Err(Error::backend("numeric attention mask rank exceeds four"));
        }
        let target = [batch, query_heads, query_sequence, key_sequence];
        let leading = 4 - mask.shape.len();
        let full_shape = (0..4)
            .map(|current| {
                if current < leading {
                    1
                } else {
                    mask.shape[current - leading] as usize
                }
            })
            .collect::<Vec<_>>();
        if full_shape
            .iter()
            .zip(target)
            .any(|(actual, expected)| *actual != 1 && *actual != expected)
        {
            return Err(Error::backend(format!(
                "numeric attention mask shape {:?} does not broadcast to {:?}",
                mask.shape, target
            )));
        }
        let target_coordinate = [batch_index, query_head, query_position, key_position];
        let coordinate = (leading..4)
            .map(|current| {
                if full_shape[current] == 1 {
                    0
                } else {
                    target_coordinate[current]
                }
            })
            .collect::<Vec<_>>();
        Ok(mask.data[offset(&coordinate, &mask.shape)])
    };
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
                            + match mask {
                                Some(mask) => mask_value(
                                    mask,
                                    batch_index,
                                    query_head,
                                    query_position,
                                    key_position,
                                )?,
                                None => 0.0,
                            };
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
    attention_with_softcap(queries, keys, values, scale, mask, sinks, None)
}

fn attention_with_softcap(
    queries: &NumericTensor,
    keys: &NumericTensor,
    values: &NumericTensor,
    scale: f32,
    mask: Option<&NumericTensor>,
    sinks: Option<&NumericTensor>,
    softcap: Option<f32>,
) -> Result<NumericTensor, Error> {
    if sinks.is_none() && softcap.is_none() {
        return attention(queries, keys, values, scale, mask, None, 0);
    }
    if queries.shape.len() != 4
        || keys.shape.len() != 4
        || values.shape != keys.shape
        || queries.shape[0] != keys.shape[0]
        || queries.shape[3] != keys.shape[3]
        || keys.shape[1] <= 0
        || queries.shape[1] % keys.shape[1] != 0
    {
        return Err(Error::backend("numeric sink-attention geometry mismatch"));
    }
    let batch = queries.shape[0] as usize;
    let heads = queries.shape[1] as usize;
    let key_heads = keys.shape[1] as usize;
    let groups = heads / key_heads;
    let query_tokens = queries.shape[2] as usize;
    let key_tokens = keys.shape[2] as usize;
    let dimensions = queries.shape[3] as usize;
    if sinks.is_some_and(|sinks| sinks.shape != [heads as i32])
        || mask.is_some_and(|mask| mask.shape != [query_tokens as i32, key_tokens as i32])
    {
        return Err(Error::backend(
            "numeric sink-attention mask or sink mismatch",
        ));
    }
    let mut output = NumericTensor::zeros(queries.shape.clone());
    for b in 0..batch {
        for head in 0..heads {
            let key_head = head / groups;
            for query in 0..query_tokens {
                let query_base = ((b * heads + head) * query_tokens + query) * dimensions;
                let mut scores = (0..key_tokens)
                    .map(|key| {
                        let key_base = ((b * key_heads + key_head) * key_tokens + key) * dimensions;
                        let score = (0..dimensions)
                            .map(|dimension| {
                                queries.data[query_base + dimension]
                                    * keys.data[key_base + dimension]
                            })
                            .sum::<f32>()
                            * scale;
                        softcap.map_or(score, |cap| cap * (score / cap).tanh())
                            + mask.map_or(0.0, |mask| mask.data[query * key_tokens + key])
                    })
                    .collect::<Vec<_>>();
                if let Some(sinks) = sinks {
                    scores.push(sinks.data[head]);
                }
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
                                * values.data[((b * key_heads + key_head) * key_tokens + key)
                                    * dimensions
                                    + dimension]
                        })
                        .sum::<f32>()
                        / denominator;
                }
            }
        }
    }
    Ok(output)
}

fn attention_with_sinks_windowed(
    queries: &NumericTensor,
    keys: &NumericTensor,
    values: &NumericTensor,
    scale: f32,
    sinks: Option<&NumericTensor>,
    window: i32,
    query_offset: i32,
) -> Result<NumericTensor, Error> {
    attention_with_softcap_windowed(
        queries,
        keys,
        values,
        scale,
        sinks,
        window,
        query_offset,
        None,
    )
}

fn attention_with_softcap_windowed(
    queries: &NumericTensor,
    keys: &NumericTensor,
    values: &NumericTensor,
    scale: f32,
    sinks: Option<&NumericTensor>,
    window: i32,
    query_offset: i32,
    softcap: Option<f32>,
) -> Result<NumericTensor, Error> {
    if window <= 0 || queries.shape.len() != 4 || keys.shape.len() != 4 {
        return Err(Error::backend(
            "numeric sliding sink-attention geometry mismatch",
        ));
    }
    let query_tokens = queries.shape[2];
    let key_tokens = keys.shape[2];
    let key_offset = query_offset + query_tokens - key_tokens;
    if key_offset < 0 {
        return Err(Error::backend(
            "numeric sliding attention key origin precedes position zero",
        ));
    }
    let mut mask = NumericTensor::zeros(vec![query_tokens, key_tokens]);
    for query in 0..query_tokens {
        let absolute_query = query_offset + query;
        let first_visible = (absolute_query - window + 1).max(0);
        for key in 0..key_tokens {
            let absolute_key = key_offset + key;
            if absolute_key < first_visible || absolute_key > absolute_query {
                mask.data[(query * key_tokens + key) as usize] = f32::NEG_INFINITY;
            }
        }
    }
    attention_with_softcap(queries, keys, values, scale, Some(&mask), sinks, softcap)
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
    selection: TopKGroupSelectionSpec,
    correction_bias: Option<(NumericTensor, ParameterMetadata)>,
    input_transform: Option<(f32, NumericTensor, ParameterMetadata, bool)>,
    coefficient_scale: Option<(NumericTensor, ParameterMetadata)>,
}

impl Parameterized<NumericTensor> for NumericRouter {


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

__source_result = __source_result.and(        self.linear.visit_parameter_sources(visitor));
        if let Some((value, metadata)) = &self.correction_bias {
            visit(metadata, value, visitor);
        }
        if let Some((_, value, metadata, _)) = &self.input_transform {
            visit(metadata, value, visitor);
        }
        if let Some((value, metadata)) = &self.coefficient_scale {
            visit(metadata, value, visitor);
        }

 __source_result
}

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        self.linear.visit_parameters_mut(visitor);
        if let Some((value, metadata)) = &mut self.correction_bias {
            visit_mut(metadata, value, visitor);
        }
        if let Some((_, value, metadata, _)) = &mut self.input_transform {
            visit_mut(metadata, value, visitor);
        }
        if let Some((value, metadata)) = &mut self.coefficient_scale {
            visit_mut(metadata, value, visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.linear.set_trainable(trainable);
        if let Some((_, metadata)) = &mut self.correction_bias {
            metadata.trainable = trainable;
        }
        if let Some((_, _, metadata, _)) = &mut self.input_transform {
            metadata.trainable = trainable;
        }
        if let Some((_, metadata)) = &mut self.coefficient_scale {
            metadata.trainable = trainable;
        }
    }
}

impl NumericRouter {
    fn transformed_input(&self, input: &NumericTensor) -> NumericTensor {
        let Some((epsilon, scale, _, inverse_sqrt_dimensions)) = &self.input_transform else {
            return input.clone();
        };
        let width = input.shape.last().copied().unwrap() as usize;
        let width_scale = if *inverse_sqrt_dimensions {
            (width as f32).sqrt().recip()
        } else {
            1.0
        };
        let mut output = input.clone();
        for row in output.data.chunks_mut(width) {
            let rms = (row.iter().map(|value| value * value).sum::<f32>() / width as f32 + epsilon)
                .sqrt()
                .recip();
            for (dimension, value) in row.iter_mut().enumerate() {
                *value *= rms * scale.data[dimension] * width_scale;
            }
        }
        output
    }
}

impl GroupSelectionOperator<NumericTensor> for NumericRouter {
    fn select(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<GroupSelection<NumericTensor>, Error> {
        let input = self.transformed_input(input);
        let logits = self.linear.forward(&input, context)?;
        let experts = self.selection.group_count() as usize;
        let top_k = self.selection.top_k() as usize;
        let tokens = logits.data.len() / experts;
        let route_shape = vec![tokens as i32, top_k as i32];
        let mut group_indices = NumericTensor::zeros(route_shape.clone());
        let mut selected_scores = NumericTensor::zeros(route_shape.clone());
        let mut coefficients = NumericTensor::zeros(route_shape);
        for token in 0..tokens {
            let row = &logits.data[token * experts..(token + 1) * experts];
            let scores: Vec<f32> = match self.selection.scoring() {
                eredu_nn::GroupScoring::Softmax => {
                    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exponentials = row
                        .iter()
                        .map(|value| (*value - maximum).exp())
                        .collect::<Vec<_>>();
                    let sum = exponentials.iter().sum::<f32>();
                    exponentials.iter().map(|value| *value / sum).collect()
                }
                eredu_nn::GroupScoring::SelectedSoftmax => row.to_vec(),
                eredu_nn::GroupScoring::Sigmoid => row
                    .iter()
                    .map(|value| 1.0 / (1.0 + (-value).exp()))
                    .collect(),
                eredu_nn::GroupScoring::SqrtSoftplus => row
                    .iter()
                    .map(|value| (1.0 + value.exp()).ln().sqrt())
                    .collect(),
                _ => return Err(Error::backend("unsupported group scoring policy")),
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
            let groups = self.selection.selection_partitions() as usize;
            let selected_groups = self.selection.selected_groups() as usize;
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
            let selected = order[..top_k]
                .iter()
                .map(|expert| scores[*expert])
                .collect::<Vec<_>>();
            let selected = if self.selection.scoring() == eredu_nn::GroupScoring::SelectedSoftmax {
                let maximum = selected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exponentials = selected
                    .iter()
                    .map(|value| (*value - maximum).exp())
                    .collect::<Vec<_>>();
                let sum = exponentials.iter().sum::<f32>();
                exponentials.into_iter().map(|value| value / sum).collect()
            } else {
                selected
            };
            let selected_sum = selected.iter().sum::<f32>();
            for (route, expert) in order.iter().copied().take(top_k).enumerate() {
                let selected = selected[route];
                let index = token * top_k + route;
                group_indices.data[index] = expert as f32;
                selected_scores.data[index] = selected;
                coefficients.data[index] = if self.selection.normalize_selected() {
                    selected / (selected_sum + self.selection.normalization_epsilon())
                } else {
                    selected
                } * self.selection.coefficient_scale()
                    * self
                        .coefficient_scale
                        .as_ref()
                        .map_or(1.0, |(scale, _)| scale.data[expert]);
            }
        }
        Ok(GroupSelection::new(
            group_indices,
            selected_scores,
            coefficients,
        ))
    }

    fn select_indices(
        &mut self,
        input: &NumericTensor,
        group_indices: &NumericTensor,
        context: &NumericContext,
    ) -> Result<GroupSelection<NumericTensor>, Error> {
        let input = self.transformed_input(input);
        let logits = self.linear.forward(&input, context)?;
        let experts = self.selection.group_count() as usize;
        let top_k = self.selection.top_k() as usize;
        let tokens = logits.data.len() / experts;
        if group_indices.shape != [tokens as i32, top_k as i32] {
            return Err(Error::backend("caller-selected route geometry mismatch"));
        }
        let mut selected_scores = NumericTensor::zeros(group_indices.shape.clone());
        let mut coefficients = NumericTensor::zeros(group_indices.shape.clone());
        for token in 0..tokens {
            let row = &logits.data[token * experts..(token + 1) * experts];
            let scores = match self.selection.scoring() {
                eredu_nn::GroupScoring::Softmax => {
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
                eredu_nn::GroupScoring::SelectedSoftmax => row.to_vec(),
                eredu_nn::GroupScoring::Sigmoid => row
                    .iter()
                    .map(|value| 1.0 / (1.0 + (-value).exp()))
                    .collect(),
                eredu_nn::GroupScoring::SqrtSoftplus => row
                    .iter()
                    .map(|value| (1.0 + value.exp()).ln().sqrt())
                    .collect(),
                _ => return Err(Error::backend("unsupported group scoring policy")),
            };
            let mut sum = 0.0;
            for route in 0..top_k {
                let index = token * top_k + route;
                let expert = group_indices.data[index] as usize;
                if expert >= experts || expert as f32 != group_indices.data[index] {
                    return Err(Error::backend("caller-selected expert id is invalid"));
                }
                selected_scores.data[index] = scores[expert];
                sum += scores[expert];
            }
            if self.selection.scoring() == eredu_nn::GroupScoring::SelectedSoftmax {
                let start = token * top_k;
                let maximum = selected_scores.data[start..start + top_k]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut denominator = 0.0;
                for value in &mut selected_scores.data[start..start + top_k] {
                    *value = (*value - maximum).exp();
                    denominator += *value;
                }
                for value in &mut selected_scores.data[start..start + top_k] {
                    *value /= denominator;
                }
                sum = 1.0;
            }
            for route in 0..top_k {
                let index = token * top_k + route;
                let expert = group_indices.data[index] as usize;
                coefficients.data[index] = if self.selection.normalize_selected() {
                    selected_scores.data[index] / (sum + self.selection.normalization_epsilon())
                } else {
                    selected_scores.data[index]
                } * self.selection.coefficient_scale()
                    * self
                        .coefficient_scale
                        .as_ref()
                        .map_or(1.0, |(scale, _)| scale.data[expert]);
            }
        }
        Ok(GroupSelection::new(
            group_indices.clone(),
            selected_scores,
            coefficients,
        ))
    }
}

#[derive(Debug, Clone)]
struct NumericExpert {
    gate: NumericTensor,
    gate_bias: Option<NumericTensor>,
    up: NumericTensor,
    up_bias: Option<NumericTensor>,
    down: NumericTensor,
    down_bias: Option<NumericTensor>,
}

#[derive(Debug, Clone)]
struct NumericExpertBank {
    experts: Vec<NumericExpert>,
    parameters: Vec<(NumericTensor, ParameterMetadata)>,
    policy: eredu_nn::GatedProductPolicy,
    spec: GroupedGatedProductSpec,
}

fn numeric_expert_bank_spec(
    expert_count: i32,
    hidden: i32,
    intermediate: i32,
    policy: GatedProductPolicy,
) -> GroupedGatedProductSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    GroupedGatedProductSpec::new(
        expert_count,
        hidden,
        intermediate,
        hidden,
        policy,
        GatedProductGroupLayout::Packed {
            gate_up: eredu_nn::GroupedProjectionSpec::new(
                parameter("test.experts.gate_up_proj"),
                None,
                dense_linear_format(),
            )
            .unwrap(),
            down: eredu_nn::GroupedProjectionSpec::new(
                parameter("test.experts.down_proj"),
                None,
                dense_linear_format(),
            )
            .unwrap(),
        },
    )
    .unwrap()
}

fn numeric_grouped_projection_parameters(
    projection: &eredu_nn::GroupedProjectionSpec,
    logical: NumericTensor,
    group_count: i32,
    output: i32,
    input: i32,
    context: &NumericContext,
) -> Result<Vec<(NumericTensor, ParameterMetadata)>, Error> {
    let mut parameters = Vec::new();
    match projection.format().encoding() {
        eredu_checkpoint::LinearFormat::MxFp4 => {
            let groups = input.checked_div(32).ok_or_else(|| {
                Error::backend("numeric grouped MXFP4 companion shape overflowed")
            })?;
            parameters.push((
                logical,
                ParameterMetadata::from_spec(projection.weight(), projection.weight().trainable),
            ));
            if let Some(companion) = projection.format().scale() {
                parameters.push((
                    local_parameter(companion, vec![group_count, output, groups], false, context)?,
                    numeric_companion_metadata(companion, projection.weight()),
                ));
            }
        }
        eredu_checkpoint::LinearFormat::Affine(format) => {
            let groups = input.checked_div(format.group_size).ok_or_else(|| {
                Error::backend("numeric grouped affine companion shape overflowed")
            })?;
            parameters.push((
                logical,
                ParameterMetadata::from_spec(projection.weight(), projection.weight().trainable),
            ));
            for companion in [
                projection.format().scale(),
                projection.format().affine_bias(),
            ]
            .into_iter()
            .flatten()
            {
                parameters.push((
                    local_parameter(companion, vec![group_count, output, groups], false, context)?,
                    numeric_companion_metadata(companion, projection.weight()),
                ));
            }
        }
        _ => parameters.push((
            logical,
            ParameterMetadata::from_spec(projection.weight(), projection.weight().trainable),
        )),
    }
    Ok(parameters)
}

fn numeric_independent_projection_parameters(
    projection: &eredu_nn::GroupedProjectionSpec,
    logical: NumericTensor,
    output: i32,
    input: i32,
    context: &NumericContext,
) -> Result<Vec<(NumericTensor, ParameterMetadata)>, Error> {
    let mut parameters = Vec::new();
    match projection.format().encoding() {
        eredu_checkpoint::LinearFormat::Affine(format) => {
            let groups = input.checked_div(format.group_size).ok_or_else(|| {
                Error::backend("numeric expert affine companion shape overflowed")
            })?;
            parameters.push((
                logical,
                ParameterMetadata::from_spec(projection.weight(), projection.weight().trainable),
            ));
            for companion in [
                projection.format().scale(),
                projection.format().affine_bias(),
            ]
            .into_iter()
            .flatten()
            {
                parameters.push((
                    local_parameter(companion, vec![output, groups], false, context)?,
                    numeric_companion_metadata(companion, projection.weight()),
                ));
            }
        }
        _ => parameters.push((
            logical,
            ParameterMetadata::from_spec(projection.weight(), projection.weight().trainable),
        )),
    }
    Ok(parameters)
}

#[derive(Debug, Clone)]
struct NumericRelu2Groups {
    spec: GroupedRelu2Spec,
    expert_count: usize,
    hidden: usize,
    intermediate: usize,
    up: (NumericTensor, ParameterMetadata),
    down: (NumericTensor, ParameterMetadata),
}

impl Parameterized<NumericTensor> for NumericRelu2Groups {


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

        visit(&self.up.1, &self.up.0, visitor);
        visit(&self.down.1, &self.down.0, visitor);

 __source_result
}

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        visit_mut(&self.up.1, &mut self.up.0, visitor);
        visit_mut(&self.down.1, &mut self.down.0, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.up.1.trainable = trainable;
        self.down.1.trainable = trainable;
    }
}

impl GroupedRelu2Operator<NumericTensor> for NumericRelu2Groups {
    fn spec(&self) -> &GroupedRelu2Spec {
        &self.spec
    }

    fn forward_grouped(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let tokens = input.data.len() / self.hidden;
        let top_k = routes.group_indices().shape[1] as usize;
        if routes.group_indices().shape != [tokens as i32, top_k as i32]
            || routes.coefficients().shape != routes.group_indices().shape
        {
            return Err(Error::backend("numeric ReLU2 route geometry mismatch"));
        }
        let mut output = NumericTensor::zeros(input.shape.clone());
        for token in 0..tokens {
            let token_input = NumericTensor::new(
                vec![1, self.hidden as i32],
                input.data[token * self.hidden..(token + 1) * self.hidden].to_vec(),
            );
            for route in 0..top_k {
                let route_index = token * top_k + route;
                let expert = routes.group_indices().data[route_index] as usize;
                if expert >= self.expert_count {
                    return Err(Error::backend("numeric ReLU2 expert id is out of range"));
                }
                let up_start = expert * self.intermediate * self.hidden;
                let up = NumericTensor::new(
                    vec![self.intermediate as i32, self.hidden as i32],
                    self.up.0.data[up_start..up_start + self.intermediate * self.hidden].to_vec(),
                );
                let down_start = expert * self.hidden * self.intermediate;
                let down = NumericTensor::new(
                    vec![self.hidden as i32, self.intermediate as i32],
                    self.down.0.data[down_start..down_start + self.hidden * self.intermediate]
                        .to_vec(),
                );
                let activated =
                    linear(&token_input, &up, None)?.map(|value| value.max(0.0).powi(2));
                let expert_output = linear(&activated, &down, None)?;
                let weight = routes.coefficients().data[route_index];
                for dimension in 0..self.hidden {
                    output.data[token * self.hidden + dimension] +=
                        weight * expert_output.data[dimension];
                }
            }
        }
        Ok(output)
    }
    fn forward_grouped_with_unit_observer(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        context: &NumericContext,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<NumericTensor>>,
    ) -> Result<NumericTensor, Error> {
        match observer {
            Some(observer) => routed_units::relu(self, input, routes, observer),
            None => self.forward_grouped(input, routes, context),
        }
    }
}

impl TensorParallelGroupedRelu2Operator<NumericTensor> for NumericRelu2Groups {
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        _: usize,
        context: &NumericContext,
    ) -> Result<TensorParallelGroupedOutput<NumericTensor>, Error> {
        Ok(TensorParallelGroupedOutput::new(
            self.forward_grouped(input, routes, context)?,
            None,
        ))
    }
    fn forward_grouped_tensor_parallel_with_unit_observer(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        partitions: usize,
        context: &NumericContext,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<NumericTensor>>,
    ) -> Result<TensorParallelGroupedOutput<NumericTensor>, Error> {
        match observer {
            Some(observer) => routed_units::relu(self, input, routes, observer)
                .map(|output| TensorParallelGroupedOutput::new(output, None)),
            None => self.forward_grouped_tensor_parallel(input, routes, partitions, context),
        }
    }
}

impl Parameterized<NumericTensor> for NumericExpertBank {


    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), eredu_nn::ParameterSourceError>
    where
        V: eredu_nn::ParameterSourceVisitor<'a, NumericTensor>,
    {
 let mut __source_result = Ok(());

        for (parameter, metadata) in &self.parameters {
            visit(metadata, parameter, visitor);
        }



        for expert in &self.experts {
            for value in [
                Some(&expert.gate),
                expert.gate_bias.as_ref(),
                Some(&expert.up),
                expert.up_bias.as_ref(),
                Some(&expert.down),
                expert.down_bias.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                visitor.retained(value);
            }
        }

__source_result
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

impl NumericExpertBank {
    // Parameter visitors bind canonical packed tensors. Refresh scalar views
    // before execution so checkpoint-backed tests consume the admitted payload.
    fn refresh_bound_experts(&mut self) -> Result<(), Error> {
        if self.parameters.is_empty() {
            return Ok(());
        }
        let value = |spec: &ParameterSpec| {
            self.parameters
                .iter()
                .find(|(_, metadata)| metadata.id == spec.id)
                .map(|(value, _)| value.clone())
                .ok_or_else(|| {
                    Error::backend(format!(
                        "missing scalar expert parameter {}",
                        spec.id.as_str()
                    ))
                })
        };
        let optional = |spec: Option<&ParameterSpec>| spec.map(&value).transpose();
        let mut experts = Vec::new();
        match self.spec.layout() {
            GatedProductGroupLayout::Packed { gate_up, down } => {
                let gu = value(gate_up.weight())?;
                let d = value(down.weight())?;
                let gu_bias = optional(gate_up.bias())?;
                let d_bias = optional(down.bias())?;
                let width = self.spec.intermediate_dimensions() as usize;
                let member = |tensor: &NumericTensor, expert| {
                    let tensor = tensor.axis_slice(0, expert, expert + 1);
                    NumericTensor::new(tensor.shape[1..].to_vec(), tensor.data)
                };
                for expert in 0..self.spec.group_count() as usize {
                    let gu = member(&gu, expert);
                    experts.push(NumericExpert {
                        gate: gu.axis_slice(0, 0, width),
                        up: gu.axis_slice(0, width, width * 2),
                        down: member(&d, expert),
                        gate_bias: gu_bias
                            .as_ref()
                            .map(|b| member(b, expert).axis_slice(0, 0, width)),
                        up_bias: gu_bias
                            .as_ref()
                            .map(|b| member(b, expert).axis_slice(0, width, width * 2)),
                        down_bias: d_bias.as_ref().map(|b| member(b, expert)),
                    });
                }
            }
            GatedProductGroupLayout::Independent(specs) => {
                for spec in specs {
                    experts.push(NumericExpert {
                        gate: value(spec.gate().weight())?,
                        up: value(spec.up().weight())?,
                        down: value(spec.down().weight())?,
                        gate_bias: optional(spec.gate().bias())?,
                        up_bias: optional(spec.up().bias())?,
                        down_bias: optional(spec.down().bias())?,
                    });
                }
            }
            _ => return Err(Error::backend("unsupported scalar expert layout")),
        }
        self.experts = experts;
        Ok(())
    }

    fn forward_bound(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
    ) -> Result<NumericTensor, Error> {
        let hidden = input.shape.last().copied().unwrap() as usize;
        let tokens = input.data.len() / hidden;
        let top_k = routes.group_indices().shape[1] as usize;
        if routes.group_indices().shape != [tokens as i32, top_k as i32]
            || routes.coefficients().shape != routes.group_indices().shape
        {
            return Err(Error::backend("numeric expert route geometry mismatch"));
        }
        let mut output = NumericTensor::zeros(input.shape.clone());
        for token in 0..tokens {
            let token_input = NumericTensor::new(
                vec![1, hidden as i32],
                input.data[token * hidden..(token + 1) * hidden].to_vec(),
            );
            let mut order = (0..top_k).collect::<Vec<_>>();
            if self.spec.reduction() == eredu_nn::GroupReduction::SequentialGroupOrder {
                order.sort_by_key(|&route| {
                    routes.group_indices().data[token * top_k + route] as usize
                });
            }
            for route in order {
                let route_index = token * top_k + route;
                let expert_id = routes.group_indices().data[route_index] as usize;
                let expert = self
                    .experts
                    .get(expert_id)
                    .ok_or_else(|| Error::backend("numeric expert id is out of range"))?;
                let gate =
                    linear(&token_input, &expert.gate, expert.gate_bias.as_ref())?.map(|value| {
                        self.policy
                            .gate_upper_bound()
                            .map_or(value, |bound| value.min(bound))
                    });
                let gate = gate.map(|value| match self.policy.activation() {
                    eredu_nn::GatedProductActivation::Silu => {
                        value / (1.0 + (-self.policy.sigmoid_multiplier() * value).exp())
                    }
                    eredu_nn::GatedProductActivation::GeluApproximate => {
                        0.5 * value
                            * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
                    }
                    _ => panic!("unsupported gated-product activation in numeric backend"),
                });
                let up = linear(&token_input, &expert.up, expert.up_bias.as_ref())?.map(|value| {
                    self.policy
                        .up_absolute_bound()
                        .map_or(value, |bound| value.clamp(-bound, bound))
                        + self.policy.up_offset()
                });
                let activated = gate.zip(&up, |left, right| left * right)?;
                let expert_output = linear(&activated, &expert.down, expert.down_bias.as_ref())?;
                let weight = routes.coefficients().data[route_index];
                for dimension in 0..hidden {
                    output.data[token * hidden + dimension] +=
                        weight * expert_output.data[dimension];
                }
            }
        }
        Ok(output)
    }
}

impl GroupedGatedProductOperator<NumericTensor> for NumericExpertBank {
    fn spec(&self) -> &GroupedGatedProductSpec {
        &self.spec
    }
    fn forward_grouped(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        if context.bind_checkpoint_values {
            self.refresh_bound_experts()?;
        }
        self.forward_bound(input, routes)
    }
    fn forward_grouped_with_unit_observer(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        context: &NumericContext,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<NumericTensor>>,
    ) -> Result<NumericTensor, Error> {
        match observer {
            Some(observer) => routed_units::gated(self, input, routes, context, observer, false)
                .map(|output| output.into_parts().0),
            None => self.forward_grouped(input, routes, context),
        }
    }
}

impl TensorParallelGroupedGatedProductOperator<NumericTensor> for NumericExpertBank {
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        _: usize,
        context: &NumericContext,
    ) -> Result<TensorParallelGroupedOutput<NumericTensor>, Error> {
        if context.bind_checkpoint_values {
            self.refresh_bound_experts()?;
        }
        let mut local = self.clone();
        let has_down_bias = local
            .experts
            .iter()
            .any(|expert| expert.down_bias.is_some());
        for expert in &mut local.experts {
            expert.down_bias = None;
        }
        let reducible = local.forward_bound(input, routes)?;
        let post_reduce = if has_down_bias {
            let hidden = input.shape.last().copied().unwrap() as usize;
            let tokens = input.data.len() / hidden;
            let top_k = routes.group_indices().shape[1] as usize;
            let mut bias = NumericTensor::zeros(input.shape.clone());
            for token in 0..tokens {
                for route in 0..top_k {
                    let route_index = token * top_k + route;
                    let expert = routes.group_indices().data[route_index] as usize;
                    let down_bias = self
                        .experts
                        .get(expert)
                        .and_then(|expert| expert.down_bias.as_ref())
                        .ok_or_else(|| Error::backend("numeric expert down-bias mismatch"))?;
                    let weight = routes.coefficients().data[route_index];
                    for dimension in 0..hidden {
                        bias.data[token * hidden + dimension] += weight * down_bias.data[dimension];
                    }
                }
            }
            Some(bias)
        } else {
            None
        };
        Ok(TensorParallelGroupedOutput::new(reducible, post_reduce))
    }
    fn forward_grouped_tensor_parallel_with_unit_observer(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        partitions: usize,
        context: &NumericContext,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<NumericTensor>>,
    ) -> Result<TensorParallelGroupedOutput<NumericTensor>, Error> {
        match observer {
            Some(observer) => routed_units::gated(self, input, routes, context, observer, true),
            None => self.forward_grouped_tensor_parallel(input, routes, partitions, context),
        }
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

impl SamplingBackend for NumericBackend {
    type Logits = NumericTensor;
    type Token = NumericTensor;
    type RandomState = i32;
    type Context = NumericContext;
    type Error = Error;

    fn error(message: String) -> Self::Error {
        Error::backend(message)
    }

    fn validate_token(
        token: &Self::Token,
        domain: TokenDomain,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        if token.data.iter().all(|value| {
            *value >= 0.0 && value.fract() == 0.0 && (*value as usize) < domain.cardinality()
        }) {
            Ok(token.clone())
        } else {
            Err(Error::backend(
                "numeric token is outside its decision domain",
            ))
        }
    }

    fn scale_temperature(
        logits: &Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn apply_penalties(
        logits: &Self::Logits,
        _: &[u32],
        _: PenaltyConfig,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn apply_top_k(
        logits: Self::Logits,
        _: i32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_top_p(
        logits: Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_min_p(
        logits: Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_token_filter(
        logits: &Self::Logits,
        _: &TokenFilter,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn apply_mirostat(
        logits: &Self::Logits,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn sample_raw(
        logits: &Self::Logits,
        _: f32,
        random: Option<&mut Self::RandomState>,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        let vocabulary = usize::try_from(*logits.shape.last().unwrap()).map_err(Error::backend)?;
        let rows = logits.data.len() / vocabulary;
        let mut shape = logits.shape.clone();
        shape.pop();
        let mut tokens = Vec::with_capacity(rows);
        for row in logits.data.chunks_exact(vocabulary) {
            let index = row
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .unwrap_or_default();
            tokens.push(index as f32);
        }
        if let Some(random) = random {
            *random += 1;
        }
        Ok(NumericTensor::new(shape, tokens))
    }

    fn sample_processed(
        logits: &Self::Logits,
        temperature: f32,
        random: Option<&mut Self::RandomState>,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        Self::sample_raw(logits, temperature, random, context)
    }

    fn token_id(token: &Self::Token, _: &Self::Context) -> Result<u32, Self::Error> {
        let value = *token
            .data
            .first()
            .ok_or_else(|| Error::backend("numeric token is empty"))?;
        if value < 0.0 || value.fract() != 0.0 {
            return Err(Error::backend("numeric token is invalid"));
        }
        Ok(value as u32)
    }

    fn token_probability(_: &Self::Logits, _: u32, _: &Self::Context) -> Result<f32, Self::Error> {
        Ok(1.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct NumericSampler;

impl Sampler<NumericBackend> for NumericSampler {
    fn sample(
        &mut self,
        logits: &NumericTensor,
        temperature: f32,
        random: Option<&mut i32>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        NumericBackend::sample_raw(logits, temperature, random, context)
    }
}

impl GroupedNeuralBackend for NumericBackend {
    type LinearGroups = grouped_linear::NumericLinearGroups;
    fn grouped_linear_bank(
        spec: eredu_nn::GroupedLinearSpec,
        context: &NumericContext,
    ) -> Result<Self::LinearGroups, Error> {
        grouped_linear::NumericLinearGroups::new(spec, context)
    }

    type Selector = NumericRouter;
    type GatedProductGroups = NumericExpertBank;
    type Relu2Groups = NumericRelu2Groups;

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

    fn joint_group_selection(
        input: JointGroupSelectionInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<JointGroupSelection<NumericTensor>, Error> {
        input.validate()?;
        let hidden = input.hidden().shape.last().copied().unwrap() as usize;
        let tokens = input.hidden().data.len() / hidden;
        let groups = input.weight().shape[0] as usize;
        let selectable = input.selectable_groups() as usize;
        let always_on = groups - selectable;
        let top_k = input.top_k() as usize;
        let global_scale = *input
            .global_scale()
            .data
            .first()
            .ok_or_else(|| Error::backend("numeric global coefficient scale is empty"))?;
        let mut primary_indices = NumericTensor::zeros(vec![tokens as i32, input.top_k()]);
        let mut primary_coefficients = NumericTensor::zeros(vec![tokens as i32, input.top_k()]);
        let mut always_on_coefficients =
            NumericTensor::zeros(vec![tokens as i32, always_on as i32]);
        for token in 0..tokens {
            let logits = (0..groups)
                .map(|group| {
                    (0..hidden)
                        .map(|column| {
                            input.hidden().data[token * hidden + column]
                                * input.weight().data[group * hidden + column]
                        })
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let mut order = (0..selectable).collect::<Vec<_>>();
            order.sort_by(|left, right| {
                let score = |group: usize| {
                    1.0 / (1.0 + (-logits[group]).exp()) + input.correction_bias().data[group]
                };
                score(*right)
                    .total_cmp(&score(*left))
                    .then_with(|| left.cmp(right))
            });
            order.truncate(top_k);
            let mut selected = order
                .iter()
                .map(|group| 1.0 / (1.0 + (-logits[*group]).exp()))
                .collect::<Vec<_>>();
            selected.extend(
                logits[selectable..]
                    .iter()
                    .map(|logit| 1.0 / (1.0 + (-logit).exp())),
            );
            let scale = input.coefficient_scale() * global_scale / selected.iter().sum::<f32>();
            for (slot, group) in order.into_iter().enumerate() {
                primary_indices.data[token * top_k + slot] = group as f32;
                primary_coefficients.data[token * top_k + slot] = selected[slot] * scale;
            }
            for group in 0..always_on {
                always_on_coefficients.data[token * always_on + group] =
                    selected[top_k + group] * scale;
            }
        }
        Ok(JointGroupSelection::new(
            primary_indices,
            primary_coefficients,
            always_on_coefficients,
        ))
    }

    fn top_k_group_selector(
        spec: TopKGroupSelectorSpec,
        context: &NumericContext,
    ) -> Result<Self::Selector, Error> {
        spec.validate()?;
        let routing = spec.selection();
        let linear = Self::linear(
            LinearSpec {
                input: spec.input_dimensions(),
                output: routing.group_count(),
                weight: spec.weight().clone(),
                bias: spec.bias().cloned(),
                format: spec.format().clone(),
            },
            context,
        )?;
        let correction_bias = spec.correction_bias().map(|parameter_spec| {
            let value = parameter(parameter_spec, vec![routing.group_count()], true);
            let metadata = ParameterMetadata::from_spec(parameter_spec, parameter_spec.trainable);
            (value, metadata)
        });
        let input_transform = spec.input_transform().map(|transform| {
            let value = parameter(transform.scale(), vec![spec.input_dimensions()], true);
            let metadata =
                ParameterMetadata::from_spec(transform.scale(), transform.scale().trainable);
            (
                transform.epsilon(),
                value,
                metadata,
                transform.inverse_sqrt_dimensions(),
            )
        });
        let coefficient_scale = spec.coefficient_scale().map(|parameter_spec| {
            let value = parameter(parameter_spec, vec![routing.group_count()], true);
            let metadata = ParameterMetadata::from_spec(parameter_spec, parameter_spec.trainable);
            (value, metadata)
        });
        Ok(NumericRouter {
            linear,
            selection: routing,
            correction_bias,
            input_transform,
            coefficient_scale,
        })
    }

    fn grouped_gated_product(
        spec: GroupedGatedProductSpec,
        context: &NumericContext,
    ) -> Result<Self::GatedProductGroups, Error> {
        spec.validate()?;
        let construction_spec = spec.clone();
        let expert_count = spec.group_count() as usize;
        let hidden = spec.input_dimensions();
        let intermediate = spec.intermediate_dimensions();
        let policy = spec.policy();
        let mut experts = Vec::with_capacity(expert_count);
        let mut parameters = Vec::new();
        match spec.layout() {
            GatedProductGroupLayout::Packed { gate_up, down } => {
                let packed_gate_up = local_parameter(
                    gate_up.weight(),
                    vec![spec.group_count(), 2 * intermediate, hidden],
                    false,
                    context,
                )?;
                let packed_down = local_parameter(
                    down.weight(),
                    vec![spec.group_count(), spec.output_dimensions(), intermediate],
                    false,
                    context,
                )?;
                let packed_gate_up_bias = gate_up
                    .bias()
                    .map(|bias| {
                        local_parameter(
                            bias,
                            vec![spec.group_count(), 2 * intermediate],
                            false,
                            context,
                        )
                    })
                    .transpose()?;
                let packed_down_bias = down
                    .bias()
                    .map(|bias| {
                        local_parameter(
                            bias,
                            vec![spec.group_count(), spec.output_dimensions()],
                            false,
                            context,
                        )
                    })
                    .transpose()?;
                let gate_up_per_expert = (2 * intermediate * hidden) as usize;
                let projection_per_expert = (intermediate * hidden) as usize;
                let down_per_expert = (spec.output_dimensions() * intermediate) as usize;
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
                        gate_bias: packed_gate_up_bias.as_ref().map(|bias| {
                            NumericTensor::new(
                                vec![intermediate],
                                bias.data[expert * 2 * intermediate as usize
                                    ..expert * 2 * intermediate as usize + intermediate as usize]
                                    .to_vec(),
                            )
                        }),
                        up: NumericTensor::new(
                            vec![intermediate, hidden],
                            packed_gate_up.data[gate_up_start + projection_per_expert
                                ..gate_up_start + 2 * projection_per_expert]
                                .to_vec(),
                        ),
                        up_bias: packed_gate_up_bias.as_ref().map(|bias| {
                            let start = expert * 2 * intermediate as usize + intermediate as usize;
                            NumericTensor::new(
                                vec![intermediate],
                                bias.data[start..start + intermediate as usize].to_vec(),
                            )
                        }),
                        down: NumericTensor::new(
                            vec![spec.output_dimensions(), intermediate],
                            packed_down.data[down_start..down_start + down_per_expert].to_vec(),
                        ),
                        down_bias: packed_down_bias.as_ref().map(|bias| {
                            let width = spec.output_dimensions() as usize;
                            NumericTensor::new(
                                vec![spec.output_dimensions()],
                                bias.data[expert * width..(expert + 1) * width].to_vec(),
                            )
                        }),
                    });
                }
                parameters.extend(numeric_grouped_projection_parameters(
                    gate_up,
                    packed_gate_up,
                    spec.group_count(),
                    2 * intermediate,
                    hidden,
                    context,
                )?);
                if let (Some(parameter_spec), Some(value)) = (gate_up.bias(), packed_gate_up_bias) {
                    parameters.push((
                        value,
                        ParameterMetadata::from_spec(parameter_spec, parameter_spec.trainable),
                    ));
                }
                if let (Some(parameter_spec), Some(value)) = (down.bias(), packed_down_bias) {
                    parameters.push((
                        value,
                        ParameterMetadata::from_spec(parameter_spec, parameter_spec.trainable),
                    ));
                }
                parameters.extend(numeric_grouped_projection_parameters(
                    down,
                    packed_down,
                    spec.group_count(),
                    spec.output_dimensions(),
                    intermediate,
                    context,
                )?);
            }
            GatedProductGroupLayout::Independent(specs) => {
                for expert_spec in specs {
                    let gate = local_parameter(
                        expert_spec.gate().weight(),
                        vec![intermediate, hidden],
                        false,
                        context,
                    )?;
                    let up = local_parameter(
                        expert_spec.up().weight(),
                        vec![intermediate, hidden],
                        false,
                        context,
                    )?;
                    let down = local_parameter(
                        expert_spec.down().weight(),
                        vec![spec.output_dimensions(), intermediate],
                        false,
                        context,
                    )?;
                    let gate_bias = expert_spec
                        .gate()
                        .bias()
                        .map(|bias| local_parameter(bias, vec![intermediate], false, context))
                        .transpose()?;
                    let up_bias = expert_spec
                        .up()
                        .bias()
                        .map(|bias| local_parameter(bias, vec![intermediate], false, context))
                        .transpose()?;
                    let down_bias = expert_spec
                        .down()
                        .bias()
                        .map(|bias| {
                            local_parameter(bias, vec![spec.output_dimensions()], false, context)
                        })
                        .transpose()?;
                    experts.push(NumericExpert {
                        gate: gate.clone(),
                        gate_bias: gate_bias.clone(),
                        up: up.clone(),
                        up_bias: up_bias.clone(),
                        down: down.clone(),
                        down_bias: down_bias.clone(),
                    });
                    parameters.extend(numeric_independent_projection_parameters(
                        expert_spec.gate(),
                        gate,
                        intermediate,
                        hidden,
                        context,
                    )?);
                    parameters.extend(numeric_independent_projection_parameters(
                        expert_spec.up(),
                        up,
                        intermediate,
                        hidden,
                        context,
                    )?);
                    parameters.extend(numeric_independent_projection_parameters(
                        expert_spec.down(),
                        down,
                        spec.output_dimensions(),
                        intermediate,
                        context,
                    )?);
                    for (projection, value) in [
                        (expert_spec.gate(), gate_bias),
                        (expert_spec.up(), up_bias),
                        (expert_spec.down(), down_bias),
                    ] {
                        if let (Some(parameter_spec), Some(value)) = (projection.bias(), value) {
                            parameters.push((
                                value,
                                ParameterMetadata::from_spec(
                                    parameter_spec,
                                    parameter_spec.trainable,
                                ),
                            ));
                        }
                    }
                }
            }
            _ => return Err(Error::backend("unsupported grouped parameter layout")),
        }
        Ok(NumericExpertBank {
            experts,
            parameters,
            policy,
            spec: construction_spec,
        })
    }

    fn grouped_relu2(
        spec: GroupedRelu2Spec,
        context: &NumericContext,
    ) -> Result<Self::Relu2Groups, Error> {
        spec.validate()?;
        let up = local_parameter(
            spec.up().weight(),
            vec![
                spec.group_count(),
                spec.intermediate_dimensions(),
                spec.hidden_dimensions(),
            ],
            true,
            context,
        )?;
        let down = local_parameter(
            spec.down().weight(),
            vec![
                spec.group_count(),
                spec.hidden_dimensions(),
                spec.intermediate_dimensions(),
            ],
            true,
            context,
        )?;
        Ok(NumericRelu2Groups {
            spec: spec.clone(),
            expert_count: spec.group_count() as usize,
            hidden: spec.hidden_dimensions() as usize,
            intermediate: spec.intermediate_dimensions() as usize,
            up: (
                up,
                ParameterMetadata::from_spec(spec.up().weight(), spec.up().weight().trainable),
            ),
            down: (
                down,
                ParameterMetadata::from_spec(spec.down().weight(), spec.down().weight().trainable),
            ),
        })
    }
}

#[derive(Debug, Clone)]
struct NumericCache {
    offset: i32,
    window: Option<i32>,
    keys: Option<NumericTensor>,
    values: Option<NumericTensor>,
    // In this mode updates expose only the current submission, like a paged
    // cache. The attention operation must read its retained history itself.
    attention_history: Option<(NumericTensor, NumericTensor)>,
}

impl NumericCache {
    fn new(window: Option<i32>) -> Self {
        Self {
            offset: 0,
            window,
            keys: None,
            values: None,
            attention_history: None,
        }
    }

    fn retained(&self) -> i32 {
        self.keys.as_ref().map_or(0, |keys| keys.shape[2])
    }
}

impl AttentionCache<NumericTensor> for NumericCache {
    fn uses_blockwise_attention(&self) -> bool {
        self.attention_history.is_some()
    }

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
        let submitted = context
            .cache_owned_attention
            .then(|| (keys.clone(), values.clone()));
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
        if let Some(submitted) = submitted {
            self.attention_history = Some((attention_keys, attention_values));
            Ok(submitted)
        } else {
            Ok((attention_keys, attention_values))
        }
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let request = if let Some((keys, values)) = &self.attention_history {
            assert_eq!(
                request.keys.shape[2], request.queries.shape[2],
                "only submitted keys escape a cache-owned history"
            );
            AttentionRequest {
                keys: keys.clone(),
                values: values.clone(),
                ..request
            }
        } else {
            request
        };
        request.validate()?;
        let query_offset = self.offset - request.queries.shape[2];
        if let Some(cap) = request.softcap {
            return match self.window {
                Some(window) => attention_with_softcap_windowed(
                    &request.queries,
                    &request.keys,
                    &request.values,
                    request.scale,
                    request.sinks,
                    window,
                    query_offset,
                    Some(cap),
                ),
                None => attention_with_softcap(
                    &request.queries,
                    &request.keys,
                    &request.values,
                    request.scale,
                    request.mask,
                    request.sinks,
                    Some(cap),
                ),
            };
        }
        match request.sinks {
            Some(sinks) if self.window.is_some() => attention_with_sinks_windowed(
                &request.queries,
                &request.keys,
                &request.values,
                request.scale,
                Some(sinks),
                self.window.unwrap(),
                query_offset,
            ),
            Some(sinks) => attention_with_sinks(
                &request.queries,
                &request.keys,
                &request.values,
                request.scale,
                request.mask,
                Some(sinks),
            ),
            None => attention(
                &request.queries,
                &request.keys,
                &request.values,
                request.scale,
                request.mask,
                self.window,
                query_offset,
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct NumericHybridLayerState {
    attention: Option<NumericCache>,
    compressed: Option<NumericCompressedCache>,
    pooling: Option<NumericPoolingCache>,
    fixed: BTreeMap<StateTensorRole, Option<NumericTensor>>,
    fixed_offset: i32,
    resets: usize,
}

impl NumericHybridLayerState {
    fn new(policy: &LayerCachePolicy) -> Self {
        let pooling_required = matches!(
            policy,
            LayerCachePolicy::KeyOnly { .. } | LayerCachePolicy::KeyOnlyWithFixedState { .. }
        );
        let pooling_ratios = policy
            .fixed_state()
            .iter()
            .filter_map(|tensor| match tensor.role {
                StateTensorRole::Pooling {
                    stream,
                    component: eredu_core::cache::PoolingStateComponent::Pooled,
                } => tensor.shape.iter().find_map(|dimension| match dimension {
                    eredu_core::cache::StateTensorDimension::PrefixTokensDiv(ratio) => {
                        Some((stream, i32::try_from(ratio.get()).unwrap()))
                    }
                    _ => None,
                }),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        Self {
            attention: match policy {
                LayerCachePolicy::KeyValue { attention, .. }
                | LayerCachePolicy::KeyValueWithFixedState { attention, .. }
                | LayerCachePolicy::KeyOnly { attention, .. }
                | LayerCachePolicy::KeyOnlyWithFixedState { attention, .. } => {
                    Some(NumericCache::new(attention.sliding_window_i32().unwrap()))
                }
                _ => None,
            },
            compressed: matches!(policy, LayerCachePolicy::CompressedLatentRotary { .. })
                .then(NumericCompressedCache::resident),
            pooling: pooling_required.then(|| {
                let ratios = pooling_ratios.values().copied().collect::<Vec<_>>();
                NumericPoolingCache::new(
                    policy
                        .attention()
                        .expect("pooling state requires local attention")
                        .sliding_window_i32()
                        .unwrap()
                        .unwrap_or(i32::MAX),
                    &ratios,
                )
            }),
            fixed: policy
                .fixed_state()
                .iter()
                .map(|tensor| (tensor.role, None))
                .collect(),
            fixed_offset: 0,
            resets: 0,
        }
    }
}

impl RuntimeLayerState<NumericBackend> for NumericHybridLayerState {
    type RetainedValues<'a> = std::vec::IntoIter<&'a NumericTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        let mut values = Vec::new();
        if let Some(attention) = &self.attention {
            values.extend(attention.keys.iter());
            values.extend(attention.values.iter());
        }
        if let Some(compressed) = &self.compressed {
            values.extend(
                compressed
                    .state
                    .iter()
                    .flat_map(|state| [&state.latent, &state.rotary]),
            );
        }
        if let Some(pooling) = &self.pooling {
            values.extend(pooling.local.iter());
            for stream in &pooling.streams {
                values.extend(stream.pending_values.iter());
                values.extend(stream.pending_gates.iter());
                values.extend(stream.pooled.iter());
                values.extend(stream.overlap_values.iter());
                values.extend(stream.overlap_gates.iter());
            }
        }
        values.extend(self.fixed.values().filter_map(Option::as_ref));
        values.into_iter()
    }
}

impl ResettableRuntimeLayerState<NumericBackend> for NumericHybridLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        self.resets += 1;
        if let Some(attention) = &mut self.attention {
            attention.offset = 0;
            attention.keys = None;
            attention.values = None;
        }
        if let Some(compressed) = &mut self.compressed {
            compressed.state = None;
            compressed.offset = 0;
        }
        if let Some(pooling) = &mut self.pooling {
            let ratios = pooling
                .streams
                .iter()
                .map(|stream| stream.ratio)
                .collect::<Vec<_>>();
            *pooling = NumericPoolingCache::new(pooling.window, &ratios);
        }
        self.fixed.values_mut().for_each(|value| *value = None);
        self.fixed_offset = 0;
        Ok(())
    }
}

impl RuntimeStateComponents<NumericBackend> for NumericHybridLayerState {
    fn position(&self) -> i32 {
        [
            self.attention.as_ref().map(AttentionCache::offset),
            self.compressed
                .as_ref()
                .map(CompressedAttentionCache::offset),
            self.pooling.as_ref().map(PoolingAttentionCache::offset),
            Some(self.fixed_offset),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or_default()
    }

    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<NumericTensor>, StateError> {
        self.fixed
            .get_mut(&role)
            .ok_or(StateError::UnknownComponent { role })
    }

    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        if self.attention.is_some() || self.pooling.is_some() || tokens <= 0 {
            return Err(StateError::InvalidAdvance(format!(
                "invalid fixed-state advance {tokens}"
            )));
        }
        self.fixed_offset += tokens;
        Ok(())
    }
}

impl CompressedAttentionCache<NumericTensor> for NumericHybridLayerState {
    type Checkpoint = NumericCompressedCache;

    fn offset(&self) -> i32 {
        self.position()
    }

    fn is_paged(&self) -> bool {
        self.compressed
            .as_ref()
            .is_some_and(CompressedAttentionCache::is_paged)
    }

    fn append(
        &mut self,
        state: CompressedAttentionState<NumericTensor>,
        context: &NumericContext,
    ) -> Result<CompressedAttentionView<NumericTensor>, Error> {
        self.compressed
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no compressed attention state"))?
            .append(state, context)
    }

    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &NumericContext,
        visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<NumericTensor>) -> Result<u64, Error>,
    {
        self.compressed
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no compressed attention state"))?
            .visit_blocks(query_tokens, context, visitor)
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.compressed
            .as_ref()
            .expect("compressed checkpoint requested for compressed layer")
            .clone()
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &NumericContext,
    ) -> Result<(), Error> {
        self.compressed
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no compressed attention state"))?
            .restore(checkpoint, context)
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.compressed
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no compressed attention state"))?
            .finalize()
    }

    fn clear(&mut self) -> Result<(), Error> {
        self.compressed
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no compressed attention state"))?
            .clear()
    }
}

impl PoolingAttentionCache<NumericTensor> for NumericHybridLayerState {
    type Checkpoint = NumericPoolingCache;

    fn offset(&self) -> i32 {
        self.position()
    }

    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.pooling
            .as_ref()
            .and_then(|pooling| pooling.pooling_ratio(stream))
    }

    fn append_local(
        &mut self,
        keys: NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.pooling
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .append_local(keys, context)
    }

    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.pooling
            .as_ref()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .local_mask(query_tokens, offset, context)
    }

    fn accumulate_pooling_windows(
        &mut self,
        stream: u32,
        values: NumericTensor,
        gates: NumericTensor,
        absolute_offset: i32,
        context: &NumericContext,
    ) -> Result<PoolingWindows<NumericTensor>, Error> {
        self.pooling
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .accumulate_pooling_windows(stream, values, gates, absolute_offset, context)
    }

    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: NumericTensor,
        gates: NumericTensor,
    ) -> Result<PoolingOverlap<NumericTensor>, Error> {
        self.pooling
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .replace_pooling_overlap(stream, values, gates)
    }

    fn append_pooled(
        &mut self,
        stream: u32,
        values: NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.pooling
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .append_pooled(stream, values, context)
    }

    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &NumericContext,
    ) -> Result<Option<NumericTensor>, Error> {
        self.pooling
            .as_ref()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .pooling_mask(stream, query_tokens, offset, context)
    }

    fn checkpoint(&self) -> Result<Self::Checkpoint, Error> {
        self.pooling
            .as_ref()
            .ok_or_else(|| Error::backend("pooling checkpoint requested for non-pooling layer"))
            .cloned()
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &NumericContext,
    ) -> Result<(), Error> {
        self.pooling
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .restore(checkpoint, context)
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.pooling
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .finalize()
    }

    fn clear(&mut self) -> Result<(), Error> {
        self.pooling
            .as_mut()
            .ok_or_else(|| Error::backend("layer has no pooling attention state"))?
            .clear()
    }
}

impl AttentionCache<NumericTensor> for NumericHybridLayerState {
    fn uses_blockwise_attention(&self) -> bool {
        self.attention
            .as_ref()
            .is_some_and(AttentionCache::uses_blockwise_attention)
    }

    fn offset(&self) -> i32 {
        self.position()
    }

    fn max_size(&self) -> Option<i32> {
        self.attention.as_ref().and_then(AttentionCache::max_size)
    }

    fn update_for_attention(
        &mut self,
        keys: NumericTensor,
        values: NumericTensor,
        context: &NumericContext,
    ) -> Result<(NumericTensor, NumericTensor), Error> {
        self.attention
            .as_mut()
            .ok_or_else(|| Error::backend("fixed layer has no attention state"))?
            .update_for_attention(keys, values, context)
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.attention
            .as_mut()
            .ok_or_else(|| Error::backend("fixed layer has no attention state"))?
            .attention(request, context)
    }
}

impl eredu_nn::AuxiliaryConvolutionState<NumericTensor> for NumericHybridLayerState {
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<NumericTensor>, Error> {
        self.fixed_component(StateTensorRole::Convolution { slot })
            .map_err(Error::backend_retained_source)
    }
}
