//! Multi-family numerical reference tests using a deterministic scalar backend.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
};

use eredu_architectures::{
    decoder, deepseek, gemma4, gpt_oss, inkling, kimi_linear, lfm2, llama, moshi, muse_glimmer,
    nemotron_h, qwen,
};
use eredu_core::cache::{LayerCachePolicy, PromptCacheTopology, StateTensorRole};
use eredu_core::{Completion, LayerSchedule, ParallelRankTopology, ParallelTopology, TokenFilter};
use eredu_nn::{
    reference_gated_delta_scan, reference_selective_state_space_scan, validate_parameter_topology,
    AttentionCache, AttentionMask, AttentionRequest, BlockwiseAttentionBackend,
    BlockwiseAttentionSpec, CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec,
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, ConvolutionActivation,
    EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, FusedProjectionLayout,
    FusedProjectionSegment, GatedDeltaScanInput, GatedDeltaScanOutput,
    GatedProductExpertBankOperator, GatedProductExpertBankSpec, GatedProductExpertLayout,
    GatedProductPolicy, GatedShortConvolution, GatedShortConvolutionSpec, HyperConnection,
    HyperConnectionOperator, HyperConnectionSpec, HyperConnectionState, HyperHead,
    HyperHeadOperator, HyperHeadSpec, HyperNeuralBackend, Index, IndexedAttentionInput,
    JointExpertRoutingInput, JointExpertRoutingResult, LinearOperator, LinearSpec,
    LowRankProjection, LowRankProjectionSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, PadMode, ParameterMetadata, ParameterSpec,
    ParameterVisitor, ParameterVisitorMut, Parameterized, PooledAttentionInput,
    PooledPositionInput, PoolingAttentionCache, PoolingOverlap, PoolingWindows,
    RelativeAttentionInput, Relu2ExpertBankOperator, Relu2ExpertBankSpec, RotaryOperator,
    RotaryPosition, RotarySpec, RotarySubspace, RoutedNeuralBackend, RoutingOperator,
    RoutingResult, SelectiveStateSpaceScanInput, SelectiveStateSpaceScanOutput, Tensor,
    TensorParallelExpertOutput, TopKRouterSpec, TopKRoutingSpec, VocabularyParallelRange,
};
use eredu_runtime::{
    ArchitectureParameters, CompositeLayeredTraversalHook, DeviceState, ExecutionUnitAddress,
    ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredTraversalHook,
    LayerwiseAcquireError, LayerwisePolicy, LayerwiseRuntime, LocalModelLayout, LocalTensorLayout,
    MemberSharding, ParameterGroupSpec, ParameterRole, PenaltyConfig, PredictionDirective,
    ResettableRuntimeLayerState, ResidentRuntime, ResidentUnitWindow, RoutedExpertProvider,
    RoutedExpertRequest, RoutedExpertTensorParallelOutput, RuntimeLayerState,
    RuntimeStateComponents, Sampler, SamplingBackend, SequentialDecisionDriver,
    SequentialDecisionPlan, SequentialDecisionSource, SequentialDecisionTraversal, StateError,
    SubmissionBackend, TensorPlacement, TokenDomain,
};

fn dense_linear_format() -> eredu_nn::LinearFormatSpec {
    eredu_nn::LinearFormatSpec::unscaled(eredu_nn::LinearFormat::Dense).unwrap()
}

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

    fn from_i32_slice(values: &[i32], shape: &[i32], _: &NumericContext) -> Result<Self, Error> {
        Ok(Self::new(
            shape.to_vec(),
            values.iter().map(|value| *value as f32).collect(),
        ))
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

    fn tanh(&self, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(f32::tanh))
    }

    fn maximum_scalar(&self, rhs: f32, _: &NumericContext) -> Result<Self, Error> {
        Ok(self.map(|value| value.max(rhs)))
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
        TensorPlacement::Omit
        | TensorPlacement::Rank { .. }
        | TensorPlacement::PipelineStage { .. } => Err(Error::backend(
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
    let selected = select_parameter(&parameter(spec, global_shape, norm), layout.placement())?;
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
                ),
            );
        }
    }
    Ok(layout)
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
    vocabulary_range: Option<VocabularyParallelRange>,
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
        let vocabulary = self
            .vocabulary_range
            .as_ref()
            .map_or(self.weight.shape[0] as usize, |range| {
                range.global_vocabulary
            });
        let dimensions = self.weight.shape[1] as usize;
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
                    .copy_from_slice(
                        &self.weight.data[local * dimensions..(local + 1) * dimensions],
                    );
            }
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
        let vocabulary = self
            .vocabulary_range
            .as_ref()
            .map_or(self.weight.shape[0] as usize, |range| {
                range.global_vocabulary
            });
        let dimensions = self.weight.shape[1] as usize;
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
                    .copy_from_slice(
                        &self.weight.data[local * dimensions..(local + 1) * dimensions],
                    );
            }
        }
        Ok(output)
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

#[derive(Default, Clone)]
struct NumericContext {
    sliding_attention_calls: Cell<usize>,
    local_layout: Option<Arc<LocalModelLayout>>,
}

impl NumericContext {
    fn with_local_layout(layout: LocalModelLayout) -> Self {
        Self {
            sliding_attention_calls: Cell::new(0),
            local_layout: Some(Arc::new(layout)),
        }
    }

    fn tensor_layout(&self, target: &str) -> Option<&LocalTensorLayout> {
        self.local_layout
            .as_deref()
            .and_then(|layout| layout.tensor(target))
    }
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
    group: Arc<NumericParallelGroup>,
    next_sequence: Cell<usize>,
    trace: RefCell<Vec<NumericCollectiveTrace>>,
}

impl NumericParallelContext {
    fn new(rank: usize, group: Arc<NumericParallelGroup>) -> Self {
        assert!(rank < group.size);
        Self {
            rank,
            group,
            next_sequence: Cell::new(0),
            trace: RefCell::new(Vec::new()),
        }
    }

    fn collective(
        &self,
        kind: NumericCollectiveKind,
        value: NumericTensor,
    ) -> Result<NumericTensor, Error> {
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

#[derive(Debug, Clone, Copy)]
struct NumericBackend;

#[derive(Debug)]
struct NumericCompletion;

impl Completion for NumericCompletion {
    type Error = std::convert::Infallible;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SubmissionBackend for NumericBackend {
    type Executor = NumericContext;
    type OwnedExecutor = NumericContext;
    type Completion = NumericCompletion;

    fn fork_executors(
        executor: &Self::Executor,
        count: usize,
    ) -> Result<Vec<Self::OwnedExecutor>, std::convert::Infallible> {
        Ok(vec![executor.clone(); count])
    }

    fn submit<'a, I>(_: &Self::Executor, _: I) -> Result<Self::Completion, std::convert::Infallible>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        Ok(NumericCompletion)
    }

    fn order_after(
        _: &Self::Completion,
        _: &Self::Executor,
    ) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn retain_until_complete<T: Send + 'static>(
        _: &Self::Executor,
        _: &Self::Completion,
        _: T,
    ) -> Result<(), std::convert::Infallible> {
        Ok(())
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
    const OPERATOR_CAPABILITIES: eredu_nn::NeuralOperatorCapabilities =
        eredu_nn::NeuralOperatorCapabilities::ALL;

    type Tensor = NumericTensor;
    type Linear = NumericLinear;
    type Embedding = NumericEmbedding;
    type Normalization = NumericNorm;
    type Rotary = NumericRotary;
    type ParallelContext = NumericParallelContext;

    fn linear(spec: LinearSpec, context: &NumericContext) -> Result<Self::Linear, Error> {
        let weight = local_parameter(&spec.weight, vec![spec.output, spec.input], false, context)?;
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
        })
    }

    fn embedding(spec: EmbeddingSpec, context: &NumericContext) -> Result<Self::Embedding, Error> {
        Ok(NumericEmbedding {
            weight: local_parameter(
                &spec.weight,
                vec![spec.vocabulary, spec.dimensions],
                false,
                context,
            )?,
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            vocabulary_range: None,
        })
    }

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
        };
        let local = linear.forward(input, context)?;
        parallel.collective(NumericCollectiveKind::GatherVocabulary, local)
    }

    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &NumericContext,
    ) -> Result<Self::Normalization, Error> {
        spec.validate()?;
        let dimensions = spec.dimensions;
        let epsilon = spec.epsilon;
        let (weight, metadata) = match spec.scale {
            NormalizationScale::Learned(weight) => {
                let value = local_parameter(&weight, vec![dimensions], true, context)?;
                let metadata = ParameterMetadata::from_spec(&weight, weight.trainable);
                (value, metadata)
            }
            NormalizationScale::LearnedOffset { weight, offset } => {
                let value = local_parameter(&weight, vec![dimensions], false, context)?
                    .map(|value| value + offset);
                let metadata = ParameterMetadata::from_spec(&weight, weight.trainable);
                (value, metadata)
            }
            NormalizationScale::Unit => {
                let parameter =
                    ParameterSpec::trainable("numeric.unit_norm.weight").map_err(Error::backend)?;
                (
                    NumericTensor::new(vec![dimensions], vec![1.0; dimensions as usize]),
                    ParameterMetadata::from_spec(&parameter, false),
                )
            }
        };
        Ok(NumericNorm {
            weight,
            metadata,
            epsilon,
        })
    }

    fn rotary(spec: RotarySpec, _: &NumericContext) -> Result<Self::Rotary, Error> {
        Ok(NumericRotary {
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

    fn softplus(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(|value| value.exp().ln_1p()))
    }

    fn exp(input: Self::Tensor, _: &NumericContext) -> Result<Self::Tensor, Error> {
        Ok(input.map(f32::exp))
    }

    fn l2_normalize(
        input: &Self::Tensor,
        epsilon: f32,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        let width = usize::try_from(
            *input
                .shape
                .last()
                .ok_or_else(|| Error::backend("numeric L2 normalization requires rank"))?,
        )
        .map_err(Error::backend)?;
        let mut output = input.clone();
        for (source, target) in input
            .data
            .chunks_exact(width)
            .zip(output.data.chunks_exact_mut(width))
        {
            let norm = source
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .max(epsilon)
                .sqrt();
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
        if input.shape != gate.shape || weight.shape != [*input.shape.last().unwrap()] {
            return Err(Error::backend("numeric gated RMS geometry mismatch"));
        }
        let width = *input.shape.last().unwrap() as usize;
        let groups = usize::try_from(groups).map_err(Error::backend)?;
        if groups == 0 || !width.is_multiple_of(groups) {
            return Err(Error::backend(
                "numeric gated RMS groups do not divide width",
            ));
        }
        let group_width = width / groups;
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
        if input.shape != gate.shape {
            return Err(Error::backend("numeric SiLU-gated RMS geometry mismatch"));
        }
        let width = usize::try_from(*input.shape.last().unwrap()).map_err(Error::backend)?;
        let groups = usize::try_from(groups).map_err(Error::backend)?;
        if groups == 0 || !width.is_multiple_of(groups) {
            return Err(Error::backend(
                "numeric SiLU-gated RMS groups do not divide width",
            ));
        }
        let group_width = width / groups;
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
                let key_head = head % key_heads as usize;
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

    fn joint_expert_routing(
        input: JointExpertRoutingInput<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<JointExpertRoutingResult<NumericTensor>, Error> {
        input.validate()?;
        let hidden = input.hidden.shape.last().copied().unwrap() as usize;
        let tokens = input.hidden.data.len() / hidden;
        let experts = input.weight.shape[0] as usize;
        let routed = input.routed_experts as usize;
        let shared = experts - routed;
        let top_k = input.top_k as usize;
        let global_scale = *input
            .global_scale
            .data
            .first()
            .ok_or_else(|| Error::backend("numeric global route scale is empty"))?;
        let mut routed_ids = NumericTensor::zeros(vec![tokens as i32, input.top_k]);
        let mut routed_weights = NumericTensor::zeros(vec![tokens as i32, input.top_k]);
        let mut shared_weights = NumericTensor::zeros(vec![tokens as i32, shared as i32]);
        for token in 0..tokens {
            let logits = (0..experts)
                .map(|expert| {
                    (0..hidden)
                        .map(|column| {
                            input.hidden.data[token * hidden + column]
                                * input.weight.data[expert * hidden + column]
                        })
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let mut order = (0..routed).collect::<Vec<_>>();
            order.sort_by(|left, right| {
                let score = |expert: usize| {
                    1.0 / (1.0 + (-logits[expert]).exp()) + input.correction_bias.data[expert]
                };
                score(*right)
                    .total_cmp(&score(*left))
                    .then_with(|| left.cmp(right))
            });
            order.truncate(top_k);
            let mut selected = order
                .iter()
                .map(|expert| 1.0 / (1.0 + (-logits[*expert]).exp()))
                .collect::<Vec<_>>();
            selected.extend(
                logits[routed..]
                    .iter()
                    .map(|logit| 1.0 / (1.0 + (-logit).exp())),
            );
            let denominator = selected.iter().sum::<f32>();
            let scale = input.route_scale * global_scale / denominator;
            for (slot, expert) in order.into_iter().enumerate() {
                routed_ids.data[token * top_k + slot] = expert as f32;
                routed_weights.data[token * top_k + slot] = selected[slot] * scale;
            }
            for expert in 0..shared {
                shared_weights.data[token * shared + expert] = selected[top_k + expert] * scale;
            }
        }
        Ok(JointExpertRoutingResult {
            routed_ids,
            routed_weights,
            shared_weights,
        })
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
        attention_with_sinks(
            &request.queries,
            &request.keys,
            &request.values,
            request.scale,
            request.mask,
            request.sinks,
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
        let attended = attention_with_sinks_windowed(
            &request.queries,
            &request.keys,
            &request.values,
            request.scale,
            request.sinks,
            window,
            position_offset,
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

    fn sum_parallel(
        value: Self::Tensor,
        parallel: &Self::ParallelContext,
        _: &NumericContext,
    ) -> Result<Self::Tensor, Error> {
        parallel.collective(NumericCollectiveKind::Sum, value)
    }

    fn parallel_size(parallel: &Self::ParallelContext) -> usize {
        parallel.group.size
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
    if sinks.is_none() {
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
            let key_head = head / groups;
            for query in 0..query_tokens {
                let query_base = ((b * heads + head) * query_tokens + query) * dimensions;
                let mut scores = (0..key_tokens)
                    .map(|key| {
                        let key_base = ((b * key_heads + key_head) * key_tokens + key) * dimensions;
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
    attention_with_sinks(queries, keys, values, scale, Some(&mask), sinks)
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
    input_transform: Option<(f32, NumericTensor, ParameterMetadata, bool)>,
    route_scale: Option<(NumericTensor, ParameterMetadata)>,
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
        if let Some((_, value, metadata, _)) = &self.input_transform {
            visit(metadata, value, visitor);
        }
        if let Some((value, metadata)) = &self.route_scale {
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
        if let Some((_, value, metadata, _)) = &mut self.input_transform {
            visit_mut(metadata, value, visitor);
        }
        if let Some((value, metadata)) = &mut self.route_scale {
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
        if let Some((_, metadata)) = &mut self.route_scale {
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

impl RoutingOperator<NumericTensor> for NumericRouter {
    fn route(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<RoutingResult<NumericTensor>, Error> {
        let input = self.transformed_input(input);
        let logits = self.linear.forward(&input, context)?;
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
                eredu_nn::RoutingScoring::SelectedSoftmax => row.to_vec(),
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
            let selected = order[..top_k]
                .iter()
                .map(|expert| scores[*expert])
                .collect::<Vec<_>>();
            let selected = if self.routing.scoring() == eredu_nn::RoutingScoring::SelectedSoftmax {
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
                expert_ids.data[index] = expert as f32;
                selected_scores.data[index] = selected;
                route_weights.data[index] = if self.routing.normalize_selected() {
                    selected / (selected_sum + self.routing.normalization_epsilon())
                } else {
                    selected
                } * self.routing.routed_scaling()
                    * self
                        .route_scale
                        .as_ref()
                        .map_or(1.0, |(scale, _)| scale.data[expert]);
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
        let input = self.transformed_input(input);
        let logits = self.linear.forward(&input, context)?;
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
                eredu_nn::RoutingScoring::SelectedSoftmax => row.to_vec(),
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
            if self.routing.scoring() == eredu_nn::RoutingScoring::SelectedSoftmax {
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
                let expert = expert_ids.data[index] as usize;
                route_weights.data[index] = if self.routing.normalize_selected() {
                    selected_scores.data[index] / (sum + self.routing.normalization_epsilon())
                } else {
                    selected_scores.data[index]
                } * self.routing.routed_scaling()
                    * self
                        .route_scale
                        .as_ref()
                        .map_or(1.0, |(scale, _)| scale.data[expert]);
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
    spec: GatedProductExpertBankSpec,
}

fn numeric_expert_bank_spec(
    expert_count: i32,
    hidden: i32,
    intermediate: i32,
    policy: GatedProductPolicy,
) -> GatedProductExpertBankSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    GatedProductExpertBankSpec {
        expert_count,
        input_dimensions: hidden,
        intermediate_dimensions: intermediate,
        output_dimensions: hidden,
        policy,
        layout: GatedProductExpertLayout::Packed {
            gate_up: eredu_nn::ExpertProjectionSpec {
                weight: parameter("test.experts.gate_up_proj"),
                bias: None,
                format: dense_linear_format(),
            },
            down: eredu_nn::ExpertProjectionSpec {
                weight: parameter("test.experts.down_proj"),
                bias: None,
                format: dense_linear_format(),
            },
        },
    }
}

#[derive(Debug, Clone)]
struct NumericRelu2ExpertBank {
    expert_count: usize,
    hidden: usize,
    intermediate: usize,
    up: (NumericTensor, ParameterMetadata),
    down: (NumericTensor, ParameterMetadata),
}

impl Parameterized<NumericTensor> for NumericRelu2ExpertBank {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        visit(&self.up.1, &self.up.0, visitor);
        visit(&self.down.1, &self.down.0, visitor);
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

impl Relu2ExpertBankOperator<NumericTensor> for NumericRelu2ExpertBank {
    fn forward_routed(
        &mut self,
        input: &NumericTensor,
        routes: &RoutingResult<NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let tokens = input.data.len() / self.hidden;
        let top_k = routes.expert_ids.shape[1] as usize;
        if routes.expert_ids.shape != [tokens as i32, top_k as i32]
            || routes.route_weights.shape != routes.expert_ids.shape
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
                let expert = routes.expert_ids.data[route_index] as usize;
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
                let weight = routes.route_weights.data[route_index];
                for dimension in 0..self.hidden {
                    output.data[token * self.hidden + dimension] +=
                        weight * expert_output.data[dimension];
                }
            }
        }
        Ok(output)
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        input: &NumericTensor,
        routes: &RoutingResult<NumericTensor>,
        _: usize,
        context: &NumericContext,
    ) -> Result<TensorParallelExpertOutput<NumericTensor>, Error> {
        Ok(TensorParallelExpertOutput {
            reducible: self.forward_routed(input, routes, context)?,
            post_reduce: None,
        })
    }
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

impl GatedProductExpertBankOperator<NumericTensor> for NumericExpertBank {
    fn spec(&self) -> &GatedProductExpertBankSpec {
        &self.spec
    }

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
                });
                let up = linear(&token_input, &expert.up, expert.up_bias.as_ref())?.map(|value| {
                    self.policy
                        .up_absolute_bound()
                        .map_or(value, |bound| value.clamp(-bound, bound))
                        + self.policy.up_offset()
                });
                let activated = gate.zip(&up, |left, right| left * right)?;
                let expert_output = linear(&activated, &expert.down, expert.down_bias.as_ref())?;
                let weight = routes.route_weights.data[route_index];
                for dimension in 0..hidden {
                    output.data[token * hidden + dimension] +=
                        weight * expert_output.data[dimension];
                }
            }
        }
        Ok(output)
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        input: &NumericTensor,
        routes: &RoutingResult<NumericTensor>,
        _: usize,
        context: &NumericContext,
    ) -> Result<TensorParallelExpertOutput<NumericTensor>, Error> {
        let mut local = self.clone();
        let has_down_bias = local
            .experts
            .iter()
            .any(|expert| expert.down_bias.is_some());
        for expert in &mut local.experts {
            expert.down_bias = None;
        }
        let reducible = local.forward_routed(input, routes, context)?;
        let post_reduce = if has_down_bias {
            let hidden = input.shape.last().copied().unwrap() as usize;
            let tokens = input.data.len() / hidden;
            let top_k = routes.expert_ids.shape[1] as usize;
            let mut bias = NumericTensor::zeros(input.shape.clone());
            for token in 0..tokens {
                for route in 0..top_k {
                    let route_index = token * top_k + route;
                    let expert = routes.expert_ids.data[route_index] as usize;
                    let down_bias = self
                        .experts
                        .get(expert)
                        .and_then(|expert| expert.down_bias.as_ref())
                        .ok_or_else(|| Error::backend("numeric expert down-bias mismatch"))?;
                    let weight = routes.route_weights.data[route_index];
                    for dimension in 0..hidden {
                        bias.data[token * hidden + dimension] += weight * down_bias.data[dimension];
                    }
                }
            }
            Some(bias)
        } else {
            None
        };
        Ok(TensorParallelExpertOutput {
            reducible,
            post_reduce,
        })
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

impl RoutedNeuralBackend for NumericBackend {
    type Router = NumericRouter;
    type GatedProductExpertBank = NumericExpertBank;
    type Relu2ExpertBank = NumericRelu2ExpertBank;

    fn top_k_router(spec: TopKRouterSpec, context: &NumericContext) -> Result<Self::Router, Error> {
        spec.validate()?;
        let routing = spec.routing;
        let linear = Self::linear(
            LinearSpec {
                input: spec.input_dimensions,
                output: routing.expert_count(),
                weight: spec.weight,
                bias: spec.bias,
                format: spec.format,
            },
            context,
        )?;
        let correction_bias = spec.correction_bias.map(|parameter_spec| {
            let value = parameter(&parameter_spec, vec![routing.expert_count()], true);
            let metadata = ParameterMetadata::from_spec(&parameter_spec, parameter_spec.trainable);
            (value, metadata)
        });
        let input_transform = spec.input_transform.map(|transform| {
            let value = parameter(&transform.scale, vec![spec.input_dimensions], true);
            let metadata =
                ParameterMetadata::from_spec(&transform.scale, transform.scale.trainable);
            (
                transform.epsilon,
                value,
                metadata,
                transform.inverse_sqrt_dimensions,
            )
        });
        let route_scale = spec.route_scale.map(|parameter_spec| {
            let value = parameter(&parameter_spec, vec![routing.expert_count()], true);
            let metadata = ParameterMetadata::from_spec(&parameter_spec, parameter_spec.trainable);
            (value, metadata)
        });
        Ok(NumericRouter {
            linear,
            routing,
            correction_bias,
            input_transform,
            route_scale,
        })
    }

    fn gated_product_expert_bank(
        spec: GatedProductExpertBankSpec,
        context: &NumericContext,
    ) -> Result<Self::GatedProductExpertBank, Error> {
        spec.validate()?;
        let construction_spec = spec.clone();
        let expert_count = spec.expert_count as usize;
        let hidden = spec.input_dimensions;
        let intermediate = spec.intermediate_dimensions;
        let policy = spec.policy;
        let mut experts = Vec::with_capacity(expert_count);
        let mut parameters = Vec::new();
        match spec.layout {
            GatedProductExpertLayout::Packed { gate_up, down } => {
                let packed_gate_up = local_parameter(
                    &gate_up.weight,
                    vec![spec.expert_count, 2 * intermediate, hidden],
                    false,
                    context,
                )?;
                let packed_down = local_parameter(
                    &down.weight,
                    vec![spec.expert_count, spec.output_dimensions, intermediate],
                    false,
                    context,
                )?;
                let packed_gate_up_bias = gate_up
                    .bias
                    .as_ref()
                    .map(|bias| {
                        local_parameter(
                            bias,
                            vec![spec.expert_count, 2 * intermediate],
                            false,
                            context,
                        )
                    })
                    .transpose()?;
                let packed_down_bias = down
                    .bias
                    .as_ref()
                    .map(|bias| {
                        local_parameter(
                            bias,
                            vec![spec.expert_count, spec.output_dimensions],
                            false,
                            context,
                        )
                    })
                    .transpose()?;
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
                            vec![spec.output_dimensions, intermediate],
                            packed_down.data[down_start..down_start + down_per_expert].to_vec(),
                        ),
                        down_bias: packed_down_bias.as_ref().map(|bias| {
                            let width = spec.output_dimensions as usize;
                            NumericTensor::new(
                                vec![spec.output_dimensions],
                                bias.data[expert * width..(expert + 1) * width].to_vec(),
                            )
                        }),
                    });
                }
                parameters.push((
                    packed_gate_up,
                    ParameterMetadata::from_spec(&gate_up.weight, gate_up.weight.trainable),
                ));
                if let (Some(parameter_spec), Some(value)) =
                    (gate_up.bias.as_ref(), packed_gate_up_bias)
                {
                    parameters.push((
                        value,
                        ParameterMetadata::from_spec(parameter_spec, parameter_spec.trainable),
                    ));
                }
                if let (Some(parameter_spec), Some(value)) = (down.bias.as_ref(), packed_down_bias)
                {
                    parameters.push((
                        value,
                        ParameterMetadata::from_spec(parameter_spec, parameter_spec.trainable),
                    ));
                }
                parameters.push((
                    packed_down,
                    ParameterMetadata::from_spec(&down.weight, down.weight.trainable),
                ));
            }
            GatedProductExpertLayout::Independent(specs) => {
                for expert_spec in specs {
                    let gate = local_parameter(
                        &expert_spec.gate.weight,
                        vec![intermediate, hidden],
                        false,
                        context,
                    )?;
                    let up = local_parameter(
                        &expert_spec.up.weight,
                        vec![intermediate, hidden],
                        false,
                        context,
                    )?;
                    let down = local_parameter(
                        &expert_spec.down.weight,
                        vec![spec.output_dimensions, intermediate],
                        false,
                        context,
                    )?;
                    let gate_bias = expert_spec
                        .gate
                        .bias
                        .as_ref()
                        .map(|bias| local_parameter(bias, vec![intermediate], false, context))
                        .transpose()?;
                    let up_bias = expert_spec
                        .up
                        .bias
                        .as_ref()
                        .map(|bias| local_parameter(bias, vec![intermediate], false, context))
                        .transpose()?;
                    let down_bias = expert_spec
                        .down
                        .bias
                        .as_ref()
                        .map(|bias| {
                            local_parameter(bias, vec![spec.output_dimensions], false, context)
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
                    for (projection, value) in [
                        (&expert_spec.gate, gate_bias),
                        (&expert_spec.up, up_bias),
                        (&expert_spec.down, down_bias),
                    ] {
                        if let (Some(parameter_spec), Some(value)) =
                            (projection.bias.as_ref(), value)
                        {
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
        }
        Ok(NumericExpertBank {
            experts,
            parameters,
            policy,
            spec: construction_spec,
        })
    }

    fn relu2_expert_bank(
        spec: Relu2ExpertBankSpec,
        context: &NumericContext,
    ) -> Result<Self::Relu2ExpertBank, Error> {
        spec.validate()?;
        let up = local_parameter(
            &spec.up.weight,
            vec![
                spec.expert_count,
                spec.intermediate_dimensions,
                spec.hidden_dimensions,
            ],
            true,
            context,
        )?;
        let down = local_parameter(
            &spec.down.weight,
            vec![
                spec.expert_count,
                spec.hidden_dimensions,
                spec.intermediate_dimensions,
            ],
            true,
            context,
        )?;
        Ok(NumericRelu2ExpertBank {
            expert_count: spec.expert_count as usize,
            hidden: spec.hidden_dimensions as usize,
            intermediate: spec.intermediate_dimensions as usize,
            up: (
                up,
                ParameterMetadata::from_spec(&spec.up.weight, spec.up.weight.trainable),
            ),
            down: (
                down,
                ParameterMetadata::from_spec(&spec.down.weight, spec.down.weight.trainable),
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
        request: AttentionRequest<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        request.validate()?;
        let query_offset = self.offset - request.queries.shape[2];
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
    fixed: BTreeMap<StateTensorRole, Option<NumericTensor>>,
    fixed_offset: i32,
    resets: usize,
}

impl NumericHybridLayerState {
    fn new(policy: &LayerCachePolicy) -> Self {
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
        self.fixed.values_mut().for_each(|value| *value = None);
        self.fixed_offset = 0;
        Ok(())
    }
}

impl RuntimeStateComponents<NumericBackend> for NumericHybridLayerState {
    fn position(&self) -> i32 {
        self.attention
            .as_ref()
            .map(AttentionCache::offset)
            .or_else(|| {
                self.compressed
                    .as_ref()
                    .map(CompressedAttentionCache::offset)
            })
            .unwrap_or(self.fixed_offset)
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
        if self.attention.is_some() || tokens <= 0 {
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

impl AttentionCache<NumericTensor> for NumericHybridLayerState {
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
            .map_err(Error::backend)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum StreamPolicyEvent {
    Begin,
    Acquire(usize, usize, usize),
    Complete(usize, usize, usize),
    Finish,
}

struct RebuiltUnitLease<U>(U);

impl<U> std::ops::Deref for RebuiltUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<U> std::ops::DerefMut for RebuiltUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Default)]
struct RebuildingUnitPolicy {
    events: Vec<StreamPolicyEvent>,
    fail_acquire: Option<usize>,
}

impl RebuildingUnitPolicy {
    fn failing_at(ordinal: usize) -> Self {
        Self {
            events: Vec::new(),
            fail_acquire: Some(ordinal),
        }
    }
}

impl<U> LayerwisePolicy<NumericBackend, U> for RebuildingUnitPolicy {
    type Lease = RebuiltUnitLease<U>;
    type Error = &'static str;

    fn begin(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Self::Error> {
        self.events.push(StreamPolicyEvent::Begin);
        Ok(())
    }

    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&NumericContext) -> Result<U, E>,
    {
        self.events.push(StreamPolicyEvent::Acquire(
            ordinal,
            address.group(),
            address.index(),
        ));
        if self.fail_acquire == Some(ordinal) {
            return Err(LayerwiseAcquireError::Policy(
                "injected dense-stream acquisition failure",
            ));
        }
        build(context)
            .map(RebuiltUnitLease)
            .map_err(LayerwiseAcquireError::Architecture)
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        _: Self::Lease,
        _: &'a NumericTensor,
        _: StateValues,
        _: ContextValues,
        _: &NumericContext,
    ) -> Result<(), Self::Error>
    where
        NumericTensor: 'a,
        StateValues: Iterator<Item = &'a NumericTensor>,
        ContextValues: Iterator<Item = &'a NumericTensor>,
    {
        self.events.push(StreamPolicyEvent::Complete(
            ordinal,
            address.group(),
            address.index(),
        ));
        Ok(())
    }

    fn finish(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Self::Error> {
        self.events.push(StreamPolicyEvent::Finish);
        Ok(())
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

fn numeric_moshi_config() -> moshi::MoshiConfig {
    moshi::MoshiConfig::from_json(
        r#"{
            "model_type":"moshi", "dim":4, "text_card":5,
            "n_q":2, "dep_q":1, "generated_audio_codebooks":1, "card":4,
            "num_heads":1, "num_layers":1, "dim_feedforward":6,
            "causal":true, "context":3, "max_period":10000.0,
            "positional_embedding":"rope", "depformer_dim":4,
            "depformer_dim_feedforward":6, "depformer_num_heads":1,
            "depformer_num_layers":1, "depformer_context":2,
            "depformer_max_period":10000.0, "depformer_pos_emb":"none",
            "delays":[0,0,1]
        }"#,
    )
    .unwrap()
}

fn explicit_numeric_linear(name: &str, output: i32, input: i32, values: &[f32]) -> NumericLinear {
    let spec = ParameterSpec::trainable(name).unwrap();
    NumericLinear {
        weight: NumericTensor::new([output, input], values.to_vec()),
        weight_metadata: ParameterMetadata::from_spec(&spec, spec.trainable),
        bias: None,
    }
}

#[test]
fn fused_and_split_qkv_are_numerically_identical_through_cached_attention() {
    let context = NumericContext::default();
    let query = [0.5, -0.25, 0.75, 0.125];
    let key = [-0.5, 0.375, 0.25, 0.625];
    let value = [0.125, 0.5, -0.75, 0.25];
    let output = [1.0, 0.0, 0.0, 1.0];
    let mut split = decoder::Attention::<NumericBackend>::from_parts(
        1,
        1,
        2,
        explicit_numeric_linear("split.q", 2, 2, &query),
        explicit_numeric_linear("split.k", 2, 2, &key),
        explicit_numeric_linear("split.v", 2, 2, &value),
        explicit_numeric_linear("split.out", 2, 2, &output),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut fused = decoder::Attention::<NumericBackend>::from_parts(
        1,
        1,
        2,
        explicit_numeric_linear("unused.q", 2, 2, &query),
        explicit_numeric_linear("unused.k", 2, 2, &key),
        explicit_numeric_linear("unused.v", 2, 2, &value),
        explicit_numeric_linear("fused.out", 2, 2, &output),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let fused_weight = query
        .into_iter()
        .chain(key)
        .chain(value)
        .collect::<Vec<_>>();
    fused.input_projection =
        decoder::AttentionInputProjection::Fused(decoder::FusedAttentionProjection {
            projection: explicit_numeric_linear("fused.qkv", 6, 2, &fused_weight),
            layout: FusedProjectionLayout::new([
                FusedProjectionSegment::new("query", 2).unwrap(),
                FusedProjectionSegment::new("key", 2).unwrap(),
                FusedProjectionSegment::new("value", 2).unwrap(),
            ])
            .unwrap(),
        });
    let mut split_cache = NumericCache::new(None);
    let mut fused_cache = NumericCache::new(None);
    for hidden in [
        NumericTensor::new([1, 2, 2], vec![0.25, -0.5, 0.75, 0.125]),
        NumericTensor::new([1, 1, 2], vec![-0.25, 0.875]),
    ] {
        let split_output = split
            .forward(
                decoder::AttentionInput {
                    hidden: &hidden,
                    mask: None,
                    cache: Some(&mut split_cache),
                    allow_sliding_prefill: false,
                    rotary_position: None,
                },
                &context,
            )
            .unwrap();
        let fused_output = fused
            .forward(
                decoder::AttentionInput {
                    hidden: &hidden,
                    mask: None,
                    cache: Some(&mut fused_cache),
                    allow_sliding_prefill: false,
                    rotary_position: None,
                },
                &context,
            )
            .unwrap();
        assert_tensor_close(&fused_output, &split_output, "fused QKV attention");
    }
    assert_eq!(split_cache.offset, 3);
    assert_eq!(fused_cache.offset, 3);
}

#[test]
fn fused_and_split_gate_up_are_numerically_identical_through_silu_and_output() {
    let context = NumericContext::default();
    let input = NumericTensor::new([1, 2, 2], vec![0.25, -0.5, 0.75, 0.125]);
    let gate_weight = [0.5, -0.25, 0.75, 0.125, -0.5, 0.375];
    let up_weight = [0.125, 0.5, -0.75, 0.25, 0.625, -0.125];
    let down_weight = [0.5, -0.25, 0.75, -0.125, 0.625, 0.25];
    let gate = explicit_numeric_linear("split.gate", 3, 2, &gate_weight)
        .forward(&input, &context)
        .unwrap();
    let up = explicit_numeric_linear("split.up", 3, 2, &up_weight)
        .forward(&input, &context)
        .unwrap();
    let split_hidden =
        NumericBackend::gated_product(gate, up, GatedProductPolicy::ordinary_silu(), &context)
            .unwrap();

    let fused_weight = gate_weight.into_iter().chain(up_weight).collect::<Vec<_>>();
    let fused_projected = explicit_numeric_linear("fused.gate_up", 6, 2, &fused_weight)
        .forward(&input, &context)
        .unwrap();
    let layout = FusedProjectionLayout::new([
        FusedProjectionSegment::new("gate", 3).unwrap(),
        FusedProjectionSegment::new("up", 3).unwrap(),
    ])
    .unwrap();
    let mut fused_components = layout
        .split(&fused_projected, &context)
        .unwrap()
        .into_iter();
    let fused_hidden = NumericBackend::gated_product(
        fused_components.next().unwrap(),
        fused_components.next().unwrap(),
        GatedProductPolicy::ordinary_silu(),
        &context,
    )
    .unwrap();
    assert!(fused_components.next().is_none());
    assert_tensor_close(&fused_hidden, &split_hidden, "fused gate/up product");

    let mut split_down = explicit_numeric_linear("split.down", 2, 3, &down_weight);
    let mut fused_down = explicit_numeric_linear("fused.down", 2, 3, &down_weight);
    let split_output = split_down.forward(&split_hidden, &context).unwrap();
    let fused_output = fused_down.forward(&fused_hidden, &context).unwrap();
    assert_tensor_close(&fused_output, &split_output, "fused gate/up output");
}

#[test]
fn zero_sentinel_and_ordered_multi_table_sum_have_exact_scalar_results() {
    let context = NumericContext::default();
    let parameter = ParameterSpec::trainable("sentinel.weight").unwrap();
    let mut sentinel = NumericEmbedding {
        weight: NumericTensor::new([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        metadata: ParameterMetadata::from_spec(&parameter, parameter.trainable),
        vocabulary_range: None,
    };
    let looked_up = sentinel
        .lookup(
            &NumericTensor::new([1, 3], vec![-1.0, 0.0, 2.0]),
            EmbeddingLookupPolicy::ZeroSentinel(-1),
            &context,
        )
        .unwrap();
    assert_eq!(looked_up.data, [0.0, 0.0, 1.0, 2.0, 5.0, 6.0]);
    for invalid in [-2.0, 3.0] {
        assert!(sentinel
            .lookup(
                &NumericTensor::new([1, 1], vec![invalid]),
                EmbeddingLookupPolicy::ZeroSentinel(-1),
                &context,
            )
            .is_err());
    }

    let specs = [
        decoder::NamedEmbeddingSpec {
            name: "first".into(),
            embedding: EmbeddingSpec {
                vocabulary: 3,
                dimensions: 2,
                weight: ParameterSpec::trainable("first.weight").unwrap(),
                format: dense_linear_format(),
            },
            lookup: EmbeddingLookupPolicy::ZeroSentinel(-1),
        },
        decoder::NamedEmbeddingSpec {
            name: "second".into(),
            embedding: EmbeddingSpec {
                vocabulary: 3,
                dimensions: 2,
                weight: ParameterSpec::trainable("second.weight").unwrap(),
                format: dense_linear_format(),
            },
            lookup: EmbeddingLookupPolicy::ZeroSentinel(-1),
        },
    ];
    let mut sum = decoder::MultiTableEmbedding::<NumericBackend>::new(specs, &context).unwrap();
    sum.tables[0].embedding.weight =
        NumericTensor::new([3, 2], vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    sum.tables[1].embedding.weight =
        NumericTensor::new([3, 2], vec![0.5, 5.0, 1.5, 15.0, 2.5, 25.0]);
    let first = NumericTensor::new([1, 2], vec![0.0, -1.0]);
    let second = NumericTensor::new([1, 2], vec![2.0, 1.0]);
    let actual = sum.forward(&[&first, &second], &context).unwrap();
    assert_eq!(sum.names().collect::<Vec<_>>(), ["first", "second"]);
    assert_eq!(actual.shape, [1, 2, 2]);
    assert_eq!(actual.data, [3.5, 35.0, 1.5, 15.0]);
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct StatefulNumericSampler {
    calls: usize,
    invalid: bool,
}

impl Sampler<NumericBackend> for StatefulNumericSampler {
    fn sample(
        &mut self,
        logits: &NumericTensor,
        temperature: f32,
        random: Option<&mut i32>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.calls += 1;
        if self.invalid {
            if let Some(random) = random {
                *random += 1;
            }
            return Ok(NumericTensor::token_ids(&[usize::MAX / 2]));
        }
        NumericBackend::sample_raw(logits, temperature, random, context)
    }
}

#[derive(Debug)]
struct NumericMoshiFrame {
    sources: Vec<SequentialDecisionSource>,
    tokens: Vec<usize>,
    diagnostic_shapes: Vec<Vec<i32>>,
    text_shape: Vec<i32>,
    previous_depth_token: Vec<f32>,
    samplers: Vec<StatefulNumericSampler>,
    random: Option<i32>,
}

#[derive(Default)]
struct NumericMoshiObservationCapture {
    order: Vec<String>,
    values: BTreeMap<String, NumericTensor>,
}

impl NumericMoshiObservationCapture {
    fn capture(
        &mut self,
        point: moshi::ObservationPoint,
        value: &NumericTensor,
    ) -> Result<(), Error> {
        let path = point.path();
        self.order.push(path.clone());
        if self.values.insert(path.clone(), value.clone()).is_some() {
            return Err(Error::backend(format!(
                "numeric Moshi observation {path} was captured twice"
            )));
        }
        Ok(())
    }
}

impl LayeredTraversalHook<NumericBackend, moshi::ForwardContext<NumericTensor>, Error>
    for NumericMoshiObservationCapture
{
    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        _: usize,
        value: &NumericTensor,
        _: &mut moshi::ForwardContext<NumericTensor>,
        _: &NumericContext,
    ) -> Result<eredu_runtime::LayeredUnitAction, Error> {
        if (group, index) == (0, 0) {
            self.capture(moshi::ObservationPoint::TemporalInput, value)?;
        }
        Ok(eredu_runtime::LayeredUnitAction::Execute)
    }

    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &NumericTensor,
        _: &mut moshi::ForwardContext<NumericTensor>,
        _: &NumericContext,
    ) -> Result<(), Error> {
        match group {
            0 => self.capture(
                moshi::ObservationPoint::TemporalLayer { layer: index },
                value,
            ),
            1 => self.capture(
                moshi::ObservationPoint::DepthSliceLogits { slice: index },
                value,
            ),
            _ => Err(Error::backend(format!(
                "numeric Moshi observed unknown execution group {group}"
            ))),
        }
    }

    fn after_group(
        &mut self,
        group: usize,
        _: &NumericTensor,
        forward: &mut moshi::ForwardContext<NumericTensor>,
        _: &NumericContext,
    ) -> Result<(), Error> {
        if group == 0 {
            let logits = forward
                .text_logits()
                .ok_or_else(|| Error::backend("numeric Moshi text logits are unavailable"))?
                .clone();
            self.capture(moshi::ObservationPoint::TextLogits, &logits)?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_numeric_moshi_frame(
    runtime: &mut ResidentRuntime<
        moshi::LayeredModel<NumericBackend>,
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
    >,
    state: &mut DeviceState<NumericBackend, NumericHybridLayerState>,
    config: &moshi::MoshiConfig,
    text_id: usize,
    audio_ids: [usize; 2],
    directives: [PredictionDirective<NumericTensor>; 2],
    retain_diagnostics: bool,
    samplers: Vec<StatefulNumericSampler>,
    temperatures: Vec<f32>,
    random: Option<i32>,
    context: &NumericContext,
) -> Result<NumericMoshiFrame, String> {
    let plan = SequentialDecisionPlan::new(directives, retain_diagnostics, true)
        .map_err(|error| error.to_string())?;
    let mut driver = SequentialDecisionDriver::new(plan, samplers, temperatures, random)
        .map_err(|error| error.to_string())?;
    let mut boundary = moshi::DecisionBoundary::new(config).map_err(|error| error.to_string())?;
    let text = NumericTensor::token_ids(&[text_id]);
    let audio_values = audio_ids.map(|id| NumericTensor::token_ids(&[id]));
    let audio = audio_values.iter().collect::<Vec<_>>();
    let (text_logits, forward) = {
        let mut traversal = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                state,
                context,
                &mut traversal,
            )
            .map_err(|error| error.to_string())?
    };
    driver.finish().map_err(|error| error.to_string())?;
    let sources = driver
        .decisions()
        .iter()
        .map(|decision| decision.source())
        .collect();
    let tokens = driver
        .decisions()
        .iter()
        .map(|decision| decision.token().data[0] as usize)
        .collect();
    let diagnostic_shapes = driver
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.logits().shape.clone())
        .collect();
    let previous_depth_token = forward
        .previous_depth_token()
        .ok_or_else(|| "numeric Moshi frame has no accepted depth token".to_string())?
        .data
        .clone();
    let (samplers, random) = driver
        .finish_into_sampling_state()
        .map_err(|error| error.to_string())?;
    Ok(NumericMoshiFrame {
        sources,
        tokens,
        diagnostic_shapes,
        text_shape: text_logits.shape,
        previous_depth_token,
        samplers,
        random,
    })
}

#[test]
fn moshi_numeric_autoregressive_decisions_continue_one_cache_and_roll_back_failures() {
    let config = numeric_moshi_config();
    let context = NumericContext::default();
    let layout = moshi::state_layout(&config).unwrap();
    let mut state =
        DeviceState::<NumericBackend, NumericHybridLayerState>::create(layout, |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
    let architecture =
        moshi::LayeredModel::<NumericBackend>::new(config.clone(), &context).unwrap();
    let mut runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let mut samplers = vec![
        StatefulNumericSampler {
            calls: 0,
            invalid: false,
        };
        2
    ];
    let mut random = None;

    let greedy = execute_numeric_moshi_frame(
        &mut runtime,
        &mut state,
        &config,
        1,
        [2, 3],
        [PredictionDirective::Sample, PredictionDirective::Sample],
        true,
        samplers,
        vec![0.0; 2],
        random,
        &context,
    )
    .unwrap();
    assert_eq!(greedy.sources, [SequentialDecisionSource::Sampled; 2]);
    assert_eq!(greedy.tokens, [4, 3]);
    assert_eq!(greedy.diagnostic_shapes, [vec![1, 1, 5], vec![1, 1, 4]]);
    assert_eq!(greedy.text_shape, [1, 1, 5]);
    assert_eq!(greedy.previous_depth_token, [3.0]);
    assert_eq!(state.as_ref()[0].position(), 1);
    assert_eq!((state.as_ref()[0].resets, state.as_ref()[1].resets), (0, 1));
    samplers = greedy.samplers;
    random = Some(10);

    let seeded = execute_numeric_moshi_frame(
        &mut runtime,
        &mut state,
        &config,
        2,
        [1, 0],
        [PredictionDirective::Sample, PredictionDirective::Sample],
        true,
        samplers,
        vec![0.7; 2],
        random,
        &context,
    )
    .unwrap();
    assert_eq!(seeded.sources, [SequentialDecisionSource::Sampled; 2]);
    assert_eq!(seeded.tokens, [3, 2]);
    assert_eq!(seeded.diagnostic_shapes, [vec![1, 1, 5], vec![1, 1, 4]]);
    assert_eq!(seeded.random, Some(12));
    assert_eq!(
        seeded
            .samplers
            .iter()
            .map(|sampler| sampler.calls)
            .collect::<Vec<_>>(),
        [2, 2]
    );
    assert_eq!(state.as_ref()[0].position(), 2);
    assert_eq!(state.as_ref()[1].position(), 1);
    assert_eq!(state.as_ref()[1].resets, 2);
    samplers = seeded.samplers;
    random = seeded.random;

    let forced = execute_numeric_moshi_frame(
        &mut runtime,
        &mut state,
        &config,
        3,
        [0, 1],
        [
            PredictionDirective::Force(NumericTensor::token_ids(&[1])),
            PredictionDirective::Force(NumericTensor::token_ids(&[3])),
        ],
        false,
        samplers,
        vec![0.7; 2],
        random,
        &context,
    )
    .unwrap();
    assert_eq!(
        forced.sources,
        [
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::ForcedTailSkipped,
        ]
    );
    assert_eq!(forced.tokens, [1, 3]);
    assert!(forced.diagnostic_shapes.is_empty());
    assert_eq!(forced.previous_depth_token, [3.0]);
    assert_eq!(forced.random, Some(12));
    assert_eq!(
        forced
            .samplers
            .iter()
            .map(|sampler| sampler.calls)
            .collect::<Vec<_>>(),
        [2, 2]
    );
    assert_eq!(state.as_ref()[0].position(), 3);
    assert_eq!(state.as_ref()[1].position(), 0);
    assert_eq!(state.as_ref()[1].resets, 3);
    samplers = forced.samplers;
    random = forced.random;

    let canonical_state = state.clone();
    let canonical_samplers = samplers.clone();
    let canonical_random = random;
    let mut rejected_state = state.clone();
    let mut rejected_samplers = samplers.clone();
    rejected_samplers[0].invalid = true;
    let error = execute_numeric_moshi_frame(
        &mut runtime,
        &mut rejected_state,
        &config,
        4,
        [1, 2],
        [PredictionDirective::Sample, PredictionDirective::Sample],
        true,
        rejected_samplers,
        vec![0.7; 2],
        random,
        &context,
    )
    .unwrap_err();
    assert!(error.contains("outside its decision domain"));
    assert_eq!(
        state
            .as_ref()
            .iter()
            .map(|layer| (layer.position(), layer.resets))
            .collect::<Vec<_>>(),
        canonical_state
            .as_ref()
            .iter()
            .map(|layer| (layer.position(), layer.resets))
            .collect::<Vec<_>>()
    );
    assert_eq!(samplers, canonical_samplers);
    assert_eq!(random, canonical_random);

    let partial = execute_numeric_moshi_frame(
        &mut runtime,
        &mut state,
        &config,
        4,
        [1, 2],
        [
            PredictionDirective::Force(NumericTensor::token_ids(&[2])),
            PredictionDirective::Sample,
        ],
        true,
        samplers,
        vec![0.7; 2],
        random,
        &context,
    )
    .unwrap();
    assert_eq!(
        partial.sources,
        [
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Sampled,
        ]
    );
    assert_eq!(partial.tokens, [2, 2]);
    assert_eq!(partial.diagnostic_shapes, [vec![1, 1, 5], vec![1, 1, 4]]);
    assert_eq!(partial.previous_depth_token, [2.0]);
    assert_eq!(partial.random, Some(13));
    assert_eq!(
        partial
            .samplers
            .iter()
            .map(|sampler| sampler.calls)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    assert_eq!(state.as_ref()[0].position(), 4);
    assert_eq!(state.as_ref()[1].position(), 1);
    assert_eq!((state.as_ref()[0].resets, state.as_ref()[1].resets), (0, 4));
}

#[test]
fn moshi_numeric_teacher_forced_logits_are_exact_across_continuation() {
    let config = numeric_moshi_config();
    let context = NumericContext::default();
    let layout = moshi::state_layout(&config).unwrap();
    let mut state =
        DeviceState::<NumericBackend, NumericHybridLayerState>::create(layout, |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
    let architecture =
        moshi::LayeredModel::<NumericBackend>::new(config.clone(), &context).unwrap();
    let mut runtime = ResidentRuntime::new(architecture, &context).unwrap();

    let text = NumericTensor::token_ids(&[1]);
    let audio_values = [
        NumericTensor::token_ids(&[2]),
        NumericTensor::token_ids(&[3]),
    ];
    let audio = audio_values.iter().collect::<Vec<_>>();
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(NumericTensor::token_ids(&[2])),
            PredictionDirective::Force(NumericTensor::token_ids(&[1])),
        ],
        true,
        false,
    )
    .unwrap();
    let mut driver =
        SequentialDecisionDriver::new(plan, vec![NumericSampler; 2], vec![0.0; 2], None).unwrap();
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    let (first_text, _) = {
        let mut traversal = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &context,
                &mut traversal,
            )
            .unwrap()
    };
    driver.finish().unwrap();
    let first_audio = driver.diagnostics()[1].logits().clone();

    let text = NumericTensor::token_ids(&[2]);
    let audio_values = [
        NumericTensor::token_ids(&[1]),
        NumericTensor::token_ids(&[0]),
    ];
    let audio = audio_values.iter().collect::<Vec<_>>();
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(NumericTensor::token_ids(&[3])),
            PredictionDirective::Force(NumericTensor::token_ids(&[2])),
        ],
        true,
        false,
    )
    .unwrap();
    let mut driver =
        SequentialDecisionDriver::new(plan, vec![NumericSampler; 2], vec![0.0; 2], None).unwrap();
    let (second_text, observations) = {
        let decisions = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        let capture = NumericMoshiObservationCapture::default();
        let mut traversal = CompositeLayeredTraversalHook::new(decisions, capture);
        let (text_logits, _) = runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &context,
                &mut traversal,
            )
            .unwrap();
        let (_, observations) = traversal.into_parts();
        (text_logits, observations)
    };
    driver.finish().unwrap();
    let second_audio = driver.diagnostics()[1].logits().clone();

    assert_tensor_close(
        &first_text,
        &NumericTensor::new(
            vec![1, 1, 5],
            vec![
                -0.06355038,
                -0.033500876,
                -0.06604099,
                -0.06899042,
                -0.008079363,
            ],
        ),
        "first Moshi text logits",
    );
    assert_tensor_close(
        &first_audio,
        &NumericTensor::new(
            vec![1, 1, 4],
            vec![-0.0025060514, -0.0005943443, 0.0011221073, -0.0010868304],
        ),
        "first Moshi audio logits",
    );
    assert_tensor_close(
        &second_text,
        &NumericTensor::new(
            vec![1, 1, 5],
            vec![
                0.007951072,
                -0.020567559,
                0.017594583,
                0.07032819,
                0.010250918,
            ],
        ),
        "continued Moshi text logits",
    );
    assert_tensor_close(
        &second_audio,
        &NumericTensor::new(
            vec![1, 1, 4],
            vec![-0.001689149, -0.0018042361, 0.0036688647, 0.0020801048],
        ),
        "continued Moshi audio logits",
    );
    assert_eq!(
        observations.order.as_slice(),
        moshi::observation_points(&config)
            .into_iter()
            .map(moshi::ObservationPoint::path)
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_tensor_close(
        observations.values.get("temporal.input").unwrap(),
        &NumericTensor::new(
            vec![1, 1, 4],
            vec![0.012679999, 0.058, 0.064959995, -0.029040001],
        ),
        "continued Moshi temporal input",
    );
    assert_tensor_close(
        observations
            .values
            .get("transformer.layers.0.output")
            .unwrap(),
        &NumericTensor::new(
            vec![1, 1, 4],
            vec![0.012538117, 0.059593383, 0.06597139, -0.030165773],
        ),
        "continued Moshi temporal layer output",
    );
    assert_tensor_close(
        observations.values.get("text_linear.logits").unwrap(),
        &NumericTensor::new(
            vec![1, 1, 5],
            vec![
                0.007951072,
                -0.020567559,
                0.017594583,
                0.07032819,
                0.010250918,
            ],
        ),
        "observed continued Moshi text logits",
    );
    assert_tensor_close(
        observations
            .values
            .get("depformer.slices.0.logits")
            .unwrap(),
        &NumericTensor::new(
            vec![1, 1, 4],
            vec![-0.001689149, -0.0018042361, 0.0036688647, 0.0020801048],
        ),
        "observed continued Moshi depth-slice logits",
    );
    assert_eq!(state.as_ref()[0].position(), 2);
    assert_eq!(state.as_ref()[1].position(), 1);
}

#[test]
fn moshi_numeric_rejects_out_of_range_tokens_before_cache_mutation() {
    let config = numeric_moshi_config();
    let context = NumericContext::default();
    let layout = moshi::state_layout(&config).unwrap();
    let mut state =
        DeviceState::<NumericBackend, NumericHybridLayerState>::create(layout, |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
    let architecture = moshi::LayeredModel::<NumericBackend>::new(config, &context).unwrap();
    let mut runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let invalid_text = NumericTensor::token_ids(&[99]);
    let audio_values = [
        NumericTensor::token_ids(&[1]),
        NumericTensor::token_ids(&[2]),
    ];
    let audio = audio_values.iter().collect::<Vec<_>>();
    let error = runtime
        .forward(
            moshi::Input {
                text: &invalid_text,
                audio: &audio,
                mask: None,
            },
            &mut state,
            &context,
        )
        .expect_err("out-of-range token must fail");
    assert!(error.to_string().contains("embedding token is invalid"));
    assert_eq!(state.as_ref()[0].position(), 0);
    assert_eq!(state.as_ref()[1].position(), 0);
}

#[derive(Debug, Clone)]
struct SinkDecoderConfig(qwen::ModelArgs);

impl decoder::Config for SinkDecoderConfig {
    fn model_family(&self) -> &'static str {
        "sink_decoder_fixture"
    }

    fn model_identity(&self) -> &str {
        decoder::Config::model_identity(&self.0)
    }
    fn architecture_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "reference_sink_decoder",
            [
                ("base", decoder::Config::architecture_fingerprint(&self.0)),
                ("learned_attention_sinks", "true".into()),
            ],
        )
    }
    fn parameter_root(&self) -> &str {
        decoder::Config::parameter_root(&self.0)
    }
    fn validate_config(&self) -> Result<(), Error> {
        decoder::Config::validate_config(&self.0)
    }
    fn hidden_size(&self) -> i32 {
        decoder::Config::hidden_size(&self.0)
    }
    fn num_hidden_layers(&self) -> i32 {
        decoder::Config::num_hidden_layers(&self.0)
    }
    fn intermediate_size(&self) -> i32 {
        decoder::Config::intermediate_size(&self.0)
    }
    fn num_attention_heads(&self) -> i32 {
        decoder::Config::num_attention_heads(&self.0)
    }
    fn num_key_value_heads(&self) -> i32 {
        decoder::Config::num_key_value_heads(&self.0)
    }
    fn head_dim(&self) -> i32 {
        decoder::Config::head_dim(&self.0)
    }
    fn rms_norm_epsilon(&self) -> f32 {
        decoder::Config::rms_norm_epsilon(&self.0)
    }
    fn vocabulary_size(&self) -> i32 {
        decoder::Config::vocabulary_size(&self.0)
    }
    fn attention_bias(&self, projection: decoder::AttentionProjection) -> bool {
        decoder::Config::attention_bias(&self.0, projection)
    }
    fn learned_attention_sinks(&self) -> bool {
        true
    }
    fn query_key_norm_epsilon(&self) -> Option<f32> {
        decoder::Config::query_key_norm_epsilon(&self.0)
    }
    fn mlp_bias(&self) -> bool {
        decoder::Config::mlp_bias(&self.0)
    }
    fn tie_word_embeddings(&self) -> bool {
        decoder::Config::tie_word_embeddings(&self.0)
    }
    fn attention_schedule(&self) -> &eredu_core::LayerSchedule<eredu_core::AttentionPolicy> {
        decoder::Config::attention_schedule(&self.0)
    }
    fn weight_quantization(&self, name: &str) -> Option<eredu_checkpoint::WeightQuantization> {
        decoder::Config::weight_quantization(&self.0, name)
    }
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        decoder::Config::rotary_spec(&self.0, dimensions)
    }
}

#[test]
fn shared_decoder_constructs_optional_trainable_attention_sinks() {
    let args = qwen::model_args_from_config_value(&config("qwen2", false)).unwrap();
    let context = NumericContext::default();
    let ordinary = decoder::Attention::<NumericBackend>::new(&args, 0, &context).unwrap();
    assert!(ordinary.sinks.is_none());

    let sink_aware =
        decoder::Attention::<NumericBackend>::new(&SinkDecoderConfig(args), 0, &context).unwrap();
    assert_eq!(sink_aware.sinks.as_ref().unwrap().as_ref().shape, [2]);
    assert!(validate_parameter_topology::<NumericTensor, _>(&sink_aware)
        .unwrap()
        .iter()
        .any(
            |parameter| parameter.id.as_str() == "model.layers.0.self_attn.sinks"
                && parameter.trainable
        ));
}

#[test]
fn sink_aware_request_matches_cached_full_and_sliding_scalar_references() {
    let context = NumericContext::default();
    let queries = NumericTensor::zeros(vec![1, 1, 3, 1]);
    let keys = NumericTensor::zeros(vec![1, 1, 3, 1]);
    let values = NumericTensor::new(vec![1, 1, 3, 1], vec![10.0, 20.0, 30.0]);
    let sinks = NumericTensor::zeros(vec![1]);
    let mask = NumericBackend::causal_mask(3, 0, None, &context).unwrap();

    let uncached = NumericBackend::attention_with_sinks(
        AttentionRequest {
            queries: queries.clone(),
            keys: keys.clone(),
            values: values.clone(),
            scale: 1.0,
            mask: Some(&mask),
            sinks: Some(&sinks),
        },
        &context,
    )
    .unwrap();
    assert_tensor_close(
        &uncached,
        &NumericTensor::new(vec![1, 1, 3, 1], vec![5.0, 10.0, 15.0]),
        "uncached attention sinks",
    );

    let mut full = NumericCache::new(None);
    let (cached_keys, cached_values) = full
        .update_for_attention(keys.clone(), values.clone(), &context)
        .unwrap();
    let cached = full
        .attention(
            AttentionRequest {
                queries: queries.clone(),
                keys: cached_keys,
                values: cached_values,
                scale: 1.0,
                mask: Some(&mask),
                sinks: Some(&sinks),
            },
            &context,
        )
        .unwrap();
    assert_tensor_close(&cached, &uncached, "cached attention sinks");

    let sliding = NumericBackend::sliding_window_attention_with_sinks(
        AttentionRequest {
            queries,
            keys,
            values,
            scale: 1.0,
            mask: None,
            sinks: Some(&sinks),
        },
        2,
        0,
        &context,
    )
    .unwrap();
    assert_tensor_close(
        &sliding,
        &NumericTensor::new(vec![1, 3, 1], vec![5.0, 10.0, 50.0 / 3.0]),
        "sliding attention sinks",
    );

    let malformed_sinks = NumericTensor::zeros(vec![2]);
    let malformed = AttentionRequest {
        queries: NumericTensor::zeros(vec![1, 1, 1, 1]),
        keys: NumericTensor::zeros(vec![1, 1, 1, 1]),
        values: NumericTensor::zeros(vec![1, 1, 1, 1]),
        scale: 1.0,
        mask: None,
        sinks: Some(&malformed_sinks),
    };
    assert!(malformed.validate().is_err());
}

struct ForwardResult {
    prefill: Vec<f32>,
    decode: Vec<f32>,
    retained: Vec<i32>,
}

fn forward(model_type: &str, tied: bool) -> Result<ForwardResult, Error> {
    let args =
        qwen::model_args_from_config_value(&config(model_type, tied)).map_err(Error::backend)?;
    let context = NumericContext::default();
    let architecture = qwen::RoutedLayeredModel::<NumericBackend>::new(args.clone(), &context)?;
    let units = (0..usize::try_from(args.num_hidden_layers).map_err(Error::backend)?)
        .map(|layer| architecture.construct_unit(layer, &context))
        .collect::<Result<Vec<_>, _>>()?;
    let mut state =
        DeviceState::<NumericBackend, _>::create(qwen::state_layout(&args)?, |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })?;
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));

    let prefill_tokens = NumericTensor::token_ids(&[1, 4, 2]);
    let prefill_logits = runtime
        .forward(
            decoder::LayeredInput {
                tokens: &prefill_tokens,
                mask: None,
            },
            &mut state,
            &context,
        )
        .map_err(Error::backend)?;

    let decode_tokens = NumericTensor::token_ids(&[3]);
    let decode_logits = runtime
        .forward(
            decoder::LayeredInput {
                tokens: &decode_tokens,
                mask: None,
            },
            &mut state,
            &context,
        )
        .map_err(Error::backend)?;
    let cache_state = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            state
                .layer(layer)
                .map_err(Error::backend)?
                .attention
                .as_ref()
                .map(NumericCache::retained)
                .ok_or_else(|| Error::backend(format!("Qwen layer {layer} has no attention cache")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ForwardResult {
        prefill: prefill_logits.data,
        decode: decode_logits.data,
        retained: cache_state,
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
    }
}

#[test]
fn dense_qwen_construction_rejects_moe_configuration() {
    let args = qwen::model_args_from_config_value(&config("qwen3_moe", false)).unwrap();
    let context = NumericContext::default();
    let errors = [
        qwen::new_block::<NumericBackend>(&args, 0, &context)
            .err()
            .unwrap(),
        qwen::LayeredModel::<NumericBackend>::new(args, &context)
            .err()
            .unwrap(),
    ];
    for error in errors {
        assert!(error
            .to_string()
            .contains("dense Qwen construction does not accept a routed MoE configuration"));
    }
}

#[test]
fn qwen_expert_realization_owns_assignment_and_tp_local_bank_geometry() {
    let mut value = config("qwen3_moe", false);
    value["num_attention_heads"] = 4.into();
    value["num_key_value_heads"] = 2.into();
    value["head_dim"] = 2.into();
    let args = qwen::model_args_from_config_value(&value).unwrap();
    let context = NumericContext::default();
    let seed = qwen::RoutedLayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = qwen::static_parallel_parameter_groups::<NumericBackend>(
        &seed.static_modules().embeddings,
        &seed.static_modules().norm,
        seed.static_modules().lm_head.as_ref(),
        &args.parameter_root,
    )
    .unwrap();
    let unit = seed.construct_unit(0, &context).unwrap();
    groups.extend(qwen::routed_layer_parallel_parameter_groups(&unit, &args, 0).unwrap());
    let layout = numeric_local_layout(&groups, 2, 1).unwrap();
    let geometry = qwen::local_geometry(&args, &layout).unwrap();
    let local_context = NumericContext::with_local_layout(layout);
    let architecture =
        qwen::RoutedLayeredModel::<NumericBackend>::new_parallel(args, geometry, &local_context)
            .unwrap();
    let topology = ParallelTopology::new(2, 1, 2, 1).unwrap();
    let rank = ParallelRankTopology::new(topology, 3).unwrap();
    let plan = qwen::expert_realization_plan(&architecture, rank)
        .unwrap()
        .unwrap();

    assert_eq!(plan.global_expert_count(), 4);
    assert_eq!(plan.owners(), [0, 0, 1, 1]);
    assert_eq!(plan.local_global_expert_ids(), [2, 3]);
    let bank = plan.unit_spec("text_decoder", 0).unwrap();
    assert_eq!(bank.expert_count, 2);
    assert_eq!(bank.intermediate_dimensions, 3);
}

#[test]
fn qwen_prompt_snapshot_reopens_with_global_identity_and_continues_exactly() {
    let mut value = config("qwen3", false);
    value["num_hidden_layers"] = 2.into();
    let args = qwen::model_args_from_config_value(&value).unwrap();
    let context = NumericContext::default();
    let layout = qwen::state_layout(&args).unwrap();
    let new_state = || {
        DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let new_runtime = || {
        ResidentRuntime::new(
            qwen::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
            &context,
        )
        .unwrap()
    };

    let prefix = NumericTensor::token_ids(&[1, 4, 2]);
    let continuation = NumericTensor::token_ids(&[3]);
    let all = NumericTensor::token_ids(&[1, 4, 2, 3]);
    let mut persisted_state = new_state();
    let mut writer = new_runtime();
    writer
        .forward(
            decoder::LayeredInput {
                tokens: &prefix,
                mask: None,
            },
            &mut persisted_state,
            &context,
        )
        .unwrap();
    assert_eq!(persisted_state.layer(0).unwrap().position(), 3);
    assert_eq!(persisted_state.layer(1).unwrap().position(), 3);

    // A reopened runtime owns a new architecture instance while the restored
    // state is the exact backend-neutral prompt snapshot.
    let mut reopened_state = persisted_state.clone();
    let mut reopened = new_runtime();
    let reopened_logits = reopened
        .forward(
            decoder::LayeredInput {
                tokens: &continuation,
                mask: None,
            },
            &mut reopened_state,
            &context,
        )
        .unwrap();

    let mut uninterrupted_state = new_state();
    let mut uninterrupted = new_runtime();
    let uninterrupted_logits = uninterrupted
        .forward(
            decoder::LayeredInput {
                tokens: &all,
                mask: None,
            },
            &mut uninterrupted_state,
            &context,
        )
        .unwrap()
        .axis_slice(1, 3, 4);
    assert_tensor_close(
        &reopened_logits,
        &uninterrupted_logits,
        "reopened Qwen prompt continuation",
    );
    assert_eq!(reopened_state.layer(0).unwrap().position(), 4);
    assert_eq!(reopened_state.layer(1).unwrap().position(), 4);

    // A pipeline rank that owns only the second global layer records global
    // coordinates and distributed rank identity, never a local zero-based
    // shadow identity.
    let local_layout = eredu_runtime::StateLayout::new(
        LayerSchedule::new(1, vec![layout.layers().get(1).unwrap().clone()]).unwrap(),
    )
    .unwrap();
    let topology = PromptCacheTopology {
        pipeline: Some((2, 1)),
        tensor_parallel: Some((2, 0)),
        ..PromptCacheTopology::default()
    };
    let model_identity = qwen::state_identity(&args, &local_layout, 1, topology.clone()).unwrap();
    let prompt_identity = model_identity.prompt_cache_identity(&local_layout).unwrap();
    assert_eq!(prompt_identity.global_layer_start, 1);
    assert_eq!(prompt_identity.global_layer_end, 2);
    assert_eq!(prompt_identity.topology, topology);
    assert_eq!(prompt_identity.layer_layout.len(), 1);

    // Identity and input failures are rejected before mutating the restored
    // state. This is the portable failure-atomicity boundary used before any
    // persistent tensors are materialized.
    let mut malformed_identity = model_identity;
    malformed_identity.global_layer_start = 2;
    assert!(malformed_identity
        .prompt_cache_identity(&local_layout)
        .is_err());
    let positions = [
        reopened_state.layer(0).unwrap().position(),
        reopened_state.layer(1).unwrap().position(),
    ];
    let invalid = NumericTensor::token_ids(&[args.vocab_size as usize]);
    assert!(reopened
        .forward(
            decoder::LayeredInput {
                tokens: &invalid,
                mask: None,
            },
            &mut reopened_state,
            &context,
        )
        .is_err());
    assert_eq!(
        [
            reopened_state.layer(0).unwrap().position(),
            reopened_state.layer(1).unwrap().position(),
        ],
        positions,
    );
}

fn expected_stream_events(addresses: &[(usize, usize)]) -> Vec<StreamPolicyEvent> {
    let mut events = vec![StreamPolicyEvent::Begin];
    for (ordinal, &(group, index)) in addresses.iter().enumerate() {
        events.push(StreamPolicyEvent::Acquire(ordinal, group, index));
        events.push(StreamPolicyEvent::Complete(ordinal, group, index));
    }
    events.push(StreamPolicyEvent::Finish);
    events
}

#[test]
fn real_decoder_and_hybrid_models_match_resident_and_dense_streamed_traversal() {
    let context = NumericContext::default();

    let mut qwen_value = config("qwen3", false);
    qwen_value["num_hidden_layers"] = 2.into();
    let qwen_args = qwen::model_args_from_config_value(&qwen_value).unwrap();
    let qwen_layout = qwen::state_layout(&qwen_args).unwrap();
    let qwen_state = || {
        DeviceState::<NumericBackend, _>::create(qwen_layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let mut qwen_resident_state = qwen_state();
    let mut qwen_streamed_state = qwen_state();
    let mut qwen_resident = ResidentRuntime::new(
        qwen::LayeredModel::<NumericBackend>::new(qwen_args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut qwen_streamed = LayerwiseRuntime::new(
        qwen::LayeredModel::<NumericBackend>::new(qwen_args, &context).unwrap(),
        RebuildingUnitPolicy::default(),
    );
    let qwen_tokens = NumericTensor::token_ids(&[1, 4, 2]);
    let resident_logits = qwen_resident
        .forward(
            decoder::LayeredInput {
                tokens: &qwen_tokens,
                mask: None,
            },
            &mut qwen_resident_state,
            &context,
        )
        .unwrap();
    let streamed_logits = qwen_streamed
        .forward(
            decoder::LayeredInput {
                tokens: &qwen_tokens,
                mask: None,
            },
            &mut qwen_streamed_state,
            &context,
        )
        .unwrap();
    assert_tensor_close(
        &streamed_logits,
        &resident_logits,
        "Qwen resident/dense-stream logits",
    );
    assert_eq!(
        qwen_streamed.policy().events,
        expected_stream_events(&[(0, 0), (0, 1)])
    );
    for layer in 0..2 {
        assert_eq!(
            qwen_streamed_state.layer(layer).unwrap().position(),
            qwen_resident_state.layer(layer).unwrap().position()
        );
    }

    let lfm_args = lfm2::model_args_from_config_value(&serde_json::json!({
        "model_type":"lfm2", "vocab_size":17, "hidden_size":8,
        "intermediate_size":10, "num_hidden_layers":2,
        "num_attention_heads":4, "num_key_value_heads":2,
        "max_position_embeddings":32, "layer_types":["conv","full_attention"],
        "conv_L_cache":3, "block_multiple_of":2,
        "block_ffn_dim_multiplier":1.0, "block_auto_adjust_ff_dim":true,
        "tie_word_embeddings":false
    }))
    .unwrap();
    let lfm_layout = lfm2::state_layout(&lfm_args).unwrap();
    let lfm_state = || {
        DeviceState::<NumericBackend, _>::create(lfm_layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let mut lfm_resident_state = lfm_state();
    let mut lfm_streamed_state = lfm_state();
    let mut lfm_resident = ResidentRuntime::new(
        lfm2::LayeredModel::<NumericBackend>::new(lfm_args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut lfm_streamed = LayerwiseRuntime::new(
        lfm2::LayeredModel::<NumericBackend>::new(lfm_args, &context).unwrap(),
        RebuildingUnitPolicy::default(),
    );
    let lfm_tokens = NumericTensor::token_ids(&[2, 5, 7]);
    let resident_logits = lfm_resident
        .forward(
            decoder::LayeredInput {
                tokens: &lfm_tokens,
                mask: None,
            },
            &mut lfm_resident_state,
            &context,
        )
        .unwrap();
    let streamed_logits = lfm_streamed
        .forward(
            decoder::LayeredInput {
                tokens: &lfm_tokens,
                mask: None,
            },
            &mut lfm_streamed_state,
            &context,
        )
        .unwrap();
    assert_tensor_close(
        &streamed_logits,
        &resident_logits,
        "LFM2 resident/dense-stream logits",
    );
    assert_eq!(
        lfm_streamed.policy().events,
        expected_stream_events(&[(0, 0), (0, 1)])
    );
    for layer in 0..2 {
        assert_eq!(
            lfm_streamed_state.layer(layer).unwrap().position(),
            lfm_resident_state.layer(layer).unwrap().position()
        );
    }
}

#[test]
fn dense_stream_acquisition_failure_is_atomic_before_first_real_unit() {
    let mut value = config("qwen3", false);
    value["num_hidden_layers"] = 2.into();
    let args = qwen::model_args_from_config_value(&value).unwrap();
    let context = NumericContext::default();
    let mut state = DeviceState::<NumericBackend, _>::create(
        qwen::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut runtime = LayerwiseRuntime::new(
        qwen::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        RebuildingUnitPolicy::failing_at(0),
    );
    let tokens = NumericTensor::token_ids(&[1, 2]);
    assert!(runtime
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut state,
            &context,
        )
        .is_err());
    assert_eq!(state.layer(0).unwrap().position(), 0);
    assert_eq!(state.layer(1).unwrap().position(), 0);
    assert_eq!(
        runtime.policy().events,
        [
            StreamPolicyEvent::Begin,
            StreamPolicyEvent::Acquire(0, 0, 0)
        ]
    );
}

#[test]
fn neutral_gpt_oss_prefill_decode_is_chunk_invariant() {
    let args = gpt_oss::model_args_from_config_value(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 17,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001,
        "sliding_window": 2,
        "max_position_embeddings": 64,
        "rope_theta": 150000.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "quantization_config": {"quant_method": "mxfp4"},
        "swiglu_limit": 7.0
    }))
    .unwrap();
    let context = NumericContext::default();
    let layout = gpt_oss::state_layout(&args).unwrap();
    let mut whole_state = DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let mut chunked_state = DeviceState::<NumericBackend, _>::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let mut whole = ResidentRuntime::new(
        gpt_oss::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut chunked = ResidentRuntime::new(
        gpt_oss::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        &context,
    )
    .unwrap();

    let whole_tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let whole_logits = whole
        .forward(
            decoder::LayeredInput {
                tokens: &whole_tokens,
                mask: None,
            },
            &mut whole_state,
            &context,
        )
        .unwrap();
    let prefix_tokens = NumericTensor::token_ids(&[1, 2]);
    let prefix_logits = chunked
        .forward(
            decoder::LayeredInput {
                tokens: &prefix_tokens,
                mask: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let decode_tokens = NumericTensor::token_ids(&[3]);
    let decode_logits = chunked
        .forward(
            decoder::LayeredInput {
                tokens: &decode_tokens,
                mask: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();

    let final_whole = whole_logits
        .axis_slice(1, 2, 3)
        .reshape(&[1, 1, 17], &context)
        .unwrap();
    let whole_prefix = whole_logits.axis_slice(1, 0, 2);
    assert_tensor_close(&prefix_logits, &whole_prefix, "neutral GPT-OSS prefix");
    assert_tensor_close(&decode_logits, &final_whole, "neutral GPT-OSS decode");
    assert!(checksum(&decode_logits.data)
        .iter()
        .all(|value| value.is_finite()));
    for layer in 0..2 {
        assert_eq!(AttentionCache::offset(whole_state.layer(layer).unwrap()), 3);
        assert_eq!(
            AttentionCache::offset(chunked_state.layer(layer).unwrap()),
            3
        );
    }
}

#[test]
fn gemma4_sparse_shared_kv_prefill_decode_is_chunk_invariant() {
    let args = gemma4::ModelArgs::from_hf_json(
        br#"{
            "model_type":"gemma4_unified",
            "hidden_size":8,
            "num_hidden_layers":4,
            "intermediate_size":12,
            "num_attention_heads":2,
            "rms_norm_eps":0.00001,
            "vocab_size":19,
            "num_key_value_heads":1,
            "max_position_embeddings":64,
            "head_dim":4,
            "attention_k_eq_v":true,
            "num_kv_shared_layers":1,
            "layer_types":["sliding_attention","full_attention","full_attention","full_attention"],
            "sliding_window":4,
            "enable_moe_block":true,
            "num_experts":3,
            "top_k_experts":2,
            "moe_intermediate_size":5,
            "final_logit_softcapping":7.0
        }"#,
    )
    .unwrap();
    let layout = gemma4::state_layout(&args).unwrap();
    let family = gemma4::FamilyConfig {
        model_type: args.model_type.clone(),
        text: args,
        vision: None,
        image_token_id: None,
        video_token_id: None,
        audio: None,
        audio_token_id: None,
    };
    let context = NumericContext::default();
    let mut whole_state = DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let mut chunked_state = DeviceState::<NumericBackend, _>::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let mut whole = ResidentRuntime::new(
        gemma4::LayeredModel::<NumericBackend>::new(family.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut chunked = ResidentRuntime::new(
        gemma4::LayeredModel::<NumericBackend>::new(family, &context).unwrap(),
        &context,
    )
    .unwrap();
    let whole_tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let whole_parts = [gemma4::DecoderInputPart::Text(&whole_tokens)];
    let expected = whole
        .forward(
            gemma4::ModelInput {
                parts: &whole_parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            &mut whole_state,
            &context,
        )
        .unwrap();
    let prefix_tokens = NumericTensor::token_ids(&[1, 2]);
    let prefix_parts = [gemma4::DecoderInputPart::Text(&prefix_tokens)];
    let prefix = chunked
        .forward(
            gemma4::ModelInput {
                parts: &prefix_parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let decode_tokens = NumericTensor::token_ids(&[3]);
    let decode_parts = [gemma4::DecoderInputPart::Text(&decode_tokens)];
    let decode = chunked
        .forward(
            gemma4::ModelInput {
                parts: &decode_parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let actual = NumericTensor::concatenate(&[prefix, decode], 1, &context).unwrap();
    assert_tensor_close(&actual, &expected, "Gemma 4 sparse shared-KV logits");
    assert_eq!(actual.shape, [1, 3, 19]);
    assert_eq!(chunked_state.layer(0).unwrap().position(), 3);
    assert_eq!(chunked_state.layer(1).unwrap().position(), 3);
    assert_eq!(chunked_state.layer(2).unwrap().position(), 3);
    assert_eq!(chunked_state.layer(3).unwrap().position(), 0);
}

#[test]
fn gemma4_tp2_text_matches_replicated_composite_graph() {
    let text = gemma4::ModelArgs::from_hf_json(
        br#"{
        "model_type":"gemma4_unified","hidden_size":8,"num_hidden_layers":2,
        "intermediate_size":10,"num_attention_heads":2,"rms_norm_eps":0.00001,
        "vocab_size":7,"num_key_value_heads":2,"max_position_embeddings":64,"head_dim":4,
        "attention_k_eq_v":false,"num_kv_shared_layers":0,
        "layer_types":["sliding_attention","full_attention"],"sliding_window":4,
        "enable_moe_block":false,"final_logit_softcapping":7.0
    }"#,
    )
    .unwrap();
    let family = gemma4::FamilyConfig {
        model_type: text.model_type.clone(),
        text,
        vision: None,
        image_token_id: None,
        video_token_id: None,
        audio: None,
        audio_token_id: None,
    };
    let context = NumericContext::default();
    let architecture =
        gemma4::LayeredModel::<NumericBackend>::new(family.clone(), &context).unwrap();
    let mut groups = gemma4::static_parameter_groups(&family.text).unwrap();
    for layer in 0..2 {
        groups.extend(gemma4::layer_parameter_groups(&family.text, layer).unwrap());
    }
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let parts = [gemma4::DecoderInputPart::Text(&tokens)];
    let expected = expected_runtime
        .forward(
            gemma4::ModelInput {
                parts: &parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();
    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = gemma4::local_geometry(&family, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = gemma4::LayeredModel::<NumericBackend>::new_parallel(
        family.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = [(2, 0), (2, 1)]
        .into_iter()
        .map(|(group, index)| {
            <gemma4::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, group, index, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_parts = [gemma4::DecoderInputPart::Text(&tokens)];
    let tp1_logits = tp1_runtime
        .forward_parallel(
            gemma4::ModelInput {
                parts: &tp1_parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Gemma4 TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 2, "Gemma4 TP1 state");
    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let family = family.clone();
                let tokens = tokens.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = gemma4::local_geometry(&family, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = gemma4::LayeredModel::<NumericBackend>::new_parallel(
                        family, geometry, &context,
                    )
                    .unwrap();
                    let units = [(2, 0), (2, 1)]
                        .into_iter()
                        .map(|(group, index)| {
                            <gemma4::LayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(
                                &architecture, group, index, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let parts = [gemma4::DecoderInputPart::Text(&tokens)];
                    let logits = runtime
                        .forward_parallel(
                            gemma4::ModelInput {
                                parts: &parts,
                                vision: None,
                                audio: None,
                                per_layer_tokens: None,
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace())
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(&outputs[0].0, &expected, "Gemma4 TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "Gemma4 TP2 rank 1 logits");
    assert_eq!(outputs[0].1.len(), 6);
    assert_eq!(outputs[1].1.len(), 6);
}

#[test]
fn gemma4_tp2_ordered_vision_audio_text_matches_replicated_multimodal_graph() {
    let family = gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "model_type":"gemma4_unified", "tie_word_embeddings":false,
            "image_token_id":5, "audio_token_id":6,
            "text_config":{
                "model_type":"gemma4_text", "hidden_size":8,
                "num_hidden_layers":2, "intermediate_size":10,
                "num_attention_heads":2, "num_key_value_heads":2, "head_dim":4,
                "rms_norm_eps":0.00001, "vocab_size":7,
                "max_position_embeddings":64, "attention_k_eq_v":false,
                "num_kv_shared_layers":0,
                "layer_types":["sliding_attention","full_attention"],
                "sliding_window":4, "enable_moe_block":false,
                "final_logit_softcapping":7.0
            },
            "vision_config":{
                "hidden_size":8, "intermediate_size":10,
                "num_hidden_layers":1, "num_attention_heads":2,
                "num_key_value_heads":2, "head_dim":4, "patch_size":2,
                "pooling_kernel_size":2, "position_embedding_size":2,
                "rms_norm_eps":0.00001
            },
            "audio_config":{
                "hidden_size":8, "num_hidden_layers":1,
                "num_attention_heads":2, "output_proj_dims":8,
                "conv_kernel_size":3, "attention_chunk_size":4,
                "attention_context_left":5, "attention_context_right":0,
                "attention_invalid_logits_value":-1000000000.0,
                "attention_logit_cap":50.0, "residual_weight":0.5,
                "rms_norm_eps":0.00001, "subsampling_conv_channels":[4,8]
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        gemma4::LayeredModel::<NumericBackend>::new(family.clone(), &context).unwrap();
    let groups = architecture
        .parameter_description(&context)
        .unwrap()
        .groups()
        .iter()
        .map(|owned| owned.group().clone())
        .collect::<Vec<_>>();
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let text_before = NumericTensor::token_ids(&[0]);
    let image_tokens = NumericTensor::token_ids(&[5]);
    let audio_tokens = NumericTensor::token_ids(&[6]);
    let text_after = NumericTensor::token_ids(&[4]);
    let patches = NumericTensor::new(
        [1, 4, 12],
        (0..48).map(|index| (index as f32 - 24.0) / 100.0).collect(),
    );
    let position_ids = NumericTensor::new([1, 4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let position_valid = NumericTensor::new([1, 4, 1], vec![1.0; 4]);
    let vision_mask = NumericTensor::zeros([1, 1, 1, 4]);
    let grid_extents = [(2, 2)];
    let audio_features = NumericTensor::new(
        [1, 4, 128],
        (0..512)
            .map(|index| (index as f32 % 17.0 - 8.0) / 100.0)
            .collect(),
    );
    let audio_input_mask = NumericTensor::new([1, 4, 1], vec![1.0; 4]);
    let audio_first_mask = NumericTensor::new([1, 2, 1, 1], vec![1.0; 2]);
    let audio_valid = [1];
    let parts = [
        gemma4::DecoderInputPart::Text(&text_before),
        gemma4::DecoderInputPart::Image(&image_tokens),
        gemma4::DecoderInputPart::Audio(&audio_tokens),
        gemma4::DecoderInputPart::Text(&text_after),
    ];
    let input = || gemma4::ModelInput {
        parts: &parts,
        vision: Some(gemma4::VisionInput {
            patches: &patches,
            position_ids: &position_ids,
            position_valid: &position_valid,
            key_mask: &vision_mask,
            grid_extents: &grid_extents,
        }),
        audio: Some(gemma4::AudioInput {
            features: &audio_features,
            input_mask: &audio_input_mask,
            first_stage_mask: &audio_first_mask,
            valid_subsampled_frames: &audio_valid,
        }),
        per_layer_tokens: None,
        mask: None,
    };
    let expected = expected_runtime
        .forward(input(), &mut expected_state, &context)
        .unwrap();
    assert_eq!(expected.shape, [1, 4, 7]);

    let addresses = [(0, 0), (1, 0), (2, 0), (2, 1)];
    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = gemma4::local_geometry(&family, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = gemma4::LayeredModel::<NumericBackend>::new_parallel(
        family.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = addresses
        .into_iter()
        .map(|(group, index)| {
            <gemma4::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, group, index, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(input(), &mut tp1_state, &tp1_parallel, &tp1_context)
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Gemma4 multimodal TP1 logits");
    assert_state_exact(
        &tp1_state,
        &expected_state,
        2,
        "Gemma4 multimodal TP1 state",
    );

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let family = family.clone();
                let text_before = text_before.clone();
                let image_tokens = image_tokens.clone();
                let audio_tokens = audio_tokens.clone();
                let text_after = text_after.clone();
                let patches = patches.clone();
                let position_ids = position_ids.clone();
                let position_valid = position_valid.clone();
                let vision_mask = vision_mask.clone();
                let audio_features = audio_features.clone();
                let audio_input_mask = audio_input_mask.clone();
                let audio_first_mask = audio_first_mask.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = gemma4::local_geometry(&family, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = gemma4::LayeredModel::<NumericBackend>::new_parallel(
                        family, geometry, &context,
                    )
                    .unwrap();
                    let units = addresses
                        .into_iter()
                        .map(|(group, index)| {
                            <gemma4::LayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(
                                &architecture, group, index, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let grid_extents = [(2, 2)];
                    let audio_valid = [1];
                    let parts = [
                        gemma4::DecoderInputPart::Text(&text_before),
                        gemma4::DecoderInputPart::Image(&image_tokens),
                        gemma4::DecoderInputPart::Audio(&audio_tokens),
                        gemma4::DecoderInputPart::Text(&text_after),
                    ];
                    let logits = runtime
                        .forward_parallel(
                            gemma4::ModelInput {
                                parts: &parts,
                                vision: Some(gemma4::VisionInput {
                                    patches: &patches,
                                    position_ids: &position_ids,
                                    position_valid: &position_valid,
                                    key_mask: &vision_mask,
                                    grid_extents: &grid_extents,
                                }),
                                audio: Some(gemma4::AudioInput {
                                    features: &audio_features,
                                    input_mask: &audio_input_mask,
                                    first_stage_mask: &audio_first_mask,
                                    valid_subsampled_frames: &audio_valid,
                                }),
                                per_layer_tokens: None,
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (rank, (logits, trace, state)) in outputs.iter().enumerate() {
        assert_tensor_close(
            logits,
            &expected,
            &format!("Gemma4 multimodal TP2 rank {rank} logits"),
        );
        assert_eq!(state.as_ref()[0].position(), 4);
        assert_eq!(state.as_ref()[1].position(), 4);
        assert_eq!(
            trace.last().unwrap().kind,
            NumericCollectiveKind::GatherVocabulary
        );
        assert_eq!(trace.last().unwrap().output_shape, [1, 4, 7]);
    }
}

#[test]
fn gemma4_external_assistant_draft_traversal_and_rollback_are_backend_neutral() {
    let config = gemma4::AssistantConfig::from_json(
        br#"{
          "model_type":"gemma4_assistant", "backbone_hidden_size":8,
          "use_ordered_embeddings":false, "tie_word_embeddings":false,
          "block_size":4,
          "text_config":{
            "model_type":"gemma4_text", "hidden_size":8,
            "num_hidden_layers":2, "intermediate_size":10,
            "num_attention_heads":2, "num_key_value_heads":2, "head_dim":4,
            "rms_norm_eps":0.00001, "vocab_size":7,
            "max_position_embeddings":64, "tie_word_embeddings":false,
            "attention_k_eq_v":false,
            "layer_types":["full_attention","sliding_attention"],
            "sliding_window":4
          }
        }"#,
    )
    .unwrap();
    let context = NumericContext::default();
    let mut assistant = gemma4::Assistant::<NumericBackend>::new(config, &context).unwrap();
    assert_eq!(assistant.max_proposals(), 3);
    let shared = std::collections::HashMap::from([
        (
            eredu_core::AttentionPolicy::Full,
            (
                NumericTensor::zeros([1, 2, 3, 4]),
                NumericTensor::zeros([1, 2, 3, 4]),
            ),
        ),
        (
            eredu_core::AttentionPolicy::Sliding {
                window: std::num::NonZeroU32::new(4).unwrap(),
            },
            (
                NumericTensor::zeros([1, 2, 3, 4]),
                NumericTensor::zeros([1, 2, 3, 4]),
            ),
        ),
    ]);
    let hidden = NumericTensor::new([1, 1, 8], (0..8).map(|value| value as f32 / 8.0).collect());
    let mut canonical = assistant.begin_round(shared, 2, hidden);
    let first_embedding = NumericTensor::new(
        [1, 1, 8],
        (0..8).map(|value| (value + 1) as f32 / 9.0).collect(),
    );
    let first = assistant
        .draft_step::<NumericHybridLayerState>(&first_embedding, &mut canonical, &context)
        .unwrap();
    assert_eq!(first.shape, [1, 1, 7]);
    assert_eq!(canonical.kv_offset, 3);
    assert_eq!(canonical.hidden.shape, [1, 1, 8]);

    let checkpoint = canonical.clone();
    let mut transaction = eredu_runtime::DraftStateTransaction::fork(&checkpoint);
    let second_embedding = NumericTensor::new(
        [1, 1, 8],
        (0..8).map(|value| (value + 2) as f32 / 10.0).collect(),
    );
    let second = assistant
        .draft_step::<NumericHybridLayerState>(&second_embedding, transaction.draft_mut(), &context)
        .unwrap();
    assert_eq!(second.shape, [1, 1, 7]);
    assert_eq!(transaction.draft_mut().kv_offset, 4);
    transaction.rollback(&mut canonical);
    assert_eq!(canonical.kv_offset, 3);
    assert_tensor_exact(
        &canonical.hidden,
        &checkpoint.hidden,
        "Gemma assistant rollback",
    );
}

#[test]
fn inkling_fixed_state_and_sparse_decode_are_chunk_invariant() {
    let args = inkling::ModelArgs::from_hf_json(
        br#"{
          "model_type":"inkling_mm_model",
          "text_config":{
            "hidden_size":8,"num_hidden_layers":2,"vocab_size":19,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "sliding_window_size":4,
            "layer_types":["sliding_attention","full_attention"],
            "mlp_layer_types":["dense","moe"],"sconv_kernel_size":3,
            "d_rel":2,"rel_extent":8,"intermediate_size":12,
            "dense_intermediate_size":12,"moe_intermediate_size":6,
            "n_routed_experts":3,"num_experts_per_tok":2,"n_shared_experts":1,
            "unpadded_vocab_size":19
          }
        }"#,
    )
    .unwrap();
    let layout = inkling::state_layout(&args).unwrap();
    let context = NumericContext::default();
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let mut whole_state = make_state();
    let mut chunked_state = make_state();
    let mut whole = ResidentRuntime::new(
        inkling::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut chunked = ResidentRuntime::new(
        inkling::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        &context,
    )
    .unwrap();
    let whole_tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let whole_parts = [inkling::DecoderInputPart::Text(&whole_tokens)];
    let expected = whole
        .forward(
            inkling::ModelInput {
                parts: &whole_parts,
                vision_patches: None,
                audio: None,
            },
            &mut whole_state,
            &context,
        )
        .unwrap();
    let prefix_tokens = NumericTensor::token_ids(&[1, 2]);
    let prefix_parts = [inkling::DecoderInputPart::Text(&prefix_tokens)];
    let prefix = chunked
        .forward(
            inkling::ModelInput {
                parts: &prefix_parts,
                vision_patches: None,
                audio: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let decode_tokens = NumericTensor::token_ids(&[3]);
    let decode_parts = [inkling::DecoderInputPart::Text(&decode_tokens)];
    let decode = chunked
        .forward(
            inkling::ModelInput {
                parts: &decode_parts,
                vision_patches: None,
                audio: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let actual = NumericTensor::concatenate(&[prefix, decode], 1, &context).unwrap();
    assert_tensor_close(&actual, &expected, "Inkling fixed-state sparse logits");
    assert_eq!(actual.shape, [1, 3, 19]);
    for layer in 0..2 {
        let state = chunked_state.layer(layer).unwrap();
        assert_eq!(state.position(), 3);
        assert_eq!(
            state.fixed.values().filter(|value| value.is_some()).count(),
            4
        );
    }
}

fn assert_llama_compatible_tp2_reconstructs_uneven_tied_vocabulary_and_exact_collectives(
    model_type: &str,
) {
    let args = llama::model_args_from_config_value(&serde_json::json!({
        "model_type": model_type, "hidden_size": 8, "num_hidden_layers": 2,
        "intermediate_size": 10, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 2, "vocab_size": 7,
        "max_position_embeddings": 64, "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0, "tie_word_embeddings": true
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture = llama::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = llama::static_parallel_parameter_groups::<NumericBackend>(
        &architecture.static_modules().embeddings,
        &architecture.static_modules().norm,
        architecture.static_modules().lm_head.as_ref(),
        "model",
    )
    .unwrap();
    for layer in 0..args.num_hidden_layers as usize {
        let unit = <llama::LayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::build_unit(&architecture, 0, layer, &context)
        .unwrap();
        groups.extend(
            llama::layer_parallel_parameter_groups::<NumericBackend>(&unit, &args, layer).unwrap(),
        );
    }

    let units = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            <llama::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&architecture, 0, layer, &context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        llama::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = llama::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = llama::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            <llama::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, 0, layer, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Llama TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 2, "Llama TP1 state");

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        layouts[0]
            .tensor("model.embed_tokens.weight")
            .unwrap()
            .local_shape(),
        [4, 8]
    );
    assert_eq!(
        layouts[1]
            .tensor("model.embed_tokens.weight")
            .unwrap()
            .local_shape(),
        [3, 8]
    );
    let group = NumericParallelGroup::new(2);
    let mut outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let tokens = tokens.clone();
                let group = Arc::clone(&group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = llama::local_geometry(&args, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = llama::LayeredModel::<NumericBackend>::new_parallel(
                        args.clone(),
                        geometry,
                        &context,
                    )
                    .unwrap();
                    let units = (0..args.num_hidden_layers as usize)
                        .map(|layer| {
                            <llama::LayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(
                                &architecture, 0, layer, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, group);
                    let logits = runtime
                        .forward_parallel(
                            decoder::LayeredInput {
                                tokens: &tokens,
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert_eq!(outputs[0].0.shape, [1, 3, 7]);
    assert_tensor_close(&outputs[0].0, &expected, "Llama TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "Llama TP2 rank 1 logits");
    assert_eq!(outputs[0].1.len(), outputs[1].1.len());
    for (left, right) in outputs[0].1.iter().zip(&outputs[1].1) {
        assert_eq!(left.sequence, right.sequence);
        assert_eq!(left.kind, right.kind);
        assert_eq!(left.output_shape, right.output_shape);
    }
    assert_eq!(
        outputs[0]
            .1
            .iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>(),
        [
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::GatherVocabulary,
        ]
    );
    assert_eq!(outputs[0].1.last().unwrap().input_shape, [1, 3, 4]);
    assert_eq!(outputs[1].1.last().unwrap().input_shape, [1, 3, 3]);
    assert_eq!(outputs[0].1.last().unwrap().output_shape, [1, 3, 7]);
    for (_, _, state) in &mut outputs {
        assert_eq!(state.layer(0).unwrap().position(), 3);
        assert_eq!(state.layer(1).unwrap().position(), 3);
    }
}

#[test]
fn llama_and_mistral_tp2_reconstruct_uneven_tied_vocabulary_and_exact_collectives() {
    assert_llama_compatible_tp2_reconstructs_uneven_tied_vocabulary_and_exact_collectives("llama");
    assert_llama_compatible_tp2_reconstructs_uneven_tied_vocabulary_and_exact_collectives(
        "mistral",
    );
}

fn assert_shared_qwen_tp2(model_type: &str, tied: bool) {
    let mut value = config(model_type, tied);
    value["num_hidden_layers"] = 2.into();
    value["vocab_size"] = 7.into();
    value["num_attention_heads"] = 4.into();
    value["num_key_value_heads"] = 2.into();
    value["head_dim"] = 2.into();
    value["intermediate_size"] = 10.into();
    if model_type == "qwen2" {
        value["use_sliding_window"] = false.into();
    } else if model_type == "qwen3_moe" {
        value["intermediate_size"] = 0.into();
        value["moe_intermediate_size"] = 6.into();
    }
    let args = qwen::model_args_from_config_value(&value).unwrap();
    let context = NumericContext::default();
    let architecture =
        qwen::RoutedLayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = qwen::static_parallel_parameter_groups::<NumericBackend>(
        &architecture.static_modules().embeddings,
        &architecture.static_modules().norm,
        architecture.static_modules().lm_head.as_ref(),
        &args.parameter_root,
    )
    .unwrap();
    for layer in 0..args.num_hidden_layers as usize {
        let unit = <qwen::RoutedLayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::build_unit(&architecture, 0, layer, &context)
        .unwrap();
        groups.extend(qwen::routed_layer_parallel_parameter_groups(&unit, &args, layer).unwrap());
    }
    let units = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            <qwen::RoutedLayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&architecture, 0, layer, &context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        qwen::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = qwen::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = qwen::RoutedLayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            <qwen::RoutedLayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, 0, layer, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, &format!("{model_type} TP1 logits"));
    assert_state_exact(
        &tp1_state,
        &expected_state,
        2,
        &format!("{model_type} TP1 state"),
    );

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let vocabulary = if tied {
        format!("{}.embed_tokens.weight", args.parameter_root)
    } else {
        "lm_head.weight".into()
    };
    assert_eq!(layouts[0].tensor(&vocabulary).unwrap().local_shape()[0], 4);
    assert_eq!(layouts[1].tensor(&vocabulary).unwrap().local_shape()[0], 3);
    let collective_group = NumericParallelGroup::new(2);
    let mut outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let tokens = tokens.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = qwen::local_geometry(&args, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = qwen::RoutedLayeredModel::<NumericBackend>::new_parallel(
                        args.clone(),
                        geometry,
                        &context,
                    )
                    .unwrap();
                    let units = (0..args.num_hidden_layers as usize)
                        .map(|layer| {
                            <qwen::RoutedLayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(
                                &architecture, 0, layer, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let logits = runtime
                        .forward_parallel(
                            decoder::LayeredInput {
                                tokens: &tokens,
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (rank, (logits, trace, state)) in outputs.iter_mut().enumerate() {
        assert_tensor_close(
            logits,
            &expected,
            &format!("{model_type} TP2 rank {rank} logits"),
        );
        assert_eq!(state.layer(0).unwrap().position(), 3);
        assert_eq!(state.layer(1).unwrap().position(), 3);
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::GatherVocabulary,
            ]
        );
        assert_eq!(trace.last().unwrap().input_shape[2], 4 - rank as i32);
        assert_eq!(trace.last().unwrap().output_shape, [1, 3, 7]);
    }
}

#[test]
fn shared_qwen2_qwen3_and_routed_moe_tp2_match_replicated_with_exact_collectives() {
    assert_shared_qwen_tp2("qwen2", false);
    assert_shared_qwen_tp2("qwen3", true);
    assert_shared_qwen_tp2("qwen3_moe", false);
}

#[test]
fn deepseek_v3_dense_tp2_matches_replicated_with_uneven_vocabulary() {
    let args = deepseek::parse_v3_config(&serde_json::json!({
        "model_type": "deepseek_v3", "hidden_size": 8,
        "intermediate_size": 10, "moe_intermediate_size": 4,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "vocab_size": 7, "max_position_embeddings": 64,
        "q_lora_rank": 2, "kv_lora_rank": 2,
        "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 2,
        "first_k_dense_replace": 2, "n_routed_experts": 2,
        "n_shared_experts": 1, "num_experts_per_tok": 1,
        "n_group": 1, "topk_group": 1, "tie_word_embeddings": false
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture = deepseek::v3::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = deepseek::parallel::v3_static_parameter_groups(&args).unwrap();
    for layer in 0..args.num_hidden_layers as usize {
        groups.extend(deepseek::parallel::v3_layer_parameter_groups(&args, layer).unwrap());
    }
    let units = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            <deepseek::v3::Model<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericCompressedCache>,
            >>::build_unit(&architecture, 0, layer, &context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected_state =
        DeviceState::<NumericBackend, _>::create(architecture.state_layout().unwrap(), |_, _| {
            Ok::<_, Error>(NumericCompressedCache::resident())
        })
        .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = deepseek::parallel::v3_local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = deepseek::v3::Model::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = (0..args.num_hidden_layers as usize)
        .map(|layer| {
            <deepseek::v3::Model<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericCompressedCache>,
            >>::build_unit(&tp1_architecture, 0, layer, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, _| {
        Ok::<_, Error>(NumericCompressedCache::resident())
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "DeepSeek-V3 TP1 logits");
    assert_retained_state_exact(
        &tp1_state,
        &expected_state,
        args.num_hidden_layers as usize,
        "DeepSeek-V3 TP1 state",
    );
    for layer in 0..args.num_hidden_layers as usize {
        assert_eq!(
            tp1_state.as_ref()[layer].offset,
            expected_state.as_ref()[layer].offset,
            "DeepSeek-V3 TP1 state layer {layer} offset"
        );
    }

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        layouts[0]
            .tensor("model.embed_tokens.weight")
            .unwrap()
            .local_shape(),
        [4, 8]
    );
    assert_eq!(
        layouts[1]
            .tensor("model.embed_tokens.weight")
            .unwrap()
            .local_shape(),
        [3, 8]
    );
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let tokens = tokens.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = deepseek::parallel::v3_local_geometry(&args, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = deepseek::v3::Model::<NumericBackend>::new_parallel(
                        args.clone(),
                        geometry,
                        &context,
                    )
                    .unwrap();
                    let units = (0..args.num_hidden_layers as usize)
                        .map(|layer| {
                            <deepseek::v3::Model<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericCompressedCache>,
                            >>::build_unit(
                                &architecture, 0, layer, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, _| {
                            Ok::<_, Error>(NumericCompressedCache::resident())
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let logits = runtime
                        .forward_parallel(
                            deepseek::mtp::EmbeddedInput::target(&tokens, None),
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(&outputs[0].0, &expected, "DeepSeek-V3 TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "DeepSeek-V3 TP2 rank 1 logits");
    for (rank, (_, trace, _)) in outputs.iter().enumerate() {
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::GatherVocabulary,
            ]
        );
        assert_eq!(trace.last().unwrap().input_shape, [1, 3, 4 - rank as i32]);
        assert_eq!(trace.last().unwrap().output_shape, [1, 3, 7]);
    }
}

#[test]
fn deepseek_v4_tp2_matches_replicated_hyper_and_routed_block() {
    let args = deepseek::parse_v4_config(&serde_json::json!({
        "model_type": "deepseek_v4", "hidden_size": 4,
        "moe_intermediate_size": 4, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "qk_rope_head_dim": 2, "q_lora_rank": 2,
        "o_lora_rank": 2, "o_groups": 2, "vocab_size": 7,
        "max_position_embeddings": 64, "sliding_window": 4,
        "compress_ratios": [0], "index_n_heads": 2, "index_head_dim": 4,
        "index_topk": 1, "hc_mult": 2, "hc_sinkhorn_iters": 2,
        "n_routed_experts": 2, "n_shared_experts": 1,
        "num_experts_per_tok": 1, "num_hash_layers": 0,
        "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
        "norm_topk_prob": true, "routed_scaling_factor": 1.0, "swiglu_limit": 4.0
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture = deepseek::v4::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = deepseek::parallel::v4_static_parameter_groups(&args).unwrap();
    groups.extend(deepseek::parallel::v4_layer_parameter_groups(&args, 0).unwrap());
    let unit = <deepseek::v4::Model<NumericBackend> as LayeredArchitecture<
        NumericBackend,
        DeviceState<NumericBackend, NumericPoolingCache>,
    >>::build_unit(&architecture, 0, 0, &context)
    .unwrap();
    let mut expected_state =
        DeviceState::<NumericBackend, _>::create(architecture.state_layout().unwrap(), |_, _| {
            Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &[]))
        })
        .unwrap();
    let mut expected_runtime =
        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(vec![unit]));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = deepseek::parallel::v4_local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = deepseek::v4::Model::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_unit = <deepseek::v4::Model<NumericBackend> as LayeredArchitecture<
        NumericBackend,
        DeviceState<NumericBackend, NumericPoolingCache>,
    >>::build_unit(&tp1_architecture, 0, 0, &tp1_context)
    .unwrap();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(vec![tp1_unit]));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, _| {
        Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &[]))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "DeepSeek-V4 TP1 logits");
    assert_retained_state_exact(&tp1_state, &expected_state, 1, "DeepSeek-V4 TP1 state");
    let tp1_cache = &tp1_state.as_ref()[0];
    let expected_cache = &expected_state.as_ref()[0];
    assert_eq!(tp1_cache.offset, expected_cache.offset);
    assert_eq!(
        tp1_cache.attention_local_tokens,
        expected_cache.attention_local_tokens
    );
    match (&tp1_cache.local, &expected_cache.local) {
        (Some(actual), Some(expected)) => {
            assert_tensor_exact(actual, expected, "DeepSeek-V4 TP1 local state")
        }
        (None, None) => {}
        _ => panic!("DeepSeek-V4 TP1 local state presence mismatch"),
    }

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        layouts[0].tensor("embed.weight").unwrap().local_shape(),
        [4, 4]
    );
    assert_eq!(
        layouts[1].tensor("embed.weight").unwrap().local_shape(),
        [3, 4]
    );
    assert_eq!(
        layouts[0]
            .tensor("layers.0.ffn.switch_mlp.gate_up_proj")
            .unwrap()
            .local_shape(),
        [2, 4, 4]
    );
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let tokens = tokens.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = deepseek::parallel::v4_local_geometry(&args, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = deepseek::v4::Model::<NumericBackend>::new_parallel(
                        args.clone(),
                        geometry,
                        &context,
                    )
                    .unwrap();
                    let unit = <deepseek::v4::Model<NumericBackend> as LayeredArchitecture<
                        NumericBackend,
                        DeviceState<NumericBackend, NumericPoolingCache>,
                    >>::build_unit(&architecture, 0, 0, &context)
                    .unwrap();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(vec![unit]));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, _| {
                            Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &[]))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let logits = runtime
                        .forward_parallel(
                            deepseek::mtp::EmbeddedInput::target(&tokens, None),
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(&outputs[0].0, &expected, "DeepSeek-V4 TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "DeepSeek-V4 TP2 rank 1 logits");
    for (_, trace, _) in &outputs {
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::GatherVocabulary,
            ]
        );
        for (sequence, event) in trace.iter().enumerate() {
            assert_eq!(event.sequence, sequence);
            assert_eq!(
                event.output_shape,
                if sequence == 3 {
                    vec![1, 3, 7]
                } else {
                    vec![1, 3, 4]
                }
            );
        }
    }
    assert_eq!(
        outputs[0].1.last().unwrap().kind,
        NumericCollectiveKind::GatherVocabulary
    );
    assert_eq!(outputs[0].1.last().unwrap().input_shape, [1, 3, 4]);
    assert_eq!(outputs[1].1.last().unwrap().input_shape, [1, 3, 3]);
    assert_eq!(outputs[0].1.last().unwrap().output_shape, [1, 3, 7]);
}

#[test]
fn gpt_oss_tp2_matches_replicated_with_biased_packed_experts() {
    let args = gpt_oss::model_args_from_config_value(&serde_json::json!({
        "model_type": "gpt_oss", "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "vocab_size": 7, "num_local_experts": 2, "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001, "sliding_window": 4,
        "max_position_embeddings": 64, "rope_theta": 150000.0,
        "layer_types": ["full_attention"],
        "quantization_config": {"quant_method": "mxfp4"}, "swiglu_limit": 7.0
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        gpt_oss::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups =
        gpt_oss::static_parameter_groups(architecture.static_modules(), &args).unwrap();
    let unit = <gpt_oss::LayeredModel<NumericBackend> as LayeredArchitecture<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
    >>::build_unit(&architecture, 0, 0, &context)
    .unwrap();
    groups.extend(gpt_oss::layer_parallel_parameter_groups(&unit, &args, 0).unwrap());
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime =
        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(vec![unit.clone()]));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = gpt_oss::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = gpt_oss::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let mut tp1_runtime = ResidentRuntime::new(tp1_architecture, &tp1_context).unwrap();
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_logits = tp1_runtime
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut tp1_state,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "GPT-OSS TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 1, "GPT-OSS TP1 state");

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        layouts[0]
            .tensor("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap()
            .local_shape(),
        [2, 64, 32]
    );
    assert_eq!(
        layouts[1]
            .tensor("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap()
            .local_shape(),
        [2, 64, 32]
    );
    let expert_input = NumericTensor::new([1, 32], vec![0.25; 32]);
    let mut global_expert = unit.mlp.clone();
    let global_routes = global_expert.router.route(&expert_input, &context).unwrap();
    let global_expert_output = global_expert
        .experts
        .forward_routed(&expert_input, &global_routes, &context)
        .unwrap();
    let mut partials = layouts
        .iter()
        .map(|layout| {
            let local_context = NumericContext::with_local_layout(layout.clone());
            let geometry = gpt_oss::local_geometry(&args, layout).unwrap();
            let architecture = gpt_oss::LayeredModel::<NumericBackend>::new_parallel(
                args.clone(),
                geometry,
                &local_context,
            )
            .unwrap();
            let mut local = <gpt_oss::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&architecture, 0, 0, &local_context)
            .unwrap();
            let routes = local
                .mlp
                .router
                .route(&expert_input, &local_context)
                .unwrap();
            local
                .mlp
                .experts
                .forward_routed_tensor_parallel(&expert_input, &routes, 2, &local_context)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let expert_reconstructed = partials
        .remove(0)
        .reducible
        .add(&partials[0].reducible, &context)
        .unwrap()
        .add(partials[0].post_reduce.as_ref().unwrap(), &context)
        .unwrap();
    assert_tensor_close(
        &expert_reconstructed,
        &global_expert_output,
        "GPT-OSS packed expert TP2",
    );
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let tokens = tokens.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = gpt_oss::local_geometry(&args, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = gpt_oss::LayeredModel::<NumericBackend>::new_parallel(
                        args, geometry, &context,
                    )
                    .unwrap();
                    let unit = <gpt_oss::LayeredModel<NumericBackend> as LayeredArchitecture<
                        NumericBackend,
                        DeviceState<NumericBackend, NumericHybridLayerState>,
                    >>::build_unit(&architecture, 0, 0, &context)
                    .unwrap();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(vec![unit]));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let logits = runtime
                        .forward_parallel(
                            decoder::LayeredInput {
                                tokens: &tokens,
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(&outputs[0].0, &outputs[1].0, "GPT-OSS TP2 rank agreement");
    assert_tensor_close(&outputs[0].0, &expected, "GPT-OSS TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "GPT-OSS TP2 rank 1 logits");
    for (rank, (_, trace, _)) in outputs.iter().enumerate() {
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::GatherVocabulary,
            ]
        );
        assert_eq!(trace.last().unwrap().input_shape, [1, 3, 4 - rank as i32]);
        assert_eq!(trace.last().unwrap().output_shape, [1, 3, 7]);
    }
}

#[test]
fn qwen3_vl_tp2_runs_full_vision_and_text_lifecycle() {
    let args = qwen::vl::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3_vl", "image_token_id": 5, "video_token_id": 6,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_vl_text", "hidden_size": 16,
            "num_hidden_layers": 1, "intermediate_size": 10,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
            "rms_norm_eps": 0.000001, "vocab_size": 7,
            "max_position_embeddings": 64, "rope_theta": 1000000.0,
            "rope_scaling": {"mrope_section": [1, 1, 2], "mrope_interleaved": true}
        },
        "vision_config": {
            "depth": 1, "hidden_size": 8, "intermediate_size": 10,
            "num_heads": 2, "num_position_embeddings": 16,
            "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "out_hidden_size": 16,
            "deepstack_visual_indexes": []
        }
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        qwen::vl::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let static_modules = <qwen::vl::LayeredModel<NumericBackend> as LayeredArchitecture<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
    >>::static_modules(&architecture);
    let mut groups = qwen::static_parallel_parameter_groups::<NumericBackend>(
        &static_modules.text.embeddings,
        &static_modules.text.norm,
        static_modules.text.lm_head.as_ref(),
        &args.text.parameter_root,
    )
    .unwrap();
    groups.extend(
        qwen::vision::static_parallel_parameter_groups(
            &static_modules.vision,
            &args.vision,
            "model.visual",
        )
        .unwrap(),
    );
    let vision_unit = <qwen::vl::LayeredModel<NumericBackend> as LayeredArchitecture<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
    >>::build_unit(&architecture, 0, 0, &context)
    .unwrap();
    let qwen::vl::Unit::Vision(vision_block) = &vision_unit else {
        unreachable!()
    };
    groups.extend(
        qwen::vision::block_parallel_parameter_groups(
            vision_block,
            &args.vision,
            "model.visual",
            0,
        )
        .unwrap(),
    );
    let text_unit = <qwen::vl::LayeredModel<NumericBackend> as LayeredArchitecture<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
    >>::build_unit(&architecture, 1, 0, &context)
    .unwrap();
    let qwen::vl::Unit::Text(text_block) = &text_unit else {
        unreachable!()
    };
    groups.extend(qwen::routed_layer_parallel_parameter_groups(text_block, &args.text, 0).unwrap());
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(
        architecture,
        ResidentUnitWindow::new(vec![vision_unit, text_unit]),
    );
    let text_tokens = NumericTensor::token_ids(&[1, 2]);
    let image_tokens = NumericTensor::token_ids(&[5]);
    let grid = [(1, 2, 2)];
    let pixels = NumericTensor::new(
        [4, 24],
        (0..96).map(|index| (index as f32 - 48.0) / 100.0).collect(),
    );
    let parts = [
        qwen::vl::InputPart::Text(&text_tokens),
        qwen::vl::InputPart::Image {
            tokens: &image_tokens,
            grid: &grid,
        },
    ];
    let expected = expected_runtime
        .forward(
            qwen::vl::ModelInput {
                parts: &parts,
                pixels: Some(&pixels),
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = qwen::vl::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = qwen::vl::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let mut tp1_runtime = ResidentRuntime::new(tp1_architecture, &tp1_context).unwrap();
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parts = [
        qwen::vl::InputPart::Text(&text_tokens),
        qwen::vl::InputPart::Image {
            tokens: &image_tokens,
            grid: &grid,
        },
    ];
    let tp1_logits = tp1_runtime
        .forward(
            qwen::vl::ModelInput {
                parts: &tp1_parts,
                pixels: Some(&pixels),
                mask: None,
            },
            &mut tp1_state,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Qwen3-VL TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 1, "Qwen3-VL TP1 state");

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let text_tokens = text_tokens.clone();
                let image_tokens = image_tokens.clone();
                let pixels = pixels.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = qwen::vl::local_geometry(&args, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = qwen::vl::LayeredModel::<NumericBackend>::new_parallel(
                        args, geometry, &context,
                    )
                    .unwrap();
                    let units = [(0, 0), (1, 0)]
                        .into_iter()
                        .map(|(group, index)| {
                            <qwen::vl::LayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(
                                &architecture, group, index, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let grid = [(1, 2, 2)];
                    let parts = [
                        qwen::vl::InputPart::Text(&text_tokens),
                        qwen::vl::InputPart::Image {
                            tokens: &image_tokens,
                            grid: &grid,
                        },
                    ];
                    let logits = runtime
                        .forward_parallel(
                            qwen::vl::ModelInput {
                                parts: &parts,
                                pixels: Some(&pixels),
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(&outputs[0].0, &expected, "Qwen3-VL TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "Qwen3-VL TP2 rank 1 logits");
    for (rank, (_, trace, _)) in outputs.iter().enumerate() {
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::GatherVocabulary,
            ]
        );
        assert_eq!(
            trace[..6]
                .iter()
                .map(|event| event.input_shape.clone())
                .collect::<Vec<_>>(),
            [
                vec![1, 2, 16],
                vec![4, 8],
                vec![4, 8],
                vec![1, 16],
                vec![1, 3, 16],
                vec![3, 16],
            ]
        );
        assert!(trace[..6]
            .iter()
            .enumerate()
            .all(|(sequence, event)| event.sequence == sequence
                && event.output_shape == event.input_shape));
        assert_eq!(
            trace.last().unwrap().kind,
            NumericCollectiveKind::GatherVocabulary
        );
        assert_eq!(trace.last().unwrap().input_shape, [1, 3, 4 - rank as i32]);
        assert_eq!(trace.last().unwrap().output_shape, [1, 3, 7]);
    }
}

#[test]
fn qwen_hybrid_constructed_graph_owns_embedded_prediction_depth() {
    let config = qwen::hybrid::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3_5_text",
        "vocab_size": 8,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "mtp_num_hidden_layers": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 16,
        "intermediate_size": 16,
        "num_experts": 0,
        "tie_word_embeddings": true,
        "layer_types": ["full_attention", "full_attention"]
    }))
    .unwrap()
    .text;
    let architecture =
        qwen::hybrid::LayeredModel::<NumericBackend>::new(config, &NumericContext::default())
            .unwrap();

    assert_eq!(architecture.mtp_len(), 2);
    assert_eq!(architecture.unit_layout().unwrap().group_count(), 3);
}

#[test]
fn qwen35_conditional_tp2_runs_full_vision_and_text_lifecycle() {
    let parsed = qwen::hybrid::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3_5", "image_token_id": 5, "video_token_id": 6,
        "text_config": {
            "model_type": "qwen3_5_text", "vocab_size": 7, "hidden_size": 8,
            "num_hidden_layers": 1, "mtp_num_hidden_layers": 0,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 2,
            "max_position_embeddings": 64, "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 2, "linear_value_head_dim": 2,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "intermediate_size": 10, "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4, "num_experts_per_tok": 1,
            "num_experts": 2, "layer_types": ["full_attention"],
            "tie_word_embeddings": false
        },
        "vision_config": {
            "depth": 1, "hidden_size": 8, "intermediate_size": 10,
            "num_heads": 2, "num_position_embeddings": 16,
            "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "out_hidden_size": 8
        }
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        qwen::hybrid::ConditionalLayeredModel::<NumericBackend>::new(parsed.clone(), &context)
            .unwrap();
    let static_modules =
        <qwen::hybrid::ConditionalLayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::static_modules(&architecture);
    let mut groups = decoder::static_parallel_parameter_groups::<NumericBackend>(
        &static_modules.text.embeddings,
        &static_modules.text.norm,
        static_modules.text.lm_head.as_ref(),
        "model",
    )
    .unwrap();
    let vision = parsed.vision.as_ref().unwrap();
    groups.extend(
        qwen::vision::static_parallel_parameter_groups(
            &static_modules.vision,
            vision,
            "model.visual",
        )
        .unwrap(),
    );
    let vision_unit =
        <qwen::hybrid::ConditionalLayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::build_unit(&architecture, 0, 0, &context)
        .unwrap();
    let qwen::hybrid::ConditionalUnit::Vision(vision_block) = &vision_unit else {
        unreachable!()
    };
    groups.extend(
        qwen::vision::block_parallel_parameter_groups(vision_block, vision, "model.visual", 0)
            .unwrap(),
    );
    let target_unit =
        <qwen::hybrid::ConditionalLayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::build_unit(&architecture, 1, 0, &context)
        .unwrap();
    let qwen::hybrid::ConditionalUnit::Target(target_block) = &target_unit else {
        unreachable!()
    };
    let wrapped = qwen::hybrid::Unit::Target(target_block.clone());
    groups.extend(
        qwen::hybrid::unit_parallel_parameter_groups(&wrapped, &parsed.text, 0, 0).unwrap(),
    );
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(
        architecture,
        ResidentUnitWindow::new(vec![vision_unit, target_unit]),
    );
    let text_tokens = NumericTensor::token_ids(&[1, 2]);
    let image_tokens = NumericTensor::token_ids(&[5]);
    let grid = [(1, 2, 2)];
    let pixels = NumericTensor::new(
        [4, 24],
        (0..96).map(|index| (index as f32 - 48.0) / 100.0).collect(),
    );
    let parts = [
        qwen::vl::InputPart::Text(&text_tokens),
        qwen::vl::InputPart::Image {
            tokens: &image_tokens,
            grid: &grid,
        },
    ];
    let expected = expected_runtime
        .forward(
            qwen::hybrid::ConditionalInput::Target {
                parts: &parts,
                pixels: Some(&pixels),
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = qwen::hybrid::conditional_local_geometry(&parsed, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = qwen::hybrid::ConditionalLayeredModel::<NumericBackend>::new_parallel(
        parsed.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let mut tp1_runtime = ResidentRuntime::new(tp1_architecture, &tp1_context).unwrap();
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parts = [
        qwen::vl::InputPart::Text(&text_tokens),
        qwen::vl::InputPart::Image {
            tokens: &image_tokens,
            grid: &grid,
        },
    ];
    let tp1_logits = tp1_runtime
        .forward(
            qwen::hybrid::ConditionalInput::Target {
                parts: &tp1_parts,
                pixels: Some(&pixels),
                mask: None,
            },
            &mut tp1_state,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Qwen3.5 conditional TP1 logits");
    assert_state_exact(
        &tp1_state,
        &expected_state,
        1,
        "Qwen3.5 conditional TP1 state",
    );

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let parsed = parsed.clone();
                let text_tokens = text_tokens.clone();
                let image_tokens = image_tokens.clone();
                let pixels = pixels.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry =
                        qwen::hybrid::conditional_local_geometry(&parsed, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture =
                        qwen::hybrid::ConditionalLayeredModel::<NumericBackend>::new_parallel(
                            parsed, geometry, &context,
                        )
                        .unwrap();
                    let units = [(0, 0), (1, 0)]
                        .into_iter()
                        .map(|(group, index)| {
                            <qwen::hybrid::ConditionalLayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(&architecture, group, index, &context)
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let grid = [(1, 2, 2)];
                    let parts = [
                        qwen::vl::InputPart::Text(&text_tokens),
                        qwen::vl::InputPart::Image {
                            tokens: &image_tokens,
                            grid: &grid,
                        },
                    ];
                    let logits = runtime
                        .forward_parallel(
                            qwen::hybrid::ConditionalInput::Target {
                                parts: &parts,
                                pixels: Some(&pixels),
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(
        &outputs[0].0,
        &expected,
        "Qwen3.5 conditional TP2 rank 0 logits",
    );
    assert_tensor_close(
        &outputs[1].0,
        &expected,
        "Qwen3.5 conditional TP2 rank 1 logits",
    );
    for (rank, (_, trace, _)) in outputs.iter().enumerate() {
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::GatherVocabulary,
            ]
        );
        assert_eq!(
            trace[..6]
                .iter()
                .map(|event| event.input_shape.clone())
                .collect::<Vec<_>>(),
            [
                vec![1, 2, 8],
                vec![4, 8],
                vec![4, 8],
                vec![1, 8],
                vec![1, 3, 8],
                vec![1, 3, 8],
            ]
        );
        assert!(trace[..6]
            .iter()
            .enumerate()
            .all(|(sequence, event)| event.sequence == sequence
                && event.output_shape == event.input_shape));
        assert_eq!(
            trace.last().unwrap().kind,
            NumericCollectiveKind::GatherVocabulary
        );
        assert_eq!(trace.last().unwrap().input_shape, [1, 3, 4 - rank as i32]);
        assert_eq!(trace.last().unwrap().output_shape, [1, 3, 7]);
    }
}

#[test]
fn qwen_hybrid_tp2_reconstructs_uneven_untied_vocabulary_and_exact_collectives() {
    let config = qwen::hybrid::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3_5_text", "vocab_size": 7, "hidden_size": 8,
        "num_hidden_layers": 2, "mtp_num_hidden_layers": 0,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 2,
        "max_position_embeddings": 64, "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 2, "linear_value_head_dim": 2,
        "linear_num_key_heads": 2, "linear_num_value_heads": 2,
        "intermediate_size": 10, "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4, "num_experts_per_tok": 1,
        "num_experts": 2, "layer_types": ["linear_attention", "full_attention"],
        "tie_word_embeddings": false
    }))
    .unwrap()
    .text;
    let context = NumericContext::default();
    let architecture =
        qwen::hybrid::LayeredModel::<NumericBackend>::new(config.clone(), &context).unwrap();
    let mut groups = decoder::static_parallel_parameter_groups::<NumericBackend>(
        &architecture.static_modules().embeddings,
        &architecture.static_modules().norm,
        architecture.static_modules().lm_head.as_ref(),
        "model",
    )
    .unwrap();
    for layer in 0..config.num_hidden_layers as usize {
        let unit = <qwen::hybrid::LayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::build_unit(&architecture, 0, layer, &context)
        .unwrap();
        groups.extend(
            qwen::hybrid::unit_parallel_parameter_groups(&unit, &config, 0, layer).unwrap(),
        );
    }
    let units = (0..config.num_hidden_layers as usize)
        .map(|layer| {
            <qwen::hybrid::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&architecture, 0, layer, &context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            qwen::hybrid::EmbeddedInput::Target {
                tokens: &tokens,
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = qwen::hybrid::local_geometry(&config, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = qwen::hybrid::LayeredModel::<NumericBackend>::new_parallel(
        config.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = (0..config.num_hidden_layers as usize)
        .map(|layer| {
            <qwen::hybrid::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, 0, layer, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            qwen::hybrid::EmbeddedInput::Target {
                tokens: &tokens,
                mask: None,
            },
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Qwen hybrid TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 2, "Qwen hybrid TP1 state");

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        layouts[0].tensor("lm_head.weight").unwrap().local_shape(),
        [4, 8]
    );
    assert_eq!(
        layouts[1].tensor("lm_head.weight").unwrap().local_shape(),
        [3, 8]
    );
    let group = NumericParallelGroup::new(2);
    let mut outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let config = config.clone();
                let tokens = tokens.clone();
                let group = Arc::clone(&group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = qwen::hybrid::local_geometry(&config, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = qwen::hybrid::LayeredModel::<NumericBackend>::new_parallel(
                        config.clone(),
                        geometry,
                        &context,
                    )
                    .unwrap();
                    let units =
                        (0..config.num_hidden_layers as usize)
                            .map(|layer| {
                                <qwen::hybrid::LayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(&architecture, 0, layer, &context)
                            .unwrap()
                            })
                            .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, group);
                    let logits = runtime
                        .forward_parallel(
                            qwen::hybrid::EmbeddedInput::Target {
                                tokens: &tokens,
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert_tensor_close(&outputs[0].0, &expected, "Qwen hybrid TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "Qwen hybrid TP2 rank 1 logits");
    assert_eq!(outputs[0].1.len(), outputs[1].1.len());
    for (left, right) in outputs[0].1.iter().zip(&outputs[1].1) {
        assert_eq!((left.sequence, left.kind), (right.sequence, right.kind));
        assert_eq!(left.output_shape, right.output_shape);
    }
    assert_eq!(
        outputs[0]
            .1
            .iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>(),
        [
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::GatherVocabulary,
        ]
    );
    assert_eq!(outputs[0].1.last().unwrap().input_shape, [1, 3, 4]);
    assert_eq!(outputs[1].1.last().unwrap().input_shape, [1, 3, 3]);
    assert_eq!(outputs[0].1.last().unwrap().output_shape, [1, 3, 7]);
    for (_, _, state) in &mut outputs {
        assert_eq!(state.layer(0).unwrap().position(), 3);
        assert_eq!(state.layer(1).unwrap().position(), 3);
    }
}

#[test]
fn inkling_tensor_parallel_size_one_matches_replicated_multimodal_graph() {
    let args = inkling::ModelArgs::from_hf_json(
        br#"{
          "model_type":"inkling_mm_model",
          "text_config":{
            "hidden_size":8,"num_hidden_layers":2,"vocab_size":19,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "sliding_window_size":4,
            "layer_types":["sliding_attention","full_attention"],
            "mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,
            "d_rel":2,"rel_extent":8,"intermediate_size":12,
            "dense_intermediate_size":12,"moe_intermediate_size":6,
            "n_routed_experts":3,"num_experts_per_tok":2,"n_shared_experts":1,
            "unpadded_vocab_size":19
          }
        }"#,
    )
    .unwrap();
    let context = NumericContext::default();
    let state_layout = inkling::state_layout(&args).unwrap();
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(state_layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };

    let mut replicated = ResidentRuntime::new(
        inkling::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut replicated_state = make_state();

    let tensor = |logical: &str, role, shape: Vec<usize>| {
        LocalTensorLayout::new(
            logical,
            role,
            shape.clone(),
            shape,
            eredu_runtime::TensorPlacement::Replicated,
            None,
            None,
            false,
        )
    };
    let mut layout = LocalModelLayout::default();
    layout.insert(
        "model.embed_tokens.weight".into(),
        tensor("model.embed_tokens", ParameterRole::Vocabulary, vec![19, 8]),
    );
    layout.insert(
        "lm_head.weight".into(),
        tensor("lm_head", ParameterRole::Vocabulary, vec![19, 8]),
    );
    for layer in 0..2 {
        layout.insert(
            format!("model.layers.{layer}.self_attn.q_proj.weight"),
            tensor("query", ParameterRole::AttentionHeads, vec![8, 8]),
        );
        layout.insert(
            format!("model.layers.{layer}.self_attn.k_proj.weight"),
            tensor("key", ParameterRole::AttentionHeads, vec![4, 8]),
        );
    }
    layout.insert(
        "model.layers.0.dense.gate_proj.weight".into(),
        tensor("dense", ParameterRole::FeedForwardIntermediate, vec![12, 8]),
    );
    layout.insert(
        "model.layers.1.dense.gate_proj.weight".into(),
        tensor("dense", ParameterRole::FeedForwardIntermediate, vec![12, 8]),
    );
    let geometry = Arc::new(inkling::local_geometry(&args, &layout).unwrap());
    let parallel_architecture =
        inkling::LayeredModel::<NumericBackend>::new_parallel(args, geometry, &context).unwrap();
    let units = (0..2)
        .map(|index| {
            <inkling::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&parallel_architecture, 1, index, &context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut parallel = LayerwiseRuntime::new(parallel_architecture, ResidentUnitWindow::new(units));
    let mut parallel_state = make_state();

    let tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let parts = [inkling::DecoderInputPart::Text(&tokens)];
    let expected = replicated
        .forward(
            inkling::ModelInput {
                parts: &parts,
                vision_patches: None,
                audio: None,
            },
            &mut replicated_state,
            &context,
        )
        .unwrap();
    let actual = parallel
        .forward_parallel(
            inkling::ModelInput {
                parts: &parts,
                vision_patches: None,
                audio: None,
            },
            &mut parallel_state,
            &NumericParallelContext::new(0, NumericParallelGroup::new(1)),
            &context,
        )
        .unwrap();
    assert_tensor_exact(&actual, &expected, "Inkling TP1 logits");
    assert_state_exact(&parallel_state, &replicated_state, 2, "Inkling TP1 state");
}

#[test]
fn inkling_tp1_tp2_trim_padded_vocab_and_match_replicated_multimodal_graph() {
    let args = inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
          "model_type":"inkling_mm_model", "image_token_id":5, "audio_token_id":6,
          "text_config":{
            "hidden_size":8,"num_hidden_layers":2,"vocab_size":8,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,
            "sliding_window_size":4,
            "layer_types":["sliding_attention","full_attention"],
            "mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,
            "d_rel":2,"rel_extent":8,"intermediate_size":10,
            "dense_intermediate_size":10,"moe_intermediate_size":4,
            "n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":1,
            "unpadded_vocab_size":7
          },
          "audio_config":{"text_hidden_size":8,"num_codebooks":2,"codebook_size":4},
          "vision_config":{"text_hidden_size":8,"patch_size":40,"temporal_patch_size":2,
            "num_channels":3,"num_hidden_layers":4}
        }))
        .unwrap(),
    )
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        inkling::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = inkling::static_parameter_groups(&args).unwrap();
    for layer in 0..2 {
        groups.extend(inkling::layer_parameter_groups(&args, layer).unwrap());
    }
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        inkling::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let text_before = NumericTensor::token_ids(&[1]);
    let image_tokens = NumericTensor::token_ids(&[5]);
    let audio_tokens = NumericTensor::token_ids(&[6, 6]);
    let text_after = NumericTensor::token_ids(&[2]);
    let parts = [
        inkling::DecoderInputPart::Text(&text_before),
        inkling::DecoderInputPart::Image(&image_tokens),
        inkling::DecoderInputPart::Audio(&audio_tokens),
        inkling::DecoderInputPart::Text(&text_after),
    ];
    let patches = NumericTensor::zeros([1, 2, 40, 40, 3]);
    let audio_codes = NumericTensor::new([1, 3, 2], vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0]);
    let expected = expected_runtime
        .forward(
            inkling::ModelInput {
                parts: &parts,
                vision_patches: Some(&patches),
                audio: Some(inkling::AudioInput {
                    code_ids: &audio_codes,
                    valid_frames: 2,
                }),
            },
            &mut expected_state,
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = Arc::new(inkling::local_geometry(&args, &tp1_layout).unwrap());
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = inkling::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
        .into_iter()
        .map(|(group, index)| {
            <inkling::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, group, index, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_logits = tp1_runtime
        .forward_parallel(
            inkling::ModelInput {
                parts: &parts,
                vision_patches: Some(&patches),
                audio: Some(inkling::AudioInput {
                    code_ids: &audio_codes,
                    valid_frames: 2,
                }),
            },
            &mut tp1_state,
            &NumericParallelContext::new(0, NumericParallelGroup::new(1)),
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Inkling multimodal TP1 logits");
    assert_state_exact(
        &tp1_state,
        &expected_state,
        2,
        "Inkling multimodal TP1 state",
    );

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let text_before = text_before.clone();
                let image_tokens = image_tokens.clone();
                let audio_tokens = audio_tokens.clone();
                let text_after = text_after.clone();
                let patches = patches.clone();
                let audio_codes = audio_codes.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = Arc::new(inkling::local_geometry(&args, &layout).unwrap());
                    let state_layout = geometry.state_layout().clone();
                    let architecture = inkling::LayeredModel::<NumericBackend>::new_parallel(
                        args, geometry, &context,
                    )
                    .unwrap();
                    let addresses = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)];
                    let units = addresses
                        .into_iter()
                        .map(|(group, index)| {
                            <inkling::LayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(
                                &architecture, group, index, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let parts = [
                        inkling::DecoderInputPart::Text(&text_before),
                        inkling::DecoderInputPart::Image(&image_tokens),
                        inkling::DecoderInputPart::Audio(&audio_tokens),
                        inkling::DecoderInputPart::Text(&text_after),
                    ];
                    let logits = runtime
                        .forward_parallel(
                            inkling::ModelInput {
                                parts: &parts,
                                vision_patches: Some(&patches),
                                audio: Some(inkling::AudioInput {
                                    code_ids: &audio_codes,
                                    valid_frames: 2,
                                }),
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    (logits, parallel.trace())
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(
        &outputs[0].0,
        &expected,
        "Inkling TP2 rank 0 multimodal logits",
    );
    assert_tensor_close(
        &outputs[1].0,
        &expected,
        "Inkling TP2 rank 1 multimodal logits",
    );
    assert_eq!(outputs[0].1.len(), 7);
    assert_eq!(outputs[1].1.len(), 7);
    assert_eq!(
        outputs[0]
            .1
            .iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>(),
        [
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::GatherVocabulary,
        ]
    );
    assert_eq!(expected.shape, [1, 5, 7]);
    assert_eq!(outputs[0].1.last().unwrap().output_shape, [1, 5, 8]);
}

#[test]
fn inkling_embedded_mtp_traversal_and_rollback_are_backend_neutral() {
    let args = inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
          "model_type":"inkling_mm_model",
          "text_config":{
            "hidden_size":8,"num_hidden_layers":2,"vocab_size":7,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,
            "sliding_window_size":4,
            "layer_types":["sliding_attention","full_attention"],
            "mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,
            "d_rel":2,"rel_extent":8,"intermediate_size":10,
            "dense_intermediate_size":10,"moe_intermediate_size":4,
            "n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":1,
            "unpadded_vocab_size":7
          },
          "mtp_config":{
            "num_nextn_predict_layers":2,"local_layer_ids":[1],
            "chain_hidden_post_norm":true
          }
        }))
        .unwrap(),
    )
    .unwrap();
    let context = NumericContext::default();
    let layout = inkling::mtp_state_layout(&args)
        .unwrap()
        .expect("Inkling MTP state layout");
    let mut model = inkling::LayeredModel::<NumericBackend>::new(args, &context).unwrap();
    let ingress_layout = model.ingress_state_layout().unwrap();
    let persistence_layout = ArchitectureParameters::state_layout(&model).unwrap();
    assert_eq!(ingress_layout.len(), 2);
    assert_eq!(persistence_layout.len(), 4);
    assert_ne!(ingress_layout, persistence_layout);
    assert_eq!(model.mtp_len(), 2);
    assert_eq!(model.mtp_policy(0), Some(eredu_core::AttentionPolicy::Full));
    assert!(model.mtp_policy(1).unwrap().window().is_some());
    let mut state = DeviceState::<NumericBackend, _>::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let hidden = NumericTensor::new([1, 1, 8], (0..8).map(|value| value as f32 / 8.0).collect());
    let first_token = NumericTensor::token_ids(&[2]);
    let first = model
        .forward_partition_mtp(&hidden, &first_token, 0, state.as_mut(), None, &context)
        .unwrap();
    assert_eq!(first.logits.shape, [1, 1, 7]);
    assert_eq!(first.hidden.shape, [1, 1, 8]);
    assert_tensor_exact(&first.tokens, &first_token, "Inkling MTP depth-0 tokens");
    assert_eq!(state.layer(0).unwrap().position(), 1);
    assert_eq!(state.layer(1).unwrap().position(), 0);

    let checkpoint = state.clone();
    let mut transaction = eredu_runtime::DraftStateTransaction::fork(&checkpoint);
    let second_token = NumericTensor::token_ids(&[3]);
    let second = model
        .forward_partition_mtp(
            &first.hidden,
            &second_token,
            3,
            transaction.draft_mut().as_mut(),
            None,
            &context,
        )
        .unwrap();
    assert_eq!(second.logits.shape, [1, 1, 7]);
    assert_tensor_exact(&second.tokens, &second_token, "Inkling MTP cyclic tokens");
    assert_eq!(transaction.draft_mut().layer(0).unwrap().position(), 1);
    assert_eq!(transaction.draft_mut().layer(1).unwrap().position(), 1);
    let mut canonical = checkpoint.clone();
    transaction.rollback(&mut canonical);
    assert_eq!(canonical.layer(0).unwrap().position(), 1);
    assert_eq!(canonical.layer(1).unwrap().position(), 0);
}

#[test]
fn real_multimodal_model_matches_resident_and_dense_streamed_traversal() {
    let args = inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
          "model_type":"inkling_mm_model", "image_token_id":5,
          "text_config":{
            "hidden_size":8,"num_hidden_layers":1,"vocab_size":7,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,
            "sliding_window_size":4, "layer_types":["full_attention"],
            "mlp_layer_types":["dense"],"sconv_kernel_size":3,
            "d_rel":2,"rel_extent":8,"intermediate_size":10,
            "dense_intermediate_size":10,"moe_intermediate_size":4,
            "n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":1,
            "unpadded_vocab_size":7
          },
          "vision_config":{"text_hidden_size":8,"patch_size":40,"temporal_patch_size":2,
            "num_channels":3,"num_hidden_layers":4}
        }))
        .unwrap(),
    )
    .unwrap();
    let context = NumericContext::default();
    let layout = inkling::state_layout(&args).unwrap();
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let mut resident_state = make_state();
    let mut streamed_state = make_state();
    let mut resident = ResidentRuntime::new(
        inkling::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut streamed = LayerwiseRuntime::new(
        inkling::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        RebuildingUnitPolicy::default(),
    );
    let text_tokens = NumericTensor::token_ids(&[1, 2]);
    let image_tokens = NumericTensor::token_ids(&[5]);
    let parts = [
        inkling::DecoderInputPart::Text(&text_tokens),
        inkling::DecoderInputPart::Image(&image_tokens),
    ];
    let patches = NumericTensor::zeros([1, 2, 40, 40, 3]);
    let input = || inkling::ModelInput {
        parts: &parts,
        vision_patches: Some(&patches),
        audio: None,
    };
    let expected = resident
        .forward(input(), &mut resident_state, &context)
        .unwrap();
    let actual = streamed
        .forward(input(), &mut streamed_state, &context)
        .unwrap();
    assert_tensor_close(
        &actual,
        &expected,
        "Inkling resident/dense-stream multimodal logits",
    );
    assert_eq!(
        streamed.policy().events,
        expected_stream_events(&[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)])
    );
    assert_eq!(streamed_state.layer(0).unwrap().position(), 3);
    assert_eq!(resident_state.layer(0).unwrap().position(), 3);
}

#[test]
fn muse_text_decode_skips_vision_and_is_chunk_invariant() {
    let value = serde_json::json!({
        "architectures":["MuseGlimmerForConditionalGeneration"],"model_type":"muse_glimmer",
        "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":8,
        "text_config":{"model_type":"muse_glimmer_text","hidden_size":8,"num_hidden_layers":2,
          "intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
          "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,
          "max_position_embeddings":64,"rope_theta":10000.0,
          "layer_types":["sliding_attention","full_attention"],
          "layer_rope_theta":[10000.0,0.0],"sliding_window":4,"tie_word_embeddings":false,
          "hidden_act":"silu","attention_dropout":0.0,"qk_scale_factor":1.0,
          "output_multiplier":1.0,"final_logit_softcapping":7.0},
        "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,
          "intermediate_size":12,"num_attention_heads":2,"num_hidden_layers":1,
          "patch_size":2,"patch_temporal":1,"merge_size":1,"pos_emb_height":2,
          "pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
          "hidden_act":"gelu","layer_types":["full_attention"],
          "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
    });
    let args = muse_glimmer::DecoderConfig::from_hf_value(&value).unwrap();
    let layout = muse_glimmer::state_layout(&args).unwrap();
    let context = NumericContext::default();
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let mut whole_state = make_state();
    let mut chunked_state = make_state();
    let mut whole = ResidentRuntime::new(
        muse_glimmer::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut chunked = ResidentRuntime::new(
        muse_glimmer::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        &context,
    )
    .unwrap();
    let whole_tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let whole_parts = [muse_glimmer::DecoderInputPart::Text(&whole_tokens)];
    let expected = whole
        .forward(
            muse_glimmer::ModelInput {
                parts: &whole_parts,
                vision: None,
                mask: None,
            },
            &mut whole_state,
            &context,
        )
        .unwrap();
    let prefix_tokens = NumericTensor::token_ids(&[1, 2]);
    let prefix_parts = [muse_glimmer::DecoderInputPart::Text(&prefix_tokens)];
    let prefix = chunked
        .forward(
            muse_glimmer::ModelInput {
                parts: &prefix_parts,
                vision: None,
                mask: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let decode_tokens = NumericTensor::token_ids(&[3]);
    let decode_parts = [muse_glimmer::DecoderInputPart::Text(&decode_tokens)];
    let decode = chunked
        .forward(
            muse_glimmer::ModelInput {
                parts: &decode_parts,
                vision: None,
                mask: None,
            },
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let actual = NumericTensor::concatenate(&[prefix, decode], 1, &context).unwrap();
    assert_tensor_close(&actual, &expected, "Muse-Glimmer text logits");
    assert_eq!(actual.shape, [1, 3, 24]);
}

#[test]
fn muse_glimmer_tp2_ordered_vision_text_matches_replicated_multimodal_graph() {
    let args = muse_glimmer::DecoderConfig::from_hf_value(&serde_json::json!({
        "architectures":["MuseGlimmerForConditionalGeneration"],"model_type":"muse_glimmer",
        "image_token_id":5,"video_token_id":6,"out_hidden_size":32,"projector_hidden_size":8,
        "text_config":{"model_type":"muse_glimmer_text","hidden_size":8,"num_hidden_layers":2,
          "intermediate_size":10,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,
          "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":7,
          "max_position_embeddings":64,"rope_theta":10000.0,
          "layer_types":["sliding_attention","full_attention"],
          "layer_rope_theta":[10000.0,0.0],"sliding_window":4,"tie_word_embeddings":false,
          "hidden_act":"silu","attention_dropout":0.0,"qk_scale_factor":1.0,
          "output_multiplier":1.0,"final_logit_softcapping":7.0},
        "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,
          "intermediate_size":10,"num_attention_heads":2,"num_hidden_layers":1,
          "patch_size":2,"patch_temporal":1,"merge_size":1,"pos_emb_height":2,
          "pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
          "hidden_act":"gelu","layer_types":["full_attention"],
          "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        muse_glimmer::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = muse_glimmer::static_parameter_groups(&args).unwrap();
    for layer in 0..2 {
        groups.extend(muse_glimmer::layer_parameter_groups(&args, layer).unwrap());
    }
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        muse_glimmer::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let text_before = NumericTensor::token_ids(&[0]);
    let media_tokens = NumericTensor::token_ids(&[5, 5, 5, 5]);
    let text_after = NumericTensor::token_ids(&[6]);
    let pixels = NumericTensor::new(
        [4, 12],
        (0..48).map(|index| (index as f32 - 24.0) / 100.0).collect(),
    );
    let grid = [(1, 2, 2)];
    let parts = [
        muse_glimmer::DecoderInputPart::Text(&text_before),
        muse_glimmer::DecoderInputPart::Media(&media_tokens),
        muse_glimmer::DecoderInputPart::Text(&text_after),
    ];
    let expected = expected_runtime
        .forward(
            muse_glimmer::ModelInput {
                parts: &parts,
                vision: Some(muse_glimmer::VisionInput {
                    pixels: &pixels,
                    grid: &grid,
                }),
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();
    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = muse_glimmer::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = muse_glimmer::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = [(0, 0), (1, 0), (1, 1)]
        .into_iter()
        .map(|(group, index)| {
            <muse_glimmer::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, group, index, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parts = [
        muse_glimmer::DecoderInputPart::Text(&text_before),
        muse_glimmer::DecoderInputPart::Media(&media_tokens),
        muse_glimmer::DecoderInputPart::Text(&text_after),
    ];
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            muse_glimmer::ModelInput {
                parts: &tp1_parts,
                vision: Some(muse_glimmer::VisionInput {
                    pixels: &pixels,
                    grid: &grid,
                }),
                mask: None,
            },
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Muse-Glimmer TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 2, "Muse-Glimmer TP1 state");
    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let collective_group = NumericParallelGroup::new(2);
    let outputs =
        std::thread::scope(|scope| {
            let handles =
                layouts
                    .into_iter()
                    .enumerate()
                    .map(|(rank, layout)| {
                        let args = args.clone();
                        let text_before = text_before.clone();
                        let media_tokens = media_tokens.clone();
                        let text_after = text_after.clone();
                        let pixels = pixels.clone();
                        let collective_group = Arc::clone(&collective_group);
                        scope.spawn(move || {
                            let context = NumericContext::with_local_layout(layout.clone());
                            let geometry = muse_glimmer::local_geometry(&args, &layout).unwrap();
                            let state_layout = geometry.state_layout().clone();
                            let architecture =
                                muse_glimmer::LayeredModel::<NumericBackend>::new_parallel(
                                    args, geometry, &context,
                                )
                                .unwrap();
                            let addresses = [(0, 0), (1, 0), (1, 1)];
                            let units = addresses.into_iter().map(|(group, index)| {
                    <muse_glimmer::LayeredModel<NumericBackend> as LayeredArchitecture<
                        NumericBackend,
                        DeviceState<NumericBackend, NumericHybridLayerState>,
                    >>::build_unit(&architecture, group, index, &context).unwrap()
                }).collect::<Vec<_>>();
                            let mut runtime =
                                LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                            let mut state = DeviceState::<NumericBackend, _>::create(
                                state_layout,
                                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                            )
                            .unwrap();
                            let parallel = NumericParallelContext::new(rank, collective_group);
                            let grid = [(1, 2, 2)];
                            let parts = [
                                muse_glimmer::DecoderInputPart::Text(&text_before),
                                muse_glimmer::DecoderInputPart::Media(&media_tokens),
                                muse_glimmer::DecoderInputPart::Text(&text_after),
                            ];
                            let logits = runtime
                                .forward_parallel(
                                    muse_glimmer::ModelInput {
                                        parts: &parts,
                                        vision: Some(muse_glimmer::VisionInput {
                                            pixels: &pixels,
                                            grid: &grid,
                                        }),
                                        mask: None,
                                    },
                                    &mut state,
                                    &parallel,
                                    &context,
                                )
                                .unwrap();
                            (logits, parallel.trace())
                        })
                    })
                    .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
    assert_tensor_close(&outputs[0].0, &expected, "Muse-Glimmer TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "Muse-Glimmer TP2 rank 1 logits");
    assert_eq!(outputs[0].0.shape, [1, 6, 7]);
    for (_, trace) in &outputs {
        assert_eq!(trace.len(), 7);
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::Sum,
                NumericCollectiveKind::GatherVocabulary,
            ]
        );
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
        input_transform: None,
        route_scale: None,
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
                gate_bias: None,
                up: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                up_bias: None,
                down: NumericTensor::new(vec![2, 1], vec![1.0, 2.0]),
                down_bias: None,
            },
            NumericExpert {
                gate: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                gate_bias: None,
                up: NumericTensor::new(vec![1, 2], vec![1.0, 0.0]),
                up_bias: None,
                down: NumericTensor::new(vec![2, 1], vec![2.0, -1.0]),
                down_bias: None,
            },
        ],
        parameters: Vec::new(),
        policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        spec: numeric_expert_bank_spec(2, 2, 1, eredu_nn::GatedProductPolicy::ordinary_silu()),
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

#[derive(Default)]
struct RecordingNumericExpertProvider {
    calls: Vec<(usize, ExpertPass, Vec<f32>, Vec<f32>)>,
}

impl RoutedExpertProvider<NumericBackend> for RecordingNumericExpertProvider {
    type Error = Error;

    fn forward_routed(
        &mut self,
        resident_bank: &mut NumericExpertBank,
        request: RoutedExpertRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Self::Error> {
        self.calls.push((
            request.layer,
            request.pass,
            request.routes.expert_ids.data.clone(),
            request.routes.route_weights.data.clone(),
        ));
        resident_bank.forward_routed(request.input, request.routes, context)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut NumericRelu2ExpertBank,
        request: RoutedExpertRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Self::Error> {
        resident_bank.forward_routed(request.input, request.routes, context)
    }
}

#[test]
fn external_expert_provider_preserves_route_order_weights_bias_and_telemetry() {
    let context = NumericContext::default();
    let mut bank = NumericExpertBank {
        experts: vec![
            NumericExpert {
                gate: NumericTensor::new(vec![1, 2], vec![1.0, 0.0]),
                gate_bias: Some(NumericTensor::new(vec![1], vec![0.25])),
                up: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                up_bias: Some(NumericTensor::new(vec![1], vec![0.5])),
                down: NumericTensor::new(vec![2, 1], vec![1.0, -2.0]),
                down_bias: Some(NumericTensor::new(vec![2], vec![3.0, -4.0])),
            },
            NumericExpert {
                gate: NumericTensor::new(vec![1, 2], vec![0.0, 1.0]),
                gate_bias: Some(NumericTensor::new(vec![1], vec![-0.5])),
                up: NumericTensor::new(vec![1, 2], vec![1.0, 0.0]),
                up_bias: Some(NumericTensor::new(vec![1], vec![-0.25])),
                down: NumericTensor::new(vec![2, 1], vec![-1.5, 0.75]),
                down_bias: Some(NumericTensor::new(vec![2], vec![-2.0, 5.0])),
            },
        ],
        parameters: Vec::new(),
        policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        spec: numeric_expert_bank_spec(2, 2, 1, eredu_nn::GatedProductPolicy::ordinary_silu()),
    };
    let input = NumericTensor::new(vec![2, 2], vec![2.0, 1.0, -1.0, 3.0]);
    let routes = RoutingResult {
        // Deliberately reverse the route order for the second token. The
        // provider receives architecture-selected order rather than sorting
        // by expert identity for its cache acquisition.
        expert_ids: NumericTensor::new(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]),
        selected_scores: NumericTensor::new(vec![2, 2], vec![0.8, 0.2, 0.6, 0.4]),
        route_weights: NumericTensor::new(vec![2, 2], vec![0.75, 0.25, 0.6, 0.4]),
    };
    let mut direct_bank = bank.clone();
    let expected = direct_bank
        .forward_routed(&input, &routes, &context)
        .unwrap();

    let mut provider = RecordingNumericExpertProvider::default();
    let actual = provider
        .forward_routed(
            &mut bank,
            RoutedExpertRequest {
                layer: 7,
                input: &input,
                routes: &routes,
                pass: ExpertPass::Prefill,
            },
            &context,
        )
        .unwrap();
    assert_tensor_close(&actual, &expected, "external routed expert output");
    assert_eq!(
        provider.calls,
        [(
            7,
            ExpertPass::Prefill,
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.75, 0.25, 0.6, 0.4],
        )]
    );

    // Down-projection biases are route-weighted exactly once. Removing them
    // must change the delegated result even though route IDs and weights are
    // unchanged.
    for expert in &mut direct_bank.experts {
        expert.down_bias = None;
    }
    let without_bias = direct_bank
        .forward_routed(&input, &routes, &context)
        .unwrap();
    assert_ne!(actual.data, without_bias.data);
}

#[test]
fn selected_softmax_router_applies_rms_input_and_per_expert_scales() {
    let weight = ParameterSpec::trainable("router.selected.weight").unwrap();
    let input_scale = ParameterSpec::trainable("router.selected.input_scale").unwrap();
    let route_scale = ParameterSpec::trainable("router.selected.route_scale").unwrap();
    let mut router = NumericRouter {
        linear: NumericLinear {
            weight: NumericTensor::new(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
            weight_metadata: ParameterMetadata::from_spec(&weight, true),
            bias: None,
        },
        routing: TopKRoutingSpec::new(3, 2, eredu_nn::RoutingScoring::SelectedSoftmax, false)
            .unwrap(),
        correction_bias: None,
        input_transform: Some((
            0.0,
            NumericTensor::new(vec![2], vec![2.0, 1.0]),
            ParameterMetadata::from_spec(&input_scale, true),
            true,
        )),
        route_scale: Some((
            NumericTensor::new(vec![3], vec![2.0, 3.0, 5.0]),
            ParameterMetadata::from_spec(&route_scale, true),
        )),
    };
    let routes = router
        .route(
            &NumericTensor::new(vec![1, 2], vec![3.0, 4.0]),
            &NumericContext::default(),
        )
        .unwrap();
    let first = 0.4_f32.exp() / (0.4_f32.exp() + 1.0);
    assert_eq!(routes.expert_ids.data, [0.0, 1.0]);
    assert_close(routes.selected_scores.data[0], first);
    assert_close(routes.selected_scores.data[1], 1.0 - first);
    assert_close(routes.route_weights.data[0], 2.0 * first);
    assert_close(routes.route_weights.data[1], 3.0 * (1.0 - first));
}

#[test]
fn selected_softmax_router_projection_bias_affects_ids_and_weights() {
    let mut router = NumericBackend::top_k_router(
        TopKRouterSpec {
            input_dimensions: 1,
            weight: ParameterSpec::trainable("router.biased.weight").unwrap(),
            bias: Some(ParameterSpec::trainable("router.biased.bias").unwrap()),
            correction_bias: Some(
                ParameterSpec::trainable("router.biased.correction_bias").unwrap(),
            ),
            input_transform: None,
            route_scale: None,
            format: dense_linear_format(),
            routing: TopKRoutingSpec::new(3, 2, eredu_nn::RoutingScoring::SelectedSoftmax, false)
                .unwrap(),
        },
        &NumericContext::default(),
    )
    .unwrap();
    assert_eq!(
        validate_parameter_topology::<NumericTensor, _>(&router)
            .unwrap()
            .into_iter()
            .map(|parameter| parameter.id.as_str().to_owned())
            .collect::<Vec<_>>(),
        [
            "router.biased.weight",
            "router.biased.bias",
            "router.biased.correction_bias",
        ]
    );
    router.linear.weight = NumericTensor::new(vec![3, 1], vec![0.0; 3]);
    router.linear.bias.as_mut().unwrap().0 =
        NumericTensor::new(vec![3], vec![0.0, 2.0_f32.ln(), 4.0_f32.ln()]);
    router.correction_bias.as_mut().unwrap().0 = NumericTensor::new(vec![3], vec![10.0, 0.0, 0.0]);

    let routes = router
        .route(
            &NumericTensor::new(vec![1, 1], vec![1.0]),
            &NumericContext::default(),
        )
        .unwrap();

    assert_eq!(routes.expert_ids.data, [0.0, 2.0]);
    assert_close(routes.selected_scores.data[0], 0.2);
    assert_close(routes.selected_scores.data[1], 0.8);
    assert_close(routes.route_weights.data[0], 0.2);
    assert_close(routes.route_weights.data[1], 0.8);
}

#[test]
fn routed_gated_experts_support_approximate_gelu() {
    let mut experts = NumericExpertBank {
        experts: vec![NumericExpert {
            gate: NumericTensor::new(vec![1, 1], vec![1.0]),
            gate_bias: None,
            up: NumericTensor::new(vec![1, 1], vec![2.0]),
            up_bias: None,
            down: NumericTensor::new(vec![1, 1], vec![3.0]),
            down_bias: None,
        }],
        parameters: Vec::new(),
        policy: eredu_nn::GatedProductPolicy::ordinary_gelu_approximate(),
        spec: numeric_expert_bank_spec(
            1,
            1,
            1,
            eredu_nn::GatedProductPolicy::ordinary_gelu_approximate(),
        ),
    };
    let routes = RoutingResult {
        expert_ids: NumericTensor::new(vec![1, 1], vec![0.0]),
        selected_scores: NumericTensor::new(vec![1, 1], vec![1.0]),
        route_weights: NumericTensor::new(vec![1, 1], vec![1.0]),
    };
    let output = experts
        .forward_routed(
            &NumericTensor::new(vec![1, 1], vec![1.0]),
            &routes,
            &NumericContext::default(),
        )
        .unwrap();
    let gelu = 0.5 * (1.0 + (0.797_884_6_f32 * 1.044_715).tanh());
    assert_close(output.data[0], gelu * 2.0 * 3.0);
}

#[test]
fn gated_product_policy_and_projection_biases_match_analytical_value() {
    let parameter = |name: &str| ParameterSpec::trainable(name).unwrap();
    let policy = eredu_nn::GatedProductPolicy::new(
        eredu_nn::GatedProductActivation::Silu,
        Some(2.0),
        Some(1.5),
        1.7,
        1.0,
    )
    .unwrap();
    let mut bank = NumericBackend::gated_product_expert_bank(
        GatedProductExpertBankSpec {
            expert_count: 1,
            input_dimensions: 1,
            intermediate_dimensions: 1,
            output_dimensions: 1,
            policy,
            layout: GatedProductExpertLayout::Packed {
                gate_up: eredu_nn::ExpertProjectionSpec {
                    weight: parameter("experts.gate_up_proj"),
                    bias: Some(parameter("experts.gate_up_proj_bias")),
                    format: dense_linear_format(),
                },
                down: eredu_nn::ExpertProjectionSpec {
                    weight: parameter("experts.down_proj"),
                    bias: Some(parameter("experts.down_proj_bias")),
                    format: dense_linear_format(),
                },
            },
        },
        &NumericContext::default(),
    )
    .unwrap();
    let topology = validate_parameter_topology::<NumericTensor, _>(&bank)
        .unwrap()
        .into_iter()
        .map(|parameter| parameter.id.as_str().to_owned())
        .collect::<Vec<_>>();
    assert!(topology.contains(&"experts.gate_up_proj_bias".to_owned()));
    assert!(topology.contains(&"experts.down_proj_bias".to_owned()));

    bank.experts[0].gate = NumericTensor::new(vec![1, 1], vec![0.0]);
    bank.experts[0].gate_bias = Some(NumericTensor::new(vec![1], vec![2.5]));
    bank.experts[0].up = NumericTensor::new(vec![1, 1], vec![0.0]);
    bank.experts[0].up_bias = Some(NumericTensor::new(vec![1], vec![-3.0]));
    bank.experts[0].down = NumericTensor::new(vec![1, 1], vec![2.0]);
    bank.experts[0].down_bias = Some(NumericTensor::new(vec![1], vec![5.0]));
    let routes = RoutingResult {
        expert_ids: NumericTensor::new(vec![1, 1], vec![0.0]),
        selected_scores: NumericTensor::new(vec![1, 1], vec![0.25]),
        route_weights: NumericTensor::new(vec![1, 1], vec![0.25]),
    };
    let output = bank
        .forward_routed(
            &NumericTensor::new(vec![1, 1], vec![7.0]),
            &routes,
            &NumericContext::default(),
        )
        .unwrap();
    let gate = 2.0 / (1.0 + (-3.4_f32).exp());
    let expected = 0.25 * (2.0 * gate * -0.5 + 5.0);
    assert_close(output.data[0], expected);
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
                format: dense_linear_format(),
            }),
            normalization: NormalizationConstructionSpec::learned(
                2,
                1e-6,
                parameter("low_rank.norm.weight"),
            ),
            second: LinearSpec {
                input: 2,
                output: 1,
                weight: parameter("low_rank.second.weight"),
                bias: None,
                format: dense_linear_format(),
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
    assert_close(output.data[0], 2.0 / (12.5_f32 + 1e-6).sqrt());
}

#[test]
fn causal_depthwise_convolution_matches_prefill_and_incremental_reference() {
    let parameter = |name: &str| ParameterSpec::trainable(name).unwrap();
    let mut convolution = CausalDepthwiseConvolution::<NumericBackend>::new(
        CausalDepthwiseConvolutionSpec {
            channels: 2,
            kernel_size: 3,
            weight: parameter("conv.weight"),
            bias: Some(parameter("conv.bias")),
            activation: ConvolutionActivation::Identity,
        },
        &NumericContext::default(),
    )
    .unwrap();

    struct Loader;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Loader {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            value.data = match metadata.id.as_str() {
                "conv.weight" => vec![1.0, 2.0, 1.0, 1.0, 0.0, -1.0],
                "conv.bias" => vec![0.5, -0.5],
                unexpected => panic!("unexpected convolution parameter {unexpected}"),
            };
        }
    }
    convolution.visit_parameters_mut(&mut Loader);

    let prefill = convolution
        .forward(
            &NumericTensor::new(
                vec![1, 4, 2],
                vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            ),
            None,
            &NumericContext::default(),
        )
        .unwrap();
    assert_eq!(
        prefill.output.data,
        [1.5, -10.5, 4.5, -20.5, 8.5, -20.5, 12.5, -20.5]
    );
    assert_eq!(
        prefill.history.as_ref().unwrap().data,
        [3.0, 30.0, 4.0, 40.0]
    );

    let decode = convolution
        .forward(
            &NumericTensor::new(vec![1, 1, 2], vec![5.0, 50.0]),
            prefill.history.as_ref(),
            &NumericContext::default(),
        )
        .unwrap();
    assert_eq!(decode.output.data, [16.5, -20.5]);
    assert_eq!(
        decode.history.as_ref().unwrap().data,
        [4.0, 40.0, 5.0, 50.0]
    );
}

#[test]
fn gated_short_convolution_matches_chunked_state_continuation() {
    let parameter = |name: &str| ParameterSpec::trainable(name).unwrap();
    let linear = |input, output, name: &str| LinearSpec {
        input,
        output,
        weight: parameter(name),
        bias: None,
        format: dense_linear_format(),
    };
    let spec = GatedShortConvolutionSpec {
        input_dimensions: 2,
        channels: 2,
        output_dimensions: 2,
        input_projection: linear(2, 6, "short.in.weight"),
        output_projection: linear(2, 2, "short.out.weight"),
        convolution: CausalDepthwiseConvolutionSpec {
            channels: 2,
            kernel_size: 3,
            weight: parameter("short.conv.weight"),
            bias: Some(parameter("short.conv.bias")),
            activation: ConvolutionActivation::Identity,
        },
    };
    let mut unbiased_spec = spec.clone();
    unbiased_spec.convolution.bias = None;
    let context = NumericContext::default();
    let mut whole = GatedShortConvolution::<NumericBackend>::new(spec.clone(), &context).unwrap();
    let mut chunked = GatedShortConvolution::<NumericBackend>::new(spec, &context).unwrap();

    struct Loader;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Loader {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            value.data = match metadata.id.as_str() {
                // B=x, C swaps the two channels, and the final projection is identity.
                "short.in.weight" => {
                    vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
                }
                "short.out.weight" => vec![1.0, 0.0, 0.0, 1.0],
                "short.conv.weight" => vec![0.25, 0.5, 1.0, -0.5, 0.25, 1.0],
                "short.conv.bias" => vec![0.1, -0.2],
                unexpected => panic!("unexpected gated short-convolution parameter {unexpected}"),
            };
        }
    }
    whole.visit_parameters_mut(&mut Loader);
    chunked.visit_parameters_mut(&mut Loader);

    let input = NumericTensor::new(
        vec![1, 5, 2],
        vec![0.5, -1.0, 1.5, 0.25, -0.75, 2.0, 1.0, 1.25, -2.0, 0.5],
    );
    let expected = whole.forward(&input, None, &context).unwrap();
    let mut history = None;
    let mut outputs = Vec::new();
    for (start, end) in [(0, 2), (2, 4), (4, 5)] {
        let next = chunked
            .forward(&input.axis_slice(1, start, end), history.as_ref(), &context)
            .unwrap();
        outputs.push(next.output);
        history = next.history;
    }
    let actual = NumericTensor::concatenate(&outputs, 1, &context).unwrap();
    assert_eq!(actual.shape, expected.output.shape);
    for (index, (expected, actual)) in expected.output.data.iter().zip(&actual.data).enumerate() {
        assert!(
            (*expected - *actual).abs() <= 1.0e-6,
            "gated short-convolution output {index}: expected {expected}, got {actual}"
        );
    }
    assert_eq!(history.unwrap().data, expected.history.unwrap().data);

    let mut unbiased_whole =
        GatedShortConvolution::<NumericBackend>::new(unbiased_spec.clone(), &context).unwrap();
    let mut unbiased_chunked =
        GatedShortConvolution::<NumericBackend>::new(unbiased_spec, &context).unwrap();
    unbiased_whole.visit_parameters_mut(&mut Loader);
    unbiased_chunked.visit_parameters_mut(&mut Loader);
    let expected = unbiased_whole.forward(&input, None, &context).unwrap();
    let first = unbiased_chunked
        .forward(&input.axis_slice(1, 0, 3), None, &context)
        .unwrap();
    let second = unbiased_chunked
        .forward(&input.axis_slice(1, 3, 5), first.history.as_ref(), &context)
        .unwrap();
    let actual = NumericTensor::concatenate(&[first.output, second.output], 1, &context).unwrap();
    assert_tensor_close(
        &actual,
        &expected.output,
        "unbiased gated short convolution",
    );
    assert_eq!(second.history.unwrap().data, expected.history.unwrap().data);
}

#[test]
fn lfm2_mixed_schedule_advances_attention_and_fixed_state_together() {
    let args = lfm2::model_args_from_config_value(&serde_json::json!({
        "model_type": "lfm2",
        "vocab_size": 16,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
        "layer_types": ["conv", "full_attention", "conv"],
        "conv_L_cache": 3,
        "block_multiple_of": 2,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": true,
        "tie_word_embeddings": true
    }))
    .unwrap();
    let layout = lfm2::state_layout(&args).unwrap();
    let mut state = DeviceState::<NumericBackend, _>::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let context = NumericContext::default();
    let model = lfm2::LayeredModel::<NumericBackend>::new(args, &context).unwrap();
    let mut runtime = ResidentRuntime::new(model, &context).unwrap();

    for tokens in [
        NumericTensor::token_ids(&[1, 3, 2]),
        NumericTensor::token_ids(&[4]),
    ] {
        let output = runtime
            .forward(
                eredu_architectures::decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut state,
                &context,
            )
            .unwrap();
        assert_eq!(output.shape, [1, tokens.dim(1), 16]);
        assert!(output.data.iter().all(|value| value.is_finite()));
    }

    for layer in 0..3 {
        assert_eq!(state.layer(layer).unwrap().position(), 4);
    }
    for layer in [0, 2] {
        let history = state
            .layer(layer)
            .unwrap()
            .fixed_component(StateTensorRole::Convolution { slot: 0 })
            .unwrap()
            .as_ref()
            .unwrap();
        assert_eq!(history.shape, [1, 2, 4]);
    }
}

fn assert_lfm2_tp2_mixed_state_matches_replicated_and_rolls_back_invalid_tokens(sparse: bool) {
    let mut config = serde_json::json!({
        "model_type":"lfm2", "vocab_size":7, "hidden_size":8,
        "intermediate_size":10, "num_hidden_layers":2,
        "num_attention_heads":4, "num_key_value_heads":2,
        "max_position_embeddings":32, "layer_types":["conv","full_attention"],
        "conv_L_cache":3, "block_multiple_of":2,
        "block_ffn_dim_multiplier":1.0, "block_auto_adjust_ff_dim":true,
        "tie_word_embeddings":false
    });
    if sparse {
        config["model_type"] = "lfm2_moe".into();
        config["num_dense_layers"] = 1.into();
        config["moe_intermediate_size"] = 6.into();
        config["num_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
    }
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let architecture = lfm2::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = lfm2::static_parallel_parameter_groups(architecture.static_modules()).unwrap();
    for layer in 0..2 {
        let unit = <lfm2::LayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::build_unit(&architecture, 0, layer, &context)
        .unwrap();
        groups.extend(lfm2::layer_parallel_parameter_groups(&unit, &args, layer).unwrap());
    }
    let units = (0..2)
        .map(|layer| {
            <lfm2::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&architecture, 0, layer, &context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();
    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = lfm2::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = lfm2::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = (0..2)
        .map(|layer| {
            <lfm2::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, 0, layer, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "LFM2 TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 2, "LFM2 TP1 state");
    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let group = NumericParallelGroup::new(2);
    let mut outputs = std::thread::scope(|scope| {
        let handles = layouts
            .into_iter()
            .enumerate()
            .map(|(rank, layout)| {
                let args = args.clone();
                let tokens = tokens.clone();
                let group = Arc::clone(&group);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = lfm2::local_geometry(&args, &layout).unwrap();
                    let state_layout = geometry.state_layout().clone();
                    let architecture = lfm2::LayeredModel::<NumericBackend>::new_parallel(
                        args, geometry, &context,
                    )
                    .unwrap();
                    let units = (0..2)
                        .map(|layer| {
                            <lfm2::LayeredModel<NumericBackend> as LayeredArchitecture<
                                NumericBackend,
                                DeviceState<NumericBackend, NumericHybridLayerState>,
                            >>::build_unit(
                                &architecture, 0, layer, &context
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let parallel = NumericParallelContext::new(rank, group);
                    let logits = runtime
                        .forward_parallel(
                            decoder::LayeredInput {
                                tokens: &tokens,
                                mask: None,
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    let positions = [
                        state.layer(0).unwrap().position(),
                        state.layer(1).unwrap().position(),
                    ];
                    let bad = NumericTensor::token_ids(&[7]);
                    assert!(runtime
                        .forward_parallel(
                            decoder::LayeredInput {
                                tokens: &bad,
                                mask: None
                            },
                            &mut state,
                            &parallel,
                            &context,
                        )
                        .is_err());
                    assert_eq!(
                        [
                            state.layer(0).unwrap().position(),
                            state.layer(1).unwrap().position()
                        ],
                        positions,
                    );
                    (logits, parallel.trace(), state)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(&outputs[0].0, &expected, "LFM2 TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "LFM2 TP2 rank 1 logits");
    assert_eq!(outputs[0].1.len(), 6);
    assert_eq!(outputs[1].1.len(), 6);
    assert_eq!(outputs[0].1.last().unwrap().output_shape, [1, 3, 7]);
    for (_, _, state) in &mut outputs {
        assert_eq!(state.layer(0).unwrap().position(), 3);
        assert_eq!(state.layer(1).unwrap().position(), 3);
    }
}

#[test]
fn lfm2_and_lfm2_moe_tp2_mixed_state_match_replicated_and_rollback_invalid_tokens() {
    assert_lfm2_tp2_mixed_state_matches_replicated_and_rolls_back_invalid_tokens(false);
    assert_lfm2_tp2_mixed_state_matches_replicated_and_rolls_back_invalid_tokens(true);
}

fn assert_tensor_close(actual: &NumericTensor, expected: &NumericTensor, label: &str) {
    assert_eq!(actual.shape, expected.shape, "{label} shape");
    for (index, (actual, expected)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 2.0e-4,
            "{label}[{index}]: expected {expected}, got {actual}"
        );
    }
}

fn assert_tensor_exact(actual: &NumericTensor, expected: &NumericTensor, label: &str) {
    assert_eq!(actual.shape, expected.shape, "{label} shape");
    assert_eq!(actual.data, expected.data, "{label} values");
}

fn assert_retained_state_exact<S>(
    actual: &DeviceState<NumericBackend, S>,
    expected: &DeviceState<NumericBackend, S>,
    layers: usize,
    label: &str,
) where
    S: RuntimeLayerState<NumericBackend>,
{
    for layer in 0..layers {
        let actual = actual.as_ref().get(layer).unwrap();
        let expected = expected.as_ref().get(layer).unwrap();
        let actual_values = RuntimeLayerState::retained_values(actual).collect::<Vec<_>>();
        let expected_values = RuntimeLayerState::retained_values(expected).collect::<Vec<_>>();
        assert_eq!(
            actual_values.len(),
            expected_values.len(),
            "{label} layer {layer} retained tensor count"
        );
        for (index, (actual, expected)) in
            actual_values.into_iter().zip(expected_values).enumerate()
        {
            assert_tensor_exact(
                actual,
                expected,
                &format!("{label} layer {layer} retained tensor {index}"),
            );
        }
    }
}

fn assert_state_exact(
    actual: &DeviceState<NumericBackend, NumericHybridLayerState>,
    expected: &DeviceState<NumericBackend, NumericHybridLayerState>,
    layers: usize,
    label: &str,
) {
    assert_retained_state_exact(actual, expected, layers, label);
    for layer in 0..layers {
        let actual = &actual.as_ref()[layer];
        let expected = &expected.as_ref()[layer];
        assert_eq!(
            actual.position(),
            expected.position(),
            "{label} layer {layer} position"
        );
        assert_eq!(
            actual.fixed_offset, expected.fixed_offset,
            "{label} layer {layer} fixed offset"
        );
        assert_eq!(
            actual.resets, expected.resets,
            "{label} layer {layer} reset count"
        );
    }
}

#[test]
fn kimi_linear_mixed_kda_mla_prefill_decode_uses_one_heterogeneous_state() {
    let args = kimi_linear::model_args_from_config_value(&serde_json::json!({
        "model_type":"kimi_linear", "vocab_size":16, "hidden_size":12,
        "num_hidden_layers":2, "num_attention_heads":3, "num_key_value_heads":3,
        "intermediate_size":17, "head_dim":4, "model_max_length":64,
        "linear_attn_config":{
            "kda_layers":[1], "full_attn_layers":[2], "num_heads":3,
            "head_dim":4, "short_conv_kernel_size":3
        },
        "num_experts":2, "moe_intermediate_size":9, "kv_lora_rank":6,
        "qk_nope_head_dim":4, "qk_rope_head_dim":2, "v_head_dim":4,
        "mla_use_nope":true, "num_experts_per_token":1, "num_shared_experts":1,
        "routed_scaling_factor":1.0, "first_k_dense_replace":1,
        "num_expert_group":1, "topk_group":1, "tie_word_embeddings":false
    }))
    .unwrap();
    let new_state = || {
        DeviceState::<NumericBackend, _>::create(
            kimi_linear::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap()
    };
    let mut state = new_state();
    let mut whole_state = new_state();
    let context = NumericContext::default();
    let model = kimi_linear::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut runtime = ResidentRuntime::new(model, &context).unwrap();

    let mut chunked_logits = Vec::new();
    for tokens in [
        NumericTensor::token_ids(&[1, 3, 2]),
        NumericTensor::token_ids(&[4]),
    ] {
        let logits = runtime
            .forward(
                eredu_architectures::decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut state,
                &context,
            )
            .unwrap();
        assert_eq!(logits.shape, [1, tokens.dim(1), 16]);
        assert!(logits.data.iter().all(|value| value.is_finite()));
        chunked_logits.push(logits);
    }

    let mut whole = ResidentRuntime::new(
        kimi_linear::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        &context,
    )
    .unwrap();
    let whole_logits = whole
        .forward(
            eredu_architectures::decoder::LayeredInput {
                tokens: &NumericTensor::token_ids(&[1, 3, 2, 4]),
                mask: None,
            },
            &mut whole_state,
            &context,
        )
        .unwrap();
    let chunked_logits = NumericTensor::concatenate(&chunked_logits, 1, &context).unwrap();
    assert_tensor_close(&chunked_logits, &whole_logits, "Kimi chunked target logits");

    assert_eq!(state.layer(0).unwrap().position(), 4);
    for slot in 0..3 {
        let history = state
            .layer(0)
            .unwrap()
            .fixed_component(StateTensorRole::Convolution { slot })
            .unwrap()
            .as_ref()
            .unwrap();
        assert_eq!(history.shape[0], 1);
        assert_eq!(history.shape[1], 2);
    }
    assert!(state
        .layer(0)
        .unwrap()
        .fixed_component(StateTensorRole::Recurrent)
        .unwrap()
        .is_some());
    assert_eq!(state.layer(1).unwrap().position(), 4);
    assert_eq!(
        state.layer(1).unwrap().compressed.as_ref().unwrap().offset,
        4
    );
    for layer in 0..2 {
        assert_eq!(
            state.layer(layer).unwrap().position(),
            whole_state.layer(layer).unwrap().position()
        );
    }
}

#[test]
fn kimi_tp2_kda_mla_matches_replicated_and_rolls_back_invalid_tokens() {
    let args = kimi_linear::model_args_from_config_value(&serde_json::json!({
        "model_type":"kimi_linear", "vocab_size":7, "hidden_size":8,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":2,
        "intermediate_size":10, "head_dim":4, "model_max_length":64,
        "linear_attn_config":{
            "kda_layers":[1], "full_attn_layers":[2], "num_heads":2,
            "head_dim":4, "short_conv_kernel_size":3
        },
        "num_experts":2, "moe_intermediate_size":6, "kv_lora_rank":4,
        "qk_nope_head_dim":4, "qk_rope_head_dim":2, "v_head_dim":4,
        "mla_use_nope":true, "num_experts_per_token":1, "num_shared_experts":1,
        "routed_scaling_factor":1.0, "first_k_dense_replace":1,
        "num_expert_group":1, "topk_group":1, "tie_word_embeddings":false
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        kimi_linear::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups =
        kimi_linear::static_parallel_parameter_groups(architecture.static_modules()).unwrap();
    for layer in 0..2 {
        let unit = <kimi_linear::LayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::build_unit(&architecture, 0, layer, &context)
        .unwrap();
        groups.extend(kimi_linear::layer_parallel_parameter_groups(&unit, &args, layer).unwrap());
    }
    let units = (0..2)
        .map(|layer| {
            <kimi_linear::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&architecture, 0, layer, &context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let expected = expected_runtime
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut expected_state,
            &context,
        )
        .unwrap();
    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = kimi_linear::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = kimi_linear::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = (0..2)
        .map(|layer| {
            <kimi_linear::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, 0, layer, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let tp1_logits = tp1_runtime
        .forward_parallel(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_logits, &expected, "Kimi TP1 logits");
    assert_state_exact(&tp1_state, &expected_state, 2, "Kimi TP1 state");
    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let group = NumericParallelGroup::new(2);
    let mut outputs =
        std::thread::scope(|scope| {
            let handles =
                layouts
                    .into_iter()
                    .enumerate()
                    .map(|(rank, layout)| {
                        let args = args.clone();
                        let tokens = tokens.clone();
                        let group = Arc::clone(&group);
                        scope.spawn(move || {
                            let context = NumericContext::with_local_layout(layout.clone());
                            let geometry = kimi_linear::local_geometry(&args, &layout).unwrap();
                            let state_layout = geometry.state_layout().clone();
                            let architecture =
                                kimi_linear::LayeredModel::<NumericBackend>::new_parallel(
                                    args, geometry, &context,
                                )
                                .unwrap();
                            let units = (0..2).map(|layer| {
                    <kimi_linear::LayeredModel<NumericBackend> as LayeredArchitecture<
                        NumericBackend,
                        DeviceState<NumericBackend, NumericHybridLayerState>,
                    >>::build_unit(&architecture, 0, layer, &context).unwrap()
                }).collect::<Vec<_>>();
                            let mut runtime =
                                LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                            let mut state = DeviceState::<NumericBackend, _>::create(
                                state_layout,
                                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                            )
                            .unwrap();
                            let parallel = NumericParallelContext::new(rank, group);
                            let logits = runtime
                                .forward_parallel(
                                    decoder::LayeredInput {
                                        tokens: &tokens,
                                        mask: None,
                                    },
                                    &mut state,
                                    &parallel,
                                    &context,
                                )
                                .unwrap();
                            let positions = [
                                state.layer(0).unwrap().position(),
                                state.layer(1).unwrap().position(),
                            ];
                            let bad = NumericTensor::token_ids(&[7]);
                            assert!(runtime
                                .forward_parallel(
                                    decoder::LayeredInput {
                                        tokens: &bad,
                                        mask: None
                                    },
                                    &mut state,
                                    &parallel,
                                    &context,
                                )
                                .is_err());
                            assert_eq!(
                                [
                                    state.layer(0).unwrap().position(),
                                    state.layer(1).unwrap().position()
                                ],
                                positions,
                            );
                            (logits, parallel.trace(), state)
                        })
                    })
                    .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
    assert_tensor_close(&outputs[0].0, &expected, "Kimi TP2 rank 0 logits");
    assert_tensor_close(&outputs[1].0, &expected, "Kimi TP2 rank 1 logits");
    assert_eq!(outputs[0].1.len(), 7);
    assert_eq!(outputs[1].1.len(), 7);
    assert_eq!(
        outputs[0]
            .1
            .iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>(),
        [
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::Sum,
            NumericCollectiveKind::GatherVocabulary,
        ]
    );
    assert_eq!(outputs[0].1.last().unwrap().output_shape, [1, 3, 7]);
    for (_, _, state) in &mut outputs {
        assert_eq!(state.layer(0).unwrap().position(), 3);
        assert_eq!(state.layer(1).unwrap().position(), 3);
    }
}

#[derive(Default)]
struct TypedRelu2ProviderProbe {
    replicated_calls: usize,
    tensor_parallel_partitions: Vec<usize>,
}

impl RoutedExpertProvider<NumericBackend> for TypedRelu2ProviderProbe {
    type Error = std::convert::Infallible;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut NumericExpertBank,
        request: RoutedExpertRequest<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Self::Error> {
        Ok(request.input.clone())
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut NumericRelu2ExpertBank,
        request: RoutedExpertRequest<'_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Self::Error> {
        self.replicated_calls += 1;
        Ok(request.input.clone())
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut NumericRelu2ExpertBank,
        request: RoutedExpertRequest<'_, NumericTensor>,
        partitions: usize,
        _: &NumericContext,
    ) -> Result<RoutedExpertTensorParallelOutput<NumericTensor>, Self::Error> {
        self.tensor_parallel_partitions.push(partitions);
        Ok(RoutedExpertTensorParallelOutput::Partial(
            TensorParallelExpertOutput {
                reducible: request.input.clone(),
                post_reduce: None,
            },
        ))
    }
}

#[test]
fn nemotron_h_relu2_provider_uses_typed_tensor_parallel_results() {
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":16, "hidden_size":8,
        "intermediate_size":12, "num_hidden_layers":1,
        "hybrid_override_pattern":"E", "num_attention_heads":2,
        "num_key_value_heads":1, "head_dim":4, "mamba_num_heads":2,
        "n_groups":1, "mamba_head_dim":4, "ssm_state_size":2,
        "conv_kernel":2, "chunk_size":2, "n_routed_experts":4,
        "n_shared_experts":1, "moe_intermediate_size":4,
        "moe_shared_expert_intermediate_size":4, "num_experts_per_tok":2,
        "n_group":2, "topk_group":1, "tie_word_embeddings":false
    }))
    .unwrap();
    let input = NumericTensor::new([1, args.hidden_size], vec![0.25; args.hidden_size as usize]);
    let collective_group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|rank| {
                let args = args.clone();
                let input = input.clone();
                let collective_group = Arc::clone(&collective_group);
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut moe = nemotron_h::SparseMoe::<NumericBackend>::new(
                        &args,
                        0,
                        args.moe_intermediate_size,
                        args.moe_shared_expert_intermediate_size,
                        &context,
                    )
                    .unwrap();
                    let mut provider = TypedRelu2ProviderProbe::default();
                    let parallel = NumericParallelContext::new(rank, collective_group);
                    let output = moe
                        .forward_parallel_with_provider(
                            &input,
                            ExpertPass::Decode,
                            &parallel,
                            &context,
                            &mut provider,
                        )
                        .unwrap();
                    (output, provider, parallel.trace())
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_tensor_close(&outputs[0].0, &outputs[1].0, "TP+EP provider output");
    for (output, provider, trace) in outputs {
        assert_eq!(output.shape, input.shape);
        assert_eq!(provider.replicated_calls, 0);
        assert_eq!(provider.tensor_parallel_partitions, [2]);
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [NumericCollectiveKind::Sum, NumericCollectiveKind::Sum]
        );
        assert!(trace.iter().all(|event| event.input_shape == input.shape));
    }
}

#[test]
fn nemotron_h_chunked_target_and_mtp_transactions_are_backend_neutral() {
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
        "intermediate_size":24, "num_hidden_layers":4,
        "hybrid_override_pattern":"M*-E", "num_attention_heads":4,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":4,
        "n_shared_experts":1, "moe_intermediate_size":8,
        "moe_shared_expert_intermediate_size":8, "num_experts_per_tok":2,
        "n_group":2, "topk_group":1, "num_nextn_predict_layers":1,
        "mtp_hybrid_override_pattern":"*E", "tie_word_embeddings":false,
        "residual_in_fp32":true
    }))
    .unwrap();
    let context = NumericContext::default();
    let new_state = || {
        DeviceState::<NumericBackend, _>::create(
            nemotron_h::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap()
    };
    let mut whole_state = new_state();
    let mut chunked_state = new_state();
    let mut whole = ResidentRuntime::new(
        nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut chunked = ResidentRuntime::new(
        nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let tokens = NumericTensor::token_ids(&[2, 5, 7]);
    let expected = whole
        .forward(
            nemotron_h::EmbeddedInput::target(&tokens, None),
            &mut whole_state,
            &context,
        )
        .unwrap();
    let first_tokens = NumericTensor::token_ids(&[2, 5]);
    let second_tokens = NumericTensor::token_ids(&[7]);
    let first = chunked
        .forward(
            nemotron_h::EmbeddedInput::target(&first_tokens, None),
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let second = chunked
        .forward(
            nemotron_h::EmbeddedInput::target(&second_tokens, None),
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let actual = NumericTensor::concatenate(&[first, second], 1, &context).unwrap();
    assert_tensor_close(&actual, &expected, "Nemotron-H chunked target logits");
    for (layer, expected_position) in [3, 3, 0, 0].into_iter().enumerate() {
        assert_eq!(
            whole_state.layer(layer).unwrap().position(),
            expected_position
        );
        assert_eq!(
            chunked_state.layer(layer).unwrap().position(),
            expected_position
        );
    }
    for layer in 4..6 {
        assert_eq!(chunked_state.layer(layer).unwrap().position(), 0);
    }

    let checkpoint = chunked_state.clone();
    let mut rollback = eredu_runtime::DraftStateTransaction::fork(&checkpoint);
    let prior = NumericTensor::new(vec![1, 1, 16], (0..16).map(|i| i as f32 / 16.0).collect());
    let draft_token = NumericTensor::token_ids(&[11]);
    let draft_logits = chunked
        .forward(
            nemotron_h::EmbeddedInput::draft(&draft_token, &prior, 0),
            rollback.draft_mut(),
            &context,
        )
        .unwrap();
    assert_eq!(draft_logits.shape, [1, 1, 32]);
    assert_eq!(rollback.draft_mut().layer(4).unwrap().position(), 1);
    assert_eq!(rollback.draft_mut().layer(5).unwrap().position(), 0);
    let mut canonical = checkpoint.clone();
    rollback.rollback(&mut canonical);
    assert_eq!(canonical.layer(4).unwrap().position(), 0);
    assert_eq!(canonical.layer(5).unwrap().position(), 0);

    let mut commit = eredu_runtime::DraftStateTransaction::fork(&checkpoint);
    chunked
        .forward(
            nemotron_h::EmbeddedInput::draft(&draft_token, &prior, 0),
            commit.draft_mut(),
            &context,
        )
        .unwrap();
    commit.commit_draft(&mut canonical);
    assert_eq!(canonical.layer(4).unwrap().position(), 1);
    assert_eq!(canonical.layer(5).unwrap().position(), 0);
    for (layer, expected_position) in [3, 3, 0, 0].into_iter().enumerate() {
        assert_eq!(
            canonical.layer(layer).unwrap().position(),
            expected_position
        );
    }
}

#[test]
fn real_prediction_graph_matches_resident_and_dense_streamed_traversal() {
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":7, "hidden_size":8,
        "intermediate_size":10, "num_hidden_layers":1,
        "hybrid_override_pattern":"*", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":2,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":2,
        "n_shared_experts":1, "moe_intermediate_size":4,
        "moe_shared_expert_intermediate_size":4, "num_experts_per_tok":1,
        "n_group":1, "topk_group":1, "num_nextn_predict_layers":1,
        "mtp_hybrid_override_pattern":"*", "tie_word_embeddings":false
    }))
    .unwrap();
    let context = NumericContext::default();
    let layout = nemotron_h::state_layout(&args).unwrap();
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let mut resident_state = make_state();
    let mut streamed_state = make_state();
    let mut resident = ResidentRuntime::new(
        nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut streamed = LayerwiseRuntime::new(
        nemotron_h::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        RebuildingUnitPolicy::default(),
    );
    let target_tokens = NumericTensor::token_ids(&[1, 4, 2]);
    let (resident_target, resident_forward) = resident
        .forward_with_context(
            nemotron_h::EmbeddedInput::target(&target_tokens, None),
            &mut resident_state,
            &context,
        )
        .unwrap();
    let (streamed_target, streamed_forward) = streamed
        .forward_with_context_hook(
            nemotron_h::EmbeddedInput::target(&target_tokens, None),
            &mut streamed_state,
            &context,
            |_, _, _| Ok(()),
        )
        .unwrap();
    assert_tensor_close(
        &streamed_target,
        &resident_target,
        "Nemotron target resident/dense-stream logits",
    );
    assert_eq!(streamed.policy().events, expected_stream_events(&[(0, 0)]));
    assert_eq!(resident_state.layer(0).unwrap().position(), 3);
    assert_eq!(streamed_state.layer(0).unwrap().position(), 3);
    assert_eq!(resident_state.layer(1).unwrap().position(), 0);
    assert_eq!(streamed_state.layer(1).unwrap().position(), 0);

    let resident_prior = resident_forward
        .target_capture()
        .unwrap()
        .axis_slice(1, 2, 3);
    let streamed_prior = streamed_forward
        .target_capture()
        .unwrap()
        .axis_slice(1, 2, 3);
    assert_tensor_close(
        &streamed_prior,
        &resident_prior,
        "Nemotron target capture resident/dense-stream",
    );
    let draft_token = NumericTensor::token_ids(&[3]);
    let resident_draft = resident
        .forward(
            nemotron_h::EmbeddedInput::draft(&draft_token, &resident_prior, 0),
            &mut resident_state,
            &context,
        )
        .unwrap();
    let streamed_draft = streamed
        .forward(
            nemotron_h::EmbeddedInput::draft(&draft_token, &streamed_prior, 0),
            &mut streamed_state,
            &context,
        )
        .unwrap();
    assert_tensor_close(
        &streamed_draft,
        &resident_draft,
        "Nemotron prediction resident/dense-stream logits",
    );
    let mut events = expected_stream_events(&[(0, 0)]);
    events.extend([
        StreamPolicyEvent::Begin,
        StreamPolicyEvent::Acquire(1, 1, 0),
        StreamPolicyEvent::Complete(1, 1, 0),
        StreamPolicyEvent::Finish,
    ]);
    assert_eq!(streamed.policy().events, events);
    assert_eq!(resident_state.layer(0).unwrap().position(), 3);
    assert_eq!(streamed_state.layer(0).unwrap().position(), 3);
    assert_eq!(resident_state.layer(1).unwrap().position(), 1);
    assert_eq!(streamed_state.layer(1).unwrap().position(), 1);
}

#[test]
fn nemotron_tp2_target_mtp_matches_replicated_and_rolls_back_draft_state() {
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":7, "hidden_size":8,
        "intermediate_size":10, "num_hidden_layers":3,
        "hybrid_override_pattern":"M*-", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":2,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":2,
        "n_shared_experts":1, "moe_intermediate_size":4,
        "moe_shared_expert_intermediate_size":4, "num_experts_per_tok":1,
        "n_group":1, "topk_group":1, "num_nextn_predict_layers":1,
        "mtp_hybrid_override_pattern":"*", "tie_word_embeddings":false
    }))
    .unwrap();
    let context = NumericContext::default();
    let architecture =
        nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let parameter_description = architecture.parameter_description(&context).unwrap();
    let groups = parameter_description
        .groups()
        .iter()
        .map(|owned| owned.group().clone())
        .collect::<Vec<_>>();
    let addresses: [(usize, usize); 4] = (0..parameter_description.unit_layout().len())
        .map(|ordinal| {
            let address = parameter_description
                .unit_layout()
                .address(ordinal)
                .unwrap();
            (address.group(), address.index())
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    assert_eq!(addresses, [(0, 0), (0, 1), (0, 2), (1, 0)]);
    let mut expected_state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut expected_runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let tokens = NumericTensor::token_ids(&[0, 4, 6]);
    let (expected_target, expected_forward) = expected_runtime
        .forward_with_context(
            nemotron_h::EmbeddedInput::target(&tokens, None),
            &mut expected_state,
            &context,
        )
        .unwrap();
    let prior = expected_forward
        .target_capture()
        .unwrap()
        .axis_slice(1, 2, 3);
    let draft_token = NumericTensor::token_ids(&[3]);
    let mut expected_draft_state = eredu_runtime::DraftStateTransaction::fork(&expected_state);
    let expected_draft = expected_runtime
        .forward(
            nemotron_h::EmbeddedInput::draft(&draft_token, &prior, 0),
            expected_draft_state.draft_mut(),
            &context,
        )
        .unwrap();

    let tp1_layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let tp1_context = NumericContext::with_local_layout(tp1_layout.clone());
    let tp1_geometry = nemotron_h::local_geometry(&args, &tp1_layout).unwrap();
    let tp1_state_layout = tp1_geometry.state_layout().clone();
    let tp1_architecture = nemotron_h::LayeredModel::<NumericBackend>::new_parallel(
        args.clone(),
        tp1_geometry,
        &tp1_context,
    )
    .unwrap();
    let tp1_units = addresses
        .iter()
        .copied()
        .map(|(group, index)| {
            <nemotron_h::LayeredModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
            >>::build_unit(&tp1_architecture, group, index, &tp1_context)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tp1_runtime =
        LayerwiseRuntime::new(tp1_architecture, ResidentUnitWindow::new(tp1_units));
    let mut tp1_state = DeviceState::<NumericBackend, _>::create(tp1_state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let tp1_parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let (tp1_target, tp1_forward) = tp1_runtime
        .forward_parallel_with_context_hook(
            nemotron_h::EmbeddedInput::target(&tokens, None),
            &mut tp1_state,
            &tp1_parallel,
            &tp1_context,
            |_, _, _| Ok(()),
        )
        .unwrap();
    assert_tensor_exact(&tp1_target, &expected_target, "Nemotron TP1 target logits");
    assert_state_exact(&tp1_state, &expected_state, 4, "Nemotron TP1 target state");
    let tp1_prior = tp1_forward.target_capture().unwrap().axis_slice(1, 2, 3);
    let mut tp1_draft_state = eredu_runtime::DraftStateTransaction::fork(&tp1_state);
    let tp1_draft = tp1_runtime
        .forward_parallel(
            nemotron_h::EmbeddedInput::draft(&draft_token, &tp1_prior, 0),
            tp1_draft_state.draft_mut(),
            &tp1_parallel,
            &tp1_context,
        )
        .unwrap();
    assert_tensor_exact(&tp1_draft, &expected_draft, "Nemotron TP1 draft logits");
    assert_state_exact(
        tp1_draft_state.draft_mut(),
        expected_draft_state.draft_mut(),
        4,
        "Nemotron TP1 draft state",
    );

    let layouts = (0..2)
        .map(|rank| numeric_local_layout(&groups, 2, rank).unwrap())
        .collect::<Vec<_>>();
    let collective_group = NumericParallelGroup::new(2);
    let outputs =
        std::thread::scope(|scope| {
            let handles =
                layouts
                    .into_iter()
                    .enumerate()
                    .map(|(rank, layout)| {
                        let args = args.clone();
                        let tokens = tokens.clone();
                        let draft_token = draft_token.clone();
                        let collective_group = Arc::clone(&collective_group);
                        scope.spawn(move || {
                            let context = NumericContext::with_local_layout(layout.clone());
                            let geometry = nemotron_h::local_geometry(&args, &layout).unwrap();
                            let state_layout = geometry.state_layout().clone();
                            let architecture =
                                nemotron_h::LayeredModel::<NumericBackend>::new_parallel(
                                    args, geometry, &context,
                                )
                                .unwrap();
                            let units = addresses.iter().copied().map(|(group, index)| {
                    <nemotron_h::LayeredModel<NumericBackend> as LayeredArchitecture<
                        NumericBackend,
                        DeviceState<NumericBackend, NumericHybridLayerState>,
                    >>::build_unit(&architecture, group, index, &context).unwrap()
                }).collect::<Vec<_>>();
                            let mut runtime =
                                LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                            let mut state = DeviceState::<NumericBackend, _>::create(
                                state_layout,
                                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                            )
                            .unwrap();
                            let parallel = NumericParallelContext::new(rank, collective_group);
                            let (target, forward) = runtime
                                .forward_parallel_with_context_hook(
                                    nemotron_h::EmbeddedInput::target(&tokens, None),
                                    &mut state,
                                    &parallel,
                                    &context,
                                    |_, _, _| Ok(()),
                                )
                                .unwrap();
                            let prior = forward.target_capture().unwrap().axis_slice(1, 2, 3);
                            let checkpoint = state.clone();
                            let mut transaction =
                                eredu_runtime::DraftStateTransaction::fork(&checkpoint);
                            let draft = runtime
                                .forward_parallel(
                                    nemotron_h::EmbeddedInput::draft(&draft_token, &prior, 0),
                                    transaction.draft_mut(),
                                    &parallel,
                                    &context,
                                )
                                .unwrap();
                            assert_eq!(transaction.draft_mut().layer(3).unwrap().position(), 1);
                            let mut canonical = checkpoint.clone();
                            transaction.rollback(&mut canonical);
                            assert_eq!(canonical.layer(3).unwrap().position(), 0);
                            (target, draft, parallel.trace(), canonical)
                        })
                    })
                    .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
    for (rank, (target, draft, trace, _)) in outputs.iter().enumerate() {
        assert_tensor_close(
            target,
            &expected_target,
            &format!("Nemotron TP2 rank {rank} target"),
        );
        assert_tensor_close(
            draft,
            &expected_draft,
            &format!("Nemotron TP2 rank {rank} draft"),
        );
        assert_eq!(trace.last().unwrap().output_shape, [1, 1, 7]);
    }
    assert_eq!(outputs[0].2.len(), 8);
    assert_eq!(outputs[1].2.len(), 8);
}

#[test]
fn nemotron_h_sliding_attention_is_chunk_invariant() {
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":16, "hidden_size":8,
        "intermediate_size":12, "num_hidden_layers":1,
        "hybrid_override_pattern":"*", "num_attention_heads":2,
        "num_key_value_heads":1, "head_dim":4, "mamba_num_heads":2,
        "n_groups":1, "mamba_head_dim":4, "ssm_state_size":2,
        "conv_kernel":3, "chunk_size":2, "sliding_window":2,
        "n_routed_experts":2, "n_shared_experts":1,
        "moe_intermediate_size":4, "moe_shared_expert_intermediate_size":4,
        "num_experts_per_tok":1, "n_group":1, "topk_group":1,
        "num_nextn_predict_layers":0, "tie_word_embeddings":true,
        "residual_in_fp32":true
    }))
    .unwrap();
    let new_state = || {
        DeviceState::<NumericBackend, _>::create(
            nemotron_h::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap()
    };
    let context = NumericContext::default();
    let mut whole_state = new_state();
    let mut chunked_state = new_state();
    let mut whole = ResidentRuntime::new(
        nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut chunked = ResidentRuntime::new(
        nemotron_h::LayeredModel::<NumericBackend>::new(args, &context).unwrap(),
        &context,
    )
    .unwrap();
    let expected = whole
        .forward(
            nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&[1, 2, 3, 4]), None),
            &mut whole_state,
            &context,
        )
        .unwrap();
    let first = chunked
        .forward(
            nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&[1, 2]), None),
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let second = chunked
        .forward(
            nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&[3, 4]), None),
            &mut chunked_state,
            &context,
        )
        .unwrap();
    let actual = NumericTensor::concatenate(&[first, second], 1, &context).unwrap();
    assert_tensor_close(&actual, &expected, "Nemotron-H sliding target logits");
    assert!(context.sliding_attention_calls.get() >= 3);
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
        input_transform: None,
        route_scale: None,
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
fn bounded_gated_product_caps_gate_and_up_before_activation() {
    let output = NumericBackend::gated_product(
        NumericTensor::new(vec![1, 2], vec![10.0, -10.0]),
        NumericTensor::new(vec![1, 2], vec![10.0, -10.0]),
        eredu_nn::GatedProductPolicy::bounded_silu(2.0).unwrap(),
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
            format: eredu_nn::LinearFormatSpec::unscaled(v4.linear_format).unwrap(),
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
        "compress_ratios": [0, 0, 0],
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
        "num_nextn_predict_layers": 1,
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
    assert_eq!(v4_runtime.architecture().mtp_len(), 1);
    assert_eq!(v4_runtime.architecture().draft_proposal_capacity(), 2);
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

    let mut transaction = eredu_runtime::DraftStateTransaction::fork(&v4_state);
    let anchor = NumericTensor::token_ids(&[3]);
    let oversized = v4_runtime
        .forward(
            deepseek::mtp::EmbeddedInput::dspark_proposal(&anchor, 3),
            transaction.draft_mut(),
            &context,
        )
        .expect_err("proposal capacity exceeds the DSpark block width");
    assert!(oversized
        .to_string()
        .contains("DSpark proposal capacity must be between 1 and 2, got 3"));
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
