use super::*;
use crate::device::CandleSettlement;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
mod ordered;
use candle_core::D;
use eredu_core::cache::mechanisms::{
    AttentionScan, CacheAttentionAccumulator, CacheAttentionInput, CacheReadSubmission,
    LiveCacheReadSubmissionBackend, LiveCacheSelectedReadBackend, PagedAttentionGeometry,
    PagedAttentionOperands, PagedKeyValueStorage,
};
use eredu_nn::{
    paged_attention::{
        mask_columns, merge_softmax, page_visibility, requires_value_pass, retained_page_ranges,
        SoftmaxPartial,
    },
    IndexedAttentionInput, RelativeAttentionInput,
};
use ordered::{OrderedCpuAttention, OrderedCpuMode};

enum OrderedAttention {
    Cpu(OrderedCpuAttention),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(metal::OrderedMetalAttention),
}

impl OrderedAttention {
    fn begin_pass(&mut self, pass: usize, end: i64) -> Result<(), BackendFailure> {
        match self {
            Self::Cpu(cpu) => cpu.begin_pass(pass, end),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(metal) => metal.begin_pass(pass, end),
        }
    }

    fn accumulate(
        &mut self,
        start: i64,
        scores: &NativeTensor,
        values: &NativeTensor,
    ) -> Result<u64, Error> {
        match self {
            Self::Cpu(cpu) => {
                cpu.accumulate(start, scores, values)?;
                Ok(cpu.scratch_bytes())
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(metal) => {
                metal.accumulate(start, scores, values)?;
                Ok(metal.scratch_bytes())
            }
        }
    }

    fn finish(self, end: i64, queue: &CandleContext) -> Result<CandleTensor, BackendFailure> {
        match self {
            Self::Cpu(cpu) => cpu.finish(end, queue),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(metal) => metal.finish(end, queue),
        }
    }
}

impl LiveCacheSelectedReadBackend for CandleCacheStorage {
    fn copy_token_slice(
        tensor: &CandleTensor,
        axis: usize,
        range: Range<i32>,
        queue: &CandleContext,
        retention: impl Sized + 'static,
    ) -> Result<CandleTensor, BackendFailure> {
        queue.check_healthy()?;
        queue.validate_tensor_device(tensor)?;
        let mut scope = CandleSettlement::new(queue, (retention, vec![tensor.clone()]));
        let slice = Self::slice_tokens(tensor, axis, range, queue)?;
        let copy = slice.compact(queue).map_err(BackendFailure::from_error)?;
        scope.resources_mut().1.push(copy.clone());
        scope.finish()?;
        Ok(copy)
    }

    fn gather_token_rows(
        tensors: [&CandleTensor; 2],
        batch: i32,
        rows: &[i32],
        queue: &CandleContext,
        retention: impl Sized + 'static,
    ) -> Result<[CandleTensor; 2], BackendFailure> {
        queue.check_healthy()?;
        for tensor in tensors {
            queue.validate_tensor_device(tensor)?;
        }
        let end = batch
            .checked_add(1)
            .ok_or_else(|| invalid("batch index overflow"))?;
        let count = i32::try_from(rows.len()).map_err(BackendFailure::from_error)?;
        let mut scope =
            CandleSettlement::new(queue, (retention, tensors.map(Clone::clone).to_vec()));
        let indices = CandleTensor::from_i32_slice(rows, &[count], queue)
            .map_err(BackendFailure::from_error)?;
        scope.resources_mut().1.push(indices.clone());
        let mut gather = |tensor| -> Result<_, BackendFailure> {
            let result = Self::slice_tokens(tensor, 0, batch..end, queue)?
                .take_axis(&indices, 2, queue)
                .map_err(BackendFailure::from_error)?;
            scope.resources_mut().1.push(result.clone());
            Ok(result)
        };
        let output = [gather(tensors[0])?, gather(tensors[1])?];
        scope.finish()?;
        Ok(output)
    }
}

impl<R: 'static> CacheReadSubmission for CandleSettlement<R> {
    fn settle(self) -> Result<(), BackendFailure> {
        self.finish().map(drop)
    }
}

impl LiveCacheReadSubmissionBackend for CandleCacheStorage {
    fn begin_cache_read(
        queue: &CandleContext,
        retention: impl Sized + 'static,
    ) -> Result<impl CacheReadSubmission, BackendFailure> {
        queue.check_healthy()?;
        Ok(CandleSettlement::new(queue, retention))
    }
}

fn unsupported(message: &'static str) -> BackendFailure {
    BackendFailure::new(BackendFailureKind::Unsupported, Error::backend(message))
}

impl PagedKeyValueStorage for CandleCacheStorage {
    fn describe_paged_append(
        invocation: &MechanismInvocation,
    ) -> Result<MechanismMemoryContract, Error> {
        MechanismMemoryContract::unknown(invocation)
    }

    fn paged_attention(
        input: PagedAttentionOperands<'_, CandleTensor>,
        geometry: PagedAttentionGeometry,
        queue: &CandleContext,
    ) -> Result<impl CacheAttentionAccumulator<CandleTensor, CandleContext>, BackendFailure> {
        Self::device(queue)?;
        if !input.scale.is_finite()
            || input.scale <= 0.
            || input
                .softcap
                .is_some_and(|cap| !cap.is_finite() || cap <= 0.)
            || input.sinks.is_some_and(|sinks| {
                input.queries.shape().get(1).map(std::slice::from_ref) != Some(sinks.shape())
            })
        {
            return Err(invalid(
                "invalid paged attention scale, softcap or sink geometry",
            ));
        }
        let dtype = input.queries.value.dtype();
        if !supports_paged_float(dtype, queue) {
            return Err(unsupported(
                "paged attention requires F16, BF16, F32, or CPU F64 tensors",
            ));
        }
        let context_end =
            usize::try_from(geometry.context_end).map_err(BackendFailure::from_error)?;
        let bf16_scores = input.queries.value.dtype() == DType::BF16
            && input.arithmetic != eredu_nn::AttentionArithmetic::Float32;
        let bf16_cpu = bf16_scores && queue.placement().is_cpu();
        let mode = if bf16_cpu {
            Some(OrderedCpuMode::Bfloat16)
        } else if crate::float_cpu::applies(&input.queries.value) {
            Some(OrderedCpuMode::ArmFloat)
        } else {
            None
        };
        let ordered_geometry = geometry.window.is_none()
            && geometry.transient_context_start.is_none()
            && input.sinks.is_none()
            && input.softcap.is_none();
        let ordered = mode
            .filter(|_| ordered_geometry)
            .map(|mode| OrderedAttention::Cpu(OrderedCpuAttention::new(mode, context_end)));
        let bf16_metal = dtype == DType::BF16
            && input.arithmetic == eredu_nn::AttentionArithmetic::InputScores
            && crate::matmul::supports_ordered_metal(&input.queries.value);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let ordered = if ordered_geometry && bf16_metal {
            Some(OrderedAttention::Metal(metal::OrderedMetalAttention::new(
                geometry.context_end,
            )))
        } else {
            ordered
        };
        Ok(CandlePagedAttention {
            queries: input.queries.clone(),
            scale: input.scale,
            mask: input.mask.cloned(),
            sinks: input.sinks.cloned(),
            softcap: input.softcap,
            relative: None,
            output_dtype: dtype,
            arithmetic: input.arithmetic,
            bf16_scores: bf16_cpu || bf16_metal,
            rounded: requires_value_pass(input.arithmetic) || bf16_scores,
            ordered,
            geometry,
            partial: None,
            value_pass: false,
            value: None,
        })
    }

    fn paged_relative_attention(
        input: &RelativeAttentionInput<'_, CandleTensor>,
        geometry: PagedAttentionGeometry,
        queue: &CandleContext,
    ) -> Result<impl CacheAttentionAccumulator<CandleTensor, CandleContext>, BackendFailure> {
        Self::device(queue)?;
        CandlePagedAttention::relative(input, geometry, queue).map_err(BackendFailure::from_error)
    }

    fn indexed_positions(
        input: &IndexedAttentionInput<'_, CandleTensor>,
        batch: i32,
        query: i32,
        _: &CandleContext,
    ) -> Result<Vec<Option<i32>>, BackendFailure> {
        crate::attention::indexed_positions(input, batch, query).map_err(BackendFailure::from_error)
    }

    fn attend_indexed_row(
        input: &IndexedAttentionInput<'_, CandleTensor>,
        batch: i32,
        query: i32,
        slots: &[Option<i32>],
        gathered: Option<[CandleTensor; 2]>,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        crate::attention::attend_indexed_row(input, batch, query, slots, gathered, queue)
            .map_err(BackendFailure::from_error)
    }
}

pub(crate) fn resident_relative_attention(
    input: &RelativeAttentionInput<'_, CandleTensor>,
    queue: &CandleContext,
) -> Result<CandleTensor, Error> {
    let start = i64::from(input.key_offset);
    let end = start + i64::from(input.keys.dim(2));
    let geometry = PagedAttentionGeometry {
        query_start: i64::from(input.query_offset),
        context_end: end,
        window: input.window,
        prefix: 0,
        transient_context_start: None,
    };
    let mut attention = CandlePagedAttention::relative(input, geometry, queue)?;
    attention.accumulate_page(start..end, start..end, [input.keys, input.values], queue)?;
    attention.finish(queue).map_err(Error::backend_source)
}

struct CandlePagedAttention {
    queries: CandleTensor,
    scale: f32,
    mask: Option<CandleTensor>,
    sinks: Option<CandleTensor>,
    softcap: Option<f32>,
    relative: Option<crate::attention::RelativeScores>,
    output_dtype: DType,
    arithmetic: eredu_nn::AttentionArithmetic,
    geometry: PagedAttentionGeometry,
    rounded: bool,
    bf16_scores: bool,
    ordered: Option<OrderedAttention>,
    partial: Option<SoftmaxPartial<CandleTensor>>,
    value_pass: bool,
    value: Option<NativeTensor>,
}

fn supports_paged_float(dtype: DType, queue: &CandleContext) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        || (dtype == DType::F64 && queue.placement().is_cpu())
}

impl CandlePagedAttention {
    fn compute_dtype(&self) -> DType {
        if self.queries.value.dtype() == DType::F64
            && self.arithmetic == eredu_nn::AttentionArithmetic::Fused
        {
            DType::F64
        } else {
            DType::F32
        }
    }

    fn relative(
        input: &RelativeAttentionInput<'_, CandleTensor>,
        geometry: PagedAttentionGeometry,
        queue: &CandleContext,
    ) -> Result<Self, Error> {
        let (queries, relative) = crate::attention::RelativeScores::prepare(input, queue)?;
        Ok(Self {
            queries,
            scale: 1.,
            mask: None,
            sinks: None,
            softcap: None,
            relative: Some(relative),
            output_dtype: input.queries.value.dtype(),
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
            geometry,
            rounded: false,
            bf16_scores: false,
            ordered: None,
            partial: None,
            value_pass: false,
            value: None,
        })
    }

    fn merge_sink(&mut self, queue: &CandleContext) -> Result<(), Error> {
        let Some(sinks) = &self.sinks else {
            return Ok(());
        };
        let partial = self
            .partial
            .as_ref()
            .ok_or_else(|| Error::backend("empty paged attention scan"))?;
        let compute = self.compute_dtype();
        let minimum = if compute == DType::F64 {
            f64::MIN
        } else {
            f64::from(f32::MIN)
        };
        let dtype = if self.rounded {
            self.queries.value.dtype()
        } else {
            compute
        };
        let maximum = CandleTensor::wrap(
            crate::tensor::sink_scores(
                sinks,
                partial
                    .maximum
                    .value
                    .dims4()
                    .map_err(Error::backend_source)?,
                dtype,
            )
            .and_then(|value| value.to_dtype(compute)),
        )?;
        let sink = SoftmaxPartial {
            mass: CandleTensor::wrap(
                maximum
                    .value
                    .gt(f64::NEG_INFINITY)
                    .and_then(|valid| valid.to_dtype(compute)),
            )?,
            maximum: CandleTensor::wrap(maximum.value.clamp(minimum, -minimum))?,
            weighted_values: CandleTensor::wrap(NativeTensor::zeros_like(
                &partial.weighted_values.value,
            ))?,
        };
        self.partial = Some(merge_softmax(
            partial,
            &sink,
            queue,
            |left, right, _| CandleTensor::wrap(left.value.broadcast_maximum(&right.value)),
            |value, _| CandleTensor::wrap(value.value.exp()),
        )?);
        Ok(())
    }

    fn accumulate_page(
        &mut self,
        page: Range<i64>,
        selected: Range<i64>,
        tensors: [&CandleTensor; 2],
        queue: &CandleContext,
    ) -> Result<u64, Error> {
        let product = if crate::float_cpu::applies(&self.queries.value) {
            crate::float_cpu::attention_product
        } else {
            crate::matmul::attention_product
        };
        let (batch, heads, queries, _) =
            self.queries.value.dims4().map_err(Error::backend_source)?;
        if selected.start < page.start || selected.end > page.end || selected.is_empty() {
            return Err(Error::backend("invalid attention page slice"));
        }
        let dtype = self.queries.value.dtype();
        let compute = self.compute_dtype();
        let product_dtype = if dtype == DType::F64
            && self.arithmetic == eredu_nn::AttentionArithmetic::InputScores
        {
            dtype
        } else {
            compute
        };
        let width = (selected.end - selected.start) as usize;
        let start = (selected.start - page.start) as usize;
        let offset = selected.start;
        let select = |tensor: &CandleTensor, dtype| {
            let slice = tensor.value.narrow(2, start, width)?.to_dtype(dtype)?;
            crate::tensor::expand_attention_heads(&slice, heads)
        };
        let (scores, visible, values) = (|| -> candle_core::Result<_> {
            let values = select(tensors[1], product_dtype)?;
            let mut scores = if self.bf16_scores {
                crate::matmul::bf16_attention_page_scores(
                    &self.queries.value,
                    &select(tensors[0], DType::BF16)?,
                    self.geometry.context_end as usize,
                )?
                .to_dtype(DType::F32)?
            } else {
                product(
                    &self.queries.value.to_dtype(product_dtype)?,
                    &select(tensors[0], product_dtype)?,
                    true,
                )?
            };
            if self.rounded {
                scores = scores.to_dtype(dtype)?.to_dtype(product_dtype)?;
            }
            scores = scores.affine(self.scale.into(), 0.)?;
            if self.rounded {
                scores = scores.to_dtype(dtype)?.to_dtype(product_dtype)?;
            }
            if let Some(cap) = self.softcap {
                scores = crate::tensor::softcap_scores(scores, cap, queue)
                    .map_err(candle_core::Error::wrap)?;
                if self.rounded {
                    scores = scores.to_dtype(dtype)?.to_dtype(product_dtype)?;
                }
            }
            if let Some(relative) = &self.relative {
                scores = scores.add(&relative.bias(selected.clone())?.to_dtype(product_dtype)?)?;
            }
            let visible = if self.geometry.transient_context_start.is_some() {
                vec![1; queries * width]
            } else {
                page_visibility(
                    self.geometry.query_start,
                    queries as i32,
                    selected.clone(),
                    self.geometry.window,
                    self.geometry.prefix,
                )
                .map_err(candle_core::Error::wrap)?
                .into_iter()
                .map(u8::from)
                .collect()
            };
            let mut visible = NativeTensor::from_vec(visible, (queries, width), &queue.device)?
                .broadcast_as(scores.shape())?;
            if let Some(mask) = &self.mask {
                let mask_width = *mask.shape().last().ok_or_else(|| {
                    candle_core::Error::Msg("attention mask lacks a key axis".into())
                })?;
                let columns = mask_columns(
                    selected,
                    self.geometry.context_end - i64::from(mask_width),
                    mask_width,
                )
                .map_err(candle_core::Error::wrap)?;
                let mask_slice = mask
                    .value
                    .narrow(D::Minus1, columns.start as usize, width)?;
                if mask.value.dtype() == DType::U8 {
                    visible = visible.broadcast_mul(&mask_slice)?;
                } else {
                    let additive = mask_slice.to_dtype(product_dtype)?;
                    visible = visible.broadcast_mul(&additive.gt(f32::NEG_INFINITY as f64)?)?;
                    scores = scores.broadcast_add(&additive)?;
                    if self.rounded {
                        scores = scores.to_dtype(dtype)?.to_dtype(product_dtype)?;
                    }
                }
            }
            let masked = match compute {
                DType::F64 => NativeTensor::full(f64::MIN, (), &queue.device)?,
                _ => NativeTensor::full(f32::MIN, (), &queue.device)?,
            }
            .broadcast_as(scores.shape())?;
            let scores = visible.where_cond(&scores.to_dtype(compute)?, &masked)?;
            Ok((scores, visible.to_dtype(compute)?, values))
        })()
        .map_err(Error::backend_source)?;

        let scalar_bytes = product_dtype.size_in_bytes();
        let per_score_bytes = scalar_bytes * if self.relative.is_some() { 8 } else { 4 };
        let scratch = scores
            .elem_count()
            .saturating_mul(per_score_bytes)
            .saturating_add(values.elem_count().saturating_mul(scalar_bytes))
            .saturating_add(
                self.relative
                    .as_ref()
                    .map_or(0, |relative| relative.prepared_bytes(&self.queries)),
            ) as u64;
        if let Some(ordered) = &mut self.ordered {
            let ordered_scratch = ordered.accumulate(offset, &scores, &values)?;
            return Ok(scratch.saturating_add(ordered_scratch));
        }

        let safe_mass = |mass: &NativeTensor| {
            mass.gt(0.)?
                .where_cond(mass, &NativeTensor::ones_like(mass)?)
        };
        if self.value_pass {
            let normalizer = self
                .partial
                .as_ref()
                .ok_or_else(|| Error::backend("attention needs its normalization pass"))?;
            let next = (|| {
                let exponent = scores.broadcast_sub(&normalizer.maximum.value)?;
                let probabilities = visible
                    .gt(0.)?
                    .where_cond(
                        &exponent,
                        &NativeTensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?,
                    )?
                    .exp()?
                    .broadcast_div(&safe_mass(&normalizer.mass.value)?)?
                    .to_dtype(dtype)?
                    .to_dtype(product_dtype)?;
                let next = product(&probabilities, &values, false)?;
                match &self.value {
                    Some(previous) => previous.add(&next),
                    None => Ok(next),
                }
            })()
            .map_err(Error::backend_source)?;
            self.value = Some(next);
        } else {
            let maximum = CandleTensor::wrap(scores.max_keepdim(D::Minus1))?;
            let weights = scores
                .broadcast_sub(&maximum.value)
                .and_then(|x| x.exp())
                .and_then(|x| x.mul(&visible))
                .map_err(Error::backend_source)?;
            let mass = CandleTensor::wrap(weights.sum_keepdim(D::Minus1))?;
            let weighted_values = CandleTensor::wrap(if self.rounded {
                NativeTensor::zeros(
                    (batch, heads, queries, values.dims()[3]),
                    compute,
                    &queue.device,
                )
            } else {
                product(&weights, &values, false)
            })?;
            let partial = SoftmaxPartial {
                maximum,
                mass,
                weighted_values,
            };
            self.partial = Some(match self.partial.as_ref() {
                None => partial,
                Some(previous) => merge_softmax(
                    previous,
                    &partial,
                    queue,
                    |left, right, _| CandleTensor::wrap(left.value.broadcast_maximum(&right.value)),
                    |value, _| CandleTensor::wrap(value.value.exp()),
                )?,
            });
        }
        Ok(scratch)
    }
}

impl CacheAttentionAccumulator<CandleTensor, CandleContext> for CandlePagedAttention {
    fn scan(&self) -> AttentionScan {
        if self.ordered.is_some() {
            AttentionScan::MaximumThenMassThenValues
        } else if self.rounded {
            AttentionScan::OnlineThenValues
        } else {
            AttentionScan::Online
        }
    }

    fn begin_pass(&mut self, index: usize, queue: &CandleContext) -> Result<(), BackendFailure> {
        if let Some(ordered) = &mut self.ordered {
            return ordered.begin_pass(index, self.geometry.context_end);
        }
        if index != 1 || !self.rounded || self.value_pass || self.partial.is_none() {
            return Err(invalid("invalid paged attention normalization pass"));
        }
        self.merge_sink(queue).map_err(BackendFailure::from_error)?;
        self.value_pass = true;
        Ok(())
    }

    fn accumulate(
        &mut self,
        _: CacheAttentionInput,
        positions: Range<i64>,
        tensors: [&CandleTensor; 2],
        queue: &CandleContext,
        _retention: impl Sized + 'static,
    ) -> Result<u64, BackendFailure> {
        let accumulated = (|| {
            if let Some(suffix) = self.geometry.transient_context_start {
                let mut peak = 0;
                for selection in
                    retained_page_ranges(positions.clone(), self.geometry.prefix, suffix)
                {
                    peak = peak.max(self.accumulate_page(
                        positions.clone(),
                        selection,
                        tensors,
                        queue,
                    )?);
                }
                Ok(peak)
            } else {
                self.accumulate_page(positions.clone(), positions, tensors, queue)
            }
        })();
        if let Err(error) = queue.synchronize() {
            queue.quarantine((
                _retention,
                tensors.map(Clone::clone),
                self.queries.clone(),
                self.mask.clone(),
                self.sinks.clone(),
                self.relative.take(),
                self.partial.take(),
                self.value.take(),
                self.ordered.take(),
            ));
            return Err(error);
        }
        accumulated.map_err(BackendFailure::from_error)
    }

    fn finish(mut self, queue: &CandleContext) -> Result<CandleTensor, BackendFailure> {
        if let Some(ordered) = self.ordered {
            return ordered.finish(self.geometry.context_end, queue);
        }
        let needs_completion = self.sinks.is_some() || self.relative.is_some();
        let result = (|| {
            let value = if self.rounded {
                self.value
                    .as_ref()
                    .cloned()
                    .ok_or_else(|| invalid("missing paged attention value pass"))?
            } else {
                self.merge_sink(queue).map_err(BackendFailure::from_error)?;
                let partial = self
                    .partial
                    .as_ref()
                    .ok_or_else(|| invalid("empty paged attention scan"))?;
                (|| {
                    let mass = &partial.mass.value;
                    let safe_mass = mass
                        .gt(0.)?
                        .where_cond(mass, &NativeTensor::ones_like(mass)?)?;
                    partial.weighted_values.value.broadcast_div(&safe_mass)
                })()
                .map_err(BackendFailure::from_error)?
            };
            CandleTensor::wrap(value.to_dtype(self.output_dtype))
                .map_err(BackendFailure::from_error)
        })();
        if needs_completion {
            if let Err(error) = queue.synchronize() {
                queue.quarantine((
                    self.sinks,
                    self.relative,
                    self.queries,
                    self.partial,
                    self.value,
                    result.ok(),
                ));
                return Err(error);
            }
        }
        result
    }
}

impl eredu_core::cache::mechanisms::CompressedCacheStorage for CandleCacheStorage {
    fn snapshot_tensor_estimate(
        tensor: &CandleTensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let bytes =
            Self::snapshot_tensor_costs()?.bytes(tensor_size(tensor), tensor.shape().len())?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }
}

/// Single-pass FP32 online attention over already selected key and value blocks.
pub struct CandleBlockwiseAttention {
    attention: CandlePagedAttention,
    last_end: i64,
}

impl eredu_nn::BlockwiseAttentionBackend for CandleNeuralBackend {
    type BlockwiseAccumulator = CandleBlockwiseAttention;

    fn begin_blockwise_attention(
        spec: eredu_nn::BlockwiseAttentionSpec<'_, CandleTensor>,
        queue: &CandleContext,
    ) -> Result<Self::BlockwiseAccumulator, Error> {
        let (_, heads, queries, _) = spec.queries.value.dims4().map_err(Error::backend_source)?;
        if !spec.scale.is_finite()
            || spec.scale <= 0.0
            || spec.query_start < 0
            || spec
                .query_start
                .checked_add(queries as i64)
                .is_none_or(|end| end > spec.context_end)
            || spec.prefix_tokens < 0
            || spec.prefix_tokens > spec.context_end
            || spec.sliding_window.is_some_and(|window| window <= 0)
            || spec
                .sinks
                .is_some_and(|value| value.value.elem_count() != heads)
            || !supports_paged_float(spec.queries.value.dtype(), queue)
        {
            return Err(Error::backend(
                "invalid blockwise attention geometry or precision",
            ));
        }
        Ok(CandleBlockwiseAttention {
            attention: CandlePagedAttention {
                queries: spec.queries.clone(),
                scale: spec.scale,
                mask: spec.mask.cloned(),
                sinks: spec.sinks.cloned(),
                softcap: None,
                relative: None,
                output_dtype: spec.queries.value.dtype(),
                arithmetic: eredu_nn::AttentionArithmetic::Fused,
                geometry: PagedAttentionGeometry {
                    query_start: spec.query_start,
                    context_end: spec.context_end,
                    window: spec.sliding_window,
                    prefix: spec.prefix_tokens,
                    transient_context_start: None,
                },
                rounded: false,
                bf16_scores: false,
                ordered: None,
                partial: None,
                value_pass: false,
                value: None,
            },
            last_end: 0,
        })
    }

    fn accumulate_blockwise_attention(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: CandleTensor,
        values: CandleTensor,
        context: &CandleContext,
    ) -> Result<u64, Error> {
        if start < accumulator.last_end
            || end <= start
            || end > accumulator.attention.geometry.context_end
        {
            return Err(Error::backend(
                "blockwise attention blocks must be ordered within the selected context",
            ));
        }
        let width = i32::try_from(end - start).map_err(Error::backend_source)?;
        if keys.shape().get(2) != Some(&width) || values.shape().get(2) != Some(&width) {
            return Err(Error::backend(
                "blockwise attention tensor length differs from its position range",
            ));
        }
        let scratch = accumulator.attention.accumulate_page(
            start..end,
            start..end,
            [&keys, &values],
            context,
        )?;
        accumulator.last_end = end;
        Ok(scratch)
    }

    fn finish_blockwise_attention(
        accumulator: Self::BlockwiseAccumulator,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        accumulator
            .attention
            .finish(context)
            .map_err(Error::backend_source)
    }
}
