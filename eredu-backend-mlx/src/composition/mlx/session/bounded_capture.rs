//! Eager bounded transformations under the session's existing submission recovery.
//!
//! No native capture handles survive the observer call. Source evaluation happens
//! only after reservation. It severs lazy dependencies before slicing/reduction;
//! views and contiguous temporaries are consumed before returning to inference.

use super::*;
use eredu_core::capture::*;
use safemlx::ops::indexing::{ArrayIndex, IntoStrideBy};

const CHUNK: u64 = 1024;

#[cfg(test)]
mod tests;

#[cfg(test)]
thread_local! {
    static HOST_READS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
pub(super) fn record_host_read(elements: usize) {
    HOST_READS.with(|reads| {
        let (total, largest) = reads.get();
        reads.set((total + elements, largest.max(elements)));
    });
}

pub(crate) fn capabilities() -> CaptureCapabilities {
    CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Preview, CaptureTransformKind::Slice,
            CaptureTransformKind::FullTensor, CaptureTransformKind::Summary, CaptureTransformKind::Histogram, CaptureTransformKind::TopCandidates],
        max_histogram_bins: 128,
        physical_native_limit: false,
        conditions: vec![
            "Ordinary committed text execution on one rank; no speculative, media, or partitioned capture".into(),
            "Logical capture storage is bounded; inference allocations and native private allocator/workspace are excluded".into(),
            "Transforms execute synchronously; no native views or lazy capture graphs are queued".into(),
            "Statistics use F32 inputs and native chunk reductions with F64 aggregation; raw integer IDs remain exact".into(),
            "Candidate scores are finite raw logits for the last row, before token filtering and sampler processing".into(),
        ],
    }
}

pub(super) struct NativeCapture<'a> {
    pub(super) stream: &'a Stream,
}

/// Reject known impossible bounds before any forward execution. Unknown shapes
/// remain subject to the exact observer-time check; they never receive a zero cost.
pub(super) fn preflight(plan: &AdmittedCapturePlan) -> Result<(), CaptureError> {
    eredu_runtime::capture::preflight(plan, estimate_shape)
}

/// Includes a conservative logical allowance for source/contiguous backing, all
/// elementwise reduction temporaries (bounded chunks), host conversion buffers,
/// and worst-case JSON numbers. Private allocator workspace is not claimed bounded.
fn estimate_shape(
    source: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    let source_elements = elements(source)?;
    let selected = elements(&slice.shape)?;
    if source_elements > i32::MAX as u64
        || source
            .iter()
            .chain(slice.strides.iter())
            .any(|n| *n > i32::MAX as u64)
    {
        return Err(CaptureError::Unsupported(
            "MLX capture shape/index exceeds signed 32-bit indexing".into(),
        ));
    }
    let chunks = selected.div_ceil(CHUNK);
    let (host_bytes, encoded_bytes, temporary_elements) = match &selection.transform {
        CaptureTransform::TopCandidates { count } => {
            if *count == 0 || source.last().is_none_or(|vocabulary| count > vocabulary) {
                return Err(CaptureError::Invalid(
                    "candidate count exceeds the runtime vocabulary".into(),
                ));
            }
            (
                add(16, mul(*count, 16)?)?,
                add(256, mul(*count, 64)?)?,
                *count,
            )
        }
        CaptureTransform::Preview { max_elements } => {
            let n = selected.min(*max_elements);
            (mul(n, 16)?, add(128, mul(n, 32)?)?, n)
        }
        CaptureTransform::Slice | CaptureTransform::FullTensor => {
            (mul(selected, 16)?, add(128, mul(selected, 32)?)?, selected)
        }
        CaptureTransform::Summary => (add(128, mul(chunks, 128)?)?, 512, selected.min(CHUNK)),
        CaptureTransform::Histogram { edges } => {
            let bins = edges.len().saturating_sub(1) as u64;
            if bins == 0 || bins > 128 {
                return Err(CaptureError::Unsupported("histogram bin count".into()));
            }
            (
                add(mul(bins, 16)?, mul(chunks, mul(add(bins, 3)?, 8)?)?)?,
                add(256, mul(add(bins, 1)?, 64)?)?,
                selected.min(CHUNK),
            )
        }
    };
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: add(
            add(mul(source_elements, 8)?, mul(selected, 16)?)?,
            add(4096, mul(temporary_elements, 128)?)?,
        )?,
        host_bytes,
        encoded_bytes,
    })
}

impl CaptureBackend for NativeCapture<'_> {
    type Tensor = MlxTensor;
    type Error = Exception;

    fn shape(&self, tensor: &MlxTensor) -> Result<Vec<u64>, Exception> {
        tensor
            .shape()
            .iter()
            .map(|n| u64::try_from(*n).map_err(|_| Exception::custom("negative capture dimension")))
            .collect()
    }

    fn estimate(
        &self,
        tensor: &MlxTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        if tensor.as_array().dtype() == Dtype::Complex64 {
            return Err(CaptureError::Unsupported("complex capture".into()));
        }
        let shape = self
            .shape(tensor)
            .map_err(|e| CaptureError::Invalid(e.to_string()))?;
        estimate_shape(&shape, selection, slice)
    }

    fn transform(
        &mut self,
        tensor: &MlxTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Exception> {
        let stream = self.stream;
        // Reservation precedes even this evaluation. No observation-owned lazy
        // source is kept while later model blocks execute.
        tensor.as_array().evaluated()?;
        let indices: Vec<_> = slice
            .starts
            .iter()
            .zip(&slice.ends)
            .zip(&slice.strides)
            .map(|((start, end), stride)| {
                ((*start as i32)..(*end as i32))
                    .stride_by(*stride as i32)
                    .index_op()
            })
            .collect();
        let selected = tensor
            .as_array()
            .try_index_device(indices.as_slice(), stream)?;
        let flat = selected.reshape(&[-1], stream)?;
        flat.evaluated()?;
        match &selection.transform {
            CaptureTransform::TopCandidates { count } => {
                let vocabulary = *tensor
                    .shape()
                    .last()
                    .ok_or_else(|| Exception::custom("scalar logits have no vocabulary"))?;
                let row = tensor
                    .as_array()
                    .reshape(&[-1, vocabulary], stream)?
                    .try_index_device((-1, ..), stream)?
                    .as_dtype(Dtype::Float32, stream)?;
                if count_mask(row.is_finite(stream)?, stream)? != vocabulary as u64 {
                    return Err(Exception::custom(
                        "candidate capture requires finite raw logits",
                    ));
                }
                let sorted = safemlx::ops::argsort(&row, stream)?;
                let indices = sorted
                    .try_index_device((vocabulary - *count as i32)..vocabulary, stream)?
                    .contiguous(false, stream)?;
                let scores = row.take(&indices, stream)?;
                #[cfg(test)]
                {
                    record_host_read(*count as usize);
                    record_host_read(*count as usize);
                }
                let indices = indices.evaluated()?;
                let scores = scores.evaluated()?;
                let candidates = indices
                    .as_slice::<u32>()
                    .iter()
                    .zip(scores.as_slice::<f32>())
                    .rev()
                    .map(|(id, score)| CaptureCandidate {
                        token_id: *id,
                        score: *score,
                    })
                    .collect();
                Ok(CapturePayload::Candidates(CaptureCandidates {
                    stage: CandidateScoreStage::RawLogitsBeforeSampling,
                    candidates,
                }))
            }
            CaptureTransform::Preview { max_elements } => {
                let n = (flat.size() as u64).min(*max_elements) as i32;
                let prefix = flat.try_index_device(0..n, stream)?;
                Ok(CapturePayload::Tensor(
                    super::observation::observe_tensor(&MlxTensor::from_array(prefix), stream)
                        .map_err(|e| Exception::custom(e.to_string()))?,
                ))
            }
            CaptureTransform::Slice | CaptureTransform::FullTensor => {
                let contiguous = selected.contiguous(false, stream)?;
                Ok(CapturePayload::Tensor(
                    super::observation::observe_tensor(&MlxTensor::from_array(contiguous), stream)
                        .map_err(|e| Exception::custom(e.to_string()))?,
                ))
            }
            CaptureTransform::Summary => Ok(CapturePayload::Summary(summary(&flat, stream)?)),
            CaptureTransform::Histogram { edges } => {
                Ok(CapturePayload::Histogram(histogram(&flat, edges, stream)?))
            }
        }
    }
}

fn scalar_f32(array: Array) -> Result<f64, Exception> {
    #[cfg(test)]
    record_host_read(1);
    Ok(f64::from(array.evaluated()?.as_slice::<f32>()[0]))
}

fn count_mask(mask: Array, stream: &Stream) -> Result<u64, Exception> {
    #[cfg(test)]
    record_host_read(1);
    let sum = mask.as_dtype(Dtype::Uint32, stream)?.sum(false, stream)?;
    Ok(u64::from(sum.evaluated()?.as_slice::<u32>()[0]))
}

fn summary(flat: &Array, stream: &Stream) -> Result<CaptureSummary, Exception> {
    let mut out = CaptureSummary {
        elements: flat.size() as u64,
        finite: 0,
        non_finite: 0,
        nan: 0,
        positive_infinity: 0,
        negative_infinity: 0,
        min: None,
        max: None,
        mean: None,
        rms: None,
    };
    let mut sum = 0.0f64;
    let mut squares = 0.0f64;
    for start in (0..flat.size()).step_by(CHUNK as usize) {
        let end = (start + CHUNK as usize).min(flat.size());
        let chunk = flat
            .try_index_device(start as i32..end as i32, stream)?
            .as_dtype(Dtype::Float32, stream)?;
        let finite = chunk.is_finite(stream)?;
        let finite_count = count_mask(finite.clone(), stream)?;
        out.finite += finite_count;
        out.nan += count_mask(chunk.is_nan(stream)?, stream)?;
        out.positive_infinity += count_mask(chunk.is_pos_inf(stream)?, stream)?;
        out.negative_infinity += count_mask(chunk.is_neg_inf(stream)?, stream)?;
        if finite_count == 0 {
            continue;
        }
        let clean = safemlx::ops::r#where(&finite, &chunk, Array::from(0.0f32), stream)?;
        let min = scalar_f32(
            safemlx::ops::r#where(&finite, &chunk, Array::from(f32::INFINITY), stream)?
                .min(false, stream)?,
        )?;
        let max = scalar_f32(
            safemlx::ops::r#where(&finite, &chunk, Array::from(f32::NEG_INFINITY), stream)?
                .max(false, stream)?,
        )?;
        out.min = Some(out.min.map_or(min, |value| value.min(min)));
        out.max = Some(out.max.map_or(max, |value| value.max(max)));
        // Normalize before summing or squaring: finite F32 extremes must not
        // overflow intermediate F32 sums. Chunk count is at most 1024.
        let scale = min.abs().max(max.abs());
        if scale != 0.0 {
            let scaled = clean.divide(Array::from(scale as f32), stream)?;
            sum += scalar_f32(scaled.sum(false, stream)?)? * scale;
            squares +=
                scalar_f32(scaled.multiply(&scaled, stream)?.sum(false, stream)?)? * scale * scale;
        }
    }
    if out.finite != 0 {
        out.mean = Some(sum / out.finite as f64);
        out.rms = Some((squares / out.finite as f64).sqrt());
    }
    out.non_finite = out.elements - out.finite;
    Ok(out)
}

fn histogram(flat: &Array, edges: &[f32], stream: &Stream) -> Result<CaptureHistogram, Exception> {
    let mut out = CaptureHistogram {
        edges: edges.to_vec(),
        counts: vec![0; edges.len() - 1],
        below: 0,
        above: 0,
        non_finite: 0,
    };
    for start in (0..flat.size()).step_by(CHUNK as usize) {
        let end = (start + CHUNK as usize).min(flat.size());
        let chunk = flat
            .try_index_device(start as i32..end as i32, stream)?
            .as_dtype(Dtype::Float32, stream)?;
        let finite = chunk.is_finite(stream)?;
        out.non_finite += count_mask(finite.logical_not(stream)?, stream)?;
        out.below += count_mask(
            chunk
                .lt(Array::from(edges[0]), stream)?
                .logical_and(&finite, stream)?,
            stream,
        )?;
        out.above += count_mask(
            chunk
                .gt(Array::from(*edges.last().unwrap()), stream)?
                .logical_and(&finite, stream)?,
            stream,
        )?;
        for (index, bounds) in edges.windows(2).enumerate() {
            let lower = chunk.ge(Array::from(bounds[0]), stream)?;
            let upper = if index + 1 == out.counts.len() {
                chunk.le(Array::from(bounds[1]), stream)?
            } else {
                chunk.lt(Array::from(bounds[1]), stream)?
            };
            out.counts[index] += count_mask(lower.logical_and(upper, stream)?, stream)?;
        }
    }
    Ok(out)
}

pub(super) struct BoundedObserver<'a> {
    pub(super) capture: &'a mut eredu_runtime::capture::CaptureSession,
    pub(super) stream: &'a Stream,
}

impl RuntimeActivationObserver<MlxTensor, Exception> for BoundedObserver<'_> {
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), Exception> {
        self.capture
            .observe(
                &mut NativeCapture {
                    stream: self.stream,
                },
                path,
                value,
            )
            .map_err(|e| Exception::custom(e.to_string()))
    }
    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, MlxTensor>,
    ) -> Result<(), Exception> {
        let mut result = Ok(());
        routing.for_each_tensor(|path, value| {
            if result.is_ok() {
                result = self.observe(&path, value);
            }
        });
        result
    }
}
