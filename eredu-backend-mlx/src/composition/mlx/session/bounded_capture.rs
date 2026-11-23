//! Eager bounded transformations under the session's existing submission recovery.
//!
//! No native capture handles survive the observer call. Source evaluation happens
//! only after reservation. It severs lazy dependencies before slicing/reduction;
//! views and contiguous temporaries are consumed before returning to inference.

use super::*;
use eredu_core::capture::*;
use safemlx::ops::indexing::{ArrayIndex, IntoStrideBy};

use super::score_readout::CHUNK;

pub(super) mod routed;
mod token_scores;

#[cfg(feature = "tensor-capture")]
pub(crate) fn capabilities() -> CaptureCapabilities {
    CaptureCapabilities {
        transformations: vec![CaptureTransformKind::RoutedUnits, CaptureTransformKind::Preview, CaptureTransformKind::Slice,
            CaptureTransformKind::FullTensor, CaptureTransformKind::Summary, CaptureTransformKind::Histogram, CaptureTransformKind::TopCandidates, CaptureTransformKind::TokenScores],
        max_histogram_bins: 128,
        physical_native_limit: false,
        conditions: vec![
            "Scoped speculative activations require complete selected hooks and phase-aware invocation admission; distributed execution also requires retained producer layouts and transport".into(),
            "Logical capture storage is bounded; inference allocations and native private allocator/workspace are excluded".into(),
            "Transforms execute synchronously; no native views or lazy capture graphs are queued".into(),
            "Statistics use F32 inputs and native chunk reductions with F64 aggregation; raw integer IDs remain exact".into(),
            "Candidate scores are finite raw logits for the selected row, before token filtering and sampler processing".into(),
            "Candidate domain membership reuses the exact tokenizer/constraint filter before forcing; unavailable provenance is explicit as domain: None".into(),
        ],
    }
}

#[doc(hidden)]
pub struct NativeCapture<'a> {
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) domain: Option<CaptureTokenDomain<'a>>,
    pub(in crate::composition::mlx) partition:
        Option<&'a crate::backend::distributed::MlxDistributedSession>,
}

/// Native allocation facts for the MLX capture operations.
pub(super) const CAPTURE_WORKSPACE: CaptureWorkspaceModel = CaptureWorkspaceModel {
    max_index: i32::MAX as u64,
    reduction_chunk: std::num::NonZeroU64::new(CHUNK).unwrap(),
    source_bytes_per_element: 8,
    selected_bytes_per_element: 16,
    workspace_base_bytes: 4096,
    workspace_bytes_per_element: 128,
    routed: Some(RoutedCaptureWorkspace {
        base_bytes: 4096,
        source_bytes_per_element: 24,
        bytes_per_route: 192,
    }),
};

fn bound_score_row(row: Option<u64>) -> Result<i32, Exception> {
    row.and_then(|row| i32::try_from(row).ok())
        .ok_or_else(|| Exception::custom("missing or unrepresentable admitted score row"))
}

impl CaptureBackend for NativeCapture<'_> {
    fn estimate_partition_routed_units(
        &self,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        CAPTURE_WORKSPACE.estimate_partition_routed(request)
    }
    fn capture_partition_routed_units(
        &mut self,
        source: &PartitionRoutedUnitCaptureSource<'_, MlxTensor>,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Option<Result<RoutedUnitCapture, Error>> {
        Some(capture_routed_units(
            &routed::NativeRoutedReadout(self.stream),
            &source.source,
            request.geometry,
            request.slice,
            Some((source, request)),
        ))
    }
    fn estimate_routed_units(
        &self,
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        CAPTURE_WORKSPACE.estimate(shape, selection, slice)
    }
    fn capture_routed_units(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        geometry: RoutedUnitGeometry,
        slice: &ResolvedCaptureSlice,
    ) -> Option<Result<RoutedUnitCapture, Error>> {
        Some(capture_routed_units(
            &routed::NativeRoutedReadout(self.stream),
            source,
            geometry,
            slice,
            None,
        ))
    }
    type Tensor = MlxTensor;
    type Error = Error;

    fn estimate_partition_source(
        &self,
        shape: &[u64],
        wait: eredu_core::BoundedCompletionWait,
    ) -> Result<CaptureUsage, CaptureError> {
        if self.partition.is_none()
            || wait.cancellation()
                != eredu_core::CompletionCancellationMode::QuarantineUntilComplete
        {
            return Err(CaptureError::Unsupported(
                "MLX capture source requires selected bounded partition completion".into(),
            ));
        }
        if shape.iter().any(|dimension| *dimension > i32::MAX as u64) {
            return Err(CaptureError::Unsupported(
                "MLX capture source exceeds signed indexing".into(),
            ));
        }
        Ok(CaptureUsage {
            // Ordinary inference owns its existing graph/state. Capture adds
            // one retained source, exact event, stream and unsplit world handle.
            retained_bytes: add(mul(elements(shape)?, 8)?, 8192)?,
            host_bytes: 4096,
            ..Default::default()
        })
    }

    fn prepare_partition_source(
        &mut self,
        tensor: &MlxTensor,
        wait: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Error> {
        self.partition
            .ok_or_else(|| {
                Error::observation(CaptureError::Invalid(
                    "partition source has no native session owner".into(),
                ))
            })?
            .prepare_capture_source(tensor, self.stream, wait)
    }

    fn source_dtype(&self, tensor: &MlxTensor) -> Option<eredu_core::checkpoint::TensorDtype> {
        Some(crate::tensor::portable_dtype(tensor.as_array().dtype()))
    }

    fn shape(&self, tensor: &MlxTensor) -> Result<Vec<u64>, Error> {
        tensor
            .shape()
            .iter()
            .map(|n| {
                u64::try_from(*n)
                    .map_err(|_| Error::from(Exception::custom("negative capture dimension")))
            })
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
        CAPTURE_WORKSPACE.estimate(&shape, selection, slice)
    }

    fn estimate_generated(
        &self,
        prototype: &MlxTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        use eredu_core::checkpoint::TensorDtype;
        match &source.source_dtype {
            Some(TensorDtype::Complex64 | TensorDtype::Encoded(_)) => Err(
                CaptureError::Unsupported("generated source precision is not capturable".into()),
            ),
            Some(_) => {
                let shape = self
                    .shape(prototype)
                    .map_err(|error| CaptureError::Invalid(error.to_string()))?;
                CAPTURE_WORKSPACE.estimate(&shape, selection, slice)
            }
            None => self.estimate(prototype, selection, slice),
        }
    }

    fn transform(
        &mut self,
        tensor: &MlxTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
        score_row: Option<u64>,
    ) -> Result<CapturePayload, Error> {
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
            CaptureTransform::RoutedUnits => Err(Error::observation(CaptureError::Invalid(
                "routed units require actual route metadata".into(),
            ))),
            CaptureTransform::TokenScores { token_ids, .. } => self
                .token_scores(tensor, token_ids, bound_score_row(score_row)?)
                .map_err(Into::into),
            CaptureTransform::TopCandidates { count } => {
                let vocabulary = *tensor
                    .shape()
                    .last()
                    .ok_or_else(|| Exception::custom("scalar logits have no vocabulary"))?;
                let row = tensor
                    .as_array()
                    .reshape(&[-1, vocabulary], stream)?
                    .try_index_device((bound_score_row(score_row)?, ..), stream)?
                    .as_dtype(Dtype::Float32, stream)?;
                if count_mask(row.is_finite(stream)?, stream)? != vocabulary as u64 {
                    return Err(
                        Exception::custom("candidate capture requires finite raw logits").into(),
                    );
                }
                let sorted = safemlx::ops::argsort(&row, stream)?;
                let indices = sorted
                    .try_index_device((vocabulary - *count as i32)..vocabulary, stream)?
                    .contiguous(false, stream)?;
                let scores = row.take(&indices, stream)?;
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
                        allowed: self.domain.is_none_or(|domain| domain.filter.allows(*id)),
                    })
                    .collect();
                Ok(CapturePayload::Candidates(CaptureCandidates {
                    stage: CandidateScoreStage::RawLogitsBeforeSampling,
                    source: CandidateLogitsSource::Original,
                    candidates,
                    domain: self.domain.map(|domain| domain.summary(vocabulary as u32)),
                }))
            }
            CaptureTransform::Preview { max_elements } => {
                let n = (flat.size() as u64).min(*max_elements) as i32;
                let prefix = flat.try_index_device(0..n, stream)?;
                Ok(CapturePayload::Tensor(super::observation::observe_tensor(
                    &MlxTensor::from_array(prefix),
                    stream,
                )?))
            }
            CaptureTransform::Slice | CaptureTransform::FullTensor => {
                let contiguous = selected.contiguous(false, stream)?;
                Ok(CapturePayload::Tensor(super::observation::observe_tensor(
                    &MlxTensor::from_array(contiguous),
                    stream,
                )?))
            }
            CaptureTransform::Summary => Ok(CapturePayload::Summary(summary(&flat, stream)?)),
            CaptureTransform::Histogram { edges } => {
                Ok(CapturePayload::Histogram(histogram(&flat, edges, stream)?))
            }
        }
    }
}

fn scalar_f32(array: Array) -> Result<f64, Exception> {
    Ok(f64::from(array.evaluated()?.as_slice::<f32>()[0]))
}

fn count_mask(mask: Array, stream: &Stream) -> Result<u64, Exception> {
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

/// Native observations, placement, and estimates used by the portable speculative collector.
#[doc(hidden)]
pub struct MlxSpeculativeCaptureMechanisms;

impl CaptureWorkspace for MlxSpeculativeCaptureMechanisms {
    fn capture_workspace() -> CaptureWorkspaceModel {
        CAPTURE_WORKSPACE
    }
}

impl eredu_core::capture::SpeculativeCapturePrimitives for MlxSpeculativeCaptureMechanisms {
    type Queues = crate::composition::mlx::speculative::autoregressive::MlxAutoregressiveMechanisms;
    type Transport = MlxDistributedSession;
    type InvocationTensor = Array;
    type ObserverError = Error;
    type Scores = Array;
    type Tensor = MlxTensor;
    type Error = Error;
    type InvocationError = Exception;
    type Backend<'a> = NativeCapture<'a>;

    fn capture_backend<'a>(
        stream: &'a Stream,
        partition: Option<&'a Self::Transport>,
    ) -> NativeCapture<'a> {
        NativeCapture {
            stream,
            domain: None,
            partition,
        }
    }

    fn intervention_mechanisms() -> eredu_core::intervention::InterventionMechanisms {
        super::intervention::mechanisms()
    }

    fn intervention_estimator(
    ) -> std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator> {
        std::sync::Arc::new(super::intervention::NativeInterventionEstimator)
    }

    fn generated_creation_bytes(shape: &[u64]) -> Result<u64, CaptureError> {
        super::model_session::partition_capture::generated_creation_bytes(shape)
    }

    fn observation_failure(error: eredu_core::BackendFailure) -> Error {
        Error::observation(error)
    }

    fn invocation_failure(error: &dyn std::error::Error) -> Exception {
        Exception::custom(error.to_string())
    }
}
