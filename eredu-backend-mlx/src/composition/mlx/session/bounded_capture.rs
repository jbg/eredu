//! Eager bounded transformations under the session's existing submission recovery.
//!
//! No native capture handles survive the observer call. Source evaluation happens
//! only after reservation. It severs lazy dependencies before slicing/reduction;
//! views and contiguous temporaries are consumed before returning to inference.

use super::*;
use eredu_core::capture::*;
use safemlx::ops::indexing::{ArrayIndex, IntoStrideBy};
use std::mem::size_of;

const CHUNK: u64 = 1024;

#[cfg(test)]
mod tests;

mod original_speculative;
mod scheduled;
pub(in crate::composition::mlx) mod partition;
pub(in crate::composition::mlx) mod funded_model;
pub(in crate::composition::mlx) use scheduled::ScheduledNativeCapture;
mod routed;
pub(in crate::composition::mlx) use original_speculative::prepare as original_speculative_capture_with_error;
mod token_scores;

pub(in crate::composition::mlx) struct SpeculativeCaptureProvider {
    stream: Stream,
    partition: Option<MlxDistributedSession>,
}
impl SpeculativeCaptureProvider {
    pub(in crate::composition::mlx) fn new(
        stream: Stream,
        partition: Option<MlxDistributedSession>,
    ) -> Self {
        Self { stream, partition }
    }
}

impl eredu_runtime::capture::CaptureBackendProvider for SpeculativeCaptureProvider {
    type Tensor = MlxTensor;
    type Error = Error;
    type Backend<'a> = NativeCapture<'a>;
    fn backend(&mut self) -> NativeCapture<'_> {
        NativeCapture {
            stream: &self.stream,
            domain: None,
            partition: self.partition.as_ref(),
        }
    }
}

pub(in crate::composition::mlx) fn speculative_capture(
    plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
    request: eredu_core::SpeculativeRequestId,
    stream: &Stream,
) -> Result<
    Option<Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, Exception>>>,
    eredu_core::speculative::SpeculativeControlError,
> {
    speculative_capture_with_error(
        plan,
        request,
        stream,
        |error: &eredu_runtime::capture::CaptureExecutionError<Error>| {
            Exception::custom(error.to_string())
        },
    )
}

pub(in crate::composition::mlx) fn speculative_capture_with_error<E: 'static, F>(
    plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
    request: eredu_core::SpeculativeRequestId,
    stream: &Stream,
    map_error: F,
) -> Result<
    Option<Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, E>>>,
    eredu_core::speculative::SpeculativeControlError,
>
where
    F: eredu_runtime::capture::SpeculativeCaptureErrorTransport<Error, E> + 'static,
{
    Ok(
        eredu_runtime::capture::SpeculativeCaptureObserver::from_admitted(
            plan,
            SpeculativeCaptureProvider::new(stream.clone(), None),
            map_error,
            request,
            std::sync::Arc::new(super::intervention::NativeInterventionEstimator),
        )?
        .map(|observer| {
            Box::new(observer)
                as Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, E>>
        }),
    )
}

#[cfg(test)]
pub(crate) fn fixture_speculative_capture(
    session: eredu_runtime::capture::CaptureSession,
    stream: Stream,
    captures: Vec<eredu_runtime::capture::SpeculativeCaptureScope>,
    interventions: Vec<eredu_runtime::capture::SpeculativeCaptureScope>,
) -> Result<
    impl eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, Error>,
    CaptureError,
> {
    eredu_runtime::capture::SpeculativeCaptureObserver::new(
        session,
        SpeculativeCaptureProvider::new(stream, None),
        |error: &eredu_runtime::capture::CaptureExecutionError<Error>| {
            Error::Exception(Exception::custom(error.to_string()))
        },
        eredu_core::SpeculativeRequestId::new(0),
        captures,
        interventions,
    )
}

#[cfg(test)]
thread_local! {
    static HOST_READS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    static TRANSFORM_FAILURE: std::cell::RefCell<Option<Exception>> = const { std::cell::RefCell::new(None) };
    static TRANSFORM_FAILURE_AFTER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) struct TestCaptureFailure;
#[cfg(test)]
impl Drop for TestCaptureFailure {
    fn drop(&mut self) {
        TRANSFORM_FAILURE.with(|slot| {
            slot.borrow_mut().take();
        });
        TRANSFORM_FAILURE_AFTER.set(0);
    }
}
#[cfg(test)]
pub(super) fn fail_next_transform(error: Exception) -> TestCaptureFailure {
    fail_transform_after(0, error)
}
#[cfg(test)]
pub(crate) fn fail_transform_after(successful: usize, error: Exception) -> TestCaptureFailure {
    TRANSFORM_FAILURE_AFTER.set(successful);
    TRANSFORM_FAILURE.with(|slot| {
        assert!(slot.borrow_mut().replace(error).is_none());
    });
    TestCaptureFailure
}

#[cfg(test)]
fn test_transform_failure() -> Result<(), Exception> {
    TRANSFORM_FAILURE.with(|slot| {
        if slot.borrow().is_none() {
            return Ok(());
        }
        let remaining = TRANSFORM_FAILURE_AFTER.get();
        if remaining != 0 {
            TRANSFORM_FAILURE_AFTER.set(remaining - 1);
            return Ok(());
        }
        Err(slot.borrow_mut().take().expect("installed error"))
    })
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
        transformations: vec![CaptureTransformKind::RoutedUnits, CaptureTransformKind::Preview, CaptureTransformKind::Slice,
            CaptureTransformKind::FullTensor, CaptureTransformKind::Summary, CaptureTransformKind::Histogram, CaptureTransformKind::TopCandidates, CaptureTransformKind::TokenScores],
        max_histogram_bins: 128,
        conditions: vec![
            "Scoped speculative activations require complete selected hooks and phase-aware invocation admission; distributed execution also requires retained producer layouts and transport".into(),
            "Logical capture storage is bounded; inference allocations and native private allocator/workspace are excluded".into(),
            "Transforms execute synchronously; no native views or lazy capture graphs are queued".into(),
            "Statistics use F32 inputs and native chunk reductions with F64 aggregation; raw integer IDs remain exact".into(),
            "Candidate scores are finite raw logits for the last row, before token filtering and sampler processing".into(),
            "Candidate domain membership reuses the exact tokenizer/constraint filter before forcing; unavailable provenance is explicit as domain: None".into(),
        ],
    }
}

pub(in crate::composition::mlx) struct NativeCapture<'a> {
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) domain: Option<CaptureTokenDomain<'a>>,
    pub(in crate::composition::mlx) partition:
        Option<&'a crate::backend::distributed::MlxDistributedSession>,
}

/// Includes a conservative logical allowance for source/contiguous backing, all
/// elementwise reduction temporaries (bounded chunks), host conversion buffers,
/// and worst-case JSON numbers. Private allocator workspace is not claimed bounded.
pub(super) fn estimate_shape(
    source: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    if matches!(selection.transform, CaptureTransform::RoutedUnits) {
        return routed::estimate(source, slice);
    }
    estimate_selected_shape(source, selection, &slice.shape, &slice.strides)
}
fn estimate_selected_shape(
    source: &[u64],
    selection: &CaptureSelection,
    selected_shape: &[u64],
    strides: &[u64],
) -> Result<CaptureUsage, CaptureError> {
    let source_elements = elements(source)?;
    let selected = elements(selected_shape)?;
    if source_elements > i32::MAX as u64
        || source
            .iter()
            .chain(strides.iter())
            .any(|n| *n > i32::MAX as u64)
    {
        return Err(CaptureError::Unsupported(
            "MLX capture shape/index exceeds signed 32-bit indexing".into(),
        ));
    }
    if let Some(output) = raw_tensor_output(&selection.transform, selected) {
        return estimate_raw_tensor_counts(source_elements, selected, output);
    }
    let chunks = selected.div_ceil(CHUNK);
    let (host_bytes, encoded_bytes, temporary_elements) = match &selection.transform {
        CaptureTransform::RoutedUnits => unreachable!("handled before dense estimation"),
        CaptureTransform::TokenScores { token_ids } => {
            if token_ids.is_empty()
                || token_ids.len() > 64
                || source
                    .last()
                    .is_none_or(|width| token_ids.iter().any(|id| u64::from(*id) >= *width))
            {
                return Err(CaptureError::Invalid(
                    "selected score IDs exceed the runtime vocabulary or count limit".into(),
                ));
            }
            let count = token_ids.len() as u64;
            (
                add(256, add(mul(count, 128)?, mul(chunks, 8)?)?)?,
                add(1024, mul(count, 512)?)?,
                selected.min(CHUNK),
            )
        }
        CaptureTransform::TopCandidates { count } => {
            if *count == 0 || source.last().is_none_or(|vocabulary| count > vocabulary) {
                return Err(CaptureError::Invalid(
                    "candidate count exceeds the runtime vocabulary".into(),
                ));
            }
            (
                add(80, mul(*count, 20)?)?,
                add(384, mul(*count, 80)?)?,
                *count,
            )
        }
        CaptureTransform::Preview { .. }
        | CaptureTransform::Slice
        | CaptureTransform::FullTensor => unreachable!("raw tensor handled before reductions"),
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
    // Candidate capture touches only the last logits row: its temporaries
    // (dtype copy, finite mask, argsort indices) are all row-sized, and the
    // source backing is an inference allocation that exists regardless of
    // capture. Charging the whole source would veto prefill candidate capture
    // — where the source is [batch, prompt, vocabulary] — for any real prompt.
    let (retained_source, retained_selected) = match &selection.transform {
        CaptureTransform::TopCandidates { .. } => {
            let vocabulary = source.last().copied().unwrap_or(0);
            (vocabulary, vocabulary)
        }
        _ => (source_elements, selected),
    };
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: add(
            add(mul(retained_source, 8)?, mul(retained_selected, 16)?)?,
            add(4096, mul(temporary_elements, 128)?)?,
        )?,
        host_bytes,
        encoded_bytes,
    })
}

pub(in crate::composition::mlx) fn estimate_summary(
    geometry: &CaptureSummaryGeometry<'_>,
) -> Result<CaptureUsage, CaptureError> {
    let rank = geometry.source_shape().len();
    let mut source = [0u64; 32];
    let mut selected = [0u64; 32];
    for axis in 0..rank {
        source[axis] = geometry.source_shape()[axis] as u64;
        selected[axis] = geometry.shape()[axis] as u64;
    }
    estimate_selected_shape(
        &source[..rank],
        &geometry.admission().plan().selections[geometry.selection_index()],
        &selected[..rank],
        geometry.strides(),
    )
}
pub(in crate::composition::mlx) fn estimate_prefill_summary(
    plan: &CapturePrefillTransformPlan<'_>,
) -> Result<CaptureUsage, CaptureError> {
    eredu_runtime::capture::summary_prefill_usage(plan, |fragment| {
        estimate_selected_shape(
            fragment.source_shape(),
            plan.selection(),
            fragment.selected_shape(),
            fragment.strides(),
        )
    })
}
pub(in crate::composition::mlx) fn estimate_histogram(
    geometry: &CaptureHistogramGeometry<'_>,
) -> Result<CaptureUsage, CaptureError> {
    let rank = geometry.source_shape().len();
    let mut source = [0u64; 32];
    let mut selected = [0u64; 32];
    for axis in 0..rank {
        source[axis] = geometry.source_shape()[axis] as u64;
        selected[axis] = geometry.shape()[axis] as u64;
    }
    estimate_selected_shape(
        &source[..rank],
        &geometry.admission().plan().selections[geometry.selection_index()],
        &selected[..rank],
        geometry.strides(),
    )
}
pub(in crate::composition::mlx) fn estimate_prefill_histogram(
    plan: &CapturePrefillTransformPlan<'_>,
) -> Result<CaptureUsage, CaptureError> {
    eredu_runtime::capture::histogram_prefill_usage(plan, |fragment| {
        estimate_selected_shape(
            fragment.source_shape(),
            plan.selection(),
            fragment.selected_shape(),
            fragment.strides(),
        )
    })
}
pub(in crate::composition::mlx) fn estimate_candidates(
    geometry: &CaptureCandidateGeometry<'_>,
) -> Result<CaptureUsage, CaptureError> {
    let k = geometry.count() as u64;
    let v = geometry.vocabulary() as u64;
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: add(mul(v, 24)?, add(4096, mul(k, 128)?)?)?,
        host_bytes: add(80, mul(k, 20)?)?,
        encoded_bytes: add(384, mul(k, 80)?)?,
    })
}

pub(in crate::composition::mlx) fn estimate_token_scores(
    geometry: &CaptureTokenScoreGeometry<'_>,
) -> Result<CaptureUsage, CaptureError> {
    let shape = geometry.source_shape().map(|n| n as u64);
    let selection = &geometry.admission().plan().selections[geometry.selection_index()];
    estimate_selected_shape(&shape, selection, &shape, &[1, 1, 1])
}

fn raw_tensor_output(transform: &CaptureTransform, selected: u64) -> Option<u64> {
    match transform {
        CaptureTransform::Preview { max_elements } => Some(selected.min(*max_elements)),
        CaptureTransform::Slice | CaptureTransform::FullTensor => Some(selected),
        _ => None,
    }
}

// Existing logical quota, shared by legacy and scheduled observers. These terms
// are not a physical native bound or a host construction permission.
fn estimate_raw_tensor_counts(
    source: u64,
    selected: u64,
    output: u64,
) -> Result<CaptureUsage, CaptureError> {
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: add(
            add(mul(source, 8)?, mul(selected, 16)?)?,
            add(4096, mul(output, 128)?)?,
        )?,
        host_bytes: mul(output, 16)?,
        encoded_bytes: add(128, mul(output, 32)?)?,
    })
}

/// The same raw-tensor logical quota from an actual admitted geometry, without
/// shape/slice Vecs or native observation. Preview retains the complete selected
/// count for source/selection terms, separately from its delivered prefix count.
pub(in crate::composition::mlx) fn estimate_tensor_geometry(
    geometry: &CaptureTensorGeometry<'_>,
) -> Result<CaptureUsage, CaptureError> {
    let mut source = [0u64; 32];
    let mut selected = [0u64; 32];
    let rank = geometry.source_shape().len();
    let mut wide_stride = false;
    for (i, &extent) in geometry.source_shape().iter().enumerate() {
        source[i] = u64::try_from(extent).map_err(|_| CaptureError::Overflow)?;
        let stride = geometry.strides()[i];
        wide_stride |= stride > i32::MAX as u64;
        if stride == 0 {
            return Err(CaptureError::Overflow);
        }
        selected[i] = geometry.ends()[i]
            .checked_sub(geometry.starts()[i])
            .ok_or(CaptureError::Overflow)?
            .div_ceil(stride);
    }
    let source_elements = elements(&source[..rank])?;
    let selected_elements = elements(&selected[..rank])?;
    if source_elements > i32::MAX as u64
        || source[..rank].iter().any(|n| *n > i32::MAX as u64)
        || wide_stride
    {
        return Err(CaptureError::Unsupported(
            "MLX capture shape/index exceeds signed 32-bit indexing".into(),
        ));
    }
    let output =
        raw_tensor_output(geometry.native_transform(), selected_elements).ok_or_else(|| {
            CaptureError::Unsupported("capture requires a raw tensor transform".into())
        })?;
    estimate_raw_tensor_counts(source_elements, selected_elements, output)
}

impl CaptureBackend for NativeCapture<'_> {
    fn validate_capture_prefill_transform_source(
        &self,
        tensor: &MlxTensor,
        fragment: &eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
    ) -> Option<Result<eredu_core::checkpoint::TensorDtype, Error>> {
        Some((|| {
            if tensor.shape().len() != fragment.source_shape().len()
                || tensor
                    .shape()
                    .iter()
                    .zip(fragment.source_shape())
                    .any(|(&a, &b)| u64::try_from(a).ok() != Some(b))
            {
                return Err(Error::observation(CaptureError::Invalid(
                    "ordinary transform source shape differs".into(),
                )));
            }
            let dtype = crate::tensor::portable_dtype(tensor.as_array().dtype());
            if !matches!(
                dtype,
                eredu_core::checkpoint::TensorDtype::F16
                    | eredu_core::checkpoint::TensorDtype::Bf16
                    | eredu_core::checkpoint::TensorDtype::F32
            ) {
                return Err(Error::observation(CaptureError::Unsupported(
                    "ordinary transform source precision".into(),
                )));
            }
            Ok(dtype)
        })())
    }
    fn estimate_capture_prefill_transform(
        &self,
        fragment: &eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError> {
        if fragment
            .source_shape()
            .iter()
            .chain(fragment.starts())
            .chain(fragment.ends())
            .chain(fragment.strides())
            .any(|&v| i32::try_from(v).is_err())
        {
            return Err(CaptureError::Overflow);
        }
        let mut usage = estimate_selected_shape(
            fragment.source_shape(),
            fragment.plan().selection(),
            fragment.selected_shape(),
            fragment.strides(),
        )?;
        usage.host_bytes = add(
            usage.host_bytes,
            mul(
                fragment.source_shape().len() as u64,
                (4 * size_of::<u64>() + size_of::<safemlx::ops::indexing::ArrayIndexOp<'_>>())
                    as u64,
            )?,
        )?;
        Ok(usage)
    }
    fn capture_prefill_transform(
        &mut self,
        tensor: &MlxTensor,
        fragment: &eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
    ) -> Option<Result<CapturePayload, Error>> {
        Some((|| {
            self.validate_capture_prefill_transform_source(tensor, fragment)
                .expect("native validation")?;
            let slice = ResolvedCaptureSlice {
                starts: fragment.starts().to_vec(),
                ends: fragment.ends().to_vec(),
                strides: fragment.strides().to_vec(),
                shape: fragment.selected_shape().to_vec(),
            };
            self.transform(tensor, fragment.plan().selection(), &slice)
        })())
    }
    fn estimate_capture_prefill_candidates(
        &self,
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate_candidates(geometry)
    }
    fn capture_prefill_candidates(
        &mut self,
        tensor: &MlxTensor,
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Option<Result<CaptureCandidates, Error>> {
        Some((|| {
            let program = crate::backend::array_copy::CandidateExtraction::from_geometry(geometry)
                .map_err(|e| Error::Other(Box::new(e)))?;
            program
                .validate_source(tensor.as_array())
                .map_err(|e| Error::Other(Box::new(e)))?;
            tensor.as_array().evaluated()?;
            #[cfg(test)]
            test_transform_failure()?;
            self.candidate_values(tensor, geometry.count() as u64)
        })())
    }

    fn estimate_capture_prefill_token_scores(
        &self,
        geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate_token_scores(geometry)
    }
    fn capture_prefill_token_scores(
        &mut self,
        tensor: &MlxTensor,
        geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Option<Result<CaptureTokenScores, Error>> {
        Some((|| {
            let program = crate::backend::array_copy::TokenScoreProgram::from_geometry(geometry)
                .map_err(|e| Error::Other(Box::new(e)))?;
            program
                .validate_source(tensor.as_array())
                .map_err(|e| Error::Other(Box::new(e)))?;
            tensor.as_array().evaluated()?;
            #[cfg(test)]
            test_transform_failure()?;
            match self.token_scores(tensor, geometry.token_ids())? {
                CapturePayload::TokenScores(scores) => Ok(scores),
                _ => unreachable!("closed selected-token score worker"),
            }
        })())
    }

    fn validate_capture_prefill_source(
        &self,
        tensor: &MlxTensor,
        fragment: &eredu_core::capture::CapturePrefillFragment<'_, '_>,
    ) -> Option<Result<eredu_core::checkpoint::TensorDtype, Error>> {
        Some((|| {
            if tensor.shape().len() != fragment.source_shape().len()
                || tensor
                    .shape()
                    .iter()
                    .zip(fragment.source_shape())
                    .any(|(&a, &b)| usize::try_from(a).ok() != Some(b))
            {
                return Err(Error::observation(CaptureError::Invalid(
                    "prepared-media physical hook shape differs".into(),
                )));
            }
            let dtype = crate::tensor::portable_dtype(tensor.as_array().dtype());
            if !matches!(
                dtype,
                eredu_core::checkpoint::TensorDtype::F32
                    | eredu_core::checkpoint::TensorDtype::F16
                    | eredu_core::checkpoint::TensorDtype::Bf16
            ) {
                return Err(Error::observation(CaptureError::Unsupported(
                    "prepared-media capture requires a floating decoder source".into(),
                )));
            }
            Ok(dtype)
        })())
    }
    fn estimate_capture_prefill_fragment(
        &self,
        fragment: &eredu_core::capture::CapturePrefillFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError> {
        // Native slice descriptors use signed 32-bit indices. Reject cold,
        // before the logical row reservation or any selected native factory.
        if fragment
            .source_shape()
            .iter()
            .any(|&d| i32::try_from(d).is_err())
            || (0..fragment.source_shape().len()).any(|axis| {
                fragment.selection_axis(axis).is_none_or(|part| {
                    i32::try_from(part.start()).is_err()
                        || i32::try_from(part.end()).is_err()
                        || i32::try_from(part.stride()).is_err()
                })
            })
        {
            return Err(CaptureError::Overflow);
        }
        let source = fragment
            .source_shape()
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
            .ok_or(CaptureError::Overflow)?;
        let mut usage = estimate_raw_tensor_counts(
            source,
            fragment.output_elements() as u64,
            fragment.output_elements() as u64,
        )?;
        // The actual four slice buffers and index-operation Vec are constructed
        // only after this physical quota joins the one cumulative row charge.
        usage.host_bytes = usage
            .host_bytes
            .checked_add(
                (fragment.source_shape().len() as u64)
                    .checked_mul(
                        (4 * std::mem::size_of::<u64>()
                            + std::mem::size_of::<safemlx::ops::indexing::ArrayIndexOp<'_>>())
                            as u64,
                    )
                    .ok_or(CaptureError::Overflow)?,
            )
            .ok_or(CaptureError::Overflow)?;
        Ok(usage)
    }
    fn capture_prefill_fragment(
        &mut self,
        tensor: &MlxTensor,
        fragment: &eredu_core::capture::CapturePrefillFragment<'_, '_>,
    ) -> Option<Result<eredu_core::TensorObservation, Error>> {
        Some((|| {
            self.validate_capture_prefill_source(tensor, fragment)
                .expect("native validation")?;
            let rank = fragment.source_shape().len();
            let mut slice = ResolvedCaptureSlice {
                starts: Vec::with_capacity(rank),
                ends: Vec::with_capacity(rank),
                strides: Vec::with_capacity(rank),
                shape: Vec::with_capacity(rank),
            };
            for axis in 0..rank {
                let part = fragment
                    .selection_axis(axis)
                    .ok_or_else(|| Error::observation(CaptureError::Overflow))?;
                slice.starts.push(part.start() as u64);
                slice.ends.push(part.end() as u64);
                slice.strides.push(part.stride() as u64);
                slice.shape.push(part.elements() as u64);
            }
            let logical = fragment.assembly().logical_geometry();
            let selection = &logical.admission().plan().selections[logical.selection_index()];
            match self.transform(tensor, selection, &slice)? {
                CapturePayload::Tensor(value) => Ok(value),
                _ => Err(Error::observation(CaptureError::Invalid(
                    "prepared-media fragment returned another transform".into(),
                ))),
            }
        })())
    }

    fn estimate_partition_routed_units(
        &self,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        routed::estimate_partition(request)
    }
    fn capture_partition_routed_units(
        &mut self,
        source: &PartitionRoutedUnitCaptureSource<'_, MlxTensor>,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Option<Result<RoutedUnitCapture, Error>> {
        Some(routed::capture_partition(source, request, self.stream))
    }
    fn estimate_routed_units(
        &self,
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate_shape(shape, selection, slice)
    }
    fn capture_routed_units(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        geometry: RoutedUnitGeometry,
        slice: &ResolvedCaptureSlice,
    ) -> Option<Result<RoutedUnitCapture, Error>> {
        Some(routed::capture(source, geometry, slice, self.stream))
    }
    type Tensor = MlxTensor;
    type Error = Error;

    fn estimate_partition_source(
        &self,
        shape: &[u64],
        wait: eredu_core::BoundedCompletionWait,
    ) -> Result<CaptureUsage, CaptureError> {
        if self.partition.is_none() {
            return Err(CaptureError::Unsupported(
                "MLX capture source requires selected bounded partition completion".into(),
            ));
        }
        estimate_partition_source(shape, wait)
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
        estimate_shape(&shape, selection, slice)
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
                estimate_shape(&shape, selection, slice)
            }
            None => self.estimate(prototype, selection, slice),
        }
    }

    fn transform(
        &mut self,
        tensor: &MlxTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Error> {
        let stream = self.stream;
        // Reservation precedes even this evaluation. No observation-owned lazy
        // source is kept while later model blocks execute.
        tensor.as_array().evaluated()?;
        #[cfg(test)]
        test_transform_failure()?;
        if let CaptureTransform::TopCandidates { count } = selection.transform {
            return self
                .candidate_values(tensor, count)
                .map(CapturePayload::Candidates);
        }
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
            CaptureTransform::TokenScores { token_ids } => {
                self.token_scores(tensor, token_ids).map_err(Into::into)
            }
            CaptureTransform::TopCandidates { .. } => {
                unreachable!("terminal extraction handled before generic slicing")
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

fn summary(flat: &Array, stream: &Stream) -> Result<CaptureSummary, Exception> {
    use crate::backend::array_copy::{CaptureCompletion, CaptureTensorNativeError, SummaryProgram};
    let elements = i32::try_from(flat.size())
        .map_err(|_| Exception::custom("summary extent exceeds native indexing"))?;
    SummaryProgram::new(elements)
        .and_then(|program| {
            program.execute(
                flat,
                stream,
                CaptureCompletion::Ordinary,
                &mut |_| Ok(()),
                &mut |count| {
                    #[cfg(test)]
                    record_host_read(count);
                    #[cfg(not(test))]
                    let _ = count;
                },
            )
        })
        .map_err(|cause| match cause {
            CaptureTensorNativeError::Native(cause) => cause,
            cause => Exception::custom(cause.to_string()),
        })
}

fn histogram(flat: &Array, edges: &[f32], stream: &Stream) -> Result<CaptureHistogram, Exception> {
    use crate::backend::array_copy::{
        CaptureCompletion, CaptureTensorNativeError, HistogramProgram,
    };
    let elements = i32::try_from(flat.size())
        .map_err(|_| Exception::custom("histogram extent exceeds native indexing"))?;
    let mut out = CaptureHistogram {
        edges: edges.to_vec(),
        counts: vec![0; edges.len() - 1],
        below: 0,
        above: 0,
        non_finite: 0,
    };
    let result = HistogramProgram::new(elements, edges).and_then(|program| {
        program.execute(
            flat,
            stream,
            CaptureCompletion::Ordinary,
            &mut |_| Ok(()),
            &mut |count| {
                #[cfg(test)]
                record_host_read(count);
                #[cfg(not(test))]
                let _ = count;
            },
            &mut |index, count| {
                out.counts[index] += count;
                Ok(())
            },
        )
    });
    let totals = result.map_err(|cause| match cause {
        CaptureTensorNativeError::Native(cause) => cause,
        cause => Exception::custom(cause.to_string()),
    })?;
    out.below = totals.below;
    out.above = totals.above;
    out.non_finite = totals.non_finite;
    Ok(out)
}

pub(super) fn observer<'a>(
    capture: &'a mut eredu_runtime::capture::CaptureSession,
    stream: &'a Stream,
    domain: Option<CaptureTokenDomain<'a>>,
    prediction: u64,
) -> impl RuntimeActivationObserver<MlxTensor, Error> + 'a {
    let host = capture.ordinary_error_custody().cloned();
    eredu_runtime::intervention::CaptureObserver::for_step(
        capture,
        NativeCapture {
            partition: None,
            stream,
            domain,
        },
        prediction,
        move |error| capture_error(error).retain_ordinary_capture(host.clone()),
    )
}

/// Portable policy failures remain directly discoverable in the public source
/// chain; native failures keep their original error and creation location.
pub(crate) fn capture_error(error: eredu_runtime::capture::CaptureExecutionError<Error>) -> Error {
    match error {
        eredu_runtime::capture::CaptureExecutionError::Admission(error) => {
            Error::observation(error)
        }
        eredu_runtime::capture::CaptureExecutionError::Backend(error) => error,
    }
}

impl NativeCapture<'_> {
    fn candidate_values(
        &mut self,
        tensor: &MlxTensor,
        count: u64,
    ) -> Result<CaptureCandidates, Error> {
        let stream = self.stream;
        // Legacy callers also supply a vocabulary vector or other leading
        // dimensions. Preserve their old last-flattened-row normalization.
        // Original-funded workers enter the direct [1, rows, V] program
        // separately and never construct this compatibility reshape.
        let normalized;
        let source = if matches!(tensor.shape(), [1, rows, _] if *rows > 0) {
            tensor.as_array()
        } else {
            let width = *tensor
                .shape()
                .last()
                .ok_or_else(|| Exception::custom("candidate source has no vocabulary"))?;
            if width <= 0 {
                return Err(Exception::custom("candidate vocabulary is empty").into());
            }
            normalized = tensor
                .as_array()
                .reshape(&[-1, width], stream)?
                .try_index_device((-1, ..), stream)?
                .reshape(&[1, 1, width], stream)?;
            &normalized
        };
        let program =
            crate::backend::array_copy::CandidateExtraction::borrowed(source.shape(), count)?;
        #[cfg(test)]
        record_host_read(1);
        let (indices, scores) = program.execute(source, stream, |_| {})?;
        #[cfg(test)]
        {
            record_host_read(count as usize);
            record_host_read(count as usize);
        }
        let indices = indices.evaluated()?;
        let scores = scores.evaluated()?;
        let candidates = indices
            .as_slice::<u32>()
            .iter()
            .zip(scores.as_slice::<f32>())
            .rev()
            .map(|(&token_id, &score)| CaptureCandidate {
                token_id,
                score,
                allowed: self
                    .domain
                    .is_none_or(|domain| domain.filter.allows(token_id)),
            })
            .collect();
        Ok(CaptureCandidates {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: CandidateLogitsSource::Original,
            candidates,
            domain: self
                .domain
                .map(|d| d.summary(*tensor.shape().last().expect("validated vocabulary") as u32)),
        })
    }
}

pub(in crate::composition::mlx) fn estimate_routed_geometry(
    geometry: &eredu_core::capture::CaptureRoutedUnitsGeometry<'_>,
) -> Result<CaptureUsage, CaptureError> {
    routed::estimate_geometry(geometry)
}

/// The shared logical sparse cost policy for a retained distributed fragment.
pub(in crate::composition::mlx::session) fn estimate_partition_routed_geometry(
    request: &PartitionRoutedUnitCaptureRequest<'_>,
) -> Result<CaptureUsage, CaptureError> {
    routed::estimate_partition(request)
}

/// Shared source dependency quota for the existing original partition worker.
/// The enclosing caller separately proves its retained partition and completion owner.
pub(crate) fn estimate_partition_source(
    shape: &[u64],
    wait: eredu_core::BoundedCompletionWait,
) -> Result<CaptureUsage, CaptureError> {
    if wait.cancellation() != eredu_core::CompletionCancellationMode::QuarantineUntilComplete {
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
