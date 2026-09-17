//! Exact terminal-row extraction. Legacy and original-funded callers share it.
#[cfg(test)]
mod cpu_tests;

use super::capture_tensor::{CaptureCompletion, CaptureTensorNativeError};
use super::*;
use eredu_core::capture::CaptureCandidateGeometry;
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceOperationKind};
use safemlx::{Dtype, ops::indexing::TryIndexOp};

/// Borrowed source-shape program. Count/width are actual original selection facts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CandidateExtraction {
    shape: [i32; 3],
    count: i32,
}
#[cfg(test)]
thread_local! { static CALLS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) }; }
impl CandidateExtraction {
    #[cfg(test)]
    pub(crate) fn reset_test_counts() {
        CALLS.set((0, 0));
    }
    #[cfg(test)]
    pub(crate) fn test_counts() -> (usize, usize) {
        CALLS.get()
    }
    #[cfg(test)]
    pub(crate) fn record_host_values(count: usize) {
        CALLS.set((CALLS.get().0, CALLS.get().1 + count));
    }

    pub(crate) const ROOTS: usize = 9;
    /// Finite count, selected IDs and selected scores are separate scalar/read frontiers.
    pub(crate) const COMPLETIONS: usize = 3;
    pub(crate) fn from_geometry(
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<Self, capture_tensor::CaptureTensorNativeError> {
        let mut shape = [0; 3];
        for (out, &n) in shape.iter_mut().zip(geometry.source_shape()) {
            *out = i32::try_from(n)
                .map_err(|_| capture_tensor::CaptureTensorNativeError::GeometryOverflow)?;
        }
        Ok(Self {
            shape,
            count: geometry.count() as i32,
        })
    }
    pub(crate) fn borrowed(shape: &[i32], count: u64) -> Result<Self, Exception> {
        let [batch, rows, vocabulary] = shape else {
            return Err(Exception::custom(
                "candidate logits require batch/sequence/vocabulary axes",
            ));
        };
        if *batch != 1 || *rows <= 0 || *vocabulary <= 0 || count == 0 || count > *vocabulary as u64
        {
            return Err(Exception::custom("candidate count/source geometry differs"));
        }
        Ok(Self {
            shape: [*batch, *rows, *vocabulary],
            count: count as i32,
        })
    }
    pub(crate) fn validate_source(
        &self,
        source: &Array,
    ) -> Result<(), capture_tensor::CaptureTensorNativeError> {
        if source.shape() != self.shape {
            return Err(capture_tensor::CaptureTensorNativeError::ShapeMismatch);
        }
        if !matches!(
            source.dtype(),
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ) {
            return Err(capture_tensor::CaptureTensorNativeError::UnsupportedDtype(
                source.dtype(),
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_workspace(
        &self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        use eredu_nn::Tensor;
        context.validate_values([source])?;
        if source.shape() != self.shape || source.layout().dtype() != WorkspaceDtype::Float32 {
            return Err(context.metadata_error(format_args!("candidate workspace source differs")));
        }
        Ok(())
    }
    pub(crate) fn trace(
        &self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<Vec<WorkspaceTensor>, eredu_nn::Error> {
        self.validate_workspace(source, context)?;
        let mut outputs = context.metadata_vec(2)?;
        outputs.push(context.layout(&[self.count], WorkspaceDtype::Uint32)?);
        outputs.push(context.layout(&[self.count], WorkspaceDtype::Float32)?);
        context.execute(
            WorkspaceOperationKind::CandidateExtraction {
                vocabulary: self.shape[2] as u32,
                count: self.count as u32,
            },
            &[source],
            outputs,
        )
    }
    /// Retain each actual output before the next fallible native operation. The
    /// caller owns original source/partial recovery through failure and unwind.
    pub(crate) fn execute(
        &self,
        source: &Array,
        stream: &Stream,
        mut retain: impl FnMut(&Array),
    ) -> Result<(Array, Array), Exception> {
        self.execute_with_completion(source, stream, CaptureCompletion::Ordinary, &mut |array| {
            retain(array);
            Ok(())
        })
        .map_err(|cause| match cause {
            CaptureTensorNativeError::Native(cause) => cause,
            cause => Exception::custom(cause.to_string()),
        })
    }
    /// Same ordinary finite-check/sort/take equation, with the caller's actual
    /// completion owner and fallible retained-prefix destination.
    pub(crate) fn execute_with_completion(
        &self,
        source: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        retain: &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<(Array, Array), CaptureTensorNativeError> {
        if source.shape() != self.shape {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        // Ordinary callers keep the existing AsType semantics. The original
        // supplied-stream profile separately binds the audited floating source.
        if matches!(completion, CaptureCompletion::Original(_)) {
            self.validate_source(source)?;
        }
        completion.validate()?;
        #[cfg(test)]
        CALLS.set((CALLS.get().0 + 1, CALLS.get().1));
        let view = source.try_index_device((0, -1, ..), stream)?;
        retain(&view)?;
        let row = view.as_dtype(Dtype::Float32, stream)?;
        retain(&row)?;
        let finite = row.is_finite(stream)?;
        retain(&finite)?;
        let mask = finite.as_dtype(Dtype::Uint32, stream)?;
        retain(&mask)?;
        let count = mask.sum(false, stream)?;
        retain(&count)?;
        let count = completion.settle(&count, stream)?;
        if count.try_iter::<u32>()?.next() != Some(self.shape[2] as u32) {
            return Err(CaptureTensorNativeError::NonFiniteScores);
        }
        let sorted = safemlx::ops::argsort(&row, stream)?;
        retain(&sorted)?;
        let selected =
            sorted.try_index_device((self.shape[2] - self.count)..self.shape[2], stream)?;
        retain(&selected)?;
        let ids = selected.contiguous(false, stream)?;
        retain(&ids)?;
        // These are the actual ArgSort IDs; high-level embedding/index-input
        // validation is a different producer and is not substituted here.
        let scores = row.take(&ids, stream)?;
        retain(&scores)?;
        Ok((ids, scores))
    }
    /// Reads exactly K IDs and K scores after separate exact completions. The
    /// descending order (including equal-score tie behavior) is unchanged.
    pub(crate) fn read_with_completion(
        &self,
        ids: &Array,
        scores: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        mut emit: impl FnMut(u32, f32) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<(), CaptureTensorNativeError> {
        let ids = completion.settle(ids, stream)?;
        let scores = completion.settle(scores, stream)?;
        let ids = ids.try_as_slice::<u32>()?;
        let scores = scores.try_as_slice::<f32>()?;
        if ids.len() != self.count as usize || scores.len() != ids.len() {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        #[cfg(test)]
        Self::record_host_values(ids.len() + scores.len());
        for (&id, &score) in ids.iter().zip(scores).rev() {
            emit(id, score)?;
        }
        Ok(())
    }
    /// Extra leaf controls only; original source/claim/publication is independent.
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<(Array, Array)>(),
            size_of::<Result<(Array, Array), CaptureTensorNativeError>>(),
            size_of::<Result<(), CaptureTensorNativeError>>(),
            size_of::<
                std::iter::Rev<
                    std::iter::Zip<std::slice::Iter<'static, u32>, std::slice::Iter<'static, f32>>,
                >,
            >(),
            size_of::<(&[u32], &[f32])>(),
            size_of::<Result<&[u32], safemlx::error::AsSliceError>>(),
            size_of::<Result<&[f32], safemlx::error::AsSliceError>>(),
            safemlx::OperationEvent::nested_completion_control_bytes::<1>()?.checked_mul(Self::COMPLETIONS)?,
            safemlx::EvaluatedArray::iteration_control_bytes::<u32>()?,
            safemlx::EvaluatedArray::iteration_control_bytes::<f32>()?,
            safemlx::ops::indexing::inline_basic_index_control_bytes()?.checked_mul(2)?,
            safemlx::PreparedArrayClone::control_bytes()?
                .checked_add(Array::inspection_clone_handle_bytes())?
                .checked_mul(Self::ROOTS)?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
    use eredu_nn::Tensor;
    #[test]
    fn actual_terminal_program_and_trace_cover_sort_branch_full_backing_and_failure_prefix() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        for vocabulary in [2048i32, 2049] {
            for count in [1, vocabulary] {
                let mut values = vec![f32::INFINITY; 6 * vocabulary as usize];
                for (i, value) in values[5 * vocabulary as usize..].iter_mut().enumerate() {
                    *value = i as f32 * 0.25;
                }
                let source = Array::from_slice(&values, &[1, 6, vocabulary]);
                let program = CandidateExtraction::borrowed(source.shape(), count as u64).unwrap();
                let mut roots = Vec::with_capacity(CandidateExtraction::ROOTS);
                let (ids, scores) = program
                    .execute(&source, &stream, |a| roots.push(a.clone()))
                    .unwrap();
                assert_eq!(roots.len(), CandidateExtraction::ROOTS);
                drop(source);
                let ids = ids.evaluated().unwrap();
                let scores = scores.evaluated().unwrap();
                for (i, (&id, &score)) in ids
                    .as_slice::<u32>()
                    .iter()
                    .zip(scores.as_slice::<f32>())
                    .rev()
                    .enumerate()
                {
                    assert_eq!(id, vocabulary as u32 - 1 - i as u32);
                    assert_eq!(score, id as f32 * 0.25);
                }
                let mut quoted = None;
                for rows in [1, 6] {
                    let context =
                        WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
                    let source = WorkspaceTensor::from_f32_slice(
                        &values[..rows as usize * vocabulary as usize],
                        &[1, rows, vocabulary],
                        &context,
                    )
                    .unwrap();
                    context.begin_span();
                    let program =
                        CandidateExtraction::borrowed(&[1, rows, vocabulary], count as u64)
                            .unwrap();
                    let outputs = program.trace(&source, &context).unwrap();
                    let report = context.report(&outputs).unwrap();
                    assert_eq!(report.host_workspace_bytes, Some(0));
                    assert!(report.tensor_buffers.total_bytes.unwrap() > vocabulary as u64 * 4);
                    if let Some(previous) = quoted {
                        assert_eq!(report.tensor_buffers.total_bytes, previous);
                    }
                    quoted = Some(report.tensor_buffers.total_bytes);
                }
            }
        }
        let source = Array::from_slice(&[0.0f32, 1.0, f32::NAN, 2.0], &[1, 1, 4]);
        let program = CandidateExtraction::borrowed(source.shape(), 2).unwrap();
        let mut roots = Vec::with_capacity(CandidateExtraction::ROOTS);
        let error = program
            .execute(&source, &stream, |a| roots.push(a.clone()))
            .unwrap_err();
        assert!(error.to_string().contains("finite raw logits"));
        assert_eq!(
            roots.len(),
            5,
            "all actual finite-test roots, before sort or result allocation"
        );
        drop(source);
        assert_eq!(roots[4].evaluated().unwrap().as_slice::<u32>(), &[3]);
    }
}
