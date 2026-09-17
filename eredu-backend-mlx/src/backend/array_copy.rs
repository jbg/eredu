//! One borrowed isolated-copy program, shared by native work and cold tracing.

use crate::backend::nn::workspace::ExistingArrayProjection;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTensor};
use eredu_nn::{IsolatedCopyMechanism, WorkspaceIsolatedCopy, isolated_copy};
use safemlx::{Array, Stream, error::Exception};
use std::cell::RefCell;

mod host;
mod routed;
pub(crate) use routed::{execute_speculative_routed_capture, speculative_routed_capture_control_bytes, CompletedRoutedCaptureSource, CompletedPartitionRoutedCaptureSource, PartitionRoutedCaptureLayout, RoutedCaptureTransfer, RoutedInterventionReadError};
pub(crate) use host::PreparedHostArrayCopy;

mod original;
pub(crate) use original::{
    RealtimeCopyPlan, RealtimeCopyContext,
    OriginalCopyCause, OriginalCopyExecution, OriginalCopyFailure, OriginalCopyLayoutBuilder,
    OriginalCopyPlan, OriginalResumeCopyPopulation, PreparedOriginalCopy, PreparedSavedHostCopy,
    SavedHostCopyPlan,
};

pub(crate) use capture_tensor::{PreparedCaptureHistogram, PreparedCaptureSummary};
mod histogram;
pub(crate) use histogram::{HistogramProgram, HistogramTotals};
mod summary;
pub(crate) use summary::{SummaryPopulation, SummaryProgram};

mod candidates;
mod capture_tensor;
pub(crate) use candidates::CandidateExtraction;
pub(crate) use capture_tensor::{
    CaptureCompletion, CaptureTensorExecutionError, CaptureTensorNativeError,
    GeneratedCaptureRetention, PreparedCaptureFragment, PreparedCaptureTensor,
    ScheduledCaptureTensorExecutionError, Selection as CaptureTensorSelection,
    capture_original_control_bytes, capture_original_error, capture_original_error_control_bytes,
    execute_speculative_candidates, execute_speculative_candidates_cpu, execute_speculative_capture, execute_speculative_capture_cpu, execute_speculative_histogram, execute_speculative_histogram_cpu,
    execute_speculative_summary, execute_speculative_summary_cpu, execute_speculative_token_scores, execute_speculative_token_scores_cpu,
    speculative_candidate_control_bytes, speculative_capture_control_bytes,
    speculative_histogram_control_bytes, speculative_summary_control_bytes,
    speculative_token_score_control_bytes,
};

mod prepared_source;
pub(crate) use prepared_source::OriginalPreparedArrayCopySource;
mod pending_input;
pub(crate) use pending_input::{
    OriginalPendingTokenCopy, PendingTokenInputError, PendingTokenMetadataError,
    PendingTokenNativePopulation, PendingTokenSourceCause, PreparedPendingTokenArray,
    PreparedPendingTokenInput, RegisteredArrayCopy, RegisteredArrayCopyCustody,
};

/// Binds the exact source without cloning a handle or inspecting its descriptor.
/// The caller retains source custody and allocation authority through completion
/// and failure recovery, and keeps the source settled and exclusive while using
/// its trace. A borrow does not prevent external aliases changing native data.
/// This program grants neither admission nor ownership.
pub(crate) struct IsolatedArrayCopy<'a> {
    source: &'a Array,
}

impl<'a> IsolatedArrayCopy<'a> {
    /// Actual borrowed operand; inspection does not grant copy or storage authority.
    pub(crate) fn source(&self)->&'a Array {self.source}

    pub(crate) const fn new(source: &'a Array) -> Self {
        Self { source }
    }

    /// Allocates metadata only. One projection must cover the complete source
    /// inventory so distinct logical views preserve their shared backing root.
    pub(crate) fn trace(
        &self,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<WorkspaceTensor, eredu_nn::Error> {
        let source = projection.project(self.source)?;
        isolated_copy(WorkspaceIsolatedCopy::new(source, projection.context()))
    }

    pub(crate) fn copy(self, stream: &Stream) -> Result<Array, Exception> {
        isolated_copy(NativeCopy {
            source: self.source,
            stream,
            roots: None,
        })
    }

    /// Retains each produced descriptor before the following fallible operation.
    /// The caller owns this collector through exact native recovery; temporary
    /// roots are not automatically published as saved destination allocations.
    pub(crate) fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<Array, Exception> {
        roots.borrow_mut().reserve(2);
        isolated_copy(NativeCopy {
            source: self.source,
            stream,
            roots: Some(roots),
        })
    }
}

struct NativeCopy<'a> {
    source: &'a Array,
    stream: &'a Stream,
    roots: Option<&'a RefCell<Vec<Array>>>,
}

impl IsolatedCopyMechanism for NativeCopy<'_> {
    type Value = Array;
    type Error = Exception;

    fn contiguous(&self) -> Result<Array, Exception> {
        if safemlx::OriginalScopeObserver::try_current()?.is_some() {
            safemlx::OperationEvent::validate_nested_completion(1)?;
        }
        let contiguous = self.source.contiguous(false, self.stream)?;
        if let Some(roots) = self.roots {
            let retained = contiguous.try_clone_handle()?;
            roots.borrow_mut().push(retained);
        }
        Ok(contiguous)
    }

    fn deep_copy(&self, contiguous: Array) -> Result<Array, Exception> {
        let copy = if let Some(observer) = safemlx::OriginalScopeObserver::try_current()? {
            // Complete this one actual frontier, then borrow that completion.
            // Array::deep_clone would evaluate again through into_evaluated,
            // creating a second unplanned tracking/worker population.
            safemlx::OperationEvent::complete_nested([&contiguous], self.stream)?;
            contiguous
                .completed_in_original_scope(&observer)?
                .deep_clone()?
                .into_array()
                .expect("deep copy owns its completed storage")
        } else {
            contiguous.deep_clone()?
        };
        if let Some(roots) = self.roots {
            let retained = copy.try_clone_handle()?;
            roots.borrow_mut().push(retained);
        }
        Ok(copy)
    }
}

#[cfg(test)]
mod tests;

mod token_scores;
pub(crate) use token_scores::TokenScoreProgram;

mod capture_population;
pub(crate) use capture_population::CaptureNativePopulation;
