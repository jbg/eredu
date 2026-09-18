//! Actual destinations for the shared logical aggregate constructor.
use super::*;
use eredu_nn::workspace::{WorkspaceMetadataError, HostMetadataFunding};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(in crate::capture) enum ConstructionError {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("{0}")]
    Unsupported(&'static str),
    #[error("aggregate metadata arithmetic overflow")]
    Overflow,
    #[error(transparent)]
    Capture(#[from] CaptureError),
    #[error(transparent)]
    Window(#[from] CaptureWindowError),
    #[error(transparent)]
    Preview(#[from] preview::PreviewError),
    #[error(transparent)]
    Metadata(#[from] eredu_nn::Error),
    #[error(transparent)]
    Summary(#[from] reduction::CaptureSummaryError),
    #[error(transparent)]
    Histogram(#[from] reduction::CaptureHistogramError),
}
impl ConstructionError {
    pub(in crate::capture) fn into_capture(self) -> CaptureError {
        match self {
            Self::Invalid(message) => CaptureError::Invalid(message.into()),
            Self::Unsupported(message) => CaptureError::Unsupported(message.into()),
            Self::Overflow => CaptureError::Overflow,
            Self::Capture(error) => error,
            Self::Window(error) => error.into(),
            Self::Preview(error) => error.into(),
            Self::Summary(error) => error.into(),
            Self::Histogram(error) => error.into(),
            // The ordinary destination never emits Metadata. Keep a total
            // ordinary adapter without discarding a future producer's cause.
            Self::Metadata(error) => CaptureError::Invalid(error.to_string()),
        }
    }
}
#[derive(Clone, Copy)]
pub(in crate::capture) struct Metadata<'a>(Option<&'a HostMetadataFunding>);
impl<'a> Metadata<'a> {
    pub(in crate::capture) fn ordinary() -> Self {
        Self(None)
    }
    pub(in crate::capture) fn original(funding: &'a HostMetadataFunding) -> Self {
        Self(Some(funding))
    }
    pub(in crate::capture) fn controls(self, bytes: usize) -> Result<(), ConstructionError> {
        match self.0 {
            None => Ok(()),
            Some(funding) => funding.reserve_metadata(bytes).map_err(|cause| {
                ConstructionError::Metadata(WorkspaceMetadataError::from(cause).into())
            }),
        }
    }
    pub(in crate::capture) fn vec<T>(self, capacity: usize) -> Result<Vec<T>, ConstructionError> {
        match self.0 {
            None => Ok(Vec::with_capacity(capacity)),
            Some(funding) => funding.metadata_vec(capacity).map_err(Into::into),
        }
    }
    pub(in crate::capture) fn deferred_vec<T>(self, count: usize) -> Result<(), ConstructionError> {
        self.controls(
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<T>(count)
                .ok_or(ConstructionError::Overflow)?,
        )
    }
    pub(in crate::capture) fn copy<T: Copy>(
        self,
        input: &[T],
    ) -> Result<Vec<T>, ConstructionError> {
        let mut out = self.vec(input.len())?;
        out.extend_from_slice(input);
        Ok(out)
    }
    pub(in crate::capture) fn zeros<T: Copy>(
        self,
        count: usize,
        zero: T,
    ) -> Result<Vec<T>, ConstructionError> {
        let mut out = self.vec(count)?;
        out.resize(count, zero);
        Ok(out)
    }
    pub(in crate::capture) fn text(self, input: &str) -> Result<String, ConstructionError> {
        match self.0 {
            None => Ok(input.to_owned()),
            Some(funding) => funding
                .metadata_string(format_args!("{input}"))
                .map_err(Into::into),
        }
    }
}
impl WindowReductions {
    pub(super) fn construction_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Option<Self>, ConstructionError>>(),
            size_of::<SpeculativePrefillReductions>(),
            size_of::<SpeculativePrefillReduction>(),
            size_of::<State>(),
            size_of::<Geometry>(),
            size_of::<preview::Plan>(),
            size_of::<Vec<State>>(),
            size_of::<Vec<SpeculativePrefillReduction>>(),
            size_of::<CaptureUsage>(),
            size_of::<Result<CaptureUsage, CaptureError>>(),
            size_of::<(usize, u64, u64, bool)>(),
            size_of::<[SpeculativeActivationPhase; 2]>(),
            size_of::<(
                &AdmittedCapturePlan,
                &mut CaptureLedger,
                &[SpeculativeCaptureScope],
                SpeculativePrefillReductionGeometry,
                SpeculativeActivationOrigin,
                u64,
                bool,
                Metadata<'_>,
            )>(),
            size_of::<ConstructionError>(),
            size_of::<CaptureWindowError>(),
            size_of::<preview::PreviewError>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
