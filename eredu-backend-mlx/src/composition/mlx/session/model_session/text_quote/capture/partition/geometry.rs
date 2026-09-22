//! Exact invocation coordinates retained beside the original loaded source.
use super::*;
use eredu_core::capture::{CaptureInvocationShape, CaptureInvocationWindow};

/// Physical model axes and their optional logical row placement. These facts
/// describe the admitted callback; they create no role or transport authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::composition::mlx) struct PartitionCaptureInvocation {
    pub(in crate::composition::mlx) phase: CapturePhase,
    pub(in crate::composition::mlx) prediction: u64,
    pub(in crate::composition::mlx) physical: CaptureInvocationShape,
    pub(in crate::composition::mlx) window: Option<CaptureInvocationWindow>,
}
impl PartitionCaptureInvocation {
    pub(super) fn receipt_window(
        self,
    ) -> Option<eredu_core::capture::PartitionCaptureInvocationWindow> {
        self.window.map(
            |window| eredu_core::capture::PartitionCaptureInvocationWindow {
                physical: self.physical,
                window,
            },
        )
    }

    pub(in crate::composition::mlx) fn logical(
        self,
    ) -> Result<CaptureInvocationShape, eredu_core::capture::CaptureError> {
        self.window
            .map_or(Ok(self.physical), |window| window.validate(self.physical))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FrameGeometry {
    Text(eredu_core::InferenceGeometry),
    Invocation(PartitionCaptureInvocation),
}
impl FrameGeometry {
    pub(super) fn receipt_window(
        self,
    ) -> Option<eredu_core::capture::PartitionCaptureInvocationWindow> {
        match self {
            Self::Text(_) => None,
            Self::Invocation(value) => value.receipt_window(),
        }
    }

    pub(super) fn phase(self, prediction: u64) -> CapturePhase {
        match self {
            Self::Text(_) if prediction == 0 => CapturePhase::Prefill,
            Self::Text(_) => CapturePhase::Decode,
            Self::Invocation(value) => value.phase,
        }
    }
    pub(super) fn prefill(self, prediction: u64) -> Option<eredu_core::InferenceGeometry> {
        match self {
            Self::Text(value) if prediction == 0 => Some(value),
            _ => None,
        }
    }
    pub(super) fn invocation(
        self,
    ) -> Result<Option<CaptureInvocationShape>, eredu_core::capture::CaptureError> {
        match self {
            Self::Text(_) => Ok(None),
            Self::Invocation(value) => value.logical().map(Some),
        }
    }
    pub(super) fn matches_prediction(self, prediction: u64) -> bool {
        match self {
            Self::Text(_) => true,
            Self::Invocation(value) => value.prediction == prediction,
        }
    }
}
