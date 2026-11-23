//! MLX memory observations for the portable text forecast driver.

use super::*;
use eredu_core::capability::{NativeExecutionMemory, NativeWorkspaceEstimate};

impl<Dispatch> NativeExecutionMemory<MlxBackend<'_, Dispatch>> for MlxSessionMechanisms {
    fn refinement_workspace() -> NativeWorkspaceEstimate {
        NativeWorkspaceEstimate {
            lower: 0,
            upper: 64,
            evidence: "native full-score/feedback, retained selection rows, categorical/noise and processing scratch envelope; F32 upper width",
        }
    }

    fn sampling_workspace() -> NativeWorkspaceEstimate {
        NativeWorkspaceEstimate {
            lower: 0,
            upper: 128,
            evidence: "MLX speculative sampling calibration v1: 32 float32/index rows per live distribution for logits processing, filtering, normalization and residual sampling; planning envelope, not a kernel allocation guarantee",
        }
    }

    fn capture_completes_per_step() -> bool {
        // Transforms evaluate and transfer synchronously.
        true
    }

    fn placement_memory(
        backend: &MlxBackend<'_, Dispatch>,
    ) -> Result<eredu_core::capability::ParameterMemoryFacts, BackendFailure> {
        crate::composition::mlx::capability::placement_memory(backend.stream())
    }
}
