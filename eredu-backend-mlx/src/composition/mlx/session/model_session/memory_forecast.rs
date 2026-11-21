//! MLX memory observations for the portable text forecast driver.

use super::*;
use crate::composition::mlx::capability::MlxAllocatorObservations;
use eredu_core::capability::{NativeExecutionMemory, NativeWorkspaceEstimate};

impl<Dispatch> NativeExecutionMemory<MlxBackend<'_, Dispatch>> for MlxSessionMechanisms {
    type Allocator = MlxAllocatorObservations;

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

    fn shares_allocator() -> bool {
        true
    }

    fn capture_completes_per_step() -> bool {
        // Transforms evaluate and transfer synchronously.
        true
    }

    fn host_execution(backend: &MlxBackend<'_, Dispatch>) -> Result<bool, Error> {
        backend
            .stream()
            .get_device()
            .and_then(|d| d.get_type())
            .map(|device| device == safemlx::DeviceType::Cpu)
            .map_err(|error| {
                BackendFailure::from_error(error)
                    .with_operation("forecast execution device")
                    .into()
            })
    }
}
