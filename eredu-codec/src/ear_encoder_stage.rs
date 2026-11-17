//! Portable execution of a bound pipeline stage.

use crate::ear_encoder_layered::EncoderKernel;
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::Tensor;
use eredu_runtime::StatelessComponentState;
use eredu_runtime::{BoundComponentPipeline, ComponentAdapter, ComponentDefinition};

/// Typed architecture retained by a selected stage.
pub type EarEncoderPipelineArchitecture<B> =
    ComponentAdapter<ComponentDefinition, EncoderKernel<B>>;

/// One codec-selected finite encoder stage. The shared distributed driver owns
/// parameter leases, transport, exact completion and commit agreement.
pub struct EarEncoderPipeline<P> {
    execution: P,
}
impl<P> EarEncoderPipeline<P> {
    /// Wraps a bound component and its retained stage facts.
    pub fn new(execution: P) -> Self {
        Self { execution }
    }

    /// Borrows the bound execution for residency and disk telemetry.
    pub fn pipeline(&self) -> &P {
        &self.execution
    }

    /// Converts a finite spectrum to completed frame-major latents on every rank.
    /// Every rank supplies the same recorded request descriptor and commit epoch.
    /// The enclosing driver consumes or publishes latents only for `Committed`.
    /// Spectral preprocessing and the borrowed quantizer remain separate components.
    pub fn forward<B>(
        &mut self,
        spectrum: &B::Tensor,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DistributedCommitOutcome), BackendFailure>
    where
        B: eredu_nn::DistributedNeuralBackend,
        P: BoundComponentPipeline<EarEncoderPipelineArchitecture<B>, B, StatelessComponentState>,
    {
        let mut state = StatelessComponentState::new().map_err(BackendFailure::from_error)?;
        self.execution.execute(
            spectrum,
            vec![spectrum.clone()],
            &mut state,
            descriptor,
            epoch,
            stream,
        )
    }
}
