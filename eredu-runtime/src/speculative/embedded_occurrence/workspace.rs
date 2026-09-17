//! Exact geometry handoff from a selected equation/state projection to accounting.
use super::{EmbeddedInvocation, EmbeddedOccurrenceError, Phase};
use eredu_core::{InferenceGeometry, OutputDemand};

/// A checked coordinate descriptor, not proof of source identity or an execution
/// grant. Native consumers must bind the actual state revision and selected
/// architecture equation; equal geometry alone cannot authorize their work.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct EmbeddedInvocationWorkspace {
    invocation: EmbeddedInvocation,
    geometry: InferenceGeometry,
}
impl EmbeddedInvocationWorkspace {
    /// Target state coordinates and vocabulary demand already present in the
    /// shared invocation. Each prefill invocation is one actual bounded span.
    pub fn target(invocation: EmbeddedInvocation) -> Result<Self, EmbeddedOccurrenceError> {
        let output = invocation
            .target_output()
            .ok_or(EmbeddedOccurrenceError::Geometry)?;
        Self::target_with_readout(invocation, output)
    }
    /// Actual target readout selected after composing observation requirements.
    /// It may widen the invocation's demand, but cannot discard required scores.
    /// Phase, width and target frontier remain those of the original invocation.
    pub fn target_with_readout(
        invocation: EmbeddedInvocation,
        actual_output: OutputDemand,
    ) -> Result<Self, EmbeddedOccurrenceError> {
        let minimum = invocation.target_output().ok_or(EmbeddedOccurrenceError::Geometry)?;
        let accepted = match minimum {
            OutputDemand::StateOnly => true,
            OutputDemand::LastPosition => actual_output != OutputDemand::StateOnly,
            OutputDemand::Sequence => actual_output == OutputDemand::Sequence,
        };
        if !accepted { return Err(EmbeddedOccurrenceError::Geometry); }
        Self::new(invocation, invocation.target_frontier(), actual_output)
    }
    /// Prediction-local frontier read from the actual architecture state, and
    /// output demand of the actual selected equation/observation path. Neither
    /// is inferred from target positions or an independent-draft offset.
    pub fn prediction(
        invocation: EmbeddedInvocation,
        actual_state_frontier: u64,
        output: OutputDemand,
    ) -> Result<Self, EmbeddedOccurrenceError> {
        if invocation.target_output().is_some() {
            return Err(EmbeddedOccurrenceError::Geometry);
        }
        if let Some(span) = invocation.prefill_span() {
            if invocation.phase() != Phase::PredictionPrefill
                || actual_state_frontier != span.seed_start
            {
                return Err(EmbeddedOccurrenceError::Geometry);
            }
        }
        Self::new(invocation, actual_state_frontier, output)
    }
    fn new(
        invocation: EmbeddedInvocation,
        frontier: u64,
        output: OutputDemand,
    ) -> Result<Self, EmbeddedOccurrenceError> {
        let positions =
            u64::try_from(invocation.positions()).map_err(|_| EmbeddedOccurrenceError::Overflow)?;
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: frontier,
            input_positions: positions,
            max_output_tokens: 0,
            prefill_chunk_positions: positions,
            output,
        };
        geometry
            .validate_fixed()
            .map_err(|_| EmbeddedOccurrenceError::Geometry)?;
        Ok(Self {
            invocation,
            geometry,
        })
    }
    /// Exact phase, canonical target coordinate, depth and sequence descriptor.
    pub const fn invocation(self) -> EmbeddedInvocation {
        self.invocation
    }
    /// Full geometry to compare with the completed architecture equation report.
    pub const fn geometry(self) -> InferenceGeometry {
        self.geometry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observation_readout_preserves_target_coordinates_and_required_scores() {
        let target = EmbeddedInvocation {
            phase: Phase::TargetPrefill, target_frontier: 11, positions: 3,
            span: None, target_output: Some(OutputDemand::StateOnly),
        };
        let observed = EmbeddedInvocationWorkspace::target_with_readout(target, OutputDemand::Sequence).unwrap();
        assert_eq!(observed.invocation(), target);
        assert_eq!(observed.geometry().cached_positions, 11);
        assert_eq!(observed.geometry().input_positions, 3);
        assert_eq!(observed.geometry().output, OutputDemand::Sequence);
        let verification = EmbeddedInvocation { target_output: Some(OutputDemand::Sequence), ..target };
        assert!(EmbeddedInvocationWorkspace::target_with_readout(verification, OutputDemand::LastPosition).is_err());
        let prediction = EmbeddedInvocation { target_output: None, ..target };
        assert!(EmbeddedInvocationWorkspace::target_with_readout(prediction, OutputDemand::Sequence).is_err());
    }
    #[test]
    fn prediction_frontier_is_separate_from_target_and_seed_is_exact() {
        use eredu_core::speculative::SpeculativePrefillSpan;
        let span = SpeculativePrefillSpan {
            prompt_tokens: 7,
            input_start: 3,
            input_end: 6,
            position: 14,
            hidden_start: 2,
            token_start: 3,
            sequence: 3,
            seed_start: 31,
        };
        let seed = EmbeddedInvocation {
            phase: Phase::PredictionPrefill,
            target_frontier: 14,
            positions: 3,
            span: Some(span),
            target_output: None,
        };
        assert!(EmbeddedInvocationWorkspace::target(seed).is_err());
        assert!(
            EmbeddedInvocationWorkspace::prediction(seed, 14, OutputDemand::StateOnly).is_err()
        );
        let projected =
            EmbeddedInvocationWorkspace::prediction(seed, 31, OutputDemand::StateOnly).unwrap();
        assert_eq!(projected.geometry().cached_positions, 31);
        assert_eq!(projected.geometry().input_positions, 3);
        let proposal = EmbeddedInvocation {
            phase: Phase::FusedProposal,
            target_frontier: 18,
            positions: 2,
            span: None,
            target_output: None,
        };
        let projected =
            EmbeddedInvocationWorkspace::prediction(proposal, 37, OutputDemand::Sequence).unwrap();
        assert_eq!(projected.invocation().source_positions(), 1);
        assert_eq!(projected.geometry().input_positions, 2);
        assert_eq!(projected.geometry().cached_positions, 37);
    }
}
