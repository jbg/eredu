//! Causal evidence shared by ordinary append and every draft-source verification.
use super::autoregressive::{AutoregressiveMechanisms, AutoregressivePass};
use eredu_core::Submission;

/// Completed causal output and the exact input tokens whose K/V can be retained.
/// Proposal confidence or likelihood evidence belongs to the separate draft block.
#[derive(Clone)]
pub struct CausalVerification<O, I = Vec<u32>> {
    pub(super) output: O,
    pub(super) inputs: I,
}
impl<O, I> CausalVerification<O, I> {
    /// Retains one invocation's output and exact backend-native or host input.
    pub fn new(output: O, inputs: I) -> Self {
        Self { output, inputs }
    }
    /// Exact input geometry retained for accepted-prefix publication.
    pub fn inputs(&self) -> &I {
        &self.inputs
    }
    /// Borrows output bookkeeping for completed telemetry delivery.
    pub fn output_mut(&mut self) -> &mut O {
        &mut self.output
    }
    /// Moves completed output and inputs into source-specific cache commitment.
    pub fn into_parts(self) -> (O, I) {
        (self.output, self.inputs)
    }
    /// Immutable causal output for sampling, observation and accepted-prefix decisions.
    pub fn output(&self) -> &O {
        &self.output
    }
}

pub(super) fn submit<M: AutoregressiveMechanisms>(
    model: &mut M::Model,
    tokens: &[u32],
    cache: &mut M::State,
    context: M::Context<'_>,
) -> Result<Submission<CausalVerification<M::Output>, M::Completion>, M::Error> {
    if tokens.is_empty() {
        return Err(M::invalid("causal invocation requires nonempty input"));
    }
    let output = M::decode(
        model,
        tokens,
        cache,
        AutoregressivePass::Verification,
        context,
    )?;
    let completion = M::completion(&output, context)?;
    Ok(Submission {
        output: CausalVerification {
            output,
            inputs: tokens.to_vec(),
        },
        completion,
    })
}

pub(super) fn logits<M: AutoregressiveMechanisms>(
    output: &CausalVerification<M::Output>,
    position: usize,
    context: M::Context<'_>,
) -> Result<M::Logits, M::Error> {
    if position >= output.inputs.len() {
        return Err(M::invalid("causal score position exceeds its invocation"));
    }
    M::logits(&output.output, position, context)
}
