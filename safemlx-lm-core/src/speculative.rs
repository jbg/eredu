//! High-level contract for whole-session speculative execution backends.

use crate::backend::{Completion, Submission};
use serde::{Deserialize, Serialize};

/// Relationship between target and assistant execution placements.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeExecutionTopology {
    /// Target and assistant operations share one ordered execution queue.
    #[default]
    Single,
    /// Distinct queues share one device and can use ordered handoffs.
    SameDeviceSplit,
    /// Target and assistant use different devices and require transfers.
    CrossDeviceSplit,
}

impl std::fmt::Display for SpeculativeExecutionTopology {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Single => "single",
            Self::SameDeviceSplit => "same-device-split",
            Self::CrossDeviceSplit => "cross-device-split",
        })
    }
}

/// Backend-owned first-token output and assistant seed state.
#[derive(Debug)]
pub struct SpeculativePrefill<State, Logits> {
    /// Opaque logits used by the selected backend sampler.
    pub logits: Logits,
    /// Backend state from which the first proposal round begins.
    pub state: State,
    /// Number of prompt tokens evaluated by the target.
    pub evaluated_tokens: usize,
}

/// Result of committing one exact target verification transaction.
#[derive(Debug)]
pub struct SpeculativeCommit<State> {
    /// Assistant seed state matching the committed target cache.
    pub state: State,
    /// Target tokens replayed while restoring the exact retained prefix.
    pub replayed_tokens: usize,
}

/// Whole-session speculative execution contract.
///
/// Tensor values, execution queues, caches, model state, logits, native
/// completions, and errors remain opaque associated types. The contract models
/// only high-level prefill, proposal, verification, and exact commit actions;
/// it deliberately does not define primitive tensor operations.
pub trait SpeculativeExecutor {
    /// Backend-owned model input accepted by prefill submission.
    type Input;
    /// Complete backend-owned target cache.
    type Cache;
    /// Target state used to seed one proposal round.
    type TargetState;
    /// Private, discardable assistant state.
    type DraftState: Clone;
    /// Exact target-cache checkpoint marker.
    type CacheCheckpoint;
    /// Retained target verification output.
    type Verification;
    /// Opaque logits consumed by the backend's sampling adapter.
    type Logits;
    /// Backend execution assignment for one operation.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact completion for submitted verification work.
    type Completion: Completion<Error = Self::Error>;
    /// Optional backend-specific component telemetry.
    type Telemetry: Default;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Maximum proposals supported in one verification transaction.
    fn max_proposals(&self) -> usize {
        usize::MAX
    }

    /// Enables optional component telemetry.
    fn set_telemetry_enabled(&mut self, _enabled: bool) {}

    /// Whether optional component telemetry is available.
    fn supports_telemetry(&self) -> bool {
        false
    }

    /// Resolves and drains assistant telemetry since the previous call.
    fn take_telemetry(&mut self) -> Result<Self::Telemetry, Self::Error> {
        Ok(Self::Telemetry::default())
    }

    /// Resolves telemetry retained by one target verification output.
    fn take_verification_telemetry(
        &mut self,
        _output: &mut Self::Verification,
    ) -> Result<Self::Telemetry, Self::Error> {
        Ok(Self::Telemetry::default())
    }

    /// Whether cloned assistant state can be promoted after an exact bonus match.
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    /// Prefills the target and returns first-token logits plus assistant seed state.
    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'context;

    /// Starts one private proposal round sized to the available output budget.
    fn begin_proposal<'a>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>;

    /// Produces opaque next-token logits and advances private assistant state.
    fn proposal_logits<'a>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Captures the exact cache boundary before target verification.
    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint;

    /// Submits verification of the last committed token and proposal block.
    ///
    /// Implementations materialize token tensors internally and return an exact
    /// completion retaining every resource required by the submission.
    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>;

    /// Borrows the opaque verification logits.
    fn verification_logits(output: &Self::Verification) -> &Self::Logits;

    /// Commits exactly the requested verified inputs and restores matching seed state.
    fn commit_verification<'a>(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[derive(Debug, Clone, Copy)]
    struct Done;

    impl Completion for Done {
        type Error = Infallible;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    struct MockExecutor;

    struct MockVerification {
        tokens: Vec<u32>,
        logits: Vec<f32>,
    }

    impl SpeculativeExecutor for MockExecutor {
        type Input = Vec<u32>;
        type Cache = Vec<u32>;
        type TargetState = usize;
        type DraftState = Vec<u32>;
        type CacheCheckpoint = usize;
        type Verification = MockVerification;
        type Logits = Vec<f32>;
        type Context<'a> = ();
        type Completion = Done;
        type Telemetry = ();
        type Error = Infallible;

        fn prefill<'context>(
            &mut self,
            input: Self::Input,
            cache: &mut Self::Cache,
            _: Self::Context<'context>,
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
        where
            Self: 'context,
        {
            cache.extend_from_slice(&input);
            Ok(SpeculativePrefill {
                logits: vec![0.0, 1.0],
                state: cache.len(),
                evaluated_tokens: input.len(),
            })
        }

        fn begin_proposal<'a>(
            &mut self,
            _: &Self::TargetState,
            last_token: u32,
            _: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::DraftState, Self::Error> {
            Ok(vec![last_token])
        }

        fn proposal_logits<'a>(
            &mut self,
            state: &mut Self::DraftState,
            last_token: u32,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            state.push(last_token + 1);
            Ok(vec![0.0, 1.0])
        }

        fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
            cache.len()
        }

        fn submit_verification<'a>(
            &mut self,
            input_tokens: &[u32],
            cache: &mut Self::Cache,
            _: Self::Context<'a>,
        ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
            cache.extend_from_slice(input_tokens);
            Ok(Submission {
                output: MockVerification {
                    tokens: input_tokens.to_vec(),
                    logits: vec![0.0, 1.0],
                },
                completion: Done,
            })
        }

        fn verification_logits(output: &Self::Verification) -> &Self::Logits {
            &output.logits
        }

        fn commit_verification<'a>(
            &mut self,
            output: Self::Verification,
            draft_state: Self::DraftState,
            cache: &mut Self::Cache,
            checkpoint: Self::CacheCheckpoint,
            verified_inputs: usize,
            _: Self::Context<'a>,
        ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
            assert!(!output.tokens.is_empty());
            cache.truncate(checkpoint + verified_inputs);
            Ok(SpeculativeCommit {
                state: draft_state.len(),
                replayed_tokens: 0,
            })
        }
    }

    #[test]
    fn mock_executor_prefill_propose_verify_and_commit_without_a_tensor_runtime() {
        let mut executor = MockExecutor;
        let mut cache = Vec::new();
        let prefill = executor.prefill(vec![4, 5], &mut cache, ()).unwrap();
        let mut draft = executor.begin_proposal(&prefill.state, 5, 2, ()).unwrap();
        assert_eq!(
            executor.proposal_logits(&mut draft, 5, ()).unwrap(),
            [0.0, 1.0]
        );
        let checkpoint = MockExecutor::checkpoint(&cache);
        let submission = executor
            .submit_verification(&[5, 6], &mut cache, ())
            .unwrap();
        submission.completion.wait().unwrap();
        let commit = executor
            .commit_verification(submission.output, draft, &mut cache, checkpoint, 1, ())
            .unwrap();
        assert_eq!(cache, [4, 5, 5]);
        assert_eq!(commit.replayed_tokens, 0);
    }

    #[test]
    fn execution_topology_is_a_portable_schema() {
        let topology = SpeculativeExecutionTopology::CrossDeviceSplit;
        let encoded = serde_json::to_string(&topology).unwrap();
        assert_eq!(encoded, "\"cross_device_split\"");
        assert_eq!(
            serde_json::from_str::<SpeculativeExecutionTopology>(&encoded).unwrap(),
            topology
        );
    }
}
