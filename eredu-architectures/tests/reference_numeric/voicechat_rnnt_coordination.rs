//! One decision owner and failure agreement over the shared RNNT label loop.
use super::voicechat_rnnt::{assert_state, fixture, frame, limits};
use super::voicechat_rnnt_residency::{provider, same_state, take_acquisitions, Policy};
use super::*;
use eredu_architectures::voicechat::{
    rnnt::{RnntDimensions, RnntError, RnntExecution, RnntPredictionState, RnntState},
    rnnt_coordination,
    rnnt_layered::LayerwiseRnnt,
};
use eredu_core::BackendFailure;
use eredu_runtime::replicated_execution::ReplicatedCoordinator;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc::{self, Receiver, Sender},
};
use std::time::Duration;

enum Exchange {
    Agreement([u64; 8], bool),
    Publication(NumericTensor),
}
pub(super) struct Coordinator {
    owner: bool,
    send: Sender<Exchange>,
    receive: Receiver<Exchange>,
    timeouts: Arc<AtomicUsize>,
}
impl Coordinator {
    fn exchange(&self, local: Exchange) -> Result<Exchange, BackendFailure> {
        self.send.send(local).map_err(|error| {
            BackendFailure::from_error(std::io::Error::other(error.to_string()))
        })?;
        self.receive
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| {
                if matches!(error, mpsc::RecvTimeoutError::Timeout) {
                    self.timeouts.fetch_add(1, Ordering::SeqCst);
                }
                BackendFailure::from_error(std::io::Error::other(error.to_string()))
            })
    }
}
impl ReplicatedCoordinator<NumericTensor, NumericContext> for Coordinator {
    fn is_owner(&self) -> bool {
        self.owner
    }
    fn agree(
        &mut self,
        descriptor: [u64; 8],
        success: bool,
        _: &NumericContext,
    ) -> Result<bool, BackendFailure> {
        match self.exchange(Exchange::Agreement(descriptor, success))? {
            Exchange::Agreement(peer, peer_success) => {
                Ok(success && peer_success && descriptor == peer)
            }
            Exchange::Publication(_) => Err(BackendFailure::from_error(std::io::Error::other(
                "agreement/publication order differs",
            ))),
        }
    }
    fn publish(
        &mut self,
        value: NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, BackendFailure> {
        match self.exchange(Exchange::Publication(value.clone()))? {
            Exchange::Publication(peer) => Ok(if self.owner { value } else { peer }),
            Exchange::Agreement(_, _) => Err(BackendFailure::from_error(std::io::Error::other(
                "publication/agreement order differs",
            ))),
        }
    }
}
struct Model {
    inner: LayerwiseRnnt<NumericBackend, Policy>,
    owner: bool,
    selections: usize,
    malformed_prediction: bool,
}
impl RnntExecution<NumericBackend> for Model {
    fn dimensions(&self) -> RnntDimensions {
        self.inner.dimensions()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &NumericContext,
    ) -> Result<NumericTensor, RnntError> {
        self.inner.embed_token(token, context)
    }
    fn predictor_step(
        &mut self,
        input: &NumericTensor,
        previous: Option<&RnntPredictionState<NumericTensor>>,
        context: &NumericContext,
    ) -> Result<RnntPredictionState<NumericTensor>, RnntError> {
        self.inner.predictor_step(input, previous, context)
    }
    fn encode_frame(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, RnntError> {
        self.inner.encode_frame(input, context)
    }
    fn project_prediction(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, RnntError> {
        let value = self.inner.project_prediction(input, context)?;
        // The additive joint operation lies between wrapped primitives. An invalid
        // result forces an asymmetric tensor error at that shared policy boundary.
        Ok(if self.malformed_prediction {
            NumericTensor::zeros([1, self.dimensions().joint_width + 1])
        } else {
            value
        })
    }
    fn project_output(
        &mut self,
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, RnntError> {
        self.inner.project_output(input, context)
    }
    fn select_symbol(
        &mut self,
        logits: &NumericTensor,
        context: &NumericContext,
    ) -> Result<usize, RnntError> {
        self.selections += 1;
        if !self.owner {
            return Err(RnntError::InvalidState(
                "nonowner greedy selection is forbidden",
            ));
        }
        self.inner.select_symbol(logits, context)
    }
}

pub(super) fn coordinator_pair() -> ([Coordinator; 2], Arc<AtomicUsize>) {
    let (send0, receive1) = mpsc::channel();
    let (send1, receive0) = mpsc::channel();
    let timeouts = Arc::new(AtomicUsize::new(0));
    let coordinators = [
        Coordinator {
            owner: true,
            send: send0,
            receive: receive0,
            timeouts: timeouts.clone(),
        },
        Coordinator {
            owner: false,
            send: send1,
            receive: receive1,
            timeouts: timeouts.clone(),
        },
    ];
    (coordinators, timeouts)
}

#[test]
fn voicechat_rnnt_coordination_owner_decisions_failure_rollback_and_replay() {
    let fixture = fixture();
    let (coordinators, timeouts) = coordinator_pair();
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, mut coordinator)| {
                let fixture = &fixture;
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut inner = provider(fixture, false, &context);
                    let parent = inner
                        .decode_frame(
                            &frame(&fixture["streaming_frames"][0]),
                            &RnntState::default(),
                            limits(32),
                            &context,
                        )
                        .unwrap()
                        .state;
                    let saved = parent.clone();
                    take_acquisitions(&mut inner);
                    let mut model = Model {
                        inner,
                        owner: rank == 0,
                        selections: 0,
                        malformed_prediction: false,
                    };
                    let expected = &fixture["streaming_frames"][1];
                    let encoder = frame(expected);

                    // Admission disagreement cannot acquire a parameter or modify a parent.
                    assert!(rnnt_coordination::decode_frame::<NumericBackend, _, _>(
                        &mut model,
                        &mut coordinator,
                        [9, 2, 10 + rank as u64],
                        &encoder,
                        &parent,
                        limits(32),
                        &context
                    )
                    .is_err());
                    assert!(model.inner.runtime().policy().acquired.is_empty());
                    same_state(&parent, &saved);

                    // Rank one fails only after rank-local layer zero has produced a
                    // tentative hidden/cell pair, and after a coordinated token decision.
                    if rank == 1 {
                        model.inner.runtime_mut().policy_mut().fail_ordinal = Some(2);
                    }
                    assert!(rnnt_coordination::decode_frame::<NumericBackend, _, _>(
                        &mut model,
                        &mut coordinator,
                        [9, 2, 20],
                        &encoder,
                        &parent,
                        limits(32),
                        &context
                    )
                    .is_err());
                    same_state(&parent, &saved);
                    let policy = model.inner.runtime_mut().policy_mut();
                    assert!(policy.completed.contains(&1));
                    if rank == 1 {
                        assert!(policy.aborted > 0);
                    }
                    policy.fail_ordinal = None;
                    policy.acquired.clear();
                    policy.completed.clear();

                    // An unwrapped add failure must fence both participants without a
                    // second final agreement after the peer observes phase disagreement.
                    model.malformed_prediction = rank == 1;
                    assert!(rnnt_coordination::decode_frame::<NumericBackend, _, _>(
                        &mut model,
                        &mut coordinator,
                        [9, 2, 30],
                        &encoder,
                        &parent,
                        limits(32),
                        &context
                    )
                    .is_err());
                    same_state(&parent, &saved);
                    model.malformed_prediction = false;
                    take_acquisitions(&mut model.inner);

                    let mut replay = None;
                    for invocation in [40, 41] {
                        let output = rnnt_coordination::decode_frame::<NumericBackend, _, _>(
                            &mut model,
                            &mut coordinator,
                            [9, 2, invocation],
                            &encoder,
                            &parent,
                            limits(32),
                            &context,
                        )
                        .unwrap();
                        let emitted = expected["emitted"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|token| token.as_u64().unwrap() as usize)
                            .collect::<Vec<_>>();
                        assert_eq!(output.emitted, emitted);
                        assert_eq!(
                            output.first_blank,
                            expected["first_blank"].as_bool().unwrap()
                        );
                        assert_eq!(
                            output.ended_blank,
                            expected["ended_blank"].as_bool().unwrap()
                        );
                        assert_eq!(
                            output.hit_symbol_limit,
                            expected["hit_symbol_limit"].as_bool().unwrap()
                        );
                        assert_eq!(output.state.frames(), parent.frames() + 1);
                        assert_state(output.state.predictor().unwrap(), expected, fixture);
                        if let Some(previous) = &replay {
                            same_state(previous, &output.state);
                        }
                        replay = Some(output.state);
                        take_acquisitions(&mut model.inner);
                        same_state(&parent, &saved);
                    }
                    if rank == 0 {
                        assert!(model.selections > 0);
                    } else {
                        assert_eq!(model.selections, 0);
                    }
                    replay.unwrap()
                })
            })
            .collect::<Vec<_>>();
        let outputs = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        same_state(&outputs[0], &outputs[1]);
    });
    assert_eq!(
        timeouts.load(Ordering::SeqCst),
        0,
        "failure agreement must not strand a participant"
    );
}
