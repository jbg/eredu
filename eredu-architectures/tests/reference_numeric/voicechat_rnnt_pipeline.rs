//! Shared RNNT greedy control over genuinely transported stage-local recurrence.
use super::voicechat_mog_residency::weights;
use super::voicechat_rnnt::{dimensions, fixture, frame, limits};
use super::voicechat_rnnt_coordination::coordinator_pair;
use super::voicechat_rnnt_predictor::{compare_state, PipelinePredictor};
use super::voicechat_rnnt_residency::{same_state, Policy};
use super::*;
use eredu_architectures::voicechat::{
    rnnt::{RnntDimensions, RnntError, RnntProjectionExecution, RnntState},
    rnnt_pipeline::{self, RnntPipelineCursor},
    rnnt_projection_layered::{
        rnnt_projection_parameter_description, LayerwiseRnntProjection, RnntProjectionArchitecture,
    },
};
use eredu_core::DistributedCommitEpoch;
use eredu_runtime::derive_partitioned_local_layout;

struct Projections {
    inner: LayerwiseRnntProjection<NumericBackend, Policy>,
    owner: bool,
    selections: usize,
}
impl RnntProjectionExecution<NumericBackend> for Projections {
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
        self.inner.project_prediction(input, context)
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
        input: &NumericTensor,
        context: &NumericContext,
    ) -> Result<usize, RnntError> {
        assert!(self.owner, "nonowner must consume the coordinated decision");
        self.selections += 1;
        self.inner.select_symbol(input, context)
    }
}

// Generic RNNT snapshots permit floating precisions beyond the F32 wire policy.
// This fixture constructs such a valid ordinary parent without private state access.
struct Restored {
    dimensions: RnntDimensions,
    rank: usize,
    readout_only: bool,
}
impl eredu_architectures::voicechat::rnnt::RnntExecution<NumericBackend> for Restored {
    fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    fn predictor_ownership(&self) -> std::ops::Range<usize> {
        self.rank..self.rank + 1
    }
    fn embed_token(&mut self, _: usize, _: &NumericContext) -> Result<NumericTensor, RnntError> {
        Ok(NumericTensor::zeros([1, self.dimensions.predictor_width]))
    }
    fn predictor_step(
        &mut self,
        _: &NumericTensor,
        _: Option<&eredu_architectures::voicechat::rnnt::RnntPredictionState<NumericTensor>>,
        _: &NumericContext,
    ) -> Result<eredu_architectures::voicechat::rnnt::RnntPredictionState<NumericTensor>, RnntError>
    {
        use eredu_core::checkpoint::TensorDtype;
        use eredu_nn::recurrent::{LstmLayerState, LstmState};
        let full = NumericTensor::zeros([1, self.dimensions.predictor_width]);
        let local = if self.rank == 0 && !self.readout_only {
            full.clone().with_dtype(TensorDtype::F16)
        } else {
            full.clone()
        };
        eredu_architectures::voicechat::rnnt::RnntPredictionState::new(
            self.dimensions,
            self.rank..self.rank + 1,
            LstmState {
                layers: vec![LstmLayerState {
                    hidden: local.clone(),
                    cell: local,
                }],
            },
            (self.rank == 0).then(|| full.with_dtype(TensorDtype::F16)),
        )
    }
    fn encode_frame(
        &mut self,
        _: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, RnntError> {
        unreachable!()
    }
    fn project_prediction(
        &mut self,
        _: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, RnntError> {
        unreachable!()
    }
    fn project_output(
        &mut self,
        _: &NumericTensor,
        _: &NumericContext,
    ) -> Result<NumericTensor, RnntError> {
        unreachable!()
    }
}

#[test]
fn voicechat_rnnt_pipeline_greedy_preserves_local_parents_and_spends_explicit_epochs() {
    let fixture = fixture();
    let (coordinators, timeouts) = coordinator_pair();
    let world = Arc::new(NumericPartitionWorld::default());
    let barrier = Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, mut coordinator)| {
                let fixture = &fixture;
                let world = world.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut projections = Projections {
                        inner: LayerwiseRnntProjection::new(
                            RnntProjectionArchitecture::new(dimensions(fixture)).unwrap(),
                            Policy::bounded(weights(&fixture["parameters"])),
                        ),
                        owner: rank == 0,
                        selections: 0,
                    };
                    let mut predictor = PipelinePredictor::new(fixture, 1, rank, world.clone());
                    let first = &fixture["streaming_frames"][0];
                    let mut cursor =
                        RnntPipelineCursor::new(100, DistributedCommitEpoch::new(10).unwrap(), 4);
                    let empty = RnntState::default();
                    assert!(matches!(
                        rnnt_pipeline::decode_frame::<NumericBackend, _, _, _>(
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [3, 7, 0],
                            &mut cursor,
                            &frame(first),
                            &empty,
                            limits(32),
                            &context
                        ),
                        Err(RnntError::PipelineCapacity {
                            required: 5,
                            remaining: 4
                        })
                    ));
                    assert_eq!(cursor.next_step(), 100);
                    assert_eq!(cursor.next_epoch().value(), 10);
                    assert!(predictor.calls.is_empty());
                    assert!(projections.inner.runtime().policy().acquired.is_empty());
                    let mut cursor =
                        RnntPipelineCursor::new(100, DistributedCommitEpoch::new(10).unwrap(), 128);
                    assert!(rnnt_pipeline::decode_frame::<NumericBackend, _, _, _>(
                        &mut projections,
                        &mut predictor,
                        &mut coordinator,
                        [3, 7, 1 + rank as u64],
                        &mut cursor,
                        &frame(first),
                        &empty,
                        limits(32),
                        &context
                    )
                    .is_err());
                    assert_eq!(cursor.next_step(), 105);
                    assert_eq!(cursor.next_epoch().value(), 15);
                    assert_eq!(predictor.acquisitions.load(Ordering::Relaxed), 0);
                    assert!(projections.inner.runtime().policy().acquired.is_empty());
                    let initial = rnnt_pipeline::decode_frame::<NumericBackend, _, _, _>(
                        &mut projections,
                        &mut predictor,
                        &mut coordinator,
                        [3, 7, 3],
                        &mut cursor,
                        &frame(first),
                        &empty,
                        limits(32),
                        &context,
                    )
                    .unwrap();
                    assert!(initial.emitted.is_empty());
                    compare_state(initial.state.predictor().unwrap(), rank, first, fixture);
                    assert_eq!(predictor.calls, vec![(105, 15), (106, 16)]);
                    assert_eq!(cursor.next_step(), 110, "unused blank slots remain spent");
                    let mut committed = initial.state;
                    let saved = committed.clone();
                    let next = &fixture["streaming_frames"][1];
                    barrier.wait();
                    if rank == 0 {
                        world.arm_fault_one_per_agreement_group(
                            NumericPartitionFault::LocalExecution,
                        );
                    }
                    barrier.wait();
                    assert!(rnnt_pipeline::decode_frame::<NumericBackend, _, _, _>(
                        &mut projections,
                        &mut predictor,
                        &mut coordinator,
                        [3, 7, 4],
                        &mut cursor,
                        &frame(next),
                        &committed,
                        limits(32),
                        &context
                    )
                    .is_err());
                    barrier.wait();
                    same_state(&committed, &saved);
                    assert_eq!(cursor.next_step(), 113);
                    assert_eq!(cursor.next_epoch().value(), 23);
                    // A candidate with no exact committed outcome cannot enter the label loop.
                    predictor.reject_commit = rank == 1;
                    assert!(rnnt_pipeline::decode_frame::<NumericBackend, _, _, _>(
                        &mut projections,
                        &mut predictor,
                        &mut coordinator,
                        [3, 7, 5],
                        &mut cursor,
                        &frame(next),
                        &committed,
                        limits(32),
                        &context
                    )
                    .is_err());
                    predictor.reject_commit = false;
                    same_state(&committed, &saved);
                    assert_eq!(cursor.next_step(), 116);
                    let mut replay = None;
                    for identity in [6, 7] {
                        let result = rnnt_pipeline::decode_frame::<NumericBackend, _, _, _>(
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [3, 7, identity],
                            &mut cursor,
                            &frame(next),
                            &committed,
                            limits(32),
                            &context,
                        )
                        .unwrap();
                        let tokens = next["emitted"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .collect::<Vec<_>>();
                        assert_eq!(result.emitted, tokens);
                        compare_state(result.state.predictor().unwrap(), rank, next, fixture);
                        if let Some(previous) = &replay {
                            same_state(previous, &result.state);
                        }
                        replay = Some(result.state);
                        same_state(&committed, &saved);
                    }
                    committed = replay.unwrap();
                    for (index, expected) in fixture["streaming_frames"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .skip(2)
                    {
                        let result = rnnt_pipeline::decode_frame::<NumericBackend, _, _, _>(
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [3, 7, index as u64 + 10],
                            &mut cursor,
                            &frame(expected),
                            &committed,
                            limits(32),
                            &context,
                        )
                        .unwrap();
                        assert_eq!(
                            result.emitted,
                            expected["emitted"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_u64().unwrap() as usize)
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(
                            result.first_blank,
                            expected["first_blank"].as_bool().unwrap()
                        );
                        assert_eq!(
                            result.ended_blank,
                            expected["ended_blank"].as_bool().unwrap()
                        );
                        assert_eq!(
                            result.hit_symbol_limit,
                            expected["hit_symbol_limit"].as_bool().unwrap()
                        );
                        compare_state(result.state.predictor().unwrap(), rank, expected, fixture);
                        committed = result.state;
                    }
                    for readout_only in [false,true] {
                        let mut restored=Restored {dimensions:dimensions(fixture),rank,readout_only};
                        let foreign=eredu_architectures::voicechat::rnnt::RnntExecution::prime(&mut restored,&RnntState::default(),&context).unwrap();
                        let before=projections.inner.runtime().policy().acquired.len();
                        let recurrent_before=predictor.calls.len();
                        let step_before=cursor.next_step();
                        assert!(rnnt_pipeline::decode_frame::<NumericBackend,_,_,_>(&mut projections,&mut predictor,&mut coordinator,[3,7,90+u64::from(readout_only)],&mut cursor,&frame(first),&foreign,limits(32),&context).is_err());
                        assert_eq!(projections.inner.runtime().policy().acquired.len(),before,"non-F32 restored carry/readout rejects even on a blank frame before projection leases");
                        assert_eq!(predictor.calls.len(),recurrent_before);
                        assert_eq!(cursor.next_step(),step_before+3);
                    }
                    assert!(predictor
                        .calls
                        .windows(2)
                        .all(|pair| pair[0].0 < pair[1].0 && pair[0].1 < pair[1].1));
                    assert_eq!(projections.selections > 0, rank == 0);
                    assert!(
                        projections
                            .inner
                            .runtime()
                            .policy()
                            .acquired
                            .iter()
                            .all(|&ordinal| ordinal < 4),
                        "compact projection traversal cannot address a recurrent layer"
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(
        timeouts.load(Ordering::SeqCst),
        0,
        "preflight and neural failures must not strand a peer"
    );
}

#[test]
fn voicechat_rnnt_projection_tp_lease_failure_agrees_before_vocabulary_collectives() {
    use super::voicechat_rnnt::assert_values;
    use super::voicechat_rnnt_coordination::Coordinator;
    use eredu_architectures::voicechat::{
        rnnt_layered::RnntForward, rnnt_projection_layered::RnntProjectionInput,
    };
    use eredu_runtime::replicated_execution::ReplicatedCoordinator;
    use eredu_runtime::{LayeredLocalPhase, LayeredTraversalHook};
    struct Hook {
        coordinator: Coordinator,
        sequence: u64,
    }
    impl LayeredTraversalHook<NumericBackend, RnntForward<NumericTensor>, RnntError> for Hook {
        fn local_phase(
            &mut self,
            phase: LayeredLocalPhase,
            success: bool,
            context: &NumericContext,
        ) -> Result<(), RnntError> {
            let (phase, group, index) = match phase {
                LayeredLocalPhase::Initialization => (0, 0, 0),
                LayeredLocalPhase::PolicyBegin => (1, 0, 0),
                LayeredLocalPhase::GroupIngress { group } => (2, group, 0),
                LayeredLocalPhase::BeforeUnit { group, index } => (3, group, index),
                LayeredLocalPhase::Acquisition { group, index } => (4, group, index),
                LayeredLocalPhase::UnitExecution { group, index } => (5, group, index),
                LayeredLocalPhase::UnitCompletion { group, index } => (6, group, index),
                LayeredLocalPhase::GroupCompletion { group } => (7, group, 0),
                LayeredLocalPhase::Finish => (8, 0, 0),
            };
            let descriptor = [
                0x52544e4e,
                self.sequence,
                phase,
                group as u64,
                index as u64,
                0,
                0,
                0,
            ];
            self.sequence += 1;
            if self.coordinator.agree(descriptor, success, context)? {
                Ok(())
            } else {
                Err(RnntError::InvalidState(
                    "projection cohort failed local preparation",
                ))
            }
        }
    }
    let fixture = fixture();
    let dim = dimensions(&fixture);
    let description = rnnt_projection_parameter_description(dim).unwrap();
    let (coordinators, timeouts) = coordinator_pair();
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, coordinator)| {
                let fixture = &fixture;
                let description = &description;
                let group = group.clone();
                scope.spawn(move || {
                    let layout = derive_partitioned_local_layout(
                        description,
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap(),
                    )
                    .unwrap();
                    let context = NumericContext::with_local_layout(layout.clone());
                    let parallel = NumericParallelContext::new(rank, group);
                    let parameters = weights(&fixture["parameters"])
                        .into_iter()
                        .filter_map(|(name, value)| {
                            let placement = layout.tensor(&name)?.placement();
                            Some((name, select_parameter(&value, placement).unwrap()))
                        })
                        .collect();
                    let mut model = LayerwiseRnntProjection::new(
                        RnntProjectionArchitecture::new_parallel(dim, &layout).unwrap(),
                        Policy::bounded(parameters),
                    );
                    let mut hook = Hook {
                        coordinator,
                        sequence: 0,
                    };
                    let reference = &fixture["streaming_frames"][0]["joint_calls"][0];
                    let hidden = NumericTensor::new(
                        [1, dim.joint_width],
                        reference["hidden_sum"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| (v.as_f64().unwrap() as f32).max(0.))
                            .collect(),
                    );
                    let saved = hidden.clone();
                    for ordinal in [0, 3] {
                        model.runtime_mut().policy_mut().fail_ordinal =
                            if rank == 0 { Some(ordinal) } else { None };
                        let before = parallel.trace().len();
                        let input = || {
                            if ordinal == 0 {
                                RnntProjectionInput::Embed(2)
                            } else {
                                RnntProjectionInput::Output(&hidden)
                            }
                        };
                        assert!(model
                            .invoke_parallel_with_hook(input(), &parallel, &context, &mut hook)
                            .is_err());
                        assert_eq!(
                            parallel.trace().len(),
                            before,
                            "peer must not enter a vocabulary collective after failed acquisition"
                        );
                        assert!(model.runtime().policy().aborted > 0);
                        assert_eq!(hidden.data, saved.data);
                        model.runtime_mut().policy_mut().fail_ordinal = None;
                        let output = model
                            .invoke_parallel_with_hook(input(), &parallel, &context, &mut hook)
                            .unwrap();
                        let expected = if ordinal == 0 {
                            serde_json::Value::Array(
                                fixture["parameters"]
                                    ["stt_model.rnnt_decoder.prediction.embed.weight"]["values"]
                                    .as_array()
                                    .unwrap()[4..6]
                                    .to_vec(),
                            )
                        } else {
                            reference["logits"].clone()
                        };
                        assert_values(&output.data, &expected, fixture);
                        assert!(parallel.trace().len() > before);
                        assert_eq!(hidden.data, saved.data);
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
