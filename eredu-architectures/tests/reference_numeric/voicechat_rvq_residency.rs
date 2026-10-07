//! Pinned feedback and quantization through the shared bank/projection providers.
use super::voicechat_mog_residency::{dimensions, weights};
use super::voicechat_tts_generator::{codes, compare, compare_cache, model, tensor};
use super::*;
use eredu_architectures::voicechat::{
    tts_mog::MogHead,
    tts_rvq::{RefinementPolicy, RvqDimensions, RvqRefinement},
    tts_rvq_layered::{LayerwiseRvq, RvqArchitecture, RvqExecution, RvqUnit},
};
use eredu_core::scheduler::SemanticStateTransaction;
use serde_json::Value;

type Unit = RvqUnit<NumericBackend>;
pub(super) struct BoundFixturePolicy {
    inner: RebuildingUnitPolicy,
    parameters: BTreeMap<String, NumericTensor>,
    bank_acquisitions: usize,
    fail_bank_acquisition: Option<usize>,
    fail_projection: bool,
    aborted: usize,
}
impl LayerwisePolicy<NumericBackend, Unit> for BoundFixturePolicy {
    type Lease = RebuiltUnitLease<Unit>;
    type Error = Error;
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        _: &NumericContext,
    ) {
        self.aborted += 1;
        drop(active);
    }
    fn begin(&mut self, input: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::begin(
            &mut self.inner,
            input,
            context,
        )
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<Unit, E>,
    {
        if ordinal == 0 {
            self.bank_acquisitions += 1;
        }
        self.inner.fail_acquire = ((ordinal == 0
            && self.fail_bank_acquisition == Some(self.bank_acquisitions))
            || (ordinal == 1 && self.fail_projection))
            .then_some(ordinal);
        let parameters = &self.parameters;
        self.inner.acquire(
            ordinal,
            address,
            |context| {
                let mut unit = build(context)?;
                replace_numeric_parameters(&mut unit, parameters);
                Ok(unit)
            },
            context,
        )
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a NumericTensor,
        state: S,
        retained: C,
        context: &NumericContext,
    ) -> Result<(), Error>
    where
        NumericTensor: 'a,
        S: Iterator<Item = &'a NumericTensor>,
        C: Iterator<Item = &'a NumericTensor>,
    {
        self.inner
            .complete(ordinal, address, lease, output, state, retained, context)
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::finish(
            &mut self.inner,
            output,
            context,
        )
    }
}
fn geometry() -> RvqDimensions {
    RvqDimensions {
        cardinality: 5,
        latent: 4,
        hidden: 4,
    }
}
pub(super) fn bounded(
    parameters: &Value,
    d: RvqDimensions,
) -> LayerwiseRvq<NumericBackend, BoundFixturePolicy> {
    LayerwiseRvq::new(
        RvqArchitecture::new(d).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters: weights(parameters),
            bank_acquisitions: 0,
            fail_bank_acquisition: None,
            fail_projection: false,
            aborted: 0,
        },
    )
}
fn resident(
    parameters: &Value,
    context: &NumericContext,
) -> LayerwiseRvq<NumericBackend, NumericReplicatedPolicy<Unit>> {
    let architecture = RvqArchitecture::new(geometry()).unwrap();
    let parameters = weights(parameters);
    let mut units = Vec::new();
    for group in 0..2 {
        let mut unit = architecture.build_unit(group, 0, context).unwrap();
        replace_numeric_parameters(&mut unit, &parameters);
        units.push(unit);
    }
    LayerwiseRvq::new(architecture, NumericReplicatedPolicy::resident(units))
}
fn events(project: bool) -> Vec<StreamPolicyEvent> {
    let mut events = vec![
        StreamPolicyEvent::Begin,
        StreamPolicyEvent::Acquire(0, 0, 0),
        StreamPolicyEvent::Complete(0, 0, 0),
    ];
    if project {
        events.extend([
            StreamPolicyEvent::Acquire(1, 1, 0),
            StreamPolicyEvent::Complete(1, 1, 0),
        ]);
    }
    events.push(StreamPolicyEvent::Finish);
    events
}

#[test]
fn voicechat_rvq_resident_and_bounded_feedback_and_quantization_match_nvidia_without_unused_projection()
 {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mut ordinary = RvqRefinement::<NumericBackend>::new(5, 4, 4, &context).unwrap();
    replace_numeric_parameters(&mut ordinary, &weights(&fixture["parameters"]));
    let mut resident = resident(&fixture["parameters"], &context);
    let mut bounded = bounded(&fixture["parameters"], geometry());
    let case = &fixture["released_policy"];
    let hidden = tensor(&case["hidden"]);
    let conditional = hidden.axis_slice(0, 0, 1);
    let unconditional = hidden.axis_slice(0, 1, 2);
    let policy = RefinementPolicy {
        guidance_scale: 0.2,
        noise_scale: 0.001,
        top_p: 0.95,
    };
    let mut state = ordinary.start(1, 1, &context).unwrap();
    let initial = state.clone();
    assert!(bounded.runtime().policy().inner.events.is_empty());
    for step in case["steps"].as_array().unwrap() {
        let reference = ordinary.embed_codes(state.codes(), &context).unwrap();
        let resident_embedding = resident.embed_codes(state.codes(), &context).unwrap();
        let bounded_embedding = bounded.embed_codes(state.codes(), &context).unwrap();
        assert_eq!(reference.data, resident_embedding.data);
        assert_eq!(reference.data, bounded_embedding.data);
        assert_eq!(bounded.runtime().policy().inner.events, events(true));
        bounded.runtime_mut().policy_mut().inner.events.clear();
        // The independently generated input includes the learned feedback
        // projection and both guidance lanes, anchoring embedding parity.
        compare(
            &bounded
                .input(&state, &conditional, Some(&unconditional), policy, &context)
                .unwrap(),
            &step["input"],
            &fixture,
        );
        assert_eq!(bounded.runtime().policy().inner.events, events(true));
        bounded.runtime_mut().policy_mut().inner.events.clear();
        let latent = tensor(&step["latent"]);
        let direct = ordinary.advance(&state, &latent, &context).unwrap();
        let leased = resident.advance(&state, &latent, &context).unwrap();
        let candidate = bounded.advance(&state, &latent, &context).unwrap();
        assert_eq!(candidate.codes().data, tensor(&step["codes_after"]).data);
        assert_eq!(direct.codes().data, candidate.codes().data);
        assert_eq!(leased.codes().data, candidate.codes().data);
        assert_eq!(state.codes().data, tensor(&step["codes_before"]).data);
        assert_eq!(bounded.runtime().policy().inner.events, events(false));
        bounded.runtime_mut().policy_mut().inner.events.clear();
        for sum in [
            ordinary.depthsum(candidate.codes(), &context).unwrap(),
            resident.depthsum(candidate.codes(), &context).unwrap(),
            bounded.depthsum(candidate.codes(), &context).unwrap(),
        ] {
            compare(&sum, &step["summed_feedback"], &fixture);
        }
        assert_eq!(bounded.runtime().policy().inner.events, events(false));
        bounded.runtime_mut().policy_mut().inner.events.clear();
        state = candidate;
    }
    assert_eq!(state.completed_depth(), 31);
    assert_eq!(state.next_slot(), None);
    assert_eq!(initial.codes().data, vec![5.; 31]);
    let mut head =
        MogHead::<NumericBackend>::new(dimensions(&fixture["dimensions"]), &context).unwrap();
    replace_numeric_parameters(&mut head, &weights(&fixture["parameters"]));
    let steps = case["steps"].as_array().unwrap();
    let mut draws = 0;
    let generated = bounded
        .generate(
            &conditional,
            Some(&unconditional),
            policy,
            &mut head,
            |slot, logits: &NumericTensor, _| {
                let step = &steps[draws];
                draws += 1;
                assert_eq!(slot, step["slot"].as_u64().unwrap() as usize);
                compare(logits, &step["logits"], &fixture);
                let mixture = codes(&step["mixture_indices"], &context)
                    .reshape(&logits.shape()[..2], &context)
                    .unwrap();
                Ok((mixture, tensor(&step["gaussian_draw"])))
            },
            &context,
        )
        .unwrap();
    assert_eq!(
        draws, 5,
        "the first three empty slots acquire no parameters or draws"
    );
    assert_eq!(generated.codes().data, tensor(&case["codes"]).data);
    assert_eq!(generated.codes().data, state.codes().data);
    assert_eq!(
        bounded.runtime().policy().inner.events,
        [events(true), events(false)].concat().repeat(5)
    );
}

#[test]
fn voicechat_rvq_quantization_acquisition_failure_after_draws_preserves_frame_rng_and_retry() {
    use super::voicechat_speech_sampling::{Recorded, Tape, generation};
    use super::voicechat_transformer::CausalCache;
    use eredu_architectures::voicechat::{
        tts_generator::{TtsGenerator, TtsGeneratorParts},
        tts_ingress::{PromptLayout, SpeakerPrompt},
    };
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let ordinary = model(&fixture, &context);
    let config = ordinary.config().clone();
    let parts = ordinary.into_parts();
    let mut generator = TtsGenerator::from_parts(
        &config,
        16,
        5,
        TtsGeneratorParts::new(
            parts.conditioning,
            parts.character_transformer,
            parts.transformer,
            bounded(
                &fixture["parameters"],
                RvqDimensions {
                    cardinality: config.codebook_size as i32,
                    latent: config.latent_size as i32,
                    hidden: config.backbone.hidden_size as i32,
                },
            ),
            parts.mog,
        ),
    )
    .unwrap();
    let prompt = &fixture["calls"][0];
    let call = &fixture["calls"][1];
    let mask = prompt["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect::<Vec<_>>();
    let mut caches = [
        CausalCache(NumericCache::new(Some(3))),
        CausalCache(NumericCache::new(None)),
    ];
    generator
        .prefill(
            &codes(&prompt["input_codes"], &context),
            &PromptLayout::new(1, 5, &mask).unwrap(),
            SpeakerPrompt::Aria,
            Some(&tensor(&prompt["supplied_subwords"])),
            &mut caches,
            &context,
        )
        .unwrap();
    let frame = generator
        .begin_frame(
            &codes(&call["input_codes"], &context),
            Some(&tensor(&call["supplied_subwords"])),
            &mut caches,
            &context,
        )
        .unwrap();
    let tape = Tape::new(call);
    let canonical = generation(Some(tape.clone()));
    let mut branch = canonical.branch().unwrap();
    let policy = generator.refinement.runtime_mut().policy_mut();
    policy.inner.events.clear();
    policy.fail_bank_acquisition = Some(policy.bank_acquisitions + 2);
    let error = generator
        .advance_sampled_frame::<Recorded, _, _, _, _, _>(&frame, &mut branch, None, &context)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected dense-stream acquisition failure")
    );
    assert_eq!(
        tape.issued.get(),
        2,
        "quantization-bank refusal follows both recorded draws"
    );
    assert_eq!(branch.random_state().unwrap().cursor, 0);
    assert_eq!(canonical.random_state().unwrap().cursor, 0);
    assert_eq!(frame.refinement().next_slot(), Some(3));
    assert_eq!(frame.refinement().codes().data, vec![5.; 31]);
    compare_cache(&caches, &call["cache"], &fixture);
    assert_eq!(
        generator.refinement.runtime().policy().inner.events,
        [events(true), events(false)[..2].to_vec()].concat()
    );
    generator
        .refinement
        .runtime_mut()
        .policy_mut()
        .fail_bank_acquisition = None;
    let mut controlled = frame.clone();
    for expected in call["refinement"].as_array().unwrap() {
        controlled = generator
            .advance_sampled_frame::<Recorded, _, _, _, _, _>(
                &controlled,
                &mut branch,
                None,
                &context,
            )
            .unwrap();
        assert_eq!(
            controlled.refinement().codes().data,
            codes(&expected["codes"], &context).data
        );
    }
    let mut uninterrupted = canonical.branch().unwrap();
    let complete = generator
        .generate_sampled_frame::<Recorded, _, _, _, _, _>(
            &frame,
            &mut uninterrupted,
            None,
            &context,
        )
        .unwrap();
    assert_eq!(
        complete.codes().unwrap().data,
        codes(&call["codes"], &context).data
    );
    assert_eq!(
        complete.codes().unwrap().data,
        controlled.codes().unwrap().data
    );
    assert_eq!(branch.random_state().unwrap().cursor, 10);
    assert_eq!(uninterrupted.random_state().unwrap().cursor, 10);
    assert_eq!(canonical.random_state().unwrap().cursor, 0);
    compare_cache(&caches, &call["cache"], &fixture);
}

fn parallel_models(
    dimensions: RvqDimensions,
    parameters: BTreeMap<String, NumericTensor>,
    layout: &LocalModelLayout,
    context: &NumericContext,
) -> (
    LayerwiseRvq<NumericBackend, NumericReplicatedPolicy<Unit>>,
    LayerwiseRvq<NumericBackend, BoundFixturePolicy>,
) {
    let parameters: BTreeMap<_, _> = parameters
        .into_iter()
        .filter_map(|(name, value)| {
            layout.tensor(&name).map(|tensor| {
                let selected = select_parameter(&value, tensor.placement()).unwrap();
                (name, selected)
            })
        })
        .collect();
    let architecture = || RvqArchitecture::new_parallel(dimensions, layout).unwrap();
    let units = (0..2)
        .map(|group| {
            let mut unit = architecture().build_unit(group, 0, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            unit
        })
        .collect();
    (
        LayerwiseRvq::new(architecture(), NumericReplicatedPolicy::resident(units)),
        LayerwiseRvq::new(
            architecture(),
            BoundFixturePolicy {
                inner: RebuildingUnitPolicy::default(),
                parameters,
                bank_acquisitions: 0,
                fail_bank_acquisition: None,
                fail_projection: false,
                aborted: 0,
            },
        ),
    )
}

#[test]
fn voicechat_rvq_parallel_tp2_matches_nvidia_prefixes_feedback_and_bank_retry() {
    use eredu_architectures::voicechat::{
        tts_mog::TtsError, tts_rvq_layered::rvq_parameter_description,
    };
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let description = rvq_parameter_description(geometry()).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let world = world.clone();
                let group = group.clone();
                let fixture = &fixture;
                let description = &description;
                scope.spawn(move || {
                    let topology =
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap();
                    let layout =
                        eredu_runtime::derive_partitioned_local_layout(description, topology)
                            .unwrap();
                    let context =
                        NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let parallel =
                        NumericParallelContext::new_partitioned(rank, rank, group, world);
                    let (mut resident, mut bounded) = parallel_models(
                        geometry(),
                        weights(&fixture["parameters"]),
                        &layout,
                        &context,
                    );
                    let case = &fixture["released_policy"];
                    let first = &case["steps"][0];
                    let mut state = bounded.start(1, 1, &context).unwrap();
                    let initial = state.clone();
                    let latent = tensor(&first["latent"]);
                    for wrong in [
                        NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                        NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                    ] {
                        assert!(matches!(
                            bounded.depthsum_parallel(state.codes(), &wrong, &context),
                            Err(TtsError::Configuration(_))
                        ));
                        assert!(matches!(
                            bounded.embed_codes_parallel(state.codes(), &wrong, &context),
                            Err(TtsError::Configuration(_))
                        ));
                        assert!(matches!(
                            bounded.advance_parallel(&state, &latent, &wrong, &context),
                            Err(TtsError::Configuration(_))
                        ));
                    }
                    assert!(matches!(
                        bounded.depthsum(state.codes(), &context),
                        Err(TtsError::Configuration(_))
                    ));
                    assert!(matches!(
                        bounded.embed_codes(state.codes(), &context),
                        Err(TtsError::Configuration(_))
                    ));
                    assert!(matches!(
                        bounded.advance(&state, &latent, &context),
                        Err(TtsError::Configuration(_))
                    ));
                    for malformed in [
                        NumericTensor::zeros([1, 1, 30])
                            .with_dtype(eredu_core::checkpoint::TensorDtype::I32),
                        NumericTensor::zeros([1, 1, 31]),
                    ] {
                        assert!(matches!(
                            bounded.depthsum_parallel(&malformed, &parallel, &context),
                            Err(TtsError::Shape(_))
                        ));
                        assert!(matches!(
                            bounded.embed_codes_parallel(&malformed, &parallel, &context),
                            Err(TtsError::Shape(_))
                        ));
                    }
                    for malformed in [
                        NumericTensor::zeros([1, 1, 3]),
                        NumericTensor::zeros([1, 1, 4])
                            .with_dtype(eredu_core::checkpoint::TensorDtype::I32),
                    ] {
                        assert!(matches!(
                            bounded.advance_parallel(&state, &malformed, &parallel, &context),
                            Err(TtsError::Shape(_))
                        ));
                    }
                    assert!(bounded.runtime().policy().inner.events.is_empty());
                    // Both ranks refuse their bank before collective execution.
                    // The caller retains the same parent and explicit latent on retry.
                    bounded.runtime_mut().policy_mut().fail_bank_acquisition = Some(1);
                    assert!(
                        bounded
                            .advance_parallel(&state, &latent, &parallel, &context)
                            .is_err()
                    );
                    assert_eq!(bounded.runtime().policy().inner.events, events(false)[..2]);
                    assert_eq!(state.codes().data, vec![5.; 31]);
                    assert_eq!(state.next_slot(), Some(3));
                    bounded.runtime_mut().policy_mut().fail_bank_acquisition = None;
                    bounded.runtime_mut().policy_mut().inner.events.clear();
                    assert_eq!(
                        bounded
                            .depthsum_parallel(state.codes(), &parallel, &context)
                            .unwrap()
                            .data,
                        vec![0.; 4]
                    );
                    assert_eq!(
                        bounded
                            .embed_codes_parallel(state.codes(), &parallel, &context)
                            .unwrap()
                            .data,
                        vec![0.; 4]
                    );
                    let hidden = tensor(&case["hidden"]);
                    let conditional = hidden.axis_slice(0, 0, 1);
                    let unconditional = hidden.axis_slice(0, 1, 2);
                    for step in case["steps"].as_array().unwrap() {
                        assert_eq!(
                            state.next_slot(),
                            Some(step["slot"].as_u64().unwrap() as usize)
                        );
                        bounded.runtime_mut().policy_mut().inner.events.clear();
                        let embedding = bounded
                            .embed_codes_parallel(state.codes(), &parallel, &context)
                            .unwrap();
                        let resident_embedding = resident
                            .embed_codes_parallel(state.codes(), &parallel, &context)
                            .unwrap();
                        assert_eq!(embedding.data, resident_embedding.data);
                        let input = NumericTensor::concatenate(
                            &[
                                embedding.add(&conditional, &context).unwrap(),
                                embedding.add(&unconditional, &context).unwrap(),
                            ],
                            0,
                            &context,
                        )
                        .unwrap();
                        compare(&input, &step["input"], fixture);
                        assert_eq!(bounded.runtime().policy().inner.events, events(true));
                        bounded.runtime_mut().policy_mut().inner.events.clear();
                        let latent = tensor(&step["latent"]);
                        let candidate = bounded
                            .advance_parallel(&state, &latent, &parallel, &context)
                            .unwrap();
                        let resident_candidate = resident
                            .advance_parallel(&state, &latent, &parallel, &context)
                            .unwrap();
                        assert_eq!(candidate.codes().data, tensor(&step["codes_after"]).data);
                        assert_eq!(candidate.codes().data, resident_candidate.codes().data);
                        assert_eq!(state.codes().data, tensor(&step["codes_before"]).data);
                        assert_eq!(bounded.runtime().policy().inner.events, events(false));
                        bounded.runtime_mut().policy_mut().inner.events.clear();
                        for sum in [
                            bounded
                                .depthsum_parallel(candidate.codes(), &parallel, &context)
                                .unwrap(),
                            resident
                                .depthsum_parallel(candidate.codes(), &parallel, &context)
                                .unwrap(),
                        ] {
                            compare(&sum, &step["summed_feedback"], fixture);
                        }
                        assert_eq!(bounded.runtime().policy().inner.events, events(false));
                        state = candidate;
                    }
                    assert_eq!(state.completed_depth(), 31);
                    assert_eq!(state.next_slot(), None);
                    assert_eq!(state.codes().data, tensor(&case["codes"]).data);
                    assert_eq!(initial.codes().data, vec![5.; 31]);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert!(world.tensor_collective_calls() > 0);
}

#[test]
fn voicechat_rvq_parallel_uneven_latent_reduces_distances_before_argmin_and_preserves_first_tie() {
    use eredu_architectures::voicechat::tts_rvq_layered::rvq_parameter_description;
    let dimensions = RvqDimensions {
        cardinality: 3,
        latent: 5,
        hidden: 2,
    };
    let description = rvq_parameter_description(dimensions).unwrap();
    // With zero latent, rank 0 prefers code 0 and rank 1 prefers code 1.
    // Full distances are [16, 9, 8], so only reduction before argmin selects 2.
    let mut bank = vec![0., 0., 0., 4., 0., 3., 0., 0., 0., 0., 2., 0., 0., 2., 0.].repeat(31);
    // The next slot starts from a fresh zero latent, without subtracting the
    // previous prefix. Its first depth ties at distance 2 and must select 0.
    bank[15..30].copy_from_slice(&[-1., 0., 0., 1., 0., 1., 0., 0., -1., 0., 2., 0., 0., 2., 0.]);
    let parameters = BTreeMap::from([
        (
            "tts_model.tts_model.rvq_embs".into(),
            NumericTensor::new([31, 3, 5], bank),
        ),
        (
            "tts_model.tts_model.embed_code.weight".into(),
            NumericTensor::new([2, 5], vec![1., 2., 3., 4., 5., -1., -2., -3., -4., -5.]),
        ),
    ]);
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let world = world.clone();
                let group = group.clone();
                let description = &description;
                let parameters = parameters.clone();
                scope.spawn(move || {
                    let topology =
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap();
                    let layout =
                        eredu_runtime::derive_partitioned_local_layout(description, topology)
                            .unwrap();
                    let context =
                        NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let parallel =
                        NumericParallelContext::new_partitioned(rank, rank, group, world);
                    let (_, mut bounded) =
                        parallel_models(dimensions, parameters, &layout, &context);
                    let initial = bounded.start(1, 1, &context).unwrap();
                    let zero = NumericTensor::zeros([1, 1, 5]);
                    let first = bounded
                        .advance_parallel(&initial, &zero, &parallel, &context)
                        .unwrap();
                    assert_eq!(first.codes().data, [vec![2.], vec![3.; 30]].concat());
                    let sum = bounded
                        .depthsum_parallel(first.codes(), &parallel, &context)
                        .unwrap();
                    assert_eq!(sum.shape(), &[1, 1, 5]);
                    assert_eq!(sum.data, vec![2., 0., 0., 2., 0.]);
                    assert_eq!(
                        bounded
                            .embed_codes_parallel(first.codes(), &parallel, &context)
                            .unwrap()
                            .data,
                        vec![10., -10.]
                    );
                    let next = bounded
                        .advance_parallel(&first, &zero, &parallel, &context)
                        .unwrap();
                    assert_eq!(&next.codes().data[..2], &[2., 0.]);
                    assert_eq!(&first.codes().data[..2], &[2., 3.]);
                    assert_eq!(initial.codes().data, vec![3.; 31]);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert!(world.tensor_collective_calls() > 0);
}

mod guarded {
    use super::super::voicechat_rnnt_coordination::{Coordinator, coordinator_pair};
    use super::*;
    use eredu_architectures::voicechat::{
        tts_mog::TtsError,
        tts_rvq_layered::{RvqForward, RvqInput, rvq_parameter_description},
    };
    use eredu_runtime::{
        LayeredLocalPhase, LayeredTraversalHook, replicated_execution::ReplicatedCoordinator,
    };
    use std::{cell::RefCell, rc::Rc, sync::atomic::Ordering};

    struct Hook(Rc<RefCell<Coordinator>>);
    impl LayeredTraversalHook<NumericBackend, RvqForward<NumericTensor>, TtsError> for Hook {
        fn local_phase(
            &mut self,
            phase: LayeredLocalPhase,
            success: bool,
            context: &NumericContext,
        ) -> Result<(), TtsError> {
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
            if !self.0.borrow_mut().agree(
                [
                    0x525651484f4f4b30,
                    phase,
                    group as u64,
                    index as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                success,
                context,
            )? {
                return Err(TtsError::Configuration(
                    "RVQ participant failed traversal phase",
                ));
            }
            Ok(())
        }
    }
    #[test]
    fn voicechat_rvq_tp_preparation_failures_agree_and_retry() {
        let (coordinators, timeouts) = coordinator_pair();
        let group = NumericParallelGroup::new(2);
        std::thread::scope(|scope| {
            let workers = coordinators
                .into_iter()
                .enumerate()
                .map(|(rank, coordinator)| {
                    let group = group.clone();
                    scope.spawn(move || worker(rank, coordinator, group))
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        assert_eq!(timeouts.load(Ordering::SeqCst), 0);
    }
    fn worker(rank: usize, coordinator: Coordinator, group: Arc<NumericParallelGroup>) {
        let fixture: Value =
            serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
        let description = rvq_parameter_description(geometry()).unwrap();
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank).unwrap();
        let layout =
            eredu_runtime::derive_partitioned_local_layout(&description, topology).unwrap();
        let context = NumericContext::with_local_layout(layout.clone());
        let parallel = NumericParallelContext::new(rank, group);
        let (_, mut model) = parallel_models(
            geometry(),
            weights(&fixture["parameters"]),
            &layout,
            &context,
        );
        let coordinator = Rc::new(RefCell::new(coordinator));
        let mut hook = Hook(coordinator.clone());
        let agree = |descriptor, success, context: &NumericContext| {
            coordinator
                .borrow_mut()
                .agree(descriptor, success, context)
                .map_err(TtsError::from)
        };
        let mut state = model.start(1, 1, &context).unwrap();
        let initial = state.codes().clone();
        let mismatch = if rank == 0 {
            RvqInput::Sum(state.codes())
        } else {
            RvqInput::Embed(state.codes())
        };
        assert!(
            model
                .invoke_parallel_guarded(mismatch, &parallel, &context, &mut hook, agree)
                .is_err()
        );
        assert!(model.runtime().policy().inner.events.is_empty());
        assert!(parallel.trace().is_empty());
        // The bank has completed before a projection acquisition fails on one rank.
        model.runtime_mut().policy_mut().fail_projection = rank == 0;
        assert!(
            model
                .invoke_parallel_guarded(
                    RvqInput::Embed(state.codes()),
                    &parallel,
                    &context,
                    &mut hook,
                    agree
                )
                .is_err()
        );
        assert_eq!(model.runtime().policy().aborted, 1);
        assert!(
            model
                .runtime()
                .policy()
                .inner
                .events
                .contains(&StreamPolicyEvent::Complete(0, 0, 0))
        );
        assert!(parallel.trace().is_empty());
        assert_eq!(state.codes().data, initial.data);
        model.runtime_mut().policy_mut().fail_projection = false;
        // F32 precision is checked locally before any projection reduction.
        let projection = "tts_model.tts_model.embed_code.weight";
        let saved = model.runtime().policy().parameters[projection].clone();
        if rank == 0 {
            model.runtime_mut().policy_mut().parameters.insert(
                projection.into(),
                saved
                    .clone()
                    .with_dtype(eredu_core::checkpoint::TensorDtype::F16),
            );
        }
        assert!(
            model
                .invoke_parallel_guarded(
                    RvqInput::Embed(state.codes()),
                    &parallel,
                    &context,
                    &mut hook,
                    agree
                )
                .is_err()
        );
        assert!(parallel.trace().is_empty());
        model
            .runtime_mut()
            .policy_mut()
            .parameters
            .insert(projection.into(), saved);
        for step in fixture["released_policy"]["steps"].as_array().unwrap() {
            let latent = tensor(&step["latent"]);
            if state.next_slot() == Some(7) {
                // The final slot contains 22 ordered score reductions. A malformed
                // bound bank must reject before the first one on every rank.
                let bank = "tts_model.tts_model.rvq_embs";
                let saved = model.runtime().policy().parameters[bank].clone();
                if rank == 0 {
                    model
                        .runtime_mut()
                        .policy_mut()
                        .parameters
                        .insert(bank.into(), NumericTensor::zeros([31, 5, 3]));
                }
                let before = parallel.trace().len();
                assert!(
                    model
                        .invoke_parallel_guarded(
                            RvqInput::Advance {
                                state: &state,
                                latent: &latent
                            },
                            &parallel,
                            &context,
                            &mut hook,
                            agree
                        )
                        .is_err()
                );
                assert_eq!(parallel.trace().len(), before);
                assert_eq!(state.codes().data, tensor(&step["codes_before"]).data);
                model
                    .runtime_mut()
                    .policy_mut()
                    .parameters
                    .insert(bank.into(), saved);
            }
            let (_, candidate) = model
                .invoke_parallel_guarded(
                    RvqInput::Advance {
                        state: &state,
                        latent: &latent,
                    },
                    &parallel,
                    &context,
                    &mut hook,
                    agree,
                )
                .unwrap();
            let candidate = candidate.into_state().unwrap();
            assert_eq!(candidate.codes().data, tensor(&step["codes_after"]).data);
            assert_eq!(state.codes().data, tensor(&step["codes_before"]).data);
            let (sum, _) = model
                .invoke_parallel_guarded(
                    RvqInput::Sum(candidate.codes()),
                    &parallel,
                    &context,
                    &mut hook,
                    agree,
                )
                .unwrap();
            compare(&sum, &step["summed_feedback"], &fixture);
            let (embedded, _) = model
                .invoke_parallel_guarded(
                    RvqInput::Embed(state.codes()),
                    &parallel,
                    &context,
                    &mut hook,
                    agree,
                )
                .unwrap();
            let hidden = tensor(&fixture["released_policy"]["hidden"]).axis_slice(0, 0, 1);
            compare(
                &embedded.add(&hidden, &context).unwrap(),
                &serde_json::json!({"shape":[1,1,4],"values":tensor(&step["input"]).axis_slice(0,0,1).data}),
                &fixture,
            );
            state = candidate;
        }
        assert_eq!(state.completed_depth(), 31);
    }
}
