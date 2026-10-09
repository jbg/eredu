//! Independent NVIDIA MoG/RVQ equations through ordinary bounded unit providers.
use super::voicechat_tts_generator::{codes, compare, compare_cache, model, tensor};
use super::*;
use eredu_architectures::voicechat::{
    tts_mog::{MogDimensions, MogHead},
    tts_mog_layered::{LayerwiseMog, MogArchitecture, MogExecution, MogUnit},
    tts_rvq::{RefinementPolicy, RvqRefinement},
};
use eredu_core::scheduler::SemanticStateTransaction;
use serde_json::Value;

type Unit = MogUnit<NumericBackend>;
pub(super) struct BoundFixturePolicy {
    inner: RebuildingUnitPolicy,
    parameters: BTreeMap<String, NumericTensor>,
}
impl LayerwisePolicy<NumericBackend, Unit> for BoundFixturePolicy {
    type Lease = RebuiltUnitLease<Unit>;
    type Error = Error;
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

pub(super) fn weights(parameters: &Value) -> BTreeMap<String, NumericTensor> {
    parameters
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| (name.clone(), tensor(value)))
        .collect()
}
pub(super) fn dimensions(value: &Value) -> MogDimensions {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    MogDimensions {
        hidden: i("hidden"),
        intermediate: i("intermediate"),
        latent: i("latent"),
        mixtures: i("mixtures"),
        rank: i("low_rank"),
        layers: i("mlp_layers") as usize,
        epsilon: value["eps"].as_f64().unwrap() as f32,
        min_log_std: value["min_log_std"].as_f64().unwrap() as f32,
    }
}
pub(super) fn bounded(
    d: MogDimensions,
    parameters: &Value,
) -> LayerwiseMog<NumericBackend, BoundFixturePolicy> {
    LayerwiseMog::new(
        MogArchitecture::new(d).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters: weights(parameters),
        },
    )
}
fn resident(
    d: MogDimensions,
    parameters: &Value,
    context: &NumericContext,
) -> LayerwiseMog<NumericBackend, NumericReplicatedPolicy<Unit>> {
    let architecture = MogArchitecture::new(d).unwrap();
    let parameters = weights(parameters);
    let mut units = Vec::new();
    for group in 0..3 {
        for index in 0..architecture.group_unit_count(group).unwrap() {
            let mut unit = architecture.build_unit(group, index, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            units.push(unit);
        }
    }
    LayerwiseMog::new(architecture, NumericReplicatedPolicy::resident(units))
}
fn prediction_events(layers: usize) -> Vec<StreamPolicyEvent> {
    let mut events = vec![StreamPolicyEvent::Begin];
    for layer in 0..layers {
        events.extend([
            StreamPolicyEvent::Acquire(layer, 0, layer),
            StreamPolicyEvent::Complete(layer, 0, layer),
        ]);
    }
    events.extend([
        StreamPolicyEvent::Acquire(layers, 1, 0),
        StreamPolicyEvent::Complete(layers, 1, 0),
        StreamPolicyEvent::Finish,
    ]);
    events
}
fn selected_events(layers: usize) -> Vec<StreamPolicyEvent> {
    vec![
        StreamPolicyEvent::Begin,
        StreamPolicyEvent::Acquire(layers + 1, 2, 0),
        StreamPolicyEvent::Complete(layers + 1, 2, 0),
        StreamPolicyEvent::Finish,
    ]
}
fn mixture(step: &Value, context: &NumericContext) -> NumericTensor {
    // NVIDIA records flattened mixture IDs; selected readout consumes one ID
    // per batch/time row, retaining both singleton axes.
    let hidden = tensor(&step["guided_hidden"]);
    codes(&step["mixture_indices"], context)
        .reshape(&hidden.shape()[..2], context)
        .unwrap()
}
fn parity(
    head: &mut impl MogExecution<NumericBackend>,
    step: &Value,
    noise: f32,
    fixture: &Value,
    context: &NumericContext,
) {
    let prediction = head
        .predict(
            &tensor(&step["input"]),
            step["guidance_scale"].as_f64().unwrap() as f32,
            context,
        )
        .unwrap();
    compare(&prediction.hidden, &step["guided_hidden"], fixture);
    compare(&prediction.logits, &step["logits"], fixture);
    let sample = head
        .selected(
            &prediction,
            &mixture(step, context),
            &tensor(&step["gaussian_draw"]),
            noise,
            context,
        )
        .unwrap();
    compare(&sample.mean, &step["mu"], fixture);
    compare(&sample.log_std, &step["logs"], fixture);
    compare(&sample.latent, &step["latent"], fixture);
}

#[test]
fn voicechat_mog_resident_and_bounded_match_nvidia_without_reacquiring_prediction_for_selected_readout(
) {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let d = dimensions(&fixture["dimensions"]);
    let mut ordinary = MogHead::<NumericBackend>::new(d, &context).unwrap();
    replace_numeric_parameters(&mut ordinary, &weights(&fixture["parameters"]));
    let mut resident = resident(d, &fixture["parameters"], &context);
    let mut bounded = bounded(d, &fixture["parameters"]);
    for case in ["released_policy", "top_k_without_guidance"] {
        let case = &fixture[case];
        let noise = case["policy"]["noise_scale"].as_f64().unwrap() as f32;
        for step in case["steps"].as_array().unwrap() {
            parity(&mut ordinary, step, noise, &fixture, &context);
            parity(&mut resident, step, noise, &fixture, &context);
            bounded.runtime_mut().policy_mut().inner.events.clear();
            parity(&mut bounded, step, noise, &fixture, &context);
            let expected = [prediction_events(d.layers), selected_events(d.layers)].concat();
            assert_eq!(bounded.runtime().policy().inner.events, expected);
        }
    }
}

#[test]
fn voicechat_mog_bounded_refinement_matches_all_nvidia_code_prefixes_and_uninterrupted_output() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let d = dimensions(&fixture["dimensions"]);
    let mut head = bounded(d, &fixture["parameters"]);
    let mut refinement = RvqRefinement::<NumericBackend>::new(5, 4, 4, &context).unwrap();
    replace_numeric_parameters(&mut refinement, &weights(&fixture["parameters"]));
    let policy = RefinementPolicy {
        guidance_scale: 0.2,
        noise_scale: 0.001,
        top_p: 0.95,
    };
    let case = &fixture["released_policy"];
    let steps = case["steps"].as_array().unwrap();
    let hidden = tensor(&case["hidden"]);
    let conditional = hidden.axis_slice(0, 0, 1);
    let unconditional = hidden.axis_slice(0, 1, 2);
    let initial = refinement.start(1, 1, &context).unwrap();
    let mut state = initial.clone();
    for step in steps {
        assert_eq!(
            state.next_slot(),
            Some(step["slot"].as_u64().unwrap() as usize)
        );
        state = refinement
            .advance_sampled(
                &state,
                &conditional,
                Some(&unconditional),
                policy,
                &mut head,
                |slot, logits: &NumericTensor, top_p| {
                    assert_eq!(slot, step["slot"].as_u64().unwrap() as usize);
                    assert_eq!(top_p, 0.95);
                    compare(logits, &step["logits"], &fixture);
                    Ok((mixture(step, &context), tensor(&step["gaussian_draw"])))
                },
                &context,
            )
            .unwrap();
        assert_eq!(state.codes().data, tensor(&step["codes_after"]).data);
        compare(
            &refinement.depthsum(state.codes(), &context).unwrap(),
            &step["summed_feedback"],
            &fixture,
        );
    }
    let mut draws = 0;
    let generated = refinement
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
                Ok((mixture(step, &context), tensor(&step["gaussian_draw"])))
            },
            &context,
        )
        .unwrap();
    assert_eq!(
        draws, 5,
        "three empty slots acquire no parameters or random draws"
    );
    assert_eq!(initial.next_slot(), Some(3));
    assert_eq!(initial.codes().data, vec![5.; 31]);
    assert_eq!(generated.next_slot(), None);
    assert_eq!(generated.completed_depth(), 31);
    assert_eq!(generated.codes().data, tensor(&case["codes"]).data);
    assert_eq!(generated.codes().data, state.codes().data);
    let one_slot = [prediction_events(d.layers), selected_events(d.layers)].concat();
    assert_eq!(head.runtime().policy().inner.events, one_slot.repeat(10));
}

#[test]
fn voicechat_mog_selected_bank_failure_preserves_frame_and_shared_rng_until_retry() {
    use super::voicechat_speech_sampling::{generation, Recorded, Tape};
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
    let d = MogDimensions::from_config(&config).unwrap();
    let parts = ordinary.into_parts();
    let mut generator = TtsGenerator::from_parts(
        &config,
        16,
        5,
        TtsGeneratorParts::new(
            parts.conditioning,
            parts.character_transformer,
            parts.transformer,
            parts.refinement,
            bounded(d, &fixture["parameters"]),
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
    generator.mog.runtime_mut().policy_mut().inner.fail_acquire = Some(d.layers + 1);
    let error = generator
        .advance_sampled_frame::<Recorded, _, _, _, _, _>(&frame, &mut branch, None, &context)
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("injected dense-stream acquisition failure"));
    assert_eq!(
        tape.issued.get(),
        2,
        "selected-bank refusal follows both recorded draws"
    );
    assert_eq!(branch.random_state().unwrap().cursor, 0);
    assert_eq!(canonical.random_state().unwrap().cursor, 0);
    assert_eq!(frame.refinement().next_slot(), Some(3));
    assert_eq!(frame.refinement().codes().data, vec![5.; 31]);
    compare_cache(&caches, &call["cache"], &fixture);
    let failed_events = [
        prediction_events(d.layers),
        selected_events(d.layers)[..2].to_vec(),
    ]
    .concat();
    assert_eq!(generator.mog.runtime().policy().inner.events, failed_events);
    generator.mog.runtime_mut().policy_mut().inner.fail_acquire = None;
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

#[test]
fn voicechat_mog_parallel_tp2_resident_and_bounded_match_nvidia_guidance_and_selected_moments() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    parallel_mog_parity(&fixture, true);

    // Split a nonzero low-rank term into two equal contributions. Per mixture,
    // mus rows [a, b] become [a, b, a], and low_mat columns [x, y] become
    // [x / 2, y, x / 2]. The contraction is unchanged, so the pinned NVIDIA
    // moments remain an independent oracle for uneven (2 + 1) segmented TP.
    let mut uneven = fixture.clone();
    uneven["dimensions"]["low_rank"] = 3.into();
    let parameters = &mut uneven["parameters"];
    let mus = &mut parameters["tts_model.tts_model.mog_head.proj_mus.weight"];
    let original = mus["values"].as_array().unwrap().clone();
    mus["shape"][0] = 12.into();
    mus["values"] = Value::Array(
        original
            .chunks_exact(8)
            .flat_map(|mixture| mixture.iter().chain(&mixture[..4]).cloned())
            .collect(),
    );
    let low = &mut parameters["tts_model.tts_model.mog_head.low_mat"];
    let original = low["values"].as_array().unwrap().clone();
    low["shape"][2] = 3.into();
    low["values"] = Value::Array(
        original
            .chunks_exact(2)
            .flat_map(|row| {
                let half = Value::from(row[0].as_f64().unwrap() / 2.0);
                [half.clone(), row[1].clone(), half]
            })
            .collect(),
    );
    parallel_mog_parity(&uneven, false);
}

fn parallel_mog_parity(fixture: &Value, include_resident: bool) {
    use eredu_architectures::voicechat::{
        tts_mog::TtsError, tts_mog_layered::mog_parameter_description,
    };

    let dimensions = dimensions(&fixture["dimensions"]);
    let description = mog_parameter_description(dimensions).unwrap();
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
                    let topology = ParallelRankTopology::new(
                        ParallelTopology::new(2, 1, 1, 1).unwrap(),
                        rank,
                    )
                    .unwrap();
                    let layout = eredu_runtime::derive_partitioned_local_layout(description, topology).unwrap();
                    let context = NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let parallel =
                        NumericParallelContext::new_partitioned(rank, rank, group, world);
                    let parameters: BTreeMap<_, _> = weights(&fixture["parameters"])
                        .into_iter()
                        // The fixture also contains RVQ parameters; this head owns
                        // only the residual stack and its two independently leased heads.
                        .filter_map(|(name, value)| {
                            layout.tensor(&name).map(|tensor| {
                                let selected = select_parameter(&value, tensor.placement()).unwrap();
                                (name, selected)
                            })
                        })
                        .collect();
                    let architecture = || {
                        MogArchitecture::<NumericBackend>::new_parallel(dimensions, &layout).unwrap()
                    };
                    let mut resident = include_resident.then(|| {
                        let mut units = Vec::new();
                        for group in 0..3 {
                            for index in 0..architecture().group_unit_count(group).unwrap() {
                                let mut unit = architecture().build_unit(group, index, &context).unwrap();
                                replace_numeric_parameters(&mut unit, &parameters);
                                units.push(unit);
                            }
                        }
                        LayerwiseMog::new(architecture(), NumericReplicatedPolicy::resident(units))
                    });
                    let mut bounded = LayerwiseMog::new(
                        architecture(),
                        BoundFixturePolicy {
                            inner: RebuildingUnitPolicy::default(),
                            parameters,
                        },
                    );
                    let first = &fixture["released_policy"]["steps"][0];
                    let input = tensor(&first["input"]);
                    let guidance = first["guidance_scale"].as_f64().unwrap() as f32;
                    // A selected TP owner cannot silently execute with ordinary
                    // kernels or acquire parameters under a different communicator.
                    assert!(matches!(
                        bounded.predict(&input, guidance, &context),
                        Err(TtsError::Configuration(_))
                    ));
                    for wrong in [
                        NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                        NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                    ] {
                        assert!(matches!(
                            bounded.predict_parallel(&input, guidance, &wrong, &context),
                            Err(TtsError::Configuration(_))
                        ));
                    }
                    assert!(bounded.runtime().policy().inner.events.is_empty());
                    // Both ranks reject their first lease before collective work;
                    // retry uses the same execution object and explicit decisions.
                    bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(0);
                    assert!(bounded
                        .predict_parallel(&input, guidance, &parallel, &context)
                        .is_err());
                    assert_eq!(
                        bounded.runtime().policy().inner.events,
                        [StreamPolicyEvent::Begin, StreamPolicyEvent::Acquire(0, 0, 0)]
                    );
                    bounded.runtime_mut().policy_mut().inner.fail_acquire = None;

                    for case_name in ["released_policy", "top_k_without_guidance"] {
                        if !include_resident && case_name != "released_policy" {
                            continue;
                        }
                        let case = &fixture[case_name];
                        let noise = case["policy"]["noise_scale"].as_f64().unwrap() as f32;
                        for (step_index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
                            if !include_resident && step_index > 0 {
                                break;
                            }
                            let input = tensor(&step["input"]);
                            let guidance = step["guidance_scale"].as_f64().unwrap() as f32;
                            let selected = mixture(step, &context);
                            let gaussian = tensor(&step["gaussian_draw"]);
                            if let Some(resident) = &mut resident {
                            let prediction = resident
                                .predict_parallel(&input, guidance, &parallel, &context)
                                .unwrap();
                            compare(&prediction.hidden, &step["guided_hidden"], fixture);
                            compare(&prediction.logits, &step["logits"], fixture);
                            let sample = resident
                                .selected_parallel(
                                    &prediction, &selected, &gaussian, noise, &parallel, &context,
                                )
                                .unwrap();
                            compare(&sample.mean, &step["mu"], fixture);
                            compare(&sample.log_std, &step["logs"], fixture);
                            compare(&sample.latent, &step["latent"], fixture);

                            }

                            bounded.runtime_mut().policy_mut().inner.events.clear();
                            let prediction = bounded
                                .predict_parallel(&input, guidance, &parallel, &context)
                                .unwrap();
                            compare(&prediction.hidden, &step["guided_hidden"], fixture);
                            compare(&prediction.logits, &step["logits"], fixture);
                            assert_eq!(
                                bounded.runtime().policy().inner.events,
                                prediction_events(dimensions.layers),
                                "prediction does not acquire the selected-component bank"
                            );
                            bounded.runtime_mut().policy_mut().inner.events.clear();
                            if case_name == "released_policy" && step_index == 0 {
                                assert!(matches!(
                                    bounded.selected(&prediction, &selected, &gaussian, noise, &context),
                                    Err(TtsError::Configuration(_))
                                ));
                                for wrong in [
                                    NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                                    NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                                ] {
                                    assert!(matches!(
                                        bounded.selected_parallel(
                                            &prediction, &selected, &gaussian, noise, &wrong, &context,
                                        ),
                                        Err(TtsError::Configuration(_))
                                    ));
                                }
                                assert!(bounded.runtime().policy().inner.events.is_empty());
                                bounded.runtime_mut().policy_mut().inner.fail_acquire =
                                    Some(dimensions.layers + 1);
                                assert!(bounded
                                    .selected_parallel(
                                        &prediction, &selected, &gaussian, noise, &parallel, &context,
                                    )
                                    .is_err());
                                assert_eq!(
                                    bounded.runtime().policy().inner.events,
                                    [StreamPolicyEvent::Begin,
                                     StreamPolicyEvent::Acquire(dimensions.layers + 1, 2, 0)]
                                );
                                bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
                                bounded.runtime_mut().policy_mut().inner.events.clear();
                            }
                            let sample = bounded
                                .selected_parallel(
                                    &prediction, &selected, &gaussian, noise, &parallel, &context,
                                )
                                .unwrap();
                            compare(&sample.mean, &step["mu"], fixture);
                            compare(&sample.log_std, &step["logs"], fixture);
                            compare(&sample.latent, &step["latent"], fixture);
                            assert_eq!(
                                bounded.runtime().policy().inner.events,
                                selected_events(dimensions.layers),
                                "selected readout does not repeat residual prediction or sample decisions"
                            );
                        }
                    }
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
fn voicechat_mog_tp_preparation_failures_agree_and_retry() {
    use super::voicechat_rnnt_coordination::{coordinator_pair, Coordinator};
    use eredu_architectures::voicechat::{
        tts_mog::TtsError,
        tts_mog_layered::{mog_parameter_description, MogForward, MogInput},
    };
    use eredu_runtime::{
        replicated_execution::ReplicatedCoordinator, LayeredLocalPhase, LayeredTraversalHook,
    };
    use std::{cell::RefCell, rc::Rc, sync::atomic::Ordering};
    struct Hook(Rc<RefCell<Coordinator>>);
    impl LayeredTraversalHook<NumericBackend, MogForward<NumericTensor>, TtsError> for Hook {
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
                    0x4d4f47484f4f4b,
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
                    "MoG participant failed local traversal phase",
                ));
            }
            Ok(())
        }
    }
    let (coordinators, timeouts) = coordinator_pair();
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, coordinator)| {
                let group = group.clone();
                scope.spawn(move || {
                    let fixture: Value =
                        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON)
                            .unwrap();
                    let d = dimensions(&fixture["dimensions"]);
                    let description = mog_parameter_description(d).unwrap();
                    let topology =
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap();
                    let layout =
                        eredu_runtime::derive_partitioned_local_layout(&description, topology)
                            .unwrap();
                    let context = NumericContext::with_local_layout(layout.clone());
                    let parallel = NumericParallelContext::new(rank, group);
                    let parameters = weights(&fixture["parameters"])
                        .into_iter()
                        .filter_map(|(name, value)| {
                            layout.tensor(&name).map(|tensor| {
                                (name, select_parameter(&value, tensor.placement()).unwrap())
                            })
                        })
                        .collect();
                    let mut model = LayerwiseMog::new(
                        MogArchitecture::new_parallel(d, &layout).unwrap(),
                        BoundFixturePolicy {
                            inner: RebuildingUnitPolicy::default(),
                            parameters,
                        },
                    );
                    let coordinator = Rc::new(RefCell::new(coordinator));
                    let mut hook = Hook(coordinator.clone());
                    let agree = |descriptor, success, context: &NumericContext| {
                        coordinator
                            .borrow_mut()
                            .agree(descriptor, success, context)
                            .map_err(TtsError::from)
                    };
                    let step = &fixture["released_policy"]["steps"][0];
                    let input = tensor(&step["input"]);
                    let guidance = step["guidance_scale"].as_f64().unwrap() as f32;
                    model.runtime_mut().policy_mut().inner.fail_acquire = (rank == 0).then_some(1);
                    assert!(model
                        .invoke_parallel_guarded(
                            MogInput::Predict {
                                hidden: &input,
                                guidance
                            },
                            &parallel,
                            &context,
                            &mut hook,
                            agree
                        )
                        .is_err());
                    assert_eq!(
                        parallel.trace().len(),
                        1,
                        "only first completed block reduces"
                    );
                    model.runtime_mut().policy_mut().inner.fail_acquire = None;
                    // A malformed local projection fails after acquisition, before its peer enters sum.
                    let key = "tts_model.tts_model.mog_head.mlp_stack.1.mlp.down_proj.weight";
                    let saved = model.runtime().policy().parameters[key].clone();
                    if rank == 0 {
                        model
                            .runtime_mut()
                            .policy_mut()
                            .parameters
                            .insert(key.into(), NumericTensor::zeros([d.hidden, 1]));
                    }
                    let before = parallel.trace().len();
                    assert!(model
                        .invoke_parallel_guarded(
                            MogInput::Predict {
                                hidden: &input,
                                guidance
                            },
                            &parallel,
                            &context,
                            &mut hook,
                            agree
                        )
                        .is_err());
                    assert_eq!(parallel.trace().len() - before, 1);
                    model
                        .runtime_mut()
                        .policy_mut()
                        .parameters
                        .insert(key.into(), saved);
                    let (_, prediction) = model
                        .invoke_parallel_guarded(
                            MogInput::Predict {
                                hidden: &input,
                                guidance,
                            },
                            &parallel,
                            &context,
                            &mut hook,
                            agree,
                        )
                        .unwrap();
                    let prediction = prediction.into_prediction().unwrap();
                    compare(&prediction.hidden, &step["guided_hidden"], &fixture);
                    compare(&prediction.logits, &step["logits"], &fixture);
                    let mixture = mixture(step, &context);
                    let gaussian = tensor(&step["gaussian_draw"]);
                    let noise_scale = fixture["released_policy"]["policy"]["noise_scale"]
                        .as_f64()
                        .unwrap() as f32;
                    let key = "tts_model.tts_model.mog_head.low_mat";
                    let saved = model.runtime().policy().parameters[key].clone();
                    if rank == 0 {
                        model.runtime_mut().policy_mut().parameters.insert(
                            key.into(),
                            NumericTensor::zeros([d.mixtures, d.latent, saved.dim(2) + 1]),
                        );
                    }
                    let before = parallel.trace().len();
                    assert!(model
                        .invoke_parallel_guarded(
                            MogInput::Selected {
                                prediction: &prediction,
                                mixture: &mixture,
                                gaussian: &gaussian,
                                noise_scale
                            },
                            &parallel,
                            &context,
                            &mut hook,
                            agree
                        )
                        .is_err());
                    assert_eq!(
                        parallel.trace().len(),
                        before,
                        "malformed selected moments cannot enter sum"
                    );
                    model
                        .runtime_mut()
                        .policy_mut()
                        .parameters
                        .insert(key.into(), saved);
                    let (_, sampled) = model
                        .invoke_parallel_guarded(
                            MogInput::Selected {
                                prediction: &prediction,
                                mixture: &mixture,
                                gaussian: &gaussian,
                                noise_scale,
                            },
                            &parallel,
                            &context,
                            &mut hook,
                            agree,
                        )
                        .unwrap();
                    compare(
                        &sampled.into_sample().unwrap().latent,
                        &step["latent"],
                        &fixture,
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
