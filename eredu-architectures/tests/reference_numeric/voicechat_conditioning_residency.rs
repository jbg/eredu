//! Conditioning equations with independently leased character, ingress and fusion units.
use super::voicechat_mog_residency::weights;
use super::voicechat_tts_generator::{codes, compare, compare_cache, model, tensor};
use super::*;
use eredu_architectures::voicechat::{
    tts_conditioning::CharacterDimensions,
    tts_conditioning_layered::{
        ConditioningArchitecture, ConditioningDimensions, ConditioningExecution, ConditioningUnit,
        LayerwiseConditioning, ResidentConditioning,
    },
    tts_ingress::{PromptLayout, SpeakerPrompt},
    tts_rvq::RvqRefinement,
};
use serde_json::Value;

type Unit = ConditioningUnit<NumericBackend>;
struct BoundFixturePolicy {
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
fn dimensions(characters: i32, width: i32, output_width: i32) -> ConditioningDimensions {
    ConditioningDimensions::new(
        CharacterDimensions {
            characters,
            width,
            output_width,
            use_continuation: true,
            use_special: true,
        },
        5,
    )
    .unwrap()
}
fn bounded(
    d: ConditioningDimensions,
    parameters: BTreeMap<String, NumericTensor>,
) -> LayerwiseConditioning<NumericBackend, BoundFixturePolicy> {
    LayerwiseConditioning::new(
        ConditioningArchitecture::new(d).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters,
        },
    )
}
fn events(group: usize) -> Vec<StreamPolicyEvent> {
    vec![
        StreamPolicyEvent::Begin,
        StreamPolicyEvent::Acquire(group, group, 0),
        StreamPolicyEvent::Complete(group, group, 0),
        StreamPolicyEvent::Finish,
    ]
}

#[test]
fn voicechat_conditioning_resident_and_bounded_character_pooling_and_fusion_match_nvidia() {
    use super::voicechat_conditioning::{compare as character_compare, integers, mapping};
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CONDITIONING_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mapping = mapping(&fixture);
    let tokens = integers(&fixture["tokens"]);
    let selected = fixture["selected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect::<Vec<_>>();
    let batch = mapping
        .prepare(&tokens, Some(&selected), 2, 4, 1024)
        .unwrap();
    let d = dimensions(mapping.characters().len() as i32, 4, 4);
    let mut resident = ResidentConditioning::<NumericBackend>::new(d, &context).unwrap();
    replace_numeric_parameters(&mut resident, &weights(&fixture["parameters"]));
    let mut bounded = bounded(d, weights(&fixture["parameters"]));
    let resident_output = resident
        .character_forward(
            &batch,
            |embedded, _| {
                character_compare(embedded, &fixture["embedded"], &fixture);
                Ok(tensor(&fixture["hidden"]))
            },
            &context,
        )
        .unwrap()
        .unwrap();
    let output = bounded
        .character_forward(
            &batch,
            |embedded, _| {
                character_compare(embedded, &fixture["embedded"], &fixture);
                Ok(tensor(&fixture["hidden"]))
            },
            &context,
        )
        .unwrap()
        .unwrap();
    character_compare(&resident_output, &fixture["character_output"], &fixture);
    character_compare(&output, &fixture["character_output"], &fixture);
    assert_eq!(bounded.runtime().policy().inner.events, events(0).repeat(2));
    bounded.runtime_mut().policy_mut().inner.events.clear();
    for fused in [
        resident
            .fuse(&tensor(&fixture["audio"]), &resident_output, &context)
            .unwrap(),
        bounded
            .fuse(&tensor(&fixture["audio"]), &output, &context)
            .unwrap(),
    ] {
        character_compare(&fused, &fixture["fusion_output"], &fixture);
    }
    assert_eq!(bounded.runtime().policy().inner.events, events(2));
    bounded.runtime_mut().policy_mut().inner.events.clear();
    let empty = mapping
        .prepare(&tokens, Some(&[false; 8]), 2, 4, 0)
        .unwrap();
    assert!(
        bounded
            .character_forward(
                &empty,
                |_, _| panic!("empty character selection cannot execute its backbone"),
                &context
            )
            .unwrap()
            .is_none()
    );
    assert!(bounded.runtime().policy().inner.events.is_empty());
}

#[test]
fn voicechat_conditioning_bounded_ingress_matches_nvidia_speaker_prompt_and_decode_lanes() {
    use super::voicechat_tts_ingress::compare as ingress_compare;
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::INGRESS_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mut parameters = weights(&fixture["parameters"]);
    parameters.insert(
        "tts_model.audio_prompt_latents.Aria".into(),
        tensor(&fixture["cases"][0]["supplied_speaker"]),
    );
    let d = dimensions(16, 4, 4);
    let mut resident = ResidentConditioning::<NumericBackend>::new(d, &context).unwrap();
    replace_numeric_parameters(&mut resident, &parameters);
    let mut bounded = bounded(d, parameters.clone());
    let mut feedback = RvqRefinement::<NumericBackend>::new(5, 4, 4, &context).unwrap();
    replace_numeric_parameters(&mut feedback, &parameters);
    for case in fixture["cases"].as_array().unwrap() {
        let code = codes(&case["code"], &context);
        let condition = tensor(&case["supplied_subwords"]);
        let any = case["subword_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_bool().unwrap());
        let condition = any.then_some(&condition);
        let (ordinary, leased) = if case["audio_mask"].is_null() {
            (
                resident
                    .decode(&mut feedback, &code, condition, true, &context)
                    .unwrap(),
                bounded
                    .decode(&mut feedback, &code, condition, true, &context)
                    .unwrap(),
            )
        } else {
            let mask = case["audio_mask"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect::<Vec<_>>();
            let layout = PromptLayout::new(code.dim(0), code.dim(1), &mask).unwrap();
            let speaker = if case["supplied_speaker"].is_null() {
                SpeakerPrompt::ProjectedCodes
            } else {
                SpeakerPrompt::Aria
            };
            (
                resident
                    .prefill(
                        &mut feedback,
                        &code,
                        &layout,
                        speaker,
                        condition,
                        true,
                        &context,
                    )
                    .unwrap(),
                bounded
                    .prefill(
                        &mut feedback,
                        &code,
                        &layout,
                        speaker,
                        condition,
                        true,
                        &context,
                    )
                    .unwrap(),
            )
        };
        for lanes in [&ordinary, &leased] {
            ingress_compare(&lanes.audio, &case["audio_embedding"], &fixture, &context);
            ingress_compare(&lanes.text, &case["conditioning"], &fixture, &context);
        }
        assert_eq!(bounded.runtime().policy().inner.events, events(1));
        bounded.runtime_mut().policy_mut().inner.events.clear();
        ingress_compare(
            &bounded.fuse(&leased.audio, &leased.text, &context).unwrap(),
            &case["backbone"]["inputs"],
            &fixture,
            &context,
        );
        assert_eq!(bounded.runtime().policy().inner.events, events(2));
        bounded.runtime_mut().policy_mut().inner.events.clear();
    }
}

#[test]
fn voicechat_conditioning_fusion_acquisition_failure_preserves_causal_parent_and_cached_retry() {
    use super::voicechat_transformer::CausalCache;
    use super::{
        voicechat_mog_residency, voicechat_rvq_residency, voicechat_transformer_residency,
    };
    use eredu_architectures::voicechat::{
        tts_generator::{TtsGenerator, TtsGeneratorParts},
        tts_mog::MogDimensions,
        tts_rvq::RvqDimensions,
        tts_transformer::TransformerConfig,
    };
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let config = model(&fixture, &context).config().clone();
    let mut generator = TtsGenerator::from_parts(
        &config,
        16,
        5,
        TtsGeneratorParts::new(
            bounded(
                ConditioningDimensions::new(
                    CharacterDimensions {
                        characters: 16,
                        width: config.character_encoder.hidden_size as i32,
                        output_width: config.backbone.hidden_size as i32,
                        use_continuation: config.use_subword_flag_emb,
                        use_special: config.use_bos_eos_emb,
                    },
                    5,
                )
                .unwrap(),
                weights(&fixture["parameters"]),
            ),
            voicechat_transformer_residency::bounded(
                TransformerConfig::new(&config.character_encoder).unwrap(),
                &fixture["parameters"],
            ),
            voicechat_transformer_residency::bounded(
                TransformerConfig::new(&config.backbone).unwrap(),
                &fixture["parameters"],
            ),
            voicechat_rvq_residency::bounded(
                &fixture["parameters"],
                RvqDimensions::from_config(&config).unwrap(),
            ),
            voicechat_mog_residency::bounded(
                MogDimensions::from_config(&config).unwrap(),
                &fixture["parameters"],
            ),
        ),
    )
    .unwrap();
    let prompt = &fixture["calls"][0];
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
    let hidden = generator
        .prefill(
            &codes(&prompt["input_codes"], &context),
            &PromptLayout::new(1, 5, &mask).unwrap(),
            SpeakerPrompt::Aria,
            Some(&tensor(&prompt["supplied_subwords"])),
            &mut caches,
            &context,
        )
        .unwrap();
    compare(&hidden, &prompt["hidden"], &fixture);
    compare_cache(&caches, &prompt["cache"], &fixture);
    assert_eq!(
        generator.conditioning.runtime().policy().inner.events,
        [events(1), events(2)].concat()
    );
    generator
        .conditioning
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    generator
        .conditioning
        .runtime_mut()
        .policy_mut()
        .inner
        .fail_acquire = Some(2);
    let call = &fixture["calls"][1];
    let error = generator
        .begin_frame(
            &codes(&call["input_codes"], &context),
            Some(&tensor(&call["supplied_subwords"])),
            &mut caches,
            &context,
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected dense-stream acquisition failure")
    );
    compare_cache(&caches, &prompt["cache"], &fixture);
    assert_eq!(
        generator.conditioning.runtime().policy().inner.events,
        [events(1), events(2)[..2].to_vec()].concat()
    );
    generator
        .conditioning
        .runtime_mut()
        .policy_mut()
        .inner
        .fail_acquire = None;
    generator
        .conditioning
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    for call in &fixture["calls"].as_array().unwrap()[1..3] {
        let any = call["subword_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_bool().unwrap());
        let supplied = tensor(&call["supplied_subwords"]);
        let frame = generator
            .begin_frame(
                &codes(&call["input_codes"], &context),
                any.then_some(&supplied),
                &mut caches,
                &context,
            )
            .unwrap();
        let hidden = NumericTensor::concatenate(
            &[
                frame.conditional().clone(),
                frame.unconditional().unwrap().clone(),
            ],
            0,
            &context,
        )
        .unwrap();
        compare(&hidden, &call["hidden"], &fixture);
        compare_cache(&caches, &call["cache"], &fixture);
        let mut draws = call["refinement"].as_array().unwrap().iter();
        let complete = generator
            .generate_frame(
                &frame,
                |slot, logits: &NumericTensor, _| {
                    let expected = draws.next().unwrap();
                    assert_eq!(slot, expected["slot"].as_u64().unwrap() as usize);
                    compare(logits, &expected["logits"], &fixture);
                    Ok((
                        codes(&expected["mixture_indices"], &context)
                            .reshape(&logits.shape()[..2], &context)
                            .unwrap(),
                        tensor(&expected["gaussian_draw"]),
                    ))
                },
                &context,
            )
            .unwrap();
        assert!(draws.next().is_none());
        assert_eq!(
            complete.codes().unwrap().data,
            codes(&call["codes"], &context).data
        );
        compare_cache(&caches, &call["cache"], &fixture);
    }
    assert_eq!(
        generator.conditioning.runtime().policy().inner.events,
        [events(1), events(2)].concat().repeat(2),
        "supplied character conditioning never acquires the character unit"
    );
}

type ParallelResident = LayerwiseConditioning<NumericBackend, NumericReplicatedPolicy<Unit>>;
type ParallelBounded = LayerwiseConditioning<NumericBackend, BoundFixturePolicy>;

fn with_parallel_conditioning<F>(
    dimensions: ConditioningDimensions,
    parameters: BTreeMap<String, NumericTensor>,
    run: F,
) where
    F: Fn(
            usize,
            &NumericParallelContext,
            &NumericContext,
            &mut ParallelResident,
            &mut ParallelBounded,
        ) + Sync,
{
    use eredu_architectures::voicechat::tts_conditioning_layered::conditioning_parameter_description;
    let description = conditioning_parameter_description(dimensions).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let world = world.clone();
                let group = group.clone();
                let parameters = &parameters;
                let description = &description;
                let run = &run;
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
                    let parameters = parameters
                        .iter()
                        .filter_map(|(name, value)| {
                            layout.tensor(name).map(|tensor| {
                                (
                                    name.clone(),
                                    select_parameter(value, tensor.placement()).unwrap(),
                                )
                            })
                        })
                        .collect::<BTreeMap<_, _>>();
                    let architecture =
                        || ConditioningArchitecture::new_parallel(dimensions, &layout).unwrap();
                    let units = (0..3)
                        .map(|group| {
                            let mut unit = architecture().build_unit(group, 0, &context).unwrap();
                            replace_numeric_parameters(&mut unit, &parameters);
                            unit
                        })
                        .collect();
                    let mut resident = LayerwiseConditioning::new(
                        architecture(),
                        NumericReplicatedPolicy::resident(units),
                    );
                    let mut bounded = LayerwiseConditioning::new(
                        architecture(),
                        BoundFixturePolicy {
                            inner: RebuildingUnitPolicy::default(),
                            parameters,
                        },
                    );
                    run(rank, &parallel, &context, &mut resident, &mut bounded);
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
fn voicechat_conditioning_parallel_shards_cannot_be_repackaged_as_ordinary_resident_units() {
    use eredu_architectures::voicechat::{
        tts_conditioning_layered::conditioning_parameter_description, tts_mog::TtsError,
    };
    let d = dimensions(2, 4, 4);
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
    let layout = eredu_runtime::derive_partitioned_local_layout(
        &conditioning_parameter_description(d).unwrap(),
        topology,
    )
    .unwrap();
    let context = NumericContext::default();
    let ordinary = ConditioningArchitecture::<NumericBackend>::new(d).unwrap();
    let sharded = ConditioningArchitecture::<NumericBackend>::new_parallel(d, &layout).unwrap();
    // Each public unit can be extracted independently. No mixture of full and
    // local units may establish an ordinary executable that publishes partial features.
    for selected in 0..3 {
        let build = |group| {
            if group == selected {
                &sharded
            } else {
                &ordinary
            }
            .build_unit(group, 0, &context)
            .unwrap()
        };
        let ConditioningUnit::Character(character) = build(0) else {
            panic!("character group built another unit")
        };
        let ConditioningUnit::Ingress(ingress) = build(1) else {
            panic!("ingress group built another unit")
        };
        let ConditioningUnit::Fusion(fusion) = build(2) else {
            panic!("fusion group built another unit")
        };
        assert!(matches!(
            ResidentConditioning::from_parts(d, character, ingress, fusion),
            Err(TtsError::Configuration(_))
        ));
    }
}

#[test]
fn voicechat_conditioning_parallel_tp2_matches_nvidia_character_flags_fusion_and_retry() {
    use super::voicechat_conditioning::{compare, integers, mapping};
    use eredu_architectures::voicechat::tts_mog::TtsError;
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CONDITIONING_REFERENCE_JSON).unwrap();
    let mapping = mapping(&fixture);
    let selected = fixture["selected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_bool().unwrap())
        .collect::<Vec<_>>();
    let batch = mapping
        .prepare(&integers(&fixture["tokens"]), Some(&selected), 2, 4, 1024)
        .unwrap();
    with_parallel_conditioning(
        dimensions(mapping.characters().len() as i32, 4, 4),
        weights(&fixture["parameters"]),
        |rank, parallel, context, resident, bounded| {
            let hidden = tensor(&fixture["hidden"]);
            let audio = tensor(&fixture["audio"]);
            let text = tensor(&fixture["character_output"]);
            for wrong in [
                NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
            ] {
                assert!(matches!(
                    bounded.character_embed_parallel(&batch, &wrong, context),
                    Err(TtsError::Configuration(_))
                ));
                assert!(matches!(
                    bounded.character_finish_parallel(&batch, &hidden, &wrong, context),
                    Err(TtsError::Configuration(_))
                ));
                assert!(matches!(
                    bounded.fuse_parallel(&audio, &text, &wrong, context),
                    Err(TtsError::Configuration(_))
                ));
            }
            assert!(matches!(
                bounded.character_embed(&batch, context),
                Err(TtsError::Configuration(_))
            ));
            assert!(bounded.runtime().policy().inner.events.is_empty());
            for embedding in [
                resident
                    .character_embed_parallel(&batch, parallel, context)
                    .unwrap(),
                bounded
                    .character_embed_parallel(&batch, parallel, context)
                    .unwrap(),
            ] {
                compare(&embedding, &fixture["embedded"], &fixture);
            }
            for output in [
                resident
                    .character_finish_parallel(&batch, &hidden, parallel, context)
                    .unwrap(),
                bounded
                    .character_finish_parallel(&batch, &hidden, parallel, context)
                    .unwrap(),
            ] {
                // The oracle includes flags at unselected token positions.
                compare(&output, &fixture["character_output"], &fixture);
            }
            assert_eq!(bounded.runtime().policy().inner.events, events(0).repeat(2));
            bounded.runtime_mut().policy_mut().inner.events.clear();
            // Fail on both ranks before a collective, then retry the same inputs.
            bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(2);
            assert!(
                bounded
                    .fuse_parallel(&audio, &text, parallel, context)
                    .is_err()
            );
            assert_eq!(bounded.runtime().policy().inner.events, events(2)[..2]);
            bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
            bounded.runtime_mut().policy_mut().inner.events.clear();
            for fused in [
                resident
                    .fuse_parallel(&audio, &text, parallel, context)
                    .unwrap(),
                bounded
                    .fuse_parallel(&audio, &text, parallel, context)
                    .unwrap(),
            ] {
                compare(&fused, &fixture["fusion_output"], &fixture);
            }
            assert_eq!(bounded.runtime().policy().inner.events, events(2));
        },
    );
}

#[test]
fn voicechat_conditioning_parallel_tp2_matches_nvidia_speaker_prompt_and_decode_lanes() {
    use super::voicechat_tts_ingress::compare;
    use eredu_architectures::voicechat::tts_ingress::shift_prompt_codes;
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::INGRESS_REFERENCE_JSON).unwrap();
    let mut parameters = weights(&fixture["parameters"]);
    parameters.insert(
        "tts_model.audio_prompt_latents.Aria".into(),
        tensor(&fixture["cases"][0]["supplied_speaker"]),
    );
    with_parallel_conditioning(
        dimensions(16, 4, 4),
        parameters.clone(),
        |_, parallel, context, resident, bounded| {
            let feedback_context = NumericContext::default();
            let mut feedback =
                RvqRefinement::<NumericBackend>::new(5, 4, 4, &feedback_context).unwrap();
            replace_numeric_parameters(&mut feedback, &parameters);
            for case in fixture["cases"].as_array().unwrap() {
                let code = codes(&case["code"], &feedback_context);
                let condition = tensor(&case["supplied_subwords"]);
                let any = case["subword_mask"]["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v.as_bool().unwrap());
                let condition = any.then_some(&condition);
                let (ordinary, leased) = if case["audio_mask"].is_null() {
                    let embedded = feedback.embed_codes(&code, &feedback_context).unwrap();
                    (
                        resident
                            .decode_embedded_parallel(&embedded, condition, true, parallel, context)
                            .unwrap(),
                        bounded
                            .decode_embedded_parallel(&embedded, condition, true, parallel, context)
                            .unwrap(),
                    )
                } else {
                    let mask = case["audio_mask"]["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_bool().unwrap())
                        .collect::<Vec<_>>();
                    let layout = PromptLayout::new(code.dim(0), code.dim(1), &mask).unwrap();
                    let speaker = if case["supplied_speaker"].is_null() {
                        SpeakerPrompt::ProjectedCodes
                    } else {
                        SpeakerPrompt::Aria
                    };
                    let shifted = shift_prompt_codes(&code, &layout, &feedback_context).unwrap();
                    let embedded = feedback.embed_codes(&shifted, &feedback_context).unwrap();
                    (
                        resident
                            .prefill_embedded_parallel(
                                &embedded, &layout, speaker, condition, true, parallel, context,
                            )
                            .unwrap(),
                        bounded
                            .prefill_embedded_parallel(
                                &embedded, &layout, speaker, condition, true, parallel, context,
                            )
                            .unwrap(),
                    )
                };
                for lanes in [&ordinary, &leased] {
                    compare(&lanes.audio, &case["audio_embedding"], &fixture, context);
                    compare(&lanes.text, &case["conditioning"], &fixture, context);
                }
                assert_eq!(bounded.runtime().policy().inner.events, events(1));
                bounded.runtime_mut().policy_mut().inner.events.clear();
                compare(
                    &bounded
                        .fuse_parallel(&leased.audio, &leased.text, parallel, context)
                        .unwrap(),
                    &case["backbone"]["inputs"],
                    &fixture,
                    context,
                );
                assert_eq!(bounded.runtime().policy().inner.events, events(2));
                bounded.runtime_mut().policy_mut().inner.events.clear();
            }
        },
    );
}

#[test]
fn voicechat_conditioning_parallel_uneven_width_gathers_biases_before_global_fusion_norm() {
    use eredu_architectures::voicechat::tts_conditioning::{CharacterFlags, CharacterMapping};
    let prefix = "tts_model.tts_model.gated_fusion_audio_text";
    let mut parameters = BTreeMap::new();
    let mapping = CharacterMapping::new(
        &BTreeMap::from([(0, "a".to_owned())]),
        CharacterFlags {
            continuation: vec![0, 0],
            continuation_pad: 1,
            special: vec![0],
            special_pad: 0,
        },
    )
    .unwrap();
    let batch = mapping.prepare(&[0], None, 1, 1, 1).unwrap();
    parameters.insert(
        "tts_model.tts_model.embed_subword.embed_tokens.weight".into(),
        NumericTensor::new(vec![2, 3], vec![1., 2., 3., 0., 0., 0.]),
    );
    parameters.insert(
        "tts_model.tts_model.embed_subword.proj_embedding.weight".into(),
        NumericTensor::new(
            vec![5, 3],
            vec![1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., -1., 2., -3.],
        ),
    );
    let mut audio_weight = vec![0.0; 25];
    let mut text_weight = vec![0.0; 25];
    for index in 0..5 {
        audio_weight[index * 5 + index] = (index + 1) as f32;
        text_weight[index * 5 + index] = 1.0;
    }
    for (name, shape, data) in [
        ("audio_proj.weight", vec![5, 5], audio_weight),
        ("text_proj.weight", vec![5, 5], text_weight),
        ("audio_proj.bias", vec![5], vec![1., 2., 3., 4., 5.]),
        ("text_proj.bias", vec![5], vec![-1., -2., -3., -4., -5.]),
        ("gate", vec![5], vec![0.; 5]),
        ("residual_scale", vec![], vec![0.]),
        ("final_norm.weight", vec![5], vec![0., 1., -0.5, 2., -1.]),
    ] {
        parameters.insert(format!("{prefix}.{name}"), NumericTensor::new(shape, data));
    }
    with_parallel_conditioning(
        ConditioningDimensions::new(
            CharacterDimensions {
                characters: 1,
                width: 3,
                output_width: 5,
                use_continuation: false,
                use_special: false,
            },
            5,
        )
        .unwrap(),
        parameters,
        |_, parallel, context, resident, bounded| {
            for embedding in [
                resident
                    .character_embed_parallel(&batch, parallel, context)
                    .unwrap(),
                bounded
                    .character_embed_parallel(&batch, parallel, context)
                    .unwrap(),
            ] {
                assert_eq!(embedding.shape(), [1, 1, 3]);
                assert_eq!(embedding.data, [1., 2., 3.]);
            }
            let hidden = NumericTensor::new(vec![1, 1, 3], vec![1., 2., 3.]);
            for output in [
                resident
                    .character_finish_parallel(&batch, &hidden, parallel, context)
                    .unwrap(),
                bounded
                    .character_finish_parallel(&batch, &hidden, parallel, context)
                    .unwrap(),
            ] {
                assert_eq!(output.shape(), [1, 1, 5]);
                assert_eq!(output.data, [1., 2., 3., 6., -6.]);
            }
            assert_eq!(bounded.runtime().policy().inner.events, events(0).repeat(2));
            bounded.runtime_mut().policy_mut().inner.events.clear();
            let audio = NumericTensor::new(vec![1, 1, 5], vec![31., 62., 93., 124., 155.]);
            let text = NumericTensor::new(vec![1, 1, 5], vec![5., -4., 3., -2., 1.]);
            // Mixed features [1.5, 0, 3, 3.5, 6.5] have mean square 13.15.
            // Literal scalar oracle includes the learned offset norm scale.
            let expected = [0.41364556, 0., 0.41364556, 2.895519, 0.];
            for output in [
                resident
                    .fuse_parallel(&audio, &text, parallel, context)
                    .unwrap(),
                bounded
                    .fuse_parallel(&audio, &text, parallel, context)
                    .unwrap(),
            ] {
                assert_eq!(output.shape(), [1, 1, 5]);
                for (actual, expected) in output.data.iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "{actual} versus {expected}"
                    );
                }
            }
            assert_eq!(bounded.runtime().policy().inner.events, events(2));
        },
    );
}
