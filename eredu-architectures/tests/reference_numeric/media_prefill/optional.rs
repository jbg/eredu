//! Actual optional roots use the shared selected traversal, not synthetic embeddings.
use super::*;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Gemma,
    Inkling,
}
fn config(family: Family, raw_vision: bool) -> serde_json::Value {
    match family {
        Family::Gemma => serde_json::json!({
            "model_type":"gemma4_unified","tie_word_embeddings":false,"image_token_id":5,"audio_token_id":6,
            "text_config":{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":4,"intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"rms_norm_eps":0.00001,"vocab_size":19,"max_position_embeddings":64,"attention_bias":true,"attention_k_eq_v":false,"num_kv_shared_layers":2,"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],"sliding_window":4,"enable_moe_block":false,"hidden_size_per_layer_input":4,"vocab_size_per_layer_input":19,"final_logit_softcapping":7.0},
            "vision_config":{"hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"patch_size":2,"pooling_kernel_size":2,"position_embedding_size":2,"rms_norm_eps":0.00001},
            "audio_config":{"hidden_size":8,"num_hidden_layers":2,"num_attention_heads":2,"output_proj_dims":8,"conv_kernel_size":3,"attention_chunk_size":4,"attention_context_left":5,"attention_context_right":0,"attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,"residual_weight":0.5,"rms_norm_eps":0.00001,"subsampling_conv_channels":[4,8]}
        }),
        Family::Inkling => {
            let mut c = serde_json::json!({"model_type":"inkling_mm_model","image_token_id":5,"audio_token_id":6,
                "text_config":{"hidden_size":8,"num_hidden_layers":2,"vocab_size":19,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"sliding_window_size":4,"layer_types":["sliding_attention","full_attention"],"mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,"d_rel":2,"rel_extent":8,"intermediate_size":12,"dense_intermediate_size":12,"n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":1,"unpadded_vocab_size":19},
                "audio_config":{"text_hidden_size":8,"num_codebooks":2,"codebook_size":4}});
            if raw_vision {
                c["vision_config"] = serde_json::json!({"text_hidden_size":8,"patch_size":40,"temporal_patch_size":2,"num_channels":3,"num_hidden_layers":4});
            }
            c
        }
    }
}
fn input(family: Family, image: bool, audio: bool) -> Input {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart as Part, PreparedInputPayload as Payload};
    let text = |tokens: &[usize]| {
        Part::new(
            InputModality::Text,
            Payload::TokenIds(NumericTensor::token_ids(tokens)),
            [],
        )
        .unwrap()
    };
    let mut parts = vec![text(&[1, 2])];
    if image {
        parts.push(match family {
            Family::Gemma => Part::new_with_extents(
                InputModality::Image,
                Payload::Tensor(NumericTensor::new(
                    [1, 4, 12],
                    (0..48).map(|i| (i as f32 - 24.) / 100.).collect(),
                )),
                [
                    (
                        InputMetadataKey::PatchGrid,
                        NumericTensor::new([1, 3], vec![1., 2., 2.]),
                    ),
                    (
                        InputMetadataKey::PatchPositions,
                        NumericTensor::new([1, 4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]),
                    ),
                ],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 2,
                }],
            )
            .unwrap(),
            Family::Inkling => Part::new(
                InputModality::Image,
                Payload::Tensor(NumericTensor::new(
                    [1, 2, 40, 40, 3],
                    (0..9600).map(|i| ((i % 37) as f32 - 18.) / 80.).collect(),
                )),
                [],
            )
            .unwrap(),
        });
    }
    if audio {
        parts.push(match family {
            Family::Gemma => Part::new_with_extents(
                InputModality::Audio,
                Payload::Tensor(NumericTensor::new(
                    [1, 8, 128],
                    (0..1024)
                        .map(|i| {
                            if i < 640 {
                                ((i % 17) as f32 - 8.) / 100.
                            } else {
                                7.
                            }
                        })
                        .collect(),
                )),
                [(
                    InputMetadataKey::AudioMask,
                    NumericTensor::new([1, 8], vec![1., 1., 1., 1., 1., 0., 0., 0.]),
                )],
                [InputExtent::AudioValidFrames(5)],
            )
            .unwrap(),
            Family::Inkling => Part::new_with_extents(
                InputModality::Audio,
                Payload::Tensor(NumericTensor::new([1, 3, 2], vec![0., 1., 2., 3., 1., 0.])),
                [(
                    InputMetadataKey::AudioMask,
                    NumericTensor::new([1, 3], vec![1., 1., 0.]),
                )],
                [InputExtent::AudioValidFrames(2)],
            )
            .unwrap(),
        });
    }
    parts.push(text(&[4, 3, 1]));
    Input::new(parts, |tensor| {
        eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
    })
    .unwrap()
}
// Two actual raw patches precede the same ordered text tokens. Axis zero of
// the raw payload is a patch population, while the admitted decoder batch is one.
fn inkling_image_first_input() -> Input {
    use eredu_runtime::{PreparedInputPart as Part, PreparedInputPayload as Payload};
    let mut parts = input(Family::Inkling, false, false).into_parts();
    parts.insert(
        0,
        Part::new(
            eredu_core::InputModality::Image,
            Payload::Tensor(NumericTensor::new(
                [2, 2, 40, 40, 3],
                (0..19200).map(|i| ((i % 43) as f32 - 21.) / 90.).collect(),
            )),
            [],
        )
        .unwrap(),
    );
    Input::new(parts, |tensor| {
        eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
    })
    .unwrap()
}
fn encoders(family: Family, projections: &[(String, Vec<i32>)]) -> Vec<(String, Vec<i32>)> {
    projections
        .iter()
        .filter(|(name, _)| match family {
            Family::Gemma => {
                name.starts_with("model.vision_")
                    || name.starts_with("model.audio_")
                    || name.starts_with("model.embed_vision")
                    || name.starts_with("model.embed_audio")
            }
            Family::Inkling => name.starts_with("visual.") || name.starts_with("audio."),
        })
        .cloned()
        .collect()
}
// Gemma's vision and audio stacks are independent: retained ingress starts
// each input projection at its group, while ordinary input starts both eagerly.
// Keep every operation/shape and its order within each independent stack.
fn same_encoder_trace(
    family: Family,
    actual: &[(String, Vec<i32>)],
    expected: &[(String, Vec<i32>)],
) {
    if family == Family::Gemma {
        for audio in [false, true] {
            let lane = |trace: &[(String, Vec<i32>)]| {
                trace
                    .iter()
                    .filter(|(name, _)| {
                        (name.starts_with("model.audio_") || name.starts_with("model.embed_audio"))
                            == audio
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                lane(actual),
                lane(expected),
                "same ordered independent Gemma encoder"
            );
        }
    } else {
        assert_eq!(actual, expected);
    }
}

fn head(family: Family, name: &str) -> bool {
    match family {
        Family::Gemma => name == "lm_head.weight",
        Family::Inkling => name == "lm_head.weight",
    }
}
fn trace(
    family: Family,
    report: &Report,
    schedule: Schedule,
    owns_output: bool,
    owns_ingress: bool,
    image: bool,
    audio: bool,
) {
    trace_with_vision_rows(
        family,
        report,
        schedule,
        owns_output,
        owns_ingress,
        image,
        audio,
        1,
    );
}
fn trace_with_vision_rows(
    family: Family,
    report: &Report,
    schedule: Schedule,
    owns_output: bool,
    owns_ingress: bool,
    image: bool,
    audio: bool,
    vision_rows: i32,
) {
    let end = schedule.cancel_after.unwrap_or(report.input_positions);
    let spans = (0..end)
        .step_by(schedule.chunk as usize)
        .map(|start| {
            let stop = (start + schedule.chunk).min(report.input_positions);
            PrefillChunk {
                input: start..stop,
                position: start,
                output: schedule.output.for_chunk(stop == report.input_positions),
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(report.trace.announced, spans);
    assert_eq!(report.trace.delivered, spans);
    assert_eq!(
        report.locally_requested,
        schedule.locally_cancel && schedule.cancel_after.is_some()
    );
    assert_eq!(
        report.outcome,
        if schedule.cancel_after.is_some() {
            PrefillOutcome::Cancelled
        } else {
            PrefillOutcome::Complete
        }
    );
    let first = encoders(family, &report.trace.projections[0]);
    for projections in &report.trace.projections {
        assert_eq!(
            encoders(family, projections),
            first,
            "encoders run only in the first committed span"
        );
    }
    if !image && !audio {
        assert!(first.is_empty());
    }
    if family == Family::Inkling && !image {
        assert!(
            first.is_empty(),
            "zero-unit audio has no invented linear projection"
        );
    }
    if owns_ingress {
        let roots = &report.trace.roots[0];
        if image {
            assert!(
                roots
                    .iter()
                    .any(|(shape, values)| shape == &[1, vision_rows, 8]
                        && values.iter().any(|x| x.abs() > 1e-8)
                        && values.iter().all(|x| x.is_finite())),
                "first span completes actual future vision rows"
            );
        }
        if audio {
            assert!(
                roots.iter().any(|(shape, values)| shape == &[1, 2, 8]
                    && values.iter().any(|x| x.abs() > 1e-8)
                    && values.iter().all(|x| x.is_finite())),
                "first span completes actual future audio rows"
            );
        }
    }
    let mut heads = Vec::new();
    for (span, projections) in spans.iter().zip(&report.trace.projections) {
        if owns_output && span.output != OutputDemand::StateOnly {
            heads.push(vec![
                1,
                if span.output == OutputDemand::Sequence {
                    (span.input.end - span.input.start) as i32
                } else {
                    1
                },
                8,
            ]);
        }
        assert_eq!(
            projections
                .iter()
                .filter(|(name, _)| head(family, name))
                .map(|(_, shape)| shape.clone())
                .collect::<Vec<_>>(),
            heads,
            "only demanded decoder rows reach vocabulary"
        );
    }
    if audio {
        assert!(
            audio_observations(report) <= 1,
            "one actual producer traversal; a receiver need not re-observe its root"
        );
    }
}
fn audio_observations(report: &Report) -> usize {
    report
        .trace
        .observations
        .iter()
        .filter(|(p, shape)| {
            p == eredu_core::AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH && shape == &vec![1, 2, 8]
        })
        .count()
}
fn local_matrix(family: Family) {
    let c = config(family, false);
    let artifact = selected::fixture(&c);
    let combinations = if family == Family::Gemma {
        vec![(false, false), (true, false), (false, true), (true, true)]
    } else {
        vec![(false, false), (false, true)]
    };
    for (image, audio) in combinations {
        for residency in selected::residencies() {
            for output in [OutputDemand::LastPosition, OutputDemand::Sequence] {
                let reference = ordinary::run_with_input(
                    &c,
                    &artifact,
                    residency.clone(),
                    None,
                    false,
                    output,
                    input(family, image, audio),
                );
                for width in [1, 2, 3] {
                    for stepped in [false, true] {
                        let schedule = Schedule {
                            output,
                            chunk: width,
                            stepped,
                            cancel_after: None,
                            locally_cancel: false,
                            follow_decode: true,
                        };
                        let actual = ordinary::run_with_input(
                            &c,
                            &artifact,
                            residency.clone(),
                            Some(schedule),
                            false,
                            output,
                            input(family, image, audio),
                        );
                        assert_eq!(actual.outputs.len(), 4);
                        for (a, e) in actual.outputs.iter().zip(&reference.outputs) {
                            nonzero(a);
                            assert_tensor_close(
                                a,
                                e,
                                "optional media ordinary/full versus shared spans",
                            );
                        }
                        same_state(&actual.state, &reference.state);
                        let report = actual.media.as_ref().unwrap();
                        trace(family, report, schedule, true, true, image, audio);
                        if image || (family == Family::Gemma && audio) {
                            assert!(!encoders(family, &report.trace.projections[0]).is_empty());
                        }
                        if audio {
                            assert_eq!(audio_observations(report), 1);
                        }
                        assert_eq!(
                            report.output.as_ref().unwrap().shape,
                            [
                                1,
                                if output == OutputDemand::Sequence {
                                    report.input_positions as i32
                                } else {
                                    1
                                },
                                19
                            ]
                        );
                        if !image && !audio && width == 2 {
                            assert_eq!(
                                report
                                    .trace
                                    .delivered
                                    .iter()
                                    .map(|c| c.input.end - c.input.start)
                                    .collect::<Vec<_>>(),
                                [2, 2, 1]
                            );
                        }
                    }
                }
            }
        }
    }
}
fn partition_case(
    family: Family,
    c: &serde_json::Value,
    artifact: &tempfile::TempDir,
    topology: ParallelTopology,
    residency: eredu_core::ResidencyPlan,
    image: bool,
    audio: bool,
    stepped: bool,
    cancel_after: Option<u64>,
) -> Vec<Report> {
    partition_case_with_input(
        family,
        c,
        artifact,
        topology,
        residency,
        image,
        audio,
        stepped,
        cancel_after,
        input(family, image, audio),
        1,
    )
}
fn assert_inkling_vision_final_norm_binding(
    family: Family,
    config: &serde_json::Value,
    topology: ParallelRankTopology,
) {
    if family != Family::Inkling || config.get("vision_config").is_none() {
        return;
    }
    let units = config["vision_config"]["num_hidden_layers"]
        .as_u64()
        .unwrap() as usize;
    assert_eq!(units, 4, "this fixture retains every released hMLP stage");
    let last_owner = units.min(topology.pipeline_parallel_size()) - 1;
    REFERENCE_STAGE_EVIDENCE.with(|evidence| {
        let evidence = evidence.borrow();
        let bound = evidence.bound_parameters.get("visual.final_norm.weight");
        assert_eq!(
            bound.is_some(),
            topology.pipeline_parallel_rank() == last_owner,
            "actual selected checkpoint final norm belongs only to the last hMLP owner"
        );
        if let Some((shape, bits)) = bound {
            assert_eq!(shape, &[8]);
            let spec = ParameterSpec::trainable("model.visual.final_norm.weight").unwrap();
            let expected = deterministic_values(&spec, 8, true);
            assert_finite_values(&expected, "checkpoint final norm");
            assert_eq!(
                *bits,
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "last owner binds the original checkpoint values"
            );
        }
    });
}

fn partition_case_with_input(
    family: Family,
    c: &serde_json::Value,
    artifact: &tempfile::TempDir,
    topology: ParallelTopology,
    residency: eredu_core::ResidencyPlan,
    image: bool,
    audio: bool,
    stepped: bool,
    cancel_after: Option<u64>,
    input: Input,
    vision_rows: i32,
) -> Vec<Report> {
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let description = numeric_composite_parameter_description(c);
    let plan = prepared_adapter::plan(None)
        .with_topology(topology)
        .with_residency(residency);
    let world = Arc::new(NumericPartitionWorld::default());
    let reports: Vec<Report> = std::thread::scope(|scope| {
        let workers=(0..topology.world_size()).map(|rank|{let(inspection,description,plan,input)=(&inspection,&description,&plan,&input);let world=Arc::clone(&world);std::thread::Builder::new().name(format!("reference-optional-media-{rank}")).stack_size(32 * 1024 * 1024).spawn_scoped(scope, move||{
            let rank_topology=ParallelRankTopology::new(topology,rank).unwrap();let layout=eredu_architectures::partitioned_execution::derive_partitioned_local_layout(description,rank_topology).unwrap();let mut context=NumericContext::with_partition(layout,rank,world);context.bind_checkpoint_values=true;
            let prepare=||{let p=partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(inspection,plan,rank,std::time::Duration::from_secs(30),None,16).unwrap();assert!(matches!((plan.residency(),p.selected().text_realization().residency()),(eredu_core::ResidencyPlan::FullyResident,LayerWeightResidency::FullyResident)|(eredu_core::ResidencyPlan::LayerwiseHost{..},LayerWeightResidency::LayerwiseHost(_))|(eredu_core::ResidencyPlan::DenseDiskStream{..},LayerWeightResidency::DenseDiskStream(_))));p};
            let mut reference=partitioned_adapter::composite(prepare(),&context).unwrap();
            assert_inkling_vision_final_norm_binding(family,c,rank_topology);
            let expected=match cancel_after{
                Some(2)=>Some(reference.forward(&numeric_text_prepared_input(&[1,2]),true).unwrap()),
                Some(4) if !image&&!audio=>Some(reference.forward(&numeric_text_prepared_input(&[1,2,4,3]),true).unwrap()),
                Some(4)=>{let p=reference.media_prefill.as_mut().unwrap()(input,Schedule{output:OutputDemand::LastPosition,chunk:4,stepped:false,cancel_after:Some(4),locally_cancel:rank==0,follow_decode:false}).unwrap();assert_eq!(p.outcome,PrefillOutcome::Cancelled);None},
                None=>Some(reference.forward(input,true).unwrap()),_=>unreachable!()};
            let before=reference.snapshot().unwrap();let original_encoders=encoders(family,&context.projections.lock().unwrap());
            let mut actual=partitioned_adapter::composite(prepare(),&context).unwrap();context.projections.lock().unwrap().clear();context.media_completions.lock().unwrap().clear();
            let schedule=Schedule{output:OutputDemand::LastPosition,chunk:2,stepped,cancel_after,locally_cancel:rank==0,follow_decode:true};
            let report=actual.media_prefill.as_mut().unwrap()(input,schedule).unwrap();trace_with_vision_rows(family,&report,schedule,rank_topology.owns_output_head(),rank_topology.owns_embedding(),image,audio,vision_rows);same_state(&report.state,&before);
            if cancel_after!=Some(2){same_encoder_trace(family,&encoders(family,&report.trace.projections[0]),&original_encoders);}
            if let Some(output)=&report.output{nonzero(output);assert_tensor_close(output,expected.as_ref().unwrap(),"optional-media selected ordinary versus shared");}
            assert_eq!(report.cached.len(),3);for(token,output)in [2,6,1].into_iter().zip(&report.cached){let expected=reference.forward(&numeric_text_prepared_input(&[token]),false).unwrap();nonzero(output);assert_tensor_close(output,&expected,"optional-media cancelled-prefix cached state");}same_state(&report.final_state,&reference.snapshot().unwrap());report
        }).expect("spawn reference optional-media rank worker")}).collect::<Vec<_>>();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    if image || (family == Family::Gemma && audio) {
        assert!(
            reports
                .iter()
                .any(|r| !encoders(family, &r.trace.projections[0]).is_empty()),
            "actual selected encoder owners must execute"
        );
    }
    if audio {
        assert!(
            reports.iter().any(|r| audio_observations(r) == 1),
            "actual audio producer is observed"
        );
    }
    reports
}

fn parallel_matrix(family: Family) {
    let c = config(family, false);
    let artifact = selected::fixture(&c);
    let combinations = if family == Family::Gemma {
        vec![(false, false), (true, false), (false, true), (true, true)]
    } else {
        vec![(false, false), (false, true)]
    };
    for (image, audio) in combinations {
        for (tp, pp) in [(2, 1), (1, 2), (2, 2)] {
            for residency in selected::residencies() {
                for cancelled in [None, Some(2), Some(4)] {
                    let topology = ParallelTopology::new(tp, pp, 1, 1).unwrap();
                    let run = partition_case(
                        family,
                        &c,
                        &artifact,
                        topology,
                        residency.clone(),
                        image,
                        audio,
                        false,
                        cancelled,
                    );
                    let step = partition_case(
                        family,
                        &c,
                        &artifact,
                        topology,
                        residency.clone(),
                        image,
                        audio,
                        true,
                        cancelled,
                    );
                    for (run, step) in run.iter().zip(&step) {
                        assert_eq!(run.trace, step.trace);
                        same_state(&run.state, &step.state);
                        same_state(&run.final_state, &step.final_state);
                        match (&run.output, &step.output) {
                            (Some(a), Some(b)) => {
                                assert_tensor_close(a, b, "optional root run/step")
                            }
                            (None, None) => (),
                            _ => panic!("output presence"),
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn gemma_optional_roots_keep_shared_kv_per_layer_identity_and_full_rows_across_residencies() {
    local_matrix(Family::Gemma);
}
#[test]
fn gemma_optional_roots_match_selected_tp_pp_cancellation_and_controlled_steps() {
    parallel_matrix(Family::Gemma);
}
#[test]
fn inkling_zero_unit_audio_and_inactive_vision_keep_full_state_across_residencies() {
    local_matrix(Family::Inkling);
}
#[test]
fn inkling_zero_unit_audio_matches_selected_tp_pp_cancellation_and_controlled_steps() {
    parallel_matrix(Family::Inkling);
}
#[test]
fn inkling_released_hmlp_shape_keeps_nonzero_retained_media_in_bounded_residencies_and_pp() {
    let mut c = config(Family::Inkling, true);
    // Four real decoder units satisfy the existing one-unit-per-PP-owner rule.
    // The exact released hMLP tower and its fixed 8192×4800 matrix are unchanged.
    c["text_config"]["num_hidden_layers"] = serde_json::json!(4);
    c["text_config"]["layer_types"] = serde_json::json!([
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention"
    ]);
    c["text_config"]["mlp_layer_types"] = serde_json::json!(["dense", "dense", "dense", "dense"]);
    let artifact = selected::fixture(&c);
    // The exact fixed 8192×4800 matrix is retained. These are deliberately real
    // 512 MiB fixture limits, not claims that the released tower fits 1 MiB.
    let residencies = [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(512 << 20),
            host_budget_bytes: Some(512 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 512 << 20,
            host_budget_bytes: 512 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ];
    for residency in &residencies {
        let reference = ordinary::run_with_input(
            &c,
            &artifact,
            residency.clone(),
            None,
            false,
            OutputDemand::LastPosition,
            input(Family::Inkling, true, true),
        );
        for stepped in [false, true] {
            let schedule = Schedule {
                output: OutputDemand::LastPosition,
                chunk: 2,
                stepped,
                cancel_after: None,
                locally_cancel: false,
                follow_decode: true,
            };
            let actual = ordinary::run_with_input(
                &c,
                &artifact,
                residency.clone(),
                Some(schedule),
                false,
                OutputDemand::LastPosition,
                input(Family::Inkling, true, true),
            );
            assert_eq!(actual.outputs.len(), 4);
            assert_eq!(reference.outputs.len(), 4);
            for (a, e) in actual.outputs.iter().zip(&reference.outputs) {
                nonzero(a);
                assert_tensor_close(a, e, "real hMLP ordinary versus retained spans");
            }
            same_state(&actual.state, &reference.state);
            trace(
                Family::Inkling,
                actual.media.as_ref().unwrap(),
                schedule,
                true,
                true,
                true,
                true,
            );
        }
    }
    for pp in [2, 4] {
        for stepped in [false, true] {
            let _ = partition_case(
                Family::Inkling,
                &c,
                &artifact,
                ParallelTopology::new(1, pp, 1, 1).unwrap(),
                residencies[1].clone(),
                true,
                false,
                stepped,
                Some(4),
            );
        }
    }
    // Exercise the batch/patch distinction with an independent ordinary path,
    // then all three hMLP cuts, without reinterpreting the input or its ordering.
    let reference = ordinary::run_with_input(
        &c,
        &artifact,
        residencies[1].clone(),
        None,
        false,
        OutputDemand::LastPosition,
        inkling_image_first_input(),
    );
    for stepped in [false, true] {
        let schedule = Schedule {
            output: OutputDemand::LastPosition,
            chunk: 2,
            stepped,
            cancel_after: None,
            locally_cancel: false,
            follow_decode: true,
        };
        let actual = ordinary::run_with_input(
            &c,
            &artifact,
            residencies[1].clone(),
            Some(schedule),
            false,
            OutputDemand::LastPosition,
            inkling_image_first_input(),
        );
        assert_eq!(actual.outputs.len(), 4);
        assert_eq!(reference.outputs.len(), 4);
        for (a, e) in actual.outputs.iter().zip(&reference.outputs) {
            nonzero(a);
            assert_tensor_close(
                a,
                e,
                "two raw image-first patches retain exact decoder batch/order",
            );
        }
        same_state(&actual.state, &reference.state);
        trace_with_vision_rows(
            Family::Inkling,
            actual.media.as_ref().unwrap(),
            schedule,
            true,
            true,
            true,
            false,
            2,
        );
        let _ = partition_case_with_input(
            Family::Inkling,
            &c,
            &artifact,
            ParallelTopology::new(1, 4, 1, 1).unwrap(),
            residencies[1].clone(),
            true,
            false,
            stepped,
            None,
            inkling_image_first_input(),
            2,
        );
    }
}
