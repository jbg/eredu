//! Fused cache preparation, proposal masks and dynamic score addition.
use super::*;

#[test]
fn dspark_multiblock_observed_execution_reconstructs_scores_and_preserves_context() {
    let mut config = tiny_v4_config();
    config["num_nextn_predict_layers"] = 2.into();
    config["compress_ratios"] = serde_json::json!([0, 4, 128, 0, 0]);
    config["dspark_block_size"] = 3.into();
    config["dspark_noise_token_id"] = 0.into();
    config["dspark_target_layer_ids"] = serde_json::json!([0, 2]);
    config["dspark_markov_rank"] = 3.into();
    let args = deepseek::parse_v4_config(&config).unwrap();
    let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let scope = &graph.component_scopes[0];
    let context = NumericContext::default();
    let parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let strategy =
        eredu_architectures::prediction_extension::DsparkPredictionStrategy::from_args(&args)
            .unwrap();
    let make = || {
        let source = deepseek::v4::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
        let target =
            deepseek::v4::Model::<NumericBackend>::new(args.prediction_target().unwrap(), &context)
                .unwrap();
        let pinned = source.static_modules().dspark.as_ref().unwrap().clone();
        let units: Vec<_> = (0..2)
            .map(|depth| Box::new(source.construct_unit(depth + 1, 0, &context).unwrap()))
            .collect();
        let caches: Vec<_> = (0..2)
            .map(|_| NumericPoolingCache::new(args.sliding_window, &[]))
            .collect();
        (target, pinned, units, caches)
    };
    let (model, pinned, units, _) = make();
    let mut parameters = Parameters::default();
    model.static_modules().visit_parameters(&mut parameters);
    pinned.visit_parameters(&mut parameters);
    for unit in units {
        unit.visit_parameters(&mut parameters);
    }
    for tensor_parallel in [false, true] {
        let parallel = tensor_parallel.then_some(&parallel);
        let mut survivors = Vec::new();
        for mode in 0..6 {
            let (mut model, mut pinned, mut units, mut caches) = make();
            let (mut ordinary, mut ordinary_pinned, mut ordinary_units, mut ordinary_caches) =
                make();
            for (step, rows) in [4, 1, 1].into_iter().enumerate() {
                let captures = NumericTensor::new(
                    [1, rows, 8],
                    (0..rows * 8)
                        .map(|i| ((i * 11 + step as i32 * 7) % 29) as f32 * 0.05 - 0.7)
                        .collect(),
                );
                let mut context_capture = Components {
                    zero: (mode == 1 && step == 0).then_some(("dspark.context.normalized", 1, 1)),
                    ..Components::strict()
                };
                model
                    .pipeline_prefill_dspark_extension_context_observed(
                        &strategy,
                        &mut pinned,
                        &mut units,
                        &captures,
                        &mut caches,
                        &context,
                        Some(&mut context_capture),
                    )
                    .unwrap();
                ordinary
                    .pipeline_prefill_dspark_extension_context(
                        &strategy,
                        &mut ordinary_pinned,
                        &mut ordinary_units,
                        &captures,
                        &mut ordinary_caches,
                        &context,
                    )
                    .unwrap();
                for path in context_capture.values.keys() {
                    let point = graph.observations.get(path).unwrap();
                    assert_eq!(
                        eredu_architectures::speculative_execution::speculative_capture_scope(
                            &graph,
                            &point.node_id
                        )
                        .unwrap(),
                        eredu_core::speculative::SpeculativeCaptureScope::PredictionContext
                    );
                }
                let anchor = NumericTensor::token_ids(&[step + 1]);
                let snapshot = caches.clone();
                let offsets: Vec<_> = caches.iter().map(|cache| cache.offset()).collect();
                let mut proposal_state = snapshot.clone();
                let mut capture = Components {
                    zero: match mode {
                        2 | 5 => Some((
                            "dspark.proposal.layers.0.compressed_attention.channels",
                            1,
                            1,
                        )),
                        4 => Some(("dspark.proposal.markov.input", 0, 1)),
                        _ => None,
                    },
                    keep: matches!(mode, 3 | 5).then_some((
                        "dspark.proposal.layers.0.feed_forward.shared.units",
                        1,
                        0,
                    )),
                    ..Components::strict()
                };
                let actual = model
                    .pipeline_dspark_extension_proposal_observed(
                        &strategy,
                        &mut pinned,
                        &mut units,
                        &anchor,
                        3,
                        &mut proposal_state,
                        parallel,
                        &context,
                        Some(&mut capture),
                    )
                    .unwrap();
                let expected = ordinary
                    .pipeline_dspark_extension_proposal_observed(
                        &strategy,
                        &mut ordinary_pinned,
                        &mut ordinary_units,
                        &anchor,
                        3,
                        &mut ordinary_caches.clone(),
                        parallel,
                        &context,
                        None,
                    )
                    .unwrap();
                if mode == 0 {
                    assert_tensor_exact(
                        &actual,
                        &expected,
                        "observed fused proposal equals ordinary",
                    );
                } else if step == 0 {
                    assert_ne!(
                        actual.data, expected.data,
                        "causal fused/context edit {mode}"
                    );
                }
                assert_eq!(
                    caches
                        .iter()
                        .map(|cache| cache.offset())
                        .collect::<Vec<_>>(),
                    offsets
                );
                let mut replay = Components {
                    zero: capture.zero,
                    keep: capture.keep,
                    ..Components::strict()
                };
                let repeated = model
                    .pipeline_dspark_extension_proposal_observed(
                        &strategy,
                        &mut pinned,
                        &mut units,
                        &anchor,
                        3,
                        &mut snapshot.clone(),
                        parallel,
                        &context,
                        Some(&mut replay),
                    )
                    .unwrap();
                assert_tensor_exact(&actual, &repeated, "temporary proposal state replay");
                for path in capture.values.keys() {
                    let point = graph.observations.get(path).unwrap();
                    assert_eq!(
                        eredu_architectures::speculative_execution::speculative_capture_scope(
                            &graph,
                            &point.node_id
                        )
                        .unwrap(),
                        eredu_core::speculative::SpeculativeCaptureScope::FusedProposal
                    );
                }
                for (selection, keep) in [(capture.zero, false), (capture.keep, true)] {
                    if let Some((path, row, component)) = selection {
                        let before = &capture.values[path];
                        let after = &capture.values[&format!("{path}.effective")];
                        let width = *before.shape.last().unwrap() as usize;
                        for i in 0..before.data.len() {
                            let changed = i / width == row && (i % width == component) != keep;
                            assert_eq!(after.data[i], if changed { 0.0 } else { before.data[i] });
                        }
                    }
                }
                prediction::reconstruct_scope(scope, &capture, &parameters, &actual);
                if step == 0 && matches!(mode, 3 | 5) {
                    survivors.push(
                        capture.values
                            ["dspark.proposal.layers.0.feed_forward.shared.units.effective"]
                            .data[args.moe_intermediate_size as usize],
                    );
                }
                assert!(capture.values[&scope.readout.score_writes[0].output]
                    .data
                    .iter()
                    .any(|v| v.abs() > 1e-6));
            }
        }
        assert_ne!(
            survivors[0], survivors[1],
            "surviving shared unit is recomputed after channel removal"
        );
    }
}
