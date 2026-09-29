//! Independent MTP role joined with both container targets, through shared sessions.
use super::executor::{Materializer, SourceBinding};
use super::*;
use eredu_architectures::prediction_extension::MaterializedPredictionExecutor;
use eredu_architectures::qwen4_exp::prepared::{PreparationError, PreparedPredictionSource};
use eredu_runtime::*;

fn run(target: &PreparedTarget, residency: LayerWeightResidency) -> Vec<NumericTensor> {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let p = target.prediction(limits()).unwrap();
    let plan = super::super::gguf::plan(target)
        .with_prediction(limits(), super::selected::streams(p.spec()))
        .unwrap();
    let capabilities =
        super::super::cold::capabilities(plan.requirements(), None).with_indexed_movement(true);
    let state_capabilities = super::selected::state_capabilities(&plan);
    let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
            .with_session(eredu_core::SessionCapabilities::new(false, true, true))
            .with_exact_completion(true),
        WeightResidency::with_layers(residency),
    )
    .unwrap();
    let before = target
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let selected = plan
        .select(&request, &capabilities, Some(&state_capabilities))
        .unwrap();
    let joint = selected
        .clone()
        .prepare_prediction_execution::<NumericBackend, State>(&ctx)
        .unwrap();
    super::installed::assert_joint_sources(
        &selected,
        target.artifact(),
        joint.target_source(),
        joint.provider_source(),
    );
    assert_eq!(
        target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        before
    );
    let banks = joint.prediction_banks();
    let (target_handoff, prediction_handoff, source, provider_source) = joint.into_parts();
    let mut binding = SourceBinding {
        source: provider_source,
        context: &ctx,
        residency,
        roles: vec![],
    };
    let (mut session, provider) =
        super::installed::bind_target(target_handoff, source, residency, &banks, &ctx);
    let mut executor = prediction_handoff
        .materialize_executor::<Materializer, _>(&mut binding, provider.unwrap(), |_, state| {
            super::installed::realize(state)
        })
        .unwrap();
    let mut lane = <super::installed::Executor as MaterializedPredictionExecutor<
        TargetModel<NumericBackend>,
        NumericBackend,
        Materializer,
    >>::new_state(&executor);
    let mut outputs = Vec::new();
    let ids = [3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut offset = 0;
    for length in [1, 2, 4, 3, 9]
        .into_iter()
        .chain(std::iter::repeat_n(1, 16))
    {
        let token = if offset < ids.len() {
            NumericTensor::token_ids(&ids[offset..offset + length])
        } else {
            NumericTensor::token_ids(&[(offset - ids.len()) % 16])
        };
        let (logits, capture) = if offset < ids.len() {
            session
                .prefill_prediction_target(&token, None, &ctx)
                .unwrap()
        } else {
            session.decode_prediction_target(&token, &ctx).unwrap()
        };
        outputs.push(logits);
        let saved_target = session.checkpoint(&ctx).unwrap();
        let saved_lane = lane.clone();
        let mut invoke = |lane: &mut super::executor::Lane| {
            executor
                .logits::<State, _>(
                    &mut super::installed::Invoker {
                        session: &mut session,
                        ctx: &ctx,
                    },
                    &capture,
                    &token,
                    0,
                    lane,
                )
                .unwrap()
        };
        let actual = invoke(&mut lane);
        let mut replay = saved_lane;
        let repeated = invoke(&mut replay);
        assert_tensor_exact(
            &actual.0,
            &repeated.0,
            "separate prediction rollback logits",
        );
        assert_tensor_exact(
            &actual.1,
            &repeated.1,
            "separate prediction rollback capture",
        );
        super::super::session::assert_checkpoint(&lane.0, &replay.0);
        super::super::session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &saved_target);
        outputs.extend([actual.0, actual.1]);
        offset += length;
    }
    outputs
}

#[test]
fn separate_prediction_source_matches_both_containers_and_bounded_sessions() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(separate_source_case)
        .unwrap()
        .join()
        .unwrap();
}
fn separate_source_case() {
    let (dir, gguf, st) = super::super::gguf::fixtures(false);
    let mut config = eredu_architectures::qwen4_exp::config::Config::from_gguf(
        &eredu_gguf::Checkpoint::open(dir.path().join("target.gguf")).unwrap(),
    )
    .unwrap();
    config.prediction = Some(PredictionGeometry {
        layers: eredu_core::LayerSchedule::new(1, vec![LayerKind::Indexed]).unwrap(),
        rope_theta: 7777.,
    });
    let ctx = NumericContext::default();
    let spec = spec(&config);
    let mut parameters = Parameters::default();
    PredictionShared::<NumericBackend>::new(&spec, &ctx)
        .unwrap()
        .visit_parameters_mut(&mut parameters);
    PredictionUnit::<NumericBackend>::new(spec.units[0].clone(), &ctx)
        .unwrap()
        .visit_parameters_mut(&mut parameters);
    let values: Vec<_> = parameters
        .0
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                safetensors::Dtype::F32,
                value.shape.iter().map(|n| *n as usize).collect::<Vec<_>>(),
                value
                    .data
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let (_mtp_dir, source) = super::super::transforms::fixture(&values);
    let encoding = || SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap();
    let prediction =
        PreparedPredictionSource::safetensors(source.clone(), config.clone(), encoding()).unwrap();
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    assert!(prediction
        .artifact()
        .source_keys()
        .iter()
        .all(|n| n.starts_with("mtp.")));
    assert!(matches!(
        gguf.prediction(limits()),
        Err(PreparationError::MissingPrediction)
    ));
    let mut wrong = config.clone();
    wrong.vocabulary += 1;
    let wrong = PreparedPredictionSource::safetensors(source.clone(), wrong, encoding()).unwrap();
    assert!(matches!(
        gguf.clone().with_prediction_source(wrong),
        Err(PreparationError::PredictionMismatch {
            field: "vocabulary"
        })
    ));
    let mut wrong = config.clone();
    wrong.attention.rotary.base *= 2.;
    let wrong = PreparedPredictionSource::safetensors(source.clone(), wrong, encoding()).unwrap();
    assert!(matches!(
        gguf.clone().with_prediction_source(wrong),
        Err(PreparationError::PredictionMismatch { field: "attention" })
    ));
    let identity = gguf.bound_spec().unwrap().state_fingerprint().to_owned();
    let target_source = gguf.artifact().clone();
    let joined = gguf.with_prediction_source(prediction.clone()).unwrap();
    assert_eq!(joined.bound_spec().unwrap().state_fingerprint(), identity);
    assert!(matches!(
        joined.clone().with_prediction_source(prediction.clone()),
        Err(PreparationError::PredictionAlreadyPresent)
    ));
    assert_eq!(
        joined.artifact().source_diagnostics().unwrap().backend,
        eredu_checkpoint::store::WeightStoreBackend::Composite
    );
    for name in source.source_keys() {
        assert_eq!(
            joined.artifact().source_provenance(&name).unwrap(),
            source.source_provenance(&name).unwrap()
        );
    }
    let owner = joined.prediction(limits()).unwrap();
    assert!(owner
        .unit(0)
        .unwrap()
        .source()
        .source_metadata("token_embd.weight")
        .is_err());
    for (name, recipe) in owner.vocabulary().recipes() {
        assert_eq!(recipe, &joined.static_parameters().recipes()[name]);
    }
    let reference = run(
        &st.with_prediction_source(prediction).unwrap(),
        LayerWeightResidency::FullyResident,
    );
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let actual = run(&joined, residency);
        assert_eq!(actual.len(), reference.len());
        for (a, b) in actual.iter().zip(&reference) {
            assert_tensor_close(a, b, "separate GGUF prediction");
        }
    }
    assert_eq!(
        joined
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        target_source.source_diagnostics().unwrap().physical_reads
            + source.source_diagnostics().unwrap().physical_reads
    );
    let mut missing = values.clone();
    missing.retain(|(n, ..)| n != "mtp.fc_hidden.weight");
    let (_dir, bad) = super::super::transforms::fixture(&missing);
    assert!(PreparedPredictionSource::safetensors(bad, config.clone(), encoding()).is_err());
    let mut malformed = values;
    malformed
        .iter_mut()
        .find(|(n, ..)| n == "mtp.fc_hidden.weight")
        .unwrap()
        .2 = vec![16, 64];
    let (_dir, bad) = super::super::transforms::fixture(&malformed);
    assert!(PreparedPredictionSource::safetensors(bad, config, encoding()).is_err());
}
