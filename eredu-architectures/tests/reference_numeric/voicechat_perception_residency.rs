//! Pinned perception equations through ordinary resident and leased-unit execution.
use super::*;
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    conformer::ConformerDimensions,
    perception::{Perception, PerceptionDimensions, PerceptionState},
    perception_layered::{
        LayerwisePerception, PerceptionArchitecture, PerceptionExecution, PerceptionUnit,
    },
    perception_rnnt::{PerceptionRnnt, PerceptionRnntLimits, PerceptionRnntState},
    rnnt::{Rnnt, RnntDimensions, RnntExecution, RnntLimits},
    rnnt_layered::{LayerwiseRnnt, RnntArchitecture},
    silence::PreparedSilenceConditioning,
    subsampling::SubsamplingDimensions,
};
use serde_json::Value;

pub(super) fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
}

pub(super) fn dimensions(fixture: &Value) -> PerceptionDimensions {
    let integer =
        |part: &str, field: &str| fixture[part]["geometry"][field].as_i64().unwrap() as i32;
    PerceptionDimensions {
        subsampling: SubsamplingDimensions {
            features: integer("subsampling", "features"),
            channels: integer("subsampling", "channels"),
            width: integer("subsampling", "width"),
        },
        block: ConformerDimensions {
            width: integer("conformer", "width"),
            heads: integer("conformer", "heads"),
            ff_hidden: integer("conformer", "feed_forward"),
            conv_kernel: integer("conformer", "kernel"),
            left_context: integer("conformer", "left_context"),
        },
        layers: 1,
        output_width: fixture["perception"]["output_width"].as_i64().unwrap() as i32,
    }
}

pub(super) fn parameters(fixture: &Value) -> BTreeMap<String, NumericTensor> {
    ["subsampling", "conformer", "perception"]
        .into_iter()
        .flat_map(|part| {
            fixture[part]["parameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| (name.clone(), tensor(value)))
        })
        .collect()
}

// The existing rebuilding policy owns acquisition, completion and failure. This
// adapter only installs independent fixture values into each newly built unit.
struct BoundFixturePolicy {
    inner: RebuildingUnitPolicy,
    parameters: BTreeMap<String, NumericTensor>,
}

type Unit = PerceptionUnit<NumericBackend>;
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

fn bounded(fixture: &Value) -> LayerwisePerception<NumericBackend, BoundFixturePolicy> {
    LayerwisePerception::new(
        PerceptionArchitecture::new(dimensions(fixture)).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters: parameters(fixture),
        },
    )
}

pub(super) fn assert_tensor(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual.data.iter().zip(expected.data).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "perception element {index}: {actual} != NVIDIA {expected}"
        );
    }
}

pub(super) fn assert_state(
    state: &PerceptionState<NumericTensor>,
    expected: &Value,
    frames: u64,
    fixture: &Value,
    context: &NumericContext,
) {
    assert_eq!(state.frames(), frames);
    assert_eq!(state.blocks().len(), 1);
    let cache = &state.blocks()[0];
    assert_eq!(
        cache.valid_channel(),
        expected["valid_cache"].as_u64().unwrap() as usize
    );
    assert_tensor(
        cache.channel().unwrap(),
        &expected["channel_cache"],
        fixture,
    );
    let temporal = cache
        .convolution()
        .unwrap()
        .transpose_axes(&[0, 2, 1], context)
        .unwrap();
    assert_tensor(&temporal, &expected["time_cache"], fixture);
}

fn exercise(
    model: &mut impl PerceptionExecution<NumericBackend>,
    fixture: &Value,
    context: &NumericContext,
) {
    let windows = fixture["subsampling"]["steps"].as_array().unwrap();
    let expected = fixture["perception"]["steps"].as_array().unwrap();
    let initial = PerceptionState::default();
    let full = model
        .forward(
            &tensor(&fixture["subsampling"]["input"]),
            0,
            &initial,
            context,
        )
        .unwrap();
    assert_tensor(
        &full.encoder,
        &fixture["perception"]["encoder_output"],
        fixture,
    );
    assert_tensor(
        &full.projected,
        &fixture["perception"]["projected_output"],
        fixture,
    );
    assert_state(&full.state, expected.last().unwrap(), 6, fixture, context);
    let mut previous = initial;
    for (index, (window, expected)) in windows.iter().zip(expected).enumerate() {
        let next = model
            .forward(
                &tensor(&window["input"]),
                window["drop"].as_u64().unwrap() as usize,
                &previous,
                context,
            )
            .unwrap();
        assert_tensor(&next.encoder, &expected["encoder_output"], fixture);
        assert_tensor(&next.projected, &expected["projected_output"], fixture);
        assert_state(&next.state, expected, index as u64 + 1, fixture, context);
        assert_eq!(previous.frames(), index as u64);
        previous = next.state;
    }
}

fn successful_events() -> Vec<StreamPolicyEvent> {
    vec![
        StreamPolicyEvent::Begin,
        StreamPolicyEvent::Acquire(0, 0, 0),
        StreamPolicyEvent::Complete(0, 0, 0),
        StreamPolicyEvent::Acquire(1, 1, 0),
        StreamPolicyEvent::Complete(1, 1, 0),
        StreamPolicyEvent::Acquire(2, 2, 0),
        StreamPolicyEvent::Complete(2, 2, 0),
        StreamPolicyEvent::Finish,
    ]
}

#[test]
fn voicechat_perception_resident_and_bounded_units_match_pinned_full_and_cached_state() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let weights = parameters(&fixture);
    let mut ordinary = Perception::new(dimensions(&fixture), &context).unwrap();
    replace_numeric_parameters(&mut ordinary, &weights);
    exercise(&mut ordinary, &fixture, &context);

    let architecture = PerceptionArchitecture::<NumericBackend>::new(dimensions(&fixture)).unwrap();
    let units = (0..3)
        .map(|group| {
            let mut unit = architecture.build_unit(group, 0, &context).unwrap();
            replace_numeric_parameters(&mut unit, &weights);
            unit
        })
        .collect();
    let mut resident =
        LayerwisePerception::new(architecture, NumericReplicatedPolicy::resident(units));
    exercise(&mut resident, &fixture, &context);
    let mut streamed = bounded(&fixture);
    exercise(&mut streamed, &fixture, &context);
    assert_eq!(
        streamed.runtime().policy().inner.events,
        successful_events().repeat(7)
    );
}

#[test]
fn voicechat_perception_failed_projection_acquisition_preserves_parent_and_retries() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let windows = fixture["subsampling"]["steps"].as_array().unwrap();
    let expected = fixture["perception"]["steps"].as_array().unwrap();
    let mut streamed = bounded(&fixture);
    let prefix = streamed
        .forward(
            &tensor(&windows[0]["input"]),
            0,
            &PerceptionState::default(),
            &context,
        )
        .unwrap()
        .state;
    streamed.runtime_mut().policy_mut().inner.events.clear();
    streamed.runtime_mut().policy_mut().inner.fail_acquire = Some(2);
    let input = tensor(&windows[1]["input"]);
    let drop = windows[1]["drop"].as_u64().unwrap() as usize;
    let failure = streamed
        .forward(&input, drop, &prefix, &context)
        .unwrap_err();
    assert!(failure
        .to_string()
        .contains("injected dense-stream acquisition failure"));
    assert_eq!(
        streamed.runtime().policy().inner.events,
        successful_events()[..6]
    );
    assert_state(&prefix, &expected[0], 1, &fixture, &context);
    streamed.runtime_mut().policy_mut().inner.fail_acquire = None;
    let retry = streamed.forward(&input, drop, &prefix, &context).unwrap();
    assert_tensor(&retry.projected, &expected[1]["projected_output"], &fixture);
    assert_state(&retry.state, &expected[1], 2, &fixture, &context);
    assert_state(&prefix, &expected[0], 1, &fixture, &context);
}

#[test]
fn voicechat_startup_silence_uses_the_same_bounded_perception_units() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TOOL_SILENCE_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let prepared = PreparedSilenceConditioning::with_coefficients(
        &config.perception.preprocessor,
        tensor(&fixture["preprocessing"]["window"]).data,
        tensor(&fixture["preprocessing"]["filters"]).data,
    )
    .unwrap();
    let mut streamed = bounded(&fixture);
    let actual = prepared
        .execute(&mut streamed, eredu_nn::TensorElementType::Bf16, &context)
        .unwrap();
    assert_eq!(actual.shape(), &[1, 1, 6]);
    assert_eq!(
        actual.data,
        tensor(&fixture["perception"]["bf16_embedding"]).data
    );
    assert_eq!(
        streamed.runtime().policy().inner.events,
        successful_events()
    );
}

fn transcription(context: &NumericContext) -> Rnnt<NumericBackend> {
    struct Bind;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind {
        fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut NumericTensor) {
            // This analytic joint boundary at raw encoder coordinate1=1.12
            // separates the first two independently generated NVIDIA frames.
            let values = match metadata.id.as_str() {
                "stt_model.rnnt_joint.enc.weight" => vec![0., 1., 0., 0.],
                "stt_model.rnnt_joint.enc.bias" => vec![0.],
                "stt_model.rnnt_joint.pred.weight" | "stt_model.rnnt_joint.pred.bias" => vec![0.],
                "stt_model.rnnt_joint.joint_net.2.weight" => vec![1., 0.],
                "stt_model.rnnt_joint.joint_net.2.bias" => vec![-1.12, 0.],
                name if name.starts_with("stt_model.rnnt_decoder.") => vec![0.1; slot.data.len()],
                name => panic!("unexpected RNNT parameter {name}"),
            };
            *slot = NumericTensor::new(slot.shape(), values);
        }
    }
    let mut rnnt = Rnnt::new(
        RnntDimensions {
            vocab_size: 1,
            predictor_width: 1,
            predictor_layers: 1,
            encoder_width: 4,
            joint_width: 1,
        },
        context,
    )
    .unwrap();
    rnnt.visit_parameters_mut(&mut Bind);
    rnnt
}

#[test]
fn voicechat_bounded_perception_feeds_the_same_cached_rnnt_once_per_audio_frame() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mut direct = Perception::new(dimensions(&fixture), &context).unwrap();
    replace_numeric_parameters(&mut direct, &parameters(&fixture));
    let mut ordinary = PerceptionRnnt::from_parts(direct, transcription(&context)).unwrap();
    let rnnt = transcription(&context);
    struct Collect<'a>(&'a mut BTreeMap<String, NumericTensor>);
    impl<'a> ParameterVisitor<'a, NumericTensor> for Collect<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, slot: &'a NumericTensor) {
            self.0.insert(metadata.id.as_str().into(), slot.clone());
        }
    }
    let mut policy = NumericReplicatedPolicy::bounded();
    rnnt.visit_parameters(&mut Collect(&mut policy.parameter_overrides));
    let leased_rnnt = LayerwiseRnnt::new(RnntArchitecture::new(rnnt.dimensions()).unwrap(), policy);
    let mut streamed = PerceptionRnnt::from_parts(bounded(&fixture), leased_rnnt).unwrap();
    let mut ordinary_state = PerceptionRnntState::default();
    let mut streamed_state = PerceptionRnntState::default();
    for index in 0..2 {
        let window = &fixture["subsampling"]["steps"][index];
        let expected = &fixture["perception"]["steps"][index];
        let input = tensor(&window["input"]);
        let drop = window["drop"].as_u64().unwrap() as usize;
        let limits = PerceptionRnntLimits {
            frame_capacity: 2,
            rnnt: RnntLimits {
                max_symbols_per_frame: 2,
                hypothesis_capacity: 2,
            },
        };
        let direct = ordinary
            .advance(&input, drop, &ordinary_state, limits, &context)
            .unwrap();
        let bounded = streamed
            .advance(&input, drop, &streamed_state, limits, &context)
            .unwrap();
        assert_tensor(&bounded.encoder, &expected["encoder_output"], &fixture);
        assert_tensor(&bounded.projected, &expected["projected_output"], &fixture);
        assert_eq!(
            bounded.emitted,
            if index == 0 { vec![0, 0] } else { vec![] }
        );
        assert_eq!(bounded.emitted, direct.emitted);
        assert_eq!(bounded.state.rnnt().hypothesis(), &[0, 0]);
        assert_eq!(bounded.state.rnnt().frames(), index as u64 + 1);
        assert_eq!(
            (
                bounded.first_blank,
                bounded.ended_blank,
                bounded.hit_symbol_limit
            ),
            (index != 0, index != 0, index == 0)
        );
        assert_state(
            bounded.state.perception(),
            expected,
            index as u64 + 1,
            &fixture,
            &context,
        );
        for (actual, expected) in bounded
            .state
            .rnnt()
            .predictor()
            .unwrap()
            .local_state()
            .layers
            .iter()
            .zip(
                &direct
                    .state
                    .rnnt()
                    .predictor()
                    .unwrap()
                    .local_state()
                    .layers,
            )
        {
            assert_eq!(actual.hidden.data, expected.hidden.data);
            assert_eq!(actual.cell.data, expected.cell.data);
        }
        assert_eq!(streamed_state.frames(), index as u64);
        ordinary_state = direct.state;
        streamed_state = bounded.state;
    }
    let (perception, _) = streamed.into_parts();
    assert_eq!(
        perception.runtime().policy().inner.events,
        successful_events().repeat(2)
    );
}
