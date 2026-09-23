use super::*;
use eredu_runtime::execution_topology::TextExecutionTopology;
use eredu_runtime::memory_forecast::{SpeculativeForecastBackend, SpeculativeMemoryProfile};
use eredu_runtime::prediction_resources::{
    EmbeddedPredictionTopology, PredictionExecutionMode, PredictionStateLayer,
};
use eredu_runtime::{SpeculativeCaptureEntry, SpeculativeCaptureSchema, SpeculativeIdentity};

pub(super) struct EmbeddedFixture;

pub(super) fn topology() -> EmbeddedPredictionTopology {
    let projection = |name, input, output| {
        serde_json::json!({
            "input": input, "output": output, "format": {"kind":"dense"}, "bias":false, "parameter":name,
        })
    };
    let execution: TextExecutionTopology = serde_json::from_value(serde_json::json!({
        "hidden_size":32, "vocabulary_size":128, "missing":[],
        "output":projection("shared-output",32,128),
        "layers":[{
            "normalization_count":2,
            "mixer":{"kind":"attention","query_heads":4,"kv_heads":1,"key_width":8,"value_width":8,
                "input_scores":false,"softcap":false,"sinks":false,"query_key_normalization":false,"rotary":true,
                "projections":[projection("query",32,32),projection("key",32,8),projection("value",32,8),projection("out",32,32)]},
            "feed_forward":{"kind":"gated","intermediate_size":64,
                "projections":[projection("gate",32,64),projection("up",32,64),projection("down",64,32)]}
        }]
    })).unwrap();
    let id = |s| SpeculativeIdentity::new(s).unwrap();
    EmbeddedPredictionTopology {
        execution_topology: Some(execution),
        missing: vec![],
        mode: PredictionExecutionMode::Sequential,
        proposal_capacity: 4,
        nodes: vec![],
        edges: vec![],
        parameters: vec![],
        invocations: vec![],
        state: vec![PredictionStateLayer {
            layer: 0,
            policy: eredu_core::cache::LayerCachePolicy::key_only(
                eredu_core::AttentionPolicy::Full,
                1,
                8,
            )
            .unwrap(),
            processed_token_offset: -1,
        }],
        target_features: SpeculativeCaptureSchema::new(
            id("fixture"),
            [SpeculativeCaptureEntry::new(
                id("hidden"),
                vec![1, 1024, 32],
                id("target"),
                id("hidden-observation"),
            )
            .unwrap()
            .with_bounded_dimension(1)
            .unwrap()],
        )
        .unwrap(),
    }
}

impl SpeculativeForecastBackend<EmbeddedFixture> for MockBackend {
    fn speculative_memory_profile(
        _: &ModelRuntime<Self>,
        _: &eredu_core::SpeculativeDraft<'_, EmbeddedFixture>,
    ) -> Result<Option<SpeculativeMemoryProfile>, GenerationForecastError> {
        Ok(Some(SpeculativeMemoryProfile {
            embedded: Some(topology()),
            parameter_conversions: Some(vec![]),
            draft: None,
            auxiliary_bytes_per_position: MemoryBytes::exact(0),
            sampling_bytes_per_vocabulary_entry: MemoryBytes::estimated(
                0,
                128,
                "neutral fixture sampling",
            ),
            proposal_capacity: 4,
            shared_allocator: true,
        }))
    }
}

#[test]
fn embedded_facade_forecast_is_recomputable_and_retains_one_residency_owner() {
    let (model, _, settings) = setup();
    let forecast = model
        .forecast_speculative_token_ids(
            &[1; 17],
            settings,
            &eredu_core::SpeculativeDraft::<EmbeddedFixture>::Embedded,
            Default::default(),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    assert!(forecast.speculative.as_ref().unwrap().embedded.is_some());
    assert!(forecast.speculative.as_ref().unwrap().draft.is_none());
    let prediction = forecast
        .speculative
        .as_ref()
        .unwrap()
        .embedded
        .as_ref()
        .unwrap();
    assert_eq!(prediction.execution.state_layout.layer_layout().len(), 1);
    assert_eq!(
        prediction.execution.state_layout.layer_prefix_offsets(),
        [-1]
    );
    assert_eq!(
        forecast.request.domains[0].executions[0]
            .state_layout
            .layer_layout()
            .len(),
        2
    );
    assert_eq!(
        forecast.estimate.domains[0].phases[0]
            .parameters
            .lower_bytes,
        4096
    );
    let decoded: eredu::api::GenerationForecast =
        serde_json::from_str(&serde_json::to_string(&forecast).unwrap()).unwrap();
    let recomputed = decoded.with_max_output_tokens(32).unwrap();
    assert_eq!(
        recomputed.estimate,
        forecast.with_max_output_tokens(32).unwrap().estimate
    );
    let short = forecast.with_max_output_tokens(1).unwrap();
    assert!(
        short.estimate.domains[0]
            .generation_peak
            .upper_bytes
            .unwrap()
            <= recomputed.estimate.domains[0]
                .generation_peak
                .upper_bytes
                .unwrap()
    );
    assert!(forecast.execution.full_pass_reason.is_some());
    assert_eq!(forecast.request.prefill_chunk_tokens, 17);
}
