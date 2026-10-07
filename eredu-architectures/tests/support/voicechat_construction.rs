//! Released shape-only construction coverage against the pinned physical header.

use super::*;
use eredu_architectures::voicechat::{
    checkpoint::{component_for_key, Component},
    config::VoiceChatConfig,
    construction::{ConstructionError, SpeechConstructionPlan, SpeechUnitId},
    controls::VOICECHAT_INTEGER_BUFFERS,
};

#[test]
fn voicechat_construction_covers_released_learned_parameters_once_by_unit() {
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let plan = SpeechConstructionPlan::from_config(&config).unwrap();
    let header: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/voicechat/official-header.json")).unwrap();
    let expected = header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| {
            !matches!(
                component_for_key(name),
                Some(Component::Backbone | Component::Controls)
            ) && !name.starts_with("stt_model.perception.preprocessor.")
                && !VOICECHAT_INTEGER_BUFFERS
                    .iter()
                    .any(|buffer| buffer.key == name.as_str())
        })
        .map(|(name, value)| {
            let shape = value["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|axis| axis.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            (name.clone(), shape)
        })
        .collect::<BTreeMap<_, _>>();
    struct Census {
        unit: SpeechUnitId,
        parameters: BTreeMap<String, Vec<i32>>,
    }
    impl<'a> ParameterVisitor<'a, ReferenceTensor> for Census {
        fn visit(&mut self, metadata: ParameterMetadata, tensor: &'a ReferenceTensor) {
            let name = metadata.id.as_str();
            assert_eq!(
                component_for_key(name),
                Some(self.unit.component()),
                "{name}"
            );
            assert!(
                self.parameters
                    .insert(name.into(), tensor.shape().to_vec())
                    .is_none(),
                "parameter {name} is duplicated across independently bindable units"
            );
        }
    }
    let mut census = Census {
        unit: SpeechUnitId::Rnnt,
        parameters: BTreeMap::new(),
    };
    struct BindMarker {
        marker: i32,
        bindings: BTreeMap<String, i32>,
    }
    impl<'a> ParameterVisitorMut<'a, ReferenceTensor> for BindMarker {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut ReferenceTensor) {
            assert!(self
                .bindings
                .insert(metadata.id.as_str().into(), self.marker)
                .is_none());
            // The reference tensor reserves a trailing value below -1000 as a
            // payload marker excluded from its shape. This models replacement
            // during binding without allocating released-size weight payloads.
            value.0.push(self.marker);
        }
    }
    let mut binder = BindMarker {
        marker: -1001,
        bindings: BTreeMap::new(),
    };
    let mut visited = Vec::new();
    let assembled = plan
        .assemble_with::<AssemblyBackend, Infallible>(&(), |id, module| {
            visited.push(id);
            census.unit = id;
            let topology = eredu_nn::validate_parameter_topology(module).unwrap();
            assert!(!topology.is_empty(), "empty declared unit {id:?}");
            module.visit_parameters(&mut census);
            module.visit_parameters_mut(&mut binder);
            binder.marker -= 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, plan.units());
    assert_eq!(census.parameters, expected);
    struct AssembledCensus<'a> {
        bindings: &'a BTreeMap<String, i32>,
        parameters: BTreeMap<String, Vec<i32>>,
    }
    impl<'a> ParameterVisitor<'a, ReferenceTensor> for AssembledCensus<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, tensor: &'a ReferenceTensor) {
            let name = metadata.id.as_str();
            assert_eq!(
                tensor.0.last(),
                self.bindings.get(name),
                "binding lost for {name}"
            );
            assert!(self
                .parameters
                .insert(name.into(), tensor.shape().to_vec())
                .is_none());
        }
    }
    let mut assembled_census = AssembledCensus {
        bindings: &binder.bindings,
        parameters: BTreeMap::new(),
    };
    assembled.visit_parameters(&mut assembled_census);
    assert_eq!(assembled_census.parameters, expected);
    // Unit indices are geometry, not opaque physical tensor names; an invalid
    // index must fail before constructing any partial module or reading payloads.
    for invalid in [
        SpeechUnitId::PerceptionBlock(config.perception.encoder.n_layers),
        SpeechUnitId::CharacterBlock(config.tts.character_encoder.num_hidden_layers),
        SpeechUnitId::TtsBlock(config.tts.backbone.num_hidden_layers),
        SpeechUnitId::CodecCodebook(config.codec.num_quantizers),
        SpeechUnitId::CodecEncoder(usize::MAX),
    ] {
        assert!(matches!(plan.build_unit::<ReferenceBackend>(invalid, &()),
            Err(ConstructionError::Unit(id)) if id == invalid));
    }
}

#[test]
fn voicechat_provider_assembly_rejects_a_function_head_from_another_geometry() {
    use eredu_architectures::voicechat::{
        assembly::{AssemblyError, SpeechProviderKind, SpeechProviderParts},
        channels::ChannelError,
        function_head_layered::{FunctionHeadDimensions, FunctionHeadExecution},
    };

    // A selected provider must match the admitted graph even if it does not
    // construct or lease its learned projection until the first invocation.
    struct DeferredFunction(FunctionHeadDimensions);
    impl FunctionHeadExecution<AssemblyBackend> for DeferredFunction {
        fn dimensions(&self) -> FunctionHeadDimensions {
            self.0
        }
        fn project(
            &mut self,
            _: &ReferenceTensor,
            _: &(),
        ) -> Result<ReferenceTensor, ChannelError> {
            panic!("assembly must not invoke a selected provider")
        }
    }
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let plan = SpeechConstructionPlan::from_config(&config).unwrap();
    let assembled = plan
        .assemble_with::<AssemblyBackend, Infallible>(&(), |_, _| Ok(()))
        .unwrap();
    let (perception, rnnt) = assembled.input.into_parts();
    let mut dimensions = assembled.function.dimensions();
    dimensions.vocabulary -= 1;
    let result = plan.assemble_providers(
        SpeechProviderParts {
            perception,
            rnnt,
            function: DeferredFunction(dimensions),
            generator: assembled.generator,
            encoder: assembled.encoder.neural,
            decoder: assembled.decoder.neural,
            quantizer: assembled.decoder.quantizer,
        },
        &(),
    );
    assert!(matches!(
        result,
        Err(AssemblyError::ProviderGeometry(
            SpeechProviderKind::FunctionHead
        ))
    ));
}

#[test]
fn voicechat_construction_stops_at_binding_failure() {
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let plan = SpeechConstructionPlan::from_config(&config).unwrap();
    let stop = SpeechUnitId::Rnnt;
    let mut visited = Vec::new();
    let result = plan.assemble_with::<AssemblyBackend, Error>(&(), |id, _| {
        visited.push(id);
        if id == stop {
            Err(Error::backend("injected binding failure"))
        } else {
            Ok(())
        }
    });
    assert!(result.is_err());
    let expected = plan.units().iter().position(|id| *id == stop).unwrap();
    assert_eq!(visited, &plan.units()[..=expected]);
}

// This wrapper permits construction of FFT constants, not numerical execution.
#[test]
fn voicechat_missing_spectral_mechanisms_reject_before_binding() {
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let plan = SpeechConstructionPlan::from_config(&config).unwrap();
    let mut bindings = 0;
    let result = plan.assemble_with::<ReferenceBackend, Infallible>(&(), |_, _| {
        bindings += 1;
        Ok(())
    });
    assert!(matches!(
        result,
        Err(
            eredu_architectures::voicechat::assembly::AssemblyError::Construction(
                ConstructionError::Neural(_)
            )
        )
    ));
    assert_eq!(bindings, 0);
}

// This wrapper permits construction of FFT constants, not numerical execution.
// Keep the shared reference backend's advertised execution facts unchanged.
struct AssemblyBackend;
impl NeuralBackend for AssemblyBackend {
    const OPERATOR_CAPABILITIES: eredu_nn::NeuralOperatorCapabilities =
        ReferenceBackend::OPERATOR_CAPABILITIES
            .union(eredu_nn::NeuralOperatorCapabilities::RFFT)
            .union(eredu_nn::NeuralOperatorCapabilities::IRFFT)
            .union(eredu_nn::NeuralOperatorCapabilities::SIN_COS)
            .union(eredu_nn::NeuralOperatorCapabilities::FULL_F32)
            .union(eredu_nn::NeuralOperatorCapabilities::FROM_I32_SLICE)
            .union(eredu_nn::NeuralOperatorCapabilities::CAST_FLOAT)
            .union(eredu_nn::NeuralOperatorCapabilities::SOFTPLUS)
            .union(eredu_nn::NeuralOperatorCapabilities::EXP)
            .union(eredu_nn::NeuralOperatorCapabilities::CLIP)
            .union(eredu_nn::NeuralOperatorCapabilities::BROADCAST_TO);
    type Tensor = ReferenceTensor;
    type Linear = ReferenceLinear;
    type Embedding = ReferenceEmbedding;
    type Normalization = ReferenceNorm;
    type Rotary = ReferenceRotary;
    type ParallelContext = ();

    fn linear(spec: LinearSpec, context: &()) -> Result<Self::Linear, Error> {
        ReferenceBackend::linear(spec, context)
    }
    fn embedding(spec: EmbeddingSpec, context: &()) -> Result<Self::Embedding, Error> {
        ReferenceBackend::embedding(spec, context)
    }
    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &(),
    ) -> Result<Self::Normalization, Error> {
        ReferenceBackend::normalization(spec, context)
    }
    fn rotary(spec: RotarySpec, context: &()) -> Result<Self::Rotary, Error> {
        ReferenceBackend::rotary(spec, context)
    }
    fn silu(input: Self::Tensor, context: &()) -> Result<Self::Tensor, Error> {
        ReferenceBackend::silu(input, context)
    }
    fn gated_product(
        gate: Self::Tensor,
        up: Self::Tensor,
        policy: eredu_nn::GatedProductPolicy,
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        ReferenceBackend::gated_product(gate, up, policy, context)
    }
    fn attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        mask: Option<&Self::Tensor>,
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        ReferenceBackend::attention(queries, keys, values, scale, mask, context)
    }
    fn sliding_window_attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        ReferenceBackend::sliding_window_attention(
            queries,
            keys,
            values,
            scale,
            window,
            position_offset,
            context,
        )
    }
    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        ReferenceBackend::causal_mask(sequence, offset, window, context)
    }
    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        parallel: &(),
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        ReferenceBackend::row_parallel_linear(linear, input, parallel, context)
    }
}
