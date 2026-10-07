use super::*;
use crate::voicechat::rnnt::RnntDimensions;
use crate::voicechat::{
    artifact::tests::official_parameters,
    checkpoint,
    codec::{waveform_decoder_config, waveform_encoder_config},
    config::VoiceChatConfig,
    controls::VOICECHAT_INTEGER_BUFFERS,
    function_head_layered::FunctionHeadDimensions,
    perception::PerceptionDimensions,
    tts_conditioning_layered::ConditioningDimensions,
    tts_mog::MogDimensions,
    tts_rvq::RvqDimensions,
    tts_transformer::TransformerConfig,
};
use eredu_checkpoint::store::{
    CheckpointLease, CheckpointSource, TensorMetadata, TensorReadRequest, WeightStoreDiagnostics,
};
use eredu_checkpoint::{
    recipe::RecipeDtype,
    store::{SharedCheckpointSource, StoreError},
};
use std::sync::Arc;

struct HeaderSource(BTreeMap<String, TensorMetadata>);
impl CheckpointSource for HeaderSource {
    fn source_keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("cold speech preparation must not read released weight payloads")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        Err(StoreError::Internal(
            "header fixture has no payload store".into(),
        ))
    }
}

fn fixture() -> (
    PreparedSpeechProviders,
    BTreeMap<String, ParameterAdmission>,
) {
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let parameters = official_parameters();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let encoder = waveform_encoder_config(&config.codec).unwrap();
    let decoder = waveform_decoder_config(&config.codec).unwrap();
    let recipes = parameters
        .iter()
        .map(|(name, p)| (name.clone(), p.recipe.clone()))
        .collect();
    let mut prepared = PreparedSpeechProviders {
        plan: SpeechConstructionPlan::from_config(&config).unwrap(),
        perception: PreparedPerceptionBindings::from_parameters(
            PerceptionDimensions::from_config(&config.perception).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        function: PreparedFunctionHeadBindings::from_parameters(
            FunctionHeadDimensions::new(4480, 131072).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        rnnt: PreparedRnntBindings::from_parameters(
            RnntDimensions::from_config(&config.rnnt).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        character: PreparedTtsTransformerBindings::from_parameters(
            TransformerConfig::new(&config.tts.character_encoder).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        transformer: PreparedTtsTransformerBindings::from_parameters(
            TransformerConfig::new(&config.tts.backbone).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        mog: PreparedMogBindings::from_parameters(
            MogDimensions::from_config(&config.tts).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        rvq: PreparedRvqBindings::from_parameters(
            RvqDimensions::from_config(&config.tts).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        conditioning: PreparedConditioningBindings::from_parameters(
            ConditioningDimensions::from_config(&config.tts, 256, 37).unwrap(),
            source.clone(),
            &parameters,
        )
        .unwrap(),
        encoder: PreparedEncoderBindings::from_recipes(
            encoder.neural,
            "tts_model.audio_codec.encoder",
            source.clone(),
            &recipes,
        )
        .unwrap(),
        decoder: PreparedDecoderBindings::from_recipes(
            decoder.neural,
            "tts_model.audio_codec.decoder",
            source.clone(),
            &recipes,
        )
        .unwrap(),
        quantizer: PreparedQuantizerBindings::from_recipes(
            decoder.quantizer,
            "tts_model.audio_codec.prvq",
            source,
            &recipes,
        )
        .unwrap(),
        resources: BTreeMap::new(),
    };
    prepared.prepare_resources(&parameters).unwrap();
    (prepared, parameters)
}

#[test]
fn released_provider_census_charges_every_selected_learned_parameter_without_payload_reads() {
    let (prepared, parameters) = fixture();
    let expected: u64 = parameters
        .iter()
        .filter(|(name, parameter)| {
            !matches!(
                parameter.component,
                Component::Backbone | Component::Controls
            ) && !name.starts_with("stt_model.perception.preprocessor.")
                && !VOICECHAT_INTEGER_BUFFERS
                    .iter()
                    .any(|buffer| buffer.key == name.as_str())
        })
        .map(|(_, p)| p.output.shape.iter().product::<usize>() as u64 * 4)
        .sum();
    let actual = prepared
        .resources()
        .values()
        .map(|r| r.parameter_bytes)
        .sum::<u64>();
    assert_eq!(actual, expected);
    assert_eq!(
        prepared.resources()[&SpeechProvider::FunctionHead].parameter_bytes,
        2_348_810_240
    );
    assert_eq!(
        prepared.resources()[&SpeechProvider::Rnnt].parameter_bytes,
        35_772_420
    );
    for provider in [
        SpeechProvider::Rnnt,
        SpeechProvider::Perception,
        SpeechProvider::CharacterEncoder,
        SpeechProvider::TtsTransformer,
        SpeechProvider::Mog,
        SpeechProvider::CodecEncoder,
        SpeechProvider::CodecDecoder,
        SpeechProvider::CodecQuantizer,
    ] {
        let resources = prepared.resources()[&provider];
        assert!(
            resources.largest_unit_bytes < resources.parameter_bytes,
            "{provider:?} pins its whole stack"
        );
        assert!(resources.materialization_peak_bytes >= resources.largest_unit_bytes);
    }
}

#[test]
fn rnnt_retains_exact_units_and_bounded_conversion_without_acquiring_payloads() {
    use crate::voicechat::rnnt_preparation::RnntPreparationError;
    let (prepared, mut parameters) = fixture();
    let rnnt = prepared.rnnt;
    assert_eq!(rnnt.unit_bindings().iter().map(Vec::len).sum::<usize>(), 15);
    let resources =
        SpeechProviderResources::from_bindings(rnnt.source(), rnnt.unit_bindings(), &parameters)
            .unwrap();
    assert!(resources.largest_unit_bytes < resources.parameter_bytes);
    assert!(resources.materialization_peak_bytes >= resources.largest_unit_bytes);
    for binding in rnnt.unit_bindings().iter().flatten() {
        let output = binding
            .source_recipe()
            .infer(rnnt.source().as_ref())
            .unwrap();
        assert_eq!(output.dtype, RecipeDtype::F32);
        assert_eq!(
            binding.expected_bytes(),
            output.shape.iter().product::<usize>() as u64 * 4
        );
    }
    let dimensions = rnnt.dimensions();
    let retained = rnnt.source().clone();
    let (_, local, layout, source, units) = rnnt.into_parts();
    assert!(local.is_none());
    assert!(Arc::ptr_eq(&retained, &source));
    assert_eq!(layout.len(), units.len());
    assert_eq!(source.source_keys().len(), 15);
    assert!(matches!(
        source.source_metadata("stt_model.function_head.weight"),
        Err(StoreError::UnauthorizedTensor { .. })
    ));
    let mut recipes = parameters
        .iter()
        .map(|(name, p)| (name.clone(), p.recipe.clone()))
        .collect::<BTreeMap<_, _>>();
    // Half-precision storage remains retained while each leased unit admits its
    // F32 output and conversion temporary without reading checkpoint payloads.
    let mut half_headers = checkpoint::tests::official().tensors().clone();
    for name in source.source_keys() {
        let metadata = half_headers.get_mut(&name).unwrap();
        metadata.stored_dtype = eredu_checkpoint::StoredDtype::F16;
        metadata.encoded_byte_len = metadata.logical_shape.iter().product::<usize>() as u64 * 2;
    }
    let half_source: SharedCheckpointSource = Arc::new(HeaderSource(half_headers));
    let half = PreparedRnntBindings::from_recipes(dimensions, half_source, &recipes).unwrap();
    for binding in half.unit_bindings().iter().flatten() {
        let stored = half.source().source_metadata(binding.name()).unwrap();
        let recipe = binding.source_recipe();
        assert_eq!(stored.stored_dtype, eredu_checkpoint::StoredDtype::F16);
        assert_eq!(
            recipe.infer(half.source().as_ref()).unwrap().dtype,
            RecipeDtype::F32
        );
        assert!(
            recipe
                .peak_materialization_bytes(half.source().as_ref())
                .unwrap()
                >= stored.encoded_byte_len + binding.expected_bytes()
        );
    }
    let name = "stt_model.rnnt_joint.enc.weight";
    recipes.remove(name);
    assert!(matches!(
        PreparedRnntBindings::from_recipes(dimensions, source.clone(), &recipes),
        Err(RnntPreparationError::MissingParameter(key)) if key == name
    ));
    parameters.get_mut(name).unwrap().output.shape[0] += 1;
    assert!(matches!(
        PreparedRnntBindings::from_parameters(dimensions, source, &parameters),
        Err(RnntPreparationError::Geometry(key)) if key == name
    ));
}

#[test]
fn released_rnnt_tp_preparation_preserves_blank_rows_and_charges_recurrent_replicas() {
    use crate::voicechat::rnnt_preparation::RnntPreparationError;
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let dimensions = RnntDimensions::from_config(&config.rnnt).unwrap();
    let parameters = official_parameters();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let prepare =
        || PreparedRnntBindings::from_parameters(dimensions, source.clone(), &parameters).unwrap();
    let mut rank_layouts = Vec::new();
    let mut total_bytes = 0;
    for rank in 0..2 {
        let original = prepare();
        let retained = original.source().clone();
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank).unwrap();
        let (selected, layout) = original.select_parallel(topology).unwrap();
        assert!(Arc::ptr_eq(&retained, selected.source()));
        assert_eq!(selected.local_layout(), Some(&layout));
        let rows = if rank == 0 { 513 } else { 512 };
        for (name, shape) in [
            (
                "stt_model.rnnt_decoder.prediction.embed.weight",
                vec![rows, 640],
            ),
            ("stt_model.rnnt_joint.joint_net.2.weight", vec![rows, 640]),
            ("stt_model.rnnt_joint.joint_net.2.bias", vec![rows]),
            (
                "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_hh_l1",
                vec![2560, 640],
            ),
        ] {
            let binding = selected
                .unit_bindings()
                .iter()
                .flatten()
                .find(|b| b.name() == name)
                .unwrap();
            let output = binding
                .source_recipe()
                .infer(selected.source().as_ref())
                .unwrap();
            assert_eq!(output.shape, shape);
            assert_eq!(output.dtype, RecipeDtype::F32);
        }
        assert!(matches!(
            selected
                .source()
                .source_metadata("stt_model.function_head.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let resources = SpeechProviderResources::from_bindings(
            selected.source(),
            selected.unit_bindings(),
            &parameters,
        )
        .unwrap();
        // Both LSTM layers and the encoder/predictor joint projections remain
        // complete replicas. Only the embedding and final vocabulary affine split.
        let replicated_bytes = 30_520_320;
        assert_eq!(
            resources.parameter_bytes,
            replicated_bytes + rows as u64 * (640 + 640 + 1) * 4
        );
        assert_eq!(resources.largest_unit_bytes, 13_127_680);
        assert!(resources.materialization_peak_bytes >= resources.largest_unit_bytes);
        total_bytes += resources.parameter_bytes;
        let (selected, repeated) = selected.select_parallel(topology).unwrap();
        assert_eq!(repeated, layout);
        assert!(Arc::ptr_eq(&retained, selected.source()));
        assert!(matches!(
            selected.into_parallel(&layout),
            Err(RnntPreparationError::Parallel(_))
        ));
        rank_layouts.push(layout);
    }
    assert_eq!(total_bytes, 35_772_420 + 30_520_320);
    let mut mixed = rank_layouts[0].clone();
    let bias = "stt_model.rnnt_joint.joint_net.2.bias";
    mixed.insert(bias.into(), rank_layouts[1].tensor(bias).unwrap().clone());
    assert!(
        prepare().into_parallel(&mixed).is_err(),
        "vocabulary weight and bias must retain the same rows"
    );
    let pipeline =
        ParallelRankTopology::new(ParallelTopology::new(1, 2, 1, 1).unwrap(), 0).unwrap();
    assert!(matches!(
        prepare().select_parallel(pipeline),
        Err(RnntPreparationError::Parallel(_))
    ));
}

#[test]
fn rnnt_projection_sources_and_tensor_census_exclude_recurrent_replicas() {
    use crate::voicechat::rnnt_projection_preparation::PreparedRnntProjectionBindings;
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let dimensions = RnntDimensions::from_config(&config.rnnt).unwrap();
    let parameters = official_parameters();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let full =
        || PreparedRnntBindings::from_parameters(dimensions, source.clone(), &parameters).unwrap();
    let ordinary = PreparedRnntProjectionBindings::from_rnnt(full()).unwrap();
    assert_eq!(
        ordinary
            .unit_bindings()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        [1, 2, 2, 2]
    );
    let resources = SpeechProviderResources::from_bindings(
        ordinary.source(),
        ordinary.unit_bindings(),
        &parameters,
    )
    .unwrap();
    assert_eq!(resources.parameter_bytes, 9_517_060);
    for rank in 0..2 {
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank).unwrap();
        let (selected, layout) = PreparedRnntProjectionBindings::from_rnnt(full())
            .unwrap()
            .select_parallel(topology)
            .unwrap();
        let (full_selected, _) = full().select_parallel(topology).unwrap();
        let projected = PreparedRnntProjectionBindings::from_rnnt(full_selected).unwrap();
        assert_eq!(projected.local_layout(), Some(&layout));
        assert_eq!(layout.tensors().count(), 7);
        assert_eq!(projected.source().source_keys().len(), 7);
        for layer in 0..2 {
            for stem in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                let name =
                    format!("stt_model.rnnt_decoder.prediction.dec_rnn.lstm.{stem}_l{layer}");
                assert!(layout.tensor(&name).is_none());
                assert!(matches!(
                    projected.source().source_metadata(&name),
                    Err(StoreError::UnauthorizedTensor { .. })
                ));
            }
        }
        for (expected, actual) in selected
            .unit_bindings()
            .iter()
            .flatten()
            .zip(projected.unit_bindings().iter().flatten())
        {
            assert_eq!(actual.name(), expected.name());
            assert_eq!(actual.source_recipe(), expected.source_recipe());
            assert_eq!(actual.expected_bytes(), expected.expected_bytes());
        }
        let resources = SpeechProviderResources::from_bindings(
            projected.source(),
            projected.unit_bindings(),
            &parameters,
        )
        .unwrap();
        let rows = if rank == 0 { 513 } else { 512 };
        assert_eq!(resources.parameter_bytes, 4_264_960 + rows * 1281 * 4);
        let (projected, repeated) = projected.select_parallel(topology).unwrap();
        assert_eq!(repeated, layout);
        let other = ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1 - rank)
            .unwrap();
        assert!(projected.select_parallel(other).is_err());
    }
}

#[test]
fn rnnt_tp_communication_admits_exact_shapes_and_rejects_incomplete_cold_contracts() {
    use crate::voicechat::rnnt_preparation::{
        validate_parallel_communication, RnntPreparationError,
    };
    use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode};
    use eredu_runtime::{
        CommunicationCapabilityError, CommunicationCompletionPolicy, CommunicationGroupDescriptor,
        CommunicationGroupRequirements, CommunicationManifest, CommunicationOperation as Op,
        CommunicationOperationRequirement as Requirement, CommunicationTensorLimits,
    };
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let dimensions = RnntDimensions::from_config(&config.rnnt).unwrap();
    let group = CollectiveGroupId::new(47);
    let tensor = |operation, dtype, rank, input, output, exact| {
        Requirement::tensors(
            operation,
            [dtype],
            CommunicationTensorLimits::new(1, rank, input, None)
                .unwrap()
                .with_output_tensor_elements(output)
                .unwrap(),
            exact,
        )
        .unwrap()
    };
    let operations = vec![
        Requirement::failure_agreement(true),
        tensor(Op::Broadcast, TensorDtype::I32, 1, 1, 1, true),
        tensor(Op::AllReduceSum, TensorDtype::F32, 2, 640, 640, true),
        tensor(Op::AllGatherUneven, TensorDtype::F32, 2, 513, 1025, true),
    ];
    let manifest = |operations, member: bool, bounded: bool| {
        let descriptor = CommunicationGroupDescriptor::new(
            group,
            0,
            vec![0, 1],
            member.then_some(1),
            CommunicationGroupRequirements::new(operations).unwrap(),
        )
        .unwrap();
        let manifest =
            CommunicationManifest::new(3, if member { 1 } else { 2 }, vec![descriptor], vec![])
                .unwrap();
        if bounded {
            manifest.with_completion_policy(
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
        } else {
            manifest
        }
    };
    // Exactly the logical operations suffice: no padded AllGatherEven declaration.
    validate_parallel_communication(dimensions, &manifest(operations.clone(), true, true), group)
        .unwrap();
    // Projection-only owners do not admit unrelated decision operations.
    let projections = operations
        .iter()
        .filter(|item| matches!(item.operation(), Op::AllReduceSum | Op::AllGatherUneven,))
        .cloned()
        .collect();
    crate::voicechat::rnnt_projection_preparation::validate_projection_parallel_communication(
        dimensions,
        &manifest(projections, true, true),
        group,
    )
    .unwrap();
    for omitted in [
        Op::FailureAgreement,
        Op::Broadcast,
        Op::AllReduceSum,
        Op::AllGatherUneven,
    ] {
        let reduced = operations
            .iter()
            .filter(|item| item.operation() != omitted)
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            matches!(validate_parallel_communication(dimensions, &manifest(reduced, true, true), group),
            Err(RnntPreparationError::Communication(CommunicationCapabilityError::MissingOperation { operation })) if operation == omitted)
        );
    }
    for replacement in [
        tensor(Op::Broadcast, TensorDtype::F32, 1, 1, 1, true),
        tensor(Op::AllReduceSum, TensorDtype::F16, 2, 640, 640, true),
        tensor(Op::AllReduceSum, TensorDtype::F32, 2, 639, 640, true),
        tensor(Op::AllReduceSum, TensorDtype::F32, 2, 640, 639, true),
        tensor(Op::AllGatherUneven, TensorDtype::F32, 1, 513, 1025, true),
        tensor(Op::AllGatherUneven, TensorDtype::F32, 2, 512, 1025, true),
        tensor(Op::AllGatherUneven, TensorDtype::F32, 2, 513, 1024, true),
        tensor(Op::Broadcast, TensorDtype::I32, 1, 1, 1, false),
        Requirement::failure_agreement(false),
    ] {
        let mut altered = operations.clone();
        let operation = replacement.operation();
        *altered
            .iter_mut()
            .find(|item| item.operation() == operation)
            .unwrap() = replacement;
        assert!(matches!(
            validate_parallel_communication(dimensions, &manifest(altered, true, true), group),
            Err(RnntPreparationError::Communication(_))
        ));
    }
    assert!(validate_parallel_communication(
        dimensions,
        &manifest(operations.clone(), false, true),
        group
    )
    .is_err());
    assert!(validate_parallel_communication(
        dimensions,
        &manifest(operations.clone(), true, false),
        group
    )
    .is_err());
    assert!(validate_parallel_communication(
        dimensions,
        &manifest(operations, true, true),
        CollectiveGroupId::new(99)
    )
    .is_err());
}

#[test]
fn released_quantizer_tp_preparation_retains_channel_slices_variance_and_source_bounds() {
    use eredu_codec::prvq_preparation::QuantizerPreparationError;
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let geometry = waveform_decoder_config(&config.codec).unwrap().quantizer;
    let parameters = official_parameters();
    let recipes = parameters
        .iter()
        .map(|(name, p)| (name.clone(), p.recipe.clone()))
        .collect();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let prepare = || {
        PreparedQuantizerBindings::from_recipes(
            geometry,
            "tts_model.audio_codec.prvq",
            source.clone(),
            &recipes,
        )
        .unwrap()
    };
    let prepared = prepare();
    let retained = prepared.source().clone();
    let global = SpeechProviderResources::from_bindings(
        prepared.source(),
        prepared.unit_bindings(),
        &parameters,
    )
    .unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
    let (selected, layout) = prepared.select_parallel(topology).unwrap();
    assert!(Arc::ptr_eq(&retained, selected.source()));
    assert!(matches!(
        selected
            .source()
            .source_metadata("tts_model.tts_model.rvq_embs"),
        Err(StoreError::UnauthorizedTensor { .. })
    ));
    for (name, shape) in [
        ("tts_model.audio_codec.prvq.mus_list.0", vec![1024, 256]),
        ("tts_model.audio_codec.prvq.mus_list.30", vec![1024, 256]),
        (
            "tts_model.audio_codec.prvq._variance_list.0.variance",
            vec![],
        ),
    ] {
        let binding = selected
            .unit_bindings()
            .iter()
            .flatten()
            .find(|b| b.name() == name)
            .unwrap();
        let output = binding
            .source_recipe()
            .infer(selected.source().as_ref())
            .unwrap();
        assert_eq!(output.shape, shape);
        assert_eq!(output.dtype, eredu_checkpoint::recipe::RecipeDtype::F32);
        assert_eq!(
            binding.expected_bytes(),
            shape.iter().product::<usize>() as u64 * 4
        );
    }
    let resources = SpeechProviderResources::from_bindings(
        selected.source(),
        selected.unit_bindings(),
        &parameters,
    )
    .unwrap();
    assert_eq!(resources.parameter_bytes, 31 * (1024 * 256 + 1) * 4);
    assert_eq!(
        2 * resources.parameter_bytes,
        global.parameter_bytes + 31 * 4
    );
    assert!(resources.materialization_peak_bytes > resources.largest_unit_bytes);
    let (selected, same_layout) = selected.select_parallel(topology).unwrap();
    assert_eq!(same_layout, layout);
    assert!(Arc::ptr_eq(&retained, selected.source()));
    assert!(matches!(
        selected.into_parallel(&layout),
        Err(QuantizerPreparationError::Parallel(_))
    ));
    let opposite =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
    let (_, first) = prepare().select_parallel(opposite).unwrap();
    let mut mismatched = layout;
    let mean = "tts_model.audio_codec.prvq.mus_list.30";
    mismatched.insert(mean.into(), first.tensor(mean).unwrap().clone());
    assert!(prepare().into_parallel(&mismatched).is_err());
}

#[test]
fn released_decoder_tp_preparation_retains_convolution_slices_replicas_and_source_bounds() {
    use eredu_codec::ear_decoder_preparation::DecoderPreparationError;
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let decoder = waveform_decoder_config(&config.codec).unwrap().neural;
    let parameters = official_parameters();
    let recipes = parameters
        .iter()
        .map(|(name, p)| (name.clone(), p.recipe.clone()))
        .collect();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let prepare = || {
        PreparedDecoderBindings::from_recipes(
            decoder.clone(),
            "tts_model.audio_codec.decoder",
            source.clone(),
            &recipes,
        )
        .unwrap()
    };
    let prepared = prepare();
    let retained = prepared.source().clone();
    let global = SpeechProviderResources::from_bindings(
        prepared.source(),
        prepared.unit_bindings(),
        &parameters,
    )
    .unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
    let (selected, layout) = prepared.select_parallel(topology).unwrap();
    assert!(Arc::ptr_eq(&retained, selected.source()));
    assert!(matches!(
        selected
            .source()
            .source_metadata("tts_model.audio_codec.prvq.mus_list.0"),
        Err(StoreError::UnauthorizedTensor { .. })
    ));
    for (name, shape) in [
        (
            "tts_model.audio_codec.decoder.layers.0.weight",
            vec![512, 768, 9],
        ),
        (
            "tts_model.audio_codec.decoder.layers.1.pwconv1.weight",
            vec![3072, 1536, 1],
        ),
        (
            "tts_model.audio_codec.decoder.layers.1.pwconv1.bias",
            vec![3072],
        ),
        (
            "tts_model.audio_codec.decoder.layers.1.pwconv2.weight",
            vec![1536, 3072, 1],
        ),
        (
            "tts_model.audio_codec.decoder.layers.1.pwconv2.bias",
            vec![1536],
        ),
        (
            "tts_model.audio_codec.decoder.layers.1.dwconv.weight",
            vec![1536, 1, 7],
        ),
        (
            "tts_model.audio_codec.decoder.layers.12.weight",
            vec![9, 384, 1],
        ),
    ] {
        let binding = selected
            .unit_bindings()
            .iter()
            .flatten()
            .find(|b| b.name() == name)
            .unwrap();
        let output = binding
            .source_recipe()
            .infer(selected.source().as_ref())
            .unwrap();
        assert_eq!(output.shape, shape);
        assert_eq!(output.dtype, eredu_checkpoint::recipe::RecipeDtype::F32);
        assert_eq!(
            binding.expected_bytes(),
            shape.iter().product::<usize>() as u64 * 4
        );
    }
    let resources = SpeechProviderResources::from_bindings(
        selected.source(),
        selected.unit_bindings(),
        &parameters,
    )
    .unwrap();
    // Nine full-width depthwise kernels, biases, normalization and post-reduction biases.
    assert_eq!(
        2 * resources.parameter_bytes,
        global.parameter_bytes + 354_816
    );
    assert!(resources.materialization_peak_bytes > resources.largest_unit_bytes);
    let (selected, same_layout) = selected.select_parallel(topology).unwrap();
    assert_eq!(same_layout, layout);
    assert!(Arc::ptr_eq(&retained, selected.source()));
    assert!(matches!(
        selected.into_parallel(&layout),
        Err(DecoderPreparationError::Parallel(_))
    ));
    let opposite =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
    let (_, first) = prepare().select_parallel(opposite).unwrap();
    let mut mismatched = layout;
    let projection = "tts_model.audio_codec.decoder.layers.1.pwconv2.weight";
    mismatched.insert(projection.into(), first.tensor(projection).unwrap().clone());
    assert!(prepare().into_parallel(&mismatched).is_err());
}

#[test]
fn released_encoder_tp_preparation_retains_convolution_slices_replicas_and_source_bounds() {
    use eredu_codec::ear_encoder_preparation::EncoderPreparationError;
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let encoder = waveform_encoder_config(&config.codec).unwrap().neural;
    let parameters = official_parameters();
    let recipes = parameters
        .iter()
        .map(|(name, p)| (name.clone(), p.recipe.clone()))
        .collect();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let prepare = || {
        PreparedEncoderBindings::from_recipes(
            encoder.clone(),
            "tts_model.audio_codec.encoder",
            source.clone(),
            &recipes,
        )
        .unwrap()
    };
    let prepared = prepare();
    let retained = prepared.source().clone();
    let global = SpeechProviderResources::from_bindings(
        prepared.source(),
        prepared.unit_bindings(),
        &parameters,
    )
    .unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
    let (selected, layout) = prepared.select_parallel(topology).unwrap();
    assert!(Arc::ptr_eq(&retained, selected.source()));
    assert!(matches!(
        selected
            .source()
            .source_metadata("tts_model.audio_codec.prvq.mus_list.0"),
        Err(StoreError::UnauthorizedTensor { .. })
    ));
    for (name, shape) in [
        (
            "tts_model.audio_codec.encoder.layers.0.weight",
            vec![192, 18, 1],
        ),
        (
            "tts_model.audio_codec.encoder.layers.1.pwconv1.weight",
            vec![768, 384, 1],
        ),
        (
            "tts_model.audio_codec.encoder.layers.1.pwconv1.bias",
            vec![768],
        ),
        (
            "tts_model.audio_codec.encoder.layers.1.pwconv2.weight",
            vec![384, 768, 1],
        ),
        (
            "tts_model.audio_codec.encoder.layers.1.pwconv2.bias",
            vec![384],
        ),
        (
            "tts_model.audio_codec.encoder.layers.1.dwconv.weight",
            vec![384, 1, 7],
        ),
        (
            "tts_model.audio_codec.encoder.layers.12.weight",
            vec![256, 1536, 9],
        ),
    ] {
        let binding = selected
            .unit_bindings()
            .iter()
            .flatten()
            .find(|b| b.name() == name)
            .unwrap();
        let output = binding
            .source_recipe()
            .infer(selected.source().as_ref())
            .unwrap();
        assert_eq!(output.shape, shape);
        assert_eq!(output.dtype, eredu_checkpoint::recipe::RecipeDtype::F32);
        assert_eq!(
            binding.expected_bytes(),
            shape.iter().product::<usize>() as u64 * 4
        );
    }
    let resources = SpeechProviderResources::from_bindings(
        selected.source(),
        selected.unit_bindings(),
        &parameters,
    )
    .unwrap();
    // Nine full-width depthwise kernels, biases, normalization and post-reduction biases.
    assert_eq!(
        2 * resources.parameter_bytes,
        global.parameter_bytes + 354_816
    );
    assert!(resources.materialization_peak_bytes > resources.largest_unit_bytes);
    let (selected, same_layout) = selected.select_parallel(topology).unwrap();
    assert_eq!(same_layout, layout);
    assert!(Arc::ptr_eq(&retained, selected.source()));
    assert!(matches!(
        selected.into_parallel(&layout),
        Err(EncoderPreparationError::Parallel(_))
    ));
    let opposite =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
    let (_, first) = prepare().select_parallel(opposite).unwrap();
    let mut mismatched = layout;
    let projection = "tts_model.audio_codec.encoder.layers.1.pwconv2.weight";
    mismatched.insert(projection.into(), first.tensor(projection).unwrap().clone());
    assert!(prepare().into_parallel(&mismatched).is_err());
}

#[test]
fn released_encoder_pipeline_preparation_restricts_sources_and_rejects_state_ownership() {
    use eredu_codec::{
        ear_encoder_layered::{encoder_parameter_description, encoder_state_layout},
        ear_encoder_partition::EncoderBoundarySchema,
    };
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    use eredu_runtime::{
        ArchitecturePartition, ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule,
        PartitionOwnership,
    };
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let config = waveform_encoder_config(&config.codec).unwrap().neural;
    let parameters = official_parameters();
    let recipes = parameters
        .iter()
        .map(|(name, p)| (name.clone(), p.recipe.clone()))
        .collect();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let prepare = || {
        PreparedEncoderBindings::from_recipes(
            config.clone(),
            "tts_model.audio_codec.encoder",
            source.clone(),
            &recipes,
        )
        .unwrap()
    };
    let description =
        encoder_parameter_description(&config, "tts_model.audio_codec.encoder").unwrap();
    let state = encoder_state_layout().unwrap();
    let count = config.unit_specs().unwrap().len();
    let partition = |state_plan: &ArchitectureStatePartitionPlan| {
        ArchitecturePartition::from_description(
            &description,
            [(description.graph().groups()[0].id(), 4..count)],
            PartitionOwnership::new(false, true, std::iter::empty::<String>()).unwrap(),
            &state,
            state_plan,
            (),
            EncoderBoundarySchema::new(&config).unwrap(),
        )
        .unwrap()
    };
    let selected = partition(&ArchitectureStatePartitionPlan::new([]));
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
    let (prepared, layout) = prepare().select_parallel(topology).unwrap();
    let first = prepared.unit_bindings()[4][0].name().to_owned();
    let expected_bytes = prepared.unit_bindings()[4..]
        .iter()
        .flatten()
        .map(|binding| binding.expected_bytes())
        .sum::<u64>();
    let stage = prepared.into_partition(&selected).unwrap();
    assert_eq!(stage.local_layout(), Some(&layout));
    assert_eq!(
        stage
            .unit_bindings()
            .iter()
            .map(|(address, _)| address.index())
            .collect::<Vec<_>>(),
        (4..count).collect::<Vec<_>>()
    );
    assert_eq!(
        stage
            .unit_bindings()
            .iter()
            .flat_map(|(_, bindings)| bindings)
            .map(|binding| binding.expected_bytes())
            .sum::<u64>(),
        expected_bytes
    );
    assert!(stage.source().source_metadata(&first).is_ok());
    for unowned in [
        "tts_model.audio_codec.encoder.layers.0.weight",
        "tts_model.audio_codec.decoder.layers.0.weight",
        "tts_model.audio_codec.prvq.mus_list.0",
    ] {
        assert!(matches!(
            stage.source().source_metadata(unowned),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
    }
    // This finite stage cannot acquire a persistent state slot, even a NoState marker.
    let wrong = partition(&ArchitectureStatePartitionPlan::new([
        ArchitectureStatePartitionRule::output_owner(0..1),
    ]));
    assert!(prepare().into_partition(&wrong).is_err());
}

#[test]
fn released_decoder_pipeline_preparation_restricts_sources_and_requires_exact_local_state_ownership(
) {
    use eredu_codec::{
        ear_decoder_layered::{decoder_parameter_description, decoder_state_layout},
        ear_decoder_partition::DecoderBoundarySchema,
    };
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    use eredu_runtime::{
        ArchitecturePartition, ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule,
        PartitionOwnership,
    };
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    let config = waveform_decoder_config(&config.codec).unwrap().neural;
    let parameters = official_parameters();
    let recipes = parameters
        .iter()
        .map(|(name, p)| (name.clone(), p.recipe.clone()))
        .collect();
    let source: SharedCheckpointSource = Arc::new(HeaderSource(
        checkpoint::tests::official().tensors().clone(),
    ));
    let prepare = || {
        PreparedDecoderBindings::from_recipes(
            config.clone(),
            "tts_model.audio_codec.decoder",
            source.clone(),
            &recipes,
        )
        .unwrap()
    };
    let description =
        decoder_parameter_description(&config, "tts_model.audio_codec.decoder").unwrap();
    let state = decoder_state_layout(&config).unwrap();
    let count = config.unit_specs().unwrap().len();
    let partition = |state_plan: &ArchitectureStatePartitionPlan| {
        ArchitecturePartition::from_description(
            &description,
            [(description.graph().groups()[0].id(), 4..count)],
            PartitionOwnership::new(false, true, std::iter::empty::<String>()).unwrap(),
            &state,
            state_plan,
            (),
            DecoderBoundarySchema::new(&config).unwrap(),
        )
        .unwrap()
    };
    let selected = partition(&ArchitectureStatePartitionPlan::new([
        ArchitectureStatePartitionRule::group_units(0, 0..count),
    ]));
    let local_state = selected.state().unwrap();
    assert_eq!(local_state.global_layers(), 4..count);
    assert_eq!(local_state.layout(), &state.slice(4..count).unwrap());
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
    let (prepared, layout) = prepare().select_parallel(topology).unwrap();
    let first = prepared.unit_bindings()[4][0].name().to_owned();
    let expected_bytes = prepared.unit_bindings()[4..]
        .iter()
        .flatten()
        .map(|binding| binding.expected_bytes())
        .sum::<u64>();
    let stage = prepared.into_partition(&selected).unwrap();
    assert_eq!(stage.local_layout(), Some(&layout));
    assert_eq!(
        stage
            .unit_bindings()
            .iter()
            .map(|(address, _)| address.index())
            .collect::<Vec<_>>(),
        (4..count).collect::<Vec<_>>()
    );
    assert_eq!(
        stage
            .unit_bindings()
            .iter()
            .flat_map(|(_, bindings)| bindings)
            .map(|binding| binding.expected_bytes())
            .sum::<u64>(),
        expected_bytes
    );
    assert!(stage.source().source_metadata(&first).is_ok());
    for unowned in [
        "tts_model.audio_codec.decoder.layers.0.weight",
        "tts_model.audio_codec.encoder.layers.0.weight",
        "tts_model.audio_codec.prvq.mus_list.0",
    ] {
        assert!(matches!(
            stage.source().source_metadata(unowned),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
    }
    // A stage cannot retain convolution caches for units owned by an upstream rank.
    let wrong = partition(&ArchitectureStatePartitionPlan::new([
        ArchitectureStatePartitionRule::output_owner(0..count),
    ]));
    assert!(prepare().into_partition(&wrong).is_err());
}
