use super::*;
use crate::artifact_preparation::PhysicalArtifactId;
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{MemoryWeightStore, ReadPolicy, TensorReadRequest, TensorSelection},
};
use eredu_nn::{LayerNorm, Parameter, ParameterSpec};

// Ordinary shared module traversal, with host placeholders because preparation
// inspects identities only; backend slot compatibility is checked before binding.
fn normalization(name: &str) -> LayerNorm<()> {
    LayerNorm {
        epsilon: 1e-6,
        weight: Some(Parameter::new(ParameterSpec::trainable(name).unwrap(), ())),
        bias: None,
    }
}

fn fixtures() -> (SharedCheckpointSource, BTreeMap<String, ParameterAdmission>) {
    // Physical names intentionally differ from module identities. The transformed
    // source must remain a recipe, rather than being reduced to a direct lease.
    let source: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([
            (
                "stored.block0".into(),
                safetensors::Dtype::F32,
                vec![2],
                [3.0f32, -2.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
            (
                "stored.block1".into(),
                safetensors::Dtype::F32,
                vec![1],
                9.0f32.to_le_bytes().to_vec(),
            ),
        ])
        .unwrap(),
    );
    let parameters = [
        (
            "tts_model.tts_model.backbone.layers.0.input_layernorm.weight",
            DerivedWeightRecipe::SubtractOne {
                input: Box::new(DerivedWeightRecipe::source(
                    "stored.block0",
                    TensorSelection::Full,
                )),
            },
        ),
        (
            "tts_model.tts_model.backbone.layers.1.input_layernorm.weight",
            DerivedWeightRecipe::source("stored.block1", TensorSelection::Full),
        ),
    ]
    .into_iter()
    .map(|(name, recipe)| {
        (
            name.into(),
            ParameterAdmission {
                linear_format: eredu_checkpoint::LinearFormat::Dense,
                name: name.into(),
                component: Component::Tts,
                artifact: PhysicalArtifactId::Primary,
                output: recipe.infer(source.as_ref()).unwrap(),
                stored_bytes: if name.contains("layers.0.") { 8 } else { 4 },
                conversion_workspace_bytes: 0,
                materialization_peak_bytes: 24,
                recipe,
            },
        )
    })
    .collect();
    (source, parameters)
}

#[test]
fn speech_parameter_unit_retains_only_its_exact_transformed_inputs() {
    let (source, parameters) = fixtures();
    let name = "tts_model.tts_model.backbone.layers.0.input_layernorm.weight";
    let original_provenance = source.source_provenance("stored.block0").unwrap();
    let unit = project_unit(&parameters, source, Component::Tts, &normalization(name)).unwrap();
    assert_eq!(unit.source().source_keys(), ["stored.block0"]);
    assert_eq!(unit.parameters().len(), 1);
    assert_eq!(unit.bindings().len(), 1);
    assert_eq!(unit.bindings()[0].name(), name);
    assert_eq!(unit.bindings()[0].recipe(), Some(&parameters[name].recipe));
    assert_eq!(
        unit.source().source_provenance("stored.block0").unwrap(),
        original_provenance
    );
    assert_eq!(unit.parameters()[name].materialization_peak_bytes, 24);
    assert!(unit
        .source()
        .acquire_lease(TensorReadRequest {
            key: "stored.block0".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .is_ok());
    assert!(unit
        .source()
        .acquire_lease(TensorReadRequest {
            key: "stored.block1".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .is_err());

    let (source, mut parameters) = fixtures();
    parameters.remove(name);
    assert!(matches!(
        project_unit(&parameters, source, Component::Tts, &normalization(name)),
        Err(SpeechParameterError::MissingParameter { name: missing, .. }) if missing == name
    ));
}

#[test]
fn speech_parameter_units_reject_host_buffers_and_preserve_declared_aliases() {
    let (source, mut parameters) = fixtures();
    for host in [
        "stt_model.perception.preprocessor.featurizer.fb",
        "tts_model.tts_model.embed_subword.bos_eos_emb.special_flags",
    ] {
        assert!(matches!(
            project_unit(&parameters, source.clone(), Component::Tts, &normalization(host)),
            Err(SpeechParameterError::HostParameter(name)) if name == host
        ));
    }
    let owner = "tts_model.tts_model.backbone.layers.0.input_layernorm.weight";
    let alias = "tts_model.tts_model.backbone.layers.1.input_layernorm.weight";
    let owner_spec = ParameterSpec::trainable(owner).unwrap();
    let mut alias_spec = ParameterSpec::trainable(alias).unwrap();
    alias_spec.alias_of = Some(owner_spec.id.clone());
    let slots = vec![
        Parameter::new(owner_spec, ()),
        Parameter::new(alias_spec, ()),
    ];
    assert!(matches!(
        project_unit(&parameters, source.clone(), Component::Tts, &slots),
        Err(SpeechParameterError::AliasMismatch { .. })
    ));
    let mut aliased_parameter = parameters[owner].clone();
    aliased_parameter.name = alias.into();
    parameters.insert(alias.into(), aliased_parameter);
    let unit = project_unit(&parameters, source, Component::Tts, &slots).unwrap();
    assert_eq!(unit.source().source_keys(), ["stored.block0"]);
    assert_eq!(unit.bindings().len(), 2);
    assert!(unit
        .bindings()
        .iter()
        .any(|binding| binding.name() == alias && binding.alias_of() == Some(owner)));
}

#[test]
fn speech_parameter_units_normalize_execution_precision_without_replacing_sources() {
    use eredu_checkpoint::store::EncodedTensorLease;

    // Both source formats can expose half-precision floating recipes. These
    // encoded values are 3 and -2; source identity and original bytes remain
    // available while the backend receives the explicit F32 recipe output.
    for (component, name, dtype, encoded, promotes) in [
        (
            Component::Tts,
            "tts_model.tts_model.backbone.layers.0.input_layernorm.weight",
            safetensors::Dtype::F16,
            [0x4200u16, 0xc000],
            true,
        ),
        (
            Component::Perception,
            "stt_model.perception.encoder.layers.0.norm_out.weight",
            safetensors::Dtype::BF16,
            [0x4040, 0xc000],
            true,
        ),
        (
            Component::CodecEncoder,
            "tts_model.audio_codec.encoder.norm.weight",
            safetensors::Dtype::F16,
            [0x4200, 0xc000],
            true,
        ),
        (
            Component::CodecDecoder,
            "tts_model.audio_codec.decoder.norm.weight",
            safetensors::Dtype::F16,
            [0x4200, 0xc000],
            true,
        ),
        (
            Component::CodecQuantizer,
            "tts_model.audio_codec.prvq.codebook.0.weight",
            safetensors::Dtype::F16,
            [0x4200, 0xc000],
            true,
        ),
        (
            Component::Rnnt,
            "stt_model.rnnt_joint.enc.weight",
            safetensors::Dtype::BF16,
            [0x4040, 0xc000],
            false,
        ),
        (
            Component::FunctionHead,
            "stt_model.function_head.weight",
            safetensors::Dtype::F16,
            [0x4200, 0xc000],
            false,
        ),
    ] {
        let encoded: Vec<_> = encoded.into_iter().flat_map(u16::to_le_bytes).collect();
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "physical".into(),
                dtype,
                vec![1, 2],
                encoded.clone(),
            )])
            .unwrap(),
        );
        let provenance = source.source_provenance("physical").unwrap();
        let recipe = DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source(
                "physical",
                TensorSelection::Full,
            )),
            shape: vec![2],
        };
        let parameter = ParameterAdmission {
            linear_format: eredu_checkpoint::LinearFormat::Dense,
            name: name.into(),
            component,
            artifact: PhysicalArtifactId::Primary,
            output: recipe.infer(source.as_ref()).unwrap(),
            stored_bytes: 4,
            conversion_workspace_bytes: 0,
            // Encoded source (4) plus conservative reshape overlap (4 + 4).
            materialization_peak_bytes: 12,
            recipe: recipe.clone(),
        };
        let parameters = BTreeMap::from([(name.into(), parameter)]);
        let unit = project_unit(&parameters, source, component, &normalization(name)).unwrap();
        let prepared = &unit.parameters()[name];
        assert_eq!(prepared.output.shape, [2]);
        assert_eq!(prepared.stored_bytes, 4);
        assert_eq!(prepared.conversion_workspace_bytes, 0);
        assert_eq!(prepared.artifact, PhysicalArtifactId::Primary);
        assert_eq!(
            unit.source().source_provenance("physical").unwrap(),
            provenance
        );
        let lease = unit
            .source()
            .acquire_lease(TensorReadRequest {
                key: "physical".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(lease.encoded_bytes().unwrap(), encoded);
        if promotes {
            assert_eq!(prepared.output.dtype, RecipeDtype::F32);
            assert_eq!(prepared.output.byte_len, 8);
            // Encoded source (4), half input (4), expanded output (8).
            assert_eq!(prepared.materialization_peak_bytes, 16);
            assert_eq!(
                unit.bindings()[0].recipe(),
                Some(&DerivedWeightRecipe::Cast {
                    input: Box::new(recipe),
                    dtype: RecipeDtype::F32,
                })
            );
        } else {
            assert_eq!(prepared.output, parameters[name].output);
            assert_eq!(prepared.materialization_peak_bytes, 12);
            assert_eq!(unit.bindings()[0].recipe(), Some(&recipe));
        }
    }
}
