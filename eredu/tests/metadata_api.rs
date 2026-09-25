//! Application coverage: all eredu contracts are imported through the facade.
use eredu::api::{
    inspect_model_metadata, ArtifactMetadata, BackendId, CheckpointMetadata,
    MetadataInspectionError, MetadataInspectionOptions, MetadataProvenance, SafetensorsHeader,
};
use std::collections::BTreeMap;

fn metadata() -> ArtifactMetadata {
    let shapes: &[(&str, &[usize])] = &[
        ("model.embed_tokens.weight", &[32, 32]),
        ("lm_head.weight", &[32, 32]),
        ("model.norm.weight", &[32]),
        ("model.layers.0.input_layernorm.weight", &[32]),
        ("model.layers.0.post_attention_layernorm.weight", &[32]),
        ("model.layers.0.self_attn.q_proj.weight", &[32, 32]),
        ("model.layers.0.self_attn.k_proj.weight", &[16, 32]),
        ("model.layers.0.self_attn.v_proj.weight", &[16, 32]),
        ("model.layers.0.self_attn.o_proj.weight", &[32, 32]),
        ("model.layers.0.mlp.gate_proj.weight", &[64, 32]),
        ("model.layers.0.mlp.up_proj.weight", &[64, 32]),
        ("model.layers.0.mlp.down_proj.weight", &[32, 64]),
    ];
    let mut offset = 0;
    let entries = shapes
        .iter()
        .map(|(name, shape)| {
            let size = shape.iter().product::<usize>() * 4;
            let entry = format!(
                r#""{name}":{{"dtype":"F32","shape":{shape:?},"data_offsets":[{offset},{}]}}"#,
                offset + size
            );
            offset += size;
            entry
        })
        .collect::<Vec<_>>();
    let header = format!("{{{}}}", entries.join(",")).into_bytes();
    let bytes = [(header.len() as u64).to_le_bytes().as_slice(), &header].concat();
    let file_len = bytes.len() as u64 + offset as u64;
    ArtifactMetadata {
        provenance: MetadataProvenance { source: "metadata-only/llama-fixture".into(), revision: "pinned-fixture".into() },
        checkpoint: CheckpointMetadata::SafeTensors {
            config: br#"{"model_type":"llama","hidden_size":32,"num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"rms_norm_eps":0.00001,"vocab_size":32,"max_position_embeddings":4096,"rope_theta":10000.0,"tie_word_embeddings":false}"#.to_vec(),
            index: None,
            headers: vec![SafetensorsHeader {member: "model.safetensors".into(), bytes, file_len}],
        },
        sidecars: BTreeMap::new(),
    }
}

#[test]
fn unknown_backend_is_explicit_and_never_falls_back() {
    let options = MetadataInspectionOptions::new(BackendId::new("unknown-test-backend").unwrap());
    let error = inspect_model_metadata(&metadata(), &options).unwrap_err();
    assert_eq!(
        error,
        MetadataInspectionError::UnknownBackend {
            backend: options.backend().clone()
        }
    );
}

#[test]
#[cfg(not(feature = "mlx"))]
fn known_backend_requires_its_compiled_feature() {
    let options = MetadataInspectionOptions::new(BackendId::new("mlx").unwrap());
    assert_eq!(
        inspect_model_metadata(&metadata(), &options).unwrap_err(),
        MetadataInspectionError::BackendNotCompiled {
            backend: options.backend().clone(),
            feature: "mlx",
        }
    );
}

#[test]
#[cfg(feature = "mlx")]
fn explicit_backend_inspection_and_forecast_use_only_the_facade() {
    use eredu::api::{
        forecast_inspected_generation, GenerationMemoryOptions, GenerationMemoryPlacement,
        InputTokenCount, NormalizedLoadRequest, ParameterConversionRetentionPolicy,
    };
    let request = NormalizedLoadRequest::default()
        .with_parameter_conversion_retention(Some(ParameterConversionRetentionPolicy::Unlimited));
    let options = MetadataInspectionOptions::new(BackendId::new("mlx").unwrap())
        .with_load_request(request.clone());
    assert_eq!(options.load_request(), &request);
    let inspection = inspect_model_metadata(&metadata(), &options).unwrap();
    assert!(
        inspection.report().is_compatible(),
        "{:?}",
        inspection.report().issues
    );
    assert!(!inspection.report().is_loadable());
    assert_eq!(
        inspection
            .selected()
            .unwrap()
            .preparation()
            .execution()
            .text_realization()
            .parameter_conversion_retention(),
        Some(ParameterConversionRetentionPolicy::Unlimited),
    );
    assert_eq!(
        inspection
            .report()
            .metadata_provenance
            .as_ref()
            .unwrap()
            .revision,
        "pinned-fixture"
    );
    for positions in [4, 128, 2_000] {
        let mut memory = GenerationMemoryOptions::new(
            InputTokenCount::text(positions),
            GenerationMemoryPlacement::Unified,
        );
        memory.max_output_tokens = Some(32);
        memory.budget.application_limit_bytes = Some(8 << 30);
        let forecast =
            forecast_inspected_generation(&inspection, &memory, &Default::default()).unwrap();
        assert_eq!(forecast.request.input.model_positions, positions);
        assert!(
            forecast.estimate.domains[0]
                .additional_generation_peak
                .lower_bytes
                > 0
        );
        assert!(forecast
            .request
            .domains
            .iter()
            .flat_map(|domain| &domain.executions)
            .filter_map(|execution| execution.execution_topology.as_ref())
            .all(|topology| topology.projection_storage.is_empty()));
    }
}

#[test]
#[cfg(feature = "mlx")]
fn selected_backend_reports_invalid_metadata_without_an_execution_selection() {
    use eredu::api::{InspectionSeverity, ModelInspectionReport};

    let options = MetadataInspectionOptions::new(BackendId::new("mlx").unwrap());
    let mut metadata = metadata();
    let CheckpointMetadata::SafeTensors { headers, .. } = &mut metadata.checkpoint else {
        unreachable!()
    };
    headers.clear();
    let inspection = inspect_model_metadata(&metadata, &options).unwrap();
    assert!(inspection.selected().is_none());
    assert!(!inspection.report().is_compatible());
    let report: &ModelInspectionReport = inspection.report();
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.severity == InspectionSeverity::Error));
}
