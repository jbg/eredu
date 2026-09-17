use super::*;

#[test]
fn cold_destinations_preserve_packed_geometry_and_companions_without_payload_reads() {
    let mut config = config("llama", true);
    config["hidden_size"] = 32.into();
    config["intermediate_size"] = 64.into();
    config["head_dim"] = 16.into();
    config["num_hidden_layers"] = 2.into();
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    for quantization in [
        None,
        Some(eredu_core::QuantizationRequest::Affine {
            group_size: 32,
            bits: 4,
        }),
    ] {
        let plan = prepared_adapter::plan(quantization).with_residency(
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: None,
                host_budget_bytes: None,
            },
        );
        let sources = prepared_adapter::prepare(
            &inspection,
            &plan,
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let before = sources.target().source_diagnostics().unwrap();
        let context = WorkspaceContext::new(Facts::default());
        let destinations =
            eredu_architectures::prepared_execution::project_replicated_text_binding_destinations(
                &sources, &context,
            )
            .unwrap();
        assert_eq!(destinations.units().len(), 2);
        for (ordinal, unit) in destinations.units().iter().enumerate() {
            let prefix = format!("model.layers.{ordinal}.self_attn.q_proj");
            let weight = &unit[&format!("{prefix}.weight")];
            if quantization.is_some() {
                assert_eq!(weight.shape(), &[32, 4]);
                assert_eq!(weight.dtype(), WorkspaceDtype::Uint32);
                assert_eq!(unit[&format!("{prefix}.scales")].shape(), &[32, 1]);
                assert_eq!(unit[&format!("{prefix}.biases")].shape(), &[32, 1]);
            } else {
                assert_eq!(weight.shape(), &[32, 32]);
                assert_eq!(weight.dtype(), WorkspaceDtype::Float32);
                assert!(!unit.contains_key(&format!("{prefix}.scales")));
            }
        }
        let after = sources.target().source_diagnostics().unwrap();
        assert_eq!(before.physical_reads, after.physical_reads);
        assert_eq!(before.physical_read_bytes, after.physical_read_bytes);
        assert_eq!(before.payload_shard_paths, after.payload_shard_paths);
    }
}
