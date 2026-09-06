//! Independent proof of per-profile visitor conversion without identity boilerplate.

use super::*;
use std::cell::RefCell;
use std::rc::Rc;

struct AttentionVisitor<'a> {
    original: NumericReplicatedVisitor<'a>,
    events: Rc<RefCell<Vec<&'static str>>>,
}

impl
    ReplicatedTextArchitectureVisitor<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
    > for AttentionVisitor<'_>
{
    type Output = NumericReplicatedRun;
    type Error = String;

    fn construction_started(&mut self) {
        self.events.borrow_mut().push("construction_started");
        self.original.construction_started();
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        checkpoint: SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
                Error = Error,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        self.events.borrow_mut().push("visit");
        self.original.visit(prepared, checkpoint)
    }
}

#[test]
fn one_profile_conversion_changes_only_its_typed_visitor() {
    use safetensors::tensor::{serialize_to_file, TensorView};

    for (layer_types, overridden) in [
        (["full_attention", "full_attention"], true),
        (["conv", "conv"], false),
    ] {
        let mut config = heterogeneous_replicated_configs().remove(0);
        config["layer_types"] = serde_json::json!(layer_types);
        let artifact = tempfile::tempdir().unwrap();
        std::fs::write(
            artifact.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let tensors = required_safetensors_parameters(&config)
            .into_iter()
            .map(|(name, shape)| {
                let bytes = vec![0_u8; shape.iter().product::<usize>() * 4];
                (name, shape, bytes)
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &artifact.path().join("model.safetensors")).unwrap();
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        assert_eq!(
            requirements.state_access(),
            if overridden {
                eredu_runtime::ReplicatedTextStateAccess::KeyValue
            } else {
                eredu_runtime::ReplicatedTextStateAccess::Fixed
            }
        );
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        );
        let capabilities = eredu_runtime::synthesize_replicated_text_capabilities(
            &requirements,
            &request,
            &NumericMechanismSupport::default(),
        );
        let selected = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request,
            &capabilities,
        )
        .unwrap();
        let checkpoint: SharedCheckpointSource = Arc::new(
            eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
        );
        let context = NumericContext::default();
        let tokens = NumericTensor::token_ids(&[1, 3, 2]);
        let events = Rc::new(RefCell::new(Vec::new()));
        let conversion_events = Rc::clone(&events);
        let visitor = SharedReplicatedTextVisitor::<NumericReplicatedStateProfiles, _>::new(
            NumericReplicatedVisitor {
                context: &context,
                tokens: &tokens,
                construction_started: false,
            },
        )
        .with_attention_conversion(move |original| {
            conversion_events.borrow_mut().push("conversion");
            AttentionVisitor {
                original,
                events: conversion_events,
            }
        });
        reset_reference_stage_evidence("SafeTensors");
        let output = dispatch_replicated_text_architecture(
            inspection.architecture_plan(),
            selected,
            checkpoint,
            &context,
            visitor,
        )
        .unwrap();
        assert_eq!(output.outputs.len(), 3);
        assert_eq!(
            events.borrow().as_slice(),
            if overridden {
                &["conversion", "construction_started", "visit"][..]
            } else {
                &[]
            }
        );
        assert_eq!(
            last_reference_stage_evidence().stages,
            [
                "construction_started",
                "typed_architecture",
                "materialization",
                "session_constructed",
                "prefill",
                "decode",
                "decode",
                "report",
            ]
        );
    }
}
