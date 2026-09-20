use super::*;

fn residencies() -> [eredu_core::ResidencyPlan; 3] {
    [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(1 << 20),
            host_budget_bytes: Some(1 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 20,
            host_budget_bytes: 1 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ]
}
#[test]
fn conditional_qwen_actual_prepared_sources_bind_group1_readout_and_reject_media_origins() {
    for hybrid in [false, true] {
        for routed in [false, true] {
            for edge in [false, true] {
                let f = Fixture::new(configuration(hybrid, routed, edge, false), hybrid);
                let c = NumericContext::default();
                let (_, d, paths) = f.model(&c);
                let (_, _, foreign_paths) = f.model(&c);
                let inspection =
                    eredu_architectures::configuration::inspect_artifact(f.artifact.path())
                        .unwrap();
                for residency in residencies() {
                    let sources = prepared_adapter::prepare(
                        &inspection,
                        &prepared_adapter::plan(None).with_residency(residency),
                        &prepared_adapter::NumericPreparationProvider { addressable: false },
                    )
                    .unwrap();
                    let discovery = sources.prepare_discovery(
                        ObservationMechanisms {
                            activation_tensors: true,
                            floating_to_f32: true,
                            ..Default::default()
                        },
                        CaptureCapabilities {
                            transformations: vec![
                                CaptureTransformKind::FullTensor,
                                CaptureTransformKind::Preview,
                            ],
                            ..Default::default()
                        },
                    );
                    let before = sources.target().source_diagnostics().unwrap();
                    for declaration in &d {
                        for preview in [false, true] {
                            let source =
                                admission(&discovery, &[declaration.path()], preview, true);
                            let selected = discovery
                                .prepare_capture_selection(&source, paths.source())
                                .unwrap();
                            assert!(selected.source().same_storage(&source));
                            assert!(selected.paths().same_storage(paths.source()));
                            assert_eq!(selected.declaration(0).unwrap(), Some(declaration));
                            let mut geometry = request(4);
                            geometry.prefill_chunk_positions = 2;
                            geometry.output = selected.physical_output(OutputDemand::LastPosition);
                            let expected = if declaration.readout_stage() == Stage::BeforeReadout {
                                OutputDemand::LastPosition
                            } else {
                                OutputDemand::Sequence
                            };
                            assert_eq!(geometry.output, expected);
                            assert_eq!(
                                selected.bind_geometry(geometry).unwrap().geometry(),
                                geometry
                            );
                            if expected == OutputDemand::Sequence {
                                let mut short = geometry;
                                short.output = OutputDemand::LastPosition;
                                assert!(matches!(
                                    selected.bind_geometry(short),
                                    Err(SelectionError::Readout)
                                ));
                            }
                            let mut changed = geometry;
                            changed.cached_positions += 1;
                            assert!(selected.bind_geometry(changed).is_err());
                            let foreign =
                                admission(&discovery, &[declaration.path()], preview, true);
                            assert!(matches!(
                                selected.validate_sources(&foreign, paths.source()),
                                Err(SelectionError::Identity)
                            ));
                            assert!(matches!(
                                selected.validate_sources(&source, foreign_paths.source()),
                                Err(SelectionError::Identity)
                            ));
                        }
                    }
                    let capture = discovery.capture().unwrap();
                    let projector = capture
                        .catalog
                        .get(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH)
                        .unwrap();
                    assert!(projector
                        .requirements
                        .contains(&eredu_core::ObservationRequirement::MediaInput));
                    for point in &capture.catalog.points {
                        if point
                            .requirements
                            .contains(&eredu_core::ObservationRequirement::MediaInput)
                        {
                            assert!(paths.source().prefill_observation(&point.path).is_none());
                        }
                    }
                    assert!(paths
                        .source()
                        .prefill_observation(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH)
                        .is_none());
                    let ordinary = admission(&discovery, &["readout.embedding"], false, true);
                    let invocation = SharedCapturePlan::new(
                        ordinary
                            .admission()
                            .plan()
                            .clone()
                            .admit_invocations(
                                &capture.catalog,
                                &capture.support,
                                &capture.support.capture,
                                CaptureInvocationBounds {
                                    batch: 1,
                                    max_sequence: 5,
                                    max_context: Some(16),
                                    max_predictions: 4,
                                },
                            )
                            .unwrap(),
                    );
                    assert!(invocation.admission().text_origin().is_none());
                    assert!(matches!(
                        discovery.prepare_capture_selection(&invocation, paths.source()),
                        Err(SelectionError::Identity)
                    ));
                    assert_eq!(
                        before.physical_reads,
                        sources
                            .target()
                            .source_diagnostics()
                            .unwrap()
                            .physical_reads
                    );
                }
            }
        }
    }
}
#[test]
fn conditional_qwen_rejects_foreign_or_invalidated_runtime_before_any_callback() {
    for hybrid in [false, true] {
        for routed in [false, true] {
            let f = Fixture::new(configuration(hybrid, routed, true, false), hybrid);
            let c = NumericContext {
                bind_checkpoint_values: true,
                ..Default::default()
            };
            let (mut model, d, paths) = f.model(&c);
            let (_foreign_model, _, foreign) = f.model(&c);
            let mut state = f.state();
            let before = state.clone();
            let mut rows = Rows::new(&d);
            assert!(matches!(
                model.run(
                    &[1, 3],
                    &mut state,
                    &c,
                    &mut rows,
                    &foreign,
                    OutputDemand::Sequence
                ),
                Err(PreparedLayeredObservationError::BindingMismatch)
            ));
            assert!(rows.values.is_empty());
            same_state(&state, &before);
            assert!(matches!(
                model.run(
                    &[1, 3],
                    &mut state,
                    &c,
                    &mut rows,
                    &paths,
                    OutputDemand::LastPosition
                ),
                Err(PreparedLayeredObservationError::ReadoutDemand)
            ));
            assert!(rows.values.is_empty());
            same_state(&state, &before);
            model.invalidate();
            assert!(matches!(
                model.run(
                    &[1, 3],
                    &mut state,
                    &c,
                    &mut rows,
                    &paths,
                    OutputDemand::Sequence
                ),
                Err(PreparedLayeredObservationError::BindingMismatch)
            ));
            assert!(rows.values.is_empty());
            same_state(&state, &before);
        }
    }
}
#[test]
fn conditional_hybrid_target_declarations_preserve_all_existing_mtp_hook_gates() {
    use eredu_runtime::inspection::ObservationHookSite;
    for routed in [false, true] {
        for width in [1, 4] {
            let mut value = configuration(true, routed, false, false);
            value["text_config"]["mtp_num_hidden_layers"] = 1.into();
            value["text_config"]["linear_conv_kernel_dim"] = width.into();
            let a = Hybrid::new(
                hybrid::model_args_from_config_value(&value).unwrap(),
                &NumericContext::default(),
            )
            .unwrap();
            let d = tensor_row_declarations(<Hybrid as LayeredArchitecture<NumericBackend, State>>::prefill_observation_declarations(&a, None).unwrap());
            assert!(d.len() >= 18);
            let prediction =
                <Hybrid as LayeredArchitecture<NumericBackend, State>>::unit_path(&a, 2, 0, None)
                    .unwrap();
            assert!(d.iter().all(|d| !d.path().starts_with(&prediction)));
            let hooks = [
                <Hybrid as LayeredArchitecture<NumericBackend, State>>::observation_hooks(&a),
                <Hybrid as eredu_runtime::ParallelLayeredArchitecture<NumericBackend, State>>::parallel_observation_hooks(&a),
                <Hybrid as eredu_runtime::PartitionedLayeredArchitecture<NumericBackend, State>>::partition_observation_hooks(&a, false),
                <Hybrid as eredu_runtime::PartitionedLayeredArchitecture<NumericBackend, State>>::partition_observation_hooks(&a, true),
            ];
            for hooks in hooks {
                for site in [
                    ObservationHookSite::Input,
                    ObservationHookSite::Unit,
                    ObservationHookSite::Readout,
                ] {
                    assert!(!hooks.supports(site));
                }
            }
        }
    }
}
