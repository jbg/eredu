use super::*;
use eredu_architectures::{gpt_oss, k2_horizon as k2};
use eredu_runtime::{layered::PrefillObservationDeclaration, StateLayout};

type CustomState = DeviceState<NumericBackend, NumericHybridLayerState>;
type CustomModel<C, P> = decoder::LayeredModel<NumericBackend, C, P>;

// Every declared value, including effective companions, must be emitted by the
// real component/traversal observer. This collector never manufactures aliases.
struct DeclaredRows<'a> {
    declarations: &'a [PrefillObservationDeclaration],
    values: BTreeMap<String, NumericTensor>,
}
impl<'a> DeclaredRows<'a> {
    fn new(declarations: &'a [PrefillObservationDeclaration]) -> Self {
        Self {
            declarations,
            values: BTreeMap::new(),
        }
    }
    fn record(&mut self, path: &str, value: &NumericTensor) {
        if self.declarations.iter().any(|d| d.path() == path) {
            assert!(
                self.values.insert(path.into(), value.clone()).is_none(),
                "duplicate {path}"
            );
        }
    }
    fn assert_complete(&self, positions: i32) {
        assert_eq!(self.values.len(), self.declarations.len());
        for declaration in self.declarations {
            let path = declaration.path();
            let value = self
                .values
                .get(path)
                .unwrap_or_else(|| panic!("missing {path}"));
            assert_eq!(declaration.sequence_axis(), 1);
            assert_eq!(value.shape.len(), 3, "{path}");
            assert_eq!(value.shape[..2], [1, positions], "{path}");
            assert!(value.data.iter().all(|v| v.is_finite()), "finite {path}");
            assert!(value.data.iter().any(|v| v.abs() > 1e-6), "nonzero {path}");
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for DeclaredRows<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.record(path, value);
        Ok(())
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &NumericTensor,
        _: &GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<NumericTensor, Error>,
    ) -> Result<(), Error> {
        assert!(self.declarations.iter().all(|d| d.path() != path));
        Ok(())
    }
}

fn assert_custom_state(actual: &CustomState, expected: &CustomState) {
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (layer, (a, b)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert!(a.fixed.is_empty() && b.fixed.is_empty());
        assert!(a.compressed.is_none() && b.compressed.is_none());
        assert!(a.pooling.is_none() && b.pooling.is_none());
        let (a, b) = (a.attention.as_ref().unwrap(), b.attention.as_ref().unwrap());
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.window, b.window);
        // Default NumericContext does not enable the separate paged-history
        // fixture. Compare all actual retained KV rows, including window tails.
        assert!(a.attention_history.is_none() && b.attention_history.is_none());
        for (name, a, b) in [("keys", &a.keys, &b.keys), ("values", &a.values, &b.values)] {
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(a), Some(b)) = (a, b) {
                assert_tensor_close(a, b, &format!("layer {layer} {name}"));
            }
        }
    }
}

fn compare_equations<C, P>(args: C, layout: StateLayout)
where
    C: decoder::Config,
    P: decoder::BlockFactory<NumericBackend, C>,
{
    let vocabulary = args.vocabulary_size();
    let context = NumericContext::default();
    let architecture = CustomModel::<C, P>::new(args, &context).unwrap();
    let declarations = <CustomModel<C, P> as LayeredArchitecture<NumericBackend, CustomState>>::
        prefill_observation_declarations(&architecture).unwrap();
    let mut model = ResidentRuntime::new(architecture, &context).unwrap();
    let paths = model.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        assert_eq!(
            paths.source().prefill_observation(declaration.path()),
            Some(declaration)
        );
    }
    let mut full_state = CustomState::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    model
        .forward(
            decoder::LayeredInput {
                tokens: &NumericTensor::token_ids(&[4, 2]),
                mask: None,
            },
            &mut full_state,
            &context,
        )
        .unwrap();
    let mut chunk_state = full_state.clone();
    let mut full = DeclaredRows::new(&declarations);
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            decoder::LayeredInput {
                tokens: &NumericTensor::token_ids(&[1, 3, 5, 2, 6]),
                mask: None,
            },
            &mut full_state,
            &context,
            &mut full,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
    let scores = eredu_runtime::observe_model_logits(&mut full, &scores.unwrap()).unwrap();
    assert_eq!(scores.shape, [1, 5, vocabulary]);
    full.assert_complete(5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = DeclaredRows::new(&declarations);
        let (scores, _) = model
            .forward_with_prepared_observer_and_context_with_readout(
                decoder::LayeredInput {
                    tokens: &NumericTensor::token_ids(ids),
                    mask: None,
                },
                &mut chunk_state,
                &context,
                &mut rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
        let scores = eredu_runtime::observe_model_logits(&mut rows, &scores.unwrap()).unwrap();
        assert_eq!(scores.shape, [1, ids.len() as i32, vocabulary]);
        rows.assert_complete(ids.len() as i32);
        for declaration in &declarations {
            assembled
                .entry(declaration.path().into())
                .or_default()
                .extend_from_slice(&rows.values[declaration.path()].data);
        }
    }
    assert_eq!(assembled.len(), declarations.len());
    for declaration in &declarations {
        let path = declaration.path();
        let expected = &full.values[path];
        assert_tensor_close(
            &NumericTensor::new(expected.shape.clone(), assembled.remove(path).unwrap()),
            expected,
            path,
        );
    }
    assert_custom_state(&chunk_state, &full_state);
    for id in [7, 8, 9] {
        let tokens = NumericTensor::token_ids(&[id]);
        let mut expected = DeclaredRows::new(&declarations);
        let mut actual = DeclaredRows::new(&declarations);
        for (state, rows) in [
            (&mut full_state, &mut expected),
            (&mut chunk_state, &mut actual),
        ] {
            let (scores, _) = model
                .forward_with_prepared_observer_and_context_with_readout(
                    decoder::LayeredInput {
                        tokens: &tokens,
                        mask: None,
                    },
                    state,
                    &context,
                    rows,
                    &paths,
                    OutputDemand::Sequence,
                )
                .unwrap();
            let scores = eredu_runtime::observe_model_logits(rows, &scores.unwrap()).unwrap();
            assert_eq!(scores.shape, [1, 1, vocabulary]);
            rows.assert_complete(1);
        }
        for declaration in &declarations {
            let path = declaration.path();
            assert_tensor_close(&actual.values[path], &expected.values[path], path);
        }
        assert_custom_state(&chunk_state, &full_state);
    }
    assert!(full_state
        .as_ref()
        .iter()
        .all(|state| state.position() == 10));
}

fn gpt_configuration(yarn: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"],
        "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "vocab_size": 17, "num_local_experts": 3, "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001, "sliding_window": 2,
        "max_position_embeddings": 64, "rope_theta": 150000.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "quantization_config": {"quant_method": "mxfp4"}, "swiglu_limit": 7.0
    });
    if yarn {
        config["num_experts_per_tok"] = 2.into();
        config["rope_scaling"] = serde_json::json!({
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 16, "beta_fast": 8.0,
            "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 0.0
        });
    }
    config
}
fn k2_configuration(kind: &str) -> serde_json::Value {
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/k2_horizon/reference.json"
    ))
    .unwrap();
    let mut config = reference[kind]["config"].clone();
    config["num_hidden_layers"] = 2.into();
    config
}

#[test]
fn gpt_oss_declared_rows_match_sink_attention_and_biased_experts_across_chunks() {
    for yarn in [false, true] {
        let args = gpt_oss::model_args_from_config_value(&gpt_configuration(yarn)).unwrap();
        let layout = gpt_oss::state_layout(&args).unwrap();
        compare_equations::<_, gpt_oss::GptOssBlockFactory>(args, layout);
    }
}
#[test]
fn k2_dense_declared_rows_match_grouped_norm_and_gated_attention_across_chunks() {
    for kind in ["dense", "grouped"] {
        let mut config = k2_configuration(kind);
        if kind == "grouped" {
            config["use_sliding_window"] = true.into();
            config["sliding_window"] = 2.into();
            config["attention_gate_func"] = "silu".into();
            config["tie_word_embeddings"] = true.into();
        }
        let args = k2::model_args_from_config_value(&config).unwrap();
        let layout = decoder::state_layout(&args).unwrap();
        compare_equations::<_, k2::DenseBlockFactory>(args.clone(), layout.clone());
        compare_equations::<_, k2::BlockFactory>(args, layout);
    }
}
#[test]
fn k2_routed_declared_rows_match_moe_mova_and_shared_experts_across_chunks() {
    for kind in ["moe", "mova"] {
        let args = k2::model_args_from_config_value(&k2_configuration(kind)).unwrap();
        let layout = decoder::state_layout(&args).unwrap();
        compare_equations::<_, k2::BlockFactory>(args, layout);
    }
    let mut config = k2_configuration("mova");
    config["mova_num_experts_per_tok"] = 1.into();
    config["router_score_func"] = "softmax".into();
    config["norm_topk_prob"] = true.into();
    config["moe_gate_bias"] = false.into();
    config["num_shared_experts"] = 0.into();
    config["use_sliding_window"] = true.into();
    config["sliding_window"] = 2.into();
    let args = k2::model_args_from_config_value(&config).unwrap();
    let layout = decoder::state_layout(&args).unwrap();
    compare_equations::<_, k2::BlockFactory>(args, layout);
}

// Same schema-driven SafeTensors fixture policy used by the existing routed
// conformance entry. Preserve exact U8 MXFP4 storage instead of writing F32 in
// byte slots; this artifact serves cold discovery, not the numerical oracle.
fn gpt_artifact(config: &serde_json::Value) -> tempfile::TempDir {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(config)
        .unwrap();
    let checkpoint = resolved
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let constraints = checkpoint.common_tensors.iter().chain(
        checkpoint
            .layout_groups
            .iter()
            .filter(|g| g.required)
            .filter_map(|g| g.variants.first())
            .flat_map(|v| v.tensors.iter()),
    );
    let mut tensors = BTreeMap::<String, (Vec<usize>, Dtype, Vec<u8>)>::new();
    for tensor in constraints
        .filter(|t| t.requirement == eredu_checkpoint::schema::TensorRequirement::Required)
    {
        let count = tensor.shape.iter().product::<usize>();
        let (dtype, bytes) = if tensor.dtype.accepts(&eredu_checkpoint::StoredDtype::F32) {
            let value = parameter(
                &ParameterSpec::trainable(&tensor.key).unwrap(),
                tensor.shape.iter().map(|&d| d as i32).collect(),
                tensor.key.contains("norm"),
            );
            (
                Dtype::F32,
                value.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
        } else {
            assert!(tensor.dtype.accepts(&eredu_checkpoint::StoredDtype::U8));
            (
                Dtype::U8,
                vec![
                    if tensor.key.ends_with("scales") {
                        127
                    } else {
                        0x22
                    };
                    count
                ],
            )
        };
        assert!(tensors
            .insert(tensor.key.clone(), (tensor.shape.clone(), dtype, bytes))
            .is_none());
    }
    let views = tensors
        .iter()
        .map(|(name, (shape, dtype, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &artifact.path().join("model.safetensors")).unwrap();
    artifact
}

fn check_binding<C, P>(artifact: &std::path::Path, args: C)
where
    C: decoder::Config,
    P: decoder::BlockFactory<NumericBackend, C>,
{
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let before = sources.target().source_diagnostics().unwrap();
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
    let context = NumericContext::default();
    let architecture = CustomModel::<C, P>::new(args, &context).unwrap();
    let declarations = <CustomModel<C, P> as LayeredArchitecture<NumericBackend, CustomState>>::prefill_observation_declarations(&architecture).unwrap();
    let runtime =
        ResidentRuntime::<_, NumericBackend, CustomState>::new(architecture, &context).unwrap();
    let paths = runtime.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        for preview in [false, true] {
            let source = admission(&discovery, &[declaration.path()], preview, true);
            let selected = discovery
                .prepare_capture_selection(&source, paths.source())
                .unwrap();
            assert!(selected.source().same_storage(&source));
            assert!(selected.paths().same_storage(paths.source()));
            assert_eq!(selected.declaration(0).unwrap(), Some(declaration));
            let mut geometry = request(4);
            geometry.prefill_chunk_positions = 2;
            geometry.output = selected.physical_output(OutputDemand::LastPosition);
            assert_eq!(
                geometry.output,
                if declaration.readout_stage() == Stage::BeforeReadout {
                    OutputDemand::LastPosition
                } else {
                    OutputDemand::Sequence
                }
            );
            let bound = selected.bind_geometry(geometry).unwrap();
            assert_eq!(bound.geometry(), geometry);
            assert!(std::ptr::eq(bound.selection(), &selected));
            if declaration.readout_stage() != Stage::BeforeReadout {
                geometry.output = OutputDemand::LastPosition;
                assert!(matches!(
                    selected.bind_geometry(geometry),
                    Err(SelectionError::Readout)
                ));
            }
            let independent = admission(&discovery, &[declaration.path()], preview, true);
            assert!(matches!(
                selected.validate_sources(&independent, paths.source()),
                Err(SelectionError::Identity)
            ));
        }
    }
    // Router and component internals are not declared by the outer factory.
    let catalog = discovery.capture().unwrap();
    let internal = catalog
        .catalog
        .points
        .iter()
        .find(|point| {
            point.prefill
                && point.axes.as_ref().is_some_and(|axes| axes.len() == 3)
                && paths.source().prefill_observation(&point.path).is_none()
                && catalog.support.points.iter().any(|s| {
                    s.path == point.path && s.prefill == ObservationSupportStatus::Supported
                })
        })
        .unwrap();
    let source = admission(&discovery, &[&internal.path], false, true);
    assert!(matches!(
        discovery.prepare_capture_selection(&source, paths.source()),
        Err(SelectionError::Undeclared { index: 0 })
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
#[test]
fn gpt_oss_actual_discovery_and_shared_paths_bind_all_declared_readout_stages() {
    let config = gpt_configuration(true);
    let artifact = gpt_artifact(&config);
    check_binding::<_, gpt_oss::GptOssBlockFactory>(
        artifact.path(),
        gpt_oss::model_args_from_config_value(&config).unwrap(),
    );
}
#[test]
fn k2_actual_discovery_and_shared_paths_bind_dense_and_routed_readout_stages() {
    for kind in ["dense", "grouped", "moe", "mova"] {
        let config = k2_configuration(kind);
        let (artifact, _) = k2_horizon::checkpoint_fixture(&config, 1.0);
        let args = k2::model_args_from_config_value(&config).unwrap();
        check_binding::<_, k2::BlockFactory>(artifact.path(), args.clone());
        if matches!(kind, "dense" | "grouped") {
            check_binding::<_, k2::DenseBlockFactory>(artifact.path(), args);
        } else {
            assert!(
                CustomModel::<_, k2::DenseBlockFactory>::new(args, &NumericContext::default())
                    .is_err()
            );
        }
    }
}
