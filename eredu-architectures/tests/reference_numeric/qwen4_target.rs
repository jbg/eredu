//! Target composition through the shared layered driver and exact request transport.
use super::qwen4_recurrent::Parameters;
use super::*;
use eredu_architectures::qwen4_exp::{
    config::LayerKind,
    input::{RequestBoundarySchema, RequestContext, RequestError},
    qsa::QsaExecutionLimits,
    target::{BoundTargetSpec, TargetInput, TargetLimits, TargetModel, TargetSpec, Unit, UnitSpec},
};
use eredu_runtime::{
    AppendStreamLimits, ArchitectureBoundary, ArchitectureParameters, OriginalTokenIds,
    ParameterBankAccess, ParameterProviders, ResidentAppendStream, ResidentExpertProvider,
    ResidentUnitWindow, RowLookupError, RowLookupProvider, RowLookupSpec, TokenVisibility,
};
use std::sync::Arc;
type State = DeviceState<NumericBackend, NumericHybridLayerState>;

#[test]
fn qwen4_prepared_sources_bind_nonzero_target_without_cross_owner_reads() {
    use eredu_architectures::qwen4_exp::{
        checkpoint::schema::SafetensorsEncoding,
        prepared::{PreparedParameters, SafetensorsTargetPlan},
    };
    use eredu_checkpoint::store::{SafetensorsWeightStore, SharedCheckpointSource};
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let spec = specification();
    let mut original = TargetModel::new(bind_spec(spec.clone()), &ctx).unwrap();
    let mut parameters = Parameters::default();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut original).visit_parameters_mut(&mut parameters);
    let original_units: Vec<_> = (0..spec.units.len())
        .map(|i| {
            let mut unit = original.construct_unit(i, &ctx).unwrap();
            unit.visit_parameters_mut(&mut parameters);
            unit
        })
        .collect();
    let mut values = table_values();
    for shard in 0..2 {
        let table = values
            .iter_mut()
            .find(|(name, _, _, _)| name.ends_with(&format!("shard_{shard}.weight")))
            .unwrap();
        table.3 = (shard * 12..(shard + 1) * 12)
            .flat_map(|r| [(r as f32 + 0.25) / 17., -(r as f32 + 0.5) / 19.])
            .flat_map(f32::to_le_bytes)
            .collect();
    }
    for (name, value) in parameters.0 {
        let mut shape: Vec<usize> = value.shape().iter().map(|n| *n as usize).collect();
        if name.ends_with(".conv1d.weight") && shape.len() == 2 {
            shape.insert(1, 1);
        }
        // Exercise the official conditional namespace rather than identity names.
        let physical = name
            .strip_prefix("model.")
            .map_or_else(|| name.clone(), |s| format!("model.language_model.{s}"));
        values.push((
            physical,
            safetensors::Dtype::F32,
            shape,
            value.data.into_iter().flat_map(f32::to_le_bytes).collect(),
        ));
    }
    let directory = tempfile::tempdir().unwrap();
    let views = values.iter().map(|(name, dtype, shape, bytes)| {
        (
            name.as_str(),
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
        )
    });
    safetensors::tensor::serialize_to_file(
        views,
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source: SharedCheckpointSource =
        Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let header = SafetensorsTargetPlan::prepare(
        source.as_ref(),
        configuration(),
        SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap(),
    )
    .unwrap();
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    let prepared = header.bind(source.clone(), spec.limits).unwrap();
    struct Bind<'a> {
        ordinary: &'a PreparedParameters,
        experts: Vec<PreparedParameters>,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
        fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
            let name = meta.id.as_str();
            let data = if self.ordinary.recipes().contains_key(name) {
                read_owner_parameter(self.ordinary, name)
            } else {
                self.experts
                    .iter()
                    .flat_map(|e| read_owner_parameter(e, name))
                    .collect()
            };
            assert_eq!(data.len(), value.data.len(), "{name}");
            value.data = data;
        }
    }
    assert!(matches!(
        prepared.prediction(eredu_architectures::qwen4_exp::mtp::PredictionLimits {
            qsa: spec.limits.qsa,
            tile_blocks: spec.limits.tile_blocks,
            selection_workspace: spec.limits.selection_workspace,
            element: spec.limits.element,
        }),
        Err(eredu_architectures::qwen4_exp::prepared::PreparationError::MissingPrediction)
    ));
    let mut rebuilt = TargetModel::new(prepared.bound_spec().unwrap(), &ctx).unwrap();
    cold::check(&prepared, &rebuilt, source.clone(), &ctx);
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut rebuilt)
        .visit_parameters_mut(&mut Bind { ordinary: prepared.static_parameters(), experts: vec![] });
    let rebuilt_units: Vec<_> = (0..spec.units.len())
        .map(|i| {
            let mut unit = rebuilt.construct_unit(i, &ctx).unwrap();
            let experts = if matches!(spec.units[i], UnitSpec::Decoder { .. }) {
                (0..configuration().experts.count as usize)
                    .map(|e| prepared.expert(i, e).unwrap())
                    .collect()
            } else {
                vec![]
            };
            unit.visit_parameters_mut(&mut Bind {
                ordinary: prepared.unit(i).unwrap(),
                experts,
            });
            unit
        })
        .collect();
    let mut expected = LayerwiseRuntime::new(original, ResidentUnitWindow::new(original_units));
    let mut actual = LayerwiseRuntime::new(rebuilt, ResidentUnitWindow::new(rebuilt_units));
    let (ids, visible, media) = input_values();
    let run = |runtime: &mut LayerwiseRuntime<
        TargetModel<NumericBackend>,
        NumericBackend,
        State,
        ResidentUnitWindow<Unit<NumericBackend>>,
    >,
               bank| {
        let mut provider = ParameterProviders {
            grouped: ResidentExpertProvider,
            rows: Rows {
                bank,
                ..Default::default()
            },
        };
        runtime
            .forward_with_provider_and_observer_and_context(
                TargetInput {
                    position_delta: None,
                    ids: Some(OriginalTokenIds::Host(&ids)),
                    batch: 2,
                    tokens: 19,
                    embeddings: Some(&media),
                    visible: Some(TokenVisibility::Host(&visible)),
                    rotary: None,
                },
                &mut state(&spec),
                ExpertPass::Prefill,
                &mut provider,
                &ctx,
                &mut Observe::default(),
            )
            .unwrap()
            .0
    };
    assert_eq!(run(&mut actual, 1).data, run(&mut expected, 2).data);
}
fn configuration() -> eredu_architectures::qwen4_exp::config::Config {
    let mut c = super::qwen4_recurrent::config();
    c.vocabulary = 32;
    c.eos = vec![7];
    c.hidden_size = 2;
    c.layers = eredu_core::LayerSchedule::new(
        4,
        vec![
            LayerKind::Recurrent,
            LayerKind::Indexed,
            LayerKind::Recurrent,
            LayerKind::Indexed,
        ],
    )
    .unwrap();
    c.prediction = None;
    c.vision = None;
    c.media = None;
    c.attention.heads = 2;
    c.attention.kv_heads = 1;
    c.attention.head_dim = 2;
    c.attention.rotary.dimensions = 2;
    c.attention.rotary_dim = 2;
    c.attention.index_heads = 2;
    c.attention.index_head_dim = 2;
    c.attention.ratio = 3;
    c.attention.budget = 6;
    c.experts.count = 3;
    c.experts.selected = 2;
    c.experts.intermediate = 2;
    c.experts.shared_intermediate = 4;
    c.ngram.layers = vec![1];
    c.ngram.order = 3;
    c.ngram.heads = 1;
    c.ngram.embedding_dim = 4;
    if let eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors {
        shards,
        vocabulary_alignment,
        ..
    } = &mut c.ngram.source
    {
        *shards = 2;
        *vocabulary_alignment = 4;
    }
    c.ngram.kernel = 3;
    c
}
fn table_values() -> Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> {
    let root = "model.layers.1.ple.ple_embedding";
    let mut values = vec![];
    for (name, ints) in [
        (
            "layer_multipliers",
            vec![23703573157769i64, 20109073645365, 8052911324071],
        ),
        ("ngram_heads_vocab_sizes", vec![11, 13]),
        ("ngram_heads_offsets", vec![0, 11]),
    ] {
        values.push((
            format!("{root}.{name}"),
            safetensors::Dtype::I64,
            vec![ints.len()],
            ints.into_iter().flat_map(i64::to_le_bytes).collect(),
        ));
    }
    for shard in 0..2 {
        values.push((
            format!("{root}.ngram_embedding.shard_{shard}.weight"),
            safetensors::Dtype::F32,
            vec![12, 2],
            vec![0u8; 96],
        ));
    }
    values
}
fn specification() -> TargetSpec {
    specification_for(configuration())
}
fn specification_for(c: eredu_architectures::qwen4_exp::config::Config) -> TargetSpec {
    let table = RowLookupSpec {
        parameter: eredu_nn::ParameterId::new(
            "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
        )
        .unwrap(),
        bank: 2,
        unit: 1,
        rows: 24,
        dimensions: 2,
        encoding: eredu_runtime::RowEncoding::Dense,
        output_type: eredu_nn::TensorElementType::F32,
    };
    let projection = |name| {
        eredu_nn::GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(),
            None,
            dense_linear_format(),
        )
        .unwrap()
    };
    let experts = |layer, _| {
        GroupedGatedProductSpec::new(
            c.experts.count,
            c.hidden_size,
            c.experts.intermediate,
            c.hidden_size,
            eredu_nn::GatedProductPolicy::ordinary_silu(),
            eredu_nn::GatedProductGroupLayout::Packed {
                gate_up: projection(format!("model.layers.{layer}.mlp.experts.gate_up_proj")),
                down: projection(format!("model.layers.{layer}.mlp.experts.down_proj")),
            },
        )
    };
    TargetSpec::from_headers(
        c.clone(),
        &BTreeMap::from([(1, table)]),
        TargetLimits {
            history_tokens: 128,
            qsa: QsaExecutionLimits {
                batch: 2,
                tokens: 32,
                workspace_bytes: 1048576,
            },
            tile_blocks: 2,
            selection_workspace: 16384,
            invocation_tokens: 64,
            lookup_rows: 128,
            element: eredu_nn::TensorElementType::F32,
        },
        experts,
        |_| Ok(dense_linear_format()),
    )
    .unwrap()
}
fn stream_bindings(spec: &TargetSpec) -> Vec<eredu_runtime::AppendStreamBinding> {
    spec.units
        .iter()
        .enumerate()
        .flat_map(|(layer, unit)| match unit {
            UnitSpec::Decoder {
                mixer: eredu_architectures::qwen4_exp::target::MixerSpec::Indexed(a),
                ..
            } => a
                .state
                .streams()
                .into_iter()
                .map(|spec| eredu_runtime::AppendStreamBinding {
                    layer,
                    lanes: 2,
                    spec,
                    limits: AppendStreamLimits {
                        entries: 64,
                        page_entries: 2,
                        read_entries: 2,
                    },
                    payload_bytes: 65536,
                    scratch_bytes: 65536,
                    catalog_bytes: 65536,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}
fn state(spec: &TargetSpec) -> State {
    DeviceState::create(spec.state_layout().unwrap(), |index, policy| {
        let mut state = NumericHybridLayerState::new(policy);
        if let UnitSpec::Decoder {
            mixer: eredu_architectures::qwen4_exp::target::MixerSpec::Indexed(a),
            ..
        } = &spec.units[index]
        {
            for lane in 0..2 {
                for stream in a.state.streams() {
                    state.streams.push((
                        stream.slot,
                        lane,
                        ResidentAppendStream::new(
                            stream,
                            AppendStreamLimits {
                                entries: 64,
                                page_entries: 2,
                                read_entries: 2,
                            },
                            65536,
                            65536,
                            65536,
                        )
                        .unwrap(),
                    ));
                }
            }
        }
        Ok::<_, Error>(state)
    })
    .unwrap()
}
fn fixture_hash() -> eredu_architectures::qwen4_exp::ngram::NGramHashSpec {
    eredu_architectures::qwen4_exp::ngram::NGramHashSpec::new(
        32,
        7,
        3,
        1,
        vec![23703573157769, 20109073645365, 8052911324071],
        vec![11, 13],
        vec![0, 11],
        24,
    )
    .unwrap()
}
fn bind_spec(spec: TargetSpec) -> BoundTargetSpec {
    BoundTargetSpec::new(spec, BTreeMap::from([(1, fixture_hash())])).unwrap()
}
fn model(spec: TargetSpec, ctx: &NumericContext) -> TargetModel<NumericBackend> {
    let mut model = TargetModel::new(bind_spec(spec), ctx).unwrap();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut Parameters::default());
    model
}
struct Rows {
    calls: Vec<Vec<u64>>,
    bank: usize,
}
impl Default for Rows {
    fn default() -> Self {
        Self {
            calls: vec![],
            bank: 2,
        }
    }
}
impl RowLookupProvider<NumericBackend> for Rows {
    fn has_row_parameter(&self, id: &eredu_nn::ParameterId) -> bool {
        id.as_str() == "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        _: ParameterBankAccess,
        _: &NumericContext,
    ) -> Result<NumericTensor, RowLookupError> {
        assert_eq!((spec.bank, spec.unit), (self.bank, 1));
        assert!(rows.len() <= 128);
        assert!(rows.iter().all(|r| *r < 24));
        self.calls.push(rows.to_vec());
        Ok(NumericTensor::new(
            [rows.len() as i32, 2],
            rows.iter()
                .flat_map(|r| [(*r as f32 + 0.25) / 17., -(*r as f32 + 0.5) / 19.])
                .collect(),
        ))
    }
}
#[derive(Default)]
struct Rebuild {
    acquired: Vec<usize>,
    complete: usize,
}
impl LayerwisePolicy<NumericBackend, Unit<NumericBackend>> for Rebuild {
    type Lease = RebuiltUnitLease<Unit<NumericBackend>>;
    type Error = Error;
    fn begin(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Error> {
        Ok(())
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        _: ExecutionUnitAddress,
        build: F,
        ctx: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<Unit<NumericBackend>, E>,
    {
        self.acquired.push(ordinal);
        let mut unit = build(ctx).map_err(LayerwiseAcquireError::Architecture)?;
        unit.visit_parameters_mut(&mut Parameters::default());
        Ok(RebuiltUnitLease(unit))
    }
    fn complete<'a, SV, CV>(
        &mut self,
        _: usize,
        _: ExecutionUnitAddress,
        _: Self::Lease,
        _: &'a NumericTensor,
        _: SV,
        retained: CV,
        _: &NumericContext,
    ) -> Result<(), Error>
    where
        NumericTensor: 'a,
        SV: Iterator<Item = &'a NumericTensor>,
        CV: Iterator<Item = &'a NumericTensor>,
    {
        let values: Vec<_> = retained.collect();
        assert!(values.len() >= 5);
        assert_eq!(
            values[0].element_type(),
            Some(eredu_nn::TensorElementType::I32)
        );
        self.complete += 1;
        Ok(())
    }
    fn finish(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Error> {
        Ok(())
    }
}
#[derive(Default)]
struct Observe {
    paths: Vec<String>,
    fail: bool,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observe {
    fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
        self.paths.push(path.into());
        if self.fail && path == "model.layers.1.lexical.write" {
            return Err(Error::backend("injected target failure"));
        }
        Ok(())
    }
}
fn input_values() -> (Vec<u64>, Vec<bool>, NumericTensor) {
    let ids = (0..2)
        .flat_map(|lane| {
            (0..19).map(move |t| {
                if t == 5 || t == 12 {
                    7
                } else {
                    (t * 3 + lane * 5) % 31
                }
            })
        })
        .collect();
    let visible = (0..2)
        .flat_map(|lane| (0..19).map(move |t| !(lane == 0 && t < 2) && t != 9))
        .collect();
    let media = NumericTensor::new(
        [2, 19, 2],
        (0..76)
            .map(|i| ((i * 11 % 43) as f32 - 21.) / 17.)
            .collect(),
    );
    (ids, visible, media)
}
#[test]
fn qwen4_target_shared_resident_and_rebuilt_execution_preserves_all_unit_state() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let spec = specification();
    assert_eq!(spec.units.len(), 5);
    assert_eq!(spec.units[1].path(), "model.layers.1.ple");
    assert_eq!(spec.units[2].path(), "model.layers.1");
    let architecture = model(spec.clone(), &ctx);
    assert!(architecture.parameter_description(&ctx).is_ok());
    let units = (0..5)
        .map(|i| {
            let mut u = architecture.construct_unit(i, &ctx).unwrap();
            u.visit_parameters_mut(&mut Parameters::default());
            u
        })
        .collect();
    let mut resident = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut whole = state(&spec);
    let (ids, visible, media) = input_values();
    let mut provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: Rows::default(),
    };
    let mut observed = Observe::default();
    let (expected, forward) = resident
        .forward_with_provider_and_observer_and_context(
            TargetInput {
                position_delta: None,
                ids: Some(OriginalTokenIds::Host(&ids)),
                batch: 2,
                tokens: 19,
                embeddings: Some(&media),
                visible: Some(TokenVisibility::Host(&visible)),
                rotary: None,
            },
            &mut whole,
            ExpertPass::Prefill,
            &mut provider,
            &ctx,
            &mut observed,
        )
        .unwrap();
    assert_eq!(forward.request.ids(), ids);
    assert_eq!(<TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::prediction_target_capture(&forward).unwrap().shape(),[2,19,2,2]);
    assert_eq!(expected.shape(), [2, 19, 32]);
    assert!(expected.data.iter().any(|v| v.abs() > 0.01));
    assert!(observed
        .paths
        .contains(&"model.layers.1.lexical.write".into()));
    assert!(observed
        .paths
        .contains(&"model.layers.1.attention.selected_positions".into()));
    for chunks in [vec![1, 2, 1, 3, 1, 11], vec![1; 19]] {
        let mut runtime = LayerwiseRuntime::new(model(spec.clone(), &ctx), Rebuild::default());
        let mut split = state(&spec);
        let mut pieces = vec![];
        let mut start = 0;
        for length in chunks {
            let end = start + length;
            let chunk_ids: Vec<_> = (0..2)
                .flat_map(|lane| ids[lane * 19 + start..lane * 19 + end].iter().copied())
                .collect();
            let mask: Vec<_> = (0..2)
                .flat_map(|lane| visible[lane * 19 + start..lane * 19 + end].iter().copied())
                .collect();
            pieces.push(
                runtime
                    .forward_with_provider_and_observer(
                        TargetInput {
                            position_delta: None,
                            ids: Some(OriginalTokenIds::Host(&chunk_ids)),
                            batch: 2,
                            tokens: length as i32,
                            embeddings: Some(&media.axis_slice(1, start, end)),
                            visible: Some(TokenVisibility::Host(&mask)),
                            rotary: None,
                        },
                        &mut split,
                        ExpertPass::Decode,
                        &mut provider,
                        &ctx,
                        &mut Observe::default(),
                    )
                    .unwrap(),
            );
            start = end;
        }
        assert_tensor_close(
            &NumericTensor::concatenate(&pieces, 1, &ctx).unwrap(),
            &expected,
            "target layered chunk invariance",
        );
        assert_eq!(runtime.policy().complete, runtime.policy().acquired.len());
        for index in 0..5 {
            let a = whole.layer(index).unwrap();
            let b = split.layer(index).unwrap();
            assert_eq!(a.position(), 19);
            assert_eq!(b.position(), 19);
            for (role, value) in &a.fixed {
                assert_tensor_close(
                    value.as_ref().unwrap(),
                    b.fixed[role].as_ref().unwrap(),
                    "complete target fixed state",
                );
            }
            if let Some(cache) = &a.attention {
                assert_tensor_close(
                    cache.keys.as_ref().unwrap(),
                    b.attention.as_ref().unwrap().keys.as_ref().unwrap(),
                    "target K/V",
                );
            }
        }
    }
    let checkpoint = whole.clone();
    let before = provider.rows.calls.len();
    assert!(resident
        .forward_with_provider_and_observer(
            TargetInput {
                position_delta: None,
                ids: None,
                batch: 2,
                tokens: 1,
                embeddings: Some(&media.axis_slice(1, 0, 1)),
                visible: None,
                rotary: None
            },
            &mut whole,
            ExpertPass::Decode,
            &mut provider,
            &ctx,
            &mut Observe::default()
        )
        .is_err());
    assert_eq!(provider.rows.calls.len(), before);
    let call = |runtime: &mut LayerwiseRuntime<_, NumericBackend, State, _>,
                state: &mut State,
                provider: &mut ParameterProviders<_, _>,
                fail| {
        runtime.forward_with_provider_and_observer(
            TargetInput {
                position_delta: None,
                ids: Some(OriginalTokenIds::Host(&[3, 8])),
                batch: 2,
                tokens: 1,
                embeddings: None,
                visible: None,
                rotary: None,
            },
            state,
            ExpertPass::Decode,
            provider,
            &ctx,
            &mut Observe {
                fail,
                ..Default::default()
            },
        )
    };
    assert!(call(&mut resident, &mut whole, &mut provider, true).is_err());
    assert!(provider.rows.calls.len() > before);
    whole = checkpoint.clone();
    let actual = call(&mut resident, &mut whole, &mut provider, false).unwrap();
    let mut fork = checkpoint;
    let replay = call(&mut resident, &mut fork, &mut provider, false).unwrap();
    assert_tensor_exact(&actual, &replay, "target rollback and fork");
}
#[test]
fn qwen4_request_boundary_preserves_exact_ids_media_and_rejects_drift() {
    let ctx = NumericContext::default();
    let schema = RequestBoundarySchema::new(
        eredu_nn::residual_streams::ResidualStreamGeometry::new(2, 2).unwrap(),
        2,
    )
    .unwrap();
    let ids = [16_777_217u64, 7, 16_777_219, 2];
    let visible = [true, false, true, true];
    let cosine = NumericTensor::new([1, 2, 2], vec![0.6001234, 0.7001234, 0.8001234, 0.9001234]);
    let sine = NumericTensor::new([1, 2, 2], vec![0.4001234, 0.3001234, 0.2001234, 0.1001234]);
    let request = RequestContext::new(
        Some(OriginalTokenIds::Host(&ids)),
        Some(TokenVisibility::Host(&visible)),
        2,
        2,
        17,
        20_000_000,
        4,
        2,
        Some(RotaryPosition::Embeddings {
            cosine: &cosine,
            sine: &sine,
        }),
        -3,
        &ctx,
    )
    .unwrap();
    let wire = schema.wire_schema().unwrap();
    assert_eq!(wire.primary().shape().len(), 4);
    assert_eq!(
        wire.auxiliary()[3].dtype(),
        eredu_runtime::BoundaryTensorDtype::Float32
    );
    let encoded = schema.encode(request.boundary()).unwrap();
    let restored = schema
        .decode(encoded.into_iter().map(|v| v.into_parts().1).collect())
        .unwrap();
    let decoded =
        RequestContext::from_boundary(restored, 2, 2, 17, 20_000_000, 4, 2, &ctx).unwrap();
    assert_eq!(decoded.ids(), ids);
    assert_eq!(decoded.visible(), visible);
    assert_eq!(decoded.position_delta(), -3);
    assert_tensor_exact(
        &decoded.boundary().cosine,
        &cosine.broadcast_to(&[2, 2, 2], &ctx).unwrap(),
        "media FP32 provenance",
    );
    assert!(
        RequestContext::from_boundary(decoded.boundary(), 2, 2, 18, 20_000_000, 4, 2, &ctx)
            .is_err()
    );
    let mut bad = decoded.boundary();
    bad.ids = NumericTensor::new([2, 2], vec![1.; 4]);
    assert!(RequestContext::from_boundary(bad, 2, 2, 17, 20_000_000, 4, 2, &ctx).is_err());
    assert!(matches!(
        RequestContext::<NumericTensor>::new(None, None, 2, 2, 0, 32, 4, 2, None, 0, &ctx),
        Err(RequestError::Tokens(_))
    ));
    assert!(schema.decode::<NumericTensor>(vec![]).is_err());
}

#[test]
fn qwen4_target_request_handoffs_cross_lexical_and_mixer_boundaries() {
    use eredu_runtime::RoutedLayeredArchitecture;
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let spec = specification();
    let schema = spec.boundary_schema().unwrap();
    let (ids, visible, media) = input_values();
    let cosine = NumericTensor::new(
        [2, 19, 2],
        (0..2)
            .flat_map(|b| (0..19).flat_map(move |t| [((t + b * 3) as f32 / 7.).cos(); 2]))
            .collect(),
    );
    let sine = NumericTensor::new(
        [2, 19, 2],
        (0..2)
            .flat_map(|b| (0..19).flat_map(move |t| [((t + b * 3) as f32 / 7.).sin(); 2]))
            .collect(),
    );
    let mut reference = None;
    for handoff in [false, true] {
        let mut model = model(spec.clone(), &ctx);
        let mut state = state(&spec);
        let mut forward = model
            .begin_forward(
                TargetInput {
                    position_delta: Some(-3),
                    ids: Some(OriginalTokenIds::Host(&ids)),
                    batch: 2,
                    tokens: 19,
                    embeddings: Some(&media),
                    visible: Some(TokenVisibility::Host(&visible)),
                    rotary: Some(RotaryPosition::Embeddings {
                        cosine: &cosine,
                        sine: &sine,
                    }),
                },
                &mut state,
                &ctx,
            )
            .unwrap();
        let mut provider = ParameterProviders {
            grouped: ResidentExpertProvider,
            rows: Rows::default(),
        };
        for index in 0..spec.units.len() {
            if handoff {
                let values = schema.encode(forward.context.request.boundary()).unwrap();
                let boundary = schema
                    .decode(values.into_iter().map(|v| v.into_parts().1).collect())
                    .unwrap();
                forward = model
                    .resume_request(
                        forward.hidden,
                        boundary,
                        state.layer(index).unwrap().position(),
                        &ctx,
                    )
                    .unwrap();
                assert_eq!(forward.context.request.ids(), ids);
                assert_eq!(forward.context.request.position_delta(), -3);
            }
            let mut unit = model.construct_unit(index, &ctx).unwrap();
            unit.visit_parameters_mut(&mut Parameters::default());
            forward.hidden = model
                .forward_unit_with_provider(
                    0,
                    index,
                    &mut unit,
                    &forward.hidden,
                    &mut state,
                    &mut forward.context,
                    ExpertPass::Prefill,
                    &mut provider,
                    &ctx,
                )
                .unwrap();
        }
        let output = model
            .finish_forward(&forward.hidden, &mut state, &forward.context, &ctx)
            .unwrap();
        if let Some(expected) = &reference {
            assert_tensor_exact(
                &output,
                expected,
                "original-ID/media handoff across every target unit",
            );
        } else {
            reference = Some(output);
        }
    }
    let first = model(spec.clone(), &ctx);
    let second = model(specification(), &ctx);
    let state = eredu_runtime::PartitionState::new(spec.state_layout().unwrap(), 0).unwrap();
    assert_eq!(
        first.state_identity(&state, Default::default()).unwrap(),
        second.state_identity(&state, Default::default()).unwrap(),
        "independently parsed config maps have stable cache identity"
    );
    let mut bad = spec.clone();
    bad.units.swap(1, 2);
    assert!(
        BoundTargetSpec::new(bad, BTreeMap::from([(1, fixture_hash())]))
            .and_then(|bound| TargetModel::<NumericBackend>::new(bound, &ctx))
            .is_err()
    );
    let mut bad = spec;
    bad.units.pop();
    assert!(
        BoundTargetSpec::new(bad, BTreeMap::from([(1, fixture_hash())]))
            .and_then(|bound| TargetModel::<NumericBackend>::new(bound, &ctx))
            .is_err()
    );
}

#[path = "qwen4_target/session.rs"]
mod session;

#[path = "qwen4_target/cold.rs"]
mod cold;

#[path = "qwen4_target/transforms.rs"]
mod transforms;

fn read_owner_parameter(
    owner: &eredu_architectures::qwen4_exp::prepared::PreparedParameters,
    name: &str,
) -> Vec<f32> {
    let recipe = &owner.recipes()[name];
    let meta = recipe.infer(owner.source().as_ref()).unwrap();
    let mut bytes = vec![0u8; meta.byte_len as usize];
    if let Some(read) = recipe
        .prepare_encoded_read(owner.source().as_ref())
        .unwrap()
    {
        read.read_into(&mut bytes).unwrap();
    } else {
        use eredu_checkpoint::{
            recipe::DerivedWeightRecipe,
            store::{EncodedTensorLease, ReadPolicy, TensorReadRequest},
        };
        let DerivedWeightRecipe::Source { key, selection } = recipe else {
            panic!("fixture expects direct packed sources")
        };
        let lease = owner
            .source()
            .acquire_lease(TensorReadRequest {
                key: key.clone(),
                selection: selection.clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert!(lease.bounded_read_proof().physically_bounded);
        bytes.copy_from_slice(lease.encoded_bytes().unwrap());
    }
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[path = "qwen4_target/prediction.rs"]
mod prediction;

#[path = "qwen4_target/gguf.rs"]
mod gguf;

#[path = "qwen4_target/vision.rs"]
mod vision;

#[path = "qwen4_target/load_policy.rs"]
mod load_policy;

#[path = "qwen4_target/safetensors_admission.rs"]
mod safetensors_admission;

#[test]
fn qwen4_bound_target_rejects_missing_extra_and_mismatched_hash_controls() {
    let geometry = specification();
    let incompatible = eredu_architectures::qwen4_exp::ngram::NGramHashSpec::new(
        33,
        7,
        3,
        1,
        vec![23703573157769, 20109073645365, 8052911324071],
        vec![11, 13],
        vec![0, 11],
        24,
    )
    .unwrap();
    let wrong_table = eredu_architectures::qwen4_exp::ngram::NGramHashSpec::new(
        32,
        7,
        3,
        1,
        vec![23703573157769, 20109073645365, 8052911324071],
        vec![11, 13],
        vec![0, 11],
        25,
    )
    .unwrap();
    // This validation has no backend type, context or tensor-construction authority.
    for hashes in [
        BTreeMap::new(),
        BTreeMap::from([(1, fixture_hash()), (3, fixture_hash())]),
        BTreeMap::from([(0, fixture_hash())]),
        BTreeMap::from([(1, incompatible)]),
        BTreeMap::from([(1, wrong_table)]),
    ] {
        assert!(BoundTargetSpec::new(geometry.clone(), hashes).is_err());
    }
    assert!(BoundTargetSpec::new(geometry, BTreeMap::from([(1, fixture_hash())])).is_ok());
}

#[test]
fn qwen4_bound_target_rejects_matching_lexical_hash_that_changes_retained_config() {
    let mut geometry = specification();
    let rows = geometry.limits.lookup_rows;
    let UnitSpec::Lexical { spec, .. } = &mut geometry.units[1] else {
        panic!("lexical unit")
    };
    spec.embedding = eredu_architectures::qwen4_exp::ngram::NGramEmbeddingSpec::new(
        32,
        8,
        3,
        1,
        spec.embedding.lookup_spec().clone(),
        0,
        rows,
    )
    .unwrap();
    let changed_hash = eredu_architectures::qwen4_exp::ngram::NGramHashSpec::new(
        32,
        8,
        3,
        1,
        vec![23703573157769, 20109073645365, 8052911324071],
        vec![11, 13],
        vec![0, 11],
        24,
    )
    .unwrap();
    // The replacement fits this lexical spec and has unchanged width and state;
    // the separately retained family configuration still requires reset token 7.
    assert!(eredu_architectures::qwen4_exp::ngram::NGramEmbedding::new(
        spec.embedding.clone(),
        changed_hash.clone()
    )
    .is_ok());
    assert!(BoundTargetSpec::new(geometry, BTreeMap::from([(1, changed_hash)])).is_err());
}

#[path = "qwen4_target/gguf_admission.rs"]
mod gguf_admission;

#[path = "qwen4_target/registry.rs"]
mod registry;

#[path = "qwen4_target/tensor_parallel.rs"]
mod tensor_parallel;

#[path = "qwen4_target/parallel_geometry.rs"]
mod parallel_geometry;

#[path = "qwen4_target/prepared_tensor_parallel.rs"]
mod prepared_tensor_parallel;

#[path = "qwen4_target/partition_input.rs"]
mod partition_input;

#[path = "qwen4_target/partition_selection.rs"]
mod partition_selection;

#[path = "qwen4_target/partitioned.rs"]
mod partitioned;

#[path = "qwen4_target/ordinary_partition.rs"]
mod ordinary_partition;

#[path = "qwen4_target/expert_geometry.rs"]
mod expert_geometry;
