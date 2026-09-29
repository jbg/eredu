//! Selected module formats against independently rounded dense checkpoint weights.
use super::*;
use eredu_architectures::{
    qwen4_exp::{
        checkpoint::schema::SafetensorsEncoding,
        prepared::{PreparedTarget, TargetExecutionPlan},
    },
    routed_text::{
        PreparedRoutedTextArchitecture, RoutedTextArchitectureVisitor, RoutedTextSelectionRequest,
    },
};
use eredu_nn::TensorElementType;
use eredu_runtime::*;

type Values = Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>;
const QUANTIZATION: eredu_core::QuantizationRequest = eredu_core::QuantizationRequest::Affine {
    group_size: 16,
    bits: 4,
};

pub(super) fn fixture(values: &Values) -> (tempfile::TempDir, SharedCheckpointSource) {
    let directory = tempfile::tempdir().unwrap();
    safetensors::tensor::serialize_to_file(
        values.iter().map(|(name, dtype, shape, bytes)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source =
        Arc::new(eredu_checkpoint::store::SafetensorsWeightStore::open(directory.path()).unwrap());
    (directory, source)
}
fn plan(source: SharedCheckpointSource, spec: &TargetSpec) -> TargetExecutionPlan {
    let mut config = configuration();
    config.hidden_size = 32;
    config.experts.intermediate = 32;
    let target = PreparedTarget::safetensors(
        source,
        config,
        SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap(),
        spec.limits,
    )
    .unwrap();
    target
        .execution_plan(
            stream_bindings(spec),
            eredu_runtime::SelectedRowLookupPlans::select(
                target
                    .row_lookups(
                        RowLookupLimits {
                            requests: 128,
                            rows_per_acquisition: 2,
                            acquisition_bytes: 16,
                            host_bytes: 16384,
                            output_bytes: 3072,
                        },
                        eredu_core::residency::ResidencyPolicy::Cacheable,
                    )
                    .unwrap()
                    .descriptors()
                    .clone(),
                ParameterBankLoadOptions::new(
                    eredu_core::residency::OffloadConfig::new(Some(256), Some(0), 1).unwrap(),
                    32768,
                    32768,
                )
                .unwrap(),
                0,
                &super::cold::Support,
            )
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn qwen4_selected_transforms_match_independent_dense_weights_across_residency() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut config = configuration();
    config.hidden_size = 32;
    config.experts.intermediate = 32;
    let spec = specification_for(config);
    let mut model = TargetModel::<NumericBackend>::new(bind_spec(spec.clone()), &ctx).unwrap();
    let mut parameters = Parameters::default();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model)
        .visit_parameters_mut(&mut parameters);
    for i in 0..spec.units.len() {
        model
            .construct_unit(i, &ctx)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    let mut values = table_values();
    for shard in 0..2 {
        let row = values
            .iter_mut()
            .find(|(n, _, _, _)| n.ends_with(&format!("shard_{shard}.weight")))
            .unwrap();
        row.3 = (shard * 12..(shard + 1) * 12)
            .flat_map(|r| [(r as f32 + 0.25) / 17., -(r as f32 + 0.5) / 19.])
            .flat_map(f32::to_le_bytes)
            .collect();
    }
    for (name, value) in parameters.0 {
        let mut shape: Vec<usize> = value.shape.iter().map(|n| *n as usize).collect();
        if name.ends_with(".conv1d.weight") && shape.len() == 2 {
            shape.insert(1, 1);
        }
        values.push((
            name,
            safetensors::Dtype::F32,
            shape,
            value.data.into_iter().flat_map(f32::to_le_bytes).collect(),
        ));
    }
    let (_source_dir, source) = fixture(&values);
    let plan = plan(source.clone(), &spec);
    let transformed: BTreeSet<_> = plan
        .requirements()
        .text()
        .parameters()
        .iter()
        .filter(|p| p.transform_target(QUANTIZATION).unwrap().is_some())
        .map(|p| p.name().to_owned())
        .collect();
    assert!(transformed.len() > 40);
    assert!(transformed.contains("model.embed_tokens.weight"));
    assert!(transformed.contains("model.layers.0.mlp.experts.down_proj"));
    assert!(!transformed.contains("model.layers.1.ple.value_proj.weight"));
    assert!(!transformed.contains("model.hyper_connection_mixer.input_mix_weight_up.weight"));
    // Independent affine oracle: round a dense source copy. No selected tasks,
    // format companions or production/scalar materializer are used to decode it.
    let mut changed = 0;
    for (name, _, _, bytes) in &mut values {
        if !transformed.contains(name) {
            continue;
        }
        for group in bytes.chunks_exact_mut(16 * 4) {
            let input: Vec<f32> = group
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let low = input.iter().copied().reduce(f32::min).unwrap();
            let high = input.iter().copied().reduce(f32::max).unwrap();
            let step = (high - low) / 15.;
            for (value, out) in input.iter().zip(group.chunks_exact_mut(4)) {
                let code = if step == 0. {
                    0.
                } else {
                    ((*value - low) / step).round().clamp(0., 15.)
                };
                let rounded = low + step * code;
                changed += usize::from(rounded.to_bits() != value.to_bits());
                out.copy_from_slice(&rounded.to_le_bytes());
            }
        }
    }
    assert!(changed > 1000);
    let (_dense_dir, dense_source) = fixture(&values);
    let dense_plan = self::plan(dense_source.clone(), &spec);
    let dense_facts = super::cold::capabilities(dense_plan.requirements(), None);
    let run = |plan: TargetExecutionPlan,
               source: SharedCheckpointSource,
               residency,
               quantization,
               facts: &BackendMechanismCapabilities| {
        let mut text = ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
            .with_session(eredu_core::SessionCapabilities::new(false, true, true))
            .with_exact_completion(true);
        if let Some(q) = quantization {
            text = text.with_quantization(q);
        }
        let request =
            RoutedTextSelectionRequest::new(text, WeightResidency::with_layers(residency)).unwrap();
        let reads = source.source_diagnostics().unwrap().physical_reads;
        let selected = plan.select(&request, facts, None).unwrap();
        assert_eq!(reads, source.source_diagnostics().unwrap().physical_reads);
        selected
            .visit::<NumericBackend, State, _>(
                &ctx,
                TraceBinding {
                    ctx: &ctx,
                    residency,
                },
            )
            .unwrap()
    };
    let expected = run(
        dense_plan,
        dense_source,
        LayerWeightResidency::FullyResident,
        None,
        &dense_facts,
    );
    let facts = super::cold::capabilities(plan.requirements(), Some(QUANTIZATION));
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 20, 0, 0, 0).unwrap(),
        ),
    ] {
        let actual = run(
            plan.clone(),
            source.clone(),
            residency,
            Some(QUANTIZATION),
            &facts,
        );
        assert_ne!(actual.identity, expected.identity);
        assert_eq!(actual.outputs.len(), expected.outputs.len());
        for (a, e) in actual.outputs.iter().zip(&expected.outputs) {
            assert_tensor_close(a, e, "selected target versus independently rounded weights");
        }
        super::session::assert_checkpoint(&actual.state, &expected.state);
    }
    // MXFP4 has a distinct one-companion layout. Validate typed source/selected
    // construction without claiming native packed numerical coverage.
    let mx = eredu_core::QuantizationRequest::MxFp4;
    let mx_request = RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_quantization(mx),
        WeightResidency::with_layers(LayerWeightResidency::FullyResident),
    )
    .unwrap();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let count = plan
        .clone()
        .select(
            &mx_request,
            &super::cold::capabilities(plan.requirements(), Some(mx)),
            None,
        )
        .unwrap()
        .visit::<NumericBackend, State, _>(&ctx, InspectMx)
        .unwrap();
    assert!(count > 40);
    assert_eq!(reads, source.source_diagnostics().unwrap().physical_reads);
    // Native-only facts must reject a requested transform before any payload read.
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let request = RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_quantization(QUANTIZATION),
        WeightResidency::with_layers(LayerWeightResidency::FullyResident),
    )
    .unwrap();
    assert!(plan
        .clone()
        .select(
            &request,
            &super::cold::capabilities(plan.requirements(), None),
            None
        )
        .is_err());
    assert_eq!(reads, source.source_diagnostics().unwrap().physical_reads);
}

struct Trace {
    outputs: Vec<NumericTensor>,
    state: State,
    identity: String,
}
struct TraceBinding<'a> {
    ctx: &'a NumericContext,
    residency: LayerWeightResidency,
}
fn fixture_state(layout: &StateLayout) -> Result<State, Error> {
    DeviceState::create(layout.clone(), |_, policy| {
        let mut state = NumericHybridLayerState::new(policy);
        for lane in 0..2 {
            for stream in policy.append_streams() {
                state.streams.push((
                    stream.slot(),
                    lane,
                    ResidentAppendStream::new(
                        AppendStreamSpec {
                            slot: stream.slot(),
                            width: stream.width(),
                            element: match stream.dtype() {
                                eredu_core::cache::StateTensorDtype::Floating
                                | eredu_core::cache::StateTensorDtype::Float32 => {
                                    TensorElementType::F32
                                }
                                eredu_core::cache::StateTensorDtype::Int32 => {
                                    TensorElementType::I32
                                }
                                other => panic!("unexpected fixture stream dtype {other:?}"),
                            },
                        },
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
        Ok::<_, Error>(state)
    })
}
impl RoutedTextArchitectureVisitor<NumericBackend, State> for TraceBinding<'_> {
    type Output = Trace;
    type Error = String;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        source: SharedCheckpointSource,
    ) -> Result<Trace, String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        let identity = prepared
            .text()
            .prompt_cache_identity()
            .architecture_fingerprint()
            .to_owned();
        let ctx = self.ctx;
        let mut mechanisms = if self.residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(source)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(source)
        };
        mechanisms.fixture_state_factory = Some(fixture_state);
        macro_rules! run {
            ($session:expr) => {{
                let mut session = $session;
                let mut outputs = Vec::new();
                let ids = [3, 4, 7, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
                for part in [&ids[..3], &ids[3..8], &ids[8..]] {
                    outputs.push(
                        session
                            .prefill(&NumericTensor::token_ids(part), None, ctx)
                            .unwrap(),
                    );
                }
                for token in 0..16 {
                    outputs.push(
                        session
                            .decode(&NumericTensor::token_ids(&[token]), ctx)
                            .unwrap(),
                    );
                }
                assert!(outputs.iter().all(|o| o.data.iter().all(|v| v.is_finite())));
                assert!(outputs
                    .iter()
                    .any(|o| o.data.iter().any(|v| v.abs() > 1e-4)));
                let checkpoint = session.checkpoint(ctx).unwrap();
                let token = NumericTensor::token_ids(&[5]);
                let draft = session.decode(&token, ctx).unwrap();
                session.rollback(checkpoint, ctx).unwrap();
                assert_tensor_close(
                    &draft,
                    &session.decode(&token, ctx).unwrap(),
                    "selected target rollback",
                );
                Ok::<_, String>(Trace {
                    outputs,
                    state: session.checkpoint(ctx).unwrap(),
                    identity: identity.clone(),
                })
            }};
        }
        eredu_architectures::prepared_execution::construct_selected_routed_session(
            prepared,
            mechanisms,
            ctx,
            |_, _, rows| {
                let rows = rows.unwrap();
                let providers = rows
                    .prepared()
                    .bind(
                        rows.prepared()
                            .entries()
                            .iter()
                            .map(|(id, entry)| {
                                (id.clone(), crate::row_bank::SourceRows::new(entry))
                            })
                            .collect(),
                    )
                    .unwrap();
                Ok::<_, String>(
                    (
                        BTreeMap::<
                            RoutedBankId,
                            (NumericGroupedBankMechanism, NumericIndexedMovement),
                        >::new(),
                        Some(providers),
                    ),
                )
            },
            (),
            |_, session, _| run!(session),
            |_, session, _| run!(session),
        )
        .map_err(|e| e.to_string())
    }
}

struct InspectMx;
impl RoutedTextArchitectureVisitor<NumericBackend, State> for InspectMx {
    type Output = usize;
    type Error = String;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        _: SharedCheckpointSource,
    ) -> Result<usize, String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        let tasks = prepared.text().selected().materialization_tasks();
        let mut count = 0;
        for task in tasks {
            if task.executable() == eredu_checkpoint::LinearFormat::MxFp4 {
                assert!(matches!(
                    task.lowering(),
                    WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
                ));
                assert_eq!(task.output_companions().len(), 1);
                count += 1;
            }
        }
        let (mut modules, _, _, _) = prepared.into_parts();
        assert!(modules.take_source_architecture().is_some());
        Ok(count)
    }
}
