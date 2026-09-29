//! Retained prediction tasks selected with target weights, then generic binding.
use super::executor::{Materializer, SourceBinding};
use super::*;
use eredu_architectures::routed_text::RoutedTextSelectionRequest;
use eredu_runtime::*;

pub(super) fn streams(spec: &PredictionSpec) -> Vec<AppendStreamBinding> {
    spec.units
        .iter()
        .flat_map(|u| {
            let eredu_architectures::qwen4_exp::target::MixerSpec::Indexed(a) = &u.mixer else {
                return vec![];
            };
            a.state
                .streams()
                .into_iter()
                .map(|spec| AppendStreamBinding {
                    layer: u.depth,
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
                .collect::<Vec<_>>()
        })
        .collect()
}
pub(super) fn state_capabilities(
    plan: &eredu_architectures::qwen4_exp::prepared::TargetExecutionPlan,
) -> StateMechanismCapabilities {
    synthesize_state_capabilities(
        plan.prediction_state_requirements().unwrap(),
        &CacheResidencyPolicy::Device,
        &NumericMechanismSupport {
            persistent_session: true,
            ..Default::default()
        },
    )
}

pub(super) fn check(
    target: &PreparedTarget,
    source: &SharedCheckpointSource,
    ctx: &NumericContext,
) {
    let prepared = target.prediction(limits()).unwrap();
    let options = ParameterBankLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(0), 1).unwrap(),
        32768,
        32768,
    )
    .unwrap();
    let plan = target
        .execution_plan(
            stream_bindings(target.spec()),
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
                options,
                0,
                &super::super::cold::Support,
            )
            .unwrap(),
        )
        .unwrap();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let target_only_plan = plan.clone();
    assert!(plan.clone().with_prediction(limits(), vec![]).is_err());
    let plan = plan
        .with_prediction(limits(), streams(prepared.spec()))
        .unwrap();
    assert!(plan
        .clone()
        .with_prediction(limits(), streams(prepared.spec()))
        .is_err());
    let header = eredu_architectures::qwen4_exp::prepared::SafetensorsTargetPlan::prepare(
        source.as_ref(),
        target.spec().configuration().clone(),
        SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap(),
    )
    .unwrap();
    let header_prediction = header.prediction_spec(limits()).unwrap();
    assert_eq!(
        format!("{header_prediction:?}"),
        format!("{:?}", prepared.spec())
    );
    let header_plan = header
        .execution_plan(
            target.spec().limits,
            stream_bindings(target.spec()),
            plan.requirements().row_lookups().unwrap().clone(),
            super::super::safetensors_admission::physical_sources(source.as_ref()),
        )
        .unwrap()
        .with_prediction(limits(), streams(&header_prediction))
        .unwrap();
    assert_eq!(header_plan.requirements(), plan.requirements());
    assert_eq!(
        header_plan.capability_estimate(),
        plan.capability_estimate()
    );
    assert_eq!(
        plan.capability_estimate()
            .state_layout()
            .layer_layout()
            .len(),
        target.spec().units.len() + prepared.spec().units.len()
    );
    assert_eq!(
        plan.capability_estimate().speculative_draft_source(),
        Some(eredu_core::SpeculativeDraftSource::Embedded)
    );
    assert_eq!(
        header_plan.prediction_state_requirements(),
        plan.prediction_state_requirements()
    );

    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    assert_eq!(plan.requirements().banks().len(), 2);
    let auxiliary = plan.requirements().text().auxiliary_parameters();
    assert!(auxiliary.iter().all(|p| p.name().starts_with("mtp.")));
    let mut roles = BTreeMap::new();
    for parameter in auxiliary {
        let role = parameter.auxiliary_residency().unwrap();
        roles.insert(role.module().as_str(), role.shared());
    }
    assert_eq!(roles.len(), prepared.spec().units.len() + 1);
    assert_eq!(roles.values().filter(|v| **v).count(), 1);
    let prediction_capabilities = state_capabilities(&plan);
    let capabilities =
        super::super::cold::capabilities(plan.requirements(), None).with_indexed_movement(true);
    // Each role fits u64 independently, but the retained pair must reject the
    // combined allowance before any source payload or native construction.
    let mut large_target_streams = stream_bindings(target.spec());
    let mut large_prediction_streams = streams(prepared.spec());
    for stream in large_target_streams
        .iter_mut()
        .chain(&mut large_prediction_streams)
    {
        stream.catalog_bytes = u64::MAX / 8;
    }
    let large = target
        .execution_plan(
            large_target_streams,
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
                options,
                0,
                &super::super::cold::Support,
            )
            .unwrap(),
        )
        .unwrap()
        .with_prediction(limits(), large_prediction_streams)
        .unwrap();
    let ordinary = RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        ),
        WeightResidency::with_layers(LayerWeightResidency::FullyResident),
    )
    .unwrap();
    assert!(matches!(
        large.select(&ordinary, &capabilities, Some(&prediction_capabilities)),
        Err(eredu_architectures::qwen4_exp::prepared::TargetSelectionError::Resources(_))
    ));
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 20, 0, 0, 0).unwrap(),
        ),
    ] {
        let request = RoutedTextSelectionRequest::new(
            ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
                .with_session(eredu_core::SessionCapabilities::new(false, true, true))
                .with_exact_completion(true),
            WeightResidency::with_layers(residency),
        )
        .unwrap();
        let before = source.source_diagnostics().unwrap().physical_reads;
        use eredu_architectures::qwen4_exp::prepared::TargetSelectionError;
        assert!(matches!(
            plan.clone().select(&request, &capabilities, None),
            Err(TargetSelectionError::PredictionStatePresence)
        ));
        // Target component ordinals include the injection unit and cannot serve as
        // prediction component facts, even when some tensor dimensions agree.
        assert!(matches!(
            plan.clone()
                .select(&request, &capabilities, Some(capabilities.state())),
            Err(TargetSelectionError::PredictionState(_))
        ));
        for broken in [
            prediction_capabilities
                .clone()
                .with_transactions(true, false),
            prediction_capabilities.clone().with_reset(false),
            prediction_capabilities
                .clone()
                .with_observation_retention(false),
        ] {
            assert!(matches!(
                plan.clone().select(&request, &capabilities, Some(&broken)),
                Err(TargetSelectionError::PredictionState(_))
            ));
        }
        let selected = plan
            .clone()
            .select(&request, &capabilities, Some(&prediction_capabilities))
            .unwrap();
        assert!(matches!(
            header_plan.clone().select(&request, &capabilities, None),
            Err(TargetSelectionError::PredictionStatePresence)
        ));
        let selected_header = header_plan
            .clone()
            .select(&request, &capabilities, Some(&prediction_capabilities))
            .unwrap();
        assert_eq!(selected_header.selected(), selected.realization());
        assert_eq!(
            selected_header.prediction_state(),
            selected.prediction_state()
        );
        assert_eq!(
            format!("{:?}", selected_header.prediction_spec().unwrap()),
            format!("{:?}", selected.prediction_spec().unwrap())
        );
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, before);

        super::resources::check(&selected, source);
        use eredu_architectures::prepared_execution::{
            PreparedExecutionError, RetainedArchitectureConstruction, RoutedRoute,
        };
        let result = RoutedRoute::<NumericBackend, (), (), State, _, _, _, _>::new(
            ctx,
            ctx,
            (),
            (),
            (),
            TargetHandoff(false),
        )
        .construct_retained(
            RetainedArchitectureConstruction::Qwen4Exp(Box::new(selected.clone())),
            None,
        );
        assert!(matches!(
            result,
            Err(PreparedExecutionError::PredictionSourceMismatch)
        ));
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, before);
        selected
            .clone()
            .visit::<NumericBackend, State, _>(ctx, TargetHandoff(true))
            .unwrap();
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            before,
            "joint state and weight selection must not read payloads"
        );
        super::installed::check(&selected, target, ctx);
        super::driver::check(&selected, target, ctx);
        let before = source.source_diagnostics().unwrap().physical_reads;
        let selected_spec = selected.prediction_spec().unwrap();
        let handoff = selected
            .prepare_prediction_weights::<NumericBackend>(ctx)
            .unwrap();
        assert_eq!(handoff.streams(), streams(prepared.spec()));
        assert_eq!(Some(handoff.state()), selected.prediction_state());
        assert_eq!(
            handoff.state().layout(),
            &prepared.spec().state_layout().unwrap()
        );
        assert_ne!(
            handoff.state().layout(),
            selected.realization().text().state().layout()
        );
        assert!(Arc::ptr_eq(handoff.source(), target.artifact()));
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            before,
            "selection and construction read no payload"
        );
        let prediction_bank = selected.realization().bank(RoutedBankId::new(2)).unwrap();
        assert_eq!(
            prediction_bank.addressable_members().len(),
            3 * prepared.spec().units.len()
        );
        assert!(prediction_bank
            .addressable_members()
            .iter()
            .all(|m| m.key().bank() == 2));
        let mut binding = SourceBinding {
            source: handoff.source().clone(),
            context: ctx,
            residency,
            roles: vec![],
        };
        let (mut actual_shared, mut units) =
            handoff.materialize::<Materializer>(&mut binding).unwrap();
        assert_eq!(
            binding.roles[0],
            eredu_architectures::prediction_extension::PredictionModuleRole::Shared
        );
        assert_eq!(binding.roles.len(), prepared.spec().units.len() + 1);
        let mut expected_shared = shared(&prepared, ctx);
        let mut actual_state = prediction_state(&selected_spec);
        let mut expected_state = prediction_state(prepared.spec());
        for (depth, actual) in units.iter_mut().enumerate() {
            let mut expected = unit(&prepared, depth, ctx);
            for position in 0..19 {
                let embedding = NumericTensor::new([1, 1, 2], vec![position as f32 / 19., -0.3]);
                let residual =
                    NumericTensor::new([1, 1, 2, 2], vec![0.1, 0.3, -0.2, position as f32 / 21.]);
                let run = |module: &mut PredictionUnit<NumericBackend>,
                           shared: &mut PredictionShared<NumericBackend>,
                           state: &mut State| {
                    module
                        .forward(
                            shared,
                            PredictionInput {
                                embeddings: &embedding,
                                residual: &residual,
                                visible: None,
                                rotary: None,
                            },
                            state.layer(depth).unwrap(),
                            &mut ResidentExpertProvider,
                            ctx,
                            &mut ComponentInstrumentation::disabled(),
                        )
                        .unwrap()
                };
                let a = run(&mut actual.0, &mut actual_shared.0, &mut actual_state);
                let b = run(&mut expected, &mut expected_shared, &mut expected_state);
                assert_tensor_exact(&a.capture, &b.capture, "selected prediction capture");
                assert_tensor_exact(&a.hidden, &b.hidden, "selected prediction hidden");
            }
        }
    }
    let mut wider_prediction = limits();
    wider_prediction.qsa.tokens = target.spec().limits.qsa.tokens * 2;
    let prediction = target.prediction(wider_prediction).unwrap();
    let plan = target_only_plan
        .with_prediction(wider_prediction, streams(prediction.spec()))
        .unwrap();
    let capabilities =
        super::super::cold::capabilities(plan.requirements(), None).with_indexed_movement(true);
    let prediction_capabilities = state_capabilities(&plan);
    let request = RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_session(eredu_core::SessionCapabilities::new(false, true, true))
        .with_exact_completion(true),
        WeightResidency::with_layers(LayerWeightResidency::FullyResident),
    )
    .unwrap();
    let selected = plan
        .select(&request, &capabilities, Some(&prediction_capabilities))
        .unwrap();
    super::driver::check_joint_prefill_cap(&selected, target, ctx);
}

#[test]
fn qwen4_prediction_selected_formats_retain_source_expert_and_fusion_layout() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut config = configuration();
    config.hidden_size = 32;
    config.experts.intermediate = 32;
    config.prediction = Some(PredictionGeometry {
        layers: eredu_core::LayerSchedule::new(1, vec![LayerKind::Indexed]).unwrap(),
        rope_theta: 7777.,
    });
    let target_spec = specification_for(config.clone());
    let prediction_spec = spec(&config);
    let mut parameters = Parameters::default();
    let mut target =
        TargetModel::<NumericBackend>::new(bind_spec(target_spec.clone()), &ctx).unwrap();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut target).visit_parameters_mut(&mut parameters);
    for index in 0..target_spec.units.len() {
        target
            .construct_unit(index, &ctx)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    PredictionShared::<NumericBackend>::new(&prediction_spec, &ctx)
        .unwrap()
        .visit_parameters_mut(&mut parameters);
    PredictionUnit::<NumericBackend>::new(prediction_spec.units[0].clone(), &ctx)
        .unwrap()
        .visit_parameters_mut(&mut parameters);
    let mut values = table_values();
    for (name, value) in &parameters.0 {
        let mut shape: Vec<_> = value.shape.iter().map(|n| *n as usize).collect();
        if name.ends_with(".conv1d.weight") && shape.len() == 2 {
            shape.insert(1, 1);
        }
        values.push((
            name.clone(),
            safetensors::Dtype::F32,
            shape,
            value.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    }
    let (_directory, source) = super::super::transforms::fixture(&values);
    let prepared = PreparedTarget::safetensors(
        source.clone(),
        config,
        SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap(),
        target_spec.limits,
    )
    .unwrap();
    let options = ParameterBankLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(0), 1).unwrap(),
        1 << 20,
        1 << 20,
    )
    .unwrap();
    let plan = prepared
        .execution_plan(
            stream_bindings(&target_spec),
            eredu_runtime::SelectedRowLookupPlans::select(
                prepared
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
                options,
                0,
                &super::super::cold::Support,
            )
            .unwrap(),
        )
        .unwrap()
        .with_prediction(limits(), streams(&prediction_spec))
        .unwrap();
    let quantization = eredu_core::QuantizationRequest::Affine {
        group_size: 16,
        bits: 4,
    };
    let request = RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_session(eredu_core::SessionCapabilities::new(false, true, true))
        .with_exact_completion(true)
        .with_quantization(quantization),
        WeightResidency::with_layers(LayerWeightResidency::FullyResident),
    )
    .unwrap();
    let prediction_capabilities = state_capabilities(&plan);
    let reads = source.source_diagnostics().unwrap().physical_reads;
    assert!(plan
        .clone()
        .select(
            &request,
            &super::super::cold::capabilities(plan.requirements(), None),
            Some(&prediction_capabilities)
        )
        .is_err());
    let capabilities = super::super::cold::capabilities(plan.requirements(), Some(quantization))
        .with_indexed_movement(true);
    let selected = plan
        .clone()
        .select(&request, &capabilities, Some(&prediction_capabilities))
        .unwrap();
    let spec = selected.prediction_spec().unwrap();
    assert_eq!(
        spec.fusion.hidden_projection.format.encoding(),
        eredu_checkpoint::LinearFormat::Affine(
            eredu_checkpoint::AffineQuantization::new(16, 4).unwrap()
        )
    );
    let tasks = selected
        .realization()
        .text()
        .auxiliary_materialization_tasks();
    assert!(tasks
        .iter()
        .any(|t| t.name() == "mtp.fc_hidden.weight" && !t.output_companions().is_empty()));
    let handoff = selected
        .prepare_prediction_weights::<NumericBackend>(&ctx)
        .unwrap();
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    let mut binding = SourceBinding {
        source: handoff.source().clone(),
        context: &ctx,
        residency: LayerWeightResidency::FullyResident,
        roles: vec![],
    };
    let (mut shared, mut units) = handoff.materialize::<Materializer>(&mut binding).unwrap();
    let mut state = prediction_state(&spec);
    let output = units[0]
        .0
        .forward(
            &mut shared.0,
            PredictionInput {
                embeddings: &NumericTensor::new([1, 1, 32], vec![0.2; 32]),
                residual: &NumericTensor::new(
                    [1, 1, 2, 32],
                    (0..64).map(|i| i as f32 / 64. - 0.5).collect(),
                ),
                visible: None,
                rotary: None,
            },
            state.layer(0).unwrap(),
            &mut ResidentExpertProvider,
            &ctx,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap();
    assert!(output.capture.data.iter().all(|v| v.is_finite()));
    assert!(output.hidden.data.iter().any(|v| v.abs() > 1e-4));
    // The same selected recipes independently address prediction experts under a
    // shared cache policy. This is cold admission, not a cached-provider run.
    let request = RoutedTextSelectionRequest::new(
        request.text().clone(),
        WeightResidency::with_independent_parameter_banks(
            OrdinaryWeightResidency::FullyResident,
            options,
        ),
    )
    .unwrap();
    let selected = plan
        .select(&request, &capabilities, Some(&prediction_capabilities))
        .unwrap();
    let bank = selected.realization().bank(RoutedBankId::new(2)).unwrap();
    assert_eq!(bank.addressable_members().len(), 3);
    assert!(bank.addressable_members().iter().all(|m| {
        m.parameters()
            .iter()
            .all(|p| p.task().name().starts_with("mtp."))
    }));
}

struct TargetHandoff(bool);
impl eredu_architectures::routed_text::RoutedTextArchitectureVisitor<NumericBackend, State>
    for TargetHandoff
{
    type Output = ();
    type Error = String;
    fn construction_started(&mut self) {
        assert!(
            self.0,
            "rejected prediction must not reach target construction"
        );
    }
    fn visit<A>(
        self,
        _: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        _: SharedCheckpointSource,
    ) -> Result<(), String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        Ok(())
    }
}
