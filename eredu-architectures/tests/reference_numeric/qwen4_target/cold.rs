//! Exact checkpoint-derived requirements through neutral selection and binding.
use super::*;
use eredu_architectures::qwen4_exp::prepared::PreparedTarget;
use eredu_architectures::routed_text::{
    PreparedRoutedTextArchitecture, RoutedTextArchitectureVisitor, RoutedTextSelectionRequest,
};
use eredu_runtime::*;
pub(super) struct Support;
impl RowLookupMechanismSupport for Support {
    fn storage(&self) -> Option<AddressableStorageCapabilities> {
        Some(AddressableStorageCapabilities::new(
            true,
            true,
            true,
            1 << 20,
        ))
    }
    fn workspace(
        &self,
        _: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
        Ok(Some(RowLookupWorkspace {
            decode_bytes: 64,
            scalar_bytes: 0,
        }))
    }
}
pub(super) fn check(
    prepared: &PreparedTarget,
    model: &TargetModel<NumericBackend>,
    source: SharedCheckpointSource,
    ctx: &NumericContext,
) {
    let before_header = source.source_diagnostics().unwrap().physical_reads;
    let header = eredu_architectures::qwen4_exp::prepared::SafetensorsTargetPlan::prepare(
        source.as_ref(),
        configuration(),
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &serde_json::json!({}),
        )
        .unwrap(),
    )
    .unwrap();
    let physical = header
        .resolution()
        .source_keys()
        .iter()
        .map(|key| {
            let provenance = source.source_provenance(key).unwrap();
            let metadata = source.source_metadata(key).unwrap();
            (
                key.clone(),
                ReplicatedTextPhysicalSource::new(
                    provenance.catalog_key,
                    provenance.physical_tensor,
                    provenance.backing_shard.unwrap(),
                    provenance.output,
                    provenance.source_encoding,
                    metadata.encoded_byte_len,
                )
                .unwrap(),
            )
        })
        .collect();
    let descriptors = header
        .row_descriptors(
            prepared.spec().limits,
            RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 16,
                host_bytes: 16384,
                output_bytes: 3072,
            },
            eredu_core::residency::ResidencyPolicy::Cacheable,
        )
        .unwrap();
    let row_plan = SelectedRowLookupPlans::select(
        descriptors,
        ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(256), Some(0), 1).unwrap(),
            32768,
            32768,
        )
        .unwrap(),
        0,
        &Support,
    )
    .unwrap();
    let plan = header
        .execution_plan(
            prepared.spec().limits,
            stream_bindings(prepared.spec()),
            row_plan,
            physical,
        )
        .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        before_header
    );
    let text = plan.requirements().text();
    let description = model.parameter_description(ctx).unwrap();
    let actual: BTreeMap<_, _> = description
        .groups()
        .iter()
        .flat_map(|g| g.members())
        .map(|m| (m.target(), m.global_shape()))
        .collect();
    let expected: BTreeMap<_, _> = text
        .parameters()
        .iter()
        .map(|p| (p.name(), p.logical_shape()))
        .collect();
    assert_eq!(actual, expected);
    let capabilities = capabilities(plan.requirements(), None);
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 20, 0, 0, 0).unwrap(),
        ),
    ] {
        let before = source.source_diagnostics().unwrap().physical_reads;
        let request = RoutedTextSelectionRequest::new(
            ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
                .with_session(eredu_core::SessionCapabilities::new(false, true, true))
                .with_exact_completion(true),
            WeightResidency::with_layers(residency),
        )
        .unwrap();
        assert!(plan
            .clone()
            .select(
                &request,
                &capabilities.clone().with_exact_completion(false),
                None
            )
            .is_err());
        let selected = plan.clone().select(&request, &capabilities, None).unwrap();
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            before,
            "cold selection must not read weights"
        );
        let selected = selected.bind(source.clone(), None).unwrap();
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            before + 3,
            "binding acquires only three integer control vectors"
        );
        use eredu_architectures::prepared_execution::{
            RetainedArchitectureConstruction, RoutedRoute,
        };
        RoutedRoute::<NumericBackend, (), (), State, _, _, _, _>::new(
            ctx,
            ctx,
            (),
            (),
            (),
            Binding {
                ctx,
                source: source.clone(),
                residency,
            },
        )
        .construct_retained(
            RetainedArchitectureConstruction::Qwen4Exp(Box::new(selected)),
            None,
        )
        .unwrap();
    }
}

struct Binding<'a> {
    ctx: &'a NumericContext,
    source: SharedCheckpointSource,
    residency: LayerWeightResidency,
}
impl RoutedTextArchitectureVisitor<NumericBackend, State> for Binding<'_> {
    type Output = ();
    type Error = String;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        source: SharedCheckpointSource,
    ) -> Result<(), String>
    where
        A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
    {
        let ctx = self.ctx;
        let before = self.source.source_diagnostics().unwrap().physical_reads;
        let mut mechanisms = if self.residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(source)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(source)
        };
        mechanisms.fixture_state_factory = Some(|layout| {
            assert_eq!(*layout, specification().state_layout().unwrap());
            Ok(state(&specification()))
        });
        macro_rules! exercise {
            ($actual:expr, $facts:expr) => {{
                let (_, capability, model_type, residency) = $facts.into_parts();
                assert_eq!(model_type, "qwen4_exp_text");
                assert_eq!(residency, self.residency);
                assert_eq!(
                    capability.state_layout().layer_layout(),
                    specification().state_layout().unwrap().layers()
                );
                assert_eq!(
                    capability.capabilities().estimation,
                    eredu_core::EstimationCompleteness::PersistentStateOnly
                );
                let mut actual = $actual;
                let mut expected = super::session::session(ctx);
                let ids = [3, 4, 7, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
                for part in [&ids[..3], &ids[3..8], &ids[8..]] {
                    let tokens =
                        NumericTensor::from_i32_slice(part, &[1, part.len() as i32], ctx).unwrap();
                    assert_tensor_close(
                        &actual.prefill(&tokens, None, ctx).unwrap(),
                        &expected.prefill(&tokens, None, ctx).unwrap(),
                        "checkpoint bound prefill",
                    );
                }
                for token in 0..16 {
                    let token = NumericTensor::from_i32_slice(&[token], &[1, 1], ctx).unwrap();
                    assert_tensor_close(
                        &actual.decode(&token, ctx).unwrap(),
                        &expected.decode(&token, ctx).unwrap(),
                        "checkpoint bound cached decode",
                    );
                }
                let saved = actual.checkpoint(ctx).unwrap();
                super::session::assert_checkpoint(&saved, &expected.checkpoint(ctx).unwrap());
                let token = NumericTensor::from_i32_slice(&[5], &[1, 1], ctx).unwrap();
                let first = actual.decode(&token, ctx).unwrap();
                let charged = self.source.source_diagnostics().unwrap().physical_reads;
                actual.rollback(saved, ctx).unwrap();
                assert_eq!(
                    self.source.source_diagnostics().unwrap().physical_reads,
                    charged
                );
                assert_tensor_close(
                    &first,
                    &actual.decode(&token, ctx).unwrap(),
                    "checkpoint bound rollback",
                );
                assert!(self.source.source_diagnostics().unwrap().physical_reads > before);
                assert!(actual
                    .execution_strategy()
                    .provider()
                    .rows
                    .as_ref()
                    .unwrap()
                    .providers()
                    .values()
                    .all(|p| p.bank().completed() > 0));
                Ok::<_, String>(())
            }};
        }
        eredu_architectures::prepared_execution::construct_selected_routed_session(
            prepared,
            mechanisms,
            ctx,
            |_, _, rows| {
                let rows = rows.expect("the retained target has a lexical table");
                let providers = rows
                    .prepared()
                    .bind(
                        rows.prepared()
                            .entries()
                            .iter()
                            .map(|(id, entry)| {
                                (id.clone(), super::super::row_bank::SourceRows::new(entry))
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
            |_, actual, facts| exercise!(actual, facts),
            |_, actual, facts| exercise!(actual, facts),
        )
        .map_err(|e| e.to_string())
    }
}

pub(super) fn capabilities(
    requirements: &eredu_architectures::routed_text::RoutedTextRequirements,
    quantization: Option<eredu_core::QuantizationRequest>,
) -> BackendMechanismCapabilities {
    let text = requirements.text();
    BackendMechanismCapabilities::new(
        text.operators(),
        text.parameters()
            .iter()
            .chain(text.auxiliary_parameters())
            .flat_map(|p| {
                let mut lowerings = vec![WeightLoweringCapability::new(
                    p.lowering_descriptor(p.native_executable()).unwrap(),
                    WeightLoweringKind::Direct,
                )];
                if let Some(target) = quantization.and_then(|q| p.transform_target(q).unwrap()) {
                    lowerings.push(WeightLoweringCapability::new(
                        target.descriptor().clone(),
                        WeightLoweringKind::Transform,
                    ));
                }
                lowerings
            })
            .collect(),
        vec![
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
            WeightResidencyMechanism::DiskStreamed,
        ],
        StateMechanismCapabilities::new(text.state_layout().layers().iter().enumerate().flat_map(
            |(layer, p)| {
                p.components().into_iter().map(move |c| {
                    StateComponentMechanism::new(
                        layer,
                        c,
                        Some(StateComponentPlacement::Device),
                        None,
                    )
                })
            },
        ))
        .with_floating_state_dtype(
            eredu_core::checkpoint::TensorDtype::F32,
            StateStorageDtype::F32,
        )
        .with_transactions(true, true)
        .with_reset(true)
        .with_observation_retention(true),
    )
    .with_grouped_operations([GroupedOperationRequirement::GatedProduct])
    .with_session(eredu_core::SessionCapabilities::new(false, true, true))
    .with_exact_completion(true)
    .with_chunked_prefill(true)
    .with_addressable_storage(Support.storage().unwrap())
}
