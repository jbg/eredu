//! Selected grouped/row provider composition through the ordinary session driver.
use super::*;
use eredu_architectures::routed_text::{
    self, RoutedBankRequirements, RoutedTextRequirements, RoutedTextSelectionRequest,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    rows::PreparedRowSource,
    store::{MemoryWeightStore, SharedCheckpointSource, TensorSelection},
};
use eredu_core::residency::{OffloadConfig, OffloadUnitId, OffloadUnitRange, ResidencyPolicy};

struct Support;
impl RowLookupMechanismSupport for Support {
    fn storage(&self) -> Option<AddressableStorageCapabilities> {
        Some(AddressableStorageCapabilities::new(
            true,
            true,
            true,
            u64::MAX,
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
use crate::row_bank::SourceRows;
fn row_selection(
    source: SharedCheckpointSource,
    options: ParameterBankLoadOptions,
) -> SelectedRowLookups {
    let table = PreparedRowSource::new(
        source,
        DerivedWeightRecipe::source("table", TensorSelection::Full),
    )
    .unwrap();
    let range = RowResidencyRange::new(
        OffloadUnitRange::new(
            OffloadUnitId::new("rows").unwrap(),
            0,
            24,
            8,
            ResidencyPolicy::Cacheable,
        )
        .unwrap(),
        table,
        "value",
    )
    .unwrap();
    let entry = PreparedRowLookup::new(
        range,
        RowLookupSpec {
            parameter: eredu_nn::ParameterId::new(
                "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
            )
            .unwrap(),
            bank: 2,
            unit: 1,
            rows: 24,
            dimensions: 2,
            encoding: RowEncoding::Dense,
            output_type: eredu_nn::TensorElementType::F32,
        },
        None,
        RowLookupLimits {
            requests: 128,
            rows_per_acquisition: 2,
            acquisition_bytes: 16,
            host_bytes: 16384,
            output_bytes: 3072,
        },
    )
    .unwrap();
    SelectedRowLookups::select(
        PreparedRowLookups::new([entry], 5).unwrap(),
        options,
        0,
        &Support,
    )
    .unwrap()
}

#[test]
fn qwen4_selected_sessions_bind_retained_rows_for_resident_and_cached_experts() {
    let ctx = NumericContext::default();
    let directory = tempfile::tempdir().unwrap();
    let data: Vec<u8> = (0..24)
        .flat_map(|r| [(r as f32 + 0.25) / 17., -(r as f32 + 0.5) / 19.])
        .flat_map(f32::to_le_bytes)
        .collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![24, 2], &data).unwrap();
    safetensors::tensor::serialize_to_file(
        [("table", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source: SharedCheckpointSource =
        Arc::new(eredu_checkpoint::store::SafetensorsWeightStore::open(directory.path()).unwrap());
    let options = ParameterBankLoadOptions::new(
        OffloadConfig::new(Some(256), Some(0), 1).unwrap(),
        32768,
        32768,
    )
    .unwrap();
    let rows = row_selection(source.clone(), options);
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    for cached in [false, true] {
        let (architecture, text, capabilities, identity) =
            setup(&ctx, model(specification(), &ctx), false);
        let capabilities = capabilities
            .with_grouped_operations([GroupedOperationRequirement::GatedProduct])
            .with_indexed_movement(true)
            .with_addressable_storage(Support.storage().unwrap());
        let owner = text.execution_units().group_id(0).unwrap().clone();
        let source_values = text
            .parameters()
            .iter()
            .map(|p| {
                (
                    p.name().to_owned(),
                    safetensors::Dtype::F32,
                    p.logical_shape().to_vec(),
                    vec![0u8; p.logical_shape().iter().product::<usize>() * 4],
                )
            })
            .collect::<Vec<_>>();
        let catalog_source = MemoryWeightStore::from_safetensors(source_values).unwrap();
        let spec = specification();
        let mut units = vec![];
        let mut specs = BTreeMap::new();
        let mut routes = BTreeMap::new();
        for (unit, value) in spec.units.iter().enumerate() {
            let UnitSpec::Decoder { feed_forward, .. } = value else {
                continue;
            };
            let experts = feed_forward.feed_forward.experts.clone();
            specs.insert((owner.clone(), unit), experts.clone());
            routes.insert(unit, configuration().experts.selected as usize);
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = experts.layout()
            else {
                panic!("packed fixture")
            };
            for expert in 0..3 {
                units.push(
                    ExpertResidencyUnit::new(
                        ParameterBankKey::new(0, unit, expert),
                        owner.clone(),
                        unit,
                        value.path(),
                        ExpertResidencyDistribution::ExpertParallel,
                        [("gate_up_proj", gate_up), ("down_proj", down)].map(
                            |(binding, projection)| {
                                let name = projection.weight().id.as_str();
                                ExpertParameterRecipe::new(
                                    binding,
                                    name,
                                    DerivedWeightRecipe::source(
                                        name,
                                        TensorSelection::Range {
                                            axis: 0,
                                            start: expert,
                                            end: expert + 1,
                                        },
                                    ),
                                    ExpertParameterRole::Preserved,
                                )
                                .unwrap()
                            },
                        ),
                    )
                    .unwrap(),
                );
            }
        }
        let catalog = ExpertResidencyCatalog::new(units)
            .unwrap()
            .with_inferred_byte_geometry(&catalog_source)
            .unwrap();
        let plan = ExpertRealizationPlan::balanced(
            3,
            eredu_core::topology::ParallelRankTopology::new(
                eredu_core::topology::ParallelTopology::new(1, 1, 1, 1).unwrap(),
                0,
            )
            .unwrap(),
            specs,
        )
        .unwrap();
        let bank = RoutedBankRequirements::new(owner, plan.into(), catalog, routes).unwrap();
        let plain =
            RoutedTextRequirements::new(text, [(RoutedBankId::new(0), bank)], &catalog_source)
                .unwrap();
        let requirements = plain.clone().with_row_lookups(rows.plan().clone()).unwrap();
        let residency = if cached {
            WeightResidency::with_independent_parameter_banks(
                OrdinaryWeightResidency::FullyResident,
                options,
            )
        } else {
            WeightResidency::fully_resident()
        };
        let request = RoutedTextSelectionRequest::new(
            ReplicatedTextSelectionRequest::new(
                LayerWeightResidency::FullyResident,
                CacheResidencyPolicy::Device,
            )
            .with_session(eredu_core::SessionCapabilities::new(false, true, true))
            .with_exact_completion(true),
            residency,
        )
        .unwrap();
        let selected =
            routed_text::select_routed_text_realization(&requirements, &request, &capabilities)
                .unwrap();
        assert_eq!(selected.row_lookups(), Some(rows.plan()));
        let estimate = eredu_architectures::capability::qwen4_exp_target(&spec).unwrap();
        let prepare = |expected, selected| {
            routed_text::prepare_routed_architecture_handoff::<NumericBackend, State, _>(
                model(specification(), &ctx),
                None,
                expected,
                selected,
                Some(rows.prepared().clone()),
                estimate.clone(),
                "qwen4_exp_text".into(),
                identity.clone(),
                &ctx,
            )
        };
        assert!(
            prepare(plain, selected.clone()).is_err(),
            "handoff cannot drop row authority"
        );
        let replacement = row_selection(source.clone(), options);
        assert_ne!(
            rows, replacement,
            "equal geometry must not replace the prepared binding"
        );
        assert_eq!(
            rows.plan(),
            replacement.plan(),
            "cold requirements retain descriptors, never prepared source identity"
        );
        let rebound = routed_text::prepare_routed_architecture_handoff::<NumericBackend, State, _>(
            model(specification(), &ctx),
            None,
            requirements.clone(),
            selected.clone(),
            Some(replacement.prepared().clone()),
            estimate.clone(),
            "qwen4_exp_text".into(),
            identity.clone(),
            &ctx,
        )
        .unwrap();
        assert_eq!(rebound.row_lookups(), Some(&replacement));
        assert_ne!(rebound.row_lookups(), Some(&rows));
        assert!(
            routed_text::prepare_routed_architecture_handoff::<NumericBackend, State, _>(
                model(specification(), &ctx),
                None,
                requirements.clone(),
                selected.clone(),
                None,
                estimate.clone(),
                "qwen4_exp_text".into(),
                identity.clone(),
                &ctx,
            )
            .is_err(),
            "selected rows require an explicit prepared source binding"
        );
        let mismatch = RoutedTextSelectionRequest::new(
            request.text().clone(),
            WeightResidency::with_independent_parameter_banks(
                OrdinaryWeightResidency::FullyResident,
                ParameterBankLoadOptions::default(),
            ),
        )
        .unwrap();
        assert!(
            routed_text::select_routed_text_realization(&requirements, &mismatch, &capabilities)
                .is_err(),
            "table and expert policies cannot be different"
        );
        let missing = prepare(requirements.clone(), selected.clone()).unwrap()
            .construct_resident_session::<NumericBackend, NumericReplicatedMechanisms, NoRowLookups>(NumericReplicatedMechanisms::default(), Some(NoRowLookups), &[], &ctx).err().unwrap();
        assert!(missing.contains("missing selected row provider"));
        let absent = prepare(requirements.clone(), selected.clone()).unwrap()
            .construct_resident_session::<NumericBackend, NumericReplicatedMechanisms, NoRowLookups>(NumericReplicatedMechanisms::default(), None, &[], &ctx).err().unwrap();
        assert!(absent.contains("presence differs"));
        let prepared = prepare(requirements, selected).unwrap();
        drop(architecture);
        fn initialize(metadata: ParameterMetadata, value: &mut NumericTensor) {
            Parameters::default().visit_mut(metadata, value);
        }
        fn make_state(layout: &StateLayout) -> Result<State, Error> {
            assert_eq!(*layout, specification().state_layout().unwrap());
            Ok(state(&specification()))
        }
        macro_rules! exercise {
            ($session:expr) => {{
                let mut actual = $session;
                let mut expected = session(&ctx);
                let ids = [3, 4, 7, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
                for part in [&ids[..3], &ids[3..8], &ids[8..]] {
                    let tokens =
                        NumericTensor::from_i32_slice(part, &[1, part.len() as i32], &ctx).unwrap();
                    assert_tensor_close(
                        &actual.prefill(&tokens, None, &ctx).unwrap(),
                        &expected.prefill(&tokens, None, &ctx).unwrap(),
                        "prepared row chunk",
                    );
                }
                for token in 0..16 {
                    let token = NumericTensor::from_i32_slice(&[token], &[1, 1], &ctx).unwrap();
                    assert_tensor_close(
                        &actual.decode(&token, &ctx).unwrap(),
                        &expected.decode(&token, &ctx).unwrap(),
                        "prepared row cached decode",
                    );
                }
                let saved = actual.checkpoint(&ctx).unwrap();
                let reference = expected.checkpoint(&ctx).unwrap();
                assert_checkpoint(&saved, &reference);
                let token = NumericTensor::from_i32_slice(&[5], &[1, 1], &ctx).unwrap();
                let reads = source.source_diagnostics().unwrap().physical_read_bytes;
                assert!(actual
                    .decode_with_observer(
                        &token,
                        &ctx,
                        &mut Observe {
                            paths: vec![],
                            fail: true
                        }
                    )
                    .is_err());
                assert_checkpoint(&actual.checkpoint(&ctx).unwrap(), &saved);
                assert!(
                    source.source_diagnostics().unwrap().physical_read_bytes > reads,
                    "failed observations do not refund table reads"
                );
                let first = actual.decode(&token, &ctx).unwrap();
                let charged = source.source_diagnostics().unwrap().physical_read_bytes;
                actual.rollback(saved.clone(), &ctx).unwrap();
                assert_eq!(
                    source.source_diagnostics().unwrap().physical_read_bytes,
                    charged
                );
                assert_tensor_close(
                    &first,
                    &actual.decode(&token, &ctx).unwrap(),
                    "prepared row rollback",
                );
                let provider = actual
                    .execution_strategy()
                    .provider()
                    .rows
                    .as_ref()
                    .unwrap();
                assert!(provider
                    .providers()
                    .values()
                    .all(|p| p.bank().completed() > 0));
                Ok::<_, String>(())
            }};
        }
        eredu_architectures::prepared_execution::construct_selected_routed_session(
            prepared,
            NumericReplicatedMechanisms {
                fixture_parameter_init: Some(initialize),
                fixture_state_factory: Some(make_state),
                ..Default::default()
            },
            &ctx,
            |banks, residency, selected| {
                let selected = selected.expect("retained nonempty row selection");
                assert_eq!(selected, &rows);
                let native = selected
                    .prepared()
                    .entries()
                    .iter()
                    .map(|(id, entry)| (id.clone(), SourceRows::new(entry)))
                    .collect();
                let providers = selected.prepared().bind(native).unwrap();
                let grouped = if matches!(residency, ParameterBankResidency::IndependentCache(_)) {
                    banks
                        .iter()
                        .map(|(id, bank)| {
                            Ok((
                                *id,
                                (
                                    numeric_addressable_gated_bank(
                                        bank.plan().gated().unwrap(),
                                        bank.catalog(),
                                        bank.addressable_members(),
                                        5,
                                        &ctx,
                                    )?,
                                    NumericIndexedMovement,
                                ),
                            ))
                        })
                        .collect::<Result<_, String>>()?
                } else {
                    BTreeMap::new()
                };
                Ok::<_, String>((grouped, Some(providers)))
            },
            (),
            |_, session, _| exercise!(session),
            |_, session, _| exercise!(session),
        )
        .unwrap();
    }
    let reads = source.source_diagnostics().unwrap();
    assert!(reads.physical_read_bytes > 0);
    assert_eq!(
        reads.physical_read_bytes,
        reads.physical_reads * 8,
        "every physical read is one requested row"
    );
}
