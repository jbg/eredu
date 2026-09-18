//! Completed source and local-cache declarations use real selected partitions.
use super::*;
use crate::preparation_selection::{
    select_preparation,
    tests::{BoundedIndependentAdapter, inspected_config, routed_config},
};
use eredu_nn::{Tensor, workspace::*};
use std::{cell::RefCell, rc::Rc};
mod source_binding;
#[derive(Debug)]
struct NoPayload;
impl WorkspaceMechanisms for NoPayload {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
type Captured = Rc<
    RefCell<
        Option<(
            crate::partitioned_execution::PreparedRoutedExecutionHandoff,
            BTreeMap<eredu_runtime::RoutedBankId, SelectedRoutedBank>,
        )>,
    >,
>;
struct Probe(Captured);
impl
    crate::partitioned_execution::RoutedPartitionedProductionVisitor<
        WorkspaceBackend,
        ResidentState,
    > for Probe
{
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<WorkspaceBackend,A,G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend,ResidentState>>::Boundary>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: TextPartitionArchitecture<WorkspaceBackend, ResidentState>
            + eredu_runtime::ReplicatedTextArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = Error,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        *self.0.borrow_mut() = Some((
            prepared.execution_handoff().clone(),
            prepared.banks().clone(),
        ));
        Ok(collect_banks(
            prepared.bank_residency(),
            prepared.banks(),
            Some(prepared.layout()),
        ))
    }
}
#[test]
fn completed_independent_partition_retains_exact_source_and_local_cache_equation() {
    let mut config = routed_config();
    config["num_key_value_heads"] = 2.into();
    let tensor_size = 2;
    let mut empty_owners = 0;
    let mut whole_unit_refusals = 0;
    for (pipeline_size, expert_size, dense_stage) in [(2, 2, false), (1, 2, false), (2, 2, true)] {
        let mut fixture = config.clone();
        if dense_stage {
            // This family explicitly supports a dense-only first decoder stage.
            // Expert ownership remains nonempty; there is no local bank invocation.
            fixture["model_type"] = "k2_horizon".into();
            fixture.as_object_mut().unwrap().remove("architectures");
            fixture["mlp_only_layers"] = serde_json::json!([0]);
        }
        let (_directory, inspection) = inspected_config(fixture);
        let topology =
            eredu_core::ParallelTopology::new(tensor_size, pipeline_size, expert_size, 1).unwrap();
        let ranks = tensor_size * pipeline_size * expert_size;
        for index in 0..ranks {
            let rank = eredu_core::ParallelRankTopology::new(topology, index).unwrap();
            let completion = eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap();
            let parallel = eredu_runtime::ParallelLoadRequest::new(
                rank,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                32,
                completion,
            )
            .unwrap();
            let options = eredu_runtime::ParameterBankLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1152), Some(1 << 20), 1).unwrap(),
                1152,
                1152,
            )
            .unwrap();
            let request = eredu_runtime::NormalizedLoadRequest::default()
                .with_parallel_execution(parallel)
                .unwrap()
                .with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::FullyResident,
                        options,
                    ),
                );
            let selected =
                select_preparation(&inspection, &request, &BoundedIndependentAdapter::default())
                    .unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection.clone(),
                request.preparation_policy().unwrap(),
                selected.session_capabilities(),
            )
            .unwrap();
            let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
            assert!(
                sources
                    .construction_semantics()
                    .direct_partition
                    .get()
                    .is_none()
            );
            let captured = Captured::default();
            let second = captured.clone();
            let context = WorkspaceContext::new(NoPayload);
            let routes =
                PreparedExecutionRoutes::new().with_partitioned_routed(PartitionedRoutedRoute::<
                    WorkspaceBackend,
                    ResidentState,
                    ResidentState,
                    _,
                    _,
                >::new(
                    &context,
                    &context,
                    |_: PreparedPartitionResources<()>| Probe(captured.clone()),
                    |_: PreparedPartitionResources<()>| Probe(second),
                ));
            let output = construct_prepared_execution(
                sources.clone(),
                Some(()),
                routes,
                AddressableAssembler {
                    manifest: sources.selected().communication_manifest(),
                    dtype: Some(eredu_runtime::StateStorageDtype::F32),
                },
            )
            .unwrap();
            let retained = sources
                .construction_semantics()
                .direct_partition
                .get()
                .unwrap();
            assert!(retained.tensor_waves(sources.selected(), rank).is_ok());
            let foreign =
                eredu_core::ParallelRankTopology::new(topology, (index + 1) % ranks).unwrap();
            assert!(retained.tensor_waves(sources.selected(), foreign).is_err());
            assert!(
                retained
                    .routed_resident_source(sources.selected(), rank)
                    .is_err()
            );
            let (execution, banks) = captured.borrow_mut().take().unwrap();
            let (plans, group) = execution.expert_region_source();
            let (&id, bank) = banks.iter().next().unwrap();
            let plan = plans[&id].gated().unwrap();
            assert_eq!(output.is_some(), !bank.addressable_members().is_empty());
            let physical = source_binding::physical_bytes(&banks);
            if physical.is_empty() {
                assert!(dense_stage);
                assert_eq!(rank.layer_range(2).unwrap(), 0..1);
                assert!(!plan.local_global_group_indices().is_empty());
                assert!(plan.unit_spec(bank.owner_group().as_str(), 0).is_none());
                let source = bank.plan().partition_source().unwrap();
                assert!(source.require_completion().is_err());
                source_binding::bind(&banks, options, &physical, &physical).unwrap();
                source_binding::bind(&banks, options, &physical, &physical).unwrap();
                source.require_completion().unwrap();
                assert!(std::sync::Arc::ptr_eq(
                    bank.plan().retained_partition_source().unwrap(),
                    plans[&id].retained_partition_source().unwrap()
                ));
                // Completion of no local members does not authorize an invented
                // routed unit or provide a maximum for an absent invocation.
                assert!(source.maximum(0, options).is_err());
                empty_owners += 1;
                continue;
            }
            let unit = bank.addressable_members().first().unwrap().key().unit();
            let spec = plan.unit_spec(bank.owner_group().as_str(), unit).unwrap();
            let width = WorkspaceExpertKernel::Gated(spec).dimensions().0;
            let (&key, &bytes) = physical.iter().next().unwrap();
            let mut missing = physical.clone();
            missing.remove(&key);
            source_binding::refused(source_binding::bind(&banks, options, &missing, &physical));
            let mut wrong = physical.clone();
            wrong.insert(key, bytes + 1);
            source_binding::refused(source_binding::bind(&banks, options, &wrong, &physical));
            let mut foreign = physical.clone();
            foreign.insert(
                eredu_runtime::ParameterBankKey::new(key.bank(), key.unit(), usize::MAX),
                bytes,
            );
            source_binding::refused(source_binding::bind(&banks, options, &foreign, &foreign));
            if physical.keys().any(|other| other.unit() != key.unit()) {
                let without_unit = physical
                    .iter()
                    .filter(|(other, _)| other.unit() != key.unit())
                    .map(|(key, bytes)| (*key, *bytes))
                    .collect();
                source_binding::refused(source_binding::bind(
                    &banks,
                    options,
                    &without_unit,
                    &physical,
                ));
                whole_unit_refusals += 1;
            }
            // Repeated construction must adopt the winning published source owner,
            // rather than leave the new native provider paired with a fresh slot.
            let retry = Captured::default();
            let second = retry.clone();
            let routes =
                PreparedExecutionRoutes::new().with_partitioned_routed(PartitionedRoutedRoute::<
                    WorkspaceBackend,
                    ResidentState,
                    ResidentState,
                    _,
                    _,
                >::new(
                    &context,
                    &context,
                    |_: PreparedPartitionResources<()>| Probe(retry.clone()),
                    |_: PreparedPartitionResources<()>| Probe(second),
                ));
            construct_prepared_execution(
                sources.clone(),
                Some(()),
                routes,
                AddressableAssembler {
                    manifest: sources.selected().communication_manifest(),
                    dtype: Some(eredu_runtime::StateStorageDtype::F32),
                },
            )
            .unwrap();
            let (retry_execution, retry_banks) = retry.borrow_mut().take().unwrap();
            for (id, bank) in &banks {
                assert!(std::sync::Arc::ptr_eq(
                    bank.plan().retained_partition_source().unwrap(),
                    retry_banks[id].plan().retained_partition_source().unwrap()
                ));
                assert!(std::sync::Arc::ptr_eq(
                    bank.plan().retained_partition_source().unwrap(),
                    retry_execution.expert_region_source().0[id]
                        .retained_partition_source()
                        .unwrap()
                ));
            }
            for completed in [false, true] {
                if completed {
                    source_binding::bind(&retry_banks, options, &physical, &physical).unwrap();
                    source_binding::bind(&banks, options, &physical, &physical).unwrap();
                    // A second genuinely consistent mechanism may not replace the
                    // physical source already retained by this prepared partition.
                    source_binding::refused(source_binding::bind(&banks, options, &wrong, &wrong));
                }
                for source_rows in [0, 3] {
                    let input =
                        WorkspaceTensor::full_f32(0.5, &[source_rows, width], &context).unwrap();
                    let count = i32::try_from(bank.routes_by_unit()[&unit]).unwrap();
                    let ids =
                        WorkspaceTensor::full_i32(0, &[source_rows, count], &context).unwrap();
                    let scores =
                        WorkspaceTensor::full_f32(0.25, &[source_rows, count], &context).unwrap();
                    let routes = eredu_nn::GroupSelection::new(ids, scores.clone(), scores);
                    for pass in [
                        eredu_runtime::ExpertPass::Prefill,
                        eredu_runtime::ExpertPass::Decode,
                    ] {
                        let request = eredu_runtime::RoutedExpertRequest {
                            bank: id,
                            layer: unit,
                            input: &input,
                            routes: &routes,
                            pass,
                            unit_observer: None,
                        };
                        let (tensor, wave) = execution.expert_provider_groups().unwrap();
                        let region = crate::partitioned_execution::partition_region_declaration::<
                            WorkspaceBackend,
                            _,
                        >(
                            plan,
                            &request,
                            group.unwrap(),
                            (tensor_size > 1).then_some(tensor_size),
                            tensor,
                            wave,
                            WorkspaceExpertKernel::Gated,
                            &context,
                        );
                        let direct = bank.addressable_workspace_source(
                            &request,
                            options,
                            Some(tensor_size),
                            false,
                            &context,
                        );
                        if !completed {
                            assert!(
                                region.is_err(),
                                "EP quote must require successful physical completion, including decode"
                            );
                            assert!(
                                direct.is_err(),
                                "direct quote must require successful physical completion, including decode"
                            );
                            continue;
                        }
                        let region=region.unwrap_or_else(|cause| panic!("rank {index} unit {unit} rows {source_rows} groups {} local {:?}: {cause:?}",spec.group_count(),plan.local_global_group_indices()));
                        region.validate().unwrap();
                        if plan.local_global_group_indices().is_empty() {
                            assert_eq!(region.maximum_received_rows(), Some(0));
                            assert!(region.addressable.is_none());
                            continue;
                        }
                        let addressable = region.addressable.unwrap();
                        assert_eq!(addressable.chunks.rows, source_rows as usize * expert_size);
                        assert_eq!(addressable.chunks.routes, 1);
                        assert_eq!(
                            addressable.local_members,
                            Some(plan.local_global_group_indices())
                        );
                        let maximum = physical
                            .iter()
                            .filter(|(key, _)| key.unit() == unit)
                            .map(|(_, bytes)| *bytes)
                            .max()
                            .unwrap();
                        let expected = eredu_runtime::expert::AddressableChunkPlan::new(
                            source_rows as usize * expert_size,
                            1,
                            plan.local_global_group_indices().len(),
                            pass.parameter_bank_access(),
                            Some(maximum),
                            1152,
                        )
                        .unwrap()
                        .workspace_source();
                        assert_eq!(addressable.chunks, expected);
                        let direct = direct.unwrap();
                        let expected_direct = eredu_runtime::expert::AddressableChunkPlan::new(
                            source_rows as usize,
                            1,
                            plan.local_global_group_indices().len(),
                            pass.parameter_bank_access(),
                            Some(maximum),
                            1152,
                        )
                        .unwrap()
                        .workspace_source();
                        assert_eq!(direct.chunks, expected_direct);
                        if matches!(pass, eredu_runtime::ExpertPass::Prefill) {
                            let old_maximum = bank
                                .addressable_members()
                                .iter()
                                .filter(|member| member.key().unit() == unit)
                                .map(|member| member.selected_bytes())
                                .max()
                                .unwrap();
                            let stale = eredu_runtime::expert::AddressableChunkPlan::new(
                                source_rows as usize * expert_size,
                                1,
                                plan.local_global_group_indices().len(),
                                pass.parameter_bank_access(),
                                Some(old_maximum),
                                1152,
                            )
                            .unwrap();
                            assert_ne!(
                                addressable.chunks.chunk_rows,
                                stale.workspace_source().chunk_rows
                            );
                        }
                        crate::architecture_parameter_metadata_tests::boundary(|paid| {
                            let retained = region.retain(paid.unwrap_or(&context))?;
                            assert_eq!(retained.as_view(), region);
                            Ok(())
                        });
                        let mut wrong = region;
                        wrong.addressable.as_mut().unwrap().chunks.rows += 1;
                        assert!(wrong.validate().is_err());
                        wrong = region;
                        wrong.addressable.as_mut().unwrap().bank += 1;
                        assert!(wrong.validate().is_err());
                    }
                }
            }
            source_binding::bind(&retry_banks, options, &physical, &physical).unwrap();
            source_binding::refused(source_binding::bind(&retry_banks, options, &wrong, &wrong));
        }
    }
    assert!(empty_owners > 0);
    assert!(whole_unit_refusals > 0);
}
