//! Actual prepared partitions must reach the same exact binding worker cold.
use super::*;
use crate::preparation_selection::{
    select_preparation,
    tests::{BoundedIndependentAdapter, composite_config, inspected_config, routed_config},
};
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};

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

fn check(config: serde_json::Value) {
    let (_directory, inspection) = inspected_config(config);
    let mut excluded_required = 0;
    let mut completed = 0;
    for pipeline in [1, 2] {
        let topology = eredu_core::ParallelTopology::new(2, pipeline, 2, 1).unwrap();
        for rank in 0..4 * pipeline {
            let topology = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
            let parallel = eredu_runtime::ParallelLoadRequest::new(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                32,
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
            .unwrap();
            let bank = eredu_runtime::ParameterBankLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
                1 << 20,
                1 << 20,
            )
            .unwrap();
            let request = eredu_runtime::NormalizedLoadRequest::default()
                .with_parallel_execution(parallel)
                .unwrap()
                .with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(
                            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1)
                                .unwrap(),
                        ),
                        bank,
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
            let context = WorkspaceContext::new(NoPayload);
            let projection =
                project_partitioned_text_binding_destinations(&sources, &context).unwrap();
            let repeated =
                project_partitioned_text_binding_destinations(&sources, &context).unwrap();
            assert_eq!(
                projection.excluded_parameters(),
                repeated.excluded_parameters()
            );
            assert_eq!(projection.units(), repeated.units());
            assert_eq!(projection.static_parameters(), repeated.static_parameters());
            assert_eq!(projection.physical_layout(), repeated.physical_layout());
            assert_eq!(
                projection.addresses().len(),
                projection.local_layout().len()
            );
            assert_eq!(projection.units().len(), projection.local_layout().len());
            assert!(!projection.excluded_parameters().is_empty());
            let plan = eredu_runtime::plan_local_replicated_text_materialization_tasks(
                projection.tasks(),
                projection.global_layout(),
                projection.addresses(),
            )
            .unwrap();
            let static_tasks = plan.static_tasks(projection.tasks()).unwrap();
            let unit_tasks = plan.unit_tasks(projection.tasks()).unwrap();
            for (destinations, tasks) in
                std::iter::once((projection.static_parameters(), &static_tasks))
                    .chain(projection.units().iter().zip(&unit_tasks))
            {
                let targets = destinations
                    .iter()
                    .map(|(name, layout)| {
                        assert_eq!(layout.dtype(), WorkspaceDtype::Float32);
                        (
                            name.clone(),
                            eredu_runtime::ParameterBindingTarget {
                                shape: layout
                                    .shape()
                                    .iter()
                                    .map(|&extent| usize::try_from(extent).unwrap())
                                    .collect(),
                                dtype: eredu_checkpoint::recipe::RecipeDtype::F32,
                                permitted_source_dtypes: Vec::new(),
                            },
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                let bind = |excluded: &BTreeSet<String>| {
                    eredu_runtime::build_exact_replicated_text_bindings_for_targets(&targets, sources.target().as_ref(),
                        tasks, excluded, Some(projection.physical_layout()),
                        |_, _, _| -> Result<eredu_checkpoint::recipe::DerivedWeightRecipe, &'static str> {
                            panic!("F32 destination fixture must not invoke a transformed producer")
                        })
                };
                let bindings = bind(projection.excluded_parameters()).unwrap();
                assert!(
                    bindings
                        .iter()
                        .all(|binding| !projection.excluded_parameters().contains(binding.name()))
                );
                // An excluded actual module slot still requires the retained
                // ownership declaration; missing source rows do not authorize it.
                if let Some(name) = targets
                    .keys()
                    .find(|name| projection.excluded_parameters().contains(*name))
                {
                    let mut incomplete = projection.excluded_parameters().clone();
                    assert!(incomplete.remove(name));
                    assert!(bind(&incomplete).is_err());
                    excluded_required += 1;
                }
                completed += 1;
            }
        }
    }
    assert!(completed > 0);
    assert!(excluded_required > 0);
}

#[test]
fn routed_disk_partition_destinations_bind_exact_selected_exclusions() {
    let mut config = routed_config();
    config["num_key_value_heads"] = 2.into();
    check(config);
}

#[test]
fn composite_disk_partition_destinations_bind_exact_selected_exclusions() {
    check(composite_config());
}
