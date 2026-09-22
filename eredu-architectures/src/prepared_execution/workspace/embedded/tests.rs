use super::super::prediction::tests::{
    parallel_request, retained_communication, selected_state_layout,
};
use super::*;
use crate::preparation_selection::{
    select_preparation,
    tests::{BoundedIndependentAdapter, inspected_config, prediction_config},
};
use eredu_core::{
    generation::SpeculativeSchedulerOptions, speculative::PredictionPrefillAlignment,
};
use eredu_nn::workspace::*;
use eredu_runtime::{
    ArchitectureStateFactory, NormalizedLoadRequest,
    prefill::PrefillControlPlan,
    speculative::embedded_occurrence::{EmbeddedPredictionShape, EmbeddedSchedulePlan},
};
use std::num::{NonZeroU32, NonZeroUsize};

const TENSOR_FACT: &str = "fixture allocations with five temporary bytes";
const HOST_FACT: &str = "fixture three host bytes";
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 5,
            assumptions: TENSOR_FACT.into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 3,
            assumptions: HOST_FACT.into(),
        }))
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(Some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: operation.outputs.len(),
                aliases: 0,
                assumption_bytes: TENSOR_FACT.len(),
            },
            scratch_bytes: 5,
        }))
    }
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        let facts = self.operation_facts(operation)?.unwrap();
        destination.validate(facts.layout).unwrap();
        destination
            .assumptions
            .copy_from_slice(TENSOR_FACT.as_bytes());
        for (slot, layout) in destination.outputs.iter_mut().zip(operation.outputs.iter()) {
            *slot = WorkspaceOutputEffect::Allocate(layout.bytes().unwrap());
        }
        Ok(Some(facts))
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(Some(WorkspaceHostFacts {
            bytes: 3,
            assumption_bytes: HOST_FACT.len(),
        }))
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        let facts = self.host_facts(operation)?.unwrap();
        destination.validate(facts).unwrap();
        destination
            .assumptions
            .copy_from_slice(HOST_FACT.as_bytes());
        Ok(Some(facts))
    }
}
fn funded_context() -> WorkspaceContext {
    let capacity = 1 << 24;
    let pool = crate::memory_fixture::ledger(capacity, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &eredu_runtime::working_memory::InferenceExecutionIdentity::default(),
            crate::memory_fixture::limits(&pool, capacity),
        )
        .unwrap();
    WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap()
}
struct TargetTrace {
    sequence: i32,
    score_positions: Option<i32>,
    calls: usize,
    collectives: usize,
}
impl InferenceEquationTraceObserver for TargetTrace {
    fn observe(
        &mut self,
        _: &InferenceWorkspaceSpan,
        _: &WorkspaceTraceReport,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<(), Error> {
        panic!("embedded target must lend its actual hidden root separately")
    }
    fn observe_target_with_storage(
        &mut self,
        _: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        _: usize,
        output_roots: usize,
        _: usize,
        output: Option<WorkspaceStoragePopulation>,
        capture: &WorkspaceTensor,
        scores: Option<&WorkspaceTensor>,
    ) -> Result<(), Error> {
        assert_eq!(capture.shape(), [1, self.sequence, 16]);
        assert_eq!(scores.is_some(), self.score_positions.is_some());
        if let Some(scores) = scores {
            assert_eq!(scores.shape(), [1, self.score_positions.unwrap(), 32]);
        }
        assert_eq!(
            output_roots,
            1 + usize::from(self.score_positions.is_some())
        );
        assert!(output.is_some());
        assert!(!report.operations.is_empty());
        self.collectives += report
            .operations
            .iter()
            .filter(|row| matches!(row.kind, WorkspaceOperationKind::Collective(_)))
            .count();
        self.calls += 1;
        Ok(())
    }
}

#[test]
fn selected_embedded_target_keeps_hidden_without_scores_and_does_not_materialize_prediction() {
    for rank in [None, Some(0), Some(1)] {
        let (_directory, inspection) = inspected_config(prediction_config());
        let request = rank.map(parallel_request).unwrap_or_default();
        let selected =
            select_preparation(&inspection, &request, &BoundedIndependentAdapter::default())
                .unwrap();
        let communication = rank.map(|rank| {
            let manifests = (0..2)
                .map(|peer| {
                    select_preparation(
                        &inspection,
                        &parallel_request(peer),
                        &BoundedIndependentAdapter::default(),
                    )
                    .unwrap()
                    .communication_manifest()
                    .unwrap()
                    .clone()
                })
                .collect::<Vec<_>>();
            retained_communication(rank, &manifests)
        });
        let plan = eredu_core::plan_model_preparation(
            inspection,
            request.preparation_policy().unwrap(),
            selected.session_capabilities(),
        )
        .unwrap();
        let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
        let blueprint = PreparedInferenceBlueprint::new(sources.clone());
        let local_layout = selected_state_layout(blueprint.selected());
        if rank.is_some() {
            let source_context = WorkspaceContext::new(Facts);
            crate::prepared_execution::workspace::destinations::project_partitioned_text_binding_destinations_for(&sources,&source_context,ConstructionPurpose::TargetEquations).unwrap();
        }
        assert!(sources.graph().prediction_placement.get().is_none());
        for (chunk, score_positions) in [(2, None), (3, Some(1)), (2, Some(2))] {
            let context = if rank.is_some() {
                funded_context()
            } else {
                WorkspaceContext::new(Facts)
            };
            let mut factory = WorkspaceResidentStateFactory::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU32::new(4).unwrap(),
                &context,
            )
            .unwrap();
            let state = factory.realize(&local_layout).unwrap();
            let geometry = InferenceGeometry {
                batch_size: 1,
                input_positions: 3,
                cached_positions: 0,
                max_output_tokens: 2,
                prefill_chunk_positions: chunk,
                output: eredu_core::OutputDemand::LastPosition,
            };
            let config = eredu_core::generation::SpeculativeConfig {
                max_tokens: 2,
                max_draft_tokens: 1,
                ..Default::default()
            };
            let schedule = EmbeddedSchedulePlan::new(
                blueprint.selected().prediction_realization().unwrap(),
                EmbeddedPredictionShape::Sequential {
                    depth: NonZeroUsize::new(1).unwrap(),
                },
                PredictionPrefillAlignment::NextToken,
                0,
                PrefillControlPlan::new(geometry, true).unwrap(),
                128,
                &config,
                SpeculativeSchedulerOptions::default(),
            )
            .unwrap();
            let (target, _) = schedule.prefill_invocations(0).unwrap();
            let output = match score_positions {
                None => eredu_core::OutputDemand::StateOnly,
                Some(1) => eredu_core::OutputDemand::LastPosition,
                Some(_) => eredu_core::OutputDemand::Sequence,
            };
            let workspace =
                EmbeddedInvocationWorkspace::target_with_readout(target, output).unwrap();
            let mut trace = TargetTrace {
                sequence: chunk as i32,
                score_positions,
                calls: 0,
                collectives: 0,
            };
            blueprint
                .quote_embedded_target_invocation(
                    workspace,
                    WorkspaceDtype::Uint32,
                    None,
                    &state,
                    &context,
                    None,
                    communication.as_ref(),
                    None,
                    &mut trace,
                )
                .unwrap_or_else(|cause| panic!("target rank {rank:?} chunk {chunk}: {cause:?}"));
            assert_eq!(trace.calls, 1);
            if rank.is_some() {
                assert!(
                    trace.collectives > 0,
                    "actual TP target forward must retain its collectives"
                );
            }
            validate_frontier(&state, 0).unwrap();
            assert!(sources.graph().prediction_placement.get().is_none());
            assert!(blueprint.has_selected_sources(&sources));
        }

        if rank.is_some() {
            continue;
        }
        // An ordinary warm-up on this same loaded selection executes only target
        // equations. It must not attempt the unsupported prediction materializer
        // in this route or publish an extension placement as a quote side effect.
        let context = if rank.is_some() {
            funded_context()
        } else {
            WorkspaceContext::new(Facts)
        };
        let mut factory = WorkspaceResidentStateFactory::new(
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(4).unwrap(),
            &context,
        )
        .unwrap();
        let state = factory.realize(&local_layout).unwrap();
        let geometry = InferenceGeometry {
            batch_size: 1,
            input_positions: 3,
            cached_positions: 0,
            max_output_tokens: 2,
            prefill_chunk_positions: 2,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let report = blueprint
            .quote_replicated_resident_text(geometry, &state, &context)
            .unwrap();
        assert_eq!(report.geometry(), geometry);
        assert!(report.completed_spans() > 1);
        assert!(report.tensor_transient_peak_bytes().unwrap() > 0);
        validate_frontier(&state, 0).unwrap();
        assert!(sources.graph().prediction_placement.get().is_none());
        assert!(blueprint.has_selected_sources(&sources));
    }
}
