use super::*;
use crate::prediction_extension::workspace::tests::ProjectedFixtureParameters;
use crate::prediction_extension::workspace::{
    WorkspacePredictionInvocation, WorkspacePredictionInvocations,
};
use crate::preparation_selection::{
    select_preparation,
    tests::{BoundedIndependentAdapter, inspected_config, prediction_config},
};
use eredu_core::{
    generation::SpeculativeSchedulerOptions, speculative::PredictionPrefillAlignment,
};
use eredu_nn::{CompressedAttentionCache, CompressedAttentionState, Index, workspace::*};
use eredu_runtime::{
    ArchitectureStateFactory, NormalizedLoadRequest,
    prefill::PrefillControlPlan,
    speculative::embedded_occurrence::{EmbeddedPredictionShape, EmbeddedSchedulePlan},
    working_memory::WorkspaceCompressedCache,
};
use std::{
    cell::Cell,
    num::{NonZeroU32, NonZeroUsize},
};

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
            scratch_bytes: 0,
            assumptions: "explicit fixture backing".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture has no disjoint native host worker".into(),
        }))
    }
}
struct RecordedParameters;
impl WorkspacePredictionParameterSource for RecordedParameters {
    type Context<'a> = (&'a Cell<usize>, &'a WorkspacePredictionInvocations);
    fn bind<U: eredu_nn::Parameterized<WorkspaceTensor>>(
        source: &mut Self::Context<'_>,
        index: usize,
        declaration: &U,
        local: &mut U,
        tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
        layout: Option<&eredu_runtime::LocalModelLayout>,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        ProjectedFixtureParameters::bind(
            &mut source.0,
            index,
            declaration,
            local,
            tasks,
            layout,
            context,
        )
    }
    fn invocation(
        source: &mut Self::Context<'_>,
        index: usize,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspacePredictionInvocation>, Error> {
        source.1.module(index, context).map(Some)
    }
}

struct Tails {
    tokens: usize,
    rows: usize,
}
impl WorkspacePredictionEquationTails for Tails {
    fn token(&mut self, id: u32, context: &WorkspaceContext) -> Result<WorkspaceTensor, Error> {
        assert_eq!(id, 7);
        self.tokens += 1;
        WorkspaceTensor::initialized(&[1, 1], WorkspaceDtype::Int32, context)
    }
    fn readout(
        &mut self,
        output: &PredictionEquationOutput<WorkspaceTensor>,
        rows: &mut Vec<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        match output {
            PredictionEquationOutput::Sequential { logits, hidden } => {
                assert_eq!(logits.shape(), [1, 1, 32]);
                assert_eq!(hidden.shape(), [1, 1, 16]);
                self.rows += 1;
                context.reserve_metadata_vec(rows, 1)?;
                rows.push(logits.index(&[Index::At(0), Index::At(0), Index::Full], context)?);
                Ok(())
            }
            PredictionEquationOutput::StateOnly => Ok(()),
            PredictionEquationOutput::Fused(_) => panic!("selected fixture is sequential"),
        }
    }
}
struct Observation {
    expected: u64,
    active: bool,
    began: usize,
    ended: usize,
    callbacks: usize,
}
impl eredu_runtime::ActivationObserver<WorkspaceTensor, Error> for Observation {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn observe(&mut self, _: &str, _: &WorkspaceTensor) -> Result<(), Error> {
        assert!(self.active, "callbacks require the real begin_span");
        self.callbacks += 1;
        Ok(())
    }
}
impl InferenceWorkspaceObserver for Observation {
    fn begin_span(
        &mut self,
        _: eredu_core::InferenceGeometry,
        _: &InferenceWorkspaceSpan,
        prediction: u64,
        _: &WorkspaceContext,
    ) -> Result<bool, Error> {
        assert_eq!(prediction, self.expected);
        assert!(!self.active);
        self.active = true;
        self.began += 1;
        Ok(true)
    }
    fn visit_retained(&self, _: &mut dyn FnMut(&WorkspaceTensor)) {}
    fn end_span(&mut self, _: &InferenceWorkspaceSpan, _: &WorkspaceContext) -> Result<(), Error> {
        assert!(self.active);
        self.active = false;
        self.ended += 1;
        Ok(())
    }
}
struct Trace {
    collectives: usize,
    calls: usize,
    outputs: usize,
}
impl InferenceEquationTraceObserver for Trace {
    fn observe(
        &mut self,
        _: &InferenceWorkspaceSpan,
        _: &WorkspaceTraceReport,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<(), Error> {
        panic!("actual prepared source cannot skip an invented input")
    }
    fn observe_prepared_with_storage(
        &mut self,
        _: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained: usize,
        outputs: usize,
        input: Option<usize>,
        population: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        assert_eq!(input, None);
        assert_eq!(outputs, self.outputs);
        assert!(retained > 0);
        assert!(population.is_some());
        assert!(!report.operations.is_empty());
        self.calls += 1;
        self.collectives += report
            .operations
            .iter()
            .filter(|operation| matches!(operation.kind, WorkspaceOperationKind::Collective(_)))
            .count();
        Ok(())
    }
}
fn cache(
    policy: &eredu_core::cache::LayerCachePolicy,
    position: i32,
    context: &WorkspaceContext,
) -> WorkspaceCompressedCache {
    let eredu_core::cache::LayerCachePolicy::CompressedLatentRotary {
        latent_dim,
        rotary_dim,
        ..
    } = policy
    else {
        panic!("fixture source profile")
    };
    let mut cache = WorkspaceCompressedCache::new(
        NonZeroU32::new(1).unwrap(),
        *latent_dim,
        *rotary_dim,
        NonZeroU32::new(4).unwrap(),
        context,
    )
    .unwrap();
    if position > 0 {
        cache
            .append(
                CompressedAttentionState {
                    latent: WorkspaceTensor::initialized(
                        &[1, position, i32::try_from(latent_dim.get()).unwrap()],
                        WorkspaceDtype::Float32,
                        context,
                    )
                    .unwrap(),
                    rotary: WorkspaceTensor::initialized(
                        &[1, position, i32::try_from(rotary_dim.get()).unwrap()],
                        WorkspaceDtype::Float32,
                        context,
                    )
                    .unwrap(),
                },
                context,
            )
            .unwrap();
    }
    cache
}

#[test]
fn retained_prediction_quote_runs_prefill_proposal_and_replay_without_rebinding_placement() {
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
        // The loaded source retains its actual declaration once, before request quotes.
        let source_context = WorkspaceContext::new(Facts);
        let initial = crate::prediction_extension::prepare::<WorkspaceBackend>(
            blueprint.selected().prediction_extension().unwrap(),
            blueprint
                .selected()
                .execution()
                .parallel_topology()
                .unwrap_or(
                    ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0)
                        .unwrap(),
                ),
            blueprint
                .selected()
                .text_realization()
                .auxiliary_materialization_tasks(),
            &source_context,
            &source_context,
        )
        .unwrap();
        let topology = blueprint
            .selected()
            .execution()
            .parallel_topology()
            .unwrap_or(
                ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
            );
        assert!(
            sources
                .graph()
                .prediction_placement
                .set(initial.retained_placement(topology))
                .is_ok()
        );
        let placement = std::sync::Arc::clone(sources.graph().prediction_placement.get().unwrap());
        let target_inspection = sources.execution_inspection();
        let source_alias = sources.clone();
        assert!(std::ptr::eq(
            target_inspection,
            source_alias.execution_inspection()
        ));
        assert!(
            target_inspection
                .admission_token()
                .same_admission(&sources.inspection().admission_token())
        );
        drop(initial);
        drop(source_context);
        let geometry = InferenceGeometry {
            batch_size: 1,
            input_positions: 3,
            cached_positions: 0,
            max_output_tokens: 3,
            prefill_chunk_positions: 2,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let config = eredu_core::generation::SpeculativeConfig {
            max_tokens: 3,
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
        for phase in 0..3 {
            let context = WorkspaceContext::new(Facts);
            let invocation = match phase {
                0 => schedule.prefill_invocations(0).unwrap().1.unwrap(),
                1 => schedule
                    .decode_invocation(SpeculativeActivationPhase::Proposal { depth: 0 }, 3, 1)
                    .unwrap(),
                _ => schedule
                    .decode_invocation(SpeculativeActivationPhase::PredictionReplay, 3, 1)
                    .unwrap(),
            };
            let prediction_position = if phase == 0 { 0 } else { 2 };
            let actual_target_position = if phase == 0 {
                2
            } else if phase == 1 {
                3
            } else {
                4
            };
            let mut factory = WorkspaceResidentStateFactory::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU32::new(4).unwrap(),
                &context,
            )
            .unwrap();
            let mut target = factory.realize(&local_layout).unwrap();
            let policies = &local_layout;
            for (index, layer) in target.as_mut().iter_mut().enumerate() {
                *layer = WorkspaceResidentLayerState::Compressed(cache(
                    policies.layer(index).unwrap(),
                    actual_target_position,
                    &context,
                ));
            }
            let tensor = |positions| {
                WorkspaceTensor::initialized(&[1, positions, 16], WorkspaceDtype::Float32, &context)
                    .unwrap()
            };
            let ids =
                || WorkspaceTensor::initialized(&[1, 1], WorkspaceDtype::Int32, &context).unwrap();
            let equation = match phase {
                0 => PredictionEquation::Prefill {
                    target_capture: tensor(2),
                    hidden: tensor(1),
                    tokens: ids(),
                },
                1 => PredictionEquation::Sequential {
                    hidden: tensor(1),
                    token: 7,
                    depth: 0,
                },
                _ => PredictionEquation::Replay {
                    captures: tensor(1),
                    tokens: ids(),
                },
            };
            let output = if phase == 1 {
                eredu_core::OutputDemand::Sequence
            } else {
                eredu_core::OutputDemand::StateOnly
            };
            let workspace = EmbeddedInvocationWorkspace::prediction(
                invocation,
                u64::try_from(prediction_position).unwrap(),
                output,
            )
            .unwrap();
            let projected = Cell::new(0);
            let bound = Cell::new(0);
            let calls = WorkspacePredictionInvocations::new(1, &context).unwrap();
            let mut tails = Tails { tokens: 0, rows: 0 };
            let mut trace = Trace {
                collectives: 0,
                calls: 0,
                outputs: if phase == 1 { 3 } else { 0 },
            };
            let mut observation = Observation {
                expected: 7 + phase,
                active: false,
                began: 0,
                ended: 0,
                callbacks: 0,
            };
            let observation_prediction = observation.expected;
            if communication.is_some() && phase == 1 {
                let before = context.operation_count();
                let failure = blueprint
                    .quote_embedded_prediction_invocation::<RecordedParameters, _>(
                        workspace,
                        equation.clone(),
                        &target,
                        &context,
                        None,
                        None,
                        (&bound, &calls),
                        |_, _| -> Result<WorkspacePredictionState, Error> {
                            panic!("missing communication must reject before projection")
                        },
                        None,
                        &mut tails,
                        &mut trace,
                    )
                    .unwrap_err();
                assert!(failure.to_string().contains("communication"));
                assert_eq!(context.operation_count(), before);
                assert_eq!(bound.get(), 0);
            }
            let report = blueprint
                .quote_embedded_prediction_invocation::<RecordedParameters, _>(
                    workspace,
                    equation,
                    &target,
                    &context,
                    None,
                    communication.as_ref(),
                    (&bound, &calls),
                    |layout, context| {
                        projected.set(projected.get() + 1);
                        let PredictionStateSourceLayout::Sequential(rows) = layout else {
                            panic!("selected profile")
                        };
                        assert_eq!(rows.len(), 1);
                        let mut values = context.metadata_vec(rows.len())?;
                        for (_, policy) in rows {
                            values.push(cache(policy, prediction_position, context));
                        }
                        Ok(WorkspacePredictionState::Sequential(values))
                    },
                    Some(EmbeddedPredictionWorkspaceObservation::new(
                        &mut observation,
                        observation_prediction,
                    )),
                    &mut tails,
                    &mut trace,
                )
                .unwrap_or_else(|cause| {
                    panic!("prediction rank {rank:?} phase {phase}: {cause:?}")
                });
            assert_eq!(observation.began, 1);
            assert_eq!(observation.ended, 1);
            assert!(!observation.active);
            assert_eq!(report.completed_spans(), 1);
            assert_eq!(report.geometry(), workspace.geometry());
            assert_eq!(projected.get(), 1);
            assert_eq!(bound.get(), 1);
            assert_eq!(trace.calls, 1);
            calls
                .with_completed(|calls| {
                    assert_eq!(
                        calls.len(),
                        1,
                        "one selected physical module per actual V3 equation"
                    );
                    assert_eq!(calls[0].module, 0);
                    assert_eq!(calls[0].completion, Some(0));
                    assert!(calls[0].retained_roots.is_some_and(|roots| roots > 0));
                    Ok(())
                })
                .unwrap();
            assert_eq!(tails.tokens, usize::from(phase == 1));
            assert_eq!(tails.rows, tails.tokens);
            validate_frontier(&target, u64::try_from(actual_target_position).unwrap()).unwrap();
            assert!(std::sync::Arc::ptr_eq(
                &placement,
                sources.graph().prediction_placement.get().unwrap()
            ));
            assert!(blueprint.has_selected_sources(&sources));
        }
    }
}

pub(in crate::prepared_execution::workspace) fn parallel_request(
    rank: usize,
) -> NormalizedLoadRequest {
    let rank = ParallelRankTopology::new(ParallelTopology::new(1, 2, 1, 1).unwrap(), rank).unwrap();
    NormalizedLoadRequest::default()
        .with_parallel_execution(
            eredu_runtime::ParallelLoadRequest::new(
                rank,
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
            .unwrap(),
        )
        .unwrap()
}
pub(in crate::prepared_execution::workspace) fn retained_communication(
    rank: usize,
    manifests: &[eredu_runtime::CommunicationManifest],
) -> eredu_runtime::RetainedCommunicationSource {
    use crate::preparation_selection::PreparationMechanismProvider;
    struct Transport {
        rank: usize,
        payloads: Vec<Vec<u8>>,
        step: Cell<usize>,
    }
    impl eredu_core::consensus::ConsensusTransport for Transport {
        type Error = std::convert::Infallible;
        fn participant_count(&self) -> usize {
            self.payloads.len()
        }
        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
            let step = self.step.replace(self.step.get() + 1);
            Ok(match step {
                0 => {
                    assert_eq!(
                        local,
                        [u32::try_from(self.payloads[self.rank].len()).unwrap(), 0]
                    );
                    self.payloads
                        .iter()
                        .flat_map(|bytes| [u32::try_from(bytes.len()).unwrap(), 0])
                        .collect()
                }
                1 => self
                    .payloads
                    .iter()
                    .flat_map(|bytes| {
                        let mut words = vec![0; local.len()];
                        for (i, &byte) in bytes.iter().enumerate() {
                            words[i / 4] |= u32::from(byte) << ((i % 4) * 8);
                        }
                        words
                    })
                    .collect(),
                _ => panic!("only source setup gathers"),
            })
        }
    }
    let transport = Transport {
        rank,
        step: Cell::new(0),
        payloads: manifests
            .iter()
            .enumerate()
            .map(|(peer, manifest)| {
                let nonce = [u8::try_from(peer + 1).unwrap(); 32];
                serde_json::to_vec(
                    &serde_json::json!({"schema_version":1,"nonce":nonce,"manifest":manifest}),
                )
                .unwrap()
            })
            .collect(),
    };
    let agreed = eredu_runtime::establish_communication_session(
        &transport,
        &manifests[rank],
        Some([u8::try_from(rank + 1).unwrap(); 32]),
    )
    .unwrap();
    assert_eq!(transport.step.get(), 2);
    eredu_runtime::prepare_communication_realization(
        &manifests[rank],
        agreed.manifests(),
        &BoundedIndependentAdapter::default().communication_capabilities(),
        eredu_runtime::CommunicationTopologyCapabilities::RingWithWorldWaves,
    )
    .unwrap()
    .into_retained_source(agreed.identity())
    .unwrap()
}

pub(in crate::prepared_execution::workspace) fn selected_state_layout(
    selected: &crate::SelectedPreparation,
) -> eredu_runtime::StateLayout {
    struct Layout;
    impl<'a> crate::SelectedExecutionBorrowedDispatcher<'a> for Layout {
        type Output = eredu_runtime::StateLayout;
        type Error = std::convert::Infallible;
        fn replicated(
            self,
            source: &'a eredu_runtime::SelectedReplicatedTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            Ok(source.state().layout().clone())
        }
        fn routed(
            self,
            source: &'a crate::SelectedRoutedTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            Ok(source.text().state().layout().clone())
        }
        fn composite(
            self,
            source: &'a crate::replicated_text::SelectedCompositeTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            Ok(source.execution().state().layout().clone())
        }
        fn partitioned_dense(
            self,
            source: &'a crate::SelectedDensePartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            Ok(source.requirements().state().unwrap().layout().clone())
        }
        fn partitioned_routed(
            self,
            source: &'a crate::SelectedRoutedPartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            Ok(source.requirements().state().unwrap().layout().clone())
        }
        fn partitioned_composite(
            self,
            source: &'a crate::SelectedCompositePartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            Ok(source.requirements().state().unwrap().layout().clone())
        }
    }
    selected.execution().dispatch_ref(Layout).unwrap()
}
