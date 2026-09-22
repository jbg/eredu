use super::*;
use eredu_nn::{workspace::*, Tensor};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, MemoryLedger, SamplingWorkspacePhase, WorkspaceSamplingInput,
};
use std::convert::Infallible;

const TENSOR_FACT: &str = "independent packed output and seven-byte scratch";
const HOST_FACT: &str = "three-byte host scratch";

#[derive(Debug)]
struct Facts {
    known: bool,
    ordinary: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        assert!(
            self.ordinary,
            "checked sampling must not use ordinary facts"
        );
        Ok(self.known.then(|| WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| WorkspaceOutputStorage::Allocate(layout.bytes().unwrap()))
                .collect(),
            scratch_bytes: 7,
            assumptions: TENSOR_FACT.into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        assert!(self.ordinary);
        Ok(self.known.then(|| WorkspaceHostBound {
            bytes: 3,
            assumptions: HOST_FACT.into(),
        }))
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(self.known.then_some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: operation.outputs.len(),
                aliases: 0,
                assumption_bytes: TENSOR_FACT.len(),
            },
            scratch_bytes: 7,
        }))
    }
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
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
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(self.known.then_some(WorkspaceHostFacts {
            bytes: 3,
            assumption_bytes: HOST_FACT.len(),
        }))
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        let facts = self.host_facts(operation)?.unwrap();
        destination.validate(facts).unwrap();
        destination
            .assumptions
            .copy_from_slice(HOST_FACT.as_bytes());
        Ok(Some(facts))
    }
}

#[derive(Default)]
struct Phases(Vec<(SamplingWorkspacePhase, usize, usize)>);
impl SamplingWorkspaceObserver for Phases {
    fn observe(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
    ) -> Result<(), Error> {
        let keys = report
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation.kind,
                    WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::CreateRandomKey)
                )
            })
            .count();
        self.0.push((phase, keys, report.operations.len()));
        Ok(())
    }
}

#[test]
fn configured_sampling_after_finished_equations_prices_fresh_seed_and_keeps_report_closed() {
    for known in [false, true] {
        let capacity = 1 << 24;
        let pool = crate::memory_fixture::ledger(capacity, 0).unwrap();
        let funding = pool
            .prepare_workspace_metadata(
                &InferenceExecutionIdentity::default(),
                crate::memory_fixture::limits(&pool, capacity),
            )
            .unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(
            Facts {
                known,
                ordinary: false,
            },
            funding,
        )
        .unwrap();
        let logits = context.layout(&[1, 37], WorkspaceDtype::Float32).unwrap();
        let mut previous = crate::memory_fixture::used(&pool).unwrap();
        for temperature in [0.0, 0.8] {
            for steps in [0, 2] {
                context.begin_state_span([]).unwrap();
                let marker = WorkspaceTensor::full_i32(7, &[1], &context).unwrap();
                let equations = context
                    .finish_report(std::slice::from_ref(&marker))
                    .unwrap();
                assert_eq!(equations.operations.len(), 1);
                let config = TextGenerationConfig::new(
                    eredu_core::resolve_generation_config(
                        None,
                        eredu_core::GenerationConfigOverrides {
                            temperature: Some(temperature),
                            max_new_tokens: Some(2),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                );
                let mut phases = Phases::default();
                let sampling = TextSamplingInput::Configured(
                    config.clone(),
                    TextFilterWorkspace::Exact(&eredu_core::TokenFilter::All),
                )
                .quote(
                    WorkspaceSamplingInput {
                        layout: &logits,
                        backing_capacity_bytes: Some(148),
                    }
                    .with_backing_population(1),
                    steps,
                    &context,
                    Some(&mut phases),
                )
                .unwrap();
                // Independent entry: ordinary facts and a fresh key in an open
                // context feed the existing runtime sampler directly. Compare the
                // full peaks, not just whether checked construction succeeds.
                let oracle = WorkspaceContext::new(Facts {
                    known,
                    ordinary: true,
                });
                let random = (temperature > 0.0)
                    .then(|| WorkspaceSamplingRandomState::from_seed(&oracle).unwrap());
                let sampler = ConfiguredTextSampler::from_config(config).unwrap();
                let expected = quote_sampling_workspace_with_observer(
                    &sampler,
                    temperature,
                    random.as_ref(),
                    WorkspaceSamplingInput {
                        layout: &logits,
                        backing_capacity_bytes: Some(148),
                    }
                    .with_backing_population(1),
                    &eredu_core::TokenFilter::All,
                    steps,
                    &oracle,
                    None,
                )
                .unwrap();
                assert_eq!(sampling.tensor_peak_bytes, expected.tensor_peak_bytes);
                assert_eq!(sampling.host_peak_bytes, expected.host_peak_bytes);
                assert_eq!(sampling.peak.bytes(), expected.peak.bytes());
                assert_eq!(sampling.final_history_bytes, expected.final_history_bytes);
                if known && temperature > 0.0 && steps == 0 {
                    assert_eq!(
                        sampling.tensor_peak_bytes,
                        Some(8 + 7),
                        "the two-word seed and real scratch stay priced"
                    );
                    assert_eq!(
                        sampling.host_peak_bytes,
                        Some(std::mem::size_of::<ConfiguredTextSampler>() as u64 + 3)
                    );
                }
                assert_eq!(sampling.steps, steps);
                assert_eq!(phases.0.len(), steps as usize + 1);
                assert_eq!(
                    phases.0[0],
                    (
                        SamplingWorkspacePhase::Preparation,
                        usize::from(temperature > 0.0),
                        usize::from(temperature > 0.0)
                    )
                );
                for (index, (phase, keys, operations)) in phases.0[1..].iter().enumerate() {
                    assert_eq!(
                        *phase,
                        SamplingWorkspacePhase::Step {
                            index: index as u64
                        }
                    );
                    assert_eq!(*keys, 0);
                    assert!(*operations > 0);
                }
                if !known && (temperature > 0.0 || steps > 0) {
                    assert!(
                        sampling.peak.bytes().is_none(),
                        "unknown facts cannot become a bound"
                    );
                }
                let error = context.report_scalars(&[]).unwrap_err();
                assert!(matches!(
                    std::error::Error::source(&error)
                        .unwrap()
                        .downcast_ref::<WorkspaceMetadataError>(),
                    Some(WorkspaceMetadataError::ReportFinished)
                ));
                assert_eq!(
                    equations.operations.len(),
                    1,
                    "sampling cannot reopen the finished equation report"
                );
                let used = crate::memory_fixture::used(&pool).unwrap();
                assert!(
                    used > previous,
                    "new spans do not refund prior metadata spending"
                );
                previous = used;
            }
        }
        drop(logits);
        drop(context);
        assert_eq!(crate::memory_fixture::used(&pool).unwrap(), 0);
    }
}
