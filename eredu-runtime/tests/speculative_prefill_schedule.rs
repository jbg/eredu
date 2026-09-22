//! The same completion-gated scheduler retains real speculative host custody.
use eredu_core::{
    generation::{SpeculativeConfig, SpeculativeSchedulerOptions},
    Completion, DomainMemoryRequirements, GenerationCancellationToken, InferenceGeometry,
    MemoryDomainDescription, MemoryLimit, MemoryLimits, MemoryLocation, MemoryTopology,
    OutputDemand, Submission,
};
use eredu_runtime::{prefill::*, speculative::external_occurrence::*, working_memory::*, *};
use std::{cell::Cell, convert::Infallible, num::NonZeroUsize, rc::Rc, sync::Arc};
fn selected() -> SelectedSpeculativeRealization {
    let id = |s: &str| SpeculativeIdentity::new(s).unwrap();
    let target = id("target");
    let strategy = id("assistant");
    let capture = SpeculativeCaptureSchema::new(
        id("capture"),
        [
            SpeculativeCaptureEntry::new(id("hidden"), vec![1, 2, 8], id("rank"), id("seam"))
                .unwrap(),
        ],
    )
    .unwrap();
    let state = SpeculativeStateCacheIdentityIngredients::new(
        target.clone(),
        strategy.clone(),
        Some(id("assistant-source")),
        Some([7; 32]),
        id("artifact"),
        id("format"),
        id("placement"),
        0,
        id("text"),
        vec![id("target-state"), id("assistant-state")],
    )
    .unwrap();
    let requirements = SpeculativeRealizationRequirements::new(
        target.clone(),
        SpeculativeStrategyRequirements::external(
            strategy.clone(),
            NonZeroUsize::new(2).unwrap(),
            [7; 32],
        ),
        capture.clone(),
        SpeculativeMechanismRequirements::new([]),
        state,
    )
    .unwrap();
    let request =
        SpeculativeSelectionRequest::new(SpeculativePlacementRequest::Single, capture.clone())
            .with_architecture_proof(SpeculativeArchitectureCompatibilityProof::new(
                target,
                strategy,
                capture.identity().clone(),
            ))
            .with_tokenizer_proof(
                eredu_core::TokenizerCompatibilityProof::prove([7; 32], [7; 32]).unwrap(),
            );
    select_speculative_realization(
        &requirements,
        &request,
        &SpeculativeMechanismCapabilities::new(
            requirements.mechanisms().mechanisms().iter().copied(),
        ),
    )
    .unwrap()
}
fn schedule(
    selected: &SelectedSpeculativeRealization,
    shape: ExternalPredictionShape,
) -> ExternalSchedulePlan<'_> {
    ExternalSchedulePlan::new(
        selected,
        shape,
        PrefillControlPlan::new(
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 11,
                input_positions: 7,
                max_output_tokens: 5,
                prefill_chunk_positions: 3,
                output: OutputDemand::LastPosition,
            },
            true,
        )
        .unwrap(),
        23,
        &SpeculativeConfig {
            max_tokens: 5,
            max_draft_tokens: 2,
            ..Default::default()
        },
        SpeculativeSchedulerOptions {
            lookahead_blocks: 1,
            ..Default::default()
        },
    )
    .unwrap()
}

fn ledger(limit: MemoryLimit) -> (MemoryLedger, MemoryLimits) {
    let topology = Arc::new(
        MemoryTopology::new(vec![MemoryDomainDescription {
            name: "host".into(),
            locations: vec![MemoryLocation::Host],
        }])
        .unwrap(),
    );
    let limits = MemoryLimits::resolve(&topology, [(topology.host_domain(), limit)]).unwrap();
    let baseline = DomainMemoryRequirements::zero(&topology);
    (
        MemoryLedger::new(topology, limits.clone(), baseline).unwrap(),
        limits,
    )
}
fn charge(ledger: &MemoryLedger) -> u64 {
    ledger.snapshot().unwrap().domains[0].current_charge_bytes
}
struct Ready {
    complete: Rc<Cell<bool>>,
    _authority: SpeculativePrefillScheduleAuthority,
}
impl Completion for Ready {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(self.complete.get())
    }
    fn wait(&self) -> Result<(), Infallible> {
        self.complete.set(true);
        Ok(())
    }
}
struct Numerical {
    execution: InferenceExecutionIdentity,
    expected: SpeculativePrefillScheduleAuthority,
    complete: Rc<Cell<bool>>,
    chunks: Vec<PrefillChunk>,
    sum: u64,
}
impl PrefillExecutor<SpeculativePrefillScheduleAuthority> for Numerical {
    type Output = u64;
    type Completion = Ready;
    type Error = WorkingMemoryError;
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        authority: SpeculativePrefillScheduleAuthority,
    ) -> Result<Submission<Option<u64>, Ready>, WorkingMemoryError> {
        authority.validate(&self.execution, self.expected.geometry())?;
        self.expected.validate_same_request(&authority)?;
        // Nonzero fixture arithmetic is owned by this neutral executor. The
        // schedule scope grants no backend tensor allocation or native role.
        for input in chunk.input.clone() {
            self.sum += (input + 1) * (input + 2);
        }
        self.chunks.push(chunk.clone());
        Ok(Submission {
            output: (chunk.output != OutputDemand::StateOnly).then_some(self.sum),
            completion: Ready {
                complete: self.complete.clone(),
                _authority: authority,
            },
        })
    }
}
#[test]
fn controlled_and_uninterrupted_share_funded_schedule_and_output() {
    for limit in [MemoryLimit::Finite(1 << 22), MemoryLimit::Unlimited] {
        let (pool, limits) = ledger(limit);
        let baseline = charge(&pool);
        for controlled in [false, true] {
            let source = selected();
            let plan = schedule(&source, ExternalPredictionShape::Sequential);
            let execution = InferenceExecutionIdentity::default();
            let request = OriginalSpeculativeRequest::prepare_external(
                &pool,
                &execution,
                &plan,
                limits.clone(),
            )
            .unwrap();
            let funding = pool
                .prepare_workspace_metadata(&execution, limits.clone())
                .unwrap();
            let authority = request
                .prepare_external_prefill_schedule(&plan, &funding)
                .unwrap();
            let mut executor = Numerical {
                execution: execution.clone(),
                expected: authority.clone(),
                complete: Rc::new(Cell::new(true)),
                chunks: vec![],
                sum: 0,
            };
            let mut driver = PrefillDriver::new_speculative_schedule(
                &execution,
                authority,
                GenerationCancellationToken::new(),
            )
            .unwrap();
            drop(request);
            drop(funding);
            assert!(charge(&pool) > baseline);
            let output = if controlled {
                let mut output = None;
                loop {
                    match driver.step(&mut executor).unwrap() {
                        PrefillProgress::Chunk { output: value, .. } => {
                            if value.is_some() {
                                output = value;
                            }
                        }
                        PrefillProgress::Complete => break,
                        other => panic!("unexpected progress: {other:?}"),
                    }
                }
                output
            } else {
                let (outcome, output) = driver.run_final(&mut executor).unwrap();
                assert_eq!(outcome, PrefillOutcome::Complete);
                output
            };
            assert_eq!(output, Some(168));
            assert_eq!(
                executor
                    .chunks
                    .iter()
                    .map(|c| (c.input.clone(), c.position, c.output))
                    .collect::<Vec<_>>(),
                vec![
                    (0..3, 11, OutputDemand::StateOnly),
                    (3..6, 14, OutputDemand::StateOnly),
                    (6..7, 17, OutputDemand::LastPosition)
                ]
            );
            drop(driver);
            drop(executor);
            assert_eq!(charge(&pool), baseline);
        }
    }
}
#[test]
fn exact_issuer_execution_and_one_start_are_required() {
    let (pool, limits) = ledger(MemoryLimit::Unlimited);
    let source = selected();
    let plan = schedule(&source, ExternalPredictionShape::Sequential);
    let foreign_plan = schedule(&source, ExternalPredictionShape::Sequential);
    let execution = InferenceExecutionIdentity::default();
    let request =
        OriginalSpeculativeRequest::prepare_external(&pool, &execution, &plan, limits.clone())
            .unwrap();
    let funding = pool.prepare_workspace_metadata(&execution, limits).unwrap();
    let before = pool.snapshot().unwrap();
    assert!(request
        .prepare_external_prefill_schedule(&foreign_plan, &funding)
        .is_err());
    assert_eq!(pool.snapshot().unwrap(), before);
    let authority = request
        .prepare_external_prefill_schedule(&plan, &funding)
        .unwrap();
    assert!(PrefillDriver::<u64, Ready, _>::new_speculative_schedule(
        &InferenceExecutionIdentity::default(),
        authority.clone(),
        GenerationCancellationToken::new()
    )
    .is_err());
    let driver = PrefillDriver::<u64, Ready, _>::new_speculative_schedule(
        &execution,
        authority.clone(),
        GenerationCancellationToken::new(),
    )
    .unwrap();
    assert!(PrefillDriver::<u64, Ready, _>::new_speculative_schedule(
        &execution,
        authority,
        GenerationCancellationToken::new()
    )
    .is_err());
    drop(driver);
}
#[test]
fn cancellation_settles_pending_chunk_before_releasing_schedule() {
    let (pool, limits) = ledger(MemoryLimit::Finite(1 << 22));
    let baseline = charge(&pool);
    let source = selected();
    let plan = schedule(&source, ExternalPredictionShape::Sequential);
    let execution = InferenceExecutionIdentity::default();
    let request =
        OriginalSpeculativeRequest::prepare_external(&pool, &execution, &plan, limits.clone())
            .unwrap();
    let funding = pool.prepare_workspace_metadata(&execution, limits).unwrap();
    let authority = request
        .prepare_external_prefill_schedule(&plan, &funding)
        .unwrap();
    let complete = Rc::new(Cell::new(false));
    let mut executor = Numerical {
        execution: execution.clone(),
        expected: authority.clone(),
        complete: complete.clone(),
        chunks: vec![],
        sum: 0,
    };
    let cancellation = GenerationCancellationToken::new();
    let mut driver =
        PrefillDriver::new_speculative_schedule(&execution, authority, cancellation.clone())
            .unwrap();
    drop(request);
    drop(funding);
    assert!(matches!(
        driver.step(&mut executor).unwrap(),
        PrefillProgress::Pending
    ));
    let pending_charge = charge(&pool);
    cancellation.cancel();
    assert!(matches!(
        driver.step(&mut executor).unwrap(),
        PrefillProgress::Pending
    ));
    assert_eq!(charge(&pool), pending_charge);
    assert_eq!(executor.chunks.len(), 1);
    complete.set(true);
    assert!(matches!(
        driver.step(&mut executor).unwrap(),
        PrefillProgress::Cancelled
    ));
    assert_eq!(driver.completed_positions(), 3);
    drop(driver);
    drop(executor);
    assert_eq!(charge(&pool), baseline);
}
