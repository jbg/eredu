//! Test instrumentation around the existing SessionPrefill/PrefillDriver pair.
//! This is ordinary, unbudgeted numerical conformance, not managed admission.
use super::*;
use eredu_architectures::composite_execution::{
    CompositeArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use eredu_runtime::{
    prefill::{PrefillChunk, PrefillDriver, PrefillOutcome, PrefillProgress},
    replicated_session::{PreparedPrefillSource, SessionPrefill},
};
use std::rc::Rc;

type Projections = Vec<(String, Vec<i32>)>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BoundaryTrace {
    pub prepared: Vec<PrefillChunk>,
    pub announced: Vec<PrefillChunk>,
    pub delivered: Vec<PrefillChunk>,
    pub projections_after_delivery: Vec<Projections>,
    pub transactions_prepared: usize,
    pub transactions_coordinated: usize,
    pub transactions_completed: usize,
    pub transactions_finished: Vec<bool>,
    pub observations: Vec<(String, Vec<i32>)>,
}

pub(crate) struct BoundarySchedule {
    pub chunk: u64,
    pub stepped: bool,
    pub cancel_after_first: bool,
    pub max_output_tokens: u64,
    pub transactional: bool,
}

pub(crate) struct BoundaryReport {
    pub outcome: PrefillOutcome,
    pub output: Option<NumericTensor>,
    pub trace: BoundaryTrace,
    pub requested_locally: bool,
    pub terminal_cancelled: bool,
}

struct RecordingSource<P> {
    source: P,
    trace: Rc<RefCell<BoundaryTrace>>,
}
impl<A, P> PreparedPrefillSource<PreparedCompositeArchitecture<A>, NumericBackend, ReadoutState>
    for RecordingSource<P>
where
    A: CompositeArchitecture<NumericBackend, ReadoutState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    P: PreparedPrefillSource<PreparedCompositeArchitecture<A>, NumericBackend, ReadoutState>,
{
    type Chunk = P::Chunk;
    fn geometry(&self) -> eredu_core::InferenceGeometry {
        self.source.geometry()
    }
    fn prepare_chunk(
        &self,
        chunk: &PrefillChunk,
        context: &NumericContext,
    ) -> Result<Self::Chunk, Error> {
        self.trace.borrow_mut().prepared.push(chunk.clone());
        self.source.prepare_chunk(chunk, context)
    }
    fn input<'a>(
        &'a self,
        chunk: &'a Self::Chunk,
    ) -> PreparedCompositeInput<'a, NumericTensor, A::InputPartPlan> {
        self.source.input(chunk)
    }
    fn shared_cache_identity(&self) -> Option<eredu_runtime::SharedPreparedInputCacheIdentity> {
        self.source.shared_cache_identity()
    }
}

struct BoundaryObserver {
    trace: Rc<RefCell<BoundaryTrace>>,
    transactional: bool,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for BoundaryObserver {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn transactional(&self) -> bool {
        self.transactional
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        self.trace.borrow_mut().announced.push(chunk.clone());
        Ok(())
    }
    fn prepare_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.trace.borrow_mut().transactions_prepared += 1;
        Ok(())
    }
    fn coordinate_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), Error> {
        self.trace.borrow_mut().transactions_coordinated += 1;
        Ok(())
    }
    fn complete_transaction(&mut self, _: eredu_core::DistributedCommitEpoch) -> Result<(), Error> {
        self.trace.borrow_mut().transactions_completed += 1;
        Ok(())
    }
    fn finish_transaction(&mut self, _: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.trace
            .borrow_mut()
            .transactions_finished
            .push(committed);
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.trace
            .borrow_mut()
            .observations
            .push((path.to_owned(), value.shape.clone()));
        Ok(())
    }
}

/// Explicit advancement for scheduled composite text, shared with
/// run/cancellation instrumentation. Both forms invoke the production scheduler.
/// The observer accepts only the readout rows actually requested by that driver.
pub(crate) fn scheduled<A, D>(
    session: &mut eredu_runtime::ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        NumericReplicatedMechanisms,
        D,
    >,
    admission: &A::AdmissionConfig,
    input: &eredu_runtime::PreparedModelInput<NumericTensor>,
    context: &NumericContext,
    schedule: BoundarySchedule,
) -> Result<BoundaryReport, String>
where
    A: CompositeArchitecture<NumericBackend, ReadoutState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        ReadoutState,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    let BoundarySchedule {
        chunk,
        stepped,
        cancel_after_first,
        max_output_tokens,
        transactional,
    } = schedule;
    let tokens: Vec<i32> = input
        .parts()
        .iter()
        .flat_map(|part| {
            assert_eq!(part.modality(), eredu_core::InputModality::Text);
            let eredu_runtime::PreparedInputPayload::TokenIds(tokens) = part.payload() else {
                panic!("text fixture requires token IDs");
            };
            tokens.data.iter().map(|value| *value as i32)
        })
        .collect();
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: tokens.len() as u64,
        max_output_tokens,
        prefill_chunk_positions: chunk.min(tokens.len() as u64),
        output: OutputDemand::LastPosition,
    };
    let source = eredu_architectures::prefill::PreparedCompositeTextPrefill::from_token_ids(
        Arc::from(tokens),
        geometry,
        admission.clone(),
        NumericInputInspector,
    )
    .map_err(|error| error.to_string())?;
    let trace = Rc::new(RefCell::new(BoundaryTrace::default()));
    let source = RecordingSource {
        source,
        trace: Rc::clone(&trace),
    };
    let execution = session.inference_execution_identity().clone();
    let request = eredu_runtime::working_memory::InferenceRequest::without_memory_budget(
        &execution, geometry,
    )
    .map_err(|error| error.to_string())?;
    let mut observer = BoundaryObserver {
        trace: Rc::clone(&trace),
        transactional,
    };
    let mut executor = SessionPrefill::new(session, source, &request, context, &mut observer)
        .map_err(|error| error.to_string())?;
    let cancellation = eredu_core::GenerationCancellationToken::new();
    let mut driver = PrefillDriver::new(&execution, request, geometry, cancellation.clone())
        .map_err(|error| error.to_string())?;
    let mut final_output = None;
    let mut requested_locally = false;
    let mut consume = |span: PrefillChunk, output| {
        if span.input.end == geometry.input_positions {
            assert!(final_output.is_none());
            final_output = output;
        } else {
            assert!(output.is_none(), "intermediate text spans are state-only");
        }
        let mut trace = trace.borrow_mut();
        trace.delivered.push(span);
        trace
            .projections_after_delivery
            .push(context.projections.lock().unwrap().clone());
        // Only the selected rank requests cancellation, after the first Chunk
        // has passed completion and post-chunk agreement and reached its caller.
        if cancel_after_first && trace.delivered.len() == 1 {
            requested_locally = true;
            cancellation.cancel();
        }
    };
    let outcome = if stepped {
        loop {
            match driver
                .step(&mut executor)
                .map_err(|error| error.to_string())?
            {
                PrefillProgress::Chunk { chunk, output } => consume(chunk, output),
                PrefillProgress::Complete => break PrefillOutcome::Complete,
                PrefillProgress::Cancelled => break PrefillOutcome::Cancelled,
                PrefillProgress::Pending => panic!("eager numeric completion unexpectedly pending"),
            }
        }
    } else {
        driver
            .run(&mut executor, &mut consume)
            .map_err(|error| error.to_string())?
    };
    drop(consume);
    // Terminal advances must not enter another source preparation, model
    // transaction, projection, or collective, even on a locally uncancelled rank.
    let terminal_trace = trace.borrow().clone();
    let terminal_projections = context.projections.lock().unwrap().clone();
    let terminal_mechanisms = context.mechanism_trace();
    for _ in 0..2 {
        let terminal = driver
            .step(&mut executor)
            .map_err(|error| error.to_string())?;
        assert!(matches!(
            (outcome, terminal),
            (PrefillOutcome::Complete, PrefillProgress::Complete)
                | (PrefillOutcome::Cancelled, PrefillProgress::Cancelled)
        ));
        assert_eq!(*trace.borrow(), terminal_trace);
        assert_eq!(*context.projections.lock().unwrap(), terminal_projections);
        assert_eq!(context.mechanism_trace(), terminal_mechanisms);
    }
    Ok(BoundaryReport {
        outcome,
        output: final_output,
        trace: terminal_trace,
        requested_locally,
        terminal_cancelled: cancellation.is_cancelled(),
    })
}
