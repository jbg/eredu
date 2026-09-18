//! Nonzero selected media execution through the one shared prefill driver.
//! These are ordinary mechanisms; no native/original media admission is implied.
use super::*;
use eredu_architectures::composite_execution::{
    CompositeArchitecture, CompositeMediaIngressArchitecture, PreparedCompositeArchitecture,
    PreparedCompositeInput,
};
use eredu_core::OutputDemand;
use eredu_runtime::{
    media_prefill::MediaTextExecutionStrategy,
    prefill::{PrefillChunk, PrefillDriver, PrefillOutcome, PrefillProgress},
    replicated_session::SessionPrefill,
};
use std::rc::Rc;

type State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Input = eredu_runtime::PreparedModelInput<NumericTensor>;
type Session<A, D> = eredu_runtime::ReplicatedTextSession<
    PreparedCompositeArchitecture<A>,
    NumericBackend,
    NumericReplicatedMechanisms,
    D,
>;
pub(super) type PartitionMedia = dyn FnMut(&Input, Schedule) -> Result<Report, Error>;

#[derive(Clone, Copy)]
pub(super) struct Schedule {
    output: OutputDemand,
    chunk: u64,
    stepped: bool,
    cancel_after: Option<u64>,
    locally_cancel: bool,
    follow_decode: bool,
}
#[derive(Clone, Debug, Default, PartialEq)]
struct Trace {
    announced: Vec<PrefillChunk>,
    delivered: Vec<PrefillChunk>,
    observations: Vec<(String, Vec<i32>)>,
    projections: Vec<Vec<(String, Vec<i32>)>>,
    roots: Vec<Vec<(Vec<i32>, Vec<f32>)>>,
}
pub(super) struct Report {
    input_positions: u64,
    outcome: PrefillOutcome,
    output: Option<NumericTensor>,
    state: State,
    trace: Trace,
    cached: Vec<NumericTensor>,
    final_state: State,
    locally_requested: bool,
}
struct Observer(Rc<RefCell<Trace>>);
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        self.0.borrow_mut().announced.push(chunk.clone());
        Ok(())
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if path.starts_with("visual.layers.") || path == "model.vision.projector.output" {
            assert_finite_values(&value.data, path);
        }
        self.0
            .borrow_mut()
            .observations
            .push((path.into(), value.shape.clone()));
        Ok(())
    }
}

pub(super) fn attach<A, D>(
    session: &Rc<RefCell<Session<A, D>>>,
    admission: &A::AdmissionConfig,
    context: &NumericContext,
) -> Box<PartitionMedia>
where
    A: CompositeMediaIngressArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: Clone + 'static,
    D: MediaTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            NumericBackend,
            State,
            NumericReplicatedPolicy<A::Unit>,
            NumericReplicatedPolicy<A::Unit>,
        > + 'static,
    D::Runtime: 'static,
{
    let session = Rc::clone(session);
    let admission = admission.clone();
    let context = context.clone();
    Box::new(move |input, schedule| {
        scheduled::<A, D>(
            &mut *session.borrow_mut(),
            &admission,
            input,
            &context,
            schedule,
        )
    })
}

fn snapshot<A, D>(session: &Session<A, D>) -> Result<State, Error>
where
    A: CompositeArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    session
        .inspect_runtime_state(|state| Ok(state.clone()))
        .map_err(|e| Error::backend(e.to_string()))
}
fn decode<A, D>(
    session: &mut Session<A, D>,
    admission: &A::AdmissionConfig,
    token: usize,
    context: &NumericContext,
) -> Result<NumericTensor, Error>
where
    A: CompositeArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    let input = numeric_text_prepared_input(&[token]);
    let admitted = A::admit_prepared_input(admission, &input, &NumericInputInspector)
        .map_err(Error::backend_retained_source)?;
    let paired = PreparedCompositeInput::new(&input, &admitted).map_err(Error::backend)?;
    session
        .decode_input(paired, context)
        .map_err(|e| Error::backend(e.to_string()))
}

fn scheduled<A, D>(
    session: &mut Session<A, D>,
    admission: &A::AdmissionConfig,
    input: &Input,
    context: &NumericContext,
    schedule: Schedule,
) -> Result<Report, Error>
where
    A: CompositeMediaIngressArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    D: MediaTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    let shape = A::admit_prepared_input(admission, input, &NumericInputInspector)
        .map_err(Error::backend_retained_source)?
        .decoder_shape();
    let geometry = eredu_core::InferenceGeometry {
        batch_size: shape[0],
        cached_positions: 0,
        input_positions: shape[1],
        max_output_tokens: 3,
        prefill_chunk_positions: schedule.chunk.min(shape[1]),
        output: schedule.output,
    };
    let plan = A::prepare_ingress_plan(admission, input.clone(), &NumericInputInspector, geometry)?;
    let mut source = session
        .prepare_media_prefill_unbudgeted(plan)
        .map_err(|e| Error::backend(e.to_string()))?;
    let request = source.request().expect("ordinary source retains its inference request").clone();
    let execution = session.inference_execution_identity().clone();
    let cancellation = eredu_core::GenerationCancellationToken::new();
    let trace = Rc::new(RefCell::new(Trace::default()));
    let mut observer = Observer(Rc::clone(&trace));
    let mut executor = SessionPrefill::new_media(session, &mut source, context, &mut observer)
        .map_err(|e| Error::backend(e.to_string()))?;
    let mut driver = PrefillDriver::new(&execution, request, geometry, cancellation.clone())
        .map_err(|e| Error::backend(e.to_string()))?;
    let mut output = None;
    let mut sequence_rows = Vec::new();
    let mut locally_requested = false;
    let mut consume = |span: PrefillChunk, value: Option<NumericTensor>| {
        if geometry.output == OutputDemand::Sequence {
            sequence_rows.push(value.expect("every full-sequence span delivers its actual rows"));
        } else if span.input.end == geometry.input_positions {
            output = value;
        } else {
            assert!(value.is_none(), "no intermediate vocabulary output");
        }
        let mut trace = trace.borrow_mut();
        trace.delivered.push(span.clone());
        trace
            .projections
            .push(context.projections.lock().unwrap().clone());
        let roots = context
            .media_completions
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        for (_, values) in &roots {
            assert_finite_values(values, "completed retained media root");
        }
        trace.roots.push(roots);
        if schedule.locally_cancel && schedule.cancel_after == Some(span.input.end) {
            locally_requested = true;
            cancellation.cancel();
        }
    };
    let outcome = if schedule.stepped {
        loop {
            match driver
                .step(&mut executor)
                .map_err(|e| Error::backend(e.to_string()))?
            {
                PrefillProgress::Chunk { chunk, output } => consume(chunk, output),
                PrefillProgress::Complete => break PrefillOutcome::Complete,
                PrefillProgress::Cancelled => break PrefillOutcome::Cancelled,
                PrefillProgress::Pending => panic!("eager fixture unexpectedly pending"),
            }
        }
    } else {
        driver
            .run(&mut executor, &mut consume)
            .map_err(|e| Error::backend(e.to_string()))?
    };
    drop(consume);
    if geometry.output == OutputDemand::Sequence && outcome == PrefillOutcome::Complete {
        output = Some(NumericTensor::concatenate(&sequence_rows, 1, context)?);
    }
    let completed_trace = trace.borrow().clone();
    let projections = context.projections.lock().unwrap().clone();
    let mechanisms = context.mechanism_trace();
    let roots = context.media_completions.lock().unwrap().clone();
    for _ in 0..2 {
        let terminal = driver
            .step(&mut executor)
            .map_err(|e| Error::backend(e.to_string()))?;
        assert!(matches!(
            (outcome, terminal),
            (PrefillOutcome::Complete, PrefillProgress::Complete)
                | (PrefillOutcome::Cancelled, PrefillProgress::Cancelled)
        ));
        assert_eq!(*trace.borrow(), completed_trace);
        assert_eq!(*context.projections.lock().unwrap(), projections);
        assert_eq!(context.mechanism_trace(), mechanisms);
        assert_eq!(*context.media_completions.lock().unwrap(), roots);
    }
    drop(executor);
    let state = snapshot::<A, D>(session)?;
    let mut cached = Vec::new();
    // This entry is explicitly ordinary: unfunded requests do not install an
    // InferenceStateAdmission. Preserve cached decode from the actually committed
    // prefix; funded prompt-end authority remains a separate Unit C obligation.
    if schedule.follow_decode {
        let checkpoint = session
            .checkpoint_complete_distributed(context)
            .map_err(|e| Error::backend(e.to_string()))?;
        for token in [2, 6, 1] {
            cached.push(decode::<A, D>(session, admission, token, context)?);
        }
        let after = snapshot::<A, D>(session)?;
        session
            .rollback_complete_distributed(checkpoint, context)
            .map_err(|e| Error::backend(e.to_string()))?;
        same_state(&snapshot::<A, D>(session)?, &state);
        for (token, expected) in [2, 6, 1].into_iter().zip(&cached) {
            assert_tensor_close(
                &decode::<A, D>(session, admission, token, context)?,
                expected,
                "restored media cached decode",
            );
        }
        same_state(&snapshot::<A, D>(session)?, &after);
    }
    Ok(Report {
        input_positions: geometry.input_positions,
        outcome,
        output,
        state,
        trace: completed_trace,
        cached,
        final_state: snapshot::<A, D>(session)?,
        locally_requested,
    })
}

fn nonzero(value: &NumericTensor) {
    assert!(!value.data.is_empty());
    assert!(value.data.iter().all(|v| v.is_finite()));
    assert!(value.data.iter().any(|v| v.abs() > 1e-9));
}
fn same_state(a: &State, b: &State) {
    assert_eq!(a.layout(), b.layout());
    same_layers(a.as_ref(), b.as_ref());
}
fn same_layers(a: &[NumericHybridLayerState], b: &[NumericHybridLayerState]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        let av = RuntimeLayerState::retained_values(a).collect::<Vec<_>>();
        let bv = RuntimeLayerState::retained_values(b).collect::<Vec<_>>();
        assert_eq!(av.len(), bv.len());
        for (a, b) in av.into_iter().zip(bv) {
            assert_eq!(a.dtype, b.dtype);
            assert_tensor_close(a, b, "all local retained media state");
        }
        assert_eq!(a.attention.is_some(), b.attention.is_some());
        if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.window, b.window);
            assert_eq!(a.attention_history.is_some(), b.attention_history.is_some());
            if let (Some((ak, av)), Some((bk, bv))) = (&a.attention_history, &b.attention_history) {
                assert_tensor_close(ak, bk, "attention-owned media keys");
                assert_tensor_close(av, bv, "attention-owned media values");
            }
        }
    }
}

#[path = "media_prefill/selected.rs"]
mod selected;

#[path = "media_prefill/ordinary.rs"]
mod ordinary;

#[path = "media_prefill/muse.rs"]
mod muse;

#[path = "media_prefill/optional.rs"]
mod optional;
