//! Final-row ownership changes only the score handoff of the existing driver.
use super::*;
use eredu_runtime::replicated_session::{
    OrdinaryPrefillSpan, PrefillScoreLayout, PrefillSpanOperation,
};

struct SelectedPositions;
impl
    PrefillSpanOperation<
        Rows,
        FakeBackend,
        ReferenceTextMechanisms,
        eredu_runtime::DirectReplicatedTextExecution,
        Input,
        SpanObserver,
    > for SelectedPositions
{
    fn score_layout(&self) -> PrefillScoreLayout {
        PrefillScoreLayout::SelectedPositions
    }

    fn execute<'s>(
        &mut self,
        session: &mut Session,
        source: &'s Input,
        prepared: Option<&'s FakeTensor>,
        input: Result<&'s FakeTensor, SessionError>,
        identity: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        chunk: &PrefillChunk,
        context: &(),
        observer: &mut SpanObserver,
    ) -> Result<(Option<FakeTensor>, Option<DistributedCommitOutcome>), SessionError> {
        OrdinaryPrefillSpan.execute(
            session, source, prepared, input, identity, chunk, context, observer,
        )
    }
}

// Reuse the existing terminal/index trace and cancellation observer, allowing
// the real source's multi-position spans instead of the scalar fixture's width1.
struct SpanObserver(Observer);
impl ActivationObserver<FakeTensor, Error> for SpanObserver {
    fn requires_sequence_readout(&self) -> bool {
        // This observer only records lifecycle callbacks and checks nonempty
        // values; it has no row-sensitive capture. The operation independently
        // requests Sequence scores below. Do not claim unauthenticated capture.
        false
    }
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn transactional(&self) -> bool {
        true
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        check_inference_scope();
        self.0.current = chunk.input.start;
        self.0
            .trace
            .borrow_mut()
            .events
            .push(Event::Begin(chunk.input.start));
        Ok(())
    }
    fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Error> {
        self.0.observe(path, value)
    }
    fn finish_transaction(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        self.0.finish_transaction(epoch, committed);
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.0.finish_prefill(committed);
    }
}

fn run<K>(operation: K, preserve: bool, cancel_after: Option<u64>)
where
    K: PrefillSpanOperation<
            Rows,
            FakeBackend,
            ReferenceTextMechanisms,
            eredu_runtime::DirectReplicatedTextExecution,
            Input,
            SpanObserver,
        >,
{
    // This fixture already has two cached nonzero positions (sum17, sentinel29).
    // Install the trace after that setup so only this continuation is counted.
    let mut session = fixture::session();
    let trace = Fixture::new(Fault::None);
    let cancellation = GenerationCancellationToken::new();
    let mut observer = SpanObserver(trace.observer(&cancellation, cancel_after, true));
    let events = Rc::new(RefCell::new(Events::default()));
    let progress = session
        .try_prefill_unbudgeted_source_with_operation(
            Some([1, 5]),
            std::num::NonZeroU64::new(3),
            OutputDemand::Sequence,
            |geometry| {
                assert_eq!(geometry.cached_positions, 2);
                assert_eq!(geometry.output, OutputDemand::Sequence);
                Ok(Some(Input {
                    geometry,
                    events: events.clone(),
                    fail: Fail::None,
                    identity: Arc::new(()),
                }))
            },
            &cancellation,
            &(),
            &mut observer,
            operation,
        )
        .unwrap();

    let first_only = cancel_after == Some(0);
    let completed = if first_only { 3 } else { 5 };
    assert_eq!(progress.completed_positions, completed);
    assert_eq!(
        events.borrow().prepared,
        if first_only {
            vec![0..3]
        } else {
            vec![0..3, 3..5]
        },
    );
    assert_eq!(
        state(&session),
        if first_only {
            (5, vec![31, 29])
        } else {
            (7, vec![55, 29])
        },
        "cancellation preserves the same already committed prefix in both layouts",
    );
    if cancel_after.is_some() {
        assert!(matches!(progress.outcome, PrefillSourceOutcome::Cancelled));
    } else {
        let PrefillSourceOutcome::Complete(Some(scores)) = progress.outcome else {
            panic!("completed score prefill");
        };
        // The final span contains inputs11/13, yielding distinct rows95/121.
        // Preserve both actual selected rows; the default hands off only121.
        assert_eq!(scores.0, if preserve { vec![95, 121] } else { vec![121] });
    }
    let indexed = !preserve && cancel_after.is_none();
    trace.finish(cancel_after.is_none(), indexed);
    let events = &trace.0.borrow().events;
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Index(_)))
            .count(),
        usize::from(indexed),
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::Chunk(true))
            .count(),
        if first_only { 1 } else { 2 },
    );
}

#[test]
fn selected_positions_preserve_final_span_and_default_indexing_and_cancellation_agree() {
    for cancel_after in [None, Some(0), Some(3)] {
        // Exercise the real default implementation, not an explicit FinalScores
        // override, and delegate selected execution to that same span worker.
        run(OrdinaryPrefillSpan, false, cancel_after);
        run(SelectedPositions, true, cancel_after);
    }
}
