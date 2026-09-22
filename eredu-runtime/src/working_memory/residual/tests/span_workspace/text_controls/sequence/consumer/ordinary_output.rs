use super::*;
use eredu_core::{GenerationOutput, GenerationTiming};

fn ordinary_layout() -> GenerationSequenceConsumerLayout {
    // Actual neutral harness specialization, not a facade type stand-in.
    GenerationSequenceConsumerLayout::for_driver_with_ordinary_output::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap()
}

#[test]
fn ordinary_result_original_exact_and_minus_one_use_real_admission_before_any_work() {
    let consumer = ordinary_layout();
    assert_ne!(consumer, layout());
    for route in 0..3 {
        for capture in [false, true] {
            let options = || {
                capture.then(|| TextPreparationOptions {
                    interventions: None,
                    capture: Some(capture_source()),
                })
            };
            let (mut actual, state) = runtime(Mode {
                capture,
                ..Mode::default()
            });
            let sequence = with_consumer(&mut actual, route, options(), &consumer).unwrap();
            assert_eq!(sequence.consumer_layout(), Some(&consumer));
            let accepted_r = state.borrow().r;
            assert_eq!(
                state.borrow().order,
                ["admit", "bind", "extract", "prompt", "sampling"]
            );
            let (pool, held) = retire_request(&state);
            drop(actual);
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            drop(sequence);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);

            let (mut baseline, base) = runtime(Mode {
                capture,
                ..Mode::default()
            });
            let sequence = with_consumer(&mut baseline, route, options(), &layout()).unwrap();
            assert_eq!(
                accepted_r - base.borrow().r,
                (consumer.retention_peak_bytes() - layout().retention_peak_bytes()) as u64
            );
            drop(sequence);
            retire_request(&base);
            drop(baseline);

            let (mut short, rejected) = runtime(Mode {
                capture,
                short: true,
                ..Mode::default()
            });
            let error = with_consumer(&mut short, route, options(), &consumer).unwrap_err();
            let mut cause: &(dyn std::error::Error + 'static) = &error;
            let mut budget = false;
            loop {
                if let Some((required_bytes, available_bytes)) = cause
                    .downcast_ref::<WorkingMemoryError>()
                    .and_then(capacity_numbers)
                {
                    assert_eq!(required_bytes, available_bytes + 1);
                    budget = true;
                }
                match cause.source() {
                    Some(next) => cause = next,
                    None => break,
                }
            }
            assert!(
                budget,
                "must preserve actual width-one budget rejection: {error:?}"
            );
            let facts = rejected.borrow();
            assert_eq!(facts.order, ["admit"]);
            assert_eq!(facts.votes, [(Stage::Admission, Status::Failed)]);
            assert_eq!(facts.pool.payload_used_bytes().unwrap(), 64);
            assert!(facts.active.is_none());
        }
    }
}

#[test]
fn ordinary_result_keeps_original_raw_tail_after_full_source_retirement_and_partial_iteration() {
    let consumer = ordinary_layout();
    let (mut actual, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let mut sequence = with_consumer(&mut actual, 1, None, &consumer)
        .unwrap()
        .prepare_storage()
        .unwrap();
    sequence.commit(4, TokenTerminalSignals::default()).unwrap();
    sequence.commit(5, TokenTerminalSignals::default()).unwrap();
    let pointer = sequence.tokens().as_ptr();
    let reason = sequence.finish_reason().unwrap();
    let (pool, held) = retire_request(&state);
    drop(actual);
    assert_eq!(pool.payload_used_bytes().unwrap(), held + 64);
    let timing = GenerationTiming::new(Some(std::time::Duration::from_micros(7)));
    let output = GenerationOutput::from_retained(sequence.into_token_ids(), reason, timing);
    assert_eq!(output.token_ids(), [4, 5]);
    assert_eq!(output.token_ids().as_ptr(), pointer);
    assert_eq!(output.finish_reason(), eredu_core::FinishReason::MaxTokens);
    assert_eq!(*output.timing(), timing);
    assert_eq!(pool.payload_used_bytes().unwrap(), held);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut iter = output.token_ids.into_iter();
    assert_eq!(iter.next(), Some(4));
    assert_eq!(pool.payload_used_bytes().unwrap(), held);
    drop(iter);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
