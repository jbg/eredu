use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::working_memory::OriginalGenerationDecoderInput;
use eredu_text::{decoder_storage::PreparedDecodeSource, tokenizer::Tokenizer};

pub(super) fn tokenizer() -> Tokenizer {
    let mut vocab = serde_json::Map::new();
    for id in 0..256 {
        vocab.insert(format!("t{id}"), id.into());
    }
    vocab.insert("[UNK]".into(), 256.into());
    let json = serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}});
    Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}
pub(super) fn input(maximum: usize) -> OriginalGenerationDecoderInput {
    OriginalGenerationDecoderInput::new(
        PreparedDecodeSource::prepare(&tokenizer().snapshot()).unwrap(),
        maximum,
        false,
    )
    .unwrap()
}
#[test]
fn original_decoder_uses_actual_native_predictions_residency_and_capture_result_bank() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    super::run_real_prediction(true);
}
#[test]
fn original_native_decoder_exact_minus_one_has_one_precandidate_source_take() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    for (capture, rows) in [(false, false), (true, false), (true, true)] {
        let mut required = None;
        for pass in 0..3 {
            let pool = crate::tests::support::test_utils::initialize_original_sources();
            let (mut runtime, _artifact) = load(&stream, &pool, 0);
            let source = capture.then(|| source(&runtime));
            let probe = Probe::new(&runtime, source.as_ref(), rows);
            let baseline = pool.fixture_host_charge().unwrap();
            let before = paths::session_input_creation_attempts();
            let native = paths::snapshot();
            let decoder = input(4);
            let capacity = required.map_or(u64::MAX, |n| baseline + n - u64::from(pass == 1));
            let result = start_with_decoder(
                &mut runtime,
                4,
                &[],
                2,
                capacity,
                source.as_ref(),
                Some(&decoder),
            );
            assert_eq!(probe.0.decoder_takes.get(), 1);
            if pass == 1 {
                let error = result.unwrap_err();
                assert!(matches!(
                    cause::<WorkingMemoryError>(&error),
                    WorkingMemoryError::Domain(
                        eredu_core::MemoryDomainError::BudgetExceeded { .. }
                    )
                ));
                assert!(probe.0.admitted.borrow().is_none());
                assert_eq!(paths::session_input_creation_attempts(), before);
                assert_eq!(paths::snapshot(), native);
                assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
                // The source is terminally spent even though no original account
                // was accepted. An increased capacity cannot refill this input.
                let error = start_with_decoder(
                    &mut runtime,
                    4,
                    &[],
                    0,
                    u64::MAX,
                    source.as_ref(),
                    Some(&decoder),
                )
                .unwrap_err();
                assert_eq!(cause::<Rejection>(&error), &Rejection::Unavailable);
                assert_eq!(probe.0.decoder_takes.get(), 1);
                assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
            } else {
                let sequence = result.unwrap();
                let facts = probe.facts();
                if pass == 0 {
                    required = Some(facts.required);
                } else {
                    assert_eq!(required, Some(facts.required));
                }
                assert!(sequence.matches_decoder_input(Some(&decoder)));
                drop((sequence, probe.take()));
            }
            drop((probe, source, decoder));
            finish(runtime, &stream);
            settle(&pool, 0);
        }
    }
}
#[test]
fn native_decoder_staging_destroys_source_before_accepted_owners_on_error_and_unwind() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    for mode in [
        Mode::DecoderAcceptedError,
        Mode::DecoderAcceptedPanic,
        Mode::DecoderFundingError,
        Mode::DecoderFundingPanic,
    ] {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let (mut runtime, _artifact) = load(&stream, &pool, 0);
        let baseline = pool.fixture_host_charge().unwrap();
        let probe = Probe::new(&runtime, None, false);
        probe.mode(mode);
        let decoder = input(4);
        let native = paths::snapshot();
        let before = paths::session_input_creation_attempts();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            start_with_decoder(&mut runtime, 4, &[], 0, u64::MAX, None, Some(&decoder))
        }));
        if matches!(mode, Mode::DecoderAcceptedPanic | Mode::DecoderFundingPanic) {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert_eq!(probe.0.decoder_takes.get(), 1);
        assert_eq!(probe.0.decoder_retirements.get(), 1);
        assert!(probe.0.decoder_expected.borrow().as_ref().unwrap().1 > baseline);
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
        assert_eq!(paths::session_input_creation_attempts(), before);
        assert_eq!(paths::snapshot(), native);
        assert!(probe.0.admitted.borrow().is_none());
        drop((probe, decoder));
        finish(runtime, &stream);
        settle(&pool, 0);
    }
}

mod loaded;

mod shared_plain;
