use super::*;
use crate::{TokenFilter, TokenSamplingDecision};

#[test]
fn old_candidate_json_defaults_to_explicitly_unknown_domain() {
    let payload: CapturePayload = serde_json::from_str(
        r#"{
        "kind":"candidates", "value":{
            "stage":"raw_logits_before_sampling",
            "candidates":[{"token_id":7,"score":3.5}]
        }
    }"#,
    )
    .unwrap();
    let CapturePayload::Candidates(candidates) = payload else {
        panic!()
    };
    assert_eq!(candidates.source, CandidateLogitsSource::Original);
    assert_eq!(candidates.domain, None);
    assert!(candidates.candidates[0].allowed);
    let mut candidates = candidates;
    candidates.candidates[0].allowed = false;
    candidates.domain = Some(CandidateDomain {
        allowed_tokens: 3,
        vocabulary: 8,
        constrained: true,
    });
    let json = serde_json::to_string(&candidates).unwrap();
    assert_eq!(
        serde_json::from_str::<CaptureCandidates>(&json).unwrap(),
        candidates
    );
    assert_eq!(CAPTURE_SCHEMA_VERSION, 1);
}

#[test]
fn domain_uses_output_width_and_distinguishes_validity_from_constraints_and_forcing() {
    let validity = TokenFilter::allowed(vec![true, false, true, true]).unwrap();
    let filter = TokenFilter::allowed(vec![true, false, true, false]).unwrap();
    let mut decision =
        TokenSamplingDecision::new(filter.clone()).with_tokenizer_validity(&validity);
    let domain = decision.capture_domain().unwrap();
    assert_eq!(
        domain.summary(6),
        CandidateDomain {
            allowed_tokens: 2,
            vocabulary: 6,
            constrained: true
        }
    );
    // The only semantic exclusion lies beyond this narrower model output.
    assert_eq!(
        domain.summary(3),
        CandidateDomain {
            allowed_tokens: 2,
            vocabulary: 3,
            constrained: false
        }
    );
    let forced = TokenFilter::allowed(vec![true, false, false, false]).unwrap();
    decision.override_filter(forced.clone());
    assert_eq!(decision.filter(), &forced);
    assert_eq!(decision.capture_domain().unwrap().filter, &filter);
    assert!(decision.capture_domain().unwrap().filter.allows(2));
    let text = TokenSamplingDecision::new(validity.clone()).with_tokenizer_validity(&validity);
    assert_eq!(
        text.capture_domain().unwrap().summary(6),
        CandidateDomain {
            allowed_tokens: 3,
            vocabulary: 6,
            constrained: false
        }
    );
    assert!(TokenSamplingDecision::new(filter)
        .capture_domain()
        .is_none());
}
