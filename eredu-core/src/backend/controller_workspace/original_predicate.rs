use super::*;
use crate::{OriginalSourceWitness, TokenSamplingDecision};
#[test]
fn original_tag_cannot_reduce_legacy_capacity_validation_and_owned_masks_stay_separate() {
    let mask = TokenFilter::Allowed(vec![true; 7]);
    let witness = 5u8; // A caller tag is explicitly not runtime source evidence.
    let contract = TextControllerContract::from_workspace(
        crate::TextControllerWorkspace {
            filter: (&mask).into(),
            additional_host_bytes: 0,
        },
        7,
    )
    .unwrap();
    let mut decision = TokenSamplingDecision::new(mask.clone())
        .with_original_tokenizer_validity(&mask, OriginalSourceWitness::new(&witness));
    assert!(matches!(
        contract.validate_decision(&decision),
        Err(TextControllerContractError::AdditionalPayloadExceeded {
            required_bytes: 7,
            bound_bytes: 0
        })
    ));
    // This is only the owned-payload predicate. Runtime must first reject the tag.
    contract.validate_owned_decision_payload(&decision).unwrap();
    decision.override_filter(mask.clone());
    assert!(matches!(
        contract.validate_owned_decision_payload(&decision),
        Err(TextControllerContractError::AdditionalPayloadExceeded {
            required_bytes: 7,
            bound_bytes: 0
        })
    ));
    let contract = TextControllerContract::from_workspace(
        crate::TextControllerWorkspace {
            filter: (&mask).into(),
            additional_host_bytes: 7,
        },
        7,
    )
    .unwrap();
    contract.validate_owned_decision_payload(&decision).unwrap();
    decision.override_filter(TokenFilter::Allowed(vec![true; 8]));
    assert!(matches!(
        contract.validate_owned_decision_payload(&decision),
        Err(TextControllerContractError::FilterCapacityExceeded { .. })
    ));
}
