use super::*;

fn mask(capacity: usize, values: &[bool]) -> TokenFilter {
    let mut mask = Vec::with_capacity(capacity);
    mask.extend_from_slice(values);
    TokenFilter::allowed(mask).unwrap()
}

fn contract(filter: &TokenFilter, additional_host_bytes: u64) -> TextControllerContract {
    TextControllerContract::from_workspace(
        TextControllerWorkspace {
            filter: filter.into(),
            additional_host_bytes,
        },
        4,
    )
    .unwrap()
}

#[test]
fn changing_allowed_values_and_lengths_fit_the_same_cold_contract() {
    let witness = mask(16, &[true, false, false, false]);
    let bound = contract(&witness, 0);
    let capacity = capacity_bytes(&witness).unwrap();
    drop(witness);
    assert_eq!(bound.output_width(), 4);
    assert!(bound.uses_mask());
    assert!(bound.requires_mask());
    assert_eq!(bound.filter_capacity_bytes(), capacity);
    for values in [vec![false, true], vec![false, false, true, false, true]] {
        let decision = TokenSamplingDecision::new(TokenFilter::allowed(values).unwrap());
        bound.validate_decision(&decision).unwrap();
    }
}

#[test]
fn different_filter_mechanisms_require_a_different_quote() {
    let allowed = mask(4, &[true, false, false, false]);
    assert!(!contract(&TokenFilter::All, 0).uses_mask());
    assert!(!contract(&TokenFilter::All, 0).requires_mask());
    assert_eq!(
        contract(&TokenFilter::All, 100)
            .validate_decision(&TokenSamplingDecision::new(allowed.clone())),
        Err(TextControllerContractError::FilterMechanismMismatch)
    );
    assert_eq!(
        contract(&allowed, 100).validate_decision(&TokenSamplingDecision::new(TokenFilter::All)),
        Err(TextControllerContractError::FilterMechanismMismatch)
    );
    contract(&TokenFilter::All, 0)
        .validate_decision(&TokenSamplingDecision::new(TokenFilter::All))
        .unwrap();
}

#[test]
fn retained_spare_capacity_cannot_hide_behind_equal_mask_values() {
    let witness = mask(4, &[true, false, false, false]);
    let actual = mask(257, &[true, false, false, false]);
    let expected = TextControllerContractError::FilterCapacityExceeded {
        required_bytes: capacity_bytes(&actual).unwrap(),
        bound_bytes: capacity_bytes(&witness).unwrap(),
    };
    assert_eq!(
        contract(&witness, 0).validate_decision(&TokenSamplingDecision::new(actual)),
        Err(expected)
    );
}

#[test]
fn forced_decision_keeps_original_payload_even_without_capture_provenance() {
    let witness = mask(4, &[true, false, false, false]);
    let original = mask(131, &[true, true, false, false]);
    let original_bytes = capacity_bytes(&original).unwrap();
    let mut decision = TokenSamplingDecision::new(original);
    decision.override_filter(mask(4, &[false, true, false, false]));
    assert!(decision.capture_domain().is_none());
    assert_eq!(
        contract(&witness, original_bytes - 1).validate_decision(&decision),
        Err(TextControllerContractError::AdditionalPayloadExceeded {
            required_bytes: original_bytes,
            bound_bytes: original_bytes - 1,
        })
    );
    contract(&witness, original_bytes)
        .validate_decision(&decision)
        .unwrap();
    assert!(!decision.filter().allows(0));
    assert!(decision.filter().allows(1));
}

#[test]
fn tokenizer_validity_and_original_filter_share_the_additional_allowance() {
    let witness = mask(4, &[true, false, false, false]);
    let original = mask(17, &[true, true, false, false]);
    let validity = mask(29, &[true, true, true, false]);
    let additional = capacity_bytes(&original).unwrap() + capacity_bytes(&validity).unwrap();
    let mut decision = TokenSamplingDecision::new(original).with_tokenizer_validity(&validity);
    decision.override_filter(mask(4, &[false, true, false, false]));
    let bound = contract(&witness, additional);
    assert_eq!(bound.additional_host_bytes(), additional);
    bound.validate_decision(&decision).unwrap();
    assert_eq!(
        contract(&witness, additional - 1).validate_decision(&decision),
        Err(TextControllerContractError::AdditionalPayloadExceeded {
            required_bytes: additional,
            bound_bytes: additional - 1,
        })
    );
    let domain = decision.capture_domain().unwrap();
    assert!(domain.filter.allows(0));
    assert!(!decision.filter().allows(0));
    assert!(domain.tokenizer_validity.allows(2));
    assert!(!domain.filter.allows(2));
}

#[test]
fn executable_domain_validation_and_checked_capacity_remain_mandatory() {
    let witness = mask(8, &[true, false, false, false]);
    let outside = TokenSamplingDecision::new(mask(8, &[false, false, false, false, true]));
    assert_eq!(
        contract(&witness, 0).validate_decision(&outside),
        Err(TextControllerContractError::InvalidFilter(
            TokenFilterError::NoExecutableToken { output_width: 4 }
        ))
    );
    assert_eq!(
        TextControllerContract::from_workspace(
            TextControllerWorkspace {
                filter: (&witness).into(),
                additional_host_bytes: 0
            },
            0,
        ),
        Err(TextControllerContractError::InvalidFilter(
            TokenFilterError::EmptyVocabulary
        ))
    );
    assert_eq!(
        TextControllerContract::from_workspace(
            TextControllerWorkspace {
                filter: (&witness).into(),
                additional_host_bytes: u64::MAX
            },
            4,
        ),
        Err(TextControllerContractError::CapacityOverflow)
    );
}

fn optional(
    positions: usize,
    capacity: u64,
    additional: u64,
    output_width: usize,
) -> Result<TextControllerContract, TextControllerContractError> {
    TextControllerContract::from_workspace(
        TextControllerWorkspace {
            filter: TextFilterWorkspace::OptionalMask {
                max_mask_positions: positions,
                mask_capacity_bytes: capacity,
            },
            additional_host_bytes: additional,
        },
        output_width,
    )
}

#[test]
fn optional_masks_allow_mixed_decisions_without_restricting_unfiltered_membership() {
    let declaration = TextFilterWorkspace::OptionalMask {
        max_mask_positions: 2,
        mask_capacity_bytes: 8,
    };
    assert_eq!(declaration.mask_capacity_bytes().unwrap(), 8);
    declaration.validate_output_width(16).unwrap();
    let bound = optional(2, 8, 0, 16).unwrap();
    assert!(bound.uses_mask());
    assert!(!bound.requires_mask());
    assert_eq!(bound.filter_capacity_bytes(), 8);
    for filter in [
        TokenFilter::All,
        mask(8, &[false, true]),
        TokenFilter::All,
        mask(3, &[true]),
    ] {
        let decision = TokenSamplingDecision::new(filter);
        bound.validate_decision(&decision).unwrap();
        if matches!(decision.filter(), TokenFilter::All) {
            assert!(decision.filter().allows(15));
        }
    }
}

#[test]
fn optional_masks_enforce_logical_extent_capacity_and_executable_membership_separately() {
    let bound = optional(4, 16, 0, 4).unwrap();
    assert_eq!(
        bound.validate_decision(&TokenSamplingDecision::new(mask(
            8,
            &[true, false, false, false, false],
        ))),
        Err(TextControllerContractError::FilterPositionsExceeded {
            required_positions: 5,
            bound_positions: 4,
        })
    );
    assert_eq!(
        bound.validate_decision(&TokenSamplingDecision::new(mask(
            17,
            &[true, false, false, false],
        ))),
        Err(TextControllerContractError::FilterCapacityExceeded {
            required_bytes: 17,
            bound_bytes: 16,
        })
    );
    assert_eq!(
        optional(8, 16, 0, 4)
            .unwrap()
            .validate_decision(&TokenSamplingDecision::new(mask(
                8,
                &[false, false, false, false, true],
            ))),
        Err(TextControllerContractError::InvalidFilter(
            TokenFilterError::NoExecutableToken { output_width: 4 }
        ))
    );
    for filter in [
        TokenFilter::Allowed(vec![]),
        TokenFilter::Allowed(vec![false; 4]),
    ] {
        assert_eq!(
            bound.validate_decision(&TokenSamplingDecision::new(filter)),
            Err(TextControllerContractError::InvalidFilter(
                TokenFilterError::NoExecutableToken { output_width: 4 }
            ))
        );
    }
}

#[test]
fn optional_metadata_requires_nonzero_coherent_host_extent_and_checked_total_capacity() {
    for (positions, capacity) in [(0, 0), (0, 8), (4, 3)] {
        let declaration = TextFilterWorkspace::OptionalMask {
            max_mask_positions: positions,
            mask_capacity_bytes: capacity,
        };
        let expected = TextControllerContractError::InvalidOptionalMask {
            max_mask_positions: positions,
            mask_capacity_bytes: capacity,
        };
        assert_eq!(declaration.mask_capacity_bytes(), Err(expected.clone()));
        assert_eq!(declaration.validate_output_width(4), Err(expected.clone()));
        assert_eq!(optional(positions, capacity, 0, 4), Err(expected));
    }
    for (positions, capacity) in [(usize::MAX, u64::MAX), (1, (isize::MAX as u64) + 1)] {
        assert_eq!(
            optional(positions, capacity, 0, 4),
            Err(TextControllerContractError::CapacityOverflow)
        );
    }
    assert_eq!(
        optional(4, 4, 0, 0),
        Err(TextControllerContractError::InvalidFilter(
            TokenFilterError::EmptyVocabulary
        ))
    );
    assert_eq!(
        optional(4, 4, u64::MAX, 4),
        Err(TextControllerContractError::CapacityOverflow)
    );
    optional(4, 4, 0, 1).unwrap();
    assert_eq!(
        TextFilterWorkspace::from(&TokenFilter::All).mask_capacity_bytes(),
        Ok(0)
    );
    let exact = mask(13, &[true, false]);
    assert_eq!(
        TextFilterWorkspace::from(&exact).mask_capacity_bytes(),
        Ok(13)
    );
}

#[test]
fn optional_forcing_preserves_shared_provenance_and_additional_payload_checks() {
    let validity = crate::SharedTokenFilter::new(mask(29, &[true, true, true, false]));
    let original = mask(17, &[true, true, false, false]);
    let additional = capacity_bytes(&original).unwrap() + validity.capacity_bytes().unwrap();
    let mut decision =
        TokenSamplingDecision::new(original).with_shared_tokenizer_validity(&validity);
    decision.override_filter(mask(4, &[false, true, false, false]));
    optional(4, 8, additional, 4)
        .unwrap()
        .validate_decision(&decision)
        .unwrap();
    assert_eq!(
        optional(4, 8, additional - 1, 4)
            .unwrap()
            .validate_decision(&decision),
        Err(TextControllerContractError::AdditionalPayloadExceeded {
            required_bytes: additional,
            bound_bytes: additional - 1,
        })
    );
    assert!(decision
        .shared_tokenizer_validity()
        .unwrap()
        .same_storage(&validity));
    let provenance = decision.capture_domain().unwrap();
    assert!(provenance.filter.allows(0));
    assert!(!decision.filter().allows(0));
    assert!(provenance.tokenizer_validity.allows(2));
    decision.override_filter(TokenFilter::All);
    optional(4, 8, additional, 4)
        .unwrap()
        .validate_decision(&decision)
        .unwrap();
    assert!(decision
        .shared_tokenizer_validity()
        .unwrap()
        .same_storage(&validity));
    assert_eq!(
        capacity_bytes(decision.pre_override_filter.as_ref().unwrap()).unwrap(),
        17
    );
}
