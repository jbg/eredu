use super::*;
use crate::cache::StateResidencyClass;
use std::num::NonZeroU32;
mod legacy;

fn layout(tensor: Option<StateTensorPolicy>) -> StateMemoryLayout {
    let mut policies = vec![
        LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 3).unwrap(),
        LayerCachePolicy::key_only(AttentionPolicy::sliding(9).unwrap(), 1, 5).unwrap(),
        LayerCachePolicy::key_value(AttentionPolicy::sliding(2).unwrap(), 1, 4).unwrap(),
        LayerCachePolicy::key_only(AttentionPolicy::sliding(9).unwrap(), 2, 2).unwrap(),
    ];
    if let Some(tensor) = tensor {
        policies.push(LayerCachePolicy::fixed_only(vec![tensor]).unwrap());
    }
    let count = policies.len();
    let offsets = (0..count).map(|n| -(n as i32)).collect();
    StateMemoryLayout::new(
        LayerSchedule::new(count, policies).unwrap(),
        offsets,
        7,
        8,
        EstimationCompleteness::Complete,
    )
    .unwrap()
}
fn tensor(shape: Vec<StateTensorDimension>) -> StateTensorPolicy {
    StateTensorPolicy {
        role: StateTensorRole::Recurrent,
        shape,
        dtype: StateTensorDtype::Floating,
        residency: StateResidencyClass::LayerScopedOffloadable,
        presence: StateTensorPresence::Required,
    }
}
fn compare(layout: &StateMemoryLayout, input: InputTokenCount, output: u64, batch: u64, width: u8) {
    let width = NonZeroU8::new(width).unwrap();
    let expected = legacy::estimate_runtime_state(layout, input, output, batch, width);
    let actual = estimate_runtime_state_facts(layout, input, output, batch, width)
        .map(RuntimeStateFacts::into_estimate)
        .map_err(CapabilityError::from);
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual, expected);
            assert_eq!(
                actual.completeness,
                EstimationCompleteness::PersistentStateOnly
            );
            assert!(actual.selected_state_backing.is_none());
            assert!(actual.execution_workspace.is_none());
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(
                std::mem::discriminant(&actual),
                std::mem::discriminant(&expected)
            );
            assert_eq!(actual.to_string(), expected.to_string());
        }
        (actual, expected) => panic!("state fact mismatch: {actual:?} vs {expected:?}"),
    }
}

#[test]
fn borrowed_state_facts_match_independent_legacy_dimensions_windows_and_presence() {
    use StateTensorDimension as D;
    let divisor = NonZeroU32::new(3).unwrap();
    let shapes = [
        vec![D::Scalar],
        vec![D::Batch, D::Fixed(NonZeroU32::new(5).unwrap())],
        vec![D::Batch, D::PrefixTokens, D::Fixed(divisor)],
        vec![D::Batch, D::PrefixTokensDiv(divisor), D::Fixed(divisor)],
        vec![D::Batch, D::PrefixTokensRem(divisor), D::Fixed(divisor)],
    ];
    for shape in shapes {
        for presence in [
            StateTensorPresence::Required,
            StateTensorPresence::Optional,
            StateTensorPresence::PrefixRemainderNonZero(divisor),
            StateTensorPresence::PrefixAtLeast(divisor),
        ] {
            for dtype in [
                StateTensorDtype::Floating,
                StateTensorDtype::Float32,
                StateTensorDtype::Int32,
                StateTensorDtype::Uint32,
            ] {
                let mut value = tensor(shape.clone());
                value.presence = presence;
                value.dtype = dtype;
                let layout = layout(Some(value));
                for positions in [0, 1, 2, 3, 4, 17] {
                    for batch in [1, 3] {
                        for width in [2, 4] {
                            compare(
                                &layout,
                                InputTokenCount::prepared(
                                    positions,
                                    2,
                                    positions + 4,
                                    11,
                                    ObservationKind::Conservative,
                                ),
                                2,
                                batch,
                                width,
                            );
                        }
                    }
                }
            }
        }
    }
    let mut prefix = tensor(vec![D::Batch, D::PrefixTokens, D::Fixed(divisor)]);
    prefix.role = StateTensorRole::PrefixEmbedding;
    prefix.residency = StateResidencyClass::AlwaysDeviceMutable;
    prefix.presence = StateTensorPresence::Optional;
    compare(
        &layout(Some(prefix)),
        InputTokenCount::prepared(3, 5, 8, 9, ObservationKind::Exact),
        4,
        1,
        4,
    );
}

#[test]
fn borrowed_state_conversion_errors_keep_legacy_precedence_over_products_and_zero() {
    use StateTensorDimension as D;
    let max = NonZeroU32::new(i32::MAX as u32).unwrap();
    let invalid = NonZeroU32::new(u32::MAX).unwrap();
    let required = tensor(vec![
        D::Fixed(max),
        D::Fixed(max),
        D::Fixed(max),
        D::Fixed(invalid),
    ]);
    let late_invalid = layout(Some(required));
    let error = estimate_runtime_state_facts(
        &late_invalid,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        AdmissionPolicyError::InvalidConfiguration {
            field: "layer_layout",
            ..
        }
    ));
    compare(&late_invalid, InputTokenCount::text(1), 0, 1, 4);
    let zero = layout(Some(tensor(vec![
        D::PrefixTokensRem(NonZeroU32::new(2).unwrap()),
        D::Fixed(invalid),
    ])));
    compare(&zero, InputTokenCount::text(4), 0, 1, 4);
    for positions in [0, i32::MAX as u64, i32::MAX as u64 + 1, u64::MAX] {
        for batch in [0, 1, i32::MAX as u64 + 1, u64::MAX] {
            compare(
                &layout(Some(tensor(vec![D::Batch, D::PrefixTokens]))),
                InputTokenCount::text(positions),
                1,
                batch,
                4,
            );
        }
    }
}

#[test]
fn borrowed_window_plan_has_exact_no_write_short_or_long_destinations() {
    let source = layout(None);
    let facts = estimate_runtime_state_facts(
        &source,
        InputTokenCount::text(7),
        3,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let windows = facts.sliding_windows;
    assert_eq!(windows.len(), 2);
    assert_eq!(windows.iter().collect::<Vec<_>>(), [2, 9]);
    let mut exact = [99; 2];
    windows.fill(&mut exact).unwrap();
    assert_eq!(exact, [2, 9]);
    let mut short = [77];
    assert!(windows.fill(&mut short).is_err());
    assert_eq!(short, [77]);
    let mut long = [88; 3];
    assert!(windows.fill(&mut long).is_err());
    assert_eq!(long, [88; 3]);
    let empty = StateMemoryLayout::new(
        LayerSchedule::empty(),
        Vec::new(),
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let empty = estimate_runtime_state_facts(
        &empty,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    assert!(empty.sliding_windows.is_empty());
    empty.sliding_windows.fill(&mut []).unwrap();
    let single = StateMemoryLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_only(AttentionPolicy::sliding(7).unwrap(), 1, 2).unwrap()],
        )
        .unwrap(),
        vec![0],
        2,
        4,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let facts = estimate_runtime_state_facts(
        &single,
        InputTokenCount::text(3),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let mut one = [91];
    facts.sliding_windows.fill(&mut one).unwrap();
    assert_eq!(one, [7]);
    let mut sentinel = [93];
    let failed = estimate_runtime_state_facts(
        &single,
        InputTokenCount::text(u64::MAX),
        1,
        1,
        NonZeroU8::new(4).unwrap(),
    );
    assert!(failed.is_err());
    if let Ok(facts) = failed {
        facts.sliding_windows.fill(&mut sentinel).unwrap();
    }
    assert_eq!(sentinel, [93]);
}
