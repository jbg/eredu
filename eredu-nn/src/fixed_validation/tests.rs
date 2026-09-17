use super::*;
use crate::{
    GatedProductGroupLayout, GatedProductGroupParameters, GatedProductPolicy,
    GroupedGatedProductSpec, GroupedProjectionSpec, GroupedRelu2Spec,
};

fn parameter(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}

fn projection(name: &str) -> GroupedProjectionSpec {
    GroupedProjectionSpec::new(
        parameter(name),
        None,
        LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
    )
    .unwrap()
}

fn ordinary_error(result: Result<(), Error>, expected: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.to_string(), expected);
    assert!(std::error::Error::source(&error).is_none());
}

#[test]
fn fixed_format_validation_preserves_precedence_and_borrows_companion_identities() {
    let mut source = LinearFormatSpec::scaled(LinearFormat::MxFp4, parameter("scale")).unwrap();
    source.row_layout = LinearRowLayout::equal_partitions(2).unwrap();
    source.affine_bias = Some(parameter("unexpected"));
    assert_eq!(
        source.validate_fixed(),
        Err(LinearFormatValidationError::RowLayout)
    );
    ordinary_error(
        source.validate(),
        "independent row blocks require a block-FP8 encoding",
    );
    source.row_layout = LinearRowLayout::Contiguous;
    assert!(matches!(
        source.validate_fixed(),
        Err(LinearFormatValidationError::CompanionCardinality {
            expected: (true, false),
            actual: (true, true),
            ..
        })
    ));
    source.affine_bias = None;
    source.scale.as_mut().unwrap().linear_companion = None;
    assert_eq!(
        source.validate_fixed(),
        Err(LinearFormatValidationError::CompanionRole)
    );
    source.scale.as_mut().unwrap().linear_companion = Some(LinearCompanionRole::Scale);
    let primary = parameter("scale");
    let pointer = source.scale().unwrap() as *const ParameterSpec;
    assert_eq!(source.validate_fixed(), Ok(()));
    assert_eq!(
        source.validate_for_weight_fixed(&primary),
        Err(LinearWeightValidationError::PrimaryIdentity)
    );
    ordinary_error(
        source.validate_for_weight(&primary),
        "linear format companion reuses primary weight identity scale",
    );
    assert_eq!(source.scale().unwrap() as *const ParameterSpec, pointer);

    let projection =
        GroupedProjectionSpec::new(parameter("weight"), Some(parameter("bias")), source).unwrap();
    let borrowed = projection.parameters_borrowed().collect::<Vec<_>>();
    let owned = projection.parameters();
    assert_eq!(borrowed.len(), 3);
    for (left, right) in borrowed.iter().zip(&owned) {
        assert!(std::ptr::eq(*left, *right));
    }
    assert!(std::ptr::eq(borrowed[0], projection.weight()));
    assert!(std::ptr::eq(borrowed[1], projection.bias().unwrap()));
    assert!(std::ptr::eq(
        borrowed[2],
        projection.format().scale().unwrap()
    ));
}

#[test]
fn fixed_grouped_checks_keep_projection_and_bank_error_order_distinct() {
    let group =
        GatedProductGroupParameters::new(projection("gate"), projection("up"), projection("down"));
    let mut gated = GroupedGatedProductSpec::new(
        1,
        3,
        5,
        3,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Independent(vec![group]),
    )
    .unwrap();
    let GatedProductGroupLayout::Independent(groups) = &mut gated.layout else {
        unreachable!()
    };
    groups[0].up.weight.id = groups[0].gate.weight.id.clone();
    groups[0].down.format.scale = Some(parameter("unexpected"));
    // Gated banks admit each projection's identities before checking the next
    // projection. The earlier repeated gate identity wins over malformed down.
    assert_eq!(
        gated.validate_fixed(),
        Err(GroupedBankValidationError::ParameterIdentity {
            projection: 1,
            parameter: 0
        })
    );
    ordinary_error(
        gated.validate(),
        "gated-product group parameter identity gate is duplicated",
    );
    let GatedProductGroupLayout::Independent(groups) = &mut gated.layout else {
        unreachable!()
    };
    groups[0].up.weight.id = parameter("up").id;
    assert!(matches!(
        gated.validate_fixed(),
        Err(GroupedBankValidationError::Projection { projection: 2, .. })
    ));

    let mut relu = GroupedRelu2Spec::new(1, 3, 5, projection("up"), projection("down")).unwrap();
    relu.down.weight.id = relu.up.weight.id.clone();
    relu.down.format.scale = Some(parameter("unexpected"));
    // ReLU2 checks both complete projections before bank-wide duplicates.
    assert!(matches!(
        relu.validate_fixed(),
        Err(GroupedRelu2ValidationError::Projection { projection: 1, .. })
    ));
    ordinary_error(
        relu.validate(),
        "linear format Dense requires scale/bias companions (false, false), got (true, false)",
    );
    relu.down.format.scale = None;
    assert_eq!(
        relu.validate_fixed(),
        Err(GroupedRelu2ValidationError::ParameterIdentity {
            projection: 1,
            parameter: 0
        })
    );
    ordinary_error(
        relu.validate(),
        "ReLU2 group parameter identity up is duplicated",
    );
}

#[test]
fn fixed_grouped_independent_source_has_no_group_cap_and_preserves_first_duplicate() {
    let groups = (0..257)
        .map(|index| {
            GatedProductGroupParameters::new(
                projection(&format!("g{index}.gate")),
                projection(&format!("g{index}.up")),
                projection(&format!("g{index}.down")),
            )
        })
        .collect();
    let mut source = GroupedGatedProductSpec::new(
        257,
        3,
        5,
        3,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Independent(groups),
    )
    .unwrap();
    assert_eq!(source.validate_fixed(), Ok(()));
    assert_eq!(source.projections_borrowed().count(), 771);
    let GatedProductGroupLayout::Independent(groups) = &mut source.layout else {
        unreachable!()
    };
    let first = groups[0].gate.weight.id.clone();
    groups[256].up.weight.id = first;
    groups[256].down.weight.id = groups[0].up.weight.id.clone();
    assert_eq!(
        source.validate_fixed(),
        Err(GroupedBankValidationError::ParameterIdentity {
            projection: 769,
            parameter: 0
        })
    );
    ordinary_error(
        source.validate(),
        "gated-product group parameter identity g0.gate is duplicated",
    );
}
