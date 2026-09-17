use super::*;
use crate::{
    LinearFormatSpec, ParameterId, ParameterSpec, VocabularyParallelRange, VocabularyRangeError,
};

#[test]
fn borrowed_names_preserve_unicode_empty_and_segment_boundary_identity() {
    for (a, b) in [("", ""), ("\u{2003}", "\n"), (" \t", "\u{3000}")] {
        let borrowed = ParameterNameView::joined(a, b);
        let ordinary = format!("{a}{b}");
        assert_eq!(borrowed.validate_fixed(), Err(ParameterNameError::Empty));
        assert!(matches!(
            ParameterId::new(ordinary),
            Err(crate::ParameterTopologyError::EmptyId)
        ));
    }
    for (a, b) in [
        ("α", ".weight"),
        ("", "μ.scales"),
        ("two words", ""),
        ("\u{200b}", ""),
    ] {
        let name = ParameterNameView::joined(a, b);
        let ordinary = format!("{a}{b}");
        assert_eq!(name, ParameterNameView::new(&ordinary));
        assert_eq!(name.validate_fixed(), Ok(()));
        assert_eq!(name.byte_len(), Some(ordinary.len()));
        assert_eq!(name.to_string(), ordinary);
        assert_eq!(name.parts()[0].as_ptr(), a.as_ptr());
        assert_eq!(name.parts()[1].as_ptr(), b.as_ptr());
        assert!(ParameterId::new(ordinary).is_ok());
    }
    assert_eq!(
        ParameterNameView::joined("ab", "c"),
        ParameterNameView::joined("a", "bc")
    );
    assert_ne!(
        ParameterNameView::joined("a", "bc"),
        ParameterNameView::joined("ab", "d")
    );
}

// Independent operation50 owning kernel, retained before the borrowed-view change.
fn legacy_format(source: &LinearFormatSpec) -> Result<(), LinearFormatValidationError> {
    source.format.validate_fixed()?;
    if source.row_layout != LinearRowLayout::Contiguous
        && !matches!(source.format, LinearFormat::E4M3BlockFp8(_))
    {
        return Err(LinearFormatValidationError::RowLayout);
    }
    let expected = match source.format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => (false, false),
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => (true, false),
        LinearFormat::Affine(_) => (true, true),
    };
    let actual = (source.scale.is_some(), source.affine_bias.is_some());
    if actual != expected {
        return Err(LinearFormatValidationError::CompanionCardinality {
            format: source.format,
            expected,
            actual,
        });
    }
    if source
        .scale
        .as_ref()
        .zip(source.affine_bias.as_ref())
        .is_some_and(|(a, b)| a.id == b.id)
    {
        return Err(LinearFormatValidationError::CompanionIdentity);
    }
    if source
        .scale
        .as_ref()
        .is_some_and(|p| p.linear_companion != Some(LinearCompanionRole::Scale))
        || source
            .affine_bias
            .as_ref()
            .is_some_and(|p| p.linear_companion != Some(LinearCompanionRole::AffineBias))
    {
        return Err(LinearFormatValidationError::CompanionRole);
    }
    Ok(())
}

#[test]
fn borrowed_format_checks_match_independent_owning_order_and_complete_names() {
    let mut owned = LinearFormatSpec::affine(
        LinearFormat::Affine(eredu_checkpoint::AffineQuantization::default()),
        ParameterSpec::trainable("long.scales").unwrap(),
        ParameterSpec::trainable("long.biases").unwrap(),
    )
    .unwrap();
    let initial = owned.clone();
    for stage in 0..7 {
        owned = initial.clone();
        match stage {
            0 => {}
            1 => {
                owned.format = LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
                    group_size: 0,
                    ..Default::default()
                })
            }
            2 => owned.row_layout = LinearRowLayout::equal_partitions(2).unwrap(),
            3 => owned.affine_bias = None,
            4 => owned.affine_bias.as_mut().unwrap().id = owned.scale.as_ref().unwrap().id.clone(),
            5 => owned.scale.as_mut().unwrap().linear_companion = None,
            6 => {
                owned.scale.as_mut().unwrap().linear_companion = None;
                owned.affine_bias.as_mut().unwrap().id = owned.scale.as_ref().unwrap().id.clone();
            }
            _ => unreachable!(),
        }
        assert_eq!(owned.validate_fixed(), legacy_format(&owned));
        let result = owned.validate();
        assert_eq!(result.is_ok(), legacy_format(&owned).is_ok());
        if let (Err(actual), Err(expected)) = (result, legacy_format(&owned)) {
            assert_eq!(actual.to_string(), expected.to_string());
            assert!(std::error::Error::source(&actual).is_none());
        }
    }
    let mut view = initial.validation_view();
    view.scale.as_mut().unwrap().name = ParameterNameView::joined("long", ".scales");
    view.affine_bias.as_mut().unwrap().name = ParameterNameView::joined("long.", "scales");
    view.scale.as_mut().unwrap().role = None;
    assert_eq!(
        view.validate_fixed(),
        Err(LinearFormatValidationError::CompanionIdentity)
    );
    view.affine_bias.as_mut().unwrap().name = ParameterNameView::joined("long", ".biases");
    assert_eq!(
        view.validate_fixed(),
        Err(LinearFormatValidationError::CompanionRole)
    );
    view.scale.as_mut().unwrap().role = Some(LinearCompanionRole::Scale);
    assert_eq!(
        view.validate_for_weight_fixed(ParameterNameView::joined("long.s", "cales")),
        Err(LinearWeightValidationError::PrimaryIdentity)
    );
    assert_eq!(
        view.validate_for_weight_fixed(ParameterNameView::new("long.weight")),
        Ok(())
    );
}

#[test]
fn fixed_vocabulary_range_checks_preserve_ordinary_precedence_and_messages() {
    for (global, start, end, rows, expected) in [
        (0, 0, 0, -1, "invalid vocabulary-parallel range 0..0 of 0"),
        (11, 8, 4, -1, "invalid vocabulary-parallel range 8..4 of 11"),
        (
            11,
            4,
            12,
            0,
            "invalid vocabulary-parallel range 4..12 of 11",
        ),
        (
            11,
            4,
            8,
            -1,
            "vocabulary-parallel operator declares -1 rows but ownership covers 11",
        ),
        (
            11,
            4,
            8,
            10,
            "vocabulary-parallel operator declares 10 rows but ownership covers 11",
        ),
    ] {
        let range = VocabularyParallelRange {
            global_vocabulary: global,
            local: start..end,
        };
        let fixed = range.validate_global_rows_fixed(rows).unwrap_err();
        let error = range.validate_global_rows(rows).unwrap_err();
        assert_eq!(fixed.to_string(), expected);
        assert_eq!(error.to_string(), expected);
        assert!(std::error::Error::source(&error).is_none());
        if start >= end || end > global {
            assert!(matches!(fixed, VocabularyRangeError::InvalidRange { .. }));
        }
    }
    let valid = VocabularyParallelRange {
        global_vocabulary: 11,
        local: 4..8,
    };
    assert_eq!(valid.validate_global_rows_fixed(11), Ok(()));
    assert_eq!(valid.balanced_peer_widths(3, 1).unwrap(), [4, 4, 3]);
    let too_large = VocabularyParallelRange {
        global_vocabulary: usize::MAX,
        local: 1..2,
    };
    assert!(matches!(
        too_large.validate_global_rows_fixed(i32::MAX),
        Err(VocabularyRangeError::GlobalRows { .. })
    ));
}
