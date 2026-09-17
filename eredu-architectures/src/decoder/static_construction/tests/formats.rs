use super::super::format::{
    standard_linear_format_with, CompanionSuffix, FormatStorage, OrdinaryFormat,
};
use super::*;
use eredu_nn::{LinearCompanionRole, LinearFormatSpec, ParameterNameView, ParameterSpec};

// Independent original worker retained before policy factoring. No shared naming
// or construction helper is called by this reference.
fn legacy_format(weight_name: &str, format: LinearFormat) -> Result<LinearFormatSpec, Error> {
    fn companion(weight: &str, name: String, component: &str) -> Result<ParameterSpec, Error> {
        let mut p = ParameterSpec::trainable(name).map_err(Error::backend)?;
        p.group = Some(weight.to_owned());
        if p.id.as_str() == weight {
            return Err(Error::backend(format!(
                "linear {component} companion reuses weight identity {weight:?}"
            )));
        }
        Ok(p)
    }
    let prefix = weight_name.strip_suffix(".weight").ok_or_else(|| {
        Error::backend(format!(
            "encoded ordinary linear parameter {weight_name:?} must end in .weight"
        ))
    });
    match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => LinearFormatSpec::unscaled(format),
        LinearFormat::E4M3BlockFp8(_) => {
            let prefix = prefix?;
            LinearFormatSpec::scaled(
                format,
                companion(weight_name, format!("{prefix}.weight_scale_inv"), "scale")?,
            )
        }
        LinearFormat::MxFp4 => {
            let prefix = prefix?;
            LinearFormatSpec::scaled(
                format,
                companion(weight_name, format!("{prefix}.scales"), "scale")?,
            )
        }
        LinearFormat::Affine(_) => {
            let prefix = prefix?;
            LinearFormatSpec::affine(
                format,
                companion(weight_name, format!("{prefix}.scales"), "scale")?,
                companion(weight_name, format!("{prefix}.biases"), "affine-bias")?,
            )
        }
    }
}
fn encodings() -> [LinearFormat; 5] {
    [
        LinearFormat::Dense,
        LinearFormat::Affine(Default::default()),
        LinearFormat::MxFp4,
        LinearFormat::E4M3BlockFp8(
            eredu_checkpoint::BlockFp8Format::new(
                32,
                64,
                eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint,
            )
            .unwrap(),
        ),
        LinearFormat::GgufIQuant {
            ggml_type: eredu_gguf::GgmlType::Q4_0,
            endian: eredu_gguf::Endian::Little,
        },
    ]
}
fn compare_row(row: StaticParameterDeclaration<'_>, expected: &ParameterSpec) {
    assert_eq!(row.name.to_string(), expected.id.as_str());
    assert_eq!(row.trainable, expected.trainable);
    assert_eq!(
        row.alias_of.map(|n| n.to_string()),
        expected.alias_of.as_ref().map(ToString::to_string)
    );
    assert_eq!(row.group.map(|n| n.to_string()), expected.group);
    assert_eq!(row.linear_companion, expected.linear_companion);
    assert_eq!(
        row.linear_companion_of.map(|n| n.to_string()),
        expected
            .linear_companion_of
            .as_ref()
            .map(ToString::to_string)
    );
    assert_eq!(row.linear_row_layout, expected.linear_row_layout);
}
#[test]
fn ordinary_and_borrowed_static_formats_match_independent_complete_declarations() {
    for format in encodings() {
        for weight in ["model.α.weight", "weight", ".weight"] {
            let reference = legacy_format(weight, format);
            let ordinary = crate::linear_format::standard_linear_format(weight, format);
            let fixed = static_linear_format(weight, format);
            match (reference, ordinary, fixed) {
                (Ok(reference), Ok(ordinary), Ok(fixed)) => {
                    assert_eq!(ordinary, reference);
                    assert_eq!(fixed.format, reference.encoding());
                    assert_eq!(fixed.row_layout, reference.row_layout());
                    assert_eq!(fixed.scale.is_some(), reference.scale().is_some());
                    assert_eq!(
                        fixed.affine_bias.is_some(),
                        reference.affine_bias().is_some()
                    );
                    for (row, p) in fixed
                        .scale
                        .into_iter()
                        .chain(fixed.affine_bias)
                        .zip(reference.scale().into_iter().chain(reference.affine_bias()))
                    {
                        compare_row(row, p);
                        assert_eq!(row.name.parts()[0].as_ptr(), weight.as_ptr());
                        assert_eq!(row.group.unwrap().parts()[0].as_ptr(), weight.as_ptr());
                        assert!(row.linear_companion_of.is_none()); // Backend binding fills this later.
                    }
                    assert_eq!(
                        fixed
                            .validation_view()
                            .validate_for_weight_fixed(ParameterNameView::new(weight)),
                        Ok(())
                    );
                }
                (Err(reference), Err(ordinary), Err(fixed)) => {
                    assert_eq!(ordinary.to_string(), reference.to_string());
                    assert_eq!(fixed.to_string(), reference.to_string());
                    assert!(std::error::Error::source(&ordinary).is_none());
                }
                _ => panic!("ordinary/reference/fixed result differs for {weight:?} {format:?}"),
            }
        }
    }
    // Keep exact literal-format allocation policy as well as resulting strings.
    for prefix in ["", "α", "model.decoder.long_module_name"] {
        for (suffix, old) in [
            (
                CompanionSuffix::BlockScale,
                format!("{prefix}.weight_scale_inv"),
            ),
            (CompanionSuffix::Scale, format!("{prefix}.scales")),
            (CompanionSuffix::AffineBias, format!("{prefix}.biases")),
        ] {
            let new = OrdinaryFormat
                .companion("owner.weight", prefix, suffix, "scale")
                .unwrap();
            // ParameterId consumes the same String unchanged. Compare the policy
            // directly because ParameterId intentionally hides owning capacity.
            let direct = suffix.ordinary_name(prefix);
            assert_eq!(new.id.as_str(), old);
            assert_eq!(direct.capacity(), old.capacity());
        }
    }
}

struct ObservedError {
    ordinary: Error,
    _retirement: Owner,
}
struct ObservedFormat {
    events: Events,
    fail_bias: bool,
}
impl<'a> FormatStorage<'a> for ObservedFormat {
    type Parameter = (ParameterSpec, Owner);
    type Output = LinearFormatSpec;
    type Error = ObservedError;
    fn missing_suffix(&mut self, weight: &'a str) -> Self::Error {
        self.events.borrow_mut().push("missing-suffix");
        ObservedError {
            ordinary: OrdinaryFormat.missing_suffix(weight),
            _retirement: Owner {
                label: "suffix-error-retire",
                events: self.events.clone(),
            },
        }
    }
    fn companion(
        &mut self,
        weight: &'a str,
        prefix: &'a str,
        suffix: CompanionSuffix,
        component: &'static str,
    ) -> Result<Self::Parameter, Self::Error> {
        let bias = matches!(suffix, CompanionSuffix::AffineBias);
        self.events
            .borrow_mut()
            .push(if bias { "bias" } else { "scale" });
        if bias && self.fail_bias {
            return Err(ObservedError {
                ordinary: Error::backend("bias failed"),
                _retirement: Owner {
                    label: "bias-error-retire",
                    events: self.events.clone(),
                },
            });
        }
        let p = OrdinaryFormat
            .companion(weight, prefix, suffix, component)
            .unwrap();
        Ok((
            p,
            Owner {
                label: if bias { "bias-retire" } else { "scale-retire" },
                events: self.events.clone(),
            },
        ))
    }
    fn unscaled(&mut self, format: LinearFormat) -> Result<Self::Output, Self::Error> {
        self.events.borrow_mut().push("unscaled");
        Ok(OrdinaryFormat.unscaled(format).unwrap())
    }
    fn scaled(
        &mut self,
        format: LinearFormat,
        scale: Self::Parameter,
    ) -> Result<Self::Output, Self::Error> {
        self.events.borrow_mut().push("scaled");
        Ok(OrdinaryFormat.scaled(format, scale.0).unwrap())
    }
    fn affine(
        &mut self,
        format: LinearFormat,
        scale: Self::Parameter,
        bias: Self::Parameter,
    ) -> Result<Self::Output, Self::Error> {
        self.events.borrow_mut().push("affine");
        Ok(OrdinaryFormat.affine(format, scale.0, bias.0).unwrap())
    }
}
#[test]
fn shared_format_policy_preserves_eager_discard_and_partial_companion_retirement() {
    for format in [encodings()[0], encodings()[4]] {
        let events = Events::default();
        let mut policy = ObservedFormat {
            events: events.clone(),
            fail_bias: false,
        };
        assert!(standard_linear_format_with("no_suffix", format, &mut policy).is_ok());
        assert_eq!(
            *events.borrow(),
            ["missing-suffix", "unscaled", "suffix-error-retire"]
        );
    }
    let events = Events::default();
    let mut policy = ObservedFormat {
        events: events.clone(),
        fail_bias: true,
    };
    let error = standard_linear_format_with(
        "model.weight",
        LinearFormat::Affine(Default::default()),
        &mut policy,
    )
    .err()
    .unwrap();
    assert_eq!(error.ordinary.to_string(), "bias failed");
    assert_eq!(*events.borrow(), ["scale", "bias", "scale-retire"]);
    drop(error);
    assert_eq!(events.borrow().last(), Some(&"bias-error-retire"));
    let error = static_linear_format(
        "no_suffix",
        LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
            group_size: 0,
            ..Default::default()
        }),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StaticDeclarationError::MissingWeightSuffix("no_suffix")
    ));
    let error = static_linear_format(
        "model.weight",
        LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
            group_size: 0,
            ..Default::default()
        }),
    )
    .unwrap_err();
    assert!(matches!(error, StaticDeclarationError::Format(_)));
}
