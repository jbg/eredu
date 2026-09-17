//! Shared ordinary-matrix naming with explicit ordinary and borrowed storage.
use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    LinearCompanionRole, LinearCompanionView, LinearFormatSpec, LinearFormatValidationError,
    LinearFormatView, LinearRowLayout, ParameterNameError, ParameterNameView,
};

/// Complete logical row emitted by this static-module construction convention.
///
/// Borrowed names and scalar metadata own no parameter, native object or charge.
/// Destination bytes, rows and all native construction controls need separate
/// source-bound ownership before this can participate in original admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticParameterDeclaration<'a> {
    /// Complete authoritative logical identity.
    pub name: ParameterNameView<'a>,
    /// Default trainability from construction.
    pub trainable: bool,
    /// Logical alias destination, if declared.
    pub alias_of: Option<ParameterNameView<'a>>,
    /// Atomic physical encoding group, if present.
    pub group: Option<ParameterNameView<'a>>,
    /// Physical companion role.
    pub linear_companion: Option<LinearCompanionRole>,
    /// Construction-level primary attachment, before the backend binds metadata.
    pub linear_companion_of: Option<ParameterNameView<'a>>,
    /// Independent primary row-block origins.
    pub linear_row_layout: LinearRowLayout,
}
impl<'a> StaticParameterDeclaration<'a> {
    /// Declares the same trainable default as ordinary ParameterSpec::trainable.
    pub fn trainable(name: ParameterNameView<'a>) -> Result<Self, ParameterNameError> {
        name.validate_fixed()?;
        Ok(Self {
            name,
            trainable: true,
            alias_of: None,
            group: None,
            linear_companion: None,
            linear_companion_of: None,
            linear_row_layout: LinearRowLayout::Contiguous,
        })
    }
}

/// Complete borrowed companions produced by the ordinary-matrix naming worker.
#[derive(Debug, Clone, Copy)]
pub struct StaticLinearFormatDeclaration<'a> {
    /// Physical checkpoint encoding.
    pub format: LinearFormat,
    /// Exact scale declaration.
    pub scale: Option<StaticParameterDeclaration<'a>>,
    /// Exact affine-bias declaration.
    pub affine_bias: Option<StaticParameterDeclaration<'a>>,
    /// Independent row-block origins.
    pub row_layout: LinearRowLayout,
}
impl StaticLinearFormatDeclaration<'_> {
    /// Borrows the closed view used by ordinary and fixed format validation.
    pub fn validation_view(&self) -> LinearFormatView<'_> {
        fn companion(row: StaticParameterDeclaration<'_>) -> LinearCompanionView<'_> {
            LinearCompanionView {
                name: row.name,
                role: row.linear_companion,
            }
        }
        LinearFormatView {
            format: self.format,
            row_layout: self.row_layout,
            scale: self.scale.map(companion),
            affine_bias: self.affine_bias.map(companion),
        }
    }
}

/// Fixed failure while creating borrowed ordinary-matrix declarations.
/// Source text is borrowed; no diagnostic String or ParameterSpec is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StaticDeclarationError<'a> {
    /// The parameter identity is empty.
    #[error(transparent)]
    Name(#[from] ParameterNameError),
    /// An encoding needs a companion suffix, but the primary has no .weight suffix.
    #[error("encoded ordinary linear parameter {0:?} must end in .weight")]
    MissingWeightSuffix(&'a str),
    /// A generated physical companion reused the primary identity.
    #[error("linear {component} companion reuses weight identity {weight:?}")]
    CompanionIdentity {
        /// Architecture-declared physical component.
        component: &'static str,
        /// Actual borrowed primary identity.
        weight: &'a str,
    },
    /// The actual format, roles or cardinality are invalid.
    #[error(transparent)]
    Format(#[from] LinearFormatValidationError),
}

/// Creates exact borrowed companion declarations through the ordinary naming
/// worker. It does not validate/create a primary parameter or native module.
pub fn static_linear_format(
    weight_name: &str,
    format: LinearFormat,
) -> Result<StaticLinearFormatDeclaration<'_>, StaticDeclarationError<'_>> {
    standard_linear_format_with(weight_name, format, &mut BorrowedFormat)
}

// Keep literal-format capacity policy as well as names in one declaration.
macro_rules! companion_suffixes {
    ($($variant:ident => $literal:literal),+ $(,)?) => {
        #[derive(Debug, Clone, Copy)]
        pub(crate) enum CompanionSuffix { $($variant),+ }
        impl CompanionSuffix {
            pub(crate) fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $literal),+ }
            }
            pub(super) fn ordinary_name(self, prefix: &str) -> String {
                match self {
                    $(Self::$variant => format!(concat!("{prefix}", $literal), prefix = prefix)),+
                }
            }
        }
    };
}
companion_suffixes! {
    BlockScale => ".weight_scale_inv",
    Scale => ".scales",
    AffineBias => ".biases",
}

pub(crate) trait FormatStorage<'a> {
    type Parameter;
    type Output;
    type Error;
    fn missing_suffix(&mut self, weight: &'a str) -> Self::Error;
    fn companion(
        &mut self,
        weight: &'a str,
        prefix: &'a str,
        suffix: CompanionSuffix,
        component: &'static str,
    ) -> Result<Self::Parameter, Self::Error>;
    fn unscaled(&mut self, format: LinearFormat) -> Result<Self::Output, Self::Error>;
    fn scaled(
        &mut self,
        format: LinearFormat,
        scale: Self::Parameter,
    ) -> Result<Self::Output, Self::Error>;
    fn affine(
        &mut self,
        format: LinearFormat,
        scale: Self::Parameter,
        bias: Self::Parameter,
    ) -> Result<Self::Output, Self::Error>;
}

pub(crate) fn standard_linear_format_with<'a, S: FormatStorage<'a>>(
    weight_name: &'a str,
    format: LinearFormat,
    storage: &mut S,
) -> Result<S::Output, S::Error> {
    // Intentionally eager, even when Dense/GGUF discard this error. The ordinary
    // policy's error construction and retirement are existing observable work.
    let prefix = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| storage.missing_suffix(weight_name));
    match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => storage.unscaled(format),
        LinearFormat::E4M3BlockFp8(_) => {
            let prefix = prefix?;
            let scale =
                storage.companion(weight_name, prefix, CompanionSuffix::BlockScale, "scale")?;
            storage.scaled(format, scale)
        }
        LinearFormat::MxFp4 => {
            let prefix = prefix?;
            let scale = storage.companion(weight_name, prefix, CompanionSuffix::Scale, "scale")?;
            storage.scaled(format, scale)
        }
        LinearFormat::Affine(_) => {
            let prefix = prefix?;
            let scale = storage.companion(weight_name, prefix, CompanionSuffix::Scale, "scale")?;
            let bias = storage.companion(
                weight_name,
                prefix,
                CompanionSuffix::AffineBias,
                "affine-bias",
            )?;
            storage.affine(format, scale, bias)
        }
    }
}

pub(crate) struct OrdinaryFormat;
impl<'a> FormatStorage<'a> for OrdinaryFormat {
    type Parameter = eredu_nn::ParameterSpec;
    type Output = LinearFormatSpec;
    type Error = eredu_nn::Error;
    fn missing_suffix(&mut self, weight: &str) -> Self::Error {
        eredu_nn::Error::backend(format!(
            "encoded ordinary linear parameter {weight:?} must end in .weight"
        ))
    }
    fn companion(
        &mut self,
        weight: &str,
        prefix: &str,
        suffix: CompanionSuffix,
        component: &'static str,
    ) -> Result<Self::Parameter, Self::Error> {
        crate::linear_format::companion(weight, suffix.ordinary_name(prefix), component)
    }
    fn unscaled(&mut self, format: LinearFormat) -> Result<Self::Output, Self::Error> {
        LinearFormatSpec::unscaled(format)
    }
    fn scaled(
        &mut self,
        format: LinearFormat,
        scale: Self::Parameter,
    ) -> Result<Self::Output, Self::Error> {
        LinearFormatSpec::scaled(format, scale)
    }
    fn affine(
        &mut self,
        format: LinearFormat,
        scale: Self::Parameter,
        bias: Self::Parameter,
    ) -> Result<Self::Output, Self::Error> {
        LinearFormatSpec::affine(format, scale, bias)
    }
}

struct BorrowedFormat;
impl<'a> FormatStorage<'a> for BorrowedFormat {
    type Parameter = StaticParameterDeclaration<'a>;
    type Output = StaticLinearFormatDeclaration<'a>;
    type Error = StaticDeclarationError<'a>;
    fn missing_suffix(&mut self, weight: &'a str) -> Self::Error {
        StaticDeclarationError::MissingWeightSuffix(weight)
    }
    fn companion(
        &mut self,
        weight: &'a str,
        prefix: &'a str,
        suffix: CompanionSuffix,
        component: &'static str,
    ) -> Result<Self::Parameter, Self::Error> {
        let mut row = StaticParameterDeclaration::trainable(ParameterNameView::joined(
            prefix,
            suffix.as_str(),
        ))?;
        row.group = Some(ParameterNameView::new(weight));
        if row.name == ParameterNameView::new(weight) {
            return Err(StaticDeclarationError::CompanionIdentity { component, weight });
        }
        Ok(row)
    }
    fn unscaled(&mut self, format: LinearFormat) -> Result<Self::Output, Self::Error> {
        finish_borrowed(format, None, None)
    }
    fn scaled(
        &mut self,
        format: LinearFormat,
        mut scale: Self::Parameter,
    ) -> Result<Self::Output, Self::Error> {
        scale.linear_companion = Some(LinearCompanionRole::Scale);
        scale.linear_companion_of = None;
        finish_borrowed(format, Some(scale), None)
    }
    fn affine(
        &mut self,
        format: LinearFormat,
        mut scale: Self::Parameter,
        mut bias: Self::Parameter,
    ) -> Result<Self::Output, Self::Error> {
        scale.linear_companion = Some(LinearCompanionRole::Scale);
        scale.linear_companion_of = None;
        bias.linear_companion = Some(LinearCompanionRole::AffineBias);
        bias.linear_companion_of = None;
        finish_borrowed(format, Some(scale), Some(bias))
    }
}
fn finish_borrowed<'a>(
    format: LinearFormat,
    scale: Option<StaticParameterDeclaration<'a>>,
    affine_bias: Option<StaticParameterDeclaration<'a>>,
) -> Result<StaticLinearFormatDeclaration<'a>, StaticDeclarationError<'a>> {
    let declaration = StaticLinearFormatDeclaration {
        format,
        scale,
        affine_bias,
        row_layout: LinearRowLayout::Contiguous,
    };
    declaration.validation_view().validate_fixed()?;
    Ok(declaration)
}
