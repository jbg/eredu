//! The caller's metadata destination for the shared standard module worker.
use super::static_construction::format::{CompanionSuffix, FormatStorage};
use super::*;
use eredu_nn::workspace::WorkspaceContext;

#[derive(Clone, Copy)]
pub(crate) struct ModuleMetadata<'a>(Option<&'a WorkspaceContext>);
impl<'a> ModuleMetadata<'a> {
    pub(crate) fn funded(context: &'a WorkspaceContext) -> Self {
        Self(Some(context))
    }
    pub(crate) fn ordinary() -> Self {
        Self(None)
    }
    pub(crate) fn new<B: NeuralBackend>(context: &'a <B::Tensor as Tensor>::Context) -> Self {
        Self(B::construction_metadata(context).filter(|context| context.uses_checked_metadata()))
    }
    pub(crate) fn vector<T>(self, capacity: usize) -> Result<Vec<T>, Error> {
        self.controls::<Vec<T>>()?;
        match self.0 {
            Some(context) => context.metadata_vec(capacity),
            None => Ok(Vec::with_capacity(capacity)),
        }
    }
    pub(crate) fn text(self, args: std::fmt::Arguments<'_>) -> Result<String, Error> {
        match self.0 {
            Some(context) => context.metadata_string(args),
            None => Ok(args.to_string()),
        }
    }
    pub(crate) fn error(self, args: std::fmt::Arguments<'_>) -> Error {
        match self.0 {
            Some(context) => context.metadata_error(args),
            None => Error::backend(args),
        }
    }
    pub(crate) fn controls<T>(self) -> Result<(), Error> {
        if let Some(context) = self.0 {
            context.charge_metadata(
                size_of::<T>() + size_of::<Result<T, Error>>() + size_of::<Self>(),
            )?;
        }
        Ok(())
    }
    pub(crate) fn borrowed_controls<T>(self, value: &T) -> Result<(), Error> {
        if let Some(context) = self.0 {
            context.charge_metadata(std::mem::size_of_val(value))?;
        }
        Ok(())
    }
    pub(crate) fn require<B: NeuralBackend>(
        self,
        name: &'static str,
        capabilities: eredu_nn::NeuralOperatorCapabilities,
    ) -> Result<(), Error> {
        self.controls::<(
            eredu_nn::NeuralOperatorCapabilities,
            eredu_nn::NeuralOperatorCapabilities,
        )>()?;
        match self.0 {
            Some(_) => {
                B::OPERATOR_CAPABILITIES.require_with(name, capabilities, |args| self.error(args))
            }
            None => B::require_operator_capabilities(name, capabilities),
        }
    }
    pub(super) fn parameter(
        self,
        config: &impl Config,
        name: String,
    ) -> Result<ParameterSpec, Error> {
        let Some(context) = self.0 else {
            return parameter_spec(config, name).map_err(Error::backend);
        };
        self.controls::<ParameterSpec>()?;
        let alias = config
            .parameter_alias_with_metadata(&name, context)?
            .map(eredu_nn::ParameterId::new)
            .transpose()
            .map_err(|cause| self.error(format_args!("{cause}")))?;
        let mut parameter =
            ParameterSpec::trainable(name).map_err(|cause| self.error(format_args!("{cause}")))?;
        parameter.alias_of = alias;
        Ok(parameter)
    }
    pub(crate) fn plain_parameter(self, name: &str) -> Result<ParameterSpec, Error> {
        self.named_parameter(format_args!("{name}"))
    }
    pub(crate) fn named_parameter(
        self,
        name: std::fmt::Arguments<'_>,
    ) -> Result<ParameterSpec, Error> {
        self.controls::<Result<ParameterSpec, eredu_nn::ParameterTopologyError>>()?;
        let name = self.text(name)?;
        ParameterSpec::trainable(name).map_err(|cause| self.error(format_args!("{cause}")))
    }
    pub(super) fn construction_error(
        self,
        cause: static_construction::StaticConstructionError,
    ) -> Error {
        match self.0 {
            Some(_) => self.error(format_args!("{cause}")),
            None => cause.into(),
        }
    }
    pub(super) fn weight_quantization(
        self,
        config: &impl Config,
        name: &str,
    ) -> Result<Option<WeightQuantization>, Error> {
        match self.0 {
            Some(context) => config.weight_quantization_with_metadata(name, context),
            None => Ok(config.weight_quantization(name)),
        }
    }
    pub(super) fn linear_format(
        self,
        config: &impl Config,
        name: &str,
    ) -> Result<LinearFormat, Error> {
        match self.0 {
            Some(context) => config.linear_format_with_metadata(name, context),
            None => Ok(config.linear_format(name)),
        }
    }
    pub(crate) fn format(
        mut self,
        name: &str,
        format: LinearFormat,
    ) -> Result<eredu_nn::LinearFormatSpec, Error> {
        self.controls::<(eredu_nn::LinearFormatSpec, Result<&str, Error>)>()?;
        match self.0 {
            Some(_) => {
                static_construction::format::standard_linear_format_with(name, format, &mut self)
            }
            None => crate::linear_format::standard_linear_format(name, format),
        }
    }
    pub(super) fn normalization_name(
        self,
        config: &impl Config,
        layer: usize,
        kind: parameter_metadata::NormalizationName,
    ) -> Result<Option<String>, Error> {
        use parameter_metadata::NormalizationName::*;
        match self.0 {
            Some(context) => match kind {
                Block => config.block_output_normalization_with_metadata(layer, context),
                Attention => config.attention_output_normalization_with_metadata(layer, context),
                FeedForward => {
                    config.feed_forward_output_normalization_with_metadata(layer, context)
                }
            },
            None => Ok(match kind {
                Block => config.block_output_normalization(layer),
                Attention => config.attention_output_normalization(layer),
                FeedForward => config.feed_forward_output_normalization(layer),
            }),
        }
    }
    pub(super) fn with_groups(
        self,
        mut spec: NormalizationConstructionSpec,
        groups: i32,
    ) -> Result<NormalizationConstructionSpec, Error> {
        match self.0 {
            Some(_) => {
                self.controls::<NormalizationConstructionSpec>()?;
                spec.groups = Some(groups);
                spec.validate_fixed()
                    .map_err(|cause| self.error(format_args!("{cause}")))?;
                Ok(spec)
            }
            None => spec.with_groups(groups),
        }
    }
    pub(super) fn segment(self, name: &str, width: i32) -> Result<FusedProjectionSegment, Error> {
        match self.0 {
            Some(context) => {
                FusedProjectionSegment::new_with_metadata(format_args!("{name}"), width, context)
            }
            None => FusedProjectionSegment::new(name, width),
        }
    }
    pub(super) fn fused<const N: usize>(
        self,
        segments: [FusedProjectionSegment; N],
    ) -> Result<FusedProjectionLayout, Error> {
        match self.0 {
            Some(context) => {
                let mut owned = context.metadata_vec(N)?;
                owned.extend(segments);
                FusedProjectionLayout::from_owned_with_metadata(owned, context)
            }
            None => FusedProjectionLayout::new(segments),
        }
    }
    pub(super) fn rotary(self, config: &impl Config, dimensions: i32) -> Result<RotarySpec, Error> {
        self.controls::<RotarySpec>()?;
        match self.0 {
            Some(context) => config.rotary_spec_with_metadata(dimensions, context),
            None => Ok(config.rotary_spec(dimensions)),
        }
    }
}
impl<'a> FormatStorage<'a> for ModuleMetadata<'_> {
    type Parameter = ParameterSpec;
    type Output = eredu_nn::LinearFormatSpec;
    type Error = Error;
    fn missing_suffix(&mut self, weight: &str) -> Error {
        self.error(format_args!(
            "encoded ordinary linear parameter {weight:?} must end in .weight"
        ))
    }
    fn companion(
        &mut self,
        weight: &str,
        prefix: &str,
        suffix: CompanionSuffix,
        component: &'static str,
    ) -> Result<ParameterSpec, Error> {
        self.controls::<ParameterSpec>()?;
        let name = self.text(format_args!("{prefix}{}", suffix.as_str()))?;
        let mut companion =
            ParameterSpec::trainable(name).map_err(|cause| self.error(format_args!("{cause}")))?;
        companion.group = Some(self.text(format_args!("{weight}"))?);
        if companion.id.as_str() == weight {
            return Err(self.error(format_args!(
                "linear {component} companion reuses weight identity {weight:?}"
            )));
        }
        Ok(companion)
    }
    fn unscaled(&mut self, format: LinearFormat) -> Result<Self::Output, Error> {
        self.finish_format(format, None, None)
    }
    fn scaled(
        &mut self,
        format: LinearFormat,
        scale: ParameterSpec,
    ) -> Result<Self::Output, Error> {
        self.finish_format(format, Some(scale), None)
    }
    fn affine(
        &mut self,
        format: LinearFormat,
        scale: ParameterSpec,
        bias: ParameterSpec,
    ) -> Result<Self::Output, Error> {
        self.finish_format(format, Some(scale), Some(bias))
    }
}
impl ModuleMetadata<'_> {
    pub(crate) fn declared_format(
        self,
        format: LinearFormat,
        scale: Option<ParameterSpec>,
        bias: Option<ParameterSpec>,
    ) -> Result<eredu_nn::LinearFormatSpec, Error> {
        match self.0 {
            Some(context) => {
                eredu_nn::LinearFormatSpec::from_parts_with_metadata(format, scale, bias, context)
            }
            None => match (scale, bias) {
                (None, None) => eredu_nn::LinearFormatSpec::unscaled(format),
                (Some(scale), None) => eredu_nn::LinearFormatSpec::scaled(format, scale),
                (Some(scale), Some(bias)) => {
                    eredu_nn::LinearFormatSpec::affine(format, scale, bias)
                }
                (None, Some(_)) => {
                    Err(self.error(format_args!("linear affine bias lacks its scale")))
                }
            },
        }
    }
    fn finish_format(
        self,
        format: LinearFormat,
        scale: Option<ParameterSpec>,
        bias: Option<ParameterSpec>,
    ) -> Result<eredu_nn::LinearFormatSpec, Error> {
        eredu_nn::LinearFormatSpec::from_parts_with_metadata(
            format,
            scale,
            bias,
            self.0.expect("checked format worker"),
        )
    }
}
