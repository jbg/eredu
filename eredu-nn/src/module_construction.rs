//! Checked destinations for the shared owned module-specification constructors.
use super::*;
use workspace::WorkspaceContext;

#[derive(Clone, Copy)]
struct Destination<'a>(Option<&'a WorkspaceContext>);
impl Destination<'_> {
    fn error(self, args: std::fmt::Arguments<'_>) -> Error {
        match self.0 {
            Some(context) => context.metadata_error(args),
            None => Error::backend(args),
        }
    }
    fn vector<T>(self, count: usize) -> Result<Vec<T>, Error> {
        match self.0 {
            Some(context) => context.metadata_vec(count),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    fn controls<T>(self) -> Result<(), Error> {
        if let Some(context) = self.0 {
            context.charge_metadata(
                size_of::<T>() + size_of::<Result<T, Error>>() + size_of::<Self>(),
            )?;
        }
        Ok(())
    }
}

impl LinearFormatSpec {
    pub(crate) fn from_parts_with(
        format: LinearFormat,
        mut scale: Option<ParameterSpec>,
        mut affine_bias: Option<ParameterSpec>,
        context: Option<&WorkspaceContext>,
    ) -> Result<Self, Error> {
        let destination = Destination(context);
        destination.controls::<Self>()?;
        if let Some(scale) = &mut scale {
            scale.linear_companion = Some(LinearCompanionRole::Scale);
            scale.linear_companion_of = None;
        }
        if let Some(bias) = &mut affine_bias {
            bias.linear_companion = Some(LinearCompanionRole::AffineBias);
            bias.linear_companion_of = None;
        }
        let spec = Self {
            format,
            scale,
            affine_bias,
            row_layout: LinearRowLayout::Contiguous,
        };
        spec.validate_fixed()
            .map_err(|cause| destination.error(format_args!("{cause}")))?;
        Ok(spec)
    }
    /// Moves already admitted companion declarations through the ordinary
    /// role assignment and format validator. The caller retains their custody.
    pub fn from_parts_with_metadata(
        format: LinearFormat,
        scale: Option<ParameterSpec>,
        affine_bias: Option<ParameterSpec>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::from_parts_with(format, scale, affine_bias, Some(context))
    }
}

impl FusedProjectionSegment {
    pub(crate) fn from_owned_with(
        name: String,
        width: i32,
        context: Option<&WorkspaceContext>,
    ) -> Result<Self, Error> {
        let destination = Destination(context);
        destination.controls::<Self>()?;
        if name.trim().is_empty() || width <= 0 {
            return Err(destination.error(format_args!("fused projection segments require a name and positive width, got name={name:?} width={width}")));
        }
        Ok(Self { name, width })
    }
    /// Constructs the same named segment, admitting the exact name first.
    pub fn new_with_metadata(
        name: std::fmt::Arguments<'_>,
        width: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::from_owned_with(context.metadata_string(name)?, width, Some(context))
    }
}
impl FusedProjectionLayout {
    pub(crate) fn from_owned_with(
        segments: Vec<FusedProjectionSegment>,
        context: Option<&WorkspaceContext>,
    ) -> Result<Self, Error> {
        let destination = Destination(context);
        destination.controls::<Self>()?;
        if segments.is_empty() {
            return Err(destination.error(format_args!(
                "fused projection layout must contain at least one segment"
            )));
        }
        // Names remain in their original owners; only indices need scratch.
        let mut order = destination.vector::<usize>(segments.len())?;
        order.extend(0..segments.len());
        order.sort_unstable_by(|a, b| segments[*a].name.cmp(&segments[*b].name).then(a.cmp(b)));
        let duplicate = order
            .windows(2)
            .filter(|pair| segments[pair[0]].name == segments[pair[1]].name)
            .map(|pair| pair[1])
            .min();
        let mut output_width = 0i32;
        for (index, segment) in segments.iter().enumerate() {
            if duplicate == Some(index) {
                return Err(destination.error(format_args!(
                    "fused projection segment {:?} is duplicated",
                    segment.name
                )));
            }
            output_width = output_width.checked_add(segment.width).ok_or_else(|| {
                destination.error(format_args!(
                    "fused projection output width overflowed signed 32-bit geometry"
                ))
            })?;
        }
        Ok(Self {
            segments,
            output_width,
        })
    }
    /// Moves an already admitted ordered segment vector into the shared validator.
    pub fn from_owned_with_metadata(
        segments: Vec<FusedProjectionSegment>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::from_owned_with(segments, Some(context))
    }
}

struct MissingCapabilities(NeuralOperatorCapabilities, NeuralOperatorCapabilities);
impl std::fmt::Display for MissingCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, name) in self.0.iter_missing_capability_names(self.1).enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            f.write_str(name)?;
        }
        Ok(())
    }
}
impl NeuralOperatorCapabilities {
    /// Applies the ordinary exact support predicate and ordered diagnostic,
    /// using the caller's owning diagnostic destination.
    pub fn require_with<E>(
        self,
        architecture: &'static str,
        required: Self,
        invalid: impl FnOnce(std::fmt::Arguments<'_>) -> E,
    ) -> Result<(), E> {
        if self.contains(required) {
            return Ok(());
        }
        Err(invalid(format_args!(
            "{architecture} requires unsupported backend operators: {}",
            MissingCapabilities(self, required)
        )))
    }
}

// Only these closed neural declarations can use the shared metadata clone census.
macro_rules! prepared_spec_copy {
    ($($ty:ty),+ $(,)?) => { $(impl $ty {
        /// Copies this exact immutable declaration after admitting all nested
        /// names, policy buffers and controls. The caller retains Context custody.
        pub fn clone_with_metadata(&self, context: &WorkspaceContext) -> Result<Self, Error> {
            context.clone_metadata(self)
        }
    })+ };
}
prepared_spec_copy!(
    LinearSpec,
    NormalizationConstructionSpec,
    TopKGroupSelectorSpec,
    GroupedGatedProductSpec,
    GroupedRelu2Spec,
    LowRankProjectionSpec
);

prepared_spec_copy!(ParameterSpec, CausalDepthwiseConvolutionSpec);

prepared_spec_copy!(EmbeddingSpec, HyperConnectionSpec, HyperHeadSpec);
