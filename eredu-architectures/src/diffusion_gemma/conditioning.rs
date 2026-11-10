//! Decoder-only feedback slots, separate from shared-stack invocation ownership.
use crate::gemma4::ModelArgs;
use eredu_nn::{
    DeclareLogicalParameters, Error, LogicalParameterDeclaration, NormalizationConstructionSpec,
    ParameterSpec,
};

/// Learned feedback normalization and a GELU-tanh gated product before residual normalization.
#[derive(Debug, Clone)]
pub struct Declaration {
    /// Normalization of feedback; evaluated for both supplied and zero feedback.
    pub pre_norm: NormalizationConstructionSpec,
    /// Complete GELU-tanh gated product, including affine slots and connectivity.
    pub product: eredu_nn::SplitGatedProductSpec,
    /// Independent weightless RMS equation after adding the feedback signal.
    pub output_epsilon: f32,
}
impl Declaration {
    /// Parameter owner, independent of context and refinement state identities.
    pub const ROOT: &'static str = "model.decoder.self_conditioning";
    const PRODUCT_POLICY: eredu_nn::GatedProductPolicy =
        eredu_nn::GatedProductPolicy::ordinary_gelu_approximate();
    /// Declares the released global feedback equation without allocating tensors.
    pub(super) fn new(args: &ModelArgs) -> Result<Self, Error> {
        let width = args
            .layer_policy(0)
            .ok_or_else(|| Error::backend("empty diffusion stack"))?
            .intermediate_size
            .get() as i32;
        let linear = |field, input, output| {
            crate::gemma4::text::specs::linear_spec(
                args,
                &format!("{}.{field}.weight", Self::ROOT),
                input,
                output,
                false,
            )
        };
        let declaration = Self {
            pre_norm: args.normalization(
                args.hidden_size,
                ParameterSpec::trainable(format!("{}.pre_norm.weight", Self::ROOT))
                    .map_err(Error::backend)?,
            ),
            product: eredu_nn::SplitGatedProductSpec::new(
                linear("gate_proj", args.hidden_size, width)?,
                linear("up_proj", args.hidden_size, width)?,
                linear("down_proj", width, args.hidden_size)?,
                Self::PRODUCT_POLICY,
            )?,
            output_epsilon: args.rms_norm_eps,
        };
        declaration.validate_for(args)?;
        Ok(declaration)
    }
    /// Retains parameter identities and arithmetic at a placement-selected intermediate width.
    pub fn with_intermediate(mut self, width: i32) -> Result<Self, Error> {
        if width <= 0 || width > self.product.gate.output {
            return Err(Error::backend(
                "diffusion feedback width exceeds its retained source",
            ));
        }
        self.product.gate.output = width;
        self.product.up.output = width;
        self.product.down.input = width;
        self.validate()?;
        Ok(self)
    }
    /// Rejects incompatible geometry, equations and unresolved or conflicting identities.
    pub fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        self.parameter_groups()?;
        Ok(())
    }
    /// Projects physical encodings while preserving source geometry and scalar equations.
    pub(super) fn project_formats(&self, args: &ModelArgs) -> Result<Self, Error> {
        let mut exact = self.clone();
        for spec in [
            &mut exact.product.gate,
            &mut exact.product.up,
            &mut exact.product.down,
        ] {
            let mut format = crate::linear_format::standard_linear_format(
                spec.weight.id.as_str(),
                args.linear_format_for(spec.weight.id.as_str()),
            )?
            .with_arithmetic(spec.format.arithmetic())?;
            if spec.format.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            spec.format = format;
        }
        exact.validate_for(args)?;
        Ok(exact)
    }
    /// Resolves named storage extensions against the declared feedback matrix identities.
    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), Error> {
        let width = args
            .layer_policy(0)
            .ok_or_else(|| Error::backend("empty diffusion stack"))?
            .intermediate_size
            .get() as i32;
        if self.product.gate.input != args.hidden_size
            || self.product.gate.output != width
            || self.output_epsilon != args.rms_norm_eps
            || self.pre_norm.epsilon != args.rms_norm_eps
        {
            return Err(Error::backend(
                "diffusion feedback source policy changes retained geometry or normalization",
            ));
        }
        self.validate()?;
        let parameters = self.logical_parameters()?;
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|f| f.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            if name.starts_with(&format!("{}.", Self::ROOT))
                && !parameters
                    .iter()
                    .any(|p| p.parameter.id.as_str() == name && p.linear_format.is_some())
            {
                return Err(Error::backend(format!(
                    "diffusion feedback format references undeclared projection {name}"
                )));
            }
        }
        Ok(())
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        self.pre_norm.validate()?;
        if self.pre_norm.dimensions != self.product.gate.input
            || self.product.down.output != self.product.gate.input
            || self.product.policy != Self::PRODUCT_POLICY
            || !self.output_epsilon.is_finite()
            || self.output_epsilon <= 0.0
        {
            return Err(Error::backend(
                "incompatible diffusion feedback geometry, equation or normalization",
            ));
        }
        Ok(())
    }

    pub(crate) fn parameter_groups(&self) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, Error> {
        use eredu_runtime::{logical_parameter_group, MemberSharding, ParameterRole};
        let norm = logical_parameter_group(
            format!("{}.norm", Self::ROOT),
            ParameterRole::Replicated,
            None,
            &self.pre_norm,
            |_| MemberSharding::Replicated,
        )
        .map_err(Error::backend_source)?;
        let bank = logical_parameter_group(
            format!("{}.intermediate", Self::ROOT),
            ParameterRole::FeedForwardIntermediate,
            Some(self.product.gate.output as usize),
            &self.product,
            |p| {
                if p.parameter.id == self.product.down.weight.id {
                    MemberSharding::Partitioned { axis: 1 }
                } else if self
                    .product
                    .down
                    .bias
                    .as_ref()
                    .is_some_and(|bias| bias.id == p.parameter.id)
                {
                    MemberSharding::Replicated
                } else {
                    MemberSharding::Partitioned { axis: 0 }
                }
            },
        )
        .map_err(Error::backend_source)?;
        Ok(vec![norm, bank])
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut p = self.pre_norm.logical_parameters()?;
        p.extend(self.product.logical_parameters()?);
        Ok(p)
    }
}
