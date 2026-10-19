//! Parameterized Muse normalization and residual equations.

use eredu_nn::{
    Error, NeuralBackend, NormalizationOperator, Parameter, ParameterSpec, Parameterized, Tensor,
    TensorElementType,
};

use super::WeightConvention;

/// Meaning of an independently stored normalization gain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsGain {
    /// Stored values are offsets from one.
    Centered,
    /// Stored values are the effective gain.
    Direct,
}

/// Ordering of learned scaling and weightless RMS normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormEquation {
    /// Normalize first, then multiply by the checkpoint's effective scale.
    OutputScaled(WeightConvention),
    /// Normalize and apply the effective gain in FP32, then round once.
    /// Centered gains are formed in the stored parameter dtype first.
    OutputScaledFp32(WeightConvention),
    /// Scale in FP32, round to the input dtype, then normalize in FP32.
    InputScaled(WeightConvention),
    /// Round input to the parameter's compute dtype, then normalize and apply
    /// the centered gain entirely in FP32. Return the parameter's compute dtype.
    OutputScaledComputeFp32(RmsGain),
    /// Form the effective gain and scaled input in FP32 without an intermediate
    /// activation rounding boundary; normalize and return the input dtype.
    InputScaledFp32(RmsGain),
}

/// RMS normalization with an authoritative, replaceable scale parameter.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LearnedRmsNorm<B: NeuralBackend> {
    /// Stored scale, centered or direct according to the equation.
    pub weight: Parameter<B::Tensor>,
    #[parameter(skip)]
    epsilon: f32,
    #[parameter(skip)]
    equation: RmsNormEquation,
}

impl<B: NeuralBackend> LearnedRmsNorm<B> {
    /// Constructs an unloaded normalization parameter.
    pub fn new(
        dimensions: i32,
        epsilon: f32,
        equation: RmsNormEquation,
        name: impl Into<String>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if dimensions <= 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::backend(
                "Muse RMS normalization requires positive width and epsilon",
            ));
        }
        if !matches!(equation, RmsNormEquation::OutputScaled(_)) {
            B::require_operator_capabilities(
                "ordered RMS normalization",
                eredu_nn::NeuralOperatorCapabilities::CASCADED_RMS
                    .union(eredu_nn::NeuralOperatorCapabilities::CAST_FLOAT),
            )?;
        }
        Ok(Self {
            weight: Parameter::unloaded(
                ParameterSpec::trainable(name.into()).map_err(Error::backend)?,
                &[dimensions],
                context,
            )?,
            epsilon,
            equation,
        })
    }

    /// Evaluates the selected equation using the current parameter slot.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gain = match self.equation {
            RmsNormEquation::OutputScaled(convention)
            | RmsNormEquation::OutputScaledFp32(convention)
            | RmsNormEquation::InputScaled(convention) => match convention {
                WeightConvention::HuggingFace => RmsGain::Centered,
                WeightConvention::Gguf => RmsGain::Direct,
            },
            RmsNormEquation::OutputScaledComputeFp32(gain)
            | RmsNormEquation::InputScaledFp32(gain) => gain,
        };
        match self.equation {
            RmsNormEquation::OutputScaled(_) => {
                // Retain the backend's established output-normalization arithmetic.
                // Never cache derived gains across parameter replacement or unloading.
                let effective = if gain == RmsGain::Centered {
                    Some(self.weight.as_ref().add(
                        &B::Tensor::full_f32(1.0, self.weight.as_ref().shape(), context)?,
                        context,
                    )?)
                } else {
                    None
                };
                B::rms_norm_with_weight(
                    input,
                    effective.as_ref().unwrap_or_else(|| self.weight.as_ref()),
                    self.epsilon,
                    context,
                )
            }
            RmsNormEquation::InputScaled(_)
            | RmsNormEquation::OutputScaledFp32(_)
            | RmsNormEquation::OutputScaledComputeFp32(_)
            | RmsNormEquation::InputScaledFp32(_) => {
                let weight_type = floating_type(self.weight.as_ref())?;
                let input = match self.equation {
                    RmsNormEquation::OutputScaledComputeFp32(_) => {
                        input.cast_float(weight_type, context)?
                    }
                    _ => input.clone(),
                };
                let dtype = floating_type(&input)?;
                let weight = self
                    .weight
                    .as_ref()
                    .cast_float(TensorElementType::F32, context)?;
                let scale = if gain == RmsGain::Centered {
                    let centered =
                        weight.add(&B::Tensor::full_f32(1.0, weight.shape(), context)?, context)?;
                    match self.equation {
                        RmsNormEquation::OutputScaledComputeFp32(_)
                        | RmsNormEquation::InputScaledFp32(_) => centered,
                        _ => centered
                            .cast_float(weight_type, context)?
                            .cast_float(TensorElementType::F32, context)?,
                    }
                } else {
                    weight
                };
                let input = input.cast_float(TensorElementType::F32, context)?;
                let output = match self.equation {
                    RmsNormEquation::InputScaled(_) => {
                        let scaled = input
                            .multiply(&scale, context)?
                            .cast_float(dtype, context)?
                            .cast_float(TensorElementType::F32, context)?;
                        weightless_rms_norm::<B>(&scaled, self.epsilon, context)?
                    }
                    RmsNormEquation::InputScaledFp32(_) => weightless_rms_norm::<B>(
                        &input.multiply(&scale, context)?,
                        self.epsilon,
                        context,
                    )?,
                    RmsNormEquation::OutputScaledFp32(_)
                    | RmsNormEquation::OutputScaledComputeFp32(_) => {
                        weightless_rms_norm::<B>(&input, self.epsilon, context)?
                            .multiply(&scale, context)?
                    }
                    RmsNormEquation::OutputScaled(_) => unreachable!(),
                };
                output.cast_float(dtype, context)
            }
        }
    }
}

/// Normalizes in FP32 and rounds to the input dtype without a learned scale.
pub fn weightless_rms_norm<B: NeuralBackend>(
    input: &B::Tensor,
    epsilon: f32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let dtype = floating_type(input)?;
    B::require_operator_capabilities(
        "ordered RMS normalization",
        eredu_nn::NeuralOperatorCapabilities::CASCADED_RMS
            .union(eredu_nn::NeuralOperatorCapabilities::CAST_FLOAT),
    )?;
    let dimensions = input
        .shape()
        .last()
        .copied()
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            Error::backend("ordered RMS normalization requires a positive final axis")
        })?;
    let mut norm = B::normalization(
        eredu_nn::NormalizationConstructionSpec {
            dimensions,
            epsilon,
            groups: Some(1),
            reduction: eredu_nn::NormalizationReduction::Cascaded,
            scale: eredu_nn::NormalizationScale::Unit,
        },
        context,
    )?;
    norm.forward(&input.cast_float(TensorElementType::F32, context)?, context)?
        .cast_float(dtype, context)
}

/// Normalizes each final-axis head and applies a sigmoid gate with one final store.
pub fn weightless_rms_sigmoid_gate<B: NeuralBackend>(
    input: &B::Tensor,
    gate: &B::Tensor,
    epsilon: f32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    if input.shape() != gate.shape() {
        return Err(Error::backend("normalized head gate geometry mismatch"));
    }
    let dtype = floating_type(input)?;
    let normalized = weightless_rms_norm::<B>(
        &input.cast_float(TensorElementType::F32, context)?,
        epsilon,
        context,
    )?;
    normalized
        .multiply(
            &B::sigmoid(gate.cast_float(TensorElementType::F32, context)?, context)?,
            context,
        )?
        .cast_float(dtype, context)
}

/// An architecture-declared branch norm with an exact parameter topology.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum BranchNorm<B: NeuralBackend> {
    /// A learned scale with an explicit arithmetic equation.
    Learned(LearnedRmsNorm<B>),
    /// Weightless FP32 normalization; contains no checkpoint parameter.
    Weightless {
        /// Positive normalization epsilon.
        #[parameter(skip)]
        epsilon: f32,
    },
}

impl<B: NeuralBackend> BranchNorm<B> {
    /// Returns the learned normalization when this branch declares a scale.
    pub fn learned(&self) -> Option<&LearnedRmsNorm<B>> {
        match self {
            Self::Learned(norm) => Some(norm),
            Self::Weightless { .. } => None,
        }
    }

    /// Returns the replaceable learned normalization when present.
    pub fn learned_mut(&mut self) -> Option<&mut LearnedRmsNorm<B>> {
        match self {
            Self::Learned(norm) => Some(norm),
            Self::Weightless { .. } => None,
        }
    }

    /// Evaluates the declared normalization.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Learned(norm) => norm.forward(input, context),
            Self::Weightless { epsilon } => weightless_rms_norm::<B>(input, *epsilon, context),
        }
    }
}

/// Residual arithmetic and its associated checkpoint parameters.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum ResidualCombination<B: NeuralBackend> {
    /// Ordinary addition, with no learned parameters.
    Add,
    /// Learned temperature-scaled variance-preserving gate.
    Learned(LearnedResidualCombination<B>),
    /// Artifact-preserved coefficients without inverse reconstruction of gate logits.
    Stored(StoredResidualCombination<B>),
}

impl<B: NeuralBackend> ResidualCombination<B> {
    /// Combines the branch and residual using the declared arithmetic.
    pub fn forward(
        &mut self,
        residual: &B::Tensor,
        branch: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Add => residual.add(branch, context),
            Self::Learned(gate) => gate.forward(residual, branch, context),
            Self::Stored(coefficients) => coefficients.forward(residual, branch, context),
        }
    }
}

/// Stream and branch multipliers retained exactly from a converted checkpoint.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StoredResidualCombination<B: NeuralBackend> {
    /// Multiplier of the incoming residual stream.
    pub alpha: Parameter<B::Tensor>,
    /// Multiplier of the transformed branch.
    pub beta: Parameter<B::Tensor>,
}

impl<B: NeuralBackend> StoredResidualCombination<B> {
    /// Constructs independent, replaceable coefficient parameters.
    pub fn new(
        dimensions: i32,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if dimensions <= 0 {
            return Err(Error::backend("Muse residual width must be positive"));
        }
        B::require_operator_capabilities(
            "FP32 residual coefficients",
            eredu_nn::NeuralOperatorCapabilities::CAST_FLOAT,
        )?;
        let parameter = |suffix| {
            Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.{suffix}")).map_err(Error::backend)?,
                &[dimensions],
                context,
            )
        };
        Ok(Self {
            alpha: parameter("alpha")?,
            beta: parameter("beta")?,
        })
    }

    /// Applies both coefficients in FP32 and returns the residual dtype.
    pub fn forward(
        &mut self,
        residual: &B::Tensor,
        branch: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        combine_residual::<B>(
            residual,
            branch,
            self.alpha.as_ref(),
            self.beta.as_ref(),
            context,
        )
    }
}

fn combine_residual<B: NeuralBackend>(
    residual: &B::Tensor,
    branch: &B::Tensor,
    alpha: &B::Tensor,
    beta: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    if residual.shape() != branch.shape()
        || alpha.shape() != beta.shape()
        || alpha.shape().len() != 1
        || residual.shape().last() != alpha.shape().last()
    {
        return Err(Error::backend(
            "Muse residual combination has incompatible geometry",
        ));
    }
    residual
        .cast_float(TensorElementType::F32, context)?
        .multiply(&alpha.cast_float(TensorElementType::F32, context)?, context)?
        .add(
            &branch
                .cast_float(TensorElementType::F32, context)?
                .multiply(&beta.cast_float(TensorElementType::F32, context)?, context)?,
            context,
        )?
        .cast_float(floating_type(residual)?, context)
}

fn floating_type<T: Tensor>(input: &T) -> Result<TensorElementType, Error> {
    match input.element_type() {
        Some(
            dtype @ (TensorElementType::F16
            | TensorElementType::Bf16
            | TensorElementType::F32
            | TensorElementType::F64),
        ) => Ok(dtype),
        _ => Err(Error::backend(
            "Muse equations require a known floating input dtype",
        )),
    }
}

/// Learned elementwise residual combination evaluated entirely in FP32.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LearnedResidualCombination<B: NeuralBackend> {
    /// Unscaled gate logits.
    pub gate: Parameter<B::Tensor>,
    #[parameter(skip)]
    temperature: f32,
    #[parameter(skip)]
    minimum_variance: f32,
}

impl<B: NeuralBackend> LearnedResidualCombination<B> {
    /// Constructs an unloaded residual gate with explicit temperature and clamp.
    pub fn new(
        dimensions: i32,
        temperature: f32,
        minimum_variance: f32,
        name: impl Into<String>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if dimensions <= 0
            || !temperature.is_finite()
            || temperature <= 0.0
            || !minimum_variance.is_finite()
            || minimum_variance <= 0.0
        {
            return Err(Error::backend(
                "Muse residual combination requires positive width, temperature, and clamp",
            ));
        }
        B::require_operator_capabilities(
            "learned residual combination",
            eredu_nn::NeuralOperatorCapabilities::CAST_FLOAT
                .union(eredu_nn::NeuralOperatorCapabilities::SQRT)
                .union(eredu_nn::NeuralOperatorCapabilities::SIGMOID)
                .union(eredu_nn::NeuralOperatorCapabilities::FULL_F32),
        )?;
        Ok(Self {
            gate: Parameter::unloaded(
                ParameterSpec::trainable(name.into()).map_err(Error::backend)?,
                &[dimensions],
                context,
            )?,
            temperature,
            minimum_variance,
        })
    }

    /// Combines a branch with the residual and returns the residual dtype.
    pub fn forward(
        &mut self,
        residual: &B::Tensor,
        branch: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if residual.shape() != branch.shape()
            || residual.shape().last() != self.gate.as_ref().shape().last()
        {
            return Err(Error::backend(
                "Muse residual combination has incompatible geometry",
            ));
        }
        let logits = self
            .gate
            .as_ref()
            .cast_float(TensorElementType::F32, context)?
            .divide(
                &B::Tensor::full_f32(self.temperature, &[], context)?,
                context,
            )?;
        let beta = B::sigmoid(logits.clone(), context)?;
        let alpha = B::sigmoid(logits.multiply_scalar(-1.0, context)?, context)?
            .multiply(
                &beta.add(&B::Tensor::full_f32(1.0, beta.shape(), context)?, context)?,
                context,
            )?
            .maximum_scalar(self.minimum_variance, context)?
            .sqrt(context)?;
        combine_residual::<B>(residual, branch, &alpha, &beta, context)
    }
}
