//! Typed normalization and residual construction slots, independent of native tensors.

use super::{RmsGain, RmsNormEquation};
use eredu_nn::{
    DeclareLogicalParameters, Error, LogicalParameterDeclaration, LogicalParameterRole,
    NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
};

/// Learned RMS geometry and the family-owned ordering of scaling and reduction.
#[derive(Debug, Clone)]
pub struct LearnedRmsNormSpec {
    pub(super) normalization: NormalizationConstructionSpec,
    pub(super) equation: RmsNormEquation,
}
impl LearnedRmsNormSpec {
    /// Declares the exact stored gain and arithmetic before backend allocation.
    pub fn new(
        dimensions: i32,
        epsilon: f32,
        equation: RmsNormEquation,
        weight: ParameterSpec,
    ) -> Result<Self, Error> {
        let normalization = NormalizationConstructionSpec {
            dimensions,
            epsilon,
            groups: if matches!(equation, RmsNormEquation::OutputScaled(_)) {
                None
            } else {
                Some(1)
            },
            reduction: if matches!(equation, RmsNormEquation::OutputScaled(_)) {
                eredu_nn::NormalizationReduction::Native
            } else {
                eredu_nn::NormalizationReduction::Cascaded
            },
            scale: if equation.gain() == RmsGain::Centered {
                NormalizationScale::LearnedOffset {
                    weight,
                    offset: 1.0,
                }
            } else {
                NormalizationScale::Learned(weight)
            },
        };
        let spec = Self {
            normalization,
            equation,
        };
        spec.logical_parameters()?;
        Ok(spec)
    }
    /// Exact normalization geometry, stored gain convention, and reduction policy.
    pub fn normalization(&self) -> &NormalizationConstructionSpec {
        &self.normalization
    }
    /// Ordering of scaling, rounding, and normalization.
    pub const fn equation(&self) -> RmsNormEquation {
        self.equation
    }
    /// Authoritative replaceable gain slot.
    pub fn weight(&self) -> &ParameterSpec {
        match &self.normalization.scale {
            NormalizationScale::Learned(weight)
            | NormalizationScale::LearnedOffset { weight, .. } => weight,
            NormalizationScale::Unit => {
                unreachable!("learned RMS construction always declares a gain")
            }
        }
    }
}
impl DeclareLogicalParameters for LearnedRmsNormSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.normalization.logical_parameters()
    }
}
/// A branch's learned or weightless normalization, including its execution epsilon.
#[derive(Debug, Clone)]
pub enum BranchNormSpec {
    /// Learned gain with exact arithmetic.
    Learned(LearnedRmsNormSpec),
    /// Weightless FP32 normalization.
    Weightless {
        /// Positive additive mean-square epsilon.
        epsilon: f32,
    },
}
impl DeclareLogicalParameters for BranchNormSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Learned(spec) => spec.logical_parameters(),
            Self::Weightless { epsilon } if epsilon.is_finite() && *epsilon > 0.0 => Ok(Vec::new()),
            _ => Err(Error::backend(
                "Muse weightless normalization epsilon must be finite and positive",
            )),
        }
    }
}
/// Independent per-channel logits for a learned residual combination.
#[derive(Debug, Clone)]
pub struct LearnedResidualSpec {
    /// Residual channel width.
    pub dimensions: i32,
    /// Positive gate-logit divisor.
    pub temperature: f32,
    /// Positive clamp before the residual coefficient square root.
    pub minimum_variance: f32,
    /// Authoritative stored logits.
    pub gate: ParameterSpec,
}
impl DeclareLogicalParameters for LearnedResidualSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if !self.temperature.is_finite()
            || self.temperature <= 0.0
            || !self.minimum_variance.is_finite()
            || self.minimum_variance <= 0.0
        {
            return Err(Error::backend(
                "Muse residual temperature and clamp must be finite and positive",
            ));
        }
        Ok(vec![LogicalParameterDeclaration::new(
            &self.gate,
            &[self.dimensions],
            LogicalParameterRole::ResidualMixBias,
        )?])
    }
}
/// Artifact-preserved coefficients; these are never reconstructed as gate logits.
#[derive(Debug, Clone)]
pub struct StoredResidualSpec {
    /// Residual channel width.
    pub dimensions: i32,
    /// Incoming-stream multiplier.
    pub alpha: ParameterSpec,
    /// Transformed-branch multiplier.
    pub beta: ParameterSpec,
}
impl DeclareLogicalParameters for StoredResidualSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        [&self.alpha, &self.beta]
            .into_iter()
            .map(|parameter| {
                LogicalParameterDeclaration::new(
                    parameter,
                    &[self.dimensions],
                    LogicalParameterRole::ResidualMixScale,
                )
            })
            .collect()
    }
}
/// Exact residual parameter ownership and arithmetic for one invocation.
#[derive(Debug, Clone)]
pub enum ResidualSpec {
    /// Unparameterized stream addition.
    Add,
    /// Learned temperature-scaled variance-preserving coefficients.
    Learned(LearnedResidualSpec),
    /// Independently stored stream and branch coefficients.
    Stored(StoredResidualSpec),
}
impl DeclareLogicalParameters for ResidualSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Add => Ok(Vec::new()),
            Self::Learned(spec) => spec.logical_parameters(),
            Self::Stored(spec) => spec.logical_parameters(),
        }
    }
}
