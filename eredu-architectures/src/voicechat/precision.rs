//! Portable storage and arithmetic policy for speech and perception projections.

use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized, Tensor, TensorElementType};

/// Floating storage or projection precision. Sensitive reductions and state stay F32.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpeechFloat {
    /// Released F32 execution.
    #[default]
    F32,
    /// IEEE half precision.
    F16,
    /// Brain floating point.
    Bf16,
}
impl SpeechFloat {
    /// Physical floating tensor type.
    pub const fn element(self) -> TensorElementType {
        match self {
            Self::F32 => TensorElementType::F32,
            Self::F16 => TensorElementType::F16,
            Self::Bf16 => TensorElementType::Bf16,
        }
    }
    fn recipe(self) -> eredu_checkpoint::recipe::RecipeDtype {
        use eredu_checkpoint::recipe::RecipeDtype;
        match self {
            Self::F32 => RecipeDtype::F32,
            Self::F16 => RecipeDtype::F16,
            Self::Bf16 => RecipeDtype::BF16,
        }
    }
}

/// Independent narrow weight storage and projection/convolution execution choices.
/// Norms, residuals, perception histories, mixture banks, RVQ scoring, codec and
/// recurrent transcription remain F32. Narrow projection results return to F32.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeechPrecision {
    /// Eligible learned matrix/convolution storage.
    pub weights: SpeechFloat,
    /// Projection and convolution inputs/results before restoration to F32.
    pub compute: SpeechFloat,
}
impl SpeechPrecision {
    /// Rejects narrow arithmetic paired with another weight dtype.
    pub fn validate(self) -> Result<(), &'static str> {
        if self.compute != SpeechFloat::F32 && self.compute != self.weights {
            return Err("narrow speech compute requires matching narrow weight storage");
        }
        Ok(())
    }
    /// Explicit native projection request; ordinary F32 activations need no override.
    pub fn projection_element(self) -> Option<TensorElementType> {
        (self.compute != SpeechFloat::F32).then_some(self.compute.element())
    }
    /// Whether this architecture-owned parameter is an eligible learned projection.
    pub fn narrows(self, name: &str, rank: usize) -> bool {
        name.ends_with(".weight")
            && rank >= 2
            && ((name.starts_with("stt_model.perception.")
                && (rank == 2 || self.compute != SpeechFloat::F32))
                || (name.starts_with("tts_model.tts_model.")
                    && (name.contains(".self_attn.") || name.contains(".mlp."))))
    }
    /// Validates precision without submitting neural work.
    pub(crate) fn require<T: Tensor>(
        self,
        module: &impl Parameterized<T>,
        input: &T,
    ) -> Result<(), &'static str> {
        if input.element_type() != Some(TensorElementType::F32) {
            return Err("speech boundary requires F32 activations");
        }
        struct Check {
            policy: SpeechPrecision,
            valid: bool,
        }
        impl<'a, T: Tensor> ParameterVisitor<'a, T> for Check {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a T) {
                let element = if self
                    .policy
                    .narrows(metadata.id.as_str(), value.shape().len())
                {
                    self.policy.weights.element()
                } else {
                    TensorElementType::F32
                };
                self.valid &= value.element_type() == Some(element);
            }
        }
        let mut check = Check {
            policy: self,
            valid: true,
        };
        module.visit_parameters(&mut check);
        if !check.valid {
            return Err("speech parameter precision differs from selected portable policy");
        }
        Ok(())
    }
    pub(crate) fn bindings(
        self,
        source: &eredu_checkpoint::store::SharedCheckpointSource,
        units: &mut [Vec<eredu_runtime::WeightBinding>],
    ) -> Result<(), eredu_runtime::RecipeBindingError> {
        use eredu_checkpoint::recipe::DerivedWeightRecipe;
        for unit in units {
            for binding in unit {
                if binding.is_alias() {
                    continue;
                }
                let recipe = binding.source_recipe();
                let output = recipe.infer(source.as_ref())?;
                if !matches!(
                    output.dtype,
                    eredu_checkpoint::recipe::RecipeDtype::F32
                        | eredu_checkpoint::recipe::RecipeDtype::F16
                        | eredu_checkpoint::recipe::RecipeDtype::BF16
                ) || !self.narrows(
                    binding.logical_target().unwrap_or(binding.name()),
                    output.shape.len(),
                ) || output.dtype == self.weights.recipe()
                {
                    continue;
                }
                let recipe = DerivedWeightRecipe::Cast {
                    input: Box::new(recipe),
                    dtype: self.weights.recipe(),
                };
                let bytes = recipe.infer(source.as_ref())?.byte_len();
                *binding = binding.clone().with_source_recipe(recipe, bytes)?;
            }
        }
        Ok(())
    }
    pub(crate) fn restore<T: Tensor>(
        self,
        value: T,
        context: &T::Context,
    ) -> Result<T, eredu_nn::Error> {
        if self.compute == SpeechFloat::F32 {
            Ok(value)
        } else {
            value.cast_float(TensorElementType::F32, context)
        }
    }
    pub(crate) fn conv_input<T: Tensor>(
        self,
        input: &T,
        context: &T::Context,
    ) -> Result<T, eredu_nn::Error> {
        if self.compute == SpeechFloat::F32 {
            Ok(input.clone())
        } else {
            input.cast_float(self.compute.element(), context)
        }
    }
}

pub(crate) fn require_float32<T: Tensor>(
    module: &impl Parameterized<T>,
    input: &T,
) -> Result<(), &'static str> {
    SpeechPrecision::default().require(module, input)
}
