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

/// Storage eligibility projected from the component's typed neural slots.
/// This extension grants neither mutability nor observation/backend support.
#[derive(Debug, Clone, Default)]
pub(crate) struct SpeechPrecisionDeclaration {
    slots: std::collections::BTreeMap<String, NarrowStorage>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NarrowStorage {
    Matrix,
    Convolution,
}
impl SpeechPrecisionDeclaration {
    fn register(&mut self, name: &str, role: Option<eredu_nn::LogicalParameterRole>, affine: bool) {
        if role == Some(eredu_nn::LogicalParameterRole::LinearWeight) {
            self.slots.insert(
                name.to_owned(),
                if affine {
                    NarrowStorage::Matrix
                } else {
                    NarrowStorage::Convolution
                },
            );
        }
    }
    pub(crate) fn from_spec(
        spec: &impl eredu_nn::DeclareLogicalParameters,
    ) -> Result<Self, eredu_nn::Error> {
        let slots = spec.logical_parameters()?;
        eredu_nn::validate_logical_parameter_scope(&slots)
            .map_err(eredu_nn::Error::backend_source)?;
        let mut declaration = Self::default();
        for slot in slots {
            declaration.register(
                slot.parameter.id.as_str(),
                Some(slot.role),
                slot.linear_format.is_some(),
            );
        }
        Ok(declaration)
    }
    pub(crate) fn from_groups<'a>(
        groups: impl IntoIterator<Item = &'a eredu_runtime::OwnedParameterGroupSpec>,
    ) -> Self {
        let mut declaration = Self::default();
        for member in groups.into_iter().flat_map(|group| group.members()) {
            if member.linear_companion().is_none() {
                declaration.register(
                    member.target(),
                    member.logical_role(),
                    member.logical_linear_format().is_some(),
                );
            }
        }
        declaration
    }
    pub(crate) fn narrows(&self, policy: SpeechPrecision, name: &str) -> bool {
        match self.slots.get(name) {
            Some(NarrowStorage::Matrix) => true,
            Some(NarrowStorage::Convolution) => policy.compute != SpeechFloat::F32,
            None => false,
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
    /// Validates precision without submitting neural work.
    pub(crate) fn require<T: Tensor>(
        self,
        module: &impl Parameterized<T>,
        input: &T,
        declaration: &SpeechPrecisionDeclaration,
    ) -> Result<(), &'static str> {
        if input.element_type() != Some(TensorElementType::F32) {
            return Err("speech boundary requires F32 activations");
        }
        struct Check<'d> {
            declaration: &'d SpeechPrecisionDeclaration,
            policy: SpeechPrecision,
            valid: bool,
        }
        impl<'a, T: Tensor> ParameterVisitor<'a, T> for Check<'_> {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a T) {
                let element = if self.declaration.narrows(self.policy, metadata.id.as_str()) {
                    self.policy.weights.element()
                } else {
                    TensorElementType::F32
                };
                self.valid &= value.element_type() == Some(element);
            }
        }
        let mut check = Check {
            declaration,
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
        declaration: &SpeechPrecisionDeclaration,
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
                ) || !declaration
                    .narrows(self, binding.logical_target().unwrap_or(binding.name()))
                    || output.dtype == self.weights.recipe()
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
    SpeechPrecision::default().require(module, input, &SpeechPrecisionDeclaration::default())
}
