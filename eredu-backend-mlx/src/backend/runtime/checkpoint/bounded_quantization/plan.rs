use super::preflight::{output_names_for, quantization_error, validate_output_names};
use super::*;

/// One dense semantic weight that will be replaced by its packed representation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BoundedQuantizationTarget {
    pub(super) weight_name: String,
    pub(super) scales_name: String,
    pub(super) biases_name: Option<String>,
    pub(super) source: DerivedWeightRecipe,
    pub(super) affine_companion_dtype: RecipeDtype,
}

impl BoundedQuantizationTarget {
    /// Quantizes a checkpoint tensor in place under exact output identities.
    pub fn direct(
        weight_name: impl Into<String>,
        scales_name: impl Into<String>,
        biases_name: Option<impl Into<String>>,
    ) -> Result<Self, Error> {
        let weight_name = weight_name.into();
        Self::from_recipe(
            weight_name.clone(),
            scales_name,
            biases_name,
            DerivedWeightRecipe::source(weight_name, TensorSelection::Full),
        )
    }

    /// Quantizes a semantic recipe under exact packed runtime identities.
    pub fn from_recipe(
        weight_name: impl Into<String>,
        scales_name: impl Into<String>,
        biases_name: Option<impl Into<String>>,
        source: DerivedWeightRecipe,
    ) -> Result<Self, Error> {
        let weight_name = weight_name.into();
        let scales_name = scales_name.into();
        let biases_name = biases_name.map(Into::into);
        validate_output_names(&weight_name, &scales_name, biases_name.as_deref())?;
        Ok(Self {
            weight_name,
            scales_name,
            biases_name,
            source,
            affine_companion_dtype: RecipeDtype::F32,
        })
    }

    /// Selects the runtime dtype required for affine scales and biases.
    pub fn with_affine_companion_dtype(mut self, dtype: RecipeDtype) -> Result<Self, Error> {
        if !matches!(
            dtype,
            RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
        ) {
            return Err(quantization_error(format!(
                "bounded affine companions require F16, BF16, or F32, got {dtype:?}"
            )));
        }
        self.affine_companion_dtype = dtype;
        Ok(self)
    }

    /// Returns the packed weight's canonical logical name.
    pub fn weight_name(&self) -> &str {
        &self.weight_name
    }

    /// Returns the semantic source recipe.
    pub fn source(&self) -> &DerivedWeightRecipe {
        &self.source
    }

    /// Returns the scalar width selected for affine scale and bias outputs.
    pub fn affine_companion_bytes(&self) -> u64 {
        match self.affine_companion_dtype {
            RecipeDtype::F16 | RecipeDtype::BF16 => 2,
            RecipeDtype::F32 => 4,
            _ => unreachable!("validated bounded affine companion dtype"),
        }
    }

    /// Returns the exact packed scale tensor name.
    pub fn scales_name(&self) -> &str {
        &self.scales_name
    }

    /// Returns the exact packed affine-bias tensor name, when declared.
    pub fn biases_name(&self) -> Option<&str> {
        self.biases_name.as_deref()
    }
}

/// A validated collection of source-bounded, memory-resident transformations.
#[derive(Debug, Clone)]
pub struct BoundedQuantizationPlan {
    pub(super) quantization: WeightQuantization,
    pub(super) max_working_set_bytes: u64,
    pub(super) targets: Vec<BoundedQuantizationTarget>,
}

impl BoundedQuantizationPlan {
    /// Creates a non-empty plan with an explicit conversion working-set bound.
    pub fn new(
        quantization: impl Into<WeightQuantization>,
        max_working_set_bytes: u64,
        targets: impl IntoIterator<Item = BoundedQuantizationTarget>,
    ) -> Result<Self, Error> {
        let quantization = quantization.into();
        quantization.validate()?;
        if quantization.gguf_iquant().is_some() {
            return Err(quantization_error(
                "checkpoint-native GGUF IQ encodings cannot be produced by load-time quantization",
            ));
        }
        if max_working_set_bytes == 0 {
            return Err(quantization_error(
                "bounded quantization working-set bytes must be nonzero",
            ));
        }
        let mut targets = targets.into_iter().collect::<Vec<_>>();
        if targets.is_empty() {
            return Err(quantization_error(
                "bounded quantization requires at least one target",
            ));
        }
        targets.sort_by(|left, right| left.weight_name.cmp(&right.weight_name));
        let mut output_names = BTreeSet::new();
        for target in &targets {
            if quantization.has_biases() && target.biases_name.is_none() {
                return Err(quantization_error(format!(
                    "bounded affine quantization target {:?} has no affine-bias identity",
                    target.weight_name
                )));
            }
            for name in output_names_for(target, quantization)? {
                if !output_names.insert(name.clone()) {
                    return Err(quantization_error(format!(
                        "bounded quantization output {name:?} is produced more than once"
                    )));
                }
            }
        }
        Ok(Self {
            quantization,
            max_working_set_bytes,
            targets,
        })
    }

    /// Returns the final packed encoding.
    pub const fn quantization(&self) -> WeightQuantization {
        self.quantization
    }

    /// Returns the admitted conversion working-set bound.
    pub const fn max_working_set_bytes(&self) -> u64 {
        self.max_working_set_bytes
    }

    /// Returns targets in deterministic logical-name order.
    pub fn targets(&self) -> &[BoundedQuantizationTarget] {
        &self.targets
    }
}
