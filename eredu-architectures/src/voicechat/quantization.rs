//! Architecture-owned eligibility for bounded load-time speech quantization.

use super::artifact::ParameterAdmission;
use eredu_checkpoint::{
    quantization::{BoundedQuantizationPlan, BoundedQuantizationTarget, PreparedQuantizedSource},
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::{SharedCheckpointSource, StoreError, TensorSelection},
    LinearFormat, WeightQuantization,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Independent formats for the coupled model's eligible matrix groups.
/// Sensitive recurrent parameters, perception, MoG and codec retain their selected
/// floating policy. Quantized causal projections use F32 companions and arithmetic.
/// Packed source matrices require no load-time transformation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeechQuantizationPolicy {
    /// Nemotron embedding, layer projections and agent readout.
    pub backbone: Option<WeightQuantization>,
    /// Independent function-channel readout.
    pub function_head: Option<WeightQuantization>,
    /// Causal speech transformer attention and MLP projections.
    pub tts: Option<WeightQuantization>,
}

pub(crate) fn project_parameters(
    source: SharedCheckpointSource,
    parameters: &BTreeMap<String, ParameterAdmission>,
    quantization: WeightQuantization,
    eligible: &BTreeSet<String>,
    companion_dtype: RecipeDtype,
) -> Result<(SharedCheckpointSource, BTreeMap<String, ParameterAdmission>), StoreError> {
    let mut targets = Vec::new();
    for name in eligible {
        let parameter = parameters
            .get(name)
            .ok_or_else(|| StoreError::UnknownTensor { key: name.clone() })?;
        if parameter.linear_format != LinearFormat::Dense {
            return Err(StoreError::InvalidPreparation(format!(
                "load-time quantization requires dense source matrices: {name}; implicit transcoding is unsupported"
            )));
        }
        let prefix = name.strip_suffix(".weight").ok_or_else(|| {
            StoreError::InvalidPreparation(format!("matrix identity lacks weight suffix: {name}"))
        })?;
        targets.push(
            BoundedQuantizationTarget::from_recipe(
                name,
                format!("{prefix}.scales"),
                quantization
                    .has_biases()
                    .then(|| format!("{prefix}.biases")),
                if matches!(parameter.output.dtype, RecipeDtype::F16 | RecipeDtype::BF16) {
                    DerivedWeightRecipe::Cast {
                        input: Box::new(parameter.recipe.clone()),
                        dtype: RecipeDtype::F32,
                    }
                } else {
                    parameter.recipe.clone()
                },
            )?
            .with_affine_companion_dtype(companion_dtype.clone())?,
        );
    }
    if targets.is_empty() {
        return Err(StoreError::InvalidPreparation(
            "load-time quantization has no eligible matrices".into(),
        ));
    }
    let plan = BoundedQuantizationPlan::new(quantization, 64 * 1024 * 1024, targets)?;
    let outputs = plan.output_metadata(source.as_ref())?;
    let cache = source.prepared_cache_policy().cloned().unwrap_or_default();
    let source: SharedCheckpointSource = Arc::new(PreparedQuantizedSource::new(
        source,
        plan.with_prepared_cache(cache),
    )?);
    let mut result = parameters.clone();
    for weight in eligible {
        let primary = parameters[weight].clone();
        let prefix = weight.strip_suffix(".weight").unwrap();
        for name in std::iter::once(weight.clone())
            .chain(std::iter::once(format!("{prefix}.scales")))
            .chain(
                quantization
                    .has_biases()
                    .then(|| format!("{prefix}.biases")),
            )
        {
            let metadata = &outputs[&name];
            let mut parameter = primary.clone();
            parameter.name = name.clone();
            parameter.linear_format = if name == *weight {
                quantization.into()
            } else {
                LinearFormat::Dense
            };
            parameter.recipe = DerivedWeightRecipe::source(&name, TensorSelection::Full);
            parameter.output = parameter
                .recipe
                .infer(source.as_ref())
                .map_err(|e| StoreError::InvalidPreparation(e.to_string()))?;
            parameter.stored_bytes = metadata.encoded_byte_len;
            parameter.conversion_workspace_bytes = 0;
            parameter.materialization_peak_bytes = metadata.encoded_byte_len;
            result.insert(name, parameter);
        }
    }
    Ok((source, result))
}
