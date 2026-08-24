//! Architecture-owned parsing and normalization of rotary configuration.

use std::collections::HashMap;

use eredu_nn::RotaryAlgorithm;

/// One scalar accepted in external RoPE configuration maps.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(untagged)]
pub enum RopeValue {
    /// Floating-point metadata.
    Float(f32),
    /// String metadata.
    String(String),
    /// Boolean metadata.
    Bool(bool),
}

/// Converts an external configuration map into a complete rotary algorithm.
pub(crate) fn normalize_algorithm(
    values: Option<&HashMap<String, RopeValue>>,
) -> Result<RotaryAlgorithm, String> {
    let Some(values) = values else {
        return Ok(RotaryAlgorithm::Default);
    };
    let from_type = values.get("type");
    let from_rope_type = values.get("rope_type");
    let string = |value: &RopeValue| match value {
        RopeValue::String(value) => Ok(value.clone()),
        _ => Err("RoPE type or rope_type must be a string".to_string()),
    };
    let kind = match (from_type, from_rope_type) {
        (Some(left), Some(right)) => {
            let left = string(left)?;
            let right = string(right)?;
            if left != right {
                return Err(format!(
                    "conflicting RoPE type {left:?} and rope_type {right:?}"
                ));
            }
            left
        }
        (Some(value), None) | (None, Some(value)) => string(value)?,
        (None, None) => "default".to_string(),
    };
    let number = |key: &str| -> Result<Option<f32>, String> {
        match values.get(key) {
            None => Ok(None),
            Some(RopeValue::Float(value)) if value.is_finite() => Ok(Some(*value)),
            Some(RopeValue::String(value)) => value
                .parse::<f32>()
                .ok()
                .filter(|value| value.is_finite())
                .map(Some)
                .ok_or_else(|| format!("RoPE {key} must be a finite number")),
            Some(_) => Err(format!("RoPE {key} must be a finite number")),
        }
    };
    let required_positive = |key: &str| {
        number(key)?
            .filter(|value| *value > 0.0)
            .ok_or_else(|| format!("RoPE {key} must be provided as a finite positive number"))
    };
    let original_positions = || -> Result<i32, String> {
        let value = required_positive("original_max_position_embeddings")?;
        if value.fract() != 0.0 || value >= i32::MAX as f32 {
            return Err(
                "RoPE original_max_position_embeddings must be an exact positive integer"
                    .to_string(),
            );
        }
        Ok(value as i32)
    };
    let boolean = |key: &str, default: bool| match values.get(key) {
        None => Ok(default),
        Some(RopeValue::Bool(value)) => Ok(*value),
        Some(_) => Err(format!("RoPE {key} must be a boolean")),
    };

    let algorithm = match kind.as_str() {
        "none" | "default" => RotaryAlgorithm::Default,
        "linear" => RotaryAlgorithm::Linear {
            factor: required_positive("factor")?,
        },
        "llama3" => {
            let low_frequency_factor = required_positive("low_freq_factor")?;
            let high_frequency_factor = required_positive("high_freq_factor")?;
            if high_frequency_factor <= low_frequency_factor {
                return Err("RoPE high_freq_factor must be greater than low_freq_factor".into());
            }
            RotaryAlgorithm::Llama3 {
                factor: required_positive("factor")?,
                low_frequency_factor,
                high_frequency_factor,
                original_max_positions: original_positions()?,
            }
        }
        "proportional" => {
            let rotary_fraction = number("partial_rotary_factor")?.unwrap_or(1.0);
            let factor = number("factor")?.unwrap_or(1.0);
            if !(0.0 < rotary_fraction && rotary_fraction <= 1.0) {
                return Err("RoPE partial_rotary_factor must be in (0, 1]".into());
            }
            if factor <= 0.0 {
                return Err("RoPE factor must be a finite positive number".into());
            }
            RotaryAlgorithm::Proportional {
                factor,
                rotary_fraction,
            }
        }
        "yarn" => {
            let beta_fast = number("beta_fast")?.unwrap_or(32.0);
            let beta_slow = number("beta_slow")?.unwrap_or(1.0);
            let concentration = number("mscale")?.unwrap_or(1.0);
            let attention_factor = number("mscale_all_dim")?.unwrap_or(0.0);
            if beta_fast <= 0.0
                || beta_slow <= 0.0
                || concentration <= 0.0
                || attention_factor < 0.0
            {
                return Err("RoPE YaRN scalar values are outside their valid ranges".into());
            }
            RotaryAlgorithm::Yarn {
                factor: required_positive("factor")?,
                original_max_positions: original_positions()?,
                beta_fast,
                beta_slow,
                concentration,
                attention_factor,
                truncate: boolean("truncate", true)?,
            }
        }
        "longrope" => {
            return Err(
                "RoPE scaling type \"longrope\" is unsupported; LongRoPE is not implemented".into(),
            );
        }
        other => return Err(format!("RoPE scaling type {other:?} is unsupported")),
    };
    algorithm.validate().map_err(|error| error.to_string())?;
    Ok(algorithm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yarn_defaults_are_resolved_before_backend_construction() {
        let values = HashMap::from([
            ("rope_type".into(), RopeValue::String("yarn".into())),
            ("factor".into(), RopeValue::String("4".into())),
            (
                "original_max_position_embeddings".into(),
                RopeValue::Float(4_096.0),
            ),
        ]);
        assert_eq!(
            normalize_algorithm(Some(&values)).unwrap(),
            RotaryAlgorithm::Yarn {
                factor: 4.0,
                original_max_positions: 4_096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                concentration: 1.0,
                attention_factor: 0.0,
                truncate: true,
            }
        );
    }

    #[test]
    fn aliases_and_integer_geometry_are_normalized_strictly() {
        let conflicting = HashMap::from([
            ("type".into(), RopeValue::String("linear".into())),
            ("rope_type".into(), RopeValue::String("yarn".into())),
        ]);
        assert!(normalize_algorithm(Some(&conflicting))
            .unwrap_err()
            .contains("conflicting RoPE type"));

        let fractional = HashMap::from([
            ("type".into(), RopeValue::String("llama3".into())),
            ("factor".into(), RopeValue::Float(8.0)),
            ("low_freq_factor".into(), RopeValue::Float(1.0)),
            ("high_freq_factor".into(), RopeValue::Float(4.0)),
            (
                "original_max_position_embeddings".into(),
                RopeValue::Float(8_192.5),
            ),
        ]);
        assert!(normalize_algorithm(Some(&fractional))
            .unwrap_err()
            .contains("exact positive integer"));
    }
}
