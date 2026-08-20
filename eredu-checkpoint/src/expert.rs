//! Family-neutral alternative-layout recipes for routed expert banks.

use crate::recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeError};
use crate::store::TensorSelection;

/// Physical names for one independently stored SwiGLU expert.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IndependentSwiGluExpertNames {
    /// Gate projection source.
    pub gate: String,
    /// Up projection source.
    pub up: String,
    /// Down projection source.
    pub down: String,
}

/// Alternative physical names and canonical targets for one SwiGLU expert bank.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SwiGluExpertLayoutNames {
    /// Canonical fused gate/up target.
    pub target_gate_up: String,
    /// Canonical packed down target.
    pub target_down: String,
    /// Native fused gate/up physical source.
    pub packed_gate_up: String,
    /// Native packed down physical source.
    pub packed_down: String,
    /// Alternative packed gate source.
    pub separate_gate: String,
    /// Alternative packed up source.
    pub separate_up: String,
    /// Alternative packed down source.
    pub separate_down: String,
    /// Optional independently stored experts in global expert order.
    pub independent: Vec<IndependentSwiGluExpertNames>,
}

/// Physical layout selected for a derived expert bank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SwiGluExpertStorageLayout {
    /// Checkpoint already stores the canonical fused bank.
    Packed,
    /// Checkpoint stores packed gate and up tensors separately.
    SeparatePacked,
    /// Checkpoint stores one gate/up/down triple per expert.
    Independent,
}

/// Canonical recipes selected for one supported expert layout.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SwiGluExpertRecipes {
    /// Canonical parameter identity produced by [`Self::gate_up`].
    pub target_gate_up: String,
    /// Canonical parameter identity produced by [`Self::down`].
    pub target_down: String,
    /// Selected physical organization.
    pub layout: SwiGluExpertStorageLayout,
    /// Recipe producing the canonical fused gate/up bank.
    pub gate_up: DerivedWeightRecipe,
    /// Recipe producing the canonical packed down bank.
    pub down: DerivedWeightRecipe,
}

/// Failure to select or validate an alternative expert layout.
#[derive(Debug, thiserror::Error)]
pub enum ExpertLayoutError {
    /// The catalog contains no complete supported layout.
    #[error(
        "checkpoint has no complete packed, separate-packed, or independent SwiGLU expert layout"
    )]
    MissingLayout,
    /// A selected layout has incompatible shape or scalar encodings.
    #[error(transparent)]
    InvalidRecipe(#[from] RecipeError),
}

/// Selects a complete physical layout and validates its canonical recipes.
pub fn resolve_swiglu_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    names: &SwiGluExpertLayoutNames,
) -> Result<SwiGluExpertRecipes, ExpertLayoutError> {
    let full = || TensorSelection::Full;
    let packed = catalog.tensor_metadata(&names.packed_gate_up).is_ok()
        && catalog.tensor_metadata(&names.packed_down).is_ok();
    let (layout, gate_up, down) = if packed {
        (
            SwiGluExpertStorageLayout::Packed,
            DerivedWeightRecipe::source(&names.packed_gate_up, full()),
            DerivedWeightRecipe::source(&names.packed_down, full()),
        )
    } else if catalog.tensor_metadata(&names.separate_gate).is_ok()
        && catalog.tensor_metadata(&names.separate_up).is_ok()
        && catalog.tensor_metadata(&names.separate_down).is_ok()
    {
        (
            SwiGluExpertStorageLayout::SeparatePacked,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(&names.separate_gate, full()),
                    DerivedWeightRecipe::source(&names.separate_up, full()),
                ],
            },
            DerivedWeightRecipe::source(&names.separate_down, full()),
        )
    } else if !names.independent.is_empty()
        && names.independent.iter().all(|expert| {
            catalog.tensor_metadata(&expert.gate).is_ok()
                && catalog.tensor_metadata(&expert.up).is_ok()
                && catalog.tensor_metadata(&expert.down).is_ok()
        })
    {
        (
            SwiGluExpertStorageLayout::Independent,
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: names
                    .independent
                    .iter()
                    .map(|expert| DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![
                            DerivedWeightRecipe::source(&expert.gate, full()),
                            DerivedWeightRecipe::source(&expert.up, full()),
                        ],
                    })
                    .collect(),
            },
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: names
                    .independent
                    .iter()
                    .map(|expert| DerivedWeightRecipe::source(&expert.down, full()))
                    .collect(),
            },
        )
    } else {
        return Err(ExpertLayoutError::MissingLayout);
    };
    // Inference rejects mismatched shapes and scalar representations before a
    // backend allocates or reads any expert payload.
    gate_up.infer(catalog)?;
    down.infer(catalog)?;
    Ok(SwiGluExpertRecipes {
        target_gate_up: names.target_gate_up.clone(),
        target_down: names.target_down.clone(),
        layout,
        gate_up,
        down,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::store::{StoreError, TensorMetadata};
    use crate::StoredDtype;

    struct Catalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>, dtype: StoredDtype) -> TensorMetadata {
        let width = match dtype {
            StoredDtype::F32 => 4,
            _ => 2,
        };
        TensorMetadata {
            name: name.into(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * width,
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: dtype,
            backing_shard: None,
        }
    }

    fn names() -> SwiGluExpertLayoutNames {
        SwiGluExpertLayoutNames {
            target_gate_up: "target.gate_up".into(),
            target_down: "target.down".into(),
            packed_gate_up: "packed.gate_up".into(),
            packed_down: "packed.down".into(),
            separate_gate: "packed.gate".into(),
            separate_up: "packed.up".into(),
            separate_down: "packed.down".into(),
            independent: vec![
                IndependentSwiGluExpertNames {
                    gate: "e0.gate".into(),
                    up: "e0.up".into(),
                    down: "e0.down".into(),
                },
                IndependentSwiGluExpertNames {
                    gate: "e1.gate".into(),
                    up: "e1.up".into(),
                    down: "e1.down".into(),
                },
            ],
        }
    }

    #[test]
    fn independent_experts_derive_canonical_packed_banks() {
        let mut tensors = BTreeMap::new();
        for expert in 0..2 {
            for (projection, shape) in [
                ("gate", vec![3, 4]),
                ("up", vec![3, 4]),
                ("down", vec![4, 3]),
            ] {
                let name = format!("e{expert}.{projection}");
                tensors.insert(name.clone(), metadata(&name, shape, StoredDtype::F16));
            }
        }
        let catalog = Catalog(tensors);
        let recipes = resolve_swiglu_expert_recipes(&catalog, &names()).unwrap();
        assert_eq!(recipes.layout, SwiGluExpertStorageLayout::Independent);
        assert_eq!(recipes.target_gate_up, "target.gate_up");
        assert_eq!(recipes.target_down, "target.down");
        assert_eq!(recipes.gate_up.infer(&catalog).unwrap().shape(), &[2, 6, 4]);
        assert_eq!(recipes.down.infer(&catalog).unwrap().shape(), &[2, 4, 3]);
    }

    #[test]
    fn incompatible_gate_up_encodings_fail_before_materialization() {
        let tensors = BTreeMap::from([
            (
                "packed.gate".into(),
                metadata("packed.gate", vec![2, 3, 4], StoredDtype::F16),
            ),
            (
                "packed.up".into(),
                metadata("packed.up", vec![2, 3, 4], StoredDtype::F32),
            ),
            (
                "packed.down".into(),
                metadata("packed.down", vec![2, 4, 3], StoredDtype::F16),
            ),
        ]);
        assert!(matches!(
            resolve_swiglu_expert_recipes(&Catalog(tensors), &names()),
            Err(ExpertLayoutError::InvalidRecipe(RecipeError::DtypeMismatch))
        ));
    }
}
