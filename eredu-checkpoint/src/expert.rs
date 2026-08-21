//! Family-neutral alternative-layout recipes for routed expert banks.

use crate::recipe::{
    ordered_axis_selection, AtomicRecipeSet, DerivedWeightRecipe, RecipeCatalog, RecipeDtype,
    RecipeError, RecipeMetadata,
};
use crate::store::TensorSelection;

/// Physical names for one independently stored gated-product expert.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IndependentGatedProductExpertNames {
    /// Gate projection source.
    pub gate: String,
    /// Up projection source.
    pub up: String,
    /// Down projection source.
    pub down: String,
}

/// Alternative physical names and canonical targets for one gated-product expert bank.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GatedProductExpertLayoutNames {
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
    pub independent: Vec<IndependentGatedProductExpertNames>,
}

/// Physical layout selected for a derived expert bank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GatedProductExpertStorageLayout {
    /// Checkpoint already stores the canonical fused bank.
    Packed,
    /// Checkpoint stores packed gate and up tensors separately.
    SeparatePacked,
    /// Checkpoint stores one gate/up/down triple per expert.
    Independent,
}

/// Canonical recipes selected for one supported expert layout.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GatedProductExpertRecipes {
    /// Canonical parameter identity produced by [`Self::gate_up`].
    pub target_gate_up: String,
    /// Canonical parameter identity produced by [`Self::down`].
    pub target_down: String,
    /// Selected physical organization.
    pub layout: GatedProductExpertStorageLayout,
    /// Recipe producing the canonical fused gate/up bank.
    pub gate_up: DerivedWeightRecipe,
    /// Recipe producing the canonical packed down bank.
    pub down: DerivedWeightRecipe,
}

/// Physical or canonical names for one packed expert projection and its
/// scale and ordinary output-bias companions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertProjectionFamilyNames {
    /// Packed weight or block tensor.
    pub weight: String,
    /// Per-group scale tensor.
    pub scales: String,
    /// Ordinary learned output bias.
    pub bias: String,
}

/// Alternative physical layouts and canonical targets for one gated expert
/// projection family.
///
/// The packed layout stores gate and up rows in alternating order. The
/// separate layout stores component-major gate and up projections. Both are
/// normalized to the same component-major gate/up targets alongside one down
/// projection family.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GatedExpertProjectionFamilyNames {
    /// Canonical component-major fused gate/up outputs.
    pub target_gate_up: ExpertProjectionFamilyNames,
    /// Canonical down-projection outputs.
    pub target_down: ExpertProjectionFamilyNames,
    /// Alternating-row fused gate/up physical sources.
    pub alternating_gate_up: ExpertProjectionFamilyNames,
    /// Down-projection sources paired with the alternating layout.
    pub alternating_down: ExpertProjectionFamilyNames,
    /// Separate gate-projection physical sources.
    pub separate_gate: ExpertProjectionFamilyNames,
    /// Separate up-projection physical sources.
    pub separate_up: ExpertProjectionFamilyNames,
    /// Down-projection sources paired with the separate layout.
    pub separate_down: ExpertProjectionFamilyNames,
}

/// Atomically validated canonical recipes for a gated expert projection
/// family.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GatedExpertProjectionFamilyRecipes {
    /// Selected physical organization.
    pub layout: GatedProductExpertStorageLayout,
    /// Six canonical gate/up and down weight, scale, and bias outputs.
    pub outputs: AtomicRecipeSet,
}

/// Failure to select or validate an alternative expert layout.
#[derive(Debug, thiserror::Error)]
pub enum ExpertLayoutError {
    /// The catalog contains no complete supported layout.
    #[error(
        "checkpoint has no complete packed, separate-packed, or independent gated-product expert layout"
    )]
    MissingLayout,
    /// Projection companions disagree on expert, row, or scalar geometry.
    #[error("invalid gated expert projection family: {detail}")]
    InvalidProjectionFamily {
        /// Description of the rejected metadata relationship.
        detail: String,
    },
    /// A selected layout has incompatible shape or scalar encodings.
    #[error(transparent)]
    InvalidRecipe(#[from] RecipeError),
}

impl GatedExpertProjectionFamilyRecipes {
    /// Returns one validated canonical output recipe.
    pub fn get(&self, target: &str) -> Option<&DerivedWeightRecipe> {
        self.outputs.get(target)
    }

    /// Consumes the family into its atomic canonical output set.
    pub fn into_outputs(self) -> AtomicRecipeSet {
        self.outputs
    }
}

/// Selects a complete physical layout and validates its canonical recipes.
pub fn resolve_gated_product_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    names: &GatedProductExpertLayoutNames,
) -> Result<GatedProductExpertRecipes, ExpertLayoutError> {
    let full = || TensorSelection::Full;
    let packed = catalog.tensor_metadata(&names.packed_gate_up).is_ok()
        && catalog.tensor_metadata(&names.packed_down).is_ok();
    let (layout, gate_up, down) = if packed {
        (
            GatedProductExpertStorageLayout::Packed,
            DerivedWeightRecipe::source(&names.packed_gate_up, full()),
            DerivedWeightRecipe::source(&names.packed_down, full()),
        )
    } else if catalog.tensor_metadata(&names.separate_gate).is_ok()
        && catalog.tensor_metadata(&names.separate_up).is_ok()
        && catalog.tensor_metadata(&names.separate_down).is_ok()
    {
        (
            GatedProductExpertStorageLayout::SeparatePacked,
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
            GatedProductExpertStorageLayout::Independent,
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
    Ok(GatedProductExpertRecipes {
        target_gate_up: names.target_gate_up.clone(),
        target_down: names.target_down.clone(),
        layout,
        gate_up,
        down,
    })
}

/// Normalizes one complete gated-expert projection family to canonical
/// component-major gate/up rows.
///
/// Alternating fused rows are gathered in gate-then-up order with the exact
/// same permutation applied to the packed weight, scales, and output bias.
/// Separate gate/up projections are concatenated in component-major order.
/// The associated down weight, scales, and bias remain unchanged. No output
/// is returned until every source, companion, target name, shape, and dtype
/// has validated as one [`AtomicRecipeSet`].
pub fn canonical_gated_expert_projection_family_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    names: &GatedExpertProjectionFamilyNames,
) -> Result<GatedExpertProjectionFamilyRecipes, ExpertLayoutError> {
    let alternating = projection_exists(catalog, &names.alternating_gate_up)
        && projection_exists(catalog, &names.alternating_down);
    let separate = projection_exists(catalog, &names.separate_gate)
        && projection_exists(catalog, &names.separate_up)
        && projection_exists(catalog, &names.separate_down);
    let layout = if alternating {
        GatedProductExpertStorageLayout::Packed
    } else if separate {
        GatedProductExpertStorageLayout::SeparatePacked
    } else {
        return Err(ExpertLayoutError::MissingLayout);
    };

    let weights = resolve_projection_member(
        catalog,
        &names.target_gate_up.weight,
        &names.target_down.weight,
        &names.alternating_gate_up.weight,
        &names.alternating_down.weight,
        &names.separate_gate.weight,
        &names.separate_up.weight,
        &names.separate_down.weight,
        layout,
    )?;
    let scales = resolve_projection_member(
        catalog,
        &names.target_gate_up.scales,
        &names.target_down.scales,
        &names.alternating_gate_up.scales,
        &names.alternating_down.scales,
        &names.separate_gate.scales,
        &names.separate_up.scales,
        &names.separate_down.scales,
        layout,
    )?;
    let biases = resolve_projection_member(
        catalog,
        &names.target_gate_up.bias,
        &names.target_down.bias,
        &names.alternating_gate_up.bias,
        &names.alternating_down.bias,
        &names.separate_gate.bias,
        &names.separate_up.bias,
        &names.separate_down.bias,
        layout,
    )?;

    let (gate_up_weight, gate_up_scales, gate_up_bias) = match layout {
        GatedProductExpertStorageLayout::Packed => {
            let rows = alternating_row_count(catalog, &names.alternating_gate_up)?;
            let permutation = component_major_permutation(rows);
            (
                ordered_axis_selection(
                    catalog,
                    &names.alternating_gate_up.weight,
                    1,
                    permutation.clone(),
                )?,
                ordered_axis_selection(
                    catalog,
                    &names.alternating_gate_up.scales,
                    1,
                    permutation.clone(),
                )?,
                ordered_axis_selection(catalog, &names.alternating_gate_up.bias, 1, permutation)?,
            )
        }
        GatedProductExpertStorageLayout::SeparatePacked => {
            validate_separate_component_cardinality(
                catalog,
                &names.separate_gate,
                &names.separate_up,
            )?;
            (weights.gate_up, scales.gate_up, biases.gate_up)
        }
        GatedProductExpertStorageLayout::Independent => {
            return Err(ExpertLayoutError::InvalidProjectionFamily {
                detail: "independent experts are not a packed projection family".into(),
            });
        }
    };
    let gate_up_weight = canonical_mxfp4_storage(catalog, gate_up_weight, &gate_up_scales)?;
    let down_weight = canonical_mxfp4_storage(catalog, weights.down, &scales.down)?;

    validate_canonical_projection_family(
        catalog,
        &gate_up_weight,
        &gate_up_scales,
        &gate_up_bias,
        &down_weight,
        &scales.down,
        &biases.down,
    )?;

    let outputs = AtomicRecipeSet::new(
        catalog,
        [
            (names.target_gate_up.weight.clone(), gate_up_weight),
            (names.target_gate_up.scales.clone(), gate_up_scales),
            (names.target_gate_up.bias.clone(), gate_up_bias),
            (names.target_down.weight.clone(), down_weight),
            (names.target_down.scales.clone(), scales.down),
            (names.target_down.bias.clone(), biases.down),
        ],
    )?;
    Ok(GatedExpertProjectionFamilyRecipes { layout, outputs })
}

fn projection_exists<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    names: &ExpertProjectionFamilyNames,
) -> bool {
    [&names.weight, &names.scales, &names.bias]
        .into_iter()
        .all(|name| catalog.tensor_metadata(name).is_ok())
}

#[allow(clippy::too_many_arguments)]
fn resolve_projection_member<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    target_gate_up: &str,
    target_down: &str,
    alternating_gate_up: &str,
    alternating_down: &str,
    separate_gate: &str,
    separate_up: &str,
    separate_down: &str,
    layout: GatedProductExpertStorageLayout,
) -> Result<GatedProductExpertRecipes, ExpertLayoutError> {
    let unavailable = "\0unavailable packed projection family";
    let (packed_gate_up, packed_down) = if layout == GatedProductExpertStorageLayout::Packed {
        (alternating_gate_up, alternating_down)
    } else {
        (unavailable, unavailable)
    };
    let recipes = resolve_gated_product_expert_recipes(
        catalog,
        &GatedProductExpertLayoutNames {
            target_gate_up: target_gate_up.into(),
            target_down: target_down.into(),
            packed_gate_up: packed_gate_up.into(),
            packed_down: packed_down.into(),
            separate_gate: separate_gate.into(),
            separate_up: separate_up.into(),
            separate_down: separate_down.into(),
            independent: Vec::new(),
        },
    )?;
    if recipes.layout != layout {
        return Err(ExpertLayoutError::InvalidProjectionFamily {
            detail: "weight, scale, and bias members selected different physical layouts".into(),
        });
    }
    Ok(recipes)
}

fn alternating_row_count<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    names: &ExpertProjectionFamilyNames,
) -> Result<usize, ExpertLayoutError> {
    let metadata = projection_source_metadata(catalog, names)?;
    let rows = metadata[0].shape[1];
    if rows % 2 != 0 {
        return Err(ExpertLayoutError::InvalidProjectionFamily {
            detail: format!("alternating gate/up row count must be even, got {rows}"),
        });
    }
    Ok(rows)
}

fn component_major_permutation(rows: usize) -> Vec<usize> {
    (0..rows).step_by(2).chain((1..rows).step_by(2)).collect()
}

fn canonical_mxfp4_storage<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    weight: DerivedWeightRecipe,
    scales: &DerivedWeightRecipe,
) -> Result<DerivedWeightRecipe, ExpertLayoutError> {
    let weight_metadata = weight.infer(catalog)?;
    let scale_metadata = scales.infer(catalog)?;
    let mut packed_shape = scale_metadata.shape;
    let packed_width =
        packed_shape
            .last_mut()
            .ok_or_else(|| ExpertLayoutError::InvalidProjectionFamily {
                detail: "MXFP4 scales have no packed input axis".into(),
            })?;
    // One MXFP4 group contains 32 four-bit values: 16 bytes, represented by
    // the MLX affine kernels as four U32 storage units. SafeTensors exposes
    // those bytes as [..., groups, 16] U8 blocks, while converted GGUF
    // exposes [..., groups * 4] U32 directly. Canonical parameters use the
    // latter shape so translated GGUF remains a source/concatenate recipe and
    // bounded expert selection can reach its physical sources unchanged.
    *packed_width =
        packed_width
            .checked_mul(4)
            .ok_or_else(|| ExpertLayoutError::InvalidProjectionFamily {
                detail: "MXFP4 packed input width overflowed".into(),
            })?;
    if weight_metadata.dtype == RecipeDtype::U32 && weight_metadata.shape == packed_shape {
        return Ok(weight);
    }
    let viewed = DerivedWeightRecipe::View {
        input: Box::new(weight),
        dtype: RecipeDtype::U32,
        shape: packed_shape,
    };
    viewed
        .infer(catalog)
        .map_err(|error| ExpertLayoutError::InvalidProjectionFamily {
            detail: format!(
                "packed weight cannot be viewed as U32 MXFP4 storage matching its scales: {error}"
            ),
        })?;
    Ok(viewed)
}

fn validate_separate_component_cardinality<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    gate: &ExpertProjectionFamilyNames,
    up: &ExpertProjectionFamilyNames,
) -> Result<(), ExpertLayoutError> {
    let gate = projection_source_metadata(catalog, gate)?;
    let up = projection_source_metadata(catalog, up)?;
    for (kind, gate, up) in [
        ("weight", &gate[0], &up[0]),
        ("scales", &gate[1], &up[1]),
        ("bias", &gate[2], &up[2]),
    ] {
        if gate.shape != up.shape {
            return Err(ExpertLayoutError::InvalidProjectionFamily {
                detail: format!("separate gate/up {kind} shapes differ"),
            });
        }
        if gate.dtype != up.dtype {
            return Err(RecipeError::DtypeMismatch.into());
        }
    }
    Ok(())
}

fn projection_source_metadata<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    names: &ExpertProjectionFamilyNames,
) -> Result<[RecipeMetadata; 3], ExpertLayoutError> {
    let full = || TensorSelection::Full;
    let metadata = [
        DerivedWeightRecipe::source(&names.weight, full()).infer(catalog)?,
        DerivedWeightRecipe::source(&names.scales, full()).infer(catalog)?,
        DerivedWeightRecipe::source(&names.bias, full()).infer(catalog)?,
    ];
    validate_projection_metadata(&metadata)?;
    Ok(metadata)
}

fn validate_projection_metadata(metadata: &[RecipeMetadata; 3]) -> Result<(), ExpertLayoutError> {
    let [weight, scales, bias] = metadata;
    if weight.shape.len() < 3 || scales.shape.len() < 3 || bias.shape.len() != 2 {
        return Err(ExpertLayoutError::InvalidProjectionFamily {
            detail: format!(
                "expected weight/scales rank >= 3 and bias rank 2, got {}/{}/{}",
                weight.shape.len(),
                scales.shape.len(),
                bias.shape.len()
            ),
        });
    }
    if weight.shape[..2] != scales.shape[..2] || weight.shape[..2] != bias.shape[..2] {
        return Err(ExpertLayoutError::InvalidProjectionFamily {
            detail: "weight, scales, and bias disagree on expert/output-row cardinality".into(),
        });
    }
    if !matches!(
        bias.dtype,
        RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32 | RecipeDtype::F64
    ) {
        return Err(ExpertLayoutError::InvalidProjectionFamily {
            detail: format!(
                "ordinary projection bias must be floating, got {:?}",
                bias.dtype
            ),
        });
    }
    Ok(())
}

fn validate_canonical_projection_family<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    gate_up_weight: &DerivedWeightRecipe,
    gate_up_scales: &DerivedWeightRecipe,
    gate_up_bias: &DerivedWeightRecipe,
    down_weight: &DerivedWeightRecipe,
    down_scales: &DerivedWeightRecipe,
    down_bias: &DerivedWeightRecipe,
) -> Result<(), ExpertLayoutError> {
    let gate_up = [
        gate_up_weight.infer(catalog)?,
        gate_up_scales.infer(catalog)?,
        gate_up_bias.infer(catalog)?,
    ];
    let down = [
        down_weight.infer(catalog)?,
        down_scales.infer(catalog)?,
        down_bias.infer(catalog)?,
    ];
    validate_projection_metadata(&gate_up)?;
    validate_projection_metadata(&down)?;
    if gate_up[0].shape[0] != down[0].shape[0] {
        return Err(ExpertLayoutError::InvalidProjectionFamily {
            detail: "gate/up and down projections have different expert cardinality".into(),
        });
    }
    Ok(())
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

    fn names() -> GatedProductExpertLayoutNames {
        GatedProductExpertLayoutNames {
            target_gate_up: "target.gate_up".into(),
            target_down: "target.down".into(),
            packed_gate_up: "packed.gate_up".into(),
            packed_down: "packed.down".into(),
            separate_gate: "packed.gate".into(),
            separate_up: "packed.up".into(),
            separate_down: "packed.down".into(),
            independent: vec![
                IndependentGatedProductExpertNames {
                    gate: "e0.gate".into(),
                    up: "e0.up".into(),
                    down: "e0.down".into(),
                },
                IndependentGatedProductExpertNames {
                    gate: "e1.gate".into(),
                    up: "e1.up".into(),
                    down: "e1.down".into(),
                },
            ],
        }
    }

    fn projection(weight: &str, scales: &str, bias: &str) -> ExpertProjectionFamilyNames {
        ExpertProjectionFamilyNames {
            weight: weight.into(),
            scales: scales.into(),
            bias: bias.into(),
        }
    }

    fn family_names() -> GatedExpertProjectionFamilyNames {
        GatedExpertProjectionFamilyNames {
            target_gate_up: projection(
                "target.gate_up.weight",
                "target.gate_up.scales",
                "target.gate_up.bias",
            ),
            target_down: projection(
                "target.down.weight",
                "target.down.scales",
                "target.down.bias",
            ),
            alternating_gate_up: projection(
                "alternating.gate_up.weight",
                "alternating.gate_up.scales",
                "alternating.gate_up.bias",
            ),
            alternating_down: projection(
                "alternating.down.weight",
                "alternating.down.scales",
                "alternating.down.bias",
            ),
            separate_gate: projection(
                "separate.gate.weight",
                "separate.gate.scales",
                "separate.gate.bias",
            ),
            separate_up: projection(
                "separate.up.weight",
                "separate.up.scales",
                "separate.up.bias",
            ),
            separate_down: projection(
                "separate.down.weight",
                "separate.down.scales",
                "separate.down.bias",
            ),
        }
    }

    fn insert_projection(
        tensors: &mut BTreeMap<String, TensorMetadata>,
        names: &ExpertProjectionFamilyNames,
        rows: usize,
        experts: usize,
        weight_dtype: StoredDtype,
        bias_dtype: StoredDtype,
    ) {
        let weight_shape = match weight_dtype {
            StoredDtype::U32 => vec![experts, rows, 4],
            _ => vec![experts, rows, 1, 16],
        };
        tensors.insert(
            names.weight.clone(),
            metadata(&names.weight, weight_shape, weight_dtype),
        );
        tensors.insert(
            names.scales.clone(),
            metadata(&names.scales, vec![experts, rows, 1], StoredDtype::U8),
        );
        tensors.insert(
            names.bias.clone(),
            metadata(&names.bias, vec![experts, rows], bias_dtype),
        );
    }

    fn alternating_catalog(names: &GatedExpertProjectionFamilyNames) -> Catalog {
        let mut tensors = BTreeMap::new();
        insert_projection(
            &mut tensors,
            &names.alternating_gate_up,
            6,
            2,
            StoredDtype::U8,
            StoredDtype::F32,
        );
        insert_projection(
            &mut tensors,
            &names.alternating_down,
            4,
            2,
            StoredDtype::U8,
            StoredDtype::F32,
        );
        Catalog(tensors)
    }

    fn separate_catalog(names: &GatedExpertProjectionFamilyNames) -> Catalog {
        let mut tensors = BTreeMap::new();
        for projection in [&names.separate_gate, &names.separate_up] {
            insert_projection(
                &mut tensors,
                projection,
                3,
                2,
                StoredDtype::U32,
                StoredDtype::F16,
            );
        }
        insert_projection(
            &mut tensors,
            &names.separate_down,
            4,
            2,
            StoredDtype::U32,
            StoredDtype::F16,
        );
        Catalog(tensors)
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
        let recipes = resolve_gated_product_expert_recipes(&catalog, &names()).unwrap();
        assert_eq!(recipes.layout, GatedProductExpertStorageLayout::Independent);
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
            resolve_gated_product_expert_recipes(&Catalog(tensors), &names()),
            Err(ExpertLayoutError::InvalidRecipe(RecipeError::DtypeMismatch))
        ));
    }

    #[test]
    fn alternating_projection_companions_share_one_component_major_permutation() {
        let names = family_names();
        let catalog = alternating_catalog(&names);
        let recipes = canonical_gated_expert_projection_family_recipes(&catalog, &names).unwrap();
        assert_eq!(recipes.layout, GatedProductExpertStorageLayout::Packed);
        assert_eq!(recipes.outputs.iter().count(), 6);

        let expected = vec![0, 2, 4, 1, 3, 5];
        assert!(matches!(
            recipes.get(&names.target_gate_up.weight).unwrap(),
            DerivedWeightRecipe::View { input, dtype: RecipeDtype::U32, shape }
                if shape == &[2, 6, 4]
                    && input.as_ref() == &DerivedWeightRecipe::source(
                        &names.alternating_gate_up.weight,
                        TensorSelection::Indices {
                            axis: 1,
                            indices: expected.clone(),
                        },
                    )
        ));
        for (target, source) in [
            (
                &names.target_gate_up.scales,
                &names.alternating_gate_up.scales,
            ),
            (&names.target_gate_up.bias, &names.alternating_gate_up.bias),
        ] {
            assert_eq!(
                recipes.get(target).unwrap(),
                &DerivedWeightRecipe::source(
                    source,
                    TensorSelection::Indices {
                        axis: 1,
                        indices: expected.clone(),
                    },
                )
            );
        }
        assert_eq!(
            recipes
                .get(&names.target_gate_up.weight)
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[2, 6, 4]
        );
        assert_eq!(
            recipes.get(&names.target_down.bias).unwrap(),
            &DerivedWeightRecipe::source(&names.alternating_down.bias, TensorSelection::Full,)
        );
    }

    #[test]
    fn separate_projection_companions_assemble_one_component_major_family() {
        let names = family_names();
        let catalog = separate_catalog(&names);
        let recipes = canonical_gated_expert_projection_family_recipes(&catalog, &names).unwrap();
        assert_eq!(
            recipes.layout,
            GatedProductExpertStorageLayout::SeparatePacked
        );
        for (target, expected_shape) in [
            (&names.target_gate_up.weight, vec![2, 6, 4]),
            (&names.target_gate_up.scales, vec![2, 6, 1]),
            (&names.target_gate_up.bias, vec![2, 6]),
            (&names.target_down.weight, vec![2, 4, 4]),
            (&names.target_down.scales, vec![2, 4, 1]),
            (&names.target_down.bias, vec![2, 4]),
        ] {
            assert_eq!(
                recipes
                    .get(target)
                    .unwrap()
                    .infer(&catalog)
                    .unwrap()
                    .shape(),
                expected_shape
            );
        }
        assert!(matches!(
            recipes.get(&names.target_gate_up.weight).unwrap(),
            DerivedWeightRecipe::Concatenate { axis: 1, inputs }
                if inputs.len() == 2
        ));
    }

    #[test]
    fn malformed_projection_shapes_dtypes_and_cardinality_fail_atomically() {
        let names = family_names();

        let mut odd = alternating_catalog(&names);
        insert_projection(
            &mut odd.0,
            &names.alternating_gate_up,
            5,
            2,
            StoredDtype::U8,
            StoredDtype::F32,
        );
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&odd, &names),
            Err(ExpertLayoutError::InvalidProjectionFamily { .. })
        ));

        let mut shape = alternating_catalog(&names);
        shape.0.insert(
            names.alternating_gate_up.scales.clone(),
            metadata(
                &names.alternating_gate_up.scales,
                vec![2, 4, 1],
                StoredDtype::U8,
            ),
        );
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&shape, &names),
            Err(ExpertLayoutError::InvalidProjectionFamily { .. })
        ));

        let mut dtype = separate_catalog(&names);
        dtype.0.insert(
            names.separate_up.weight.clone(),
            metadata(&names.separate_up.weight, vec![2, 3, 4], StoredDtype::F32),
        );
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&dtype, &names),
            Err(ExpertLayoutError::InvalidRecipe(RecipeError::DtypeMismatch))
        ));

        let mut cardinality = alternating_catalog(&names);
        insert_projection(
            &mut cardinality.0,
            &names.alternating_down,
            4,
            3,
            StoredDtype::U8,
            StoredDtype::F32,
        );
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&cardinality, &names),
            Err(ExpertLayoutError::InvalidProjectionFamily { .. })
        ));

        let mut bias_dtype = separate_catalog(&names);
        for bias in [&names.separate_gate.bias, &names.separate_up.bias] {
            bias_dtype
                .0
                .insert(bias.clone(), metadata(bias, vec![2, 3], StoredDtype::U8));
        }
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&bias_dtype, &names),
            Err(ExpertLayoutError::InvalidProjectionFamily { .. })
        ));
    }

    #[test]
    fn missing_companion_and_target_collision_publish_no_partial_family() {
        let names = family_names();
        let mut missing = alternating_catalog(&names);
        missing.0.remove(&names.alternating_gate_up.bias);
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&missing, &names),
            Err(ExpertLayoutError::MissingLayout)
        ));

        let catalog = alternating_catalog(&names);
        let mut collision = names.clone();
        collision.target_down.weight = collision.target_gate_up.weight.clone();
        assert!(matches!(
            canonical_gated_expert_projection_family_recipes(&catalog, &collision),
            Err(ExpertLayoutError::InvalidRecipe(
                RecipeError::DuplicateOutput { .. }
            ))
        ));
    }
}
