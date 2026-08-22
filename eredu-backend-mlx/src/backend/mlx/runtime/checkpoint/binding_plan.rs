//! Declarative runtime targets backed by checkpoint-derived recipes.

#[cfg(test)]
use eredu_checkpoint::StoredDtype;

use eredu_runtime::WeightBinding;
use std::collections::{BTreeMap, BTreeSet};

use super::{
    recipe::{DerivedWeightRecipe, RecipeDtype, WeightRecipeError},
    store::TensorSelection,
};
use crate::backend::mlx::runtime::residency::manager::ResidencyError;

/// One runtime target and the physical recipe that produces it.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlannedBinding {
    pub target_name: String,
    pub expected_shape: Vec<usize>,
    pub expected_dtype: RecipeDtype,
    pub recipe: DerivedWeightRecipe,
}

impl PlannedBinding {
    #[cfg(test)]
    pub fn direct(
        target_name: impl Into<String>,
        source_key: impl Into<String>,
        expected_shape: impl Into<Vec<usize>>,
        expected_dtype: RecipeDtype,
    ) -> Self {
        Self {
            target_name: target_name.into(),
            expected_shape: expected_shape.into(),
            expected_dtype,
            recipe: DerivedWeightRecipe::source(source_key, TensorSelection::Full),
        }
    }
}

/// Deterministic binding plan shared by loading and residency policies.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BindingPlan {
    bindings: Vec<PlannedBinding>,
    shared_source_keys: BTreeSet<String>,
}

impl BindingPlan {
    /// Builds a plan in which physical source claims are exclusive.
    pub fn new(bindings: Vec<PlannedBinding>) -> Result<Self, BindingPlanError> {
        Self::allowing_shared_sources(bindings, BTreeSet::new())
    }

    /// Builds a plan with an explicit allowlist for intentionally shared sources.
    pub fn allowing_shared_sources(
        mut bindings: Vec<PlannedBinding>,
        shared_source_keys: BTreeSet<String>,
    ) -> Result<Self, BindingPlanError> {
        bindings.sort_by(|left, right| left.target_name.cmp(&right.target_name));
        let mut targets = BTreeSet::new();
        let mut claims = BTreeMap::<String, String>::new();
        for binding in &bindings {
            if binding.target_name.trim().is_empty() {
                return Err(BindingPlanError::EmptyTarget);
            }
            // An empty shape is a scalar. Zero-sized tensor dimensions remain
            // invalid because residency bindings cannot carry zero bytes.
            if binding.expected_shape.contains(&0) {
                return Err(BindingPlanError::InvalidTargetShape {
                    target: binding.target_name.clone(),
                    shape: binding.expected_shape.clone(),
                });
            }
            binding
                .expected_shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| BindingPlanError::TargetShapeOverflow {
                    target: binding.target_name.clone(),
                })?;
            if !targets.insert(binding.target_name.clone()) {
                return Err(BindingPlanError::DuplicateTarget {
                    target: binding.target_name.clone(),
                });
            }
            let source_keys = binding.recipe.source_keys();
            if source_keys.is_empty() {
                return Err(BindingPlanError::EmptyRecipeSources {
                    target: binding.target_name.clone(),
                });
            }
            for source in source_keys {
                if source.trim().is_empty() {
                    return Err(BindingPlanError::InvalidSourceKey {
                        target: binding.target_name.clone(),
                    });
                }
                if shared_source_keys.contains(source) {
                    continue;
                }
                if let Some(first) = claims.insert(source.into(), binding.target_name.clone()) {
                    return Err(BindingPlanError::DuplicateSourceClaim {
                        source_key: source.into(),
                        first,
                        second: binding.target_name.clone(),
                    });
                }
            }
        }
        Ok(Self {
            bindings,
            shared_source_keys,
        })
    }

    #[cfg(test)]
    pub fn bindings(&self) -> &[PlannedBinding] {
        &self.bindings
    }

    #[cfg(test)]
    pub fn source_keys(&self) -> Vec<&str> {
        let mut keys = BTreeSet::new();
        for binding in &self.bindings {
            keys.extend(binding.recipe.source_keys());
        }
        keys.into_iter().collect()
    }

    #[cfg(test)]
    pub fn shared_source_keys(&self) -> &BTreeSet<String> {
        &self.shared_source_keys
    }

    /// Infers every recipe and constructs residency bindings.
    ///
    /// An authoritative materialized target supplied by an overlay store wins
    /// over the semantic source recipe, matching existing loader behavior.
    pub fn build_bindings(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, BindingPlanError> {
        let mut output = Vec::with_capacity(self.bindings.len());
        for planned in &self.bindings {
            let recipe = if store.is_authoritative_materialized_key(&planned.target_name) {
                DerivedWeightRecipe::source(&planned.target_name, TensorSelection::Full)
            } else {
                planned.recipe.clone()
            };
            let metadata = recipe.infer(store)?;
            if metadata.shape() != planned.expected_shape {
                return Err(BindingPlanError::ShapeMismatch {
                    target: planned.target_name.clone(),
                    expected: planned.expected_shape.clone(),
                    actual: metadata.shape().to_vec(),
                });
            }
            if !recipe_dtype_matches(&planned.expected_dtype, metadata.dtype()) {
                return Err(BindingPlanError::DtypeMismatch {
                    target: planned.target_name.clone(),
                    expected: planned.expected_dtype.clone(),
                    actual: metadata.dtype().clone(),
                });
            }
            let binding = match recipe {
                DerivedWeightRecipe::Source { key, selection } => {
                    WeightBinding::new(&planned.target_name, key, selection, metadata.byte_len())?
                }
                recipe => {
                    WeightBinding::from_recipe(&planned.target_name, recipe, metadata.byte_len())?
                }
            };
            output.push(binding.with_logical_target(&planned.target_name)?);
        }
        Ok(output)
    }
}

fn recipe_dtype_matches(expected: &RecipeDtype, actual: &RecipeDtype) -> bool {
    expected == actual
        || matches!((expected, actual), (RecipeDtype::U8, RecipeDtype::F8E4M3))
        // Unloaded dense module parameters use floating placeholders, then
        // adopt the checkpoint-native floating dtype. Keep derived bindings
        // consistent with direct bindings so aliases and layout recipes do
        // not reject valid BF16/F16 checkpoints during the second plan check.
        || (is_floating_recipe_dtype(expected) && is_floating_recipe_dtype(actual))
}

fn is_floating_recipe_dtype(dtype: &RecipeDtype) -> bool {
    matches!(
        dtype,
        RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32 | RecipeDtype::F64
    )
}

/// Invalid declarative bindings or inferred recipe outputs.
#[derive(Debug, thiserror::Error)]
pub enum BindingPlanError {
    #[error("binding target must not be empty")]
    EmptyTarget,
    #[error("binding target {target:?} has invalid shape {shape:?}")]
    InvalidTargetShape { target: String, shape: Vec<usize> },
    #[error("binding target {target:?} shape element count overflows")]
    TargetShapeOverflow { target: String },
    #[error("binding plan contains duplicate target {target:?}")]
    DuplicateTarget { target: String },
    #[error("binding target {target:?} has a recipe with no checkpoint sources")]
    EmptyRecipeSources { target: String },
    #[error("binding target {target:?} has an empty checkpoint source key")]
    InvalidSourceKey { target: String },
    #[error("checkpoint source {source_key:?} is claimed by both {first:?} and {second:?}")]
    DuplicateSourceClaim {
        source_key: String,
        first: String,
        second: String,
    },
    #[error("binding target {target:?} expects shape {expected:?}, recipe produces {actual:?}")]
    ShapeMismatch {
        target: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("binding target {target:?} expects dtype {expected:?}, recipe produces {actual:?}")]
    DtypeMismatch {
        target: String,
        expected: RecipeDtype,
        actual: RecipeDtype,
    },
    #[error(transparent)]
    Recipe(#[from] WeightRecipeError),
    #[error(transparent)]
    NeutralRecipe(#[from] eredu_checkpoint::recipe::RecipeError),
    #[error(transparent)]
    Declaration(#[from] eredu_runtime::ResidencyDeclarationError),
    #[error(transparent)]
    Residency(#[from] ResidencyError),
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
        WeightStoreBackend, WeightStoreDiagnostics,
    };

    struct MetadataStore {
        metadata: BTreeMap<String, TensorMetadata>,
        authoritative: BTreeSet<String>,
    }

    impl MetadataStore {
        fn new(entries: impl IntoIterator<Item = (&'static str, Vec<usize>, StoredDtype)>) -> Self {
            Self {
                metadata: entries
                    .into_iter()
                    .map(|(key, shape, stored_dtype)| {
                        (
                            key.into(),
                            TensorMetadata {
                                name: key.into(),
                                encoded_byte_len: (shape.iter().product::<usize>() * 4) as u64,
                                logical_shape: shape.clone(),
                                physical_shape: shape,
                                stored_dtype,
                                backing_shard: None,
                            },
                        )
                    })
                    .collect(),
                authoritative: BTreeSet::new(),
            }
        }

        fn with_authoritative(mut self, key: &str) -> Self {
            self.authoritative.insert(key.into());
            self
        }
    }

    impl CheckpointSource for MetadataStore {
        fn source_keys(&self) -> Vec<String> {
            self.metadata.keys().cloned().collect()
        }

        fn is_authoritative_materialized_key(&self, key: &str) -> bool {
            self.authoritative.contains(key)
        }

        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.metadata
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }

        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            Err(StoreError::UnknownTensor { key: request.key })
        }

        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Ok(WeightStoreDiagnostics {
                backend: WeightStoreBackend::Memory,
                mapping_hits: 0,
                mapping_misses: 0,
                evictions: 0,
                currently_mapped_shards: 0,
                touched_shard_paths: Vec::<PathBuf>::new(),
                physical_reads: 0,
                physical_read_bytes: 0,
                coalesced_group_hits: 0,
            })
        }
    }

    #[test]
    fn duplicate_targets_and_source_claims_are_rejected() {
        let direct = || PlannedBinding::direct("target", "source", vec![2], RecipeDtype::F32);
        assert!(matches!(
            BindingPlan::new(vec![direct(), direct()]),
            Err(BindingPlanError::DuplicateTarget { .. })
        ));
        assert!(matches!(
            BindingPlan::new(vec![
                direct(),
                PlannedBinding::direct("other", "source", vec![2], RecipeDtype::F32),
            ]),
            Err(BindingPlanError::DuplicateSourceClaim { .. })
        ));
        assert!(matches!(
            BindingPlan::new(vec![PlannedBinding {
                target_name: "empty".into(),
                expected_shape: vec![1],
                expected_dtype: RecipeDtype::F32,
                recipe: DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: Vec::new(),
                },
            }]),
            Err(BindingPlanError::EmptyRecipeSources { .. })
        ));
    }

    #[test]
    fn sources_are_enumerated_and_direct_recipes_build_bindings() {
        let store = MetadataStore::new([("source", vec![2, 2], StoredDtype::F32)]);
        let plan = BindingPlan::new(vec![PlannedBinding::direct(
            "target",
            "source",
            vec![2, 2],
            RecipeDtype::F32,
        )])
        .unwrap();
        assert_eq!(plan.bindings().len(), 1);
        assert!(plan.shared_source_keys().is_empty());
        assert_eq!(plan.source_keys(), ["source"]);
        let bindings = plan.build_bindings(&store).unwrap();
        assert_eq!(bindings[0].name(), "target");
        // Direct tensors are represented as source recipes in the plan, but
        // retain the direct residency binding representation after validation.
        assert!(matches!(
            plan.bindings()[0].recipe,
            DerivedWeightRecipe::Source { .. }
        ));
        assert!(bindings[0].recipe().is_none());
    }

    #[test]
    fn authoritative_materialized_target_supersedes_the_semantic_recipe() {
        let store = MetadataStore::new([
            ("source", vec![2], StoredDtype::F32),
            ("target", vec![2], StoredDtype::F32),
        ])
        .with_authoritative("target");
        let plan = BindingPlan::new(vec![PlannedBinding::direct(
            "target",
            "source",
            vec![2],
            RecipeDtype::F32,
        )])
        .unwrap();
        let bindings = plan.build_bindings(&store).unwrap();
        assert_eq!(bindings[0].checkpoint_keys(), ["target"]);
    }

    #[test]
    fn encoded_e4m3_source_can_target_runtime_u8() {
        let store = MetadataStore::new([("encoded", vec![4], StoredDtype::F8E4M3)]);
        let plan = BindingPlan::new(vec![PlannedBinding::direct(
            "bytes",
            "encoded",
            vec![4],
            RecipeDtype::U8,
        )])
        .unwrap();
        assert!(plan.build_bindings(&store).is_ok());
    }

    #[test]
    fn native_floating_source_can_replace_unloaded_f32_placeholder() {
        let store = MetadataStore::new([("bf16", vec![4], StoredDtype::BF16)]);
        let plan = BindingPlan::new(vec![PlannedBinding::direct(
            "weight",
            "bf16",
            vec![4],
            RecipeDtype::F32,
        )])
        .unwrap();
        assert!(plan.build_bindings(&store).is_ok());
    }

    #[test]
    fn recipe_inference_covers_select_join_shape_transforms_cast_and_view() {
        let store = MetadataStore::new([
            ("matrix", vec![2, 2], StoredDtype::F32),
            ("left", vec![2], StoredDtype::F32),
            ("right", vec![2], StoredDtype::F32),
            ("bytes", vec![4], StoredDtype::U8),
        ]);
        let source = DerivedWeightRecipe::source("matrix", TensorSelection::Full);
        let recipes = [
            (
                DerivedWeightRecipe::Select {
                    input: Box::new(source.clone()),
                    selection: TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                },
                vec![1, 2],
                RecipeDtype::F32,
            ),
            (
                DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![
                        DerivedWeightRecipe::source("left", TensorSelection::Full),
                        DerivedWeightRecipe::source("right", TensorSelection::Full),
                    ],
                },
                vec![4],
                RecipeDtype::F32,
            ),
            (
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: vec![
                        DerivedWeightRecipe::source("left", TensorSelection::Full),
                        DerivedWeightRecipe::source("right", TensorSelection::Full),
                    ],
                },
                vec![2, 2],
                RecipeDtype::F32,
            ),
            (
                DerivedWeightRecipe::Reshape {
                    input: Box::new(source.clone()),
                    shape: vec![4],
                },
                vec![4],
                RecipeDtype::F32,
            ),
            (
                DerivedWeightRecipe::Transpose {
                    input: Box::new(source.clone()),
                    axes: vec![1, 0],
                },
                vec![2, 2],
                RecipeDtype::F32,
            ),
            (
                DerivedWeightRecipe::Cast {
                    input: Box::new(source),
                    dtype: RecipeDtype::F16,
                },
                vec![2, 2],
                RecipeDtype::F16,
            ),
            (
                DerivedWeightRecipe::View {
                    input: Box::new(DerivedWeightRecipe::source("bytes", TensorSelection::Full)),
                    dtype: RecipeDtype::F32,
                    shape: vec![1],
                },
                vec![1],
                RecipeDtype::F32,
            ),
        ];
        for (recipe, shape, dtype) in recipes {
            let metadata = recipe
                .infer(&store as &dyn eredu_checkpoint::store::CheckpointSource)
                .unwrap();
            assert_eq!(metadata.shape(), shape);
            assert_eq!(metadata.dtype(), &dtype);
        }
    }

    #[test]
    fn invalid_recipe_geometry_and_target_overflow_fail_before_materialization() {
        let store = MetadataStore::new([
            ("matrix", vec![2, 2], StoredDtype::F32),
            ("scalar", vec![], StoredDtype::F32),
        ]);
        let scalar = BindingPlan::new(vec![PlannedBinding::direct(
            "scalar_target",
            "scalar",
            vec![],
            RecipeDtype::F32,
        )])
        .unwrap();
        assert!(scalar.build_bindings(&store).is_ok());
        assert!(matches!(
            BindingPlan::new(vec![PlannedBinding::direct(
                "zero",
                "matrix",
                vec![2, 0],
                RecipeDtype::F32,
            )]),
            Err(BindingPlanError::InvalidTargetShape { .. })
        ));
        let invalid = DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source("matrix", TensorSelection::Full)),
            shape: vec![3],
        };
        assert!(invalid
            .infer(&store as &dyn eredu_checkpoint::store::CheckpointSource)
            .is_err());
        assert!(matches!(
            BindingPlan::new(vec![PlannedBinding::direct(
                "overflow",
                "matrix",
                vec![usize::MAX, 2],
                RecipeDtype::F32,
            )]),
            Err(BindingPlanError::TargetShapeOverflow { .. })
        ));
    }
}
