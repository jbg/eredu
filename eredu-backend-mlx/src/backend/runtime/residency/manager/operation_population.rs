//! Borrowed source counts for one materialization attempt of retained bindings.
//! These are source populations, not a complete request bank or retry bound.
use super::*;
use eredu_checkpoint::recipe::DerivedWeightRecipe;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MaterializationPopulation {
    /// Non-alias destinations. Host fallback creates one recovery per destination.
    pub(crate) bindings: usize,
    /// Source occurrences, including repeated references to the same key.
    pub(crate) pending_weights: usize,
    /// Actual NegLog validation owners in the recursive materializer.
    pub(crate) validation_materializations: usize,
    /// Per recipe invocation, including the actual group retry branches.
    pub(crate) recipe_pending_weight_bound: usize,
    /// Validation and child-detach submissions, including group retries.
    pub(crate) recipe_materialization_bound: usize,
    /// Deepest recipe recursion, including its source node.
    pub(crate) recipe_depth: usize,
}
impl MaterializationPopulation {
    fn add(self, other: Self) -> Option<Self> {
        Some(Self {
            bindings: self.bindings.checked_add(other.bindings)?,
            pending_weights: self.pending_weights.checked_add(other.pending_weights)?,
            validation_materializations: self
                .validation_materializations
                .checked_add(other.validation_materializations)?,
            recipe_pending_weight_bound: self
                .recipe_pending_weight_bound
                .checked_add(other.recipe_pending_weight_bound)?,
            recipe_materialization_bound: self
                .recipe_materialization_bound
                .checked_add(other.recipe_materialization_bound)?,
            recipe_depth: self.recipe_depth.max(other.recipe_depth),
        })
    }
    fn recipe(recipe: &DerivedWeightRecipe) -> Option<Self> {
        use DerivedWeightRecipe::*;
        let mut count = match recipe {
            Source { .. } => Self {
                pending_weights: 1,
                recipe_pending_weight_bound: 1,
                ..Self::default()
            },
            Concatenate { inputs, .. } | Stack { inputs, .. } => {
                let mut count = inputs.iter().try_fold(Self::default(), |total, input| {
                    total.add(Self::recipe(input)?)
                })?;
                // materialize_inputs retries only before its once-only
                // detach_remaining transition. Two copies of each child cover
                // that partial prefix/retry. At this level each successful child
                // has at most one additional source-detach submission.
                count.recipe_pending_weight_bound =
                    count.recipe_pending_weight_bound.checked_mul(2)?;
                count.recipe_materialization_bound = count
                    .recipe_materialization_bound
                    .checked_mul(2)?
                    .checked_add(inputs.len())?;
                count
            }
            NegLog { input } => {
                let mut count = Self::recipe(input)?;
                count.validation_materializations =
                    count.validation_materializations.checked_add(1)?;
                count.recipe_materialization_bound =
                    count.recipe_materialization_bound.checked_add(1)?;
                count
            }
            Select { input, .. }
            | Reshape { input, .. }
            | Transpose { input, .. }
            | Cast { input, .. }
            | View { input, .. }
            | SubtractOne { input } => Self::recipe(input)?,
        };
        count.recipe_depth = count.recipe_depth.checked_add(1)?;
        Some(count)
    }

    /// Reuses borrowed exact declarations; no source-key Vec, cloned recipe,
    /// shape buffer, native object, metadata inference or payload read occurs.
    /// Recipe bounds include internal group retries; raw source/validation
    /// counts remain descriptive occurrences. Manager binding/whole-unit
    /// retries, consumer tickets and repeated forwards are separate caller
    /// contributions. Direct routes may consume fewer slots.
    pub(crate) fn bindings(bindings: &[WeightBinding]) -> Option<Self> {
        bindings
            .iter()
            .filter(|binding| !binding.is_alias())
            .try_fold(Self::default(), |total, binding| {
                let mut one = match binding.recipe() {
                    Some(recipe) => Self::recipe(recipe)?,
                    None => Self {
                        pending_weights: 1,
                        recipe_pending_weight_bound: 1,
                        recipe_depth: 1,
                        ..Self::default()
                    },
                };
                one.bindings = 1;
                total.add(one)
            })
    }
}

impl ResidencyManager {
    /// Cold bounded try-borrow of the actual retained unit declaration. Busy or
    /// arithmetic overflow remains unknown; this does not authorize a producer.
    /// The selected caller must retain this manager/layout identity separately.
    pub(crate) fn unit_operation_population(
        &self,
        id: &OffloadUnitId,
    ) -> Result<Option<MaterializationPopulation>, ResidencyError> {
        let state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ResidencyError::StatePoisoned),
        };
        let unit = state
            .control
            .unit(id)
            .ok_or(ResidencyError::StatePoisoned)?;
        Ok(MaterializationPopulation::bindings(unit.bindings()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{recipe::RecipeDtype, store::TensorSelection};

    #[test]
    fn repeated_sources_and_nested_validation_are_distinct_actual_operations() {
        let source = || DerivedWeightRecipe::source("same", TensorSelection::Full);
        let recipe = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::NegLog {
                    input: Box::new(source()),
                },
                DerivedWeightRecipe::Cast {
                    input: Box::new(DerivedWeightRecipe::NegLog {
                        input: Box::new(source()),
                    }),
                    dtype: RecipeDtype::F32,
                },
            ],
        };
        assert_eq!(
            MaterializationPopulation::recipe(&recipe),
            Some(MaterializationPopulation {
                bindings: 0,
                pending_weights: 2,
                validation_materializations: 2,
                recipe_pending_weight_bound: 4,
                recipe_materialization_bound: 6,
                recipe_depth: 4,
            })
        );
    }

    #[test]
    fn retained_binding_aliases_add_no_source_or_host_materialization_slot() {
        let bindings = [
            WeightBinding::new("owner", "physical", TensorSelection::Full, 16).unwrap(),
            WeightBinding::alias("alias", "owner", 16).unwrap(),
            WeightBinding::from_recipe(
                "derived",
                DerivedWeightRecipe::NegLog {
                    input: Box::new(DerivedWeightRecipe::source(
                        "physical",
                        TensorSelection::Full,
                    )),
                },
                16,
            )
            .unwrap(),
        ];
        assert_eq!(
            MaterializationPopulation::bindings(&bindings),
            Some(MaterializationPopulation {
                bindings: 2,
                pending_weights: 2,
                validation_materializations: 1,
                recipe_pending_weight_bound: 2,
                recipe_materialization_bound: 1,
                recipe_depth: 2,
            })
        );
    }

    #[test]
    fn empty_recipe_group_and_overflow_do_not_invent_source_owners() {
        let recipe = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![],
        };
        let empty = MaterializationPopulation::recipe(&recipe).unwrap();
        assert_eq!(empty.pending_weights, 0);
        assert_eq!(empty.validation_materializations, 0);
        let full = MaterializationPopulation {
            pending_weights: usize::MAX,
            ..empty
        };
        assert!(full
            .add(MaterializationPopulation {
                pending_weights: 1,
                ..empty
            })
            .is_none());
    }
}
