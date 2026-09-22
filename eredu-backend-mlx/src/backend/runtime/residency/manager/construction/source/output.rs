//! Output declarations stay separate from the detached source-read ordinals.
use super::*;
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};

pub(super) struct BindingOutput {
    name: String,
    value: OutputValue,
}
enum OutputValue {
    Direct(usize),
    Materialized {
        metadata: RecipeMetadata,
        range: Range<usize>,
        leaves: Vec<LeafIdentity>,
    },
}
struct LeafIdentity {
    key: String,
    selection: TensorSelection,
}

impl BindingOutput {
    pub(super) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<OutputValue>(),
            size_of::<LeafIdentity>(),
            size_of::<RecipeMetadata>(),
            size_of::<Range<usize>>(),
            size_of::<Result<Self, ConstructionCause>>(),
            size_of::<(&PlannedRead, &str, Range<usize>)>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(super) fn name(&self) -> &str {
        &self.name
    }
    pub(super) fn metadata<'a>(
        &'a self,
        reads: &'a DetachedEncodedReads<ManagerCustody>,
    ) -> Option<&'a RecipeMetadata> {
        match &self.value {
            OutputValue::Direct(index) => reads.output(*index),
            OutputValue::Materialized { metadata, .. } => Some(metadata),
        }
    }
    pub(super) fn matches(
        &self,
        binding: &WeightBinding,
        source: &dyn CheckpointSource,
        reads: &DetachedEncodedReads<ManagerCustody>,
    ) -> Result<bool, WeightRecipeError> {
        if self.name != binding.name() {
            return Ok(false);
        }
        let recipe = binding.source_recipe();
        match &self.value {
            OutputValue::Direct(index) => Ok(DirectRecipeRead::prepare(&recipe, source)?
                .is_some_and(|read| reads.matches_read(*index, read.encoded()))),
            OutputValue::Materialized {
                metadata,
                range,
                leaves,
            } => {
                if recipe.infer(source)? != *metadata || range.len() != leaves.len() {
                    return Ok(false);
                }
                for (index, leaf) in range.clone().zip(leaves) {
                    let recipe = DerivedWeightRecipe::Source {
                        key: leaf.key.clone(),
                        selection: leaf.selection.clone(),
                    };
                    let Some(read) = DirectRecipeRead::prepare(&recipe, source)? else {
                        return Ok(false);
                    };
                    if !reads.matches_read(index, read.encoded()) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }
    pub(super) fn payload_bytes(row: &PlannedRead, name: &str) -> Option<usize> {
        let mut bytes = name.len();
        if let read_source_plan::ReadValue::Materialized(plan) = &row.read {
            bytes = bytes
                .checked_add(Layout::array::<usize>(plan.output.shape.len()).ok()?.size())?
                .checked_add(match &plan.output.dtype {
                    eredu_checkpoint::recipe::RecipeDtype::Other(name) => name.len(),
                    _ => 0,
                })?
                .checked_add(
                    Layout::array::<LeafIdentity>(plan.leaves.len())
                        .ok()?
                        .size(),
                )?;
            for leaf in &plan.leaves {
                bytes = bytes
                    .checked_add(leaf.key.len())?
                    .checked_add(selection_bytes(&leaf.selection)?)?;
            }
        }
        Some(bytes)
    }
    pub(super) fn construct(
        row: &PlannedRead,
        name: &str,
        range: Range<usize>,
    ) -> Result<Self, ConstructionCause> {
        if range.len() != row.read.leaf_count() {
            return Err(ResidencyError::StatePoisoned.into());
        }
        let value = match &row.read {
            read_source_plan::ReadValue::Direct(_) => OutputValue::Direct(range.start),
            read_source_plan::ReadValue::Materialized(plan) => {
                let mut leaves = Vec::new();
                leaves.try_reserve_exact(plan.leaves.len())?;
                for leaf in &plan.leaves {
                    leaves.push(LeafIdentity {
                        key: leaf.key.clone(),
                        selection: leaf.selection.clone(),
                    });
                }
                OutputValue::Materialized {
                    metadata: plan.output.clone(),
                    range,
                    leaves,
                }
            }
        };
        Ok(Self {
            name: name.to_owned(),
            value,
        })
    }
}

fn selection_bytes(selection: &TensorSelection) -> Option<usize> {
    let count = match selection {
        TensorSelection::Indices { indices, .. } => indices.len(),
        TensorSelection::Contiguous { shape, .. } => shape.len(),
        TensorSelection::Full | TensorSelection::Range { .. } => 0,
    };
    Some(Layout::array::<usize>(count).ok()?.size())
}
