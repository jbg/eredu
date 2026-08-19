//! Backend-neutral immutable-weight residency declarations.

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_core::residency::OffloadUnitId;

/// One named checkpoint selection within an atomic resident unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightBinding {
    name: String,
    logical_target: Option<String>,
    checkpoint_key: String,
    selection: TensorSelection,
    recipe: Option<DerivedWeightRecipe>,
    expected_bytes: u64,
}

impl WeightBinding {
    /// Creates a direct binding with a stable local name and selected size.
    pub fn new(
        name: impl Into<String>,
        checkpoint_key: impl Into<String>,
        selection: TensorSelection,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        let name = validate_name(name.into())?;
        let checkpoint_key = checkpoint_key.into();
        if checkpoint_key.trim().is_empty() {
            return Err(ResidencyDeclarationError::InvalidCheckpointKey { name });
        }
        validate_size(&name, expected_bytes)?;
        Ok(Self {
            name,
            logical_target: None,
            checkpoint_key,
            selection,
            recipe: None,
            expected_bytes,
        })
    }

    /// Creates a binding backed by a composable derived-weight recipe.
    pub fn from_recipe(
        name: impl Into<String>,
        recipe: DerivedWeightRecipe,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        let name = validate_name(name.into())?;
        validate_size(&name, expected_bytes)?;
        let checkpoint_key = first_source(&name, &recipe)?;
        Ok(Self {
            name,
            logical_target: None,
            checkpoint_key,
            selection: TensorSelection::Full,
            recipe: Some(recipe),
            expected_bytes,
        })
    }

    /// Returns the stable name used to look up a resident value.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the architecture-logical parameter destination.
    pub fn logical_target(&self) -> Option<&str> {
        self.logical_target.as_deref()
    }

    /// Attaches the architecture-logical parameter destination.
    pub fn with_logical_target(
        mut self,
        target: impl Into<String>,
    ) -> Result<Self, ResidencyDeclarationError> {
        self.logical_target = Some(validate_name(target.into())?);
        Ok(self)
    }

    /// Returns the first physical checkpoint source.
    pub fn checkpoint_key(&self) -> &str {
        &self.checkpoint_key
    }

    /// Returns the direct checkpoint selection.
    pub fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Returns the derived recipe when this is not a direct binding.
    pub const fn recipe(&self) -> Option<&DerivedWeightRecipe> {
        self.recipe.as_ref()
    }

    /// Returns the complete source recipe represented by this binding.
    pub fn source_recipe(&self) -> DerivedWeightRecipe {
        self.recipe.clone().unwrap_or_else(|| {
            DerivedWeightRecipe::source(self.checkpoint_key.clone(), self.selection.clone())
        })
    }

    /// Returns every checkpoint key consumed by this binding.
    pub fn checkpoint_keys(&self) -> Vec<&str> {
        self.recipe.as_ref().map_or_else(
            || vec![self.checkpoint_key.as_str()],
            DerivedWeightRecipe::source_keys,
        )
    }

    /// Returns the exact logical materialized byte length.
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }

    /// Replaces the physical source with an equivalent validated recipe.
    pub fn with_source_recipe(
        mut self,
        recipe: DerivedWeightRecipe,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        validate_size(&self.name, expected_bytes)?;
        self.checkpoint_key = first_source(&self.name, &recipe)?;
        self.selection = TensorSelection::Full;
        self.recipe = Some(recipe);
        self.expected_bytes = expected_bytes;
        Ok(self)
    }
}

/// A deterministic group of weight bindings managed as one atomic unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadUnit {
    id: OffloadUnitId,
    bindings: Vec<WeightBinding>,
}

impl OffloadUnit {
    /// Creates a non-empty unit and sorts bindings by local name.
    pub fn new(
        id: OffloadUnitId,
        bindings: impl IntoIterator<Item = WeightBinding>,
    ) -> Result<Self, ResidencyDeclarationError> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        if bindings.is_empty() {
            return Err(ResidencyDeclarationError::EmptyUnit { id });
        }
        bindings.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = bindings
            .windows(2)
            .find(|pair| pair[0].name == pair[1].name)
        {
            return Err(ResidencyDeclarationError::DuplicateBindingName {
                id,
                name: pair[0].name.clone(),
            });
        }
        Ok(Self { id, bindings })
    }

    /// Returns the plan identifier for this unit.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns bindings in stable local-name order.
    pub fn bindings(&self) -> &[WeightBinding] {
        &self.bindings
    }
}

fn validate_name(name: String) -> Result<String, ResidencyDeclarationError> {
    if name.trim().is_empty() {
        Err(ResidencyDeclarationError::InvalidBindingName)
    } else {
        Ok(name)
    }
}

fn validate_size(name: &str, expected_bytes: u64) -> Result<(), ResidencyDeclarationError> {
    if expected_bytes == 0 {
        Err(ResidencyDeclarationError::ZeroSizedBinding {
            name: name.to_owned(),
        })
    } else {
        Ok(())
    }
}

fn first_source(
    name: &str,
    recipe: &DerivedWeightRecipe,
) -> Result<String, ResidencyDeclarationError> {
    recipe
        .source_keys()
        .first()
        .map(|key| (*key).to_owned())
        .ok_or_else(|| ResidencyDeclarationError::EmptyRecipeSources {
            name: name.to_owned(),
        })
}

/// Invalid backend-neutral residency declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ResidencyDeclarationError {
    /// A binding name was empty.
    #[error("weight binding names must not be empty")]
    InvalidBindingName,
    /// A binding checkpoint key was empty.
    #[error("weight binding {name:?} has an empty checkpoint key")]
    InvalidCheckpointKey {
        /// Invalid local name.
        name: String,
    },
    /// A recipe had no physical source.
    #[error("weight binding {name:?} has no checkpoint recipe source")]
    EmptyRecipeSources {
        /// Invalid local name.
        name: String,
    },
    /// A binding declared no bytes.
    #[error("weight binding {name:?} must contain at least one byte")]
    ZeroSizedBinding {
        /// Invalid local name.
        name: String,
    },
    /// A unit had no bindings.
    #[error("residency unit {id} must contain at least one binding")]
    EmptyUnit {
        /// Unit identifier.
        id: OffloadUnitId,
    },
    /// Two bindings in one unit had the same local name.
    #[error("residency unit {id} has duplicate binding name {name:?}")]
    DuplicateBindingName {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Duplicated local name.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations_are_validated_and_deterministic() {
        let b = WeightBinding::new("b", "b.weight", TensorSelection::Full, 4).unwrap();
        let a = WeightBinding::new("a", "a.weight", TensorSelection::Full, 8).unwrap();
        let id = OffloadUnitId::new("layer.0").unwrap();
        let unit = OffloadUnit::new(id, [b, a]).unwrap();
        assert_eq!(unit.bindings()[0].name(), "a");
        assert_eq!(unit.bindings()[1].name(), "b");
    }
}
