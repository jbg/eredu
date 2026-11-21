//! Exact checkpoint selections and local aliases for atomic parameter materialization.

use std::{borrow::Cow, collections::BTreeMap};

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeError},
    store::{CheckpointSource, TensorSelection},
};

use super::OffloadUnitId;
use crate::checkpoint_mechanisms::CheckpointParameterBackend;

/// One named checkpoint selection within an atomic resident unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightBinding {
    name: String,
    alias_of: Option<String>,
    logical_target: Option<String>,
    checkpoint_key: String,
    selection: TensorSelection,
    recipe: Option<DerivedWeightRecipe>,
    quantization_companions: Option<QuantizationCompanionBindings>,
    expected_bytes: u64,
    storage_layout: eredu_nn::ParameterStorageLayout,
}

/// Exact local outputs created by one load-time packed-weight transform.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct QuantizationCompanionBindings {
    scale: String,
    affine_bias: Option<String>,
}

impl QuantizationCompanionBindings {
    /// Creates the role-keyed local companion identities.
    pub fn new(
        scale: impl Into<String>,
        affine_bias: Option<String>,
    ) -> Result<Self, ResidencyDeclarationError> {
        let scale = validate_name(scale.into())?;
        let affine_bias = affine_bias.map(validate_name).transpose()?;
        if affine_bias.as_ref() == Some(&scale) {
            return Err(ResidencyDeclarationError::InvalidQuantizationCompanions {
                name: "packed weight".into(),
                scales: scale,
                biases: affine_bias.unwrap(),
            });
        }
        Ok(Self { scale, affine_bias })
    }

    /// Returns the required scale binding.
    pub fn scale(&self) -> &str {
        &self.scale
    }
    /// Returns the affine-bias binding when the selected format has one.
    pub fn affine_bias(&self) -> Option<&str> {
        self.affine_bias.as_deref()
    }
}

impl WeightBinding {
    /// Selects an optional physical layout without changing source or byte accounting.
    pub fn with_storage_layout(mut self, layout: eredu_nn::ParameterStorageLayout) -> Self {
        self.storage_layout = layout;
        self
    }

    /// Preferred immutable storage, retaining canonical logical geometry.
    pub const fn storage_layout(&self) -> eredu_nn::ParameterStorageLayout {
        self.storage_layout
    }

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
            alias_of: None,
            logical_target: None,
            checkpoint_key,
            selection,
            recipe: None,
            quantization_companions: None,
            expected_bytes,
            storage_layout: eredu_nn::ParameterStorageLayout::Canonical,
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
            alias_of: None,
            logical_target: None,
            checkpoint_key,
            selection: TensorSelection::Full,
            recipe: Some(recipe),
            quantization_companions: None,
            expected_bytes,
            storage_layout: eredu_nn::ParameterStorageLayout::Canonical,
        })
    }

    /// Creates one logical destination that shares an already materialized
    /// owner binding in the same atomic unit.
    pub fn alias(
        name: impl Into<String>,
        owner: impl Into<String>,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        let name = validate_name(name.into())?;
        let owner = validate_name(owner.into())?;
        validate_size(&name, expected_bytes)?;
        Ok(Self {
            name,
            alias_of: Some(owner),
            logical_target: None,
            checkpoint_key: String::new(),
            selection: TensorSelection::Full,
            recipe: None,
            quantization_companions: None,
            expected_bytes,
            storage_layout: eredu_nn::ParameterStorageLayout::Canonical,
        })
    }

    /// Returns the stable name used to look up a resident value.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the logical owner when this binding is an alias.
    pub fn alias_of(&self) -> Option<&str> {
        self.alias_of.as_deref()
    }

    /// Returns whether this binding reuses another logical binding's native
    /// materialization.
    pub const fn is_alias(&self) -> bool {
        self.alias_of.is_some()
    }

    /// Replaces the stable name used to address this value inside its resident unit.
    pub fn with_name(mut self, name: impl Into<String>) -> Result<Self, ResidencyDeclarationError> {
        self.name = validate_name(name.into())?;
        Ok(self)
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

    /// Declares exact local companion names produced by load-time quantization.
    pub fn with_quantization_companions(
        mut self,
        scales_binding: impl Into<String>,
        biases_binding: Option<String>,
    ) -> Result<Self, ResidencyDeclarationError> {
        let scales_binding = validate_name(scales_binding.into())?;
        let biases_binding = biases_binding.map(validate_name).transpose()?;
        if scales_binding == self.name
            || biases_binding.as_ref() == Some(&self.name)
            || biases_binding.as_ref() == Some(&scales_binding)
        {
            return Err(ResidencyDeclarationError::InvalidQuantizationCompanions {
                name: self.name,
                scales: scales_binding,
                biases: biases_binding.unwrap_or_default(),
            });
        }
        self.quantization_companions = Some(QuantizationCompanionBindings {
            scale: scales_binding,
            affine_bias: biases_binding,
        });
        Ok(self)
    }

    /// Returns exact local scale and affine-bias bindings for load-time quantization.
    pub const fn quantization_companions(&self) -> Option<&QuantizationCompanionBindings> {
        self.quantization_companions.as_ref()
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
        self.recipe_view().into_owned()
    }

    /// Consumes a physical binding and transfers its complete recipe.
    /// Panics for logical aliases, which have no physical source.
    pub fn into_source_recipe(self) -> DerivedWeightRecipe {
        assert!(
            self.alias_of.is_none(),
            "logical aliases have no physical recipe"
        );
        self.recipe
            .unwrap_or_else(|| DerivedWeightRecipe::source(self.checkpoint_key, self.selection))
    }

    /// Borrows a derived recipe or constructs the direct source expression.
    /// Panics for logical aliases, which have no physical source.
    pub fn recipe_view(&self) -> Cow<'_, DerivedWeightRecipe> {
        assert!(
            self.alias_of.is_none(),
            "logical aliases have no physical recipe"
        );
        match &self.recipe {
            Some(recipe) => Cow::Borrowed(recipe),
            None => Cow::Owned(DerivedWeightRecipe::source(
                self.checkpoint_key.clone(),
                self.selection.clone(),
            )),
        }
    }

    /// Returns every checkpoint key consumed by this binding.
    pub fn checkpoint_keys(&self) -> Vec<&str> {
        if self.alias_of.is_some() {
            return Vec::new();
        }
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
        if self.alias_of.is_some() {
            return Err(ResidencyDeclarationError::AliasHasPhysicalSource { name: self.name });
        }
        validate_size(&self.name, expected_bytes)?;
        self.checkpoint_key = first_source(&self.name, &recipe)?;
        self.selection = TensorSelection::Full;
        self.recipe = Some(recipe);
        self.expected_bytes = expected_bytes;
        Ok(self)
    }

    /// Rewrites one logical output selection into bounded physical sources.
    pub fn select_bounded_output<C: RecipeCatalog + ?Sized>(
        self,
        catalog: &C,
        selection: TensorSelection,
    ) -> Result<Self, WeightBindingSelectionError> {
        let recipe = self.recipe_view().select_bounded(catalog, selection)?;
        let bytes = recipe.infer(catalog)?.byte_len();
        Ok(self.with_source_recipe(recipe, bytes)?)
    }
}

/// Checked executable parameter payload for a binding selection.
///
/// Logical aliases add no storage. Distinct physical owners remain separately
/// charged even when their checkpoint keys or recipes are equal. Source payload,
/// loading scratch and native host capacity are separate estimates. This does not
/// validate alias topology; a selection may include aliases to another resource.
pub fn physical_binding_bytes(
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyDeclarationError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                ResidencyDeclarationError::BindingBytesOverflow {
                    target: binding
                        .logical_target()
                        .unwrap_or_else(|| binding.name())
                        .into(),
                }
            })
        })
}

/// Validated owner/alias partition for one atomic binding unit.
#[derive(Debug)]
pub struct WeightBindingPlan<'a> {
    bindings: Cow<'a, [WeightBinding]>,
    owners: Vec<usize>,
    aliases: Vec<(usize, usize)>,
}

impl<'a> WeightBindingPlan<'a> {
    /// Validates unique identities, owner existence, cycles, and byte geometry.
    pub fn new(bindings: &'a [WeightBinding]) -> Result<Self, ResidencyDeclarationError> {
        Self::build(Cow::Borrowed(bindings))
    }

    /// Retains validated declarations and their partition for deferred loading.
    pub fn owned(
        bindings: Vec<WeightBinding>,
    ) -> Result<WeightBindingPlan<'static>, ResidencyDeclarationError> {
        WeightBindingPlan::build(Cow::Owned(bindings))
    }

    fn build(bindings: Cow<'a, [WeightBinding]>) -> Result<Self, ResidencyDeclarationError> {
        let mut by_name = BTreeMap::new();
        for (index, binding) in bindings.iter().enumerate() {
            if by_name.insert(binding.name(), index).is_some() {
                return Err(ResidencyDeclarationError::DuplicateLogicalBinding {
                    name: binding.name().into(),
                });
            }
        }
        fn resolve(
            index: usize,
            bindings: &[WeightBinding],
            by_name: &BTreeMap<&str, usize>,
            resolved: &mut [Option<usize>],
            visiting: &mut [bool],
        ) -> Result<usize, ResidencyDeclarationError> {
            if let Some(owner) = resolved[index] {
                return Ok(owner);
            }
            let binding = &bindings[index];
            if visiting[index] {
                return Err(ResidencyDeclarationError::BindingAliasCycle {
                    name: binding.name().into(),
                });
            }
            visiting[index] = true;
            let owner = match binding.alias_of() {
                None => index,
                Some(name) => {
                    let owner = *by_name.get(name).ok_or_else(|| {
                        ResidencyDeclarationError::UnknownBindingAliasOwner {
                            alias: binding.name().into(),
                            owner: name.into(),
                        }
                    })?;
                    resolve(owner, bindings, by_name, resolved, visiting)?
                }
            };
            visiting[index] = false;
            resolved[index] = Some(owner);
            Ok(owner)
        }
        let mut resolved = vec![None; bindings.len()];
        let mut visiting = vec![false; bindings.len()];
        let mut owners = Vec::new();
        let mut aliases = Vec::new();
        for (index, binding) in bindings.iter().enumerate() {
            let owner = resolve(index, &bindings, &by_name, &mut resolved, &mut visiting)?;
            if owner == index {
                owners.push(index);
            } else {
                if binding.expected_bytes() != bindings[owner].expected_bytes() {
                    return Err(ResidencyDeclarationError::BindingAliasByteMismatch {
                        alias: binding.name().into(),
                        owner: bindings[owner].name().into(),
                        alias_bytes: binding.expected_bytes(),
                        owner_bytes: bindings[owner].expected_bytes(),
                    });
                }
                aliases.push((index, owner));
            }
        }
        Ok(Self {
            bindings,
            owners,
            aliases,
        })
    }

    /// Checks source geometry and native recipe support without reading payloads.
    pub fn validate_source<B: CheckpointParameterBackend>(
        self,
        source: &dyn CheckpointSource,
    ) -> Result<Self, BindingPreflightError<B::ParameterError>> {
        for binding in self.owners() {
            let actual = binding.recipe_view().infer(source)?.byte_len();
            if actual != binding.expected_bytes() {
                return Err(BindingPreflightError::ByteMismatch {
                    parameter: binding.name().to_owned(),
                    expected: binding.expected_bytes(),
                    actual,
                });
            }
        }
        for binding in self.owners() {
            B::preflight_recipe(&binding.recipe_view(), source)
                .map_err(BindingPreflightError::Backend)?;
        }
        Ok(self)
    }

    /// Canonical bindings which require physical materialization.
    pub fn owners(&self) -> impl Iterator<Item = &WeightBinding> {
        self.owners.iter().map(|index| &self.bindings[*index])
    }

    /// Logical aliases paired with their resolved canonical owners.
    pub fn aliases(&self) -> impl Iterator<Item = (&WeightBinding, &WeightBinding)> {
        self.alias_indices()
            .map(|(alias, owner)| (&self.bindings[alias], &self.bindings[owner]))
    }

    /// Alias indices paired with their physical owner indices in binding order.
    pub fn alias_indices(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.aliases.iter().copied()
    }
}

/// Validates a complete binding unit before acquiring a checkpoint lease.
pub fn preflight_bindings<'a, B: CheckpointParameterBackend>(
    source: &dyn CheckpointSource,
    bindings: &'a [WeightBinding],
) -> Result<WeightBindingPlan<'a>, BindingPreflightError<B::ParameterError>> {
    WeightBindingPlan::new(bindings)?.validate_source::<B>(source)
}

/// Failure validating exact checkpoint binding inputs.
#[derive(Debug, thiserror::Error)]
pub enum BindingPreflightError<E: std::error::Error + 'static> {
    /// Binding declarations or local aliases are invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
    /// The checkpoint recipe cannot be inferred.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Inferred and declared materialized byte sizes disagreed.
    #[error("parameter {parameter:?} declares {expected} bytes but its recipe produces {actual}")]
    ByteMismatch {
        /// Stable binding name.
        parameter: String,
        /// Declared bytes.
        expected: u64,
        /// Inferred bytes.
        actual: u64,
    },
    /// The native backend does not support the exact recipe.
    #[error("backend parameter operation failed: {0}")]
    Backend(#[source] E),
}

/// Failure while rewriting a binding into bounded physical selections.
#[derive(Debug, thiserror::Error)]
pub enum WeightBindingSelectionError {
    /// Neutral recipe inference or selection pushdown failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// The rewritten binding declaration was invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
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
    /// Load-time quantization companion names collide with one another.
    #[error(
        "weight binding {name:?} has invalid quantization companions {scales:?} and {biases:?}"
    )]
    InvalidQuantizationCompanions {
        /// Quantizable weight binding.
        name: String,
        /// Declared scale binding.
        scales: String,
        /// Declared affine-bias binding.
        biases: String,
    },
    /// A binding declared no bytes.
    #[error("weight binding {name:?} must contain at least one byte")]
    ZeroSizedBinding {
        /// Invalid local name.
        name: String,
    },
    /// Executable payload accounting exceeded its representable byte count.
    #[error("physical binding byte total overflows at parameter {target:?}")]
    BindingBytesOverflow {
        /// Canonical destination when present, otherwise the module-local name.
        target: String,
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
    /// A general binding list repeated one logical identity.
    #[error("duplicate logical weight binding {name:?}")]
    DuplicateLogicalBinding {
        /// Repeated logical identity.
        name: String,
    },
    /// An alias named no binding in its atomic unit.
    #[error("weight binding alias {alias:?} has unknown owner {owner:?}")]
    UnknownBindingAliasOwner {
        /// Alias identity.
        alias: String,
        /// Missing owner identity.
        owner: String,
    },
    /// An alias destination did not identify one unique global owner.
    #[error("weight binding alias {alias:?} has ambiguous owner {owner:?}")]
    AmbiguousBindingAliasOwner {
        /// Alias identity.
        alias: String,
        /// Ambiguous owner identity.
        owner: String,
    },
    /// Alias declarations formed a cycle.
    #[error("weight binding alias cycle contains {name:?}")]
    BindingAliasCycle {
        /// One member of the cycle.
        name: String,
    },
    /// Alias and resolved owner disagreed on materialized byte geometry.
    #[error(
        "weight binding alias {alias:?} declares {alias_bytes} bytes but owner {owner:?} declares {owner_bytes}"
    )]
    BindingAliasByteMismatch {
        /// Alias identity.
        alias: String,
        /// Resolved canonical owner.
        owner: String,
        /// Alias byte declaration.
        alias_bytes: u64,
        /// Owner byte declaration.
        owner_bytes: u64,
    },
    /// An alias was incorrectly rewritten with a physical source.
    #[error("weight binding alias {name:?} cannot own a physical checkpoint source")]
    AliasHasPhysicalSource {
        /// Alias identity.
        name: String,
    },
}
