//! Backend-neutral immutable-weight residency declarations and control state.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeError},
    store::TensorSelection,
};
use eredu_core::residency::{OffloadPlan, OffloadUnitId, ResidencyLedger};

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

/// Validated backend-neutral declarations paired with residency control state.
///
/// Concrete backends own their native values separately and mirror every
/// publication or eviction through this controller's ledger. Keeping the
/// declarations here ensures checkpoint shape policy and plan identity are
/// validated before any backend allocation begins.
#[derive(Debug)]
pub struct ResidencyController {
    ledger: ResidencyLedger,
    units: BTreeMap<OffloadUnitId, OffloadUnit>,
}

impl ResidencyController {
    /// Validates declarations against checkpoint metadata and an explicit plan.
    pub fn new<C: RecipeCatalog + ?Sized>(
        catalog: &C,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
    ) -> Result<Self, ResidencyControllerError> {
        let mut definitions = BTreeMap::new();
        for unit in units {
            let id = unit.id().clone();
            if definitions.insert(id.clone(), unit).is_some() {
                return Err(ResidencyControllerError::DuplicateUnitDefinition { id });
            }
        }
        for spec in plan.units() {
            if !definitions.contains_key(spec.id()) {
                return Err(ResidencyControllerError::MissingUnitDefinition {
                    id: spec.id().clone(),
                });
            }
        }
        if let Some(id) = definitions
            .keys()
            .find(|id| plan.unit(id).is_none())
            .cloned()
        {
            return Err(ResidencyControllerError::UnexpectedUnitDefinition { id });
        }

        for spec in plan.units() {
            let unit = definitions
                .get(spec.id())
                .expect("definition identity validated above");
            let mut total = 0u64;
            for binding in unit.bindings() {
                total = total.checked_add(binding.expected_bytes()).ok_or(
                    ResidencyControllerError::ArithmeticOverflow {
                        context: "unit binding byte total",
                    },
                )?;
                let actual = binding
                    .source_recipe()
                    .infer(catalog)
                    .map_err(|source| ResidencyControllerError::Recipe {
                        binding: binding.name().to_owned(),
                        source,
                    })?
                    .byte_len();
                if actual != binding.expected_bytes() {
                    return Err(ResidencyControllerError::BindingByteMismatch {
                        id: unit.id().clone(),
                        binding: binding.name().to_owned(),
                        expected_bytes: binding.expected_bytes(),
                        actual_bytes: actual,
                    });
                }
            }
            if total != spec.bytes() {
                return Err(ResidencyControllerError::UnitByteMismatch {
                    id: unit.id().clone(),
                    planned_bytes: spec.bytes(),
                    actual_bytes: total,
                });
            }
        }

        Ok(Self {
            ledger: ResidencyLedger::new(plan),
            units: definitions,
        })
    }

    /// Returns the validated declaration for one planned unit.
    pub fn unit(&self, id: &OffloadUnitId) -> Option<&OffloadUnit> {
        self.units.get(id)
    }

    /// Returns declarations in stable unit-identifier order.
    pub fn units(&self) -> impl ExactSizeIterator<Item = &OffloadUnit> {
        self.units.values()
    }

    /// Returns immutable ownership, capacity, and telemetry state.
    pub const fn ledger(&self) -> &ResidencyLedger {
        &self.ledger
    }

    /// Returns mutable ownership, capacity, and telemetry state.
    pub fn ledger_mut(&mut self) -> &mut ResidencyLedger {
        &mut self.ledger
    }
}

/// Failure while validating a residency control plane.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyControllerError {
    /// More than one definition used the same plan identifier.
    #[error("duplicate residency unit definition: {id}")]
    DuplicateUnitDefinition {
        /// Duplicated identifier.
        id: OffloadUnitId,
    },
    /// The plan had no matching unit definition.
    #[error("offload plan unit {id} has no residency unit definition")]
    MissingUnitDefinition {
        /// Missing identifier.
        id: OffloadUnitId,
    },
    /// A definition had no matching plan entry.
    #[error("residency unit {id} is absent from the offload plan")]
    UnexpectedUnitDefinition {
        /// Unexpected identifier.
        id: OffloadUnitId,
    },
    /// Binding sizes did not sum to the plan's unit size.
    #[error(
        "residency unit {id} defines {actual_bytes} bytes but its plan reserves {planned_bytes}"
    )]
    UnitByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Bytes reserved by the plan.
        planned_bytes: u64,
        /// Sum of binding sizes.
        actual_bytes: u64,
    },
    /// A binding's selected checkpoint size contradicted its declaration.
    #[error(
        "binding {binding:?} in unit {id} selects {actual_bytes} bytes but declares {expected_bytes}"
    )]
    BindingByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Binding name.
        binding: String,
        /// Declared size.
        expected_bytes: u64,
        /// Catalog-validated size.
        actual_bytes: u64,
    },
    /// A derived-weight recipe was invalid.
    #[error("derived-weight recipe for binding {binding:?} failed: {source}")]
    Recipe {
        /// Local binding name.
        binding: String,
        /// Invalid recipe.
        #[source]
        source: RecipeError,
    },
    /// Checked byte arithmetic overflowed.
    #[error("residency arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: &'static str,
    },
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
    use eredu_checkpoint::{store::TensorMetadata, StoredDtype};
    use eredu_core::residency::{MemoryTier, OffloadConfig, OffloadUnitSpec, ResidencyPolicy};

    use super::*;

    struct Catalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(
            &self,
            key: &str,
        ) -> Result<TensorMetadata, eredu_checkpoint::store::StoreError> {
            self.0.get(key).cloned().ok_or_else(|| {
                eredu_checkpoint::store::StoreError::UnknownTensor {
                    key: key.to_owned(),
                }
            })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            name: name.to_owned(),
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: StoredDtype::F32,
            encoded_byte_len: 0,
            backing_shard: None,
        }
    }

    #[test]
    fn declarations_are_validated_and_deterministic() {
        let b = WeightBinding::new("b", "b.weight", TensorSelection::Full, 4).unwrap();
        let a = WeightBinding::new("a", "a.weight", TensorSelection::Full, 8).unwrap();
        let id = OffloadUnitId::new("layer.0").unwrap();
        let unit = OffloadUnit::new(id, [b, a]).unwrap();
        assert_eq!(unit.bindings()[0].name(), "a");
        assert_eq!(unit.bindings()[1].name(), "b");
    }

    #[test]
    fn controller_validates_catalog_bytes_before_allocating_backend_storage() {
        let catalog = Catalog(BTreeMap::from([
            ("a.weight".into(), metadata("a.weight", vec![2])),
            ("b.weight".into(), metadata("b.weight", vec![1])),
        ]));
        let id = OffloadUnitId::new("layer.0").unwrap();
        let unit = OffloadUnit::new(
            id.clone(),
            [
                WeightBinding::new("a", "a.weight", TensorSelection::Full, 8).unwrap(),
                WeightBinding::new("b", "b.weight", TensorSelection::Full, 4).unwrap(),
            ],
        )
        .unwrap();
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [
                OffloadUnitSpec::new(id.clone(), 12, ResidencyPolicy::Windowed, MemoryTier::Disk)
                    .unwrap(),
            ],
        )
        .unwrap();

        let controller = ResidencyController::new(&catalog, plan, [unit]).unwrap();
        assert_eq!(controller.units().len(), 1);
        assert_eq!(controller.unit(&id).unwrap().bindings().len(), 2);
        assert!(!controller.ledger().initialized());
    }

    #[test]
    fn controller_rejects_binding_and_plan_byte_mismatches() {
        let catalog = Catalog(BTreeMap::from([(
            "weight".into(),
            metadata("weight", vec![2]),
        )]));
        let id = OffloadUnitId::new("layer.0").unwrap();
        let plan = |bytes| {
            OffloadPlan::new(
                OffloadConfig::default(),
                [OffloadUnitSpec::new(
                    id.clone(),
                    bytes,
                    ResidencyPolicy::Windowed,
                    MemoryTier::Disk,
                )
                .unwrap()],
            )
            .unwrap()
        };

        let wrong_binding = OffloadUnit::new(
            id.clone(),
            [WeightBinding::new("weight", "weight", TensorSelection::Full, 4).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            ResidencyController::new(&catalog, plan(4), [wrong_binding]),
            Err(ResidencyControllerError::BindingByteMismatch { .. })
        ));

        let wrong_plan = OffloadUnit::new(
            id.clone(),
            [WeightBinding::new("weight", "weight", TensorSelection::Full, 8).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            ResidencyController::new(&catalog, plan(16), [wrong_plan]),
            Err(ResidencyControllerError::UnitByteMismatch { .. })
        ));
    }
}
