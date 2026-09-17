//! Finite owning construction from exact borrowed neutral declarations.
use super::*;
use eredu_checkpoint::recipe::{
    infer_recipe_bytes, RecipeInferenceError, RecipeInferenceInput, RecipeInferenceLayout,
};
use eredu_core::residency::{OffloadUnitSpec, ResidencyLedgerStorageLayout};
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

/// Source-derived constructor qualification, without allocating an owned diagnostic.
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum ResidencyConstructionError {
    /// A source catalog cannot lend the metadata used by this exact recipe.
    #[error("residency recipe has no finite borrowed metadata constructor")]
    MetadataUnavailable,
    /// An actual array/string/box layout overflowed, or group names were invalid.
    #[error("invalid residency constructor storage layout")]
    Layout,
}
/// A move-only constructor over its actual plan, unit declarations and catalogs.
/// The caller admits `required_bytes` before consuming it; no native authority is supplied.
pub struct ResidencyControllerPlan<'a, C: RecipeCatalog + ?Sized, F, G: AsRef<str>> {
    catalog: F,
    plan: &'a OffloadPlan,
    units: &'a [OffloadUnit],
    groups: &'a [G],
    bytes: usize,
    marker: std::marker::PhantomData<&'a C>,
}
impl<C: RecipeCatalog + ?Sized, F, G: AsRef<str>> std::fmt::Debug
    for ResidencyControllerPlan<'_, C, F, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidencyControllerPlan")
            .field("required_bytes", &self.bytes)
            .finish()
    }
}
impl<'a, C: RecipeCatalog + ?Sized + 'a, F: Fn(&OffloadUnitId) -> &'a C, G: AsRef<str>>
    ResidencyControllerPlan<'a, C, F, G>
{
    /// All requested clone/index/ledger/inference storage and constructor controls.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Validate the exact plan/unit identifier domain before constructing a
    /// different consumer of these declarations. This shares the ordinary
    /// controller's error order. It does not validate aliases, binding byte
    /// geometry or source recipes, and grants no construction authority.
    /// Successful validation allocates nothing; an error owns its rejected ID.
    pub fn validate_unit_definitions(&self) -> Result<(), ResidencyControllerError> {
        validate_unique_units(self.units)?;
        validate_plan_units(self.plan, self.units, |id| {
            self.units.iter().any(|unit| unit.id() == id)
        })
    }
    /// Borrows an alias's exact canonical declaration before owned construction.
    /// Uses the constructor's alias identity, cycle and byte-agreement rules;
    /// missing, ambiguous or invalid declarations return None without allocating
    /// an error or consulting a catalog. Non-alias bindings return None.
    pub fn binding_owner(
        &self,
        unit: &OffloadUnitId,
        binding: &WeightBinding,
    ) -> Option<(&'a OffloadUnitId, &'a WeightBinding)> {
        if !binding.is_alias() {
            return None;
        }
        let mut matches = self
            .units
            .iter()
            .enumerate()
            .filter(|(_, row)| row.id() == unit);
        let (unit, row) = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        let binding = row
            .bindings
            .iter()
            .position(|candidate| candidate == binding)?;
        let total = self
            .units
            .iter()
            .try_fold(0usize, |n, unit| n.checked_add(unit.bindings.len()))?;
        let mut steps = 0;
        let owner = resolve_alias(
            self.units,
            BindingLocation { unit, binding },
            |destination| {
                let mut matches = self.units.iter().enumerate().flat_map(|(unit, row)| {
                    row.bindings
                        .iter()
                        .enumerate()
                        .filter_map(move |(binding, value)| {
                            (alias_identity(value) == destination)
                                .then_some(BindingLocation { unit, binding })
                        })
                });
                match (matches.next(), matches.next()) {
                    (None, _) => Err(AliasLookupFailure::Missing),
                    (Some(_), Some(_)) => Err(AliasLookupFailure::Ambiguous),
                    (Some(owner), None) => Ok(owner),
                }
            },
            |_| {
                // A valid chain visits at most the actual declaration count.
                // No recursive stack or visited-vector storage is needed here.
                if steps == total {
                    return true;
                }
                steps += 1;
                false
            },
        )
        .ok()?;
        Some((&self.units[owner.unit].id, at(self.units, owner)))
    }
    /// Computes the same canonical unit closure before owned construction.
    /// Roots may borrow a tier-filtered plan iterator; no ID vector, catalog
    /// query or owned alias diagnostic is constructed. Returned units preserve
    /// the original input order, and scratch has one slot per input unit.
    pub fn operation_closure<'s, 'r>(
        &'s self,
        roots: impl Iterator<Item = &'r OffloadUnitId> + Clone,
        scratch: &'s mut [ResidencyClosureSlot],
    ) -> Result<ResidencyClosure<'s>, ResidencyClosureError> {
        super::operation_source::closure_with(self.units, roots, scratch, |from, visit| {
            let unit = &self.units[from];
            for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
                let (owner, _) = self
                    .binding_owner(unit.id(), binding)
                    .ok_or(ResidencyClosureError::InvalidOwner)?;
                let ordinal = self
                    .units
                    .iter()
                    .position(|unit| std::ptr::eq(unit.id(), owner))
                    .ok_or(ResidencyClosureError::InvalidOwner)?;
                visit(ordinal)?;
            }
            Ok(())
        })
    }
    /// Clones the borrowed inputs only after the caller admits this plan. Partial
    /// storage and inference prefixes retire before its error escapes the caller's owner.
    pub fn construct(self) -> Result<ResidencyController, ResidencyControllerError> {
        let plan = self.plan.clone();
        let units = self.units.to_vec();
        build(
            self.catalog,
            plan,
            units,
            Some(self.groups),
            |binding, catalog| match infer_recipe_bytes(input(binding), catalog) {
                Ok(bytes) => Ok(bytes),
                Err(RecipeInferenceError::Recipe(source)) => {
                    Err(ResidencyControllerError::Recipe {
                        binding: binding.name.clone(),
                        source,
                    })
                }
                Err(error) => Err(ResidencyControllerError::FiniteInference(error)),
            },
        )
    }
}
impl ResidencyController {
    /// Inspects actual borrowed metadata and finite declared windows without
    /// creating a recipe, adding cache entries or constructing any destination.
    pub fn prepare_constructor<
        'a,
        C: RecipeCatalog + ?Sized + 'a,
        F: Fn(&OffloadUnitId) -> &'a C,
        G: AsRef<str>,
    >(
        catalog: F,
        plan: &'a OffloadPlan,
        units: &'a [OffloadUnit],
        groups: &'a [G],
    ) -> Result<ResidencyControllerPlan<'a, C, F, G>, ResidencyConstructionError> {
        let layout = || ResidencyConstructionError::Layout;
        let mut owned = Layout::array::<OffloadUnitSpec>(plan.units().len())
            .map_err(|_| layout())?
            .size();
        for spec in plan.units() {
            owned = owned
                .checked_add(spec.id().as_str().len())
                .ok_or_else(layout)?;
        }
        owned = owned
            .checked_add(
                Layout::array::<OffloadUnit>(units.len())
                    .map_err(|_| layout())?
                    .size(),
            )
            .ok_or_else(layout)?;
        let mut bindings = 0usize;
        let mut aliases = 0usize;
        let mut inference = 0usize;
        let mut diagnostic_text = plan
            .units()
            .iter()
            .map(|spec| spec.id().as_str().len())
            .max()
            .unwrap_or(0);
        for unit in units {
            owned = owned
                .checked_add(unit.id.as_str().len())
                .and_then(|n| {
                    n.checked_add(
                        Layout::array::<WeightBinding>(unit.bindings.len())
                            .ok()?
                            .size(),
                    )
                })
                .ok_or_else(layout)?;
            diagnostic_text = diagnostic_text.max(unit.id.as_str().len());
            for binding in &unit.bindings {
                bindings = bindings.checked_add(1).ok_or_else(layout)?;
                aliases = aliases
                    .checked_add(usize::from(binding.is_alias()))
                    .ok_or_else(layout)?;
                let payload = binding_clone_bytes(binding).ok_or_else(layout)?;
                owned = owned.checked_add(payload).ok_or_else(layout)?;
                diagnostic_text = diagnostic_text
                    .max(binding.name.len())
                    .max(binding.alias_of.as_ref().map_or(0, String::len));
                if !binding.is_alias() {
                    inference = inference
                        .checked_add(
                            RecipeInferenceLayout::inspect(input(binding), catalog(unit.id()))
                                .ok_or(ResidencyConstructionError::MetadataUnavailable)?
                                .required_bytes(),
                        )
                        .ok_or_else(layout)?;
                }
            }
        }
        let mut bytes = owned
            .checked_add(inference)
            .and_then(|n| n.checked_add(diagnostic_text.checked_mul(3)?))
            .ok_or_else(layout)?;
        for allocation in [
            Layout::array::<BindingLocation>(bindings),
            Layout::array::<usize>(bindings),
            Layout::array::<bool>(bindings),
            Layout::array::<AliasRow>(aliases),
        ] {
            bytes = bytes
                .checked_add(allocation.map_err(|_| layout())?.size())
                .ok_or_else(layout)?;
        }
        bytes = bytes
            .checked_add(
                ResidencyLedgerStorageLayout::inspect(plan, groups)
                    .ok_or_else(layout)?
                    .required_bytes(),
            )
            .ok_or_else(layout)?;
        for control in [
            size_of::<ResidencyControllerPlan<'a, C, F, G>>(),
            size_of::<F>(),
            size_of::<ResidencyController>(),
            size_of::<OffloadPlan>(),
            size_of::<Vec<OffloadUnit>>(),
            size_of::<Vec<BindingLocation>>(),
            size_of::<Vec<usize>>(),
            size_of::<Vec<bool>>(),
            size_of::<Vec<AliasRow>>(),
            size_of::<ResidencyControllerError>(),
            size_of::<Result<ResidencyController, ResidencyControllerError>>(),
            size_of::<ResidencyConstructionError>(),
            size_of::<Result<(), TryReserveError>>(),
        ] {
            bytes = bytes.checked_add(control).ok_or_else(layout)?;
        }
        Ok(ResidencyControllerPlan {
            catalog,
            plan,
            units,
            groups,
            bytes,
            marker: std::marker::PhantomData,
        })
    }
}
fn input(binding: &WeightBinding) -> RecipeInferenceInput<'_> {
    binding
        .recipe
        .as_ref()
        .map(RecipeInferenceInput::Derived)
        .unwrap_or(RecipeInferenceInput::Source {
            key: &binding.checkpoint_key,
            selection: &binding.selection,
        })
}
fn selection_bytes(selection: &TensorSelection) -> Option<usize> {
    match selection {
        TensorSelection::Indices { indices, .. } => {
            Layout::array::<usize>(indices.len()).ok().map(|v| v.size())
        }
        TensorSelection::Contiguous { shape, .. } => {
            Layout::array::<usize>(shape.len()).ok().map(|v| v.size())
        }
        _ => Some(0),
    }
}
fn dtype_bytes(dtype: &eredu_checkpoint::recipe::RecipeDtype) -> usize {
    match dtype {
        eredu_checkpoint::recipe::RecipeDtype::Other(name) => name.len(),
        _ => 0,
    }
}
fn recipe_clone_bytes(recipe: &DerivedWeightRecipe) -> Option<usize> {
    use DerivedWeightRecipe::*;
    let child = |input: &DerivedWeightRecipe| {
        size_of::<DerivedWeightRecipe>().checked_add(recipe_clone_bytes(input)?)
    };
    match recipe {
        Source { key, selection } => key.len().checked_add(selection_bytes(selection)?),
        Concatenate { inputs, .. } | Stack { inputs, .. } => inputs.iter().try_fold(
            Layout::array::<DerivedWeightRecipe>(inputs.len())
                .ok()?
                .size(),
            |n, input| n.checked_add(recipe_clone_bytes(input)?),
        ),
        Select { input, selection } => child(input)?.checked_add(selection_bytes(selection)?),
        Reshape { input, shape } => {
            child(input)?.checked_add(Layout::array::<usize>(shape.len()).ok()?.size())
        }
        Transpose { input, axes } => {
            child(input)?.checked_add(Layout::array::<usize>(axes.len()).ok()?.size())
        }
        Cast { input, dtype } => child(input)?.checked_add(dtype_bytes(dtype)),
        View {
            input,
            dtype,
            shape,
        } => child(input)?
            .checked_add(dtype_bytes(dtype))?
            .checked_add(Layout::array::<usize>(shape.len()).ok()?.size()),
        NegLog { input } | SubtractOne { input } => child(input),
    }
}
fn binding_clone_bytes(binding: &WeightBinding) -> Option<usize> {
    let mut bytes = binding
        .name
        .len()
        .checked_add(binding.alias_of.as_ref().map_or(0, String::len))?
        .checked_add(binding.logical_target.as_ref().map_or(0, String::len))?
        .checked_add(binding.checkpoint_key.len())?
        .checked_add(selection_bytes(&binding.selection)?)?;
    if let Some(recipe) = &binding.recipe {
        bytes = bytes.checked_add(recipe_clone_bytes(recipe)?)?;
    }
    if let Some(companions) = &binding.quantization_companions {
        bytes = bytes
            .checked_add(companions.scale.len())?
            .checked_add(companions.affine_bias.as_ref().map_or(0, String::len))?;
    }
    Some(bytes)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct BindingLocation {
    pub unit: usize,
    pub binding: usize,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct AliasRow {
    pub alias: BindingLocation,
    pub owner: BindingLocation,
}
fn at(units: &[OffloadUnit], location: BindingLocation) -> &WeightBinding {
    &units[location.unit].bindings[location.binding]
}
fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<(), ResidencyControllerError> {
    values
        .try_reserve_exact(count)
        .map_err(ResidencyControllerError::StorageReserve)
}
fn alias_identity(binding: &WeightBinding) -> &str {
    binding.logical_target().unwrap_or(binding.name())
}
enum AliasLookupFailure {
    Missing,
    Ambiguous,
}
enum AliasResolutionFailure<'a> {
    Cycle(&'a WeightBinding),
    Lookup(&'a WeightBinding, &'a str, AliasLookupFailure),
    Bytes(&'a WeightBinding, &'a WeightBinding),
}
impl AliasResolutionFailure<'_> {
    fn into_owned(self) -> ResidencyControllerError {
        match self {
            Self::Cycle(binding) => ResidencyDeclarationError::BindingAliasCycle {
                name: binding.name.clone(),
            },
            Self::Lookup(binding, owner, AliasLookupFailure::Missing) => {
                ResidencyDeclarationError::UnknownBindingAliasOwner {
                    alias: binding.name.clone(),
                    owner: owner.to_owned(),
                }
            }
            Self::Lookup(binding, owner, AliasLookupFailure::Ambiguous) => {
                ResidencyDeclarationError::AmbiguousBindingAliasOwner {
                    alias: binding.name.clone(),
                    owner: owner.to_owned(),
                }
            }
            Self::Bytes(alias, owner) => ResidencyDeclarationError::BindingAliasByteMismatch {
                alias: alias.name.clone(),
                owner: owner.name.clone(),
                alias_bytes: alias.expected_bytes,
                owner_bytes: owner.expected_bytes,
            },
        }
        .into()
    }
}
fn resolve_alias<'a>(
    units: &'a [OffloadUnit],
    start: BindingLocation,
    lookup: impl Fn(&str) -> Result<BindingLocation, AliasLookupFailure>,
    mut seen: impl FnMut(BindingLocation) -> bool,
) -> Result<BindingLocation, AliasResolutionFailure<'a>> {
    let mut current = start;
    loop {
        let binding = at(units, current);
        let Some(destination) = binding.alias_of() else {
            break;
        };
        if seen(current) {
            return Err(AliasResolutionFailure::Cycle(binding));
        }
        current = lookup(destination)
            .map_err(|cause| AliasResolutionFailure::Lookup(binding, destination, cause))?;
    }
    let alias = at(units, start);
    let owner = at(units, current);
    if alias.expected_bytes != owner.expected_bytes {
        return Err(AliasResolutionFailure::Bytes(alias, owner));
    }
    Ok(current)
}
pub(super) fn aliases(units: &[OffloadUnit]) -> Result<Vec<AliasRow>, ResidencyControllerError> {
    let total = units
        .iter()
        .try_fold(0usize, |n, unit| n.checked_add(unit.bindings.len()))
        .ok_or(ResidencyControllerError::ArithmeticOverflow {
            context: "binding index count",
        })?;
    let count = units
        .iter()
        .flat_map(|unit| &unit.bindings)
        .filter(|binding| binding.is_alias())
        .count();
    let mut locations = Vec::new();
    reserve(&mut locations, total)?;
    for (unit, value) in units.iter().enumerate() {
        for binding in 0..value.bindings.len() {
            locations.push(BindingLocation { unit, binding });
        }
    }
    let mut identities = Vec::new();
    reserve(&mut identities, total)?;
    identities.extend(0..total);
    let identity = |index: usize| alias_identity(at(units, locations[index]));
    identities.sort_unstable_by(|a, b| identity(*a).cmp(identity(*b)).then_with(|| a.cmp(b)));
    let mut visiting = Vec::new();
    reserve(&mut visiting, total)?;
    visiting.resize(total, false);
    let mut output = Vec::new();
    reserve(&mut output, count)?;
    for &location in &locations {
        if !at(units, location).is_alias() {
            continue;
        }
        visiting.fill(false);
        let owner = resolve_alias(
            units,
            location,
            |destination| {
                let start = identities.partition_point(|&i| identity(i) < destination);
                let end = identities.partition_point(|&i| identity(i) <= destination);
                match end - start {
                    0 => Err(AliasLookupFailure::Missing),
                    1 => Ok(locations[identities[start]]),
                    _ => Err(AliasLookupFailure::Ambiguous),
                }
            },
            |location| {
                let index = locations.binary_search(&location).expect("indexed binding");
                std::mem::replace(&mut visiting[index], true)
            },
        )
        .map_err(AliasResolutionFailure::into_owned)?;
        output.push(AliasRow {
            alias: location,
            owner,
        });
    }
    Ok(output)
}
fn validate_unique_units(units: &[OffloadUnit]) -> Result<(), ResidencyControllerError> {
    // Preserve the original input-order duplicate refusal.
    for (index, unit) in units.iter().enumerate() {
        if units[..index].iter().any(|prior| prior.id == unit.id) {
            return Err(ResidencyControllerError::DuplicateUnitDefinition {
                id: unit.id.clone(),
            });
        }
    }
    Ok(())
}
fn validate_plan_units(
    plan: &OffloadPlan,
    units: &[OffloadUnit],
    contains: impl Fn(&OffloadUnitId) -> bool,
) -> Result<(), ResidencyControllerError> {
    for spec in plan.units() {
        if !contains(spec.id()) {
            return Err(ResidencyControllerError::MissingUnitDefinition {
                id: spec.id().clone(),
            });
        }
    }
    // Ordinary construction has already sorted its owned rows. A borrowed
    // consumer preserves that same first unexpected ID without sorting input.
    if let Some(unit) = units
        .iter()
        .filter(|unit| plan.unit(unit.id()).is_none())
        .min_by(|a, b| a.id.cmp(&b.id))
    {
        return Err(ResidencyControllerError::UnexpectedUnitDefinition {
            id: unit.id.clone(),
        });
    }
    Ok(())
}
pub(super) fn build<'a, C: RecipeCatalog + ?Sized + 'a, G: AsRef<str>>(
    catalog: impl Fn(&OffloadUnitId) -> &'a C,
    plan: OffloadPlan,
    mut units: Vec<OffloadUnit>,
    groups: Option<&[G]>,
    infer: impl Fn(&WeightBinding, &C) -> Result<u64, ResidencyControllerError>,
) -> Result<ResidencyController, ResidencyControllerError> {
    validate_unique_units(&units)?;
    units.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    validate_plan_units(&plan, &units, |id| {
        units.binary_search_by(|unit| unit.id.cmp(id)).is_ok()
    })?;
    let alias_owners = aliases(&units)?;
    for spec in plan.units() {
        let unit = &units[units
            .binary_search_by(|unit| unit.id.cmp(spec.id()))
            .expect("validated unit")];
        let source = catalog(unit.id());
        let mut total = 0u64;
        for binding in unit.bindings.iter().filter(|binding| !binding.is_alias()) {
            total = total.checked_add(binding.expected_bytes).ok_or(
                ResidencyControllerError::ArithmeticOverflow {
                    context: "unit binding byte total",
                },
            )?;
            let actual = infer(binding, source)?;
            if actual != binding.expected_bytes {
                return Err(ResidencyControllerError::BindingByteMismatch {
                    id: unit.id.clone(),
                    binding: binding.name.clone(),
                    expected_bytes: binding.expected_bytes,
                    actual_bytes: actual,
                });
            }
        }
        if total != spec.bytes() {
            return Err(ResidencyControllerError::UnitByteMismatch {
                id: unit.id.clone(),
                planned_bytes: spec.bytes(),
                actual_bytes: total,
            });
        }
    }
    let ledger = match groups {
        Some(groups) => ResidencyLedger::new_prepared(plan, groups)?,
        None => ResidencyLedger::new(plan),
    };
    Ok(ResidencyController {
        ledger,
        units,
        alias_owners,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{
        store::{StoreError, TensorMetadata},
        StoredDtype,
    };
    use eredu_core::residency::{OffloadConfig, OffloadUnitSpec, ResidencyPolicy};
    struct Catalog {
        metadata: TensorMetadata,
        reject_owned: bool,
    }
    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            assert!(
                !self.reject_owned,
                "prepared controller called allocating metadata API"
            );
            Ok(self.metadata.clone())
        }
        fn tensor_metadata_borrowed(&self, _: &str) -> Option<&TensorMetadata> {
            Some(&self.metadata)
        }
        fn recipe_cache(&self) -> Option<&eredu_checkpoint::recipe::RecipeInferenceCache> {
            assert!(
                !self.reject_owned,
                "prepared controller consulted ordinary inference cache"
            );
            None
        }
    }
    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }
    #[test]
    fn prepared_controller_borrows_recipe_metadata_and_preserves_alias_and_window_transitions() {
        let mut catalog = Catalog {
            metadata: TensorMetadata {
                name: "w".into(),
                logical_shape: vec![2, 3],
                physical_shape: vec![2, 3],
                stored_dtype: StoredDtype::F32,
                encoded_byte_len: 24,
                backing_shard: None,
            },
            reject_owned: false,
        };
        let recipe = DerivedWeightRecipe::Transpose {
            input: Box::new(DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::source("w", TensorSelection::Full)),
                shape: vec![3, 2],
            }),
            axes: vec![1, 0],
        };
        let owner = WeightBinding::from_recipe("weight", recipe, 24)
            .unwrap()
            .with_logical_target("shared")
            .unwrap();
        let alias = WeightBinding::alias("tied", "shared", 24).unwrap();
        let units = vec![OffloadUnit::new(id("a"), [owner, alias]).unwrap()];
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [
                OffloadUnitSpec::new(id("a"), 24, ResidencyPolicy::Windowed, MemoryTier::Host)
                    .unwrap(),
            ],
        )
        .unwrap();
        let mut ordinary = ResidencyController::new(&catalog, plan.clone(), units.clone()).unwrap();
        catalog.reject_owned = true;
        let prepared =
            ResidencyController::prepare_constructor(|_| &catalog, &plan, &units, &["execution"])
                .unwrap();
        assert!(prepared.required_bytes() > 0);
        prepared.validate_unit_definitions().unwrap();
        let alias = units[0]
            .bindings()
            .iter()
            .find(|binding| binding.is_alias())
            .unwrap();
        let (owner_unit, owner) = prepared.binding_owner(units[0].id(), alias).unwrap();
        assert!(std::ptr::eq(owner_unit, units[0].id()));
        assert!(std::ptr::eq(
            owner,
            units[0]
                .bindings()
                .iter()
                .find(|binding| !binding.is_alias())
                .unwrap()
        ));
        assert!(prepared.binding_owner(units[0].id(), owner).is_none());
        assert!(prepared.binding_owner(&id("absent"), alias).is_none());
        let mut finite = prepared.construct().unwrap();
        let unit = finite.unit(&id("a")).unwrap();
        let alias = unit
            .bindings()
            .iter()
            .find(|value| value.is_alias())
            .unwrap();
        assert_eq!(
            finite.binding_owner(unit.id(), alias).unwrap().1.name(),
            "weight"
        );
        for control in [&mut ordinary, &mut finite] {
            control
                .ledger_mut()
                .set_group_window("execution", &[id("a")], MemoryTier::Host)
                .unwrap();
            control
                .ledger_mut()
                .set_group_window("execution", &[id("a")], MemoryTier::Device)
                .unwrap();
            control
                .ledger_mut()
                .set_group_window("execution", &[], MemoryTier::Host)
                .unwrap();
        }
        assert_eq!(
            ordinary.ledger().active_window(),
            finite.ledger().active_window()
        );
        assert!(matches!(
            finite
                .ledger_mut()
                .set_group_window("unprepared", &[id("a")], MemoryTier::Host),
            Err(ResidencyLedgerError::UnknownPreparedGroup)
        ));
        assert_eq!(finite.ledger().active_window().len(), 1);
        finite
            .ledger_mut()
            .set_group_window("execution", &[], MemoryTier::Device)
            .unwrap();
        assert!(finite.ledger().active_window().is_empty());
        let missing = [OffloadUnit::new(
            id("a"),
            [WeightBinding::alias("alias", "missing", 24).unwrap()],
        )
        .unwrap()];
        let failure =
            ResidencyController::prepare_constructor(|_| &catalog, &plan, &missing, &[] as &[&str])
                .unwrap()
                .construct()
                .unwrap_err();
        assert!(
            matches!(failure, ResidencyControllerError::Declaration(ResidencyDeclarationError::UnknownBindingAliasOwner { alias, owner }) if alias == "alias" && owner == "missing")
        );
    }
    #[test]
    fn borrowed_unit_domain_preserves_constructor_error_priority() {
        let catalog = Catalog {
            metadata: TensorMetadata {
                name: "w".into(),
                logical_shape: vec![2, 3],
                physical_shape: vec![2, 3],
                stored_dtype: StoredDtype::F32,
                encoded_byte_len: 24,
                backing_shard: None,
            },
            reject_owned: true,
        };
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [
                OffloadUnitSpec::new(id("b"), 24, ResidencyPolicy::Windowed, MemoryTier::Disk)
                    .unwrap(),
            ],
        )
        .unwrap();
        fn cause(error: ResidencyControllerError) -> (&'static str, OffloadUnitId) {
            match error {
                ResidencyControllerError::DuplicateUnitDefinition { id } => ("duplicate", id),
                ResidencyControllerError::MissingUnitDefinition { id } => ("missing", id),
                ResidencyControllerError::UnexpectedUnitDefinition { id } => ("unexpected", id),
                other => panic!("unexpected domain failure: {other}"),
            }
        }
        for (names, expected) in [
            (&["z", "a", "z", "a"][..], ("duplicate", id("z"))),
            (&["z", "a"][..], ("missing", id("b"))),
            (&["z", "b", "a"][..], ("unexpected", id("a"))),
        ] {
            let units: Vec<_> = names
                .iter()
                .map(|name| {
                    OffloadUnit::new(
                        id(name),
                        [WeightBinding::from_recipe(
                            "weight",
                            DerivedWeightRecipe::source("w", TensorSelection::Full),
                            24,
                        )
                        .unwrap()],
                    )
                    .unwrap()
                })
                .collect();
            let prepared = ResidencyController::prepare_constructor(
                |_| &catalog,
                &plan,
                &units,
                &[] as &[&str],
            )
            .unwrap();
            assert_eq!(
                cause(prepared.validate_unit_definitions().unwrap_err()),
                expected
            );
            assert_eq!(
                cause(ResidencyController::new(&catalog, plan.clone(), units).unwrap_err()),
                expected
            );
        }
    }
    #[test]
    fn prepared_initial_tier_closure_includes_cross_tier_canonical_owners() {
        let catalog = Catalog {
            metadata: TensorMetadata {
                name: "w".into(),
                logical_shape: vec![2, 3],
                physical_shape: vec![2, 3],
                stored_dtype: StoredDtype::F32,
                encoded_byte_len: 24,
                backing_shard: None,
            },
            reject_owned: true,
        };
        let owner = |target| {
            WeightBinding::new("weight", "w", TensorSelection::Full, 24)
                .unwrap()
                .with_logical_target(target)
                .unwrap()
        };
        // Valid cyclic unit dependencies, with no tensor alias cycle. The host
        // root needs the canonical owner whose selected initial tier is Device.
        let units = [
            OffloadUnit::new(
                id("b"),
                [
                    owner("owner.b"),
                    WeightBinding::alias("tied", "owner.a", 24).unwrap(),
                ],
            )
            .unwrap(),
            OffloadUnit::new(id("c"), [owner("owner.c")]).unwrap(),
            OffloadUnit::new(
                id("a"),
                [
                    owner("owner.a"),
                    WeightBinding::alias("tied", "owner.b", 24).unwrap(),
                ],
            )
            .unwrap(),
        ];
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [
                OffloadUnitSpec::new(id("a"), 24, ResidencyPolicy::Windowed, MemoryTier::Host)
                    .unwrap(),
                OffloadUnitSpec::new(id("b"), 24, ResidencyPolicy::Windowed, MemoryTier::Device)
                    .unwrap(),
                OffloadUnitSpec::new(id("c"), 24, ResidencyPolicy::Windowed, MemoryTier::Device)
                    .unwrap(),
            ],
        )
        .unwrap();
        let prepared =
            ResidencyController::prepare_constructor(|_| &catalog, &plan, &units, &[] as &[&str])
                .unwrap();
        let roots = || {
            plan.units()
                .iter()
                .filter(|row| row.tier() == MemoryTier::Host)
                .map(OffloadUnitSpec::id)
        };
        let mut scratch = [ResidencyClosureSlot::default(); 3];
        let closure = prepared.operation_closure(roots(), &mut scratch).unwrap();
        assert_eq!(closure.len(), 2);
        let reached = closure.units().collect::<Vec<_>>();
        assert!(std::ptr::eq(reached[0], &units[0]));
        assert!(std::ptr::eq(reached[1], &units[2]));
        let reached_ids = reached
            .iter()
            .map(|unit| unit.id().clone())
            .collect::<BTreeSet<_>>();
        let before = scratch;
        assert!(matches!(
            prepared.operation_closure(roots(), &mut scratch[..2]),
            Err(ResidencyClosureError::DestinationLength)
        ));
        assert_eq!(scratch, before);
        let missing = id("missing");
        assert!(matches!(
            prepared.operation_closure(std::iter::once(&missing), &mut scratch),
            Err(ResidencyClosureError::UnknownRoot)
        ));
        assert_eq!(scratch, before);
        let constructed = prepared.construct().unwrap();
        let root_ids = roots().cloned().collect::<Vec<_>>();
        let closure = constructed
            .operation_closure(&root_ids, &mut scratch)
            .unwrap();
        assert_eq!(
            closure
                .units()
                .map(|unit| unit.id().clone())
                .collect::<BTreeSet<_>>(),
            reached_ids
        );
    }
}
