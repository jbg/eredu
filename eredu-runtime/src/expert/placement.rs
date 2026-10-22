//! Cold rank-local projection of independently addressable parameter members.

use super::{AddressableBankMember, AddressableBankMemberError, AddressableBankParameter};
use crate::{BindingPlacementError, LocalModelLayout, ResidencyDeclarationError, WeightBinding};
use eredu_checkpoint::recipe::{RecipeCatalog, RecipeError};

/// Failure projecting selected member recipes into their remaining local placement.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankPlacementError {
    /// A member's source binding declaration is invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
    /// The selected logical layout cannot be applied to a member recipe.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Member-local source geometry cannot be inferred.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Selected task, companion, owner or payload validation failed.
    #[error(transparent)]
    Member(#[from] AddressableBankMemberError),
}

/// Applies the remaining selected placements to already member-selected recipes.
///
/// The consumed bank-member axis is skipped by the shared binding placement
/// mechanism. Canonical task authority, local companion bindings, member order,
/// identity and rank ownership are retained. Payload geometry is recomputed from
/// the projected recipe and selected executable without reading tensor payloads.
pub fn place_addressable_bank_members<C: RecipeCatalog + ?Sized>(
    members: &[AddressableBankMember],
    source: &C,
    layout: &LocalModelLayout,
) -> Result<Vec<AddressableBankMember>, AddressableBankPlacementError> {
    members
        .iter()
        .map(|member| {
            let source_bindings = member
                .parameters()
                .iter()
                .map(|parameter| {
                    WeightBinding::from_recipe(
                        parameter.binding_name(),
                        parameter.recipe().clone(),
                        parameter.source_bytes(),
                    )?
                    .with_logical_target(parameter.task().name())
                })
                .collect::<Result<Vec<_>, ResidencyDeclarationError>>()?;
            let bindings =
                crate::place_addressable_member_bindings(source_bindings, source, layout)?;
            let parameters = member
                .parameters()
                .iter()
                .zip(bindings)
                .map(|(parameter, binding)| {
                    let recipe = binding.source_recipe();
                    let metadata = recipe.infer(source)?;
                    let selected_bytes =
                        super::selected_addressable_parameter_bytes(parameter.task(), &metadata)?;
                    Ok(AddressableBankParameter::from_shared_task(
                        parameter.binding_name(),
                        parameter.shared_task().clone(),
                        recipe,
                        metadata,
                        selected_bytes,
                        parameter.quantization_companions().cloned(),
                    )?)
                })
                .collect::<Result<Vec<_>, AddressableBankPlacementError>>()?;
            Ok(AddressableBankMember::new(
                member.key(),
                member.placement().clone(),
                parameters,
            )?)
        })
        .collect()
}

/// Inconsistent selected addressable storage before native materialization.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankSelectionError {
    /// The retained ownership keys differ from the selected physical entries.
    #[error("selected bank placements omit {missing:?} and unexpectedly retain {unexpected:?}")]
    PlacementCoverage {
        /// Selected members without retained ownership.
        missing: std::collections::BTreeSet<crate::ParameterBankKey>,
        /// Ownership records without selected physical entries.
        unexpected: std::collections::BTreeSet<crate::ParameterBankKey>,
    },
    /// A selected member omitted its admitted executable extent.
    #[error("selected bank member {key:?} has no executable byte total")]
    MissingBytes {
        /// Original global member identity.
        key: crate::ParameterBankKey,
    },
    /// A selected transform recipe cannot be inferred against its source.
    #[error("selected bank member {key:?} binding {binding:?}: {source}")]
    Recipe {
        /// Original global member identity.
        key: crate::ParameterBankKey,
        /// Local binding whose recipe failed.
        binding: String,
        /// Original recipe or source failure.
        #[source]
        source: RecipeError,
    },
    /// The selected executable payload cannot be calculated.
    #[error("selected bank member {key:?}: {source}")]
    Geometry {
        /// Original global member identity.
        key: crate::ParameterBankKey,
        /// Exact invalid parameter geometry.
        #[source]
        source: AddressableBankMemberError,
    },
    /// The selected member's simultaneously retained binding extents overflow.
    #[error("selected bank member {key:?} executable byte total overflowed")]
    ByteOverflow {
        /// Original global member identity.
        key: crate::ParameterBankKey,
    },
    /// Binding geometry differs from the retained selected executable extent.
    #[error(
        "selected bank member {key:?} bytes differ: expected {expected}, projected {projected}"
    )]
    ByteMismatch {
        /// Original global member identity.
        key: crate::ParameterBankKey,
        /// Retained executable payload extent.
        expected: u64,
        /// Extent projected from final bindings and selected transforms.
        projected: u64,
    },
}

/// Checks exact ownership coverage and selected executable payloads before any
/// transformation reads payloads. Direct bindings use their retained extents;
/// selected transforms infer source metadata and use shared packed geometry.
/// Namespace, residency-budget and native capability admission remain separate.
pub fn validate_addressable_bank_selection<'a, C: RecipeCatalog + ?Sized>(
    entries: impl Iterator<Item = (crate::ParameterBankKey, &'a [WeightBinding])> + Clone,
    placements: &std::collections::BTreeMap<
        crate::ParameterBankKey,
        super::AddressableBankMemberPlacement,
    >,
    expected_bytes: &std::collections::BTreeMap<crate::ParameterBankKey, u64>,
    transformations: &std::collections::BTreeMap<
        (crate::ParameterBankKey, String),
        super::AddressableBindingTransform,
    >,
    source: &C,
) -> Result<(), AddressableBankSelectionError> {
    let selected_keys = entries
        .clone()
        .map(|(key, _)| key)
        .collect::<std::collections::BTreeSet<_>>();
    let placement_keys = placements
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if selected_keys != placement_keys {
        return Err(AddressableBankSelectionError::PlacementCoverage {
            missing: selected_keys.difference(&placement_keys).copied().collect(),
            unexpected: placement_keys.difference(&selected_keys).copied().collect(),
        });
    }
    for (key, bindings) in entries {
        let expected = expected_bytes
            .get(&key)
            .ok_or(AddressableBankSelectionError::MissingBytes { key })?;
        let mut projected = 0u64;
        for binding in bindings {
            let bytes =
                if let Some(transform) = transformations.get(&(key, binding.name().to_owned())) {
                    let metadata = binding.source_recipe().infer(source).map_err(|source| {
                        AddressableBankSelectionError::Recipe {
                            key,
                            binding: binding.name().to_owned(),
                            source,
                        }
                    })?;
                    super::addressable_projection_bytes(
                        binding.name(),
                        metadata.shape(),
                        transform.quantization(),
                        transform.companion_dtype(),
                    )
                    .map_err(|source| AddressableBankSelectionError::Geometry { key, source })?
                } else {
                    binding.expected_bytes()
                };
            projected = projected
                .checked_add(bytes)
                .ok_or(AddressableBankSelectionError::ByteOverflow { key })?;
        }
        if projected != *expected {
            return Err(AddressableBankSelectionError::ByteMismatch {
                key,
                expected: *expected,
                projected,
            });
        }
    }
    Ok(())
}
