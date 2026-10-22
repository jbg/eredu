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
