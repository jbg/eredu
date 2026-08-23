//! Architecture-owned access to pinned parameter modules.

use std::collections::BTreeMap;

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::{NeuralBackend, Parameterized};

/// Retains only recipe destinations exposed by one architecture module.
pub fn module_recipes<T: 'static, M>(
    module: &M,
    mut recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String>
where
    M: Parameterized<T>,
{
    let parameters = eredu_nn::validate_parameter_topology(module)
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|metadata| metadata.id.as_str().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    recipes.retain(|name, _| parameters.contains(name));
    Ok(recipes)
}

/// Statically dispatched visitor over one immutable pinned parameter module.
pub trait StaticParameterVisitor<B: NeuralBackend> {
    /// Failure returned by the consumer.
    type Error;

    /// Visits the module bound to one architecture-declared static role.
    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<B::Tensor>;
}

/// Statically dispatched visitor over one mutable pinned parameter module.
pub trait StaticParameterVisitorMut<B: NeuralBackend> {
    /// Failure returned by the consumer.
    type Error;

    /// Visits the mutable module bound to one architecture-declared static role.
    fn visit_mut<M>(&mut self, role: &str, module: &mut M) -> Result<(), Self::Error>
    where
        M: Parameterized<B::Tensor>;
}

/// Architecture-owned binding between semantic static roles and modules.
///
/// Parameter descriptions decide which roles a partition owns. This interface
/// resolves those roles to the actual modules without exposing family fields or
/// checkpoint paths to a concrete backend.
pub trait BindableStaticParameters<B: NeuralBackend> {
    /// Returns architecture-owned checkpoint rewrites for pinned parameters.
    fn static_parameter_recipes(
        &self,
        _source: &dyn CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
        Ok(BTreeMap::new())
    }

    /// Visits every available pinned parameter module exactly once.
    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<B>;

    /// Mutably visits every available pinned parameter module exactly once.
    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<B>;
}
