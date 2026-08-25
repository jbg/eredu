//! Shared filtering for architecture-owned pinned-parameter recipes.

use std::collections::BTreeMap;

use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_nn::Parameterized;

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
