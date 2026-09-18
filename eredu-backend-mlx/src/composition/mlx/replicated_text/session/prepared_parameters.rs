//! Parameter ownership retained from the exact selected module/binding pair.
use super::*;
use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeMetadata};
use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized};
use eredu_runtime::parameter_operations::{PreparedParameterLocation, PreparedParameterSlot};

struct Metadata(Vec<ParameterMetadata>);
impl<'a> ParameterVisitor<'a, MlxTensor> for Metadata {
    fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, _value: &'a MlxTensor) {
        self.0.push(metadata.to_owned());
    }
}

pub(in crate::composition::mlx::replicated_text) fn collect_module<M: Parameterized<MlxTensor>>(
    module: &M,
    bindings: &[WeightBinding],
    location: PreparedParameterLocation,
    store: &dyn CheckpointSource,
    output: &mut Vec<PreparedParameterSlot>,
    declarations: &mut Vec<ParameterMetadata>,
) -> Result<(), Error> {
    let mut metadata = Metadata(Vec::new());
    module.visit_parameters(&mut metadata)?;
    for parameter in metadata.0 {
        declarations.push(parameter.clone());
        let Some(binding) = bindings.iter().find(|binding| {
            binding.logical_target().unwrap_or(binding.name()) == parameter.id.as_str()
        }) else {
            // A partition or independent parameter bank owns this slot.
            continue;
        };
        let materialized = binding_metadata(binding, bindings, store)?;
        output.push(PreparedParameterSlot {
            parameter,
            materialized,
            location: location.clone(),
        });
    }
    Ok(())
}

fn binding_metadata<'a>(
    mut binding: &'a WeightBinding,
    bindings: &'a [WeightBinding],
    store: &dyn CheckpointSource,
) -> Result<RecipeMetadata, Error> {
    let mut remaining = bindings.len();
    while let Some(owner) = binding.alias_of() {
        if remaining == 0 {
            return Err(Error::ArchitectureModel(
                "cyclic prepared parameter binding".into(),
            ));
        }
        remaining -= 1;
        binding = bindings
            .iter()
            .find(|candidate| candidate.name() == owner)
            .ok_or_else(|| {
                Error::ArchitectureModel(format!(
                    "prepared parameter alias owner {owner} is absent"
                ))
            })?;
    }
    let recipe = binding.recipe().cloned().unwrap_or_else(|| {
        DerivedWeightRecipe::source(binding.checkpoint_key(), binding.selection().clone())
    });
    recipe
        .infer(store)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

pub(super) fn collect<A, S>(
    architecture: &A,
    units: &[A::Unit],
    addresses: &[eredu_runtime::ExecutionUnitAddress],
    static_bindings: &[WeightBinding],
    unit_bindings: &[Vec<WeightBinding>],
    store: &dyn CheckpointSource,
) -> Result<(Vec<PreparedParameterSlot>, Vec<ParameterMetadata>), Error>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S>,
{
    struct Static<'a> {
        bindings: &'a [WeightBinding],
        store: &'a dyn CheckpointSource,
        output: &'a mut Vec<PreparedParameterSlot>,
        declarations: &'a mut Vec<ParameterMetadata>,
    }
    impl eredu_runtime::StaticParameterVisitor<MlxNeuralBackend> for Static<'_> {
        type Error = Error;
        fn visit<M: Parameterized<MlxTensor>>(
            &mut self,
            role: &str,
            module: &M,
        ) -> Result<(), Error> {
            collect_module(
                module,
                self.bindings,
                PreparedParameterLocation::Static { role: role.into() },
                self.store,
                self.output,
                self.declarations,
            )
        }
    }
    if units.len() != addresses.len() || units.len() != unit_bindings.len() {
        return Err(Error::ArchitectureModel(
            "prepared parameter owners differ from execution units".into(),
        ));
    }
    let mut output = Vec::new();
    let mut declarations = Vec::new();
    architecture.visit_static_parameters(&mut Static {
        bindings: static_bindings,
        store,
        output: &mut output,
        declarations: &mut declarations,
    })?;
    for (ordinal, ((unit, address), bindings)) in
        units.iter().zip(addresses).zip(unit_bindings).enumerate()
    {
        collect_module(
            unit,
            bindings,
            PreparedParameterLocation::Unit {
                ordinal,
                address: *address,
            },
            store,
            &mut output,
            &mut declarations,
        )?;
    }
    output.sort_unstable_by(|left, right| left.parameter.id.cmp(&right.parameter.id));
    if output
        .windows(2)
        .any(|pair| pair[0].parameter.id == pair[1].parameter.id)
    {
        return Err(Error::ArchitectureModel(
            "duplicate prepared parameter slot".into(),
        ));
    }
    Ok((output, declarations))
}
