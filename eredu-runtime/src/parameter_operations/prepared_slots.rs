//! Parameter ownership retained from the exact selected module/binding pair.
use super::{PreparedParameterLocation, PreparedParameterSlot};
use crate::{LayeredArchitecture, RuntimeState};
use eredu_checkpoint::recipe::RecipeMetadata;
use eredu_checkpoint::store::CheckpointSource;
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_core::BackendFailure;
use eredu_nn::NeuralBackend;
use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized};

struct Metadata(Vec<ParameterMetadata>);
impl<'a, T: 'a> ParameterVisitor<'a, T> for Metadata {
    fn visit(&mut self, metadata: ParameterMetadata, _value: &'a T) {
        self.0.push(metadata);
    }
}

/// Collects retained slot metadata and its exact checkpoint binding.
pub(crate) fn collect_prepared_module_parameters<T: 'static, M: Parameterized<T>>(
    module: &M,
    bindings: &[WeightBinding],
    location: PreparedParameterLocation,
    store: &dyn CheckpointSource,
    output: &mut Vec<PreparedParameterSlot>,
    declarations: &mut Vec<ParameterMetadata>,
) -> Result<(), BackendFailure> {
    let mut metadata = Metadata(Vec::new());
    module.visit_parameters(&mut metadata);
    for parameter in metadata.0 {
        declarations.push(parameter.clone());
        let Some(binding) = bindings.iter().find(|binding| {
            binding.logical_target().unwrap_or(binding.name()) == parameter.id.as_str()
        }) else {
            // A partition or independent parameter bank owns this slot.
            continue;
        };
        let (materialized, backing) = binding_metadata(binding, bindings, store)?;
        output.push(PreparedParameterSlot {
            parameter,
            materialized,
            backing: Some(backing),
            location: location.clone(),
        });
    }
    Ok(())
}

fn binding_metadata<'a>(
    mut binding: &'a WeightBinding,
    bindings: &'a [WeightBinding],
    store: &dyn CheckpointSource,
) -> Result<(RecipeMetadata, String), BackendFailure> {
    let mut remaining = bindings.len();
    while let Some(owner) = binding.alias_of() {
        if remaining == 0 {
            return Err(invalid("cyclic prepared parameter binding"));
        }
        remaining -= 1;
        binding = bindings
            .iter()
            .find(|candidate| candidate.name() == owner)
            .ok_or_else(|| invalid(format!("prepared parameter alias owner {owner} is absent")))?;
    }
    let metadata = binding
        .recipe_view()
        .infer(store)
        .map_err(BackendFailure::from_error)?;
    Ok((metadata, binding.name().to_owned()))
}

/// Collects the declared and bound parameter topology in selected storage order.
pub(crate) fn collect_prepared_layered_parameters<A, B, S>(
    architecture: &A,
    units: &[A::Unit],
    addresses: &[crate::ExecutionUnitAddress],
    static_bindings: &[WeightBinding],
    unit_bindings: &[Vec<WeightBinding>],
    store: &dyn CheckpointSource,
) -> Result<(Vec<PreparedParameterSlot>, Vec<ParameterMetadata>), BackendFailure>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    struct Static<'a> {
        bindings: &'a [WeightBinding],
        store: &'a dyn CheckpointSource,
        output: &'a mut Vec<PreparedParameterSlot>,
        declarations: &'a mut Vec<ParameterMetadata>,
    }
    impl<B: NeuralBackend> crate::StaticParameterVisitor<B> for Static<'_> {
        type Error = BackendFailure;
        fn visit<M: Parameterized<B::Tensor>>(
            &mut self,
            role: &str,
            module: &M,
        ) -> Result<(), BackendFailure> {
            collect_prepared_module_parameters::<B::Tensor, _>(
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
        return Err(invalid(
            "prepared parameter owners differ from execution units",
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
        collect_prepared_module_parameters::<B::Tensor, _>(
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
        return Err(invalid("duplicate prepared parameter slot"));
    }
    Ok((output, declarations))
}

fn invalid(detail: impl Into<String>) -> BackendFailure {
    BackendFailure::from_error(eredu_nn::Error::backend(detail.into()))
}
