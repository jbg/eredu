//! Prepared publication over the same numerical reference slots and future loaders.
use super::*;
use eredu_nn::workspace::*;
use eredu_runtime::parameter_operations::{
    ParameterPublication, ParameterReplacementValues, PreparedParameterPublication,
};

#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("parameter publication does not execute tensor equations")
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
}

fn clone_value(value: &NumericTensor, context: &WorkspaceContext) -> Result<NumericTensor, Error> {
    context.charge_metadata(std::mem::size_of::<(
        NumericTensor,
        Result<NumericTensor, Error>,
    )>())?;
    let mut shape = context.metadata_vec(value.shape.len())?;
    shape.extend_from_slice(&value.shape);
    let mut data = context.metadata_vec(value.data.len())?;
    data.extend_from_slice(&value.data);
    let dtype = match &value.dtype {
        eredu_core::checkpoint::TensorDtype::Encoded(name) => {
            eredu_core::checkpoint::TensorDtype::Encoded(
                context.metadata_string(format_args!("{name}"))?,
            )
        }
        value => value.clone(),
    };
    Ok(NumericTensor {
        shape,
        data,
        dtype,
        retirement_probe: value.retirement_probe.clone(),
        publication_funding: context.metadata_funding(),
    })
}

pub(super) fn publish(
    values: &BTreeMap<String, NumericTensor>,
    active: bool,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    visit: impl FnMut(&mut dyn ParameterPublication<NumericTensor>) -> Result<bool, Error>,
) -> Result<(), Error> {
    let pool = crate::memory_fixture::ledger(1 << 30, 0).map_err(Error::backend_retained_source)?;
    let funding = pool
        .prepare_workspace_metadata(execution, pool.configured_limits().clone())
        .map_err(Error::backend_retained_source)?;
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding.clone())?;
    let mut rows = context.metadata_vec(values.len())?;
    for (name, value) in values {
        rows.push((
            context.metadata_string(format_args!("{name}"))?,
            clone_value(value, &context)?,
        ));
    }
    let replacements =
        ParameterReplacementValues::from_prepared_rows(rows, funding.clone(), &context)
            .map_err(Error::backend_retained_source)?;
    exchange(replacements, active, visit, &context, funding)
}

pub(super) fn apply_retained<U: Parameterized<NumericTensor>>(
    unit: &mut U,
    replacements: &ParameterReplacementValues<NumericTensor>,
) -> Result<(), Error> {
    let Some(value) = replacements.values().next() else {
        return Ok(());
    };
    let funding = value
        .publication_funding
        .clone()
        .ok_or(WorkspaceMetadataError::Unqualified)?;
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding.clone())?;
    exchange(
        replacements.clone(),
        true,
        |visitor| {
            unit.visit_parameters_mut(
                &mut eredu_runtime::parameter_operations::ParameterPublicationVisitor(visitor),
            );
            Ok(true)
        },
        &context,
        funding,
    )
}

fn exchange(
    replacements: ParameterReplacementValues<NumericTensor>,
    active: bool,
    mut visit: impl FnMut(&mut dyn ParameterPublication<NumericTensor>) -> Result<bool, Error>,
    context: &WorkspaceContext,
    funding: HostMetadataFunding,
) -> Result<(), Error> {
    let mut publication = PreparedParameterPublication::prepare(
        replacements,
        active,
        &mut visit,
        |value| clone_value(value, context),
        context,
        funding,
    )
    .map_err(Error::backend_retained_source)?;
    publication
        .validate(&mut visit, |actual, expected| {
            // The scalar backend owns independent Vec copies, so exact current
            // values, dtype and geometry authenticate its immutable saved value.
            Ok(actual.shape == expected.shape
                && actual.dtype == expected.dtype
                && actual.data.len() == expected.data.len()
                && actual
                    .data
                    .iter()
                    .zip(&expected.data)
                    .all(|(a, b)| a.to_bits() == b.to_bits()))
        })
        .map_err(Error::backend_retained_source)?;
    publication.exchange(|visitor| {
        assert!(visit(visitor).expect("validated exclusively borrowed numerical participants"));
    });
    Ok(())
}
