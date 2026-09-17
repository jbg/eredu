//! Completed numerical transfer through the existing independent-copy worker.
use super::*;
use crate::backend::array_copy::{IsolatedArrayCopy, RegisteredArrayCopy};
use eredu_runtime::working_memory::WorkingMemoryError;

fn invalid() -> Error { Error::PrefillControl(WorkingMemoryError::IdentityMismatch) }

pub(crate) fn copy_value_to(
    value: &OriginalNumericalValue,
    origin: SamplingPlacement,
    destination: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<OriginalNumericalValue, Error> {
    let (sources, environment) = context.original_numerical_for(destination).ok_or_else(invalid)?;
    let (prior, source_environment) = context.original_numerical_for(origin).ok_or_else(invalid)?;
    let funding = sources.metadata_funding();
    let frames = [
        size_of::<(&OriginalNumericalValue, SamplingPlacement, SamplingPlacement, SpeculativeExecutionStreams<'_>)>(),
        size_of::<Result<OriginalNumericalValue, Error>>(),
        size_of::<RegisteredArrayCopy>(), size_of::<Result<RegisteredArrayCopy, Error>>(),
        size_of::<Option<(&safemlx::OriginalBufferBudget, &OriginalSpeculativeNumericalBudgetCustody)>>(),
        size_of::<Option<eredu_runtime::generation::PreparedControllerChoice>>(),
        eredu_runtime::generation::PreparedControllerChoice::metadata_bytes(),
    ];
    funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
    if !std::ptr::eq(sources, prior) || !environment.pool().same_domain(source_environment.pool()) {
        return Err(invalid());
    }
    NumericalProducer::validate_input_at(value, context, origin)?;
    sources.validate_environment(environment)?;
    let input = value.value();
    if !matches!(input.meaning, Meaning::RandomKey | Meaning::Logits) { return Err(invalid()); }
    let numerical = match (&input.original_budget, &input.provenance) {
        (Some(budget), Provenance::Numerical(custody)) => Some((budget, custody)),
        (None, Provenance::Registered(_)) if input._copy.is_some() => None,
        _ => return Err(invalid()),
    };
    let (roots, mechanisms) = sources.numerical_prerequisites();
    let copied = IsolatedArrayCopy::new(&input.array).copy_original(
        numerical, environment, roots, mechanisms, funding, sources.request().capacity_bytes(),
    )?;
    // The receiver checks the actual completed destination descriptor and
    // retains its independent registered source and native stream ownership.
    let output = registered::value_input(copied, input.meaning, sources, environment)?;
    Ok(output.with_controller_choice(value.controller_choice().cloned()))
}

pub(crate) fn copy_key_to(
    key: &OriginalNumericalKey,
    origin: SamplingPlacement,
    destination: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<OriginalNumericalKey, Error> {
    let (sources, _) = context.original_numerical_for(destination).ok_or_else(invalid)?;
    let frames = [size_of::<OriginalNumericalKey>(), size_of::<Result<OriginalNumericalKey, Error>>(),
        size_of::<(&OriginalNumericalKey, SamplingPlacement, SamplingPlacement, SpeculativeExecutionStreams<'_>)>()];
    sources.metadata_funding().reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
    copy_value_to(key.value(), origin, destination, context).map(OriginalNumericalKey::copied)
}
