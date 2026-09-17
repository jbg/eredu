//! Explicit source-bound tensor waves reuse model operators and nested completion.
use super::*;
use crate::backend::error::Error;
use std::mem::{size_of, size_of_val};
type Funding = eredu_nn::workspace::WorkspaceMetadataFunding;
type Roots = safemlx::PreparedNestedRoots<Funding>;
#[derive(Debug)]
struct NestedPreparationFailure {
    cause: safemlx::PreparedNestedRootsCause,
    _custody: Funding,
}
impl std::fmt::Display for NestedPreparationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for NestedPreparationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(&self.cause) }
}

type ArrayLoans<'a> = std::iter::Map<
    std::iter::Copied<std::slice::Iter<'a, &'a MlxTensor>>,
    fn(&'a MlxTensor) -> &'a Array,
>;
fn array(value: &MlxTensor) -> &Array { value.as_array() }

pub(crate) fn control_bytes(roots: usize) -> Option<usize> {
    if roots == 0 { return None; }
    let native = if roots <= 2 {
        crate::backend::runtime::cache::value_completion_control_bytes(roots)?
    } else {
        Roots::control_bytes(roots)?
            .checked_add(Roots::submission_control_bytes::<ArrayLoans<'_>>()?)?
            .checked_add(size_of::<Result<Roots, safemlx::PreparedNestedRootsFailure<Funding>>>())?
            .checked_add(eredu_nn::Error::retained_source_control_bytes::<NestedPreparationFailure>()?)?
    };
    let frames = [
        size_of::<(&MlxTensor, &Group, &Group, &Stream)>(),
        size_of::<(&[&MlxTensor], &[usize], usize)>(),
        size_of::<[&Array; 2]>(),
        size_of::<MlxTensor>(),
        size_of::<Option<MlxTensor>>(),
        size_of::<Result<Option<MlxTensor>, Error>>(),
        size_of::<Result<Option<()>, Error>>(),
        native,
    ];
    frames.iter().copied().try_fold(size_of_val(&frames), usize::checked_add)
}
fn charge(funding: &Funding, roots: usize) -> Result<(), Error> {
    funding.reserve_metadata(control_bytes(roots).ok_or(Error::PrefillScopeUnavailable)?)
        .map_err(Error::WorkspacePlanning)
}
fn complete_native(values: &[&MlxTensor], stream: &Stream, funding: &Funding) -> Result<(), Error> {
    match values {
        [] => Err(Error::PrefillScopeUnavailable),
        [one] => crate::backend::runtime::cache::complete_values([one.as_array()], stream).map_err(Into::into),
        [one, two] => crate::backend::runtime::cache::complete_values(
            [one.as_array(), two.as_array()], stream).map_err(Into::into),
        _ => {
            let observer = safemlx::OriginalScopeObserver::require_current()?;
            safemlx::OperationEvent::validate_nested_completion(values.len())?;
            // The caller charged this exact pointer destination before entry.
            // Its failure retains the same original account; no native root
            // or completion allowance is inferred from successful allocation.
            let mut roots = Roots::try_new(values.len(), funding.clone()).map_err(|failure| {
                let (cause, custody) = failure.into_parts();
                Error::Neural(eredu_nn::Error::backend_retained_source(
                    NestedPreparationFailure { cause, _custody: custody }))
            })?;
            roots.complete(values.iter().copied().map(array as fn(&MlxTensor)->&Array), &observer, stream)?;
            Ok(())
        }
    }
}
pub(super) fn sum(
    value: &MlxTensor,
    group: &Group,
    context: &Group,
    stream: &Stream,
) -> Result<Option<MlxTensor>, Error> {
    MlxNeuralBackend::with_parallel_control_context(context, |prepared| {
        let Some((_, funding)) = prepared else {
            return Ok(None);
        };
        charge(funding, 1)?;
        context
            .validate_model_collective_group(group)
            .map_err(Error::Neural)?;
        let output = context
            .sum_model(value.as_array(), stream)
            .map_err(Error::Neural)?;
        let output = MlxTensor::from_array(output);
        complete_native(&[&output], stream, funding)?;
        Ok(Some(output))
    })?
}
pub(super) fn gather(
    value: &MlxTensor,
    counts: &[usize],
    axis: usize,
    group: &Group,
    context: &Group,
    stream: &Stream,
) -> Result<Option<MlxTensor>, Error> {
    MlxNeuralBackend::with_parallel_control_context(context, |prepared| {
        let Some((_, funding)) = prepared else {
            return Ok(None);
        };
        charge(funding, 1)?;
        context
            .validate_model_collective_group(group)
            .map_err(Error::Neural)?;
        let output = super::super::parallel_gather::run_axis(
            value.as_array(),
            counts,
            axis,
            context,
            stream,
        )
        .map_err(Error::Neural)?;
        let output = MlxTensor::from_array(output);
        complete_native(&[&output], stream, funding)?;
        Ok(Some(output))
    })?
}
pub(super) fn complete(
    values: &[&MlxTensor],
    context: &Group,
    stream: &Stream,
) -> Result<Option<()>, Error> {
    MlxNeuralBackend::with_parallel_control_context(context, |prepared| {
        let Some((_, funding)) = prepared else {
            return Ok(None);
        };
        charge(funding, values.len())?;
        complete_native(values, stream, funding)?;
        Ok(Some(()))
    })?
}

#[path = "prepared_collectives/expert_input.rs"]
pub(super) mod expert_input;
